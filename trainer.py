"""
trainer.py
==========
Training pipeline for Few-Shot Learning — Standard and FewShot paradigms.

Classes
-------
TrainConfig             — all training HPs + backend switches
TrainingState           — mutable training state (separated for Optuna access)
TrainingHistory         — immutable per-epoch metrics log
LightningModuleWrapper  — Lightning module for pretrain phase
                          pretrain always uses Lightning
TrainerImpl             — all actual training logic
StandardTrainer         — thin wrapper → batch training
FewShotTrainer          — thin wrapper → episodic training

Backend
-------
Pretrain phase  : always Lightning (backend_pretrain='lightning')
                  EarlyStopping, ModelCheckpoint, mixed precision built in
Train phase     : PyTorch by default (backend_train='pytorch')
                  backend_train='lightning' — [FUTURE]

Training Flow
-------------
Phase 1 — Pretrain (both paradigms, identical):
    pool='pretrain', mode='batch', Lightning
    loss = F.cross_entropy(model(imgs, mode='linear'), labels)
    val  = 'val_seen', mode='batch'

Phase 2a — Standard train:
    pool='train', mode='batch', PyTorch
    loss = F.cross_entropy(model(imgs, mode='linear'), labels)
    val  = 'val_seen', mode='batch'

Phase 2b — FewShot episodic train:
    pool='train', mode='episodic', PyTorch
    s_emb = model(support, mode='embedding')
    q_emb = model(query,   mode='embedding')
    loss  = F.cross_entropy(model(s_emb, q_emb, mode='prototypical'), target)
    val   = 'val_unseen', mode='episodic'

Loss Notes
----------
CrossEntropyLoss always receives raw logits or proto distances — never softmax output.
CrossEntropyLoss applies LogSoftmax internally.
mode='linear'       → raw logits    → safe for CrossEntropyLoss
mode='prototypical' → distances     → safe for CrossEntropyLoss
mode='softmax'      → probabilities → NEVER pass to CrossEntropyLoss

device passed from notebook — never auto-detected inside any class.

Required Libraries
------------------
# torch>=2.0.0
# pytorch-lightning>=2.0.0  # pretrain phase — pip install pytorch-lightning
# tqdm>=4.0.0               # train phase — pip install tqdm
"""

import os
import time
import math
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm


# ==============================================================================
# TrainConfig
# ==============================================================================

@dataclass
class TrainConfig:
    """
    All training hyperparameters and backend configuration.

    HP levels:
        Default values   → used if not overridden
        Notebook dict    → TrainConfig(**override_dict)
        Optuna trial     → TrainConfig(lr=trial.suggest_float(...))

    Backend:
        backend_pretrain : 'lightning' (default) or 'pytorch'
                           Lightning — EarlyStopping, ModelCheckpoint,
                           mixed precision, CSVLogger built in
        backend_train    : 'pytorch' (default) or 'lightning' [FUTURE]

    Supported combinations:
        backend_pretrain='lightning' + backend_train='pytorch'  ← default
        backend_pretrain='pytorch'   + backend_train='pytorch'
    Unsupported combinations raise ValueError at TrainerImpl init.

    Scheduler options:
        'step'   → StepLR — sharp lr drop every lr_decay_step epochs
                   good when decay timing is known
        'cosine' → CosineAnnealingLR — smooth decay over all epochs
                   good default for few-shot learning
        'none'   → no scheduler    """

    # ── Checkpoint ────────────────────────────────────────────────────
    checkpoint_dir:  str  = 'checkpoints'
    run_id:          str  = 'run'       # stamped from ExperimentConfig.run_id by Runner
    pretrain_save_mode: str  = 'none'   # 'none'     — delete working ckpt, no export
                                        # 'full'     — keep working ckpt as-is (IS the export)
                                        # 'backbone' — strip head, save separately, delete working ckpt
    keep_final:         bool = True     # True  — keep _train_best.pt as-is (IS the export)
                                        # False — delete working ckpt, no export

    # ── Core ──────────────────────────────────────────────────────────
    lr:                   float = 1e-3
    weight_decay:         float = 1e-4
    epochs_pretrain:      int   = 100
    epochs_train:         int   = 100

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler:            str   = 'step'     # 'step', 'cosine', 'none'
    lr_decay_step:        int   = 20
    lr_decay_gamma:       float = 0.5

    # ── Regularization ────────────────────────────────────────────────
    grad_clip:            Optional[float] = None   # None = disabled

    # ── Early stopping ────────────────────────────────────────────────
    early_stop_patience:  int   = 10
    early_stop_metric:    str   = 'val_loss'   # 'val_loss' or 'val_acc'

    # ── Episodic protocol ─────────────────────────────────────────────
    n_way:                int   = 5
    k_shot:               int   = 5
    q_query:              int   = 15
    episodes_train:       int   = 600
    episodes_val:         int   = 200

    # ── Batch ─────────────────────────────────────────────────────────
    batch_size:           int   = 64
    num_workers:          int   = 2

    # ── Backend ───────────────────────────────────────────────────────
    # Pretrain : 'lightning' (default) or 'pytorch'
    # Train    : 'pytorch' (default) or 'lightning' [FUTURE]
    # Supported combinations:
    #   backend_pretrain='lightning' + backend_train='pytorch'  ← default
    #   backend_pretrain='pytorch'   + backend_train='pytorch'
    # Unsupported combinations caught at init by _validate_config().
    backend_pretrain: str = 'lightning'
    backend_train:    str = 'pytorch'

    # ── Verbose ───────────────────────────────────────────────────────
    verbose: bool = True   # True = print every epoch + tqdm (smoke/debug)
                           # False = phase summary only (real experiment runs)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Per-component lr override for trainable_param_groups
    # e.g. {'backbone': 1e-4, 'linear': 1e-3}
    lr_map:               Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def validate_config(self):
        """
        Validates field values at construction time.
        Catches bad config early — before any training starts.
        """
        
        # run_id must be set to something meaningful
        if not self.run_id or self.run_id.strip() == '':
            raise ValueError(
                "TrainConfig.run_id is empty. "
                "Set run_id in TrainConfig or let ExperimentRunner stamp it."
            )

        # checkpoint_dir must be a non-empty string
        if not self.checkpoint_dir:
            raise ValueError("TrainConfig.checkpoint_dir must not be empty.")

        valid_early_stop = {'val_loss', 'val_acc'}
        if self.early_stop_metric not in valid_early_stop:
            raise ValueError(
                f"TrainConfig.early_stop_metric='{self.early_stop_metric}' invalid. "
                f"Valid: {valid_early_stop}"
            )

        valid_schedulers = {'step', 'cosine', 'none'}
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"TrainConfig.scheduler='{self.scheduler}' invalid. "
                f"Valid: {valid_schedulers}"
            )

        valid_pretrain_backends = {'lightning', 'pytorch'}
        if self.backend_pretrain not in valid_pretrain_backends:
            raise ValueError(
                f"TrainConfig.backend_pretrain='{self.backend_pretrain}' invalid. "
                f"Valid: {valid_pretrain_backends}"
            )

        valid_train_backends = {'pytorch'}   # 'lightning' added when implemented
        if self.backend_train not in valid_train_backends:
            raise ValueError(
                f"TrainConfig.backend_train='{self.backend_train}' invalid. "
                f"Currently valid: {valid_train_backends}"
            )

        valid_pretrain_save_modes = {'none', 'full', 'backbone'}
        if self.pretrain_save_mode not in valid_pretrain_save_modes:
            raise ValueError(
                f"TrainConfig.pretrain_save_mode='{self.pretrain_save_mode}' invalid. "
                f"Valid: {valid_pretrain_save_modes}"
            )

        if self.n_way < 2:
            raise ValueError(f"TrainConfig.n_way={self.n_way} must be >= 2.")
        if self.k_shot < 1:
            raise ValueError(f"TrainConfig.k_shot={self.k_shot} must be >= 1.")
        if self.q_query < 1:
            raise ValueError(f"TrainConfig.q_query={self.q_query} must be >= 1.")
        if self.epochs_pretrain < 1:
            raise ValueError(f"TrainConfig.epochs_pretrain must be >= 1.")
        if self.epochs_train < 1:
            raise ValueError(f"TrainConfig.epochs_train must be >= 1.")
        if self.lr <= 0:
            raise ValueError(f"TrainConfig.lr must be > 0.")
        if self.batch_size < 1:
            raise ValueError(f"TrainConfig.batch_size must be >= 1.")
        if self.num_workers < 0:
            raise ValueError(f"TrainConfig.num_workers must be >= 0.")


# ==============================================================================
# TrainingState
# ==============================================================================

@dataclass
class TrainingState:
    """
    Mutable training state — separated from TrainerImpl logic.

    Separated for:
        Optuna access   → trial reads best_val_loss as objective
        Serialization   → saved independently in RunResult
        Resumption      → restore state to continue interrupted training

    Usage in Optuna:
        trainer = StandardTrainer(model, factory, config, device)
        trainer.pretrain()
        trainer.train()
        return trainer.impl.state.best_val_loss   # Optuna objective
    """

    # Current epoch
    epoch:                int   = 0

    # Best validation metrics — tracked for early stopping + checkpointing
    pretrain_best_val_loss: float = float('inf')
    pretrain_best_val_acc:  float = 0.0

    best_val_loss:        float = float('inf')
    best_val_acc:         float = 0.0

    # Export paths — written inside pretrain() / train() after each phase completes.
    # '' means nothing was saved (pretrain_save_mode='none' or keep_final=False).
    # Runner reads these for RunResult.
    pretrain_export_path: str   = ''   # path to kept pretrain file, or ''
    final_export_path:    str   = ''   # path to kept final file, or ''

    # Early stopping counter
    early_stop_counter:   int   = 0
    should_stop:          bool  = False

    # Phase flags
    is_pretrained:        bool  = False
    is_trained:           bool  = False

    def to_dict(self) -> dict:
        return asdict(self)

    def reset_early_stop(self):
        self.early_stop_counter   = 0
        self.should_stop          = False
        self.best_val_loss        = float('inf')
        self.best_val_acc         = 0.0



# ==============================================================================
# TrainingHistory
# ==============================================================================

@dataclass
class TrainingHistory:
    """
    Per-epoch metrics log — immutable record of training progress.
    Separated from mutable state for clean serialization.

    Phases:
        'pretrain'  → pretrain phase metrics
        'train'     → paradigm-specific train phase metrics
    """

    # Pretrain phase
    pretrain_train_loss: List[float] = field(default_factory=list)
    pretrain_train_acc:  List[float] = field(default_factory=list)
    pretrain_val_loss:   List[float] = field(default_factory=list)
    pretrain_val_acc:    List[float] = field(default_factory=list)

    # Train phase (standard or episodic)
    train_loss:          List[float] = field(default_factory=list)
    train_acc:           List[float] = field(default_factory=list)
    val_loss:            List[float] = field(default_factory=list)
    val_acc:             List[float] = field(default_factory=list)

    # Epoch indices for plotting
    pretrain_epochs:     List[int]   = field(default_factory=list)
    train_epochs:        List[int]   = field(default_factory=list)

    def log_pretrain(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.pretrain_epochs.append(epoch)
        self.pretrain_train_loss.append(train_loss)
        self.pretrain_train_acc.append(train_acc)
        self.pretrain_val_loss.append(val_loss)
        self.pretrain_val_acc.append(val_acc)

    def log_train(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.train_epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# LightningModuleWrapper — Lightning module for pretrain phase
# ==============================================================================

class LightningModuleWrapper:
    """
    Lightning LightningModule wrapper for pretrain phase.
    Pretrain is pure batch training — Lightning natural fit.
    Created internally by TrainerImpl._pretrain_lightning().
    Not exposed directly to user.

    Provides:
        EarlyStopping     — on val_loss, patience from TrainConfig
        ModelCheckpoint   — saves best checkpoint automatically
        CSVLogger         — logs metrics to CSV per epoch
        Mixed precision   — precision='16-mixed' on CUDA

    Install: pip install pytorch-lightning
    """

    def __init__(self,
                 model,
                 config:    TrainConfig,
                 factory,
                 state:     'TrainingState',
                 history:   'TrainingHistory',
                 ckpt_path: str):
        try:
            import lightning as L
            self._L = L
        except ImportError:
            raise ImportError(
                "pytorch-lightning required for pretrain.\n"
                "pip install pytorch-lightning"
            )

        self.model     = model
        self.config    = config
        self.factory   = factory
        self.state     = state
        self.history   = history
        self.ckpt_path = ckpt_path

    def build_and_fit(self):
        """
        Builds LightningModule, L.Trainer with callbacks, runs fit().
        Syncs metrics back to TrainingState and TrainingHistory after fit.
        """
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers   import CSVLogger

        config  = self.config
        factory = self.factory
        model   = self.model
        state   = self.state
        history = self.history

        # ── Inner LightningModule ──────────────────────────────────────
        class _PretrainModule(L.LightningModule):

            def __init__(self_inner):
                super().__init__()
                # Store reference to outer model — not re-wrapped
                self_inner.model = model

            def training_step(self_inner, batch, idx):
                imgs, labels = batch
                # Raw logits — CrossEntropyLoss applies softmax internally
                logits = self_inner.model(imgs, mode='linear')
                loss   = F.cross_entropy(logits, labels)
                acc    = (logits.argmax(1) == labels).float().mean()
                self_inner.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
                self_inner.log('train_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
                return loss

            def validation_step(self_inner, batch, idx):
                imgs, labels = batch
                logits = self_inner.model(imgs, mode='linear')
                loss   = F.cross_entropy(logits, labels)
                acc    = (logits.argmax(1) == labels).float().mean()
                self_inner.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
                self_inner.log('val_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)

            def configure_optimizers(self_inner):
                optimizer = torch.optim.Adam(
                    model.trainable_param_groups(
                        lr_map     = config.lr_map,
                        default_lr = config.lr
                    ),
                    lr           = config.lr,
                    weight_decay = config.weight_decay
                )
                if config.scheduler == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size = config.lr_decay_step,
                        gamma     = config.lr_decay_gamma
                    )
                    return [optimizer], [{'scheduler': scheduler, 'interval':  'epoch'}]
                elif config.scheduler == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=config.epochs_pretrain
                    )
                    return [optimizer], [{'scheduler': scheduler, 'interval':  'epoch'}]
                return optimizer

        # ── Data loaders ──────────────────────────────────────────────
        pretrain_loader = factory.get_loader(
            'pretrain', mode='batch',
            batch_size  = config.batch_size,
            num_workers = config.num_workers
        )
        val_loader = factory.get_loader(
            'val_seen', mode='batch',
            batch_size  = config.batch_size,
            num_workers = config.num_workers
        )

        # ── Callbacks ─────────────────────────────────────────────────
        early_stop = EarlyStopping(
            monitor  = 'val_loss',
            patience = config.early_stop_patience,
            mode     = 'min',
            verbose  = True
        )
        ckpt_callback = ModelCheckpoint(
            dirpath   = os.path.dirname(self.ckpt_path),
            filename  = os.path.basename(self.ckpt_path).replace('.pt', ''),
            monitor   = 'val_loss',
            mode      = 'min',
            save_top_k= 1,
            verbose   = True
        )
        logger = CSVLogger(
            save_dir = config.checkpoint_dir,
            name     = f"{config.run_id}_pretrain_logs"
        )

        # ── Precision ─────────────────────────────────────────────────
        # Mixed precision on CUDA — full precision on CPU
        precision = '16-mixed' if torch.cuda.is_available() else '32'

        # ── L.Trainer ─────────────────────────────────────────────────
        lt = L.Trainer(
            max_epochs        = config.epochs_pretrain,
            callbacks         = [early_stop, ckpt_callback],
            logger            = logger,
            precision         = precision,
            gradient_clip_val = config.grad_clip,
            enable_progress_bar = True,
            log_every_n_steps = 1,
        )

        lm = _PretrainModule()
        lt.fit(lm, pretrain_loader, val_loader)

        # ── Sync metrics back to state and history ────────────────────
        logs = lt.logged_metrics
        state.best_val_loss          = float(logs.get('val_loss',  float('inf')))
        state.best_val_acc           = float(logs.get('val_acc',   0.0))
        # ── Restore best-epoch weights and re-save in ModelFactory format ──
        # After lt.fit(), Lightning leaves model at LAST epoch, not best epoch.
        # Load the best .ckpt back via lm (lm.model IS the outer CompositeModel —
        # same Python object). Re-save as .pt with ModelFactory format so
        # _load_pretrain_best() works uniformly across both backends.
        lightning_ckpt_path = ckpt_callback.best_model_path
        mf_path = ''
        if lightning_ckpt_path and os.path.exists(lightning_ckpt_path):
            import torch as _torch
            from model_factory import ModelFactory as _MF
            device = next(self.model.parameters()).device   # ← get model device
            lightning_ckpt = _torch.load(lightning_ckpt_path, map_location=device)
            lm.load_state_dict(lightning_ckpt['state_dict'])
            mf_path = lightning_ckpt_path.replace('.ckpt', '.pt')
            _MF.save(self.model, mf_path)
            os.remove(lightning_ckpt_path)   # .ckpt no longer needed

        state.epoch = lt.current_epoch

        # Populate history from Lightning CSV log
        csv_path = os.path.join(logger.log_dir, 'metrics.csv')
        if os.path.exists(csv_path):
            self._sync_history_from_csv(csv_path, history)

        print(f"  Pretrain complete. Best val_loss: {state.best_val_loss:.4f}")
        
        # Cleanup Lightning logs dir — one level up from version_0 only
        logs_root = os.path.dirname(logger.log_dir)   # .../run_id_pretrain_logs
        if os.path.exists(logs_root):
            shutil.rmtree(logs_root)
            print(f"  [Lightning] Logs cleaned: {logs_root}")

        return mf_path



    def _sync_history_from_csv(self, csv_path: str,
                                history: 'TrainingHistory'):
        """
        Reads Lightning CSVLogger metrics.csv and populates TrainingHistory.
        Keeps TrainingHistory as single source of truth for plotting.
        """
        try:
            import csv
            train_loss_map: dict = {}
            train_acc_map:  dict = {}
            val_loss_map:   dict = {}
            val_acc_map:    dict = {}

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch = int(float(row.get('epoch', 0)))
                    for key, mapping in [
                        ('train_loss', train_loss_map),
                        ('train_acc',  train_acc_map),
                        ('val_loss',   val_loss_map),
                        ('val_acc',    val_acc_map),
                    ]:
                        if row.get(key):
                            mapping[epoch] = float(row[key])

            epochs = sorted(set(list(train_loss_map.keys()) + list(val_loss_map.keys())))
            for e in epochs:
                history.log_pretrain(
                    epoch      = e,
                    train_loss = train_loss_map.get(e, 0.0),
                    train_acc  = train_acc_map.get(e,  0.0),
                    val_loss   = val_loss_map.get(e,   0.0),
                    val_acc    = val_acc_map.get(e,    0.0),
                )
        except Exception as ex:
            print(f"  Warning: could not sync history from CSV: {ex}")


# ==============================================================================
# TrainerImpl — all actual training logic
# ==============================================================================

class TrainerImpl:
    """
    All actual training logic.
    Not used directly — accessed via StandardTrainer or FewShotTrainer.

    Backend dispatch:
        pretrain()        → _pretrain_lightning() or _pretrain_pytorch()
        train_batch()     → _run_train_pytorch() or _run_train_lightning()
        train_episodic()  → _run_train_pytorch() or _run_train_lightning()

    Unsupported backend combinations caught at init by validate_config().

    Methods:
        _pretrain_pytorch()         — pretrain via pure PyTorch loop
        _pretrain_lightning()       — pretrain via LightningModuleWrapper
        
        _run_train_pytorch()            — batch or episodic train, pure PyTorch
            _batch_epoch()              — single batch epoch (train or eval)
            _episodic_epoch()           — single episodic epoch (train or eval)
        _run_train_lightning()      — [FUTURE] standard train via Lightning
        
        _setup_optimizer()          — Adam with optional per-component lr
        _setup_scheduler()          — step/cosine/none
        _is_improved()              — val_loss improvement check
        _early_stopping_check()     — returns True if should stop
        _log_epoch()                — prints epoch metrics
        _save_checkpoint()          — via ModelFactory
        _load_pretrain_best()       — load best + handle file per pretrain_save_mode
        _load_train_best()          — load best + handle file per keep_final

    State and history public for Optuna:
        trainer.impl.state.best_val_loss   → Optuna objective
        trainer.impl.history               → for plotting
    """
    def __init__(self,
                 model,
                 factory,
                 config:    TrainConfig,
                 device:    torch.device,
                 paradigm:  str):
        """
        Args:
            model    : CompositeModel instance
            factory  : SmartDataLoaderFactory instance
            config   : TrainConfig
            device   : torch.device — from notebook
            paradigm : 'standard' or 'fewshot'
        """
        self.model    = model
        self.factory  = factory
        self.config   = config
        self.device   = device
        self.paradigm = paradigm

        # Validate backend combination upfront — before any training starts
        self.config.validate_config()

        # Public state — accessible by Optuna, ExperimentRunner
        self.state   = TrainingState()
        self.history = TrainingHistory()

        self._criterion = nn.CrossEntropyLoss()

        # Private working checkpoint paths — internal to TrainerImpl only.
        # Written by training loops on each improvement.
        # Read by _load_pretrain_best() / _load_train_best() at end of each phase.
        # Never stored in TrainingState — TrainingState holds only export paths.
        self._pretrain_best_path: str = ''
        self._train_best_path:    str = ''
        self._phase_start_time: float = 0.0   # wall clock per phase
        self._best_epoch:       int   = 0     # epoch where best checkpoint was saved

        self.validate()


    def validate(self):
        # For episodic paradigm, n_way must be satisfiable given the factory's pools
        # (light check — full pool validation happens in EpisodicBatchSampler)
        if self.paradigm == 'fewshot':
            available_pools = self.factory.valid_pools()
            for required in ('train', 'val_unseen'):
                if required not in available_pools:
                    raise ValueError(
                        f"FewShot paradigm requires pool '{required}' "
                        f"but factory only has: {available_pools}. "
                        f"Check FewShotClassSplitter split config."
                    )
        elif self.paradigm == 'standard':
            available_pools = self.factory.valid_pools()
            for required in ('train', 'val_seen'):
                if required not in available_pools:
                    raise ValueError(
                        f"Standard paradigm requires pool '{required}' "
                        f"but factory only has: {available_pools}."
                    )

    # ------------------------------------------------------------------
    # Public dispatch — routes to backend
    # ------------------------------------------------------------------

    def _model_summary_line(self) -> str:
        """Returns one-line model summary: params + estimated size in MB."""
        total  = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Estimate size: assume float32 (4 bytes) for params
        size_mb = total * 4 / 1024 / 1024
        return (f"  params={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M  "
                f"est_size={size_mb:.1f}MB")

    def _gpu_memory_mb(self) -> float:
        """Returns current GPU memory allocated in MB, or 0 on CPU."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        return 0.0

    # ------------------------------------------------------------------

    def pretrain(self):
        """
        Phase 1 — shared pretrain for both paradigms.
        Batch mode, CrossEntropyLoss on raw logits.
        Val on val_seen (batch).
        Saves backbone checkpoint after completion.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.state.reset_early_stop()
        self.state.epoch = 0

        backend_label = self.config.backend_pretrain
        print(f"\n  Phase 1: Pretrain [{backend_label} | {self.config.epochs_pretrain} epochs]")

        if self.config.backend_pretrain == 'pytorch':
            self._pretrain_pytorch()
        else:
            self._pretrain_lightning()

        # Restore best-pretrain-epoch weights before returning.
        # Caller (runner) receives model already at best state — no load_best() needed.
        self._load_pretrain_best()
        self.state.is_pretrained = True

    def train_batch(self, val_pool: str = 'val_seen'):
        """
        Phase 2a — standard batch training.
        Called by StandardTrainer.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        print(f"\n  Phase 2: Train [standard | {self.config.backend_train} | {self.config.epochs_train} epochs]")

        if self.config.backend_train == 'pytorch':
            self._run_train_pytorch(
                train_pool = 'train',
                val_pool   = val_pool,
                episodic   = False
            )
        else:
            self._run_train_lightning()

        # Restore best-train-epoch weights before returning.
        self._load_train_best()
        self.state.is_trained = True

    def train_episodic(self, val_pool: str = 'val_unseen'):
        """
        Phase 2b — episodic meta-training.
        Called by FewShotTrainer.
        Freezes linear head before training — only backbone trains.
        Unfreezes after for evaluation.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        print(f"\n  Phase 2: Train [fewshot | {self.config.backend_train} | {self.config.episodes_train} eps/epoch | {self.config.epochs_train} epochs]")

        # Freeze linear head — episodic training does not update it
        self.model.freeze('linear')
        self.model.freeze('softmax')

        if self.config.backend_train == 'pytorch':
            self._run_train_pytorch(
                train_pool = 'train',
                val_pool   = val_pool,
                episodic   = True
            )
        else:
            self._run_train_lightning()

        # Unfreeze for evaluation
        self.model.unfreeze('linear')
        self.model.unfreeze('softmax')

        # Restore best-train-epoch weights before returning.
        # Note: checkpoint was saved with linear frozen. ModelFactory.load()
        # restores frozen_names from checkpoint — linear will be frozen again.
        # This is correct: fewshot eval uses prototypical path, not linear.
        # Softmax eval uses pretrain-era linear weights — intentional diagnostic.
        self._load_train_best()
        self.state.is_trained = True

    # ------------------------------------------------------------------
    # PyTorch — pretrain
    # ------------------------------------------------------------------

    def _pretrain_pytorch(self):
        """Batch pretrain loop — pure PyTorch."""
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer, self.config.epochs_pretrain)

        pretrain_loader = self.factory.get_loader(
            'pretrain', mode='batch',
            batch_size  = self.config.batch_size,
            num_workers = self.config.num_workers
        )
        val_loader = self.factory.get_loader(
            'val_seen', mode='batch',
            batch_size  = self.config.batch_size,
            num_workers = self.config.num_workers
        )

        self._phase_start_time = time.time()
        epochs_run = 0

        for epoch in range(self.config.epochs_pretrain):
            self.state.epoch = epoch
            epochs_run = epoch + 1

            train_loss, train_acc = self._batch_epoch(pretrain_loader, optimizer, is_train=True)
            val_loss, val_acc = self._batch_epoch(val_loader, optimizer=None, is_train=False)

            if scheduler is not None:
                scheduler.step()

            self.history.log_pretrain(epoch, train_loss, train_acc, val_loss, val_acc)
            if self.config.verbose:
                self._log_epoch('pretrain', epoch, train_loss, train_acc, val_loss, val_acc)

            # Checkpoint on improvement
            if self._is_improved(val_loss, val_acc):
                path = os.path.join( self.config.checkpoint_dir, f"{self.config.run_id}_pretrain_best.pt" )
                self._save_checkpoint(path)
                self._pretrain_best_path = path
                self._best_epoch = epoch
                self.state.early_stop_counter = 0
            else:
                self.state.early_stop_counter += 1

            if self._early_stopping_check():
                if self.config.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        elapsed = (time.time() - self._phase_start_time) / 60
        print(f"  Pretrain — ran {epochs_run}/{self.config.epochs_pretrain} epochs  "
              f"best @ epoch {self._best_epoch}  "
              f"val_loss={self.state.best_val_loss:.2f}  val_acc={self.state.best_val_acc:.2f}  "
              f"time={elapsed:.1f}min")

    # ------------------------------------------------------------------
    # PyTorch — train (batch or episodic)
    # ------------------------------------------------------------------

    def _run_train_pytorch(self, train_pool: str, val_pool: str, episodic: bool):
        """
        Generic train loop — batch or episodic.
        Episodic: trains backbone via prototypical loss.
        Batch:    trains backbone + linear head via CrossEntropyLoss.
        """
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer, self.config.epochs_train)

        if episodic:
            train_loader = self.factory.get_loader(
                train_pool, mode='episodic',
                n          = self.config.n_way,
                k          = self.config.k_shot,
                q          = self.config.q_query,
                iterations = self.config.episodes_train,
                num_workers= self.config.num_workers
            )
            val_loader = self.factory.get_loader(
                val_pool, mode='episodic',
                n          = self.config.n_way,
                k          = self.config.k_shot,
                q          = self.config.q_query,
                iterations = self.config.episodes_val,
                num_workers= self.config.num_workers
            )
        else:
            train_loader = self.factory.get_loader(
                train_pool, mode='batch',
                batch_size  = self.config.batch_size,
                num_workers = self.config.num_workers
            )
            val_loader = self.factory.get_loader(
                val_pool, mode='batch',
                batch_size  = self.config.batch_size,
                num_workers = self.config.num_workers
            )

        mode_str = 'episodic' if episodic else 'batch'
        unit_str = f"{self.config.episodes_train} eps/epoch" if episodic else "batch"

        self._phase_start_time = time.time()
        self._best_epoch = 0
        epochs_run = 0

        for epoch in range(self.config.epochs_train):
            self.state.epoch = epoch
            epochs_run = epoch + 1

            if episodic:
                # Set epoch for EpisodicBatchSampler RNG variation
                if hasattr(train_loader.batch_sampler, 'set_epoch'):
                    train_loader.batch_sampler.set_epoch(epoch)

                train_loss, train_acc = self._episodic_epoch(train_loader, optimizer, is_train=True)
                val_loss, val_acc = self._episodic_epoch(val_loader, optimizer=None, is_train=False)
            else:
                train_loss, train_acc = self._batch_epoch(train_loader, optimizer, is_train=True)
                val_loss, val_acc = self._batch_epoch(val_loader, optimizer=None, is_train=False)

            if scheduler is not None:
                scheduler.step()

            self.history.log_train(epoch, train_loss, train_acc, val_loss, val_acc)
            if self.config.verbose:
                self._log_epoch('train', epoch, train_loss, train_acc, val_loss, val_acc)

            # Checkpoint on improvement
            if self._is_improved(val_loss, val_acc):
                path = os.path.join(self.config.checkpoint_dir, f"{self.config.run_id}_train_best.pt")
                self._save_checkpoint(path)
                self._train_best_path = path
                self._best_epoch = epoch
                self.state.early_stop_counter   = 0
            else:
                self.state.early_stop_counter += 1

            if self._early_stopping_check():
                if self.config.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        elapsed = (time.time() - self._phase_start_time) / 60
        print(f"  Train  — ran {epochs_run}/{self.config.epochs_train} epochs  "
              f"best @ epoch {self._best_epoch}  "
              f"val_loss={self.state.best_val_loss:.2f}  val_acc={self.state.best_val_acc:.2f}  "
              f"time={elapsed:.1f}min")

    # ------------------------------------------------------------------
    # PyTorch — single epoch loops
    # ------------------------------------------------------------------

    def _batch_epoch(self,
                     loader,
                     optimizer,
                     is_train: bool) -> Tuple[float, float]:
        """
        Single batch epoch — train or eval.
        Returns (avg_loss, avg_acc).
        """
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, leave=False,
                        desc=f"  {'train' if is_train else 'val  '}",
                        disable=not self.config.verbose)
            for imgs, labels in pbar:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                # Forward — raw logits
                logits = self.model(imgs, mode='linear')

                # Loss — CrossEntropyLoss on raw logits
                # NEVER pass softmax output here — double softmax = wrong gradients
                loss = self._criterion(logits, labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    optimizer.step()

                acc = (logits.argmax(1) == labels).float().mean().item()
                total_loss += loss.item()
                total_acc  += acc
                n_batches  += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                   'acc':  f'{acc:.4f}'})

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    def _episodic_epoch(self,
                        loader,
                        optimizer,
                        is_train: bool) -> Tuple[float, float]:
        """
        Single episodic epoch — train or eval.
        Each batch is a TaskCollator dict {support, query, target}.
        Returns (avg_loss, avg_acc).
        """
        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_episodes = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, leave=False,
                        desc=f"  {'train' if is_train else 'val  '}",
                        disable=not self.config.verbose)
            for batch in pbar:
                support = batch['support'].to(self.device)  # [N, K, C, H, W]
                query   = batch['query'].to(self.device)    # [N, Q, C, H, W]
                target  = batch['target'].to(self.device)   # [N*Q]

                N, K, C, H, W = support.shape
                Q             = query.shape[1]

                # Combine support + query into one backbone call.
                # For Hybrid (Run 6): GATRelationalLayer sees the full episode
                # graph — cross-group edges allow support→query info flow.
                # For CNN/GNN runs: no difference — each image is processed
                # independently by the backbone regardless of order.
                episode = torch.cat([
                    support.reshape(N * K, C, H, W),
                    query.reshape(N * Q, C, H, W)
                ], dim=0)                                             # [N*(K+Q), C, H, W]

                all_emb = self.model(episode, mode='embedding')      # [N*(K+Q), D]

                s_emb = all_emb[:N * K]                              # [N*K, D]
                q_emb = all_emb[N * K:]                              # [N*Q, D]

                # Prototypical distances
                # CrossEntropyLoss on distances — safe, no softmax involved
                dists = self.model( support_emb=s_emb, query_emb=q_emb, mode='prototypical' )   # [N*Q, N]

                loss = self._criterion(dists, target)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    optimizer.step()

                acc = (dists.argmax(1) == target).float().mean().item()
                total_loss += loss.item()
                total_acc  += acc
                n_episodes += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                   'acc':  f'{acc:.4f}'})

        return total_loss / max(n_episodes, 1), total_acc / max(n_episodes, 1)

    # ------------------------------------------------------------------
    # Lightning — stubs
    # ------------------------------------------------------------------
    def _pretrain_lightning(self):
        """Pretrain via LightningModuleWrapper — always used for pretrain."""
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"{self.config.run_id}_pretrain_best")
        wrapper = LightningModuleWrapper(
            model     = self.model,
            config    = self.config,
            factory   = self.factory,
            state     = self.state,
            history   = self.history,
            ckpt_path = ckpt_path
        )

        mf_path = wrapper.build_and_fit()
        if not mf_path:
            raise RuntimeError(
                f"Lightning pretrain failed to produce checkpoint. "
                f"Expected .pt at: {ckpt_path}.pt"
            )
        if not os.path.exists(mf_path):
            raise RuntimeError(
                f"Lightning pretrain checkpoint not found after save. "
                f"Expected: {mf_path}. "
                f"Dir contents: {os.listdir(self.config.checkpoint_dir)}"
            )

        self._pretrain_best_path = mf_path
        self._pretrain_best_path = mf_path
        print(f"  Pretrain — lightning  best val_loss={self.state.best_val_loss:.2f}  "
              f"val_acc={self.state.best_val_acc:.2f}")
        #print(f"  [Lightning] _pretrain_best_path = '{mf_path}'")


    def _run_train_lightning(self):
        """
        [FUTURE] Standard batch/Episodic train via Lightning.
        backend_train='lightning' not yet implemented.
        _validate_config() prevents reaching here until implemented.
        """
        pass

    # ------------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Adam optimizer.
        Uses per-component lr_map if provided in config.
        Otherwise uses single lr for all trainable params.
        """
        param_groups = self.model.trainable_param_groups(
            lr_map     = self.config.lr_map,
            default_lr = self.config.lr
        )
        return torch.optim.Adam(
            param_groups,
            lr           = self.config.lr,
            weight_decay = self.config.weight_decay
        )

    def _setup_scheduler(self, optimizer, epochs):
        """Step, cosine, or no scheduler."""
        if self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size = self.config.lr_decay_step,
                gamma     = self.config.lr_decay_gamma
            )
        elif self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        else:
            return None

    # ------------------------------------------------------------------
    # Early stopping + checkpointing
    # ------------------------------------------------------------------

    def _is_improved(self, val_loss: float, val_acc: float) -> bool:
        """
        Returns True if validation metric improved.
        early_stop_metric='val_loss' → lower is better (default)
        early_stop_metric='val_acc'  → higher is better
        """
        if self.config.early_stop_metric == 'val_loss':
            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                self.state.best_val_acc  = val_acc
                return True
        else:
            if val_acc > self.state.best_val_acc:
                self.state.best_val_acc  = val_acc
                self.state.best_val_loss = val_loss
                return True
        return False

    def _early_stopping_check(self) -> bool:
        """Returns True if training should stop."""
        if self.state.early_stop_counter >= self.config.early_stop_patience:
            self.state.should_stop = True
            return True
        return False

    def _save_checkpoint(self, path: str):
        """Internal — save full model during training loop on improvement."""
        from model_factory import ModelFactory
        ModelFactory.save(self.model, path)

    def _load_pretrain_best(self):
        """
        Internal — called at end of pretrain() before returning to caller.

        1. Load best-pretrain-epoch weights from self._pretrain_best_path into model.
        2. Handle the working checkpoint file based on pretrain_save_mode:
             'none'     — delete file.  state.pretrain_export_path = ''
             'full'     — keep file as-is, it IS the export.
             'backbone' — save backbone-only to new file, delete working ckpt.
        3. Write export path to state.pretrain_export_path ('' if none).
        """
        from model_factory import ModelFactory
        path = self._pretrain_best_path

        if not path or not os.path.exists(path):
            print(
                f"  Warning: pretrain best checkpoint not found at '{path}'. "
                f"Model remains at last epoch."
            )
            self.state.pretrain_export_path = ''
            return

        # 1. Load best weights into live model
        ModelFactory.load(self.model, path)
        self.model.to(self.device)

        # 2 + 3. Handle file and set export path
        mode = self.config.pretrain_save_mode

        if mode == 'none':
            os.remove(path)
            self.state.pretrain_export_path = ''

        elif mode == 'full':
            # File already contains full model — it IS the export, keep as-is
            self.state.pretrain_export_path = path
            print(f"  Pretrain export (full model): {path}")

        elif mode == 'backbone':
            # Strip head: save backbone-only to new file, delete working ckpt
            export_path = path.replace('_pretrain_best.pt', '_pretrain_backbone.pt')
            ModelFactory.save_backbone(self.model, export_path)
            os.remove(path)
            self.state.pretrain_export_path = export_path
            print(f"  Pretrain export (backbone only): {export_path}")

        self.state.pretrain_best_val_loss = self.state.best_val_loss
        self.state.pretrain_best_val_acc  = self.state.best_val_acc


    def _load_train_best(self):
        """
        Internal — called at end of train_batch() / train_episodic() before returning.

        1. Load best-train-epoch weights from self._train_best_path into model.
        2. Handle the working checkpoint file based on keep_final:
             True  — keep file as-is, it IS the export.
             False — delete file.  state.final_export_path = ''
        3. Write export path to state.final_export_path ('' if none).
        """
        from model_factory import ModelFactory
        path = self._train_best_path

        if not path or not os.path.exists(path):
            print(
                f"  Warning: train best checkpoint not found at '{path}'. "
                f"Model remains at last epoch."
            )
            self.state.final_export_path = ''
            return

        # 1. Load best weights into live model
        ModelFactory.load(self.model, path)
        self.model.to(self.device)

        # 2 + 3. Handle file and set export path
        if self.config.keep_final:
            # File already contains full trained model — it IS the export, keep as-is
            self.state.final_export_path = path
            print(f"  Final model export: {path}")
        else:
            os.remove(path)
            self.state.final_export_path = ''

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(self, phase: str, epoch: int,
                   train_loss: float, train_acc: float,
                   val_loss: float,   val_acc: float):
        """Prints epoch metrics."""
        print(
            f"  [{phase}] epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
            + (' ✓' if self.state.early_stop_counter == 0 else '')
        )


# ==============================================================================
# StandardTrainer — thin wrapper
# ==============================================================================

class StandardTrainer:
    """
    Thin wrapper for Standard paradigm (Runs 1, 3, 5).
    Routes to TrainerImpl batch methods.

    Usage:
        trainer = StandardTrainer(model, factory, config, device)
        trainer.pretrain()
        trainer.train()

        # Access state for Optuna
        best_loss = trainer.impl.state.best_val_loss

        # Access history for plotting
        history = trainer.impl.history
    """

    def __init__(self,
                 model,
                 factory,
                 config: TrainConfig,
                 device: torch.device):
        self.impl = TrainerImpl(
            model    = model,
            factory  = factory,
            config   = config,
            device   = device,
            paradigm = 'standard'
        )

    def pretrain(self):
        """Phase 1 — batch pretrain, shared with FewShot."""
        self.impl.pretrain()

    def train(self):
        """Phase 2a — batch training on seen classes."""
        self.impl.train_batch(val_pool='val_seen')

    @property
    def state(self) -> TrainingState:
        return self.impl.state

    @property
    def history(self) -> TrainingHistory:
        return self.impl.history


# ==============================================================================
# FewShotTrainer — thin wrapper
# ==============================================================================

class FewShotTrainer:
    """
    Thin wrapper for FewShot paradigm (Runs 2, 4, 6).
    Routes to TrainerImpl episodic methods.

    Episodic training note:
        Linear head frozen during episodic training — only backbone updates.
        Linear head unfrozen after training for evaluation.
        val_unseen used for meta-validation (different classes from train).

    Usage:
        trainer = FewShotTrainer(model, factory, config, device)
        trainer.pretrain()
        trainer.train()

        # Access state for Optuna
        best_loss = trainer.impl.state.best_val_loss
    """

    def __init__(self,
                 model,
                 factory,
                 config: TrainConfig,
                 device: torch.device):
        self.impl = TrainerImpl(
            model    = model,
            factory  = factory,
            config   = config,
            device   = device,
            paradigm = 'fewshot'
        )

    def pretrain(self):
        """Phase 1 — batch pretrain, shared with Standard."""
        self.impl.pretrain()

    def train(self):
        """Phase 2b — episodic meta-training on base classes."""
        self.impl.train_episodic(val_pool='val_unseen')

    @property
    def state(self) -> TrainingState:
        return self.impl.state

    @property
    def history(self) -> TrainingHistory:
        return self.impl.history
