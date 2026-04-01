"""
trainer.py
==========
Training pipeline for Few-Shot Learning — Standard and FewShot paradigms.

Classes
-------
TrainConfig     — all training HPs and configuration
TrainingState   — mutable training state (separated for Optuna access)
TrainingHistory — immutable per-epoch metrics log
TrainerImpl     — all actual training logic
StandardTrainer — thin wrapper → batch training
FewShotTrainer  — thin wrapper → episodic training

Training Flow
-------------
Phase 1 — Pretrain (both paradigms, identical):
    pool='pretrain', mode='batch', PyTorch
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
# tqdm>=4.0.0               # train phase — pip install tqdm

Elastic Weight Consolidation (EWC)

The primary experimental finding of this study is that the train phase 
consistently degrades novel class generalisation across all 6 runs — 
proto_novel drops from pretrain level in every architecture and paradigm 
combination. EWC directly addresses this by constraining backbone weights 
to stay close to the pretrain checkpoint during the train phase, weighted 
by their importance to pretrain performance.

Implementation requires:
  1. After pretrain() — estimate Fisher information via 50-batch 
     forward pass on pretrain pool (~2 min)
  2. Store pretrain checkpoint weights as frozen reference θ*
  3. During train phase — add EWC penalty to every loss computation:
     total_loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²

Expected benefit: proto_novel accuracy maintained near pretrain level 
(~0.82-0.86 proto_seen quality) rather than degrading to 0.63-0.69 
as observed in Run 2. This would validate the pretrain checkpoint as 
the optimal starting point for novel class generalisation and confirm 
that the train phase degrades rather than improves novel class features.

λ tuning required: start at λ=1000, reduce if train loss cannot improve.
Complementary to existing L2 regularisation (weight_decay) — addresses
a different problem (forgetting vs overfitting) and should be used 
alongside weight_decay, not as a replacement.

Reference: Kirkpatrick, J. et al. (2017). Overcoming catastrophic 
forgetting in neural networks. PNAS, 114(13), 3521–3526.
"""

import os
import time
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
    All training hyperparameters and configuration.

    HP levels:
        Default values   → used if not overridden
        Notebook dict    → TrainConfig(**override_dict)
        Optuna trial     → TrainConfig(lr=trial.suggest_float(...))

    Scheduler options:
        'step'   → StepLR — sharp lr drop every lr_decay_step epochs
                   good when decay timing is known
        'cosine' → CosineAnnealingLR — smooth decay over all epochs
                   good default for few-shot learning
        'none'   → no scheduler    """

    # ── Checkpoint ────────────────────────────────────────────────────
    checkpoint_dir:  str  = 'checkpoints'
    run_id:          str  = 'run'       # stamped from ExperimentConfig.run_id by Runner
    # Trainer always saves both pretrain and train checkpoints.
    # ExperimentRunner decides what to keep based on exec_config flags.

    # ── Core ──────────────────────────────────────────────────────────
    lr:                   float = 1e-3
    epochs_pretrain:      int   = 100
    epochs_train:         int   = 100

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler:            str   = 'step'     # 'step', 'cosine', 'none'
    lr_decay_step:        int   = 20
    lr_decay_gamma:       float = 0.5

    # ── Regularization ────────────────────────────────────────────────
    weight_decay:         float = 1e-4              # ← L2 regularisation — penalty on weight magnitude
    label_smoothing:      float = 0.1
    grad_clip:            Optional[float] = None    # None = disabled

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

    # ── Verbose ───────────────────────────────────────────────────────
    verbose: bool = True   # True = print every epoch + tqdm (smoke/debug)
                           # False = phase summary only (real experiment runs)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Per-component lr override for trainable_param_groups
    # e.g. {'backbone': 1e-4, 'linear': 1e-3}
    lr_map:               Optional[Dict[str, float]] = None

    # ── Train phase improvements (all default to disabled = current behaviour) ──
    ewc_lambda:       float = 0.0    # EWC penalty weight — 0.0 = disabled
    freeze_n_epochs:  int   = 0      # freeze backbone first N train epochs — 0 = disabled
    joint_loss_alpha: float = 0.0    # KL embedding anchor weight — 0.0 = disabled
    warm_start:       bool  = False  # init train early-stop from pretrain best — False = reset

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

    # Runtime tensors — populated during training, excluded from JSON serialization
    pretrain_emb_mean:    Any   = field(default=None, repr=False)  # [D] — KL joint loss reference
    pretrain_emb_var:     Any   = field(default=None, repr=False)  # [D] — KL joint loss reference
    fisher_diag:          Any   = field(default=None, repr=False)  # dict{name: tensor} — EWC Fisher
    pretrain_weights:     Any   = field(default=None, repr=False)  # dict{name: tensor} — EWC reference

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ('pretrain_emb_mean', 'pretrain_emb_var', 'fisher_diag', 'pretrain_weights'):
            d.pop(key, None)  # remove tensors from dict for JSON serialization
        return d

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
# TrainerImpl — all actual training logic
# ==============================================================================

class TrainerImpl:
    """
    All actual training logic.
    Not used directly — accessed via StandardTrainer or FewShotTrainer.

    Backend dispatch:
        pretrain()        → _pretrain_pytorch()
        train_batch()     → _run_train_pytorch()
        train_episodic()  → _run_train_pytorch()

    Unsupported backend combinations caught at init by validate_config().

    Methods:
        _pretrain_pytorch()         — pretrain via pure PyTorch loop
        
        _run_train_pytorch()            — batch or episodic train, pure PyTorch
            _batch_epoch()              — single batch epoch (train or eval)
            _episodic_epoch()           — single episodic epoch (train or eval)
        
        _setup_optimizer()          — AdamW with optional per-component lr
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

        self._criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

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

    def pretrain(self, optuna_trial=None):
        """
        Phase 1 — shared pretrain for both paradigms.
        Batch mode, CrossEntropyLoss on raw logits.
        Val on val_seen (batch).
        Saves backbone checkpoint after completion.
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.state.reset_early_stop()
        self.state.epoch = 0

        print(f"\n  Phase 1: Pretrain [pytorch | {self.config.epochs_pretrain} epochs]")
        self.model.freeze('prototypical')

        self._pretrain_pytorch(optuna_trial=optuna_trial)

        self.model.unfreeze('prototypical')

        # Restore best-pretrain-epoch weights before returning.
        # Caller (runner) receives model already at best state — no load_best() needed.
        self._load_pretrain_best()
        self.state.is_pretrained = True

        # Compute reference stats for train phase improvements (only if enabled)
        if self.config.joint_loss_alpha > 0:
            mean, var = self._compute_emb_stats()
            self.state.pretrain_emb_mean = mean
            self.state.pretrain_emb_var  = var
        if self.config.ewc_lambda > 0:
            self._compute_fisher()

    def train_batch(self, val_pool: str = 'val_seen', optuna_trial=None):
        """
        Phase 2a — standard batch training.
        Called by StandardTrainer.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        if self.config.warm_start:
            self.state.best_val_loss = self.state.pretrain_best_val_loss
            self.state.best_val_acc  = self.state.pretrain_best_val_acc

        print(f"\n  Phase 2: Train [standard | pytorch | {self.config.epochs_train} epochs]")

        self.model.freeze('prototypical')

        self._run_train_pytorch(
            train_pool = 'train',
            val_pool   = val_pool,
            episodic   = False,
            optuna_trial=optuna_trial
        )

        self.model.unfreeze('prototypical')

        # Restore best-train-epoch weights before returning.
        self._load_train_best()
        self.state.is_trained = True

    def train_episodic(self, val_pool: str = 'val_unseen', optuna_trial=None):
        """
        Phase 2b — episodic meta-training.
        Called by FewShotTrainer.
        Freezes linear head before training — only backbone trains.
        Unfreezes after for evaluation.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        if self.config.warm_start:
            self.state.best_val_loss = self.state.pretrain_best_val_loss
            self.state.best_val_acc  = self.state.pretrain_best_val_acc

        print(f"\n  Phase 2: Train [fewshot | pytorch | {self.config.episodes_train} eps/epoch | {self.config.epochs_train} epochs]")

        # Freeze linear head — episodic training does not update it
        self.model.freeze('linear')
        self.model.freeze('softmax')

        self._run_train_pytorch(
            train_pool = 'train',
            val_pool   = val_pool,
            episodic   = True,
            optuna_trial=optuna_trial
        )

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

    def _pretrain_pytorch(self, optuna_trial=None):
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

            # --- INSERTED CODE START for Optuna ---
            if optuna_trial is not None:
                optuna_trial.report(val_loss, step=epoch)
                if optuna_trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            # --- INSERTED CODE END ---

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

    def _run_train_pytorch(self, train_pool: str, val_pool: str, episodic: bool, optuna_trial=None):
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

            # freeze_n_epochs: freeze backbone for first N epochs, unfreeze after
            # mutual exclusion with EWC — if EWC active, freeze_n ignored
            if self.config.ewc_lambda == 0 and self.config.freeze_n_epochs > 0:
                if epoch == 0:
                    self.model.freeze('backbone')
                elif epoch == self.config.freeze_n_epochs:
                    self.model.unfreeze('backbone')

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

            # --- INSERTED CODE START for Optuna ---
            if optuna_trial is not None:
                optuna_trial.report(val_loss, step=epoch)
                if optuna_trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            # --- INSERTED CODE END ---

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

        # Ensure backbone unfrozen after train loop (in case freeze_n >= epochs_train)
        if self.config.ewc_lambda == 0 and self.config.freeze_n_epochs > 0:
            self.model.unfreeze('backbone')

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
                    if self.config.ewc_lambda > 0:
                        loss = loss + self.config.ewc_lambda * self._ewc_penalty()
                    if self.config.joint_loss_alpha > 0:
                        emb  = self.model(imgs, mode='embedding')
                        loss = loss + self.config.joint_loss_alpha * self._compute_kl_loss(emb)
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
                    if self.config.ewc_lambda > 0:
                        loss = loss + self.config.ewc_lambda * self._ewc_penalty()
                    if self.config.joint_loss_alpha > 0:
                        # all_emb already computed above — reuse for KL anchor
                        loss = loss + self.config.joint_loss_alpha * self._compute_kl_loss(all_emb)
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
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc':  f'{acc:.4f}'})

        return total_loss / max(n_episodes, 1), total_acc / max(n_episodes, 1)

    # ------------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        AdamW optimizer.
        Uses per-component lr_map if provided in config.
        Otherwise uses single lr for all trainable params.
        """
        param_groups = self.model.trainable_param_groups(
            lr_map     = self.config.lr_map,
            default_lr = self.config.lr
        )
        return torch.optim.AdamW(
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

    # ------------------------------------------------------------------
    # Train phase improvement helpers
    # ------------------------------------------------------------------

    def _compute_emb_stats(self):
        """
        Forward pass over val_seen — compute backbone embedding mean and var.
        Called once at end of pretrain() when joint_loss_alpha > 0.
        Returns (mean [D], var [D]) as detached CPU tensors.
        """
        loader = self.factory.get_loader(
            'val_seen', mode='batch',
            batch_size  = self.config.batch_size,
            num_workers = self.config.num_workers
        )
        self.model.eval()
        all_embs = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                emb  = self.model(imgs, mode='embedding')
                all_embs.append(emb.cpu())
        all_embs = torch.cat(all_embs, dim=0)   # [N, D]
        return all_embs.mean(0).to(self.device), all_embs.var(0).to(self.device)

    def _compute_fisher(self):
        """
        Compute diagonal Fisher information matrix over pretrain val_seen.
        Called once at end of pretrain() when ewc_lambda > 0.
        Stores fisher_diag and pretrain_weights in self.state.
        """
        loader = self.factory.get_loader(
            'val_seen', mode='batch',
            batch_size  = self.config.batch_size,
            num_workers = self.config.num_workers
        )
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        n_batches = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            logits = self.model(imgs, mode='linear')
            loss   = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1
        for n in fisher:
            fisher[n] /= max(n_batches, 1)
        self.state.fisher_diag     = fisher
        self.state.pretrain_weights = {n: p.detach().clone()
                                       for n, p in self.model.named_parameters()
                                       if p.requires_grad}

    def _ewc_penalty(self) -> torch.Tensor:
        """EWC regularisation penalty — sum of Fisher-weighted squared weight drift."""
        penalty = torch.tensor(0.0, device=self.device)
        if self.state.fisher_diag is None:
            return penalty
        for n, p in self.model.named_parameters():
            if n in self.state.fisher_diag:
                penalty = penalty + (
                    self.state.fisher_diag[n] *
                    (p - self.state.pretrain_weights[n]) ** 2
                ).sum()
        return penalty

    def _compute_kl_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between current batch embedding distribution and
        pretrain reference distribution (stored in state after pretrain).
        Closed-form KL between diagonal Gaussians:
            KL(N(mu1,var1) || N(mu2,var2))
        Returns scalar — 0 if reference stats not available.
        """
        if self.state.pretrain_emb_mean is None:
            return torch.tensor(0.0, device=self.device)
        mu1  = embeddings.mean(0)
        var1 = embeddings.var(0).clamp(min=1e-8)
        mu2  = self.state.pretrain_emb_mean
        var2 = self.state.pretrain_emb_var.clamp(min=1e-8)
        D    = mu1.shape[0]
        kl   = 0.5 * (
            (var1 / var2).sum() +
            ((mu2 - mu1) ** 2 / var2).sum() -
            D +
            (var2.log().sum() - var1.log().sum())
        )
        return kl

    def _load_pretrain_best(self):
        """
        Internal — called at end of pretrain() before returning to caller.

        1. Load best-pretrain-epoch weights into model.
        2. Keep checkpoint file — ExperimentRunner._cleanup() decides later.
        3. Set state.pretrain_export_path = path (always kept).
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

        # Load best weights into live model
        ModelFactory.load(self.model, path)
        self.model.to(self.device)

        # Always keep checkpoint — cleanup decision is ExperimentRunner's
        self.state.pretrain_export_path = path
        print(f"  Pretrain checkpoint: {path}")

        self.state.pretrain_best_val_loss = self.state.best_val_loss
        self.state.pretrain_best_val_acc  = self.state.best_val_acc


    def _load_train_best(self):
        """
        Internal — called at end of train_batch() / train_episodic() before returning.

        1. Load best-train-epoch weights into model.
        2. Keep checkpoint file — ExperimentRunner._cleanup() decides later.
        3. Set state.final_export_path = path (always kept).
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

        # Load best weights into live model
        ModelFactory.load(self.model, path)
        self.model.to(self.device)

        # Always keep checkpoint — cleanup decision is ExperimentRunner's
        self.state.final_export_path = path
        print(f"  Final model export: {path}")

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

    def pretrain(self, optuna_trial=None):
        """Phase 1 — batch pretrain, shared with FewShot."""
        self.impl.pretrain(optuna_trial=optuna_trial)

    def train(self, optuna_trial=None):
        """Phase 2a — batch training on seen classes."""
        self.impl.train_batch(val_pool='val_seen', optuna_trial=optuna_trial)

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

    def pretrain(self, optuna_trial=None):
        """Phase 1 — batch pretrain, shared with Standard."""
        self.impl.pretrain(optuna_trial=optuna_trial)

    def train(self, optuna_trial=None):
        """Phase 2b — episodic meta-training on base classes."""
        self.impl.train_episodic(val_pool='val_unseen', optuna_trial=optuna_trial)

    @property
    def state(self) -> TrainingState:
        return self.impl.state

    @property
    def history(self) -> TrainingHistory:
        return self.impl.history
