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

TRAINER JOINT LOSS SUITE: LCA & RKD IMPLEMENTATION

This suite provides a modular framework for Joint Loss regularization during
few-shot episodic training. It aims to preserve the 'Feature Memory' of a 
pre-trained ResNet12 backbone.

Techniques:
1. Latent Centroid Anchoring (LCA): 
   Statistical preservation of class-specific distributions using KL-Divergence.
   Requires pre-computed mean/variance for the 64 base classes.

2. Relational Knowledge Distillation (RKD):
   Structural preservation of batch topology (distances between samples) using
   a frozen Teacher backbone.

Hyperparameters (TrainConfig):
- joint_loss_alpha_lca: Weight for statistical anchoring.
- joint_loss_alpha_rkd: Weight for topological preservation.
- joint_loss_temp: Temperature scaling for RKD distance matrices.
- joint_loss_lca_var: Softness constant (variance) for student LCA distribution.

Required Libraries
------------------
# torch>=2.0.0
# tqdm>=4.0.0               # train phase — pip install tqdm

"""

import os
import time
import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


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
    epochs_pretrain:      int   = 100
    epochs_train:         int   = 100
    lr:                   float = 5e-4

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler:            str   = 'cosine'     # 'step', 'cosine', 'none'
    lr_decay_step:        int   = 20
    lr_decay_gamma:       float = 0.5

    # ── Regularization ────────────────────────────────────────────────
    weight_decay:         float = 1e-4              # ← L2 regularisation — penalty on weight magnitude
    label_smoothing:      float = 0.0
    grad_clip:            Optional[float] = None    # None = disabled

    # ── Early stopping ────────────────────────────────────────────────
    early_stop_patience:  int   = 25
    early_stop_metric:    str   = 'val_loss'   # 'val_loss' or 'val_acc'

    # ── Optimizer ─────────────────────────────────────────────────────
    # Per-component lr override for trainable_param_groups
    # e.g. {'backbone': 1e-4, 'linear': 1e-3}
    lr_map:               Optional[Dict[str, float]] = None

    # ── Train phase improvements (all default to disabled = current behaviour) ──
    freeze_n_epochs:  int   = 0      # freeze backbone's first N train epochs — 0 = disabled
    warm_start:       bool  = True   # init train early-stop from pretrain best using zero shot anchor validation — False = reset


    # ── Joint loss regularization (LCA + RKD) ─────────────────────────

    # Joint Loss Alpha "Dials" (0.0 = Off)

    #   joint_loss_alpha_lca: [0.1 - 1.0] 
    #   -> Weight for statistical identity. Start at 0.5; higher values enforce stricter class-identity preservation.
    joint_loss_alpha_lca: float = 0.0    

    #   joint_loss_alpha_rkd: [0.1 - 10.0 | 5.0 being optimum as found by tuner] 
    #   -> Weight for structural topology. Distills relative distances between samples.
    #       Highly effective for GAT relational learning.
    joint_loss_alpha_rkd: float = 0.0    

    # Stability Hyperparameters

    # joint_loss_temp: [1.0 - 5.0] 
    #   -> RKD Temperature. Higher values (e.g., 2.0) smooth the distance matrix,
    #       making the topology more flexible for novel tasks.
    joint_loss_temp:      float = 1.0    

    # joint_loss_lca_var: [0.01 - 0.2] 
    #   -> Student neighborhood variance. Lower values (0.01) make the student 
    #      anchoring "sharp"; higher values (0.1) allow more feature exploration.
    joint_loss_lca_var:   float = 0.1


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
    total_steps_run:      int   = 0     # Added: Continuous step for Optuna tracking across phases

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

    # LCA Anchors: Only store the raw tensors for checkpointing 
    # (Latent Centroid Anchoring - class wide mean/variance for KL divergence)
    # Shape: [90, 640]
    class_means: Optional[torch.Tensor] = None 
    class_vars:  Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        # 1. Deep copy of the state into a dictionary
        d = asdict(self)
        
        # 2. List of keys that contain Tensors (Non-JSON serializable)
        tensor_keys = (
            'class_means',   # Added for LCA
            'class_vars'     # Added for LCA
        )
        
        # 3. Strip them out for the JSON-safe return
        for key in tensor_keys:
            d.pop(key, None)
            
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
# JointLossManager — LCA + RKD implementation
#   LCA: Statistical preservation of class-specific distributions using KL-Divergence.
#   RKD: Structural preservation of batch topology (distances between samples) using a frozen Teacher backbone.
# ==============================================================================

class JointLossManager:
    """
    Surgical Suite for Knowledge Retention via Hybrid Regularization (LCA + RKD).

    TERMINOLOGY:
    -----------
    - Teacher (T):  A frozen, deep-copied instance of the pre-trained backbone. 
                    It represents the 'Topological Gold Standard' of the 64-dim base feature space.
    - Student (S):  The active model instance (Backbone + GAT) within the Trainer 
                    that is currently being optimized for novel episodic tasks.

    LOSS FUNCTIONS & MECHANICS:
    --------------------------
    1. Latent Centroid Anchoring (LCA):
        - Goal: Preserves categorical 'Identity.'
        - Computation: Uses KL-Divergence to anchor Student embeddings to the 
           pre-trained Gaussian distribution (mu, var) of its original class.
        - mu and var are retrieved from the TrainingState's class_means and class_vars tensors,
          which are populated after pretraining by computing the mean and variance of the backbone's 
          feature space for each of the n_classes trained.
        - mu : [n_classes, 640] tensor of class means
        - var: [n_classes, 640] tensor of class variances
        - Formula: Loss_LCA = KL(Normal(mu_s, sigma^2_s) || Normal(mu_t, sigma^2_t))

    2. Relational Knowledge Distillation (RKD):
        - Goal: Preserves structural 'Topology.'
        - Computation: Calculates the Mean Squared Error between the normalized 
          pairwise distance matrices of the Student and Teacher embeddings.
        - Formula: Loss_RKD = MSE(D_{student}, D_{teacher}) 
          where D is a mean-normalized distance matrix.

    HYPERPARAMETERS:
    ---------------
    - joint_loss_alpha_lca: Weight for statistical anchoring.
    - joint_loss_alpha_rkd: Weight for topological preservation.
    - joint_loss_temp: Temperature scaling for RKD distance matrices.
    - joint_loss_lca_var: Softness constant (variance) for student LCA distribution.

    TOTAL OBJECTIVE:
    ---------------
    The combined auxiliary loss is integrated into the episodic optimization:
    Loss_total = Loss_task + joint_loss_alpha_lca * Loss_LCA + joint_loss_alpha_rkd * Loss_RKD

    SYSTEM INTERACTION:
    ------------------
    - TrainConfig: Provides the 'Mixing Board' (alphas) and stability hyperparameters 
      (temperature, student variance).
    - TrainingState: Acts as the data persistence layer. It stores the serializable 
      [n_classes, 640] anchor tensors (means/vars) for LCA.
    - TrainingImpl: The orchestration layer. It instantiates the Manager, triggers 
      'initialize()' to clone the Teacher, and delegates 'compute_loss()' during 
      the training loop.

    Note: To save VRAM, the Teacher is only the Backbone component, not the full model.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.teacher_backbone = None

    def initialize(self, model):
        """
        Clones the backbone component for RKD teacher-student distillation.
        Uses factory-pattern 'get_component' and custom 'freeze' API.
        """
        if self.config.joint_loss_alpha_rkd > 0:
            # Safely extract backbone via factory method
            backbone_source = model.get_component('backbone')
            
            # Deepcopy to create a distinct Teacher instance
            self.teacher_backbone = copy.deepcopy(backbone_source)
            
            # Use your custom API to disable gradients
            self.teacher_backbone.freeze() 
            
            # Ensure it's in eval mode for inference stability
            self.teacher_backbone.eval()
            self.teacher_backbone.to(self.device)

    def compute_loss(self,
                     images:      torch.Tensor,    # [B, 3, H, W]
                     embeddings:  torch.Tensor,    # [B, 640]
                     state_means: torch.Tensor, 
                     state_vars: torch.Tensor, 
                     targets: torch.Tensor) -> torch.Tensor:
        """
        Aggregates active joint losses based on alpha configuration.

        Args:
            embeddings (torch.Tensor): Active student features $[B, 640]$ currently being optimized.
            state_means (torch.Tensor): Global class centroids $[90, 640]$ retrieved from pre-training.
            state_vars (torch.Tensor): Global class variances $[90, 640]$ defining anchor distribution width.
            targets (torch.Tensor): Original global class indices (0-89) used to index class-specific anchors.

        Returns:
            torch.Tensor: Weighted sum of LCA and RKD auxiliary losses.
        """
        loss = torch.tensor(0.0, device=self.device)

        # 1. Latent Centroid Anchoring (LCA)
        if self.config.joint_loss_alpha_lca > 0 and targets is not None:
            # Identify samples with non-zero anchors (variances were clamped to 1e-6)
            # targets indices the [N, D] tensor; sum across D to check for presence
            valid_mask = state_vars[targets].abs().sum(dim=1) > 0 

            if valid_mask.any():
                # Compute KL only on the subset of valid anchors
                kl_loss = self._lca_kl(
                    embeddings[valid_mask],
                    state_means[targets[valid_mask]],
                    state_vars[targets[valid_mask]]
                )
                loss += self.config.joint_loss_alpha_lca * kl_loss


        # 2. Relational Knowledge Distillation (RKD)
        if self.config.joint_loss_alpha_rkd > 0 and self.teacher_backbone is not None:
            loss += self.config.joint_loss_alpha_rkd * self._rkd_dist(images, embeddings)

        return loss

    def _lca_kl(self, embeddings, means, vars):
        """
        Calculates KL-Divergence using masked/pre-indexed tensors.
        
        Args:
            embeddings: Student features [Masked_Batch, Feat_Dim]
            means:      Teacher centroids [Masked_Batch, Feat_Dim]
            vars:       Teacher variances [Masked_Batch, Feat_Dim]
        """
        # Target Distribution (P): Frozen Teacher Anchors
        std_t = vars.sqrt()
        p = Normal(means, std_t) 
        
        # Student Distribution (Q): Active Features
        # Uses the 'softness' variance from config
        std_s = torch.ones_like(embeddings) * (self.config.joint_loss_lca_var ** 0.5) 
        q = Normal(embeddings, std_s) 
        
        # Returns the average KL divergence for the valid samples
        return kl_divergence(q, p).mean()

    def _rkd_dist(self, images, embeddings):
        """Preserves distance-wise topology using scale-invariant RKD."""
        with torch.no_grad():
            t_emb = self.teacher_backbone(images)

        # Compute Pairwise Euclidean Distance Matrices [B, B]
        # Entry [i, j] is the distance between sample i and sample j
        d_s = torch.cdist(embeddings, embeddings, p=2)
        d_t = torch.cdist(t_emb, t_emb, p=2)

        # Scale-Invariant Normalization (Mean = 1.0)
        # Mean-Normalization for Scale Invariance
        # Ensures Student is penalized for shape change, not absolute coordinate scale
        d_s_norm = d_s / (d_s.mean() * self.config.joint_loss_temp + 1e-7)
        d_t_norm = d_t / (d_t.mean() * self.config.joint_loss_temp + 1e-7)

        return F.mse_loss(d_s_norm, d_t_norm)


# ==============================================================================
# TrainerImpl — all actual training logic
# ==============================================================================

class TrainerImpl:
    """
    All actual training logic.
    Not used directly — accessed via StandardTrainer or FewShotTrainer.

    Uses JointLossManager for LCA + RKD regularization loss during episodic training.

    Backend dispatch:
        pretrain()        → _pretrain_pytorch()
        load_pretrain()
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
        _check_and_update_best()    — val_loss improvement check and update
        _early_stopping_check()     — returns True if should stop
        _log_epoch()                — prints epoch metrics
        _save_checkpoint()          — via ModelFactory
        _load_pretrain_best()       — load best + handle file per pretrain_save_mode
        _load_train_best()          — load best + handle file per keep_final

    State and history public for Optuna:
        trainer.impl.state.best_val_loss   → Optuna objective
        trainer.impl.history               → for plotting
    """

    def __init__(self, model, factory, config:   TrainConfig, device:   torch.device,
                 paradigm: str, seed: int = 42):
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
        self._seed = seed

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
        self._phase_start_time: float = 0.0   # wall clock time per phase
        self._best_epoch:       int   = 0     # epoch where best checkpoint was saved

        # Joint loss manager for LCA + RKD regularization during episodic training
        self.joint_loss_manager = JointLossManager(config, device)

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

        # Restore best-pretrain-epoch weights before returning.
        # Caller (runner) receives model already at best state — no load_best() needed.
        self._load_pretrain_best()
        self.model.unfreeze('prototypical')
        self.model.to(self.device)

        # Initialize LCA Anchors
        # Compute anchors (reference stats) for train phase improvements (only if enabled)
        if self.config.joint_loss_alpha_lca > 0:
            # Compute per-class means and variances for LCA anchoring
            means, vars = self._compute_lca_stats()
            self.state.class_means = means
            self.state.class_vars  = vars


    def load_pretrain(self, path: str):
        """
        Gateway API: Loads pre-trained weights and initializes 

        path: str — checkpoint path to load from (must exist)
        """
        from model_factory import ModelFactory
        
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"load_pretrain: Checkpoint not found at {path}")

        # Load weights into model
        checkpoint = ModelFactory.load(self.model, path)
        meta = checkpoint.get('metadata', {})

        # Rehydrate Historical Metrics and Flags
        self.state.pretrain_best_val_loss = meta.get('val_loss', float('inf'))
        self.state.pretrain_best_val_acc  = meta.get('val_acc', 0.0)
        self.state.pretrain_export_path   = path
        self.state.is_pretrained          = True

        # Reset Phase 1 masks to 'Neutral'
        self.model.unfreeze('prototypical')
        self.model.to(self.device)

        # Initialize LCA Anchors if missing
        if self.config.joint_loss_alpha_lca > 0 and self.state.class_means is None:
            means, vars = self._compute_lca_stats()
            self.state.class_means, self.state.class_vars = means, vars

        print(f"\n  Phase 1: Pretrain weights loaded from {path}")


    def train_batch(self, val_pool: str = 'val_seen', optuna_trial=None):
        """
        Phase 2a — standard batch training.
        Called by StandardTrainer.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        print(f"\n  Phase 2: Train [standard | pytorch | {self.config.epochs_train} epochs]")
        
        self.joint_loss_manager.initialize(self.model) # Snapshot Teacher

        self.model.unfreeze_all()
        self.model.freeze('prototypical')

        self._run_train_pytorch(
            train_pool = 'train',
            val_pool   = val_pool,
            episodic   = False,
            optuna_trial=optuna_trial
        )

        # Restore best-train-epoch weights before returning.
        self._load_train_best(episodic=False)
        self.model.unfreeze('prototypical')
        self.model.to(self.device)


    def train_episodic(self, val_pool: str = 'val_unseen', optuna_trial=None):
        """
        Phase 2b — episodic meta-training.
        Called by FewShotTrainer.
        Freezes linear head before training — only backbone trains.
        Unfreezes after for evaluation.
        """
        self.state.reset_early_stop()
        self.state.epoch = 0

        print(f"\n  Phase 2: Train [fewshot | episodic | pytorch | {self.config.episodes_train} eps/epoch | {self.config.epochs_train} epochs]")

        self.joint_loss_manager.initialize(self.model) # Snapshot Teacher

        # Ensure prototypical path unfrozen for episodic training 
        # Since prototypical path is frozen during pretrain, and checkpointed with frozen names 
        # Must unfreeze here to allow episodic training to update backbone via prototypical loss
        self.model.unfreeze_all()

        # Freeze linear head — episodic training does not update it
        self.model.freeze('linear')
        self.model.freeze('softmax')

        self._run_train_pytorch(
            train_pool = 'train',
            val_pool   = val_pool,
            episodic   = True,
            optuna_trial=optuna_trial
        )

        # Restore best-train-epoch weights before returning.
        # Note: checkpoint was saved with linear frozen. ModelFactory.load()
        # restores frozen_names from checkpoint — linear will be frozen again.
        # This is correct: fewshot eval uses prototypical path, not linear.
        # Softmax eval uses pretrain-era linear weights — intentional diagnostic.
        self._load_train_best(episodic=True)
        self.model.unfreeze('linear')
        self.model.unfreeze('softmax')
        self.model.to(self.device)


    # ------------------------------------------------------------------
    # PyTorch — pretrain
    # ------------------------------------------------------------------

    def _pretrain_pytorch(self, optuna_trial=None):
        """Batch pretrain loop — pure PyTorch."""
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer, self.config.epochs_pretrain)

        pretrain_loader = self.factory.get_loader(
            'pretrain', mode='batch',
            batch_size = self.config.batch_size,
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
                report_val = val_loss if self.config.early_stop_metric == 'val_loss' else val_acc
                optuna_trial.report(report_val, step=self.state.total_steps_run)
                if optuna_trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            # --- INSERTED CODE END ---

            # Checkpoint on improvement
            if self._check_and_update_best(val_loss, val_acc):
                path = os.path.join( self.config.checkpoint_dir, f"{self.config.run_id}_pretrain_best.pt" )
                self._save_checkpoint(path)
                self._pretrain_best_path = path
                self._best_epoch = epoch
                self.state.early_stop_counter = 0
            else:
                self.state.early_stop_counter += 1

            self.state.total_steps_run += 1

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

        # ── THE ZERO-SHOT ANCHOR for warm_start ──────────────────────
        if self.config.warm_start:
            if self.config.verbose:
                mode_label = 'episodic' if episodic else 'batch'
                print(f"  Establishing Zero-Shot Anchor via {mode_label} validation...")
            
            # Run ONE full validation pass to establish the REAL baseline
            v_loss, v_acc = self._episodic_epoch(loader=val_loader, optimizer=None, is_train=False) if episodic else \
                            self._batch_epoch(loader=val_loader, optimizer=None, is_train=False)

            # Calibrate best metrics to this actual starting state
            self.state.best_val_loss = v_loss
            self.state.best_val_acc  = v_acc
            
            print(f"  warm_start Anchor set: val_loss={v_loss:.4f} val_acc={v_acc:.4f}")
        # ─────────────────────────────────────────────────────────────

        mode_str = 'episodic' if episodic else 'batch'
        unit_str = f"{self.config.episodes_train} eps/epoch" if episodic else "batch"

        self._phase_start_time = time.time()
        self._best_epoch = 0
        epochs_run = 0

        prev_val_loss = float('inf')   # ADD — tracks prev best for warm_start early stop
        prev_val_acc  = 0.0            # ADD — tracks prev best for val_acc metric


        for epoch in range(self.config.epochs_train):
            self.state.epoch = epoch
            epochs_run = epoch + 1

            # freeze_n_epochs: freeze backbone for first N epochs, unfreeze after
            if self.config.freeze_n_epochs > 0:
                if epoch == 0:
                    self.model.freeze('backbone')
                elif epoch == self.config.freeze_n_epochs:
                    self.model.unfreeze('backbone')

            if episodic:
                # Set epoch for EpisodicBatchSampler RNG variation
                if hasattr(train_loader.batch_sampler, 'set_epoch'):
                    train_loader.batch_sampler.set_epoch(epoch)

                train_loss, train_acc = self._episodic_epoch(loader=train_loader, optimizer=optimizer, is_train=True)
                val_loss, val_acc = self._episodic_epoch(loader=val_loader, optimizer=None, is_train=False)
            else:
                train_loss, train_acc = self._batch_epoch(loader=train_loader, optimizer=optimizer, is_train=True)
                val_loss, val_acc = self._batch_epoch(loader=val_loader, optimizer=None, is_train=False)

            if scheduler is not None:
                scheduler.step()

            self.history.log_train(epoch, train_loss, train_acc, val_loss, val_acc)
            if self.config.verbose:
                self._log_epoch('train', epoch, train_loss, train_acc, val_loss, val_acc)

            # --- INSERTED CODE START for Optuna ---
            if optuna_trial is not None:
                report_val = val_loss if self.config.early_stop_metric == 'val_loss' else val_acc
                optuna_trial.report(report_val, step=self.state.total_steps_run)
                if optuna_trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            # --- INSERTED CODE END ---

            # Checkpoint on improvement
            if self._check_and_update_best(val_loss, val_acc):
                prev_val_loss = val_loss            # ADD — sync prev with new best
                prev_val_acc  = val_acc             # ADD — sync prev with new best
                path = os.path.join(self.config.checkpoint_dir, f"{self.config.run_id}_train_best.pt")
                self._save_checkpoint(path)
                self._train_best_path = path
                self._best_epoch = epoch
                self.state.early_stop_counter   = 0
            else:
                # Only count when not improving vs prev AND above best floor
                # Allows model to freely descend toward best without patience penalty
                if not self._is_better(val_loss, val_acc, prev_val_loss, prev_val_acc):
                    self.state.early_stop_counter += 1
                else:
                    prev_val_loss = val_loss    # ADD — update prev if actually better
                    prev_val_acc  = val_acc     # ADD — update prev if actually better

            self.state.total_steps_run += 1

            if self._early_stopping_check():
                if self.config.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # Ensure backbone unfrozen after train loop (in case freeze_n >= epochs_train)
        if self.config.freeze_n_epochs > 0:
            self.model.unfreeze('backbone')

        elapsed = (time.time() - self._phase_start_time) / 60
        print(f"  Train  — ran {epochs_run}/{self.config.epochs_train} epochs  "
              f"best @ epoch {self._best_epoch}  "
              f"val_loss={self.state.best_val_loss:.2f}  val_acc={self.state.best_val_acc:.2f}  "
              f"time={elapsed:.1f}min")

    # ------------------------------------------------------------------
    # PyTorch — single epoch loops
    # ------------------------------------------------------------------

    def _batch_epoch(self, loader, optimizer, is_train: bool) -> Tuple[float, float]:
        """
        Single batch epoch — train or eval.
        Returns (avg_loss, avg_acc).
        """
        should_optimize = is_train and self.model.is_trainable

        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = 0

        ctx = torch.enable_grad() if should_optimize else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, leave=False, desc=f"  {'train' if is_train else 'val  '}", disable=not self.config.verbose)
            for imgs, labels in pbar:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                # Forward — raw logits
                logits = self.model(imgs, mode='linear')

                # Loss — CrossEntropyLoss on raw logits
                # NEVER pass softmax output here — double softmax = wrong gradients
                loss = self._criterion(logits, labels)

                if should_optimize:
                    if self.config.joint_loss_alpha_lca > 0 or self.config.joint_loss_alpha_rkd > 0:
                        emb = self.model(imgs, mode='embedding')
                        loss += self.joint_loss_manager.compute_loss(
                            images=imgs,
                            embeddings=emb,
                            state_means=self.state.class_means,
                            state_vars=self.state.class_vars,
                            targets=labels
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    optimizer.step()

                acc = (logits.argmax(1) == labels).float().mean().item()
                total_loss += loss.item()
                total_acc  += acc
                n_batches  += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc':  f'{acc:.4f}'})

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


    def _episodic_epoch(self, loader, optimizer, is_train: bool) -> Tuple[float, float]:
        """
        Single episodic epoch — train or eval.
        Each batch is a TaskCollator dict {support, query, target}.
        Returns (avg_loss, avg_acc).
        """
        should_optimize = is_train and self.model.is_trainable

        self.model.train() if is_train else self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_episodes = 0

        ctx = torch.enable_grad() if should_optimize else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, leave=False, desc=f"  {'train' if is_train else 'val  '}", disable=not self.config.verbose)
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
                dists = self.model(support_emb=s_emb, query_emb=q_emb, mode='prototypical')   # [N*Q, N]

                loss = self._criterion(dists, target)

                if should_optimize:
                    if self.config.joint_loss_alpha_lca > 0 or self.config.joint_loss_alpha_rkd > 0:
                        # Retrieve the aligned global IDs from our new Collator logic
                        global_ids = batch.get('targets_global', None)

                        # Compute regularized loss
                        # Manager handles the 'Masked Drop' of sparse classes internally
                        loss += self.joint_loss_manager.compute_loss(
                            images      = episode,
                            embeddings  = all_emb, 
                            state_means = self.state.class_means, 
                            state_vars  = self.state.class_vars, 
                            targets     = global_ids.to(self.device) if global_ids is not None else None
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
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
    def _is_better(self, val_loss, val_acc, ref_loss, ref_acc) -> bool:
        """Pure comparison against explicit reference — no state update."""
        if self.config.early_stop_metric == 'val_loss':
            return val_loss < ref_loss
        else:
            return val_acc > ref_acc


    def _check_and_update_best(self, val_loss: float, val_acc: float) -> bool:
        """
        Compare against current best via _is_better, update state if improved.
        Returns True if improved — caller resets counter and saves checkpoint.
        """
        if self._is_better(val_loss, val_acc, self.state.best_val_loss, self.state.best_val_acc):
            self.state.best_val_loss = val_loss
            self.state.best_val_acc  = val_acc
            return True
        return False


    def _early_stopping_check(self) -> bool:
        """Returns True if training should stop."""
        if self.state.early_stop_counter >= self.config.early_stop_patience:
            self.state.should_stop = True
            return True
        return False

    def _save_checkpoint(self, path: str):
        """
        Internal — save full model during training loop on improvement.
        Metadata is passed to ModelFactory.save for Phase 2 rehydration.
        """
        from model_factory import ModelFactory

        metadata = {
            'val_loss': self.state.best_val_loss,
            'val_acc':  self.state.best_val_acc,
            'epoch':    self.state.epoch
        }
        ModelFactory.save(self.model, path, metadata=metadata)

    # ------------------------------------------------------------------
    # Train phase improvement helpers
    # ------------------------------------------------------------------

    def _compute_lca_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes LCA anchors using a 1:1 balanced ratio of Pretrain and Val_seen data.
        Dynamically infers feature dimensions and uses self._seed for reproducibility.
        """
        self.model.eval()
        class_data = defaultdict(list)
        
        # 1. Collect features from both base-class pools
        # pretrain = 40% (memorization), val_seen = 20% (generalization)
        for pool in ['pretrain', 'val_seen']:
            loader = self.factory.get_loader(pool, mode='batch', shuffle=False)
            with torch.no_grad():
                for imgs, labels in loader:
                    emb = self.model(imgs.to(self.device), mode='embedding')
                    # Move to CPU immediately to preserve A100 VRAM
                    for i in range(len(labels)):
                        class_data[labels[i].item()].append((emb[i].cpu(), pool))

        # 2. Dynamic Architecture Discovery
        if not class_data:
            raise RuntimeError("LCA Stats: No data collected from pretrain/val_seen pools.")

        # Infer feat_dim from the first collected embedding
        any_class_samples = next(iter(class_data.values()))
        feat_dim = any_class_samples[0][0].shape[0] 
        
        # Determine max class index (supports 90 classes or more)
        num_classes = int(max(class_data.keys()) + 1)

        means = torch.zeros(num_classes, feat_dim)
        vars = torch.zeros(num_classes, feat_dim)
        
        # Seeded RNG for reproducible sub-sampling
        rng = random.Random(self._seed)

        # 3. 1:1 Balanced Computation
        active_anchors = 0
        for c, samples in class_data.items():
            # Drop strategy: skip classes with total samples < 5
            if len(samples) < 5:
                continue

            p_samples = [s[0] for s in samples if s[1] == 'pretrain']
            v_samples = [s[0] for s in samples if s[1] == 'val_seen']
            
            # Determine the pivot size (usually limited by the smaller val_seen pool)
            min_size = min(len(p_samples), len(v_samples))
            
            if min_size > 0:
                # Sub-sample pretrain to match val_seen count (ensures 1:1 weight)
                rng.shuffle(p_samples)
                balanced_feat = torch.stack(v_samples + p_samples[:min_size])
            else:
                # Fallback for edge cases where a class is missing from one pool
                balanced_feat = torch.stack([s[0] for s in samples])

            # Calculate Gaussian parameters for the class
            means[c] = balanced_feat.mean(0)
            vars[c] = balanced_feat.var(0).clamp(min=1e-6) # Numerical stability guard
            active_anchors += 1

        print(f"  LCA: Computed {active_anchors}/{num_classes} valid anchors (Dropped {num_classes-active_anchors} sparse classes)")
        return means.to(self.device), vars.to(self.device)


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

        # Always keep checkpoint — cleanup decision is ExperimentRunner's
        # Load best weights and sync metrics from metadata
        checkpoint = ModelFactory.load(self.model, path)
        meta = checkpoint.get('metadata', {})

        self.state.pretrain_best_val_loss = meta.get('val_loss', self.state.best_val_loss)
        self.state.pretrain_best_val_acc  = meta.get('val_acc', self.state.best_val_acc)
        self.state.pretrain_export_path   = path
        self.state.is_pretrained          = True
        print(f"  Pretrain best loaded: {path} | acc: {self.state.pretrain_best_val_acc:.4f}")


    def _load_train_best(self, episodic: bool):
        """
        Internal — called at end of train_batch() / train_episodic() before returning.

        1. Load best-train-epoch weights into model.
        2. Keep checkpoint file — ExperimentRunner._cleanup() decides later.
        3. Set state.final_export_path = path (always kept).

        episodic : False = batch  (prototypical frozen during train)
                   True  = episodic (linear + softmax frozen during train)

        Fallback — if no train checkpoint was saved (warm_start=True and model never
        beat pretrain anchor), loads pretrain best instead and restores exact frozen state
        matching the train mode — so caller unfreeze calls behave identically.
        Guarantees: final model is always at least pretrain quality.
        """
        from model_factory import ModelFactory
        path = self._train_best_path

        if not path or not os.path.exists(path):
            # No train checkpoint saved — model never beat pretrain anchor
            # Fallback to pretrain best rather than leaving model at last epoch
            if self._pretrain_best_path and os.path.exists(self._pretrain_best_path):
                print(
                    f"  No train checkpoint saved — falling back to pretrain best "
                    f"(warm_start floor: acc={self.state.pretrain_best_val_acc:.4f})"
                )
                checkpoint = ModelFactory.load(self.model, self._pretrain_best_path)
                meta = checkpoint.get('metadata', {})
                self.state.best_val_loss     = meta.get('val_loss', self.state.pretrain_best_val_loss)
                self.state.best_val_acc      = meta.get('val_acc',  self.state.pretrain_best_val_acc)
                self.state.final_export_path = self._pretrain_best_path
                self.state.is_trained        = True

                # Restore frozen state matching train mode —
                # caller's unfreeze calls after return behave identically
                self.model.unfreeze_all()
                if episodic:
                    self.model.freeze('linear')
                    self.model.freeze('softmax')
                else:
                    self.model.freeze('prototypical')

                print(f"  Pretrain fallback loaded: {self._pretrain_best_path} | acc: {self.state.best_val_acc:.4f}")
            else:
                print(
                    f"  Warning: neither train nor pretrain best checkpoint found. "
                    f"Model remains at last epoch."
                )
                self.state.final_export_path = ''
            return

        # Load best train weights into live model
        checkpoint = ModelFactory.load(self.model, path)
        meta = checkpoint.get('metadata', {})

        self.state.best_val_loss     = meta.get('val_loss', self.state.best_val_loss)
        self.state.best_val_acc      = meta.get('val_acc', self.state.best_val_acc)
        self.state.final_export_path = path
        self.state.is_trained        = True
        print(f"  Final model best loaded: {path} | acc: {self.state.best_val_acc:.4f}")

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
        trainer = StandardTrainer(model, factory, config, device, seed=42)
        trainer.pretrain()
        trainer.train()

        # Access state for Optuna
        best_loss = trainer.impl.state.best_val_loss

        # Access history for plotting
        history = trainer.impl.history
    """

    def __init__(self, model, factory, config: TrainConfig, device: torch.device, seed: int = 42):
        self.impl = TrainerImpl(
            model    = model,
            factory  = factory,
            config   = config,
            device   = device,
            paradigm = 'standard',
            seed     = seed
        )

    def pretrain(self, optuna_trial=None):
        """Phase 1 — batch pretrain, shared with FewShot."""
        self.impl.pretrain(optuna_trial=optuna_trial)

    def load_pretrain(self, path: str):
        """Gateway to Phase 2 readiness."""
        self.impl.load_pretrain(path)

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
        trainer = FewShotTrainer(model, factory, config, device, seed=42)
        trainer.pretrain()
        trainer.train()

        # Access state for Optuna
        best_loss = trainer.impl.state.best_val_loss
    """

    def __init__(self, model, factory, config: TrainConfig, device: torch.device, seed: int = 42):
        self.impl = TrainerImpl(
            model    = model,
            factory  = factory,
            config   = config,
            device   = device,
            paradigm = 'fewshot',
            seed     = seed
        )

    def pretrain(self, optuna_trial=None):
        """Phase 1 — batch pretrain, shared with Standard."""
        self.impl.pretrain(optuna_trial=optuna_trial)

    def load_pretrain(self, path: str):
        """Gateway to Phase 2 readiness."""
        self.impl.load_pretrain(path)

    def train(self, optuna_trial=None):
        """Phase 2b — episodic meta-training on base classes."""
        self.impl.train_episodic(val_pool='val_unseen', optuna_trial=optuna_trial)

    @property
    def state(self) -> TrainingState:
        return self.impl.state

    @property
    def history(self) -> TrainingHistory:
        return self.impl.history
