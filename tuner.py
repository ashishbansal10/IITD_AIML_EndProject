"""
tuner.py
========
Hyperparameter tuning via Optuna — learning objective implementation.

Purpose
-------
Demonstrates Optuna integration with the training pipeline.
Not a full HP search — 2 HPs × 2 values = 4 trials total.
Designed to show how Optuna connects to model, trainer, and experiment.

Tuned HPs
---------
Model HP    : backbone dropout_rate → [0.0, 0.2]
              0.0 = no dropout (baseline)
              0.2 = with dropout (regularized)

Training HP : lr → [1e-4, 1e-3]
              1e-4 = low lr (slower, more stable)
              1e-3 = standard lr (faster convergence)

Total trials: 2 × 2 = 4 (full grid)

Objective
---------
Proxy: pretrain phase only — val_loss after pretrain.
Full train too expensive per trial.
Assumption: backbone that pretrains well generalizes well.
Best HPs from pretrain used for full training in ExperimentRunner.

LR and Scheduler
----------------
Optuna picks starting lr before training.
Scheduler (step/cosine) decays from that starting lr during training.
No conflict — they operate at different levels:
    Optuna  : picks lr = 1e-3
    Scheduler: 1e-3 → 5e-4 → 2.5e-4 (step decay each N epochs)

Framework Integration Notes
----------------------------
[CONFIRMED] Optuna integrated here for learning purposes.
[CONFIRMED] hydra  — cut, stub in ModelConfig.from_yaml()
[CONFIRMED] torch.fx — cut, stub in CompositeModel._execute_graph()
[CONFIRMED] Lightning — pretrain phase, LightningModuleWrapper
[CONFIRMED] Optuna — tuner.py, 4-trial grid

Required Libraries
------------------
# optuna>=3.0.0  — pip install optuna
"""

import copy
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# ==============================================================================
# TuneConfig
# ==============================================================================

@dataclass
class TuneConfig:
    """
    Hyperparameter tuning configuration.
    Kept minimal — learning objective, not production search.

    Search space (categorical — discrete choices, easy to reason about):
        dropout_rate : [0.0, 0.2]    backbone dropout
        lr           : [1e-4, 1e-3]  starting learning rate

    Storage:
        None              → in-memory study (lost after run)
        'sqlite:///hp.db' → persistent study (resume across runs)

    Pruning:
        True  → Optuna stops bad trials early (MedianPruner)
                saves time when a trial is clearly worse than median
        False → all trials run to completion

    [FUTURE] Extend search space here when more HPs needed.
             Add float ranges: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
             Add integers:     trial.suggest_int('n_layers', 2, 5)
    """

    # Trial control
    n_trials:         int            = 4        # 2×2 full grid
    study_name:       str            = 'hp_search'
    storage:          Optional[str]  = None     # None=memory, 'sqlite:///hp.db'=persistent
    pruning:          bool           = True

    # Search space — categorical choices only
    # dropout_rate: with or without dropout
    dropout_choices:  List[float]    = field(default_factory=lambda: [0.0, 0.2])
    # lr: low or standard starting lr
    lr_choices:       List[float]    = field(default_factory=lambda: [1e-4, 1e-3])

    proxy_epochs: Optional[int] = None   # None = use default max(10, epochs//5)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TuneConfig':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ==============================================================================
# HPTuner
# ==============================================================================

class HPTuner:
    """
    Optuna wrapper — finds best dropout_rate and lr via 4-trial grid.

    Flow per trial:
        1. Optuna samples dropout_rate from [0.0, 0.2]
        2. Optuna samples lr           from [1e-4, 1e-3]
        3. Apply dropout_rate to backbone via component.set_hp()
        4. Apply lr to a fresh TrainConfig copy
        5. Run pretrain phase only (proxy for full training)
        6. Return best_val_loss as Optuna objective
        7. Optuna picks best trial after n_trials

    Usage:
        tuner    = HPTuner(model_config, train_config, tune_config,
                           factory, device)
        best_hps = tuner.run()
        # best_hps = {'dropout_rate': 0.2, 'lr': 1e-3}

        # Apply best HPs before full training
        model.get_component('backbone').set_hp(
            dropout_rate=best_hps['dropout_rate']
        )
        train_config.lr = best_hps['lr']

    Note:
        Each trial creates a fresh model copy — original model untouched.
        After tuner.run() completes, caller applies best HPs to original model.
    """

    def __init__(self,
                 model_config,
                 train_config,
                 tune_config:  TuneConfig,
                 factory,
                 device:       torch.device):
        """
        Args:
            model_config : ModelConfig — architecture definition
            train_config : TrainConfig — base training config (copied per trial)
            tune_config  : TuneConfig  — search space + trial control
            factory      : SmartDataLoaderFactory
            device       : torch.device — from notebook
        """
        self.model_config = model_config
        self.train_config = train_config
        self.tune_config  = tune_config
        self.factory      = factory
        self.device       = device

        # Best HPs found — populated after run()
        self.best_hps:    Dict[str, Any] = {}
        self.best_trial:  Optional[Any]  = None

        try:
            import optuna
            self._optuna = optuna
        except ImportError:
            raise ImportError(
                "optuna required for HPTuner.\n"
                "pip install optuna\n"
                "Or set tune_config=None in ExperimentConfig to skip tuning."
            )

    def run(self) -> Dict[str, Any]:
        """
        Run Optuna study — n_trials total.
        Returns best HPs found as dict.

        Returns:
            {'dropout_rate': float, 'lr': float}
        """
        optuna = self._optuna

        # Pruner — stops bad trials early based on median performance
        pruner = (optuna.pruners.MedianPruner()
                  if self.tune_config.pruning
                  else optuna.pruners.NopPruner())

        # Create or load study
        study = optuna.create_study(
            study_name = self.tune_config.study_name,
            storage    = self.tune_config.storage,
            direction  = 'minimize',    # minimize val_loss
            pruner     = pruner,
            load_if_exists = True       # resume if storage set + study exists
        )

        print(f"\n{'='*60}")
        print(f"HP TUNING — {self.tune_config.n_trials} trials")
        print(f"Search space:")
        print(f"  dropout_rate : {self.tune_config.dropout_choices}")
        print(f"  lr           : {self.tune_config.lr_choices}")
        print(f"Objective      : pretrain val_loss (proxy)")
        print(f"{'='*60}")

        study.optimize(
            self._objective,
            n_trials  = self.tune_config.n_trials,
            callbacks = [self._trial_callback]
        )

        self.best_trial = study.best_trial
        self.best_hps   = study.best_trial.params

        print(f"\n  Best trial : {study.best_trial.number}")
        print(f"  Best val_loss : {study.best_trial.value:.4f}")
        print(f"  Best HPs   : {self.best_hps}")

        return self.best_hps

    # ------------------------------------------------------------------
    # Objective — one trial
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        """
        Single Optuna trial.
        Samples HPs → applies to fresh model copy → runs pretrain → returns val_loss.

        Fresh model per trial — original model untouched.
        TrainConfig copied per trial — original config untouched.
        """
        from model_factory import ModelFactory
        from trainer       import StandardTrainer

        # ── Sample HPs ────────────────────────────────────────────────
        dropout_rate = trial.suggest_categorical(
            'dropout_rate', self.tune_config.dropout_choices
        )
        lr = trial.suggest_categorical(
            'lr', self.tune_config.lr_choices
        )

        print(f"\n  Trial {trial.number} | dropout_rate={dropout_rate} lr={lr}")

        # ── Fresh model copy per trial ─────────────────────────────────
        # Original model_config untouched — copy with overridden dropout
        trial_model_config = self._build_trial_model_config(dropout_rate)
        trial_model        = ModelFactory.create(trial_model_config,
                                                  device=self.device)

        # ── Fresh TrainConfig copy per trial ──────────────────────────
        trial_train_config      = copy.deepcopy(self.train_config)
        trial_train_config.lr   = lr
        # Shorten pretrain for tuning — proxy, not full train
        # Use 20% of full epochs — enough signal for relative comparison
        if self.tune_config.proxy_epochs is not None:
            trial_train_config.epochs_pretrain = self.tune_config.proxy_epochs
        else:
            trial_train_config.epochs_pretrain = max(
                10,
                self.train_config.epochs_pretrain // 5
            )

        # ── Run pretrain as proxy ─────────────────────────────────────
        # StandardTrainer used for both paradigms during tuning —
        # pretrain is identical for both (batch mode)
        trainer = StandardTrainer(
            trial_model, self.factory, trial_train_config, self.device
        )
        trainer.pretrain()

        val_loss = trainer.impl.state.best_val_loss

        # ── Report intermediate value for pruning ─────────────────────
        trial.report(val_loss, step=trial_train_config.epochs_pretrain)
        if trial.should_prune():
            raise self._optuna.exceptions.TrialPruned()

        return val_loss

    def _build_trial_model_config(self, dropout_rate: float):
        """
        Builds a ModelConfig copy with dropout_rate applied to backbone.
        Original model_config untouched.

        Deep copies the config dict, overrides backbone dropout_rate.
        Works for cnn_config, gnn_config, hybrid_config — all have 'backbone'.
        """
        cfg_dict = self.model_config.to_dict()
        cfg_dict = copy.deepcopy(cfg_dict)

        backbone = cfg_dict['components']['backbone']

        if isinstance(backbone, list):
            # SubChain (hybrid) — apply to first member (CNN backbone)
            backbone[0]['dropout_rate'] = dropout_rate
        elif isinstance(backbone, dict):
            # Single backbone (cnn, gnn)
            backbone['dropout_rate'] = dropout_rate

        from model_factory import ModelConfig
        return ModelConfig.from_dict(cfg_dict)

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _trial_callback(self, study, trial):
        """Prints trial summary after each trial completes."""
        print(f"  Trial {trial.number} complete | "
              f"val_loss={trial.value:.4f} | "
              f"params={trial.params} | "
              f"best so far={study.best_value:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Prints full tuning results after run() completes."""
        if not self.best_trial:
            print("No trials completed yet. Call run() first.")
            return

        print(f"\n{'='*60}")
        print(f"TUNING SUMMARY — {self.tune_config.study_name}")
        print(f"{'='*60}")
        print(f"  Best trial    : {self.best_trial.number}")
        print(f"  Best val_loss : {self.best_trial.value:.4f}")
        print(f"  Best HPs:")
        for k, v in self.best_hps.items():
            print(f"      {k:20s} : {v}")
        print(f"{'='*60}")
