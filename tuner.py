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
[CONFIRMED] Optuna — tuner.py, 4-trial grid

Required Libraries
------------------
# optuna>=3.0.0  — pip install optuna
"""

import copy
import torch
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# ==============================================================================
# TuneConfig
# ==============================================================================

@dataclass
class TuneConfig:
    """
    Hyperparameter tuning configuration.

    User sets hp_choices and optionally model_hp_keys.
    Sampler, pruner, n_trials are auto-selected internally.

    hp_choices — {hp_name: [choice1, choice2, ...]}
        Pretrain: {'dropout_rate': [0.0, 0.1, 0.2], 'lr': [1e-4, 5e-4, 1e-3]}
        Train:    {'temperature': [5.0, 10.0, 20.0],
                   'label_smoothing': [0.0, 0.05, 0.1],
                   'weight_decay': [1e-4, 5e-4]}

    model_hp_keys — hp names going to model components (not TrainConfig):
        'dropout_rate' → backbone.set_hp(dropout_rate=v)
        'temperature'  → prototypical head set_hp(temperature=v)
        All others     → setattr(train_config, key, value)

    Sampler auto-selected:
        search_space_size <= 12 → GridSampler  (exhaustive)
        search_space_size >  12 → TPESampler   (Bayesian)

    Pruner auto-selected from effective n_trials:
        <= 10 → NopPruner
        <= 30 → MedianPruner
        >  30 → HyperbandPruner

    n_trials auto-selected:
        GridSampler → product of all choice lengths
        TPESampler  → min(search_space_size * 2, 30) unless overridden

    proxy_epochs — short pretrain proxy for pretrain-phase tuning.
        None = max(10, epochs_pretrain // 5)
        Ignored for train-phase tuning (checkpoint reloaded instead).

    [FUTURE] Extend search space here when more HPs needed.
             Add float ranges: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
             Add integers:     trial.suggest_int('n_layers', 2, 5)
    """

    # ── Search space ──────────────────────────────────────────────────
    # model_hp_choices: aliased names matching *_config() hp_overrides kwargs
    #   e.g. {'backbone_dropout': [0.0, 0.1, 0.2], 'temperature': [5.0, 10.0]}
    #   Applied via ModelFactory.create(hp_overrides=model_sampled)
    #
    # train_hp_choices: TrainConfig field names
    #   e.g. {'label_smoothing': [0.0, 0.05, 0.1], 'weight_decay': [1e-4, 5e-4]}
    #   Applied via setattr(train_config, k, v)
    model_hp_choices: Dict[str, List[Any]] = field(default_factory=dict)
    train_hp_choices: Dict[str, List[Any]] = field(default_factory=dict)

    # ── Optional ──────────────────────────────────────────────────────
    n_trials:     Optional[int] = None   # None = auto from search space
    proxy_epochs: Optional[int] = None   # None = epochs_pretrain // 5
    study_name:   str           = 'hp_search'
    storage:      Optional[str] = None   # None=memory, path=persistent

    def search_space_size(self) -> int:
        """Product of all choice lengths across both model and train HPs."""
        size = 1
        for v in {**self.model_hp_choices, **self.train_hp_choices}.values():
            size *= len(v)
        return size

    def effective_n_trials(self) -> int:
        """Auto-compute n_trials if not set by user."""
        if self.n_trials is not None:
            return self.n_trials
        size = self.search_space_size()
        if size <= 12:
            return size
        return min(size * 2, 30)

    def all_hp_choices(self) -> Dict[str, List[Any]]:
        """Combined dict for Optuna sampling — all HPs together."""
        return {**self.model_hp_choices, **self.train_hp_choices}

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
                 tune_config:          TuneConfig,
                 factory,
                 device:               torch.device,
                 run_id:               str           = 'run',
                 paradigm:             str           = 'standard',
                 phase:                str           = 'pretrain',
                 logs_dir:             str           = 'logs',
                 load_checkpoint_path: Optional[str] = None):
        """
        Args:
            model_config          : ModelConfig — architecture definition
            train_config          : TrainConfig — base training config (copied per trial)
            tune_config           : TuneConfig  — search space + trial control
            factory               : SmartDataLoaderFactory
            device                : torch.device
            run_id                : str — run identifier for log filename
            paradigm              : 'standard' or 'fewshot' — selects trainer class
            phase                 : 'pretrain' or 'train' — selects tuning phase
            logs_dir              : str — directory for detailed log files
            load_checkpoint_path  : str or None
                                    None  = pretrain-phase tuning (runs proxy pretrain)
                                    path  = train-phase tuning (reloads checkpoint, skips pretrain)
        """
        self.model_config         = model_config
        self.train_config         = train_config
        self.tune_config          = tune_config
        self.factory              = factory
        self.device               = device
        self.run_id               = run_id
        self.paradigm             = paradigm
        self.phase                = phase
        self.load_checkpoint_path = load_checkpoint_path

        # Best HPs found — populated after run()
        self.best_hps:    Dict[str, Any] = {}
        self.best_trial:  Optional[Any]  = None

        # ── Logger — detailed output to file, minimal to stdout ───────

        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{run_id}_tuner_{self.phase}.log")

        self._logger = logging.getLogger(f"tuner.{run_id}.{self.phase}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self._logger.addHandler(fh)
        self._log_path = log_path

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna stdout
            self._optuna = optuna
        except ImportError:
            raise ImportError(
                "optuna required for HPTuner.\n"
                "pip install optuna\n"
                "Or set tune_config=None in ExperimentConfig to skip tuning."
            )

    def _select_sampler(self):
        """Auto-select sampler based on combined search space size."""
        size     = self.tune_config.search_space_size()
        all_hps  = self.tune_config.all_hp_choices()
        if size <= 12:
            self._logger.info(f"Sampler: GridSampler (space_size={size})")
            return self._optuna.samplers.GridSampler(all_hps)
        self._logger.info(f"Sampler: TPESampler (space_size={size})")
        return self._optuna.samplers.TPESampler()

    def _select_pruner(self, n_trials: int):
        """Auto-select pruner based on effective n_trials."""
        if n_trials <= 10:
            self._logger.info(f"Pruner: NopPruner (n_trials={n_trials})")
            return self._optuna.pruners.NopPruner()
        elif n_trials <= 30:
            self._logger.info(f"Pruner: MedianPruner (n_trials={n_trials})")
            return self._optuna.pruners.MedianPruner()
        self._logger.info(f"Pruner: HyperbandPruner (n_trials={n_trials})")
        return self._optuna.pruners.HyperbandPruner()

    def run(self) -> Dict[str, Any]:
        """
        Run Optuna study.
        Sampler, pruner, n_trials auto-selected from TuneConfig.
        Returns best HPs found as dict.
        """
        optuna   = self._optuna
        n_trials = self.tune_config.effective_n_trials()
        sampler  = self._select_sampler()
        pruner   = self._select_pruner(n_trials)

        study = optuna.create_study(
            study_name     = self.tune_config.study_name,
            storage        = self.tune_config.storage,
            direction      = 'minimize',    # minimize val_loss
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = True       # resume if storage set + study exists
        )

        print(f"  Tuner started [{self.run_id} | {self.phase} | {n_trials} trials | space={self.tune_config.search_space_size()}]")
        self._logger.info(f"Tuner started — run_id={self.run_id} phase={self.phase} n_trials={n_trials}")
        self._logger.info(f"Search space model: {self.tune_config.model_hp_choices}")
        self._logger.info(f"Search space train: {self.tune_config.train_hp_choices}")
        self._logger.info(f"Objective: {'train val_loss (reusing pretrain ckpt)' if self.load_checkpoint_path else 'pretrain val_loss (proxy)'}")

        study.optimize(
            self._objective,
            n_trials  = n_trials,
            callbacks = [self._trial_callback],
        )

        self.best_trial = study.best_trial
        self.best_hps   = study.best_trial.params

        self._logger.info(f"Tuner complete — best_trial={study.best_trial.number} best_value={study.best_trial.value:.4f} best_hps={self.best_hps}")
        print(f"  Tuner complete [{self.run_id}] — best_val={study.best_trial.value:.4f} hps={self.best_hps} log={self._log_path}")

        return self.best_hps

    def _objective(self, trial) -> float:
        """
        Single Optuna trial.

        Pretrain tuning (load_checkpoint_path=None):
            Samples HPs from hp_choices → runs proxy pretrain → returns val_loss

        Train tuning (load_checkpoint_path set):
            Samples HPs from hp_choices → reloads checkpoint → runs train → returns val_loss
        """
        from trainer import StandardTrainer, FewShotTrainer
        import torch
        import gc

        # ── Sample train HPs always ───────────────────────────────────
        train_sampled = {}
        for hp_name, choices in self.tune_config.train_hp_choices.items():
            train_sampled[hp_name] = trial.suggest_categorical(hp_name, choices)

        # ── Sample model HPs only in pretrain-phase tuning ────────────
        # When load_checkpoint_path set (train-phase tuning), model is fixed —
        # model HPs cannot meaningfully change loaded weights.
        model_sampled = {}
        if not self.load_checkpoint_path:
            for hp_name, choices in self.tune_config.model_hp_choices.items():
                model_sampled[hp_name] = trial.suggest_categorical(hp_name, choices)

        self._logger.info(f"Trial {trial.number} | model_hps={model_sampled} train_hps={train_sampled}")

        # ── Fresh model — apply model HPs via ModelConfig.update_config ──
        from model_factory import ModelFactory, ModelConfig
        if model_sampled:
            trial_model_config = ModelConfig.update_config(self.model_config, **model_sampled)
        else:
            trial_model_config = self.model_config
        trial_model = ModelFactory.create(trial_model_config, device=self.device)

        # ── Fresh TrainConfig copy — apply train HPs via setattr ──────
        trial_train_config = copy.deepcopy(self.train_config)
        for k, v in train_sampled.items():
            if hasattr(trial_train_config, k):
                setattr(trial_train_config, k, v)
            else:
                self._logger.warning(f"Train HP '{k}' not in TrainConfig — skipped")

        if self.load_checkpoint_path:
            # ── Train-phase tuning — reload pretrain checkpoint ───────
            ModelFactory.load(trial_model, self.load_checkpoint_path)
            self._logger.info(f"Trial {trial.number} | loaded pretrain ckpt: {self.load_checkpoint_path}")
        else:
            # ── Pretrain-phase tuning — shorten epochs for proxy ──────
            epochs = (self.tune_config.proxy_epochs
                      if self.tune_config.proxy_epochs is not None
                      else max(10, self.train_config.epochs_pretrain // 5))
            trial_train_config.epochs_pretrain = epochs

        TrainerClass = StandardTrainer if self.paradigm == 'standard' else FewShotTrainer
        trainer = TrainerClass(trial_model, self.factory, trial_train_config, self.device)

        try:
            if self.load_checkpoint_path:
                trainer.impl.state.is_pretrained = True
                trainer.train()
            else:
                trainer.pretrain()

            val_metric = trainer.impl.state.best_val_loss
            self._logger.info(f"Trial {trial.number} | val_metric={val_metric:.4f}")

        except self._optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            self._logger.error(f"Trial {trial.number} failed: {e}")
            raise
        finally:
            del trainer
            del trial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        trial.report(val_metric, step=1)
        if trial.should_prune():
            raise self._optuna.exceptions.TrialPruned()

        return val_metric


    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _trial_callback(self, study, trial):
        """Logs trial summary to file — no stdout."""
        self._logger.info(
            f"Trial {trial.number} complete | "
            f"val={trial.value:.4f} | "
            f"params={trial.params} | "
            f"best_so_far={study.best_value:.4f}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Logs full tuning summary to file. Brief stdout only."""
        if not self.best_trial:
            print("  Tuner: no trials completed.")
            return
        self._logger.info(f"=== TUNING SUMMARY === study={self.tune_config.study_name}")
        self._logger.info(f"Best trial: {self.best_trial.number}")
        self._logger.info(f"Best val:   {self.best_trial.value:.4f}")
        for k, v in self.best_hps.items():
            self._logger.info(f"  {k}: {v}")
        print(f"  Tuner summary written to {self._log_path}")
