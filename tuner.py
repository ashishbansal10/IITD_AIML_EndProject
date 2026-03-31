"""
tuner.py
========
Hyperparameter tuning via Optuna.

Classes
-------
TuneConfig — search space + trial control
HPTuner    — Optuna study wrapper, runs trials, returns best HPs

Search Space
------------
model_hp_choices : List[Dict[str, List]] — HPs applied to ModelConfig
                   e.g. [{'backbone_dropout': [0.0, 0.1, 0.2]}]
                   Applied via ModelConfig.update_config(**model_sampled)

train_hp_choices : List[Dict[str, List]] — HPs applied to TrainConfig
                   e.g. [{'label_smoothing': [0.0, 0.05, 0.1],
                           'weight_decay'  : [1e-4, 5e-4]}]
                   Applied via setattr(train_config, k, v)

Both accept a list of sub-space dicts. Each sub-space is fully cross-producted.
Multiple sub-spaces allow defining mutually exclusive HP combinations:
    e.g. Group B — EWC vs freeze_n (mutually exclusive):
    train_hp_choices = [
        {'ewc_lambda': [0.0],            'freeze_n_epochs': [0]},
        {'ewc_lambda': [0.0],            'freeze_n_epochs': [5, 10, 20]},
        {'ewc_lambda': [0.1, 1.0, 10.0], 'freeze_n_epochs': [0]},
    ]
    Total trials = 1 + 3 + 3 = 7  (no wasted combinations)

Sampler auto-selected:
    total_space_size <= 12 → GridSampler (exhaustive, guaranteed coverage)
    total_space_size >  12 → TPESampler  (Bayesian, learns from trials)

Pruner auto-selected from effective n_trials:
    <= 10 → NopPruner
    <= 30 → MedianPruner
    >  30 → HyperbandPruner

Tuning Phases
-------------
phase='pretrain' : proxy pretrain per trial → val_loss objective
phase='train'    : reloads pretrain checkpoint per trial → val_loss objective
phase='full'     : full pretrain + train per trial → val_loss objective

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

    Sampler, pruner, n_trials are auto-selected internally — user sets search space only.

    model_hp_choices : List[Dict[str, List]]
        HPs that go to ModelConfig — aliased names matching *_config() kwargs.
        e.g. [{'backbone_dropout': [0.0, 0.1, 0.2]}]
        Applied via ModelConfig.update_config(**model_sampled).
        Only active in pretrain-phase tuning — model is fixed in train-phase tuning.

    train_hp_choices : List[Dict[str, List]]
        HPs that go to TrainConfig — field names directly.
        e.g. [{'label_smoothing': [0.0, 0.05, 0.1], 'weight_decay': [1e-4, 5e-4]}]
        Applied via setattr(train_config, k, v).

    Both accept a list of sub-space dicts:
        Single sub-space  → full cross-product (standard grid)
        Multiple sub-spaces → each sub-space cross-producted independently,
                              total trials = sum of sub-space sizes.
                              Use to define mutually exclusive HP combinations.

    Sampler auto-selected from total_space_size:
        <= 12 → GridSampler (exhaustive, guaranteed coverage)
        >  12 → TPESampler  (Bayesian, learns from trials)

    Pruner auto-selected from effective n_trials:
        <= 10 → NopPruner
        <= 30 → MedianPruner
        >  30 → HyperbandPruner

    n_trials: None = auto 
        GridSampler → : total space size
        TPESampler  → : max(10: min(size*2, 30))
        unless overridden by user.

    proxy_epochs: None = max(10, epochs_pretrain // 5) — pretrain-phase tuning only
    """

    # ── Search space ──────────────────────────────────────────────────
    # model_hp_choices: aliased names matching *_config() hp_overrides kwargs
    #   e.g. [{'backbone_dropout': [0.0, 0.1, 0.2], 'temperature': [5.0, 10.0]}]
    #   Applied via :
    #       model_config = ModelConfig.update_config(base_model_config, **model_sampled)
    #       model = ModelFactory.create()
    #
    # train_hp_choices: TrainConfig field names
    #   e.g. [{'label_smoothing': [0.0, 0.05, 0.1], 'weight_decay': [1e-4, 5e-4]}]
    #   Applied via setattr(train_config, k, v)
    # Each is a list of sub-space dicts. Single-element list = standard grid.
    # Multiple sub-spaces allow mutually exclusive HP combinations.
    model_hp_choices: List[Dict[str, List[Any]]] = field(default_factory=list)
    train_hp_choices: List[Dict[str, List[Any]]] = field(default_factory=list)

    # ── Optional ──────────────────────────────────────────────────────
    n_trials:     Optional[int] = None   # None = auto from search space
    proxy_epochs: Optional[int] = None   # None = max(10, epochs_pretrain // 5)
    study_name:   str           = 'hp_search'
    storage:      Optional[str] = None   # None=memory, path=persistent

    def _space_size(self, spaces: List[Dict[str, List[Any]]]) -> int:
        """Sum of cross-products across all sub-spaces in a choices list."""
        total = 0
        for space in spaces:
            size = 1
            for v in space.values():
                size *= len(v)
            total += size
        return total or 1

    def search_space_size(self) -> int:
        """Total trials = sum of all sub-space sizes across model + train choices."""
        return self._space_size(self.model_hp_choices) + self._space_size(self.train_hp_choices) - (1 if self.model_hp_choices and self.train_hp_choices else 0)

    def effective_n_trials(self) -> int:
        """Auto-compute n_trials. GridSampler = exact size. TPE = heuristic."""
        if self.n_trials is not None:
            return self.n_trials
        size = self.search_space_size()
        if size <= 12:
            return size
        return min(size * 2, 30)

    def all_spaces(self) -> List[Dict[str, List[Any]]]:
        """All sub-spaces combined — passed to GridSampler."""
        if self.model_hp_choices and self.train_hp_choices:
            # Merge each pair of sub-spaces for joint sampling
            return self.model_hp_choices + self.train_hp_choices
        return self.model_hp_choices or self.train_hp_choices

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
    Optuna study wrapper — runs HP search, returns best HPs.

    Flow per trial:
        1. Optuna samples HPs from model_hp_choices + train_hp_choices sub-spaces
        2. Apply model HPs → fresh ModelConfig via ModelConfig.update_config()
        3. Apply train HPs → fresh TrainConfig copy via setattr
        4. Run training phase (pretrain proxy / train / full per phase setting)
        5. Return best_val_loss as Optuna objective
        6. Optuna picks best trial after n_trials

    Usage:
        tuner    = HPTuner(model_config, train_config, tune_config,
                           factory, device, phase='train',
                           load_checkpoint_path=ckpt_path)
        best_hps = tuner.run()
        # best_hps = {'label_smoothing': 0.05, 'weight_decay': 5e-4}

    Note:
        Each trial creates a fresh model — original model_config untouched.
        ExperimentRunner applies best HPs after tuner.run() completes.
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
        log_path = os.path.join(
            logs_dir,
            f"{run_id}_{tune_config.study_name}_{self.phase}.log"
        )
        self._logger = logging.getLogger(
            f"tuner.{run_id}.{tune_config.study_name}.{self.phase}"
        )
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
        """Auto-select sampler based on total search space size.
        GridSampler receives list of sub-space dicts — exhausts each sub-space fully.
        TPESampler used for larger spaces — learns from trials.
        """
        size   = self.tune_config.search_space_size()
        spaces = self.tune_config.all_spaces()
        if size <= 12:
            self._logger.info(f"Sampler: GridSampler (space_size={size})")
            return self._optuna.samplers.GridSampler(spaces)
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
        self._logger.info(f"Total space size: {self.tune_config.search_space_size()} | n_trials: {n_trials}")
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
        # train_hp_choices is List[Dict[str, List]] — flatten all sub-spaces
        train_sampled = {}
        for space in self.tune_config.train_hp_choices:
            for hp_name, choices in space.items():
                train_sampled[hp_name] = trial.suggest_categorical(hp_name, choices)

        # ── Sample model HPs only in pretrain-phase tuning ────────────
        # When load_checkpoint_path set (train-phase tuning), model is fixed —
        # model HPs cannot meaningfully change loaded weights.
        model_sampled = {}
        if not self.load_checkpoint_path:
            for space in self.tune_config.model_hp_choices:
                for hp_name, choices in space.items():
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
