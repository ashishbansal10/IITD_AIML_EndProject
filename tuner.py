"""
tuner.py
========
Hyperparameter tuning via Optuna.

Purpose
-------
Flexible HP search for model architecture and trainer HPs.
Supports param_grid style inputs (sklearn-compatible).
Model HPs and trainer HPs kept fully separate — applied to different targets.

Classes
-------
TuneConfig — search space + trial control
HPTuner    — Optuna study wrapper, runs trials, returns best HPs

HP Space Input Formats  (model_hp_choices / train_hp_choices)
-------------------------------------------------------------
Format A — dict of lists (single grid, full cross product):
    train_hp_choices = {'label_smoothing': [0.0, 0.05, 0.1], 'weight_decay': [1e-4, 5e-4]}
    → 3×2 = 6 combos

Format B — list of dict of lists (multiple grids, each cross-producted then unioned):
    train_hp_choices = [
        {'ewc_lambda': [0.0],            'freeze_n_epochs': [0]},
        {'ewc_lambda': [0.0],            'freeze_n_epochs': [5, 10, 20]},
        {'ewc_lambda': [0.1, 1.0, 10.0], 'freeze_n_epochs': [0]},
    ]
    → 1 + 3 + 3 = 7 combos  (not 4×4=16 — invalid combos excluded)
    Like sklearn param_grid — list of grids, each cross-producted, results unioned.

Formats A and B are equivalent to sklearn's param_grid.

Internal Handling
-----------------
Formats A & B → enumerate all valid combos → index sampling via trial.suggest_int.
    GridSampler intentionally NOT used — fails on list-of-dict-of-lists.
    Enqueue also NOT used — TPE samples beyond queue ignore constraints.
    Index trick: TPE learns which combo index performs best. Constraints naturally
    respected because invalid combos never exist in the enumerated list.

Model HPs vs Trainer HPs
-------------------------
model_hp_choices  → applied to backbone ModelConfig fields via ModelConfig.update_config()
train_hp_choices  → applied to TrainConfig fields
Both kept in separate internal lists — never merged, never mixed.
Total trials = len(model_combos) × len(trainer_combos).
Either can be None — None means that group has no HPs to tune.

Auto Trial / Sampler / Pruner Selection
----------------------------------------
n_trials:
    None + grid inputs  → min(total_combos, MAX_TRIALS=30)
    None                → must set explicitly — raises if not set
    int                 → exact override

sampler:
    total_combos < 12   → exhaustive (index over all combos, evaluated once each)
    total_combos ≥ 12   → TPESampler (Bayesian)

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
import itertools
import logging
import os
import gc
import datetime
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


# ==============================================================================
# Constants
# ==============================================================================

MAX_TRIALS_DEFAULT   = 30    # cap on auto-derived n_trials
EXHAUSTIVE_THRESHOLD = 12    # below this → enumerate all combos exactly once
MEDIAN_PRUNER_MIN    = 10    # below this → NopPruner
HYPERBAND_MIN        = 30    # above this → HyperbandPruner (else MedianPruner)


# ==============================================================================
# HP choices expansion — param_grid semantics
# ==============================================================================

def _expand_hp_choices(
    hp_input: Union[Dict[str, List], List[Dict[str, List]], None]
) -> List[Dict]:
    """
    Expands HP choices into a flat list of combo dicts.
    Implements sklearn param_grid semantics.

    Format A — dict of lists (single grid, full cross product):
        {'lr': [1e-4, 1e-3], 'wd': [1e-4, 5e-4]}
        → [{'lr':1e-4,'wd':1e-4}, {'lr':1e-4,'wd':5e-4},
           {'lr':1e-3,'wd':1e-4}, {'lr':1e-3,'wd':5e-4}]

    Format B — list of dict of lists (multiple grids, each cross-producted then unioned):
        [{'a': [1,2], 'b': [10]}, {'a': [3], 'b': [20, 30]}]
        → [{a:1,b:10}, {a:2,b:10}, {a:3,b:20}, {a:3,b:30}]  (4 combos, not 2×2=4 full grid)

    None or empty → [{}]  (single empty combo — group contributes no HPs)

    Returns:
        List of dicts — one dict per valid HP combo, keys as given (no prefixing).
    """
    if not hp_input:
        return [{}]

    # Format A — single dict of lists
    if isinstance(hp_input, dict):
        keys   = list(hp_input.keys())
        values = [v if isinstance(v, list) else [v] for v in hp_input.values()]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Format B — list of dict of lists
    if isinstance(hp_input, list):
        if not hp_input:
            return [{}]
        all_combos = []
        for i, grid in enumerate(hp_input):
            if not isinstance(grid, dict):
                raise ValueError(
                    f"HP choices list element {i} must be a dict of lists.\n"
                    f"Got: {type(grid)}\n"
                    f"Example: [{{'lr': [1e-4, 1e-3], 'wd': [1e-4]}}]"
                )
            keys   = list(grid.keys())
            values = [v if isinstance(v, list) else [v] for v in grid.values()]
            for combo in itertools.product(*values):
                all_combos.append(dict(zip(keys, combo)))
        return all_combos

    raise ValueError(
        f"HP choices must be dict-of-lists, list-of-dict-of-lists, or callable. "
        f"Got: {type(hp_input)}"
    )




# ==============================================================================
# TuneConfig
# ==============================================================================

@dataclass
class TuneConfig:
    """
    Hyperparameter tuning configuration.

    HP Search Space
    ---------------
    model_hp_choices : Model architecture HPs → applied to backbone via set_hp().
                       dict of lists  (Format A) — single grid, cross product
                       list of dict of lists (Format B) — multi-grid union
                       None → no model HPs tuned

    train_hp_choices : Trainer HPs → applied to TrainConfig fields.
                       Same formats as model_hp_choices.
                       None → no trainer HPs tuned

    Model and trainer HPs are kept fully separate — never merged.
    Total combos = len(model_combos) × len(trainer_combos).

    Examples
    --------
    Group A — simple grid:
        TuneConfig(
            study_name       = 'groupA',
            train_hp_choices = {'label_smoothing': [0.0, 0.05, 0.1],
                                'weight_decay':    [1e-4, 5e-4]},
            proxy_epochs     = 20,
        )
        # 3×2 = 6 combos, exhaustive, NopPruner

    Group B — constrained space via multi-grid (list of dict of lists):
        TuneConfig(
            study_name       = 'groupB',
            train_hp_choices = [
                {'ewc_lambda': [0.0],             'freeze_n_epochs': [0]},
                {'ewc_lambda': [0.0],             'freeze_n_epochs': [5, 10, 20]},
                {'ewc_lambda': [0.1, 1.0, 10.0],  'freeze_n_epochs': [0]},
            ],
            proxy_epochs = 20,
        )
        # 1+3+3 = 7 combos (not 4×4=16), exhaustive, NopPruner

    Group B alternative — callable for native Optuna conditional sampling:
        def group_B_sampler(trial):
            ewc = trial.suggest_categorical('ewc_lambda', [0.0, 0.1, 1.0, 10.0])
            if ewc == 0.0:
                freeze = trial.suggest_categorical('freeze_n_epochs', [0, 5, 10, 20])
            else:
                freeze = 0
            return {'ewc_lambda': ewc, 'freeze_n_epochs': freeze}

        TuneConfig(
            study_name       = 'groupB',
            train_hp_choices = group_B_sampler,
            n_trials         = 20,   # must be explicit for callable
            proxy_epochs     = 20,
        )

    Auto Controls (all can be overridden explicitly)
    ------------------------------------------------
    n_trials    : None → auto = min(total_combos, MAX_TRIALS=20)
                          callable input → must be set explicitly
                  int  → exact override

    proxy_epochs : pretrain epochs per trial (proxy for full training quality).
                   None → max(10, epochs_pretrain // 5)

    Storage
    -------
    storage     : None → in-memory (lost after run)
                  'sqlite:///hp.db' → persistent (resume across runs)

    """

    # HP search space — model and trainer kept separate
    model_hp_choices: Optional[Union[Dict, List]] = None
    train_hp_choices: Optional[Union[Dict, List]] = None

    # Trial control
    n_trials:         Optional[int]  = None        # None → auto-derived
    study_name:       str            = 'hp_search'
    storage:          Optional[str]  = None        # None=memory, 'sqlite:///hp.db'=persistent

    # Proxy training length per trial
    proxy_epochs:     Optional[int]  = None        # None → max(10, epochs_pretrain // 5)

    def to_dict(self) -> dict:
        d = asdict(self)
        # callable fields not serializable — store type name only
        for key in ('model_hp_choices', 'train_hp_choices'):
            val = getattr(self, key, None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TuneConfig':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ==============================================================================
# HPTuner
# ==============================================================================

class HPTuner:
    """
    Optuna wrapper — flexible HP search with param_grid and callable support.

    Model HPs and trainer HPs stay fully separate throughout:
        model_hp_choices → _expand_hp_choices → model_combos list
        train_hp_choices → _expand_hp_choices → trainer_combos list
        total = len(model_combos) × len(trainer_combos)

    Internal sampling strategy:
        Formats A & B (grid) → enumerate all valid combos → trial.suggest_int
                                over combo index → TPE learns best index.
                                Constraints naturally respected — invalid combos
                                never exist in the enumerated list.

    Flow per trial:
        1. Sample model combo  (index into model_combos, or callable)
        2. Sample trainer combo (index into trainer_combos, or callable)
        3. Apply model combo   → fresh ModelConfig copy → backbone keys set
        4. Apply trainer combo → fresh TrainConfig copy → setattr per field
        5. Run pretrain phase only (proxy_epochs — 20% of full by default)
        6. Return best_val_loss as Optuna objective
        7. Optuna prunes or continues based on pruner policy

    Usage:
        tuner    = HPTuner(model_config, train_config, tune_config,
                           factory, device, phase='train', load_checkpoint_path=ckpt_path)
        best_hps = tuner.run()
        # best_hps = {'model':   {'dropout_rate': 0.2},
        #             'trainer': {'lr': 1e-3, 'weight_decay': 5e-4}}

        # Apply — caller's responsibility

    Note:
        Each trial creates a fresh model + config copy — originals untouched.
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
        # {'model': {k: v, ...}, 'trainer': {k: v, ...}}
        self.best_hps:   Dict[str, Any] = {}
        self.best_trial: Optional[Any]  = None

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

        print(f"Initializing HPTuner — run_id={self.run_id} paradigm={self.paradigm} phase={self.phase} logs_dir={logs_dir}")
        # ── Expand HP choices into internal combo lists ────────────────
        # Model and trainer kept as separate lists — never merged.
        self._model_combos   = _expand_hp_choices(tune_config.model_hp_choices)
        self._trainer_combos   = _expand_hp_choices(tune_config.train_hp_choices)

        # Total combos — None if either group uses callable
        if self._model_combos is not None and self._trainer_combos is not None:
            self._total_combos = len(self._model_combos) * len(self._trainer_combos)
        else:
            self._total_combos = max(len(self._model_combos), len(self._trainer_combos))

        # ── Resolve n_trials ──────────────────────────────────────────
        self._n_trials     = self._resolve_n_trials()

        # ── Logger setup ──────────────────────────────────────────────
        self._log_path = None
        self._logger   = self._setup_logger(logs_dir)
        if self._logger is None:
            print("Logger setup failed — check logs_dir permissions.")
        else:
            self._logger.info(f"HPTuner initialized — run_id={self.run_id} paradigm={self.paradigm} phase={self.phase} logs_dir={logs_dir}")

    # ------------------------------------------------------------------
    # Auto resolution — n_trials / sampler / pruner
    # ------------------------------------------------------------------

    def _resolve_n_trials(self) -> int:
        """
        n_trials explicitly set → use as-is.
        None + grid inputs     → min(total_combos, MAX_TRIALS_DEFAULT).
        None + callable        → raise — cannot auto-derive from callable.
        """
        if self.tune_config.n_trials is not None:
            return self.tune_config.n_trials
        if self._total_combos is None:
            raise ValueError(
                "n_trials must be set explicitly when using callable HP sampler.\n"
                "Example: TuneConfig(train_hp_choices=my_sampler, n_trials=20)"
            )
        return min(self._total_combos, MAX_TRIALS_DEFAULT)


    def _select_sampler(self):
        """Instantiates the resolved Optuna sampler."""
        self._logger.info(f"Sampler: TPESampler (size={len(self._total_combos) if self._total_combos else 0} model combos × {len(self._trainer_combos) if self._trainer_combos else 0})")
        return self._optuna.samplers.TPESampler(seed=42)

    def _select_pruner(self):
        """Auto-select pruner based on effective n_trials."""
        if self._n_trials <= 10:
            self._logger.info(f"Pruner: NopPruner (n_trials={self._n_trials})")
            return self._optuna.pruners.NopPruner()
        elif self._n_trials <= 30:
            self._logger.info(f"Pruner: MedianPruner (n_trials={self._n_trials})")
            return self._optuna.pruners.MedianPruner()
        self._logger.info(f"Pruner: HyperbandPruner (n_trials={self._n_trials})")
        return self._optuna.pruners.HyperbandPruner()

    # ------------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------------

    def _setup_logger(self, logs_dir: str) -> Optional[logging.Logger]:
        # ── Logger — detailed output to file, minimal to stdout ───────

        os.makedirs(logs_dir, exist_ok=True)
        log_fname = f"tuner.{self.run_id}_{self.tune_config.study_name}_{self.phase}.log"
        self._log_path = os.path.join(logs_dir, log_fname)
        print(f"HPTuner log_fname: {log_fname} _log_path: {self._log_path}")
        self._logger = logging.getLogger(log_fname)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fh = logging.FileHandler(self._log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s | %(message)s'))
        self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run Optuna study — n_trials total.
        Returns best HPs found as dict split by group:
            {'model': {k: v, ...}, 'trainer': {k: v, ...}}
        """
        print(f"Running HPTuner — run_id={self.run_id} phase={self.phase} n_trials={self._n_trials} total_combos={self._total_combos}")
        optuna = self._optuna

        # Silence Optuna's own verbose logging — we handle output ourselves
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = self._select_sampler()
        pruner  = self._select_pruner()

        # Create or load study
        study = optuna.create_study(
            study_name     = self.tune_config.study_name,
            storage        = self.tune_config.storage,
            direction      = 'minimize',    # minimize val_loss
            sampler        = sampler,
            pruner         = pruner,
            load_if_exists = True           # resume if storage set + study exists
        )

        print(f"  Tuner started [{self.run_id} | {self.phase} | {self._n_trials} trials | space={self._total_combos} | log={self._log_path}]")
        self._logger.info(f"Tuner started — run_id={self.run_id} phase={self.phase} n_trials={self._n_trials}")
        self._logger.info(f"Search space model: {self.tune_config.model_hp_choices}")
        self._logger.info(f"Search space train: {self.tune_config.train_hp_choices}")
        self._logger.info(f"Total space size: {self._total_combos} | n_trials: {self._n_trials}")
        self._logger.info(f"Objective: {'train val_loss (reusing pretrain ckpt)' if self.load_checkpoint_path else 'pretrain val_loss (proxy)'}")

        study.optimize(
            self._objective,
            n_trials  = self._n_trials,
            callbacks = [self._trial_callback]
        )

        self.best_trial = study.best_trial
        self.best_hps   = self._resolve_best_hps(study.best_trial.params)

        self._logger.info(f"Tuner complete [{self.run_id}] — best_trial={study.best_trial.number} best_value={study.best_trial.value:.4f} best_hps={self.best_hps} log={self._log_path}")
        print(f"  Tuner complete [{self.run_id}] — best_trial={study.best_trial.number} best_value={study.best_trial.value:.4f} best_hps={self.best_hps} log={self._log_path}")

        return self.best_hps

    def _objective(self, trial) -> float:
        """
        Single Optuna trial.
        Samples model + trainer HP combos → applies to fresh copies → runs pretrain.

        Grid inputs (Formats A & B):
            trial.suggest_int('model_combo_idx', 0, N-1) → index into model_combos
            trial.suggest_int('trainer_combo_idx', 0, M-1) → index into trainer_combos
            TPE learns which index is best. Constraints naturally respected —
            invalid combos never appear in the enumerated list.

        Fresh model + config copy per trial — originals untouched.
        TrainConfig copied per trial — original config untouched.

        Pretrain tuning (load_checkpoint_path=None):
            Samples HPs from hp_choices → runs proxy pretrain → returns val_loss

        Train tuning (load_checkpoint_path set):
            Samples HPs from hp_choices → reloads checkpoint → runs train → returns val_loss
        """
        from trainer import StandardTrainer, FewShotTrainer
        from model_factory import ModelFactory, ModelConfig

        # ── Sample model HPs only in pretrain-phase tuning ────────────
        # When load_checkpoint_path set (train-phase tuning), model is fixed —
        # model HPs cannot meaningfully change loaded weights.
        model_combo = None
        if not self.load_checkpoint_path:
            if self._model_combos and len(self._model_combos) > 1:
                idx         = trial.suggest_int('model_combo_idx', 0, len(self._model_combos) - 1)
                model_combo = self._model_combos[idx]
            else:
                # Single combo or empty — no suggest needed
                model_combo = self._model_combos[0] if self._model_combos else {}

        # ── Sample train HPs always ───────────────────────────────────
        trainer_combo = None
        if self._trainer_combos and len(self._trainer_combos) > 1:
            idx           = trial.suggest_int('trainer_combo_idx', 0, len(self._trainer_combos) - 1)
            trainer_combo = self._trainer_combos[idx]
        else:
            trainer_combo = self._trainer_combos[0] if self._trainer_combos else {}

        self._logger.info(f"\nTrial {trial.number} | model={model_combo} trainer={trainer_combo}")

        # ── Fresh model — apply model HPs via ModelConfig.update_config ──
        # Original model_config untouched — copy with overridden model HPs
        if model_combo:
            trial_model_config = ModelConfig.update_config(self.model_config, **model_combo)
        else:
            trial_model_config = self.model_config
        trial_model = ModelFactory.create(trial_model_config, device=self.device)

        # ── Fresh TrainConfig copy — apply train HPs via setattr ──────
        trial_train_config = copy.deepcopy(self.train_config)
        for k, v in trainer_combo.items():
            if hasattr(trial_train_config, k):
                setattr(trial_train_config, k, v)
            else:
                self._logger.warning(f"Train HP '{k}' not in TrainConfig — skipped")

        # Shorten pretrain for tuning — proxy, not full train
        # Use 20% of full epochs — enough signal for relative comparison
        proxy = (self.tune_config.proxy_epochs
                      if self.tune_config.proxy_epochs is not None
                      else max(10, self.train_config.epochs_pretrain // 5))
        trial_train_config.epochs_pretrain = proxy
        self._logger.info(f"Trial {trial.number} | proxy pretrain epochs: {proxy}")

        if self.load_checkpoint_path:
            # ── Train-phase tuning — reload pretrain checkpoint ───────
            ModelFactory.load(trial_model, self.load_checkpoint_path)
            self._logger.info(f"Trial {trial.number} | loaded pretrain ckpt: {self.load_checkpoint_path}")
        else:
            # ── Pretrain-phase tuning — shorten epochs for proxy ──────
                pass  # proxy epochs already set above

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

        # ── Report intermediate value for pruning ─────────────────────
        trial.report(val_metric, step=trial_train_config.epochs_pretrain)
        if trial.should_prune():
            self._logger.info(f"Trial {trial.number} PRUNED | val_loss={val_metric:.4f}")
            raise self._optuna.exceptions.TrialPruned()

        self._logger.info(f"Trial {trial.number} END | val_loss={val_metric:.4f}")
        return val_metric

    # ------------------------------------------------------------------
    # Build trial configs
    # ------------------------------------------------------------------

    def _resolve_best_hps(self, raw_params: Dict) -> Dict[str, Dict]:
        """
        Reconstructs best model + trainer combo dicts from best trial params.

        Grid inputs: raw_params has 'model_combo_idx' / 'trainer_combo_idx' keys
                     → look up actual combo dicts by index.
        Callable inputs: raw_params has raw HP keys from suggest_* calls
                         → separate by known group keys.

        Returns:
            {'model': {k: v, ...}, 'trainer': {k: v, ...}}
        """
        # Model best combo
        if self._model_combos:
            idx        = raw_params.get('model_combo_idx', 0)
            model_best = dict(self._model_combos[idx])
        else:
            model_best = {}

        # Trainer best combo
        if self._trainer_combos:
            idx          = raw_params.get('trainer_combo_idx', 0)
            trainer_best = dict(self._trainer_combos[idx])
        else:
            trainer_best = {}

        return {'model': model_best, 'trainer': trainer_best}

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _trial_callback(self, study, trial):
        """Prints concise trial summary after each trial completes."""
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
        """Prints full tuning results after run() completes."""
        if not self.best_trial:
            print("No trials completed yet. Call run() first.")
            return
        self._logger.info(f"=== TUNING SUMMARY === study={self.tune_config.study_name}")
        self._logger.info(f"Best trial: {self.best_trial.number}")
        self._logger.info(f"Best val:   {self.best_trial.value:.4f}")
        self._logger.info(f"  Best HPs:")
        for group, hps in self.best_hps.items():
            for k, v in (hps or {}).items():
                self._logger.info(f"      [{group}] {k:20s} : {v}")
        print(f"  Tuner summary written to {self._log_path}")