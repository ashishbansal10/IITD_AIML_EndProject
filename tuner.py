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
Formats A & B → enumerate all valid combos → index sampling via trial.suggest_categorical.
                GridSampler exhausts indices systematically; TPESampler learns best indices.
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
    proxy_epochs:     Optional[int]  = None        # None → max(10, epochs_pretrain // 5)
    min_resources:    Optional[int]  = None        # for HyperbandPruner — None → auto-derived

    study_name:       str            = 'hp_search'
    storage:          Optional[str]  = None        # None=memory, 'sqlite:///hp.db'=persistent

    # Proxy training length per trial

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
        Formats A & B (grid) → enumerate all valid combos → trial.suggest_categorical
                                over combo index — works for both GridSampler and TPESampler.
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

        self._validate_config()

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

        # ── Logger setup ──────────────────────────────────────────────
        self._log_path = None
        self._logger   = self._setup_logger(logs_dir)
        if self._logger is None:
            raise RuntimeError(f"HPTuner initialization failed. Check {logs_dir} permissions.")

        # ── Expand HP choices into internal combo lists ────────────────
        # Model and trainer kept as separate lists — never merged.
        # Total combos depends on whether the model is fixed or searchable
        model_input = tune_config.model_hp_choices if self.phase != 'train' else None
        self._model_combos   = _expand_hp_choices(model_input)
        self._trainer_combos   = _expand_hp_choices(tune_config.train_hp_choices)
        self._total_combos = len(self._model_combos) * len(self._trainer_combos)

        # ── Resolve n_trials ──────────────────────────────────────────
        self._n_trials = self._resolve_n_trials()
        self._sampler  = self._select_sampler()
        self._pruner   = self._select_pruner()

        self._logger.info(f"HPTuner initialized — run_id={self.run_id} paradigm={self.paradigm} phase={self.phase} logs_dir={logs_dir}")

    def _validate_config(self):
        """Validates tuning phase requirements and checkpoint constraints."""
        valid_phases = ['pretrain', 'train', 'full']
        if self.phase not in valid_phases:
            raise ValueError(f"Invalid phase '{self.phase}'. Must be one of {valid_phases}")

        if self.phase == 'train':
            if not self.load_checkpoint_path:
                raise ValueError("Phase 'train' requires 'load_checkpoint_path' (pretrain weights).")
        
        if self.phase in ['pretrain', 'full']:
            if self.load_checkpoint_path:
                 # In pretrain/full, we start from scratch. 
                 # If a path is provided, we should warn or raise. 
                 # Based on your requirements, it has to be None.
                 raise ValueError(f"Phase '{self.phase}' must start from scratch. Set 'load_checkpoint_path' to None.")

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
        """
        Auto-select sampler based on total combo space size vs n_trials:
            total_combos <= 12 OR n_trials >= total_combos
                → GridSampler (exhaustive, guaranteed coverage)
            total_combos > 12 AND n_trials < total_combos
                → TPESampler (Bayesian, seeded for reproducibility)
                  TPE learns which regions are promising after n_startup=10 random trials.
        """
        grid_space = {}
        if len(self._model_combos) > 1:
            grid_space['model_combo_idx'] = list(range(len(self._model_combos)))
        if len(self._trainer_combos) > 1:
            grid_space['trainer_combo_idx'] = list(range(len(self._trainer_combos)))

        if self._total_combos <= 12 or self._n_trials >= self._total_combos:
            self._logger.info(f"Sampler: GridSampler (exhaustive: {self._total_combos} combos)")
            return self._optuna.samplers.GridSampler(grid_space)
        else:
            self._logger.info(
                f"Sampler: TPESampler (partial scan: {self._n_trials}/{self._total_combos} combos)"
            )
            return self._optuna.samplers.TPESampler(seed=42)

    def _select_pruner(self):
        """
        Auto-select pruner based on n_trials vs total_combos.

        min_resource resolution (controls earliest epoch Optuna may prune):
            tune_config.min_resources set → use directly
            None → auto: max(5, proxy_epochs // 3)
                   e.g. proxy=30 → 10, proxy=20 → 6, proxy=10 → 5

        Pruner selected:
            n_trials <= 2
                → NopPruner (too few trials for statistics)
            n_trials >= total_combos (exhaustive)
                → MedianPruner(n_startup_trials=5, n_warmup_steps=min_res)
                   kills bottom 50% after warmup
            n_trials < total_combos (partial scan)
                → HyperbandPruner(min=min_res, max=proxy_epochs, factor=3)
                   progressive brackets, best for large partial scans
        """
        proxy   = self.tune_config.proxy_epochs or 20
        min_res = self.tune_config.min_resources
        if min_res is None:
            min_res = max(5, proxy // 3)

        max_res = proxy

        if self._n_trials <= 2:
            self._logger.info("Pruner: NopPruner (budget too low)")
            return self._optuna.pruners.NopPruner()

        if (self._n_trials >= self._total_combos) or (max_res <= min_res):
            self._logger.info(
                f"Pruner: MedianPruner (exhaustive or Hyperband not viable: "
                f"n_trials={self._n_trials} combos={self._total_combos} "
                f"min_res={min_res} max_res={max_res})"
            )
            return self._optuna.pruners.MedianPruner(
                n_startup_trials = 5,
                n_warmup_steps   = min_res,
                interval_steps   = 1,
            )
        else:
            self._logger.info(
                f"Pruner: HyperbandPruner (partial {self._n_trials}/{self._total_combos}, "
                f"min={min_res} max={max_res} factor=3)"
            )
            return self._optuna.pruners.HyperbandPruner(
                min_resource     = min_res,
                max_resource     = max_res,
                reduction_factor = 3,
            )

    # ------------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------------

    def _setup_logger(self, logs_dir: str) -> Optional[logging.Logger]:
        # ── Logger — detailed output to file, minimal to stdout ───────

        os.makedirs(logs_dir, exist_ok=True)
        log_fname = f"tuner.{self.run_id}_{self.tune_config.study_name}_{self.phase}.log"
        self._log_path = os.path.join(logs_dir, log_fname)
        print(f"HPTuner log_fname: {log_fname} _log_path: {self._log_path}")
        logger = logging.getLogger(log_fname)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        fh = logging.FileHandler(self._log_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s | %(message)s'))
        logger.addHandler(fh)
        return logger

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

        # Create or load study
        study = optuna.create_study(
            study_name     = self.tune_config.study_name,
            storage        = self.tune_config.storage,
            direction      = 'minimize',    # minimize val_loss
            sampler        = self._sampler,
            pruner         = self._pruner,
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

        self.print_summary(study)
        self._export_visualizations(study)
        self._logger.info(
            f"Tuner complete [{self.run_id}] — "
            f"best_trial={study.best_trial.number} "
            f"best_value={study.best_trial.value:.4f} "
            f"best_hps={self.best_hps} log={self._log_path}"
        )
        print(
            f"  Tuner complete [{self.run_id}] — "
            f"best_trial={study.best_trial.number} "
            f"best_value={study.best_trial.value:.4f} "
            f"best_hps={self.best_hps} log={self._log_path}"
        )
        return self.best_hps

    def _objective(self, trial) -> float:
        """
        Single Optuna trial.
        Samples model + trainer HP combos → applies to fresh copies → runs pretrain.

        Grid inputs (Formats A & B):
            trial.suggest_categorical('model_combo_idx', [0..N-1]) → index into model_combos
            trial.suggest_categorical('trainer_combo_idx', [0..M-1]) → index into trainer_combos
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
        # suggest_categorical over index list works for both GridSampler and TPESampler:
        #   GridSampler — exhausts all indices systematically
        #   TPESampler  — treats each index as discrete category, learns best ones
        m_idx = trial.suggest_categorical(
            'model_combo_idx', list(range(len(self._model_combos)))
        ) if len(self._model_combos) > 1 else 0
        model_combo = self._model_combos[m_idx]

        # ── Sample trainer HPs ────────────────────────────────────────
        t_idx = trial.suggest_categorical(
            'trainer_combo_idx', list(range(len(self._trainer_combos)))
        ) if len(self._trainer_combos) > 1 else 0
        trainer_combo = self._trainer_combos[t_idx]

        self._logger.info(f"\nTrial {trial.number} | model={model_combo} trainer={trainer_combo}")

        # ── Fresh model — apply model HPs via ModelConfig.update_config ──
        # Original model_config untouched — copy with overridden model HPs
        trial_model_config = ModelConfig.update_config(self.model_config, **model_combo) if model_combo else self.model_config
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
        base_proxy = self.tune_config.proxy_epochs if self.tune_config.proxy_epochs is not None else 20

        # 2. Calculate the Uniform Tune Factor
        # Example: 20 / 100 = 0.2
        tune_factor = max(0.1, min(1.0, base_proxy / min(self.train_config.epochs_pretrain, self.train_config.epochs_train)))

        # 3. Apply UNIVERSAL FLOORS (Applied to all paradigms)
        # These ensure that every trial—standard or few-shot—has a stable signal.
        
        # Epoch Floor: 5
        trial_train_config.epochs_pretrain = max(5, int(self.train_config.epochs_pretrain * tune_factor))
        trial_train_config.epochs_train    = max(5, int(self.train_config.epochs_train * tune_factor))

        # Episode Floors: 20 train / 10 val
        trial_train_config.episodes_train  = max(20, int(self.train_config.episodes_train * tune_factor))
        trial_train_config.episodes_val    = max(10, int(self.train_config.episodes_val * tune_factor))

        self._logger.info(
            f"Trial {trial.number} | Factor: {tune_factor:.2f} | "
            f"Workload: pretrain {trial_train_config.epochs_pretrain}ep, train {trial_train_config.epochs_train}ep, "
            f"{trial_train_config.episodes_train}tr/{trial_train_config.episodes_val}vl episodes"
        )


        # ── Run pretrain or train ─────────────────────────────────────
        TrainerClass = StandardTrainer if self.paradigm == 'standard' else FewShotTrainer
        trainer = TrainerClass(trial_model, self.factory, trial_train_config, self.device)

        val_metric = float('inf') # Initial guard
        try:
            if self.phase in ['pretrain', 'full']:
                self._logger.info(f"Trial {trial.number} | Running Pretrain")
                trainer.pretrain(optuna_trial=trial)
                val_metric = trainer.impl.state.best_val_loss

            if self.phase in ['train', 'full']:
                if self.phase == 'train':
                    self._logger.info(f"Trial {trial.number} | Loading pretrain checkpoint from {self.load_checkpoint_path}")
                    trainer.load_pretrain(self.load_checkpoint_path)

                self._logger.info(f"Trial {trial.number} | Running Train")
                trainer.train(optuna_trial=trial)
                val_metric = trainer.impl.state.best_val_loss

            # ── Capture metrics as user_attrs for post-run analysis ──
            # pretrain_best_* always populated (from pretrain() or load_pretrain())
            # best_val_* = current phase best (pretrain or train depending on phase)
            pretrain_val_loss = trainer.impl.state.pretrain_best_val_loss
            pretrain_val_acc  = trainer.impl.state.pretrain_best_val_acc
            train_val_loss    = trainer.impl.state.best_val_loss   # = val_metric
            train_val_acc     = trainer.impl.state.best_val_acc

            trial.set_user_attr('pretrain_val_loss', float(pretrain_val_loss))
            trial.set_user_attr('pretrain_val_acc',  float(pretrain_val_acc))

            if self.phase in ('train', 'full'):
                val_acc_delta = train_val_acc - pretrain_val_acc   # + = improved
                trial.set_user_attr('train_val_loss', float(train_val_loss))
                trial.set_user_attr('train_val_acc',  float(train_val_acc))
                trial.set_user_attr('val_acc_delta',  float(val_acc_delta))
            else:
                # phase='pretrain' — train_val_* not applicable
                val_acc_delta = float('nan')

            # ── Final report for pruner — val_loss only, no penalty ───
            trial.report(val_metric, step=trainer.state.total_steps_run)
            self._logger.info(
                f"Trial {trial.number} | Step={trainer.state.total_steps_run} | "
                f"objective(val_loss)={val_metric:.4f} | "
                f"pretrain_loss={pretrain_val_loss:.4f} pretrain_acc={pretrain_val_acc:.4f} | "
                + (f"train_loss={train_val_loss:.4f} train_acc={train_val_acc:.4f} "
                   f"acc_delta={val_acc_delta:+.4f}"
                   if self.phase in ('train', 'full') else "pretrain phase only")
            )

        except self._optuna.exceptions.TrialPruned:
            self._logger.info(f"Trial {trial.number} SCUTTLED (Early Exit) via Pruner")
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

    def _export_visualizations(self, study):
        """
        Export Optuna visualizations as HTML files to logs_dir.
        Requires plotly — silently skipped if not installed.

        Files produced:
            {run_id}_{study_name}_opt_history.html    — objective per trial
            {run_id}_{study_name}_param_importance.html — HP importance ranking
            {run_id}_{study_name}_parallel_coord.html — all HPs vs objective
            {run_id}_{study_name}_acc_delta.html      — val_acc delta per trial (custom)
        """
        try:
            import optuna.visualization as vis
            import plotly.graph_objects as go

            prefix = os.path.join(
                os.path.dirname(self._log_path),
                f"tuner.{self.run_id}_{self.tune_config.study_name}"
            )

            # 1. Optimization history — objective per trial
            try:
                vis.plot_optimization_history(study).write_html(f"{prefix}_opt_history.html")
            except Exception as e:
                self._logger.warning(f"opt_history plot failed: {e}")

            # 2. HP importance ranking
            try:
                vis.plot_param_importances(study).write_html(f"{prefix}_param_importance.html")
            except Exception as e:
                self._logger.warning(f"param_importance plot failed: {e}")

            # 3. Parallel coordinate — all HPs vs objective
            try:
                vis.plot_parallel_coordinate(study).write_html(f"{prefix}_parallel_coord.html")
            except Exception as e:
                self._logger.warning(f"parallel_coord plot failed: {e}")

            # 4. Custom — val_acc_delta per trial (key metric for our goal)
            try:
                trials  = [t for t in study.trials if t.value is not None]
                numbers = [t.number for t in trials]
                deltas  = [t.user_attrs.get('val_acc_delta', float('nan')) for t in trials]
                p_accs  = [t.user_attrs.get('pretrain_val_acc', float('nan')) for t in trials]
                tr_accs = [t.user_attrs.get('train_val_acc',    float('nan')) for t in trials]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=numbers, y=deltas,
                    name='val_acc_delta (train - pretrain)',
                    marker_color=['green' if d >= 0 else 'red' for d in deltas],
                ))
                fig.add_trace(go.Scatter(
                    x=numbers, y=p_accs,
                    name='pretrain_val_acc (baseline)',
                    mode='lines', line=dict(color='blue', dash='dash'),
                ))
                fig.add_trace(go.Scatter(
                    x=numbers, y=tr_accs,
                    name='train_val_acc',
                    mode='lines+markers', line=dict(color='orange'),
                ))
                fig.add_hline(y=0, line_color='black', line_dash='dot', annotation_text='no degradation')
                fig.update_layout(
                    title=f'{self.tune_config.study_name} — val_acc: pretrain vs train per trial',
                    xaxis_title='Trial', yaxis_title='val_acc',
                    barmode='overlay',
                )
                fig.write_html(f"{prefix}_acc_delta.html")
            except Exception as e:
                self._logger.warning(f"acc_delta plot failed: {e}")

            # 5. val_acc vs val_loss scatter — tradeoff view
            if self.phase in ('train', 'full'):
                try:
                    tr_losses = [t.user_attrs.get('train_val_loss', None) for t in trials]
                    tr_accs   = [t.user_attrs.get('train_val_acc',  None) for t in trials]
                    p_acc_ref = trials[0].user_attrs.get('pretrain_val_acc', None) if trials else None

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tr_losses, y=tr_accs,
                        mode='markers+text',
                        text=[str(t.number) for t in trials],
                        textposition='top center',
                        marker=dict(
                            size=10,
                            color=[t.value for t in trials],
                            colorscale='RdYlGn_r',
                            colorbar=dict(title='objective'),
                            showscale=True,
                        ),
                        name='trials',
                    ))
                    if p_acc_ref is not None:
                        fig.add_hline(
                            y=p_acc_ref, line_color='blue', line_dash='dash',
                            annotation_text=f'pretrain_val_acc={p_acc_ref:.4f}',
                            annotation_position='right',
                        )
                    fig.update_layout(
                        title=f'{self.tune_config.study_name} — val_acc vs val_loss per trial',
                        xaxis_title='train_val_loss (lower=better)',
                        yaxis_title='train_val_acc (higher=better)',
                    )
                    fig.write_html(f"{prefix}_acc_vs_loss.html")
                except Exception as e:
                    self._logger.warning(f"acc_vs_loss scatter failed: {e}")

            self._logger.info(f"Visualizations exported to {prefix}_*.html")
            print(f"  Visualizations: {prefix}_*.html")

        except ImportError:
            self._logger.info("plotly not installed — visualizations skipped")

    def _trial_callback(self, study, trial):
        """Logs per-trial summary — val_loss + val_acc + pretrain baseline + best so far."""
        if trial.value is None:
            return
        pretrain_loss = trial.user_attrs.get('pretrain_val_loss', float('nan'))
        pretrain_acc  = trial.user_attrs.get('pretrain_val_acc',  float('nan'))

        if self.phase in ('train', 'full'):
            train_loss = trial.user_attrs.get('train_val_loss', float('nan'))
            train_acc  = trial.user_attrs.get('train_val_acc',  float('nan'))
            delta      = trial.user_attrs.get('val_acc_delta',  float('nan'))
            phase_info = (
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"acc_delta={delta:+.4f} | "
            )
        else:
            phase_info = ""

        self._logger.info(
            f"Trial {trial.number} | "
            f"objective={trial.value:.4f} | "
            f"pretrain_loss={pretrain_loss:.4f} pretrain_acc={pretrain_acc:.4f} | "
            f"{phase_info}"
            f"best_so_far={study.best_value:.4f} | "
            f"params={trial.params}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self, study=None):
        """
        Prints full tuning results after run() completes.
        Logs best trial metrics + all-trial comparison table.
        Exports styled HTML table to logs_dir for visual inspection.
        """
        if not self.best_trial:
            print("No trials completed yet. Call run() first.")
            return

        bt            = self.best_trial
        pretrain_loss = bt.user_attrs.get('pretrain_val_loss', float('nan'))
        pretrain_acc  = bt.user_attrs.get('pretrain_val_acc',  float('nan'))
        train_loss    = bt.user_attrs.get('train_val_loss',    float('nan'))
        train_acc     = bt.user_attrs.get('train_val_acc',     float('nan'))
        acc_delta     = bt.user_attrs.get('val_acc_delta',     float('nan'))

        self._logger.info(f"")
        self._logger.info(f"=== TUNING SUMMARY === study={self.tune_config.study_name} ===")
        self._logger.info(f"  Best trial  : {bt.number}")
        self._logger.info(f"  Objective   : {bt.value:.4f}  (val_loss — lower is better)")
        self._logger.info(f"  Pretrain    : val_loss={pretrain_loss:.4f}  val_acc={pretrain_acc:.4f}  (baseline)")
        if self.phase in ('train', 'full'):
            self._logger.info(f"  Train       : val_loss={train_loss:.4f}  val_acc={train_acc:.4f}  acc_delta={acc_delta:+.4f}")
        self._logger.info(f"  Best HPs:")
        for group, hps in self.best_hps.items():
            for k, v in (hps or {}).items():
                self._logger.info(f"      [{group}] {k:20s} : {v}")

        # ── Build all-trials DataFrame ────────────────────────────────
        if study is None:
            print(f"  Tuner summary written to {self._log_path}")
            return

        import pandas as pd

        rows = []
        # Collect all HP keys in sorted order for consistent columns
        all_hp_keys = sorted({
            k for t in study.trials if t.value is not None
            for k in t.params.keys()
        })

        for t in sorted(study.trials, key=lambda x: x.number):
            if t.value is None:
                continue
            row = {'trial': t.number}
            for k in all_hp_keys:
                row[k] = t.params.get(k, None)
            row['pretrain_val_loss'] = t.user_attrs.get('pretrain_val_loss', float('nan'))
            row['pretrain_val_acc']  = t.user_attrs.get('pretrain_val_acc',  float('nan'))
            if self.phase in ('train', 'full'):
                row['train_val_loss'] = t.user_attrs.get('train_val_loss', float('nan'))
                row['train_val_acc']  = t.user_attrs.get('train_val_acc',  float('nan'))
                row['val_acc_delta']  = t.user_attrs.get('val_acc_delta',  float('nan'))
            row['objective'] = t.value
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('objective').reset_index(drop=True)

        # Log plain table
        self._logger.info('\n' + df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

        # ── Styled HTML table ─────────────────────────────────────────
        try:
            metric_cols = [c for c in ['pretrain_val_loss', 'pretrain_val_acc',
                                        'train_val_loss', 'train_val_acc',
                                        'val_acc_delta', 'objective'] if c in df.columns]

            styled = (
                df.style
                .format({c: '{:.4f}' for c in metric_cols})
                .background_gradient(subset=['objective'],    cmap='RdYlGn_r')
                .background_gradient(subset=['pretrain_val_acc'] if 'pretrain_val_acc' in df.columns else [], cmap='RdYlGn')
                .background_gradient(subset=['train_val_acc']    if 'train_val_acc'    in df.columns else [], cmap='RdYlGn')
                .background_gradient(subset=['val_acc_delta']    if 'val_acc_delta'    in df.columns else [], cmap='RdYlGn')
                .set_caption(
                    f"Tuner Results — {self.tune_config.study_name} | "
                    f"phase={self.phase} | best_trial={bt.number}"
                )
                .set_table_styles([{
                    'selector': 'caption',
                    'props': [('font-size', '14px'), ('font-weight', 'bold'), ('padding', '8px')]
                }, {
                    'selector': 'th',
                    'props': [('background-color', '#2c3e50'), ('color', 'white'),
                              ('padding', '6px 10px'), ('font-size', '12px')]
                }, {
                    'selector': 'td',
                    'props': [('padding', '5px 10px'), ('font-size', '12px')]
                }])
                .highlight_min(subset=['objective'], color='#d4efdf')
            )

            prefix    = os.path.splitext(self._log_path)[0]
            html_path = f"{prefix}_trials_table.html"
            styled.to_html(html_path)
            self._logger.info(f"Styled HTML table: {html_path}")
            print(f"  Trials table : {html_path}")

        except Exception as e:
            self._logger.warning(f"Styled HTML table failed: {e}")

        print(f"  Tuner summary written to {self._log_path}")