"""
experiment.py
=============
Experiment orchestration — runs all 6 experiments, collects results, saves everything.

Classes
-------
ExperimentConfig    — one run definition (model, train, eval, tuner configs)
ExecutionerConfig   — global run strategy (dirs, checkpoint policy, workers)
RunResult           — complete per-run saved data including RunScores
ExperimentSummary   — all RunResults + experiment-level runtime + hardware info
ExperimentRunner    — orchestrator: tune → pretrain → eval → train → eval → save
ResultStore         — serialize/deserialize RunResult + ExperimentSummary

Per-Run Flow in ExperimentRunner
---------------------------------
    full_tune_config set:
        → tune full pretrain+train pipeline → best_hps → apply
    else:
        pretrain_tune_config set → proxy pretrain tuning → best_hps → apply
        train_tune_config set   → full pretrain once → train tuning → best_hps → apply

    pretrain()      → model at best-pretrain-epoch on return
    eval_pretrain()
    train()         → model at best-train-epoch on return
    eval_trained()
    collect RunScores → RunResult → save JSON → _cleanup()

Tuning Modes (ExperimentConfig fields)
---------------------------------------
    Scenario 1 — pretrain_tune_config only:
        Proxy pretrain trials → best pretrain HPs
    Scenario 2 — train_tune_config only:
        Full pretrain once → train trials (reloads checkpoint) → best train HPs
    Scenario 3 — pretrain_tune_config + train_tune_config:
        Both independently
    Scenario 4 — full_tune_config (overrides scenarios 1-3):
        Full pretrain+train per trial → single best HP set covering both phases

    best_hps will capture results from tuners:
    - None if no tuning
    - full_tune_config → best_hps['full'] = {'model': {param: value}, 'trainer': {param: value}}
    - pretrain_tune_config → best_hps['pretrain'] = {'model': {param: value}, 'trainer': {param: value}}
    - train_tune_config → best_hps['train'] = {'model': {param: value}, 'trainer': {param: value}} 
        - however, 'model' is not tuned in train phase, so this will typically be empty or None.

Parallel vs Sequential
-----------------------
run_mode='sequential' — one run at a time (default, single GPU)
run_mode='parallel'   — [FUTURE] multi-GPU only

Required Libraries
------------------
# torch>=2.0.0
# pandas>=1.5.0  - ResultStore CSV
"""

import os
import json
import time
import platform
import datetime
import copy
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from trainer       import TrainConfig, TrainingState, TrainingHistory, StandardTrainer, FewShotTrainer
from evaluator     import EvalConfig, EvalResult, RunScores, Evaluator
from tuner         import TuneConfig, HPTuner
from model_factory import ModelConfig, ModelFactory


# ==============================================================================
# ExperimentConfig — one run definition
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Defines one complete run.

    Tuning modes — set at most one of:
        full_tune_config          : Tune full pretrain+train pipeline per trial.
                                    If set, pretrain_tune_config and train_tune_config ignored.
        pretrain_tune_config      : Proxy pretrain tuning only.
        train_tune_config         : Train-phase tuning (reloads pretrain checkpoint).
        Both pretrain + train     : Independent tuning per phase.

    Fields:
        run_id               : unique string e.g. 'run1_cnn_standard'
        paradigm             : 'standard' or 'fewshot'
        arch                 : 'cnn', 'gnn', 'hybrid'
        model_config         : ModelConfig instance
        train_config         : TrainConfig instance
        eval_config          : EvalConfig instance
        full_tune_config     : TuneConfig or None — full pipeline tuning
        pretrain_tune_config : TuneConfig or None — pretrain proxy tuning
        train_tune_config    : TuneConfig or None — train phase tuning
        random_seed          : for reproducibility
        notes                : free-text notes
    """
    run_id:       str
    paradigm:     str
    arch:         str            # 'cnn', 'gnn', 'hybrid'
    model_config: ModelConfig
    train_config: TrainConfig
    eval_config:  EvalConfig
    full_tune_config:     Optional[TuneConfig] = None
    pretrain_tune_config: Optional[TuneConfig] = None
    train_tune_config:    Optional[TuneConfig] = None
    random_seed:  int = 42
    notes:        str = ''

    def to_dict(self) -> dict:
        return {
            'run_id'              : self.run_id,
            'paradigm'            : self.paradigm,
            'arch'                : self.arch,
            'model_config'        : self.model_config.to_dict(),
            'train_config'        : self.train_config.to_dict(),
            'eval_config'         : self.eval_config.to_dict(),
            'full_tune_config'    : self.full_tune_config.to_dict()     if self.full_tune_config     else None,
            'pretrain_tune_config': self.pretrain_tune_config.to_dict() if self.pretrain_tune_config else None,
            'train_tune_config'   : self.train_tune_config.to_dict()    if self.train_tune_config    else None,
            'random_seed'         : self.random_seed,
            'notes'               : self.notes,
        }

    def validate_config(self):
        """Validates field values at construction time."""
        if not self.run_id or not self.run_id.strip():
            raise ValueError("ExperimentConfig.run_id must be a non-empty string.")
        if ' ' in self.run_id:
            raise ValueError(
                f"ExperimentConfig.run_id='{self.run_id}' must not contain spaces. "
                f"Use underscores: e.g. 'run1_cnn_standard'."
            )
        valid_paradigms = {'standard', 'fewshot'}
        if self.paradigm not in valid_paradigms:
            raise ValueError(
                f"ExperimentConfig.paradigm='{self.paradigm}' invalid. Valid: {valid_paradigms}"
            )
        valid_archs = {'cnn', 'gnn', 'hybrid'}
        if self.arch not in valid_archs:
            raise ValueError(
                f"ExperimentConfig.arch='{self.arch}' invalid. Valid: {valid_archs}"
            )


# ==============================================================================
# ExecutionerConfig — global run strategy
# ==============================================================================

@dataclass
class ExecutionerConfig:
    """
    Global strategy — applies equally to all runs.

    Fields:
        run_mode                 : 'sequential' (default) or 'parallel' [FUTURE]
        checkpoint_dir           : model checkpoint directory
        results_dir              : RunResult JSON + ExperimentSummary directory
        plots_dir                : saved plot directory
        logs_dir                 : tuner log files directory
        num_workers              : DataLoader workers (stamped into TrainConfig + EvalConfig)
        keep_pretrain_checkpoint : keep pretrain .pt after run completes
        keep_train_checkpoint    : keep final trained .pt after run completes
        max_parallel             : [FUTURE] max parallel runs
    """
    run_mode:       str  = 'sequential'
    checkpoint_dir: str  = 'checkpoints'
    results_dir:    str  = 'results'
    plots_dir:      str  = 'plots'
    logs_dir:       str  = 'logs'
    num_workers:    int  = 2
    max_parallel:   int  = 1

    # ── Checkpoint cleanup ────────────────────────────────────────────
    # Decided at end of each run — not mid-run
    keep_pretrain_checkpoint: bool = False
    keep_train_checkpoint:    bool = True
    save_tune_logs:           bool = True   # False = delete tuner log files after tune_all()

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# RunResult — complete per-run saved data
# ==============================================================================

@dataclass
class RunResult:
    """
    Complete saved data for one run. Serialized to JSON after each run.
    All configs stored as dicts — no class dependencies at load time.
    """

    # Identity
    run_id:               str
    paradigm:             str
    arch:                 str

    # Configs — as dicts for portability
    model_config:         dict
    train_config:         dict
    eval_config:          dict
    exec_config:          dict
    full_tune_config:     Optional[dict]
    pretrain_tune_config: Optional[dict]
    train_tune_config:    Optional[dict]

    # Best HPs from tuner
    best_hps:             dict

    # Checkpoints
    pretrain_path:        Optional[str]
    final_model_path:     Optional[str]

    # Training outcome
    training_state:       dict             # TrainingState serialized
    training_history:     dict             # TrainingHistory serialized

    # Scores — RunScores serialized
    run_scores:           dict

    # Per-run runtime
    start_time:           str
    end_time:             str
    duration_seconds:     float
    random_seed:          int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'RunResult':
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ==============================================================================
# ExperimentSummary — all runs + experiment-level info
# ==============================================================================

@dataclass
class ExperimentSummary:
    """
    Complete experiment record — all RunResults + experiment-level runtime.
    Saved once after all runs complete.

    comparison_table: {score_name: {run_id: top1_acc}}
    """
    experiment_id:    str
    runs:             List[dict]
    comparison_table: dict

    # Experiment-level runtime
    experiment_start: str
    experiment_end:   str
    total_duration:   float             # seconds

    # Hardware + library
    device_name:      str
    device_memory_gb: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentSummary':
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ==============================================================================
# ExperimentRunner — orchestrator
# ==============================================================================

class ExperimentRunner:
    """
    Orchestrates all runs end to end.

    Public methods:
        tune_all()  → run HP tuning only, return best HPs, no full training
        run_all()   → run full experiment, return ExperimentSummary

    Usage:
        run_configs = [run1, run2, run3, run4, run5, run6]
        exec_config = ExecutionerConfig()
        runner  = ExperimentRunner(run_configs, exec_config, factory, device)
        hps     = runner.tune_all()   # optional — inspect and apply in notebook
        summary = runner.run_all()

    Access individual results:
        runner.run_results['run1_cnn_standard']
    """

    def __init__(self,
                 run_configs: List[ExperimentConfig],
                 exec_config: ExecutionerConfig,
                 factory,
                 device:      torch.device):
        self.run_configs  = run_configs
        self.exec_config  = exec_config
        self.factory      = factory
        self.device       = device
        self.run_results: Dict[str, RunResult] = {}

        for d in [exec_config.checkpoint_dir, exec_config.results_dir, exec_config.plots_dir, exec_config.logs_dir]:
            os.makedirs(d, exist_ok=True)

        self._validate_and_stamp()

    # ------------------------------------------------------------------
    # Public — tune_all
    # ------------------------------------------------------------------

    def tune_all(self) -> Dict[str, dict]:
        """
        Run HP tuning only — no full training, no evaluation.

        Behavior per run_cfg:
            full_tune_config set           → full pretrain+train per trial
            pretrain_tune only             → proxy pretrain trials
            train_tune only                → full pretrain once + train trials
            pretrain_tune + train_tune     → both independently
            none set                       → skip (no-op)

        Returns:
            {run_id: {'pretrain': {...}, 'train': {...}}}
            or
            {run_id: {'full': {...}}}

        At end: dump tune_results JSON, delete ALL temp checkpoints.
        User inspects returned dict, applies HPs in notebook, then calls run_all().
        """
        has_tuning = any(self._has_any_tuner(c) for c in self.run_configs)
        if not has_tuning:
            print("  tune_all: no tune_config set — no-op")
            return {}

        tune_id    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        tune_start = time.time()
        all_hps    = {}
        temp_ckpts = []

        print(f"\n{'='*70}")
        print(f"TUNE START — {tune_id}")
        print(f"Runs: {len(self.run_configs)}")
        print(f"{'='*70}")

        for i, run_cfg in enumerate(self.run_configs):
            if not self._has_any_tuner(run_cfg):
                print(f"\n[{i+1}/{len(self.run_configs)}] {run_cfg.run_id} — no tuner, skip")
                continue

            print(f"\n[{i+1}/{len(self.run_configs)}] {run_cfg.run_id}  [{run_cfg.arch} | {run_cfg.paradigm}]")
            self._seed(run_cfg.random_seed)

            if run_cfg.full_tune_config is not None:
                full_hps = self._run_tuner(run_cfg, run_cfg.full_tune_config, phase='full', ckpt_path=None)
                all_hps[run_cfg.run_id] = {'full': full_hps}
            else:
                pretrain_hps, train_hps, ckpt = self._run_split_tuner_for_tune_all(run_cfg)
                if ckpt:
                    temp_ckpts.append(ckpt)
                all_hps[run_cfg.run_id] = {
                    'pretrain': pretrain_hps,
                    'train': train_hps,
                }

        # Cleanup temp checkpoints
        for ckpt in temp_ckpts:
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)
                print(f"  Cleaned: {ckpt}")

        tune_duration = time.time() - tune_start
        self._dump_tune_results(tune_id, tune_duration, all_hps)

        # Delete tuner log files if save_tune_logs=False
        if not self.exec_config.save_tune_logs:
            import glob
            pattern = os.path.join(self.exec_config.logs_dir, '*.log')
            for log_file in glob.glob(pattern):
                try:
                    os.remove(log_file)
                except OSError:
                    pass

        print(f"\n{'='*70}")
        print(f"TUNE COMPLETE — {tune_duration/60:.1f}min")
        print(f"{'='*70}")

        return all_hps

    # ------------------------------------------------------------------
    # Public — run_all
    # ------------------------------------------------------------------

    def run_all(self) -> ExperimentSummary:
        """
        Run all experiments sequentially.
        Returns ExperimentSummary after all complete.
        """
        if self.exec_config.run_mode == 'parallel':
            raise NotImplementedError("run_mode='parallel' not yet implemented. Use 'sequential'.")

        experiment_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_start     = time.time()
        exp_start_str = datetime.datetime.now().isoformat()

        print(f"\n{'='*70}")
        print(f"EXPERIMENT START — {experiment_id}")
        print(f"Runs: {len(self.run_configs)}")
        print(f"{'='*70}")

        for i, run_cfg in enumerate(self.run_configs):
            print(f"\n[{i+1}/{len(self.run_configs)}] {run_cfg.run_id}  [arch: {run_cfg.arch} | paradigm: {run_cfg.paradigm}]")
            result = self._run_single(run_cfg)
            self.run_results[run_cfg.run_id] = result

            path = os.path.join(self.exec_config.results_dir, f"{run_cfg.run_id}_result.json")
            result.to_json(path)
            print(f"  Saved: {path}")

            elapsed = result.duration_seconds / 60
            mem_mb  = (torch.cuda.memory_allocated(self.device) / 1024 / 1024
                       if self.device.type == 'cuda' else 0)
            print(f"\n  {run_cfg.run_id} DONE | mem={mem_mb:.1f}MB | time={elapsed:.1f}min")
            print(f"  {'─'*60}")

        exp_end     = time.time()
        exp_end_str = datetime.datetime.now().isoformat()
        duration    = exp_end - exp_start

        summary      = self._build_summary(experiment_id, exp_start_str, exp_end_str, duration)
        summary_path = os.path.join(self.exec_config.results_dir, f"experiment_{experiment_id}_summary.json")
        summary.to_json(summary_path)

        print(f"\n{'='*70}")
        print(f"EXPERIMENT COMPLETE — {duration/3600:.2f}h")
        print(f"Summary: {summary_path}")
        print(f"{'='*70}")

        return summary

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def _run_single(self, run_cfg: ExperimentConfig) -> RunResult:
        """
        Complete single run:
            tune (optional) → pretrain → eval pretrain →
            train → eval trained → collect RunResult → cleanup
        """
        self._seed(run_cfg.random_seed)

        start_time     = time.time()
        start_time_str = datetime.datetime.now().isoformat()

        # ── Create model ──────────────────────────────────────────────
        model = ModelFactory.create(run_cfg.model_config, device=self.device)
        print(f"  Model: {model}")
        best_hps = {}

        # ── HP Tuning ─────────────────────────────────────────────────
        if run_cfg.full_tune_config is not None:
            # Scenario 4: full pipeline tuning overrides split tuning
            full_hps = self._run_tuner(run_cfg, run_cfg.full_tune_config, phase='full', ckpt_path=None)
            best_hps["full"] = full_hps
            model, run_cfg = self._apply_model_hps(model, run_cfg, full_hps, run_cfg.full_tune_config)
            self._apply_train_hps(run_cfg, full_hps, run_cfg.full_tune_config)
        else:
            # Scenario 1/3: pretrain proxy tuning
            if run_cfg.pretrain_tune_config is not None:
                pretrain_hps = self._run_tuner(run_cfg, run_cfg.pretrain_tune_config, phase='pretrain', ckpt_path=None)
                best_hps["pretrain"] = pretrain_hps
                model, run_cfg = self._apply_model_hps(model, run_cfg, pretrain_hps, run_cfg.pretrain_tune_config)
                self._apply_train_hps(run_cfg, pretrain_hps, run_cfg.pretrain_tune_config)

        # ── Create trainer + pretrain + evaluate ──────────────────────
        trainer   = self._make_trainer(run_cfg, model)
        evaluator = Evaluator(self.factory, run_cfg.eval_config, self.device)

        trainer.pretrain()
        pre_softmax, pre_proto_seen, pre_proto_novel = evaluator.eval_pretrain(model)

        # ── Scenario 2/3: train-phase tuning after pretrain ───────────
        if run_cfg.full_tune_config is None and run_cfg.train_tune_config is not None:
            pretrain_ckpt = trainer.state.pretrain_export_path or None
            train_hps = self._run_tuner(run_cfg, run_cfg.train_tune_config, phase='train', ckpt_path=pretrain_ckpt)
            best_hps["train"] = train_hps
            self._apply_train_hps(run_cfg, train_hps, run_cfg.train_tune_config)
            trainer = self._make_trainer(run_cfg, model)
            # Restores pretrain_best_val_acc and inits LCA/EWC
            if pretrain_ckpt:
                trainer.load_pretrain(pretrain_ckpt)

        # ── Train + evaluate ──────────────────────────────────────────
        trainer.train()
        tr_softmax, tr_proto_seen, tr_proto_novel = evaluator.eval_trained(model)

        # ── Pack RunScores ────────────────────────────────────────────
        run_scores = evaluator.collect(
            run_id               = run_cfg.run_id,
            paradigm             = run_cfg.paradigm,
            arch                 = run_cfg.arch,
            pretrain_softmax     = pre_softmax,
            pretrain_proto_seen  = pre_proto_seen,
            pretrain_proto_novel = pre_proto_novel,
            trained_softmax      = tr_softmax,
            trained_proto_seen   = tr_proto_seen,
            trained_proto_novel  = tr_proto_novel,
        )

        # ── Build RunResult ───────────────────────────────────────────
        end_time     = time.time()
        end_time_str = datetime.datetime.now().isoformat()

        result = RunResult(
            run_id               = run_cfg.run_id,
            paradigm             = run_cfg.paradigm,
            arch                 = run_cfg.arch,
            model_config         = run_cfg.model_config.to_dict(),
            train_config         = run_cfg.train_config.to_dict(),
            eval_config          = run_cfg.eval_config.to_dict(),
            exec_config          = self.exec_config.to_dict(),
            full_tune_config     = run_cfg.full_tune_config.to_dict()     if run_cfg.full_tune_config     else None,
            pretrain_tune_config = run_cfg.pretrain_tune_config.to_dict() if run_cfg.pretrain_tune_config else None,
            train_tune_config    = run_cfg.train_tune_config.to_dict()    if run_cfg.train_tune_config    else None,
            pretrain_path        = trainer.state.pretrain_export_path or None,
            final_model_path     = trainer.state.final_export_path    or None,
            training_state       = trainer.state.to_dict(),
            training_history     = trainer.history.to_dict(),
            run_scores           = run_scores.to_dict(),
            best_hps             = best_hps or None,
            start_time           = start_time_str,
            end_time             = end_time_str,
            duration_seconds     = end_time - start_time,
            random_seed          = run_cfg.random_seed,
        )

        self._cleanup(run_cfg.run_id, trainer)
        return result

    # ------------------------------------------------------------------
    # Tuner helpers
    # ------------------------------------------------------------------

    def _run_tuner(self, run_cfg: ExperimentConfig, tune_cfg: TuneConfig,
                   phase: str, ckpt_path: Optional[str]) -> dict:
        """
        Run HPTuner for given phase. Returns best_hps dict.

        phase: 'pretrain' — proxy pretrain trials
               'train'    — reload checkpoint + train trials
               'full'     — full pretrain+train per trial
        """
        tuner = HPTuner(
            model_config         = run_cfg.model_config,
            train_config         = run_cfg.train_config,
            tune_config          = tune_cfg,
            factory              = self.factory,
            device               = self.device,
            run_id               = run_cfg.run_id,
            paradigm             = run_cfg.paradigm,
            phase                = phase,
            logs_dir             = self.exec_config.logs_dir,
            load_checkpoint_path = ckpt_path,
        )
        return tuner.run()

    def _run_split_tuner_for_tune_all(self, run_cfg: ExperimentConfig):
        """
        Run pretrain+train split tuners for tune_all().
        Returns (pretrain_hps, train_hps, pretrain_ckpt_path).
        Pretrain checkpoint created here for train tuner — caller cleans it up.
        """
        pretrain_hps, train_hps = {}, {}
        pretrain_ckpt = None

        if run_cfg.pretrain_tune_config is not None:
            pretrain_hps = self._run_tuner(run_cfg, run_cfg.pretrain_tune_config, phase='pretrain', ckpt_path=None)
            print(f"  Pretrain best HPs: {pretrain_hps}")

        if run_cfg.train_tune_config is not None:
            # Run full pretrain once to produce checkpoint for train tuner
            tmp_cfg = copy.deepcopy(run_cfg)
            if pretrain_hps and 'trainer' in pretrain_hps:
                self._apply_train_hps(tmp_cfg, pretrain_hps['trainer'], run_cfg.pretrain_tune_config)

            tmp_model   = ModelFactory.create(tmp_cfg.model_config, device=self.device)
            tmp_trainer = self._make_trainer(tmp_cfg, tmp_model)
            tmp_trainer.pretrain()
            pretrain_ckpt = tmp_trainer.state.pretrain_export_path or None
            if pretrain_ckpt:
                print(f"  Pretrain checkpoint: {pretrain_ckpt}")

            train_hps = self._run_tuner(run_cfg, run_cfg.train_tune_config, phase='train', ckpt_path=pretrain_ckpt)
            print(f"  Train best HPs: {train_hps}")

        return pretrain_hps, train_hps, pretrain_ckpt

    # ------------------------------------------------------------------
    # HP application helpers
    # ------------------------------------------------------------------

    def _apply_model_hps(self, model, run_cfg: ExperimentConfig, best_hps: dict, tune_cfg: TuneConfig):
        """
        Apply model_hp_choices HPs — rebuild model via ModelConfig.update_config.
        Returns (updated_model, updated_run_cfg).
        Model must be recreated since structural HPs require fresh instantiation.
        """
        model_hps = best_hps['model'] if 'model' in best_hps else None
        if model_hps:
            run_cfg     = copy.deepcopy(run_cfg)
            updated_cfg = ModelConfig.update_config(run_cfg.model_config, **model_hps)
            model       = ModelFactory.create(updated_cfg, device=self.device)
        return model, run_cfg

    def _apply_train_hps(self, run_cfg: ExperimentConfig, best_hps: dict, tune_cfg: TuneConfig) -> None:
        """
        Apply train_hp_choices HPs to run_cfg.train_config via setattr.
        Modifies run_cfg.train_config in place.
        """
        trainer_hps = best_hps['trainer'] if 'trainer' in best_hps else None
        if trainer_hps:
            for k, v in trainer_hps.items():
                if hasattr(run_cfg.train_config, k):
                    setattr(run_cfg.train_config, k, v)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _make_trainer(self, run_cfg: ExperimentConfig, model):
        """Create StandardTrainer or FewShotTrainer based on paradigm."""
        if run_cfg.paradigm == 'standard':
            return StandardTrainer(model, self.factory, run_cfg.train_config, self.device, seed=run_cfg.random_seed)
        elif run_cfg.paradigm == 'fewshot':
            return FewShotTrainer(model, self.factory, run_cfg.train_config, self.device, seed=run_cfg.random_seed)
        raise ValueError(f"Unknown paradigm: '{run_cfg.paradigm}'. Use 'standard' or 'fewshot'.")

    def _has_any_tuner(self, run_cfg: ExperimentConfig) -> bool:
        """True if any tuner is configured for this run."""
        return any([run_cfg.full_tune_config,
                    run_cfg.pretrain_tune_config,
                    run_cfg.train_tune_config])

    def _dump_tune_results(self, tune_id: str, duration: float, all_hps: dict) -> None:
        """Save tune_results JSON to results_dir."""
        path = os.path.join(self.exec_config.results_dir, f"tune_results_{tune_id}.json")
        os.makedirs(self.exec_config.results_dir, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'tune_id': tune_id, 'duration_seconds': duration, 'best_hps': all_hps}, f, indent=2)
        print(f"\n  Tune results saved: {path}")

    def _seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------
    # Validation + Stamping
    # ------------------------------------------------------------------

    def _validate_and_stamp(self):
        """
        Two responsibilities, called once at end of __init__:

        1. VALIDATE — pre-flight checks across all run_configs before any run starts.
           Fails fast with clear messages so user fixes all issues at once,
           not one-per-run after hours of training.

        2. STAMP — push global fields from ExperimentConfig / ExecutionerConfig
           down into each run's TrainConfig / EvalConfig so they are consistent
           when trainer and evaluator read them.

        Stamping order (later stamps win for overlapping fields):
            ExperimentConfig.run_id      → TrainConfig.run_id
            ExecutionerConfig.checkpoint_dir → TrainConfig.checkpoint_dir
            ExecutionerConfig.num_workers    → TrainConfig.num_workers
                                             → EvalConfig.num_workers
        """
        # ── 1a. Validate run_id uniqueness ────────────────────────────
        seen_ids = {}
        for cfg in self.run_configs:
            if cfg.run_id in seen_ids:
                raise ValueError(
                    f"Duplicate run_id '{cfg.run_id}' found. "
                    f"First occurrence at index {seen_ids[cfg.run_id]}, "
                    f"duplicate at index {self.run_configs.index(cfg)}."
                )
            seen_ids[cfg.run_id] = self.run_configs.index(cfg)

        for cfg in self.run_configs:
            cfg.validate_config()

        for cfg in self.run_configs:
            tc, ec = cfg.train_config, cfg.eval_config
            mismatches = []
            if tc.n_way   != ec.n_way:   mismatches.append(f"n_way: TrainConfig={tc.n_way}, EvalConfig={ec.n_way}")
            if tc.k_shot  != ec.k_shot:  mismatches.append(f"k_shot: TrainConfig={tc.k_shot}, EvalConfig={ec.k_shot}")
            if tc.q_query != ec.q_query: mismatches.append(f"q_query: TrainConfig={tc.q_query}, EvalConfig={ec.q_query}")
            if mismatches:
                raise ValueError(
                    f"run_id='{cfg.run_id}': TrainConfig and EvalConfig episodic mismatch:\n  "
                    + "\n  ".join(mismatches)
                )

        for cfg in self.run_configs:
            cfg.train_config = copy.deepcopy(cfg.train_config)
            cfg.eval_config  = copy.deepcopy(cfg.eval_config)

            cfg.train_config.run_id         = cfg.run_id
            cfg.train_config.checkpoint_dir = self.exec_config.checkpoint_dir
            cfg.train_config.num_workers    = self.exec_config.num_workers
            cfg.eval_config.num_workers     = self.exec_config.num_workers

    def _cleanup(self, run_id: str, trainer) -> None:
        """
        Called at end of each run.
        Decides which checkpoints to keep based on exec_config flags.
        Trainer always saves both checkpoints — this method decides what to delete.
        """
        pretrain_path = trainer.state.pretrain_export_path
        final_path    = trainer.state.final_export_path

        if pretrain_path and os.path.exists(pretrain_path):
            if not self.exec_config.keep_pretrain_checkpoint:
                os.remove(pretrain_path)
                print(f"  Cleanup: removed pretrain checkpoint ({run_id})")
            else:
                print(f"  Checkpoint kept: {pretrain_path}")

        if final_path and os.path.exists(final_path):
            if not self.exec_config.keep_train_checkpoint:
                os.remove(final_path)
                print(f"  Cleanup: removed train checkpoint ({run_id})")
            else:
                print(f"  Checkpoint kept: {final_path}")

    # ------------------------------------------------------------------
    # ExperimentSummary
    # ------------------------------------------------------------------

    def _build_summary(self, experiment_id: str, start_str: str,
                        end_str: str, duration: float) -> ExperimentSummary:
        """Builds ExperimentSummary from all RunResults."""

        # Comparison table — {score_name: {run_id: top1_acc}}
        score_names = ['pretrain_softmax', 'pretrain_proto_seen', 'pretrain_proto_novel',
                       'trained_softmax',  'trained_proto_seen', 'trained_proto_novel']
        comparison: Dict[str, Dict[str, float]] = {s: {} for s in score_names}

        for run_id, result in self.run_results.items():
            scores = result.run_scores
            for score_name in score_names:
                score_dict = scores.get(score_name, {})
                comparison[score_name][run_id] = score_dict.get('top1_acc', None)

        # Hardware
        device_name   = 'cpu'
        device_mem_gb = None
        if torch.cuda.is_available():
            device_name   = torch.cuda.get_device_name(0)
            device_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        return ExperimentSummary(
            experiment_id    = experiment_id,
            runs             = [r.to_dict() for r in self.run_results.values()],
            comparison_table = comparison,
            experiment_start = start_str,
            experiment_end   = end_str,
            total_duration   = duration,
            device_name      = device_name,
            device_memory_gb = device_mem_gb,
        )


# ==============================================================================
# ResultStore
# ==============================================================================

class ResultStore:
    """
    Serialize and deserialize RunResult and ExperimentSummary.
    JSON for full data, CSV for score comparison table.

    Usage:
        ResultStore.save_run(result, 'results/run1.json')
        ResultStore.save_summary(summary, 'results/summary.json')
        result  = ResultStore.load_run('results/run1.json')
        summary = ResultStore.load_summary('results/summary.json')
        ResultStore.scores_to_csv(summary, 'results/scores.csv')
    """

    @staticmethod
    def save_run(result: RunResult, path: str):
        result.to_json(path)

    @staticmethod
    def load_run(path: str) -> RunResult:
        return RunResult.from_json(path)

    @staticmethod
    def save_summary(summary: ExperimentSummary, path: str):
        summary.to_json(path)

    @staticmethod
    def load_summary(path: str) -> ExperimentSummary:
        return ExperimentSummary.from_json(path)

    @staticmethod
    def scores_to_csv(summary: ExperimentSummary, path: str):
        """Export comparison table to CSV. Rows=scores, Columns=runs."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        df = pd.DataFrame(summary.comparison_table).T
        df.index.name = 'score'
        df.to_csv(path)
        print(f"Scores CSV: {path}")

    @staticmethod
    def load_all_runs(results_dir: str) -> List[RunResult]:
        """Load all RunResult JSON files from directory."""
        results = []
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith('_result.json'):
                results.append(RunResult.from_json(os.path.join(results_dir, fname)))
        return results
