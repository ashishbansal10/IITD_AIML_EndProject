"""
experiment.py
=============
Experiment orchestration — runs all 6 experiments, collects results, saves everything.

Classes
-------
ExperimentConfig    — one run definition
ExecutionerConfig   — global run strategy (checkpoint policy, dirs, parallel flag)
RunResult           — complete per-run saved data including RunScores
ExperimentSummary   — all RunResults + experiment-level runtime + hardware info
ExperimentRunner    — executioner: tune → pretrain → eval → train → eval → save
ResultStore         — serialize/deserialize RunResult + ExperimentSummary
Plotter             — learning curves per run, score comparison across runs

6 Runs
------
Run 1 — CNN      Standard
Run 2 — CNN      FewShot
Run 3 — GNN      Standard   [PLACEHOLDER — Phase 6]
Run 4 — GNN      FewShot    [PLACEHOLDER — Phase 6]
Run 5 — Hybrid   Standard   [PLACEHOLDER — Phase 6]
Run 6 — Hybrid   FewShot    [PLACEHOLDER — Phase 6]

Per-Run Flow in ExperimentRunner
---------------------------------
    if tune_config:
        tuner.run() → best_hps → apply to model + train_config
    pretrain()        → model at best-pretrain-epoch on return (internal)
    eval_pretrain()
    train()           → model at best-train-epoch on return (internal)
    eval_trained()
    collect RunScores → RunResult → save to JSON

Eval Immediately After Each Phase
----------------------------------
Trainer internally loads best-epoch weights before pretrain()/train() return.
Model is already at best state when control returns to runner — no load_best() needed.
Export paths (if saved) are read from trainer.state.pretrain_export_path / final_export_path.

Parallel vs Sequential
-----------------------
run_mode='sequential' — one run at a time (default, single GPU)
run_mode='parallel'   — [FUTURE] multi-GPU only

[FUTURE] parallel mode: multiple runs share one GPU but sequentially is
         equivalent on single H100 — no speedup from parallelism on one device.
         Config option left in place for future multi-GPU support.

device passed from notebook — never auto-detected inside any class.

Required Libraries
------------------
# torch>=2.0.0
# matplotlib>=3.5.0  # Plotter — pip install matplotlib
# pandas>=1.5.0      # ResultStore CSV — pip install pandas
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

from trainer    import (TrainConfig, TrainingState, TrainingHistory,
                         StandardTrainer, FewShotTrainer)
from evaluator  import EvalConfig, EvalResult, RunScores, Evaluator
from tuner      import TuneConfig, HPTuner
from model_factory import ModelConfig, ModelFactory


# ==============================================================================
# ExperimentConfig — one run definition
# ==============================================================================

@dataclass
class ExperimentConfig:
    """
    Defines one complete run.
    6 of these for 6 runs — one per architecture × paradigm combination.

    Fields:
        run_id       : unique string e.g. 'run1_cnn_standard'
        paradigm     : 'standard' or 'fewshot'
        arch         : 'cnn', 'gnn', 'hybrid' — for grouping in results
        model_config : ModelConfig instance
        train_config : TrainConfig instance
        eval_config  : EvalConfig instance
        tune_config  : TuneConfig or None — None skips tuning
        random_seed  : for reproducibility
        notes        : free-text notes about this run
    """
    run_id:       str
    paradigm:     str
    arch:         str            # 'cnn', 'gnn', 'hybrid'
    model_config: ModelConfig
    train_config: TrainConfig
    eval_config:  EvalConfig
    tune_config:  Optional[TuneConfig] = None   # None = no tuning
    random_seed:  int                  = 42
    notes:        str                  = ''

    def to_dict(self) -> dict:
        return {
            'run_id':       self.run_id,
            'paradigm':     self.paradigm,
            'arch':         self.arch,
            'model_config': self.model_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'eval_config':  self.eval_config.to_dict(),
            'tune_config':  self.tune_config.to_dict() if self.tune_config else None,
            'random_seed':  self.random_seed,
            'notes':        self.notes,
        }
    
    def validate_config(self):
        """Validates field values at construction time."""
        # run_id: non-empty, no whitespace (would break filenames)
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
                f"ExperimentConfig.paradigm='{self.paradigm}' invalid. "
                f"Valid: {valid_paradigms}"
            )

        valid_archs = {'cnn', 'gnn', 'hybrid'}
        if self.arch not in valid_archs:
            raise ValueError(
                f"ExperimentConfig.arch='{self.arch}' invalid. "
                f"Valid: {valid_archs}"
            )


# ==============================================================================
# ExecutionerConfig — global run strategy
# ==============================================================================

@dataclass
class ExecutionerConfig:
    """
    Global strategy — applies equally to all runs.
    One instance for whole experiment.

    Fields:
        run_mode       : 'sequential' (default) or 'parallel' [FUTURE]
        checkpoint_dir : model checkpoint directory — stamped into each TrainConfig
        results_dir    : RunResult JSON + ExperimentSummary directory
        plots_dir      : saved plot directory
        num_workers    : DataLoader workers — stamped into TrainConfig + EvalConfig
                         override per-run by setting TrainConfig.num_workers after stamping
        max_parallel   : [FUTURE] max parallel runs, leave at 1 for now

    """
    run_mode:       str  = 'sequential'
    checkpoint_dir: str  = 'checkpoints'
    results_dir:    str  = 'results'
    plots_dir:      str  = 'plots'
    num_workers:    int  = 2        # stamped into TrainConfig + EvalConfig by Runner
    max_parallel:   int  = 1        # [FUTURE] >1 when multi-GPU available

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# RunResult — complete per-run saved data
# ==============================================================================

@dataclass
class RunResult:
    """
    Complete saved data for one run.
    Serialized to JSON immediately after each run completes.

    All configs stored as dicts — no class dependencies at load time.
    RunScores stored as nested dict — EvalResult fields directly accessible.

    Per-run runtime only — experiment-level runtime in ExperimentSummary.
    """

    # Identity
    run_id:           str
    paradigm:         str
    arch:             str

    # Configs — as dicts for portability
    model_config:     dict
    train_config:     dict
    eval_config:      dict
    exec_config:      dict
    tune_config:      Optional[dict]   # None if no tuning

    # Final HPs — documents what was actually used
    # {alias: {param: value}} for all model components
    # Useful even without Optuna — records defaults + any tuned values
    final_hps:        dict

    # Checkpoints
    pretrain_path:    Optional[str]    # None if pretrain_save_mode='none'
    final_model_path: Optional[str]    # None if keep_final=False

    # Training outcome
    training_state:   dict             # TrainingState serialized
    training_history: dict             # TrainingHistory serialized

    # Scores — RunScores serialized
    run_scores:       dict

    # Best HPs from tuner (None if no tuning)
    best_hps:         Optional[dict]

    # Per-run runtime
    start_time:       str
    end_time:         str
    duration_seconds: float
    random_seed:      int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True
        )
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

    comparison_table:
        {score_name: {run_id: top1_acc}}
        e.g. {'trained_proto_novel': {'run1_cnn_standard': 0.623,
                                       'run2_cnn_fewshot':  0.651, ...}}
        Used by Plotter for cross-run comparison.
    """

    experiment_id:    str
    runs:             List[dict]        # List of RunResult.to_dict()
    comparison_table: dict              # {score_name: {run_id: top1_acc}}

    # Experiment-level runtime
    experiment_start: str
    experiment_end:   str
    total_duration:   float             # seconds

    # Hardware + library
    device_name:      str
    device_memory_gb: Optional[float]
    torch_version:    str
    python_version:   str
    platform_info:    str
    easyfsl_version:  Optional[str] = None
    pyg_version:      Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True
        )
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentSummary':
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ==============================================================================
# ExperimentRunner — executioner
# ==============================================================================

class ExperimentRunner:
    """
    Orchestrates all runs end to end.

    Per run:
        1. HP tuning (if tune_config set) → best_hps
        2. Apply best_hps to model + train_config
        3. pretrain()     → model at best-pretrain-epoch on return
           eval_pretrain()
        4. train()        → model at best-train-epoch on return
           eval_trained()
        5. Collect RunScores → RunResult → save JSON

    After all runs:
        Build ExperimentSummary → save JSON

    Usage:
        run_configs = [run1, run2, run3, run4, run5, run6]
        exec_config = ExecutionerConfig()
        runner      = ExperimentRunner(run_configs, exec_config, factory, device)
        summary     = runner.run_all()

    Access individual results:
        runner.run_results['run1_cnn_standard']
    """

    def __init__(self,
                 run_configs:  List[ExperimentConfig],
                 exec_config:  ExecutionerConfig,
                 factory,
                 device:       torch.device):
        self.run_configs  = run_configs
        self.exec_config  = exec_config
        self.factory      = factory
        self.device       = device
        self.run_results: Dict[str, RunResult] = {}

        os.makedirs(exec_config.checkpoint_dir, exist_ok=True)
        os.makedirs(exec_config.results_dir,    exist_ok=True)
        os.makedirs(exec_config.plots_dir,      exist_ok=True)

        self.validate_and_stamp()

    def validate_and_stamp(self):
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
                    f"run_id must be unique across all ExperimentConfigs. "
                    f"First occurrence at index {seen_ids[cfg.run_id]}, "
                    f"duplicate at index {self.run_configs.index(cfg)}."
                )
            seen_ids[cfg.run_id] = self.run_configs.index(cfg)

        # ── 1b. Validate paradigm / arch consistency ──────────────────
        for cfg in self.run_configs:
            cfg.validate_config()

        # ── 1c. Validate episodic protocol consistency ────────────────
        # n_way / k_shot / q_query must agree between TrainConfig and EvalConfig.
        # They are separate fields (rule 5) but must not silently diverge.
        for cfg in self.run_configs:
            tc = cfg.train_config
            ec = cfg.eval_config
            mismatches = []
            if tc.n_way   != ec.n_way:
                mismatches.append(f"n_way: TrainConfig={tc.n_way}, EvalConfig={ec.n_way}")
            if tc.k_shot  != ec.k_shot:
                mismatches.append(f"k_shot: TrainConfig={tc.k_shot}, EvalConfig={ec.k_shot}")
            if tc.q_query != ec.q_query:
                mismatches.append(f"q_query: TrainConfig={tc.q_query}, EvalConfig={ec.q_query}")
            if mismatches:
                raise ValueError(
                    f"run_id='{cfg.run_id}': TrainConfig and EvalConfig episodic protocol "
                    f"mismatch — must be identical:\n  " + "\n  ".join(mismatches)
                )

        # ── 2. Stamp global fields into each run's sub-configs ────────
        for cfg in self.run_configs:
            cfg.train_config = copy.deepcopy(cfg.train_config)
            cfg.eval_config  = copy.deepcopy(cfg.eval_config)

            cfg.train_config.run_id         = cfg.run_id
            cfg.train_config.checkpoint_dir = self.exec_config.checkpoint_dir
            cfg.train_config.num_workers    = self.exec_config.num_workers
            cfg.eval_config.num_workers     = self.exec_config.num_workers

    def run_all(self) -> ExperimentSummary:
        """
        Run all experiments sequentially.
        Returns ExperimentSummary after all complete.
        """
        if self.exec_config.run_mode == 'parallel':
            raise NotImplementedError(
                "run_mode='parallel' not yet implemented.\n"
                "Use run_mode='sequential' (default).\n"
                "[FUTURE] Parallel support for multi-GPU setup."
            )

        experiment_id    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_start        = time.time()
        exp_start_str    = datetime.datetime.now().isoformat()

        print(f"\n{'='*70}")
        print(f"EXPERIMENT START — {experiment_id}")
        print(f"Runs: {len(self.run_configs)}")
        print(f"{'='*70}")

        for i, run_cfg in enumerate(self.run_configs):
            print(f"\n[{i+1}/{len(self.run_configs)}] {run_cfg.run_id}")
            result = self._run_single(run_cfg)
            self.run_results[run_cfg.run_id] = result

            path = os.path.join(self.exec_config.results_dir, f"{run_cfg.run_id}_result.json")
            result.to_json(path)
            print(f"  Saved: {path}")

        exp_end     = time.time()
        exp_end_str = datetime.datetime.now().isoformat()
        duration    = exp_end - exp_start

        summary = self._build_summary( experiment_id, exp_start_str, exp_end_str, duration )
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
            train → eval trained → collect RunResult
        """
        torch.manual_seed(run_cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_cfg.random_seed)

        start_time     = time.time()
        start_time_str = datetime.datetime.now().isoformat()

        # ── Create model ──────────────────────────────────────────────
        model = ModelFactory.create(run_cfg.model_config, device=self.device)
        print(f"  Model: {model}")

        # ── HP Tuning (optional) ──────────────────────────────────────
        best_hps = None
        if run_cfg.tune_config is not None:
            print(f"\n  HP Tuning...")
            tuner    = HPTuner(
                model_config = run_cfg.model_config,
                train_config = run_cfg.train_config,
                tune_config  = run_cfg.tune_config,
                factory      = self.factory,
                device       = self.device
            )
            best_hps = tuner.run()

            # Apply best HPs to model and train_config
            # dropout_rate → backbone component
            if 'dropout_rate' in best_hps:
                model.get_component('backbone').set_hp(dropout_rate=best_hps['dropout_rate'])
            # lr → train_config copy (don't mutate original)
            if 'lr' in best_hps:
                run_cfg = copy.deepcopy(run_cfg)
                run_cfg.train_config.lr = best_hps['lr']

            tuner.print_summary()

        # ── Create trainer ────────────────────────────────────────────
        if run_cfg.paradigm == 'standard':
            trainer = StandardTrainer( model, self.factory, run_cfg.train_config, self.device )
        elif run_cfg.paradigm == 'fewshot':
            trainer = FewShotTrainer( model, self.factory, run_cfg.train_config, self.device )
        else:
            raise ValueError(
                f"Unknown paradigm: '{run_cfg.paradigm}'. "
                f"Use 'standard' or 'fewshot'."
            )

        # Create evaluator
        evaluator = Evaluator(self.factory, run_cfg.eval_config, self.device)

        print(f"\n  Phase 1: Pretrain")
        trainer.pretrain()
        # pretrain() loads best-epoch weights and handles export internally.
        # state.pretrain_export_path is set (path or '' depending on pretrain_save_mode).
        pre_softmax, pre_proto = evaluator.eval_pretrain(model)

        print(f"\n  Phase 2: Train [{run_cfg.paradigm}]")
        trainer.train()
        # train() loads best-epoch weights and handles export internally.
        # state.final_export_path is set (path or '' depending on keep_final).
        tr_softmax, tr_proto_seen, tr_proto_novel = evaluator.eval_trained(model)


        # ── Pack RunScores ────────────────────────────────────────────
        run_scores = evaluator.collect(
            run_id              = run_cfg.run_id,
            paradigm            = run_cfg.paradigm,
            arch                = run_cfg.arch,
            pretrain_softmax    = pre_softmax,
            pretrain_proto      = pre_proto,
            trained_softmax     = tr_softmax,
            trained_proto_seen  = tr_proto_seen,
            trained_proto_novel = tr_proto_novel,
        )

        # ── Final HPs ────────────────────────────────────────────────
        final_hps = {
            alias: model.get_component(alias).get_hp()
            for alias in model.component_names()
        }

        # ── Build RunResult ───────────────────────────────────────────
        end_time     = time.time()
        end_time_str = datetime.datetime.now().isoformat()

        result = RunResult(
            run_id           = run_cfg.run_id,
            paradigm         = run_cfg.paradigm,
            arch             = run_cfg.arch,
            model_config     = run_cfg.model_config.to_dict(),
            train_config     = run_cfg.train_config.to_dict(),
            eval_config      = run_cfg.eval_config.to_dict(),
            exec_config      = self.exec_config.to_dict(),
            tune_config      = run_cfg.tune_config.to_dict()
                               if run_cfg.tune_config else None,
            final_hps        = final_hps,
            # '' means file was not saved (pretrain_save_mode='none'/keep_final=False).
            pretrain_path    = trainer.state.pretrain_export_path or None,
            final_model_path = trainer.state.final_export_path    or None,
            training_state   = trainer.state.to_dict(),
            training_history = trainer.history.to_dict(),
            run_scores       = run_scores.to_dict(),
            best_hps         = best_hps,
            start_time       = start_time_str,
            end_time         = end_time_str,
            duration_seconds = end_time - start_time,
            random_seed      = run_cfg.random_seed,
        )

        print(f"\n  {run_cfg.run_id} done ({(end_time-start_time)/60:.1f} min)")
        return result

    # ------------------------------------------------------------------
    # ExperimentSummary
    # ------------------------------------------------------------------

    def _build_summary(self,
                        experiment_id: str,
                        start_str:     str,
                        end_str:       str,
                        duration:      float) -> ExperimentSummary:
        """Builds ExperimentSummary from all RunResults."""

        # Comparison table — {score_name: {run_id: top1_acc}}
        score_names = ['pretrain_softmax', 'pretrain_proto',
                       'trained_softmax',  'trained_proto_seen',
                       'trained_proto_novel']
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

        # Library versions
        easyfsl_ver = None
        pyg_ver     = None
        try:
            import easyfsl
            easyfsl_ver = getattr(easyfsl, '__version__', 'unknown')
        except ImportError:
            pass
        try:
            import torch_geometric
            pyg_ver = getattr(torch_geometric, '__version__', 'unknown')
        except ImportError:
            pass

        return ExperimentSummary(
            experiment_id    = experiment_id,
            runs             = [r.to_dict() for r in self.run_results.values()],
            comparison_table = comparison,
            experiment_start = start_str,
            experiment_end   = end_str,
            total_duration   = duration,
            device_name      = device_name,
            device_memory_gb = device_mem_gb,
            torch_version    = torch.__version__,
            python_version   = platform.python_version(),
            platform_info    = platform.platform(),
            easyfsl_version  = easyfsl_ver,
            pyg_version      = pyg_ver,
        )


# ==============================================================================
# ResultStore
# ==============================================================================

class ResultStore:
    """
    Serialize and deserialize RunResult and ExperimentSummary.
    JSON for full data, CSV for score comparison table.

    Usage:
        ResultStore.save_run(result,  'results/run1.json')
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

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True
        )
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
                results.append(RunResult.from_json(
                    os.path.join(results_dir, fname)
                ))
        return results


# ==============================================================================
# Plotter
# ==============================================================================

class Plotter:
    """
    Visualizations for training results.

    Plots:
        learning_curves(result)       — train/val loss+acc for one run
        score_comparison(summary)     — bar chart: all 5 scores × all 6 runs
        score_table(summary)          — formatted text table to stdout

    Usage:
        plotter = Plotter(plots_dir='plots')
        plotter.learning_curves(result)
        plotter.score_comparison(summary)
        plotter.score_table(summary)
    """

    def __init__(self, plots_dir: str = 'plots'):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def learning_curves(self, result: RunResult, show: bool = True):
        """Train/val loss and accuracy curves for one run."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        h   = result.training_history
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Learning Curves — {result.run_id}", fontsize=14)

        # Pretrain
        if h.get('pretrain_epochs'):
            ep = h['pretrain_epochs']
            axes[0][0].plot(ep, h['pretrain_train_loss'], label='train')
            axes[0][0].plot(ep, h['pretrain_val_loss'],   label='val')
            axes[0][0].set_title('Pretrain Loss')
            axes[0][0].set_xlabel('Epoch')
            axes[0][0].legend()

            axes[1][0].plot(ep, h['pretrain_train_acc'], label='train')
            axes[1][0].plot(ep, h['pretrain_val_acc'],   label='val')
            axes[1][0].set_title('Pretrain Accuracy')
            axes[1][0].set_xlabel('Epoch')
            axes[1][0].legend()

        # Train
        if h.get('train_epochs'):
            ep = h['train_epochs']
            axes[0][1].plot(ep, h['train_loss'], label='train')
            axes[0][1].plot(ep, h['val_loss'],   label='val')
            axes[0][1].set_title(f"Train Loss [{result.paradigm}]")
            axes[0][1].set_xlabel('Epoch')
            axes[0][1].legend()

            axes[1][1].plot(ep, h['train_acc'], label='train')
            axes[1][1].plot(ep, h['val_acc'],   label='val')
            axes[1][1].set_title(f"Train Accuracy [{result.paradigm}]")
            axes[1][1].set_xlabel('Epoch')
            axes[1][1].legend()

        plt.tight_layout()
        path = os.path.join(self.plots_dir, f"{result.run_id}_curves.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        if show: plt.show()
        plt.close()

    def score_comparison(self, summary: ExperimentSummary, show: bool = True):
        """
        Bar chart: 5 scores × 6 runs.
        Primary metric trained_proto_novel highlighted.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("pip install matplotlib")

        ct       = summary.comparison_table
        scores   = list(ct.keys())
        run_ids  = sorted(next(iter(ct.values())).keys())
        x        = np.arange(len(scores))
        width    = 0.8 / max(len(run_ids), 1)

        fig, ax = plt.subplots(figsize=(16, 6))
        for i, run_id in enumerate(run_ids):
            values = [ct[s].get(run_id) or 0 for s in scores]
            ax.bar(x + i * width, values, width, label=run_id, alpha=0.8)

        ax.set_xlabel('Score')
        ax.set_ylabel('Accuracy (top-1)')
        ax.set_title('Score Comparison Across All Runs')
        ax.set_xticks(x + width * (len(run_ids) - 1) / 2)
        ax.set_xticklabels(scores, rotation=30, ha='right')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        # Highlight primary metric
        primary_idx = scores.index('trained_proto_novel') \
                      if 'trained_proto_novel' in scores else -1
        if primary_idx >= 0:
            ax.axvspan(primary_idx - 0.4, primary_idx + len(run_ids) * width + 0.1,
                       alpha=0.08, color='gold', label='primary metric')

        plt.tight_layout()
        path = os.path.join(self.plots_dir,
                            f"score_comparison_{summary.experiment_id}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        if show: plt.show()
        plt.close()

    def score_table(self, summary: ExperimentSummary):
        """Formatted 9-score comparison table printed to stdout."""
        ct      = summary.comparison_table
        scores  = list(ct.keys())
        run_ids = sorted(next(iter(ct.values())).keys())

        col_w  = 14
        header = f"{'Score':<22}" + ''.join(f"{r[:col_w]:>{col_w}}" for r in run_ids)
        div    = '─' * len(header)

        print(f"\n{'='*len(header)}")
        print(f"SCORE COMPARISON — {summary.experiment_id}")
        print(f"{'='*len(header)}")
        print(header)
        print(div)

        for score in scores:
            marker = ' ←' if score == 'trained_proto_novel' else ''
            row    = f"{score+marker:<22}"
            for run_id in run_ids:
                val = ct[score].get(run_id)
                row += f"{val:>{col_w}.4f}" if val is not None else f"{'N/A':>{col_w}}"
            print(row)

        print(f"{'='*len(header)}")
        print(f"← primary metric: novel class generalization\n")
