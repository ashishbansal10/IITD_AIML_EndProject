import os
import copy
import shutil
import torch
import json
import pandas as pd

from model_factory import ModelConfig, ModelFactory
from trainer    import TrainConfig
from evaluator  import EvalConfig
from tuner      import TuneConfig
from experiment import ExperimentConfig, ExecutionerConfig, ExperimentRunner, Plotter, ResultStore


# ==============================================================================
# Shared helpers
# ==============================================================================

def _n_classes(loader_factory):
    c2i = loader_factory.splitter.get_class_to_indices('pretrain')
    return max(c2i.keys()) + 1


def _cleanup(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Cleaned: {d}")


def _print_results(runner, summary, plots_dir):
    plotter = Plotter(plots_dir=plots_dir)
    plotter.score_table(summary)
    plotter.score_comparison(summary, show=False)
    for run_id, result in runner.run_results.items():
        ts = result.training_state
        print(f"  {result.run_id:<35} "
              f"pretrain_acc={ts.get('pretrain_best_val_acc','N/A')}  "
              f"train_acc={ts.get('best_val_acc','N/A'):.4f}  "
              f"{result.duration_seconds/60:.1f}min")


# ==============================================================================
# smoke_test0 — Forward pass only, all 3 archs × 4 modes  (<30s, no disk)
# ==============================================================================

def run_smoke_test0(loader_factory, device):
    """
    Fastest smoke — no training, no disk writes.
    Verifies model construction and all 4 forward modes
    for CNN, GNN, Hybrid architectures.
    """
    print(f"\n{'='*55}")
    print(f"Smoke Test 0 — Forward pass  |  device={device}")
    print(f"{'='*55}")

    n_classes = _n_classes(loader_factory)
    print(f"n_classes = {n_classes}\n")

    B, C, H, W = 4, 3, 84, 84
    N, K, Q    = 5, 1, 5

    configs = {
        'cnn'   : ModelConfig.cnn_config(n_classes=n_classes, n_way=N, k_shot=K),
        'gnn'   : ModelConfig.gnn_config(n_classes=n_classes, n_way=N, k_shot=K),
        'hybrid': ModelConfig.hybrid_config(n_classes=n_classes, n_way=N, k_shot=K),
    }

    imgs    = torch.randn(B, C, H, W).to(device)
    support = torch.randn(N * K, C, H, W).to(device)
    query   = torch.randn(N * Q, C, H, W).to(device)

    all_ok = True
    for arch, cfg in configs.items():
        print(f"  [{arch}]")
        try:
            model = ModelFactory.create(cfg, device=device)
            model.eval()
            with torch.no_grad():
                emb   = model(imgs, mode='embedding')
                lin   = model(imgs, mode='linear')
                soft  = model(imgs, mode='softmax')
                s_emb = model(support, mode='embedding')
                q_emb = model(query,   mode='embedding')
                dist  = model(support_emb=s_emb, query_emb=q_emb, mode='prototypical')

            assert emb.shape  == (B, 640),           f"embedding: {emb.shape}"
            assert lin.shape  == (B, n_classes),      f"linear: {lin.shape}"
            assert soft.shape == (B, n_classes),      f"softmax: {soft.shape}"
            assert abs(soft.sum(1).mean().item()-1.0) < 1e-4, "softmax sum != 1"
            assert dist.shape == (N * Q, N),          f"prototypical: {dist.shape}"

            print(f"    embedding    : {tuple(emb.shape)}  ✓")
            print(f"    linear       : {tuple(lin.shape)}  ✓")
            print(f"    softmax      : {tuple(soft.shape)}  sum={soft.sum(1).mean():.4f}  ✓")
            print(f"    prototypical : {tuple(dist.shape)}  ✓")
            del model
        except Exception as e:
            print(f"    FAILED: {e}")
            all_ok = False

    print(f"\n{'All forward modes OK' if all_ok else 'SOME TESTS FAILED'}.")
    print("Smoke Test 0 done. No disk writes.\n")
    return all_ok


# ==============================================================================
# smoke_test1 — CNN pipeline, Lightning pretrain + Optuna  (~5 min)
# ==============================================================================

def run_smoke_test1(loader_factory, device, num_workers=0):
    """
    Smoke test 1 — CNN Standard (Lightning + Optuna) + CNN FewShot (PyTorch)
    Verifies: Lightning backend, Optuna proxy, result serialisation.
    Cleans up all disk files on completion.
    """
    DIRS = ['smoke1_checkpoints', 'smoke1_results', 'smoke1_plots']

    print(f"\n{'='*55}")
    print(f"Smoke Test 1 — Lightning + Optuna  |  device={device}")
    print(f"{'='*55}\n")

    n_classes = _n_classes(loader_factory)
    m_cfg     = ModelConfig.test_config(n_classes=n_classes, n_way=5, k_shot=5)

    t_cfg = TrainConfig(
        epochs_pretrain    = 3,
        epochs_train       = 2,
        episodes_train     = 5,
        episodes_val       = 3,
        batch_size         = 32,
        num_workers        = num_workers,
        backend_pretrain   = 'lightning',
        pretrain_save_mode = 'none',
        keep_final         = False,
    )

    eval_cfg = EvalConfig(
        n_episodes_seen=3, n_episodes_novel=5,
        n_way=5, k_shot=5, q_query=15,
        batch_size=32, num_workers=num_workers,
    )

    # proxy_epochs=2 overrides the hardcoded max(10,...) in tuner._objective()
    tune_cfg = TuneConfig(
        n_trials        = 2,
        study_name      = 'smoke1_hp',
        storage         = None,
        pruning         = False,
        dropout_choices = [0.0],
        lr_choices      = [1e-4, 1e-3],
        proxy_epochs    = 2,            # ← overrides hardcoded max(10,...) in tuner
    )

    run_configs = [
        ExperimentConfig(
            run_id='smoke1_r1_cnn_standard', paradigm='standard', arch='cnn',
            model_config=m_cfg,
            train_config=copy.deepcopy(t_cfg), eval_config=copy.deepcopy(eval_cfg),
            tune_config=tune_cfg, random_seed=42,
            notes='Smoke1 — Lightning + Optuna',
        ),
        ExperimentConfig(
            run_id='smoke1_r2_cnn_fewshot', paradigm='fewshot', arch='cnn',
            model_config=m_cfg,
            train_config=copy.deepcopy(t_cfg), eval_config=copy.deepcopy(eval_cfg),
            tune_config=None, random_seed=42,
            notes='Smoke1 — no tuning',
        ),
    ]

    exec_cfg = ExecutionerConfig(
        checkpoint_dir=DIRS[0], results_dir=DIRS[1],
        plots_dir=DIRS[2], num_workers=num_workers,
    )

    runner  = ExperimentRunner(run_configs, exec_cfg, loader_factory, device)
    summary = runner.run_all()
    _print_results(runner, summary, DIRS[2])
    ResultStore.scores_to_csv(summary, f'{DIRS[1]}/scores.csv')

    print("\nCleaning up...")
    _cleanup(*DIRS)
    print("Smoke Test 1 done.\n")


# ==============================================================================
# smoke_test2 — All 6 architectures, PyTorch only  (~5 min)
# ==============================================================================

def run_smoke_test2(loader_factory, device, num_workers=0):
    """
    Smoke test 2 — All 6 runs (CNN + GNN + Hybrid x Standard + FewShot)
    Pure PyTorch, no Optuna, 2 epochs only.
    Verifies: GNN/Hybrid forward pass, episodic combined call, all eval paths.
    Cleans up all disk files on completion.
    """
    DIRS = ['smoke2_checkpoints', 'smoke2_results', 'smoke2_plots']

    print(f"\n{'='*55}")
    print(f"Smoke Test 2 — All 6 runs, PyTorch  |  device={device}")
    print(f"{'='*55}\n")

    n_classes  = _n_classes(loader_factory)
    cnn_cfg    = ModelConfig.cnn_config(n_classes=n_classes, n_way=5, k_shot=5)
    gnn_cfg    = ModelConfig.gnn_config(n_classes=n_classes, n_way=5, k_shot=5)
    hybrid_cfg = ModelConfig.hybrid_config(n_classes=n_classes, n_way=5, k_shot=5)

    t_cfg = TrainConfig(
        epochs_pretrain    = 2,
        epochs_train       = 2,
        episodes_train     = 5,
        episodes_val       = 3,
        batch_size         = 32,
        num_workers        = num_workers,
        backend_pretrain   = 'pytorch',
        pretrain_save_mode = 'none',
        keep_final         = False,
    )

    eval_cfg = EvalConfig(
        n_episodes_seen=3, n_episodes_novel=5,
        n_way=5, k_shot=5, q_query=15,
        batch_size=32, num_workers=num_workers,
    )

    runs = [
        ('smoke2_r1_cnn_standard',    'standard', 'cnn',    cnn_cfg),
        ('smoke2_r2_cnn_fewshot',     'fewshot',  'cnn',    cnn_cfg),
        ('smoke2_r3_gnn_standard',    'standard', 'gnn',    gnn_cfg),
        ('smoke2_r4_gnn_fewshot',     'fewshot',  'gnn',    gnn_cfg),
        ('smoke2_r5_hybrid_standard', 'standard', 'hybrid', hybrid_cfg),
        ('smoke2_r6_hybrid_fewshot',  'fewshot',  'hybrid', hybrid_cfg),
    ]

    run_configs = [
        ExperimentConfig(
            run_id=rid, paradigm=par, arch=arch,
            model_config=mcfg,
            train_config=copy.deepcopy(t_cfg),
            eval_config=copy.deepcopy(eval_cfg),
            tune_config=None, random_seed=42,
        )
        for rid, par, arch, mcfg in runs
    ]

    exec_cfg = ExecutionerConfig(
        checkpoint_dir=DIRS[0], results_dir=DIRS[1],
        plots_dir=DIRS[2], num_workers=num_workers,
    )

    runner  = ExperimentRunner(run_configs, exec_cfg, loader_factory, device)
    summary = runner.run_all()
    _print_results(runner, summary, DIRS[2])
    ResultStore.scores_to_csv(summary, f'{DIRS[1]}/scores.csv')

    print("\nCleaning up...")
    _cleanup(*DIRS)
    print("Smoke Test 2 done.\n")


# ==============================================================================
# smoke_test3 — Data pipeline only  (<2 min, no disk writes)
# ==============================================================================

def run_smoke_test3(loader_factory, device, num_workers=0):
    """
    Smoke test 3 — Data pipeline verification only. No model, no training.
    Verifies: pool counts, disjointness, loader shapes for batch + episodic.
    No disk writes.
    """
    print(f"\n{'='*55}")
    print(f"Smoke Test 3 — Data pipeline  |  device={device}")
    print(f"{'='*55}\n")

    splitter   = loader_factory.splitter
    pool_names = splitter.pool_names()
    all_ok     = True

    # ── Pool existence and class counts ───────────────────────────────
    expected = {
        'pretrain': 64, 'train': 64, 'val_seen': 64,
        'test': 64, 'val_unseen': 16, 'novel': 20,
    }
    print("  Pool counts:")
    for pool, exp_cls in expected.items():
        if pool not in pool_names:
            print(f"    {pool:<12} MISSING ✗")
            all_ok = False
            continue
        c2i   = splitter.get_class_to_indices(pool)
        n_cls = len(c2i)
        n_smp = sum(len(v) for v in c2i.values())
        ok    = '✓' if n_cls == exp_cls else f'✗ expected {exp_cls}'
        print(f"    {pool:<12} classes={n_cls} {ok}  samples={n_smp:,}")
        if n_cls != exp_cls:
            all_ok = False

    # ── Disjointness ──────────────────────────────────────────────────
    print("\n  Overlap checks:")
    pairs = [
        ('pretrain', 'novel'), ('pretrain', 'val_unseen'),
        ('train',    'novel'), ('val_seen', 'novel'),
    ]
    for a, b in pairs:
        ids_a = set(splitter.get_class_to_indices(a).keys())
        ids_b = set(splitter.get_class_to_indices(b).keys())
        overlap = ids_a & ids_b
        ok = '✓ disjoint' if not overlap else f'✗ {len(overlap)} shared'
        print(f"    {a} ∩ {b:<12} : {ok}")
        if overlap:
            all_ok = False

    # ── n_classes ─────────────────────────────────────────────────────
    n_classes = _n_classes(loader_factory)
    print(f"\n  n_classes = {n_classes}")

    # ── Batch loader shape ────────────────────────────────────────────
    print("\n  Batch loader (pretrain pool):")
    try:
        loader = loader_factory.get_loader(
            pool_name='pretrain', mode='batch',
            batch_size=8, num_workers=num_workers, shuffle=False,
        )
        imgs, labels = next(iter(loader))
        assert imgs.shape   == (8, 3, 84, 84), f"imgs: {imgs.shape}"
        assert labels.shape == (8,),            f"labels: {labels.shape}"
        print(f"    imgs={tuple(imgs.shape)}  labels={tuple(labels.shape)}  ✓")
    except Exception as e:
        print(f"    FAILED: {e}")
        all_ok = False

    # ── Episodic loader shape ─────────────────────────────────────────
    print("\n  Episodic loader (novel pool, 5-way 1-shot 5-query):")
    try:
        loader = loader_factory.get_loader(
            pool_name='novel', mode='episodic',
            n_way=5, k_shot=1, q_query=5,
            n_episodes=3, num_workers=num_workers,
        )
        batch   = next(iter(loader))
        support = batch['support']
        query   = batch['query']
        target  = batch['target']
        assert support.shape == (5, 1, 3, 84, 84), f"support: {support.shape}"
        assert query.shape   == (5, 5, 3, 84, 84), f"query: {query.shape}"
        assert target.shape  == (25,),              f"target: {target.shape}"
        print(f"    support={tuple(support.shape)}  query={tuple(query.shape)}"
              f"  target={tuple(target.shape)}  ✓")
    except Exception as e:
        print(f"    FAILED: {e}")
        all_ok = False

    print(f"\n{'All data pipeline checks OK' if all_ok else 'SOME CHECKS FAILED'}.")
    print("Smoke Test 3 done. No disk writes.\n")
    return all_ok

def run_all_smoke_tests(loader_factory, device, num_workers=0, stop_on_fail=True):
    """
    Runs all 4 smoke tests in recommended order.
    stop_on_fail=True  — stops at first failure (default)
    stop_on_fail=False — runs all regardless, reports summary at end
    """
    import time

    tests = [
        ('smoke_test3 — data pipeline',   lambda: run_smoke_test3(loader_factory, device, num_workers)),
        ('smoke_test0 — forward pass',    lambda: run_smoke_test0(loader_factory, device)),
        ('smoke_test2 — all 6 archs',     lambda: run_smoke_test2(loader_factory, device, num_workers)),
        ('smoke_test1 — lightning+optuna',lambda: run_smoke_test1(loader_factory, device, num_workers)),
    ]

    results = {}
    print(f"\n{'='*55}")
    print(f"Running all smoke tests  |  stop_on_fail={stop_on_fail}")
    print(f"{'='*55}\n")

    for name, fn in tests:
        print(f">>> {name}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            results[name] = ('PASS', elapsed)
            print(f"<<< {name} — PASS  ({elapsed/60:.1f} min)\n")
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = ('FAIL', elapsed)
            print(f"<<< {name} — FAIL  ({elapsed/60:.1f} min)")
            print(f"    Error: {e}\n")
            if stop_on_fail:
                print("Stopping — fix failure before proceeding.")
                break

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Smoke Test Summary:")
    print(f"{'─'*55}")
    all_pass = True
    for name, (status, elapsed) in results.items():
        icon = '✓' if status == 'PASS' else '✗'
        print(f"  {icon} {name:<38} {elapsed/60:.1f} min")
        if status == 'FAIL':
            all_pass = False
    not_run = [n for n, _ in tests if n not in results]
    for name in not_run:
        print(f"  - {name:<38} not run")
    print(f"{'='*55}")
    print(f"{'All tests passed — ready for Colab run.' if all_pass else 'Fix failures before launching.'}\n")
    return all_pass