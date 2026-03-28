"""
evaluator.py
============
Evaluation pipeline — collects scores for one run (one model + one config).

Design
------
EvalConfig  — evaluation protocol configuration
EvalResult  — single score result (one mechanism, one pool, one phase)
RunScores   — all scores for one run packed together
Evaluator   — produces RunScores for one model

Per-Run Score Collection
------------------------
Each run produces 5 scores independently:

    pretrain_softmax    — softmax accuracy on test seen, pretrain checkpoint
                          top1, no CI (full deterministic batch pass)

    pretrain_proto_seen — prototypical accuracy on test seen, pretrain checkpoint
                          mean + CI + std over n_episodes_seen episodes
    
    pretrain_proto_novel — prototypical accuracy on novel classes, pretrain checkpoint
                          mean + CI + std over n_episodes_novel episodes

    trained_softmax     — softmax accuracy on test seen, trained checkpoint
                          top1, no CI

    trained_proto_seen  — prototypical accuracy on test seen, trained checkpoint
                          mean + CI + std over n_episodes_seen episodes

    trained_proto_novel — prototypical accuracy on novel classes, trained checkpoint
                          mean + CI + std over n_episodes_novel episodes
                          ← PRIMARY metric for few-shot generalization

6 runs x 6 scores = 36 scores total.
Comparison across runs done in Plotter / ResultStore.

Softmax vs prototypical
--------------------
Softmax (full batch):
    model.eval()
    probs  = model(imgs, mode='softmax')     # direct probabilities
    top1   = argmax(probs) == labels
    Single deterministic pass — no CI needed.
    NEVER compute loss here — evaluation only.

prototypical (episodic N-way K-shot):
    per episode:
        s_emb = model(support_flat, mode='embedding')   # [N*K, D]
        q_emb = model(query_flat,   mode='embedding')   # [N*Q, D]
        dists = model(s_emb, q_emb, mode='prototypical')# [N*Q, N]
        acc   = (dists.argmax(1) == target).float().mean()
    Each episode = different random class + image sample → distribution of accs.
    CI computed via t-distribution over episode accuracies.
    NEVER compute loss here — evaluation only.

Why CI for prototypical but Not Softmax
-------------------------------------
Softmax: same images every time → deterministic → no variance → no CI.
Proto:   random episode sampling → different acc per episode →
         distribution → CI describes uncertainty of that distribution.
         Standard in few-shot benchmarks (Snell et al. 2017 and all after).

Eval Timing
-----------
Called immediately after pretrain and after train in ExperimentRunner.
No need to reload checkpoint — model already at best state via load_best().
Pretrain checkpoint deleted after pretrain eval if keep_pretrain=False.

device passed from notebook — never auto-detected inside any class.

Required Libraries
------------------
# torch>=2.0.0
# scipy>=1.9.0   # t-distribution CI — pip install scipy
#                # falls back to normal approx (z=1.96) if not available
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ==============================================================================
# EvalConfig
# ==============================================================================

@dataclass
class EvalConfig:
    """
    Evaluation protocol configuration.

    Episode counts:
        n_episodes_seen  : episodes for seen-class prototypical eval
        n_episodes_novel : episodes for novel-class prototypical eval
                           higher count → tighter CI → more reliable estimate
                           standard: 600 for novel (Snell et al. 2017)

    Protocol — must match PrototypicalNet config in ModelConfig:
        n_way   : classes per episode
        k_shot  : support images per class
        q_query : query images per class

    CI:
        ci_alpha : 0.05 = 95% CI (standard in few-shot benchmarks)
    """

    # Episode counts
    n_episodes_seen:  int   = 200    # seen-class proto eval
    n_episodes_novel: int   = 600    # novel-class proto eval (primary metric)

    # Episodic protocol
    n_way:            int   = 5
    k_shot:           int   = 5
    q_query:          int   = 15

    # Confidence interval
    ci_alpha:         float = 0.05   # 0.05 = 95% CI

    # Batch size for softmax evaluation
    batch_size:       int   = 64
    num_workers:      int   = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'EvalConfig':
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def validate_config(self):
        """Validates field values at construction time."""
        if self.n_way < 2:
            raise ValueError(f"EvalConfig.n_way={self.n_way} must be >= 2.")
        if self.k_shot < 1:
            raise ValueError(f"EvalConfig.k_shot={self.k_shot} must be >= 1.")
        if self.q_query < 1:
            raise ValueError(f"EvalConfig.q_query={self.q_query} must be >= 1.")
        if self.n_episodes_seen < 1:
            raise ValueError(f"EvalConfig.n_episodes_seen must be >= 1.")
        if self.n_episodes_novel < 1:
            raise ValueError(f"EvalConfig.n_episodes_novel must be >= 1.")
        if not (0 < self.ci_alpha < 1):
            raise ValueError(
                f"EvalConfig.ci_alpha={self.ci_alpha} must be between 0 and 1. "
                f"Typical: 0.05 for 95% CI."
            )
        if self.batch_size < 1:
            raise ValueError(f"EvalConfig.batch_size must be >= 1.")
        if self.num_workers < 0:
            raise ValueError(f"EvalConfig.num_workers must be >= 0.")

# ==============================================================================
# EvalResult — single score
# ==============================================================================

@dataclass
class EvalResult:
    """
    Result for a single evaluation score.

    Softmax scores:
        top1_acc  — top-1 accuracy over full test set  (primary)
        ci_lower, ci_upper, std_acc → None (not applicable)
        n_samples → total images evaluated

    prototypical scores:
        top1_acc  — mean accuracy over episodes  (primary)
        ci_lower  — 95% CI lower bound
        ci_upper  — 95% CI upper bound
        std_acc   — std deviation across episodes
        n_samples → number of episodes
    """

    mechanism:  str               # 'softmax' or 'prototypical'
    pool:       str               # 'test' or 'novel'
    phase:      str               # 'pretrain' or 'trained'

    # Primary metric — always present
    top1_acc:   float

    # prototypical only
    ci_lower:   Optional[float] = None
    ci_upper:   Optional[float] = None
    std_acc:    Optional[float] = None

    # Sample counts
    n_samples:  int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self):
        if self.mechanism == 'softmax':
            return (f"    softmax      [{self.phase:8s}] {self.pool:5s} | "
                    f"top1: {self.top1_acc:.2f}  "
                    f"n={self.n_samples}")
        else:
            primary = '  \u2605 primary' if self.pool == 'novel' else ''
            return (f"    prototypical [{self.phase:8s}] {self.pool:5s} | "
                    f"acc: {self.top1_acc:.2f}  "
                    f"CI: [{self.ci_lower:.2f}, {self.ci_upper:.2f}]  "
                    f"std: {self.std_acc:.2f}  "
                    f"n={self.n_samples}{primary}")


# ==============================================================================
# RunScores — all scores for one run
# ==============================================================================

@dataclass
class RunScores:
    """
    All evaluation scores for one run (one model + one config).
    5 EvalResult objects packed together.

    Stored inside RunResult — serialized to JSON as part of run record.

    Cross-run comparison done in Plotter using top1_acc values:
        run1.trained_proto_novel.top1_acc vs run2.trained_proto_novel.top1_acc
        → how episodic improves over standard on novel classes

        run1.trained_proto_novel.top1_acc vs run5.trained_proto_novel.top1_acc
        → how hybrid improves over CNN
    """

    run_id:   str
    paradigm: str   # 'standard' or 'fewshot'
    arch:     str   # 'cnn', 'gnn', 'hybrid'

    # Pretrain checkpoint scores
    pretrain_softmax:     EvalResult   # softmax,  test seen,  pretrain
    pretrain_proto_seen:  EvalResult   # prototypical, test seen, pretrain
    pretrain_proto_novel: EvalResult   # prototypical, novel, pretrain

    # Trained checkpoint scores
    trained_softmax:      EvalResult   # softmax,   test seen,  trained
    trained_proto_seen:   EvalResult   # prototypical, test seen,  trained
    trained_proto_novel:  EvalResult   # prototypical, novel,      trained ← PRIMARY

    def to_dict(self) -> dict:
        return {
            'run_id':               self.run_id,
            'paradigm':             self.paradigm,
            'arch':                 self.arch,
            'pretrain_softmax':     self.pretrain_softmax.to_dict(),
            'pretrain_proto_seen':  self.pretrain_proto_seen.to_dict(),
            'pretrain_proto_novel': self.pretrain_proto_novel.to_dict(),
            'trained_softmax':      self.trained_softmax.to_dict(),
            'trained_proto_seen':   self.trained_proto_seen.to_dict(),
            'trained_proto_novel':  self.trained_proto_novel.to_dict(),
        }

    def summary(self) -> str:
        lines = [
            f"",
            f"  Eval Summary Acc.: {self.run_id}  [{self.arch} | {self.paradigm}]",
            f"    Pretrain — softmax: {self.pretrain_softmax.top1_acc:.2f}  "
            f"proto_seen: {self.pretrain_proto_seen.top1_acc:.2f}",
            f"proto_novel: {self.pretrain_proto_novel.top1_acc:.2f}",
            f"    Trained  — softmax: {self.trained_softmax.top1_acc:.2f}  "
            f"proto_seen: {self.trained_proto_seen.top1_acc:.2f}  "
            f"proto_novel: {self.trained_proto_novel.top1_acc:.2f} \u2605  "
            f"CI:[{self.trained_proto_novel.ci_lower:.2f}, {self.trained_proto_novel.ci_upper:.2f}]",
            f"",
        ]
        return '\n'.join(lines)

# ==============================================================================
# Evaluator
# ==============================================================================

class Evaluator:
    """
    Produces RunScores for one model at one checkpoint.

    Usage — called from ExperimentRunner immediately after each phase:

        # After pretrain — 2 scores
        pre_softmax, pre_proto_seen, pre_proto_novel = evaluator.eval_pretrain(model)

        # After full training — 3 scores
        tr_softmax, tr_proto_seen, tr_proto_novel = evaluator.eval_trained(model)

        # Pack into RunScores
        run_scores = evaluator.collect(
            run_id, paradigm, arch,
            pre_softmax, pre_proto_seen, pre_proto_novel,
            tr_softmax, tr_proto_seen, tr_proto_novel
        )
    """

    def __init__(self,
                 factory,
                 config: EvalConfig,
                 device: torch.device):
        """
        Args:
            factory : SmartDataLoaderFactory
            config  : EvalConfig
            device  : torch.device — from notebook
        """
        self.factory = factory
        self.config  = config
        self.device  = device

        self.config.validate_config()
        self.validate()

    def validate(self):
        """
        Cross-object validation — checks config and factory are consistent.
        Called at end of __init__.
        """
        available_pools = self.factory.valid_pools()

        # Softmax eval needs 'test' pool (batch mode)
        if 'test' not in available_pools:
            raise ValueError(
                f"Evaluator requires pool 'test' for softmax eval "
                f"but factory has: {available_pools}."
            )

        # prototypical eval needs 'novel' pool (episodic mode)
        if 'novel' not in available_pools:
            raise ValueError(
                f"Evaluator requires pool 'novel' for novel-class proto eval "
                f"but factory has: {available_pools}."
            )
        
    # ------------------------------------------------------------------
    # Phase eval — called from ExperimentRunner
    # ------------------------------------------------------------------

    def eval_pretrain(self, model) -> Tuple[EvalResult, EvalResult]:
        """
        Evaluates at pretrain checkpoint.
        Returns (pretrain_softmax, pretrain_proto_seen, pretrain_proto_novel).
        Called immediately after pretrain() + load_best().
        """
        print(f"\n  Evaluating pretrain checkpoint...")
        softmax = self._score_softmax(model, pool='test', phase='pretrain')
        proto_seen   = self._score_prototypical(model, pool='test', phase='pretrain', n_episodes=self.config.n_episodes_seen)
        proto_novel  = self._score_prototypical(model, pool='novel', phase='pretrain', n_episodes=self.config.n_episodes_novel)
        print(f"\n    {softmax}")
        print(f"    {proto_seen}")
        print(f"    {proto_novel}")
        return softmax, proto_seen, proto_novel

    def eval_trained(self, model) -> Tuple[EvalResult, EvalResult, EvalResult]:
        """
        Evaluates at trained checkpoint.
        Returns (trained_softmax, trained_proto_seen, trained_proto_novel).
        Called immediately after train() + load_best().
        """
        print(f"\n  Evaluating trained model...")
        softmax     = self._score_softmax(model, pool='test',  phase='trained')
        proto_seen  = self._score_prototypical(model, pool='test',  phase='trained', n_episodes=self.config.n_episodes_seen)
        proto_novel = self._score_prototypical(model, pool='novel', phase='trained', n_episodes=self.config.n_episodes_novel)
        print(f"\n    {softmax}")
        print(f"    {proto_seen}")
        print(f"    {proto_novel}")
        return softmax, proto_seen, proto_novel

    def collect(self,
                run_id:               str,
                paradigm:             str,
                arch:                 str,
                pretrain_softmax:     EvalResult,
                pretrain_proto_seen:  EvalResult,
                pretrain_proto_novel: EvalResult,
                trained_softmax:      EvalResult,
                trained_proto_seen:   EvalResult,
                trained_proto_novel:  EvalResult) -> RunScores:
        """Packs all 5 EvalResults into RunScores."""
        run_scores = RunScores(
            run_id               = run_id,
            paradigm             = paradigm,
            arch                 = arch,
            pretrain_softmax     = pretrain_softmax,
            pretrain_proto_seen  = pretrain_proto_seen,
            pretrain_proto_novel = pretrain_proto_novel,
            trained_softmax      = trained_softmax,
            trained_proto_seen   = trained_proto_seen,
            trained_proto_novel  = trained_proto_novel,
        )
        print(run_scores.summary())
        return run_scores

    # ------------------------------------------------------------------
    # Softmax evaluation
    # ------------------------------------------------------------------

    def _score_softmax(self,
                       model,
                       pool:  str,
                       phase: str) -> EvalResult:
        """
        Full batch softmax evaluation.
        mode='softmax' → direct probabilities — no F.softmax() call needed.
        top1: argmax == label
        NEVER compute loss here — evaluation only.
        """
        loader = self.factory.get_loader(
            pool, mode='batch',
            batch_size  = self.config.batch_size,
            num_workers = self.config.num_workers,
            shuffle     = False
        )

        model.eval()
        total_top1    = 0
        total_samples = 0

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f'    softmax {pool} {phase}', leave=False):
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                probs  = model(imgs, mode='softmax')       # [B, n_classes]
                preds  = probs.argmax(dim=1)
                total_top1 += (preds == labels).sum().item()

                total_samples += labels.size(0)

        return EvalResult(
            mechanism = 'softmax',
            pool      = pool,
            phase     = phase,
            top1_acc  = total_top1 / max(total_samples, 1),
            n_samples = total_samples
        )

    # ------------------------------------------------------------------
    # prototypical evaluation
    # ------------------------------------------------------------------

    def _score_prototypical(self,
                         model,
                         pool:       str,
                         phase:      str,
                         n_episodes: int) -> EvalResult:
        """
        Episodic N-way K-shot prototypical evaluation.
        Each episode = different random class + image sample.
        Reports mean accuracy ± 95% CI over n_episodes.
        NEVER compute loss here — evaluation only.
        dists.argmax(1) == target → accuracy directly.
        """
        loader = self.factory.get_loader(
            pool, mode='episodic',
            n          = self.config.n_way,
            k          = self.config.k_shot,
            q          = self.config.q_query,
            iterations = n_episodes,
            num_workers= self.config.num_workers
        )

        model.eval()
        episode_accs: List[float] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'    proto {pool} {phase}', leave=False):
                support = batch['support'].to(self.device)  # [N, K, C, H, W]
                query   = batch['query'].to(self.device)    # [N, Q, C, H, W]
                target  = batch['target'].to(self.device)   # [N*Q]

                N, K, C, H, W = support.shape
                Q             = query.shape[1]

                # Combined episode call — same reason as trainer._episodic_epoch.
                # GATRelationalLayer needs full episode in one pass for eval too.
                episode = torch.cat([
                    support.reshape(N * K, C, H, W),
                    query.reshape(N * Q, C, H, W)
                ], dim=0)                                             # [N*(K+Q), C, H, W]

                all_emb = model(episode, mode='embedding')           # [N*(K+Q), D]

                s_emb = all_emb[:N * K]                              # [N*K, D]
                q_emb = all_emb[N * K:]                              # [N*Q, D]

                dists = model(support_emb=s_emb, query_emb=q_emb, mode='prototypical' ) # [N*Q, N]

                acc = (dists.argmax(1) == target).float().mean().item()
                episode_accs.append(acc)

        mean_acc, ci_lower, ci_upper, std_acc = self._compute_ci(episode_accs)

        return EvalResult(
            mechanism = 'prototypical',
            pool      = pool,
            phase     = phase,
            top1_acc  = mean_acc,
            ci_lower  = ci_lower,
            ci_upper  = ci_upper,
            std_acc   = std_acc,
            n_samples = n_episodes
        )

    # ------------------------------------------------------------------
    # 95% Confidence Interval — t-distribution
    # ------------------------------------------------------------------

    def _compute_ci(self,
                    values: List[float]) -> Tuple[float, float, float, float]:
        """
        Computes mean, 95% CI (t-distribution), and std.
        Returns (mean, ci_lower, ci_upper, std).

        t-distribution correct for episode accuracies — not normal approx.
        Falls back to z=1.96 if scipy unavailable.
        """
        n    = len(values)
        mean = sum(values) / n

        if n < 2:
            return mean, mean, mean, 0.0

        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std      = math.sqrt(variance)
        std_err  = std / math.sqrt(n)

        try:
            from scipy import stats
            t_val = stats.t.ppf(1 - self.config.ci_alpha / 2, df=n - 1)
        except ImportError:
            t_val = 1.96   # normal approx fallback

        margin = t_val * std_err
        return mean, mean - margin, mean + margin, std
