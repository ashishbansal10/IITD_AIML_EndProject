"""
plotter.py
==========
Visualization and reporting for experiment results.

Separated from experiment.py — can be imported independently
to regenerate plots from saved JSON results without rerunning training.

Usage:
    from experiment import ResultStore, ExperimentSummary
    from plotter    import Plotter
    import json

    with open('results/experiment_XXX_summary.json') as f:
        summary = ExperimentSummary(**json.load(f))

    plotter = Plotter(plots_dir='plots')
    plotter.score_comparison(summary)
    plotter.learning_curves(result)
    plotter.print_results_table(runner)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML

from experiment import ExperimentSummary, RunResult

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
        """Train/val loss and accuracy curves for one run — combined phases."""

        h   = result.training_history
        ts  = result.training_state

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Learning Curves — {result.run_id}", fontsize=12)

        has_pretrain = bool(h.get('pretrain_epochs'))
        has_train    = bool(h.get('train_epochs'))

        pretrain_len = len(h['pretrain_epochs']) if has_pretrain else 0

        # Offset train epochs to continue from pretrain
        train_ep_offset = [e + pretrain_len for e in h['train_epochs']] if has_train else []

        blue   = '#378ADD'
        orange = '#EF9F27'
        red    = '#E24B4A'
        gray   = '#B4B2A9'

        # Derive best epochs from history (min val_loss index)
        pretrain_best_ep = None
        train_best_ep    = None
        if h.get('pretrain_val_loss'):
            pretrain_best_ep = int(h['pretrain_val_loss'].index(min(h['pretrain_val_loss'])))
        if h.get('val_loss'):
            train_best_ep = int(h['val_loss'].index(min(h['val_loss'])))

        # Proto novel reference values from run_scores
        rs = result.run_scores if isinstance(result.run_scores, dict) else {}
        pre_novel_acc = rs.get('pretrain_proto_novel', {}).get('top1_acc')
        tr_novel_acc  = rs.get('trained_proto_novel',  {}).get('top1_acc')

        total_epochs = pretrain_len + (len(train_ep_offset) if train_ep_offset else 0)

        for ax_idx, metric in enumerate(['loss', 'acc']):
            ax = axes[ax_idx]

            # ── Pretrain lines ────────────────────────────────────────
            if has_pretrain:
                ep = h['pretrain_epochs']
                tr = h[f'pretrain_train_{metric}']
                vl = h[f'pretrain_val_{metric}']
                ax.plot(ep, tr, color=blue,   linewidth=1.5)
                ax.plot(ep, vl, color=orange, linewidth=1.5)

                # Best pretrain epoch marker
                if pretrain_best_ep is not None and pretrain_best_ep < len(vl):
                    ax.plot(pretrain_best_ep, vl[pretrain_best_ep],
                            'o', color=red, markersize=6, zorder=5)

            # ── Train lines (dashed, offset) ──────────────────────────
            if has_train:
                tr = h['train_loss'] if metric == 'loss' else h['train_acc']
                vl = h['val_loss']   if metric == 'loss' else h['val_acc']
                ax.plot(train_ep_offset, tr, color=blue,   linewidth=1.5, linestyle='--')
                ax.plot(train_ep_offset, vl, color=orange, linewidth=1.5, linestyle='--')

                # Best train epoch marker
                if train_best_ep is not None and train_best_ep < len(vl):
                    ax.plot(train_ep_offset[train_best_ep], vl[train_best_ep],
                            'o', color=red, markersize=6, zorder=5)

            # ── Phase separator ───────────────────────────────────────
            if has_pretrain and has_train:
                ax.axvline(x=pretrain_len - 0.5, color=gray, linestyle=':', linewidth=0.8)
                ylim = ax.get_ylim()
                ax.text(pretrain_len - 1, ylim[1], 'pretrain', ha='right', va='top', fontsize=8, color=gray)
                ax.text(pretrain_len,     ylim[1], 'train',    ha='left',  va='top', fontsize=8, color=gray)

            # ── Proto novel reference lines (accuracy plot only) ──────
            if ax_idx == 1:
                green = '#1D9E75'
                if pre_novel_acc is not None:
                    ax.axhline(y=pre_novel_acc, color=green, linewidth=1.0,
                               linestyle='-.', alpha=0.8)
                    ax.text(total_epochs + 0.5, pre_novel_acc,
                            f' pre novel={pre_novel_acc:.2f}',
                            va='center', fontsize=7.5, color=green)
                if tr_novel_acc is not None:
                    ax.axhline(y=tr_novel_acc, color=red, linewidth=1.0,
                               linestyle='-.', alpha=0.8)
                    ax.text(total_epochs + 0.5, tr_novel_acc,
                            f' tr novel={tr_novel_acc:.2f}',
                            va='center', fontsize=7.5, color=red)

            ax.set_title('Loss' if metric == 'loss' else 'Accuracy', fontsize=11)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

        # ── Shared legend ─────────────────────────────────────────────
        legend_elements = [
            mpatches.Patch(color=blue,        label='train'),
            mpatches.Patch(color=orange,      label='val'),
            plt.Line2D([0],[0], color='gray',     linestyle='--',  label='train phase'),
            plt.Line2D([0],[0], color=red,         marker='o', linestyle='None', markersize=6, label='best epoch'),
            plt.Line2D([0],[0], color='#1D9E75',  linestyle='-.',  label='pretrain proto novel'),
            plt.Line2D([0],[0], color=red,         linestyle='-.',  label='trained proto novel'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        path = os.path.join(self.plots_dir, f"{result.run_id}_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        if show: plt.show()
        plt.close()

    def score_comparison(self, summary: ExperimentSummary, show: bool = True):
        """
        Two side-by-side bar charts — standard and fewshot paradigms.
        Each chart: 3 arch × 2 phases × 3 metrics = 18 bars.
        """
        ct = summary.comparison_table

        run_order = [
            ('run1_cnn_standard', 'run2_cnn_fewshot'),
            ('run3_gnn_standard', 'run4_gnn_fewshot'),
            ('run5_hybrid_standard', 'run6_hybrid_fewshot'),
        ]
        run_order = [(s,f) for s,f in run_order
                     if s in next(iter(ct.values())) and f in next(iter(ct.values()))]

        arch_labels = ['CNN', 'GNN', 'Hybrid']

        metrics = [
            ('pretrain_softmax',     'trained_softmax',     'softmax',     '#AFA9EC', '#7F77DD'),
            ('pretrain_proto_seen',  'trained_proto_seen',  'proto seen',  '#FAC775', '#EF9F27'),
            ('pretrain_proto_novel', 'trained_proto_novel', 'proto novel', '#F09595', '#E24B4A'),
        ]

        thin_w     = 0.08
        pair_gap   = 0.03
        metric_gap = 0.08
        arch_gap   = 0.20
        pair_w     = 2 * thin_w + pair_gap
        group_w    = len(metrics) * pair_w + (len(metrics) - 1) * metric_gap
        arch_step  = group_w + arch_gap

        def _draw(ax, paradigm_idx, title):
            for ai, (std_run, few_run) in enumerate(run_order):
                run = std_run if paradigm_idx == 0 else few_run
                arch_x = ai * arch_step

                for pi, (pre_key, tr_key, label, light_c, dark_c) in enumerate(metrics):
                    x_pair = arch_x + pi * (pair_w + metric_gap)
                    pre_val = ct.get(pre_key, {}).get(run, 0) or 0
                    tr_val  = ct.get(tr_key,  {}).get(run, 0) or 0

                    ax.bar(x_pair,                     pre_val, thin_w, color=light_c)
                    ax.bar(x_pair + thin_w + pair_gap, tr_val,  thin_w, color=dark_c)

                    ax.text(x_pair + thin_w + pair_gap/2, -0.04, label, ha='center', va='top', fontsize=7, 
                            color='#888780', transform=ax.transData)

                group_mid = arch_x + group_w / 2
                ax.text(group_mid, 1.02, arch_labels[ai], ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444441')
                if ai > 0:
                    ax.axvline(arch_x - arch_gap/2, color='#B4B2A9', linewidth=0.8, linestyle='--', alpha=0.5)

            total_w = len(run_order) * arch_step - arch_gap
            ax.set_xlim(-0.1, total_w + 0.1)
            ax.set_ylim(0, 1.08)
            ax.set_xticks([])
            ax.set_ylabel('Accuracy (top-1)', fontsize=9)
            ax.set_title(title, fontsize=11, pad=10)
            ax.yaxis.grid(True, alpha=0.15)
            ax.set_axisbelow(True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        _draw(axes[0], 0, 'Standard Paradigm — Pretrain vs Trained')
        _draw(axes[1], 1, 'FewShot Paradigm — Pretrain vs Trained')

        legend_elements = [
            mpatches.Patch(facecolor='#AFA9EC', label='pretrain softmax'),
            mpatches.Patch(facecolor='#7F77DD', label='trained softmax'),
            mpatches.Patch(facecolor='#FAC775', label='pretrain proto seen'),
            mpatches.Patch(facecolor='#EF9F27', label='trained proto seen'),
            mpatches.Patch(facecolor='#F09595', label='pretrain proto novel'),
            mpatches.Patch(facecolor='#E24B4A', label='trained proto novel'),
        ]
        fig.legend(handles=legend_elements, fontsize=8, ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.04), frameon=False)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        path = os.path.join(self.plots_dir, f"score_comparison_{summary.experiment_id}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
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

    def print_results_table(self, runner, save_path='results/results_table.html'):
        """
        Prints a styled HTML results table for all 6 runs.
        Suitable for copy-paste into MS Word / project report.

        Columns:
            Arch | Paradigm | Pretrain Eval | Softmax | Proto Seen |
            Proto Novel | 95% CI | Time (min) | [Best HPs if show_hps=True]

        Highlighting:
            Green  = best value in column
            Red    = worst value in column
            Time   = green for fastest, red for slowest
            CI     = text only, no highlight
        """

        show_hps = any(r.best_hps is not None for r in runner.run_results.values())

        # ── Build rows ────────────────────────────────────────────────────
        rows = []
        for run_id, result in runner.run_results.items():
            rs  = result.run_scores

            def acc(key, field='top1_acc'):
                s = rs.get(key, {})
                v = s.get(field) if isinstance(s, dict) else None
                return round(v * 100, 2) if v is not None else None

            novel_lo = rs.get('trained_proto_novel', {}).get('ci_lower')
            novel_hi = rs.get('trained_proto_novel', {}).get('ci_upper')
            ci_str   = (f"{novel_lo*100:.2f} – {novel_hi*100:.2f}"
                        if novel_lo is not None else '—')

            row = {
                'Arch'              : result.arch.upper(),
                'Paradigm'          : result.paradigm.capitalize(),
                'Pre Softmax %'     : acc('pretrain_softmax'),
                'Pre Proto Seen %'  : acc('pretrain_proto_seen'),
                'Pre Proto Novel %' : acc('pretrain_proto_novel'),
                'Tr Softmax %'      : acc('trained_softmax'),
                'Tr Proto Seen %'   : acc('trained_proto_seen'),
                'Tr Proto Novel %'  : acc('trained_proto_novel'),
                '95% CI'            : ci_str,
                'Time (min)'        : int(round(result.duration_seconds / 60, 0)),
            }

            if show_hps and result.best_hps:
                row['Best HPs'] = str(result.best_hps)

            rows.append(row)

        df = pd.DataFrame(rows)

        # ── Highlight helpers ─────────────────────────────────────────────
        highlight_high = ['Pre Softmax %', 'Pre Proto Seen %', 'Pre Proto Novel %',
                          'Tr Softmax %', 'Tr Proto Seen %', 'Tr Proto Novel %']
        highlight_low  = ['Time (min)']   # lower is better

        GREEN_BG  = 'background-color: #c6efce; color: #276221; font-weight: bold'
        RED_BG    = 'background-color: #ffc7ce; color: #9c0006; font-weight: bold'
        BOLD      = 'font-weight: bold'
        NORMAL    = ''

        def highlight_col(col):
            styles = [NORMAL] * len(col)
            numeric = col.dropna()
            if col.name in highlight_high and len(numeric):
                best  = numeric.max()
                worst = numeric.min()
                for i, v in enumerate(col):
                    if v == best:  styles[i] = GREEN_BG
                    elif v == worst: styles[i] = RED_BG
            elif col.name in highlight_low and len(numeric):
                best  = numeric.min()
                worst = numeric.max()
                for i, v in enumerate(col):
                    if v == best:  styles[i] = GREEN_BG
                    elif v == worst: styles[i] = RED_BG
            return styles

        def highlight_text_cols(col):
            if col.name in ['Arch', 'Paradigm']:
                return [BOLD] * len(col)
            return [NORMAL] * len(col)

        # ── Build styler ──────────────────────────────────────────────────
        fmt = {
            'Pre Softmax %'     : '{:.2f}',
            'Pre Proto Seen %'  : '{:.2f}',
            'Pre Proto Novel %' : '{:.2f}',
            'Tr Softmax %'      : '{:.2f}',
            'Tr Proto Seen %'   : '{:.2f}',
            'Tr Proto Novel %'  : '{:.2f}',
            'Time (min)'        : '{:.0f}',
        }
        if show_hps:
            fmt['Best HPs'] = '{}'

        styled = (
            df.style
            .apply(highlight_col,      axis=0)
            .apply(highlight_text_cols, axis=0)
            .format(fmt, na_rep='—')
            .set_caption('Table: Few-Shot Classification Results — Mini-ImageNet (5-way 5-shot)')
            .set_table_styles([
                # Table overall
                {'selector': 'table',
                'props': [('border-collapse', 'collapse'),
                            ('font-family', 'Calibri, Arial, sans-serif'),
                            ('font-size', '13px'),
                            ('width', '100%')]},
                # Caption
                {'selector': 'caption',
                'props': [('font-size', '13px'),
                            ('font-weight', 'bold'),
                            ('text-align', 'left'),
                            ('padding-bottom', '6px'),
                            ('color', '#1a1a1a')]},
                # Header row
                {'selector': 'thead tr th',
                'props': [('background-color', '#1f4e79'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center'),
                            ('padding', '8px 10px'),
                            ('border', '1px solid #aaa')]},
                # Data cells
                {'selector': 'td',
                'props': [('text-align', 'center'),
                            ('padding', '6px 10px'),
                            ('border', '1px solid #ddd')]},
                # Arch + Paradigm cols — left align
                {'selector': 'td:nth-child(1), td:nth-child(2)',
                'props': [('text-align', 'left'),
                            ('font-weight', 'bold')]},
                # Alternating row shading
                {'selector': 'tbody tr:nth-child(even)',
                'props': [('background-color', '#f2f2f2')]},
                {'selector': 'tbody tr:hover',
                'props': [('background-color', '#e8f0fe')]},
            ])
            .hide(axis='index')
        )

        # ── Display ───────────────────────────────────────────────────────
        display(styled)

        # ── Save HTML ─────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        html_str = styled.to_html()
        with open(save_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
                    <html>
                    <head>
                    <meta charset='utf-8'>
                    <title>Results Table</title>
                    <style>
                    body {{ font-family: Calibri, Arial, sans-serif; padding: 20px; }}
                    </style>
                    </head>
                    <body>
                    {html_str}
                    </body>
                    </html>""")
        print(f"Saved: {save_path}")

        # ── Also save CSV for report ───────────────────────────────────────
        csv_path = save_path.replace('.html', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")