"""Plot priority quality metrics (Gini + top-K overlap) over training.

Produces a 2-panel figure: left=top-10% overlap, right=Gini coefficient,
with lines per task and shaded bands for seed variance.
"""

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def aggregate_by_task(data):
    """Group runs by task name, collect per-step metrics across seeds."""
    task_runs = defaultdict(list)
    for run_dir, info in data.items():
        task = info["task"]
        task_runs[task].append(info["snapshots"])
    return task_runs


def main():
    metrics = load_metrics("studies/td_error_baseline/snapshots/oracle_metrics.json")
    task_runs = aggregate_by_task(metrics)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    colors = {"reach-v3": "#2196F3", "pick-place-v3": "#F44336"}
    labels = {"reach-v3": "reach-v3 (100k, learns)", "pick-place-v3": "pick-place-v3 (300k)"}

    for task, runs in sorted(task_runs.items()):
        # Align steps across seeds
        steps = np.array([s["step"] for s in runs[0]])
        n_seeds = len(runs)

        overlap_all = np.array([[s["top_k_overlap"] for s in run] for run in runs])
        gini_all = np.array([[s["priority_gini"] for s in run] for run in runs])
        spearman_all = np.array([[s["spearman_r"] for s in run] for run in runs])

        for ax_idx, (metric, metric_name, ylabel) in enumerate([
            (overlap_all, "Top-10% Overlap", "Fraction of top-10% |TD|\nin top-10% oracle adv."),
            (gini_all, "Priority Gini", "Gini coeff. of |TD| priorities"),
            (spearman_all, "Spearman(|TD|, oracle)", "Spearman rank corr."),
        ]):
            ax = axes[ax_idx]
            mean = metric.mean(axis=0)
            std = metric.std(axis=0)
            c = colors[task]
            ax.plot(steps / 1000, mean, color=c, label=labels[task], linewidth=2)
            ax.fill_between(steps / 1000, mean - std, mean + std, alpha=0.2, color=c)

    # Chance line for top-K overlap
    axes[0].axhline(0.1, color="gray", linestyle="--", alpha=0.6, label="chance (10%)")
    axes[0].set_ylabel("Fraction of top-10% |TD|\nin top-10% oracle adv.")
    axes[0].set_title("Top-10% Overlap")
    axes[0].set_ylim(-0.02, 0.7)

    axes[1].set_ylabel("Gini coeff. of |TD| priorities")
    axes[1].set_title("Priority Concentration (Gini)")
    axes[1].set_ylim(0, 0.8)

    axes[2].axhline(0.0, color="gray", linestyle="--", alpha=0.6)
    axes[2].set_ylabel("Spearman rank corr.")
    axes[2].set_title("Spearman(|TD|, oracle advantage)")
    axes[2].set_ylim(-0.5, 0.8)

    for ax in axes:
        ax.set_xlabel("Env steps (×1000)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("TD-Error Priority Quality over Training (n=2 seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    out = "studies/td_error_baseline/figures/priority_quality_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
