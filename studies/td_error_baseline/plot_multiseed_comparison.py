"""Multi-seed mode comparison: uniform vs TD-PER vs adaptive vs RPE-PER.

5 seeds × 4 modes = 20 runs. Produces a 6-panel figure with mean±std
error bands showing that all priority signals fail to beat uniform.

Usage:
    python plot_multiseed_comparison.py                     # reach-v3 (default)
    python plot_multiseed_comparison.py --task pick-place-v3
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SNAP_ROOT = Path(__file__).parent / "snapshots"
MODES = ["uniform", "td-per", "adaptive", "rpe-per"]
MODE_LABELS = {
    "uniform": "Uniform",
    "td-per": "TD-PER",
    "adaptive": "Adaptive Mixer",
    "rpe-per": "RPE-PER",
}
MODE_COLORS = {
    "uniform": "#2ecc71",
    "td-per": "#3498db",
    "adaptive": "#e74c3c",
    "rpe-per": "#9b59b6",
}
SEEDS = [42, 123, 7, 99, 256]


def load_all_snapshots(mode, seed, task="reach-v3"):
    """Load snapshot data for a given mode+seed run."""
    snap_dir = SNAP_ROOT / f"{task}_s{seed}_{mode}" / "td_snapshots"
    results = []
    for f in sorted(snap_dir.glob("snapshot_*.npz")):
        data = dict(np.load(f, allow_pickle=True))
        step = int(data["step"])
        results.append({
            "step": step,
            "abs_td_mean": float(np.mean(data["abs_td_errors"])),
            "spearman": float(data.get("td_dense_spearman", 0)),
            "q_mean": float(data.get("q_mean", 0)),
            "ep_rew": float(data.get("episode_return_mean", 0)),
            "dense_rew": float(data.get("episode_dense_return_mean", 0)),
        })
    return results


def aggregate_seeds(mode, task="reach-v3"):
    """Load all seeds for a mode and aggregate by step."""
    all_runs = {seed: load_all_snapshots(mode, seed, task) for seed in SEEDS}

    # Find common steps
    step_sets = [set(d["step"] for d in run) for run in all_runs.values() if run]
    if not step_sets:
        return {}
    common_steps = sorted(set.intersection(*step_sets))

    agg = {}
    for step in common_steps:
        vals = {k: [] for k in ["abs_td_mean", "spearman", "q_mean", "ep_rew", "dense_rew"]}
        for seed, run in all_runs.items():
            for d in run:
                if d["step"] == step:
                    for k in vals:
                        vals[k].append(d[k])
                    break
        agg[step] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}
    return agg


def main(task="reach-v3"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    all_agg = {mode: aggregate_seeds(mode, task) for mode in MODES}

    # Panel (a): Episode Return (the key metric)
    ax = axes[0, 0]
    for mode in MODES:
        agg = all_agg[mode]
        steps = sorted(agg.keys())
        means = [agg[s]["ep_rew"][0] for s in steps]
        stds = [agg[s]["ep_rew"][1] for s in steps]
        ax.plot(steps, means, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=MODE_COLORS[mode])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return (sparse)")
    ax.set_title("(a) Learning Curves (n=5 seeds)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Q-value evolution
    ax = axes[0, 1]
    for mode in MODES:
        agg = all_agg[mode]
        steps = sorted(agg.keys())
        means = [agg[s]["q_mean"][0] for s in steps]
        stds = [agg[s]["q_mean"][1] for s in steps]
        ax.plot(steps, means, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
        ax.fill_between(steps,
                        [max(0, m - s) for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=MODE_COLORS[mode])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Q-value")
    ax.set_title("(b) Q-Value Stability")
    ax.legend(fontsize=9)
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(True, alpha=0.3)

    # Panel (c): |TD| magnitude
    ax = axes[0, 2]
    for mode in MODES:
        agg = all_agg[mode]
        steps = sorted(agg.keys())
        means = [agg[s]["abs_td_mean"][0] for s in steps]
        stds = [agg[s]["abs_td_mean"][1] for s in steps]
        ax.plot(steps, means, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
        ax.fill_between(steps,
                        [max(1e-6, m - s) for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=MODE_COLORS[mode])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean |TD-error|")
    ax.set_title("(c) TD-Error Magnitude")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel (d): Spearman correlation
    ax = axes[1, 0]
    for mode in MODES:
        agg = all_agg[mode]
        steps = sorted(agg.keys())
        means = [agg[s]["spearman"][0] for s in steps]
        stds = [agg[s]["spearman"][1] for s in steps]
        ax.plot(steps, means, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
        ax.fill_between(steps,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=MODE_COLORS[mode])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Spearman(|TD|, Oracle Advantage)")
    ax.set_title("(d) TD-Oracle Correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (e): Per-seed learning outcome bar chart
    ax = axes[1, 1]
    learn_threshold = 50  # ep_rew > 50 at any step >= 70k counts as "learned"
    mode_results = {}
    for mode in MODES:
        learned = 0
        for seed in SEEDS:
            snaps = load_all_snapshots(mode, seed, task)
            max_late_rew = max(
                (d["ep_rew"] for d in snaps if d["step"] >= 70000),
                default=0
            )
            if max_late_rew > learn_threshold:
                learned += 1
        mode_results[mode] = learned

    x = range(len(MODES))
    bars = ax.bar(x, [mode_results[m] for m in MODES],
                  color=[MODE_COLORS[m] for m in MODES], alpha=0.8, edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES])
    ax.set_ylabel("Seeds that Learned (out of 5)")
    ax.set_title("(e) Learning Success Rate")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.3)
    for bar, count in zip(bars, [mode_results[m] for m in MODES]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{count}/5", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (f): Per-seed ep_rew trajectories (all 15 runs)
    ax = axes[1, 2]
    for mode in MODES:
        for seed in SEEDS:
            snaps = load_all_snapshots(mode, seed, task)
            steps = [d["step"] for d in snaps]
            rews = [d["ep_rew"] for d in snaps]
            ax.plot(steps, rews, "-", color=MODE_COLORS[mode], alpha=0.3, linewidth=1)
        # Dummy line for legend
        ax.plot([], [], "-", color=MODE_COLORS[mode], linewidth=2,
                label=MODE_LABELS[mode])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title("(f) Individual Seed Trajectories")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Build title dynamically
    results_str = ", ".join(
        f"{MODE_LABELS[m]}: {mode_results[m]}/5"
        for m in MODES
    )
    fig.suptitle(
        f"5-Seed Mode Comparison: {task} (100k steps)\n{results_str}",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    fname = f"multiseed_mode_comparison_{task.replace('-', '_')}"
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{fname}.{ext}",
                    dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir / fname}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach-v3", choices=["reach-v3", "pick-place-v3"])
    args = parser.parse_args()
    main(task=args.task)
