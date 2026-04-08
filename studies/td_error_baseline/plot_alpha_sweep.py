"""Alpha sweep for TD-PER: does lower prioritization exponent mitigate Q-explosion?

Compares α=0.1, 0.3, 0.6 for td-per mode on reach-v3 (5 seeds each),
plus uniform baseline for reference. Tests whether Q-explosion is a tuning
issue (fixable with lower alpha) or structural (TD-error is uninformative).
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SNAP_ROOT = Path(__file__).parent / "snapshots"
TASK = "reach-v3"
SEEDS = [42, 123, 7, 99, 256]

# Alpha variants + uniform baseline
ALPHAS = [0.1, 0.3, 0.6]
ALPHA_LABELS = {0.1: "TD-PER α=0.1", 0.3: "TD-PER α=0.3", 0.6: "TD-PER α=0.6"}
ALPHA_COLORS = {0.1: "#9b59b6", 0.3: "#e67e22", 0.6: "#3498db"}


def load_snapshots(mode_dir_name, seed):
    """Load snapshot data for a given run directory."""
    snap_dir = SNAP_ROOT / f"{TASK}_s{seed}_{mode_dir_name}" / "td_snapshots"
    if not snap_dir.exists():
        return []
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
        })
    return results


def aggregate_seeds(mode_dir_name):
    """Aggregate across seeds for a given run type."""
    all_runs = {seed: load_snapshots(mode_dir_name, seed) for seed in SEEDS}
    all_runs = {k: v for k, v in all_runs.items() if v}

    if not all_runs:
        return {}

    step_sets = [set(d["step"] for d in run) for run in all_runs.values()]
    common_steps = sorted(set.intersection(*step_sets))

    agg = {}
    for step in common_steps:
        vals = {k: [] for k in ["abs_td_mean", "spearman", "q_mean", "ep_rew"]}
        for seed, run in all_runs.items():
            for d in run:
                if d["step"] == step:
                    for k in vals:
                        vals[k].append(d[k])
                    break
        agg[step] = {k: (np.mean(v), np.std(v)) for k, v in vals.items()}
    return agg


def count_learned(mode_dir_name, threshold=50, min_step=70000):
    """Count seeds that learned (ep_rew > threshold at any step >= min_step)."""
    count = 0
    for seed in SEEDS:
        snaps = load_snapshots(mode_dir_name, seed)
        max_late = max((d["ep_rew"] for d in snaps if d["step"] >= min_step), default=0)
        if max_late > threshold:
            count += 1
    return count


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Build directory names: alpha=0.6 is "td-per", others are "td-per_a{alpha}"
    def dir_name(alpha):
        if alpha == 0.6:
            return "td-per"
        return f"td-per_a{alpha}"

    all_agg = {}
    for alpha in ALPHAS:
        all_agg[alpha] = aggregate_seeds(dir_name(alpha))

    # Also load uniform baseline
    uniform_agg = aggregate_seeds("uniform")

    # Panel (a): Episode Return
    ax = axes[0, 0]
    # Uniform baseline
    if uniform_agg:
        steps = sorted(uniform_agg.keys())
        means = [uniform_agg[s]["ep_rew"][0] for s in steps]
        stds = [uniform_agg[s]["ep_rew"][1] for s in steps]
        ax.plot(steps, means, "o-", label="Uniform (baseline)",
                color="#2ecc71", linewidth=2, markersize=4)
        ax.fill_between(steps, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], alpha=0.15, color="#2ecc71")
    for alpha in ALPHAS:
        agg = all_agg[alpha]
        if not agg:
            continue
        steps = sorted(agg.keys())
        means = [agg[s]["ep_rew"][0] for s in steps]
        stds = [agg[s]["ep_rew"][1] for s in steps]
        ax.plot(steps, means, "o-", label=ALPHA_LABELS[alpha],
                color=ALPHA_COLORS[alpha], linewidth=2, markersize=4)
        ax.fill_between(steps, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], alpha=0.15, color=ALPHA_COLORS[alpha])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return (sparse)")
    ax.set_title("(a) Learning Curves (n=5 seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): Q-value
    ax = axes[0, 1]
    if uniform_agg:
        steps = sorted(uniform_agg.keys())
        means = [uniform_agg[s]["q_mean"][0] for s in steps]
        ax.plot(steps, means, "o-", label="Uniform", color="#2ecc71", linewidth=2, markersize=4)
    for alpha in ALPHAS:
        agg = all_agg[alpha]
        if not agg:
            continue
        steps = sorted(agg.keys())
        means = [agg[s]["q_mean"][0] for s in steps]
        stds = [agg[s]["q_mean"][1] for s in steps]
        ax.plot(steps, means, "o-", label=ALPHA_LABELS[alpha],
                color=ALPHA_COLORS[alpha], linewidth=2, markersize=4)
        ax.fill_between(steps, [max(0,m-s) for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], alpha=0.15, color=ALPHA_COLORS[alpha])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Q-value")
    ax.set_title("(b) Q-Value Stability")
    ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(True, alpha=0.3)

    # Panel (c): |TD| magnitude
    ax = axes[0, 2]
    if uniform_agg:
        steps = sorted(uniform_agg.keys())
        means = [uniform_agg[s]["abs_td_mean"][0] for s in steps]
        ax.plot(steps, means, "o-", label="Uniform", color="#2ecc71", linewidth=2, markersize=4)
    for alpha in ALPHAS:
        agg = all_agg[alpha]
        if not agg:
            continue
        steps = sorted(agg.keys())
        means = [agg[s]["abs_td_mean"][0] for s in steps]
        stds = [agg[s]["abs_td_mean"][1] for s in steps]
        ax.plot(steps, means, "o-", label=ALPHA_LABELS[alpha],
                color=ALPHA_COLORS[alpha], linewidth=2, markersize=4)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean |TD-error|")
    ax.set_title("(c) TD-Error Magnitude")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel (d): Spearman
    ax = axes[1, 0]
    if uniform_agg:
        steps = sorted(uniform_agg.keys())
        means = [uniform_agg[s]["spearman"][0] for s in steps]
        ax.plot(steps, means, "o-", label="Uniform", color="#2ecc71", linewidth=2, markersize=4)
    for alpha in ALPHAS:
        agg = all_agg[alpha]
        if not agg:
            continue
        steps = sorted(agg.keys())
        means = [agg[s]["spearman"][0] for s in steps]
        stds = [agg[s]["spearman"][1] for s in steps]
        ax.plot(steps, means, "o-", label=ALPHA_LABELS[alpha],
                color=ALPHA_COLORS[alpha], linewidth=2, markersize=4)
        ax.fill_between(steps, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], alpha=0.15, color=ALPHA_COLORS[alpha])
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Spearman(|TD|, Oracle Advantage)")
    ax.set_title("(d) TD-Oracle Correlation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (e): Learning success bar chart
    ax = axes[1, 1]
    labels_list = ["Uniform"] + [ALPHA_LABELS[a] for a in ALPHAS]
    counts = [count_learned("uniform")] + [count_learned(dir_name(a)) for a in ALPHAS]
    colors = ["#2ecc71"] + [ALPHA_COLORS[a] for a in ALPHAS]
    x = range(len(labels_list))
    bars = ax.bar(x, counts, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_list, fontsize=8, rotation=15)
    ax.set_ylabel("Seeds that Learned (out of 5)")
    ax.set_title("(e) Learning Success Rate")
    ax.set_ylim(0, 5.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{c}/5", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel (f): Individual seed trajectories
    ax = axes[1, 2]
    # Uniform
    for seed in SEEDS:
        snaps = load_snapshots("uniform", seed)
        if snaps:
            ax.plot([d["step"] for d in snaps], [d["ep_rew"] for d in snaps],
                    "-", color="#2ecc71", alpha=0.25, linewidth=1)
    ax.plot([], [], "-", color="#2ecc71", linewidth=2, label="Uniform")
    for alpha in ALPHAS:
        for seed in SEEDS:
            snaps = load_snapshots(dir_name(alpha), seed)
            if snaps:
                ax.plot([d["step"] for d in snaps], [d["ep_rew"] for d in snaps],
                        "-", color=ALPHA_COLORS[alpha], alpha=0.25, linewidth=1)
        ax.plot([], [], "-", color=ALPHA_COLORS[alpha], linewidth=2, label=ALPHA_LABELS[alpha])
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title("(f) Individual Seed Trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Alpha Sweep: TD-PER on reach-v3 (5 seeds, 100k steps)\n"
        "Does lower prioritization exponent (α) mitigate Q-explosion?",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"alpha_sweep_td_per.{ext}",
                    dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir / 'alpha_sweep_td_per.png'}")
    plt.close()


if __name__ == "__main__":
    main()
