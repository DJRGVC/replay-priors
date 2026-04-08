"""Plot comparison of adaptive vs td-per vs uniform modes on reach-v3.

Produces a 6-panel figure comparing TD-error dynamics, Q-value evolution,
Spearman correlation, and regime detection across three prioritization modes.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SNAP_ROOT = Path(__file__).parent / "snapshots"
MODES = ["adaptive", "td-per", "uniform"]
MODE_LABELS = {"adaptive": "Adaptive Mixer", "td-per": "TD-PER", "uniform": "Uniform"}
MODE_COLORS = {"adaptive": "#e74c3c", "td-per": "#3498db", "uniform": "#2ecc71"}
TASK = "reach-v3"
SEED = 42


def load_snapshots(mode: str):
    """Load all snapshot .npz files for a given mode."""
    snap_dir = SNAP_ROOT / f"{TASK}_s{SEED}_{mode}" / "td_snapshots"
    if not snap_dir.exists():
        return []
    results = []
    for f in sorted(snap_dir.glob("snapshot_*.npz")):
        step = int(f.stem.split("_")[1])
        data = dict(np.load(f, allow_pickle=True))
        results.append({
            "step": step,
            "abs_td_mean": float(np.mean(data["abs_td_errors"])),
            "abs_td_median": float(np.median(data["abs_td_errors"])),
            "abs_td_p90": float(np.percentile(data["abs_td_errors"], 90)),
            "spearman": float(data.get("td_dense_spearman", 0)),
            "pearson": float(data.get("td_dense_pearson", 0)),
            "q_mean": float(data.get("q_mean", 0)),
            "buffer_size": int(data.get("buffer_size", 0)),
        })
    return results


def load_regimes(mode: str):
    """Load regime JSON files for a given mode."""
    snap_dir = SNAP_ROOT / f"{TASK}_s{SEED}_{mode}" / "td_snapshots"
    if not snap_dir.exists():
        return []
    results = []
    for f in sorted(snap_dir.glob("regime_*.json")):
        step = int(f.stem.split("_")[1])
        with open(f) as jf:
            data = json.load(jf)
        data["step"] = step
        results.append(data)
    return results


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Load data
    all_data = {}
    all_regimes = {}
    for mode in MODES:
        all_data[mode] = load_snapshots(mode)
        all_regimes[mode] = load_regimes(mode)

    # Panel 1: |TD| mean over training
    ax = axes[0, 0]
    for mode in MODES:
        data = all_data[mode]
        steps = [d["step"] for d in data]
        vals = [d["abs_td_mean"] for d in data]
        ax.plot(steps, vals, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean |TD-error|")
    ax.set_title("(a) TD-Error Magnitude")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 2: Q-values
    ax = axes[0, 1]
    for mode in MODES:
        data = all_data[mode]
        steps = [d["step"] for d in data]
        vals = [d["q_mean"] for d in data]
        ax.plot(steps, vals, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Q-value")
    ax.set_title("(b) Q-Value Evolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Spearman correlation (|TD| vs oracle advantage)
    ax = axes[0, 2]
    for mode in MODES:
        data = all_data[mode]
        steps = [d["step"] for d in data]
        vals = [d["spearman"] for d in data]
        ax.plot(steps, vals, "o-", label=MODE_LABELS[mode],
                color=MODE_COLORS[mode], linewidth=2, markersize=5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Spearman(|TD|, Oracle Advantage)")
    ax.set_title("(c) TD-Oracle Correlation")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.15, 0.2)
    ax.grid(True, alpha=0.3)

    # Panel 4: |TD| percentiles for td-per mode
    ax = axes[1, 0]
    data = all_data["td-per"]
    steps = [d["step"] for d in data]
    ax.fill_between(steps,
                     [d["abs_td_median"] for d in data],
                     [d["abs_td_p90"] for d in data],
                     alpha=0.3, color=MODE_COLORS["td-per"], label="p50–p90")
    ax.plot(steps, [d["abs_td_mean"] for d in data], "o-",
            color=MODE_COLORS["td-per"], linewidth=2, markersize=5, label="Mean")
    ax.plot(steps, [d["abs_td_median"] for d in data], "s--",
            color=MODE_COLORS["td-per"], linewidth=1, markersize=4, label="Median")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("|TD-error|")
    ax.set_title("(d) TD-PER: |TD| Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: Regime detection for adaptive mode
    ax = axes[1, 1]
    regime_colors = {
        "aligned": "#2ecc71", "noise": "#f39c12",
        "inverted": "#e74c3c", "unstable": "#9b59b6",
    }
    if all_regimes.get("adaptive"):
        regimes = all_regimes["adaptive"]
        steps = [r["step"] for r in regimes]
        q_cvs = [r.get("q_cv", 0) for r in regimes]
        td_ginis = [r.get("td_gini", 0) for r in regimes]
        colors = [regime_colors.get(r.get("current", "noise"), "gray") for r in regimes]

        ax.bar(steps, q_cvs, width=8000, color=colors, alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(steps, td_ginis, "k^-", markersize=6, label="TD Gini")
        ax2.set_ylabel("TD Gini", color="black")
        ax2.set_ylim(0, 1)

        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()
                          if any(reg.get("current") == r for reg in regimes)]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Q Coefficient of Variation")
    ax.set_title("(e) Adaptive: Regime Detection")
    ax.grid(True, alpha=0.3)

    # Panel 6: Regime detection for td-per mode
    ax = axes[1, 2]
    if all_regimes.get("td-per"):
        regimes = all_regimes["td-per"]
        steps = [r["step"] for r in regimes]
        q_cvs = [r.get("q_cv", 0) for r in regimes]
        td_ginis = [r.get("td_gini", 0) for r in regimes]
        colors = [regime_colors.get(r.get("current", "noise"), "gray") for r in regimes]

        ax.bar(steps, q_cvs, width=8000, color=colors, alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(steps, td_ginis, "k^-", markersize=6, label="TD Gini")
        ax2.set_ylabel("TD Gini", color="black")
        ax2.set_ylim(0, 1)

        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()
                          if any(reg.get("current") == r for reg in regimes)]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Q Coefficient of Variation")
    ax.set_title("(f) TD-PER: Regime Detection")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Mode Comparison: {TASK} (seed={SEED}, 100k steps) — PER Active\n"
        f"Working PER destabilizes Q: adaptive Q=37 (unstable), td-per Q=2.2, uniform Q=0.5\n"
        f"Spearman(|TD|, oracle) ≈ 0 across all modes — no policy learned",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"mode_comparison_reach_v3.{ext}", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir / 'mode_comparison_reach_v3.png'}")
    plt.close()


if __name__ == "__main__":
    main()
