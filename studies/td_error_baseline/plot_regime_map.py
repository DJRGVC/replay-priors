"""TD-Error PER Regime Map — comprehensive diagnostic figure.

Creates a 6-panel figure that tells the full story of TD-error PER failure:
  Row 1: [Q-value dynamics | TD-PER regime classification | Spearman trajectory]
  Row 2: [Mutual information | Wasted sampling budget | Cross-study bridge to VLM]

This figure is designed to be self-contained and presentation-ready. It annotates
training phases, failure modes, and the motivation for VLM-based prioritization.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def load_all_metrics():
    """Load both 100k and 300k oracle metrics, merge by task+seed."""
    base = Path(__file__).parent / "snapshots"

    runs = {}  # (task, seed) -> list of snapshot dicts

    for metrics_file in ["oracle_metrics.json", "oracle_metrics_300k.json"]:
        path = base / metrics_file
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for run_dir, info in data.items():
            task = info["task"]
            # extract seed from dir name
            dirname = Path(run_dir).name
            if "_s" in dirname:
                seed = int(dirname.split("_s")[-1])
            else:
                seed = 42
            key = (task, seed)
            if key not in runs:
                runs[key] = []
            existing_steps = {s["step"] for s in runs[key]}
            for snap in info["snapshots"]:
                if snap["step"] not in existing_steps:
                    runs[key].append(snap)
            runs[key].sort(key=lambda s: s["step"])

    return runs


def compute_mutual_information_proxy(runs):
    """Compute binned MI proxy between |TD| and oracle advantage.

    We approximate MI using the top-K overlap metric:
      MI_proxy(step) = log2(overlap / chance) when overlap > chance, else 0
    This gives bits of information that |TD| ranking provides about oracle ranking.
    """
    mi_data = {}
    for key, snapshots in runs.items():
        steps = [s["step"] for s in snapshots]
        overlaps = [s["top_k_overlap"] for s in snapshots]
        chance = 0.10  # top-10%
        # Convert overlap to bits of "excess information"
        mi_proxy = [max(0, np.log2(o / chance)) if o > chance else 0 for o in overlaps]
        mi_data[key] = (np.array(steps), np.array(mi_proxy))
    return mi_data


def classify_regime(spearman, q_std_over_mean):
    """Classify each timestep into a TD-PER regime.

    Returns: list of regime labels:
      'noise'     — |spearman| < 0.15 (TD-error is uninformative)
      'aligned'   — spearman ≥ 0.15 (TD-error agrees with oracle)
      'inverted'  — spearman ≤ -0.15 (TD-error anti-correlates with oracle)
      'unstable'  — Q std/mean > 1.0 (Q-values are diverging/oscillating)
    """
    regimes = []
    for rho, q_ratio in zip(spearman, q_std_over_mean):
        if q_ratio > 1.0:
            regimes.append("unstable")
        elif rho <= -0.15:
            regimes.append("inverted")
        elif rho >= 0.15:
            regimes.append("aligned")
        else:
            regimes.append("noise")
    return regimes


def compute_wasted_budget(runs):
    """Estimate the fraction of PER's sampling budget that is wasted.

    'Wasted' = fraction of top-10% |TD| transitions that are NOT in top-10%
    oracle advantage. This is (1 - overlap). Under PER with alpha=0.6,
    these transitions get ~2-3x the sampling weight of uniform — so waste
    accumulates proportionally to how skewed (Gini) the priorities are.

    Effective waste = (1 - overlap) * gini_normalized
    """
    waste_data = {}
    for key, snapshots in runs.items():
        steps = [s["step"] for s in snapshots]
        waste = []
        for s in snapshots:
            overlap = s["top_k_overlap"]
            gini = s["priority_gini"]
            # waste = prob of sampling wrong transition * how concentrated priorities are
            # Interpretation: if gini is high and overlap is low, PER is confidently wrong
            w = (1.0 - overlap) * gini
            waste.append(w)
        waste_data[key] = (np.array(steps), np.array(waste))
    return waste_data


def main():
    runs = load_all_metrics()
    mi_data = compute_mutual_information_proxy(runs)
    waste_data = compute_wasted_budget(runs)

    # --- Figure setup ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    task_colors = {"reach-v3": "#2196F3", "pick-place-v3": "#E53935"}
    seed_styles = {42: "-", 123: "--"}
    regime_colors = {
        "noise": "#BDBDBD",
        "aligned": "#4CAF50",
        "inverted": "#E53935",
        "unstable": "#FF9800"
    }

    # =============================================
    # Panel 1 (top-left): Q-value dynamics
    # =============================================
    ax = axes[0, 0]
    for (task, seed), snapshots in sorted(runs.items()):
        steps = np.array([s["step"] for s in snapshots]) / 1000
        q_means = np.array([s["q_mean"] for s in snapshots])
        q_stds = np.array([s["q_std"] for s in snapshots])

        c = task_colors[task]
        ls = seed_styles[seed]
        label = f"{task} s{seed}"

        ax.semilogy(steps, q_means, color=c, linestyle=ls, linewidth=1.5,
                     label=label, alpha=0.8)
        ax.fill_between(steps,
                        np.maximum(q_means - q_stds, 0.001),
                        q_means + q_stds,
                        color=c, alpha=0.08)

    ax.set_ylabel("Q-value (log scale)", fontsize=10)
    ax.set_xlabel("Env steps (×1000)", fontsize=10)
    ax.set_title("A. Critic Dynamics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(alpha=0.2)

    # Add annotation for Q-instability
    ax.annotate("Q-values diverge\n(pick-place s42)",
                xy=(90, 500), fontsize=7, color="#E53935",
                ha="center", style="italic")

    # =============================================
    # Panel 2 (top-center): Regime classification heatmap
    # =============================================
    ax = axes[0, 1]

    # Order: reach-v3 seeds first, then pick-place
    run_order = sorted(runs.keys(), key=lambda k: (0 if k[0] == "reach-v3" else 1, k[1]))

    regime_matrix = []
    run_labels = []
    max_steps = 0

    for (task, seed) in run_order:
        snapshots = runs[(task, seed)]
        steps = [s["step"] for s in snapshots]
        spearman = [s["spearman_r"] for s in snapshots]
        q_means = [s["q_mean"] for s in snapshots]
        q_stds = [s["q_std"] for s in snapshots]
        q_ratio = [std / max(mean, 0.01) for mean, std in zip(q_means, q_stds)]

        regimes = classify_regime(spearman, q_ratio)
        regime_nums = {"noise": 0, "aligned": 1, "inverted": 2, "unstable": 3}
        regime_matrix.append([regime_nums[r] for r in regimes])
        run_labels.append(f"{task}\ns{seed}")
        max_steps = max(max_steps, len(steps))

    # Pad shorter runs with NaN
    for i in range(len(regime_matrix)):
        while len(regime_matrix[i]) < max_steps:
            regime_matrix[i].append(np.nan)

    regime_arr = np.array(regime_matrix, dtype=float)

    cmap = LinearSegmentedColormap.from_list("regime",
        [regime_colors["noise"], regime_colors["aligned"],
         regime_colors["inverted"], regime_colors["unstable"]], N=4)

    im = ax.imshow(regime_arr, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                   interpolation="nearest")

    ax.set_yticks(range(len(run_labels)))
    ax.set_yticklabels(run_labels, fontsize=8)

    # x-axis: steps
    step_labels = [s["step"] for s in runs[run_order[0]]]
    tick_positions = list(range(0, max_steps, max(1, max_steps // 6)))
    ax.set_xticks(tick_positions)
    max_step_val = max(s["step"] for snaps in runs.values() for s in snaps)
    ax.set_xticklabels([f"{step_labels[i]//1000}k" if i < len(step_labels) else ""
                        for i in tick_positions], fontsize=8)
    ax.set_xlabel("Env steps", fontsize=10)
    ax.set_title("B. TD-PER Regime Classification", fontsize=12, fontweight="bold")

    # Legend patches
    patches = [mpatches.Patch(color=regime_colors[r], label=r.capitalize())
               for r in ["noise", "aligned", "inverted", "unstable"]]
    ax.legend(handles=patches, fontsize=7, loc="lower right", ncol=2)

    # =============================================
    # Panel 3 (top-right): Spearman trajectory with regime coloring
    # =============================================
    ax = axes[0, 2]

    for (task, seed), snapshots in sorted(runs.items()):
        steps = np.array([s["step"] for s in snapshots]) / 1000
        spearman = np.array([s["spearman_r"] for s in snapshots])
        q_means = [s["q_mean"] for s in snapshots]
        q_stds = [s["q_std"] for s in snapshots]
        q_ratio = [std / max(mean, 0.01) for mean, std in zip(q_means, q_stds)]

        regimes = classify_regime(spearman, q_ratio)

        c = task_colors[task]
        ls = seed_styles[seed]

        # Plot line
        ax.plot(steps, spearman, color=c, linestyle=ls, linewidth=1.2, alpha=0.4,
                label=f"{task} s{seed}")

        # Color points by regime
        for i, (x, y, r) in enumerate(zip(steps, spearman, regimes)):
            ax.scatter(x, y, color=regime_colors[r], s=25, zorder=3,
                       edgecolors=c, linewidths=0.5)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.15, color="#4CAF50", linestyle=":", alpha=0.3, label="±0.15 threshold")
    ax.axhline(-0.15, color="#E53935", linestyle=":", alpha=0.3)
    ax.set_ylabel("Spearman(|TD|, oracle adv.)", fontsize=10)
    ax.set_xlabel("Env steps (×1000)", fontsize=10)
    ax.set_title("C. Correlation Trajectory", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 0.8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.2)

    # Annotate the inversion
    ax.annotate("INVERSION\n(anti-informative)",
                xy=(280, -0.31), xytext=(220, -0.42),
                fontsize=8, color="#E53935", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
                ha="center")

    # =============================================
    # Panel 4 (bottom-left): Mutual Information proxy
    # =============================================
    ax = axes[1, 0]

    for (task, seed), (steps, mi) in sorted(mi_data.items()):
        c = task_colors[task]
        ls = seed_styles[seed]
        ax.plot(steps / 1000, mi, color=c, linestyle=ls, linewidth=1.5,
                label=f"{task} s{seed}", alpha=0.8)
        ax.fill_between(steps / 1000, 0, mi, color=c, alpha=0.1)

    ax.set_ylabel("MI proxy (bits)", fontsize=10)
    ax.set_xlabel("Env steps (×1000)", fontsize=10)
    ax.set_title("D. Information Content of |TD| Ranking", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 3.0)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)

    # Annotate the information desert
    ax.annotate("Information desert\n(first 40-70k steps)",
                xy=(40, 0.05), fontsize=8, color="gray", style="italic",
                ha="center", va="bottom")

    # =============================================
    # Panel 5 (bottom-center): Wasted sampling budget
    # =============================================
    ax = axes[1, 1]

    for (task, seed), (steps, waste) in sorted(waste_data.items()):
        c = task_colors[task]
        ls = seed_styles[seed]
        ax.fill_between(steps / 1000, 0, waste * 100, color=c, alpha=0.15)
        ax.plot(steps / 1000, waste * 100, color=c, linestyle=ls, linewidth=1.5,
                label=f"{task} s{seed}", alpha=0.8)

    ax.set_ylabel("Effective waste (%)", fontsize=10)
    ax.set_xlabel("Env steps (×1000)", fontsize=10)
    ax.set_title("E. PER Sampling Budget Wasted", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    ax.annotate("PER confidently samples\nwrong transitions",
                xy=(75, 50), fontsize=7, color="#E53935", ha="center", style="italic")

    # =============================================
    # Panel 6 (bottom-right): The Case for VLM Prioritization
    # =============================================
    ax = axes[1, 2]

    # This is a conceptual/summary panel — show regime breakdown as stacked bars
    # plus VLM probe accuracy annotation

    task_regime_fracs = {}
    for (task, seed), snapshots in sorted(runs.items()):
        spearman = [s["spearman_r"] for s in snapshots]
        q_means = [s["q_mean"] for s in snapshots]
        q_stds = [s["q_std"] for s in snapshots]
        q_ratio = [std / max(mean, 0.01) for mean, std in zip(q_means, q_stds)]
        regimes = classify_regime(spearman, q_ratio)

        key = f"{task}\ns{seed}"
        total = len(regimes)
        task_regime_fracs[key] = {
            r: regimes.count(r) / total for r in ["noise", "aligned", "inverted", "unstable"]
        }

    # Stacked horizontal bar chart
    labels = list(task_regime_fracs.keys())
    y_pos = np.arange(len(labels))

    left = np.zeros(len(labels))
    for regime in ["noise", "inverted", "unstable", "aligned"]:
        widths = [task_regime_fracs[l][regime] * 100 for l in labels]
        bars = ax.barh(y_pos, widths, left=left, height=0.6,
                       color=regime_colors[regime], label=regime.capitalize(),
                       edgecolor="white", linewidth=0.5)
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% of training time", fontsize=10)
    ax.set_title("F. When TD-PER Works vs. Fails", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    # Add VLM annotation as text box
    textstr = ("VLM probe (sibling): reach-v3 K=8 MAE=41.9, within-20=35%\n"
               "→ Available from step 0, no critic dependency, no inversion")
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
            fontsize=6.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor="#4CAF50", alpha=0.9))

    # =============================================
    # Overall title and layout
    # =============================================
    fig.suptitle(
        "TD-Error PER Regime Map: Why Sparse-Reward Manipulation Needs Better Replay Prioritization",
        fontsize=14, fontweight="bold", y=1.01
    )

    fig.tight_layout()

    out_path = Path(__file__).parent / "figures" / "td_per_regime_map.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    # Also save as PDF for presentation
    pdf_path = str(out_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved to {pdf_path}")

    plt.close(fig)

    # Print summary statistics
    print("\n=== Regime Summary ===")
    for label, fracs in task_regime_fracs.items():
        clean_label = label.replace("\n", " ")
        total_bad = fracs["noise"] + fracs["inverted"] + fracs["unstable"]
        print(f"  {clean_label:25s}  aligned={fracs['aligned']:.0%}  "
              f"noise={fracs['noise']:.0%}  inverted={fracs['inverted']:.0%}  "
              f"unstable={fracs['unstable']:.0%}  "
              f"[TD-PER fails {total_bad:.0%} of training]")


if __name__ == "__main__":
    main()
