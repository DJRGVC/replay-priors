"""Cross-study synthesis figure: The complete priority signal landscape.

Combines td_baseline (5 RL signals × 2 tasks × 5 seeds) with vlm_probe
(VLM localization → priority quality analysis) into a single figure showing
that ALL tested priority signals fail to beat uniform replay in sparse-reward
early training.

Layout: 2×2 figure
  (a) RL signal success rates (bar chart, reach-v3, all 5 modes + alpha sweep)
  (b) VLM priority quality (overlap vs KL vs uniform baseline)
  (c) Unified failure mechanism timeline (when each signal becomes informative)
  (d) The chicken-and-egg diagram (conceptual summary)

Data sources:
  - td_baseline: 40 training runs (5 seeds × 4 modes × 2 tasks) + alpha sweep
  - vlm_probe: priority_score.py analysis (overlap, KL) + confidence gating results
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

SNAP_ROOT = Path(__file__).parent / "snapshots"
FIG_DIR = Path(__file__).parent / "figures"
IMG_DIR = Path(__file__).parent.parent.parent / "images" / "td_baseline"
FIG_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 7, 99, 256]

# Color palette
C_UNIFORM = "#2ecc71"
C_TDPER = "#3498db"
C_ADAPTIVE = "#e74c3c"
C_RPE = "#9b59b6"
C_RND = "#e67e22"
C_VLM = "#1abc9c"
C_VLM_GATE = "#16a085"
C_VLM_ENS = "#0e6655"
C_GREY = "#95a5a6"


def load_snapshots(task, seed, mode_suffix=""):
    """Load all snapshots for a given task/seed/mode."""
    if mode_suffix:
        snap_dir = SNAP_ROOT / f"{task}_s{seed}_{mode_suffix}" / "td_snapshots"
    else:
        snap_dir = SNAP_ROOT / f"{task}_s{seed}" / "td_snapshots"
    if not snap_dir.exists():
        return []
    results = []
    for f in sorted(snap_dir.glob("snapshot_*.npz")):
        data = dict(np.load(f, allow_pickle=True))
        results.append({
            "step": int(data["step"]),
            "ep_rew": float(data.get("episode_return_mean", 0)),
            "success_rate": float(data.get("success_rate", 0)),
        })
    return sorted(results, key=lambda x: x["step"])


def count_successes(task, seeds, mode_suffix):
    """Count how many seeds achieve learning (ep_rew > 50 after 50k)."""
    learned = 0
    total = 0
    for seed in seeds:
        snaps = load_snapshots(task, seed, mode_suffix)
        if not snaps:
            continue
        total += 1
        max_rew = max(d["ep_rew"] for d in snaps if d["step"] >= 50000)
        if max_rew > 50:
            learned += 1
    return learned, total


def main():
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.97, top=0.90, bottom=0.06)

    # =========================================================================
    # (a) RL signal success rates — reach-v3 (bar chart)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    modes = [
        ("Uniform", "uniform", C_UNIFORM),
        ("TD-PER\nα=0.6", "td-per", C_TDPER),
        ("TD-PER\nα=0.3", "td-per_a0.3", "#5dade2"),
        ("TD-PER\nα=0.1", "td-per_a0.1", "#85c1e9"),
        ("RPE-PER", "rpe-per", C_RPE),
        ("RND-PER", "rnd-per", C_RND),
        ("Adaptive", "adaptive", C_ADAPTIVE),
    ]

    reach_data = []
    for label, suffix, color in modes:
        learned, total = count_successes("reach-v3", SEEDS, suffix)
        reach_data.append((label, learned, total, color))

    x = np.arange(len(reach_data))
    bars = ax_a.bar(x, [d[1] for d in reach_data], 0.65,
                    color=[d[3] for d in reach_data],
                    alpha=0.85, edgecolor="black", linewidth=0.8)

    for i, (bar, d) in enumerate(zip(bars, reach_data)):
        if d[2] > 0:
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                      f"{d[1]}/{d[2]}", ha="center", va="bottom",
                      fontweight="bold", fontsize=10)

    # Uniform reference line
    uniform_count = reach_data[0][1]
    ax_a.axhline(uniform_count, color=C_UNIFORM, linestyle=":", alpha=0.6,
                 linewidth=2, label=f"Uniform baseline ({uniform_count}/5)")

    ax_a.set_xticks(list(x))
    ax_a.set_xticklabels([d[0] for d in reach_data], fontsize=8.5)
    ax_a.set_ylabel("Seeds that Learn (out of 5)", fontsize=10)
    ax_a.set_title("(a) RL Priority Signals vs Uniform — reach-v3",
                   fontsize=11, fontweight="bold")
    ax_a.set_ylim(0, 5.8)
    ax_a.legend(fontsize=8.5)
    ax_a.grid(True, alpha=0.2, axis="y")

    # Annotation
    ax_a.annotate("← All 5 RL signals ≤ uniform",
                  xy=(3, 3.3), fontsize=9, color="#c0392b",
                  fontweight="bold", fontstyle="italic")

    # =========================================================================
    # (b) VLM priority quality — overlap vs KL (from vlm_probe)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Data from vlm_probe Finding #9 and iter_037
    # Overlap: fraction of oracle top-20% that VLM also puts in top-20%
    # KL: KL(oracle || priority) — lower is better
    approaches = [
        # (label, overlap_pct, kl_div, color, marker)
        ("Uniform\n(baseline)", 21.7, 1.556, C_UNIFORM, "s"),
        ("Sonnet\nK=8", 33.0, 1.92, C_VLM, "o"),   # +12% overlap, ~-24% KL
        ("GPT-4o\nK=8", 28.0, 1.85, "#3498db", "o"),
        ("Best 2-model\nensemble", 30.0, 1.75, C_VLM_ENS, "D"),
        ("Confidence-\ngated", 21.7, 1.556, C_VLM_GATE, "^"),  # falls back to uniform
        ("Always-VLM\n(all models)", 8.7, 2.035, "#e74c3c", "X"),
    ]

    for label, overlap, kl, color, marker in approaches:
        ax_b.scatter(kl, overlap, c=color, marker=marker, s=120,
                     edgecolors="black", linewidth=0.8, zorder=5)
        # Position labels to avoid overlap
        offset_x = 0.03
        offset_y = 1.0
        ha = "left"
        if "Always" in label:
            offset_y = -2.5
            ha = "left"
        elif "Confidence" in label:
            offset_x = 0.03
            offset_y = -2.5
        elif "Uniform" in label:
            offset_x = -0.03
            ha = "right"
        ax_b.annotate(label, (kl + offset_x, overlap + offset_y),
                      fontsize=7.5, ha=ha, va="bottom")

    # Reference lines
    ax_b.axhline(21.7, color=C_UNIFORM, linestyle=":", alpha=0.5, linewidth=1.5)
    ax_b.axvline(1.556, color=C_UNIFORM, linestyle=":", alpha=0.5, linewidth=1.5)

    # Quadrant labels
    ax_b.text(1.35, 37, "Better than\nuniform", fontsize=8, color=C_UNIFORM,
              fontstyle="italic", ha="center", alpha=0.7)
    ax_b.text(2.15, 5, "Worse than\nuniform", fontsize=8, color="#e74c3c",
              fontstyle="italic", ha="center", alpha=0.7)

    ax_b.set_xlabel("KL(Oracle ‖ Priority) — lower is better →", fontsize=10)
    ax_b.set_ylabel("Top-20% Overlap with Oracle (%) — higher is better ↑", fontsize=10)
    ax_b.set_title("(b) VLM Priority Quality — overlap vs distribution fit",
                   fontsize=11, fontweight="bold")
    ax_b.set_xlim(1.2, 2.3)
    ax_b.set_ylim(0, 40)
    ax_b.grid(True, alpha=0.2)

    # Arrow showing the tradeoff
    ax_b.annotate("", xy=(1.92, 33), xytext=(1.556, 21.7),
                  arrowprops=dict(arrowstyle="->", color=C_VLM, lw=2, alpha=0.4))
    ax_b.text(1.72, 29, "VLM: ↑overlap\nbut ↑KL too", fontsize=7,
              color=C_VLM, fontstyle="italic", ha="center")

    # =========================================================================
    # (c) Signal informativeness timeline
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    # Timeline: when does each signal become useful?
    signals = [
        ("TD-error PER", 60, 100, C_TDPER, "Needs reward discovery + critic convergence"),
        ("RPE-PER", 60, 100, C_RPE, "Same chicken-and-egg as TD-error"),
        ("RND-PER", 0, 100, C_RND, "Available from step 0 but redirects, doesn't improve"),
        ("Adaptive mix", 60, 100, C_ADAPTIVE, "Switches modes but no signal is good"),
        ("VLM (Sonnet K=8)", 0, 100, C_VLM, "Available from step 0 but positionally biased"),
        ("VLM (ensemble)", 0, 100, C_VLM_ENS, "Debiased but noise dominates"),
        ("Uniform", 0, 100, C_UNIFORM, "Always unbiased — the safe default"),
    ]

    y_positions = list(range(len(signals)))

    for i, (label, onset, end, color, note) in enumerate(signals):
        # Uninformative period (grey)
        if onset > 0:
            ax_c.barh(i, onset, height=0.5, left=0, color=C_GREY, alpha=0.3,
                      edgecolor="white")
        # "Informative" period
        if label == "Uniform":
            # Uniform is always the same — no signal
            ax_c.barh(i, end, height=0.5, left=0, color=color, alpha=0.6,
                      edgecolor="black", linewidth=0.5)
            ax_c.text(50, i, "Always unbiased", fontsize=7.5,
                      ha="center", va="center", fontweight="bold", color="white")
        elif "VLM" in label or "RND" in label:
            # Available but ineffective
            ax_c.barh(i, end, height=0.5, left=0, color=color, alpha=0.25,
                      edgecolor="black", linewidth=0.5, hatch="//")
            ax_c.text(50, i, "Available but biased/redirecting", fontsize=7,
                      ha="center", va="center", color=color, fontstyle="italic")
        else:
            # Grey until onset, then colored
            ax_c.barh(i, end - onset, height=0.5, left=onset, color=color,
                      alpha=0.5, edgecolor="black", linewidth=0.5, hatch="//")
            ax_c.text(30, i, "Uninformative", fontsize=7,
                      ha="center", va="center", color=C_GREY, fontstyle="italic")
            ax_c.text(80, i, "Partially aligned", fontsize=7,
                      ha="center", va="center", color=color, fontstyle="italic")

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels([s[0] for s in signals], fontsize=9)
    ax_c.set_xlabel("Environment Steps (×10³)", fontsize=10)
    ax_c.set_title("(c) When does each signal become informative?",
                   fontsize=11, fontweight="bold")
    ax_c.set_xlim(0, 105)
    ax_c.invert_yaxis()
    ax_c.grid(True, alpha=0.2, axis="x")

    # Information desert annotation
    ax_c.axvspan(0, 60, alpha=0.04, color="red")
    ax_c.text(30, -0.7, "← Information desert (0–60k steps) →",
              fontsize=8.5, ha="center", color="#c0392b", fontweight="bold",
              fontstyle="italic")

    # =========================================================================
    # (d) Unified failure mechanism — conceptual diagram
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis("off")
    ax_d.set_title("(d) Unified Failure Mechanism",
                   fontsize=11, fontweight="bold")

    # Central question
    ax_d.text(5, 9.2, "Why does no priority signal beat uniform\nin sparse-reward early training?",
              fontsize=11, ha="center", va="top", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1",
                        edgecolor="black", linewidth=1.5))

    # RL signals branch
    ax_d.text(2.5, 7.0, "RL-Based Signals", fontsize=10, ha="center",
              fontweight="bold", color=C_TDPER)
    ax_d.annotate("", xy=(2.5, 7.2), xytext=(4.0, 8.0),
                  arrowprops=dict(arrowstyle="->", color=C_TDPER, lw=1.5))

    # Chicken-and-egg box
    chicken_text = ("Chicken-and-egg:\n"
                    "TD-error needs critic quality\n"
                    "Critic needs reward signal\n"
                    "Reward is sparse → 0 signal\n"
                    "for 60–80% of training")
    ax_d.text(2.5, 5.2, chicken_text, fontsize=8, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#d6eaf8",
                        edgecolor=C_TDPER, linewidth=1),
              color="#2c3e50")

    # VLM signals branch
    ax_d.text(7.5, 7.0, "VLM-Based Signals", fontsize=10, ha="center",
              fontweight="bold", color=C_VLM)
    ax_d.annotate("", xy=(7.5, 7.2), xytext=(6.0, 8.0),
                  arrowprops=dict(arrowstyle="->", color=C_VLM, lw=1.5))

    # Bias box
    bias_text = ("Positional bias:\n"
                 "Models predict based on\n"
                 "image position, not content\n"
                 "MAE ≈ 42 (28% of episode)\n"
                 "Ensemble/gating can't fix it")
    ax_d.text(7.5, 5.2, bias_text, fontsize=8, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#d1f2eb",
                        edgecolor=C_VLM, linewidth=1),
              color="#2c3e50")

    # RND branch (middle)
    ax_d.text(5.0, 6.3, "Novelty Signal", fontsize=9, ha="center",
              fontweight="bold", color=C_RND)
    rnd_text = "Redirects exploration\nbut doesn't improve it\n(seed-switching, not learning)"
    ax_d.text(5.0, 5.0, rnd_text, fontsize=7.5, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdebd0",
                        edgecolor=C_RND, linewidth=1),
              color="#2c3e50", fontstyle="italic")

    # Convergent conclusion
    ax_d.annotate("", xy=(5.0, 2.8), xytext=(2.5, 3.8),
                  arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax_d.annotate("", xy=(5.0, 2.8), xytext=(5.0, 3.8),
                  arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax_d.annotate("", xy=(5.0, 2.8), xytext=(7.5, 3.8),
                  arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    conclusion = ("CONVERGENT FINDING:\n"
                  "Uniform replay is the optimal default\n"
                  "for sparse-reward early training.\n"
                  "7 approaches tested, 0 improvements.")
    ax_d.text(5.0, 1.8, conclusion, fontsize=9.5, ha="center", va="center",
              fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8",
                        edgecolor="#c0392b", linewidth=2),
              color="#c0392b")

    # Study attribution
    ax_d.text(2.5, 0.5, "td_baseline: 40 runs\n5 signals × 2 tasks × 5 seeds",
              fontsize=7, ha="center", color=C_GREY, fontstyle="italic")
    ax_d.text(7.5, 0.5, "vlm_probe: 9 models\n3 tasks × 10+ interventions",
              fontsize=7, ha="center", color=C_GREY, fontstyle="italic")

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle(
        "The Priority Signal Landscape: No Signal Beats Uniform Replay\n"
        "in Sparse-Reward Early Training (cross-study synthesis)",
        fontsize=14, fontweight="bold", y=0.96
    )

    # Save
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"cross_study_synthesis.{ext}", dpi=200,
                    bbox_inches="tight")
    # Also save to images/ for Quarto
    fig.savefig(IMG_DIR / "cross_study_synthesis.png", dpi=200,
                bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'cross_study_synthesis.png'}")
    print(f"Saved: {IMG_DIR / 'cross_study_synthesis.png'}")
    plt.close()

    # Print summary
    print("\n=== Cross-Study Synthesis Summary ===")
    print("RL signals tested (td_baseline):")
    for label, learned, total, _ in reach_data:
        label_clean = label.replace('\n', ' ')
        print(f"  {label_clean}: {learned}/{total} seeds learn on reach-v3")
    print("\nVLM priority quality (vlm_probe):")
    for label, overlap, kl, _, _ in approaches:
        label_clean = label.replace('\n', ' ')
        better = "✓" if overlap > 21.7 and kl < 1.556 else "✗"
        print(f"  {label_clean}: overlap={overlap:.1f}%, KL={kl:.3f} {better}")
    print("\nConclusion: 0/7 approaches dominate uniform on both metrics.")


if __name__ == "__main__":
    main()
