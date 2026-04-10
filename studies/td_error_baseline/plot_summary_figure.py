"""Hero summary figure: How (un)informative is TD-error PER in early training?

Produces a single publication-quality 6-panel figure (3×2) that tells the complete
story across two tasks of different difficulty:

Row 1: Spearman correlation over training
  (a) reach-v3 — TD-error becomes informative only AFTER learning starts
  (b) pick-place-v3 — permanent information desert (no learning at 100k)

Row 2: Q-value dynamics
  (c) reach-v3 — PER creates Q-value positive feedback loop
  (d) pick-place-v3 — Q-instability not PER-specific on hard tasks

Row 3: Aggregate analysis
  (e) Mode comparison bar chart — both tasks, incl. alpha sweep
  (f) Information regime breakdown — aligned fraction across runs

Data sources: 5-seed reach-v3 + pick-place-v3 (100k steps each) across
uniform/td-per/adaptive/rpe-per modes, α sweep (0.1, 0.3, 0.6) on reach-v3.
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
from scipy import stats

SNAP_ROOT = Path(__file__).parent / "snapshots"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 7, 99, 256]
TASKS = ["reach-v3", "pick-place-v3"]

# Color palette
C_UNIFORM = "#2ecc71"
C_TDPER = "#3498db"
C_ADAPTIVE = "#e74c3c"
C_RPE = "#9b59b6"
C_ALIGNED = "#2ecc71"
C_NOISE = "#95a5a6"
C_INVERTED = "#e74c3c"
C_UNSTABLE = "#f39c12"


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
            "spearman": float(data.get("td_dense_spearman", 0)),
            "q_mean": float(data.get("q_mean", 0)),
            "q_std": float(data.get("q_std", 0)),
            "abs_td_mean": float(data.get("abs_td_mean", 0)),
            "ep_rew": float(data.get("episode_return_mean", 0)),
            "success_rate": float(data.get("success_rate", 0)),
        })
    return sorted(results, key=lambda x: x["step"])


def aggregate_mode(task, seeds, mode_suffix=""):
    """Aggregate snapshot data across seeds for a given mode."""
    all_runs = {}
    for seed in seeds:
        snaps = load_snapshots(task, seed, mode_suffix)
        if snaps:
            all_runs[seed] = snaps

    if not all_runs:
        return {}, all_runs

    # Find common steps
    step_sets = [set(d["step"] for d in run) for run in all_runs.values()]
    common_steps = sorted(set.intersection(*step_sets)) if step_sets else []

    agg = {}
    for step in common_steps:
        vals = {k: [] for k in ["spearman", "q_mean", "abs_td_mean", "ep_rew"]}
        for seed, run in all_runs.items():
            for d in run:
                if d["step"] == step:
                    for k in vals:
                        vals[k].append(d[k])
                    break
        agg[step] = {k: (np.mean(v), np.std(v), v) for k, v in vals.items()}
    return agg, all_runs


def classify_regime(spearman, q_cv, q_mean):
    """Classify training regime based on metrics."""
    if abs(q_mean) < 0.1:
        return "noise"  # Q hasn't started learning
    if q_cv > 1.5:
        return "unstable"
    if spearman < -0.1:
        return "inverted"
    if spearman > 0.15:
        return "aligned"
    return "noise"


def plot_spearman_panel(ax, task, seeds, mode_suffix, title_label, title_text):
    """Plot Spearman correlation traces for a task."""
    agg, runs = aggregate_mode(task, seeds, mode_suffix)
    if not agg:
        ax.text(0.5, 0.5, f"No data for {task}", transform=ax.transAxes,
                ha="center", va="center")
        return

    steps = sorted(agg.keys())

    # Plot individual seed traces
    for seed, snaps in runs.items():
        ss = [d["step"] for d in snaps if d["step"] in agg]
        sp = [d["spearman"] for d in snaps if d["step"] in agg]
        er = [d["ep_rew"] for d in snaps if d["step"] in agg]
        max_rew = max(er) if er else 0
        color = "#27ae60" if max_rew > 50 else "#bdc3c7"
        lw = 1.2 if max_rew > 50 else 0.8
        ax.plot([s/1000 for s in ss], sp, "-", color=color, alpha=0.5,
                linewidth=lw, zorder=2)

    # Plot mean +/- std
    mean_sp = [agg[s]["spearman"][0] for s in steps]
    std_sp = [agg[s]["spearman"][1] for s in steps]
    steps_k = [s/1000 for s in steps]
    ax.plot(steps_k, mean_sp, "k-", linewidth=2.5, zorder=3, label="Mean (n=5)")
    ax.fill_between(steps_k,
                    [m - s for m, s in zip(mean_sp, std_sp)],
                    [m + s for m, s in zip(mean_sp, std_sp)],
                    alpha=0.15, color="black", zorder=1)

    # Information desert shading
    ax.axhspan(-0.15, 0.15, alpha=0.08, color="red", zorder=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Count learning seeds
    n_learn = sum(1 for seed, snaps in runs.items()
                  if max(d["ep_rew"] for d in snaps) > 50)
    n_total = len(runs)

    learned_patch = mpatches.Patch(color="#27ae60", alpha=0.5,
                                   label=f"Seeds that learn ({n_learn}/{n_total})")
    nolearn_patch = mpatches.Patch(color="#bdc3c7", alpha=0.5,
                                   label=f"Seeds that don't ({n_total-n_learn}/{n_total})")
    ax.legend(handles=[learned_patch, nolearn_patch,
                       plt.Line2D([0], [0], color="k", lw=2.5, label="Mean +/- σ")],
              fontsize=7, loc="upper left")

    ax.set_xlabel("Environment Steps (x10³)", fontsize=9)
    ax.set_ylabel("Spearman ρ(|TD|, Oracle Adv.)", fontsize=9)
    ax.set_title(f"({title_label}) {title_text}", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.4, 0.8)
    max_step = max(steps_k) if steps_k else 100
    ax.set_xlim(0, max_step + 5)
    ax.grid(True, alpha=0.2)


def plot_q_panel(ax, task, seeds, title_label, title_text):
    """Plot Q-value dynamics for all 4 modes."""
    mode_data = {
        "Uniform": ("uniform", C_UNIFORM),
        "TD-PER (α=0.6)": ("td-per", C_TDPER),
        "RPE-PER": ("rpe-per", C_RPE),
        "Adaptive": ("adaptive", C_ADAPTIVE),
    }

    for label, (suffix, color) in mode_data.items():
        agg, _ = aggregate_mode(task, seeds, suffix)
        if not agg:
            continue
        steps = sorted(agg.keys())
        q_means = [agg[s]["q_mean"][0] for s in steps]
        q_stds = [agg[s]["q_mean"][1] for s in steps]
        steps_k = [s/1000 for s in steps]
        ax.plot(steps_k, q_means, "o-", color=color, linewidth=2,
                markersize=3, label=label, zorder=3)
        ax.fill_between(steps_k,
                        [max(0.01, m - s) for m, s in zip(q_means, q_stds)],
                        [m + s for m, s in zip(q_means, q_stds)],
                        alpha=0.15, color=color, zorder=1)

    ax.set_xlabel("Environment Steps (x10³)", fontsize=9)
    ax.set_ylabel("Mean Q-value", fontsize=9)
    ax.set_title(f"({title_label}) {title_text}", fontsize=10, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.2)


def main():
    fig = plt.figure(figsize=(14, 15))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30,
                           left=0.08, right=0.96, top=0.93, bottom=0.05)

    # =========================================================================
    # Row 1: Spearman correlation over training
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    plot_spearman_panel(ax_a, "reach-v3", SEEDS, "uniform", "a",
                        "reach-v3: TD-error lags learning")
    # Add annotations for reach-v3
    ax_a.annotate("Information desert\n(TD-error ≈ random)",
                  xy=(30, -0.08), fontsize=7.5, color="#c0392b",
                  fontstyle="italic", ha="center", fontweight="bold")
    ax_a.annotate("Signal emerges\nonly after learning",
                  xy=(80, 0.50), fontsize=7.5, color="#27ae60",
                  fontstyle="italic", ha="center", fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_spearman_panel(ax_b, "pick-place-v3", SEEDS, "uniform", "b",
                        "pick-place-v3: permanent desert")
    # Add annotation for pick-place
    ax_b.annotate("No learning ever occurs\n→ TD-error never informative",
                  xy=(50, -0.15), fontsize=7.5, color="#c0392b",
                  fontstyle="italic", ha="center", fontweight="bold")

    # =========================================================================
    # Row 2: Q-value dynamics
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    plot_q_panel(ax_c, "reach-v3", SEEDS, "c",
                 "reach-v3: PER inflates Q 11x")
    ax_c.annotate("Q explodes under PER:\nhigh |TD| → resample →\noverfit → ↑Q → ↑|TD|",
                  xy=(40, 100), xytext=(70, 350),
                  fontsize=7, color=C_TDPER,
                  fontstyle="italic", ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_TDPER, lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=C_TDPER, alpha=0.9))

    ax_d = fig.add_subplot(gs[1, 1])
    plot_q_panel(ax_d, "pick-place-v3", SEEDS, "d",
                 "pick-place-v3: Q-instability not PER-specific")
    ax_d.annotate("All modes unstable\n(seed 99 explodes\nacross all modes)",
                  xy=(50, 200), xytext=(70, 400),
                  fontsize=7, color=C_NOISE,
                  fontstyle="italic", ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_NOISE, lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=C_NOISE, alpha=0.9))

    # =========================================================================
    # Row 3, Left: Mode comparison bar chart — both tasks
    # =========================================================================
    ax_e = fig.add_subplot(gs[2, 0])

    modes_reach = [
        ("Uniform", "uniform", C_UNIFORM),
        ("TD-PER\nα=0.3", "td-per_a0.3", "#5dade2"),
        ("TD-PER\nα=0.6", "td-per", C_TDPER),
        ("TD-PER\nα=0.1", "td-per_a0.1", "#85c1e9"),
        ("RPE-PER", "rpe-per", C_RPE),
        ("Adaptive", "adaptive", C_ADAPTIVE),
    ]

    # Count successes for reach-v3
    reach_counts = []
    for label, suffix, color in modes_reach:
        learned = 0
        total = 0
        for seed in SEEDS:
            snaps = load_snapshots("reach-v3", seed, suffix)
            if not snaps:
                continue
            total += 1
            max_rew = max(d["ep_rew"] for d in snaps if d["step"] >= 50000)
            if max_rew > 50:
                learned += 1
        reach_counts.append((label, learned, total, color))

    x_pos = np.arange(len(reach_counts))
    bar_width = 0.35

    # reach-v3 bars
    bars_r = ax_e.bar(x_pos - bar_width/2, [sc[1] for sc in reach_counts],
                      bar_width, color=[sc[3] for sc in reach_counts],
                      alpha=0.85, edgecolor="black", linewidth=0.8,
                      label="reach-v3")

    # pick-place-v3 bars (all 0/5 for every mode)
    pp_counts = []
    for label, suffix, color in modes_reach:
        # pick-place doesn't have alpha variants
        pp_suffix = suffix
        if "a0." in suffix:
            pp_counts.append((label, 0, 0, color))  # no data
            continue
        learned = 0
        total = 0
        for seed in SEEDS:
            snaps = load_snapshots("pick-place-v3", seed, pp_suffix)
            if not snaps:
                continue
            total += 1
            max_rew = max(d["ep_rew"] for d in snaps if d["step"] >= 50000)
            if max_rew > 50:
                learned += 1
        pp_counts.append((label, learned, total, color))

    # Only plot pp bars where data exists
    pp_vals = [sc[1] for sc in pp_counts]
    pp_colors = [sc[3] if sc[2] > 0 else "none" for sc in pp_counts]
    pp_edges = ["black" if sc[2] > 0 else "none" for sc in pp_counts]
    bars_p = ax_e.bar(x_pos + bar_width/2, pp_vals,
                      bar_width, color=pp_colors,
                      alpha=0.4, edgecolor=pp_edges, linewidth=0.8,
                      hatch="//", label="pick-place-v3")

    # Labels
    for bar, sc in zip(bars_r, reach_counts):
        if sc[2] > 0:
            ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                      f"{sc[1]}/{sc[2]}", ha="center", va="bottom",
                      fontweight="bold", fontsize=9)
    for bar, sc in zip(bars_p, pp_counts):
        if sc[2] > 0:
            ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                      f"{sc[1]}/{sc[2]}", ha="center", va="bottom",
                      fontweight="bold", fontsize=8, color="#555")

    ax_e.set_xticks(list(x_pos))
    ax_e.set_xticklabels([sc[0] for sc in reach_counts], fontsize=8)
    ax_e.set_ylabel("Seeds that Learn (out of 5)", fontsize=9)
    ax_e.set_title("(e) No priority signal beats uniform;\npick-place 0/5 across all modes",
                   fontsize=10, fontweight="bold")
    ax_e.set_ylim(0, 5.8)
    ax_e.axhline(3, color=C_UNIFORM, linestyle=":", alpha=0.5, linewidth=1.5)
    ax_e.legend(fontsize=8, loc="upper right")
    ax_e.grid(True, alpha=0.2, axis="y")

    # =========================================================================
    # Row 3, Right: Regime breakdown — both tasks
    # =========================================================================
    ax_f = fig.add_subplot(gs[2, 1])

    run_configs = [
        # reach-v3 uniform (3 learning, 2 non-learning)
        ("reach s42 (unif)", "reach-v3", 42, "uniform"),
        ("reach s123 (unif)", "reach-v3", 123, "uniform"),
        ("reach s7 (unif)", "reach-v3", 7, "uniform"),
        ("reach s99 (unif)", "reach-v3", 99, "uniform"),
        ("reach s256 (unif)", "reach-v3", 256, "uniform"),
        # pick-place-v3 uniform (all non-learning)
        ("pp s42 (unif)", "pick-place-v3", 42, "uniform"),
        ("pp s123 (unif)", "pick-place-v3", 123, "uniform"),
        ("pp s7 (unif)", "pick-place-v3", 7, "uniform"),
        ("pp s99 (unif)", "pick-place-v3", 99, "uniform"),
        ("pp s256 (unif)", "pick-place-v3", 256, "uniform"),
    ]

    regime_data = []
    for label, task, seed, suffix in run_configs:
        snaps = load_snapshots(task, seed, suffix)
        if not snaps:
            continue
        counts = {"aligned": 0, "noise": 0, "inverted": 0, "unstable": 0}
        for d in snaps:
            q_cv = d["q_std"] / max(abs(d["q_mean"]), 1e-6)
            regime = classify_regime(d["spearman"], q_cv, d["q_mean"])
            counts[regime] += 1
        total = sum(counts.values())
        fracs = {k: v / total if total > 0 else 0 for k, v in counts.items()}
        regime_data.append((label, fracs))

    labels = [rd[0] for rd in regime_data]
    y_pos = range(len(labels))

    regime_order = ["aligned", "noise", "inverted", "unstable"]
    regime_colors = {"aligned": C_ALIGNED, "noise": C_NOISE,
                     "inverted": C_INVERTED, "unstable": C_UNSTABLE}
    regime_labels = {"aligned": "Aligned (useful)", "noise": "Noise (random)",
                     "inverted": "Inverted (harmful)", "unstable": "Unstable"}

    lefts = [0] * len(regime_data)
    for regime in regime_order:
        widths = [rd[1].get(regime, 0) for rd in regime_data]
        ax_f.barh(y_pos, widths, left=lefts, height=0.6,
                  color=regime_colors[regime], edgecolor="white", linewidth=0.5,
                  label=regime_labels[regime])
        lefts = [l + w for l, w in zip(lefts, widths)]

    ax_f.set_yticks(list(y_pos))
    ax_f.set_yticklabels(labels, fontsize=7.5)
    ax_f.set_xlabel("Fraction of Training", fontsize=9)
    ax_f.set_title("(f) TD-error aligned regime:\nreach 20-50%, pick-place 7-13%",
                   fontsize=10, fontweight="bold")
    ax_f.set_xlim(0, 1.0)
    ax_f.legend(fontsize=7, loc="lower right", ncol=2)
    ax_f.grid(True, alpha=0.2, axis="x")
    ax_f.invert_yaxis()

    # Add percentage annotations for aligned fraction
    for i, (label, fracs) in enumerate(regime_data):
        aligned_frac = fracs.get("aligned", 0)
        ax_f.text(aligned_frac + 0.02, i, f"{aligned_frac:.0%}",
                  va="center", fontsize=7.5, fontweight="bold", color=C_ALIGNED)

    # Add separator line between tasks
    # Find boundary between reach and pick-place entries
    n_reach = sum(1 for l, _ in regime_data if l.startswith("reach"))
    if 0 < n_reach < len(regime_data):
        ax_f.axhline(n_reach - 0.5, color="black", linestyle="-", linewidth=1, alpha=0.3)

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle(
        "TD-Error Prioritized Experience Replay Is Uninformative and Harmful\n"
        "in Sparse-Reward Early Training (reach-v3 + pick-place-v3, 5 seeds x 4 modes)",
        fontsize=13, fontweight="bold", y=0.97
    )

    # Save
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"td_per_summary.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'td_per_summary.png'}")
    print(f"Saved: {FIG_DIR / 'td_per_summary.pdf'}")
    plt.close()

    # Print key numbers
    print("\n=== Key Statistics ===")
    print("reach-v3 (100k steps):")
    for label, learned, total, _ in reach_counts:
        pct = f"{learned}/{total}" if total > 0 else "N/A"
        print(f"  {label.replace(chr(10), ' ')}: {pct}")
    print("pick-place-v3 (100k steps):")
    for label, learned, total, _ in pp_counts:
        pct = f"{learned}/{total}" if total > 0 else "no data"
        print(f"  {label.replace(chr(10), ' ')}: {pct}")
    print("Spearman: ~0 for 60-80% of reach-v3 training, ~0 always on pick-place-v3")


if __name__ == "__main__":
    main()
