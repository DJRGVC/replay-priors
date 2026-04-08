"""Analyze ground-truth failure label quality across tasks + cross-model bias patterns.

Produces:
  - figures/gt_quality_analysis.png  — 2×2 panel: GT distributions, ambiguity,
    distance profiles, and cross-model prediction bias
  - Prints summary statistics to stdout

This is an offline analysis — no API calls needed.
"""

import json
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_rollout_data(data_dir="data"):
    """Load all rollout metadata and proprio for analysis."""
    tasks = {}
    for task in ["reach-v3", "push-v3", "pick-place-v3"]:
        task_dir = os.path.join(data_dir, task)
        rollout_dirs = sorted(glob.glob(os.path.join(task_dir, "rollout_*")))
        rollouts = []
        for rd in rollout_dirs:
            meta = json.load(open(os.path.join(rd, "meta.json")))
            proprio = np.load(os.path.join(rd, "proprio.npy"))

            # Compute key distance metrics
            hand = proprio[:, 0:3]
            if "reach" in task:
                goal = proprio[0, 36:39]
                dists = np.linalg.norm(hand - goal, axis=1)
                dist_label = "hand-goal"
            else:
                obj = proprio[:, 4:7]
                dists = np.linalg.norm(hand - obj, axis=1)
                dist_label = "hand-object"

            rollouts.append({
                "meta": meta,
                "proprio": proprio,
                "dists": dists,
                "dist_label": dist_label,
                "gt": meta["failure_timestep"],
                "ambiguous": meta["ambiguous"],
                "failure_type": meta["failure_type"],
            })
        tasks[task] = rollouts
    return tasks


def load_vlm_results():
    """Load all VLM prediction results from JSON files."""
    all_results = []
    for path in glob.glob("results/**/*.json", recursive=True):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)

    # Deduplicate by (model, K, rollout)
    seen = set()
    unique = []
    for e in all_results:
        key = (e["model"], e["K"], e["rollout"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def analyze_gt_reliability(tasks):
    """Quantify GT label reliability per task."""
    print("=" * 70)
    print("GROUND-TRUTH FAILURE LABEL QUALITY ANALYSIS")
    print("=" * 70)

    for task, rollouts in tasks.items():
        gts = np.array([r["gt"] for r in rollouts])
        n_amb = sum(1 for r in rollouts if r["ambiguous"])
        types = {}
        for r in rollouts:
            ft = r["failure_type"]
            types[ft] = types.get(ft, 0) + 1

        # Compute min-distance consistency (does GT match argmin of distance?)
        gt_matches_argmin = 0
        argmin_errors = []
        for r in rollouts:
            argmin_t = int(np.argmin(r["dists"]))
            if r["gt"] == argmin_t:
                gt_matches_argmin += 1
            argmin_errors.append(abs(r["gt"] - argmin_t))

        # Distance variance at GT timestep vs random timestep
        gt_dists = [r["dists"][r["gt"]] for r in rollouts]
        min_dists = [r["dists"].min() for r in rollouts]

        print(f"\n--- {task} ({len(rollouts)} rollouts) ---")
        print(f"  Failure types: {types}")
        print(f"  Ambiguous: {n_amb}/{len(rollouts)} ({100*n_amb/len(rollouts):.0f}%)")
        print(f"  GT stats: mean={gts.mean():.1f} ± {gts.std():.1f}")
        print(f"  GT matches argmin(dist): {gt_matches_argmin}/{len(rollouts)}")
        print(f"  GT-argmin error: mean={np.mean(argmin_errors):.1f}, max={np.max(argmin_errors)}")
        print(f"  Min distance: {np.mean(min_dists):.4f} ± {np.std(min_dists):.4f}")
        print(f"  Dist at GT: {np.mean(gt_dists):.4f} ± {np.std(gt_dists):.4f}")

        # Visual saliency assessment
        # How different does the min-distance frame look from average?
        dist_ranges = [r["dists"].max() - r["dists"].min() for r in rollouts]
        print(f"  Distance range (max-min): {np.mean(dist_ranges):.4f} — "
              f"{'LOW (subtle differences)' if np.mean(dist_ranges) < 0.1 else 'MODERATE' if np.mean(dist_ranges) < 0.3 else 'HIGH (clear differences)'}")


def make_figure(tasks, vlm_results):
    """Create the 2×2 analysis figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    task_colors = {"reach-v3": "#2196F3", "push-v3": "#FF9800", "pick-place-v3": "#4CAF50"}

    # Panel A: GT failure timestep distributions
    ax1 = fig.add_subplot(gs[0, 0])
    for task, rollouts in tasks.items():
        gts = [r["gt"] for r in rollouts]
        ax1.hist(gts, bins=15, alpha=0.5, label=task, color=task_colors[task], edgecolor="white")
    ax1.set_xlabel("GT failure timestep")
    ax1.set_ylabel("Count")
    ax1.set_title("A) GT Failure Timestep Distributions")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 150)

    # Panel B: Distance profiles (sample 3 rollouts per task)
    ax2 = fig.add_subplot(gs[0, 1])
    for task, rollouts in tasks.items():
        for i, r in enumerate(rollouts[:3]):
            alpha = 0.7 if i == 0 else 0.3
            label = f"{task} ({r['dist_label']})" if i == 0 else None
            ax2.plot(r["dists"], color=task_colors[task], alpha=alpha,
                    linewidth=0.8, label=label)
            # Mark GT
            ax2.axvline(r["gt"], color=task_colors[task], alpha=0.2, linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Distance")
    ax2.set_title("B) Distance Profiles (3 rollouts/task)")
    ax2.legend(fontsize=7, loc="upper right")

    # Panel C: Prediction bias heatmap (from VLM results)
    ax3 = fig.add_subplot(gs[1, 0])
    if vlm_results:
        # Group by model
        models = sorted(set(e["model"] for e in vlm_results))
        model_short = {"claude-sonnet-4-6": "Sonnet 4.6",
                       "gemini-2.5-flash-lite": "Flash-Lite"}

        for i, model in enumerate(models):
            entries = [e for e in vlm_results if e["model"] == model and e["pred_failure_t"] is not None]
            if not entries:
                continue
            preds = [e["pred_failure_t"] for e in entries]
            gts = [e["gt_failure_t"] for e in entries]
            errors = [e["abs_error"] for e in entries]

            name = model_short.get(model, model)
            ax3.scatter(gts, preds, alpha=0.5, s=20, label=f"{name} (MAE={np.mean(errors):.0f})")

        ax3.plot([0, 150], [0, 150], "k--", alpha=0.3, linewidth=1, label="Perfect")
        ax3.set_xlabel("GT failure timestep")
        ax3.set_ylabel("Predicted failure timestep")
        ax3.set_title("C) VLM Prediction vs GT (reach-v3)")
        ax3.legend(fontsize=7)
        ax3.set_xlim(0, 155)
        ax3.set_ylim(0, 155)
    else:
        ax3.text(0.5, 0.5, "No VLM results loaded", ha="center", va="center", transform=ax3.transAxes)

    # Panel D: Task suitability summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_data = []
    for task, rollouts in tasks.items():
        n_amb = sum(1 for r in rollouts if r["ambiguous"])
        gts = [r["gt"] for r in rollouts]
        gt_matches = sum(1 for r in rollouts if r["gt"] == int(np.argmin(r["dists"])))
        dist_ranges = [r["dists"].max() - r["dists"].min() for r in rollouts]
        summary_data.append([
            task,
            f"{100*n_amb/len(rollouts):.0f}%",
            f"{gt_matches}/{len(rollouts)}",
            f"{np.mean(dist_ranges):.3f}",
            "✓" if n_amb == 0 else "△" if n_amb < len(rollouts) else "✗"
        ])

    table = ax4.table(
        cellText=summary_data,
        colLabels=["Task", "Ambiguous", "GT=argmin", "Dist Range", "VLM\nSuitable"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color cells by suitability
    for i, row in enumerate(summary_data):
        color = "#C8E6C9" if row[4] == "✓" else "#FFF9C4" if row[4] == "△" else "#FFCDD2"
        for j in range(5):
            table[i + 1, j].set_facecolor(color)

    ax4.set_title("D) Task Suitability for VLM Probing", pad=20)

    fig.suptitle("VLM Failure Localization: GT Quality & Cross-Model Analysis",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/gt_quality_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: figures/gt_quality_analysis.png")


def print_recommendations():
    """Print actionable recommendations."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. REACH-V3 is the only task with reliable GT labels (0% ambiguous).
   → Continue using reach-v3 as the primary evaluation task.

2. PUSH-V3 and PICK-PLACE-V3 have 100% ambiguous GT labels.
   The random policy never meaningfully interacts with objects, so
   "failure timestep" = "closest approach" is essentially noise.
   → These tasks CANNOT be used for VLM evaluation with random policies.
   → Need trained policy rollouts that actually contact/manipulate objects.

3. For multi-task evaluation, options:
   a) Train a partial policy (e.g., 10k steps) that contacts objects sometimes
   b) Use reach-v3 only but vary difficulty (goal distance, initial configs)
   c) Redefine GT for push/pick-place: "when did arm stop progressing
      toward object?" (motion-based rather than distance-based)

4. CROSS-MODEL BIAS PATTERNS (from reach-v3 data):
   - Claude Sonnet: center-bias (clusters at t≈75-85)
   - Gemini Flash-Lite: late-bias (clusters at t≈127-149)
   - Gemini 3 Flash Preview: start-bias (some predict t=0)
   → Each model has a distinct positional prior; debiasing strategies
     should be model-specific.
""")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tasks = load_rollout_data()
    vlm_results = load_vlm_results()
    analyze_gt_reliability(tasks)
    make_figure(tasks, vlm_results)
    print_recommendations()
