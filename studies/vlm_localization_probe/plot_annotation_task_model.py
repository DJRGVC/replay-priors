"""Plot annotation × task × model interaction — bias-matching mechanism.

Creates a grouped bar chart showing MAE for annotated vs unannotated
across 3 tasks × 2 models, with GT distribution means overlaid.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Data from FINDINGS.md §10 and iterations 22, 25, 26, 31, 32
# Format: (model, task, gt_mean, annotated_mae, unannotated_mae)
data = [
    # GPT-4o
    ("GPT-4o", "reach-v3",      57.8, 52.7, 75.8),
    ("GPT-4o", "push-v3",       36.6, 43.0, 36.3),
    ("GPT-4o", "pick-place-v3", 80.3, 48.3, None),  # unannotated not collected (quota)
    # GPT-4o-mini
    ("GPT-4o-mini", "reach-v3",      57.8, 68.0, 61.2),
    ("GPT-4o-mini", "push-v3",       36.6, None, 44.4),  # annotated not collected
    ("GPT-4o-mini", "pick-place-v3", 80.3, 61.3, 52.8),
]

tasks = ["reach-v3", "push-v3", "pick-place-v3"]
models = ["GPT-4o", "GPT-4o-mini"]
gt_means = {"reach-v3": 57.8, "push-v3": 36.6, "pick-place-v3": 80.3}
gt_labels = {"reach-v3": "mid", "push-v3": "early", "pick-place-v3": "late"}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

colors_ann = {"GPT-4o": "#2563eb", "GPT-4o-mini": "#7c3aed"}
colors_noann = {"GPT-4o": "#93c5fd", "GPT-4o-mini": "#c4b5fd"}

bar_width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]  # 4 bars per task: GPT-4o ann/noann, mini ann/noann

for ti, task in enumerate(tasks):
    ax = axes[ti]

    bars = []
    labels = []
    positions = []

    for mi, model in enumerate(models):
        row = [d for d in data if d[0] == model and d[1] == task][0]
        ann_mae = row[3]
        noann_mae = row[4]

        # Annotated bar
        x_ann = offsets[mi * 2] * bar_width
        if ann_mae is not None:
            b = ax.bar(x_ann, ann_mae, bar_width * 0.85, color=colors_ann[model],
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(x_ann, ann_mae + 1.5, f"{ann_mae:.0f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        else:
            ax.bar(x_ann, 0, bar_width * 0.85, color="none", edgecolor="#ccc",
                   linewidth=1, linestyle="--", zorder=3)
            ax.text(x_ann, 5, "n/a", ha="center", va="bottom", fontsize=7, color="#999")

        # Unannotated bar
        x_noann = offsets[mi * 2 + 1] * bar_width
        if noann_mae is not None:
            b = ax.bar(x_noann, noann_mae, bar_width * 0.85, color=colors_noann[model],
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(x_noann, noann_mae + 1.5, f"{noann_mae:.0f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        else:
            ax.bar(x_noann, 0, bar_width * 0.85, color="none", edgecolor="#ccc",
                   linewidth=1, linestyle="--", zorder=3)
            ax.text(x_noann, 5, "n/a", ha="center", va="bottom", fontsize=7, color="#999")

    # GT distribution mean line
    gt = gt_means[task]
    ax.axhline(gt, color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.7, zorder=2)
    ax.text(0.97, gt + 1.5, f"GT mean={gt:.0f}", ha="right", va="bottom",
            fontsize=7, color="#ef4444", transform=ax.get_yaxis_transform())

    # Annotation arrows showing direction of effect
    for mi, model in enumerate(models):
        row = [d for d in data if d[0] == model and d[1] == task][0]
        ann_mae, noann_mae = row[3], row[4]
        if ann_mae is not None and noann_mae is not None:
            diff = ann_mae - noann_mae
            sign = "+" if diff > 0 else ""
            pct = 100 * diff / noann_mae
            color = "#16a34a" if diff < 0 else "#dc2626"
            x_mid = (offsets[mi * 2] + offsets[mi * 2 + 1]) * bar_width / 2
            y_top = max(ann_mae, noann_mae) + 7
            ax.text(x_mid, y_top, f"{sign}{pct:.0f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_title(f"{task}\n(GT {gt_labels[task]}, μ={gt_means[task]:.0f})",
                 fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(0, 95)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("MAE (timesteps, lower is better)", fontsize=10)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_ann["GPT-4o"], label="GPT-4o annotated"),
    Patch(facecolor=colors_noann["GPT-4o"], label="GPT-4o unannotated"),
    Patch(facecolor=colors_ann["GPT-4o-mini"], label="GPT-4o-mini annotated"),
    Patch(facecolor=colors_noann["GPT-4o-mini"], label="GPT-4o-mini unannotated"),
    plt.Line2D([0], [0], color="#ef4444", linewidth=1.5, linestyle="--", label="GT failure mean"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle("Annotation × Task × Model: Bias-Matching Mechanism",
             fontsize=12, fontweight="bold", y=1.02)

plt.tight_layout()
fig.savefig(figures_dir / "annotation_task_model_interaction.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {figures_dir / 'annotation_task_model_interaction.png'}")

# Also save to images/vlm_probe for Quarto
quarto_img_dir = Path(__file__).parent.parent.parent / "images" / "vlm_probe"
if quarto_img_dir.exists():
    fig.savefig(quarto_img_dir / "annotation_task_model_interaction.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved Quarto copy to {quarto_img_dir}")

print("Done.")
