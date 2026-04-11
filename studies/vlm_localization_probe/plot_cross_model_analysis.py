"""Plot cross-model category comparison (Iteration 43)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

DATA_DIR = Path("results/failure_descriptions_iter39")
IMG_DIR = Path("../../images/vlm_probe")
IMG_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CATS = ["never_reached", "overshot", "oscillated", "wrong_direction", "stuck", "other"]


def load_successful(path):
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if "error" not in r and r.get("failure_category") != "parse_error"]


def main():
    datasets = {
        "reach-v3\nGPT-4o-mini\n(n=20)": load_successful(DATA_DIR / "reach-v3_gh_gpt-4o-mini_K4.json"),
        "push-v3\nPhi-4\n(n=10)": load_successful(DATA_DIR / "push-v3_gh_Phi-4-multimodal-instruct_K4.json"),
        "pick-place-v3\nPhi-4\n(n=9)": load_successful(DATA_DIR / "pick-place-v3_gh_Phi-4-multimodal-instruct_K4.json"),
    }

    # Get all categories (canonical + novel)
    all_cats = set()
    for results in datasets.values():
        for r in results:
            all_cats.add(r["failure_category"])
    novel_cats = sorted(all_cats - set(CANONICAL_CATS))
    ordered_cats = CANONICAL_CATS + novel_cats

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(ordered_cats)))
    color_map = {c: colors[i] for i, c in enumerate(ordered_cats)}

    for ax, (label, results) in zip(axes, datasets.items()):
        cats = [r["failure_category"] for r in results]
        counts = Counter(cats)
        n = len(results)

        bars = []
        bar_colors = []
        bar_labels = []
        for cat in ordered_cats:
            if counts.get(cat, 0) > 0:
                bars.append(counts[cat] / n)
                bar_colors.append(color_map[cat])
                # Mark novel categories with asterisk
                lbl = f"*{cat}" if cat in novel_cats else cat
                bar_labels.append(lbl)

        y_pos = np.arange(len(bars))
        ax.barh(y_pos, bars, color=bar_colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.set_xlabel("Proportion", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 0.55)
        ax.axvline(1/6, color="gray", linestyle="--", alpha=0.4, label="uniform")

        # Add count labels
        for i, (b, cnt) in enumerate(zip(bars, [counts.get(c, 0) for c in ordered_cats if counts.get(c, 0) > 0])):
            ax.text(b + 0.01, i, str(cnt), va="center", fontsize=8, color="gray")

    axes[0].set_ylabel("Failure Category", fontsize=10)
    fig.suptitle("Cross-Model Category Comparison\n* = novel (non-canonical) category from Phi-4",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    outpath = IMG_DIR / "cross_model_categories_iter43.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {outpath}")

    # Also make a severity comparison subplot
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    sev_order = ["mild", "moderate", "severe"]
    sev_colors = ["#4CAF50", "#FF9800", "#F44336"]

    for ax, (label, results) in zip(axes2, datasets.items()):
        sevs = Counter(r.get("severity", "unknown") for r in results)
        n = len(results)
        bars = [sevs.get(s, 0) / n for s in sev_order]
        y_pos = np.arange(len(sev_order))
        ax.barh(y_pos, bars, color=sev_colors, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sev_order, fontsize=9)
        ax.set_xlabel("Proportion", fontsize=10)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.0)

        for i, (b, cnt) in enumerate(zip(bars, [sevs.get(s, 0) for s in sev_order])):
            ax.text(b + 0.02, i, str(cnt), va="center", fontsize=8, color="gray")

    axes2[0].set_ylabel("Severity", fontsize=10)
    fig2.suptitle("Severity Calibration: GPT-4o-mini vs Phi-4\nGPT never says 'mild'; Phi-4 uses full scale",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()

    outpath2 = IMG_DIR / "severity_comparison_iter43.png"
    plt.savefig(outpath2, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {outpath2}")


if __name__ == "__main__":
    main()
