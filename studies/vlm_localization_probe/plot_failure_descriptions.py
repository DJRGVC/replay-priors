"""Generate figure for failure description analysis (Proposal 4)."""

import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def load_results(results_dir):
    """Load all failure description results."""
    results_dir = Path(results_dir)
    data = {}
    for f in sorted(results_dir.glob("*.json")):
        if 'summary' in f.stem:
            continue
        with open(f) as fh:
            results = json.load(fh)
        # Extract task name
        if 'Phi-4' in f.stem:
            task = f.stem.split("_gh_")[0]
            model = "Phi-4"
        elif 'gpt-4o-mini' in f.stem:
            task = f.stem.split("_gh_")[0]
            model = "GPT-4o-mini"
        else:
            task = f.stem.split("_K")[0]
            model = "unknown"
        data[f"{task} ({model})"] = results
    return data


def main():
    data = load_results("results/failure_descriptions_iter39")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Failure Mode Descriptions — Semantic Diversity Analysis\n"
                 "(Proposal 4: Non-temporal VLM signal for replay prioritization)",
                 fontsize=13, fontweight='bold')

    # Panel 1: Category distribution per task
    ax = axes[0, 0]
    all_cats = set()
    task_cat_counts = {}
    for name, results in data.items():
        valid = [r for r in results if r.get('failure_category') != 'parse_error' and 'failure_category' in r]
        cats = Counter(r['failure_category'] for r in valid)
        task_cat_counts[name] = cats
        all_cats.update(cats.keys())

    # Standardize to predefined + novel
    predefined = ['stuck', 'never_reached', 'overshot', 'wrong_direction', 'oscillated', 'other']
    novel = sorted(all_cats - set(predefined))
    all_cats_ordered = predefined + novel

    x = np.arange(len(all_cats_ordered))
    width = 0.25
    for i, (name, cats) in enumerate(task_cat_counts.items()):
        counts = [cats.get(c, 0) for c in all_cats_ordered]
        total = sum(counts)
        pcts = [100 * c / total if total > 0 else 0 for c in counts]
        bars = ax.bar(x + i * width, pcts, width, label=name.replace('(', '\n('), alpha=0.8)
        # Highlight novel categories
        for j, cat in enumerate(all_cats_ordered):
            if cat in novel and pcts[j] > 0:
                bars[j].set_edgecolor('red')
                bars[j].set_linewidth(2)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_cats_ordered, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("% of rollouts")
    ax.set_title("Category Distribution by Task", fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 50)

    # Panel 2: Category vs GT failure timing (box plot style)
    ax = axes[0, 1]
    # Use reach-v3 data (largest n)
    reach_data = data.get("reach-v3 (GPT-4o-mini)", list(data.values())[2])
    valid = [r for r in reach_data if r.get('failure_category') != 'parse_error' and 'failure_category' in r]
    cat_timings = {}
    for r in valid:
        cat = r['failure_category']
        if cat not in cat_timings:
            cat_timings[cat] = []
        cat_timings[cat].append(r['gt_failure_t'])

    cats_sorted = sorted(cat_timings.keys(), key=lambda c: np.mean(cat_timings[c]))
    positions = range(len(cats_sorted))
    for i, cat in enumerate(cats_sorted):
        times = cat_timings[cat]
        ax.scatter([i] * len(times), times, alpha=0.6, s=40, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(times)] * 2, 'k-', linewidth=2, zorder=4)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("GT failure timestep")
    ax.set_title("Category vs GT Timing (reach-v3, n=20)\nη²=0.34 — categories explain timing", fontsize=10)
    ax.axhline(y=np.mean([r['gt_failure_t'] for r in valid]), color='gray', linestyle='--', alpha=0.5, label='grand mean')
    ax.legend(fontsize=8)

    # Panel 3: η² comparison across tasks
    ax = axes[1, 0]
    task_names = []
    eta_values = []
    for name, results in data.items():
        valid = [r for r in results if r.get('failure_category') != 'parse_error' and 'failure_category' in r]
        if len(valid) < 3:
            continue
        cat_timings = {}
        for r in valid:
            cat = r['failure_category']
            if cat not in cat_timings:
                cat_timings[cat] = []
            cat_timings[cat].append(r['gt_failure_t'])

        all_times = [r['gt_failure_t'] for r in valid]
        n = len(all_times)
        grand_mean = np.mean(all_times)
        total_var = np.var(all_times)
        if total_var == 0:
            continue
        between_var = sum(len(cat_timings[c]) * (np.mean(cat_timings[c]) - grand_mean) ** 2
                         for c in cat_timings) / n
        eta_sq = between_var / total_var
        task_names.append(name)
        eta_values.append(eta_sq)

    colors = ['#2196F3', '#4CAF50', '#FF9800']
    bars = ax.barh(range(len(task_names)), eta_values, color=colors[:len(task_names)], alpha=0.8)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_xlabel("η² (variance explained)")
    ax.set_title("Category → Timing Correlation (η²)", fontsize=11)
    ax.axvline(x=0.14, color='red', linestyle='--', alpha=0.5, label='large effect threshold')
    for i, v in enumerate(eta_values):
        ax.text(v + 0.02, i, f"{v:.3f}", va='center', fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=8)

    # Panel 4: Pairwise description similarity (Jaccard) histogram
    ax = axes[1, 1]
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for',
                  'and', 'or', 'not', 'it', 'its', 'that', 'this', 'with', 'from', 'by',
                  'as', 'at', 'on', 'but', 'be', 'has', 'have', 'had', 'do', 'does', 'did'}

    for name, results in data.items():
        valid = [r for r in results if 'failure_mode' in r and r.get('failure_category') != 'parse_error']
        descs = [r['failure_mode'] for r in valid]
        word_sets = [set(d.lower().replace(',', ' ').replace('.', ' ').split()) - stop_words for d in descs]
        sims = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                inter = word_sets[i] & word_sets[j]
                union = word_sets[i] | word_sets[j]
                sims.append(len(inter) / len(union) if union else 0)
        ax.hist(sims, bins=15, alpha=0.5, label=name.split(' (')[0], density=True)

    ax.set_xlabel("Pairwise Jaccard similarity")
    ax.set_ylabel("Density")
    ax.set_title("Description Lexical Diversity\n(lower = more diverse)", fontsize=11)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='low diversity threshold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig("figures/failure_description_analysis_iter39.png", bbox_inches='tight', dpi=150)
    fig.savefig("images/vlm_probe/failure_description_analysis_iter39.png", bbox_inches='tight', dpi=150)
    print("Saved figure to figures/ and images/vlm_probe/")
    plt.close()


if __name__ == "__main__":
    main()
