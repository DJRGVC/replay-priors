"""Analyze failure mode descriptions for semantic diversity and clustering potential.

Takes the JSON output from failure_description_probe.py and answers:
1. Category distribution — how diverse are the VLM-assigned categories?
2. Category vs GT timing — do categories correlate with failure timing?
3. Description diversity — how many unique failure modes are described?
4. Visual cue overlap — how much do VLM observations differ across rollouts?
5. Clustering potential — could these descriptions drive diversity-weighted replay?

Usage:
    python analyze_failure_descriptions.py --results-dir results/failure_descriptions_iter39
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def word_set(text):
    """Extract set of meaningful words from text."""
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for',
                  'and', 'or', 'not', 'it', 'its', 'that', 'this', 'with', 'from', 'by',
                  'as', 'at', 'on', 'but', 'be', 'has', 'have', 'had', 'do', 'does', 'did'}
    words = set(text.lower().replace(',', ' ').replace('.', ' ').split())
    return words - stop_words


def analyze_task(results, task_name):
    """Analyze descriptions for a single task."""
    valid = [r for r in results if 'failure_mode' in r and r.get('failure_category') != 'parse_error']
    n = len(valid)

    print(f"\n{'='*70}")
    print(f"TASK: {task_name} (n={n})")
    print(f"{'='*70}")

    # 1. Category distribution
    categories = [r['failure_category'] for r in valid]
    cat_counts = Counter(categories)
    print(f"\n--- Category Distribution ---")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:20s}: {count:2d} ({100*count/n:.0f}%)")
    n_cats = len(cat_counts)
    entropy = -sum((c/n) * np.log2(c/n) for c in cat_counts.values())
    max_entropy = np.log2(6)  # 6 possible categories
    print(f"  Unique categories: {n_cats}/6")
    print(f"  Shannon entropy: {entropy:.2f} / {max_entropy:.2f} (normalized: {entropy/max_entropy:.2f})")

    # 2. Category vs GT timing
    print(f"\n--- Category vs GT Failure Timing ---")
    cat_timings = {}
    for r in valid:
        cat = r['failure_category']
        if cat not in cat_timings:
            cat_timings[cat] = []
        cat_timings[cat].append(r['gt_failure_t'])

    for cat in sorted(cat_timings.keys()):
        times = cat_timings[cat]
        print(f"  {cat:20s}: n={len(times):2d}, mean_t={np.mean(times):.1f}, "
              f"std={np.std(times):.1f}, range=[{min(times)}, {max(times)}]")

    # ANOVA-like: between-category variance vs within-category variance
    all_times = [r['gt_failure_t'] for r in valid]
    grand_mean = np.mean(all_times)
    total_var = np.var(all_times)

    between_var = sum(len(cat_timings[c]) * (np.mean(cat_timings[c]) - grand_mean)**2
                      for c in cat_timings) / n
    within_var = sum(np.var(cat_timings[c]) * len(cat_timings[c])
                     for c in cat_timings if len(cat_timings[c]) > 1) / n

    eta_squared = between_var / total_var if total_var > 0 else 0
    print(f"\n  Grand mean GT: {grand_mean:.1f}, Total var: {total_var:.1f}")
    print(f"  Between-category var: {between_var:.1f}, Within-category var: {within_var:.1f}")
    print(f"  η² (category explains timing): {eta_squared:.3f}")
    print(f"  → Category {'explains' if eta_squared > 0.14 else 'does NOT explain'} GT timing "
          f"({'large' if eta_squared > 0.14 else 'small'} effect)")

    # 3. Description diversity
    print(f"\n--- Description Diversity ---")
    descriptions = [r['failure_mode'] for r in valid]
    unique_descs = len(set(descriptions))
    print(f"  Unique descriptions: {unique_descs}/{n}")

    # Pairwise word-level similarity
    word_sets = [word_set(d) for d in descriptions]
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            sim = jaccard_similarity(word_sets[i], word_sets[j])
            similarities.append(sim)

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    print(f"  Mean pairwise word Jaccard: {mean_sim:.3f} ± {std_sim:.3f}")
    print(f"  → Descriptions are {'very similar' if mean_sim > 0.5 else 'moderately diverse' if mean_sim > 0.3 else 'quite diverse'}")

    # 4. Visual cue analysis
    print(f"\n--- Visual Cue Analysis ---")
    all_cues = []
    for r in valid:
        all_cues.extend(r.get('visual_cues', []))

    # Word-level cue frequency
    cue_words = Counter()
    for cue in all_cues:
        for w in word_set(cue):
            cue_words[w] += 1

    print(f"  Total visual cues: {len(all_cues)}")
    print(f"  Unique cue strings: {len(set(all_cues))}/{len(all_cues)}")
    print(f"  Top-10 cue words:")
    for word, count in cue_words.most_common(10):
        print(f"    {word:20s}: {count:3d} ({100*count/len(all_cues):.0f}%)")

    # 5. Severity distribution
    print(f"\n--- Severity Distribution ---")
    severities = [r.get('severity', 'unknown') for r in valid]
    sev_counts = Counter(severities)
    for sev, count in sev_counts.most_common():
        print(f"  {sev:12s}: {count:2d} ({100*count/n:.0f}%)")

    # Check severity vs GT timing
    sev_timings = {}
    for r in valid:
        sev = r.get('severity', 'unknown')
        if sev not in sev_timings:
            sev_timings[sev] = []
        sev_timings[sev].append(r['gt_failure_t'])
    for sev in sorted(sev_timings.keys()):
        times = sev_timings[sev]
        print(f"    {sev:12s} mean_t={np.mean(times):.1f}")

    # 6. Clustering potential summary
    print(f"\n--- Clustering Potential Summary ---")
    print(f"  Category diversity: {'Good' if n_cats >= 4 else 'Low'} ({n_cats} categories)")
    print(f"  Description diversity: {'Good' if unique_descs >= n*0.8 else 'Low'} ({unique_descs}/{n} unique)")
    print(f"  Category-timing correlation: {'Informative' if eta_squared > 0.14 else 'Weak'} (η²={eta_squared:.3f})")
    print(f"  Lexical diversity: {'Good' if mean_sim < 0.3 else 'Low'} (mean Jaccard={mean_sim:.3f})")

    viable = (n_cats >= 3 and unique_descs >= n*0.5 and mean_sim < 0.5)
    print(f"\n  ★ Clustering viable for replay prioritization: {'YES' if viable else 'NO'}")
    if not viable:
        problems = []
        if n_cats < 3:
            problems.append("too few categories")
        if unique_descs < n*0.5:
            problems.append("descriptions too repetitive")
        if mean_sim >= 0.5:
            problems.append("descriptions lexically too similar")
        print(f"    Problems: {', '.join(problems)}")

    return {
        'task': task_name,
        'n': n,
        'n_categories': n_cats,
        'category_entropy': round(entropy, 3),
        'eta_squared': round(eta_squared, 3),
        'unique_descriptions': unique_descs,
        'mean_jaccard': round(mean_sim, 3),
        'clustering_viable': bool(viable),
        'category_distribution': dict(cat_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/failure_descriptions_iter39")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summaries = []

    for f in sorted(results_dir.glob("*.json")):
        if 'summary' in f.stem:
            continue
        with open(f) as fh:
            results = json.load(fh)
        task_name = f.stem.split("_K")[0].replace("_gh_gpt-4o-mini", "").replace("_gh_Phi-4-multimodal-instruct", " (Phi-4)").replace("_gh_", "_")
        summary = analyze_task(results, task_name)
        summaries.append(summary)

    # Cross-task summary
    if len(summaries) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-TASK SUMMARY")
        print(f"{'='*70}")
        for s in summaries:
            print(f"  {s['task']:15s}: {s['n_categories']} cats, η²={s['eta_squared']:.3f}, "
                  f"Jaccard={s['mean_jaccard']:.3f}, viable={s['clustering_viable']}")

    # Save summary
    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
