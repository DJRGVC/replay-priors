"""Cross-model category comparison analysis (Iteration 43).

Since all VLM APIs are rate-limited, this does pure analysis on existing data:
1. Category distribution comparison: GPT-4o-mini (reach, n=20) vs Phi-4 (push n=10, pick-place n=9)
2. Description style analysis: length, vocabulary overlap, novel categories
3. Category stability bootstrap: how stable are distributions under subsampling?
4. Prepares metrics for when same-rollout cross-model data becomes available.
"""

import json
import numpy as np
from collections import Counter
from pathlib import Path
from itertools import combinations

DATA_DIR = Path("results/failure_descriptions_iter39")

# Canonical categories from the prompt
CANONICAL_CATS = {"never_reached", "overshot", "oscillated", "wrong_direction", "stuck", "other"}


def load_successful(path):
    """Load results, filtering out errors and parse failures."""
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if "error" not in r and r.get("failure_category") != "parse_error"]


def category_distribution(results):
    """Return normalized category distribution."""
    cats = [r["failure_category"] for r in results]
    counts = Counter(cats)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}, counts


def jensen_shannon(p_dict, q_dict):
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p_dict) | set(q_dict)
    p = np.array([p_dict.get(k, 0) for k in sorted(all_keys)])
    q = np.array([q_dict.get(k, 0) for k in sorted(all_keys)])
    # Add small epsilon for numerical stability
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return jsd


def category_stability_bootstrap(results, n_bootstrap=1000, subsample_frac=0.7):
    """Bootstrap category distribution stability within a single model's results."""
    n = len(results)
    k = max(2, int(n * subsample_frac))
    cats_all = [r["failure_category"] for r in results]
    all_cats = sorted(set(cats_all))

    distributions = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=k, replace=True)
        sub_cats = [cats_all[i] for i in idx]
        counts = Counter(sub_cats)
        total = sum(counts.values())
        dist = {c: counts.get(c, 0) / total for c in all_cats}
        distributions.append(dist)

    # Compute pairwise JSD between bootstrap samples
    jsds = []
    sample_pairs = min(500, len(distributions) * (len(distributions) - 1) // 2)
    pairs = np.random.choice(len(distributions), size=(sample_pairs, 2), replace=True)
    for i, j in pairs:
        if i != j:
            jsds.append(jensen_shannon(distributions[i], distributions[j]))

    return {
        "mean_jsd": float(np.mean(jsds)),
        "std_jsd": float(np.std(jsds)),
        "max_jsd": float(np.max(jsds)),
        "n_bootstrap": n_bootstrap,
        "subsample_k": k,
    }


def description_stats(results, label):
    """Compute description-level statistics."""
    descs = [r.get("failure_mode", "") for r in results]
    lengths = [len(d.split()) for d in descs]
    cues_counts = [len(r.get("visual_cues", [])) for r in results]

    # Vocabulary analysis
    all_words = set()
    for d in descs:
        all_words.update(d.lower().split())

    cats = [r["failure_category"] for r in results]
    novel = set(cats) - CANONICAL_CATS

    return {
        "label": label,
        "n": len(results),
        "mean_desc_words": round(np.mean(lengths), 1),
        "std_desc_words": round(np.std(lengths), 1),
        "mean_visual_cues": round(np.mean(cues_counts), 1),
        "unique_descriptions": len(set(descs)),
        "unique_words": len(all_words),
        "novel_categories": sorted(novel) if novel else [],
        "category_distribution": dict(Counter(cats).most_common()),
    }


def main():
    print("=" * 70)
    print("Cross-Model Category Analysis — Iteration 43")
    print("=" * 70)

    # Load all successful data
    datasets = {
        "reach-v3 / GPT-4o-mini": load_successful(DATA_DIR / "reach-v3_gh_gpt-4o-mini_K4.json"),
        "push-v3 / Phi-4": load_successful(DATA_DIR / "push-v3_gh_Phi-4-multimodal-instruct_K4.json"),
        "pick-place-v3 / Phi-4": load_successful(DATA_DIR / "pick-place-v3_gh_Phi-4-multimodal-instruct_K4.json"),
    }

    # 1. Description-level statistics
    print("\n## 1. Description Statistics by Model×Task")
    print("-" * 70)
    all_stats = []
    for label, results in datasets.items():
        stats = description_stats(results, label)
        all_stats.append(stats)
        print(f"\n  {label} (n={stats['n']}):")
        print(f"    Description length: {stats['mean_desc_words']} ± {stats['std_desc_words']} words")
        print(f"    Visual cues/episode: {stats['mean_visual_cues']}")
        print(f"    Unique descriptions: {stats['unique_descriptions']}/{stats['n']}")
        print(f"    Vocabulary size: {stats['unique_words']} unique words")
        print(f"    Novel categories: {stats['novel_categories'] or 'none'}")
        print(f"    Categories: {stats['category_distribution']}")

    # 2. Cross-distribution comparison
    print("\n\n## 2. Category Distribution Comparison")
    print("-" * 70)
    dists = {}
    for label, results in datasets.items():
        dist, _ = category_distribution(results)
        dists[label] = dist

    # Pairwise JSD
    labels = list(dists.keys())
    for i, j in combinations(range(len(labels)), 2):
        jsd = jensen_shannon(dists[labels[i]], dists[labels[j]])
        print(f"\n  JSD({labels[i]} vs {labels[j]}) = {jsd:.4f}")

    # GPT-4o-mini vs Phi-4 (pooled across tasks)
    gpt_results = datasets["reach-v3 / GPT-4o-mini"]
    phi_results = datasets["push-v3 / Phi-4"] + datasets["pick-place-v3 / Phi-4"]
    gpt_dist, _ = category_distribution(gpt_results)
    phi_dist, _ = category_distribution(phi_results)
    jsd_model = jensen_shannon(gpt_dist, phi_dist)
    print(f"\n  JSD(GPT-4o-mini pooled vs Phi-4 pooled) = {jsd_model:.4f}")
    print(f"    NOTE: confounded by task differences — need same-rollout data to deconfound")

    # 3. Category overlap analysis
    print("\n\n## 3. Category Overlap Analysis")
    print("-" * 70)
    gpt_cats = set(r["failure_category"] for r in gpt_results)
    phi_cats = set(r["failure_category"] for r in phi_results)
    print(f"  GPT-4o-mini categories used: {sorted(gpt_cats)}")
    print(f"  Phi-4 categories used: {sorted(phi_cats)}")
    print(f"  Intersection: {sorted(gpt_cats & phi_cats)}")
    print(f"  GPT-only: {sorted(gpt_cats - phi_cats)}")
    print(f"  Phi-4-only: {sorted(phi_cats - phi_cats)}")
    phi_only = phi_cats - gpt_cats
    print(f"  Phi-4-only (novel): {sorted(phi_only)}")
    print(f"  Jaccard similarity: {len(gpt_cats & phi_cats) / len(gpt_cats | phi_cats):.3f}")

    # 4. Category stability bootstrap
    print("\n\n## 4. Category Stability (Bootstrap JSD)")
    print("-" * 70)
    np.random.seed(42)
    for label, results in datasets.items():
        if len(results) >= 5:
            stability = category_stability_bootstrap(results)
            print(f"\n  {label} (n={len(results)}):")
            print(f"    Intra-model JSD: {stability['mean_jsd']:.4f} ± {stability['std_jsd']:.4f} (max={stability['max_jsd']:.4f})")
        else:
            print(f"\n  {label}: too few samples (n={len(results)})")

    # 5. Severity comparison
    print("\n\n## 5. Severity Distribution")
    print("-" * 70)
    for label, results in datasets.items():
        sevs = Counter(r.get("severity", "unknown") for r in results)
        print(f"  {label}: {dict(sevs.most_common())}")

    # 6. Key finding summary
    print("\n\n## 6. Summary")
    print("-" * 70)

    # Check if Phi-4 invents novel categories
    all_novel = set()
    for label, results in datasets.items():
        for r in results:
            if r["failure_category"] not in CANONICAL_CATS:
                all_novel.add(r["failure_category"])

    print(f"  Novel (non-canonical) categories: {sorted(all_novel) if all_novel else 'none'}")
    print(f"  Both models use all 6 canonical categories: GPT={len(gpt_cats & CANONICAL_CATS)}/6, Phi-4={len(phi_cats & CANONICAL_CATS)}/6")
    print(f"  Cross-model JSD (confounded): {jsd_model:.4f}")
    print(f"  Interpretation: JSD < 0.1 = similar, 0.1-0.3 = moderate, > 0.3 = different")

    # Save results
    output = {
        "description_stats": all_stats,
        "cross_model_jsd_confounded": jsd_model,
        "category_overlap": {
            "gpt_categories": sorted(gpt_cats),
            "phi_categories": sorted(phi_cats),
            "intersection": sorted(gpt_cats & phi_cats),
            "jaccard": len(gpt_cats & phi_cats) / len(gpt_cats | phi_cats),
        },
        "novel_categories": sorted(all_novel),
        "note": "JSD is confounded by task differences. Same-rollout comparison needed for deconfounding."
    }
    outpath = DATA_DIR / "cross_model_analysis_iter43.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
