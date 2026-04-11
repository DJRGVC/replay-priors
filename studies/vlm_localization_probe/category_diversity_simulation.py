"""Category-diversity replay simulation (Proposal 4, Step 3).

Simulates three replay strategies on all 60 rollouts:
1. Uniform: equal probability for all episodes
2. Temporal-prediction: weight by inverse distance to VLM-predicted failure time
3. Category-diversity: weight by inverse category frequency (upweight rare failure modes)

Measures GT failure time coverage and diversity metrics.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_category_data():
    """Load failure category labels from iter 39 data."""
    base = Path("results/failure_descriptions_iter39")
    data = {}

    # reach-v3: GPT-4o-mini, 20 rollouts
    with open(base / "reach-v3_gh_gpt-4o-mini_K4.json") as f:
        raw = json.load(f)
    data["reach-v3"] = [
        {"rollout": r["rollout"], "gt_t": r["gt_failure_t"],
         "category": r.get("failure_category", "unknown")}
        for r in raw if "error" not in r
    ]

    # push-v3: Phi-4, 10 rollouts (more coverage than gpt-4o-mini's 5)
    with open(base / "push-v3_gh_Phi-4-multimodal-instruct_K4.json") as f:
        raw = json.load(f)
    data["push-v3"] = [
        {"rollout": r["rollout"], "gt_t": r["gt_failure_t"],
         "category": r.get("failure_category", "unknown")}
        for r in raw if "error" not in r
    ]

    # pick-place-v3: Phi-4, 9-10 rollouts
    with open(base / "pick-place-v3_gh_Phi-4-multimodal-instruct_K4.json") as f:
        raw = json.load(f)
    data["pick-place-v3"] = [
        {"rollout": r["rollout"], "gt_t": r["gt_failure_t"],
         "category": r.get("failure_category", "unknown")}
        for r in raw if "error" not in r
    ]

    return data


def load_temporal_predictions():
    """Load VLM temporal predictions for comparison."""
    preds = {}

    # reach-v3: use GPT-4o unannotated (best model)
    with open("results/gpt4o_noannotate/results.json") as f:
        raw = json.load(f)
    preds["reach-v3"] = {r["rollout"]: r["pred_failure_t"] for r in raw}

    # push-v3: GPT-4o unannotated
    with open("results/push_v3_gpt4o_mini_iter31/results.json") as f:
        raw = json.load(f)
    preds["push-v3"] = {r["rollout"]: r["pred_failure_t"] for r in raw}

    # pick-place-v3: GPT-4o-mini unannotated (GPT-4o all None)
    with open("results/pick_place_v3_gpt4o_mini_unannotated_iter32/results.json") as f:
        raw = json.load(f)
    preds["pick-place-v3"] = {r["rollout"]: r["pred_failure_t"] for r in raw if r["pred_failure_t"] is not None}

    return preds


def compute_weights_uniform(episodes):
    """Equal weight for all episodes."""
    n = len(episodes)
    return np.ones(n) / n


def compute_weights_category_diversity(episodes):
    """Weight inversely proportional to category frequency."""
    cats = [e["category"] for e in episodes]
    counts = Counter(cats)
    weights = np.array([1.0 / counts[c] for c in cats])
    return weights / weights.sum()


def compute_weights_temporal_prediction(episodes, pred_map, max_t=150):
    """Weight by predicted failure time deviation from median.
    Simulates temporal PER: upweight episodes whose predicted failure time
    deviates most from the median (most 'surprising' episodes)."""
    preds = []
    for e in episodes:
        p = pred_map.get(e["rollout"], max_t // 2)
        preds.append(float(p))
    preds = np.array(preds, dtype=float)
    median_pred = np.median(preds)
    # Priority = deviation from median + small epsilon to avoid zero weights
    deviations = np.abs(preds - median_pred) + 1.0
    weights = deviations / deviations.sum()
    return weights


def gt_coverage_metrics(sampled_gt_times, all_gt_times, n_bins=15, max_t=150):
    """Compute coverage metrics for a sample of GT failure times."""
    bin_edges = np.linspace(0, max_t, n_bins + 1)

    # Bin the full GT distribution
    all_hist, _ = np.histogram(all_gt_times, bins=bin_edges)
    occupied_bins = set(np.where(all_hist > 0)[0])

    # Bin the sample
    sample_hist, _ = np.histogram(sampled_gt_times, bins=bin_edges)
    sample_occupied = set(np.where(sample_hist > 0)[0])

    # Coverage: fraction of occupied GT bins that appear in sample
    coverage = len(sample_occupied & occupied_bins) / max(len(occupied_bins), 1)

    # Entropy of sample distribution (higher = more diverse)
    sample_probs = sample_hist / max(sample_hist.sum(), 1)
    sample_probs = sample_probs[sample_probs > 0]
    entropy = -np.sum(sample_probs * np.log2(sample_probs))

    # Unique GT times in sample
    unique_times = len(set(sampled_gt_times))

    return {
        "coverage": coverage,
        "entropy": round(entropy, 4),
        "unique_gt_times": unique_times,
        "occupied_bins_sample": len(sample_occupied & occupied_bins),
        "occupied_bins_total": len(occupied_bins),
    }


def simulate_replay(episodes, weights, sample_size, n_trials=1000, max_t=150, n_bins=15):
    """Simulate drawing sample_size episodes according to weights, n_trials times."""
    all_gt = np.array([e["gt_t"] for e in episodes])
    n = len(episodes)

    coverages = []
    entropies = []
    unique_counts = []

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        indices = rng.choice(n, size=sample_size, replace=False, p=weights)
        sampled_gt = all_gt[indices]
        m = gt_coverage_metrics(sampled_gt, all_gt, n_bins=n_bins, max_t=max_t)
        coverages.append(m["coverage"])
        entropies.append(m["entropy"])
        unique_counts.append(m["unique_gt_times"])

    return {
        "coverage_mean": round(np.mean(coverages), 4),
        "coverage_std": round(np.std(coverages), 4),
        "entropy_mean": round(np.mean(entropies), 4),
        "entropy_std": round(np.std(entropies), 4),
        "unique_mean": round(np.mean(unique_counts), 2),
    }


def run_simulation():
    """Run full simulation across tasks and strategies."""
    cat_data = load_category_data()
    temp_preds = load_temporal_predictions()

    results = {}
    # Sample sizes: 50%, 30% of pool
    sample_fractions = [0.5, 0.3]

    for task, episodes in cat_data.items():
        n = len(episodes)
        print(f"\n{'='*60}")
        print(f"Task: {task} (n={n})")
        cats = Counter(e["category"] for e in episodes)
        print(f"  Categories: {dict(cats)}")
        gt_times = [e["gt_t"] for e in episodes]
        print(f"  GT times: min={min(gt_times)}, max={max(gt_times)}, "
              f"mean={np.mean(gt_times):.1f}, std={np.std(gt_times):.1f}")

        task_results = {"n": n, "categories": dict(cats), "gt_stats": {
            "min": min(gt_times), "max": max(gt_times),
            "mean": round(np.mean(gt_times), 1), "std": round(np.std(gt_times), 1)
        }}

        # Compute weights
        w_uniform = compute_weights_uniform(episodes)
        w_category = compute_weights_category_diversity(episodes)
        w_temporal = compute_weights_temporal_prediction(episodes, temp_preds.get(task, {}))

        # Show weight distributions
        print(f"\n  Category-diversity weights:")
        for i, e in enumerate(episodes):
            if i < 5 or i == n-1:
                print(f"    {e['rollout']}: cat={e['category']:15s} w_cat={w_category[i]:.4f} w_uni={w_uniform[i]:.4f} w_temp={w_temporal[i]:.4f}")

        # Max/min weight ratios
        task_results["weight_ratios"] = {
            "category_max_min": round(w_category.max() / max(w_category.min(), 1e-10), 2),
            "temporal_max_min": round(w_temporal.max() / max(w_temporal.min(), 1e-10), 2),
        }
        print(f"\n  Weight max/min ratios: category={task_results['weight_ratios']['category_max_min']:.1f}x, "
              f"temporal={task_results['weight_ratios']['temporal_max_min']:.1f}x")

        task_results["sweeps"] = {}
        for frac in sample_fractions:
            sample_size = max(2, int(n * frac))
            print(f"\n  --- Sample {sample_size}/{n} ({frac*100:.0f}%) ---")

            for name, weights in [("uniform", w_uniform), ("category_diversity", w_category), ("temporal_prediction", w_temporal)]:
                sim = simulate_replay(episodes, weights, sample_size)
                print(f"    {name:25s}: coverage={sim['coverage_mean']:.3f}±{sim['coverage_std']:.3f}  "
                      f"entropy={sim['entropy_mean']:.3f}±{sim['entropy_std']:.3f}  "
                      f"unique={sim['unique_mean']:.1f}")
                task_results["sweeps"].setdefault(f"frac_{frac}", {})[name] = sim

        results[task] = task_results

    # Save results
    outdir = Path("results/category_diversity_sim_iter42")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outdir / 'simulation_results.json'}")

    return results


def make_figure(results):
    """Create comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    tasks = list(results.keys())
    strategies = ["uniform", "category_diversity", "temporal_prediction"]
    colors = {"uniform": "#888888", "category_diversity": "#2196F3", "temporal_prediction": "#FF5722"}
    labels = {"uniform": "Uniform", "category_diversity": "Category-diversity", "temporal_prediction": "Temporal-pred"}

    for col, task in enumerate(tasks):
        task_data = results[task]

        for row, metric in enumerate(["coverage", "entropy"]):
            ax = axes[row, col]
            frac = 0.5
            sweep = task_data["sweeps"][f"frac_{frac}"]
            x = np.arange(len(strategies))
            vals = [sweep[s][f"{metric}_mean"] for s in strategies]
            errs = [sweep[s][f"{metric}_std"] for s in strategies]
            bars = ax.bar(x, vals, yerr=errs, color=[colors[s] for s in strategies],
                         capsize=4, alpha=0.85, edgecolor="black", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels([labels[s] for s in strategies], rotation=20, ha="right", fontsize=8)
            if row == 0:
                ax.set_title(f"{task}\n(n={task_data['n']})", fontsize=10, fontweight="bold")
                ax.set_ylabel("GT bin coverage", fontsize=9)
                ax.set_ylim(0, 1.1)
            else:
                ax.set_ylabel("Sample entropy (bits)", fontsize=9)

            # Add value labels
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Category-Diversity Replay vs Uniform vs Temporal-Prediction\n"
                 "(50% subsample, 1000 trials)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    outpath = Path("figures/category_diversity_simulation_iter42.png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {outpath}")

    # Also save for Quarto
    quarto_path = Path("images/vlm_probe/category_diversity_simulation_iter42.png")
    quarto_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(quarto_path, dpi=150, bbox_inches="tight")
    print(f"Quarto figure saved to {quarto_path}")
    plt.close()


def synthetic_scale_test():
    """Test category-diversity at realistic buffer sizes (N=100, 500, 1000).

    Generates synthetic episodes with known category-GT correlation (η²),
    then measures whether category-diversity improves coverage at scale.
    """
    print("\n" + "="*60)
    print("SYNTHETIC SCALE-UP TEST")
    print("="*60)

    results = {}
    rng = np.random.default_rng(123)

    for eta_sq, label in [(0.3, "low_eta"), (0.6, "med_eta"), (0.95, "high_eta")]:
        for N in [50, 200, 1000]:
            # Generate synthetic episodes
            n_categories = 6
            # Assign categories with imbalanced distribution (Zipf-like)
            cat_probs = np.array([1.0/k for k in range(1, n_categories+1)])
            cat_probs /= cat_probs.sum()
            categories = rng.choice(n_categories, size=N, p=cat_probs)

            # Generate GT failure times correlated with category
            # η² controls how much of GT variance is explained by category
            cat_means = np.linspace(10, 140, n_categories)
            within_var = (1 - eta_sq) * 50**2  # residual variance
            gt_times = cat_means[categories] + rng.normal(0, np.sqrt(within_var), N)
            gt_times = np.clip(gt_times, 1, 150).astype(int)

            episodes = [{"rollout": f"r_{i}", "gt_t": int(gt_times[i]),
                        "category": f"cat_{categories[i]}"} for i in range(N)]

            # Compute weights
            w_uni = compute_weights_uniform(episodes)
            w_cat = compute_weights_category_diversity(episodes)

            # Sample 10% and 30%
            for frac in [0.1, 0.3]:
                sample_size = max(2, int(N * frac))
                sim_uni = simulate_replay(episodes, w_uni, sample_size, n_trials=500)
                sim_cat = simulate_replay(episodes, w_cat, sample_size, n_trials=500)

                delta_cov = sim_cat["coverage_mean"] - sim_uni["coverage_mean"]
                delta_ent = sim_cat["entropy_mean"] - sim_uni["entropy_mean"]

                key = f"{label}_N{N}_f{frac}"
                results[key] = {
                    "eta_sq": eta_sq, "N": N, "frac": frac,
                    "uniform_cov": sim_uni["coverage_mean"],
                    "catdiv_cov": sim_cat["coverage_mean"],
                    "delta_cov": round(delta_cov, 4),
                    "delta_ent": round(delta_ent, 4),
                }

                marker = "✓" if delta_cov > 0.01 else "≈" if delta_cov > -0.01 else "✗"
                print(f"  {marker} η²={eta_sq:.1f} N={N:4d} sample={frac*100:.0f}%: "
                      f"Δcov={delta_cov:+.4f} Δent={delta_ent:+.4f} "
                      f"(uni={sim_uni['coverage_mean']:.3f} cat={sim_cat['coverage_mean']:.3f})")

    # Save
    outdir = Path("results/category_diversity_sim_iter42")
    with open(outdir / "synthetic_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outdir / 'synthetic_scale_results.json'}")
    return results


def make_scale_figure(scale_results):
    """Plot synthetic scale-up results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    eta_labels = {"low_eta": "η²=0.3", "med_eta": "η²=0.6", "high_eta": "η²=0.95"}
    Ns = [50, 200, 1000]

    for col, (eta_key, eta_label) in enumerate(eta_labels.items()):
        ax = axes[col]
        for frac, color, marker in [(0.1, "#2196F3", "o"), (0.3, "#FF5722", "s")]:
            deltas = []
            for N in Ns:
                key = f"{eta_key}_N{N}_f{frac}"
                deltas.append(scale_results[key]["delta_cov"])
            ax.plot(Ns, deltas, f"-{marker}", color=color, label=f"sample={frac*100:.0f}%",
                   markersize=8, linewidth=2)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Buffer size N", fontsize=10)
        ax.set_ylabel("Δ coverage (cat-div − uniform)", fontsize=10)
        ax.set_title(eta_label, fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(fontsize=9)

    fig.suptitle("Category-Diversity Replay: Effect of η² and Buffer Size\n"
                 "(synthetic data, 500 trials each)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    outpath = Path("figures/category_diversity_scale_iter42.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")

    quarto_path = Path("images/vlm_probe/category_diversity_scale_iter42.png")
    fig.savefig(quarto_path, dpi=150, bbox_inches="tight")
    print(f"Scale figure saved to {outpath} and {quarto_path}")
    plt.close()


if __name__ == "__main__":
    results = run_simulation()
    make_figure(results)
    scale_results = synthetic_scale_test()
    make_scale_figure(scale_results)
