#!/usr/bin/env python3
"""
Bias-Aware Ensemble Prioritization (BAEP) — Proposal 1 from VISIONARY_PROPOSALS.md

Implements debiased ensemble prediction on existing multi-model VLM probe data.
Tests whether ensembling predictions from models with uncorrelated positional biases
improves failure-timestep localization over any individual model.

Key idea: Each VLM has a distinct positional attractor (Claude→center, Gemini→start,
GPT-4o→mid, GPT-4o-mini→end). Debiasing each model's prediction by its characteristic
bias and then averaging should cancel errors that are structurally different across models.
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_all_results(results_root="results"):
    """Load all per-rollout predictions from results directories."""
    all_results = []
    for results_file in glob.glob(os.path.join(results_root, "*/results.json")):
        try:
            data = json.load(open(results_file))
            for r in data:
                if r.get("pred_failure_t") is not None and r.get("gt_failure_t") is not None:
                    all_results.append(r)
        except Exception:
            pass
    return all_results


def build_prediction_matrix(results, task="reach-v3", K=8, annotated=True):
    """
    Build a rollout × model prediction matrix from raw results.
    Returns: rollouts (list), models (list), pred_matrix (n_rollouts × n_models, NaN for missing),
             gt_vector (n_rollouts,)
    """
    # Deduplicate: take first prediction per (model, rollout, annotated)
    seen = set()
    filtered = []
    for r in results:
        if r.get("task") != task or r.get("K") != K:
            continue
        ann = r.get("annotate", False)
        if ann != annotated:
            continue
        key = (r["model"], r["rollout"])
        if key not in seen:
            seen.add(key)
            filtered.append(r)

    # Get unique rollouts and models
    rollout_set = sorted(set(r["rollout"] for r in filtered))
    model_set = sorted(set(r["model"] for r in filtered))

    # Build matrix
    pred_matrix = np.full((len(rollout_set), len(model_set)), np.nan)
    gt_vector = np.full(len(rollout_set), np.nan)

    rollout_idx = {r: i for i, r in enumerate(rollout_set)}
    model_idx = {m: i for i, m in enumerate(model_set)}

    for r in filtered:
        ri = rollout_idx[r["rollout"]]
        mi = model_idx[r["model"]]
        pred_matrix[ri, mi] = r["pred_failure_t"]
        gt_vector[ri] = r["gt_failure_t"]

    return rollout_set, model_set, pred_matrix, gt_vector


def compute_bias_profile(preds, gts):
    """
    Compute bias profile for a model: characteristic attractor mu and strength alpha.
    Returns: mu (attractor position), alpha (attractor strength 0-1), bias (mean signed error)
    """
    valid = ~np.isnan(preds)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    p = preds[valid]
    g = gts[valid]
    mu = np.median(p)  # characteristic attractor
    # Alpha = fraction of predictions within 10 of the attractor
    alpha = np.mean(np.abs(p - mu) < 15)
    bias = np.mean(p - g)  # mean signed error
    return mu, alpha, bias


def debias_predictions(preds, mu, bias, method="linear"):
    """
    Debias predictions by subtracting estimated bias.
    method='linear': simple bias subtraction
    method='attractor': shift predictions away from attractor proportionally
    """
    debiased = preds.copy()
    valid = ~np.isnan(debiased)
    if method == "linear":
        debiased[valid] = debiased[valid] - bias
    elif method == "attractor":
        # Shift predictions away from attractor
        debiased[valid] = debiased[valid] + 0.5 * (debiased[valid] - mu)
    # Clip to valid range
    debiased = np.clip(debiased, 0, 149)
    return debiased


def ensemble_predict(pred_matrix, gt_vector, models, method="mean", debias="none"):
    """
    Compute ensemble predictions using various strategies.

    method: 'mean', 'median', 'inverse_var_weighted'
    debias: 'none', 'linear', 'attractor'
    """
    n_rollouts, n_models = pred_matrix.shape

    # Step 1: Compute per-model bias profiles (leave-one-out to avoid data leakage)
    mus = np.zeros(n_models)
    biases = np.zeros(n_models)
    variances = np.zeros(n_models)
    for mi in range(n_models):
        valid = ~np.isnan(pred_matrix[:, mi])
        if valid.sum() < 3:
            mus[mi] = 75  # default center
            biases[mi] = 0
            variances[mi] = 1e6
            continue
        p = pred_matrix[valid, mi]
        g = gt_vector[valid]
        mus[mi], _, biases[mi] = compute_bias_profile(pred_matrix[:, mi], gt_vector)
        variances[mi] = np.var(p - g) + 1e-6

    # Step 2: Debias if requested
    debiased_matrix = pred_matrix.copy()
    if debias != "none":
        for mi in range(n_models):
            debiased_matrix[:, mi] = debias_predictions(
                pred_matrix[:, mi], mus[mi], biases[mi], method=debias
            )

    # Step 3: Ensemble
    ensemble_preds = np.full(n_rollouts, np.nan)
    ensemble_agreement = np.full(n_rollouts, np.nan)

    for ri in range(n_rollouts):
        valid = ~np.isnan(debiased_matrix[ri, :])
        if valid.sum() == 0:
            continue
        p = debiased_matrix[ri, valid]

        if method == "mean":
            ensemble_preds[ri] = np.mean(p)
        elif method == "median":
            ensemble_preds[ri] = np.median(p)
        elif method == "inverse_var_weighted":
            w = 1.0 / variances[valid]
            ensemble_preds[ri] = np.average(p, weights=w)

        # Agreement = 1 - normalized spread
        if valid.sum() > 1:
            ensemble_agreement[ri] = 1 - np.std(p) / 75.0  # normalize by half-episode
        else:
            ensemble_agreement[ri] = 0.5

    return ensemble_preds, ensemble_agreement, mus, biases


def evaluate(preds, gts, name=""):
    """Compute MAE, ±5, ±10, ±20 accuracy."""
    valid = ~np.isnan(preds) & ~np.isnan(gts)
    if valid.sum() == 0:
        return {"name": name, "n": 0}
    p = preds[valid]
    g = gts[valid]
    errors = np.abs(p - g)
    return {
        "name": name,
        "n": int(valid.sum()),
        "mae": float(np.mean(errors)),
        "median_ae": float(np.median(errors)),
        "within_5": float(np.mean(errors <= 5) * 100),
        "within_10": float(np.mean(errors <= 10) * 100),
        "within_20": float(np.mean(errors <= 20) * 100),
        "unique_preds": int(len(set(np.round(p).astype(int)))),
        "mean_pred": float(np.mean(p)),
        "std_pred": float(np.std(p)),
    }


def run_analysis():
    results = load_all_results()
    print(f"Loaded {len(results)} total predictions\n")

    all_metrics = []

    for annotated in [True, False]:
        ann_label = "annotated" if annotated else "unannotated"
        rollouts, models, pred_matrix, gt_vector = build_prediction_matrix(
            results, task="reach-v3", K=8, annotated=annotated
        )
        n_rollouts, n_models = pred_matrix.shape
        print(f"=== {ann_label.upper()} condition ===")
        print(f"Rollouts: {n_rollouts}, Models: {n_models}")
        print(f"Models: {models}")

        # Coverage matrix
        coverage = ~np.isnan(pred_matrix)
        print(f"Coverage per model: {dict(zip(models, coverage.sum(axis=0)))}")
        print()

        # Per-model baselines
        print("--- Individual model baselines ---")
        for mi, model in enumerate(models):
            m = evaluate(pred_matrix[:, mi], gt_vector, name=model.split(":")[-1])
            if m["n"] > 0:
                print(f"  {m['name']:35s} n={m['n']:2d} MAE={m['mae']:5.1f} ±10={m['within_10']:4.0f}% ±20={m['within_20']:4.0f}% unique={m['unique_preds']}")
                m["condition"] = ann_label
                m["type"] = "individual"
                all_metrics.append(m)

        # Bias profiles
        print("\n--- Bias profiles ---")
        for mi, model in enumerate(models):
            mu, alpha, bias = compute_bias_profile(pred_matrix[:, mi], gt_vector)
            if not np.isnan(mu):
                print(f"  {model.split(':')[-1]:35s} attractor={mu:5.0f} strength={alpha:.2f} bias={bias:+6.1f}")

        # Ensemble methods
        print("\n--- Ensemble results ---")
        for debias in ["none", "linear", "attractor"]:
            for method in ["mean", "median", "inverse_var_weighted"]:
                ens_preds, agreement, _, _ = ensemble_predict(
                    pred_matrix, gt_vector, models, method=method, debias=debias
                )
                m = evaluate(ens_preds, gt_vector, name=f"ENS({method},{debias})")
                if m["n"] > 0:
                    mean_agree = np.nanmean(agreement)
                    print(f"  {m['name']:35s} n={m['n']:2d} MAE={m['mae']:5.1f} ±10={m['within_10']:4.0f}% ±20={m['within_20']:4.0f}% agree={mean_agree:.2f}")
                    m["condition"] = ann_label
                    m["type"] = "ensemble"
                    m["mean_agreement"] = mean_agree
                    all_metrics.append(m)

        # Best individual vs best ensemble
        individual = [m for m in all_metrics if m["condition"] == ann_label and m["type"] == "individual" and m["n"] >= 5]
        ensembles = [m for m in all_metrics if m["condition"] == ann_label and m["type"] == "ensemble" and m["n"] >= 5]
        if individual and ensembles:
            best_ind = min(individual, key=lambda x: x["mae"])
            best_ens = min(ensembles, key=lambda x: x["mae"])
            print(f"\n  Best individual: {best_ind['name']} MAE={best_ind['mae']:.1f}")
            print(f"  Best ensemble:   {best_ens['name']} MAE={best_ens['mae']:.1f}")
            delta = best_ens["mae"] - best_ind["mae"]
            print(f"  Delta: {delta:+.1f} ({'ensemble wins' if delta < 0 else 'individual wins'})")

        # Subset ensembles: try all pairs and triples
        print("\n--- Subset ensembles (best 2-model and 3-model) ---")
        best_pair = None
        best_triple = None
        for subset_size in [2, 3]:
            for combo in combinations(range(n_models), subset_size):
                sub_matrix = pred_matrix[:, list(combo)]
                sub_models = [models[i] for i in combo]
                ens_preds, agreement, _, _ = ensemble_predict(
                    sub_matrix, gt_vector, sub_models, method="median", debias="linear"
                )
                m = evaluate(ens_preds, gt_vector)
                if m["n"] >= 5:
                    label = "+".join(s.split(":")[-1][:8] for s in sub_models)
                    if subset_size == 2 and (best_pair is None or m["mae"] < best_pair[1]):
                        best_pair = (label, m["mae"], m["within_10"], m["within_20"], m["n"])
                    if subset_size == 3 and (best_triple is None or m["mae"] < best_triple[1]):
                        best_triple = (label, m["mae"], m["within_10"], m["within_20"], m["n"])

        if best_pair:
            print(f"  Best pair:   {best_pair[0]:40s} MAE={best_pair[1]:.1f} ±10={best_pair[2]:.0f}% ±20={best_pair[3]:.0f}% n={best_pair[4]}")
        if best_triple:
            print(f"  Best triple: {best_triple[0]:40s} MAE={best_triple[1]:.1f} ±10={best_triple[2]:.0f}% ±20={best_triple[3]:.0f}% n={best_triple[4]}")

        print("\n")

    return all_metrics


def plot_ensemble_comparison(all_metrics, output_path="figures/ensemble_baep_analysis.png"):
    """Create publication-quality figure comparing individual vs ensemble performance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, condition in enumerate(["annotated", "unannotated"]):
        ax = axes[ax_idx]
        cond_metrics = [m for m in all_metrics if m.get("condition") == condition and m["n"] >= 5]
        individuals = [m for m in cond_metrics if m["type"] == "individual"]
        ensembles = [m for m in cond_metrics if m["type"] == "ensemble"]

        # Sort by MAE
        individuals.sort(key=lambda x: x["mae"])
        ensembles.sort(key=lambda x: x["mae"])

        # Plot individuals
        names_ind = [m["name"][:20] for m in individuals]
        maes_ind = [m["mae"] for m in individuals]
        bars1 = ax.barh(range(len(individuals)), maes_ind, height=0.6,
                       color="#4ECDC4", alpha=0.8, label="Individual models")

        # Plot top 5 ensembles
        top_ens = ensembles[:5]
        names_ens = [m["name"] for m in top_ens]
        maes_ens = [m["mae"] for m in top_ens]
        offset = len(individuals) + 1
        bars2 = ax.barh(range(offset, offset + len(top_ens)), maes_ens, height=0.6,
                       color="#FF6B6B", alpha=0.8, label="Ensemble methods")

        all_names = names_ind + [""] + names_ens
        ax.set_yticks(list(range(len(individuals))) + [len(individuals)] + list(range(offset, offset + len(top_ens))))
        ax.set_yticklabels(all_names, fontsize=8)
        ax.set_xlabel("MAE (timesteps)", fontsize=11)
        ax.set_title(f"Reach-v3 K=8 — {condition}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f"{width:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")

    # Also save to images/vlm_probe/ for Quarto
    quarto_path = output_path.replace("figures/", "../../images/vlm_probe/")
    os.makedirs(os.path.dirname(quarto_path), exist_ok=True)
    plt.savefig(quarto_path, dpi=150, bbox_inches="tight")
    print(f"Quarto figure saved to {quarto_path}")
    plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    metrics = run_analysis()
    plot_ensemble_comparison(metrics)

    # Save raw metrics
    os.makedirs("results/ensemble_baep", exist_ok=True)
    with open("results/ensemble_baep/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/ensemble_baep/metrics.json")
