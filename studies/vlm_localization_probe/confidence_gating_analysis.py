#!/usr/bin/env python3
"""
Confidence-Gated VLM-PER — Proposal 5 from VISIONARY_PROPOSALS.md

Uses inter-model disagreement as a confidence signal to gate between
VLM-derived priority and uniform replay. When models agree (low std),
trust the VLM prediction; when they disagree (high std), fall back to
uniform.

This analysis runs on existing multi-model probe data — no new API calls.

Key metrics:
- Gated oracle overlap: fraction of oracle top-K transitions captured
- Gated KL divergence: information loss vs oracle
- Coverage: fraction of rollouts where gating uses VLM vs uniform
- Comparison to always-VLM, always-uniform, and best-individual baselines
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from priority_score import (
    vlm_priority, oracle_priority, priority_kl_divergence, priority_overlap
)


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


def build_rollout_model_matrix(results, task="reach-v3", K=8, annotated=True):
    """
    Build per-rollout multi-model prediction data.
    Returns list of dicts, each with:
      rollout, gt, predictions={model: pred_t}, agreement, best_pred
    """
    # Filter and deduplicate
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

    # Group by rollout
    by_rollout = defaultdict(list)
    for r in filtered:
        by_rollout[r["rollout"]].append(r)

    rollout_data = []
    for rollout_id, preds in by_rollout.items():
        if len(preds) < 2:
            continue  # Need at least 2 models for agreement

        gt = preds[0]["gt_failure_t"]
        pred_dict = {}
        errors = {}
        for p in preds:
            model = p["model"]
            pred_t = p["pred_failure_t"]
            pred_dict[model] = pred_t
            errors[model] = abs(pred_t - gt)

        pred_values = list(pred_dict.values())
        std = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        median_pred = np.median(pred_values)

        # Best individual model for this rollout (oracle-informed, for analysis)
        best_model = min(errors, key=errors.get)

        rollout_data.append({
            "rollout": rollout_id,
            "gt": gt,
            "predictions": pred_dict,
            "errors": errors,
            "n_models": len(pred_dict),
            "pred_std": std,
            "pred_mean": mean_pred,
            "pred_median": median_pred,
            "best_model": best_model,
            "best_error": errors[best_model],
            "mean_error": np.mean(list(errors.values())),
        })

    return rollout_data


def gated_priority(
    pred_t, confidence, agreement, threshold,
    total_steps=150, sigma=10.0
):
    """
    Confidence-gated priority: use VLM when agreement > threshold,
    otherwise fall back to uniform.

    Returns (priority_vector, used_vlm: bool)
    """
    uniform = np.ones(total_steps) / total_steps
    if agreement >= threshold:
        # High confidence: use VLM prediction
        return vlm_priority(pred_t, confidence, total_steps, sigma), True
    else:
        # Low confidence: fall back to uniform
        return uniform, False


def evaluate_gating_threshold(
    rollout_data, threshold, total_steps=150, sigma=10.0,
    pred_source="median"
):
    """
    Evaluate a single gating threshold across all rollouts.

    pred_source: "median" (ensemble median), "mean" (ensemble mean),
                 or a model name (use that model's prediction)
    """
    kl_gated, kl_uniform, kl_always_vlm = [], [], []
    overlap_gated, overlap_uniform, overlap_always_vlm = [], [], []
    vlm_used_count = 0

    uniform = np.ones(total_steps) / total_steps

    for rd in rollout_data:
        gt = rd["gt"]
        oracle = oracle_priority(gt, total_steps, sigma)

        # Select prediction
        if pred_source == "median":
            pred_t = rd["pred_median"]
        elif pred_source == "mean":
            pred_t = rd["pred_mean"]
        else:
            pred_t = rd["predictions"].get(pred_source, rd["pred_median"])

        confidence = 0.5  # Fixed moderate confidence

        # Agreement metric: 1 - normalized std
        # Normalize std by half-episode (75 timesteps)
        agreement = 1.0 - rd["pred_std"] / 75.0

        # Gated priority
        gated_p, used_vlm = gated_priority(
            pred_t, confidence, agreement, threshold, total_steps, sigma
        )
        if used_vlm:
            vlm_used_count += 1

        # Always-VLM priority (no gating)
        always_vlm_p = vlm_priority(pred_t, confidence, total_steps, sigma)

        # Metrics
        kl_gated.append(priority_kl_divergence(oracle, gated_p))
        kl_uniform.append(priority_kl_divergence(oracle, uniform))
        kl_always_vlm.append(priority_kl_divergence(oracle, always_vlm_p))

        overlap_gated.append(priority_overlap(oracle, gated_p))
        overlap_uniform.append(priority_overlap(oracle, uniform))
        overlap_always_vlm.append(priority_overlap(oracle, always_vlm_p))

    n = len(rollout_data)
    vlm_frac = vlm_used_count / n if n > 0 else 0

    return {
        "threshold": threshold,
        "n": n,
        "vlm_used": vlm_used_count,
        "vlm_fraction": vlm_frac,
        "kl_gated": float(np.mean(kl_gated)),
        "kl_uniform": float(np.mean(kl_uniform)),
        "kl_always_vlm": float(np.mean(kl_always_vlm)),
        "overlap_gated": float(np.mean(overlap_gated)),
        "overlap_uniform": float(np.mean(overlap_uniform)),
        "overlap_always_vlm": float(np.mean(overlap_always_vlm)),
        # Per-rollout details for analysis
        "kl_gated_values": kl_gated,
        "overlap_gated_values": overlap_gated,
    }


def threshold_sweep(rollout_data, total_steps=150, sigma=10.0,
                    pred_source="median"):
    """Sweep confidence thresholds and evaluate gating performance."""
    thresholds = np.arange(0.0, 1.01, 0.05)
    results = []
    for th in thresholds:
        r = evaluate_gating_threshold(
            rollout_data, th, total_steps, sigma, pred_source
        )
        results.append(r)
    return results


def analyze_agreement_vs_error(rollout_data):
    """Check if agreement actually correlates with prediction quality."""
    agreements = [1.0 - rd["pred_std"] / 75.0 for rd in rollout_data]
    mean_errors = [rd["mean_error"] for rd in rollout_data]
    median_errors = [abs(rd["pred_median"] - rd["gt"]) for rd in rollout_data]
    best_errors = [rd["best_error"] for rd in rollout_data]

    corr_mean = np.corrcoef(agreements, mean_errors)[0, 1]
    corr_median = np.corrcoef(agreements, median_errors)[0, 1]
    corr_best = np.corrcoef(agreements, best_errors)[0, 1]

    return {
        "corr_agreement_vs_mean_error": corr_mean,
        "corr_agreement_vs_median_error": corr_median,
        "corr_agreement_vs_best_error": corr_best,
        "agreements": agreements,
        "mean_errors": mean_errors,
        "median_errors": median_errors,
        "best_errors": best_errors,
    }


def plot_confidence_gating(sweep_results, corr_data, rollout_data,
                           output_path="figures/confidence_gating_analysis.png",
                           condition_label=""):
    """Create publication-quality 4-panel figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Threshold sweep — KL divergence ---
    ax = axes[0, 0]
    thresholds = [r["threshold"] for r in sweep_results]
    kl_gated = [r["kl_gated"] for r in sweep_results]
    kl_uniform = [r["kl_uniform"] for r in sweep_results]
    kl_always = [r["kl_always_vlm"] for r in sweep_results]

    ax.plot(thresholds, kl_gated, 'b-o', markersize=4, label='Gated VLM', linewidth=2)
    ax.axhline(kl_uniform[0], color='gray', linestyle='--', label='Always uniform', alpha=0.7)
    ax.axhline(kl_always[0], color='red', linestyle=':', label='Always VLM', alpha=0.7)
    ax.set_xlabel('Agreement threshold τ', fontsize=11)
    ax.set_ylabel('KL(oracle ∥ priority)', fontsize=11)
    ax.set_title('KL Divergence vs Threshold (lower = better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Mark optimal threshold (lowest KL)
    best_kl_idx = np.argmin(kl_gated)
    ax.axvline(thresholds[best_kl_idx], color='blue', linestyle=':', alpha=0.4)
    ax.annotate(f'τ*={thresholds[best_kl_idx]:.2f}',
                xy=(thresholds[best_kl_idx], kl_gated[best_kl_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='blue'))

    # --- Panel 2: Threshold sweep — overlap ---
    ax = axes[0, 1]
    ov_gated = [r["overlap_gated"] for r in sweep_results]
    ov_uniform = [r["overlap_uniform"] for r in sweep_results]
    ov_always = [r["overlap_always_vlm"] for r in sweep_results]
    vlm_frac = [r["vlm_fraction"] for r in sweep_results]

    ax.plot(thresholds, ov_gated, 'b-o', markersize=4, label='Gated VLM overlap', linewidth=2)
    ax.axhline(ov_uniform[0], color='gray', linestyle='--', label='Always uniform', alpha=0.7)
    ax.axhline(ov_always[0], color='red', linestyle=':', label='Always VLM', alpha=0.7)
    ax.set_xlabel('Agreement threshold τ', fontsize=11)
    ax.set_ylabel('Top-20% oracle overlap', fontsize=11)
    ax.set_title('Oracle Overlap vs Threshold (higher = better)', fontsize=12, fontweight='bold')

    # Secondary axis: VLM usage fraction
    ax2 = ax.twinx()
    ax2.fill_between(thresholds, vlm_frac, alpha=0.15, color='green')
    ax2.plot(thresholds, vlm_frac, 'g-', alpha=0.5, linewidth=1)
    ax2.set_ylabel('Fraction using VLM (green fill)', fontsize=9, color='green')
    ax2.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower left')

    # --- Panel 3: Agreement vs error scatter ---
    ax = axes[1, 0]
    agreements = corr_data["agreements"]
    median_errors = corr_data["median_errors"]

    ax.scatter(agreements, median_errors, alpha=0.6, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Inter-model agreement (1 − σ/75)', fontsize=11)
    ax.set_ylabel('Ensemble median AE (timesteps)', fontsize=11)
    ax.set_title(f'Agreement vs Error (r={corr_data["corr_agreement_vs_median_error"]:.2f})',
                 fontsize=12, fontweight='bold')

    # Add trend line
    z = np.polyfit(agreements, median_errors, 1)
    x_line = np.linspace(min(agreements), max(agreements), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, label=f'Linear fit (slope={z[0]:.1f})')
    ax.legend(fontsize=9)

    # --- Panel 4: Per-rollout comparison —  gated vs always-VLM vs uniform ---
    ax = axes[1, 1]

    # Find optimal threshold from KL sweep
    best_th = thresholds[best_kl_idx]
    best_result = sweep_results[best_kl_idx]

    # Show rollout-level improvement
    n_rollouts = len(rollout_data)
    rollout_ids = range(n_rollouts)

    # For each rollout at optimal threshold: compute KL for gated vs always-VLM
    kl_per_rollout_gated = best_result["kl_gated_values"]
    kl_per_rollout_always = sweep_results[0]["kl_gated_values"]  # th=0 is always-VLM

    improvements = [a - g for a, g in zip(kl_per_rollout_always, kl_per_rollout_gated)]
    sorted_imp = sorted(enumerate(improvements), key=lambda x: x[1], reverse=True)
    sorted_idx = [s[0] for s in sorted_imp]
    sorted_vals = [s[1] for s in sorted_imp]

    colors = ['#4ECDC4' if v > 0 else '#FF6B6B' for v in sorted_vals]
    ax.bar(range(n_rollouts), sorted_vals, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Rollouts (sorted by improvement)', fontsize=11)
    ax.set_ylabel('KL improvement (gated − always-VLM)', fontsize=11)
    ax.set_title(f'Per-rollout KL improvement at τ*={best_th:.2f}', fontsize=12, fontweight='bold')
    n_improved = sum(1 for v in sorted_vals if v > 0)
    ax.text(0.02, 0.95, f'{n_improved}/{n_rollouts} improved',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'Confidence-Gated VLM-PER Analysis — {condition_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save Quarto copy
    quarto_path = output_path.replace("figures/", "../../images/vlm_probe/")
    os.makedirs(os.path.dirname(quarto_path), exist_ok=True)
    plt.savefig(quarto_path, dpi=150, bbox_inches='tight')
    print(f"Quarto figure saved to {quarto_path}")
    plt.close()


def run_analysis():
    results = load_all_results()
    print(f"Loaded {len(results)} total predictions\n")

    all_metrics = {}

    for annotated in [True, False]:
        ann_label = "annotated" if annotated else "unannotated"
        print(f"\n{'='*60}")
        print(f"  {ann_label.upper()} condition")
        print(f"{'='*60}")

        rollout_data = build_rollout_model_matrix(
            results, task="reach-v3", K=8, annotated=annotated
        )
        print(f"Rollouts with ≥2 models: {len(rollout_data)}")
        if not rollout_data:
            print("  No multi-model rollouts found — skipping")
            continue

        # Show model coverage
        all_models = set()
        for rd in rollout_data:
            all_models.update(rd["predictions"].keys())
        print(f"Models: {sorted(all_models)}")

        # Agreement distribution
        agreements = [1.0 - rd["pred_std"] / 75.0 for rd in rollout_data]
        print(f"Agreement: mean={np.mean(agreements):.2f}, "
              f"std={np.std(agreements):.2f}, "
              f"min={min(agreements):.2f}, max={max(agreements):.2f}")

        # Agreement-error correlation
        corr_data = analyze_agreement_vs_error(rollout_data)
        print(f"\nAgreement-error correlations:")
        print(f"  vs mean error:   r={corr_data['corr_agreement_vs_mean_error']:.3f}")
        print(f"  vs median error: r={corr_data['corr_agreement_vs_median_error']:.3f}")
        print(f"  vs best error:   r={corr_data['corr_agreement_vs_best_error']:.3f}")

        # Threshold sweep
        print(f"\nThreshold sweep (median ensemble, σ=10):")
        sweep = threshold_sweep(rollout_data, sigma=10.0, pred_source="median")
        print(f"{'τ':>6s} {'VLM%':>6s} {'KL_gated':>10s} {'KL_unif':>10s} {'KL_alwVLM':>10s} {'Ov_gated':>10s} {'Ov_unif':>10s}")
        for r in sweep:
            if r["threshold"] % 0.1 < 0.01 or r["threshold"] > 0.99:
                print(f"{r['threshold']:6.2f} {r['vlm_fraction']:6.1%} "
                      f"{r['kl_gated']:10.3f} {r['kl_uniform']:10.3f} {r['kl_always_vlm']:10.3f} "
                      f"{r['overlap_gated']:10.1%} {r['overlap_uniform']:10.1%}")

        # Find optimal threshold (minimizes KL)
        best_kl = min(sweep, key=lambda x: x["kl_gated"])
        # Find threshold that maximizes overlap while KL < uniform
        kl_uniform = sweep[0]["kl_uniform"]  # all same
        valid_for_overlap = [r for r in sweep if r["kl_gated"] <= kl_uniform]
        if valid_for_overlap:
            best_overlap = max(valid_for_overlap, key=lambda x: x["overlap_gated"])
        else:
            best_overlap = max(sweep, key=lambda x: x["overlap_gated"])

        print(f"\nOptimal thresholds:")
        print(f"  Best KL:      τ={best_kl['threshold']:.2f} "
              f"(KL={best_kl['kl_gated']:.3f}, overlap={best_kl['overlap_gated']:.1%}, "
              f"VLM used={best_kl['vlm_fraction']:.0%})")
        print(f"  Best overlap:  τ={best_overlap['threshold']:.2f} "
              f"(KL={best_overlap['kl_gated']:.3f}, overlap={best_overlap['overlap_gated']:.1%}, "
              f"VLM used={best_overlap['vlm_fraction']:.0%})")

        # Baselines
        always_vlm = sweep[0]  # threshold=0 → always use VLM
        always_uniform = sweep[-1]  # threshold=1.0 → always uniform (approximately)
        print(f"\nBaselines:")
        print(f"  Always VLM:     KL={always_vlm['kl_always_vlm']:.3f}, overlap={always_vlm['overlap_always_vlm']:.1%}")
        print(f"  Always uniform: KL={always_uniform['kl_uniform']:.3f}, overlap={always_uniform['overlap_uniform']:.1%}")
        print(f"  Gated (best):   KL={best_kl['kl_gated']:.3f}, overlap={best_kl['overlap_gated']:.1%}")

        # KL improvement over always-VLM
        kl_improvement = always_vlm['kl_always_vlm'] - best_kl['kl_gated']
        print(f"\n  Gated KL improvement over always-VLM: {kl_improvement:+.3f} "
              f"({kl_improvement/always_vlm['kl_always_vlm']:.0%})")

        # Generate figure
        plot_confidence_gating(
            sweep, corr_data, rollout_data,
            output_path=f"figures/confidence_gating_{ann_label}.png",
            condition_label=f"reach-v3 K=8 {ann_label}"
        )

        all_metrics[ann_label] = {
            "n_rollouts": len(rollout_data),
            "n_models": len(all_models),
            "models": sorted(all_models),
            "agreement_mean": float(np.mean(agreements)),
            "agreement_std": float(np.std(agreements)),
            "corr_agreement_median_error": corr_data["corr_agreement_vs_median_error"],
            "best_kl_threshold": best_kl["threshold"],
            "best_kl": best_kl["kl_gated"],
            "best_overlap_threshold": best_overlap["threshold"],
            "best_overlap": best_overlap["overlap_gated"],
            "always_vlm_kl": always_vlm["kl_always_vlm"],
            "always_vlm_overlap": always_vlm["overlap_always_vlm"],
            "uniform_kl": always_uniform["kl_uniform"],
            "uniform_overlap": always_uniform["overlap_uniform"],
            "sweep": [{k: v for k, v in r.items()
                       if k not in ("kl_gated_values", "overlap_gated_values")}
                      for r in sweep],
        }

    return all_metrics


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    metrics = run_analysis()

    # Save metrics
    os.makedirs("results/confidence_gating", exist_ok=True)
    with open("results/confidence_gating/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to results/confidence_gating/metrics.json")
