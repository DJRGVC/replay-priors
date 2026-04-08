"""Convert VLM failure-timestep predictions into per-transition replay priorities.

Maps a predicted failure timestep t_f (+ confidence) into a priority vector
p_i for each transition i in the rollout, using a Gaussian kernel:

    p_i = c * exp(-(i - t_f)^2 / (2 * sigma^2))   +  (1 - c) * uniform

This directly implements the G_i formula from the literature review (§36,
GP-LRR Gaussian kernel) and connects VLM localization accuracy to
downstream replay-buffer utility.

Usage:
    python priority_score.py --results-dir results/k_sweep_reach \
        --data-dir data --task reach-v3 --sigma 10 --output figures/priority_comparison.png
"""

import argparse
import json
from pathlib import Path

import numpy as np


def vlm_priority(
    pred_timestep: int | None,
    confidence: float,
    total_steps: int,
    sigma: float = 10.0,
) -> np.ndarray:
    """Generate per-transition priority scores from a VLM prediction.

    Args:
        pred_timestep: VLM-predicted failure timestep (None = uniform fallback)
        confidence: VLM confidence [0, 1]
        total_steps: total transitions in the rollout
        sigma: Gaussian kernel width (controls how sharply priority peaks)

    Returns:
        priority vector of shape (total_steps,), sums to 1.
    """
    uniform = np.ones(total_steps) / total_steps

    if pred_timestep is None or confidence <= 0:
        return uniform

    t = np.arange(total_steps)
    gaussian = np.exp(-0.5 * ((t - pred_timestep) / sigma) ** 2)
    gaussian /= gaussian.sum() + 1e-12

    # Blend Gaussian and uniform by confidence
    priority = confidence * gaussian + (1 - confidence) * uniform
    priority /= priority.sum()
    return priority


def oracle_priority(
    gt_timestep: int,
    total_steps: int,
    sigma: float = 10.0,
) -> np.ndarray:
    """Ground-truth oracle priority (Gaussian centered on true failure)."""
    t = np.arange(total_steps)
    gaussian = np.exp(-0.5 * ((t - gt_timestep) / sigma) ** 2)
    gaussian /= gaussian.sum() + 1e-12
    return gaussian


def priority_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(oracle || vlm) — how much info is lost using VLM instead of oracle."""
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def priority_overlap(p: np.ndarray, q: np.ndarray, top_k_frac: float = 0.2) -> float:
    """Overlap of top-K% transitions between oracle and VLM priorities.

    Returns: fraction of oracle's top-K% transitions that also appear in VLM's top-K%.
    """
    k = max(1, int(len(p) * top_k_frac))
    oracle_top = set(np.argsort(p)[-k:])
    vlm_top = set(np.argsort(q)[-k:])
    return len(oracle_top & vlm_top) / k


def evaluate_priorities(
    results: list[dict],
    total_steps: int = 150,
    sigma: float = 10.0,
    top_k_frac: float = 0.2,
) -> dict:
    """Evaluate priority quality across a set of VLM predictions.

    Returns:
        Summary dict with mean KL divergence, top-K overlap, and
        comparison to uniform baseline.
    """
    kl_vlm, kl_uniform = [], []
    overlap_vlm, overlap_uniform = [], []

    uniform = np.ones(total_steps) / total_steps

    for r in results:
        gt = r["gt_failure_t"]
        pred = r.get("pred_failure_t")
        conf = r.get("pred_confidence", 0.5)

        oracle = oracle_priority(gt, total_steps, sigma)
        vlm = vlm_priority(pred, conf, total_steps, sigma)

        kl_vlm.append(priority_kl_divergence(oracle, vlm))
        kl_uniform.append(priority_kl_divergence(oracle, uniform))

        overlap_vlm.append(priority_overlap(oracle, vlm, top_k_frac))
        overlap_uniform.append(priority_overlap(oracle, uniform, top_k_frac))

    return {
        "n": len(results),
        "sigma": sigma,
        "top_k_frac": top_k_frac,
        "kl_vlm_mean": float(np.mean(kl_vlm)),
        "kl_vlm_std": float(np.std(kl_vlm)),
        "kl_uniform_mean": float(np.mean(kl_uniform)),
        "kl_uniform_std": float(np.std(kl_uniform)),
        "kl_improvement": float(np.mean(kl_uniform) - np.mean(kl_vlm)),
        "overlap_vlm_mean": float(np.mean(overlap_vlm)),
        "overlap_vlm_std": float(np.std(overlap_vlm)),
        "overlap_uniform_mean": float(np.mean(overlap_uniform)),
        "overlap_uniform_std": float(np.std(overlap_uniform)),
        "overlap_improvement": float(np.mean(overlap_vlm) - np.mean(overlap_uniform)),
    }


def plot_priority_comparison(
    results: list[dict],
    total_steps: int = 150,
    sigma: float = 10.0,
    output_path: str | Path | None = None,
    max_panels: int = 6,
):
    """Plot oracle vs VLM priority distributions for sample rollouts."""
    import matplotlib.pyplot as plt

    n = min(len(results), max_panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(total_steps)

    for ax, r in zip(axes, results[:n]):
        gt = r["gt_failure_t"]
        pred = r.get("pred_failure_t")
        conf = r.get("pred_confidence", 0.5)

        oracle = oracle_priority(gt, total_steps, sigma)
        vlm = vlm_priority(pred, conf, total_steps, sigma)
        uniform = np.ones(total_steps) / total_steps

        ax.fill_between(t, oracle, alpha=0.3, color="green", label="Oracle")
        ax.plot(t, oracle, color="green", linewidth=1.5)
        ax.fill_between(t, vlm, alpha=0.3, color="blue", label=f"VLM (pred={pred})")
        ax.plot(t, vlm, color="blue", linewidth=1.5)
        ax.axhline(uniform[0], color="gray", linestyle="--", alpha=0.5, label="Uniform")

        ax.axvline(gt, color="green", linestyle=":", alpha=0.8)
        if pred is not None:
            ax.axvline(pred, color="blue", linestyle=":", alpha=0.8)

        kl = priority_kl_divergence(oracle, vlm)
        overlap = priority_overlap(oracle, vlm)
        ax.set_title(
            f"{r['rollout']} — gt={gt}, pred={pred}, err={r.get('abs_error','?')}, "
            f"KL={kl:.2f}, top-20% overlap={overlap:.0%}",
            fontsize=9,
        )
        ax.set_ylabel("Priority")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(
        f"Oracle vs VLM Replay Priorities (σ={sigma})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()


def sigma_sweep(
    results: list[dict],
    total_steps: int = 150,
    sigmas: list[float] | None = None,
    output_path: str | Path | None = None,
):
    """Sweep sigma values and plot KL divergence + overlap vs sigma."""
    import matplotlib.pyplot as plt

    if sigmas is None:
        sigmas = [3, 5, 8, 10, 15, 20, 30, 50]

    stats = [evaluate_priorities(results, total_steps, s) for s in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # KL divergence
    kl_vlm = [s["kl_vlm_mean"] for s in stats]
    kl_uni = [s["kl_uniform_mean"] for s in stats]
    ax1.plot(sigmas, kl_vlm, "b-o", label="VLM priority")
    ax1.plot(sigmas, kl_uni, "r--s", label="Uniform baseline")
    ax1.set_xlabel("σ (kernel width)")
    ax1.set_ylabel("KL(oracle || priority)")
    ax1.set_title("KL Divergence (lower = better)")
    ax1.legend()
    ax1.set_xscale("log")

    # Top-K overlap
    ov_vlm = [s["overlap_vlm_mean"] for s in stats]
    ov_uni = [s["overlap_uniform_mean"] for s in stats]
    ax2.plot(sigmas, ov_vlm, "b-o", label="VLM priority")
    ax2.plot(sigmas, ov_uni, "r--s", label="Uniform baseline")
    ax2.set_xlabel("σ (kernel width)")
    ax2.set_ylabel("Top-20% Overlap")
    ax2.set_title("Priority Overlap (higher = better)")
    ax2.legend()
    ax2.set_xscale("log")

    fig.suptitle("Priority Quality vs Kernel Width", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="VLM → replay priority converter and evaluator")
    parser.add_argument("--results-file", type=str, required=True, help="Path to results.json")
    parser.add_argument("--total-steps", type=int, default=150)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument("--sigma-sweep", action="store_true", help="Run sigma sweep analysis")
    parser.add_argument("--model-filter", type=str, default=None, help="Filter results by model name")
    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    if args.model_filter:
        results = [r for r in results if args.model_filter in r.get("model", "")]

    # Filter to valid predictions only
    valid = [r for r in results if r.get("pred_failure_t") is not None]
    print(f"Loaded {len(results)} results ({len(valid)} with valid predictions)")

    if not valid:
        print("No valid predictions to evaluate.")
        return

    # Evaluate
    stats = evaluate_priorities(valid, args.total_steps, args.sigma)
    print(f"\n--- Priority Quality (σ={args.sigma}) ---")
    print(f"  KL(oracle||VLM):     {stats['kl_vlm_mean']:.3f} ± {stats['kl_vlm_std']:.3f}")
    print(f"  KL(oracle||uniform): {stats['kl_uniform_mean']:.3f} ± {stats['kl_uniform_std']:.3f}")
    print(f"  KL improvement:      {stats['kl_improvement']:.3f} ({stats['kl_improvement']/stats['kl_uniform_mean']:.0%} of uniform gap)")
    print(f"  Top-20% overlap (VLM):     {stats['overlap_vlm_mean']:.1%} ± {stats['overlap_vlm_std']:.1%}")
    print(f"  Top-20% overlap (uniform): {stats['overlap_uniform_mean']:.1%} ± {stats['overlap_uniform_std']:.1%}")
    print(f"  Overlap improvement:       {stats['overlap_improvement']:+.1%}")

    if args.sigma_sweep:
        sweep_out = args.output.replace(".png", "_sigma_sweep.png") if args.output else None
        sigma_sweep(valid, args.total_steps, output_path=sweep_out)

    if args.output:
        plot_priority_comparison(valid, args.total_steps, args.sigma, args.output)


if __name__ == "__main__":
    main()
