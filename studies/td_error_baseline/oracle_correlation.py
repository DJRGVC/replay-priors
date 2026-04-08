"""Compute correlation between |TD-error| and dense-reward oracle advantage.

For each training checkpoint, we:
1. Load the critic and replay buffer snapshot
2. Sample transitions from the buffer
3. Compute |TD-error| using the current critic
4. Compute oracle advantage using the dense (shaped) MetaWorld reward
5. Report rank correlation (Spearman) between |TD| and oracle advantage

This is the central quantitative claim: TD-error should be poorly correlated
with true advantage early in training, improving as the critic learns.
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from scipy import stats


def load_snapshots(snapshot_dir: str):
    """Load all TD-error snapshots from a run."""
    pattern = os.path.join(snapshot_dir, "td_snapshots", "snapshot_*.npz")
    files = sorted(glob(pattern))
    snapshots = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        snapshots.append(data)
    return snapshots


def compute_priority_quality_metrics(abs_td: np.ndarray, rewards: np.ndarray):
    """Compute metrics of how well |TD| priorities align with reward signal.

    Returns dict with:
    - spearman_r: Spearman rank correlation between |TD| and reward
    - top_k_overlap: fraction of top-10% by |TD| that are also top-10% by reward
    - priority_concentration: Gini coefficient of |TD| priorities
    """
    n = len(abs_td)

    # Spearman correlation
    if np.std(abs_td) < 1e-10 or np.std(rewards) < 1e-10:
        spearman_r = 0.0
        spearman_p = 1.0
    else:
        spearman_r, spearman_p = stats.spearmanr(abs_td, rewards)

    # Top-K overlap: do high-|TD| transitions also have high reward?
    k = max(1, n // 10)
    top_td_idx = set(np.argsort(abs_td)[-k:])
    top_reward_idx = set(np.argsort(rewards)[-k:])
    top_k_overlap = len(top_td_idx & top_reward_idx) / k

    # Gini coefficient of |TD| priorities (how concentrated is sampling?)
    sorted_td = np.sort(abs_td)
    cumsum = np.cumsum(sorted_td)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    gini = 1.0 - 2.0 * np.sum(cumsum) / (n * total)

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "top_k_overlap": float(top_k_overlap),
        "priority_gini": float(gini),
    }


def analyze_run(snapshot_dir: str):
    """Analyze all snapshots from a single run."""
    snapshots = load_snapshots(snapshot_dir)
    if not snapshots:
        print(f"No snapshots found in {snapshot_dir}")
        return []

    results = []
    for snap in snapshots:
        step = int(snap["step"])
        abs_td = snap["abs_td_errors"]
        # Use oracle_advantage (dense-reward derived) if available, else fall back to sparse
        if "oracle_advantage" in snap:
            oracle = snap["oracle_advantage"]
        else:
            oracle = snap["sparse_rewards"]

        metrics = compute_priority_quality_metrics(abs_td, oracle)
        metrics["step"] = step
        metrics["abs_td_mean"] = float(snap["abs_td_mean"])
        metrics["abs_td_std"] = float(snap["abs_td_std"])
        metrics["q_mean"] = float(snap["q_mean"])
        metrics["q_std"] = float(snap["q_std"])
        metrics["buffer_size"] = int(snap["buffer_size"])

        # Episode stats if available
        for key in ["episode_return_mean", "success_rate", "episode_dense_return_mean"]:
            if key in snap:
                metrics[key] = float(snap[key])

        results.append(metrics)
        print(
            f"  step={step:>8d}: spearman_r={metrics['spearman_r']:+.4f} "
            f"top10_overlap={metrics['top_k_overlap']:.3f} "
            f"gini={metrics['priority_gini']:.3f} "
            f"|TD|={metrics['abs_td_mean']:.4f}"
        )

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="Directories containing td_snapshots/")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    all_results = {}
    for run_dir in args.run_dirs:
        task = os.path.basename(run_dir).split("_s")[0]
        print(f"\n=== {task} ({run_dir}) ===")
        results = analyze_run(run_dir)
        all_results[run_dir] = {"task": task, "snapshots": results}

    if args.output:
        # Convert numpy types for JSON serialization
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
