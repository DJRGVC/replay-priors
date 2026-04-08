"""Plot TD-error vs oracle-advantage correlation over training.

Reads snapshot .npz files from completed runs and produces the central figure:
  x = env steps, y = Spearman(|TD|, oracle_advantage), one line per task.

Usage:
    python plot_td_correlation.py --run-dirs snapshots/reach-v3_s42 snapshots/pick-place-v3_s42
    python plot_td_correlation.py --from-modal  # downloads from Modal volume first
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np


def load_snapshots_from_dir(snapshot_dir: str):
    """Load snapshot .npz files from a local directory."""
    pattern = os.path.join(snapshot_dir, "td_snapshots", "snapshot_*.npz")
    files = sorted(glob(pattern))
    snapshots = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        snapshots.append(data)
    return snapshots


def extract_correlation_series(snapshots):
    """Extract step, Spearman, Pearson series from snapshots."""
    steps, spearman, pearson = [], [], []
    for snap in snapshots:
        step = int(snap["step"])
        steps.append(step)
        spearman.append(float(snap.get("td_dense_spearman", 0.0)))
        pearson.append(float(snap.get("td_dense_pearson", 0.0)))
    return np.array(steps), np.array(spearman), np.array(pearson)


def download_from_modal():
    """Download snapshot files from Modal volume to local snapshots/ dir."""
    import subprocess
    local_base = Path(__file__).parent / "snapshots"

    for task, seed in [("reach-v3", 42), ("pick-place-v3", 42)]:
        local_dir = local_base / f"{task}_s{seed}"
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {task}_s{seed} from Modal volume...")
        result = subprocess.run(
            ["modal", "volume", "get", "td-error-baseline-results",
             f"{task}_s{seed}/", str(local_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Warning: {result.stderr.strip()}")
        else:
            print(f"  Downloaded to {local_dir}")

    return str(local_base)


def plot_figure(task_data: dict, output_path: str):
    """Generate the correlation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = {"reach-v3": "#2196F3", "pick-place-v3": "#F44336"}
    markers = {"reach-v3": "o", "pick-place-v3": "s"}

    # Left panel: Spearman correlation over training
    ax = axes[0]
    for task, (steps, spearman, _pearson) in task_data.items():
        ax.plot(steps, spearman, color=colors.get(task, "gray"),
                marker=markers.get(task, "o"), markersize=5,
                linewidth=1.5, label=task)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Spearman Correlation\n|TD-error| vs Oracle Advantage")
    ax.set_title("TD-Error as Priority Signal")
    ax.legend(loc="best")
    ax.set_ylim(-0.3, 0.3)
    ax.grid(alpha=0.3)

    # Right panel: Pearson correlation
    ax = axes[1]
    for task, (steps, _spearman, pearson) in task_data.items():
        ax.plot(steps, pearson, color=colors.get(task, "gray"),
                marker=markers.get(task, "o"), markersize=5,
                linewidth=1.5, label=task)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_title("Pearson Correlation")
    ax.legend(loc="best")
    ax.set_ylim(-0.3, 0.3)
    ax.grid(alpha=0.3)

    fig.suptitle("How (Un)informative Is TD-Error PER in Early Training?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="+", default=None,
                   help="Local dirs containing td_snapshots/")
    p.add_argument("--from-modal", action="store_true",
                   help="Download results from Modal volume first")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    output_path = args.output or str(fig_dir / "td_correlation_over_training.png")

    if args.from_modal:
        base = download_from_modal()
        run_dirs = sorted(glob(os.path.join(base, "*_s*")))
    elif args.run_dirs:
        run_dirs = args.run_dirs
    else:
        # Default: look in local snapshots/
        base = str(Path(__file__).parent / "snapshots")
        run_dirs = sorted(glob(os.path.join(base, "*_s*")))

    if not run_dirs:
        print("No run directories found. Use --from-modal or --run-dirs.")
        sys.exit(1)

    task_data = {}
    for run_dir in run_dirs:
        task = os.path.basename(run_dir).split("_s")[0]
        snapshots = load_snapshots_from_dir(run_dir)
        if not snapshots:
            print(f"No snapshots in {run_dir}, skipping")
            continue
        steps, spearman, pearson = extract_correlation_series(snapshots)
        task_data[task] = (steps, spearman, pearson)
        print(f"{task}: {len(snapshots)} snapshots, "
              f"spearman range [{spearman.min():.3f}, {spearman.max():.3f}]")

    if not task_data:
        print("No data to plot.")
        sys.exit(1)

    plot_figure(task_data, output_path)

    # Also dump raw data as JSON for reproducibility
    json_path = output_path.replace(".png", ".json")
    json_data = {}
    for task, (steps, spearman, pearson) in task_data.items():
        json_data[task] = {
            "steps": steps.tolist(),
            "spearman": spearman.tolist(),
            "pearson": pearson.tolist(),
        }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
