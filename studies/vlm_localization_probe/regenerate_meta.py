"""Regenerate meta.json files using updated failure detection heuristics.

Reads existing proprio.npy and meta.json, re-runs detect_failure_timestep,
and overwrites meta.json with updated failure info.
"""
import json
from pathlib import Path
import numpy as np
from collect_rollouts import detect_failure_timestep

data_dir = Path(__file__).parent / "data"

for task_dir in sorted(data_dir.iterdir()):
    if not task_dir.is_dir() or task_dir.name.startswith("."):
        continue
    task_name = task_dir.name
    print(f"\n=== {task_name} ===")
    for rollout_dir in sorted(task_dir.iterdir()):
        if not rollout_dir.is_dir():
            continue
        meta_path = rollout_dir / "meta.json"
        proprio_path = rollout_dir / "proprio.npy"
        if not meta_path.exists() or not proprio_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        proprio = np.load(proprio_path)
        failure_info = detect_failure_timestep(task_name, proprio, meta["success"])

        meta.update(failure_info)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {rollout_dir.name}: failure_t={failure_info['failure_timestep']} type={failure_info['failure_type']}")
