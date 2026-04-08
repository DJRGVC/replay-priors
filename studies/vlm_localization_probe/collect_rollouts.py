"""Collect failure rollouts from MetaWorld tasks using a random policy.

Saves per-rollout:
  - frames/  : downsampled 224x224 RGB PNGs at every timestep
  - proprio.npy : (T, obs_dim) array of proprioceptive observations
  - meta.json   : task name, num_steps, success flag, failure_timestep (if determinable)

Usage:
    python collect_rollouts.py --tasks reach-v3 push-v3 pick-place-v3 \
        --rollouts-per-task 20 --max-steps 150 --outdir data
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def detect_failure_timestep(task_name: str, proprio: np.ndarray, success: bool) -> dict:
    """Heuristic failure-timestep detection.

    For failed episodes, we identify the timestep where the agent most clearly
    diverges from the goal. This is task-specific.

    Returns dict with:
      - failure_timestep: int or None
      - failure_type: str description
      - ambiguous: bool
    """
    if success:
        return {"failure_timestep": None, "failure_type": "success", "ambiguous": False}

    T = len(proprio)

    if "reach" in task_name:
        # proprio[t, 0:3] = hand position, proprio[t, 36:39] = goal position (in reach-v3)
        # Failure = never got close enough. "Failure timestep" = when hand started
        # moving away from the closest approach.
        hand = proprio[:, 0:3]
        goal = proprio[0, 36:39]  # goal is static
        dists = np.linalg.norm(hand - goal, axis=1)
        closest_t = int(np.argmin(dists))
        return {
            "failure_timestep": closest_t,
            "failure_type": "closest_approach_then_diverged",
            "ambiguous": False,
        }

    if "push" in task_name or "pick-place" in task_name:
        # Hand position = proprio[:, 0:3], object position = proprio[:, 4:7]
        # With a random policy the hand often never contacts the object.
        # Two failure modes:
        #   1. Hand contacted object but lost it → failure = last contact timestep
        #   2. Hand never contacted object → failure = closest approach (the moment
        #      where the agent "should have" engaged but didn't)
        hand = proprio[:, 0:3]
        obj = proprio[:, 4:7]
        hand_obj_dist = np.linalg.norm(hand - obj, axis=1)
        near_mask = hand_obj_dist < 0.08
        if near_mask.any():
            last_contact = int(np.where(near_mask)[0][-1])
            return {
                "failure_timestep": last_contact,
                "failure_type": "lost_contact_with_object",
                "ambiguous": True,
            }
        else:
            closest_t = int(np.argmin(hand_obj_dist))
            return {
                "failure_timestep": closest_t,
                "failure_type": "never_contacted_object_closest_approach",
                "ambiguous": True,
            }

    return {"failure_timestep": T // 2, "failure_type": "unknown", "ambiguous": True}


def collect_single_rollout(env, task, max_steps: int, save_dir: Path, render_size: int = 224):
    """Run one episode, save frames + proprio + metadata."""
    env.set_task(task)
    obs, _ = env.reset()

    frames_dir = save_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    proprio_list = [obs.copy()]
    success = False
    truncated = False
    t = 0

    for t in range(max_steps):
        # Random policy
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        proprio_list.append(obs.copy())

        # Render and save frame
        frame = env.render()
        img = Image.fromarray(frame)
        if img.size != (render_size, render_size):
            img = img.resize((render_size, render_size), Image.LANCZOS)
        img.save(frames_dir / f"{t:04d}.png")

        if info.get("success", False):
            success = True

        if terminated:
            break

    proprio = np.array(proprio_list)
    np.save(save_dir / "proprio.npy", proprio)

    return proprio, success, t + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["reach-v3", "push-v3", "pick-place-v3"])
    parser.add_argument("--rollouts-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import metaworld

    base_dir = Path(__file__).parent / args.outdir

    for task_name in args.tasks:
        print(f"\n=== Collecting rollouts for {task_name} ===")
        ml1 = metaworld.ML1(task_name)
        env_cls = ml1.train_classes[task_name]
        env = env_cls(render_mode="rgb_array")

        tasks = ml1.train_tasks
        np.random.seed(args.seed)

        task_dir = base_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        collected = 0
        successes = 0
        failures = 0
        attempt = 0

        # We want ~rollouts_per_task failures. Collect up to 2x attempts.
        while failures < args.rollouts_per_task and attempt < args.rollouts_per_task * 3:
            task = tasks[attempt % len(tasks)]
            rollout_dir = task_dir / f"rollout_{collected:03d}"

            proprio, success, num_steps = collect_single_rollout(
                env, task, args.max_steps, rollout_dir
            )

            failure_info = detect_failure_timestep(task_name, proprio, success)

            meta = {
                "task_name": task_name,
                "rollout_idx": collected,
                "num_steps": num_steps,
                "success": success,
                "attempt_idx": attempt,
                **failure_info,
            }
            with open(rollout_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            if success:
                successes += 1
            else:
                failures += 1

            collected += 1
            attempt += 1

            status = "SUCCESS" if success else f"FAIL@t={failure_info['failure_timestep']}"
            print(f"  rollout {collected:3d} | {status} | steps={num_steps}")

        env.close()
        print(f"  → {task_name}: {failures} failures, {successes} successes out of {collected} total")

    # Write a .gitignore for data/
    gitignore_path = base_dir / ".gitignore"
    gitignore_path.write_text("# Large binary data — do not commit\n*\n!.gitignore\n")

    print("\nDone. Data saved to", base_dir)


if __name__ == "__main__":
    main()
