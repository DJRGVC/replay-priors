"""Modal app for running SAC + TD-error analysis on MetaWorld tasks.

Launches remote training runs on Modal with GPU, saves snapshots + checkpoints
to a Modal Volume for retrieval.

Usage:
    # Run both tasks (reach-v3 + pick-place-v3) at 100k steps
    modal run studies/td_error_baseline/modal_app.py

    # Run a single task
    modal run studies/td_error_baseline/modal_app.py::train_task \
        --task reach-v3 --total-steps 100000 --seed 42
"""

from pathlib import Path

import modal

app = modal.App("td-error-baseline")

# Persistent volume for results
vol = modal.Volume.from_name("td-error-baseline-results", create_if_missing=True)
RESULTS_DIR = "/results"
STUDY_DIR = "/root/study"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libosmesa6", "libglew-dev", "patchelf", "git")
    .pip_install(
        "torch==2.7.0",
        "stable-baselines3==2.7.1",
        "gymnasium==1.2.1",
        "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@master",
        "mujoco==3.6.0",
        "scipy==1.15.3",
        "numpy<2",
    )
    .env({"MUJOCO_GL": "osmesa"})
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path=STUDY_DIR,
        ignore=["modal_app.py", "__pycache__", "*.pyc", "snapshots", "figures"],
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,  # 90 min
    volumes={RESULTS_DIR: vol},
)
def train_task(task: str, total_steps: int = 100_000, seed: int = 42,
               snapshot_interval: int = 10_000):
    """Run SAC training on a single MetaWorld task with TD-error instrumentation."""
    import json
    import os
    import sys
    import time

    import numpy as np

    sys.path.insert(0, STUDY_DIR)

    from dense_reward_buffer import DenseRewardReplayBuffer
    from metaworld_env import make_env
    from td_instrumenter import TDInstrumentCallback

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    output_dir = os.path.join(RESULTS_DIR, f"{task}_s{seed}")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "task": task,
        "total_steps": total_steps,
        "seed": seed,
        "snapshot_interval": snapshot_interval,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[modal-train] task={task} steps={total_steps} seed={seed}")
    env = make_env(task, seed=seed, sparse=True)

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=1_000,
        seed=seed,
        verbose=1,
        device="auto",
        replay_buffer_class=DenseRewardReplayBuffer,
    )

    td_callback = TDInstrumentCallback(
        snapshot_interval=snapshot_interval,
        output_dir=output_dir,
        n_samples=5000,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="sac",
    )
    callbacks = CallbackList([td_callback, ckpt_callback])

    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callbacks, log_interval=10)
    elapsed = time.time() - t0

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)

    summary = {
        "task": task,
        "total_steps": model.num_timesteps,
        "seed": seed,
        "snapshots": td_callback.snapshot_count,
        "wall_time_seconds": elapsed,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    env.close()

    print(f"[modal-train] Done. {task} took {elapsed:.0f}s, {td_callback.snapshot_count} snapshots")
    return summary


@app.function(
    image=image,
    volumes={RESULTS_DIR: vol},
    timeout=300,
)
def download_snapshot_data(task: str, seed: int = 42):
    """Read snapshot data from the volume and return it."""
    import os
    import numpy as np

    vol.reload()
    snap_dir = os.path.join(RESULTS_DIR, f"{task}_s{seed}", "td_snapshots")
    if not os.path.exists(snap_dir):
        return {"error": f"No snapshots found at {snap_dir}"}

    results = []
    for fname in sorted(os.listdir(snap_dir)):
        if fname.endswith(".npz"):
            data = dict(np.load(os.path.join(snap_dir, fname), allow_pickle=True))
            entry = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        entry[k] = float(v)
                    elif v.size < 100:
                        entry[k] = v.tolist()
                    else:
                        entry[k + "_mean"] = float(np.mean(v))
                        entry[k + "_std"] = float(np.std(v))
                        entry[k + "_median"] = float(np.median(v))
                else:
                    entry[k] = v
            results.append(entry)

    return results


@app.function(
    image=image,
    gpu="T4",
    timeout=5400,  # 90 min
    volumes={RESULTS_DIR: vol},
)
def train_mixer_task(task: str, total_steps: int = 100_000, seed: int = 42,
                     mode: str = "adaptive", snapshot_interval: int = 10_000):
    """Run SAC training with Adaptive Priority Mixer on Modal."""
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, STUDY_DIR)

    from adaptive_priority_mixer import AdaptivePriorityMixer
    from dense_reward_buffer import DenseRewardReplayBuffer
    from metaworld_env import make_env
    from train_mixer import MixerInstrumentCallback

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    output_dir = os.path.join(RESULTS_DIR, f"{task}_s{seed}_{mode}")
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "task": task,
        "total_steps": total_steps,
        "seed": seed,
        "mode": mode,
        "snapshot_interval": snapshot_interval,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[modal-mixer] task={task} steps={total_steps} seed={seed} mode={mode}")
    env = make_env(task, seed=seed, sparse=True)

    # Choose buffer based on mode
    if mode == "uniform":
        buffer_cls = DenseRewardReplayBuffer
        buffer_kwargs = {}
    else:
        buffer_cls = AdaptivePriorityMixer
        buffer_kwargs = {"alpha": 0.6, "beta0": 0.4}

    model = SAC(
        "MlpPolicy",
        env,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=1_000,
        seed=seed,
        verbose=1,
        device="auto",
        replay_buffer_class=buffer_cls,
        replay_buffer_kwargs=buffer_kwargs,
    )

    td_callback = MixerInstrumentCallback(
        snapshot_interval=snapshot_interval,
        output_dir=output_dir,
        n_samples=5000,
        mode=mode,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="sac",
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([td_callback, ckpt_callback]),
        log_interval=10,
    )
    elapsed = time.time() - t0

    model.save(os.path.join(output_dir, "final_model"))

    summary = {
        "task": task,
        "total_steps": model.num_timesteps,
        "seed": seed,
        "mode": mode,
        "snapshots": td_callback.snapshot_count,
        "wall_time_seconds": elapsed,
        "regime_transitions": td_callback.regime_transitions,
    }
    if hasattr(model.replay_buffer, 'get_regime_stats'):
        summary["final_regime_stats"] = model.replay_buffer.get_regime_stats()

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    vol.commit()
    env.close()

    print(f"[modal-mixer] Done. {task} {mode} took {elapsed:.0f}s, {td_callback.snapshot_count} snapshots")
    return summary


@app.local_entrypoint()
def main(seeds: str = "42", tasks: str = "reach-v3,pick-place-v3",
         total_steps: int = 100_000, modes: str = "",
         compare: bool = False):
    """Run tasks in parallel on Modal.

    Args:
        seeds: Comma-separated seeds (e.g. "42" or "42,123")
        tasks: Comma-separated task names
        total_steps: Steps per run
        modes: Comma-separated modes for mixer comparison (e.g. "adaptive,td-per,uniform")
        compare: If True, run all 3 modes for comparison
    """
    import json

    task_list = [t.strip() for t in tasks.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    if compare:
        mode_list = ["adaptive", "td-per", "uniform"]
    elif modes:
        mode_list = [m.strip() for m in modes.split(",")]
    else:
        mode_list = []

    if mode_list:
        # Mixer comparison mode
        handles = []
        for task in task_list:
            for seed in seed_list:
                for mode in mode_list:
                    print(f"Launching {task} mode={mode} ({total_steps} steps, seed={seed})...")
                    handles.append(train_mixer_task.spawn(
                        task=task, total_steps=total_steps, seed=seed, mode=mode
                    ))

        for h in handles:
            result = h.get()
            print(f"\n=== {result['task']} mode={result['mode']} (seed={result['seed']}) ===")
            print(json.dumps(result, indent=2))
    else:
        # Original training mode (no mixer)
        handles = []
        for task in task_list:
            for seed in seed_list:
                print(f"Launching {task} ({total_steps} steps, seed={seed})...")
                handles.append(train_task.spawn(task=task, total_steps=total_steps, seed=seed))

        for h in handles:
            result = h.get()
            print(f"\n=== {result['task']} (seed={result['seed']}) ===")
            print(json.dumps(result, indent=2))
