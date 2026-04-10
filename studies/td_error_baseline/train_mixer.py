"""SAC training with Adaptive Priority Mixer.

Trains SAC with the regime-aware adaptive prioritization buffer and logs
regime transitions, priority distributions, and comparison metrics.

Usage:
    python train_mixer.py --task reach-v3 --total-steps 100000 --seed 42
    python train_mixer.py --task reach-v3 --total-steps 100000 --seed 42 --mode uniform
    python train_mixer.py --task reach-v3 --total-steps 100000 --seed 42 --mode td-per
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_priority_mixer import AdaptivePriorityMixer, REGIME_ALIGNED
from dense_reward_buffer import DenseRewardReplayBuffer
from metaworld_env import make_env
from td_instrumenter import TDInstrumentCallback


class MixerInstrumentCallback(TDInstrumentCallback):
    """Extended callback that also updates the mixer's regime detector
    and logs regime statistics alongside TD-error snapshots."""

    def __init__(self, *args, mode: str = "adaptive", **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.regime_transitions: list[dict] = []
        self._prev_regime = None

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Update regime detector every 100 steps (if using mixer buffer)
        model = self.model
        buffer = model.replay_buffer
        if (
            hasattr(buffer, 'update_regime')
            and self.num_timesteps % 100 == 0
            and buffer.size() >= 256
        ):
            # Quick critic eval on a small batch for regime detection
            with torch.no_grad():
                replay_data = buffer._get_samples(
                    np.random.randint(0, buffer.size(), size=256)
                )
                q1, q2 = model.critic(replay_data.observations, replay_data.actions)
                next_actions, next_log_prob = model.actor.action_log_prob(
                    replay_data.next_observations
                )
                q1_next, q2_next = model.critic_target(
                    replay_data.next_observations, next_actions
                )
                ent_coef = torch.exp(model.log_ent_coef).detach()
                q_next = torch.min(q1_next, q2_next) - ent_coef * next_log_prob.unsqueeze(-1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * model.gamma * q_next

                q_vals = ((q1 + q2) / 2.0).cpu().numpy().flatten()
                td1 = (q1 - target_q).cpu().numpy().flatten()
                td2 = (q2 - target_q).cpu().numpy().flatten()
                abs_td = (np.abs(td1) + np.abs(td2)) / 2.0

            buffer.update_regime(q_vals, abs_td)
            # Note: priorities are now updated in PERSAC.train() after each gradient step

            # Log regime transitions
            current = buffer.current_regime
            if current != self._prev_regime:
                self.regime_transitions.append({
                    "step": self.num_timesteps,
                    "from": self._prev_regime,
                    "to": current,
                })
                self._prev_regime = current

        return result

    def _take_snapshot(self, step: int):
        """Extended snapshot that includes regime stats."""
        super()._take_snapshot(step)

        # Add regime info to the latest snapshot
        buffer = self.model.replay_buffer
        if hasattr(buffer, 'get_regime_stats'):
            stats = buffer.get_regime_stats()
            snap_dir = os.path.join(self.output_dir, "td_snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            regime_path = os.path.join(snap_dir, f"regime_{step:08d}.json")
            with open(regime_path, "w") as f:
                json.dump({
                    **stats,
                    "mode": self.mode,
                    "regime_transitions": self.regime_transitions[-10:],
                }, f, indent=2, default=str)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="reach-v3")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="adaptive",
                   choices=["adaptive", "td-per", "uniform", "rpe-per", "rnd-per"],
                   help="Prioritization mode: adaptive (regime-switching), "
                        "td-per (always TD priorities), rpe-per (reward prediction error), "
                        "rnd-per (random network distillation novelty), "
                        "uniform (standard buffer)")
    p.add_argument("--snapshot-interval", type=int, default=10_000)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta0", type=float, default=0.4)
    p.add_argument("--checkpoint-interval", type=int, default=25_000)
    return p.parse_args()


def main():
    args = parse_args()

    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from per_sac import PERSAC
    from rpe_sac import RPESAC
    from rnd_sac import RNDSAC

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).parent / "snapshots" / f"{args.task}_s{args.seed}_{args.mode}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[train_mixer] task={args.task} steps={args.total_steps} "
          f"seed={args.seed} mode={args.mode}")

    env = make_env(args.task, seed=args.seed, sparse=True)

    # Choose buffer based on mode
    if args.mode == "uniform":
        buffer_cls = DenseRewardReplayBuffer
        buffer_kwargs = {}
    else:
        buffer_cls = AdaptivePriorityMixer
        buffer_kwargs = {
            "alpha": args.per_alpha,
            "beta0": args.per_beta0,
        }

    # Use RPESAC for rpe-per (reward prediction error priorities),
    # PERSAC for td-per/adaptive (TD-error priorities),
    # vanilla SAC for uniform mode (no PER needed).
    from stable_baselines3 import SAC as VanillaSAC
    if args.mode == "uniform":
        sac_cls = VanillaSAC
    elif args.mode == "rpe-per":
        sac_cls = RPESAC
    elif args.mode == "rnd-per":
        sac_cls = RNDSAC
    else:
        sac_cls = PERSAC

    model = sac_cls(
        "MlpPolicy",
        env,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        seed=args.seed,
        verbose=1,
        device="auto",
        replay_buffer_class=buffer_cls,
        replay_buffer_kwargs=buffer_kwargs,
    )

    # Callbacks
    td_callback = MixerInstrumentCallback(
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        n_samples=min(5000, args.buffer_size),
        mode=args.mode,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="sac",
    )

    print(f"[train_mixer] Starting training...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=CallbackList([td_callback, ckpt_callback]),
        log_interval=10,
    )

    model.save(os.path.join(args.output_dir, "final_model"))

    summary = {
        "task": args.task,
        "total_steps": model.num_timesteps,
        "seed": args.seed,
        "mode": args.mode,
        "snapshots": td_callback.snapshot_count,
        "regime_transitions": td_callback.regime_transitions,
    }
    if hasattr(model.replay_buffer, 'get_regime_stats'):
        summary["final_regime_stats"] = model.replay_buffer.get_regime_stats()

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    env.close()
    print("[train_mixer] Done.")


if __name__ == "__main__":
    main()
