"""Callback to instrument SB3 SAC critic with TD-error analysis.

At regular intervals, samples transitions from the replay buffer, computes
|TD-error| for each, and correlates with the dense-reward oracle signal.
Saves snapshots as .npz files for later analysis.
"""

import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class TDInstrumentCallback(BaseCallback):
    """Snapshot TD-error distributions and oracle advantage correlation."""

    def __init__(
        self,
        snapshot_interval: int = 10_000,
        output_dir: str = "snapshots",
        n_samples: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.snapshot_interval = snapshot_interval
        self.output_dir = output_dir
        self.n_samples = n_samples
        self.snapshot_count = 0
        self._last_snapshot_step = 0

        # Track episode returns for logging
        self._episode_rewards = []
        self._episode_dense_rewards = []
        self._current_episode_reward = 0.0
        self._current_episode_dense_reward = 0.0
        self._episode_successes = []
        self._current_episode_success = False

    def _on_step(self) -> bool:
        # Track episode stats
        info = self.locals.get("infos", [{}])
        if isinstance(info, list) and len(info) > 0:
            info = info[0]

        reward = self.locals.get("rewards", [0.0])
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[0]) if len(reward) > 0 else 0.0

        self._current_episode_reward += reward
        dense_r = info.get("dense_reward", reward)
        self._current_episode_dense_reward += dense_r
        if info.get("success", False):
            self._current_episode_success = True

        done = self.locals.get("dones", [False])
        if isinstance(done, (list, np.ndarray)):
            done = bool(done[0]) if len(done) > 0 else False

        if done:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_dense_rewards.append(self._current_episode_dense_reward)
            self._episode_successes.append(float(self._current_episode_success))
            self._current_episode_reward = 0.0
            self._current_episode_dense_reward = 0.0
            self._current_episode_success = False

        # Snapshot at intervals
        step = self.num_timesteps
        if step - self._last_snapshot_step >= self.snapshot_interval:
            self._take_snapshot(step)
            self._last_snapshot_step = step

        return True

    def _take_snapshot(self, step: int):
        """Compute TD-errors and correlate with dense reward oracle."""
        from scipy import stats as sp_stats

        model = self.model
        buffer = model.replay_buffer

        if buffer.size() < self.n_samples:
            if self.verbose:
                print(f"[snapshot] step={step}: buffer too small ({buffer.size()}), skipping")
            return

        # Sample — use dense-reward-aware buffer if available
        has_dense = hasattr(buffer, "sample_with_dense")
        if has_dense:
            replay_data, dense_rewards_np = buffer.sample_with_dense(self.n_samples)
        else:
            replay_data = buffer.sample(self.n_samples)
            dense_rewards_np = None

        # Compute TD-errors using the critic
        with torch.no_grad():
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            q1_current, q2_current = model.critic(obs, actions)
            next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
            q1_next, q2_next = model.critic_target(next_obs, next_actions)
            ent_coef = torch.exp(model.log_ent_coef).detach()
            q_next = torch.min(q1_next, q2_next) - ent_coef * next_log_prob.unsqueeze(-1)
            target_q = rewards + (1 - dones) * model.gamma * q_next

            td_error_1 = (q1_current - target_q).cpu().numpy().flatten()
            td_error_2 = (q2_current - target_q).cpu().numpy().flatten()
            abs_td = (np.abs(td_error_1) + np.abs(td_error_2)) / 2.0

            sparse_rewards_np = rewards.cpu().numpy().flatten()
            q_values = ((q1_current + q2_current) / 2.0).cpu().numpy().flatten()

        # Oracle correlation: |TD| vs dense-reward advantage proxy
        td_dense_pearson = None
        td_dense_spearman = None
        if dense_rewards_np is not None:
            oracle_adv = dense_rewards_np - np.mean(dense_rewards_np)
            if np.std(abs_td) > 1e-10 and np.std(oracle_adv) > 1e-10:
                td_dense_pearson = float(np.corrcoef(abs_td, oracle_adv)[0, 1])
            if len(abs_td) > 2:
                td_dense_spearman = float(sp_stats.spearmanr(abs_td, oracle_adv).correlation)

        snapshot = {
            "step": step,
            "abs_td_errors": abs_td,
            "td_error_1": td_error_1,
            "td_error_2": td_error_2,
            "q_values": q_values,
            "sparse_rewards": sparse_rewards_np,
            "abs_td_mean": float(np.mean(abs_td)),
            "abs_td_std": float(np.std(abs_td)),
            "abs_td_median": float(np.median(abs_td)),
            "abs_td_p90": float(np.percentile(abs_td, 90)),
            "abs_td_p99": float(np.percentile(abs_td, 99)),
            "q_mean": float(np.mean(q_values)),
            "q_std": float(np.std(q_values)),
            "buffer_size": buffer.size(),
        }

        if dense_rewards_np is not None:
            snapshot["dense_rewards"] = dense_rewards_np
            snapshot["oracle_advantage"] = dense_rewards_np - np.mean(dense_rewards_np)
        if td_dense_pearson is not None:
            snapshot["td_dense_pearson"] = td_dense_pearson
        if td_dense_spearman is not None:
            snapshot["td_dense_spearman"] = td_dense_spearman

        # Episode stats since last snapshot
        if self._episode_rewards:
            snapshot["episode_return_mean"] = float(np.mean(self._episode_rewards))
            snapshot["episode_return_std"] = float(np.std(self._episode_rewards))
            snapshot["episode_dense_return_mean"] = float(np.mean(self._episode_dense_rewards))
            snapshot["success_rate"] = float(np.mean(self._episode_successes))
            snapshot["n_episodes"] = len(self._episode_rewards)
            self._episode_rewards.clear()
            self._episode_dense_rewards.clear()
            self._episode_successes.clear()

        # Save
        snap_dir = os.path.join(self.output_dir, "td_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = os.path.join(snap_dir, f"snapshot_{step:08d}.npz")
        np.savez_compressed(snap_path, **snapshot)

        self.snapshot_count += 1
        corr_str = ""
        if td_dense_spearman is not None:
            corr_str = f" spearman(|TD|,oracle)={td_dense_spearman:.3f}"
        print(
            f"[snapshot] step={step}: |TD| mean={snapshot['abs_td_mean']:.4f} "
            f"median={snapshot['abs_td_median']:.4f} p90={snapshot['abs_td_p90']:.4f} "
            f"Q_mean={snapshot['q_mean']:.4f} buffer={snapshot['buffer_size']}{corr_str}"
        )

    def _on_training_end(self):
        """Take a final snapshot (only if the interval didn't already trigger one)."""
        if self.num_timesteps > self._last_snapshot_step:
            self._take_snapshot(self.num_timesteps)
