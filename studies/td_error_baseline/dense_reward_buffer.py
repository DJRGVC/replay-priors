"""Replay buffer that additionally stores the dense oracle reward.

SB3's standard ReplayBuffer discards info dicts. We subclass it to also
store info['dense_reward'] so the TD-error instrumenter can correlate
|TD-error| with the oracle dense signal.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer


class DenseRewardReplayBuffer(ReplayBuffer):
    """ReplayBuffer that additionally stores dense_reward from info dicts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parallel array for dense rewards
        self.dense_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Store dense reward before parent's add (which increments pos)
        dense = np.array(
            [info.get("dense_reward", 0.0) for info in infos], dtype=np.float32
        )
        self.dense_rewards[self.pos] = dense
        super().add(obs, next_obs, action, reward, done, infos)

    def sample_with_dense(self, batch_size: int):
        """Sample a batch and return (replay_data, dense_rewards)."""
        # Use the parent's sample logic for index generation
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        dense = self.dense_rewards[batch_inds, 0]  # n_envs=1
        replay_data = self._get_samples(batch_inds)
        return replay_data, dense
