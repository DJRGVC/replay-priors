"""Sparse reward wrapper for MetaWorld environments.

Converts MetaWorld's dense shaped reward to binary sparse:
  r = 1.0 if info['success'] else 0.0

The original dense reward is stored in info['dense_reward'] for oracle analysis.
"""

import gymnasium as gym
import numpy as np


class SparseRewardWrapper(gym.Wrapper):
    """Replace dense reward with sparse binary success signal."""

    def step(self, action):
        obs, dense_reward, terminated, truncated, info = self.env.step(action)
        sparse_reward = 1.0 if info.get("success", False) else 0.0
        info["dense_reward"] = dense_reward
        return obs, sparse_reward, terminated, truncated, info
