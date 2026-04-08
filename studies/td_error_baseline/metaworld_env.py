"""MetaWorld environment factory for the TD-error baseline study.

Creates MetaWorld v3 environments wrapped for SB3 compatibility.
"""

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import spaces


class MetaWorldGymWrapper(gym.Env):
    """Wrap a MetaWorld env into a proper Gymnasium env for SB3."""

    metadata = {"render_modes": []}

    def __init__(self, task_name: str, seed: int = 0):
        super().__init__()
        self._ml1 = metaworld.ML1(task_name)
        self._env = self._ml1.train_classes[task_name]()
        self._task = self._ml1.train_tasks[0]
        self._env.set_task(self._task)

        # Observation and action spaces
        obs, _ = self._env.reset()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = self._env.action_space
        self._seed = seed
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        obs, info = self._env.reset()
        self._step_count = 0
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        if self._step_count >= self._env.max_path_length:
            truncated = True
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self):
        self._env.close()


def make_env(task_name: str, seed: int = 0, sparse: bool = True):
    """Create a MetaWorld env, optionally with sparse rewards."""
    from sparse_wrapper import SparseRewardWrapper

    env = MetaWorldGymWrapper(task_name, seed=seed)
    if sparse:
        env = SparseRewardWrapper(env)
    return env
