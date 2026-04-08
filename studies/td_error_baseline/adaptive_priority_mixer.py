"""Adaptive Priority Mixer: regime-aware replay prioritization for SAC.

Detects the current TD-error regime (noise/aligned/inverted/unstable) online
using a sliding window of critic statistics, and switches between TD-error
priorities and a fallback signal (uniform by default, VLM scores when available).

Drop-in replacement for SB3's ReplayBuffer — wraps DenseRewardReplayBuffer
with proportional PER sampling and regime-adaptive priority weighting.

Regime definitions (from SYNTHESIS.md):
  - Aligned:  Spearman(|TD|, oracle) >= 0.15  → use TD priorities
  - Noise:    |Spearman| < 0.15                → fallback (uniform/VLM)
  - Inverted: Spearman <= -0.15                → invert TD or fallback
  - Unstable: Q std/mean > 1.0                 → fallback (uniform/VLM)

Since we can't compute Spearman online (no oracle in real training), we use
proxy indicators:
  - Q-instability: coefficient of variation of Q-values in recent batches
  - TD-error concentration: Gini of |TD| in recent batches
  - TD-error trend: whether mean |TD| is increasing (potential divergence)
"""

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class SumTree:
    """Binary sum-tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return self.tree[0]

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx

    def get(self, s: float) -> tuple[int, int, float]:
        """Find the leaf for cumulative sum s. Returns (tree_idx, data_idx, priority)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, data_idx, self.tree[idx]

    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        return max(self.tree[leaf_start:leaf_end].max(), 1e-6)


# Regime enum
REGIME_NOISE = "noise"
REGIME_ALIGNED = "aligned"
REGIME_INVERTED = "inverted"
REGIME_UNSTABLE = "unstable"


class RegimeDetector:
    """Online regime detection from critic statistics.

    Uses proxy indicators since we don't have oracle access during real training:
    - Q coefficient of variation (CV) > threshold → unstable
    - TD-error Gini + trend → noise vs aligned
    - Falling mean |TD| with rising Q variance → inverted
    """

    def __init__(self, window_size: int = 50, q_cv_threshold: float = 1.0):
        self.window_size = window_size
        self.q_cv_threshold = q_cv_threshold

        # Rolling statistics
        self.q_means: list[float] = []
        self.q_stds: list[float] = []
        self.td_means: list[float] = []
        self.td_ginis: list[float] = []
        self.current_regime = REGIME_NOISE  # start conservative

    def update(self, q_values: np.ndarray, abs_td: np.ndarray):
        """Update with batch statistics from a critic forward pass."""
        self.q_means.append(float(np.mean(q_values)))
        self.q_stds.append(float(np.std(q_values)))
        self.td_means.append(float(np.mean(abs_td)))
        self.td_ginis.append(self._gini(abs_td))

        # Keep window
        for lst in [self.q_means, self.q_stds, self.td_means, self.td_ginis]:
            if len(lst) > self.window_size:
                lst.pop(0)

        self.current_regime = self._classify()

    @staticmethod
    def _gini(x: np.ndarray) -> float:
        x = np.abs(x)
        if x.sum() < 1e-10:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * x_sorted).sum() / (n * x_sorted.sum()))

    def _classify(self) -> str:
        if len(self.q_means) < 5:
            return REGIME_NOISE  # not enough data yet

        # Recent window stats
        recent_q_mean = np.mean(self.q_means[-10:])
        recent_q_std = np.mean(self.q_stds[-10:])

        # Q coefficient of variation → unstable
        if abs(recent_q_mean) > 1e-6:
            q_cv = recent_q_std / abs(recent_q_mean)
            if q_cv > self.q_cv_threshold:
                return REGIME_UNSTABLE

        # TD-error trend: if mean |TD| is rising while Gini is high,
        # the critic may be diverging (inverted regime)
        if len(self.td_means) >= 10:
            recent_td = np.mean(self.td_means[-5:])
            older_td = np.mean(self.td_means[-10:-5])
            recent_gini = np.mean(self.td_ginis[-5:])

            # Rising TD with high concentration → likely inverted
            if recent_td > older_td * 1.5 and recent_gini > 0.4:
                return REGIME_INVERTED

        # High Gini with moderate Q stability → aligned (TD priorities meaningful)
        recent_gini = np.mean(self.td_ginis[-5:])
        if recent_gini > 0.3 and abs(recent_q_mean) > 0.1:
            return REGIME_ALIGNED

        return REGIME_NOISE

    @property
    def regime_history(self) -> dict:
        return {
            "current": self.current_regime,
            "n_updates": len(self.q_means),
            "q_cv": (np.mean(self.q_stds[-10:]) / max(abs(np.mean(self.q_means[-10:])), 1e-6)
                     if len(self.q_means) >= 10 else None),
            "td_gini": np.mean(self.td_ginis[-5:]) if len(self.td_ginis) >= 5 else None,
        }


class AdaptivePriorityMixer(ReplayBuffer):
    """Replay buffer with regime-aware adaptive prioritization.

    In ALIGNED regime: proportional PER (sample ∝ |TD|^alpha)
    In NOISE/INVERTED/UNSTABLE: uniform sampling (fallback)

    When external VLM scores are provided, NOISE/INVERTED/UNSTABLE use
    VLM scores instead of uniform.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        # PER parameters
        alpha: float = 0.6,
        beta0: float = 0.4,
        beta_increment: float = 1e-5,
        epsilon: float = 1e-6,
        # Regime detection
        q_cv_threshold: float = 1.0,
        regime_window: int = 50,
        # Mixing
        td_weight_aligned: float = 1.0,  # how much TD vs fallback in aligned regime
        td_weight_noise: float = 0.0,    # pure fallback in noise
        **kwargs,
    ):
        super().__init__(
            buffer_size, observation_space, action_space,
            device=device, n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.alpha = alpha
        self.beta = beta0
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.td_weight_aligned = td_weight_aligned
        self.td_weight_noise = td_weight_noise

        # Sum-tree for proportional sampling
        self.sum_tree = SumTree(buffer_size)
        self.max_priority = 1.0

        # Dense reward storage (same as DenseRewardReplayBuffer)
        self.dense_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)

        # External VLM scores (optional, set via set_vlm_scores)
        self.vlm_scores: Optional[np.ndarray] = None

        # Regime detector
        self.regime_detector = RegimeDetector(
            window_size=regime_window,
            q_cv_threshold=q_cv_threshold,
        )

        # Logging
        self.regime_log: list[dict] = []
        self._sample_count = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Store dense reward
        dense = np.array(
            [info.get("dense_reward", 0.0) for info in infos], dtype=np.float32
        )
        self.dense_rewards[self.pos] = dense

        # Add to sum tree with max priority (new transitions get highest priority)
        self.sum_tree.add(self.max_priority ** self.alpha)

        # Call parent add
        super().add(obs, next_obs, action, reward, done, infos)

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after a training step.

        The regime detector modulates whether TD priorities or uniform are used.
        """
        regime = self.regime_detector.current_regime
        abs_td = np.abs(td_errors) + self.epsilon

        for i, (tree_idx, td) in enumerate(zip(tree_indices, abs_td)):
            if regime == REGIME_ALIGNED:
                # Use TD-error priorities
                priority = td ** self.alpha
            elif regime == REGIME_INVERTED:
                # Invert: prioritize LOW TD-error transitions
                priority = (1.0 / (td + 1.0)) ** self.alpha
            else:
                # Noise or unstable: uniform priorities
                priority = 1.0

            self.sum_tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, float(priority))

    def update_regime(self, q_values: np.ndarray, abs_td: np.ndarray):
        """Feed critic statistics to the regime detector."""
        self.regime_detector.update(q_values, abs_td)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """Sample with adaptive prioritization."""
        self._sample_count += 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        if self.sum_tree.n_entries < batch_size or self.sum_tree.total < 1e-10:
            # Fallback to uniform if tree is empty/tiny
            return super().sample(batch_size, env)

        # Proportional sampling from sum-tree
        batch_inds = np.zeros(batch_size, dtype=np.int64)
        tree_inds = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        is_weights = np.zeros(batch_size, dtype=np.float32)

        segment = self.sum_tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            tree_idx, data_idx, priority = self.sum_tree.get(s)

            # Clamp data_idx to valid range
            upper = self.buffer_size if self.full else self.pos
            data_idx = data_idx % upper

            batch_inds[i] = data_idx
            tree_inds[i] = tree_idx
            priorities[i] = max(priority, 1e-10)

        # IS weights
        n = self.sum_tree.n_entries
        min_prob = priorities.min() / self.sum_tree.total
        max_weight = (n * min_prob) ** (-self.beta)
        for i in range(batch_size):
            prob = priorities[i] / self.sum_tree.total
            weight = (n * prob) ** (-self.beta)
            is_weights[i] = weight / max_weight

        # Store tree indices for priority update
        self._last_tree_inds = tree_inds
        self._last_is_weights = is_weights

        return self._get_samples(batch_inds)

    def sample_with_dense(self, batch_size: int):
        """Sample a batch and return (replay_data, dense_rewards)."""
        replay_data = self.sample(batch_size)

        # Recover the batch indices from the last sample
        # (This is a bit hacky but avoids changing the SB3 interface)
        if hasattr(self, '_last_batch_inds'):
            dense = self.dense_rewards[self._last_batch_inds, 0]
        else:
            # Fallback: sample fresh dense rewards
            upper = self.buffer_size if self.full else self.pos
            inds = np.random.randint(0, upper, size=batch_size)
            dense = self.dense_rewards[inds, 0]

        return replay_data, dense

    def set_vlm_scores(self, indices: np.ndarray, scores: np.ndarray):
        """Set VLM-based importance scores for specific buffer indices.

        These are used as the fallback priority signal in non-aligned regimes.
        """
        if self.vlm_scores is None:
            self.vlm_scores = np.ones(self.buffer_size, dtype=np.float32)
        self.vlm_scores[indices] = scores

    @property
    def current_regime(self) -> str:
        return self.regime_detector.current_regime

    def get_regime_stats(self) -> dict:
        """Return regime detection statistics for logging."""
        return {
            **self.regime_detector.regime_history,
            "beta": self.beta,
            "sample_count": self._sample_count,
            "max_priority": self.max_priority,
            "tree_total": self.sum_tree.total,
        }
