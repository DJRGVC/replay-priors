"""SAC subclass with Reward Prediction Error (RPE) prioritized replay.

Instead of using TD-error as the priority signal (which is uninformative
in sparse-reward early training), this uses the prediction error of a
learned reward predictor: RPE = |r_predicted - r_actual|.

The reward predictor is a small MLP: (obs, action, next_obs) → r_hat.
It is trained online alongside the SAC critic/actor, using the same
replay batches.

Hypothesis: RPE will also be uninformative in sparse-reward settings
because the predictor quickly learns to output 0 (the dominant reward),
making RPE ≈ 0 for most transitions and ≈ 1 only for the rare successes
(which are too infrequent to help early on).
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class RewardPredictor(nn.Module):
    """Small MLP that predicts reward from (obs, action, next_obs)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + act_dim + obs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action, next_obs):
        x = th.cat([obs, action, next_obs], dim=-1)
        return self.net(x)


class RPESAC(SAC):
    """SAC with Reward Prediction Error prioritized replay.

    Identical to SAC except:
    1. Maintains a reward predictor MLP trained on replay batches
    2. Uses RPE (not TD-error) to update buffer priorities
    3. Applies IS weights to critic loss (same as PERSAC)
    """

    def __init__(self, *args, rpe_lr: float = 3e-4, rpe_hidden: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        # Build reward predictor after parent __init__ sets up observation/action spaces
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        self.reward_predictor = RewardPredictor(obs_dim, act_dim, rpe_hidden).to(self.device)
        self.rpe_optimizer = th.optim.Adam(self.reward_predictor.parameters(), lr=rpe_lr)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        rpe_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # Entropy coefficient
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Compute target Q-values
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Current Q-values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # === RPE: train reward predictor and compute prediction errors ===
            r_hat = self.reward_predictor(
                replay_data.observations, replay_data.actions, replay_data.next_observations
            )
            rpe_loss = F.mse_loss(r_hat, replay_data.rewards)
            self.rpe_optimizer.zero_grad()
            rpe_loss.backward()
            self.rpe_optimizer.step()
            rpe_losses.append(rpe_loss.item())

            # Compute RPE for priority update (detached)
            with th.no_grad():
                r_hat_detached = self.reward_predictor(
                    replay_data.observations, replay_data.actions, replay_data.next_observations
                )
                rpe = (r_hat_detached - replay_data.rewards).abs()  # (batch, 1)

            # === IS-weighted critic loss ===
            buffer = self.replay_buffer
            has_is_weights = hasattr(buffer, '_last_is_weights') and buffer._last_is_weights is not None

            if has_is_weights:
                is_weights = th.tensor(
                    buffer._last_is_weights, dtype=th.float32, device=self.device
                ).reshape(-1, 1)
                critic_loss = 0.5 * sum(
                    (is_weights * (current_q - target_q_values) ** 2).mean()
                    for current_q in current_q_values
                )
            else:
                critic_loss = 0.5 * sum(
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                )

            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # === RPE-PER: update priorities with RPE instead of TD-error ===
            if hasattr(buffer, 'update_priorities') and hasattr(buffer, '_last_tree_inds'):
                rpe_np = rpe.cpu().numpy().flatten()
                buffer.update_priorities(buffer._last_tree_inds, rpe_np)

            # Actor loss (unchanged from SAC)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/rpe_loss", np.mean(rpe_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
