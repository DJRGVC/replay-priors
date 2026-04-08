"""SAC subclass with proper PER integration.

SB3's vanilla SAC never calls update_priorities() on the replay buffer,
so PER buffers silently degrade to max-priority (≈uniform) sampling.

This subclass overrides train() to:
1. Extract TD errors from the critic loss computation
2. Call buffer.update_priorities() with the per-sample TD errors
3. Apply importance-sampling (IS) weights to the critic loss for unbiased gradients
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class PERSAC(SAC):
    """SAC with Prioritized Experience Replay support.

    Identical to SAC except train() feeds TD errors back to the buffer
    and weights the critic loss by IS weights when the buffer provides them.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

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

            # === PER INTEGRATION: compute per-sample TD errors ===
            with th.no_grad():
                # Average TD error across critics for priority update
                td_errors_per_critic = []
                for current_q in current_q_values:
                    td_errors_per_critic.append((current_q - target_q_values).abs())
                mean_abs_td = th.stack(td_errors_per_critic).mean(dim=0)  # (batch, 1)

            # === PER INTEGRATION: IS-weighted critic loss ===
            buffer = self.replay_buffer
            has_is_weights = hasattr(buffer, '_last_is_weights') and buffer._last_is_weights is not None

            if has_is_weights:
                is_weights = th.tensor(
                    buffer._last_is_weights, dtype=th.float32, device=self.device
                ).reshape(-1, 1)
                # Weighted MSE: weight each sample's squared error by its IS weight
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

            # === PER INTEGRATION: update priorities ===
            if hasattr(buffer, 'update_priorities') and hasattr(buffer, '_last_tree_inds'):
                td_np = mean_abs_td.cpu().numpy().flatten()
                buffer.update_priorities(buffer._last_tree_inds, td_np)

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
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
