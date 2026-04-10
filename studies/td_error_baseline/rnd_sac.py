"""SAC subclass with Random Network Distillation (RND) prioritized replay.

Uses state novelty as the priority signal: a fixed random target network
maps observations to embeddings, and a predictor network is trained to
match those embeddings. The prediction error (high for novel/rarely-seen
states, low for familiar ones) replaces TD-error for priority updates.

Hypothesis: RND-PER will also fail in sparse-reward early training because
early exploration is random regardless of priority signal — novelty-based
resampling doesn't help when the agent hasn't discovered rewards yet.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class RNDNetwork(nn.Module):
    """Fixed random target or trainable predictor network for RND."""

    def __init__(self, obs_dim: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class RNDSAC(SAC):
    """SAC with Random Network Distillation prioritized replay.

    Identical to SAC except:
    1. Maintains a fixed random target network and a trainable predictor
    2. Uses RND prediction error (not TD-error) to update buffer priorities
    3. Applies IS weights to critic loss (same as PERSAC)
    """

    def __init__(self, *args, rnd_lr: float = 1e-3, rnd_embed_dim: int = 128,
                 rnd_hidden_dim: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        obs_dim = self.observation_space.shape[0]

        # Fixed random target — never trained
        self.rnd_target = RNDNetwork(obs_dim, rnd_embed_dim, rnd_hidden_dim).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False

        # Trainable predictor
        self.rnd_predictor = RNDNetwork(obs_dim, rnd_embed_dim, rnd_hidden_dim).to(self.device)
        self.rnd_optimizer = th.optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        rnd_losses = []

        for gradient_step in range(gradient_steps):
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

            # === RND: train predictor and compute novelty scores ===
            with th.no_grad():
                target_features = self.rnd_target(replay_data.observations)
            predicted_features = self.rnd_predictor(replay_data.observations)
            rnd_loss = F.mse_loss(predicted_features, target_features)
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
            rnd_losses.append(rnd_loss.item())

            # Compute per-sample RND error for priority update (detached)
            with th.no_grad():
                target_feat = self.rnd_target(replay_data.observations)
                pred_feat = self.rnd_predictor(replay_data.observations)
                # Per-sample MSE as novelty score
                rnd_error = ((pred_feat - target_feat) ** 2).mean(dim=-1, keepdim=True)  # (batch, 1)

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

            # === RND-PER: update priorities with novelty score ===
            if hasattr(buffer, 'update_priorities') and hasattr(buffer, '_last_tree_inds'):
                rnd_np = rnd_error.cpu().numpy().flatten()
                buffer.update_priorities(buffer._last_tree_inds, rnd_np)

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
        self.logger.record("train/rnd_loss", np.mean(rnd_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
