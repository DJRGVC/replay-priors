# Literature Review: Beyond TD-Error Prioritized Experience Replay

**Scope**: Alternatives to TD-error PER in sparse-reward RL, VLM/LLM-guided exploration
and replay, foundation-model-based replay prioritization. Focus on NeurIPS, ICLR, ICML,
CoRL, RSS 2023-2026 and major lab preprints.

**Last updated**: 2026-04-07 (iter_001 — §1 only)

---

## Table of Contents

1. [Alternatives to TD-Error PER in Sparse-Reward RL](#1-alternatives-to-td-error-per-in-sparse-reward-rl)
2. [VLM/LLM-Guided Exploration, Reward Shaping & Hindsight Relabeling](#2-vlmllm-guided-exploration-reward-shaping--hindsight-relabeling) *(planned — iter_002)*
3. [Foundation-Model-Based Replay Prioritization](#3-foundation-model-based-replay-prioritization) *(planned — iter_003)*
4. [Implications for This Project](#4-implications-for-this-project) *(planned — iter_004)*
5. [Open Questions & Proposed Directions](#5-open-questions--proposed-directions) *(planned — iter_005)*

---

## 1. Alternatives to TD-Error PER in Sparse-Reward RL

### 1.1 Background: Why TD-Error PER Fails in Sparse Reward

Standard Prioritized Experience Replay (PER; Schaul et al., ICLR 2016) ranks transitions
by |delta| (absolute TD error). In sparse-reward settings this is problematic:

- **Early training**: The critic is poorly calibrated, so TD errors are noisy and
  uninformative — they reflect model initialization more than transition importance.
- **Reward sparsity**: The vast majority of transitions have near-zero reward, making
  TD errors dominated by bootstrapping noise rather than meaningful surprise signals.
- **Stochastic environments**: TD-error prioritization amplifies noise from stochastic
  rewards/transitions, leading to instability unless uncertainty-aware proxies are used.

These limitations motivate the recent wave of alternatives surveyed below.

### 1.2 Reward Prediction Error Prioritization (RPE-PER)

**Paper**: "Reward Prediction Error Prioritisation in Experience Replay: The RPE-PER
Method" (arXiv:2501.18093, Jan 2025)

RPE-PER replaces TD error with *reward prediction error* as the prioritization signal.
An auxiliary EMCN (Error-Minimizing Critic Network) learns to predict immediate rewards;
transitions where the predicted reward diverges most from the observed reward are
prioritized. Key properties:

- **Decouples value learning from prioritization**: TD error conflates Q-function
  approximation error with genuine surprise. RPE-PER isolates the reward-surprise
  component.
- **Integrates with SAC/TD3**: Drop-in replacement for the priority computation; no
  architectural changes needed.
- **Sparse-reward relevance**: In environments where rewards are rare, reward prediction
  error naturally up-weights the few transitions that actually produce reward signal,
  whereas TD error might not distinguish them from noisy value estimates.

**Relevance to this project**: Directly applicable as a PER replacement in our SAC
pipeline. The auxiliary critic adds minimal overhead.

### 1.3 Dual-Stream Prioritized Experience Adaptive Replay (D-SPEAR)

**Paper**: "D-SPEAR: Dual-Stream Prioritized Experience Adaptive Replay for Stable
Reinforcement Learning in Robotic Manipulation" (arXiv:2603.27346, Mar 2026)

D-SPEAR addresses the actor-critic asymmetry in PER: the critic benefits from
high-TD-error samples, but the actor benefits from *low*-TD-error (well-understood)
samples for stable policy gradients. D-SPEAR:

- **Decouples actor and critic sampling** from a shared buffer: critic samples are
  prioritized by high TD error; actor samples are prioritized by *low* TD error.
- **Adaptive anchor mechanism**: Adjusts the TD-error threshold separating the two
  streams based on running statistics of TD-error variance.
- **Huber-based critic objective**: Reduces sensitivity to outlier TD errors.

Evaluated on robotic manipulation tasks (including sparse-reward variants). Shows
improved stability over vanilla PER in manipulation settings.

**Relevance**: Directly targets our domain (MetaWorld manipulation). The dual-stream
idea could be combined with VLM-based prioritization — VLM scores for actor stream,
TD error for critic stream.

### 1.4 Efficient Diversity-based Experience Replay (EDER)

**Paper**: "Efficient Diversity-based Experience Replay for Deep Reinforcement Learning"
(IJCAI 2025)

EDER prioritizes samples that maximize *diversity* in the training batch rather than
individual transition importance. Key ideas:

- **Diversity metric**: Uses a distance measure in the feature space of the Q-network
  to ensure sampled batches cover a wide region of the state-action space.
- **Addresses PER's convergence to narrow data**: Standard PER can over-sample a small
  set of high-TD-error transitions, causing overfitting and slow exploration.
- **Sparse-reward benefit**: In sparse-reward settings, diversity-based replay ensures
  that the rare reward-containing transitions are not drowned out by a few high-noise
  transitions that happen to have large TD errors.

**Relevance**: Diversity is orthogonal to per-transition scoring. Could be combined
with any priority signal (TD error, reward prediction error, VLM score) as a
batch-level constraint.

### 1.5 Dual-Priority Experience Replay (DPER)

**Paper**: "Autonomous navigation via dual-priority experience replay with adaptive
hybrid weighting" (Intelligent Service Robotics, Springer 2026)

DPER splits the replay buffer into high-priority and low-priority sub-buffers based
on a TD-error threshold:

- **Dynamic threshold**: The split point adapts based on the running distribution of
  TD errors across the buffer.
- **Hybrid sampling**: Each batch draws a mix from both sub-buffers, with the mixing
  ratio adapted over training — more high-priority early on, converging toward uniform
  sampling.
- **Motivation**: Prevents the pathological case where PER permanently ignores
  low-error transitions that may become informative as the policy changes.

**Relevance**: The adaptive mixing ratio is a useful design pattern for any
prioritization scheme — avoids committing fully to a priority signal that may be
miscalibrated early in training.

### 1.6 Multi-Dimensional Transition Priority Fusion (PERDP)

**Paper**: "Prioritized experience replay in path planning via multi-dimensional
transition priority fusion" (Frontiers in Neurorobotics, 2023)

PERDP fuses multiple priority signals into a single score:

- **TD error** (critic learning signal)
- **Actor loss influence** (how much the transition affects policy gradients)
- **Immediate reward** (direct reward signal)

A dynamic weighting scheme adjusts the contribution of each dimension based on the
average priority level of the buffer, so the system self-tunes as training progresses.

**Relevance**: The multi-dimensional fusion framework is a natural way to incorporate
VLM-derived scores as an additional priority dimension alongside TD error.

### 1.7 Prioritized Hindsight Experience Replay Variants

#### PHER (Prioritized HER for Manipulation)
**Paper**: "PHER: A Method for Solving the Sparse Reward Problem of a Manipulator
Grasping Task" (Technologies, 2026)

Combines DDPG/TD3/SAC with hindsight goal relabeling and TD-error-based prioritization
of the relabeled transitions. Shows faster convergence than vanilla HER on grasping tasks.

#### SPAHER (Spatial Position Attention HER)
**Paper**: "Prioritization Hindsight Experience Based on Spatial Position Attention
for Robots" (Machine Intelligence Research, 2024)

Introduces a spatial position attention mechanism for prioritizing hindsight experiences:

- Computes transition-level and trajectory-level spatial position functions.
- Prioritizes episodes where the end-effector trajectory was closest to the (relabeled)
  goal, on the reasoning that near-miss trajectories are most informative.
- Improves final mean success rate by ~3.63% over vanilla HER on challenging Hand
  environments (MetaWorld-style dexterous manipulation).

**Relevance**: The spatial attention idea is a form of domain-informed prioritization
that could be generalized — instead of hand-crafted spatial features, a VLM could
provide a richer "how close was this to success?" score.

#### MRHER (Model-based Relay HER)
**Paper**: "MRHER: Model-based Relay Hindsight Experience Replay for Sequential Object
Manipulation Tasks with Sparse Rewards" (2024)

Combines model-based RL with relay-style HER for multi-step manipulation. Uses a learned
dynamics model to generate synthetic transitions for relay goals, improving sample
efficiency in long-horizon sparse-reward tasks.

### 1.8 Other Notable Approaches

#### Z-Score Experience Replay
**Paper**: "Z-Score Experience Replay in Off-Policy Deep Reinforcement Learning"
(PMC, 2024)

Normalizes TD errors to Z-scores before prioritization, reducing sensitivity to the
absolute scale of TD errors which varies across training phases and environments.

#### Improved Exploration-Exploitation Trade-off via Adaptive PER
**Paper**: "Improved exploration-exploitation trade-off through adaptive prioritized
experience replay" (Neurocomputing, 2024)

Dynamically adjusts the PER exponent alpha and importance-sampling exponent beta based
on training progress, balancing exploration (uniform sampling) and exploitation (TD-error
sampling) over the course of training.

#### Loss-Adjusted Priorities (LAP / LA3P)
**References**: LAP (Fujimoto et al.) and LA3P extensions.

LAP replaces TD-error priorities with loss-function-based priorities, enabling PER-like
behavior through a reweighted loss rather than biased sampling. LA3P extends this to
actor-critic architectures, using *different* priority schemes for actor vs. critic
updates (conceptually similar to D-SPEAR but approached from the loss side).

### 1.9 Summary Table

| Method | Priority Signal | Actor-Critic Aware? | Sparse-Reward Focus? | Year |
|--------|----------------|--------------------|--------------------|------|
| PER (baseline) | |TD error| | No | No | 2016 |
| RPE-PER | Reward prediction error | No | Yes | 2025 |
| D-SPEAR | TD error (dual-stream) | Yes (decoupled) | Yes (manipulation) | 2026 |
| EDER | Batch diversity | N/A | Indirect | 2025 |
| DPER | TD error (dual-buffer) | No | Indirect | 2026 |
| PERDP | Multi-dim fusion | Yes (actor loss) | Partial (reward dim) | 2023 |
| PHER | TD error + HER | No | Yes (goal-cond.) | 2026 |
| SPAHER | Spatial position attn + HER | No | Yes (manipulation) | 2024 |
| MRHER | Model-based + relay HER | No | Yes (sequential) | 2024 |
| Z-Score ER | Normalized TD error | No | Indirect | 2024 |
| LAP/LA3P | Loss-adjusted | Yes | No | 2020/2023 |

### 1.10 Key Takeaways for This Project

1. **TD error is a poor priority signal in early sparse-reward training** — this is
   well-established in the literature. Our td_baseline study should confirm this
   empirically, and the literature provides clear alternatives.

2. **Reward prediction error (RPE-PER) is the most direct replacement** — it isolates
   reward surprise from value-function noise and is a drop-in for SAC.

3. **Actor-critic decoupling (D-SPEAR, LA3P) is important** — the actor and critic
   have different sampling needs. Any VLM-based priority signal we develop should
   consider which update stream it serves.

4. **Diversity (EDER) is orthogonal and complementary** — can be layered on top of
   any per-transition priority as a batch-level constraint.

5. **HER variants (SPAHER) show that domain-informed prioritization helps** — spatial
   attention for manipulation is hand-crafted; a VLM could learn this automatically
   from visual observations.

6. **Multi-dimensional fusion (PERDP) provides a framework** for combining VLM scores
   with TD error and other signals, rather than replacing TD error entirely.

---

## 2. VLM/LLM-Guided Exploration, Reward Shaping & Hindsight Relabeling

*(To be written — iter_002)*

---

## 3. Foundation-Model-Based Replay Prioritization

*(To be written — iter_003)*

---

## 4. Implications for This Project

*(To be written — iter_004)*

---

## 5. Open Questions & Proposed Directions

*(To be written — iter_005)*

---

## References

1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. ICLR. https://arxiv.org/abs/1511.05952
2. RPE-PER (2025). Reward Prediction Error Prioritisation in Experience Replay. https://arxiv.org/html/2501.18093v1
3. D-SPEAR (2026). Dual-Stream Prioritized Experience Adaptive Replay. https://arxiv.org/html/2603.27346
4. EDER (2025). Efficient Diversity-based Experience Replay. IJCAI. https://www.ijcai.org/proceedings/2025/0788.pdf
5. DPER (2026). Dual-Priority Experience Replay with Adaptive Hybrid Weighting. Intelligent Service Robotics. https://link.springer.com/article/10.1007/s11370-026-00693-7
6. PERDP (2023). Multi-Dimensional Transition Priority Fusion. Frontiers in Neurorobotics. https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1281166/full
7. PHER (2026). PHER: Sparse Reward Problem of a Manipulator Grasping Task. Technologies. https://www.mdpi.com/2227-7080/14/3/164
8. SPAHER (2024). Spatial Position Attention-based HER. Machine Intelligence Research. https://www.mi-research.net/article/doi/10.1007/s11633-023-1467-z
9. MRHER (2024). Model-based Relay HER. ResearchGate. https://www.researchgate.net/publication/383906165
10. Z-Score ER (2024). Z-Score Experience Replay. PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC11645091/
11. Adaptive PER (2024). Improved Exploration-Exploitation Trade-off. Neurocomputing. https://www.sciencedirect.com/science/article/pii/S0925231224016072
12. PERDP dynamics (2024). Dynamics Priority PER. Scientific Reports. https://www.nature.com/articles/s41598-024-56673-3
13. LA3P / Attention-Loss-Adjusted PER (2025). Complex & Intelligent Systems. https://link.springer.com/article/10.1007/s40747-025-01852-6
