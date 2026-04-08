# TD-Error Baseline Study — Notes

## Task Selection

We pick two MetaWorld v3 tasks at opposite ends of the difficulty spectrum:

### 1. `reach-v3` (easy, short-horizon)
- **What:** Move the Sawyer gripper to a target position in 3D space.
- **Why easy:** Single-step reaching — no grasping, no object manipulation.
  Random policies get dense rewards in the range [0.8, 4.4] immediately,
  though achieving `success=True` (tcp within 5 cm of target) still requires
  learning. Under sparse reward (0/1 success), this is the simplest possible
  MetaWorld task.
- **Role in study:** Provides a "best case" for TD-error PER — if TD-error
  is uninformative even here, it's damning. If it works here but not on the
  harder task, we can attribute the gap to task complexity.

### 2. `pick-place-v3` (hard, long-horizon)
- **What:** Grasp an object and place it at a target location.
- **Why hard:** Requires sequential sub-goals: approach → grasp → lift → place.
  Dense rewards are near-zero under random policy (~0.01), and sparse success
  is extremely unlikely by chance. The critic must learn value over a much
  longer horizon with sparser signal.
- **Role in study:** Represents the regime where the proposal claims TD-error
  PER fails most. Multi-step credit assignment + sparse reward = exactly the
  setting our VLM-based prioritization targets.

### Sparse reward setup
MetaWorld provides dense shaped rewards by default. We wrap the environment to
return **sparse binary rewards**: `r = 1.0 if info['success'] else 0.0`. The
dense reward is logged separately as the oracle signal for correlation analysis
but never seen by the agent during training.

### Observation / action space
Both tasks: obs ∈ ℝ³⁹, action ∈ ℝ⁴, max episode length = 500 steps.

## Open questions
- Is "early training" best measured in env steps or gradient steps? For now,
  we plot vs. env steps since that's the natural x-axis for sample efficiency.
- Should we snapshot the full buffer or a stratified sample? Full buffer is
  clean but memory-heavy; start with full and downsample if needed.
- MetaWorld's `success` threshold (5 cm for reach, task-specific for pick-place)
  may be too generous or too strict — monitor the success rate curve.

## Literature pointers

- **Schaul et al. (2016)** — *Prioritized Experience Replay*. Original PER paper.
  Proposes |TD-error| as priority. Acknowledges the "new experience" problem
  (max priority for new transitions) but doesn't quantify how uninformative
  |TD| is early in training.
- **Fujimoto et al. (2020)** — *An Equivalence between Loss Functions and
  Non-Uniform Sampling in Experience Replay*. Shows PER can be reframed as
  reweighting the loss; uniform sampling + reweighted loss can match PER.
  Raises the question of whether priority signal matters at all if the loss
  landscape can be adjusted.
- **Kumar et al. (2020)** — *Discor: Corrective Feedback in Reinforcement
  Learning via Distribution Correction*. Proposes correcting replay distribution
  rather than using TD-error priority; explicitly discusses failure modes of
  TD-error PER.
- **Sinha et al. (2022)** — *Experience Replay with Likelihood-free Importance
  Weights*. Replace TD-error with density-ratio based priorities. Partially
  motivated by TD-error noise.
- **Liu & Zou (2017)** — *The Effects of Memory Replay in Reinforcement
  Learning*. Theoretical analysis of replay distributions; shows uniform
  replay is suboptimal but the optimal distribution depends on the value
  function accuracy — a bootstrap problem early in training.
- **Lahire et al. (2022)** — *Large Batch Experience Replay*. Argues that
  large batch sizes reduce the impact of prioritization noise; relevant
  to our question of whether TD-error noise matters in practice.
