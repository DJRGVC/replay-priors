# Study: VLM localization probe

**Question.** Before wiring a VLM into a replay buffer, can a VLM actually
localize *where in a trajectory a failure occurred* from K keyframes plus a task
description? Under what conditions does it work?

**Approach (sketch).**
- Collect a small set of MetaWorld failure rollouts on 2–3 tasks. Hand-label or
  scripted-label the ground-truth failure timestep where possible.
- Build a thin VLM client that takes (task description, K keyframes) and returns
  a predicted failure timestep window.
- Sweep: K ∈ {2, 4, 8, 16}, frame resolution, prompt format, model
  (Claude / GPT-4o / Gemini), task.
- Metrics: localization accuracy (window IoU vs. GT), latency, $/episode.

**Deliverable.** Accuracy / cost / latency curves. Tells Parshawn's main
experiment which (K, model, prompt) configuration is even worth plugging into
SAC, and which tasks are hopeless before any RL training is burned.

**Status.** Empty. To be bootstrapped.
