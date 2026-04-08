# Study: TD-error baseline characterization

**Question.** The project proposal claims TD-error PER is unreliable early in
training because the critic is randomly initialized. Is that quantitatively
true on the MetaWorld tasks we plan to use?

**Approach (sketch).**
- Train SAC + TD-error PER on 2–3 sparse-reward MetaWorld tasks.
- At fixed intervals, snapshot: (a) the critic's TD-error distribution over the
  buffer, (b) correlation between |TD| and ground-truth advantage from a
  dense-reward oracle, (c) which transitions PER actually upweights.
- Plot how that correlation evolves vs. environment steps.

**Deliverable.** A single figure + 1-page note that either supports or
undermines the motivation paragraph of the proposal. Either outcome is useful.

**Status.** Empty. To be bootstrapped.
