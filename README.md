# replay-priors

Sandbox for small studies on **what makes a good replay-prioritization signal** in
sparse-reward deep RL. Companion / preliminary work for the CS285 project
*Semantic Failure Localization for Prioritized Experience Replay* (Gardea, Grant,
Gerafian).

The main project (Parshawn) tests a VLM-as-localizer pipeline end-to-end inside
SAC on MetaWorld. This repo is deliberately scoped to **complementary
prelim/diagnostic studies** that de-risk and contextualize that pipeline without
duplicating it.

## Planned studies

Each study lives in its own folder under `studies/` with a self-contained README,
config, and results. Studies are independent — pick one up without touching the
others.

- `studies/td_error_baseline/` — empirical characterization of TD-error PER on
  sparse-reward MetaWorld tasks. How noisy / uninformative is the TD signal in
  the early-training regime the proposal claims it fails in? Motivates the whole
  project quantitatively.
- `studies/vlm_localization_probe/` — calibration study: given MetaWorld failure
  rollouts, can a VLM (Claude / GPT-4o / Gemini) actually identify the failure
  timestep? Sweep K keyframes, prompt format, model, task. Produces
  accuracy / cost / latency curves before any RL is wired up.
- (more to be added — this is an umbrella repo)

## Layout

```
replay-priors/
  studies/
    td_error_baseline/
    vlm_localization_probe/
  shared/            # rollout collection, MetaWorld wrappers, plotting utils
  docs/              # notes, lit pointers, design decisions
```

## Status

Scaffold only. Studies are being bootstrapped by [c3r](https://github.com/) agents.
