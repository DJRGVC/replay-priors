# TD-Error Baseline Study — Findings

**Question:** How informative is TD-error as a prioritized experience replay (PER)
signal in the early training regime on sparse-reward manipulation tasks?

**Answer:** TD-error is essentially uninformative. On easy tasks, correlation with
oracle advantage only emerges after the policy has already learned (~60% through
training). On hard tasks, it never emerges.

## Setup

- **Algorithm:** SAC (MLP policy, 100k replay buffer, batch=256)
- **Tasks:** MetaWorld reach-v3 (easy) and pick-place-v3 (hard)
- **Reward:** Sparse binary (1.0 on success, 0.0 otherwise)
- **Oracle signal:** MetaWorld's dense shaped reward (never used by agent)
- **Metric:** Spearman rank correlation between |TD-error| and oracle advantage
  (dense_reward − mean), sampled from 5000 replay transitions every 10k steps
- **Compute:** Modal T4 GPU, ~20 min per 100k-step run
- **Seeds:** 42, 123

## Key Results (2 seeds, 100k steps)

| Metric | reach-v3 | pick-place-v3 |
|--------|----------|---------------|
| Spearman, first 50k steps | −0.11 to +0.10 (noise, both seeds) | −0.05 to +0.13 (noise) |
| Spearman, 50k–100k steps | 0.15–0.65 (signal emerges) | −0.11 to +0.24 (still noise) |
| Policy learns? | Yes, after ~50–60k (seed-dependent) | No, 0% success at 100k |
| Cross-seed consistency | Both seeds show same pattern; s123 learns ~10k earlier | Both seeds stay near zero |

## Figure

![TD-error correlation over training](figures/td_correlation_over_training.png)

**Left panel:** Spearman correlation between |TD-error| and oracle advantage vs. env steps.
**Right panel:** Pearson correlation (same data). Both show near-zero correlation in early
training, with divergence between tasks only after reach-v3's policy starts succeeding.

## Interpretation

1. **TD-error PER is a lagging indicator.** It only correlates with oracle advantage
   after the critic has already learned a reasonable value function — but by then the
   agent is already performing well, so the prioritization adds little.

2. **On hard tasks, TD-error is pure noise.** The critic never converges within 100k
   steps on pick-place-v3, so |TD| priorities are random with respect to true importance.
   Q-values diverge wildly, meaning high TD-error = high noise, not high learning signal.

3. **This motivates VLM-based prioritization.** A VLM that can identify "interesting"
   transitions (novel states, near-success, task-relevant progress) could provide a
   meaningful priority signal from the very first step, without waiting for critic
   convergence.

## Files

| File | Description |
|------|-------------|
| `figures/td_correlation_over_training.png` | Main figure (Spearman + Pearson over training) |
| `figures/td_correlation_over_training.json` | Raw correlation data behind the figure |
| `snapshots/` | Per-run snapshot data (TD errors, dense rewards, correlations) |
| `modal_app.py` | Modal app for running training on cloud GPU |
| `train.py` | Local training script |
| `plot_td_correlation.py` | Figure generation script |
| `td_instrumenter.py` | Callback that snapshots |TD|, dense reward, and computes correlations |
| `NOTES.md` | Detailed notes on task selection, literature, and methodology |

## Status

- [x] Single-seed (42) runs on reach-v3 + pick-place-v3, 100k steps
- [x] Second seed (123) for error bars — figure updated with mean ± std bands
- [ ] Gini coefficient + top-K overlap metrics (oracle_correlation.py)
- [ ] Consider 200k–500k on pick-place-v3 to check if correlation ever emerges
