# TD-Error Baseline Study — Findings

**Question:** How informative is TD-error as a prioritized experience replay (PER)
signal in the early training regime on sparse-reward manipulation tasks?

**Answer:** TD-error is essentially uninformative. On easy tasks, correlation with
oracle advantage only emerges after the policy has already learned (~60% through
training). On hard tasks, it never emerges.

## Summary Figure

![TD-PER Summary](figures/td_per_summary.png)

**Figure 1 (hero figure):** Four-panel summary of the complete TD-error PER baseline study.
(a) Spearman correlation between |TD-error| and oracle advantage over training — TD-error
is uninformative (ρ ≈ 0) for 60-80% of training, only becoming a lagging indicator after
learning has already started. (b) Mode comparison across 5 seeds: TD-PER at default α=0.6
prevents all seeds from learning (0/5), while even optimally-tuned α=0.3 only matches
uniform replay, never exceeds it. (c) Q-value dynamics showing the PER positive feedback
loop: biased resampling of high-|TD| transitions causes critic overfitting and Q-divergence
(11× higher Q than uniform). (d) Regime classification showing TD-error is in a useful
"aligned" regime for only 7-50% of training across tasks.

## Setup

- **Algorithm:** SAC (MLP policy, 100k replay buffer, batch=256)
- **Tasks:** MetaWorld reach-v3 (easy, 100k steps) and pick-place-v3 (hard, 300k steps)
- **Reward:** Sparse binary (1.0 on success, 0.0 otherwise)
- **Oracle signal:** MetaWorld's dense shaped reward (never used by agent)
- **Metric:** Spearman rank correlation between |TD-error| and oracle advantage
  (dense_reward − mean), sampled from 5000 replay transitions every 10k steps
- **Compute:** Modal T4 GPU, ~20 min per 100k steps
- **Seeds:** 42, 123 (initial); 42, 123, 7, 99, 256 (5-seed baseline)

## Key Results

### reach-v3 — 5-seed uniform baseline (100k steps, metaworld==3.0.0)

| Seed | Learns? | ep_rew@90k | Spearman@80k |
|------|---------|-----------|-------------|
| 42   | NO      | 0.0       | +0.11       |
| 123  | YES     | 469.1     | +0.62       |
| 7    | YES     | 241.0     | +0.59       |
| 99   | NO      | 0.0       | +0.09       |
| 256  | YES     | 469.7     | +0.53       |

**60% success rate** (3/5 seeds learn by 90k). Non-learning seeds show Spearman ≈ 0
throughout. Learning seeds show Spearman rising from ≈ 0 to 0.5–0.6 only AFTER episode
returns begin increasing (60–80k), confirming TD-error is a lagging indicator.

**The iter_010 "regression" was stochastic.** The seed=42 non-learning observed in
iter_010 is expected (2/5 seeds don't learn by 100k).

![5-seed baseline](figures/5seed_baseline_reach_v3.png)

### reach-v3 — original 2-seed results (100k steps)

| Metric | Result |
|--------|--------|
| Spearman, first 50k steps | −0.11 to +0.10 (noise, both seeds) |
| Spearman, 50k–100k steps | 0.15–0.65 (signal emerges with policy learning) |
| Policy learns? | Yes, after ~50–60k (seed-dependent) |
| Cross-seed consistency | Both seeds show same pattern; s123 learns ~10k earlier |

### pick-place-v3 (2 seeds, 300k steps)

| Metric | seed=42 | seed=123 |
|--------|---------|----------|
| Spearman, 0–100k | −0.04 to +0.24 (noise) | −0.18 to +0.19 (noise) |
| Spearman, 100k–200k | +0.07 to +0.28 (weak positive, brief) | −0.01 to +0.30 (weak positive) |
| Spearman, 200k–300k | **−0.30 to +0.07 (inverts!)** | −0.03 to +0.20 (back to noise) |
| Policy learns? | Marginal (ep_rew peaks ~0.7 at 185k, drops to 0.1) | No (ep_rew=0 throughout) |
| Q-value stability | Oscillates wildly (0.02→50→11→0.02) | Collapsed to ~0.0005 |
| Final Spearman at 300k | **−0.21** (anti-informative) | +0.20 (weak positive) |

## Figure

![TD-error correlation over training](figures/td_correlation_over_training.png)

**Left panel:** Spearman correlation between |TD-error| and oracle advantage vs. env steps.
**Right panel:** Pearson correlation (same data). Both show near-zero correlation in early
training, with divergence between tasks only after reach-v3's policy starts succeeding.

## Priority Quality Metrics (Gini + Top-K Overlap)

In addition to correlation, we measure two priority quality metrics across all snapshots:

- **Top-10% overlap:** What fraction of the top-10% transitions by |TD| are also in the
  top-10% by oracle advantage? Chance = 10%.
- **Priority Gini coefficient:** How concentrated are |TD| priorities? Higher = more
  skewed sampling.

| Metric | reach-v3 (first 40k) | reach-v3 (50-100k) | pick-place-v3 (0-300k) |
|--------|----------------------|---------------------|------------------------|
| Top-10% overlap | 7–20% (near chance) | Brief spike to 53–61% mid-learning, then drops back to 6–11% | 8–33% (never reliably above 2× chance) |
| Gini coefficient | 0.26–0.48 | 0.39–0.54 | 0.25–0.60 |

**Key insight — TD-error inversion:** On *both* tasks, the Spearman correlation *inverts*
(goes negative) at certain training phases:
- reach-v3 s123: −0.09 to −0.12 at 90–100k (critic overshooting after policy converges)
- pick-place-v3 s42: **−0.31 at 280k** (Q-value instability during marginal learning)

When correlation inverts, TD-PER actively *anti-prioritizes* useful transitions — it
would sample the least informative transitions most frequently.

![Priority quality metrics](figures/priority_quality_metrics.png)

## Interpretation

1. **TD-error PER is a lagging indicator.** It only correlates with oracle advantage
   after the critic has already learned a reasonable value function — but by then the
   agent is already performing well, so the prioritization adds little.

2. **On hard tasks, TD-error is noisy and can become anti-informative.** Even with 3×
   more training (300k steps), pick-place-v3 never sustains meaningful correlation.
   When the policy briefly learns (seed=42, peak ep_rew=0.7 at 185k), Q-values oscillate
   wildly (0.02→50→11) and TD-error correlation *inverts* to −0.31, meaning TD-PER would
   actively sample the worst transitions.

3. **Seed variance is extreme on hard tasks.** One seed (42) showed marginal learning
   with wild Q-value instability; the other (123) showed complete policy collapse with
   near-zero Q-values. This suggests TD-error PER would be unreliable even on tasks
   where learning is technically possible.

4. **This motivates VLM-based prioritization.** A VLM that can identify "interesting"
   transitions (novel states, near-success, task-relevant progress) could provide a
   meaningful priority signal from the very first step, without waiting for critic
   convergence — and critically, without the inversion problem where high TD-error
   transitions become anti-informative.

## Cross-Study Synthesis

See **[SYNTHESIS.md](SYNTHESIS.md)** for the full cross-study analysis combining:
- TD-error baseline (this study)
- VLM localization probe (sibling: `agent/vlm_probe`)
- Literature review (subagent: `agent/lit_review2`)

**Headline result:** TD-PER fails 50-93% of training time. We identify four failure
regimes (noise, aligned, inverted, unstable) and propose an Adaptive Priority Mixer
that uses VLM scores when TD-error is uninformative and switches to TD-error when
it's valid.

## Files

| File | Description |
|------|-------------|
| `figures/td_per_summary.png` | **4-panel hero figure** — primary deliverable |
| `figures/td_per_summary.pdf` | Same in PDF for presentations |
| `SYNTHESIS.md` | **Cross-study synthesis** — headline deliverable |
| `figures/td_per_regime_map.png` | **6-panel regime map** — detailed regime analysis |
| `figures/td_per_regime_map.pdf` | Same figure in PDF for presentations |
| `figures/td_correlation_over_training.png` | Spearman + Pearson over training |
| `figures/td_correlation_over_training.json` | Raw correlation data |
| `figures/priority_quality_metrics.png` | Top-K overlap + Gini + Spearman (3-panel) |
| `plot_regime_map.py` | Regime map figure generation script |
| `plot_td_correlation.py` | Correlation figure generation script |
| `plot_priority_quality.py` | Priority quality figure generation script |
| `figures/mode_comparison_reach_v3.png` | 4-panel mode comparison (iter_008) |
| `plot_mode_comparison.py` | Mode comparison figure generation script |
| `adaptive_priority_mixer.py` | Regime-aware PER buffer (SumTree + RegimeDetector) |
| `train_mixer.py` | Training script supporting adaptive/td-per/uniform modes |
| `snapshots/` | Per-run snapshot data (TD errors, dense rewards, correlations) |
| `modal_app.py` | Modal app for running training on cloud GPU |
| `train.py` | Local training script |
| `td_instrumenter.py` | Callback that snapshots |TD|, dense reward, and computes correlations |
| `LIT_REVIEW.md` | Literature review (§1: 11 alternative PER methods) |
| `NOTES.md` | Detailed notes on task selection, literature, and methodology |

### reach-v3 — 5-seed mode comparison (100k steps, iter_012)

**TD-PER actively hurts learning.** Uniform replay is the best strategy.

| Mode | Learns | ep_rew@90k (best seed) | Q_mean@100k | |TD| @100k | Spearman@100k |
|------|--------|----------------------|-------------|-----------|---------------|
| **Uniform** | **3/5 (60%)** | 469 | 20.8 ± 18.3 | 0.20 ± 0.16 | +0.18 ± 0.12 |
| TD-PER | **0/5 (0%)** | 31 | 228.3 ± 377.0 | 7.62 ± 12.6 | −0.00 ± 0.01 |
| Adaptive | 2/5 (40%) | 471 | 87.2 ± 72.3 | 2.18 ± 1.92 | +0.00 ± 0.02 |

**Why TD-PER hurts:**
1. **Q-value explosion:** TD-PER Q_mean=228 vs uniform Q_mean=21 (11× higher).
   PER's biased sampling creates a positive feedback loop: high-|TD| transitions
   get resampled → critic overfits to them → Q diverges → even higher |TD| errors.
2. **No seed learned with TD-PER.** Even seeds that learn with uniform (123, 7, 256)
   fail completely under TD-PER.
3. **IS weights insufficient:** Despite β annealing from 0.4→1.0, importance sampling
   corrections don't prevent the Q-divergence.

**Adaptive is better than TD-PER but worse than uniform:** 2/5 seeds learn (vs 0/5
for TD-PER), likely because regime detection occasionally falls back to uniform
sampling during unstable episodes. But the PER overhead still causes Q instability
for the seeds that don't learn.

![Multi-seed mode comparison](figures/multiseed_mode_comparison.png)

### reach-v3 — Alpha sweep for TD-PER (5 seeds, 100k steps, iter_013)

**Lower α mitigates Q-explosion but never beats uniform.**

| Mode | α | Learns | Q_mean@100k | |TD|@100k |
|------|---|--------|-------------|----------|
| **Uniform** | — | **3/5 (60%)** | 20.8 ± 18.3 | 0.20 ± 0.16 |
| TD-PER | 0.3 | 3/5 (60%) | 36.6 ± 26.2 | 0.46 ± 0.29 |
| TD-PER | 0.1 | 2/5 (40%) | 144.6 ± 258.0 | 1.54 ± 2.07 |
| TD-PER | 0.6 | 0/5 (0%) | 228.3 ± 377.0 | 7.62 ± 12.6 |

**Key findings:**
1. **α=0.3 ties uniform (3/5 learn)** — less aggressive PER partially mitigates
   Q-explosion (Q=36.6 vs 228.3), allowing some seeds to learn.
2. **α=0.1 is non-monotonically worse (2/5)** — very low alpha barely distinguishes
   priorities, but one seed still explodes catastrophically (Q=660).
3. **Spearman ≈ 0 across ALL alpha values** — TD-error correlation with oracle
   advantage is unchanged by prioritization strength.
4. **Best-case TD-PER matches but never exceeds uniform.** Even with tuned α,
   PER adds overhead and instability for zero benefit. The problem is the
   SIGNAL (TD-error is uninformative in sparse-reward early training), not
   the MECHANISM (prioritized sampling).

![Alpha sweep](figures/alpha_sweep_td_per.png)

## Status

- [x] Single-seed (42) runs on reach-v3 + pick-place-v3, 100k steps
- [x] Second seed (123) for error bars — figure updated with mean ± std bands
- [x] Gini coefficient + top-K overlap metrics — top-K at chance, Gini moderate, correlation inverts late
- [x] Extended 300k runs on pick-place-v3 — correlation never stabilizes, inverts under Q-instability
- [x] Literature review (§1: 11 alternative PER methods) via lit_review2 subagent
- [x] Cross-study synthesis with VLM probe + lit review → SYNTHESIS.md + regime map figure
- [x] Regime classification (4 regimes) + MI proxy + wasted budget analysis
- [x] Adaptive Priority Mixer implementation (adaptive_priority_mixer.py + train_mixer.py)
- [x] 100k reach-v3 comparison: adaptive vs td-per vs uniform (iter_008)
  - **Critical bug found:** SB3 SAC never calls `update_priorities()` → td-per ≈ uniform
  - Adaptive mode went Q-unstable at 40k despite same effective sampling
- [x] Fix SB3 PER integration — PERSAC subclass (per_sac.py) calls update_priorities()
- [x] Re-run comparison with working PER (iter_010)
  - PER active but destabilizes Q; no mode learned (seed=42 stochastic failure)
- [x] **5-seed uniform baseline (iter_011)**: 3/5 learn, 2/5 don't — confirms stochasticity
  - Pinned MetaWorld to 3.0.0 for reproducibility
  - Figure: `figures/5seed_baseline_reach_v3.png`
- [x] **5-seed mode comparison (iter_012)**: TD-PER 0/5, uniform 3/5, adaptive 2/5
  - **TD-PER actively hurts** — Q-value explosion (11×), zero seeds learn
  - Adaptive middling — regime detection helps but PER overhead still hurts
  - Figure: `figures/multiseed_mode_comparison.png`
- [x] **Alpha sweep (iter_013)**: α=0.3 ties uniform (3/5), α=0.1 worse (2/5), α=0.6 worst (0/5)
  - Q-explosion is partly tuning, but even best α never beats uniform
  - Confirms problem is the SIGNAL, not the MECHANISM
  - Figure: `figures/alpha_sweep_td_per.png`
- [ ] Run VLM probe on pick-place-v3 failure rollouts (coordinate with vlm_probe sibling)
- [ ] Head-to-head: uniform vs TD-PER vs VLM-PER vs Adaptive-Mix
- [ ] Consider RPE-PER or other non-TD priority signals as baselines
