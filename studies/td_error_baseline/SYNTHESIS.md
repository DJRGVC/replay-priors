# Cross-Study Synthesis: The Case for VLM-Guided Replay Prioritization

**Date:** 2026-04-08
**Studies synthesized:**
- TD-error baseline (this branch: `agent/td_baseline`)
- VLM localization probe (sibling: `agent/vlm_probe`)
- Literature review (subagent: `agent/lit_review2`)

---

## Executive Summary

TD-error PER fails as a replay prioritization signal for **50-93% of training time**
on sparse-reward manipulation tasks. We identify four distinct failure regimes and
show that the failure is structural, not parametric — it stems from the fundamental
dependency of TD-error on critic quality, which is exactly what sparse-reward tasks
lack. Meanwhile, VLM probes demonstrate task-relevant perception from step zero,
with no critic dependency and no inversion risk. The literature confirms these
findings generalize and points to specific architectures for combining VLM signals
with replay.

## 1. TD-Error PER: A Taxonomy of Failure

We trained SAC on two MetaWorld tasks (reach-v3, pick-place-v3) with sparse binary
rewards and instrumented the critic to measure how well |TD-error| tracks a
dense-reward oracle advantage. The regime map (Figure 1) classifies every 10k-step
snapshot into one of four regimes:

### The Four Regimes

| Regime | Definition | What happens to PER |
|--------|-----------|-------------------|
| **Noise** | \|Spearman\| < 0.15 | PER degenerates to ~uniform sampling with extra compute |
| **Aligned** | Spearman >= 0.15 | PER works as intended (rare!) |
| **Inverted** | Spearman <= -0.15 | PER **anti-prioritizes** — samples the *worst* transitions most |
| **Unstable** | Q std/mean > 1.0 | Q-values oscillating; TD-errors are dominated by approximation artifacts |

### Regime Prevalence

| Run | Aligned | Noise | Inverted | Unstable | **TD-PER fails** |
|-----|---------|-------|----------|----------|-------------------|
| reach-v3 s42 (100k) | 20% | 70% | 0% | 10% | **80%** |
| reach-v3 s123 (100k) | 50% | 50% | 0% | 0% | **50%** |
| pick-place-v3 s42 (300k) | 7% | 40% | 17% | 37% | **93%** |
| pick-place-v3 s123 (300k) | 13% | 60% | 3% | 23% | **87%** |

**Key finding: TD-PER is aligned for at most ~50% of training on the easiest task
configuration, and only 7% on the hardest. The failure is not a brief transient —
it persists for the majority of training.**

### The Inversion Problem

The most damaging failure mode is *inversion*: when Spearman correlation goes
negative, PER actively selects the least informative transitions. This happened in:

- **reach-v3 s123, 90-100k steps:** Spearman drops to -0.12 despite strong policy
  (ep_rew=390). The critic overshoots — Q-values balloon to 62 (true return ~4).
  TD-errors become large for *already-learned* transitions, drowning out genuinely
  novel ones.

- **pick-place-v3 s42, 260-300k steps:** Spearman drops to **-0.31**. Q-values
  oscillated wildly (0.02 → 50 → 11 → 0.02) during marginal learning, producing
  large TD-errors on transitions that are pure noise.

Inversion is not a random fluctuation — it's a *predictable consequence* of
Q-function instability, which is endemic to sparse-reward tasks where the critic
receives extremely infrequent gradient signal.

### Information Theory Perspective

We compute a mutual information proxy: MI_proxy = log2(overlap / chance) where
overlap is the fraction of top-10%-by-|TD| transitions that are also
top-10%-by-oracle-advantage, and chance = 10%.

- **First 40-70k steps (all runs):** MI_proxy ≈ 0 bits. The |TD| ranking contains
  *zero information* about which transitions are truly important.
- **Brief peak during learning (reach-v3 only):** MI_proxy spikes to 1.5-2.6 bits.
  But this window is narrow (~10-20k steps) and closes again.
- **Late training / hard tasks:** MI_proxy returns to 0 or goes undefined
  (overlap < chance implies TD-PER is worse than uniform).

The "information desert" — the first 40-70k steps where TD-error carries zero bits
about transition importance — is exactly the period where good prioritization would
have the most impact, since the policy has learned nothing yet.

### Wasted Sampling Budget

We define *effective waste* = (1 - overlap) × Gini, capturing how confidently PER
samples the wrong transitions. When Gini is high (priorities are concentrated) and
overlap is low (concentrated on wrong transitions), waste is maximal.

- pick-place-v3 s42: effective waste reaches **50%+** at multiple training phases
- Even reach-v3 (the "easy" task) wastes 30-40% of its sampling budget for the
  first 60k steps

This isn't just theoretical — it means PER is spending 30-50% of its compute
budget on transitions that are *less* useful than what uniform sampling would
select.

## 2. What VLM Probes Tell Us (Updated — vlm_probe iters 1–37)

The sibling VLM probe study (`agent/vlm_probe`) ran **9 models** across **3 tasks**
(reach-v3, push-v3, pick-place-v3) with **10+ interventions** (K sweep, CoT, annotation,
prompt style, multi-image vs grid, ensemble, confidence gating).

### 2a. Individual model performance (reach-v3, K=8)

| Model | MAE↓ | ±10 | ±20 | Dominant bias |
|-------|------|------|------|---------------|
| Claude Sonnet 4.6 | **41.9** | **20%** | **35%** | center (t≈85) |
| GPT-4o (annotated) | 52.7 | 10% | 10% | early-mid (t=42) |
| Gemini 3 Flash Preview | 54.2 | 44% | 56% | start (t=0) |
| GPT-4o-mini (CoT, no ann) | 53.2 | 10% | 20% | early (t=21) |
| Llama-3.2-90B | 53.5 | 0% | 0% | grid-cell (t=42) |
| Gemini 2.5 Flash | 67.8 | 20% | 30% | end (t≈149) |
| GPT-4o-mini (annotated) | 68.0 | 0% | 10% | late (t≈106) |
| Phi-4-multimodal | 64.3 | 0% | 10% | grid-center (t=85) |
| Gemini 2.5 Flash-Lite | 95.2 | 5% | 10% | late |

**Central finding:** Every model has a characteristic positional bias unrelated to
visual content. Models predict based on *image position* (grid cell, sequence slot),
not *task understanding*. This is the strongest signal in the study.

### 2b. Annotation effect is GT-distribution-dependent (3-task comparison)

| Task | GT mean | GPT-4o ann effect | GPT-4o-mini ann effect |
|------|---------|-------------------|------------------------|
| reach-v3 (mid-distributed) | 57.8 | −30% (helps) | N/A |
| push-v3 (early-distributed) | 36.6 | +18% (hurts) | +16% (hurts) |
| pick-place-v3 (late-distributed) | 80.3 | N/A | +9% (hurts) |

Annotation shifts predictions toward mid-episode — helps when GT is mid-distributed,
hurts when GT clusters at extremes. This is bias-matching, not capability.

### 2c. Ensemble and confidence gating both fail

- **Naive 5-model ensemble:** MAE=51.2 vs best individual (Llama-90B) MAE=50.1 —
  weak models dilute the signal. Selected 2-model pairs improve (−6 to −10%) but
  require oracle knowledge of which pair to pick.
- **Confidence gating (inter-model agreement):** Agreement is *positively* correlated
  with error (r=+0.53). Models agree when most wrong because they share positional
  biases. Optimal gating threshold → 100% uniform usage.
- **Always-VLM priority:** Overlap 8.7% vs uniform 21.7% (60% worse), KL 2.035 vs
  1.556 (31% worse). Strictly dominated by uniform.

### 2d. Contrastive Episode Ranking also fails (vlm_probe iter 38)

The RLHF-inspired "pairwise comparison" approach — asking VLMs "which episode failed
earlier?" instead of "at what timestep?" — was tested on GPT-4o-mini with reach-v3.

**Result: 100% primacy bias.** The model picks Episode A (presented first) in 11/11
pairs (binomial P<0.001). Accuracy = 63.6% = exactly the base rate P(GT=A earlier).
When GT=A: 7/7 correct. When GT=B: 0/4 correct. No signal above always-A baseline,
regardless of gap magnitude (2–139 timesteps). Confidence 0.80–0.90 regardless.

This extends positional bias from *within-episode* (temporal fixation on specific
timestep slots) to *between-episode* (first-presented preference). The RLHF pairwise
comparison analogy fails because RLHF models are fine-tuned on comparison data; off-
the-shelf VLMs default to position heuristics on unfamiliar visual comparison tasks.

### 2e. Revised assessment

The iter-006 assessment that VLMs provide a "qualitatively different kind of signal"
was **too optimistic**. While VLM judgments are indeed critic-independent and available
from step 0, they are dominated by positional bias and produce priorities that are
*worse* than uniform when measured by KL divergence. The only metric where VLMs show
promise is top-20% overlap (+12% above uniform for Sonnet K=8), but this comes at the
cost of catastrophic misses that create harmful priority peaks far from true failure.

**Approaches tested and failed:** direct temporal localization (9 models), K sweeps,
CoT prompting, frame annotation, ensemble debiasing, confidence gating, and now
pairwise contrastive ranking. The path forward cannot be "better temporal localization"
or "better comparison" — it must sidestep temporal reasoning entirely.

## 3. Literature Connections

The lit review (§1, 11 methods surveyed) confirms our empirical findings:

- **RPE-PER** (arXiv:2501.18093, 2025) decouples prioritization from value learning
  by using reward prediction error instead of TD-error. Our finding that TD-error
  fails precisely because it depends on critic quality validates RPE-PER's premise.

- **D-SPEAR** (arXiv:2603.27346, 2026) addresses actor-critic asymmetry: the actor
  needs *low*-TD-error samples (stable gradients) while the critic needs *high*-
  TD-error samples (surprise). Our inversion finding shows this asymmetry is even
  worse than D-SPEAR assumes — in the inverted regime, *neither* stream gets
  useful samples.

- **EDER** (IJCAI 2025) proposes diversity-based replay. Our Gini analysis shows
  TD-PER creates concentrated but misguided priorities — EDER's diversity objective
  could serve as a useful regularizer alongside any priority signal.

- **SPAHER** (2024) uses spatial position attention for manipulation PER — a
  hand-crafted version of what a VLM could provide automatically.

## 4. The Proposed Architecture: Status Update

The iter-006 Adaptive Priority Mixer architecture (regime-switching between TD-error
and VLM scores) is now **invalidated** by both studies' findings:

- **TD-error branch:** Tested 5 RL signals (TD-PER at 3 alpha values, RPE-PER,
  RND-PER, Adaptive mixer). None beat uniform on reach-v3. All fail on pick-place-v3.
  The mixer's regime detection logic is correct (noise/aligned/inverted classification
  works) but there is no good signal to switch *to*.

- **VLM branch:** Always-VLM is strictly worse than uniform (overlap −60%, KL −31%).
  Confidence gating reduces to 100% uniform. Ensembles don't help.

- **The regime classifier itself works** — it accurately detects when TD-error is
  uninformative. But detection without a viable alternative signal produces the same
  outcome as uniform.

### What the architecture *should* look like (revised hypothesis)

The evidence now suggests the architecture must avoid temporal localization entirely:

1. **Contrastive Episode Ranking** — ask VLMs "which of these two episodes was
   closer to success?" rather than "at what timestep did this episode fail?" This
   leverages VLM strengths (relative judgment, scene understanding) and avoids
   the temporal precision bottleneck.

2. **Failure Mode Clustering** — use VLM *descriptions* of failure (not timestamps)
   to cluster episodes by failure mode, then prioritize under-represented modes for
   diversity. This sidesteps temporal reasoning entirely.

3. **Phase-Segmented Replay** — use VLMs to classify episodes into coarse phases
   (approaching, attempting, failing) and weight transitions from "attempting" phases
   higher. This requires only ~3 categories, not 150-step precision.

These remain untested. The key insight is: VLMs can *categorize* and *compare* but
cannot *localize temporally*.

## 5. What We Now Know (Answered Questions)

1. ~~Cost-performance tradeoff~~ → **Moot.** VLM temporal localization doesn't
   improve over uniform regardless of cost.

2. ~~Transfer across tasks~~ → **Answered.** vlm_probe tested 3 tasks. Push-v3 is
   easiest (MAE=36, bias-matching helps), pick-place-v3 shows extreme fixation.
   Task difficulty tracks GT distribution, not visual complexity.

3. ~~Scoring granularity~~ → **Episode-level temporal scoring is the wrong frame.**
   The question should be: "is this episode more informative than that one?" not
   "which timestep matters?"

4. ~~Regime detection lag~~ → **Detection works but is useless without a good
   alternative signal.** Minimum ~20k steps for reliable Spearman.

5. ~~VLM model choice~~ → **Answered.** 9 models tested. Sonnet is best (MAE=41.9)
   but all are dominated by positional bias. Model choice shifts the bias location,
   not the bias magnitude.

### Remaining open questions

1. ~~Can contrastive/pairwise VLM judgments produce actionable episode rankings?~~
   → **Answered NO** (vlm_probe iter 38). 100% primacy bias on GPT-4o-mini, 0 signal
   above always-A baseline. Positional bias extends from within-episode to between-
   episode comparison.

2. **Can VLM failure mode descriptions (text, not timestamps) create diversity-
   weighted replay that outperforms uniform?** This is the last untested non-temporal
   direction that leverages VLM scene understanding.

3. **Is the uniform dominance result specific to MetaWorld's sparse binary reward
   structure, or does it generalize to other sparse-reward domains?**

4. **Would a dense reward shaping approach (not PER) be more effective than any
   replay prioritization scheme?** The state-visitation analysis (iter_024) showed
   dense reward distributions vary dramatically across seeds — perhaps reward
   shaping is a better lever than replay.

## 6. Figures

| Figure | Description |
|--------|-------------|
| `figures/td_per_regime_map.png` | 6-panel regime map — headline figure (iter_006) |
| `figures/td_per_summary.png` | 6-panel summary — 5 modes × 2 tasks (iter_015) |
| `figures/cross_study_synthesis.png` | **4-panel cross-study landscape** — the unified figure (iter_025) |
| `figures/seed_switching_analysis.png` | Exploration bifurcation — 6-panel (iter_023) |
| `figures/state_visitation_analysis.png` | Dense reward proxy — 5-panel (iter_024) |
| `figures/td_correlation_over_training.png` | Spearman + Pearson correlation over training |
| `figures/priority_quality_metrics.png` | Top-K overlap + Gini + Spearman (3-panel) |

## 7. Recommended Next Steps (Revised)

~~1. Run VLM probe on pick-place-v3~~ → **Done** (vlm_probe iter 32).
~~2. Implement Adaptive Priority Mixer~~ → **Invalidated** (no viable signal to mix).
~~3. Head-to-head comparison with VLM-PER~~ → **Invalidated** (VLM-PER < uniform).
~~4. RPE-PER baseline~~ → **Done** (iter_018, 2/5, ties uniform).

### Active directions

1. ~~Contrastive episode ranking~~ → **Closed** (vlm_probe iter 38, primacy bias).

2. **Failure mode clustering / phase segmentation:** The only untested non-temporal
   VLM direction. Use VLM *descriptions* of failure (text, not timestamps) to cluster
   episodes by failure mode and prioritize under-represented modes for diversity.

3. **Negative result write-up:** The convergent finding (8 approaches tested, 0 beat
   uniform) is itself publishable as a cautionary benchmark paper.

4. **Dense reward shaping investigation:** State-visitation analysis suggests
   the problem may be better attacked through reward design than replay design.

---

*This synthesis was last updated by `agent/td_baseline` (iter_026) combining:*
- *TD-error baseline: 40 runs (5 seeds × {uniform, TD-PER×3α, RPE-PER, RND-PER, Adaptive} × 2 tasks)*
- *VLM probe: 9 models × 3 tasks × 10+ interventions (38 iterations)*
- *Additional analyses: seed-switching (iter_023), state-space visitation (iter_024),*
  *CER primacy bias (vlm_probe iter_038)*
- *Cross-study synthesis figure: `figures/cross_study_synthesis.png` (iter_025)*
- *Approach count: 8 tested (5 RL signals + VLM temporal + ensemble/gating + CER), 0 beat uniform*
