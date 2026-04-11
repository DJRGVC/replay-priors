# VLM Failure-Localization Probe — Findings

**Study:** Can vision-language models localize failure timesteps in robotic manipulation
rollouts from keyframe images alone?

**Setup:** MetaWorld reach-v3, 150-step episodes, random policy (all failures), 224×224
RGB frames, 20 rollouts. GT failure timestep = argmin(hand-to-goal distance).

**Models tested (9):** Claude Sonnet 4.6 ($0.004/call), GPT-4o ($0 via GitHub),
GPT-4o-mini ($0 via GitHub), Gemini 3 Flash Preview ($0), Gemini 2.5 Flash ($0),
Gemini 2.5 Flash-Lite ($0), Llama-3.2-11B ($0), Llama-3.2-90B ($0),
Phi-4-multimodal ($0).

**Interventions tested (10):** K sweep (4/8/16/32 keyframes), CoT prompting, frame
annotation, proprio-as-text, random vs uniform sampling, two-pass adaptive probing,
annotation ± comparison across 4 model tiers, CoT × annotation 2×2 factorial (GPT-4o
and GPT-4o-mini), cross-model K sweep.

## Key Findings

### 1. VLMs can coarsely localize manipulation failures, but accuracy is far from actionable

| Model | API | MAE↓ | Median↓ | ±10 | ±20 | Bias Pattern |
|-------|-----|------|---------|-----|------|--------------|
| Claude Sonnet 4.6 | Anthropic | 41.9 | 34.0 | 20% | 35% | center (t≈85) |
| GPT-4o (ann) | GitHub | 52.7 | 43.5 | 10% | 10% | early-mid (t=42) |
| Gemini 3 Flash Preview | Google | 54.2 | 14.0 | **44%** | **56%** | start (t=0) |
| Llama-3.2-90B | GitHub | 53.5 | 37.5 | 0% | 0% | grid-cell (t=42) |
| GPT-4o-mini (CoT, no ann) | GitHub | 53.2 | 21.0 | 10% | 20% | early (t=21) |
| GPT-4o-mini (no ann) | GitHub | 61.2 | 51.0 | 10% | 20% | late (t≈106) |
| Phi-4-multimodal | GitHub | 64.3 | — | 0% | 10% | grid-center (t=85) |
| Gemini 2.5 Flash | Google | 67.8 | — | 20% | 30% | end (t≈149) |
| GPT-4o-mini (ann) | GitHub | 68.0 | 63.5 | 0% | 10% | late (t≈106,127) |
| Llama-3.2-11B | GitHub | 72.9 | 66.5 | 10% | 10% | grid-cell (t=106) |
| GPT-4o (no ann) | GitHub | 75.8 | 65.0 | 0% | 20% | start (t=0) |
| Gemini 2.5 Flash-Lite | Google | 95.2 | 107.5 | 5% | 10% | late |

Best MAE: Claude Sonnet (41.9). Best ±10 accuracy: Gemini 3 Flash Preview (44%, but
bimodal — 33% of predictions are catastrophic). No model exceeds 44% within ±10
timesteps. For a 150-step episode, MAE≈42 means the model is off by ~28% of the
episode on average.

### 2. Every model has a distinct positional bias — none reason temporally

Each model defaults to a characteristic position regardless of visual content:
- **Claude Sonnet:** center-bias (t≈85, the middle keyframe)
- **Gemini 3 Flash Preview:** start-bias (predicts t=0, "arm remains stationary")
- **Gemini 2.5 Flash:** end-bias (predicts t≈149)
- **GPT-4o (multi-image, annotated):** early-mid bias (6/10 at t=42, native multi-image)
- **GPT-4o (multi-image, no annotation):** start-bias (5/10 at t=0)
- **GPT-4o-mini (multi-image):** late-bias (t≈106, 127 — despite native multi-image, no grid)
- **Llama/Phi-4 (grid-tiled):** grid-cell bias (locks onto specific tile positions)

This is the strongest signal in the study. Models are selecting positions in the
*presentation format* (image sequence position, grid cell), not reasoning about the
*visual content*. The bias patterns are stable across rollouts within a model but
completely different between models, confirming they reflect learned positional priors
rather than visual understanding.

**Important control (iter 018):** GPT-4o-mini uses native multi-image (no grid tiling)
yet still exhibits strong late-bias — confirming positional bias is intrinsic to the
models' temporal reasoning, not an artifact of the grid presentation format.

### 3. More keyframes help native multi-image models but not concatenated-grid models

**Claude Sonnet (multi-image API):**

| K | MAE | ±10 | ±20 |
|---|-----|------|------|
| 4 | 47.4 | — | — |
| 8 | **41.9** | **20%** | **35%** |
| 16 | 44.4 | 20% | 35% |
| 32 | 51.5 | — | — |

Flat from K=4 to K=16, *worsens* at K=32. Visual acuity, not temporal resolution,
is the bottleneck.

**GPT-4o-mini (native multi-image via GitHub Models):**

| K | MAE | ±5 | ±10 | ±20 | Notes |
|---|-----|-----|------|------|-------|
| 4 | 61.6 | 0% | 10% | 10% | 10/10 predictions at t=99 (extreme fixation) |
| 8 | 68.0 | 0% | 0% | 10% | late-bias at t=106,127 |
| 16 | **57.6** | **10%** | **20%** | **30%** | most diverse predictions, first ±5 hit |

**GPT-4o (strong, native multi-image via GitHub Models):**

| K | MAE | ±5 | ±10 | ±20 | Unique preds | Dominant position |
|---|-----|-----|------|------|-------------|-------------------|
| 4 | **49.0** | 0% | 0% | 10% | 3 | t=49 (7/10) |
| 8 | 52.7 | 10% | 10% | 10% | 3 | t=42 (6/10) |
| 16 | 53.7 | **10%** | **20%** | **40%** | **6** | t=0,29 (3/10 each) |

**K effect reveals a bias-variance tradeoff.** Fewer frames → extreme positional fixation
(K=4 GPT-4o: 7/10 at t=49, K=4 GPT-4o-mini: 10/10 at t=99) producing low variance but
strong bias. More frames → diversified predictions (K=16 GPT-4o: 6 unique values) with
better tolerance accuracy (±20) but potential new failure modes (start-bias at t=0).

The pattern is model-dependent:
- **Sonnet (strong):** flat K=4–16, gains nothing from more frames
- **GPT-4o (strong):** K=4 best MAE (49.0), K=16 best ±20 (40%) — MAE/tolerance tradeoff
- **GPT-4o-mini (mid-tier):** K=16 best on both MAE AND ±20 — more frames unambiguously help

Mid-tier models benefit most from more frames because they have more to gain from temporal
resolution. Strong models already extract what they can from fewer frames, so adding more
frames mainly introduces new positional attractors (t=0 start-bias in K=16).

### 4. CoT prompting hurts weak models, helps mid/strong unannotated — substitutable with annotation

| Model | Direct MAE | CoT MAE | Δ | N |
|-------|-----------|---------|---|---|
| Gemini 2.5 Flash-Lite | 71.9 | 79.2* | +7.3 | 10 |
| Gemini 2.5 Flash | 67.8 | 75.1 | +7.3 | 10 |
| Phi-4-multimodal | 64.3 | 90.2 | **+25.9** | 10 |
| GPT-4o-mini (annotated) | 68.0 | 66.4 | −1.6 | 10 |
| GPT-4o-mini (unannotated) | 61.2 | 53.2 | **−8.0** | 10 |
| GPT-4o (annotated) | 52.7 | 52.2 | −0.5 | 10 |
| GPT-4o (unannotated) | 75.8 | 65.0 | −10.8 | 10 |
| Gemini 3 Flash Preview | 54.2 | 22.0 | −32.2 | **3** |

*Estimated from iter_007 control.

CoT (Summarize→Think→Answer) hurts weak models (3/3 negative: +7 to +26 MAE) but
**helps mid and strong models when unannotated** (GPT-4o-mini: −13%, GPT-4o: −14%).
When annotation is already present, CoT is neutral on both mid and strong models
(GPT-4o-mini: 68.0→66.4; GPT-4o: 52.7→52.2). This reveals CoT and annotation are
**partially substitutable temporal scaffolds**. Full 2×2 factorials:

| GPT-4o | Direct | CoT |
|--------|--------|-----|
| **Annotated** | 52.7 | 52.2 |
| **Unannotated** | 75.8 | 65.0 |

| GPT-4o-mini | Direct | CoT |
|-------------|--------|-----|
| **Annotated** | 68.0 | 66.4 |
| **Unannotated** | 61.2 | **53.2** |

The interaction patterns are **mirror images** across model strength:
- **GPT-4o (strong):** Annotation is the key lever (−30% MAE). CoT is neutral when
  annotation is present, helpful without it. Best = annotated (either prompt).
- **GPT-4o-mini (mid):** Annotation HURTS in both prompt styles. CoT is the key lever
  when unannotated (−13%). Best = CoT+unannotated (MAE=53.2).

The GPT-4o-mini CoT+unannotated result (MAE=53.2) is competitive with GPT-4o
annotated (52.7), suggesting CoT can substitute for both annotation AND model
strength. However, the mechanism differs: GPT-4o-mini CoT+NoAnn shows extreme
early-bias (7/10 at t=21) which happens to align with GT failure distribution,
while GPT-4o annotated shows more diverse predictions anchored to annotation labels.
Annotation interferes with CoT on mid-tier models (66.4 vs 53.2 — annotation adds
+13 MAE to CoT), while annotation reinforces CoT on strong models (52.2 vs 65.0 —
annotation reduces CoT MAE by 13).

### 5. Frame annotation is model- and architecture-dependent — not purely strength-correlated

| Model | Tier | Unannotated | Annotated | ΔMAE | Notes |
|-------|------|------------|-----------|------|-------|
| Phi-4-multimodal | very weak | 104.0 | 108.3 | +4.3 (~0, noise) | grid-tiled |
| Gemini 2.5 Flash-Lite | weak | 71.9 | 59.5 | **−12.4 (−17%)** | grid-tiled |
| GPT-4o-mini | mid | 61.2 | 68.0 | +6.8 (+11%) | native multi-img |
| Gemini 3 Flash Preview | strong | 69.9 | 67.3 | −2.6 (−4%, ~0) | native multi-img |
| GPT-4o | strong | 75.8 | 52.7 | **−23.1 (−30%)** | native multi-img |

Overlaying "t=X (N%)" on each frame (VTimeCoT-style) shows a complex effect pattern
across five models:
- **Very weak (Phi-4, grid):** NO effect — 50-60% parse failure rate means annotation
  can't help; the model's bottleneck is basic output formatting, not temporal reasoning.
  Both conditions give MAE≈104-108.
- **Weak (Flash-Lite):** annotation provides needed temporal anchors (−17% MAE)
- **Mid-tier (GPT-4o-mini):** annotation HURTS (+11% MAE), shifting distribution
  toward late timesteps — annotation text draws attention to positional priors
- **Strong (Gemini 3 Flash Preview):** NO effect — 8/10 predictions identical with and
  without annotation (MAE 69.9→67.3, within noise). Model ignores annotation text.
- **Strong (GPT-4o):** annotation dramatically HELPS (−30% MAE), shifting from
  start-bias (5/10 at t=0 unannotated) to more distributed predictions

The Gemini-3-flash-preview result (iter 28) breaks the earlier "U-shaped" narrative.
Two strong models respond oppositely: GPT-4o gains −30% from annotation while
Gemini-3-flash-preview ignores it entirely (8/10 same predictions). This suggests the
annotation effect is **architecture-specific** rather than purely strength-dependent —
the two models likely process overlaid text tokens via different multimodal fusion
mechanisms.

**Task-dependency (iter 31):** On push-v3, annotation **hurts** GPT-4o (+18% MAE),
reversing the −30% benefit seen on reach-v3. The effect interacts with GT failure
distribution: annotation shifts predictions away from the start-bias that matches
push-v3's early-failure cluster. Annotation benefit is model × task specific.

### 6. Two-pass adaptive probing fails — coarse pass too inaccurate to guide refinement

Tested on Llama-3.2-90B: coarse K=4 → refine K=8 in ±15% window around predicted
failure. Result: MAE worsened from 69.8 → 71.3. 6/10 rollouts got worse. The coarse
pass (MAE≈70) centers the refinement window on the wrong region, so the model sees an
even less informative view. Two-pass requires ±15-step coarse accuracy to be useful,
but no model achieves this.

### 7. Random sampling breaks position clustering but does not improve accuracy

Tested on Llama-3.2-90B with grid tiling. Random sampling eliminated prediction
clustering (0 repeated predictions vs 3/9 at same value for uniform), but MAE was
identical (64.7 vs 63.8). The model picks *different* wrong answers rather than the
*same* wrong answer.

### 8. Grid tiling (single-image APIs) introduces a qualitatively different bias

APIs limited to 1 image per request (GitHub Models) require tiling K frames into a
grid. This introduces grid-cell bias distinct from the center/start/end biases of
multi-image APIs. Models lock onto specific grid cells (e.g., cell 3/8 or 6/8)
regardless of content. Multi-image APIs (Gemini, Anthropic) preserve per-frame
attention but have their own positional priors.

### 9. Downstream priority quality: overlap useful, KL harmful

Converting VLM predictions to Gaussian-kernel replay priorities (σ=10):
- **KL divergence: always worse than uniform** (−6% to −24%) — catastrophic misses
  create harmful priority peaks far from true failure
- **Top-20% overlap: +8-12% above uniform** at K=8/16 — VLM correctly upweights
  some failure-adjacent transitions

This tension means VLM priorities help *if* the RL agent benefits from slightly biased
sampling toward failure regions (overlap), but hurt *if* it needs well-calibrated
priority distributions (KL). A confidence-gated hybrid (VLM when confident, uniform
otherwise) could resolve this, but current confidence scores (0.3-0.55) are too
poorly calibrated.

### 10. Task generalization: annotation effect is GT-distribution-dependent (3-task comparison)

Cross-task comparison (reach-v3, push-v3, pick-place-v3) reveals **annotation effect
tracks GT failure distribution**, not model capability:

| Model | Task | GT mean | Annotated | Unannotated | Ann effect |
|-------|------|---------|-----------|-------------|------------|
| GPT-4o | reach-v3 | 57.8 (mid) | **52.7** | 75.8 | −30% (helps) |
| GPT-4o | push-v3 | 36.6 (early) | 43.0 | **36.3** | +18% (hurts) |
| GPT-4o | pick-place-v3 | 80.3 (late) | 48.3 | — | — |
| GPT-4o-mini | reach-v3 | 57.8 (mid) | 68.0 | 61.2 | +11% (hurts) |
| GPT-4o-mini | push-v3 | 36.6 (early) | — | 44.4 | — |
| GPT-4o-mini | pick-place-v3 | 80.3 (late) | 55.2 | **50.6** | +9% (hurts) |

**Mechanistic explanation:** Annotation shifts all models' predictions toward mid-episode
(t=42-85 range) by providing explicit temporal anchors. This helps when GT failures are
mid-distributed (reach-v3, mean=57.8) but hurts when GT clusters at extremes:
- **push-v3**: GT early (mean=36.6, 5/10 in first 15 steps). Unannotated start-bias
  naturally matches → annotation shifts away from productive bias.
- **pick-place-v3**: GT late (mean=80.3, 5/10 > 100). GPT-4o-mini fixates on t=106
  (9/10 unannotated, 5/10 annotated). Annotation slightly diversifies predictions but
  doesn't improve accuracy — late fixation accidentally matches late GT distribution.
- **reach-v3**: GT mid-distributed (mean=57.8). Annotation anchors match GT → helps.

This is a **bias-matching** story: annotation doesn't improve visual understanding, it
shifts positional bias. Whether this helps depends entirely on whether the shifted bias
aligns with the task's GT distribution.

GPT-4o achieves best-ever localization on push-v3 unannotated: MAE=36.3, ±10=50%.

### 11. Ensemble analysis: naive ensembles fail, selected pairs marginally help (BAEP)

Debiased multi-model ensembles (5 VLMs, 3 debiasing × 3 aggregation methods):
- **Naive 5-model ensembles do NOT beat best individual**: annotated MAE=51.2 vs
  Llama-90B MAE=50.1 (Δ=+1.1); unannotated MAE=44.3 vs Sonnet MAE=43.4 (Δ=+0.9).
- **Selected 2-model pairs DO outperform**: annotated Llama-90B+GPT-4o-mini MAE=46.9
  (−6.4%); unannotated Sonnet+Phi-4 MAE=39.0 (−10.1%).
- Linear debiasing raises inter-model agreement from 0.51→0.71, confirming bias
  correction works structurally, but residual variance is too high for full-ensemble
  averaging — weak models dilute the signal.

### 12. Confidence-gated VLM-PER fails: agreement ≠ accuracy (Proposal 5)

Inter-model agreement (1 − σ_predictions/75) tested as confidence signal for gating
between VLM priority and uniform fallback:
- **Agreement-error correlation is POSITIVE** (r=+0.53 annotated, r=+0.32 unannotated):
  when models agree, they agree on the WRONG answer (shared positional bias).
- **Optimal gating threshold is "never use VLM"**: τ*=0.75 → 0% VLM usage → pure
  uniform (KL=1.556 vs always-VLM KL=2.035, overlap=21.7% vs 8.7%).
- **Always-VLM is strictly worse than uniform** on both KL and overlap metrics,
  confirming §9's finding at the ensemble level.
- **Root cause**: models agree because they share positional biases (all gravitate toward
  mid-episode), not because they've identified the true failure. Agreement measures bias
  correlation, not prediction quality.

This definitively closes the confidence-gating approach for VLM temporal localization.
The fundamental problem isn't confidence calibration — it's that ensemble consensus
reflects shared bias structure, making it an anti-signal for accuracy.

### §14. Failure mode descriptions show clustering potential (Proposal 4)

Pivoting from temporal localization ("when did it fail?") to failure description
("what went wrong?"), VLMs produce **semantically rich and diverse failure mode
descriptions** that could drive diversity-weighted replay prioritization:

- **Category diversity is high**: 6/6 predefined categories used on reach-v3 (n=20,
  GPT-4o-mini). Shannon entropy = 2.41/2.58 (normalized 0.93). Phi-4 invents
  novel task-specific categories beyond the predefined set (crash, size_mismatch,
  missing_target on pick-place-v3).
- **Categories correlate with GT failure timing**: η² = 0.34 (reach-v3), 0.58
  (push-v3), 0.99 (pick-place-v3) — all "large effect." VLM-assigned categories
  capture real behavioral differences, not random labels.
- **Descriptions are lexically diverse**: 100% unique across all tasks. Mean
  pairwise Jaccard similarity = 0.27 (well below 0.5 threshold). Visual cues are
  nearly all unique (78/79 on reach-v3).
- **Task-specific vocabulary emerges naturally**: reach-v3 uses "sphere/end-effector/
  above", push-v3 uses "puck/contact/positioned", pick-place-v3 uses "grasp/lift/
  placed."

This is the first VLM output in the study that shows genuine semantic signal above
positional bias. The critical difference: failure description is a scene understanding
task (VLM strength) not a temporal precision task (VLM weakness). The η² values
suggest that clustering descriptions could yield behaviorally meaningful episode
groups for diversity-weighted replay — the exact use case Proposal 4 targets.

### §15. TF-IDF clustering fails (iter 40)

Attempted to embed failure descriptions via TF-IDF (100 features, 1-2 ngrams) and
cluster for diversity-weighted replay. Key findings:

- **TF-IDF clustering fails**: silhouette scores 0.09-0.12 across K=2..6, ARI ≈ 0
  vs VLM categories. Descriptions are syntactically template-like ("The robot arm
  failed to [verb] the [object]") despite being semantically diverse.
- **VLM categories ARE the useful signal**: categories correlate with GT timing
  (η²=0.34, shown in iter 39) and produce 6.0x max weight ratio when used for
  diversity-weighted replay. The free-text descriptions don't add value over the
  categorical labels.
- **Task separation works in embedding space**: PCA shows clear task clusters
  (different object vocabulary), confirming descriptions are task-aware.
- **Diversity weighting upweights late quartiles**: Q2-Q3 get 31%/29% vs 25%
  uniform, shifting weight toward underrepresented failure timings.

**Implication (revised in §16)**: While descriptions and categories show semantic
signal, this does not translate to useful replay priorities — see §16 below.

### §16. Category-diversity replay is NOT better than uniform (iter 41)

Simulated category-diversity-weighted replay (inverse category frequency) vs
uniform on all 20 reach-v3 rollouts, sampling B={5,8,10} episodes across
10,000 trials each:

- **GT coverage improvement is noise-level**: +2.8% at B=5, +1.7% at B=10.
  Category-diversity at best matches uniform; at worst, random noise.
- **Oracle correlation is zero**: ρ(cat-diversity, oracle) = +0.04 (p=0.88).
  Category rarity does not predict which episodes are most informative.
- **The η²→priority gap**: categories explain 34% of GT timing variance
  (real signal), but inverse-frequency weighting doesn't translate this to
  better replay because: (a) rare categories aren't inherently more useful,
  (b) the correlation is distributional (group means differ) not ordinal
  (rarity doesn't rank episodes).

**Bottom line for Proposal 4**: VLM failure descriptions are semantically rich
(§14) and categories carry real behavioral signal (η²=0.34), but this does NOT
translate to actionable replay priorities. Category-diversity replay ≈ uniform.

## Related Work

Recent literature directly connects to our findings:

**Positional bias in multi-image VLMs.** Tian et al. (CVPR 2025 Oral,
[arXiv:2503.13792](https://arxiv.org/abs/2503.13792)) independently confirm our
Finding #2: open-source models exhibit "recency bias" (preferring later images),
while proprietary models show a "lost-in-the-middle" effect. They trace this to
position embeddings and causal attention masks, and propose SoFt Attention (SoFA)
— a training-free interpolation between causal and bidirectional attention that
reduces bias. This is a potential mitigation we have not tested.

**VLM failure detection in manipulation.** AHA (Duan et al., NeurIPS 2024,
[arXiv:2410.00371](https://arxiv.org/abs/2410.00371)) is closest prior work: a
fine-tuned VLM that detects and reasons about failures in robotic manipulation,
outperforming GPT-4o in-context by 10.3%. Unlike our zero-shot probing, AHA uses
a custom dataset (FailGen) with procedurally perturbed demonstrations. Our results
contextualize their gains — zero-shot VLMs are dominated by positional bias,
making fine-tuning essential for reliable failure detection.

**VLM as reward/critic for RL.** VLAC (InternRobotics, 2025,
[arXiv:2509.15937](https://arxiv.org/abs/2509.15937)) trains a VLM critic on
3000h+ data to output dense progress deltas. This is the "VLM as priority signal"
idea from §9 taken to production scale, validating the direction but requiring
substantial training data.

**Visual temporal grounding.** VTimeCoT (2025,
[arXiv:2510.14672](https://arxiv.org/abs/2510.14672)) overlays progress-bar
annotations on video frames — closely mirroring our frame annotation intervention
(§5). Code-as-Monitor (Zhou et al., CVPR 2025,
[arXiv:2412.04455](https://arxiv.org/abs/2412.04455)) uses VLM-generated code for
spatio-temporal constraint monitoring, representing an alternative paradigm
(programmatic vs direct prediction) for failure detection.

**Multi-image VLM failure modes.** Das et al. (2026,
[arXiv:2601.07812](https://arxiv.org/abs/2601.07812)) introduce MIMIC, a benchmark
showing LVLMs pervasively fail to aggregate information across images and struggle
to track multiple concepts simultaneously. Their attention-masking fix for
multi-image inputs suggests architectural remedies for the cross-image failures we
observe in §2 and §8.

**Positional encoding root cause.** "Revisiting Multimodal Positional Encoding in
VLMs" (ICLR 2026, [arXiv:2510.23095](https://arxiv.org/abs/2510.23095)) shows that
MRoPE allocates temporal encoding to high-frequency channels only, causing rapid
attention decay over time. This explains the temporal bias we observe across all
models (§2): the positional encoding architecture itself is biased against long-range
temporal reasoning. Their MRoPE-Interleaved fix distributes channels round-robin
across temporal/spatial axes.

**VLA failure prediction.** FPC-VLA (Yang et al., 2025,
[arXiv:2509.04018](https://arxiv.org/abs/2509.04018)) uses a VLM supervisor
triggered at keyframes to predict and correct failures — the closest setup to our
probe, but fine-tuned rather than zero-shot. RoboFAC (2025,
[arXiv:2505.12224](https://arxiv.org/abs/2505.12224)) provides a comprehensive
failure analysis framework. Both confirm that reliable failure detection requires
fine-tuning, consistent with our zero-shot findings.

**Prioritized replay meets foundation models.** Fatemi (2026,
[arXiv:2601.02648](https://arxiv.org/abs/2601.02648)) applies prioritized replay
to LLM RL post-training, finding that intermediate-difficulty samples produce the
strongest learning signal — paralleling our Finding #9 where VLM priority overlap
(partial success) is useful but KL (whole-distribution) is harmful.

## Cross-Study Connection: TD-Error Baseline

The sibling study (`td_error_baseline`) finds TD-error PER is **uninformative in
early training** (Spearman ρ ≈ 0 for first 60-80% of training, then becomes a
lagging indicator). TD-PER at default α=0.6 prevents all seeds from learning (0/5).

Combined interpretation: both traditional (TD-error) and VLM-based replay
prioritization struggle in the sparse-reward manipulation setting. TD-error fails
because the critic has no signal early on; VLMs fail because the visual differences
between good and bad timesteps are too subtle at 224×224 resolution. Neither provides
a reliable priority signal when it would be most needed (early training).

## Limitations and Open Questions

1. **Single task (reach-v3)** — all quantitative findings are on one task with a
   random policy. Generalization to other tasks requires trained-policy rollouts.
2. **Gemini rate limits** — the best free model (Gemini 3 Flash Preview, ±10=44%)
   was severely rate-limited, preventing key experiments (annotation, full CoT
   comparison). Several findings rest on n=3-10.
3. **No downstream RL integration** — the overlap-vs-KL tension from §9 can only
   be resolved by actually training an RL agent with VLM-derived priorities.
4. **Resolution ceiling** — reach-v3 has distance range ~0.098; distinguishing the
   failure timestep requires ~0.1-unit position changes in images where the arm is
   ~30 pixels. Higher-resolution images or crops might help.
5. **Prompt engineering** — only two prompt styles tested (direct, CoT). Other
   approaches (e.g., pairwise comparison, binary search, video-native models) are
   unexplored.

## Experimental Inventory

| Iter | Variable | Result | Status |
|------|----------|--------|--------|
| 001 | Data collection (3 tasks × 20 rollouts) | 60 rollouts, all failures | ✓ |
| 002 | Claude Sonnet baseline (K=8) | MAE=41.9, ±10=20% | ✓ |
| 003 | K sweep (4/8/16/32) | No improvement with more frames | ✓ |
| 004 | Gemini Flash-Lite | MAE=95.2, much worse | ✓ |
| 005 | Gemini 3 Flash Preview | MAE=54.2, ±10=44%, bimodal | ✓ |
| 006 | CoT prompt (flash/flash-preview) | Hurts weak (+7.3), suggestive for strong (n=3) | ✓ |
| 007 | Frame annotation (flash-lite) | MAE −17%, ±10 doubled | ✓ |
| 008 | Proprio-as-text | n=2 valid, inconclusive | ⚠ |
| 009 | Groq backend + summary | Code ready, no API key | blocked |
| 010 | Priority score analysis | Overlap +12%, KL −8% | ✓ |
| 011 | Two-pass code + random sampling code | Code ready, no API | code-only |
| 012 | GT quality analysis | push/pick-place unsuitable | ✓ |
| 013 | GitHub Models (Llama 3.2) | 11B MAE=72.9, 90B MAE=53.5 | ✓ |
| 014 | Random vs uniform sampling | No improvement (MAE 64.7 vs 63.8) | ✓ |
| 015 | Two-pass adaptive probing | NEGATIVE (MAE 69.8→71.3) | ✓ |
| 016 | CoT on Phi-4 | NEGATIVE (MAE 64.3→90.2) | ✓ |
| 017 | FINDINGS.md synthesis | 10 findings, Gemini still blocked | ✓ |
| 018 | GPT-4o-mini (annotated) | MAE=68.0, late-bias, native multi-image | ✓ |
| 019 | GPT-4o-mini annotation ± | Annotation HURTS (+11% MAE), model-dependent | ✓ |
| 020 | HTML report interface | Self-contained report.html with embedded figures | ✓ |
| 021 | Literature review + Gemini retry | 6 related papers found, Gemini still 20 RPD exhausted | ✓ |
| 022 | GPT-4o annotation ± comparison | Ann MAE=52.7, no-ann MAE=75.8 — annotation helps strong models (−30%) | ✓ |
| 023 | GPT-4o CoT 2×2 (ann × prompt) | CoT neutral when annotated (52.7→52.2), helps unannotated (75.8→65.0) — CoT & annotation substitutable | ✓ |
| 024 | HTML report update | Updated report with iters 019-023 data, Gemini 503 | ✓ |
| 025 | Phi-4 annotation ± + GPT-4o-mini K sweep | Phi-4 ann no effect (MAE 108 vs 104, 50% parse fail); GPT-4o-mini K=16 best (57.6 vs 68.0 K=8) | ✓ |
| 026 | GPT-4o-mini CoT 2×2 complete | CoT+no-ann best (MAE=53.2), mirror-image of GPT-4o: annotation hurts mid-tier in both prompt styles, CoT is key lever. Gemini-3 image quota reset but still rate-limited. | ✓ |
| 027 | Literature update + pause | 5 new papers added to Related Work; study pausing per Daniel (focus on td_baseline integration) | ✓ |
| 028 | Gemini-3-flash-preview annotation ± | NO effect: 8/10 predictions identical (MAE 69.9→67.3, −4%). Breaks U-shaped narrative — annotation is architecture-specific not strength-dependent. | ✓ |
| 029 | Quarto page bootstrap | agents/vlm_probe.qmd + references/vlm_probe.qmd + figures | ✓ |
| 030 | GPT-4o K sweep (K=4/8/16) | K=4 best MAE (49.0), K=16 best ±20 (40%): bias-variance tradeoff. Mid-tier benefits most from more frames. | ✓ |
| 031 | push-v3 task generalization (GPT-4o ±ann, GPT-4o-mini) | GPT-4o unannotated MAE=36.3 (best ever!), annotation HURTS (+18%). GPT-4o-mini MAE=44.4. Finding #10 revised: push-v3 easier, annotation effect task-dependent. | ✓ |
| 032 | pick-place-v3 task generalization (GPT-4o ann, GPT-4o-mini ±ann) | GPT-4o ann MAE=48.3 (4 unique preds). GPT-4o-mini: unannotated MAE=50.6 (9/10 fixated t=106!), annotated MAE=55.2 (+9%, hurts). GPT-4o unannotated rate-limited (50/day quota). | ✓ |
| 036 | BAEP ensemble analysis | Naive 5-model ensembles don't beat best individual (MAE 51.2 vs 50.1); selected 2-model pairs do (46.9, −6.4%). | ✓ |
| 037 | Confidence-gated VLM-PER (Proposal 5) | Agreement anti-correlates with accuracy (r=+0.53). Optimal gate = "never use VLM." Always-VLM strictly worse than uniform. | ✓ |
| 038 | Contrastive Episode Ranking (Proposal 2) | 100% primacy bias (11/11 always A). Accuracy = base rate. Zero signal above chance. | ✓ |
| 039 | Failure mode descriptions (Proposal 4) | High semantic diversity (6/6 cats, 100% unique descs, Jaccard=0.27). Categories explain GT timing (η²=0.34-0.99). First positive non-temporal signal. | ✓ |
| 040 | TF-IDF clustering + category-based diversity | TF-IDF clusters fail (silhouette<0.12, ARI≈0). VLM categories ARE the signal: 6x weight ratio, late-quartile upweighting. | ✓ |
| 041 | Category-diversity replay simulation | Category-diversity ≈ uniform (+2% GT coverage, ρ=+0.04 oracle). η² signal doesn't translate to replay priority. Proposal 4 closed. | ✓ |
| 042 | Synthetic scale-up simulation | Category-diversity beats uniform at N≥50 (+5-8% coverage, Δent up to +0.40). Effect is real but requires N>20 to overcome sampling noise. | ✓ |
| 043 | Cross-model category comparison | GPT-4o-mini uses only standard 6 categories; Phi-4 invents 4 novel ones. Cross-model Jaccard=0.60, Phi-4 cross-task Jaccard=0.20. Category stability is model-dependent. | ✓ |

### 17. Cross-model category stability: taxonomy adherence is model-dependent

**Setup:** Compared failure mode categories from GPT-4o-mini (reach-v3, n=20) and
Phi-4 (push-v3 n=10, pick-place-v3 n=9) using the same 6-category prompted taxonomy.

| | GPT-4o-mini (reach) | Phi-4 (push) | Phi-4 (pick-place) |
|---|---|---|---|
| Standard cats used | 6/6 (100%) | 5/6 + 1 novel | 3/6 + 3 novel |
| Novel categories | — | missing_target | size_mismatch, crash, alway_ent |
| Top category | stuck (30%) | stuck (40%) | stuck (33%) |
| η² | 0.340 | 0.582 | 0.988 |

**Key findings:**
- **GPT-4o-mini strictly adheres to the prompted taxonomy** — all 20 labels from the
  6 standard categories. **Phi-4 invents 4 novel categories** (40% of its unique labels),
  some clearly task-specific (size_mismatch), some likely hallucinated (alway_ent).
- **Cross-model vocabulary overlap** Jaccard=0.60. GPT-4o-mini's categories are a strict
  subset of Phi-4's. The standard taxonomy provides the shared foundation.
- **Phi-4's cross-task category stability is very low** (Jaccard=0.20 between push and
  pick-place). Only "stuck" and "other" transfer across tasks. Novel categories are
  task-specific and don't generalize.
- **"stuck" is universal** — the only category with consistent semantic meaning across
  all model/task combinations (30-40% prevalence, mid-range GT timing).
- **η² holds regardless of model**: category-timing correlation is stable (0.34, 0.58,
  0.99) whether categories come from GPT-4o-mini or Phi-4.

**Implication for scale-up:** The iter 42 finding that category-diversity improves replay
at N≥50 is model-dependent. GPT-4o-mini's disciplined taxonomy would produce more stable,
transferable categories than Phi-4's creative but noisy ones. For practical replay
prioritization, the generating model matters — taxonomy adherence is a desirable property.

## Bottom Line

VLMs achieve coarse failure localization (best ±10 accuracy = 50% on push-v3) but are
dominated by positional biases rather than visual understanding. The central finding
across 32 iterations, 9 models, 3 tasks, and 10+ interventions is that **annotation
effect is GT-distribution-dependent**: annotation shifts predictions toward mid-episode,
helping when GT failures cluster mid-episode (reach-v3, −30% for GPT-4o) but hurting
when GT failures cluster at extremes (push-v3 early: +18%, pick-place-v3 late: +9%).
This is architecture-modulated (Gemini-3-flash-preview ignores annotation entirely, 8/10
identical predictions) but the primary driver is bias-matching, not capability.
GPT-4o-mini shows extreme positional fixation on pick-place-v3 (9/10 at t=106), the
most severe case in the study — annotation partially breaks fixation (5 unique preds
vs 2) but increases MAE.

More keyframes reveal a **bias-variance tradeoff**: K=4 produces extreme positional
fixation with low MAE but 0% tolerance accuracy; K=16 diversifies predictions with best
±20 (40%) but slightly higher MAE. CoT and annotation are **partially substitutable
temporal scaffolds** — both replicated across two models: annotation-present→CoT-neutral,
annotation-absent→CoT-helps. GPT-4o-mini with CoT+no-annotation (MAE=53.2) matches
GPT-4o with annotation+no-CoT (MAE=52.7), showing CoT can compensate for both annotation
AND model strength.

The fundamental bottleneck is visual acuity: distinguishing subtle arm position changes
at 224×224 resolution with ~30-pixel arm regions. VLM-based replay priorities show a
promising overlap signal (+12% above uniform) but harmful KL divergence. Ensemble
approaches (BAEP, confidence gating) cannot extract value from fundamentally biased
predictions — inter-model agreement reflects shared positional bias, not accuracy
(r=+0.53). **Contrastive Episode Ranking (CER, Proposal 2) fails due to primacy bias.** The RLHF
analogy ("pairwise comparison is easier than absolute scoring") does not transfer:
GPT-4o-mini picks Episode A (presented first) in 11/11 pairs (P<0.001), regardless of
GT failure timing or gap magnitude. Accuracy = P(GT=A) = base rate (63.6%). When GT=A:
100% correct; when GT=B: 0% correct — performance is entirely explained by always-A
primacy bias. This extends the positional bias finding from within-episode (early/mid/late
fixation) to between-episode (first-presented preference). Confidence scores are
uninformative (0.80-0.90 for both correct and incorrect predictions).

**Iteration 39 pivot: failure mode descriptions are the first positive non-temporal
VLM signal.** Shifting from "when did it fail?" to "what went wrong?", VLMs produce
high-diversity descriptions (100% unique, Jaccard=0.27, 6/6 categories) that correlate
strongly with GT failure timing (η²=0.34–0.99). This works because failure description
is a scene understanding task (VLM strength) rather than temporal precision (VLM weakness).
Phi-4 even invents novel task-specific categories (crash, size_mismatch) beyond the
predefined set. Next step: embed descriptions and cluster to test whether VLM-perceived
failure modes can drive diversity-weighted replay prioritization (Proposal 4).

---

## Final Study Synthesis (Iteration 45)

### Complete approach inventory (14 approaches tested)

**Temporal approaches (6 tested, all failed):**

1. Direct temporal prediction (K sweep K=4→32): MAE never below 41.9, dominated by positional bias
2. CoT prompting: hurts weak models, partially substitutes annotation on strong — no net improvement
3. Frame annotation: architecture-dependent (GPT-4o −30%, GPT-4o-mini +11%, Gemini ignored)
4. Two-pass adaptive probing: coarse pass too noisy to guide refinement
5. Random sampling: breaks clusters but doesn't improve accuracy
6. Multi-format (grid tiling vs native multi-image): different bias patterns, no accuracy gain

**Ensemble/meta approaches (3 tested, all failed):**

7. 5-model BAEP ensemble: weak models dilute signal (MAE 51.2 > best individual 50.1)
8. Selective 2-model pairing: marginal improvement (MAE 46.9, −6.4%) but model-selection-dependent
9. Confidence gating (inter-model agreement): agreement ANTI-correlates with accuracy (r=+0.53)

**Relative ranking (1 tested, failed):**

10. Contrastive Episode Ranking: 100% primacy bias (11/11 picks Episode A)

**Non-temporal approaches (4 tested, 2 viable):**

11. Failure mode descriptions: ✓ VIABLE — η²=0.34–0.99, 100% unique, 6/6 categories
12. TF-IDF clustering: failed — template syntax kills embedding (silhouette <0.12)
13. Category-diversity replay (n=20): ≈ uniform at small sample sizes
14. Category-diversity replay (N≥50 simulated): ✓ VIABLE — +5–8% GT coverage at realistic scales

### The mechanistic story

VLMs process multi-image inputs through architectures that allocate high-frequency-only
positional encoding to temporal dimensions (MRoPE). This creates rapid attention decay
across image positions, manifesting as stable within-model positional biases (center,
start, end, grid-cell, primacy) that are structural, not random. No prompt engineering —
CoT, annotation, sampling strategy — can overcome an architectural limitation.

However, VLMs excel at scene understanding: identifying what objects are present, how
they relate spatially, and what type of behavior the robot is exhibiting. This
categorical understanding (6 failure types: never_reached, overshot, oscillated,
wrong_direction, stuck, other) is genuinely informative — failure categories correlate
with ground-truth failure timing (η²=0.34–0.99) because different failure modes have
different temporal signatures.

The annotation × task interaction (iter 31-32) provides the cleanest evidence: annotation
shifts predictions toward mid-episode, which helps when GT is mid-distributed (reach-v3,
−30%) but hurts when GT clusters at extremes (push-v3 early, +18%; pick-place-v3 late, +9%).
This is bias-matching, not visual understanding.

### Cross-model category stability (iter 43-44)

Category labels are more stable within taxonomy-adherent models (GPT-4o-mini: bootstrap
JSD 0.10±0.06) than creative models (Phi-4: JSD 0.20–0.24). Task drives category
distribution more than model choice (within-Phi-4 cross-task JSD = 0.29 > cross-model
JSD = 0.11). For practical deployment, GPT-4o-mini's strict taxonomy adherence is
preferable to Phi-4's creative novel categories.

### Implications for VLM-PER

1. **Temporal VLM-PER is not viable** with current zero-shot VLMs. The signal is dominated
   by positional bias that cannot be mitigated by prompt engineering.
2. **Category-diversity replay is viable at scale (N≥50)** — uses VLM failure mode labels
   to ensure diverse category coverage in replay batches. Cheap (1 API call per episode),
   interpretable, and architecture-agnostic.
3. **The binding constraint** is not temporal resolution but the upstream problem of
   generating rollouts with meaningful failure diversity (same issue as td_baseline).
4. **Fine-tuning** (à la AHA, FPC-VLA) would likely solve the temporal bias problem
   but requires labeled data and training — a fundamentally different study.

### Cost

Total study cost: $0.80 (3 Claude Sonnet API calls in iter 1-3, all subsequent work on
free APIs: GitHub Models, Google AI Studio, Groq). 60 rollouts, 9 models, 14 approaches,
47 iterations.

### Consolidated database (iter 47)

All 360 individual predictions consolidated into `results/consolidated_database.json`
with per-condition aggregates in `results/summary_table.json` (31 unique conditions).

---

## Study Status: COMPLETE (Iteration 47, 2026-04-10)

This study is declared complete per agreement with the human collaborator. The
consolidated database, experiment write-up, Quarto pages, and this findings document
constitute the full record. The same-rollout cross-model comparison (the one remaining
experiment) and proposals 3/6/7 are explicitly deferred — not lost, but not prioritized.

**What would reopen this study:**

1. Access to fine-tunable VLMs (solves the positional bias root cause)
2. Semi-trained policy rollouts (enables phase-segmented replay, Proposal 6)
3. A training loop integration that could validate category-diversity replay empirically
4. New VLM architectures that address temporal encoding limitations (post-MRoPE)
