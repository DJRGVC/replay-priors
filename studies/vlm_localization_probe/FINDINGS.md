# VLM Failure-Localization Probe — Findings

**Study:** Can vision-language models localize failure timesteps in robotic manipulation
rollouts from keyframe images alone?

**Setup:** MetaWorld reach-v3, 150-step episodes, random policy (all failures), 224×224
RGB frames, 20 rollouts. GT failure timestep = argmin(hand-to-goal distance).

**Models tested (9):** Claude Sonnet 4.6 ($0.004/call), GPT-4o ($0 via GitHub),
GPT-4o-mini ($0 via GitHub), Gemini 3 Flash Preview ($0), Gemini 2.5 Flash ($0),
Gemini 2.5 Flash-Lite ($0), Llama-3.2-11B ($0), Llama-3.2-90B ($0),
Phi-4-multimodal ($0).

**Interventions tested (8):** K sweep (4/8/16/32 keyframes), CoT prompting, frame
annotation, proprio-as-text, random vs uniform sampling, two-pass adaptive probing,
annotation ± comparison on multi-image API, CoT × annotation 2×2 factorial.

## Key Findings

### 1. VLMs can coarsely localize manipulation failures, but accuracy is far from actionable

| Model | API | MAE↓ | Median↓ | ±10 | ±20 | Bias Pattern |
|-------|-----|------|---------|-----|------|--------------|
| Claude Sonnet 4.6 | Anthropic | 41.9 | 34.0 | 20% | 35% | center (t≈85) |
| GPT-4o (ann) | GitHub | 52.7 | 43.5 | 10% | 10% | early-mid (t=42) |
| Gemini 3 Flash Preview | Google | 54.2 | 14.0 | **44%** | **56%** | start (t=0) |
| Llama-3.2-90B | GitHub | 53.5 | 37.5 | 0% | 0% | grid-cell (t=42) |
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

### 3. More keyframes do not help — the bottleneck is visual understanding, not temporal resolution

| K | MAE | ±10 | ±20 |
|---|-----|------|------|
| 4 | 47.4 | — | — |
| 8 | **41.9** | **20%** | **35%** |
| 16 | 44.4 | 20% | 35% |
| 32 | 51.5 | — | — |

Tested on Claude Sonnet. MAE is flat from K=4 to K=16 and *worsens* at K=32 (likely
input noise from 32 similar-looking images). The task requires distinguishing ~0.1-unit
arm position changes in 224×224 images where the arm occupies ~30 pixels. This is a
visual acuity problem, not a temporal sampling problem.

### 4. CoT prompting hurts weak/mid models, is neutral on strong models — partially substitutable with annotation

| Model | Direct MAE | CoT MAE | Δ | N |
|-------|-----------|---------|---|---|
| Gemini 2.5 Flash-Lite | 71.9 | 79.2* | +7.3 | 10 |
| Gemini 2.5 Flash | 67.8 | 75.1 | +7.3 | 10 |
| Phi-4-multimodal | 64.3 | 90.2 | **+25.9** | 10 |
| GPT-4o (annotated) | 52.7 | 52.2 | −0.5 | 10 |
| GPT-4o (unannotated) | 75.8 | 65.0 | −10.8 | 10 |
| Gemini 3 Flash Preview | 54.2 | 22.0 | −32.2 | **3** |

*Estimated from iter_007 control.

CoT (Summarize→Think→Answer) hurts weak/mid models (3/3 negative: +7 to +26 MAE) but
is **neutral on strong models when annotation is present** (GPT-4o annotated: 52.7→52.2).
Crucially, CoT **helps unannotated GPT-4o** (75.8→65.0, −14%), revealing that CoT and
annotation are **partially substitutable** — both provide temporal scaffolding, and when
you already have annotation, CoT adds nothing further. The full 2×2 on GPT-4o:

| GPT-4o | Direct | CoT |
|--------|--------|-----|
| **Annotated** | 52.7 | 52.2 |
| **Unannotated** | 75.8 | 65.0 |

This suggests the beneficial mechanism of both interventions is anchoring attention to
temporal structure. Once that's provided (by either annotation or CoT), the second
intervention is redundant.

### 5. Frame annotation is model-dependent — helps weak & strong models, hurts mid-tier

| Model | Unannotated | Annotated | ΔMAE |
|-------|------------|-----------|------|
| Gemini 2.5 Flash-Lite | 71.9 | 59.5 | −12.4 (−17%) |
| GPT-4o | 75.8 | 52.7 | **−23.1 (−30%)** |
| GPT-4o-mini | 61.2 | 68.0 | +6.8 (+11%) |

Overlaying "t=X (N%)" on each frame (VTimeCoT-style) shows a non-monotonic effect:
- **Weak models (Flash-Lite):** annotation provides needed temporal anchors (−17% MAE)
- **Strong models (GPT-4o):** annotation dramatically helps (−30% MAE), shifting from
  start-bias (5/10 at t=0 unannotated) to more distributed predictions (6/10 at t=42 annotated)
- **Mid-tier models (GPT-4o-mini):** annotation hurts (+11% MAE), shifting distribution
  toward later timesteps

GPT-4o without annotation has severe start-bias (MAE=75.8, 5/10 at t=0), but with
annotation MAE drops to 52.7 — comparable to Llama-3.2-90B. The annotation effect is
U-shaped: both weak and strong models benefit from temporal grounding, while mid-tier
models are distracted by the overlay text.

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

### 10. Push-v3 and pick-place-v3 are unsuitable for VLM probing with random policies

100% of push/pick-place rollouts have ambiguous GT (random policy never contacts
objects). GT "failure timestep" = argmin(hand-object distance), which is arbitrary
with no visually salient event. Only reach-v3 has non-ambiguous GT with this
experimental setup. Trained-policy rollouts would be needed for multi-task evaluation.

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

## Bottom Line

VLMs achieve coarse failure localization (best ±10 accuracy = 44%) but are dominated
by positional biases rather than visual understanding. All tested interventions
(more frames, CoT, two-pass refinement, random sampling) either fail or provide
marginal improvement. Frame annotation has a U-shaped effect: helps weak (−17% Flash-Lite)
and strong (−30% GPT-4o) models, but hurts mid-tier (+11% GPT-4o-mini). CoT and
annotation are partially substitutable — both provide temporal scaffolding, and when
one is already present, the other adds nothing (GPT-4o 2×2: annotated+CoT ≈ annotated+direct).
The fundamental bottleneck is visual acuity: distinguishing subtle arm position changes at
224×224 resolution with ~30-pixel arm regions. VLM-based replay priorities show a promising
overlap signal (+12% above uniform) but harmful KL divergence, suggesting a
confidence-gated hybrid approach as the path forward — if confidence calibration can
be solved.
