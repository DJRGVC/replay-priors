# VLM Failure-Localization Probe — Results Summary

Task: reach-v3 (MetaWorld), 150-step episodes, random policy, K=8 uniform keyframes unless noted.

## Model Comparison (K=8, direct prompt, no annotation)

| Model | N | Valid | MAE | Median | ±5 | ±10 | ±20 | Bias | Cost/call |
|-------|---|-------|-----|--------|-----|------|------|------|-----------|
| claude-sonnet-4-6 | 20 | 20 | 41.9 | 34.0 | 0% | 20% | 35% | center (t≈85) | $0.004 |
| gemini-3-flash-preview | 10 | 9 | 54.2 | 14.0 | 22% | 44% | 56% | start (t=0) | $0 |
| gemini-2.5-flash | 10 | 10 | 67.8 | — | 10% | 20% | 30% | end (t≈149) | $0 |
| gemini-2.5-flash-lite | 20 | 12 | 95.2 | 107.5 | — | 5% | 10% | late | $0 |

**Key finding:** gemini-3-flash-preview has best ±10 (44%) and median (14), but worst-case catastrophic errors inflate MAE above Claude. Claude has consistent center-bias. Each Gemini model has a distinct positional bias (start/end/late).

## K Sweep (claude-sonnet-4-6, direct, no annotation)

| K | N | MAE | ±10 | ±20 | Note |
|---|---|-----|------|------|------|
| 4 | 10 | 47.4 | — | — | partial run (rate limits) |
| 8 | 20 | 41.9 | 20% | 35% | best overall |
| 16 | 20 | 44.4 | 20% | 35% | no improvement |
| 32 | 20 | 51.5 | — | — | worst — more frames hurts |

**Finding:** More keyframes do NOT help. MAE is flat K=4–16 and worsens at K=32. Bottleneck is semantic understanding, not temporal resolution.

## Prompt Style: CoT vs Direct

| Model | Style | N | MAE | ±5 | ±10 | ±20 |
|-------|-------|---|-----|-----|------|------|
| gemini-2.5-flash | direct | 10 | 67.8 | 10% | 20% | 30% |
| gemini-2.5-flash | CoT | 10 | 75.1 | 20% | 20% | 20% |
| gemini-3-flash-preview | CoT | 3 | 22.0 | — | 67% | — |
| gemini-3-flash-preview | direct | 9 | 54.2 | 22% | 44% | 56% |

**Finding:** CoT is model-strength-dependent. Hurts weak models (+7.3 MAE on 2.5-flash, "thinking drift"). Suggestive improvement on strong models (22.0 vs 54.2 on 3-flash, but n=3). Needs larger n to confirm.

## Frame Annotation (VTimeCoT-style "t=X (N%)" overlay)

| Model | Annotated | N | MAE | Median | ±5 | ±10 | ±20 |
|-------|-----------|---|-----|--------|-----|------|------|
| gemini-2.5-flash-lite | No | 10 | 71.9 | 82.0 | 0% | 10% | 10% |
| gemini-2.5-flash-lite | Yes | 10 | 59.5 | 56.5 | 10% | 20% | 20% |

**Finding:** Annotation improves MAE by 12.4 points (-17%), ±10 doubles (10%→20%) on weak model. Helps most when GT is near episode boundaries. Not yet tested on stronger models.

## Proprio-as-Text (XYZ coords at each keyframe)

| Model | Annotated | Proprio | N | Valid | MAE |
|-------|-----------|---------|---|-------|-----|
| gemini-2.5-flash-lite | Yes | No | 10 | 10 | 59.5 |
| gemini-2.5-flash-lite | Yes | Yes | 10 | 2 | 107.5 |

**Finding:** Negative signal on flash-lite (MAE 107.5 vs 59.5), but n=2 valid due to severe rate-limiting. Inconclusive.

## Rate Limiting Status (Gemini free tier)

All Gemini models share a Google Cloud project. Practical limits:
- **gemini-2.5-flash-lite**: 20 RPD, ~2 RPM with images
- **gemini-2.5-flash**: 20 RPD, ~2 RPM with images
- **gemini-3-flash-preview**: 20 RPD, ~1-2 RPM with images (heaviest limiting)
- **gemini-2.0-flash**: 0 free RPD (unusable)

Rate limiting is the primary bottleneck. Exploring Groq (Llama 4 Scout, 30 RPM free) as alternative.

## Open Experiments (quota-gated)

1. Annotation on gemini-3-flash-preview and gemini-2.5-flash
2. Complete CoT comparison on gemini-3-flash-preview (n≥9)
3. Push-v3 / pick-place-v3 task diversity
4. Pinned sampling strategy (first+last frames anchored)
5. Retest proprio-as-text with n≥5 valid
6. Groq Llama 4 Scout as rate-limit-free alternative
