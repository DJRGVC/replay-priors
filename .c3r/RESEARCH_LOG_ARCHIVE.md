# RESEARCH_LOG_ARCHIVE.md

Verbatim entries from compacted iterations. Permanent record — never delete.

---

## iter_001 — collect failure rollouts  (2026-04-07T23:50Z)
Hypothesis: A random policy on MetaWorld reach/push/pick-place will produce all-failure rollouts suitable for VLM probing.
Change:     Created collect_rollouts.py and collected 20 failure rollouts × 3 tasks (reach-v3, push-v3, pick-place-v3). 150 steps each, 224×224 RGB frames, proprio saved as .npy, meta.json with heuristic failure timesteps.
Command:    python collect_rollouts.py --tasks reach-v3 push-v3 pick-place-v3 --rollouts-per-task 20 --max-steps 150
Result:     60 rollouts (all failures, 0 successes — expected with random policy). 500MB total data. Failure timesteps: reach uses closest-approach heuristic (non-ambiguous), push/pick-place use closest-approach-to-object (ambiguous, marked as such). Push/pick-place random policy almost never contacts object.
Decision:   Next iteration: build the VLM client (fix_plan task 3). Single-file client that takes task description + K keyframes → predicted failure window. Start with Claude claude-sonnet-4-6 vision.

## iter_002 — VLM client + first reach-v3 probe  (2026-04-08T04:40Z)
Hypothesis: Claude Sonnet 4.6 can localize failure timesteps in MetaWorld reach-v3 rollouts from 8 uniformly-sampled keyframes with ±10 accuracy >50%.
Change:     Built vlm_client.py (Anthropic + OpenAI backends, keyframe sampling) and run_probe.py (sweep harness with metrics). Ran single-task probe on reach-v3, K=8, uniform sampling, claude-sonnet-4-6.
Command:    python run_probe.py --tasks reach-v3 --K 8 --models claude-sonnet-4-6 --strategies uniform
Result:     MAE=41.9 (median 34.0), ±5=0%, ±10=20%, ±20=35%, latency=9.1s/call, cost=$0.004/call ($0.075 total for 20 rollouts). Strong bias toward t=85 (middle keyframe). Model returns valid JSON after prompt fix.
Decision:   Next iter: sweep K ∈ {4, 8, 16, 32} on reach-v3 to see if more frames help. The 85-bias suggests model can't distinguish subtle arm position differences — higher K gives finer temporal resolution. Also try push-v3 which may have more visually distinct failure modes.

## iter_003 — K sweep + API cost halt + retry logic  (2026-04-08T05:20Z)
Hypothesis: Higher K (more keyframes) will improve failure-timestep localization by giving finer temporal resolution.
Change:     Swept K ∈ {4, 8, 16, 32} on reach-v3 with claude-sonnet-4-6. Added retry+backoff for rate limits. Also fixed format bug in run_probe.py (None pred formatting). CRITICAL: discovered API calls cost real money (~$0.80 total) — Max 20x does NOT cover API. Disabled API key, notified td_baseline.
Command:    python run_probe.py --tasks reach-v3 --K 4 8 16 32 --models claude-sonnet-4-6 --strategies uniform
Result:     K=4: MAE=47.4 (n=10, partial due to rate limits), K=8: MAE=41.9 (n=20), K=16: MAE=44.4 (n=20), K=32: MAE=51.5 (n=20). **More keyframes do NOT help — MAE is flat or worsens.** ±10 accuracy peaks at 20% for K=8/K=16. K=32 is worst (MAE=51.5). Strong center-bias persists across all K. Cost scales linearly: $0.003→$0.010/call. Figure saved to figures/k_sweep_reach_v3.png.
Decision:   Next iter: find FREE VLM alternatives (Gemini free tier, local open-source VLMs). Subagent researching options. Also need to try "pinned" sampling strategy and push-v3 task — but only once we have a free inference path. The K-sweep negative result is valuable: it suggests the bottleneck is semantic understanding of subtle manipulation failures, not temporal resolution.

## iter_004 — Gemini backend + flash-lite probe  (2026-04-08T06:10Z)
Hypothesis: Gemini 2.5 Flash-Lite (free tier, $0) can achieve comparable failure-timestep localization to Claude Sonnet on reach-v3.
Change:     Added Gemini backend to vlm_client.py (google-genai SDK, JSON response mode, thinking disabled). Spawned vlm_litreview subagent via c3r spawn for literature review. Set up GOOGLE_API_KEY in .c3r/env.sh (now gitignored). Ran probe with gemini-2.5-flash-lite on reach-v3 K=8.
Command:    python run_probe.py --tasks reach-v3 --K 8 --models gemini-2.5-flash-lite --strategies uniform
Result:     **gemini-2.5-flash-lite: MAE=95.2 (median 107.5), ±10=5%, ±20=10%.** Much worse than Claude Sonnet (MAE=41.9, ±10=20%). 8/20 rollouts returned None (parse errors). Strong late-bias (predicts timesteps 106-149). Latency: 1.0s/call (fast). Cost: $0. Also discovered gemini-2.5-flash free tier = 20 RPD (not 250 as documented), and thinking tokens truncate output — fixed with thinking_budget=0. Flash probe ran but all 20 were truncated (data lost to overwrite).
Decision:   Next iter: try gemini-2.5-flash (quota resets tomorrow) with fixed thinking_budget=0. Also try gemini-3-flash-preview or gemini-2.0-flash which may have different RPD limits. flash-lite is too weak for this task. Alternatively, try improved prompting with flash-lite (include proprio as text, CoT).

## iter_005 — gemini-3-flash-preview probe  (2026-04-08T00:30Z)
Hypothesis: Gemini 3 Flash Preview, which leads VideoZeroBench temporal grounding (27.9% tIoU), will outperform Claude Sonnet and Gemini flash-lite on failure-timestep localization.
Change:     Tested gemini-3-flash-preview with existing prompt, K=8, uniform, reach-v3 (n=10, 9 valid after 1 503 error). Also confirmed: gemini-2.0-flash has 0 free RPD (unusable), gemini-2.5-flash quota still exhausted (20 RPD). Added gemini-2.0-flash and gemini-3-flash-preview to cost table.
Command:    PYTHONUNBUFFERED=1 python run_probe.py --tasks reach-v3 --K 8 --models gemini-3-flash-preview --strategies uniform --max-rollouts 10
Result:     **gemini-3-flash-preview: MAE=54.2 (median 14.0, valid-only n=9), ±5=22%, ±10=44%, ±20=56%.** Bimodal distribution: 44% of predictions within ±8 timesteps (err=2,5,6,8,14), but 33% catastrophic (err=88,107,117,141). Start-bias pattern (3/9 predict t=0 saying "arm remains stationary"). Latency: 82.8s mean (heavy rate limiting, 5 RPM?). Claude Sonnet comparison: Gemini has 2× better ±10 (44% vs 20%), much better median (14 vs 34), but worse MAE (54.2 vs 41.9) due to catastrophic misses. **No center-bias** — bias taxonomy confirmed: Claude→center, Gemini→start. Cost: $0.
Decision:   Next iter: implement structured CoT prompt (Summarize→Think→Answer from Time-R1 §12d + fraction-of-trajectory reframing from VTimeCoT §12c) and test on gemini-3-flash-preview. The start-bias could be addressed by forcing the model to describe each frame before predicting. Also: run remaining 10 rollouts on gemini-3-flash-preview to get n=20 for fair comparison.

## iter_006 — CoT prompt (Summarize→Think→Answer)  (2026-04-08T14:30Z)
Hypothesis: A structured CoT prompt (from Time-R1 §12d + VTimeCoT §12c) that forces frame-by-frame description before prediction will reduce positional bias and improve localization accuracy.
Change:     Added `prompt_style` parameter ("direct" vs "cot") to vlm_client.py and run_probe.py. CoT prompt has 3 steps: (1) SUMMARIZE each frame's visual content, (2) THINK about which transition shows failure, (3) ANSWER with fraction-of-trajectory then timestep. Disabled Gemini JSON response mode for CoT (needs free-text). Updated JSON parser to extract from longer CoT output. Tested on gemini-2.5-flash (n=10 each, direct vs CoT) and gemini-3-flash-preview (n=3 CoT only, rate-limited).
Command:    python run_probe.py --tasks reach-v3 --K 8 --models gemini-2.5-flash --strategies uniform --prompt-styles direct cot --max-rollouts 10
Result:     **CoT is model-strength-dependent — hurts weak models, may help strong ones.** gemini-2.5-flash direct: MAE=67.8, gemini-2.5-flash CoT: MAE=75.1 (+7.3). gemini-3-flash-preview CoT (n=3): MAE=22.0, suggestive but insufficient.
Decision:   Complete gemini-3-flash-preview CoT comparison (need n=7 more).

## iter_007 — frame index annotation (VTimeCoT-style)  (2026-04-08T16:00Z)
Hypothesis: Overlaying timestep index + progress fraction on keyframe images will reduce positional bias.
Change:     Added annotate_frame() to vlm_client.py (PIL text overlay). Ran paired comparison on gemini-2.5-flash-lite.
Command:    python run_probe.py --tasks reach-v3 --K 8 --models gemini-2.5-flash-lite --strategies uniform --max-rollouts 10 --annotate
Result:     **Frame annotation improves weak-model localization: MAE 71.9→59.5 (−17%), ±10 doubled (10%→20%).**
Decision:   Test annotation on stronger models once quota resets.

## iter_008 — proprio-as-text augmentation  (2026-04-08T08:16Z)
Hypothesis: Adding numeric end-effector + goal positions as text will improve localization.
Change:     Implemented --proprio flag in run_probe.py. Tested on gemini-2.5-flash-lite.
Result:     **Severe rate-limiting: only 2/10 succeeded. Valid-only MAE=107.5 — appears to HURT but n=2 too small.**
Decision:   Retest when quotas reset.

## iter_009 — Groq backend + results summary + call-delay  (2026-04-08T19:30Z)
Hypothesis: Groq (Llama 4 Scout, 30 RPM free) will bypass Gemini bottleneck.
Change:     Added Groq backend, built RESULTS_SUMMARY.md, added --call-delay flag.
Result:     **Groq implemented but untested — no API key.** Gemini image quotas still 429.
Decision:   Wait for keys or quota reset.

## iter_010 — priority score converter + priority quality analysis  (2026-04-08T21:00Z)
Hypothesis: Converting VLM predictions to Gaussian-kernel replay priorities will quantify downstream utility.
Change:     Built priority_score.py with KL divergence and top-K% overlap metrics.
Result:     **VLM priority KL always WORSE than uniform**, but **top-20% overlap beats uniform by +8-12%.**
Decision:   Downstream RL experiment needed to resolve overlap-vs-KL tension.

## iter_011 — two-pass adaptive probing + random sampling strategy  (2026-04-08T22:30Z)
Hypothesis: Coarse-to-fine two-pass will reduce MAE.
Change:     Built two_pass_probe.py and "random" sampling strategy. Code-only (all APIs blocked).
Result:     **Code verified, no API calls possible.**
Decision:   Test when APIs available.

## iter_012 — GT quality analysis + task suitability assessment  (2026-04-08T23:30Z)
Hypothesis: Push-v3 and pick-place-v3 have meaningful GT labels.
Change:     Built analyze_gt_quality.py.
Result:     **Push/pick-place UNSUITABLE with random policy** — 100% ambiguous GT (never_contacted_object).
Decision:   Need trained policy or focus on reach-v3.

## iter_013 — GitHub Models backend + Llama 3.2 Vision probe  (2026-04-08T04:45Z)
Hypothesis: Llama 3.2 Vision (11B/90B) via GitHub Models will bypass Gemini bottleneck.
Change:     Added GitHub Models backend with grid tiling for single-image APIs.
Result:     **Llama-3.2-11B MAE=72.9, 90B MAE=53.5.** Grid-position bias distinct from center/start/end biases. No rate limiting.
Decision:   Test random sampling and two-pass on 90B.

## iter_014 — random vs uniform sampling on Llama-3.2-90B  (2026-04-08T05:30Z)
Hypothesis: Random sampling breaks grid-position fixation.
Result:     **No difference: MAE 64.7 vs 63.8.** Random breaks clustering but model picks different wrong answers.
Decision:   Test two-pass or CoT instead.

## iter_015 — two-pass adaptive probing on Llama-3.2-90B  (2026-04-08T08:15Z)
Hypothesis: Coarse-to-fine will reduce MAE.
Result:     **Two-pass WORSENS: MAE 69.8→71.3.** Coarse pass too inaccurate to center refinement.
Decision:   Two-pass dead for this model class.

## iter_016 — CoT on Phi-4 + Gemini quota reset + subagent cleanup  (2026-04-08T14:45Z)
Hypothesis: CoT will improve Phi-4.
Result:     **CoT WORSENS Phi-4: MAE 64.3→90.2 (+40%).** CoT now 3/3 negative on mid/weak models.
Decision:   CoT definitively ruled out for mid/weak-tier.

## iter_017 — FINDINGS.md synthesis + Gemini quota still blocked  (2026-04-08T15:45Z)
Hypothesis: Gemini quotas have reset.
Result:     **Still blocked.** Wrote comprehensive FINDINGS.md with 10 key findings.
Decision:   Explore other free APIs.

## iter_018 — GPT-4o-mini native multi-image probe  (2026-04-08T16:30Z)
Hypothesis: GPT-4o-mini with native multi-image will reduce grid-position bias.
Result:     **MAE=68.0, still positional bias — confirming bias is intrinsic, not grid artifact.**
Decision:   Test annotation effect on GPT-4o-mini.

## iter_019 — GPT-4o-mini annotation ± comparison  (2026-04-08T17:30Z)
Hypothesis: Annotation will help GPT-4o-mini.
Result:     **Annotation HURTS: MAE 68.0→61.2 without annotation (−10%).** Overturns "annotation always helps."
Decision:   Annotation is model-dependent.

## iter_020 — self-contained HTML report interface  (2026-04-08T19:15Z)
Change:     Built report.html (2.1MB, embedded figures, interactive navigation).
Result:     Report generated successfully.
Decision:   Update when new experiments complete.
