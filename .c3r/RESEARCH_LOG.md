# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

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
Result:     **CoT is model-strength-dependent — hurts weak models, may help strong ones.**
  - gemini-2.5-flash direct: MAE=67.8, ±5=10%, ±10=20%, ±20=30% (strong end-bias, predicts t=149 repeatedly)
  - gemini-2.5-flash CoT: MAE=75.1, ±5=20%, ±10=20%, ±20=20% — **CoT worsened MAE by +7.3** despite producing coherent visual descriptions. Confirms §12f "thinking drift" warning: model confabulates plausible reasoning chains but still defaults to positional priors (center/end bias, predictions cluster at 63/85/127).
  - gemini-3-flash-preview CoT (n=3 only): errs=10,48,8 → MAE=22.0, ±10=67%. **Suggestive improvement over direct MAE=54.2**, but n=3 insufficient for significance. Start-bias partially mitigated (1/3 predict t=0 vs 3/9 in direct).
  - Rate limits: gemini-2.5-flash exhausted 20 RPD; gemini-3-flash-preview severely rate-limited with CoT (image+free-text output hits tighter quota). Could only get 3 valid CoT predictions.
  - gemini-2.5-flash also tested as new direct baseline: much weaker than gemini-3-flash-preview (MAE 67.8 vs 54.2), strong end-bias vs start-bias.
Decision:   Next iter: complete gemini-3-flash-preview CoT comparison (need n=7 more to match direct n=9). Rate limits reset daily — schedule runs accordingly. If CoT confirms improvement on gemini-3-flash-preview (n≥9), the finding is: structured reasoning only helps models strong enough to ground their CoT in visual evidence. Also try push-v3 task for visual diversity.

## iter_007 — frame index annotation (VTimeCoT-style)  (2026-04-08T16:00Z)
Hypothesis: Overlaying timestep index + progress fraction ("t=X (N%)") on keyframe images will reduce positional bias and improve failure-timestep localization (lit review §15g estimates +4-8% ±10 gain).
Change:     Added `annotate_frame()` to vlm_client.py (PIL text overlay with DejaVuSans-Bold, black outline for readability). Added `--annotate` flag to run_probe.py. Ran paired comparison on gemini-2.5-flash-lite, K=8, uniform, reach-v3 (n=10 each, same rollouts).
Command:    python run_probe.py --tasks reach-v3 --K 8 --models gemini-2.5-flash-lite --strategies uniform --max-rollouts 10 --annotate  (then same without --annotate as control)
Result:     **Frame annotation improves weak-model localization:**
  - Unannotated control: MAE=71.9 (median 82.0), ±5=0%, ±10=10%, ±20=10%
  - Annotated:           MAE=59.5 (median 56.5), ±5=10%, ±10=20%, ±20=20%
  - **MAE improved by 12.4 points (-17%), ±10 accuracy doubled (10%→20%)**
  - Per-rollout: annotated better in 4/10, unannotated better in 2/10, tie in 4/10
  - Annotation helps most when GT is near episode boundaries (rollouts 004,007,009: gt=109,149,141)
  - Note: unannotated baseline (MAE=71.9) is better than iter_004 (MAE=95.2) because thinking_budget=0 fix now prevents parse errors
  - All quotas for gemini-2.5-flash and gemini-3-flash-preview exhausted (20 RPD each). Only flash-lite available.
  - Also shared lit review highlights to Discord (standing fix_plan task).
Decision:   Next iter: test annotation on stronger models (gemini-2.5-flash or gemini-3-flash-preview) once quota resets. The annotation-helps-weak-models finding aligns with §15d "Thinking Drifts" — visual anchors provide the strongest lift where the model has weakest internal temporal reasoning. Also: run push-v3 probe (task diversity) and complete CoT comparison. Gemini quotas are the binding constraint — all 3 models share the same project, 20 RPD each.

## iter_008 — proprio-as-text augmentation  (2026-04-08T08:16Z)
Hypothesis: Adding numeric end-effector + goal positions (XYZ coords) at each keyframe as text will improve localization by giving VLMs precise spatial info invisible in 224×224 images.
Change:     Implemented `extract_proprio_text()` in vlm_client.py and `--proprio` flag in run_probe.py. Proprio text appends per-keyframe lines like "t=X: hand=(x,y,z), goal=(x,y,z), dist=D" to the prompt. Tested on gemini-2.5-flash-lite, K=8, annotated+proprio, reach-v3, n=10.
Command:    python run_probe.py --tasks reach-v3 --K 8 --models gemini-2.5-flash-lite --strategies uniform --max-rollouts 10 --annotate --proprio
Result:     **Severe rate-limiting: only 2/10 calls succeeded (8 hit 429 after 5 retries).**
  - Valid predictions: rollout_000 err=117 (gt=10, pred=127), rollout_002 err=98 (gt=8, pred=106)
  - Valid-only MAE=107.5, ±10=0%, ±20=0%
  - **Comparison to iter_007 (annotated, no proprio, n=10): MAE=59.5, ±10=20%**
  - Proprio-as-text appears to HURT flash-lite: both valid predictions show large late-bias, worse than annotation-only
  - All Gemini tiers (flash-lite, 2.5-flash, 3-flash-preview) severely rate-limited today. gemini-2.5-flash and gemini-3-flash-preview returned 429 on first attempt.
  - **Caveat: n=2 valid is too small for significance.** The rate-limiting confound means we cannot cleanly attribute the degradation to proprio text vs bad luck on those 2 rollouts.
  - Also tested gemini-3-flash-preview and gemini-2.5-flash quotas — both immediately 429'd (quotas appear not to have reset despite new day).
Decision:   Rate limiting is the binding constraint across all Gemini tiers. Next iter: (1) build a comprehensive results summary table from all experiments so far while waiting for quotas, (2) check if there are other free VLM APIs (e.g. Groq with Llama Vision, Together.ai free tier) that could bypass Gemini limits, (3) when quotas reset, retest proprio on flash-lite with n≥5 valid to confirm/deny the negative signal. The proprio code is ready and tested — just needs API access.

## iter_009 — Groq backend + results summary + call-delay  (2026-04-08T19:30Z)
Hypothesis: Adding Groq (Llama 4 Scout, 30 RPM free tier) as a backend will bypass Gemini rate-limiting bottleneck.
Change:     Three improvements: (1) Added Groq backend to vlm_client.py (OpenAI-compatible API, Llama 4 Scout 17B vision model), (2) built RESULTS_SUMMARY.md consolidating all 8 iterations into comparison tables, (3) added --call-delay flag to run_probe.py for proactive inter-call spacing. Also confirmed Gemini image quotas are STILL exhausted (text-only works, image requests 429 on all tiers). Asked human for GROQ_API_KEY — timed out after 15 min.
Command:    python -c "from groq import Groq; print('SDK OK')" (install verified); API quota tests on all Gemini models
Result:     **Groq backend implemented but untested — no API key yet.** Gemini status: text-only calls succeed on 2.5-flash and 3-flash-preview, but ALL image requests return 429. Google appears to have separate quotas for text vs image inputs, and the image quota has not reset despite 16+ hours. This is a harder constraint than the documented 20 RPD. RESULTS_SUMMARY.md captures all findings: best model is gemini-3-flash-preview (±10=44% median=14), annotation helps weak models (-17% MAE), CoT is strength-dependent, more frames don't help. [fallback after timeout] Skipping Groq for now, waiting for Gemini quotas or human to add key.
Decision:   Next iter: (1) if GROQ_API_KEY appears in env, run Llama 4 Scout probe on reach-v3 K=8 annotated, (2) if Gemini image quotas reset, run annotation on gemini-2.5-flash (baseline from iter_006), (3) otherwise explore push-v3 task preparation or implement pinned sampling strategy. The Groq backend is code-complete and ready to run.
