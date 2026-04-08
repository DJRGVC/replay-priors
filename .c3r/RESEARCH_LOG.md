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
