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
