# fix_plan.md — experiment queue for vlm_probe
#
# Replace this preamble with 3-5 concrete starting tasks (one per line,
# as a markdown bullet). Lines starting with # are kept as comments.
# Agents read the TOP of this file at the start of every iteration.
#
# Example format:
#   - [ ] Task one — one full sentence, no line breaks
#   - [ ] Task two
#
# Save and exit your editor when done. Empty file = agent picks its own direction.

- [ ] just like the td agent, occasionally add relevant literature to discord so i can review it.
- [x] Collect ~20 failure rollouts on each of 2-3 MetaWorld tasks — DONE iter_001
- [x] Build VLM client + keyframe sampling — DONE iter_002
- [x] Single-task accuracy probe on reach-v3 with K=8 — DONE iter_002 (MAE=41.9, ±10=20%)
- [x] K sweep on reach-v3 — DONE iter_003 (K=4/8/16/32: no improvement with more frames, MAE flat ~42-52)
- [ ] **PRIORITY: Add Gemini 2.5 Flash backend to vlm_client.py** — free tier, 0 cost, 250 RPD. See FREE_VLM_OPTIONS.md.
- [ ] Re-run K sweep on reach-v3 with Gemini Flash to validate cross-model consistency
- [ ] Run probe on push-v3 and pick-place-v3 (may have more visually distinct failure modes)
- [ ] Try "pinned" sampling strategy (first+last frames pinned) vs uniform
- [ ] Try improved prompt: provide proprio state as text, or use chain-of-thought prompting
- [ ] Open questions: definition of "failure" for time-out tasks; vision-only vs vision+proprio-as-text
- [ ] Budget: $0 going forward. DO NOT use Anthropic API key (costs real money). Use only free APIs.

