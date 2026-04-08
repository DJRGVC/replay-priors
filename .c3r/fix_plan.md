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
- [x] Collect ~20 failure rollouts on each of 2-3 MetaWorld tasks — DONE iter_001: 60 rollouts across reach/push/pick-place
- [ ] Build a single-file VLM client: takes (task_description, list_of_keyframe_PIL_images, K) → {failure_timestep_index,      
confidence, one_sentence_rationale}. Start with `claude-sonnet-4-6` via the Anthropic messages API (vision input).             
`ANTHROPIC_API_KEY` is in env.
- [ ] Implement uniform K-frame keyframe sampling plus one alternative (e.g. evenly spaced with first/last pinned). Make K a   
parameter; K=8 default.                                                                                                        
- [ ] Single-task accuracy probe: run the client over all rollouts on ONE task with K=8, default prompt, single model. Report 
absolute timestep error vs GT, window IoU at ±5 / ±10 tolerance, latency/call, $/call. Drop a 1-page NOTES.md. This is the     
"does the idea work at all" checkpoint that de-risks the proposal's central pipeline before Parshawn wires it into SAC — 
surface loudly in INBOX.md if accuracy is at chance.                                                                           
- [ ] If step 4 is non-trivial, sweep K ∈ {2, 4, 8, 16} on the same task and plot accuracy vs K and $ vs K. One figure in 
`studies/vlm_localization_probe/figures/`. Tells us which K Parshawn should even bother plugging into the replay-prioritization
pipeline.
- [ ] Open questions to surface (not block on): definition of "failure" for time-out tasks with no clear failure event;        
vision-only vs vision+proprio-as-text; free-form rationale vs forced JSON output.                                              
- [ ] Budget: soft cap $20 in Anthropic API spend overnight. If approaching it, pause and ask in the Discord thread. Out of 
scope tonight: wiring into SAC/replay buffers (Parshawn's territory), multi-model sweeps before single-model accuracy is       
established.

