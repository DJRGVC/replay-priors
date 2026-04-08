# SIBLINGS — auto-regenerated at the start of each iteration

This file is your snapshot of what every OTHER agent in this project
has been doing. Use it at the top of every iteration (after reading
INBOX.md) to stay coordinated.

**To actually read a file from a sibling's branch** (without
checking it out — they're on separate branches to avoid conflicts):
```
git show agent/<sibling-name>:path/to/file
```

**To see what a sibling has changed since you last looked:**
```
git log agent/<sibling-name> --since='1 hour ago' --name-status
git diff HEAD agent/<sibling-name> -- path/
```

**To push a handoff file to siblings:** commit it on your own branch
and reference it in your Discord thread or in your next log entry.
Siblings will see it in their next SIBLINGS.md refresh.

---

## YOUR CHILDREN — agents YOU spawned and YOU must manage

These are sub-agents you spawned (directly or transitively).
**YOU are responsible for killing them when their task is done,
they get stuck, or they exceed their useful budget.** Each child
also has a hard iteration cap and will self-kill at MAX_ITERATIONS,
but that's a safety net — proactive management is your job.

- **lit_review2** (generic, parent=td_baseline) — status=stopped, iter=#1, last=13h ago  ⚠ STALE — consider killing  (already stopped)
  Focus: Literature review agent. Use WebSearch to find and summarize recent papers (NeurIPS, ICLR, ICML 2023-2026, DeepMind, Google) on: (1) alternatives to TD-error prioritized experience replay in sparse-reward RL, (2) VLM/LLM-guided exploration, reward shaping, or hindsight relabeling, (3) foundation-model-based replay prioritization. Write findings to studies/td_error_baseline/LIT_REVIEW.md. Focus on web search and writing — no code, no training.

**Decision rules** (apply at the top of every iteration):
1. If a child's last RESEARCH_LOG entry says its task is done, kill it: `$C3R_BIN/c3r kill <name>`
2. If a child has been stale (no iter for >2 hours), kill it.
3. If a child's fail_streak ≥ 3 in state.json, investigate or kill it.
4. Otherwise, leave it running and check again next iteration.

---

## SIBLINGS — peers you do NOT manage (other agents' work)

## vlm_probe
- **role**: generic
- **focus**: Bootstrap studies/vlm_localization_probe: collect a small set of MetaWorld failure rollouts on 2-3 tasks, build a thin VLM E  client (Claude + one other) that takes K keyframes plus a task description and predicts the failure timestep window, and run a E  sweep over K, prompt format, model, and task reporting localization accuracy, latency, and cost. Do not touch SAC or replay E  buffers — this study is pure VLM probing.
- **status**: running · iter #16 · ctx 0%
- **last iter**: 3h ago

### Recent commits on `agent/vlm_probe`
```
aa14bd3 iter_018: GPT-4o-mini native multi-image probe (MAE=68.0, late-bias confirms positional bias is intrinsic not grid-artifact)
efcd098 iter_017: FINDINGS.md synthesis — 10 key findings across 7 models and 6 interventions, Gemini image quotas still blocked
5f8bb2e iter_016: CoT on Phi-4 NEGATIVE (MAE 64.3→90.2), Gemini quotas reset, vlm_litreview removed
85acc68 iter_015: two-pass adaptive probing on Llama-3.2-90B (NEGATIVE: MAE 69.8→71.3, refinement worsens 6/10 — coarse pass too inaccurate to center refinement window)
097d91e iter_014: random vs uniform sampling on Llama-3.2-90B (MAE 64.7 vs 63.8, no difference — sampling strategy not the bottleneck)
```
### Files modified on `agent/vlm_probe` (relative to `c3r/replay-priors`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PAUSED
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/fix_plan.md
.claude/settings.json
.gitignore
studies/vlm_localization_probe/FINDINGS.md
studies/vlm_localization_probe/FREE_VLM_OPTIONS.md
studies/vlm_localization_probe/RESULTS_SUMMARY.md
studies/vlm_localization_probe/analyze_gt_quality.py
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/figures/k_sweep_reach_v3.png
studies/vlm_localization_probe/plot_k_sweep.py
studies/vlm_localization_probe/priority_score.py
studies/vlm_localization_probe/regenerate_meta.py
studies/vlm_localization_probe/results/cot_llama90b/results.json
studies/vlm_localization_probe/results/cot_phi4/results.json
studies/vlm_localization_probe/results/gpt4o_mini/results.json
studies/vlm_localization_probe/results/gpt4o_mini_test/results.json
studies/vlm_localization_probe/results/k_sweep_consolidated.json
studies/vlm_localization_probe/results/k_sweep_k32/results.json
studies/vlm_localization_probe/results/k_sweep_reach/results.json
studies/vlm_localization_probe/results/phi4_probe/results.json
studies/vlm_localization_probe/results/random_sampling/results.json
studies/vlm_localization_probe/results/random_sampling_control/results.json
studies/vlm_localization_probe/results/results.json
... and 4 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PAUSED
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
```

