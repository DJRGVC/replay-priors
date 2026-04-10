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

- **lit_review2** (generic, parent=td_baseline) — status=running, iter=#1, last=62h ago  ⚠ STALE — consider killing
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
- **status**: running · iter #24 · ctx 100%
- **last iter**: 44h ago

### Recent commits on `agent/vlm_probe`
```
96f8b23 Iteration 27: Literature update (5 papers) + study pause per Daniel
4186349 Iteration 26: GPT-4o-mini CoT×annotation 2×2 factorial — mirror-image interaction across model strength
314b67f Iteration 26: GPT-4o-mini CoT×annotation 2×2 (CoT+unannotated best at 53.2, annotation interferes with CoT on mid-tier, substitutability replicates)
9a4d6dd Iteration 25: Phi-4 annotation ± (no effect, too weak) + GPT-4o-mini K sweep (K=16 best, more frames help mid-tier)
3586bc1 Iteration 24: HTML report update with GPT-4o results + CoT×annotation substitutability
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
studies/vlm_localization_probe/.gitignore
studies/vlm_localization_probe/FINDINGS.md
studies/vlm_localization_probe/FREE_VLM_OPTIONS.md
studies/vlm_localization_probe/RESULTS_SUMMARY.md
studies/vlm_localization_probe/analyze_gt_quality.py
studies/vlm_localization_probe/build_report.py
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/figures/k_sweep_reach_v3.png
studies/vlm_localization_probe/plot_k_sweep.py
studies/vlm_localization_probe/priority_score.py
studies/vlm_localization_probe/regenerate_meta.py
studies/vlm_localization_probe/results/cot_gpt4o/results.json
studies/vlm_localization_probe/results/cot_gpt4o_mini/results.json
studies/vlm_localization_probe/results/cot_gpt4o_mini_noannotate/results.json
studies/vlm_localization_probe/results/cot_gpt4o_noannotate/results.json
studies/vlm_localization_probe/results/cot_llama90b/results.json
studies/vlm_localization_probe/results/cot_phi4/results.json
studies/vlm_localization_probe/results/gpt4o/results.json
studies/vlm_localization_probe/results/gpt4o_mini/results.json
studies/vlm_localization_probe/results/gpt4o_mini_noannotate/results.json
... and 18 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PAUSED
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
```

## quarto-fixer
- **role**: quarto-fixer
- **focus**: Fix failed Pages build (run 24261865659)
- **status**: running · iter #0 · ctx 0%

### Recent commits on `agent/quarto-fixer`
```
3b59ea0 scaffold replay-priors umbrella
```
### Files modified on `agent/quarto-fixer` (relative to `c3r/replay-priors`)
_(none)_

