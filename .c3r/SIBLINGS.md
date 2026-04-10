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

## SIBLINGS — peers you do NOT manage (other agents' work)

## vlm_probe
- **role**: generic
- **focus**: Bootstrap studies/vlm_localization_probe: collect a small set of MetaWorld failure rollouts on 2-3 tasks, build a thin VLM E  client (Claude + one other) that takes K keyframes plus a task description and predicts the failure timestep window, and run a E  sweep over K, prompt format, model, and task reporting localization accuracy, latency, and cost. Do not touch SAC or replay E  buffers — this study is pure VLM probing.
- **status**: idle · iter #29 · ctx 100%

### Recent commits on `agent/vlm_probe`
```
9b609ca Iteration 32: Pick-place-v3 task generalization — annotation effect is GT-distribution-dependent (bias-matching mechanism)
0d2384e Iteration 31: Push-v3 task generalization — annotation effect reverses, GPT-4o achieves best-ever MAE=36.3
41a4f8f Iteration 30: GPT-4o K sweep (K=4/8/16) — bias-variance tradeoff in keyframe count
29885a7 Iteration 29: Quarto page bootstrap — agents/vlm_probe.qmd + references + figures
95c3eef Iteration 28: Gemini-3-flash-preview annotation ± — NO effect (8/10 identical predictions), breaks U-shaped narrative
```
### Files modified on `agent/vlm_probe` (relative to `c3r/replay-priors`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/fix_plan.md
.claude/settings.json
.gitignore
agents/vlm_probe.qmd
references/vlm_probe.qmd
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
studies/vlm_localization_probe/results/gemini3_annotated_iter28/results.json
studies/vlm_localization_probe/results/gemini3_unannotated_iter28/results.json
... and 30 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/SIBLINGS.md
```

## visionary
- **role**: generic
- **focus**: Survey frontier literature (2024-2026) on VLM-guided robotic learning, temporal reasoning in VLMs, and intelligent experience replay. Synthesize VISIONARY proposals for novel techniques combining VLM failure localization with replay buffer prioritization. Parent vlm_probe ran 32 iterations probing 9 VLMs on MetaWorld failure timestep localization. Read git show agent/vlm_probe:.c3r/RESEARCH_LOG.md and git show agent/vlm_probe:studies/vlm_localization_probe/FINDINGS.md for full context. Write proposals to studies/vlm_localization_probe/VISIONARY_PROPOSALS.md.
- **status**: idle · iter #0 · ctx 0%
- **parent**: vlm_probe (this is a sub-agent)

### Recent commits on `agent/visionary`
```
3b59ea0 scaffold replay-priors umbrella
```
### Files modified on `agent/visionary` (relative to `c3r/replay-priors`)
_(none)_

