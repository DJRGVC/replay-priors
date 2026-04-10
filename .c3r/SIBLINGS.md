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
- **status**: running · iter #28 · ctx 100%

### Recent commits on `agent/vlm_probe`
```
0d2384e Iteration 31: Push-v3 task generalization — annotation effect reverses, GPT-4o achieves best-ever MAE=36.3
41a4f8f Iteration 30: GPT-4o K sweep (K=4/8/16) — bias-variance tradeoff in keyframe count
29885a7 Iteration 29: Quarto page bootstrap — agents/vlm_probe.qmd + references + figures
95c3eef Iteration 28: Gemini-3-flash-preview annotation ± — NO effect (8/10 identical predictions), breaks U-shaped narrative
96f8b23 Iteration 27: Literature update (5 papers) + study pause per Daniel
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
... and 26 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/SIBLINGS.md
```

## quarto-fixer
- **role**: quarto-fixer
- **focus**: Fix failed Pages build (run 24261865659)
- **status**: running · iter #1 · ctx 0%

### Recent commits on `agent/quarto-fixer`
```
40393c0 Iteration 1: Fix ANSI ESC codes in QMD front matter + add quarto-fixer pages
3b59ea0 scaffold replay-priors umbrella
```
### Files modified on `agent/quarto-fixer` (relative to `c3r/replay-priors`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
```
### Read one with:
```
git show agent/quarto-fixer:.c3r/INBOX.md
git show agent/quarto-fixer:.c3r/INBOX_ARCHIVE.md
git show agent/quarto-fixer:.c3r/PROMPT.md
git show agent/quarto-fixer:.c3r/RESEARCH_LOG.md
git show agent/quarto-fixer:.c3r/SIBLINGS.md
```

