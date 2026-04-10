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
- **status**: running · iter #33 · ctx 100%

### Recent commits on `agent/vlm_probe`
```
5224ad8 Iteration 37: Confidence-gated VLM-PER — agreement is an anti-signal for accuracy (r=+0.53), always-VLM strictly worse than uniform
48c1612 Iteration 36: BAEP ensemble analysis — naive ensembles don't beat best individual, but selected 2-model pairs do
138e1b3 Iteration 35: Experiment write-up (bias-matching) + fix images on main
c8e63c9 Iteration 32: Pick-place-v3 task generalization — GPT-4o-mini extreme fixation (9/10 at t=106), annotation +9% MAE
7b07925 Iteration 34: Annotation × task × model figure + visionary cleanup
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
experiments/vlm_probe/2026-04-10_annotation_bias_matching.qmd
references/vlm_probe.qmd
studies/vlm_localization_probe/.gitignore
studies/vlm_localization_probe/FINDINGS.md
studies/vlm_localization_probe/FREE_VLM_OPTIONS.md
studies/vlm_localization_probe/RESULTS_SUMMARY.md
studies/vlm_localization_probe/analyze_gt_quality.py
studies/vlm_localization_probe/build_report.py
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/confidence_gating_analysis.py
studies/vlm_localization_probe/ensemble_analysis.py
studies/vlm_localization_probe/figures/k_sweep_reach_v3.png
studies/vlm_localization_probe/plot_annotation_task_model.py
studies/vlm_localization_probe/plot_k_sweep.py
studies/vlm_localization_probe/priority_score.py
studies/vlm_localization_probe/regenerate_meta.py
studies/vlm_localization_probe/results/confidence_gating/metrics.json
studies/vlm_localization_probe/results/cot_gpt4o/results.json
studies/vlm_localization_probe/results/cot_gpt4o_mini/results.json
studies/vlm_localization_probe/results/cot_gpt4o_mini_noannotate/results.json
... and 36 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/SIBLINGS.md
```

