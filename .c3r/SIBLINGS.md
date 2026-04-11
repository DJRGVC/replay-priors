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
- **status**: running · iter #38 · ctx 100%

### Recent commits on `agent/vlm_probe`
```
a91daed Iteration 45: Study synthesis — landscape figure + final FINDINGS section
7585169 Iteration 44: Cross-model analysis scripts, figures, and quantitative JSD
f81ce1d Iteration 43: Cross-model category comparison — taxonomy adherence is model-dependent
8865bc6 Iteration 42: add remaining data files
f1b3c4a Iteration 42: add simulation script, results, and figures
```
### Files modified on `agent/vlm_probe` (relative to `c3r/replay-priors`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/RESEARCH_LOG_ARCHIVE.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/fix_plan.md
.claude/settings.json
.gitignore
agents/vlm_probe.qmd
experiments/vlm_probe/2026-04-10_annotation_bias_matching.qmd
images/vlm_probe/cross_model_categories_iter43.png
images/vlm_probe/severity_comparison_iter43.png
images/vlm_probe/study_synthesis_landscape.png
references/vlm_probe.qmd
studies/vlm_localization_probe/.gitignore
studies/vlm_localization_probe/FINDINGS.md
studies/vlm_localization_probe/FREE_VLM_OPTIONS.md
studies/vlm_localization_probe/RESULTS_SUMMARY.md
studies/vlm_localization_probe/analyze_failure_descriptions.py
studies/vlm_localization_probe/analyze_gt_quality.py
studies/vlm_localization_probe/build_report.py
studies/vlm_localization_probe/category_diversity_simulation.py
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/confidence_gating_analysis.py
studies/vlm_localization_probe/contrastive_ranking_probe.py
studies/vlm_localization_probe/cross_model_category_analysis.py
studies/vlm_localization_probe/ensemble_analysis.py
studies/vlm_localization_probe/failure_description_probe.py
... and 67 more
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG_ARCHIVE.md
```

