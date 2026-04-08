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

## td_baseline
- **role**: generic
- **focus**: Bootstrap studies/td_error_baseline: set up MetaWorld + SAC with TD-error PER on 2 sparse-reward tasks using Modal for       E  training, instrument the critic to log TD-error distributions and their correlation with a dense-reward oracle advantage over  E  training, and produce a single figure quantifying how (un)informative TD-error PER is in the early training regime.
- **status**: running · iter #11 · ctx 0%
- **last iter**: 19m ago

### Recent commits on `agent/td_baseline`
```
0e9d017 iter_012: 5-seed mode comparison — TD-PER actively hurts (0/5 learn), uniform best (3/5), adaptive middling (2/5)
12ceb16 iter_011: 5-seed baseline resolves reach-v3 learning regression — stochastic, not a bug
8605f09 iter_010: 100k mode comparison with working PER — PER destabilizes Q, no mode learns reach-v3
180bc8a iter_009: fix SB3 PER integration — PERSAC subclass calls update_priorities() with TD errors + IS-weighted critic loss
9dfa19f iter_008: 100k mode comparison reveals PER integration bug — SB3 never calls update_priorities()
```
### Files modified on `agent/td_baseline` (relative to `c3r/replay-priors`)
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
studies/td_error_baseline/.gitignore
studies/td_error_baseline/FINDINGS.md
studies/td_error_baseline/LIT_REVIEW.md
studies/td_error_baseline/NOTES.md
studies/td_error_baseline/SYNTHESIS.md
studies/td_error_baseline/adaptive_priority_mixer.py
studies/td_error_baseline/dense_reward_buffer.py
studies/td_error_baseline/figures/5seed_baseline_reach_v3.pdf
studies/td_error_baseline/figures/mode_comparison_reach_v3.pdf
studies/td_error_baseline/figures/multiseed_mode_comparison.pdf
studies/td_error_baseline/figures/td_correlation_over_training.json
studies/td_error_baseline/figures/td_correlation_over_training.png
studies/td_error_baseline/figures/td_per_regime_map.pdf
studies/td_error_baseline/metaworld_env.py
studies/td_error_baseline/modal_app.py
studies/td_error_baseline/oracle_correlation.py
studies/td_error_baseline/per_sac.py
studies/td_error_baseline/plot_mode_comparison.py
studies/td_error_baseline/plot_multiseed_comparison.py
studies/td_error_baseline/plot_priority_quality.py
studies/td_error_baseline/plot_regime_map.py
... and 6 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/SIBLINGS.md
```

## vlm_litreview
- **role**: generic
- **focus**: Literature review: survey recent papers (2023-2026) on VLM-based failure detection and localization in robotic manipulation. Focus on which VLMs are used, keyframe selection methods, prompting strategies, and accuracy metrics. Summarize findings in studies/vlm_localization_probe/LITERATURE.md.
- **status**: running · iter #40 · ctx 1%
- **last iter**: 1m ago
- **parent**: vlm_probe (this is a sub-agent)

### Recent commits on `agent/vlm_litreview`
```
f8f371c iter_041: §40 episodic/trajectory-level replay prioritization survey (14 papers, HGR two-level hierarchy, VLM+episode-priority gap confirmed, two-level VLM-PER formulation) → LITERATURE.md (4716→4931 lines)
26c30f8 iter_040: §39 vlm_probe empirical summary (9 subsections, MAE=41.9 baseline, bias taxonomy, theory-vs-empirical table) → LITERATURE.md (4382→4716 lines)
0f1351d iter_039: §38 anomaly detection track survey (reconstruction/one-class/OOD/foundation) — gap confirmed: no per-timestep p_i use, MetaWorld gap, pixel-novelty failure, conformal-σ novel connection → LITERATURE.md (4280→4382 lines)
0fc5625 iter_037: §37 action segmentation + keyframe + VTG + BOCPD survey — no VLM-free method satisfies all 3 t* conditions; NC6 (BOCPD+proprio) proposed → LITERATURE.md (4110→4280 lines)
2e5ed86 iter_036: compaction (summarized iters 001-016 into archive; log shrunk 306→175 lines; fix_plan pruned)
```
### Files modified on `agent/vlm_litreview` (relative to `c3r/replay-priors`)
```
.c3r/INBOX.md
.c3r/INBOX_ARCHIVE.md
.c3r/PROMPT.md
.c3r/RESEARCH_LOG.md
.c3r/RESEARCH_LOG_ARCHIVE.md
.c3r/SIBLINGS.md
.c3r/agent.conf
.c3r/env.sh
.c3r/fix_plan.md
.claude/settings.json
studies/vlm_localization_probe/LITERATURE.md
```
### Read one with:
```
git show agent/vlm_litreview:.c3r/INBOX.md
git show agent/vlm_litreview:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_litreview:.c3r/PROMPT.md
git show agent/vlm_litreview:.c3r/RESEARCH_LOG.md
git show agent/vlm_litreview:.c3r/RESEARCH_LOG_ARCHIVE.md
```

## lit_review2
- **role**: generic
- **focus**: Literature review agent. Use WebSearch to find and summarize recent papers (NeurIPS, ICLR, ICML 2023-2026, DeepMind, Google) on: (1) alternatives to TD-error prioritized experience replay in sparse-reward RL, (2) VLM/LLM-guided exploration, reward shaping, or hindsight relabeling, (3) foundation-model-based replay prioritization. Write findings to studies/td_error_baseline/LIT_REVIEW.md. Focus on web search and writing — no code, no training.
- **status**: paused · iter #1 · ctx 0%
- **last iter**: 5h ago
- **parent**: td_baseline (this is a sub-agent)

### Recent commits on `agent/lit_review2`
```
8770137 iter_001: §1 alternatives to TD-error PER — 11 methods surveyed across 13 papers
3b59ea0 scaffold replay-priors umbrella
```
### Files modified on `agent/lit_review2` (relative to `c3r/replay-priors`)
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
studies/td_error_baseline/LIT_REVIEW.md
```
### Read one with:
```
git show agent/lit_review2:.c3r/INBOX.md
git show agent/lit_review2:.c3r/INBOX_ARCHIVE.md
git show agent/lit_review2:.c3r/PROMPT.md
git show agent/lit_review2:.c3r/RESEARCH_LOG.md
git show agent/lit_review2:.c3r/SIBLINGS.md
```

