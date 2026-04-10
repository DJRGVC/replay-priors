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

## td_baseline
- **role**: generic
- **focus**: Bootstrap studies/td_error_baseline: set up MetaWorld + SAC with TD-error PER on 2 sparse-reward tasks using Modal for       E  training, instrument the critic to log TD-error distributions and their correlation with a dense-reward oracle advantage over  E  training, and produce a single figure quantifying how (un)informative TD-error PER is in the early training regime.
- **status**: running · iter #20 · ctx 0%

### Recent commits on `agent/td_baseline`
```
e31d1c1 iter_022: Rigorous experiment write-up — 35-run study documented with full methodology, results, and figures
42b77c1 iter_021: RND-PER baseline — novelty PER ties uniform (3/5), all 5 RL signals now tested, none beat uniform
19dab3c iter_021: RND-PER implementation + 5-seed Modal training launched (awaiting results)
0044f01 iter_020: Hero figure updated with RPE-PER as 4th mode in 6-panel summary
c53c049 iter_019: Quarto page + references + INBOX catchup — full study write-up with hero figure and 7 cited papers
```
### Files modified on `agent/td_baseline` (relative to `c3r/replay-priors`)
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
.gitignore
_quarto.yml
agents/index.qmd
agents/lit_review2.qmd
agents/td_baseline.qmd
agents/vlm_probe.qmd
experiments.qmd
experiments/lit_review2/.gitkeep
experiments/td_baseline/.gitkeep
experiments/td_baseline/2026-04-10_td_error_per_baseline.qmd
experiments/vlm_probe/.gitkeep
images/README.md
images/lit_review2/.gitkeep
images/shared/.gitkeep
images/td_baseline/.gitkeep
images/td_baseline/alpha_sweep_td_per.png
images/td_baseline/multiseed_mode_comparison_pick_place_v3.png
images/td_baseline/multiseed_mode_comparison_reach_v3.png
images/td_baseline/td_per_summary.png
images/vlm_probe/.gitkeep
... and 47 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/RESEARCH_LOG_ARCHIVE.md
```

