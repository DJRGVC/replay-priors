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
- **status**: running · iter #26 · ctx 0%

### Recent commits on `agent/td_baseline`
```
03372b0 iter_028: Final cross-study synthesis — 14 approaches, 0 beat uniform (vlm_probe iters 43-45 integrated)
b2756ab iter_027: Paper outline + hero figure (10 approaches, 0 beat uniform)
be42531 iter_027: Negative result paper outline + synthesis update (10 approaches, 0 beat uniform)
59799e0 iter_026: Synthesis update — CER failure closes contrastive ranking (8 approaches tested, 0 beat uniform)
5ef2775 iter_025: SYNTHESIS.md rewrite — vlm_probe findings invalidate VLM-PER architecture, identify non-temporal directions
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
images/td_baseline/cross_study_synthesis.png
images/td_baseline/multiseed_mode_comparison_pick_place_v3.png
images/td_baseline/multiseed_mode_comparison_reach_v3.png
images/td_baseline/paper_hero_10approach.png
... and 60 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/RESEARCH_LOG_ARCHIVE.md
```

