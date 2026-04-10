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

- **visionary** (generic, parent=vlm_probe) — status=idle, iter=#0, last=never
  Focus: Survey frontier literature (2024-2026) on VLM-guided robotic learning, temporal reasoning in VLMs, and intelligent experience replay. Synthesize VISIONARY proposals for novel techniques combining VLM failure localization with replay buffer prioritization. Parent vlm_probe ran 32 iterations probing 9 VLMs on MetaWorld failure timestep localization. Read git show agent/vlm_probe:.c3r/RESEARCH_LOG.md and git show agent/vlm_probe:studies/vlm_localization_probe/FINDINGS.md for full context. Write proposals to studies/vlm_localization_probe/VISIONARY_PROPOSALS.md.

**Decision rules** (apply at the top of every iteration):
1. If a child's last RESEARCH_LOG entry says its task is done, kill it: `$C3R_BIN/c3r kill <name>`
2. If a child has been stale (no iter for >2 hours), kill it.
3. If a child's fail_streak ≥ 3 in state.json, investigate or kill it.
4. Otherwise, leave it running and check again next iteration.

---

## SIBLINGS — peers you do NOT manage (other agents' work)

## td_baseline
- **role**: generic
- **focus**: Bootstrap studies/td_error_baseline: set up MetaWorld + SAC with TD-error PER on 2 sparse-reward tasks using Modal for       E  training, instrument the critic to log TD-error distributions and their correlation with a dense-reward oracle advantage over  E  training, and produce a single figure quantifying how (un)informative TD-error PER is in the early training regime.
- **status**: running · iter #19 · ctx 0%
- **last iter**: 17m ago

### Recent commits on `agent/td_baseline`
```
0044f01 iter_020: Hero figure updated with RPE-PER as 4th mode in 6-panel summary
c53c049 iter_019: Quarto page + references + INBOX catchup — full study write-up with hero figure and 7 cited papers
2653f31 iter_018: RPE-PER baseline — reward prediction error PER also fails to beat uniform (2/5 vs 3/5), confirms problem is signal not mechanism
23d8a3b iter_017: 6-panel hero figure with pick-place-v3 data
79e6c77 iter_016: compaction (summarized iters 001-010)
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
index.qmd
... and 45 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/RESEARCH_LOG_ARCHIVE.md
```

