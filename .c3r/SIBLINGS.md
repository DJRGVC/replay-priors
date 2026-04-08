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
- **status**: running · iter #2 · ctx 0%
- **last iter**: 14m ago

### Recent commits on `agent/td_baseline`
```
0828233 iter_002: Modal app + 100k runs + TD correlation figure
9b1efaa iter_001: fix dense-reward oracle gap + smoke test pipeline
3b59ea0 scaffold replay-priors umbrella
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
studies/td_error_baseline/NOTES.md
studies/td_error_baseline/dense_reward_buffer.py
studies/td_error_baseline/figures/td_correlation_over_training.json
studies/td_error_baseline/figures/td_correlation_over_training.png
studies/td_error_baseline/metaworld_env.py
studies/td_error_baseline/modal_app.py
studies/td_error_baseline/oracle_correlation.py
studies/td_error_baseline/plot_td_correlation.py
studies/td_error_baseline/snapshots/.gitignore
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/config.json
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/summary.json
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00010000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00020000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00030000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00040000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00050000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00060000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00070000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00080000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00090000.npz
studies/td_error_baseline/snapshots/pick-place-v3_s42/pick-place-v3_s42/td_snapshots/snapshot_00100000.npz
... and 15 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/SIBLINGS.md
```

