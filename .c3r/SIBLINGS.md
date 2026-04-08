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

## vlm_probe
- **role**: generic
- **focus**: Bootstrap studies/vlm_localization_probe: collect a small set of MetaWorld failure rollouts on 2-3 tasks, build a thin VLM E  client (Claude + one other) that takes K keyframes plus a task description and predicts the failure timestep window, and run a E  sweep over K, prompt format, model, and task reporting localization accuracy, latency, and cost. Do not touch SAC or replay E  buffers — this study is pure VLM probing.
- **status**: running · iter #0 · ctx 0%

### Recent commits on `agent/vlm_probe`
```
007302f iter_001: collect 60 failure rollouts across 3 MetaWorld tasks
3b59ea0 scaffold replay-priors umbrella
```
### Files modified on `agent/vlm_probe` (relative to `c3r/replay-priors`)
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
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/regenerate_meta.py
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/SIBLINGS.md
```

