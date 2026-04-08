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
- **status**: running · iter #6 · ctx 0%
- **last iter**: 13m ago

### Recent commits on `agent/vlm_probe`
```
52cae95 iter_008: proprio-as-text augmentation (extract_proprio_text + --proprio flag), negative signal on flash-lite (MAE 107.5 vs 59.5 baseline, n=2 valid due to severe rate-limiting)
7f7d5d7 iter_007: frame annotation (VTimeCoT-style t=X overlay), MAE 71.9→59.5 on flash-lite
b14a3cd iter_006: CoT prompt (Summarize→Think→Answer), model-dependent effect
da77793 iter_005: Gemini 3 Flash probe (MAE=54.2, ±10=44%, start-bias)
0dfebb3 iter_004: Gemini backend, flash-lite probe (MAE=95.2), subagent spawn
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
studies/vlm_localization_probe/FREE_VLM_OPTIONS.md
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/figures/k_sweep_reach_v3.png
studies/vlm_localization_probe/plot_k_sweep.py
studies/vlm_localization_probe/regenerate_meta.py
studies/vlm_localization_probe/results/k_sweep_consolidated.json
studies/vlm_localization_probe/results/k_sweep_k32/results.json
studies/vlm_localization_probe/results/k_sweep_reach/results.json
studies/vlm_localization_probe/results/results.json
studies/vlm_localization_probe/run_probe.py
studies/vlm_localization_probe/vlm_client.py
```
### Read one with:
```
git show agent/vlm_probe:.c3r/INBOX.md
git show agent/vlm_probe:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_probe:.c3r/PROMPT.md
git show agent/vlm_probe:.c3r/RESEARCH_LOG.md
git show agent/vlm_probe:.c3r/SIBLINGS.md
```

## vlm_litreview
- **role**: generic
- **focus**: Literature review: survey recent papers (2023-2026) on VLM-based failure detection and localization in robotic manipulation. Focus on which VLMs are used, keyframe selection methods, prompting strategies, and accuracy metrics. Summarize findings in studies/vlm_localization_probe/LITERATURE.md.
- **status**: running · iter #26 · ctx 0%
- **last iter**: 4m ago
- **parent**: vlm_probe (this is a sub-agent)

### Recent commits on `agent/vlm_litreview`
```
9dbd607 iter_026: §30 intrinsic motivation × exploration bonus survey (ICM/RND/RIDE/POER/ELLM/IGE-LLMs/ExploRLLM/SENSEI/Task-Aware-Bisim/RPE-PER + MetaWorld pixel-novelty failure mode + POER:RND::VLM-PER:G_VLM structural predecessor + triple-empty intersection confirmed) → LITERATURE.md
7d0614b iter_025: §29 HER × VLM goal relabeling survey (HIGhER/DIAL/CAST/LGR2/AgentHER/ECHO/CodeIt + VLM-HER-PER novel formulation + triple-gap confirmed) → LITERATURE.md
22f0301 iter_024: §28 VLM dense reward generation survey (VLM-RM/EUREKA/Motif/RL-VLM-F/FuRL/ERL-VLM/MVR/LRM/VLLR + reward-vs-priority divide gap + FuRL as VLM-PER baseline) → LITERATURE.md
f135edb iter_023: §27 cross-task VLM failure localization transfer survey (SAFE/AHA/I-FailSense/FailSafe/RoboFAC/FAIL-Detect/FPC-VLA/StepEval + Δ MAE MetaWorld gap) → LITERATURE.md
e73e5a6 iter_022: §26 reward shaping × replay prioritization survey (RPE-PER/HPRS/DEMO3/GCR/PrioritizedGenReplay + three-component priority formula gap + PBRS-not-applicable clarification) → LITERATURE.md
```
### Files modified on `agent/vlm_litreview` (relative to `c3r/replay-priors`)
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
studies/vlm_localization_probe/LITERATURE.md
```
### Read one with:
```
git show agent/vlm_litreview:.c3r/INBOX.md
git show agent/vlm_litreview:.c3r/INBOX_ARCHIVE.md
git show agent/vlm_litreview:.c3r/PROMPT.md
git show agent/vlm_litreview:.c3r/RESEARCH_LOG.md
git show agent/vlm_litreview:.c3r/SIBLINGS.md
```

## lit_review2
- **role**: generic
- **focus**: Literature review agent. Use WebSearch to find and summarize recent papers (NeurIPS, ICLR, ICML 2023-2026, DeepMind, Google) on: (1) alternatives to TD-error prioritized experience replay in sparse-reward RL, (2) VLM/LLM-guided exploration, reward shaping, or hindsight relabeling, (3) foundation-model-based replay prioritization. Write findings to studies/td_error_baseline/LIT_REVIEW.md. Focus on web search and writing — no code, no training.
- **status**: paused · iter #1 · ctx 0%
- **last iter**: 4h ago
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

