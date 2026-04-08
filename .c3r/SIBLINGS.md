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
- **status**: paused · iter #13 · ctx 0%
- **last iter**: 1h ago

### Recent commits on `agent/vlm_probe`
```
85acc68 iter_015: two-pass adaptive probing on Llama-3.2-90B (NEGATIVE: MAE 69.8→71.3, refinement worsens 6/10 — coarse pass too inaccurate to center refinement window)
097d91e iter_014: random vs uniform sampling on Llama-3.2-90B (MAE 64.7 vs 63.8, no difference — sampling strategy not the bottleneck)
f38da79 iter_013: GitHub Models backend + Llama 3.2 Vision probe (11B MAE=72.9, 90B MAE=53.5, grid-position bias from tiling)
2b3c619 iter_012: GT quality analysis — push/pick-place unsuitable for VLM probing with random policy (100% ambiguous GT, no object contact)
8b34651 iter_011: two-pass adaptive probing + random sampling strategy (code-only, Gemini quotas still exhausted, no GROQ key)
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
studies/vlm_localization_probe/RESULTS_SUMMARY.md
studies/vlm_localization_probe/analyze_gt_quality.py
studies/vlm_localization_probe/collect_rollouts.py
studies/vlm_localization_probe/figures/k_sweep_reach_v3.png
studies/vlm_localization_probe/plot_k_sweep.py
studies/vlm_localization_probe/priority_score.py
studies/vlm_localization_probe/regenerate_meta.py
studies/vlm_localization_probe/results/k_sweep_consolidated.json
studies/vlm_localization_probe/results/k_sweep_k32/results.json
studies/vlm_localization_probe/results/k_sweep_reach/results.json
studies/vlm_localization_probe/results/random_sampling/results.json
studies/vlm_localization_probe/results/random_sampling_control/results.json
studies/vlm_localization_probe/results/results.json
studies/vlm_localization_probe/results/two_pass/two_pass_results.json
studies/vlm_localization_probe/run_probe.py
studies/vlm_localization_probe/two_pass_probe.py
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
- **status**: running · iter #61 · ctx 0%
- **last iter**: 54s ago
- **parent**: vlm_probe (this is a sub-agent)

### Recent commits on `agent/vlm_litreview`
```
b1eee08 iter_062: §56 Contrastive/Self-Supervised Replay Prioritization — G5 formal (C1-C4 conditions), 12-method supervision taxonomy, TW-CRL closest prior (C1+C2 only), POER structural predecessor, G16 GCR+VLM dual-signal gap (10493→10854 lines)
742b52d iter_061: §39m Two-Pass Probe Negative Result — MAE 69.8→71.3, 3-factor failure mode analysis, §54 precision ceiling (5-bit < 7.2-bit), grid-bias resolution-invariant, two-pass ruled out (10391→10493 lines)
c8750b2 iter_060: §55 Reward Shaping for Temporal Credit Assignment — PBRS/RUDDER/HCA/IRCR/HC-Dice/ReDit/GP-likelihood theoretical lineage, G_i Gaussian kernel multiply justified, info-desert unique advantage (9980→10391 lines)
711139f iter_059: §39l Cross-Reference — Llama 3.2 Vision + grid-position bias taxonomy, random vs uniform sampling (vlm_probe iters 013-014, 9872→9980 lines)
3bf6f77 iter_058: §54 Temporal Grounding and Video Understanding in VLMs — Moment-DETR/UniVTG/QD-DETR span literature, TimeChat/VTimeLLM/VidEgoThink architectures, L3 task-difficulty formal analysis, keyframe floor, G14/G15 new gaps, ~35% MAE reduction estimate (9382→9872 lines)
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
- **status**: stopped · iter #1 · ctx 0%
- **last iter**: 8h ago
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

