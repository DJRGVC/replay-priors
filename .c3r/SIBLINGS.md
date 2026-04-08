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
- **status**: running · iter #13 · ctx 0%
- **last iter**: 29m ago

### Recent commits on `agent/td_baseline`
```
104359b iter_014: hero summary figure + lit review incorporation — 4-panel pub-quality figure (Spearman/mode comparison/Q-explosion/regime breakdown), killed lit_review2 per Daniel, pulled LIT_REVIEW.md onto branch
c1bdb3e iter_013: alpha sweep — α=0.3 ties uniform (3/5), α=0.1 worse (2/5), α=0.6 worst (0/5); problem is signal not mechanism
0e9d017 iter_012: 5-seed mode comparison — TD-PER actively hurts (0/5 learn), uniform best (3/5), adaptive middling (2/5)
12ceb16 iter_011: 5-seed baseline resolves reach-v3 learning regression — stochastic, not a bug
8605f09 iter_010: 100k mode comparison with working PER — PER destabilizes Q, no mode learns reach-v3
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
studies/td_error_baseline/figures/alpha_sweep_td_per.pdf
studies/td_error_baseline/figures/mode_comparison_reach_v3.pdf
studies/td_error_baseline/figures/multiseed_mode_comparison.pdf
studies/td_error_baseline/figures/td_correlation_over_training.json
studies/td_error_baseline/figures/td_correlation_over_training.png
studies/td_error_baseline/figures/td_per_regime_map.pdf
studies/td_error_baseline/figures/td_per_summary.pdf
studies/td_error_baseline/metaworld_env.py
studies/td_error_baseline/modal_app.py
studies/td_error_baseline/oracle_correlation.py
studies/td_error_baseline/per_sac.py
studies/td_error_baseline/plot_alpha_sweep.py
studies/td_error_baseline/plot_mode_comparison.py
... and 11 more
```
### Read one with:
```
git show agent/td_baseline:.c3r/INBOX.md
git show agent/td_baseline:.c3r/INBOX_ARCHIVE.md
git show agent/td_baseline:.c3r/PROMPT.md
git show agent/td_baseline:.c3r/RESEARCH_LOG.md
git show agent/td_baseline:.c3r/SIBLINGS.md
```

## lit_review2
- **role**: generic
- **focus**: Literature review agent. Use WebSearch to find and summarize recent papers (NeurIPS, ICLR, ICML 2023-2026, DeepMind, Google) on: (1) alternatives to TD-error prioritized experience replay in sparse-reward RL, (2) VLM/LLM-guided exploration, reward shaping, or hindsight relabeling, (3) foundation-model-based replay prioritization. Write findings to studies/td_error_baseline/LIT_REVIEW.md. Focus on web search and writing — no code, no training.
- **status**: stopped · iter #1 · ctx 0%
- **last iter**: 9h ago
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

