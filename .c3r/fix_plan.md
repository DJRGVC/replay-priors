# fix_plan.md — experiment queue for vlm_probe
#
# Study declared COMPLETE at iter 47 (2026-04-10) per human agreement.

- [ ] Budget: $0. DO NOT use Anthropic API key (costs real money). Use only free APIs.

# --- COMPLETED: VLM Failure Localization Probe ---
# 47 iterations, 14 approaches, 9 models, 3 tasks, $0.80 total cost.
# All temporal approaches fail (positional bias). Non-temporal failure mode
# descriptions viable at N≥50. Full write-up on Quarto.
- [x] Baseline model comparison (9 models)
- [x] K sweep (4/8/16/32)
- [x] CoT × annotation factorial (GPT-4o, GPT-4o-mini)
- [x] Bias-matching mechanism (3 tasks)
- [x] Ensemble/gating/CER (all failed)
- [x] Failure mode descriptions (positive signal, η²=0.34-0.99)
- [x] Category-diversity replay simulation (viable at N≥50)
- [x] Cross-model category stability (JSD analysis)
- [x] Consolidated results database (360 predictions, 31 conditions)
- [x] Full experiment write-up on Quarto
- [x] Study synthesis with landscape figure

# --- DEFERRED (not prioritized) ---
# These remain if the study is reopened:
# - Same-rollout cross-model comparison (blocked by APIs at study close)
# - Proposal 3: Task-Adaptive Annotation
# - Proposal 6: Phase-Segmented Replay (needs semi-trained policy)
# - Proposal 7: Retrospective Failure Narration

# --- NEXT DIRECTION ---
# Awaiting guidance from Daniel on what to work on next.
- [ ] Ask Daniel for next research direction
