# fix_plan.md — experiment queue for td_baseline

## Completed (summary)
All core deliverables done through iter_015: pipeline built, 2 tasks × 5 seeds × 3 modes + alpha sweep run, hero summary figure produced, FINDINGS.md + SYNTHESIS.md written. TD-PER proven uninformative (Spearman≈0 for 60-80% of training) and actively harmful at default α (0/5 vs 3/5 uniform on reach-v3).

## Open tasks
- [ ] Head-to-head: uniform vs TD-PER vs VLM-PER vs Adaptive-Mix on reach-v3 + pick-place-v3
- [ ] Run VLM probe on pick-place-v3 failure rollouts (coordinate with vlm_probe sibling)
- [ ] Consider RPE-PER (arXiv:2501.18093) as additional baseline — lit review's #1 recommendation
- [ ] **Alternative priority signals**: RPE-PER, RND, or count-based novelty as baselines
- [ ] Open questions: env steps vs gradient steps for "early training"; VLM scoring frequency vs cost
- [x] Update hero summary figure to include pick-place-v3 mode comparison data (iter_017)
