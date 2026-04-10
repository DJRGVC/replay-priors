# fix_plan.md — experiment queue for td_baseline

## Completed (summary)
All core deliverables done through iter_015: pipeline built, 2 tasks × 5 seeds × 3 modes + alpha sweep run, hero summary figure produced, FINDINGS.md + SYNTHESIS.md written. TD-PER proven uninformative (Spearman≈0 for 60-80% of training) and actively harmful at default α (0/5 vs 3/5 uniform on reach-v3).

## Open tasks
- [x] **Seed-switching analysis**: exploration bifurcation under priority regimes (iter_023)
- [x] **State-space visitation analysis**: dense reward distribution proxy shows exploration divergence per seed×mode (iter_024)
- [ ] Head-to-head: uniform vs TD-PER vs VLM-PER vs Adaptive-Mix on reach-v3 + pick-place-v3
- [ ] Prototype VLM-PER integration using vlm_probe sibling's data/models
- [ ] Open questions: env steps vs gradient steps for "early training"; VLM scoring frequency vs cost

## Done
- [x] RPE-PER baseline (iter_018): 2/5 learn, confirms signal > mechanism thesis
- [x] Update hero summary figure to include pick-place-v3 mode comparison data (iter_017)
