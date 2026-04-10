# fix_plan.md — experiment queue for td_baseline

## Completed (summary)
All core deliverables done through iter_015: pipeline built, 2 tasks × 5 seeds × 3 modes + alpha sweep run, hero summary figure produced, FINDINGS.md + SYNTHESIS.md written. TD-PER proven uninformative (Spearman≈0 for 60-80% of training) and actively harmful at default α (0/5 vs 3/5 uniform on reach-v3).

## Open tasks
- [ ] Head-to-head: uniform vs TD-PER vs VLM-PER vs Adaptive-Mix on reach-v3 + pick-place-v3
- [ ] Run VLM probe on pick-place-v3 failure rollouts (coordinate with vlm_probe sibling)
- [ ] **RND-PER**: count-based novelty (Random Network Distillation) as another baseline
- [ ] Prototype VLM-PER integration using vlm_probe sibling's data/models
- [x] Update hero summary figure (plot_summary_figure.py) to include RPE-PER in 4-mode comparison (iter_020)
- [ ] Open questions: env steps vs gradient steps for "early training"; VLM scoring frequency vs cost

## Done
- [x] RPE-PER baseline (iter_018): 2/5 learn, confirms signal > mechanism thesis
- [x] Update hero summary figure to include pick-place-v3 mode comparison data (iter_017)
