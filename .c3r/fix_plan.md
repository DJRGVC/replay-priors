# fix_plan.md — experiment queue for td_baseline

- [x] Pick 2 sparse-reward MetaWorld tasks (reach-v3 + pick-place-v3) — documented in NOTES.md
- [x] Stand up Modal app running SAC on both tasks for 100k steps — modal_app.py
- [x] Instrument critic to dump |TD| histogram, top-K/bottom-K, episode returns every 10k steps
- [x] Compute correlation between |TD| and dense-reward oracle advantage per snapshot
- [x] Produce figure: x=env steps, y=|TD|↔oracle correlation, one line per task
- [x] Add a second seed (seed=123) to both tasks for robustness, regenerate figure with error bars
- [x] Create FINDINGS.md documentation + share with siblings

- [x] Run oracle_correlation.py analysis on downloaded snapshots for priority_gini + top-K overlap metrics
- [x] Keep a record of relevant literature — lit_review2 produced §1 (11 methods); pulled into branch
- [x] 300k pick-place-v3 runs — correlation never stabilizes, inverts under Q-instability
- [x] Synthesize cross-study implications with VLM probe results (sibling) → SYNTHESIS.md
- [x] Write summary comparing TD-error PER failure modes with VLM probe accuracy data → SYNTHESIS.md §2
- [x] Regime classification + MI proxy + wasted budget analysis → plot_regime_map.py, 6-panel figure
- [ ] Run VLM probe on pick-place-v3 failure rollouts (coordinate with vlm_probe sibling)
- [x] Implement Adaptive Priority Mixer (TD + VLM hybrid, regime-aware switching) — adaptive_priority_mixer.py + train_mixer.py
- [x] Run full 100k comparison: uniform vs td-per vs adaptive on reach-v3 (seed=42) via Modal
  - BUG: SB3 SAC never calls update_priorities() → PER was inactive
- [x] **Fix SB3 PER integration: subclass SAC.train() to call update_priorities() with TD errors** — per_sac.py PERSAC class
- [ ] Re-run 100k comparison with working PER (reach-v3, seed=42, all 3 modes)
- [ ] Head-to-head: uniform vs TD-PER vs VLM-PER vs Adaptive-Mix on reach-v3 + pick-place-v3
- [ ] Consider RPE-PER (arXiv:2501.18093) as additional baseline
- [ ] Open questions: env steps vs gradient steps for "early training"; VLM scoring frequency vs cost
