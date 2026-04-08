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
- [ ] Synthesize cross-study implications with VLM probe results (sibling)
- [ ] Write summary comparing TD-error PER failure modes with VLM probe accuracy data
- [ ] Open questions to surface: is "early training" measured in env vs gradient steps? full buffer vs stratified snapshots?
