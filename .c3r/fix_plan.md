# fix_plan.md — experiment queue for td_baseline
#
# Replace this preamble with 3-5 concrete starting tasks (one per line,
# as a markdown bullet). Lines starting with # are kept as comments.
# Agents read the TOP of this file at the start of every iteration.
#
# Example format:
#   - [ ] Task one — one full sentence, no line breaks
#   - [ ] Task two
#
# Save and exit your editor when done. Empty file = agent picks its own direction.

- [ ] keep a record of relevant literature to this problem so i can take a look at this later. also maybe send this in the discord occasionally.
- [ ] Pick 2 sparse-reward MetaWorld tasks of differing horizon/difficulty (e.g. `reach-v2` + `pick-place-v2`) and document the
choice + justification in `studies/td_error_baseline/NOTES.md`. Surface install/Modal blockers in INBOX.md before sinking     
hours.                                                                                                                         
- [ ] Stand up a minimal Modal app running SAC + TD-error PER on one task end-to-end for a short budget (~100k env steps),     
writing metrics + buffer snapshots to a Modal volume. Use stable-baselines3 or CleanRL — do not write SAC from scratch.        
- [ ] Instrument the critic to dump, every ~10k steps: |TD| histogram over the buffer, the top-K and bottom-K transitions PER
would actually sample, and episode return distribution. Store as parquet/npz.                                                  
- [ ] For each snapshot, compute correlation between |TD| and a ground-truth advantage estimate from MetaWorld's dense reward
(oracle only, do not change training). This is the central quantitative claim from the proposal we're testing — that TD-error  
is uninformative early in training.                       
- [ ] Produce one figure: x = env steps, y = |TD|↔oracle-advantage correlation, one line per task. Drop in                     
`studies/td_error_baseline/figures/` with a 1-paragraph interpretation in NOTES.md. Either outcome is the deliverable — this is
the empirical backbone behind the proposal's motivation paragraph and complements Parshawn's scale/complexity sweep without
overlapping it.                                                                                                                
- [ ] Open questions to surface (not block on): is "early training" measured in env vs gradient steps? full buffer vs
stratified snapshots? anything about MetaWorld sparse setup that makes the proposal's framing fragile?


