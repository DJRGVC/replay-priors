# RESEARCH_LOG.md

_(older entries auto-archived to RESEARCH_LOG_ARCHIVE.md at 2026-04-10 22:36 UTC)_

            Uniform 3/5 = RND-PER 3/5 > RPE-PER 2/5 = Adaptive 2/5 > TD-PER α=0.1 2/5
            > TD-PER α=0.6 0/5. The "all RL signals fail" argument is complete.
Decision:   The baseline study is now comprehensively finished. Five independent
            priority signals tested, none beat uniform. The unified failure mechanism
            (bootstrapped signals are uninformative before reward discovery) is well-
            supported. Next priorities: (a) write a rigorous experiment page for Quarto
            (experiments/td_baseline/), (b) begin VLM-PER prototyping using vlm_probe
            data, or (c) investigate the seed-switching phenomenon (why does RND-PER
            change WHICH seeds learn?) as this could reveal something about exploration
            diversity under different priority regimes.

## Iteration 24 — State-space visitation via dense reward proxy  (2026-04-10T23:55:00Z)
Hypothesis: Dense reward distributions (distance-to-goal proxy) can reveal the
            spatial mechanism behind seed-switching — priority signals redirect
            exploration into different state-space regions.
Change:     Created plot_state_visitation.py — 5-panel figure using per-sample
            dense rewards from all reach-v3 snapshots as exploration proxy.
            Violin plots of distributions across modes×timesteps, seed 42
            diagnostic comparison, Q-vs-dense-reward scatter, mean dense reward
            trajectories, and quantitative summary.
Command:    python3 plot_state_visitation.py
Result:     Key findings:
            • Seed 42 at 50k: uniform mean_dr=1.16 (far), RND-PER mean_dr=3.19
              (close to goal). Novelty signal literally pushed exploration toward
              reward-bearing regions (sr: 0→0.173).
            • Seed 99/RND-PER: Q→162 with low dense reward — trapped in novelty
              loop sampling "interesting but useless" states.
            • Violin plots show RND-PER creates wider, right-shifted distributions
              vs uniform's tight, low-mean distributions at 50k.
            • Q-vs-dense-reward scatter reveals decoupling: high Q does not imply
              high dense reward under priority regimes (pathological).
Decision:   State-space visitation analysis complete. The mechanism is clear:
            priority signals create bimodal exploration outcomes. Next priorities:
            (a) head-to-head comparison including VLM-PER when vlm_probe has
            a working priority scorer, (b) prototype VLM-PER integration, or
            (c) ask Daniel for direction given both baseline and mechanistic
            analyses are now comprehensive.

## Iteration 22 — Rigorous experiment write-up for Quarto  (2026-04-10T23:00:00Z)
Hypothesis: A publication-quality experiment page will make the study's findings
            accessible to collaborators and serve as a permanent record of the
            35-run, 5-signal comparison.
Change:     Created experiments/td_baseline/2026-04-10_td_error_per_baseline.qmd
            with full methodology, result tables, 4 embedded figures (hero summary,
            mode comparison, alpha sweep, pick-place), discussion of unified
            chicken-and-egg failure mechanism, and reproducibility section.
            Updated agents/td_baseline.qmd with new entry linking to experiment.
Command:    N/A (writing only)
Result:     Experiment page complete. Covers: Question, Method (6 modes, 2 tasks,
            5 seeds, instrumentation, compute), Results (headline, correlation
            analysis, 5-mode table, alpha sweep, pick-place, per-signal failure
            analysis, hero figure), Discussion (chicken-and-egg, VLM motivation,
            limitations), Reproducibility (seeds, versions, paths, ~12 GPU-hours).
Decision:   The TD-error baseline study is now comprehensively documented. Next
            priorities: (a) VLM-PER prototype using vlm_probe sibling's data,
            (b) seed-switching investigation under RND-PER, or (c) ask Daniel
            for direction on what to pursue next.

## Iteration 23 — Exploration bifurcation: seed-switching under priority regimes  (2026-04-10T23:30:00Z)
Hypothesis: Priority signals don't uniformly help/hurt — they redirect exploration,
            changing WHICH seeds learn rather than HOW MANY. This is measurable via
            per-seed outcome comparison across modes.
Change:     Created plot_seed_switching.py — 6-panel analysis figure. Extracted
            per-seed learning trajectories from all 40 existing snapshot directories
            (5 seeds × 4 main modes on reach-v3). Computed Jaccard similarity between
            "learned" seed sets across modes.
Command:    python3 plot_seed_switching.py (data extraction + figure generation only)
Result:     Key findings:
            • Uniform: {7,123,256} learn. RND-PER: {42,7,123}. RPE-PER: {7,123}.
              Adaptive: {7,256}. Same 3/5 count for Uniform/RND-PER, different seeds.
            • Mean pairwise Jaccard = 0.51 (below 1.0=identical, above 0.33=random).
            • Seed 42 diagnostic case: never learns under uniform (sr=0 at 100k),
              learns FASTEST under RND-PER (sr=0.369 at 90k).
            • Pathological Q-divergence: seed 99/RND-PER Q→162 with sr=0 (novelty
              loop), seed 123/Adaptive Q→225 with sr≈0 (priority-induced instability).
            • Seed 7 is the "robust learner" — succeeds under all 4 modes.
Decision:   This is a genuinely novel finding — priority signals create exploration
            bifurcations, not uniform improvements. Next: investigate the mechanism
            deeper — extract per-seed state-space visitation patterns (PCA of buffer
            observations) to show HOW exploration trajectories diverge. This could
            become a standalone paper figure.
