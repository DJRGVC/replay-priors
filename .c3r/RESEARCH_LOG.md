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

## Iteration 29 — Final documentation + study closure  (2026-04-11T04:00:00Z)
Hypothesis: Finalizing all documentation (experiment page + agent page) with seed-switching
            findings will make the study complete and ready for closure.
Change:     Added seed-switching bifurcation analysis (iter 23) and state-space visitation
            (iter 24) to experiment page. Added exploration bifurcation discussion section.
            Updated agent page with final status. Responded to Daniel's INBOX questions
            about why PER doesn't beat uniform. Processed self-kill instruction from
            fix-probe-findings-documentation.
Command:    N/A (documentation/writing only)
Result:     All Quarto pages finalized. Experiment page now includes: seed-switching analysis
            with diagnostic seed 42 case, state-space visitation figures, exploration
            bifurcation discussion. Agent page updated with study-complete status.
            Study summary: 35 runs, 5 priority signals, 2 tasks, 0 beat uniform.
Decision:   Study is complete per Daniel's instruction. Will commit, notify, and self-kill.

## Iteration 28 — Final cross-study synthesis: 14 approaches  (2026-04-11T03:00:00Z)
Hypothesis: Integrating vlm_probe's final findings (iters 43-45: cross-model category
            stability, JSD analysis, complete approach inventory) will produce the
            definitive synthesis of both studies.
Change:     Updated SYNTHESIS.md §2g (cross-model stability), §2f (approach count 10→14),
            §7 (closed failure-mode clustering direction). Updated PAPER_OUTLINE.md with
            14-approach framing, cross-model findings in §5.4, updated figure list.
            Regenerated hero figure as 14-approach version (paper_hero_14approach.png).
            Updated Quarto agent page with new entry.
Command:    python3 plot_paper_hero.py
Result:     SYNTHESIS.md now reflects all findings through vlm_probe iter 45. Complete
            approach inventory: 6 RL-based + 6 VLM temporal + 3 ensemble/meta +
            1 contrastive ranking + 4 non-temporal = 14 approaches tested + uniform
            baseline. Hero figure updated. TD-PER α=0.3 correctly shown as 3/5 (ties
            uniform). [fallback after timeout] Asked Daniel for direction, no response
            after 15 min. Fell back to synthesis update (most conservative option).
Decision:   Both studies are at natural conclusion. Awaiting Daniel's direction on:
            (a) paper draft writing, (b) dense reward shaping investigation, (c) cross-
            domain generalization, or (d) wind-down. Will ask again next iteration.

## Iteration 27 — Negative result paper outline + synthesis update  (2026-04-11T02:00:00Z)
Hypothesis: A structured paper outline will crystallize the convergent negative result
            (10 approaches, 0 beat uniform) into a publishable narrative.
Change:     Created PAPER_OUTLINE.md with full section structure (Abstract through
            Appendices), figure inventory, and data release plan. Updated SYNTHESIS.md
            with vlm_probe iters 39-42 findings (failure descriptions η²=0.34-0.99,
            category-diversity ≈ uniform at small n, helps at N≥50 only). Updated
            approach count from 8→10. Updated Quarto page with new entry.
Command:    python3 plot_paper_hero.py
Result:     Paper outline covers 10 approaches across 3 independent failure mechanisms.
            Estimated ~10 pages main text + 3-4 appendix. Created hero figure
            (paper_hero_10approach.png): 2-panel chart showing success rates + priority
            quality for all 10 approaches. Data correction: TD-PER α=0.3 is 3/5 (ties
            uniform), not 2/5 as previously logged — core finding unchanged.
            SYNTHESIS.md now reflects all findings through vlm_probe iter 42.
            [fallback after timeout] ask_human got HTTP 503 — fell back to creating
            hero figure (most concrete bounded task).
Decision:   Ask Daniel for direction on next iteration: paper writing, category-
            diversity at scale validation, or cross-domain generalization.

## Iteration 26 — Synthesis update: CER failure closes contrastive ranking  (2026-04-11T01:15:00Z)
Hypothesis: Integrating vlm_probe iter 38 CER findings will close the contrastive
            ranking open question and strengthen the "8 approaches, 0 beat uniform"
            narrative in SYNTHESIS.md.
Change:     Updated SYNTHESIS.md §2d (new subsection on CER primacy bias), §5
            (closed question #1), §7 (closed contrastive ranking direction, updated
            approach count to 8). Updated fix_plan.md. Updated Quarto page with
            new entry.
Command:    N/A (synthesis/writing only, no training or API calls)
Result:     SYNTHESIS.md now reflects all vlm_probe findings through iter 38.
            Approach count: 8 tested (5 RL signals: TD-PER×3α, RPE-PER, RND-PER,
            Adaptive; plus VLM temporal, ensemble/gating, CER). Zero beat uniform.
            Only untested direction: failure mode clustering via VLM descriptions.
Decision:   Two viable next steps: (a) outline the negative result paper — the
            convergent "nothing beats uniform" finding across 8 approaches is itself
            publishable, (b) prototype failure mode clustering to test the last
            non-temporal direction. Will ask Daniel for direction.

## Iteration 25 — Cross-study synthesis: unified priority signal landscape  (2026-04-11T00:30:00Z)
Hypothesis: A unified figure combining td_baseline (5 RL signals) and vlm_probe
            (VLM priority quality) will make the "nothing beats uniform" finding
            visually compelling and cross-study.
Change:     Created plot_cross_study_synthesis.py — 4-panel figure: (a) RL signal
            success rates bar chart, (b) VLM overlap vs KL scatter plot, (c) signal
            informativeness timeline, (d) unified failure mechanism diagram. Integrated
            vlm_probe findings from iters 32-37 (confidence gating, ensemble analysis,
            3-task bias matching). Updated Quarto page with new entry.
Command:    python3 plot_cross_study_synthesis.py
Result:     Figure generated. Key synthesis:
            • RL signals: 0/5 beat uniform (3/5) on reach-v3. TD-PER α=0.6 worst (0/5).
            • VLM priorities: Always-VLM overlap 8.7% vs uniform 21.7% (60% worse).
              Sonnet K=8 gets +12% overlap but +24% worse KL — tradeoff, not improvement.
            • Confidence gating: optimal threshold = 100% uniform (agreement anti-signal).
            • Three independent failure modes: chicken-and-egg (RL), positional bias
              (VLM), exploration redirection (novelty). All converge on uniform dominance.
Decision:   [fallback after timeout] Asked Daniel for direction, no response after
            15 min. Fell back to updating SYNTHESIS.md (most conservative option).
            Rewrote sections 2, 4, 5, 6, 7 with vlm_probe findings from iters 32-37.
            Key updates: VLM assessment revised from "promising" to "dominated by
            positional bias", proposed architecture invalidated, remaining directions
            identified (contrastive ranking, failure clustering, phase segmentation).
            Next iteration: await Daniel's direction or begin scoping contrastive
            episode ranking as the next experimental direction.

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
