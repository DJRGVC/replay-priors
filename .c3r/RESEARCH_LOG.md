# RESEARCH_LOG.md

Append-only, chronological log of every experiment this agent has run.
Newest entries at the bottom. Each entry follows the format in PROMPT_*.md.

---

## iter_001 — Fix dense-reward oracle gap + smoke test  (2026-04-08T04:30:00Z)
Hypothesis: SB3's standard ReplayBuffer discards info dicts, so the dense_reward
            from SparseRewardWrapper is lost — need a custom buffer to store it.
Change:     Created DenseRewardReplayBuffer (stores dense_reward alongside transitions),
            updated train.py to use it, rewrote td_instrumenter._take_snapshot to
            compute Pearson + Spearman correlation between |TD| and oracle advantage.
Command:    python studies/td_error_baseline/train.py --task reach-v3 --total-steps 3000
            --learning-starts 200 --snapshot-interval 1000 --buffer-size 1000
            --output-dir /tmp/td_smoke_test3 --seed 42
Result:     Pipeline runs end-to-end. 3 snapshots produced. Spearman(|TD|, oracle_adv)
            at steps 2000/3000: -0.027, 0.023, -0.013 — essentially zero. Dense rewards
            stored correctly (mean 1.2→2.4 as policy improves). All snapshot keys present.
Decision:   Next iteration: set up Modal app for full 100k-step runs on both reach-v3 and
            pick-place-v3. The local stack is validated.

## iter_002 — Modal app + full 100k runs + correlation figure  (2026-04-08T05:45:00Z)
Hypothesis: TD-error will be uninformative (near-zero Spearman with oracle advantage)
            for the first ~50% of training on reach-v3, and throughout training on
            pick-place-v3 where the sparse-reward policy never converges.
Change:     Created modal_app.py (image with MetaWorld+SB3+mujoco, T4 GPU, Modal Volume
            for results), plot_td_correlation.py for figure generation. Ran 100k-step
            training on both tasks via Modal, downloaded snapshots, produced figure.
Command:    modal run studies/td_error_baseline/modal_app.py  (parallel reach-v3 + pick-place-v3)
            python plot_td_correlation.py --run-dirs snapshots/reach-v3_s42/reach-v3_s42 snapshots/pick-place-v3_s42/pick-place-v3_s42
Result:     reach-v3: Spearman ≈ 0 for first 60k steps, jumps to 0.65 at 90k as policy
            learns (ep_rew 0→116). pick-place-v3: Spearman stays 0–0.24 throughout,
            policy never learns (ep_rew=0). Figure at figures/td_correlation_over_training.png.
            Wall time: reach-v3 ~19min, pick-place-v3 ~22min on T4.
Decision:   Next iteration: add seed=123 runs for robustness (error bars in figure).
            Also run oracle_correlation.py for Gini + top-K overlap metrics. Consider
            extending to 200k or 500k steps on pick-place-v3 to see if correlation
            ever emerges given enough training.

## iter_003 — Seed=123 runs + FINDINGS.md + lit_review subagent  (2026-04-08T05:50:00Z)
Hypothesis: Second seed will confirm that TD-error uninformativeness is robust, not
            seed-dependent. Seed=123 reach-v3 may learn at a slightly different rate
            but the correlation pattern should match seed=42.
Change:     Ran seed=123 on both tasks via Modal (100k steps each). Updated
            plot_td_correlation.py to support multi-seed aggregation with mean±std
            error bands. Created FINDINGS.md as concise documentation. Updated
            modal_app.py entrypoint to accept --seeds/--tasks args. Spawned
            lit_review2 subagent (opus) after killing failed lit_review.
Command:    modal run studies/td_error_baseline/modal_app.py --seeds 123
            python plot_td_correlation.py --seeds 42,123
Result:     reach-v3 s123: Spearman pattern matches s42 — near-zero until ~50k, then
            rises (0.38→0.57 at 50-60k, earlier than s42's 80-90k jump). Policy
            learned faster (ep_rew 10→379 by 100k). Interestingly, Spearman drops
            back to -0.09/-0.09 at 90-100k despite strong policy — critic overshooting.
            pick-place-v3 s123: Spearman -0.05 to +0.19, never learns (ep_rew=0).
            Consistent with s42. Figure updated with n=2 error bands, clearly showing
            the separation between tasks. Created FINDINGS.md. Killed lit_review
            (fail_streak=4), spawned lit_review2 with opus.
Decision:   Next iteration: run oracle_correlation.py for Gini + top-K overlap metrics
            on all 4 runs. This will quantify how concentrated TD-error priorities are
            (Gini) and how much overlap exists between top-K by |TD| vs top-K by oracle
            advantage.

## iter_004 — Gini + top-K overlap analysis + priority quality figure  (2026-04-08T06:15:00Z)
Hypothesis: Top-K overlap between |TD| and oracle advantage will be near chance (10%)
            early in training, and Gini will show TD priorities are concentrated but on
            the wrong transitions.
Change:     Fixed oracle_correlation.py to use oracle_advantage instead of sparse_rewards.
            Ran analysis on all 4 runs (2 tasks × 2 seeds). Created plot_priority_quality.py
            producing 3-panel figure (overlap, Gini, Spearman). Updated FINDINGS.md.
Command:    python oracle_correlation.py --run-dirs snapshots/{reach,pick-place}-v3_s{42,123}
            python plot_priority_quality.py
Result:     reach-v3: top-10% overlap at chance (7-20%) for first 40k, brief spike to 53-61%
            during learning, then drops back to 6-11%. Gini 0.26-0.54 (moderate concentration).
            pick-place-v3: overlap never exceeds ~2× chance (28% max). Gini 0.30-0.60.
            NEW FINDING: reach-v3 s123 Spearman inverts to -0.09/-0.12 at 90-100k despite
            strong policy (ep_rew=379) — critic overshooting makes TD-PER actively
            anti-informative late in training. Figure: figures/priority_quality_metrics.png.
Decision:   Next iteration: check on lit_review2 subagent progress. Then either (a) run
            extended 200-500k pick-place-v3 to see if correlation ever emerges, or (b) start
            synthesizing cross-study implications with VLM probe results from sibling.

## iter_005 — Extended 300k pick-place-v3 + lit review pull  (2026-04-08T00:30:00Z)
Hypothesis: Pick-place-v3 will remain uncorrelated at 300k steps because the policy
            cannot solve the task with sparse rewards, so the critic never calibrates.
Change:     Ran 300k-step training on pick-place-v3 (seeds 42+123) via Modal. Downloaded
            new snapshots (110k-300k), ran oracle_correlation.py, regenerated both figures.
            Pulled lit_review2 §1 (LIT_REVIEW.md) into branch. Updated FINDINGS.md.
Command:    modal run modal_app.py --tasks pick-place-v3 --seeds "42,123" --total-steps 300000
            python oracle_correlation.py + plot_td_correlation.py + plot_priority_quality.py
Result:     HYPOTHESIS PARTIALLY REFUTED — seed42 showed marginal learning (ep_rew peaked
            0.7 at 185k) but this made things WORSE: Q-values oscillated wildly
            (0.02→50→11→0.02), and Spearman INVERTED to −0.31 at 280k. TD-PER would
            have been actively anti-informative. Seed123 never learned (ep_rew=0, policy
            collapsed, Q→0.0005). Final Spearman: s42=−0.21, s123=+0.20.
            NEW FINDING: TD-error inversion is not just a reach-v3 late-training
            artifact — it's a general failure mode when Q-values are unstable.
            Wall time: s42=71min, s123=66min on T4.
Decision:   Next iteration: synthesize cross-study implications. Compare TD baseline
            failure modes with VLM probe's localization data from sibling. Write a joint
            summary that motivates VLM-based replay prioritization.

## iter_006 — Cross-study synthesis + regime map + information-theoretic analysis  (2026-04-08T07:00:00Z)
Hypothesis: Formalizing TD-PER's failure as a "regime classification" problem will
            reveal that failures are structural (not parametric) and will make the
            case for VLM-augmented PER quantitatively precise.
Change:     Created plot_regime_map.py (6-panel figure: Q-dynamics, regime heatmap,
            Spearman trajectory with regime coloring, MI proxy, wasted sampling budget,
            and regime breakdown bar chart). Created SYNTHESIS.md combining TD baseline
            results, VLM probe accuracy data, and lit review into an actionable proposal
            for an Adaptive Priority Mixer. Updated FINDINGS.md and fix_plan.md.
Command:    python studies/td_error_baseline/plot_regime_map.py
Result:     **TD-PER fails 50-93% of training time across all runs.**
            Regime breakdown:
              reach-v3 s42:  aligned=20%, noise=70%, inverted=0%, unstable=10%
              reach-v3 s123: aligned=50%, noise=50%, inverted=0%, unstable=0%
              pp-v3 s42:     aligned=7%,  noise=40%, inverted=17%, unstable=37%
              pp-v3 s123:    aligned=13%, noise=60%, inverted=3%, unstable=23%
            MI proxy shows "information desert" for first 40-70k steps (0 bits).
            Effective waste reaches 50%+ on pick-place. VLM probe (K=8, MAE=41.9,
            within-20=35%) provides complementary signal from step 0.
            Proposed Adaptive Priority Mixer: regime-aware switching between TD and
            VLM scores, with explicit inversion handling.
            6-panel regime map figure: figures/td_per_regime_map.png (+ PDF).
Decision:   Next iteration: coordinate with vlm_probe sibling to run pick-place-v3
            probes. If sibling is busy, start prototyping the Adaptive Priority
            Mixer as a drop-in SB3 component.

## iter_007 — Adaptive Priority Mixer implementation + smoke test  (2026-04-08T08:30:00Z)
Hypothesis: A regime-aware replay buffer that detects noise/aligned/inverted/unstable
            regimes online and switches between TD-error priorities and uniform can be
            implemented as a drop-in SB3 ReplayBuffer subclass.
Change:     Created adaptive_priority_mixer.py (SumTree proportional PER, RegimeDetector
            using Q coefficient-of-variation + TD Gini + trend analysis, AdaptivePriorityMixer
            buffer with regime-aware priority switching). Created train_mixer.py supporting
            3 modes: adaptive, td-per, uniform. MixerInstrumentCallback extends
            TDInstrumentCallback with online regime detection every 100 steps + regime
            stats logging.
Command:    python train_mixer.py --task reach-v3 --total-steps 3000 --learning-starts 200
            --snapshot-interval 1000 --buffer-size 1000 --output-dir /tmp/mixer_smoke_test
            --seed 42 --mode adaptive  (+ uniform + td-per modes)
Result:     All 3 modes pass smoke test. Regime detector correctly transitions:
            noise (steps 0-700) → aligned (700+) once Gini > 0.3 and Q > 0.1.
            Stats logged: q_cv=0.025, td_gini=0.42, beta annealing works.
            Uniform and td-per baselines produce comparable Spearman (≈0, expected
            at 3k steps). No crashes, correct snapshot + regime JSON output.
Decision:   Next iteration: run full 100k-step comparison on reach-v3 (all 3 modes,
            seed=42) via Modal to measure whether adaptive switching actually improves
            sample efficiency vs uniform and td-per baselines.

## iter_008 — 100k mode comparison reveals PER integration bug  (2026-04-08T09:15:00Z)
Hypothesis: Adaptive Priority Mixer will outperform vanilla TD-PER and uniform
            baselines on reach-v3 by switching to uniform in noise/unstable regimes.
Change:     Added train_mixer_task() to modal_app.py with --compare flag for
            parallel 3-mode (adaptive/td-per/uniform) runs. Launched all 3 on Modal
            T4 GPUs. Created plot_mode_comparison.py for 4-panel comparison figure.
Command:    modal run modal_app.py --tasks reach-v3 --seeds 42 --compare
            python plot_mode_comparison.py
Result:     **CRITICAL BUG: SB3's SAC never calls update_priorities()** on the buffer.
            All PER priorities remain at initial max_priority=1.0 throughout training.
            td-per and uniform produced identical |TD| and Q trajectories (confirming
            PER was inactive). Adaptive mode went Q-unstable at 40k (q_cv=3.04,
            |TD| exploded to 4.35, Q jumped to 55.29) — possibly due to sum-tree
            sampling introducing subtle ordering bias vs standard ReplayBuffer.
            None of the 3 modes learned reach-v3 (ep_rew=0 at 50-90k steps),
            contrasting with earlier runs using train.py that learned by 60-90k.
            Runs partially completed: uniform=90k, td-per=60k, adaptive=50k.
            Figure: figures/mode_comparison_reach_v3.png
Decision:   Next iteration: fix PER integration by subclassing SAC to override
            train() and call buffer.update_priorities() with actual TD errors.
            This is the minimum viable change to make PER actually work with SB3.

## iter_009 — Fix SB3 PER integration: PERSAC subclass  (2026-04-08T10:00:00Z)
Hypothesis: Subclassing SAC.train() to extract per-sample TD errors and call
            update_priorities() will make PER actually differentiate priorities,
            and IS-weighted critic loss will keep gradients unbiased.
Change:     Created per_sac.py with PERSAC class that overrides train() to:
            (1) compute per-sample |TD| averaged across both critics,
            (2) call buffer.update_priorities(tree_inds, td_errors) each gradient step,
            (3) apply IS weights to critic loss when buffer provides them.
            Updated train_mixer.py and modal_app.py to use PERSAC for td-per/adaptive
            modes and vanilla SAC for uniform mode.
Command:    python train_mixer.py --task reach-v3 --total-steps 3000 --learning-starts 200
            --snapshot-interval 1000 --buffer-size 1000 --seed 42 --mode {td-per,adaptive,uniform}
Result:     All 3 modes pass smoke test. PER now active: max_priority=2.42 (was stuck
            at 1.0), tree_total varies across snapshots (141→166→186). Regime detector
            transitions noise→aligned at step 700. IS weights applied correctly.
            Uniform mode uses vanilla SAC (no PER overhead).
Decision:   Next iteration: re-run full 100k comparison on reach-v3 (all 3 modes,
            seed=42) via Modal with working PER. This will be the first valid
            head-to-head test of adaptive vs td-per vs uniform.

## iter_010 — 100k mode comparison with working PER  (2026-04-08T11:30:00Z)
Hypothesis: With PER priorities actually updating, TD-PER will show different
            training dynamics than uniform, and adaptive will outperform both
            by switching to uniform during noise/unstable regimes.
Change:     Re-ran 100k reach-v3 (seed=42) with all 3 modes via Modal using PERSAC
            for td-per/adaptive modes. Downloaded snapshots, updated plot_mode_comparison.py
            to 6-panel figure with Spearman + regime panels, updated FINDINGS.md.
Command:    modal run modal_app.py --tasks reach-v3 --seeds 42 --total-steps 100000 --compare
            python plot_mode_comparison.py
Result:     **PER now clearly active** — td-per max_priority=6.35, adaptive=17.92 (vs 1.0 before).
            **But working PER makes things WORSE:**
            - adaptive: Q oscillates wildly (14→37), |TD| explodes 30× (0.04→1.38),
              ends in "unstable" regime (q_cv=3.1). Catastrophically unstable.
            - td-per: Q overshoots (2.2 vs uniform's 0.5), |TD| 10× higher (0.048 vs 0.0045).
              Mostly "aligned" regime but with inverted episodes at 34k, 59k, 76k.
            - uniform: stable Q decline (28→0.5), lowest |TD| (0.005), cleanest dynamics.
            **Spearman ≈ 0 across ALL modes** (range: −0.03 to +0.13). No mode learned
            reach-v3 (ep_rew≈0), contrasting with iters 002-003 where same task/seed learned
            by 90k. Possible cause: train_mixer_task code path or Modal env difference.
            Wall time: adaptive=1737s, td-per=1796s, uniform=1210s.
            Figure: figures/mode_comparison_reach_v3.png (6-panel).
Decision:   Next iteration: investigate WHY no mode learned reach-v3 when the same
            task/seed learned in iter_002. Compare train_task vs train_mixer_task code
            paths (different callback? different buffer internals?). If the uniform
            mode with DenseRewardReplayBuffer + vanilla SAC doesn't reproduce earlier
            learning, there may be a Modal image version difference. This regression
            must be understood before the mode comparison results are meaningful.

## iter_011 — 5-seed baseline resolves learning regression  (2026-04-08T08:30:00Z)
Hypothesis: The reach-v3 learning "regression" in iter_010 is stochastic (CUDA
            non-determinism + sparse reward exploration variance), not a code bug.
            Code review shows train_task and train_mixer_task(uniform) are functionally
            identical for DenseRewardReplayBuffer (same SAC, same buffer, callback
            extra code skipped via hasattr checks). Early training dynamics match
            (Q=28, dense_rew=650 at 10k for both), confirming same environment.
Change:     Pinned MetaWorld to 3.0.0 in Modal image (was @master, risking drift).
            Ran 5-seed (42,123,7,99,256) uniform baseline on reach-v3 via Modal
            (100k steps each, T4 GPU). Added --replicate flag to Modal entrypoint.
            Fixed final-snapshot overwrite bug in TDInstrumentCallback._on_training_end.
            Generated 4-panel 5-seed figure, updated FINDINGS.md.
Command:    modal run modal_app.py --tasks reach-v3 --seeds "42,123,7,99,256" --modes uniform
Result:     **HYPOTHESIS CONFIRMED — regression is stochastic:**
            - seed=42:  ep_rew=0 at 90k (NO learn) — MATCHES iter_010
            - seed=123: ep_rew=469 at 90k (LEARNS) — matches iter_003
            - seed=7:   ep_rew=241 at 90k (LEARNS)
            - seed=99:  ep_rew=0 at 90k (NO learn)
            - seed=256: ep_rew=470 at 90k (LEARNS)
            **3/5 seeds learn (60% success rate)**, 2/5 don't. The iter_010 single-seed
            failure was expected variance, not a regression. Spearman for learning seeds
            rises from 0→0.6 at 70-80k; non-learning seeds stay at 0 throughout.
            This strengthens the core finding: TD-error is a LAGGING indicator that
            only becomes informative after learning has already started.
            Wall time: ~1160-1228s per run (5 parallel on Modal T4).
            Figure: figures/5seed_baseline_reach_v3.png
Decision:   Next iteration: run multi-seed mode comparison (5 seeds × 3 modes:
            uniform, td-per, adaptive) to get statistically meaningful comparison
            of whether PER helps or hurts. Need n=5 per mode to distinguish real
            effects from stochastic noise given the 60% base success rate.

## iter_012 — 5-seed mode comparison: TD-PER hurts, uniform wins  (2026-04-08T13:00:00Z)
Hypothesis: TD-PER will not improve sample efficiency over uniform replay across
            5 seeds, because TD-error is uninformative for most of training.
Change:     Ran 5 seeds × 2 remaining modes (td-per, adaptive) on reach-v3 via
            Modal (100k steps each, T4 GPU). Uniform baseline from iter_011.
            Created plot_multiseed_comparison.py (6-panel figure). Updated FINDINGS.md.
Command:    modal run modal_app.py --tasks reach-v3 --seeds "42,123,7,99,256" --modes "td-per,adaptive"
            python plot_multiseed_comparison.py
Result:     **HYPOTHESIS CONFIRMED — TD-PER ACTIVELY HURTS LEARNING:**
            - Uniform:  3/5 seeds learn (60%) — best performer
            - TD-PER:   0/5 seeds learn (0%) — worst performer
            - Adaptive: 2/5 seeds learn (40%) — middling
            Q-value explosion under PER: td-per Q=228±377 vs uniform Q=21±18 (11×).
            PER creates positive feedback: high-|TD| transitions get resampled →
            critic overfits → Q diverges → higher |TD| → more resampling.
            IS weights (β=0.4→1.0) do not prevent this divergence.
            Adaptive's regime detection helps (2 seeds escape via uniform fallback)
            but PER overhead still causes Q instability for non-learning seeds.
            Spearman ≈ 0 across all modes for non-learning seeds.
            Wall time: ~1700-2800s per run (10 parallel on Modal T4).
            Figure: figures/multiseed_mode_comparison.png
Decision:   Next iteration: this study's core question is now answered with
            statistical power. TD-PER is harmful on sparse-reward tasks. Options:
            (a) Run same comparison on pick-place-v3 to confirm on a harder task,
            (b) test alternative priority signals (random network distillation,
            reward prediction error, or VLM-based priorities),
            (c) investigate whether lower alpha (less aggressive PER) mitigates
            Q-explosion while retaining any benefit.

## iter_013 — Alpha sweep: lower α mitigates but never beats uniform  (2026-04-08T14:30:00Z)
Hypothesis: Q-explosion under TD-PER is a tuning issue — lower α (less aggressive
            prioritization) will mitigate Q-divergence and may recover learning.
Change:     Added alpha parameter to modal_app.py train_mixer_task. Ran 5 seeds ×
            2 alpha values (0.1, 0.3) for td-per mode on reach-v3 (100k steps).
            Created plot_alpha_sweep.py for 6-panel comparison figure.
Command:    modal run modal_app.py --tasks reach-v3 --seeds "42,123,7,99,256" --modes "td-per" --alpha {0.1,0.3}
            python plot_alpha_sweep.py
Result:     **Non-monotonic alpha effect — α=0.3 is best PER setting:**
            - Uniform:       3/5 learn, Q=20.8±18.3  (baseline)
            - TD-PER α=0.3:  3/5 learn, Q=36.6±26.2  (ties uniform!)
            - TD-PER α=0.1:  2/5 learn, Q=144.6±258.0 (one seed explodes to Q=660)
            - TD-PER α=0.6:  0/5 learn, Q=228.3±377.0 (catastrophic, from iter_012)
            Spearman ≈ 0 across ALL alpha values — TD-error remains uninformative
            regardless of prioritization strength. α=0.3 mitigates Q-explosion
            enough for seeds to survive, but provides ZERO benefit over uniform.
            The problem is the SIGNAL (TD-error is uninformative in sparse-reward
            early training), not the MECHANISM (prioritized sampling).
            Wall time: ~1900-2500s per run on Modal T4.
            Figure: figures/alpha_sweep_td_per.png
Decision:   The TD-error baseline study's core thesis is now fully supported:
            (1) TD-error is uninformative early (Spearman≈0), (2) TD-PER hurts at
            default settings, (3) even tuned TD-PER only matches uniform, never
            beats it. Next: either pick-place-v3 confirmation or start prototyping
            VLM-based priority signals (the original research goal).

## iter_014 — Hero summary figure + lit review incorporation  (2026-04-08T14:15:00Z)
Hypothesis: A single 4-panel figure synthesizing all 13 iterations of data will clearly
            communicate that TD-error PER is uninformative and harmful in sparse-reward
            early training — the prompt's core deliverable.
Change:     Created plot_summary_figure.py producing a publication-quality 4-panel figure:
            (a) Spearman correlation over training with individual seed traces + info
            desert annotation, (b) mode comparison bar chart including α sweep, (c) Q-value
            explosion dynamics under PER, (d) regime breakdown showing 7-50% aligned time.
            Killed lit_review2 subagent per Daniel's request. Pulled LIT_REVIEW.md (§1:
            11 alternative PER methods) onto this branch for preservation.
            Updated FINDINGS.md with hero figure reference.
Command:    python plot_summary_figure.py
Result:     Figure saved: figures/td_per_summary.{png,pdf}. All 4 panels render correctly.
            Key numbers confirmed: Uniform 3/5, TD-PER α=0.6 0/5, α=0.3 3/5 (ties),
            α=0.1 2/5, Adaptive 2/5. Spearman ≈ 0 for 60-80% of training. Pick-place-v3
            aligned regime only 7-13% vs reach-v3 20-50%.
Decision:   Next iteration: either (a) run pick-place-v3 5-seed mode comparison to confirm
            generalization (expected: all 0/5, but Q-stability data valuable), or (b) start
            RPE-PER implementation (lit review's #1 recommendation) to test whether a
            better priority SIGNAL helps where TD-error fails.
