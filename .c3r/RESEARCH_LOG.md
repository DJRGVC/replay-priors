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
