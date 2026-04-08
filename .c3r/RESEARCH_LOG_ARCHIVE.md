# RESEARCH_LOG ARCHIVE — td_baseline

Verbatim entries archived during compaction. Newest at bottom.

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

## iter_003 — Seed=123 runs + FINDINGS.md + lit_review subagent  (2026-04-08T05:50:00Z)
Hypothesis: Second seed will confirm that TD-error uninformativeness is robust, not
            seed-dependent.
Change:     Ran seed=123 on both tasks via Modal (100k steps each). Updated
            plot_td_correlation.py to support multi-seed aggregation. Created FINDINGS.md.
            Spawned lit_review2 subagent (opus).
Command:    modal run studies/td_error_baseline/modal_app.py --seeds 123
            python plot_td_correlation.py --seeds 42,123
Result:     reach-v3 s123: Spearman pattern matches s42 — near-zero until ~50k, then
            rises (0.38→0.57 at 50-60k). Policy learned faster (ep_rew 10→379 by 100k).
            Spearman drops back to -0.09 at 90-100k despite strong policy — critic overshooting.
            pick-place-v3 s123: Spearman -0.05 to +0.19, never learns (ep_rew=0).
Decision:   Run oracle_correlation.py for Gini + top-K overlap metrics.

## iter_004 — Gini + top-K overlap analysis + priority quality figure  (2026-04-08T06:15:00Z)
Hypothesis: Top-K overlap between |TD| and oracle advantage will be near chance (10%)
            early in training.
Change:     Fixed oracle_correlation.py. Ran analysis on all 4 runs. Created plot_priority_quality.py.
Command:    python oracle_correlation.py --run-dirs snapshots/{reach,pick-place}-v3_s{42,123}
            python plot_priority_quality.py
Result:     reach-v3: top-10% overlap at chance (7-20%) for first 40k, brief spike to 53-61%
            during learning, then drops back to 6-11%. Gini 0.26-0.54.
            pick-place-v3: overlap never exceeds ~2× chance (28% max). Gini 0.30-0.60.
            NEW FINDING: reach-v3 s123 Spearman inverts to -0.09/-0.12 at 90-100k.
Decision:   Check lit_review2, then run 200-500k pick-place-v3 or synthesize.

## iter_005 — Extended 300k pick-place-v3 + lit review pull  (2026-04-08T00:30:00Z)
Hypothesis: Pick-place-v3 will remain uncorrelated at 300k steps.
Change:     Ran 300k-step training on pick-place-v3 (seeds 42+123) via Modal.
Command:    modal run modal_app.py --tasks pick-place-v3 --seeds "42,123" --total-steps 300000
Result:     HYPOTHESIS PARTIALLY REFUTED — seed42 showed marginal learning (ep_rew peaked 0.7)
            but Spearman INVERTED to −0.31 at 280k. Q-values oscillated wildly. Seed123 never
            learned. NEW FINDING: TD-error inversion is a general failure mode.
Decision:   Synthesize cross-study implications.

## iter_006 — Cross-study synthesis + regime map  (2026-04-08T07:00:00Z)
Hypothesis: Formalizing TD-PER's failure as a "regime classification" will make the
            case for VLM-augmented PER quantitatively precise.
Change:     Created plot_regime_map.py (6-panel figure), SYNTHESIS.md.
Command:    python studies/td_error_baseline/plot_regime_map.py
Result:     TD-PER fails 50-93% of training time across all runs. MI proxy shows
            "information desert" for first 40-70k steps (0 bits). Effective waste 50%+.
Decision:   Coordinate with vlm_probe or prototype Adaptive Priority Mixer.

## iter_007 — Adaptive Priority Mixer implementation + smoke test  (2026-04-08T08:30:00Z)
Hypothesis: A regime-aware replay buffer can be implemented as a drop-in SB3 subclass.
Change:     Created adaptive_priority_mixer.py, train_mixer.py (3 modes: adaptive/td-per/uniform).
Command:    python train_mixer.py --task reach-v3 --total-steps 3000 ... --mode {adaptive,td-per,uniform}
Result:     All 3 modes pass smoke test. Regime detector transitions noise→aligned correctly.
Decision:   Run full 100k comparison on reach-v3.

## iter_008 — 100k mode comparison reveals PER integration bug  (2026-04-08T09:15:00Z)
Hypothesis: Adaptive will outperform vanilla TD-PER and uniform.
Change:     Added train_mixer_task() to modal_app.py. Ran 3-mode comparison.
Command:    modal run modal_app.py --tasks reach-v3 --seeds 42 --compare
Result:     CRITICAL BUG: SB3 SAC never calls update_priorities(). PER was inactive.
            All modes produced identical dynamics. None learned reach-v3.
Decision:   Fix PER integration by subclassing SAC.

## iter_009 — Fix SB3 PER integration: PERSAC subclass  (2026-04-08T10:00:00Z)
Hypothesis: Subclassing SAC.train() to extract TD errors and call update_priorities() will fix PER.
Change:     Created per_sac.py (PERSAC class with IS-weighted critic loss).
Command:    python train_mixer.py ... --mode {td-per,adaptive,uniform}
Result:     PER now active: max_priority=2.42 (was 1.0). IS weights applied correctly.
Decision:   Re-run full 100k comparison with working PER.

## iter_010 — 100k mode comparison with working PER  (2026-04-08T11:30:00Z)
Hypothesis: Working PER will show different dynamics; adaptive will outperform.
Change:     Re-ran 100k reach-v3 (seed=42) with all 3 modes via Modal.
Command:    modal run modal_app.py --tasks reach-v3 --seeds 42 --total-steps 100000 --compare
Result:     PER active (max_priority=6.35/17.92) but makes things WORSE. Adaptive: Q oscillates
            (14→37), catastrophically unstable. TD-PER: Q overshoots (2.2 vs 0.5). Uniform: stable.
            Spearman ≈ 0 across all. No mode learned. Possible stochastic variance.
Decision:   Investigate learning regression — compare train_task vs train_mixer_task code paths.
