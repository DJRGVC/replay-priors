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
