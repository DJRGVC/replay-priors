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
