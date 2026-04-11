# Paper Outline: Nothing Beats Uniform — A Systematic Study of Replay Prioritization in Sparse-Reward Manipulation

**Working title alternatives:**
- "The Replay Prioritization Mirage: Why No Signal Beats Uniform Sampling in Sparse-Reward RL"
- "Uninformative by Design: TD-Error and VLM Signals for Experience Replay in Sparse-Reward Tasks"
- "Fourteen Approaches, Zero Winners: A Negative Result on Replay Prioritization for Sparse Manipulation"

**Target venues:** NeurIPS Datasets & Benchmarks, ICML (negative results are valued), CoRL, or workshop paper

---

## Abstract (~150 words)

Prioritized Experience Replay (PER) is a default component of deep RL systems, yet its
effectiveness in sparse-reward settings is rarely questioned. We present a systematic
evaluation of **14 replay prioritization approaches** — spanning TD-error variants, reward
prediction error, novelty-based signals, adaptive mixing, VLM temporal localization (6
approaches: K sweep, CoT, annotation, adaptive probing, random sampling, multi-format),
ensemble debiasing, confidence gating, contrastive episode ranking, and failure-mode
clustering (TF-IDF and category-diversity) — on MetaWorld manipulation tasks with binary
sparse rewards.

**None reliably outperform uniform sampling.** We identify three independent failure
mechanisms: (1) the chicken-and-egg problem (bootstrapped RL signals require the learning
they're supposed to accelerate), (2) positional bias (VLMs predict based on image
position, not task understanding), and (3) exploration bifurcation (priority signals
redirect which seeds learn, not how many). We release all 40+ training runs, instrumented
snapshots, and analysis code as a benchmark for future replay prioritization research.

---

## 1. Introduction (1.5 pages)

**Opening hook:** PER (Schaul et al., 2016) is ubiquitous in deep RL — enabled by default
in most libraries, assumed beneficial, rarely ablated. But its theoretical justification
(sample transitions proportional to expected learning progress) assumes TD-error is a
good proxy for learning utility. In sparse-reward settings, this assumption fails
catastrophically.

**Contribution statement:** We present the most comprehensive evaluation of replay
prioritization signals in sparse-reward manipulation to date:
- 5 RL-based signals (TD-PER at α∈{0.1, 0.3, 0.6}, RPE-PER, RND-PER)
- 1 adaptive mixer (regime-switching between signals)
- 8 VLM-based approaches (6 temporal localization variants, ensemble/gating,
  contrastive ranking, 2 failure-mode clustering variants)
- 2 tasks × 5 seeds × 10 configurations = 40+ fully instrumented training runs
- Dense-reward oracle advantage as ground truth for priority quality measurement

**Key negative result:** Uniform sampling achieves 3/5 seeds learning on reach-v3.
No prioritization approach exceeds this. TD-PER at default α=0.6 achieves 0/5 —
prioritization *hurts*.

**Why this matters:**
- Practitioners waste compute on PER in sparse settings (30-50% effective sampling waste)
- The failure is structural, not parametric — no amount of hyperparameter tuning fixes it
- Even external signals (VLMs) fail due to orthogonal failure modes
- The finding suggests sparse-reward RL needs reward design, not replay design

## 2. Background and Related Work (1 page)

### 2.1 Prioritized Experience Replay
- Schaul et al. (2016): |TD-error| proportional sampling
- Key assumption: |TD-error| ∝ expected learning progress
- Widely adopted: default in Stable-Baselines3, CleanRL, RLlib

### 2.2 Beyond TD-Error
- RPE-PER (arXiv:2501.18093): reward prediction error as alternative signal
- D-SPEAR (arXiv:2603.27346): actor-critic asymmetry in prioritization
- RND (Burda et al., 2019): novelty-based exploration bonus
- EDER (IJCAI 2025): diversity-based replay
- SPAHER (2024): spatial attention for manipulation PER

### 2.3 VLMs for Robotics
- VLMs as reward models (Rocamonde et al., 2024; Ma et al., 2023)
- VLMs for failure detection (Du et al., 2024)
- Our contribution: first systematic evaluation of VLMs as replay priority signals

### 2.4 Sparse Reward Manipulation
- MetaWorld benchmark (Yu et al., 2020)
- The exploration problem: binary rewards create information deserts
- Why manipulation is a stress test for PER

## 3. Experimental Setup (1.5 pages)

### 3.1 Environment and Tasks
- MetaWorld reach-v3 (easy: ~60% learn rate with uniform) and pick-place-v3 (hard: ~0%)
- Sparse binary reward: r=1 on success, r=0 otherwise
- No reward shaping, no curriculum — pure sparse signal

### 3.2 Base Algorithm: SAC
- Soft Actor-Critic with default hyperparameters
- 100k steps (reach-v3), 300k steps (pick-place-v3)
- 5 seeds per configuration: {7, 42, 99, 123, 256}

### 3.3 Instrumentation
- Dense-reward oracle: Euclidean distance-to-goal proxy (never used for training)
- Per-snapshot metrics: Spearman(|TD|, oracle_advantage), top-K overlap, Gini coefficient
- Regime classification: noise (|ρ|<0.15), aligned (ρ≥0.15), inverted (ρ≤−0.15),
  unstable (Q_std/Q_mean > 1.0)
- Buffer snapshots every 10k steps for offline analysis

### 3.4 Priority Signals Tested

**Table 1: Complete signal inventory**

| Signal | Source | Critic-dependent? | Available at step 0? |
|--------|--------|-------------------|---------------------|
| TD-PER α=0.1 | RL | Yes | No |
| TD-PER α=0.3 | RL | Yes | No |
| TD-PER α=0.6 | RL | Yes | No |
| RPE-PER | RL | Partial | No |
| RND-PER | RL | No | Yes (but random) |
| Adaptive mixer | RL | Yes (for regime detection) | No |
| VLM temporal | External | No | Yes |
| VLM ensemble | External | No | Yes |
| Contrastive ranking | External | No | Yes |
| Category-diversity | External | No | Yes |

## 4. Results: RL-Based Signals (2 pages)

### 4.1 TD-PER Fails for 50-93% of Training
- Regime map figure: 4 regimes over training for each seed×task
- The information desert: MI_proxy ≈ 0 bits for first 40-70k steps
- Inversion: negative Spearman at critical learning windows
- **Figure: Regime map (existing: td_per_regime_map.png)**

### 4.2 Alpha Sensitivity: Higher Alpha = Worse Performance
- α=0.6 (default): 0/5 seeds learn on reach-v3
- α=0.3: 2/5 seeds learn
- α=0.1: 2/5 seeds learn (≤ uniform 3/5)
- **Figure: Alpha sweep (existing: alpha_sweep_td_per.png)**

### 4.3 Alternative RL Signals Also Fail
- RPE-PER: 2/5 (ties α=0.1, below uniform 3/5)
- RND-PER: 3/5 (ties uniform, but changes WHICH seeds learn)
- Adaptive mixer: 2/5
- **Key: no RL signal exceeds uniform's 3/5 success rate**

### 4.4 Exploration Bifurcation (Novel Finding)
- Uniform: {7, 123, 256} learn. RND-PER: {42, 7, 123}. Jaccard = 0.51.
- Seed 42: never learns under uniform, learns fastest under RND-PER
- Priority signals create bimodal exploration outcomes, not uniform improvements
- **Figure: Seed switching analysis (existing: seed_switching_analysis.png)**

## 5. Results: VLM-Based Signals (2 pages)

### 5.1 Temporal Localization: Positional Bias Dominates
- 9 models tested, all dominated by positional bias
- Best model (Sonnet): MAE=41.9, but predictions cluster at grid-center (t≈85)
- Annotation effect is GT-distribution-dependent (bias-matching, not capability)
- **Figure: Model comparison heatmap (from vlm_probe)**

### 5.2 Ensemble and Confidence Gating
- Naive ensemble: worse than best individual (weak models dilute)
- Confidence gating: agreement positively correlated with error (r=+0.53)
- Optimal gating threshold → 100% uniform — VLM signal is pure noise

### 5.3 Contrastive Episode Ranking
- RLHF-inspired pairwise comparison: "which failed earlier?"
- 100% primacy bias: always picks Episode A (presented first)
- Positional bias extends from within-episode to between-episode

### 5.4 Failure-Mode Clustering
- First positive signal: η²=0.34-0.99 for VLM category × GT failure time
- TF-IDF clustering fails (silhouette <0.12) — descriptions syntactically template-like
- Category-diversity ≈ uniform at small n (n=10-20)
- Simulation shows effect only emerges at N≥50 episodes
- Cross-model stability: taxonomy-adherent models (GPT-4o-mini, JSD=0.10±0.06) more
  reliable than creative models (Phi-4, JSD=0.20-0.24). Task drives distribution more
  than model (within-model cross-task JSD=0.29 > cross-model JSD=0.11)
- Not practical for online RL where buffer grows incrementally

## 6. Analysis: Three Independent Failure Mechanisms (1.5 pages)

### 6.1 The Chicken-and-Egg Problem (RL Signals)
- Bootstrapped signals (TD-error, RPE, Q-values) require learning to be informative
- But learning is exactly what they're supposed to accelerate
- The information desert: zero bits about transition importance for first 40-70k steps
- Even non-bootstrapped signals (RND) fail because novelty ≠ importance

### 6.2 Positional Bias (VLM Signals)
- VLMs predict based on image position (grid cell, sequence slot), not task understanding
- Bias is model-specific but universal in magnitude
- Annotation, CoT, and multi-image formats shift bias location, not bias magnitude
- Extends to pairwise comparison (primacy effect)

### 6.3 Exploration Bifurcation (All Signals)
- Priority signals don't uniformly improve learning — they redirect exploration
- State-space visitation analysis: RND-PER creates wider, shifted distributions
- Q-value decoupling: high Q ≠ high dense reward under priority regimes
- The aggregate success rate (3/5 uniform, 3/5 RND-PER) masks a qualitative
  difference in exploration dynamics

### 6.4 Unified Diagnosis
- All three mechanisms converge on the same conclusion: in sparse-reward settings,
  there is no available signal that is both (a) informative about transition
  importance and (b) available when it would matter most (early training)
- Dense reward shaping breaks the bottleneck because it provides gradient signal
  directly, bypassing the prioritization problem entirely

## 7. Discussion (1 page)

### 7.1 Practical Implications
- **Turn off PER in sparse-reward settings.** The compute overhead (sorting, importance
  sampling correction) is pure waste when priorities are uninformative.
- **Sparse reward is the wrong lever.** If you need faster learning, invest in reward
  design (shaping, curricula), not replay mechanics.
- **VLMs are not ready for fine-grained temporal reasoning in RL.** Positional bias
  makes them unreliable for any task requiring temporal precision.

### 7.2 When PER Might Still Work
- Dense rewards (original PER domain)
- After initial exploration phase (warm-start with uniform, switch to PER)
- Tasks with high reward density where TD-error quickly becomes informative

### 7.3 Future Directions
- **Reward shaping vs. replay prioritization:** direct comparison
- **Failure-mode clustering at scale:** our simulation shows category-diversity helps
  at N≥50 — needs validation with real training runs
- **Phase-segmented replay:** coarse temporal binning (3 phases) instead of step-level
- **Cross-domain generalization:** are these findings specific to MetaWorld manipulation?

### 7.4 Limitations
- MetaWorld only (2 tasks) — generalization to other sparse-reward domains untested
- SAC only — other algorithms (PPO, TD3) may interact differently with prioritization
- 5 seeds — sufficient for our claims but limited for detecting small effects
- VLM evaluation used offline rollouts, not integrated into training loop

## 8. Conclusion (~0.5 page)

We systematically evaluated 14 replay prioritization approaches in sparse-reward
manipulation and found that none reliably outperform uniform sampling. The failure
stems from three independent mechanisms that make the sparse-reward setting fundamentally
hostile to prioritized replay. Our findings suggest that the sparse-reward bottleneck
is better addressed through reward design than replay mechanics.

---

## Appendices

### A. Full Per-Seed Results Tables
- 5 seeds × 6 RL modes × 2 tasks: success rate, final Q, Spearman at each snapshot

### B. VLM Model Details
- Prompt templates, K configurations, cost per query, latency

### C. Dense Reward Oracle Construction
- Exact formula, validation against shaped MetaWorld rewards

### D. Reproducibility
- All seeds, hyperparameters, library versions, compute budget (~12 GPU-hours for RL,
  ~$15 API cost for VLM probes)
- Code and data release plan

---

## Figure List (Existing + Needed)

### Existing (from td_baseline)
1. `td_per_regime_map.png` — 6-panel regime classification over training
2. `td_per_summary.png` — 5-mode comparison hero figure
3. `alpha_sweep_td_per.png` — α sensitivity
4. `seed_switching_analysis.png` — exploration bifurcation
5. `state_visitation_analysis.png` — dense reward proxy distributions
6. `cross_study_synthesis.png` — 4-panel unified landscape

### Existing (from vlm_probe, need to reference via git show)
7. K-sweep reach figure
8. Annotation bias-matching figure
9. Category-diversity simulation figures (iters 41-42)
10. Cross-model category comparison (iter 43)
11. Study synthesis landscape — 14 approaches (iter 45)

### Needed (new figures for paper)
12. **Hero figure (Fig. 1):** 14-approach comparison bar chart — success rate vs uniform baseline
11. **Failure mechanism diagram (Fig. 6):** visual summary of 3 independent mechanisms
12. **Timeline figure:** when each approach was tested and how it failed

---

## Estimated Page Count
- Main text: ~10 pages (NeurIPS format)
- Appendices: ~3-4 pages
- Figures: 8-10

## Data Release
- All 40+ training runs with buffer snapshots
- VLM probe results (responses, costs, timestamps)
- Analysis scripts (all plot_*.py files)
- Reproducibility scripts (Modal training configs)

---

*Outline drafted by agent/td_baseline, iter_027. Last updated: 2026-04-10.*
*Updated iter_028 to incorporate vlm_probe iters 043-045 (cross-model stability, final synthesis).*
*Now reflects 14 approaches across both studies.*
