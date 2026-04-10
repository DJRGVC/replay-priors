# fix_plan.md — experiment queue for vlm_probe
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

- [ ] just like the td agent, occasionally add relevant literature to discord so i can review it.
- [x] Collect ~20 failure rollouts on each of 2-3 MetaWorld tasks — DONE iter_001
- [x] Build VLM client + keyframe sampling — DONE iter_002
- [x] Single-task accuracy probe on reach-v3 with K=8 — DONE iter_002 (MAE=41.9, ±10=20%)
- [x] K sweep on reach-v3 — DONE iter_003 (K=4/8/16/32: no improvement with more frames, MAE flat ~42-52)
- [x] **Add Gemini backend to vlm_client.py** — DONE iter_004. Flash-lite MAE=95.2 (much worse than Sonnet 41.9). Flash quota exhausted (20 RPD!).
- [ ] **Re-run reach-v3 probe with gemini-2.5-flash** (thinking_budget=0 fix applied) — wait for quota reset
- [x] **Try gemini-3-flash-preview** — DONE iter_005. MAE=54.2 (median 14), ±10=44%, ±20=56%. 2× better ±10 than Sonnet. Start-bias, bimodal. gemini-2.0-flash has 0 free RPD (dead).
- [x] **Frame index annotation (VTimeCoT-style)** — DONE iter_007. MAE 71.9→59.5 (-17%), ±10 10%→20% on flash-lite. Needs test on stronger models.
- [x] **Test annotation on gemini-3-flash-preview** — DONE iter_028. NO effect: 8/10 predictions identical (MAE 69.9→67.3, −4%). Breaks U-shaped narrative.
- [ ] **Test annotation on gemini-2.5-flash** — still 503 UNAVAILABLE
- [ ] **Complete gemini-3-flash-preview CoT comparison** — need n≥9 to match direct baseline. Rate limits: run when quota resets.
- [ ] Run remaining 10 rollouts on gemini-3-flash-preview direct for n=20
- [ ] Run probe on push-v3 and pick-place-v3 (may have more visually distinct failure modes)
- [x] Test "random" sampling strategy vs uniform — DONE iter_014. No improvement (MAE 64.7 vs 63.8, both ±10=10%). Breaks grid-position clustering but doesn't help accuracy.
- [ ] Test two-pass adaptive probing (coarse K=4 → refine K=8 around t_hat, new in iter_011)
- [x] **Implement proprio-as-text augmentation** — DONE iter_008. Code ready (`--proprio` flag). Flash-lite test: n=2 valid (rate-limited), MAE=107.5 vs 59.5 baseline. Negative signal but n too small. Needs retest.
- [ ] **Retest proprio-as-text with n≥5 valid** — wait for rate limit reset
- [ ] **Groq Llama 4 Scout probe** — backend code ready, needs GROQ_API_KEY in .c3r/env.sh
- [x] **GitHub Models backend (Llama 3.2 Vision)** — DONE iter_013. Grid tiling for 1-image limit. 11B MAE=72.9, 90B MAE=53.5. Grid-position bias.
- [ ] **Test random sampling on Llama-3.2-90B** — may break grid-position bias (no rate limits, can run freely)
- [x] **Two-pass adaptive probing on Llama-3.2-90B** — DONE iter_015. NEGATIVE: coarse MAE=69.8→fine MAE=71.3, 6/10 worsened. Coarse pass too inaccurate to center refinement.
- [x] **CoT prompting on Phi-4 (GitHub Models)** — DONE iter_016. NEGATIVE: MAE 64.3→90.2 (+40%). CoT definitively ruled out for mid/weak models (3/3 negative). Only gemini-3-flash-preview suggestive (n=3).
- [x] **Explore other GitHub Models vision models** — DONE iter_013+016. Phi-4 MAE=64.3 (grid-center bias at t=85), similar to Llama-3.2-90B. Grid-position bias is universal with tiled single-image APIs.
- [ ] **Cohere aya-vision-32b** — 1000 req/month free, native multi-image (no grid needed)
- [ ] Open questions: definition of "failure" for time-out tasks; vision-only vs vision+proprio-as-text
- [ ] **Integrate TD-baseline results into HTML report** — td_baseline requests: uniform 3/5, TD-PER 0/5, RPE-PER 2/5, hero figure at studies/td_error_baseline/figures/td_per_summary.png on agent/td_baseline
- [ ] Budget: $0 going forward. DO NOT use Anthropic API key (costs real money). Use only free APIs.

