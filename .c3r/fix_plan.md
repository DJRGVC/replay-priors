# fix_plan.md — experiment queue for vlm_probe
#
# Pruned at iter_041 compaction. Completed tasks removed, forward-looking only.

- [ ] Occasionally add relevant literature to Discord for Daniel to review.
- [ ] Budget: $0. DO NOT use Anthropic API key (costs real money). Use only free APIs.

# --- Active work: Proposal 4 (Failure Mode Clustering) ---
# TF-IDF embedding failed (iter_040). VLM categories are the viable signal.
- [x] Proposal 4 Step 1: Description collection — DONE iter_039. 3 tasks, 2 models, high diversity (η²=0.34-0.99).
- [x] Proposal 4 Step 2: TF-IDF clustering — DONE iter_040. FAILED (silhouette<0.12). Categories > embeddings.
- [x] Proposal 4 Step 3: Category-diversity simulation — iter_041 (n=20: ≈uniform), iter_042 scale-up (N≥50: +5-8% coverage). Viable at realistic buffer sizes.
- [ ] Proposal 4 Step 4: Cross-model category comparison — RE-PRIORITIZED. Validate category stability across models before claiming scale-up generalizes.

# --- Remaining proposals ---
- [ ] Proposal 3: Task-Adaptive Annotation — annotation format search space, task-conditioned selection.
- [ ] Proposal 6: Phase-Segmented Replay — categorical phase labels. Needs semi-trained policy.
- [ ] Proposal 7: Retrospective Failure Narration — narration specificity as priority signal. Most speculative.

# --- Quota-gated experiments (run when available) ---
- [ ] Gemini-3-flash-preview CoT completion (need n≥9, have n=3)
- [ ] Gemini-2.5-flash annotation test (still 503)
- [ ] Groq Llama 4 Scout (needs GROQ_API_KEY)
- [ ] Cohere aya-vision-32b (1000 req/month, native multi-image)
- [ ] Integrate TD-baseline results into HTML report

