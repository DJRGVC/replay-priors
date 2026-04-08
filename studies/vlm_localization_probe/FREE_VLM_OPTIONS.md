# Free VLM Options for Failure-Timestep Probing

Researched 2026-04-08 after discovering Anthropic API costs real money.

## Recommended: Gemini 2.5 Flash (Free Tier)

- **Rate limits**: 10 RPM, **20 requests/day** (confirmed empirically 2026-04-08)
- **Vision**: Full multi-image support, images count as tokens
- **Context**: 1M token window (same as paid)
- **Cost**: $0 (free tier, no billing needed)
- **Integration**: `google-genai` Python SDK (new SDK, `google-generativeai` deprecated)
- **Caveat**: Thinking model — `thinking_budget=0` needed to avoid token waste on simple JSON tasks.
- **IMPORTANT**: 20 RPD is very low — only enough for 1 probe run per day. Use flash-lite for sweeps.

## Alternative: Gemini 2.5 Flash-Lite (Free Tier)

- 15 RPM, **1000 RPD** — much more generous, good for sweeps
- May be less capable on nuanced visual reasoning

## Alternative: NVIDIA build.nvidia.com (Qwen3.5-VL)

- Free GPU-accelerated endpoints with NVIDIA Developer registration
- Qwen3.5 VL — strong open-source VLM
- Multi-image support
- Rate limits unclear

## Alternative: Local VLMs on Modal

- Qwen2.5-VL-72B or InternVL3-78B via Modal GPU
- ~5-10% behind proprietary models on benchmarks
- Free if Modal credits available (Daniel mentioned having credits)
- More setup work but no rate limits

## Decision

Start with **Gemini 2.5 Flash free tier** — easiest integration, zero cost, sufficient rate limits.
Add Gemini backend to vlm_client.py next iteration.
