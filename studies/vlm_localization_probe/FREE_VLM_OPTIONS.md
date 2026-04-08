# Free VLM Options for Failure-Timestep Probing

Researched 2026-04-08 after discovering Anthropic API costs real money.

## Recommended: Gemini 2.5 Flash (Free Tier)

- **Rate limits**: 10 RPM, 250 requests/day, 250K TPM
- **Vision**: Full multi-image support, images count as tokens
- **Context**: 1M token window (same as paid)
- **Cost**: $0 (free tier, no billing needed)
- **Integration**: `google-generativeai` Python SDK, very similar API pattern
- **Caveat**: As of April 2026, Pro models paywalled for free users. Flash still free.

For our use case (20 rollouts × a few K values = 80-200 calls), 250 RPD is plenty.

## Alternative: Gemini 2.5 Flash-Lite (Free Tier)

- 15 RPM, 1000 RPD — even more generous
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
