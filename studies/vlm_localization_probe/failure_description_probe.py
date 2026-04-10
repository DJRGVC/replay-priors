"""Failure Mode Clustering probe (Proposal 4).

Instead of temporal localization ("when did it fail?"), ask VLMs to describe
the failure mode ("what went wrong?"). Collect free-text descriptions, then
analyze for semantic diversity and clustering structure.

Usage:
    python failure_description_probe.py --tasks reach-v3 --K 4 --models "gh:gpt-4o-mini" \
        --max-rollouts 10 --call-delay 8 --output-dir results/failure_descriptions_iter39
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image
from vlm_client import sample_keyframes, TASK_DESCRIPTIONS, _pil_to_base64, _retry_on_rate_limit


SYSTEM_PROMPT = """You are a robotics failure analysis expert. You will be shown keyframes
from a robotic manipulation episode that FAILED. Your job is to describe the failure mode —
what went wrong and why.

Respond with ONLY a JSON object (no markdown, no explanation) with these fields:
{
  "failure_mode": "A 1-2 sentence description of what went wrong",
  "failure_category": "One of: never_reached, overshot, oscillated, wrong_direction, stuck, other",
  "severity": "mild | moderate | severe",
  "visual_cues": ["list", "of", "specific", "visual", "observations"]
}"""


def describe_failure(task_name, frames_dir, K, model, call_delay=0):
    """Get VLM failure description for a single rollout."""
    from openai import OpenAI
    import subprocess

    frames, indices = sample_keyframes(frames_dir, K=K, strategy="uniform")

    token = os.environ.get("GITHUB_TOKEN") or subprocess.check_output(
        ["gh", "auth", "token"], text=True
    ).strip()

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    task_desc = TASK_DESCRIPTIONS[task_name]
    user_text = (
        f"Task: {task_desc}\n\n"
        f"This episode FAILED to complete the task. "
        f"Below are {K} keyframes sampled uniformly from the {len(indices)}-frame episode.\n\n"
        f"Analyze the keyframes and describe the failure mode."
    )

    # Build multi-image content
    is_gpt = "gpt" in model.lower()
    if is_gpt:
        content = [{"type": "text", "text": user_text}]
        for img, idx in zip(frames, indices):
            b64 = _pil_to_base64(img, fmt="JPEG")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })
            content.append({"type": "text", "text": f"↑ Frame {idx}/{max(indices)}"})
        content.append({
            "type": "text",
            "text": "Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line."
        })
    else:
        # Grid tiling for single-image models
        from vlm_client import _tile_keyframes
        grid = _tile_keyframes(frames, indices, max(indices) + 1)
        b64 = _pil_to_base64(grid, fmt="JPEG")
        content = [
            {"type": "text", "text": user_text + f"\n\nThe image below is a {K}-frame grid.\n\nNow respond with ONLY the JSON object."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
        ]

    def _do_call():
        return client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Try direct parse
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown
        import re
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {"failure_mode": text, "failure_category": "parse_error", "severity": "unknown", "visual_cues": []}
        else:
            result = {"failure_mode": text, "failure_category": "parse_error", "severity": "unknown", "visual_cues": []}

    result["_latency_s"] = round(latency, 2)
    result["_raw_response"] = text
    return result


def main():
    parser = argparse.ArgumentParser(description="Failure mode description probe")
    parser.add_argument("--tasks", nargs="+", default=["reach-v3"])
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--max-rollouts", type=int, default=10)
    parser.add_argument("--call-delay", type=int, default=8)
    parser.add_argument("--output-dir", default="results/failure_descriptions")
    args = parser.parse_args()

    data_dir = Path("data")
    os.makedirs(args.output_dir, exist_ok=True)

    for task in args.tasks:
        for model in args.models:
            model_key = model.replace("gh:", "gh_").replace("/", "_")
            print(f"\n{'='*60}")
            print(f"Config: {task}/{model}/K={args.K}")
            print(f"{'='*60}")

            results = []
            task_dir = data_dir / task
            rollout_dirs = sorted(task_dir.iterdir())[:args.max_rollouts]

            for i, rd in enumerate(rollout_dirs):
                if not rd.is_dir():
                    continue
                meta_path = rd / "meta.json"
                if not meta_path.exists():
                    continue

                with open(meta_path) as f:
                    meta = json.load(f)

                try:
                    desc = describe_failure(
                        task_name=task,
                        frames_dir=rd / "frames",
                        K=args.K,
                        model=model.replace("gh:", ""),
                    )
                    print(f"  {rd.name}: cat={desc.get('failure_category', '?'):15s} | {desc.get('failure_mode', '?')[:80]}", flush=True)
                    results.append({
                        "rollout": rd.name,
                        "gt_failure_t": meta["failure_timestep"],
                        "gt_failure_type": meta.get("failure_type", "unknown"),
                        **desc,
                    })
                except Exception as e:
                    print(f"  {rd.name}: FAILED - {e}", flush=True)
                    results.append({
                        "rollout": rd.name,
                        "gt_failure_t": meta["failure_timestep"],
                        "error": str(e),
                    })

                if i < len(rollout_dirs) - 1:
                    time.sleep(args.call_delay)

            # Save results
            outfile = Path(args.output_dir) / f"{task}_{model_key}_K{args.K}.json"
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved {len(results)} descriptions to {outfile}")

            # Summary statistics
            categories = [r.get("failure_category", "error") for r in results if "error" not in r]
            from collections import Counter
            print(f"\nCategory distribution: {dict(Counter(categories).most_common())}")
            descriptions = [r.get("failure_mode", "") for r in results if "failure_mode" in r]
            unique_descs = len(set(descriptions))
            print(f"Unique descriptions: {unique_descs}/{len(descriptions)}")


if __name__ == "__main__":
    main()
