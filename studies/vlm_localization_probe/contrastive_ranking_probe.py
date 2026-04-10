"""Contrastive Episode Ranking (CER) probe.

Instead of absolute temporal localization ("at what timestep did the robot fail?"),
CER asks a *relative* question: "which of these two episodes had its failure earlier?"

This tests whether VLMs are better at pairwise comparison (a simpler cognitive task)
than absolute temporal grounding. If pairwise accuracy is high, a Bradley-Terry model
can rank episodes by failure severity — enabling a replay buffer priority scheme that
doesn't require precise timestep prediction.

Usage:
    python contrastive_ranking_probe.py \
        --tasks reach-v3 \
        --K 4 \
        --models "gh:gpt-4o" "gh:gpt-4o-mini" \
        --max-rollouts 10 \
        --call-delay 5 \
        --output-dir results/cer_iter38
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_client import (
    TASK_DESCRIPTIONS,
    sample_keyframes,
    annotate_frame,
    _pil_to_base64,
    _retry_on_rate_limit,
    _parse_json_from_text,
)

# ── CER-specific prompts ─────────────────────────────────────────

CER_SYSTEM_PROMPT = """\
You are comparing two robot manipulation rollouts that both FAILED their task.
You will see keyframes from Episode A and Episode B, sampled at the same timestep \
positions within each episode.
Your job: determine which episode had its CRITICAL FAILURE occur EARLIER in time.

The critical failure is the moment the robot's behavior most clearly diverged from \
what would achieve the task goal — not the end of the episode, but the moment things \
first went wrong.

Respond with ONLY a valid JSON object on a single line, nothing else:
{"earlier_failure": "A" or "B", "confidence": <float 0-1>, "rationale": "<one sentence>"}
"""


def _build_cer_user_message(
    task_description: str,
    K: int,
    keyframe_indices: list[int],
    total_timesteps: int,
) -> str:
    """Build text portion of CER comparison prompt."""
    return "\n".join([
        f"Task: {task_description}",
        f"Total episode length: {total_timesteps} timesteps",
        f"Both episodes FAILED (the task was not completed).",
        f"",
        f"You are shown {K} keyframes from each episode at these timestep indices:",
        f"  {keyframe_indices}",
        f"",
        f"First you will see all {K} frames from Episode A, then all {K} frames from Episode B.",
        f"Determine which episode's critical failure occurred EARLIER in time.",
    ])


def _parse_cer_response(text: str) -> dict:
    """Parse CER JSON response."""
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    import re
    matches = list(re.finditer(r'\{[^{}]*"earlier_failure"[^{}]*\}', text))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass

    # Fallback
    return {"earlier_failure": None, "confidence": 0.0, "rationale": f"parse_error: {text[:200]}"}


# ── CER API call (GitHub Models) ─────────────────────────────────

def call_cer_github(
    task_description: str,
    frames_a: list[Image.Image],
    frames_b: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "gpt-4o",
    annotate: bool = True,
) -> dict:
    """Call GitHub Models API with a CER pairwise comparison prompt."""
    import os
    import subprocess
    from openai import OpenAI

    token = os.environ.get("GITHUB_TOKEN") or subprocess.check_output(
        ["gh", "auth", "token"], text=True
    ).strip()

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    user_text = _build_cer_user_message(
        task_description, len(frames_a), keyframe_indices, total_timesteps,
    )

    # Optionally annotate frames
    if annotate:
        frames_a = [annotate_frame(img, idx, total_timesteps) for img, idx in zip(frames_a, keyframe_indices)]
        frames_b = [annotate_frame(img, idx, total_timesteps) for img, idx in zip(frames_b, keyframe_indices)]

    content = [{"type": "text", "text": user_text}]

    # Episode A frames
    content.append({"type": "text", "text": "\n--- Episode A ---"})
    for img, idx in zip(frames_a, keyframe_indices):
        b64 = _pil_to_base64(img, fmt="JPEG")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
        content.append({"type": "text", "text": f"↑ Episode A, Timestep {idx}"})

    # Episode B frames
    content.append({"type": "text", "text": "\n--- Episode B ---"})
    for img, idx in zip(frames_b, keyframe_indices):
        b64 = _pil_to_base64(img, fmt="JPEG")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
        content.append({"type": "text", "text": f"↑ Episode B, Timestep {idx}"})

    content.append({"type": "text", "text": "\nWhich episode's critical failure occurred EARLIER? Respond with ONLY the JSON object."})

    def _do_call():
        return client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": CER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()
    result = _parse_cer_response(text)

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
    result["_output_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
    result["_model"] = model
    result["_raw_text"] = text
    return result


# ── Main sweep ────────────────────────────────────────────────────

def load_rollouts(task: str, max_rollouts: int, data_dir: str = "data"):
    """Load rollout metadata and paths."""
    task_dir = Path(data_dir) / task
    rollouts = []
    for d in sorted(task_dir.iterdir())[:max_rollouts]:
        meta = json.load(open(d / "meta.json"))
        rollouts.append({
            "name": d.name,
            "path": d,
            "failure_t": meta["failure_timestep"],
            "num_steps": meta["num_steps"],
        })
    return rollouts


def run_cer_sweep(
    task: str,
    K: int,
    models: list[str],
    max_rollouts: int = 10,
    call_delay: float = 5.0,
    output_dir: str = "results/cer",
    annotate: bool = True,
):
    """Run CER probe on all pairs of rollouts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = load_rollouts(task, max_rollouts)
    desc = TASK_DESCRIPTIONS.get(task, task)
    pairs = list(itertools.combinations(range(len(rollouts)), 2))

    print(f"Task: {task}, K={K}, {len(rollouts)} rollouts → {len(pairs)} pairs")
    print(f"Models: {models}")
    print(f"GT failure times: {[r['failure_t'] for r in rollouts]}")
    print()

    # Sample keyframes once per rollout
    all_frames = {}
    all_indices = None
    for r in rollouts:
        frames, indices = sample_keyframes(r["path"] / "frames", K=K, strategy="uniform")
        all_frames[r["name"]] = frames
        if all_indices is None:
            all_indices = indices

    all_results = {}

    for model in models:
        model_name = model.replace("gh:", "").replace("/", "_")
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        results = []
        correct = 0
        total = 0

        for pi, (i, j) in enumerate(pairs):
            ra, rb = rollouts[i], rollouts[j]
            gt_a, gt_b = ra["failure_t"], rb["failure_t"]

            # Skip pairs where GT is identical (no correct answer)
            if gt_a == gt_b:
                print(f"  Pair {pi+1}/{len(pairs)}: {ra['name']} vs {rb['name']} — GT tied ({gt_a}={gt_b}), skipping")
                continue

            gt_earlier = "A" if gt_a < gt_b else "B"
            gt_gap = abs(gt_a - gt_b)

            print(f"  Pair {pi+1}/{len(pairs)}: {ra['name']}(t={gt_a}) vs {rb['name']}(t={gt_b}) — GT: {gt_earlier} earlier (gap={gt_gap})")

            try:
                result = call_cer_github(
                    task_description=desc,
                    frames_a=all_frames[ra["name"]],
                    frames_b=all_frames[rb["name"]],
                    keyframe_indices=all_indices,
                    total_timesteps=ra["num_steps"],
                    model=model.replace("gh:", ""),
                    annotate=annotate,
                )

                pred = result.get("earlier_failure")
                is_correct = pred == gt_earlier

                result["pair"] = (ra["name"], rb["name"])
                result["gt_a"] = gt_a
                result["gt_b"] = gt_b
                result["gt_earlier"] = gt_earlier
                result["gt_gap"] = gt_gap
                result["is_correct"] = is_correct

                results.append(result)

                if pred in ("A", "B"):
                    total += 1
                    if is_correct:
                        correct += 1

                status = "✓" if is_correct else "✗"
                print(f"    {status} Pred: {pred}, Conf: {result.get('confidence', '?'):.2f}, Latency: {result['_latency_s']:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "pair": (ra["name"], rb["name"]),
                    "gt_a": gt_a, "gt_b": gt_b, "gt_earlier": gt_earlier,
                    "gt_gap": gt_gap, "error": str(e),
                })

            if call_delay > 0:
                time.sleep(call_delay)

        # Summary
        acc = correct / total if total > 0 else 0
        print(f"\n--- {model} Summary ---")
        print(f"Pairwise accuracy: {correct}/{total} = {acc:.1%}")

        # Accuracy by GT gap magnitude
        if results:
            gap_bins = [(0, 20), (20, 50), (50, 100), (100, 200)]
            for lo, hi in gap_bins:
                bin_results = [r for r in results if "is_correct" in r and lo <= r["gt_gap"] < hi]
                if bin_results:
                    bin_acc = sum(r["is_correct"] for r in bin_results) / len(bin_results)
                    print(f"  Gap [{lo},{hi}): {sum(r['is_correct'] for r in bin_results)}/{len(bin_results)} = {bin_acc:.1%}")

        all_results[model_name] = {
            "model": model,
            "task": task,
            "K": K,
            "n_rollouts": len(rollouts),
            "n_pairs": len(pairs),
            "n_valid": total,
            "correct": correct,
            "accuracy": acc,
            "results": results,
        }

        # Save per-model
        with open(out / f"{model_name}_{task}.json", "w") as f:
            json.dump(all_results[model_name], f, indent=2, default=str)

    # Save combined
    with open(out / "cer_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ── Analysis & figure ─────────────────────────────────────────────

def plot_cer_results(results_dir: str, output_path: str = None):
    """Plot CER accuracy by model and GT gap."""
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    summary_path = results_dir / "cer_summary.json"
    if not summary_path.exists():
        print("No summary found")
        return

    with open(summary_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Overall accuracy by model
    ax = axes[0]
    models = list(all_results.keys())
    accs = [all_results[m]["accuracy"] for m in models]
    bars = ax.bar(range(len(models)), accs, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"][:len(models)])
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline (50%)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Pairwise accuracy")
    ax.set_title("CER: Which episode failed earlier?")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{acc:.0%}", ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Accuracy by GT gap magnitude
    ax = axes[1]
    gap_bins = [(0, 20, "0-20"), (20, 50, "20-50"), (50, 100, "50-100"), (100, 200, "100+")]
    x = np.arange(len(gap_bins))
    width = 0.8 / len(models)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for mi, model_key in enumerate(models):
        bin_accs = []
        bin_ns = []
        for lo, hi, _ in gap_bins:
            model_results = all_results[model_key]["results"]
            bin_r = [r for r in model_results if "is_correct" in r and lo <= r.get("gt_gap", 0) < hi]
            if bin_r:
                bin_accs.append(sum(r["is_correct"] for r in bin_r) / len(bin_r))
                bin_ns.append(len(bin_r))
            else:
                bin_accs.append(0)
                bin_ns.append(0)
        bars = ax.bar(x + mi * width, bin_accs, width, label=model_key.replace("_", " "), color=colors[mi % len(colors)], alpha=0.8)
        for bar, acc, n in zip(bars, bin_accs, bin_ns):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"n={n}", ha="center", fontsize=7)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([label for _, _, label in gap_bins])
    ax.set_xlabel("GT gap (|t_A - t_B|) in timesteps")
    ax.set_ylabel("Pairwise accuracy")
    ax.set_title("CER accuracy by pair difficulty")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path is None:
        output_path = str(results_dir / "cer_accuracy.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Episode Ranking probe")
    parser.add_argument("--tasks", nargs="+", default=["reach-v3"])
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["gh:gpt-4o", "gh:gpt-4o-mini"])
    parser.add_argument("--max-rollouts", type=int, default=10)
    parser.add_argument("--call-delay", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/cer")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_cer_results(args.output_dir)
    else:
        for task in args.tasks:
            results = run_cer_sweep(
                task=task,
                K=args.K,
                models=args.models,
                max_rollouts=args.max_rollouts,
                call_delay=args.call_delay,
                output_dir=args.output_dir,
                annotate=not args.no_annotate,
            )
        plot_cer_results(args.output_dir)
