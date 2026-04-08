"""Run VLM failure-localization probe over collected rollouts.

Sweeps over tasks, K values, sampling strategies, and models.
Reports: absolute timestep error, window accuracy at ±5/±10, latency, cost.

Usage:
    # Single-task quick test
    python run_probe.py --tasks reach-v3 --K 8 --models claude-sonnet-4-6

    # Full sweep
    python run_probe.py --tasks reach-v3 push-v3 pick-place-v3 \
        --K 2 4 8 16 --models claude-sonnet-4-6 gpt-4o \
        --strategies uniform pinned
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vlm_client import predict_failure, sample_keyframes, sample_keyframes_around, TASK_DESCRIPTIONS, annotate_frame, extract_proprio_text


# ── Cost estimation (approximate, per 1M tokens) ──────────────────
# Updated as of 2026-04. Adjust if prices change.
COST_PER_1M = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},  # free tier
    "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},  # free tier
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},  # free tier
    "gemini-3-flash-preview": {"input": 0.0, "output": 0.0},  # free tier
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.0, "output": 0.0},  # Groq free tier
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def load_rollouts(data_dir: Path, task_name: str) -> list[dict]:
    """Load all rollout metadata for a task."""
    task_dir = data_dir / task_name
    rollouts = []
    for rd in sorted(task_dir.iterdir()):
        if not rd.is_dir():
            continue
        meta_path = rd / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = str(rd)
        rollouts.append(meta)
    return rollouts


def evaluate_prediction(gt_timestep: int, pred_timestep: int | None, total_steps: int) -> dict:
    """Compute error metrics for a single prediction."""
    if pred_timestep is None:
        return {
            "abs_error": total_steps,  # worst case
            "within_5": False,
            "within_10": False,
            "within_20": False,
            "normalized_error": 1.0,
        }
    abs_err = abs(gt_timestep - pred_timestep)
    return {
        "abs_error": abs_err,
        "within_5": abs_err <= 5,
        "within_10": abs_err <= 10,
        "within_20": abs_err <= 20,
        "normalized_error": abs_err / max(total_steps, 1),
    }


def run_sweep(
    data_dir: Path,
    tasks: list[str],
    K_values: list[int],
    strategies: list[str],
    models: list[str],
    output_dir: Path,
    max_rollouts: int | None = None,
    prompt_styles: list[str] | None = None,
    annotate: bool = False,
    use_proprio: bool = False,
    call_delay: float = 0.0,
):
    """Run the full sweep and save results."""
    if prompt_styles is None:
        prompt_styles = ["direct"]
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_cost = 0.0

    for task_name in tasks:
        rollouts = load_rollouts(data_dir, task_name)
        if max_rollouts:
            rollouts = rollouts[:max_rollouts]

        task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)

        for model in models:
            for K in K_values:
                for strategy in strategies:
                    for prompt_style in prompt_styles:
                        config_key = f"{task_name}/{model}/K={K}/{strategy}/{prompt_style}"
                        print(f"\n{'='*60}")
                        print(f"Config: {config_key}")
                        print(f"{'='*60}")

                        config_results = []
                        call_count = 0
                        for rollout in rollouts:
                            if rollout["success"]:
                                continue  # skip successes

                            if call_delay > 0 and call_count > 0:
                                print(f"    [delay {call_delay:.0f}s between calls]")
                                time.sleep(call_delay)
                            call_count += 1

                            rd = Path(rollout["_dir"])
                            frames_dir = rd / "frames"

                            try:
                                frames, indices = sample_keyframes(frames_dir, K=K, strategy=strategy)
                            except ValueError as e:
                                print(f"  SKIP {rd.name}: {e}")
                                continue

                            # Load proprio if requested
                            proprio_labels = None
                            if use_proprio:
                                proprio_path = rd / "proprio.npy"
                                if proprio_path.exists():
                                    proprio = np.load(proprio_path)
                                    proprio_labels = extract_proprio_text(task_name, proprio, indices)

                            try:
                                pred = predict_failure(
                                    task_description=task_desc,
                                    keyframes=frames,
                                    keyframe_indices=indices,
                                    total_timesteps=rollout["num_steps"],
                                    model=model,
                                    prompt_style=prompt_style,
                                    annotate_frames=annotate,
                                    proprio_labels=proprio_labels,
                                )
                            except Exception as e:
                                print(f"  ERROR {rd.name}: {e}")
                                pred = {
                                    "failure_timestep": None,
                                    "confidence": 0.0,
                                    "rationale": f"api_error: {e}",
                                    "_latency_s": 0,
                                    "_input_tokens": 0,
                                    "_output_tokens": 0,
                                    "_model": model,
                                }

                            gt = rollout["failure_timestep"]
                            metrics = evaluate_prediction(gt, pred.get("failure_timestep"), rollout["num_steps"])
                            cost = estimate_cost(model, pred.get("_input_tokens", 0), pred.get("_output_tokens", 0))
                            total_cost += cost

                            entry = {
                                "task": task_name,
                                "model": model,
                                "K": K,
                                "strategy": strategy,
                                "prompt_style": prompt_style,
                                "annotate": annotate,
                                "proprio": use_proprio,
                                "rollout": rd.name,
                                "gt_failure_t": gt,
                                "gt_failure_type": rollout.get("failure_type"),
                                "gt_ambiguous": rollout.get("ambiguous"),
                                "pred_failure_t": pred.get("failure_timestep"),
                                "pred_confidence": pred.get("confidence"),
                                "pred_rationale": pred.get("rationale"),
                                **metrics,
                                "latency_s": pred.get("_latency_s", 0),
                                "input_tokens": pred.get("_input_tokens", 0),
                                "output_tokens": pred.get("_output_tokens", 0),
                                "cost_usd": round(cost, 6),
                            }
                            config_results.append(entry)
                            all_results.append(entry)

                            pred_t = pred.get('failure_timestep')
                            pred_str = f"{pred_t:>4d}" if pred_t is not None else "   ?"
                            status = f"gt={gt:3d} pred={pred_str} err={metrics['abs_error']:3d} lat={pred.get('_latency_s',0):.1f}s ${cost:.4f}"
                            print(f"  {rd.name}: {status}")

                        # Config summary
                        if config_results:
                            errs = [r["abs_error"] for r in config_results]
                            w5 = np.mean([r["within_5"] for r in config_results])
                            w10 = np.mean([r["within_10"] for r in config_results])
                            w20 = np.mean([r["within_20"] for r in config_results])
                            lats = [r["latency_s"] for r in config_results]
                            costs = [r["cost_usd"] for r in config_results]
                            print(f"\n  SUMMARY ({len(config_results)} rollouts):")
                            print(f"    Mean abs error: {np.mean(errs):.1f} (median: {np.median(errs):.1f})")
                            print(f"    ±5 acc: {w5:.1%}  ±10 acc: {w10:.1%}  ±20 acc: {w20:.1%}")
                            print(f"    Mean latency: {np.mean(lats):.1f}s  Mean cost: ${np.mean(costs):.4f}/call")

    # Save all results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total API cost: ${total_cost:.4f}")

    # Generate summary table
    _print_summary_table(all_results)

    return all_results


def _print_summary_table(results: list[dict]):
    """Print a compact summary table grouped by (task, model, K, strategy)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r["task"], r["model"], r["K"], r["strategy"], r.get("prompt_style", "direct"))
        groups[key].append(r)

    print(f"\n{'='*100}")
    print(f"{'Task':<18} {'Model':<25} {'K':>2} {'Strat':<8} {'Prompt':<8} {'MAE':>5} {'±5':>5} {'±10':>5} {'±20':>5} {'Lat':>5} {'$/call':>7}")
    print(f"{'='*100}")

    for key in sorted(groups.keys()):
        task, model, K, strategy, prompt_style = key
        rs = groups[key]
        mae = np.mean([r["abs_error"] for r in rs])
        w5 = np.mean([r["within_5"] for r in rs])
        w10 = np.mean([r["within_10"] for r in rs])
        w20 = np.mean([r["within_20"] for r in rs])
        lat = np.mean([r["latency_s"] for r in rs])
        cost = np.mean([r["cost_usd"] for r in rs])
        print(f"{task:<18} {model:<25} {K:>2} {strategy:<8} {prompt_style:<8} {mae:>5.1f} {w5:>4.0%} {w10:>4.0%} {w20:>4.0%} {lat:>4.1f}s ${cost:>.4f}")

    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--tasks", nargs="+", default=["reach-v3"])
    parser.add_argument("--K", nargs="+", type=int, default=[8])
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-4-6"])
    parser.add_argument("--strategies", nargs="+", default=["uniform"])
    parser.add_argument("--prompt-styles", nargs="+", default=["direct"], choices=["direct", "cot"])
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--annotate", action="store_true",
                        help="Overlay timestep index + progress on keyframe images (VTimeCoT-style)")
    parser.add_argument("--proprio", action="store_true",
                        help="Include numeric end-effector + goal/object positions as text in prompt")
    parser.add_argument("--call-delay", type=float, default=0.0,
                        help="Seconds to wait between API calls (helps with RPM limits)")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / args.data_dir
    output_dir = Path(__file__).parent / args.output_dir

    run_sweep(
        data_dir=data_dir,
        tasks=args.tasks,
        K_values=args.K,
        strategies=args.strategies,
        models=args.models,
        output_dir=output_dir,
        max_rollouts=args.max_rollouts,
        prompt_styles=args.prompt_styles,
        annotate=args.annotate,
        use_proprio=args.proprio,
        call_delay=args.call_delay,
    )


if __name__ == "__main__":
    main()
