"""Two-pass adaptive VLM probing: coarse localization then dense refinement.

Motivation: iter_002-010 showed VLMs have high MAE (42-95) but reasonable
top-20% overlap (+12% over uniform). This suggests they can identify the
right REGION but not the precise timestep. A coarse-to-fine strategy should
improve precision by giving the model denser temporal resolution around
its initial estimate.

Protocol:
  Pass 1 (coarse): K_coarse uniform keyframes → get rough failure estimate t_hat
  Pass 2 (refine): K_fine keyframes densely sampled around t_hat → refined prediction

Usage:
    python two_pass_probe.py --tasks reach-v3 --K-coarse 4 --K-fine 8 \
        --models gemini-3-flash-preview --max-rollouts 5 --annotate
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vlm_client import (
    TASK_DESCRIPTIONS,
    annotate_frame,
    extract_proprio_text,
    predict_failure,
    sample_keyframes,
    sample_keyframes_around,
)
from run_probe import estimate_cost, evaluate_prediction, load_rollouts


def run_two_pass(
    data_dir: Path,
    tasks: list[str],
    K_coarse: int,
    K_fine: int,
    window_fraction: float,
    models: list[str],
    output_dir: Path,
    max_rollouts: int | None = None,
    annotate: bool = False,
    use_proprio: bool = False,
    call_delay: float = 0.0,
):
    """Run two-pass adaptive probe and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_cost = 0.0

    for task_name in tasks:
        rollouts = load_rollouts(data_dir, task_name)
        if max_rollouts:
            rollouts = rollouts[:max_rollouts]
        task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)

        for model in models:
            config_key = f"{task_name}/{model}/two_pass_K{K_coarse}+{K_fine}_w{window_fraction}"
            print(f"\n{'='*60}")
            print(f"Config: {config_key}")
            print(f"{'='*60}")

            config_results = []
            call_count = 0

            for rollout in rollouts:
                if rollout["success"]:
                    continue

                rd = Path(rollout["_dir"])
                frames_dir = rd / "frames"
                gt = rollout["failure_timestep"]
                total_steps = rollout["num_steps"]

                # Load proprio if requested
                proprio = None
                if use_proprio:
                    proprio_path = rd / "proprio.npy"
                    if proprio_path.exists():
                        proprio = np.load(proprio_path)

                # ── Pass 1: coarse ──
                if call_delay > 0 and call_count > 0:
                    time.sleep(call_delay)
                call_count += 1

                try:
                    frames_c, indices_c = sample_keyframes(frames_dir, K=K_coarse, strategy="uniform")
                    proprio_labels_c = extract_proprio_text(task_name, proprio, indices_c) if proprio is not None else None

                    pred_c = predict_failure(
                        task_description=task_desc,
                        keyframes=frames_c,
                        keyframe_indices=indices_c,
                        total_timesteps=total_steps,
                        model=model,
                        prompt_style="direct",
                        annotate_frames=annotate,
                        proprio_labels=proprio_labels_c,
                    )
                except Exception as e:
                    print(f"  {rd.name} PASS1 ERROR: {e}")
                    pred_c = {"failure_timestep": None, "confidence": 0.0,
                              "_latency_s": 0, "_input_tokens": 0, "_output_tokens": 0, "_model": model}

                t_hat = pred_c.get("failure_timestep")
                cost_c = estimate_cost(model, pred_c.get("_input_tokens", 0), pred_c.get("_output_tokens", 0))

                # ── Pass 2: refine around t_hat ──
                if t_hat is not None:
                    if call_delay > 0:
                        time.sleep(call_delay)
                    call_count += 1

                    try:
                        frames_f, indices_f = sample_keyframes_around(
                            frames_dir, center=t_hat, K=K_fine, window_fraction=window_fraction
                        )
                        proprio_labels_f = extract_proprio_text(task_name, proprio, indices_f) if proprio is not None else None

                        pred_f = predict_failure(
                            task_description=task_desc,
                            keyframes=frames_f,
                            keyframe_indices=indices_f,
                            total_timesteps=total_steps,
                            model=model,
                            prompt_style="direct",
                            annotate_frames=annotate,
                            proprio_labels=proprio_labels_f,
                        )
                    except Exception as e:
                        print(f"  {rd.name} PASS2 ERROR: {e}")
                        pred_f = pred_c  # fall back to coarse prediction
                else:
                    pred_f = pred_c  # coarse failed, no refinement possible

                cost_f = estimate_cost(model, pred_f.get("_input_tokens", 0), pred_f.get("_output_tokens", 0))
                total_cost += cost_c + cost_f

                # Metrics for both passes
                metrics_c = evaluate_prediction(gt, t_hat, total_steps)
                final_t = pred_f.get("failure_timestep")
                metrics_f = evaluate_prediction(gt, final_t, total_steps)

                entry = {
                    "task": task_name,
                    "model": model,
                    "strategy": "two_pass",
                    "K_coarse": K_coarse,
                    "K_fine": K_fine,
                    "window_fraction": window_fraction,
                    "annotate": annotate,
                    "proprio": use_proprio,
                    "rollout": rd.name,
                    "gt_failure_t": gt,
                    "gt_failure_type": rollout.get("failure_type"),
                    # Coarse pass results
                    "coarse_pred_t": t_hat,
                    "coarse_abs_error": metrics_c["abs_error"],
                    "coarse_confidence": pred_c.get("confidence"),
                    "coarse_rationale": pred_c.get("rationale"),
                    "coarse_latency_s": pred_c.get("_latency_s", 0),
                    # Fine pass results
                    "fine_pred_t": final_t,
                    "fine_abs_error": metrics_f["abs_error"],
                    "fine_confidence": pred_f.get("confidence"),
                    "fine_rationale": pred_f.get("rationale"),
                    "fine_latency_s": pred_f.get("_latency_s", 0),
                    # Aggregate
                    "improvement": metrics_c["abs_error"] - metrics_f["abs_error"],
                    "within_5": metrics_f["within_5"],
                    "within_10": metrics_f["within_10"],
                    "within_20": metrics_f["within_20"],
                    "total_latency_s": pred_c.get("_latency_s", 0) + pred_f.get("_latency_s", 0),
                    "total_cost_usd": round(cost_c + cost_f, 6),
                }
                config_results.append(entry)
                all_results.append(entry)

                coarse_str = f"{t_hat:>4d}" if t_hat is not None else "   ?"
                fine_str = f"{final_t:>4d}" if final_t is not None else "   ?"
                imp = entry["improvement"]
                imp_str = f"+{imp}" if imp > 0 else str(imp)
                print(f"  {rd.name}: gt={gt:3d} coarse={coarse_str}(err={metrics_c['abs_error']:3d}) → fine={fine_str}(err={metrics_f['abs_error']:3d}) Δ={imp_str}")

            # Config summary
            if config_results:
                coarse_errs = [r["coarse_abs_error"] for r in config_results]
                fine_errs = [r["fine_abs_error"] for r in config_results]
                improvements = [r["improvement"] for r in config_results]
                w10 = np.mean([r["within_10"] for r in config_results])
                w20 = np.mean([r["within_20"] for r in config_results])
                lats = [r["total_latency_s"] for r in config_results]

                print(f"\n  SUMMARY ({len(config_results)} rollouts):")
                print(f"    Coarse MAE: {np.mean(coarse_errs):.1f} → Fine MAE: {np.mean(fine_errs):.1f}")
                print(f"    Mean improvement: {np.mean(improvements):.1f} (median: {np.median(improvements):.1f})")
                print(f"    Fine ±10 acc: {w10:.1%}  ±20 acc: {w20:.1%}")
                print(f"    Improved: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
                print(f"    Worsened: {sum(1 for i in improvements if i < 0)}/{len(improvements)}")
                print(f"    Mean total latency: {np.mean(lats):.1f}s (2 API calls)")

    # Save results
    results_path = output_dir / "two_pass_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total API cost: ${total_cost:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Two-pass adaptive VLM failure localization")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/two_pass")
    parser.add_argument("--tasks", nargs="+", default=["reach-v3"])
    parser.add_argument("--K-coarse", type=int, default=4, help="Keyframes for coarse pass")
    parser.add_argument("--K-fine", type=int, default=8, help="Keyframes for refinement pass")
    parser.add_argument("--window-fraction", type=float, default=0.3,
                        help="Fraction of episode for refinement window (default 0.3 = ±15%%)")
    parser.add_argument("--models", nargs="+", default=["gemini-3-flash-preview"])
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--proprio", action="store_true")
    parser.add_argument("--call-delay", type=float, default=0.0)
    args = parser.parse_args()

    data_dir = Path(__file__).parent / args.data_dir
    output_dir = Path(__file__).parent / args.output_dir

    run_two_pass(
        data_dir=data_dir,
        tasks=args.tasks,
        K_coarse=args.K_coarse,
        K_fine=args.K_fine,
        window_fraction=args.window_fraction,
        models=args.models,
        output_dir=output_dir,
        max_rollouts=args.max_rollouts,
        annotate=args.annotate,
        use_proprio=args.proprio,
        call_delay=args.call_delay,
    )


if __name__ == "__main__":
    main()
