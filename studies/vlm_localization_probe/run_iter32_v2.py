#!/usr/bin/env python
"""Iter 32: pick-place-v3 unannotated probes - inline execution."""
import json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from vlm_client import predict_failure, sample_keyframes, TASK_DESCRIPTIONS, annotate_frame

def run_probe(model, task, K, annotate, outdir, max_rollouts=10, delay=10):
    """Run a single probe configuration."""
    data_dir = Path("data")
    task_dir = data_dir / task

    results = []
    for i in range(max_rollouts):
        rd = task_dir / f"rollout_{i:03d}"
        meta_path = rd / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        gt = meta["failure_timestep"]

        # Sample keyframes
        frames_dir = rd / "frames"
        total = meta["num_steps"]
        indices = list(range(0, total, max(1, total // K)))[:K]

        from PIL import Image
        keyframes = []
        for idx in indices:
            img_path = frames_dir / f"frame_{idx:04d}.png"
            if img_path.exists():
                keyframes.append(Image.open(img_path))
            else:
                keyframes.append(Image.new("RGB", (224, 224)))

        if annotate:
            keyframes = [annotate_frame(img, idx, total) for img, idx in zip(keyframes, indices)]

        try:
            pred_info = predict_failure(
                task_description=TASK_DESCRIPTIONS[task],
                keyframes=keyframes,
                keyframe_indices=indices,
                total_timesteps=total,
                model=model,
                prompt_style="direct",
                annotate_frames=False,  # already handled above
            )
            pred = pred_info.get("predicted_timestep")
            err = abs(pred - gt) if pred is not None else None
            lat = pred_info.get("_latency_s", 0)
            print(f"  rollout_{i:03d}: gt={gt:3d} pred={pred:4d} err={err:3d} lat={lat:.1f}s", flush=True)
            results.append({"gt": gt, "pred": pred, "err": err, "lat": lat})
        except Exception as e:
            print(f"  rollout_{i:03d}: FAILED - {e}", flush=True)
            results.append({"gt": gt, "pred": None, "err": None, "lat": 0})

        if i < max_rollouts - 1:
            time.sleep(delay)

    valid = [r for r in results if r["err"] is not None]
    if not valid:
        print("  NO VALID RESULTS")
        return

    mae = sum(r["err"] for r in valid) / len(valid)
    w5 = sum(1 for r in valid if r["err"] <= 5) / len(valid) * 100
    w10 = sum(1 for r in valid if r["err"] <= 10) / len(valid) * 100
    w20 = sum(1 for r in valid if r["err"] <= 20) / len(valid) * 100
    preds = [r["pred"] for r in valid]

    print(f"\n  MAE={mae:.1f}, ±5={w5:.0f}%, ±10={w10:.0f}%, ±20={w20:.0f}%")
    print(f"  Predictions: {preds}")
    print(f"  Valid: {len(valid)}/{len(results)}")

    # Save results
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"results": results, "mae": mae, "w5": w5, "w10": w10, "w20": w20, "preds": preds}, f, indent=2)

    return {"mae": mae, "w5": w5, "w10": w10, "w20": w20, "preds": preds, "n_valid": len(valid)}

if __name__ == "__main__":
    print("=== GPT-4o UNANNOTATED pick-place-v3 ===", flush=True)
    r1 = run_probe("gh:gpt-4o", "pick-place-v3", 8, False, "results/pick_place_v3_gpt4o_unannotated_iter32", delay=10)

    print("\n=== GPT-4o-mini UNANNOTATED pick-place-v3 ===", flush=True)
    r2 = run_probe("gh:gpt-4o-mini", "pick-place-v3", 8, False, "results/pick_place_v3_gpt4o_mini_unannotated_iter32", delay=10)

    print("\nDONE", flush=True)
