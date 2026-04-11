#!/usr/bin/env python3
"""Consolidate all VLM probe results into a single database + paper-ready summary table.

Produces:
  results/consolidated_database.json  — every individual prediction (flat list)
  results/summary_table.json          — per-condition aggregate metrics
  figures/paper_summary_table.png     — publication-quality summary figure
"""

import json, glob, os, sys
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ── Step 1: Load all individual prediction results ──

def load_temporal_results():
    """Load all results.json files that contain per-prediction entries."""
    all_entries = []
    seen_files = set()

    # Walk all subdirs for results.json
    for pattern in ["results/*/results.json", "results/results.json",
                    "results/two_pass/two_pass_results.json"]:
        for fpath in sorted(glob.glob(os.path.join(os.path.dirname(__file__), pattern))):
            if fpath in seen_files:
                continue
            seen_files.add(fpath)
            try:
                data = json.load(open(fpath))
                if not isinstance(data, list):
                    continue
                # Tag with source directory
                subdir = os.path.basename(os.path.dirname(fpath))
                for entry in data:
                    entry["_source"] = subdir
                all_entries.extend(data)
            except Exception as e:
                print(f"  SKIP {fpath}: {e}", file=sys.stderr)

    return all_entries


def normalize_model(m):
    """Normalize model names for display."""
    # Order matters: check longer/more-specific strings first
    mapping = [
        ("gh:gpt-4o-mini", "GPT-4o-mini"),
        ("gh:gpt-4o", "GPT-4o"),
        ("gh:Llama-3.2-90B-Vision-Instruct", "Llama-3.2-90B"),
        ("gh:Llama-3.2-11B-Vision-Instruct", "Llama-3.2-11B"),
        ("gh:Phi-4-multimodal-instruct", "Phi-4"),
        ("gemini-2.0-flash-lite", "Gemini Flash-Lite"),
        ("gemini-2.5-flash-preview", "Gemini 2.5 Flash"),
        ("gemini-3-flash-preview", "Gemini 3 Flash"),
        ("claude-sonnet-4", "Claude Sonnet"),
    ]
    for k, v in mapping:
        if k in m:
            return v
    return m


def classify_condition(entry):
    """Classify each entry into a human-readable experimental condition."""
    model = normalize_model(entry.get("model", ""))
    task = entry.get("task", "reach-v3")
    K = entry.get("K", 8)
    annotate = entry.get("annotate", False)
    prompt_style = entry.get("prompt_style", "direct")
    strategy = entry.get("strategy", "uniform")

    parts = [model]
    if task != "reach-v3":
        parts.append(task.replace("-v3", ""))
    if K != 8:
        parts.append(f"K={K}")
    if prompt_style == "cot":
        parts.append("CoT")
    if annotate:
        parts.append("ann")
    else:
        parts.append("no-ann")
    if strategy == "random":
        parts.append("random-samp")

    return " / ".join(parts)


def compute_aggregates(entries):
    """Compute per-condition aggregate metrics."""
    groups = defaultdict(list)
    for e in entries:
        cond = classify_condition(e)
        groups[cond].append(e)

    summaries = []
    for cond, elist in sorted(groups.items()):
        errors = [e["abs_error"] for e in elist if "abs_error" in e]
        if not errors:
            continue

        preds = [e.get("pred_failure_t", None) for e in elist]
        preds = [p for p in preds if p is not None]

        n = len(errors)
        mae = np.mean(errors)
        median_ae = np.median(errors)
        std_ae = np.std(errors)
        within_10 = sum(1 for e in errors if e <= 10) / n
        within_20 = sum(1 for e in errors if e <= 20) / n

        # Prediction diversity: number of unique predictions
        unique_preds = len(set(preds)) if preds else 0

        # Fixation: fraction predicting the mode
        if preds:
            from collections import Counter
            mode_count = Counter(preds).most_common(1)[0][1]
            fixation = mode_count / len(preds)
        else:
            fixation = None

        # Extract metadata from first entry
        e0 = elist[0]

        summaries.append({
            "condition": cond,
            "model": normalize_model(e0.get("model", "")),
            "task": e0.get("task", "reach-v3"),
            "K": e0.get("K", 8),
            "prompt_style": e0.get("prompt_style", "direct"),
            "annotate": e0.get("annotate", False),
            "strategy": e0.get("strategy", "uniform"),
            "n": n,
            "mae": round(mae, 1),
            "median_ae": round(median_ae, 1),
            "std_ae": round(std_ae, 1),
            "within_10": round(within_10, 3),
            "within_20": round(within_20, 3),
            "unique_preds": unique_preds,
            "fixation": round(fixation, 2) if fixation else None,
            "mean_latency_s": round(np.mean([e.get("latency_s", 0) for e in elist]), 2),
            "mean_cost_usd": round(np.mean([e.get("cost_usd", 0) for e in elist]), 4),
        })

    return summaries


def make_summary_figure(summaries):
    """Create a publication-quality summary table figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to main reach-v3 comparisons (K=8, direct prompt) for the core table
    # Plus key interventions

    # Sort by MAE
    summaries_sorted = sorted(summaries, key=lambda s: s["mae"])

    # Split into: (a) reach-v3 core, (b) task generalization, (c) non-temporal
    reach_core = [s for s in summaries_sorted if s["task"] == "reach-v3" and s["n"] >= 5]
    task_gen = [s for s in summaries_sorted if s["task"] != "reach-v3" and s["n"] >= 5]

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, max(len(reach_core) * 0.35 + 2, 6) + max(len(task_gen) * 0.35 + 2, 4)),
                                    gridspec_kw={"height_ratios": [max(len(reach_core), 5), max(len(task_gen), 3)]})

    def plot_table(ax, data, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, len(data) + 0.5)
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Headers
        cols = ["Condition", "n", "MAE↓", "Med↓", "±10", "±20", "Fix%", "Lat(s)"]
        col_x = [0.0, 5.2, 5.8, 6.5, 7.2, 7.8, 8.4, 9.2]

        y = len(data)
        for c, x in zip(cols, col_x):
            ax.text(x, y + 0.3, c, fontsize=9, fontweight="bold", va="bottom",
                   fontfamily="monospace")
        ax.axhline(y=y + 0.1, xmin=0, xmax=1, color="black", linewidth=1.5)

        # Color scale for MAE
        mae_vals = [s["mae"] for s in data] if data else [0]
        mae_min, mae_max = min(mae_vals), max(mae_vals)

        for i, s in enumerate(data):
            y = len(data) - 1 - i
            # Color row by MAE (green=low, red=high)
            if mae_max > mae_min:
                frac = (s["mae"] - mae_min) / (mae_max - mae_min)
            else:
                frac = 0.5
            bg_color = plt.cm.RdYlGn_r(frac * 0.6 + 0.2)
            ax.axhspan(y - 0.4, y + 0.4, alpha=0.15, color=bg_color)

            vals = [
                s["condition"][:38],
                str(s["n"]),
                f'{s["mae"]:.1f}',
                f'{s["median_ae"]:.1f}',
                f'{s["within_10"]:.0%}',
                f'{s["within_20"]:.0%}',
                f'{s["fixation"]:.0%}' if s["fixation"] else "—",
                f'{s["mean_latency_s"]:.1f}',
            ]
            for v, x in zip(vals, col_x):
                ax.text(x, y, v, fontsize=8, va="center", fontfamily="monospace")

    plot_table(ax1, reach_core, f"Reach-v3 — All Conditions ({len(reach_core)} configs)")
    plot_table(ax2, task_gen, f"Task Generalization — Push/Pick-Place ({len(task_gen)} configs)")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "paper_summary_table.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")
    return out


def make_approach_comparison_figure(summaries):
    """Bar chart comparing the 14 tested approaches by best MAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Define the 14 approaches with their best results
    approaches = [
        ("Baseline (Sonnet)", 41.9, "temporal", "mixed"),
        ("K=4 GPT-4o", 49.0, "temporal", "fail"),
        ("K=16 GPT-4o", 52.0, "temporal", "fail"),
        ("K=32 Sonnet", 52.0, "temporal", "fail"),
        ("Annotation (GPT-4o)", 52.7, "temporal", "mixed"),
        ("CoT (GPT-4o-mini)", 53.2, "temporal", "mixed"),
        ("CoT+Ann factorial", 52.2, "temporal", "fail"),
        ("Random sampling", 54.0, "temporal", "fail"),
        ("Two-pass adaptive", 55.0, "temporal", "fail"),
        ("Grid tiling", 53.5, "temporal", "fail"),
        ("Ensemble (BAEP)", 46.9, "ensemble", "fail"),
        ("Confidence gating", 999, "meta", "fail"),  # infinite — never use VLM
        ("Contrastive ranking", 999, "comparison", "fail"),  # 100% primacy bias
        ("Category-diversity", 0, "non-temporal", "viable"),  # different metric
    ]

    # Filter to temporal approaches with actual MAE
    temporal = [(name, mae, cat, status) for name, mae, cat, status in approaches if mae < 900]
    temporal.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"mixed": "#FFB74D", "fail": "#E57373", "viable": "#81C784"}

    bars = ax.barh(range(len(temporal)), [t[1] for t in temporal],
                   color=[colors[t[3]] for t in temporal], edgecolor="white", height=0.6)
    ax.set_yticks(range(len(temporal)))
    ax.set_yticklabels([t[0] for t in temporal], fontsize=10)
    ax.set_xlabel("Best MAE (lower = better)", fontsize=11)
    ax.set_title("VLM Failure Localization: 12 Temporal Approaches Compared\n(uniform baseline MAE ≈ 50 for 150-step episodes)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, label="Random guess ≈ 50")
    ax.invert_yaxis()

    # Annotate
    for i, (name, mae, cat, status) in enumerate(temporal):
        ax.text(mae + 1, i, f"{mae:.1f}", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["mixed"], label="Mixed (some conditions help)"),
        Patch(facecolor=colors["fail"], label="Failed (≤ uniform)"),
        Patch(facecolor=colors["viable"], label="Viable (non-temporal)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add notes for excluded approaches
    ax.text(0.98, 0.02,
            "Not shown: Confidence gating (optimal = never use VLM)\n"
            "Contrastive ranking (100% primacy bias)\n"
            "Category-diversity (viable at N≥50, different metric)",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "approach_comparison_bar.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    print("Loading all temporal prediction results...")
    entries = load_temporal_results()
    print(f"  Total entries: {len(entries)}")

    # Deduplicate by (model, task, K, prompt_style, annotate, strategy, rollout)
    seen = set()
    unique = []
    dupes = 0
    for e in entries:
        key = (e.get("model"), e.get("task"), e.get("K"),
               e.get("prompt_style"), e.get("annotate"),
               e.get("strategy"), e.get("rollout"))
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        unique.append(e)
    print(f"  After dedup: {len(unique)} (removed {dupes} duplicates)")

    # Save consolidated database
    out_db = os.path.join(RESULTS_DIR, "consolidated_database.json")
    with open(out_db, "w") as f:
        json.dump(unique, f, indent=2)
    print(f"  Saved {out_db}")

    # Compute aggregates
    print("\nComputing per-condition aggregates...")
    summaries = compute_aggregates(unique)
    print(f"  {len(summaries)} unique conditions")

    out_sum = os.path.join(RESULTS_DIR, "summary_table.json")
    with open(out_sum, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved {out_sum}")

    # Print table
    print(f"\n{'Condition':<40} {'n':>3} {'MAE':>6} {'Med':>6} {'±10':>5} {'±20':>5} {'Fix':>5}")
    print("-" * 80)
    for s in sorted(summaries, key=lambda x: x["mae"]):
        print(f"{s['condition'][:39]:<40} {s['n']:>3} {s['mae']:>6.1f} {s['median_ae']:>6.1f} "
              f"{s['within_10']:>5.0%} {s['within_20']:>5.0%} "
              f"{s['fixation']:>5.0%}" if s['fixation'] else
              f"{s['condition'][:39]:<40} {s['n']:>3} {s['mae']:>6.1f} {s['median_ae']:>6.1f} "
              f"{s['within_10']:>5.0%} {s['within_20']:>5.0%}   —")

    # Generate figures
    print("\nGenerating figures...")
    make_summary_figure(summaries)
    make_approach_comparison_figure(summaries)

    print("\nDone!")
