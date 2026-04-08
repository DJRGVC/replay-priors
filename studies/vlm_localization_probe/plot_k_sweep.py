"""Plot K-sweep results: accuracy and cost vs K."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_path = Path(__file__).parent / "results" / "k_sweep_consolidated.json"
figures_dir = Path(__file__).parent / "figures"
figures_dir.mkdir(exist_ok=True)

with open(results_path) as f:
    data = json.load(f)

# Group by K
from collections import defaultdict
groups = defaultdict(list)
for r in data:
    groups[r["K"]].append(r)

K_vals = sorted(groups.keys())
mae_vals = [np.mean([r["abs_error"] for r in groups[k]]) for k in K_vals]
w10_vals = [100 * np.mean([r["within_10"] for r in groups[k]]) for k in K_vals]
w20_vals = [100 * np.mean([r["within_20"] for r in groups[k]]) for k in K_vals]
cost_vals = [np.mean([r["cost_usd"] for r in groups[k]]) for k in K_vals]
n_vals = [len(groups[k]) for k in K_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: MAE and window accuracy vs K
ax1.plot(K_vals, mae_vals, "o-", color="tab:red", label="MAE", linewidth=2, markersize=8)
ax1.set_xlabel("K (number of keyframes)")
ax1.set_ylabel("Mean Absolute Error (timesteps)", color="tab:red")
ax1.set_ylim(0, 80)
ax1.set_xticks(K_vals)
ax1.tick_params(axis="y", labelcolor="tab:red")

ax1b = ax1.twinx()
ax1b.plot(K_vals, w10_vals, "s--", color="tab:blue", label="±10 acc", markersize=7)
ax1b.plot(K_vals, w20_vals, "^--", color="tab:green", label="±20 acc", markersize=7)
ax1b.set_ylabel("Window Accuracy (%)", color="tab:blue")
ax1b.set_ylim(0, 60)
ax1b.tick_params(axis="y", labelcolor="tab:blue")

# Add n annotations
for i, k in enumerate(K_vals):
    ax1.annotate(f"n={n_vals[i]}", (k, mae_vals[i]), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8, color="gray")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
ax1.set_title("Localization Accuracy vs K\n(reach-v3, claude-sonnet-4-6, uniform)")

# Right: cost vs K
ax2.bar(range(len(K_vals)), [c * 1000 for c in cost_vals], tick_label=[str(k) for k in K_vals],
        color="tab:orange", alpha=0.8)
ax2.set_xlabel("K (number of keyframes)")
ax2.set_ylabel("Cost per call (millicents)")
ax2.set_title("API Cost vs K")

# Add cost labels
for i, c in enumerate(cost_vals):
    ax2.text(i, c * 1000 + 0.1, f"${c:.4f}", ha="center", fontsize=9)

plt.tight_layout()
fig.savefig(figures_dir / "k_sweep_reach_v3.png", dpi=150, bbox_inches="tight")
print(f"Saved to {figures_dir / 'k_sweep_reach_v3.png'}")
