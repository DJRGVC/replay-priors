"""Study synthesis figure — landscape of all VLM-PER approaches tested.

Creates a comprehensive visualization showing:
1. All approaches tested, grouped by category
2. Key outcome metric for each
3. Clear pass/fail annotation
4. The 'graveyard' of temporal approaches vs the viable non-temporal path
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Define all approaches tested, grouped by category
approaches = [
    # (name, category, metric_label, metric_value, outcome, note)
    # Temporal approaches — all failed
    ("Direct prediction\n(K=8, Sonnet)", "Temporal", "MAE", 41.9, "fail", "Best MAE but\n±10 = 20%"),
    ("K sweep\n(K=4→32)", "Temporal", "MAE range", 42, "fail", "Flat (Sonnet)\nor fixation (K=4)"),
    ("CoT prompting", "Temporal", "ΔMAE", -7.3, "fail", "Hurts weak,\nsubstitutes ann."),
    ("Frame annotation", "Temporal", "ΔMAE", -30, "mixed", "Architecture-\ndependent"),
    ("Two-pass adaptive", "Temporal", "MAE", 47.5, "fail", "Coarse pass\ntoo noisy"),
    ("Random sampling", "Temporal", "ΔMAE", 0, "fail", "Breaks clusters,\nno accuracy gain"),

    # Ensemble/meta — all failed
    ("5-model ensemble\n(BAEP)", "Ensemble", "MAE", 51.2, "fail", "Weak models\ndilute signal"),
    ("2-model selective\npair", "Ensemble", "MAE", 46.9, "mixed", "−6.4% vs best\nindividual"),
    ("Confidence gating\n(agreement)", "Ensemble", "r(agree,err)", 0.53, "fail", "Agreement is\nANTI-signal"),

    # Relative/ranking — failed
    ("Contrastive Episode\nRanking (CER)", "Ranking", "Accuracy", 63.6, "fail", "100% primacy\nbias (11/11 = A)"),

    # Non-temporal — partially viable
    ("Failure mode\ndescriptions", "Non-temporal", "η²", 0.34, "pass", "6/6 cats, 100%\nunique, η²=0.34-0.99"),
    ("TF-IDF clustering", "Non-temporal", "Silhouette", 0.12, "fail", "Template syntax\nkills embedding"),
    ("Category-diversity\nreplay (n=20)", "Non-temporal", "Δcoverage", 2.8, "fail", "≈ uniform at\nsmall n"),
    ("Category-diversity\nreplay (N≥50 sim)", "Non-temporal", "Δcoverage", 5.0, "pass", "Viable at\nrealistic N"),
]

fig, ax = plt.subplots(figsize=(16, 9))

# Category positions and colors
cat_order = ["Temporal", "Ensemble", "Ranking", "Non-temporal"]
cat_colors = {
    "Temporal": "#e74c3c",
    "Ensemble": "#e67e22",
    "Ranking": "#9b59b6",
    "Non-temporal": "#27ae60"
}
cat_y_ranges = {
    "Temporal": (6.5, 11.5),
    "Ensemble": (3.5, 5.5),
    "Ranking": (2.5, 3.5),
    "Non-temporal": (0, 2.5),
}

outcome_markers = {"fail": "x", "mixed": "D", "pass": "o"}
outcome_colors = {"fail": "#c0392b", "mixed": "#f39c12", "pass": "#27ae60"}
outcome_sizes = {"fail": 120, "mixed": 100, "pass": 120}

# Group by category
cat_items = {}
for a in approaches:
    cat = a[1]
    if cat not in cat_items:
        cat_items[cat] = []
    cat_items[cat].append(a)

# Layout
y_pos = 0
positions = []
for cat in reversed(cat_order):
    items = cat_items[cat]
    for i, item in enumerate(items):
        positions.append((item, y_pos))
        y_pos += 1
    y_pos += 0.5  # gap between categories

# Draw
for item, y in positions:
    name, cat, metric_label, metric_value, outcome, note = item
    color = cat_colors[cat]

    # Background bar
    bar_color = outcome_colors[outcome]
    alpha = 0.15 if outcome == "fail" else (0.25 if outcome == "mixed" else 0.3)
    ax.barh(y, 1, height=0.7, color=bar_color, alpha=alpha, left=0)

    # Approach name (left)
    ax.text(-0.05, y, name, ha='right', va='center', fontsize=9,
            fontfamily='sans-serif', fontweight='bold')

    # Outcome marker
    ax.scatter(0.5, y, marker=outcome_markers[outcome],
               s=outcome_sizes[outcome], c=outcome_colors[outcome],
               zorder=5, linewidths=2)

    # Note (right of bar)
    ax.text(1.1, y, note, ha='left', va='center', fontsize=7.5,
            fontfamily='sans-serif', color='#555')

# Category labels
cat_boundaries = {}
for item, y in positions:
    cat = item[1]
    if cat not in cat_boundaries:
        cat_boundaries[cat] = [y, y]
    cat_boundaries[cat] = [min(cat_boundaries[cat][0], y), max(cat_boundaries[cat][1], y)]

for cat, (y_min, y_max) in cat_boundaries.items():
    mid = (y_min + y_max) / 2
    color = cat_colors[cat]
    # Category bracket
    ax.plot([-0.85, -0.85], [y_min - 0.3, y_max + 0.3], color=color, linewidth=3)
    ax.text(-0.9, mid, cat.upper(), ha='right', va='center', fontsize=10,
            fontweight='bold', color=color, rotation=90)

# Legend
legend_elements = [
    plt.scatter([], [], marker='x', s=100, c='#c0392b', linewidths=2, label='Failed (no signal above uniform)'),
    plt.scatter([], [], marker='D', s=80, c='#f39c12', linewidths=1, label='Mixed (marginal / conditional)'),
    plt.scatter([], [], marker='o', s=100, c='#27ae60', linewidths=1, label='Viable (positive signal)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

ax.set_xlim(-1.2, 2.0)
ax.set_ylim(-0.8, y_pos + 0.3)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_title("VLM-PER Approach Landscape: 14 approaches tested, 10 failed, 2 mixed, 2 viable",
             fontsize=13, fontweight='bold', pad=15)

subtitle = ("44 iterations · 9 models · 3 tasks · $0.80 total cost\n"
            "Core finding: VLMs have scene understanding (what) but not temporal precision (when)")
ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom',
        fontsize=9, color='#666', style='italic')

plt.tight_layout()
plt.savefig("../../images/vlm_probe/study_synthesis_landscape.png", dpi=150, bbox_inches='tight',
            facecolor='white')
print("Saved study_synthesis_landscape.png")
plt.close()
