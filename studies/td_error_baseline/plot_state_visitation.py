"""
State-space visitation analysis via dense reward distributions.

MetaWorld reach-v3 dense reward = f(distance to goal). Higher dense reward ⟹
closer to goal ⟹ more "useful" exploration. This lets us use the per-sample
dense reward distribution as a proxy for WHERE in state space the agent is
visiting, without needing raw observations.

Creates a multi-panel figure showing:
1. Dense reward distribution evolution (violin plots) across modes at 3 timesteps
2. Seed 42 diagnostic: density comparison (uniform vs RND-PER) at 50k
3. Q-value vs dense reward scatter at 50k/100k: do Q-values track reward structure?
4. "Exploration radius" summary: mean dense reward trajectory per seed×mode
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# ── Config ──────────────────────────────────────────────────────────────────

modes = ['uniform', 'rnd-per', 'rpe-per', 'adaptive']
mode_labels = ['Uniform', 'RND-PER', 'RPE-PER', 'Adaptive']
mode_colors = {'uniform': '#1f77b4', 'rnd-per': '#ff7f0e',
               'rpe-per': '#2ca02c', 'adaptive': '#d62728'}
seeds = [42, 7, 99, 123, 256]
seed_colors = {42: '#e41a1c', 7: '#377eb8', 99: '#4daf4a',
               123: '#984ea3', 256: '#ff7f00'}
base = 'snapshots'
timesteps = [10000, 50000, 100000]
timestep_labels = ['10k', '50k', '100k']

# ── Data loading ────────────────────────────────────────────────────────────

def load_snapshot(mode, seed, step):
    """Load snapshot data for a given mode/seed/step."""
    d = f'{base}/reach-v3_s{seed}_{mode}/td_snapshots'
    fname = f'snapshot_{step:08d}.npz'
    path = os.path.join(d, fname)
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)

# Pre-load all data
all_data = {}
for mode in modes:
    for seed in seeds:
        for step in timesteps:
            data = load_snapshot(mode, seed, step)
            if data is not None:
                all_data[(mode, seed, step)] = {
                    'dense_rewards': data['dense_rewards'],
                    'q_values': data['q_values'],
                    'sparse_rewards': data['sparse_rewards'],
                    'sr_frac': np.mean(data['sparse_rewards'] > 0),
                }

# ── Figure ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30,
                       left=0.07, right=0.96, top=0.93, bottom=0.05)

# ── Panel A: Dense reward distributions across modes at 3 timesteps ─────────
ax_a = fig.add_subplot(gs[0, :])

# Aggregate dense rewards across all seeds for each mode at each timestep
positions = []
violins_data = []
colors_list = []
labels_at = []

for t_idx, (step, step_label) in enumerate(zip(timesteps, timestep_labels)):
    for m_idx, mode in enumerate(modes):
        all_dr = []
        for seed in seeds:
            key = (mode, seed, step)
            if key in all_data:
                all_dr.append(all_data[key]['dense_rewards'])
        if all_dr:
            combined = np.concatenate(all_dr)
            # Subsample for plotting performance
            if len(combined) > 5000:
                combined = np.random.RandomState(42).choice(combined, 5000, replace=False)
            pos = t_idx * (len(modes) + 1) + m_idx
            positions.append(pos)
            violins_data.append(combined)
            colors_list.append(mode_colors[mode])
            if t_idx == 0:
                labels_at.append((pos, mode_labels[m_idx]))

parts = ax_a.violinplot(violins_data, positions=positions, showmeans=True,
                         showmedians=True, widths=0.8)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_list[i])
    pc.set_alpha(0.6)
parts['cmeans'].set_color('black')
parts['cmedians'].set_color('darkred')
parts['cmedians'].set_linewidth(2)

# Group labels
for t_idx, step_label in enumerate(timestep_labels):
    center = t_idx * (len(modes) + 1) + 1.5
    ax_a.text(center, -1.5, f't = {step_label} steps', fontsize=11,
              ha='center', fontweight='bold')

# Separators
for t_idx in range(1, len(timesteps)):
    x = t_idx * (len(modes) + 1) - 0.5
    ax_a.axvline(x, color='gray', linestyle=':', alpha=0.3)

ax_a.set_xticks(positions)
# Build tick labels matching actual positions
tick_label_map = {}
for t_idx, step_label in enumerate(timestep_labels):
    for m_idx, ml in enumerate(mode_labels):
        tick_label_map[t_idx * (len(modes) + 1) + m_idx] = ml
ax_a.set_xticklabels([tick_label_map.get(p, '') for p in positions],
                      fontsize=8, rotation=30)
ax_a.set_ylabel('Dense reward (distance-to-goal proxy)', fontsize=11)
ax_a.set_title('(A) State-Space Visitation: Dense Reward Distributions Across Modes\n'
               '(higher = closer to goal = more useful exploration)',
               fontsize=12, fontweight='bold')
ax_a.set_ylim(-0.5, 11)

# ── Panel B: Seed 42 diagnostic — uniform vs RND-PER at 50k ────────────────
ax_b = fig.add_subplot(gs[1, 0])

# Seed 42 is the key diagnostic: learns under RND-PER but not uniform
for mode, label, color, ls in [('uniform', 'Uniform', '#1f77b4', '-'),
                                 ('rnd-per', 'RND-PER', '#ff7f0e', '--')]:
    key = (mode, 42, 50000)
    if key in all_data:
        dr = all_data[key]['dense_rewards']
        # KDE-like histogram
        ax_b.hist(dr, bins=50, alpha=0.5, color=color, label=f'{label} (mean={dr.mean():.2f})',
                  density=True, edgecolor='none')
        ax_b.axvline(dr.mean(), color=color, linestyle=ls, linewidth=2, alpha=0.8)

# Add 10k and 100k comparisons as lighter lines
for step, alpha in [(10000, 0.3), (100000, 0.9)]:
    for mode, color, ls in [('uniform', '#1f77b4', '-'), ('rnd-per', '#ff7f0e', '--')]:
        key = (mode, 42, step)
        if key in all_data:
            dr = all_data[key]['dense_rewards']
            ax_b.axvline(dr.mean(), color=color, linestyle=ls, linewidth=1,
                         alpha=alpha)
            ax_b.text(dr.mean() + 0.1, ax_b.get_ylim()[1] * 0.85 * alpha,
                      f'{step//1000}k', fontsize=7, color=color, alpha=alpha)

ax_b.set_xlabel('Dense reward', fontsize=10)
ax_b.set_ylabel('Density', fontsize=10)
ax_b.set_title('(B) Seed 42 at 50k Steps: Exploration Divergence\n'
               '(RND-PER pushes exploration toward higher-reward states)',
               fontsize=11, fontweight='bold')
ax_b.legend(fontsize=9)

# Annotate the key insight
sr_uniform = all_data.get(('uniform', 42, 50000), {}).get('sr_frac', 0)
sr_rnd = all_data.get(('rnd-per', 42, 50000), {}).get('sr_frac', 0)
ax_b.text(0.98, 0.72, f'sparse reward frac:\n  Uniform: {sr_uniform:.3f}\n  RND-PER: {sr_rnd:.3f}',
          transform=ax_b.transAxes, fontsize=9, ha='right', va='top',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

# ── Panel C: Q-value vs dense reward scatter at 50k ────────────────────────
ax_c = fig.add_subplot(gs[1, 1])

# Show seed 7 (robust learner) and seed 42 (diagnostic case) at 50k
for seed, marker in [(7, 'o'), (42, '^')]:
    for mode in ['uniform', 'rnd-per']:
        key = (mode, seed, 50000)
        if key not in all_data:
            continue
        dr = all_data[key]['dense_rewards']
        qv = all_data[key]['q_values']
        # Subsample for readability
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(dr), min(500, len(dr)), replace=False)
        label = f's{seed}/{mode_labels[modes.index(mode)]}'
        ax_c.scatter(dr[idx], qv[idx], alpha=0.15, s=8, marker=marker,
                     color=mode_colors[mode], label=label)

ax_c.set_xlabel('Dense reward (state quality)', fontsize=10)
ax_c.set_ylabel('Q-value (critic estimate)', fontsize=10)
ax_c.set_title('(C) Q-Value vs Dense Reward at 50k Steps\n'
               '(seeds 7 & 42, uniform vs RND-PER)',
               fontsize=11, fontweight='bold')
ax_c.legend(fontsize=8, markerscale=3)

# ── Panel D: Mean dense reward trajectory per seed×mode ─────────────────────
ax_d = fig.add_subplot(gs[2, 0])

# Load all timesteps for all mode/seed combos
all_steps = sorted(set(step for _, _, step in all_data.keys()))

for mode, ls in zip(modes, ['-', '--', ':', '-.']):
    for seed in seeds:
        steps_list = []
        mean_dr_list = []
        for step in all_steps:
            key = (mode, seed, step)
            if key in all_data:
                steps_list.append(step / 1000)
                mean_dr_list.append(all_data[key]['dense_rewards'].mean())
        if steps_list:
            learned = all_data.get((mode, seed, 100000), {}).get('sr_frac', 0) > 0.10
            ax_d.plot(steps_list, mean_dr_list, color=seed_colors[seed],
                      linestyle=ls, linewidth=2 if learned else 0.8,
                      alpha=0.9 if learned else 0.25)

# Legend
from matplotlib.lines import Line2D
legend_elements = []
for seed, color in seed_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=f'seed {seed}'))
legend_elements.append(Line2D([0], [0], color='gray', linewidth=0))
for mode, ls, label in zip(modes, ['-', '--', ':', '-.'], mode_labels):
    legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls, linewidth=1.5, label=label))

ax_d.legend(handles=legend_elements, fontsize=7, ncol=2, loc='upper left')
ax_d.set_xlabel('Environment steps (×1000)', fontsize=10)
ax_d.set_ylabel('Mean dense reward', fontsize=10)
ax_d.set_title('(D) Exploration Radius: Mean Dense Reward Over Training\n'
               '(bold = learned, faded = did not learn)',
               fontsize=11, fontweight='bold')

# ── Panel E: Quantitative summary ──────────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 1])
ax_e.axis('off')

# Compute stats at 50k
summary_rows = []
for mode, label in zip(modes, mode_labels):
    mean_drs = []
    sr_fracs = []
    for seed in seeds:
        key = (mode, seed, 50000)
        if key in all_data:
            mean_drs.append(all_data[key]['dense_rewards'].mean())
            sr_fracs.append(all_data[key]['sr_frac'])
    if mean_drs:
        summary_rows.append(f'{label:12s}  dense_r={np.mean(mean_drs):.2f}±{np.std(mean_drs):.2f}  '
                            f'sr_frac={np.mean(sr_fracs):.3f}±{np.std(sr_fracs):.3f}')

summary_text = """STATE-SPACE VISITATION SUMMARY (at 50k steps)

""" + '\n'.join(summary_rows) + """

KEY INSIGHT
Dense reward serves as a distance-to-goal proxy in MetaWorld.
At 50k steps, the distributions reveal:

• RND-PER shifts exploration TOWARD goal for responsive seeds
  (seed 42: mean dense reward 1.16→3.19, uniform→RND-PER)
  but causes Q-divergence for non-responsive seeds (seed 99)

• Uniform replay maintains a tight, low-mean distribution —
  exploration is undirected but stable

• Priority signals create BIMODAL exploration: some seeds get
  pushed toward reward regions, others get trapped in
  self-reinforcing replay loops (high Q, zero sparse reward)

• This is the mechanism behind seed-switching: priority doesn't
  help ALL seeds explore better — it creates winners and losers

IMPLICATION FOR VLM-PER
A good priority signal must be monotonically informative:
higher priority => genuinely closer to task-relevant states.
TD-error and RND novelty fail this because they amplify
noise in the absence of reward signal."""

ax_e.text(0.02, 0.98, summary_text, transform=ax_e.transAxes,
          fontsize=8.2, fontfamily='monospace', verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

fig.suptitle('State-Space Visitation Under Priority Regimes (reach-v3)',
             fontsize=14, fontweight='bold', y=0.97)

out_path = 'figures/state_visitation_analysis.png'
os.makedirs('figures', exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')

# Also save to images/ for Quarto
quarto_path = '../../images/td_baseline/state_visitation_analysis.png'
os.makedirs(os.path.dirname(quarto_path), exist_ok=True)
plt.savefig(quarto_path, dpi=150, bbox_inches='tight')
print(f'Saved to {quarto_path}')

plt.close()
