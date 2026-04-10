"""
Seed-switching analysis: How priority signals redirect exploration.

Key finding: Different priority regimes don't just change HOW MANY seeds learn,
they change WHICH seeds learn — implying priority signals reshape the exploration
landscape rather than uniformly helping or hurting.

Creates a multi-panel figure:
1. Heatmap: seed × mode outcome matrix
2. Per-seed learning curves across modes (success fraction over time)
3. Q-value trajectories revealing pathological divergence patterns
4. Reward-discovery timing: when does each seed first find sparse reward?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Data extraction ──────────────────────────────────────────────────────────

modes = ['uniform', 'rnd-per', 'rpe-per', 'adaptive']
mode_labels = ['Uniform', 'RND-PER', 'RPE-PER', 'Adaptive']
seeds = [42, 7, 99, 123, 256]
base = 'snapshots'

# Extract trajectories: {mode: {seed: [(step, sr, q_mean), ...]}}
trajectories = {}
for mode in modes:
    trajectories[mode] = {}
    for s in seeds:
        d = f'{base}/reach-v3_s{s}_{mode}/td_snapshots'
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d) if f.endswith('.npz')])
        traj = []
        for f in files:
            data = np.load(os.path.join(d, f), allow_pickle=True)
            step = data['step'].item()
            sr = np.mean(data['sparse_rewards'] > 0)
            q = data['q_mean'].item()
            traj.append((step, sr, q))
        trajectories[mode][s] = traj

# ── Derived quantities ───────────────────────────────────────────────────────

# Final success fraction and "learned" threshold
LEARN_THRESHOLD = 0.10  # >10% of buffer has successful transitions

def final_sr(mode, seed):
    """Get final success rate for a mode/seed combo."""
    traj = trajectories.get(mode, {}).get(seed, [])
    return traj[-1][1] if traj else 0.0

def first_reward_step(mode, seed, threshold=0.005):
    """Step at which sparse reward fraction first exceeds threshold."""
    traj = trajectories.get(mode, {}).get(seed, [])
    for step, sr, q in traj:
        if sr > threshold:
            return step
    return None  # never discovered

def final_q(mode, seed):
    """Get final Q-value for a mode/seed combo."""
    traj = trajectories.get(mode, {}).get(seed, [])
    return traj[-1][2] if traj else 0.0

# Build outcome matrix
outcome_matrix = np.zeros((len(seeds), len(modes)))
for j, mode in enumerate(modes):
    for i, seed in enumerate(seeds):
        outcome_matrix[i, j] = final_sr(mode, seed)

# ── Figure ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                       left=0.08, right=0.95, top=0.93, bottom=0.06)

# Color scheme for seeds
seed_colors = {42: '#e41a1c', 7: '#377eb8', 99: '#4daf4a',
               123: '#984ea3', 256: '#ff7f00'}

# ── Panel A: Outcome heatmap ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

# Custom colormap: red = failed, white = marginal, blue = learned
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('learn',
    [(0, '#d73027'), (0.1, '#fee08b'), (0.3, '#d9ef8b'), (1.0, '#1a9850')])

im = ax1.imshow(outcome_matrix, cmap=cmap, vmin=0, vmax=0.5, aspect='auto')
ax1.set_xticks(range(len(modes)))
ax1.set_xticklabels(mode_labels, fontsize=10)
ax1.set_yticks(range(len(seeds)))
ax1.set_yticklabels([f'seed {s}' for s in seeds], fontsize=10)
ax1.set_title('(A) Seed × Mode Outcome Matrix\n(buffer success fraction at final step)',
              fontsize=11, fontweight='bold')

# Annotate cells
for i in range(len(seeds)):
    for j in range(len(modes)):
        val = outcome_matrix[i, j]
        color = 'white' if val > 0.25 else 'black'
        label = f'{val:.2f}'
        if val > LEARN_THRESHOLD:
            label += ' ✓'
        ax1.text(j, i, label, ha='center', va='center', fontsize=9,
                 color=color, fontweight='bold' if val > LEARN_THRESHOLD else 'normal')

plt.colorbar(im, ax=ax1, label='Success fraction', shrink=0.8)

# ── Panel B: Seed-switching Sankey-style summary ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

# Show which seeds switch on/off relative to uniform
uniform_learned = {s for s in seeds if final_sr('uniform', s) > LEARN_THRESHOLD}
for j, (mode, label) in enumerate(zip(modes[1:], mode_labels[1:])):
    mode_learned = {s for s in seeds if final_sr(mode, s) > LEARN_THRESHOLD}
    switched_on = mode_learned - uniform_learned
    switched_off = uniform_learned - mode_learned
    stayed_on = mode_learned & uniform_learned
    stayed_off = (set(seeds) - uniform_learned) - mode_learned

    y = 2 - j
    ax2.text(-0.1, y, label, fontsize=11, fontweight='bold', ha='right', va='center')

    x = 0
    for s in sorted(seeds):
        if s in stayed_on:
            marker, color, alpha = 'o', seed_colors[s], 1.0
        elif s in stayed_off:
            marker, color, alpha = 'x', seed_colors[s], 0.3
        elif s in switched_on:
            marker, color, alpha = '^', seed_colors[s], 1.0
        else:  # switched off
            marker, color, alpha = 'v', seed_colors[s], 1.0
        ax2.scatter(x, y, marker=marker, c=color, s=150, alpha=alpha,
                    edgecolors='black' if s in switched_on or s in switched_off else 'none',
                    linewidth=1.5, zorder=3)
        ax2.text(x, y - 0.3, str(s), fontsize=7, ha='center', va='top', color=color)
        x += 1

ax2.set_xlim(-0.5, 5)
ax2.set_ylim(-1, 3.5)
ax2.set_title('(B) Seed Switching Relative to Uniform\n'
              '(○ stayed on, △ switched on, ▽ switched off, × stayed off)',
              fontsize=11, fontweight='bold')
ax2.axis('off')

# Legend for uniform baseline
ax2.text(2, 3.2, f'Uniform baseline: seeds {sorted(uniform_learned)} learn',
         fontsize=9, ha='center', style='italic')

# ── Panel C: Learning curves (success fraction) ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

for mode, ls in zip(modes, ['-', '--', ':', '-.']):
    for seed in seeds:
        traj = trajectories.get(mode, {}).get(seed, [])
        if not traj:
            continue
        steps = [t[0] / 1000 for t in traj]
        srs = [t[1] for t in traj]
        alpha = 0.9 if srs[-1] > LEARN_THRESHOLD else 0.25
        ax3.plot(steps, srs, color=seed_colors[seed], linestyle=ls,
                 alpha=alpha, linewidth=1.5)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = []
for seed, color in seed_colors.items():
    legend_elements.append(Line2D([0], [0], color=color, linewidth=2,
                                   label=f'seed {seed}'))
legend_elements.append(Line2D([0], [0], color='gray', linewidth=1))
for mode, ls, label in zip(modes, ['-', '--', ':', '-.'], mode_labels):
    legend_elements.append(Line2D([0], [0], color='gray', linestyle=ls,
                                   linewidth=1.5, label=label))

ax3.legend(handles=legend_elements, fontsize=8, ncol=2, loc='upper left')
ax3.set_xlabel('Environment steps (×1000)', fontsize=10)
ax3.set_ylabel('Buffer success fraction', fontsize=10)
ax3.set_title('(C) Per-Seed Learning Curves by Priority Mode',
              fontsize=11, fontweight='bold')
ax3.axhline(y=LEARN_THRESHOLD, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
ax3.set_ylim(-0.02, 0.55)

# ── Panel D: Q-value trajectories ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

for mode, ls in zip(modes, ['-', '--', ':', '-.']):
    for seed in seeds:
        traj = trajectories.get(mode, {}).get(seed, [])
        if not traj:
            continue
        steps = [t[0] / 1000 for t in traj]
        qs = [t[2] for t in traj]
        ax4.plot(steps, qs, color=seed_colors[seed], linestyle=ls,
                 alpha=0.7, linewidth=1.5)

ax4.set_xlabel('Environment steps (×1000)', fontsize=10)
ax4.set_ylabel('Mean Q-value', fontsize=10)
ax4.set_title('(D) Q-Value Trajectories\n(divergence without learning = pathological)',
              fontsize=11, fontweight='bold')
ax4.set_yscale('symlog', linthresh=10)

# Annotate pathological cases
# seed 99 under RND-PER: Q→161 with sr=0
ax4.annotate('seed 99 / RND-PER\nQ→162, sr=0 (pathological)',
             xy=(80, 161.7), fontsize=7, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'),
             xytext=(60, 300))
# seed 123 under adaptive: Q→225 with sr~0
ax4.annotate('seed 123 / Adaptive\nQ→225, sr≈0 (diverged)',
             xy=(100, 225.2), fontsize=7, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'),
             xytext=(75, 500))

# ── Panel E: Reward discovery timing ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])

bar_width = 0.18
x_positions = np.arange(len(seeds))

for j, (mode, label) in enumerate(zip(modes, mode_labels)):
    discovery_steps = []
    for seed in seeds:
        step = first_reward_step(mode, seed)
        discovery_steps.append(step / 1000 if step is not None else 110)  # 110 = never

    bars = ax5.bar(x_positions + j * bar_width, discovery_steps, bar_width,
                   label=label, alpha=0.8)
    # Mark "never" bars
    for i, (ds, bar) in enumerate(zip(discovery_steps, bars)):
        if ds >= 110:
            bar.set_hatch('///')
            bar.set_alpha(0.3)

ax5.set_xticks(x_positions + 1.5 * bar_width)
ax5.set_xticklabels([f'seed {s}' for s in seeds], fontsize=10)
ax5.set_ylabel('First reward discovery (×1000 steps)', fontsize=10)
ax5.set_title('(E) Reward Discovery Timing by Seed × Mode\n'
              '(hatched = never discovered within training)',
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_ylim(0, 120)
ax5.axhline(y=100, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

# ── Panel F: Exploration bifurcation summary ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

summary_text = """KEY FINDINGS — Seed-Switching Under Priority Regimes

1. SEED IDENTITY MATTERS MORE THAN MODE
   Uniform: {7, 123, 256} learn    RND-PER: {42, 7, 123} learn
   RPE-PER: {7, 123} learn         Adaptive: {7, 256} learn

   → Same 3/5 count, completely different seed composition

2. SEED 42: THE DIAGNOSTIC CASE
   • Never learns under uniform (sr=0.000 at 100k)
   • Learns FASTEST under RND-PER (sr=0.056 at 30k, 0.369 at 90k)
   → Novelty-seeking literally redirected this seed's exploration
     into a region of state space where it found reward early

3. PATHOLOGICAL Q-DIVERGENCE
   • Seed 99 / RND-PER: Q→162 with sr=0 (pure exploration noise)
   • Seed 123 / Adaptive: Q→225 with sr≈0 (priority-induced instability)
   → Priority regimes can create self-reinforcing replay loops
     that drive Q-values up without any reward signal

4. SEED 7: THE ROBUST LEARNER
   Learns under ALL 4 modes — its initial exploration trajectory
   naturally encounters reward. Priority regime is irrelevant.

5. IMPLICATION FOR VLM-PER
   If RL-internal signals merely reshuffle which seeds succeed,
   VLM-PER must do something qualitatively different: provide
   information that is INDEPENDENT of the current Q-function's
   biases, breaking the exploration-exploitation deadlock."""

ax6.text(0.02, 0.98, summary_text, transform=ax6.transAxes,
         fontsize=8.5, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

fig.suptitle('Exploration Bifurcation: How Priority Signals Redirect Learning',
             fontsize=14, fontweight='bold', y=0.97)

out_path = 'figures/seed_switching_analysis.png'
os.makedirs('figures', exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')

# Also save to images/ for Quarto
quarto_path = '../../images/td_baseline/seed_switching_analysis.png'
os.makedirs(os.path.dirname(quarto_path), exist_ok=True)
plt.savefig(quarto_path, dpi=150, bbox_inches='tight')
print(f'Saved to {quarto_path}')
