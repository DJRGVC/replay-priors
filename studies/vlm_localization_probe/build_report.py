#!/usr/bin/env python3
"""Build a self-contained HTML report for the VLM Failure-Localization Probe study.

Embeds all figures as base64, reads results JSONs, and produces a single
report.html that can be opened locally or hosted anywhere.

Usage:
    python build_report.py          # generates report.html in this directory
    python build_report.py --open   # generates and opens in browser
"""

import base64, json, glob, os, sys, html
from pathlib import Path
from datetime import datetime

STUDY_DIR = Path(__file__).parent
FIGURES_DIR = STUDY_DIR / "figures"
RESULTS_DIR = STUDY_DIR / "results"
DATA_DIR = STUDY_DIR / "data"
OUTPUT = STUDY_DIR / "report.html"


def b64_img(path: Path) -> str:
    """Return a base64 data URI for a PNG file."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_results():
    """Load all results.json files into a dict keyed by directory name."""
    results = {}
    for rfile in sorted(RESULTS_DIR.glob("*/results.json")):
        name = rfile.parent.name
        with open(rfile) as f:
            results[name] = json.load(f)
    # also load top-level results.json
    top = RESULTS_DIR / "results.json"
    if top.exists():
        with open(top) as f:
            results["_main"] = json.load(f)
    return results


def load_sample_frames(task="reach-v3", rollout="rollout_000", indices=None):
    """Load a few sample frames as base64 for the report hero section."""
    frames_dir = DATA_DIR / task / rollout / "frames"
    if not frames_dir.exists():
        return []
    all_frames = sorted(frames_dir.glob("*.png"))
    if indices is None:
        # Pick 8 uniform frames
        n = len(all_frames)
        indices = [int(i * (n - 1) / 7) for i in range(8)]
    out = []
    for i in indices:
        if i < len(all_frames):
            out.append((i, b64_img(all_frames[i])))
    return out


def build_html():
    figures = {}
    for fig_path in sorted(FIGURES_DIR.glob("*.png")):
        figures[fig_path.stem] = b64_img(fig_path)

    results = load_results()
    sample_frames = load_sample_frames()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build sample frames HTML
    frames_html = ""
    if sample_frames:
        frames_html = '<div class="frames-strip">'
        for idx, b64 in sample_frames:
            frames_html += f'<div class="frame-item"><img src="{b64}" alt="Frame {idx}"><span>t={idx}</span></div>'
        frames_html += '</div>'

    # Build figures HTML
    def fig(name, caption=""):
        if name in figures:
            return f'''<figure>
                <img src="{figures[name]}" alt="{caption}" class="fig-img" onclick="this.classList.toggle('expanded')">
                <figcaption>{caption} <span class="click-hint">(click to expand)</span></figcaption>
            </figure>'''
        return f'<p class="missing-fig">Figure not found: {name}</p>'

    report_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLM Failure-Localization Probe — Research Report</title>
<style>
:root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2129;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --accent3: #d2a8ff;
    --red: #f85149;
    --orange: #d29922;
    --green: #3fb950;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}}
h1 {{ font-size: 2em; margin: 0.5em 0; color: var(--accent); }}
h2 {{ font-size: 1.5em; margin: 1.5em 0 0.5em; color: var(--accent3); border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
h3 {{ font-size: 1.15em; margin: 1em 0 0.4em; color: var(--accent2); }}
p {{ margin: 0.5em 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

.hero {{
    background: linear-gradient(135deg, #1a1e2e 0%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
}}
.hero h1 {{ margin-top: 0; }}
.hero .subtitle {{ color: var(--text-muted); font-size: 1.1em; margin-bottom: 16px; }}
.hero .meta {{ color: var(--text-muted); font-size: 0.85em; }}

.tldr {{
    background: var(--surface);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 16px 0;
}}
.tldr strong {{ color: var(--accent); }}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}}
.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}}
.stat-card .number {{ font-size: 2em; font-weight: 700; color: var(--accent); }}
.stat-card .label {{ font-size: 0.85em; color: var(--text-muted); }}

.frames-strip {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 12px 0;
    margin: 12px 0;
}}
.frame-item {{
    flex-shrink: 0;
    text-align: center;
}}
.frame-item img {{
    width: 120px;
    height: 120px;
    border-radius: 6px;
    border: 1px solid var(--border);
    object-fit: cover;
}}
.frame-item span {{
    display: block;
    font-size: 0.75em;
    color: var(--text-muted);
    margin-top: 4px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.9em;
}}
th, td {{
    padding: 8px 12px;
    border: 1px solid var(--border);
    text-align: left;
}}
th {{
    background: var(--surface2);
    color: var(--accent);
    font-weight: 600;
    position: sticky;
    top: 0;
}}
tr:nth-child(even) {{ background: var(--surface); }}
tr:hover {{ background: var(--surface2); }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.best {{ color: var(--green); font-weight: 700; }}
.worst {{ color: var(--red); }}
.neutral {{ color: var(--orange); }}

.finding {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}}
.finding-num {{
    display: inline-block;
    background: var(--accent);
    color: var(--bg);
    font-weight: 700;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 50%;
    margin-right: 8px;
}}
.tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    margin: 2px;
}}
.tag-positive {{ background: #0d3321; color: var(--green); border: 1px solid #1b4332; }}
.tag-negative {{ background: #3d1318; color: var(--red); border: 1px solid #5c1d24; }}
.tag-mixed {{ background: #3b2e04; color: var(--orange); border: 1px solid #5c4a0a; }}
.tag-info {{ background: #0c2d6b; color: var(--accent); border: 1px solid #1a4487; }}

figure {{
    margin: 16px 0;
    text-align: center;
}}
.fig-img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    cursor: pointer;
    transition: max-width 0.3s;
}}
.fig-img.expanded {{
    max-width: 150%;
    position: relative;
    z-index: 10;
}}
figcaption {{
    font-size: 0.85em;
    color: var(--text-muted);
    margin-top: 8px;
}}
.click-hint {{ font-size: 0.8em; opacity: 0.5; }}

.nav {{
    position: sticky;
    top: 0;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 8px 0;
    z-index: 100;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}}
.nav a {{
    padding: 4px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    font-size: 0.8em;
    white-space: nowrap;
}}
.nav a:hover {{ background: var(--surface2); text-decoration: none; }}

section {{ scroll-margin-top: 60px; }}

.experiment-timeline {{
    border-left: 3px solid var(--border);
    margin-left: 12px;
    padding-left: 20px;
}}
.experiment-timeline .entry {{
    position: relative;
    margin-bottom: 12px;
    padding: 8px 12px;
    background: var(--surface);
    border-radius: 6px;
    font-size: 0.85em;
}}
.experiment-timeline .entry::before {{
    content: '';
    position: absolute;
    left: -26px;
    top: 14px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent);
}}
.experiment-timeline .entry.negative::before {{ background: var(--red); }}
.experiment-timeline .entry.positive::before {{ background: var(--green); }}
.experiment-timeline .entry.neutral::before {{ background: var(--orange); }}

.cross-study {{
    background: linear-gradient(135deg, #1a1530 0%, var(--surface) 100%);
    border: 1px solid var(--accent3);
    border-radius: 8px;
    padding: 20px;
    margin: 16px 0;
}}

footer {{
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.85em;
    text-align: center;
}}

@media (max-width: 768px) {{
    body {{ padding: 12px; }}
    .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .frame-item img {{ width: 80px; height: 80px; }}
    table {{ font-size: 0.8em; }}
    th, td {{ padding: 4px 6px; }}
}}
</style>
</head>
<body>

<nav class="nav">
    <a href="#overview">Overview</a>
    <a href="#models">Model Comparison</a>
    <a href="#findings">Key Findings</a>
    <a href="#interventions">Interventions</a>
    <a href="#priorities">Priority Analysis</a>
    <a href="#figures">Figures</a>
    <a href="#timeline">Timeline</a>
    <a href="#cross-study">Cross-Study</a>
    <a href="#limitations">Limitations</a>
</nav>

<div class="hero">
    <h1>VLM Failure-Localization Probe</h1>
    <div class="subtitle">Can vision-language models localize failure timesteps in robotic manipulation rollouts from keyframe images alone?</div>
    <p>MetaWorld reach-v3 &bull; 150-step episodes &bull; Random policy (all failures) &bull; 224&times;224 RGB</p>
    <div class="meta">Report generated: {now} &bull; 23 iterations &bull; 9 models &bull; 8 interventions tested</div>
</div>

{frames_html}

<section id="overview">
<h2>Overview</h2>

<div class="stat-grid">
    <div class="stat-card"><div class="number">9</div><div class="label">Models Tested</div></div>
    <div class="stat-card"><div class="number">8</div><div class="label">Interventions</div></div>
    <div class="stat-card"><div class="number">44%</div><div class="label">Best &pm;10 Accuracy</div></div>
    <div class="stat-card"><div class="number">41.9</div><div class="label">Best MAE</div></div>
    <div class="stat-card"><div class="number">$0.80</div><div class="label">Total API Cost</div></div>
    <div class="stat-card"><div class="number">23</div><div class="label">Experiments Run</div></div>
</div>

<div class="tldr">
    <strong>TL;DR:</strong> VLMs achieve coarse failure localization (best &pm;10 = 44%) but are dominated by
    <strong>positional biases</strong> rather than visual understanding. Every model has a distinct bias pattern
    (center/start/end/grid-cell). Frame annotation has a <strong>U-shaped</strong> effect: helps weak (&minus;17%) and
    strong (&minus;30% GPT-4o) models, hurts mid-tier (+11% GPT-4o-mini). CoT and annotation are
    <strong>partially substitutable</strong> &mdash; both anchor temporal attention, and adding both yields no gain
    over either alone (GPT-4o 2&times;2 factorial). The fundamental bottleneck is <strong>visual acuity</strong>:
    distinguishing subtle arm position changes at 224&times;224 with ~30-pixel arm regions.
</div>
</section>

<section id="models">
<h2>Model Comparison</h2>
<h3>All models, K=8 uniform keyframes, reach-v3</h3>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>API</th>
            <th>Image Mode</th>
            <th>MAE &darr;</th>
            <th>Median &darr;</th>
            <th>&pm;10</th>
            <th>&pm;20</th>
            <th>Bias Pattern</th>
            <th>Cost/call</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Claude Sonnet 4.6</td>
            <td>Anthropic</td>
            <td>Multi-image</td>
            <td class="num best">41.9</td>
            <td class="num">34.0</td>
            <td class="num">20%</td>
            <td class="num">35%</td>
            <td>center (t&asymp;85)</td>
            <td class="num">$0.004</td>
        </tr>
        <tr>
            <td>Gemini 3 Flash Preview</td>
            <td>Google</td>
            <td>Multi-image</td>
            <td class="num">54.2</td>
            <td class="num best">14.0</td>
            <td class="num best">44%</td>
            <td class="num best">56%</td>
            <td>start (t=0)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>GPT-4o (ann.)</td>
            <td>GitHub</td>
            <td>Multi-image</td>
            <td class="num">52.7</td>
            <td class="num">43.5</td>
            <td class="num">10%</td>
            <td class="num">10%</td>
            <td>early-mid (t=42)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>Llama-3.2-90B</td>
            <td>GitHub</td>
            <td>Grid tiled</td>
            <td class="num">53.5</td>
            <td class="num">37.5</td>
            <td class="num worst">0%</td>
            <td class="num worst">0%</td>
            <td>grid-cell (t=42)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>GPT-4o-mini (no ann.)</td>
            <td>GitHub</td>
            <td>Multi-image</td>
            <td class="num">61.2</td>
            <td class="num">51.0</td>
            <td class="num">10%</td>
            <td class="num">20%</td>
            <td>late (t&asymp;106)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>Phi-4-multimodal</td>
            <td>GitHub</td>
            <td>Grid tiled</td>
            <td class="num">64.3</td>
            <td class="num">&mdash;</td>
            <td class="num worst">0%</td>
            <td class="num">10%</td>
            <td>grid-center (t=85)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>Gemini 2.5 Flash</td>
            <td>Google</td>
            <td>Multi-image</td>
            <td class="num">67.8</td>
            <td class="num">&mdash;</td>
            <td class="num">20%</td>
            <td class="num">30%</td>
            <td>end (t&asymp;149)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>GPT-4o-mini (ann.)</td>
            <td>GitHub</td>
            <td>Multi-image</td>
            <td class="num">68.0</td>
            <td class="num">63.5</td>
            <td class="num worst">0%</td>
            <td class="num">10%</td>
            <td>late (t&asymp;106,127)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>GPT-4o (no ann.)</td>
            <td>GitHub</td>
            <td>Multi-image</td>
            <td class="num">75.8</td>
            <td class="num">65.0</td>
            <td class="num worst">0%</td>
            <td class="num">20%</td>
            <td>start (t=0)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>Llama-3.2-11B</td>
            <td>GitHub</td>
            <td>Grid tiled</td>
            <td class="num">72.9</td>
            <td class="num">66.5</td>
            <td class="num">10%</td>
            <td class="num">10%</td>
            <td>grid-cell (t=106)</td>
            <td class="num">$0</td>
        </tr>
        <tr>
            <td>Gemini 2.5 Flash-Lite</td>
            <td>Google</td>
            <td>Multi-image</td>
            <td class="num worst">95.2</td>
            <td class="num worst">107.5</td>
            <td class="num">5%</td>
            <td class="num">10%</td>
            <td>late</td>
            <td class="num">$0</td>
        </tr>
    </tbody>
</table>
<p style="color: var(--text-muted); font-size: 0.85em;">
    MAE = Mean Absolute Error in timesteps (lower is better, out of 150). &pm;10/&pm;20 = fraction of predictions within 10/20 timesteps of ground truth.
    <span class="best">Green</span> = best in column. <span class="worst">Red</span> = worst.
</p>
</section>

<section id="findings">
<h2>Key Findings</h2>

<div class="finding">
    <h3><span class="finding-num">1</span> VLMs localize coarsely, but far from actionable
    <span class="tag tag-mixed">MIXED</span></h3>
    <p>Best MAE: Claude Sonnet at 41.9 (~28% of episode). Best &pm;10: Gemini 3 Flash Preview at 44%, but bimodal &mdash; 33% of its predictions are catastrophic (&gt;80 timesteps off). No model exceeds 44% within &pm;10.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">2</span> Every model has a distinct positional bias &mdash; none reason temporally
    <span class="tag tag-negative">CRITICAL</span></h3>
    <p>Models select positions in the <em>presentation format</em> (image sequence position, grid cell), not by reasoning about visual content. Bias patterns are stable within models but completely different between them:</p>
    <table>
        <tr><th>Model</th><th>Bias</th><th>Pattern</th></tr>
        <tr><td>Claude Sonnet</td><td>center</td><td>Predicts t&asymp;85 (middle keyframe)</td></tr>
        <tr><td>Gemini 3 Flash Preview</td><td>start</td><td>Predicts t=0 ("arm remains stationary")</td></tr>
        <tr><td>Gemini 2.5 Flash</td><td>end</td><td>Predicts t&asymp;149</td></tr>
        <tr><td>GPT-4o (ann.)</td><td>early-mid</td><td>6/10 at t=42 (native multi-image)</td></tr>
        <tr><td>GPT-4o (no ann.)</td><td>start</td><td>5/10 at t=0 (annotation completely shifts bias)</td></tr>
        <tr><td>GPT-4o-mini</td><td>late</td><td>t&asymp;106, 127 (despite native multi-image)</td></tr>
        <tr><td>Llama/Phi-4</td><td>grid-cell</td><td>Locks onto specific tile positions</td></tr>
    </table>
    <p><strong>Key control:</strong> GPT-4o-mini uses native multi-image (no grid) yet still shows strong late-bias &mdash; confirming bias is intrinsic, not a grid artifact.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">3</span> More keyframes do NOT help
    <span class="tag tag-negative">NEGATIVE</span></h3>
    <p>K sweep on Claude Sonnet: K=4 MAE=47.4, K=8 MAE=41.9, K=16 MAE=44.4, K=32 MAE=51.5. Flat from K=4&ndash;16, <em>worsens</em> at K=32. The bottleneck is visual understanding of subtle arm position changes (~0.1 unit, ~30 pixels), not temporal resolution.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">4</span> CoT &amp; annotation are substitutable temporal scaffolds
    <span class="tag tag-mixed">MIXED</span></h3>
    <p>CoT hurts mid/weak models (3/3 negative: Flash-Lite +7.3, Flash +7.3, Phi-4 <strong>+25.9 MAE</strong>).
    But GPT-4o 2&times;2 factorial reveals the mechanism: <strong>CoT and annotation are partially substitutable.</strong>
    Both anchor temporal attention. When annotation is already present, CoT adds nothing (52.7&rarr;52.2).
    When annotation is absent, CoT helps (75.8&rarr;65.0, &minus;14%). Once one scaffold is provided, the other is redundant.</p>
    <table>
        <tr><th>GPT-4o</th><th>Direct</th><th>CoT</th></tr>
        <tr><td><strong>Annotated</strong></td><td class="num">52.7</td><td class="num">52.2</td></tr>
        <tr><td><strong>Unannotated</strong></td><td class="num worst">75.8</td><td class="num neutral">65.0</td></tr>
    </table>
</div>

<div class="finding">
    <h3><span class="finding-num">5</span> Frame annotation has a U-shaped effect across model strength
    <span class="tag tag-mixed">U-SHAPED</span></h3>
    <p>VTimeCoT-style "t=X (N%)" overlays show a <strong>non-monotonic</strong> effect:</p>
    <table>
        <tr><th>Model</th><th>Tier</th><th>Unannotated</th><th>Annotated</th><th>&Delta;MAE</th></tr>
        <tr><td>Flash-Lite</td><td>Weak</td><td class="num">71.9</td><td class="num best">59.5</td><td class="num best">&minus;17%</td></tr>
        <tr><td>GPT-4o-mini</td><td>Mid</td><td class="num best">61.2</td><td class="num">68.0</td><td class="num worst">+11%</td></tr>
        <tr><td>GPT-4o</td><td>Strong</td><td class="num">75.8</td><td class="num best">52.7</td><td class="num best">&minus;30%</td></tr>
    </table>
    <p>Weak &amp; strong models benefit from temporal grounding; mid-tier models are distracted by the overlay text, worsening their positional bias.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">6</span> Two-pass adaptive probing fails
    <span class="tag tag-negative">NEGATIVE</span></h3>
    <p>Coarse K=4 &rarr; refine K=8 in &pm;15% window. MAE worsened 69.8 &rarr; 71.3 on Llama-3.2-90B. 6/10 rollouts worse. Coarse pass is too inaccurate (&asymp;70 MAE) to center a useful refinement window.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">7</span> Random sampling: different wrong answers, same accuracy
    <span class="tag tag-negative">NEGATIVE</span></h3>
    <p>Random breaks grid-position clustering (0 repeated vs 3/9), but MAE is identical (64.7 vs 63.8). Run-to-run variance &gt; strategy effect.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">8</span> Grid tiling adds grid-cell bias on top of model bias
    <span class="tag tag-info">INFO</span></h3>
    <p>Single-image APIs (GitHub Models) require tiling K frames into a grid. Models lock onto specific grid cells regardless of content. Multi-image APIs avoid this but have their own positional priors.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">9</span> Priority quality: overlap useful, KL harmful
    <span class="tag tag-mixed">MIXED</span></h3>
    <p>Converting VLM predictions to Gaussian-kernel replay priorities: KL always worse than uniform (&minus;6% to &minus;24%), but top-20% overlap +8&ndash;12% above uniform. Catastrophic misses poison KL, but good predictions correctly upweight failure-adjacent transitions. A confidence-gated hybrid could resolve this.</p>
</div>

<div class="finding">
    <h3><span class="finding-num">10</span> Push/pick-place unsuitable with random policies
    <span class="tag tag-info">INFO</span></h3>
    <p>100% ambiguous GT labels (random policy never contacts objects). Only reach-v3 has non-ambiguous GT. Trained-policy rollouts needed for multi-task evaluation.</p>
</div>
</section>

<section id="interventions">
<h2>Intervention Summary</h2>
<table>
    <thead>
        <tr>
            <th>Intervention</th>
            <th>Effect</th>
            <th>Verdict</th>
            <th>Details</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>More keyframes (K=4&rarr;32)</td>
            <td>Flat, then worse</td>
            <td><span class="tag tag-negative">NEGATIVE</span></td>
            <td>K=8 is optimal. K=32 worsens MAE by +10</td>
        </tr>
        <tr>
            <td>CoT prompting</td>
            <td>+7 to +26 MAE (mid/weak); neutral on strong</td>
            <td><span class="tag tag-mixed">MODEL-DEP</span></td>
            <td>3/3 negative on mid/weak. Neutral on GPT-4o when annotated. Substitutable with annotation.</td>
        </tr>
        <tr>
            <td>Frame annotation</td>
            <td>&minus;30% to +11% MAE</td>
            <td><span class="tag tag-mixed">U-SHAPED</span></td>
            <td>Helps weak (Flash-Lite &minus;17%) &amp; strong (GPT-4o &minus;30%), hurts mid (GPT-4o-mini +11%)</td>
        </tr>
        <tr>
            <td>CoT &times; Annotation (2&times;2)</td>
            <td>Substitutable, not additive</td>
            <td><span class="tag tag-info">MECHANISTIC</span></td>
            <td>GPT-4o: ann+direct&asymp;ann+CoT&asymp;52. Both anchor temporal attention; redundant together.</td>
        </tr>
        <tr>
            <td>Proprio-as-text</td>
            <td>+48 MAE (n=2)</td>
            <td><span class="tag tag-mixed">INCONCLUSIVE</span></td>
            <td>Only 2 valid predictions due to rate limits</td>
        </tr>
        <tr>
            <td>Random sampling</td>
            <td>&asymp;0 MAE change</td>
            <td><span class="tag tag-negative">NEGATIVE</span></td>
            <td>Breaks clustering but doesn't improve accuracy</td>
        </tr>
        <tr>
            <td>Two-pass refinement</td>
            <td>+1.5 MAE</td>
            <td><span class="tag tag-negative">NEGATIVE</span></td>
            <td>Coarse pass too inaccurate to guide refinement</td>
        </tr>
        <tr>
            <td>Native multi-image</td>
            <td>No grid bias, same accuracy</td>
            <td><span class="tag tag-info">CONTROL</span></td>
            <td>Confirms bias is intrinsic, not grid artifact</td>
        </tr>
    </tbody>
</table>
</section>

<section id="priorities">
<h2>Priority Quality Analysis</h2>
<p>VLM predictions converted to Gaussian-kernel replay priorities (&sigma;=10), compared against oracle and uniform baselines.</p>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>K</th>
            <th>N</th>
            <th>MAE</th>
            <th>KL (VLM)</th>
            <th>KL (Uniform)</th>
            <th>KL Imp.</th>
            <th>Top-20% Overlap</th>
            <th>Overlap (Unif)</th>
            <th>Overlap Imp.</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Claude Sonnet</td>
            <td class="num">4</td>
            <td class="num">10</td>
            <td class="num">47.4</td>
            <td class="num">1.87</td>
            <td class="num">1.50</td>
            <td class="num worst">&minus;24%</td>
            <td class="num">13.3%</td>
            <td class="num">11.7%</td>
            <td class="num">+1.7%</td>
        </tr>
        <tr>
            <td>Claude Sonnet</td>
            <td class="num">8</td>
            <td class="num">14</td>
            <td class="num">43.4</td>
            <td class="num">1.64</td>
            <td class="num">1.53</td>
            <td class="num worst">&minus;8%</td>
            <td class="num best">27.1%</td>
            <td class="num">15.5%</td>
            <td class="num best">+11.7%</td>
        </tr>
        <tr>
            <td>Claude Sonnet</td>
            <td class="num">16</td>
            <td class="num">20</td>
            <td class="num">44.4</td>
            <td class="num">1.59</td>
            <td class="num">1.50</td>
            <td class="num worst">&minus;6%</td>
            <td class="num">26.7%</td>
            <td class="num">15.8%</td>
            <td class="num">+10.8%</td>
        </tr>
        <tr>
            <td>Claude Sonnet</td>
            <td class="num">32</td>
            <td class="num">32</td>
            <td class="num">55.4</td>
            <td class="num">1.72</td>
            <td class="num">1.49</td>
            <td class="num worst">&minus;15%</td>
            <td class="num">17.7%</td>
            <td class="num">13.5%</td>
            <td class="num">+4.2%</td>
        </tr>
    </tbody>
</table>
<p><strong>Key tension:</strong> KL says VLM priorities are always <em>worse</em> than uniform (catastrophic misses dominate). But top-20% overlap says VLM priorities <em>help</em> sample near failures (+8&ndash;12%). Resolution requires a downstream RL experiment or a confidence-gated hybrid approach.</p>
</section>

<section id="figures">
<h2>Figures</h2>

{fig("k_sweep_reach_v3", "K sweep on Claude Sonnet (reach-v3): MAE is flat K=4-16, worsens at K=32")}

{fig("gt_quality_analysis", "Ground-truth quality analysis: only reach-v3 has reliable failure labels")}

{fig("priority_good_vs_bad_sonnet_k8", "Priority distributions: good vs bad VLM predictions (Claude Sonnet, K=8)")}

{fig("sigma_sweep_sonnet_k8", "Sigma sensitivity analysis: σ=10-15 is the sweet spot for priority specificity")}

{fig("priority_comparison_sonnet_k8", "VLM vs oracle vs uniform priority distributions (Claude Sonnet, K=8)")}

{fig("priority_comparison_sonnet_k8_sigma_sweep", "Sigma sweep: wider σ reduces KL gap but also overlap advantage")}
</section>

<section id="timeline">
<h2>Experiment Timeline</h2>
<div class="experiment-timeline">
    <div class="entry positive">
        <strong>iter_001</strong> &mdash; Data collection: 60 rollouts &times; 3 tasks (reach, push, pick-place)
    </div>
    <div class="entry positive">
        <strong>iter_002</strong> &mdash; Claude Sonnet baseline: MAE=41.9, &pm;10=20% on reach-v3
    </div>
    <div class="entry negative">
        <strong>iter_003</strong> &mdash; K sweep (4/8/16/32): no improvement with more frames
    </div>
    <div class="entry negative">
        <strong>iter_004</strong> &mdash; Gemini Flash-Lite: MAE=95.2 (much worse)
    </div>
    <div class="entry positive">
        <strong>iter_005</strong> &mdash; Gemini 3 Flash Preview: MAE=54.2, &pm;10=44% (best &pm;10!)
    </div>
    <div class="entry negative">
        <strong>iter_006</strong> &mdash; CoT prompt: hurts Flash (+7.3 MAE), suggestive for Flash Preview (n=3)
    </div>
    <div class="entry positive">
        <strong>iter_007</strong> &mdash; Frame annotation: &minus;17% MAE on Flash-Lite
    </div>
    <div class="entry neutral">
        <strong>iter_008</strong> &mdash; Proprio-as-text: n=2 valid, inconclusive (rate-limited)
    </div>
    <div class="entry neutral">
        <strong>iter_009</strong> &mdash; Groq backend built, results summary. API key pending.
    </div>
    <div class="entry positive">
        <strong>iter_010</strong> &mdash; Priority analysis: top-20% overlap +12%, but KL &minus;8%
    </div>
    <div class="entry neutral">
        <strong>iter_011</strong> &mdash; Two-pass + random sampling code (API-blocked)
    </div>
    <div class="entry positive">
        <strong>iter_012</strong> &mdash; GT quality: push/pick-place unsuitable with random policy
    </div>
    <div class="entry positive">
        <strong>iter_013</strong> &mdash; GitHub Models: Llama 3.2 11B/90B probes. Grid-position bias.
    </div>
    <div class="entry negative">
        <strong>iter_014</strong> &mdash; Random vs uniform sampling: no difference (MAE 64.7 vs 63.8)
    </div>
    <div class="entry negative">
        <strong>iter_015</strong> &mdash; Two-pass adaptive: NEGATIVE (MAE 69.8&rarr;71.3)
    </div>
    <div class="entry negative">
        <strong>iter_016</strong> &mdash; CoT on Phi-4: NEGATIVE (MAE 64.3&rarr;90.2, +40%)
    </div>
    <div class="entry positive">
        <strong>iter_017</strong> &mdash; FINDINGS.md synthesis: 10 key findings crystallized
    </div>
    <div class="entry neutral">
        <strong>iter_018</strong> &mdash; GPT-4o-mini: MAE=68.0, confirms positional bias is intrinsic
    </div>
    <div class="entry neutral">
        <strong>iter_019</strong> &mdash; Annotation &pm; comparison: annotation hurts GPT-4o-mini (+11% MAE)
    </div>
    <div class="entry positive">
        <strong>iter_020</strong> &mdash; Self-contained HTML report interface (build_report.py &rarr; report.html)
    </div>
    <div class="entry positive">
        <strong>iter_021</strong> &mdash; Literature review: 6 related papers, CVPR 2025 confirms positional bias
    </div>
    <div class="entry positive">
        <strong>iter_022</strong> &mdash; GPT-4o annotation &pm;: annotation helps strong model (&minus;30% MAE), U-shaped effect
    </div>
    <div class="entry positive">
        <strong>iter_023</strong> &mdash; GPT-4o CoT&times;annotation 2&times;2: CoT &amp; annotation substitutable temporal scaffolds
    </div>
</div>
</section>

<section id="cross-study">
<h2>Cross-Study Connection</h2>
<div class="cross-study">
    <h3>VLM Probe &times; TD-Error Baseline</h3>
    <p>The sibling study (<code>td_error_baseline</code>) finds TD-error PER is <strong>uninformative in early training</strong>
    (Spearman &rho; &asymp; 0 for first 60&ndash;80% of training). TD-PER at default &alpha;=0.6 prevents all seeds from learning (0/5).</p>
    <p><strong>Combined interpretation:</strong> Both traditional (TD-error) and VLM-based replay prioritization struggle in the
    sparse-reward manipulation setting. TD-error fails because the critic has no signal early on; VLMs fail because the visual
    differences between good and bad timesteps are too subtle at 224&times;224 resolution. <strong>Neither provides a reliable priority
    signal when it would be most needed (early training).</strong></p>
</div>
</section>

<section id="limitations">
<h2>Limitations &amp; Open Questions</h2>
<ol>
    <li><strong>Single task (reach-v3)</strong> &mdash; all quantitative findings on one task with random policy</li>
    <li><strong>Gemini rate limits</strong> &mdash; best free model (Gemini 3 Flash Preview) severely rate-limited, several findings rest on n=3&ndash;10</li>
    <li><strong>No downstream RL</strong> &mdash; overlap-vs-KL tension can only be resolved by training with VLM priorities</li>
    <li><strong>Resolution ceiling</strong> &mdash; arm is ~30 pixels in 224&times;224; higher resolution or crops may help</li>
    <li><strong>Limited prompt engineering</strong> &mdash; only direct and CoT tested; pairwise comparison, binary search, video-native models unexplored</li>
</ol>

<h3>Next Steps</h3>
<ul>
    <li>Test annotation on Gemini 3 Flash Preview (best model + untested annotation) &mdash; quota-gated</li>
    <li>Complete CoT comparison on Gemini 3 Flash Preview (n&ge;9) &mdash; quota-gated</li>
    <li>Explore Cohere aya-vision-32b (1000 req/month, native multi-image, no grid)</li>
    <li>Test SoFA (training-free positional bias mitigation from CVPR 2025)</li>
    <li>Confidence-gated hybrid priority scheme</li>
    <li>Trained-policy rollouts for multi-task evaluation</li>
</ul>
</section>

<footer>
    <p>VLM Failure-Localization Probe &bull; replay-priors project &bull; Generated {now}</p>
    <p>Agent: vlm_probe &bull; Branch: agent/vlm_probe &bull; 23 iterations</p>
</footer>

</body>
</html>'''

    with open(OUTPUT, "w") as f:
        f.write(report_html)
    print(f"Report written to {OUTPUT}")
    print(f"  Figures embedded: {len(figures)}")
    print(f"  Sample frames: {len(sample_frames)}")
    print(f"  File size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")


if __name__ == "__main__":
    build_html()
    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(f"file://{OUTPUT.resolve()}")
