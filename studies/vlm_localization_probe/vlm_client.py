"""Thin VLM client for failure-timestep localization.

Takes a task description + K keyframe images and asks the VLM to predict
which timestep window contains the critical failure moment.

Supports:
  - claude-sonnet-4-6 (Anthropic vision API)
  - gpt-4o (OpenAI vision API)

Usage:
    from vlm_client import predict_failure, sample_keyframes

    frames, indices = sample_keyframes("data/reach-v3/rollout_000/frames", K=8)
    result = predict_failure(
        task_description="reach-v3: robot arm must reach a target position",
        keyframes=frames,
        keyframe_indices=indices,
        total_timesteps=150,
        model="claude-sonnet-4-6",
    )
    print(result)
    # {"failure_timestep": 42, "confidence": 0.7, "rationale": "..."}
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


# ── Rate-limit retry config ──────────────────────────────────────
MAX_RETRIES = 5
RETRY_BASE_DELAY = 15  # seconds — conservative for 5 RPM limit


def _retry_on_rate_limit(fn, *args, **kwargs):
    """Call fn with exponential backoff on 429 rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Rate limited after {MAX_RETRIES} retries")

# ── Keyframe sampling ──────────────────────────────────────────────

def annotate_frame(img: Image.Image, timestep: int, total_timesteps: int) -> Image.Image:
    """Overlay timestep index and progress fraction on a keyframe image.

    Inspired by VTimeCoT (ICCV 2025): visual progress anchors reduce positional
    bias by embedding temporal context directly in the image.
    """
    from PIL import ImageDraw, ImageFont

    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Use a legible font size (~5% of image height)
    font_size = max(12, h // 18)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    frac = timestep / max(total_timesteps - 1, 1)
    label = f"t={timestep} ({frac:.0%})"

    # Draw with black outline for readability
    x, y = 4, 4
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), label, fill="black", font=font)
    draw.text((x, y), label, fill="white", font=font)

    return img


def sample_keyframes(
    frames_dir: str | Path,
    K: int = 8,
    strategy: Literal["uniform", "random"] = "uniform",
) -> tuple[list[Image.Image], list[int]]:
    """Sample K keyframes from a rollout's frames directory.

    Strategies:
      - "uniform": evenly spaced indices from 0..T-1
      - "random": K random indices (breaks positional priors)

    Returns (list of PIL Images, list of timestep indices).
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("*.png"))
    T = len(all_frames)
    if T == 0:
        raise ValueError(f"No frames found in {frames_dir}")
    K = min(K, T)

    if strategy == "uniform":
        indices = np.linspace(0, T - 1, K, dtype=int).tolist()
    elif strategy == "random":
        indices = sorted(np.random.choice(T, size=K, replace=False).tolist())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)
    indices = unique_indices

    images = [Image.open(all_frames[i]) for i in indices]
    return images, indices


def sample_keyframes_around(
    frames_dir: str | Path,
    center: int,
    K: int = 8,
    window_fraction: float = 0.3,
) -> tuple[list[Image.Image], list[int]]:
    """Sample K keyframes densely around a center timestep.

    Used for the refinement pass in two-pass adaptive probing.
    Samples K frames from [center - window/2, center + window/2],
    clamped to episode bounds.

    Args:
        center: timestep to center the dense window on
        K: number of frames to sample
        window_fraction: fraction of episode length for the window
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("*.png"))
    T = len(all_frames)
    if T == 0:
        raise ValueError(f"No frames found in {frames_dir}")
    K = min(K, T)

    half_window = int(T * window_fraction / 2)
    lo = max(0, center - half_window)
    hi = min(T - 1, center + half_window)
    # Ensure we have at least K frames in the window
    if hi - lo + 1 < K:
        lo = max(0, hi - K + 1)
        if hi - lo + 1 < K:
            hi = min(T - 1, lo + K - 1)

    indices = np.linspace(lo, hi, K, dtype=int).tolist()

    # Deduplicate
    seen = set()
    unique_indices = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)
    indices = unique_indices

    images = [Image.open(all_frames[i]) for i in indices]
    return images, indices


# ── Image encoding ─────────────────────────────────────────────────

def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ── Prompt construction ────────────────────────────────────────────

SYSTEM_PROMPT_DIRECT = """\
You are analyzing a robot manipulation rollout that FAILED its task.
You will see K keyframes sampled from the episode at known timestep indices.
Your job: identify the single timestep INDEX (from the provided list of indices) \
where the critical failure occurred — the moment the robot's behavior most \
clearly diverged from what would achieve the task goal.

IMPORTANT: You MUST pick one of the provided timestep indices. \
Respond with ONLY a valid JSON object on a single line, nothing else:
{"failure_timestep": <int from the index list>, "confidence": <float 0-1>, "rationale": "<one sentence>"}
"""

SYSTEM_PROMPT_COT = """\
You are analyzing a robot manipulation rollout that FAILED its task.
You will see K keyframes sampled from the episode at known timestep indices.
Your job: identify when the critical failure occurred — the moment the robot's \
behavior most clearly diverged from what would achieve the task goal.

You MUST follow this exact 3-step reasoning process:

**Step 1 — SUMMARIZE:** For EACH keyframe image, describe in 1-2 sentences what \
you see: the robot arm's position, gripper state (open/closed), the object's \
location relative to the target, and any notable changes from the previous frame. \
Reference specific visual details — do NOT skip any frame.

**Step 2 — THINK:** Based on your frame-by-frame descriptions, identify which \
transition between consecutive keyframes shows the clearest sign that the task \
will fail. What specific visual evidence supports this? Is the arm moving away \
from the target? Has it missed the object? Is it stuck?

**Step 3 — ANSWER:** First estimate what fraction of the episode (0.0 to 1.0) has \
elapsed when the failure occurs. Then convert that fraction to the nearest \
timestep index from the provided list.

After your reasoning, output the final answer as a JSON object on its own line:
{"failure_timestep": <int from the index list>, "confidence": <float 0-1>, "rationale": "<one sentence>"}
"""

def extract_proprio_text(
    task_name: str,
    proprio: np.ndarray,
    keyframe_indices: list[int],
) -> list[str]:
    """Extract human-readable proprio features at each keyframe index.

    Returns one string per keyframe describing end-effector and goal/object positions.
    """
    labels = []
    for idx in keyframe_indices:
        t = min(idx, len(proprio) - 1)
        hand = proprio[t, 0:3]

        if "reach" in task_name:
            goal = proprio[0, 36:39]  # static goal
            dist = np.linalg.norm(hand - goal)
            labels.append(
                f"t={idx}: hand=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f}), "
                f"goal=({goal[0]:.3f},{goal[1]:.3f},{goal[2]:.3f}), "
                f"dist={dist:.3f}"
            )
        elif "push" in task_name or "pick-place" in task_name:
            obj = proprio[t, 4:7]
            hand_obj_dist = np.linalg.norm(hand - obj)
            labels.append(
                f"t={idx}: hand=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f}), "
                f"object=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f}), "
                f"hand-obj dist={hand_obj_dist:.3f}"
            )
        else:
            labels.append(
                f"t={idx}: hand=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f})"
            )
    return labels


def _build_user_message(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> str:
    """Build the text portion of the user message."""
    lines = [
        f"Task: {task_description}",
        f"Total episode length: {total_timesteps} timesteps",
        f"This episode FAILED (the task was not completed).",
        f"",
        f"You are shown {len(keyframes)} keyframes at these timestep indices:",
        f"  {keyframe_indices}",
    ]
    if proprio_labels:
        lines += [
            f"",
            f"Numeric state at each keyframe (XYZ coordinates in meters):",
        ]
        for label in proprio_labels:
            lines.append(f"  {label}")
    if prompt_style == "cot":
        lines += [
            f"",
            f"Follow the 3-step process (SUMMARIZE → THINK → ANSWER) described in your instructions.",
            f"Describe each frame carefully before making your prediction.",
        ]
    else:
        lines += [
            f"",
            f"For each image below, the timestep index is labeled.",
            f"Identify which timestep index marks the critical failure point.",
        ]
    return "\n".join(lines)


def _get_system_prompt(prompt_style: str = "direct") -> str:
    """Return the appropriate system prompt for the given style."""
    if prompt_style == "cot":
        return SYSTEM_PROMPT_COT
    return SYSTEM_PROMPT_DIRECT


def _parse_json_from_text(text: str) -> dict:
    """Extract JSON object from text that may contain CoT reasoning before it."""
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try parsing the whole thing first (direct style)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # For CoT: find the last JSON object in the text
    import re
    # Find all JSON-like objects
    json_matches = list(re.finditer(r'\{[^{}]*"failure_timestep"[^{}]*\}', text))
    if json_matches:
        try:
            return json.loads(json_matches[-1].group())
        except json.JSONDecodeError:
            pass

    # Last resort: find any {...} at end of text
    last_brace = text.rfind('{')
    if last_brace >= 0:
        candidate = text[last_brace:]
        end = candidate.find('}')
        if end >= 0:
            try:
                return json.loads(candidate[:end+1])
            except json.JSONDecodeError:
                pass

    return {"failure_timestep": None, "confidence": 0.0, "rationale": f"parse_error: {text[:200]}"}


# ── Claude (Anthropic) backend ─────────────────────────────────────

def _call_claude(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "claude-sonnet-4-6",
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> dict:
    import anthropic

    client = anthropic.Anthropic()

    # Build content blocks: interleave images with labels
    content = []
    user_text = _build_user_message(task_description, keyframes, keyframe_indices, total_timesteps, prompt_style, proprio_labels)
    content.append({"type": "text", "text": user_text})

    for img, idx in zip(keyframes, keyframe_indices):
        b64 = _pil_to_base64(img)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        content.append({"type": "text", "text": f"↑ Timestep {idx}"})

    if prompt_style == "cot":
        content.append({"type": "text", "text": "Now follow the 3-step process. After your reasoning, output the JSON on its own line."})
    else:
        content.append({"type": "text", "text": "Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line."})

    max_tokens = 2048 if prompt_style == "cot" else 512

    def _do_call():
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_get_system_prompt(prompt_style),
            messages=[{"role": "user", "content": content}],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.content[0].text.strip()
    result = _parse_json_from_text(text)

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = response.usage.input_tokens
    result["_output_tokens"] = response.usage.output_tokens
    result["_model"] = model
    return result


# ── OpenAI backend ─────────────────────────────────────────────────

def _call_openai(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "gpt-4o",
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> dict:
    from openai import OpenAI

    client = OpenAI()

    user_text = _build_user_message(task_description, keyframes, keyframe_indices, total_timesteps, prompt_style, proprio_labels)

    # Build content parts
    content = [{"type": "text", "text": user_text}]
    for img, idx in zip(keyframes, keyframe_indices):
        b64 = _pil_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
        })
        content.append({"type": "text", "text": f"↑ Timestep {idx}"})

    if prompt_style == "cot":
        content.append({"type": "text", "text": "Now follow the 3-step process. After your reasoning, output the JSON on its own line."})
    else:
        content.append({"type": "text", "text": "Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line."})

    max_tokens = 2048 if prompt_style == "cot" else 512

    def _do_call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _get_system_prompt(prompt_style)},
                {"role": "user", "content": content},
            ],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()
    result = _parse_json_from_text(text)

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = response.usage.prompt_tokens
    result["_output_tokens"] = response.usage.completion_tokens
    result["_model"] = model
    return result


# ── Gemini (Google) backend ────────────────────────────────────────

def _call_gemini(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "gemini-2.5-flash",
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client()

    user_text = _build_user_message(task_description, keyframes, keyframe_indices, total_timesteps, prompt_style, proprio_labels)

    # Build content parts: text + interleaved images with labels
    contents = [user_text]
    for img, idx in zip(keyframes, keyframe_indices):
        contents.append(img)
        contents.append(f"↑ Timestep {idx}")

    if prompt_style == "cot":
        contents.append("Now follow the 3-step process. After your reasoning, output the JSON on its own line.")
    else:
        contents.append("Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line.")

    # For CoT, disable JSON response mode so the model can reason freely
    response_mime = None if prompt_style == "cot" else "application/json"
    max_tokens = 4096 if prompt_style == "cot" else 2048

    def _do_call():
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_get_system_prompt(prompt_style),
                max_output_tokens=max_tokens,
                response_mime_type=response_mime,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.text.strip()
    result = _parse_json_from_text(text)

    # Gemini usage metadata
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = input_tokens
    result["_output_tokens"] = output_tokens
    result["_model"] = model
    return result


# ── Groq backend ─────────────────────────────────────────────────

def _call_groq(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> dict:
    from groq import Groq

    client = Groq()

    user_text = _build_user_message(task_description, keyframes, keyframe_indices, total_timesteps, prompt_style, proprio_labels)

    # Build content parts (OpenAI-compatible format)
    content = [{"type": "text", "text": user_text}]
    for img, idx in zip(keyframes, keyframe_indices):
        b64 = _pil_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        content.append({"type": "text", "text": f"↑ Timestep {idx}"})

    if prompt_style == "cot":
        content.append({"type": "text", "text": "Now follow the 3-step process. After your reasoning, output the JSON on its own line."})
    else:
        content.append({"type": "text", "text": "Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line."})

    max_tokens = 2048 if prompt_style == "cot" else 512

    def _do_call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _get_system_prompt(prompt_style)},
                {"role": "user", "content": content},
            ],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()
    result = _parse_json_from_text(text)

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = response.usage.prompt_tokens
    result["_output_tokens"] = response.usage.completion_tokens
    result["_model"] = model
    return result


# ── Grid tiling (for single-image APIs) ──────────────────────────

def _tile_keyframes(
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    cols: int = 4,
) -> Image.Image:
    """Tile K keyframes into a single grid image with timestep labels.

    Each cell shows the frame with a small label at top-left.
    Returns a single PIL image suitable for single-image APIs.
    """
    from PIL import ImageDraw, ImageFont

    K = len(keyframes)
    rows = (K + cols - 1) // cols
    cell_w, cell_h = keyframes[0].size

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)

    font_size = max(10, cell_h // 20)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (img, idx) in enumerate(zip(keyframes, keyframe_indices)):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, r * cell_h
        grid.paste(img, (x0, y0))
        frac = idx / max(total_timesteps - 1, 1)
        label = f"t={idx} ({frac:.0%})"
        # Black outline + white text
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((x0 + 3 + dx, y0 + 3 + dy), label, fill="black", font=font)
        draw.text((x0 + 3, y0 + 3), label, fill="yellow", font=font)

    return grid


# ── GitHub Models backend ─────────────────────────────────────────

def _call_github_models(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "Llama-3.2-11B-Vision-Instruct",
    prompt_style: str = "direct",
    proprio_labels: list[str] | None = None,
) -> dict:
    """GitHub Models API — free, OpenAI-compatible, vision-capable."""
    from openai import OpenAI
    import subprocess

    # Get token from gh CLI
    token = os.environ.get("GITHUB_TOKEN") or subprocess.check_output(
        ["gh", "auth", "token"], text=True
    ).strip()

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    user_text = _build_user_message(
        task_description, keyframes, keyframe_indices,
        total_timesteps, prompt_style, proprio_labels,
    )

    # GitHub Models limits to 1 image per request — tile into grid
    grid = _tile_keyframes(keyframes, keyframe_indices, total_timesteps)
    b64 = _pil_to_base64(grid, fmt="JPEG")  # JPEG for smaller payload

    suffix = (
        "Now follow the 3-step process. After your reasoning, output the JSON on its own line."
        if prompt_style == "cot" else
        "Now respond with ONLY the JSON object. No explanation, no markdown — just the raw JSON on one line."
    )
    content = [
        {"type": "text", "text": user_text + f"\n\nThe image below is a {len(keyframes)}-frame grid (left-to-right, top-to-bottom). Each cell is labeled with its timestep.\n\n{suffix}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
    ]

    max_tokens = 2048 if prompt_style == "cot" else 512

    def _do_call():
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _get_system_prompt(prompt_style)},
                {"role": "user", "content": content},
            ],
        )

    t0 = time.time()
    response = _retry_on_rate_limit(_do_call)
    latency = time.time() - t0

    text = response.choices[0].message.content.strip()
    result = _parse_json_from_text(text)

    result["_latency_s"] = round(latency, 2)
    result["_input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
    result["_output_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
    result["_model"] = model
    return result


# ── Unified entry point ────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "reach-v3": "Sawyer robot arm must move its end-effector to reach a target 3D position marked by a small sphere. The arm starts at a neutral pose.",
    "push-v3": "Sawyer robot arm must push a small puck across the table to a target position. The arm needs to make contact with the puck and slide it to the goal.",
    "pick-place-v3": "Sawyer robot arm must pick up a small puck from the table and place it at an elevated target position. Requires grasping, lifting, and placing.",
}


def predict_failure(
    task_description: str,
    keyframes: list[Image.Image],
    keyframe_indices: list[int],
    total_timesteps: int,
    model: str = "claude-sonnet-4-6",
    prompt_style: str = "direct",
    annotate_frames: bool = False,
    proprio_labels: list[str] | None = None,
) -> dict:
    """Predict the failure timestep using a VLM.

    Args:
        prompt_style: "direct" (predict timestep immediately) or
                      "cot" (Summarize→Think→Answer structured reasoning)
        annotate_frames: If True, overlay timestep index + progress fraction
                         on each keyframe image (VTimeCoT-style visual anchor).
        proprio_labels: If provided, list of human-readable proprio state strings
                        (one per keyframe) to include as text in the prompt.

    Returns dict with: failure_timestep, confidence, rationale,
    plus _latency_s, _input_tokens, _output_tokens, _model.
    """
    if annotate_frames:
        keyframes = [annotate_frame(img, idx, total_timesteps) for img, idx in zip(keyframes, keyframe_indices)]

    kwargs = dict(
        task_description=task_description,
        keyframes=keyframes,
        keyframe_indices=keyframe_indices,
        total_timesteps=total_timesteps,
        model=model,
        prompt_style=prompt_style,
        proprio_labels=proprio_labels,
    )

    if "claude" in model or "sonnet" in model or "opus" in model or "haiku" in model:
        return _call_claude(**kwargs)
    elif "gpt" in model or "o1" in model or "o3" in model or "o4" in model:
        return _call_openai(**kwargs)
    elif "gemini" in model:
        return _call_gemini(**kwargs)
    elif model.startswith("gh:"):
        kwargs["model"] = model[3:]  # strip "gh:" prefix
        return _call_github_models(**kwargs)
    elif "llama" in model or "groq" in model:
        return _call_groq(**kwargs)
    else:
        raise ValueError(f"Unknown model family: {model}. Use a Claude, OpenAI, Gemini, Groq, or gh:<model> name.")


# ── Quick CLI test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True, help="Path to a rollout directory")
    parser.add_argument("--task", required=True, help="Task name (e.g. reach-v3)")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--strategy", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    frames, indices = sample_keyframes(rollout_dir / "frames", K=args.K, strategy=args.strategy)

    desc = TASK_DESCRIPTIONS.get(args.task, args.task)

    # Load ground truth
    with open(rollout_dir / "meta.json") as f:
        meta = json.load(f)

    print(f"Task: {args.task}")
    print(f"Ground truth failure_t: {meta['failure_timestep']} ({meta['failure_type']})")
    print(f"Keyframe indices: {indices}")
    print(f"Model: {args.model}")
    print()

    result = predict_failure(desc, frames, indices, meta["num_steps"], model=args.model)
    print("VLM prediction:")
    print(json.dumps(result, indent=2))

    if result["failure_timestep"] is not None:
        gt = meta["failure_timestep"]
        pred = result["failure_timestep"]
        print(f"\nAbsolute error: {abs(pred - gt)} timesteps")
