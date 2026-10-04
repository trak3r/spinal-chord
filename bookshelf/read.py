"""Title/author extraction from book crops (OpenRouter or local VLM)."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass

import requests
from PIL import Image

_model = None
_processor = None
_model_id: str | None = None

LOCAL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
OPENROUTER_MODEL = "qwen/qwen3.8-27b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

READ_PROMPT = (
    "This image is a single book spine or cover. "
    "Read the title and author. Reply with ONLY a JSON object: "
    '{"title": "...", "author": "..."}. '
    "Use empty strings if you cannot read them. No markdown, no extra text."
)


@dataclass
class Reading:
    title: str
    author: str
    raw: str
    backend: str  # "openrouter" | "local"


def reader_backend(prefer: str = "auto") -> str:
    """Resolve which reader to use: openrouter | local."""
    if prefer == "local":
        return "local"
    if prefer == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (required for --reader openrouter)"
            )
        return "openrouter"
    # auto
    return "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "local"


def _parse_json(text: str) -> tuple[str, str]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
        return str(data.get("title") or "").strip(), str(data.get("author") or "").strip()
    except json.JSONDecodeError:
        return "", ""


def _image_data_url(image: Image.Image, max_side: int = 1280) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _read_openrouter(
    image: Image.Image,
    *,
    model: str = OPENROUTER_MODEL,
    timeout: float = 90.0,
    retries: int = 4,
) -> Reading:
    import time

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": READ_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    },
                ],
            }
        ],
        "temperature": 0,
        # Free Qwen endpoints may spend tokens on hidden reasoning; keep headroom.
        "max_tokens": 1024,
        "reasoning": {"effort": "none"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/trak3r/spinal-chord",
        "X-Title": "spinal-chord",
    }

    last_err = ""
    for attempt in range(retries):
        resp = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
        )
        if resp.status_code == 429:
            last_err = resp.text[:400]
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            raise RuntimeError(
                f"OpenRouter error {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            if code == 429 or (isinstance(err, dict) and "rate" in str(err).lower()):
                last_err = str(err)[:400]
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenRouter error: {err}")
        try:
            raw = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected OpenRouter response: {data!r}") from e
        if isinstance(raw, list):
            raw = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw
            )
        title, author = _parse_json(str(raw))
        return Reading(
            title=title, author=author, raw=str(raw), backend="openrouter"
        )

    raise RuntimeError(
        f"OpenRouter rate-limited after {retries} tries: {last_err}"
    )


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_vlm(model_id: str = LOCAL_MODEL):
    global _model, _processor, _model_id
    if _model is not None and _model_id == model_id:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = _device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        _model = _model.to(device)
    _model.eval()
    _model_id = model_id
    return _model, _processor


def _read_local(image: Image.Image, *, model_id: str = LOCAL_MODEL) -> Reading:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _get_vlm(model_id)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": READ_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    in_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, in_len:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    title, author = _parse_json(raw)
    return Reading(title=title, author=author, raw=raw, backend="local")


def read_book(
    image: Image.Image,
    *,
    reader: str = "auto",
    model: str | None = None,
) -> Reading:
    """Extract title and author from a book crop."""
    backend = reader_backend(reader)
    if backend == "openrouter":
        return _read_openrouter(image, model=model or OPENROUTER_MODEL)
    return _read_local(image, model_id=model or LOCAL_MODEL)
