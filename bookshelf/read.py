"""Local Qwen3-VL title/author extraction from book crops."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PIL import Image

_model = None
_processor = None
_model_id: str | None = None

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
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


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_vlm(model_id: str = DEFAULT_MODEL):
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


def read_book(image: Image.Image, *, model_id: str = DEFAULT_MODEL) -> Reading:
    """Extract title and author from a book crop."""
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
    return Reading(title=title, author=author, raw=raw)
