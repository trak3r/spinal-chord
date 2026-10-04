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


class OpenRouterRetryError(RuntimeError):
    """Raised after exhausting retries on transient rate limits or network errors."""


class OpenRouterQuotaError(RuntimeError):
    """Account/daily quota or credits exhausted — do not retry."""


# Back-compat alias
OpenRouterRateLimitError = OpenRouterRetryError


def _retry_wait(attempt: int, retry_after: str | None = None) -> float:
    """Seconds to sleep before the next try (caps at 60s)."""
    if retry_after:
        try:
            return min(60.0, max(1.0, float(retry_after)))
        except ValueError:
            pass
    # 5, 10, 20, 40, 60, 60, …
    return min(60.0, 5.0 * (2 ** attempt))


def _error_blob(data: dict | None) -> tuple[dict, str, dict]:
    """Return (error_dict, message_lower, metadata)."""
    if not data or not isinstance(data, dict):
        return {}, "", {}
    err = data.get("error")
    if not isinstance(err, dict):
        return {}, str(err or "").lower(), {}
    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    return err, str(err.get("message") or "").lower(), meta


def _quota_exhausted_reason(
    resp: requests.Response, data: dict | None
) -> str | None:
    """
    Hard stop: daily free-model quota, or insufficient credits.

    Upstream provider overload / shared free pool → None (retryable).
    """
    if resp.status_code == 402:
        err, msg, _ = _error_blob(data)
        return msg or "insufficient credits (HTTP 402)"

    err, msg, meta = _error_blob(data)
    if resp.status_code != 429 and err.get("code") != 429:
        return None

    limit_source = str(meta.get("limit_source") or "").lower()
    # OpenRouter account/platform daily free-model cap.
    if (
        "free-models-per-day" in msg
        or "free-models-per-day" in limit_source
        or "per-day" in limit_source
        or "daily" in limit_source
        or ("per day" in msg and "rate limit" in msg)
    ):
        return err.get("message") or "daily free-model quota exceeded"

    # Credits / spend caps sometimes surface as 429 with these sources.
    if "credit" in limit_source or "spend" in limit_source:
        return err.get("message") or "OpenRouter credit/spend limit exceeded"

    return None


def _is_transient_rate_limit(resp: requests.Response, data: dict | None) -> bool:
    """Provider overload / per-minute soft limits — worth sleeping and retrying."""
    if _quota_exhausted_reason(resp, data):
        return False
    if resp.status_code == 429:
        return True
    err, msg, meta = _error_blob(data)
    if err.get("code") == 429:
        return True
    limit_source = str(meta.get("limit_source") or "").lower()
    if "upstream" in limit_source or "shared_pool" in limit_source:
        return True
    if "per-min" in limit_source or "per_minute" in limit_source:
        return True
    if "rate" in msg or "temporarily rate-limited" in msg:
        return True
    return False


def _is_retryable_status(status: int) -> bool:
    # 429 handled separately (quota vs transient).
    return status in {408, 425, 500, 502, 503, 504}


def _read_openrouter(
    image: Image.Image,
    *,
    model: str = OPENROUTER_MODEL,
    timeout: float = 90.0,
    retries: int = 12,
) -> Reading:
    import sys
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
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            )
        except requests.RequestException as e:
            # SSLError, ConnectionError, Timeout, etc.
            last_err = f"{type(e).__name__}: {e}"[:300]
            wait = _retry_wait(attempt)
            print(
                f"OpenRouter network error ({type(e).__name__}); "
                f"sleeping {wait:.0f}s (try {attempt + 1}/{retries})…",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        data: dict | None
        try:
            data = resp.json() if resp.content else None
        except ValueError:
            data = None

        data_dict = data if isinstance(data, dict) else None
        quota = _quota_exhausted_reason(resp, data_dict)
        if quota:
            raise OpenRouterQuotaError(
                f"OpenRouter quota/credits exhausted — not retrying. {quota}"
            )

        if _is_transient_rate_limit(resp, data_dict):
            last_err = (resp.text or str(data) or "rate limited")[:300]
            wait = _retry_wait(attempt, resp.headers.get("Retry-After"))
            _, msg, meta = _error_blob(data_dict)
            src = meta.get("limit_source") or meta.get("provider_name") or "provider"
            print(
                f"OpenRouter busy ({src}); sleeping {wait:.0f}s "
                f"(try {attempt + 1}/{retries})…",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        if _is_retryable_status(resp.status_code):
            last_err = (resp.text or f"HTTP {resp.status_code}")[:300]
            wait = _retry_wait(attempt, resp.headers.get("Retry-After"))
            print(
                f"OpenRouter HTTP {resp.status_code}; sleeping {wait:.0f}s "
                f"(try {attempt + 1}/{retries})…",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        if not resp.ok:
            raise RuntimeError(
                f"OpenRouter error {resp.status_code}: {resp.text[:400]}"
            )
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected OpenRouter response: {resp.text[:400]}")
        if data.get("error"):
            raise RuntimeError(f"OpenRouter error: {data['error']}")

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

    raise OpenRouterRetryError(
        f"OpenRouter still failing after {retries} tries. "
        f"Wait a bit and re-run, or use --reader local. Last error: {last_err}"
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
