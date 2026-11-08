"""Curated model lineup, GPU-based recommendation, and downloads."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

import ollama

# Hugging Face models Ollama can pull directly: GGUF repos, referenced as
# hf.co/<user>/<repo>[:<quantization>], e.g. hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
_HF_REF = re.compile(r"^hf\.co/[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}(:[\w.-]{1,64})?$")

# fit_gb = rough total memory at an 8K context (weights + KV cache + buffers).
# llama3.1:8b was measured on an RTX 4070 Laptop (5.8 GB); the rest are
# estimates from model size and architecture. hardware.fit_check() reports
# the real number once a model is loaded.
CATALOG: list[dict] = [
    {
        "tag": "llama3.2:3b", "label": "Llama 3.2 3B", "size_gb": 2.0, "fit_gb": 3.2,
        "thinking": False, "license": "Llama 3.2 Community License",
        "blurb": "Small and quick. Good for older or smaller GPUs.",
    },
    {
        # The plain qwen3:4b tag is the thinking-only 2507 build (slow for agent
        # work); the instruct build answers directly and handles tools well.
        "tag": "qwen3:4b-instruct", "label": "Qwen3 4B", "size_gb": 2.5, "fit_gb": 4.0,
        "thinking": False, "license": "Apache 2.0",
        "blurb": "Strong for its size; the most reliable tool use in the lineup.",
    },
    {
        "tag": "llama3.1:8b", "label": "Llama 3.1 8B", "size_gb": 4.9, "fit_gb": 5.8,
        "thinking": False, "license": "Llama 3.1 Community License",
        "blurb": "Good prose for plain chat; less reliable with library tools.",
    },
    {
        "tag": "qwen3:8b", "label": "Qwen3 8B", "size_gb": 5.2, "fit_gb": 6.6,
        "thinking": True, "license": "Apache 2.0",
        "blurb": "Sharper reasoning; slightly too big for 8 GB at 8K context.",
    },
    {
        "tag": "gemma4:12b-it-qat", "label": "Gemma 4 12B", "size_gb": 7.2, "fit_gb": 8.8,
        "thinking": False, "license": "Gemma terms (see model page)",
        "blurb": "Polished prose. Needs a 12 GB GPU.",
    },
    {
        "tag": "qwen3:14b", "label": "Qwen3 14B", "size_gb": 9.3, "fit_gb": 10.9,
        "thinking": True, "license": "Apache 2.0",
        "blurb": "Best of the lineup. Needs a 12–16 GB GPU.",
    },
]

# GPU memory the driver/desktop keep for themselves (Ollama reported 6.5 GiB
# available of 8 GiB on the dev machine).
RESERVED_GB = 1.5
FALLBACK = "llama3.2:3b"


def by_tag(tag: str) -> dict | None:
    return next((m for m in CATALOG if m["tag"] == tag), None)


def budget_gb(vram_gb: float) -> float:
    return max(vram_gb - RESERVED_GB, 0.0)


def fits(model: dict, vram_gb: float) -> bool:
    return model["fit_gb"] <= budget_gb(vram_gb)


# Preference order for the library agent, best first. Based on testing the
# read -> save -> revise workflow: Qwen3 completed it reliably at every size;
# Llama 3.1 8B writes well but often fumbles multi-step tool calls (calls a
# tool without its arguments, or claims to have saved when it hasn't).
# Gemma 4's tool use hasn't been tested here yet, so tested models rank first.
AGENT_PREFERENCE = [
    "qwen3:14b", "qwen3:8b", "qwen3:4b-instruct", "gemma4:12b-it-qat", "llama3.1:8b", "llama3.2:3b",
]


def recommend(vram_gb: float) -> str:
    """Best model for the library agent that should run fully on this GPU."""
    for tag in AGENT_PREFERENCE:
        m = by_tag(tag)
        if m and fits(m, vram_gb):
            return tag
    return FALLBACK


def chat_options(tag: str) -> dict:
    """Extra ollama.chat() kwargs for a model's reasoning mode.

    * Hybrid Qwen3 (8b/14b, marked thinking=True): think=False turns
      reasoning off, which is much faster.
    * Other models that report a 'thinking' capability (e.g. thinking-only
      builds): think=True, so the reasoning arrives in its own field and the
      app can hide it instead of it leaking into the reply.
    * Everything else: no think argument at all.
    """
    m = by_tag(tag)
    if m:
        return {"think": False} if m["thinking"] else {}
    try:
        caps = ollama.show(tag).capabilities or []
    except Exception:
        return {}
    return {"think": True} if "thinking" in caps else {}


def _with_aliases(name: str) -> list[str]:
    # Ollama lists "hf.co/user/repo" as "hf.co/user/repo:latest"; treat both as the same model.
    return [name, name[: -len(":latest")]] if name.endswith(":latest") else [name]


def installed() -> set[str]:
    return set(installed_sizes())


@lru_cache(maxsize=64)
def capabilities(tag: str) -> tuple[str, ...]:
    """What Ollama says a downloaded model can do, e.g. ('completion', 'tools')."""
    return tuple(ollama.show(tag).capabilities or ())


def supports_tools(tag: str) -> bool:
    try:
        return "tools" in capabilities(tag)
    except Exception:
        return False


def template_supports_system(template: str) -> bool:
    """Does a chat template place a system message anywhere?

    Works for Ollama's Go templates ({{ .System }}, eq .Role "system") and for
    Jinja templates taken from GGUF files (messages[0].role == 'system').
    Templates that raise an error for a system role, or never mention it,
    count as unsupported: Ollama would drop the system message silently.
    Templates that fold the system text into the first user turn (Gemma)
    count as supported, because the text still reaches the model.
    """
    low = (template or "").lower()
    # Only errors that reject the system role itself count. Templates also raise
    # for narrower things ("System message cannot contain images", "System
    # message must be at the beginning") while supporting system prompts fine.
    if _REJECTS_SYSTEM.search(low):
        return False
    return ".system" in low or bool(re.search(r"""['"]system['"]""", low))


_REJECTS_SYSTEM = re.compile(
    r"""raise_exception\(\s*['"][^'"]*system[^'"]*(not supported|unsupported|not allowed|is not permitted)[^'"]*['"]"""
)


@lru_cache(maxsize=64)
def template_info(tag: str) -> dict:
    """{'system': bool, 'tools': bool} for a downloaded model, read from its chat template."""
    try:
        info = ollama.show(tag)
    except Exception:
        return {"system": True, "tools": False}
    return {
        "system": template_supports_system(info.template or ""),
        "tools": "tools" in (info.capabilities or []),
    }


def installed_sizes() -> dict[str, float]:
    """Downloaded models and their size in GB."""
    try:
        return {
            alias: round((m.size or 0) / 1024**3, 1)
            for m in ollama.list().models
            for alias in _with_aliases(m.model)
        }
    except Exception:
        return {}


def friendly_pull_error(tag: str, error: str) -> str:
    low = error.lower()
    if is_hf(tag) and any(k in low for k in ("manifest", "not found", "realm host", "401", "403", "404")):
        return ("Couldn't find a downloadable GGUF model at that address. Check the spelling, that the "
                "repo is public and contains .gguf files, and (if you added one) that the quantization "
                "tag exists, e.g. :Q4_K_M.")
    if "connect" in low or "timeout" in low:
        return "Couldn't reach the download server. Check your internet connection and try again."
    return error


def normalize_hf(ref: str) -> str | None:
    """Turn a pasted Hugging Face reference into hf.co/<user>/<repo>[:<quant>], or None.

    Accepts hf.co/..., huggingface.co/..., or a full https:// URL. Anything with
    extra path parts, odd characters or another host is rejected.
    """
    ref = str(ref or "").strip()
    ref = re.sub(r"^https?://", "", ref, flags=re.I)
    ref = re.sub(r"^(www\.)?huggingface\.co/", "hf.co/", ref, flags=re.I)
    ref = ref.rstrip("/")
    return ref if _HF_REF.match(ref) else None


def is_hf(tag: str) -> bool:
    return bool(_HF_REF.match(tag or ""))


def hf_label(tag: str) -> str:
    repo = tag.split("/", 2)[2]
    name, _, quant = repo.partition(":")
    name = re.sub(r"[-_.]?gguf$", "", name, flags=re.I)
    return f"{name} ({quant})" if quant else name


def delete(tag: str) -> None:
    ollama.delete(tag)
    capabilities.cache_clear()
    template_info.cache_clear()


def pull(tag: str, on_progress: Callable[[str, int, int], None]) -> None:
    """Download a model, reporting (status, completed_bytes, total_bytes)."""
    for p in ollama.pull(tag, stream=True):
        on_progress(p.status or "", p.completed or 0, p.total or 0)
