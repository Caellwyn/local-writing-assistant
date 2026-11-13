"""Curated model lineup, GPU-based recommendation, and downloads."""

from __future__ import annotations

import os
import re
import sys
import threading
from functools import lru_cache
from typing import Callable

import ollama

# Hugging Face models Ollama can pull directly: GGUF repos, referenced as
# hf.co/<user>/<repo>[:<quantization>], e.g. hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
_HF_REF = re.compile(r"^hf\.co/[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}(:[\w.-]{1,64})?$")

# kv_elems = values the model keeps in its context (KV) cache per token:
#   (layers with a full KV cache) x (KV heads) x (key dim + value dim)
# from each model's architecture. Memory is estimated by estimate_fit_gb();
# hardware.fit_check() reports the real split once a model is loaded.
# fit_gb is only a fallback for models whose architecture isn't known.
CATALOG: list[dict] = [
    {
        "tag": "llama3.2:3b", "label": "Llama 3.2 3B", "size_gb": 2.0, "kv_elems": 28 * 8 * 256,
        "fit_gb": 3.2, "thinking": False, "license": "Llama 3.2 Community License",
        "blurb": "Small and quick. Good for older or smaller GPUs.",
    },
    {
        # The plain qwen3:4b tag is the thinking-only 2507 build (slow for agent
        # work); the instruct build answers directly and handles tools well.
        "tag": "qwen3:4b-instruct", "label": "Qwen3 4B", "size_gb": 2.5, "kv_elems": 36 * 8 * 256,
        "fit_gb": 4.0, "thinking": False, "license": "Apache 2.0",
        "blurb": "Strong for its size; the most reliable tool use in the lineup.",
    },
    {
        # Qwen3.5 is a hybrid: only every 4th layer keeps a context cache (8 x 4 heads x 512).
        # Reasoning is switched off for chat (thinking=True -> think=False), like Qwen3 8B/14B.
        "tag": "qwen3.5:4b", "label": "Qwen3.5 4B", "size_gb": 3.3, "kv_elems": 8 * 4 * 512,
        "vision_gb": 0.68, "fit_gb": 4.0, "thinking": True, "license": "Apache 2.0",
        "blurb": "Newer small Qwen with a small context cache. Not yet tested with the library tools.",
    },
    {
        "tag": "llama3.1:8b", "label": "Llama 3.1 8B", "size_gb": 4.9, "kv_elems": 32 * 8 * 256,
        "fit_gb": 5.8, "thinking": False, "license": "Llama 3.1 Community License",
        "blurb": "Good prose for plain chat; less reliable with library tools.",
    },
    {
        "tag": "qwen3:8b", "label": "Qwen3 8B", "size_gb": 5.2, "kv_elems": 36 * 8 * 256,
        "fit_gb": 6.6, "thinking": True, "license": "Apache 2.0",
        "blurb": "Sharper reasoning; needs more than 8 GB at 8K context.",
    },
    {
        "tag": "qwen3.5:9b", "label": "Qwen3.5 9B", "size_gb": 6.6, "kv_elems": 8 * 4 * 512,
        "vision_gb": 0.92, "fit_gb": 7.2, "thinking": True, "license": "Apache 2.0",
        "blurb": "Strong writer and the default research model. Fits 8 GB with Ollama's default 1 slot.",
    },
    {
        # Gemma uses sliding-window attention on most layers; its exact cache
        # size isn't modelled here, so the fallback fit_gb is used.
        "tag": "gemma4:12b-it-qat", "label": "Gemma 4 12B", "size_gb": 7.2, "kv_elems": None,
        "fit_gb": 8.8, "thinking": False, "license": "Gemma terms (see model page)",
        "blurb": "Polished prose. Needs a 12 GB GPU.",
    },
    {
        "tag": "qwen3:14b", "label": "Qwen3 14B", "size_gb": 9.3, "kv_elems": 40 * 8 * 256,
        "fit_gb": 10.9, "thinking": True, "license": "Apache 2.0",
        "blurb": "Best of the lineup. Needs a 16 GB GPU.",
    },
]

# GPU memory not available for model data: driver/desktop use plus Ollama's own
# working buffers. On the dev machine (8 GiB) Ollama reported 6.5 GiB available
# but placed at most ~5.9 GiB of a model on the GPU, hence 2.0.
RESERVED_GB = 2.0
FALLBACK = "llama3.2:3b"
DEFAULT_CTX = 8192
OVERHEAD_GB = 0.3  # compute buffers etc.; calibrated against `ollama ps` on the dev machine
# bytes per cached value for each OLLAMA_KV_CACHE_TYPE (q8_0/q4_0 store a scale per 32 values)
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


# -- Models this app has loaded ------------------------------------------------
# Ollama keeps a model in memory for minutes after its last request (keep_alive). Every place the
# app sends a request records the model here, so closing the app can free them straight away.
_used: set[str] = set()
_used_lock = threading.Lock()


def note_used(tag: str) -> None:
    with _used_lock:
        _used.add(tag)


def _full_tag(tag: str) -> str:
    """Ollama lists `name` as `name:latest`."""
    return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"


def unload(host: str | None = None, *, only_used: bool = True, gpu_only: bool = False,
           timeout: float = 5.0, tags: list[str] | None = None) -> list[str]:
    """Ask an Ollama server (default: the user's) to unload loaded models (keep_alive=0).

    only_used: just the models this app used, never another program's (False for the app's private
    research server, where every model is the app's). gpu_only: keep models running on the CPU (the
    embedder), which don't compete for GPU memory. tags: only these models. Only loaded models are asked: an unload request
    for one that isn't loaded would load it first. Returns the tags unloaded."""
    if only_used:
        with _used_lock:
            used = {_full_tag(t): t for t in _used}
        if not used:
            return []
    client = ollama.Client(host=host, timeout=timeout) if host else ollama.Client(timeout=timeout)
    try:
        loaded = client.ps().models
    except Exception:
        return []  # the server isn't running: nothing to free
    freed = []
    wanted = {_full_tag(t) for t in tags} if tags else None
    for m in loaded:
        if wanted is not None and m.model not in wanted:
            continue
        if only_used and m.model not in used:
            continue
        if gpu_only and not getattr(m, "size_vram", 0):
            continue
        tag = used[m.model] if only_used else m.model
        try:
            client.generate(model=tag, keep_alive=0)
            freed.append(tag)
        except Exception:
            pass
    return freed


def unload_used(timeout: float = 5.0) -> list[str]:
    """Unload every model this app used from the user's Ollama (when the app closes)."""
    return unload(timeout=timeout)


def by_tag(tag: str) -> dict | None:
    return next((m for m in CATALOG if m["tag"] == tag), None)


def budget_gb(vram_gb: float) -> float:
    return max(vram_gb - RESERVED_GB, 0.0)


# -- What the Ollama server is configured with --------------------------------

def _ollama_setting(name: str) -> str:
    """An Ollama server setting: this process's environment, else the user's saved one."""
    if os.environ.get(name):
        return os.environ[name]
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            pass
    return ""


def server_parallel() -> int:
    try:
        return max(int(_ollama_setting("OLLAMA_NUM_PARALLEL") or 1), 1)
    except ValueError:
        return 1


def kv_cache_type() -> str:
    t = _ollama_setting("OLLAMA_KV_CACHE_TYPE").lower() or "f16"
    return t if t in KV_BYTES else "f16"


# -- Memory estimate ----------------------------------------------------------

def kv_elems_from_info(info: dict) -> int | None:
    """Cached values per token, from GGUF model info (handles hybrid models)."""
    arch = info.get("general.architecture")
    if not arch:
        return None
    get = lambda key: info.get(f"{arch}.{key}")
    layers, kv_heads = get("block_count"), get("attention.head_count_kv")
    if isinstance(kv_heads, list):  # some models list KV heads per layer
        kv_heads = max(kv_heads) if kv_heads else None
    if not layers or not kv_heads:
        return None
    head_dim = None
    if get("embedding_length") and get("attention.head_count"):
        head_dim = get("embedding_length") // get("attention.head_count")
    k_dim, v_dim = get("attention.key_length") or head_dim, get("attention.value_length") or head_dim
    if not k_dim or not v_dim:
        return None
    interval = get("full_attention_interval")  # hybrid models: only every Nth layer has a KV cache
    kv_layers = layers // interval if interval else layers
    return int(kv_layers * kv_heads * (k_dim + v_dim))


@lru_cache(maxsize=64)
def _installed_kv_elems(tag: str) -> int | None:
    try:
        return kv_elems_from_info(dict(ollama.show(tag).modelinfo or {}))
    except Exception:
        return None


def estimate_fit_gb(
    weights_gb: float, kv_elems: int | None, num_ctx: int = DEFAULT_CTX,
    parallel: int | None = None, kv_type: str | None = None, fallback_gb: float | None = None,
) -> float:
    """Weights + context cache (x parallel slots) + overhead, in GiB."""
    if not kv_elems:
        return fallback_gb if fallback_gb is not None else weights_gb + 1.2
    parallel = parallel or server_parallel()
    kv_gb = kv_elems * KV_BYTES[kv_type or kv_cache_type()] * num_ctx * parallel / 1024**3
    return round(weights_gb + kv_gb + OVERHEAD_GB, 2)


def model_fit_gb(model: dict, num_ctx: int = DEFAULT_CTX) -> float:
    """Estimated memory for a catalog entry (or a downloaded model's entry).

    The image encoder of a vision model (vision_gb, part of the download) isn't counted: Ollama leaves it
    out when it decides placement (`ollama ps` showed Qwen3.5 9B at 5.24 GiB with it loaded), and the
    measured loads agree. On the 8 GB dev machine, at 8K and q8_0: Qwen3.5 9B ran 100% on the GPU with
    1 slot and 79% with 4; Qwen3 8B 100% with 1 and 76% with 4; Qwen3.5 4B 100% with 4."""
    weights = (model["size_gb"] - model.get("vision_gb", 0)) * 1e9 / 1024**3  # download sizes: decimal GB
    kv = model.get("kv_elems")
    if kv is None and model.get("installed"):
        kv = _installed_kv_elems(model["tag"])
    return estimate_fit_gb(weights, kv, num_ctx, fallback_gb=model.get("fit_gb"))


def installed_fit_gb(tag: str, size_gib: float, num_ctx: int = DEFAULT_CTX) -> float:
    """Estimated memory for a downloaded model, from its real architecture."""
    return estimate_fit_gb(size_gib, _installed_kv_elems(tag), num_ctx)


def fits(model: dict, vram_gb: float) -> bool:
    return model_fit_gb(model) <= budget_gb(vram_gb)


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


def can_think(tag: str) -> bool:
    """Whether a model can reason before it answers (catalog flag, else Ollama's 'thinking' capability)."""
    m = by_tag(tag)
    if m:
        return bool(m["thinking"])
    try:
        return "thinking" in capabilities(tag)
    except Exception:
        return False


def thinking_only(tag: str) -> bool:
    """Builds that always reason (e.g. qwen3:4b-thinking-2507): asking them not to leaks the reasoning
    into the reply instead."""
    return "thinking" in tag.lower() and can_think(tag)


def chat_options(tag: str, think: bool = False) -> dict:
    """Extra ollama.chat() kwargs for a model's reasoning mode.

    * Models that can reason: think=<the user's Thinking setting>. Off is much faster: the DavidAU
      Qwen3.5 9B took 16.7 s to its first visible word with reasoning on, 0.1 s with it off. With it on,
      the reasoning arrives in its own field, which the app shows collapsed above the reply.
    * Thinking-only builds: always think=True, so their reasoning stays out of the reply.
    * Everything else: no think argument at all.
    """
    if not can_think(tag):
        return {}
    return {"think": True if thinking_only(tag) else bool(think)}


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
    # Models with a built-in renderer (Ollama's own qwen3.5 builds) show a placeholder template,
    # {{ .Prompt }}; the renderer formats every role itself, system messages included.
    renderer = any(line.startswith("RENDERER ") for line in (info.modelfile or "").splitlines())
    return {
        "system": renderer or template_supports_system(info.template or ""),
        "tools": "tools" in (info.capabilities or []),
    }


def installed_sizes() -> dict[str, float]:
    """Downloaded models and their size in GB."""
    try:
        return {
            alias: round((m.size or 0) / 1024**3, 1)
            for m in ollama.list().models
            if not m.model.startswith("text-only/")  # the app's own text-only copies (runtime.py)
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
    _installed_kv_elems.cache_clear()


def pull(tag: str, on_progress: Callable[[str, int, int], None]) -> None:
    """Download a model, reporting (status, completed_bytes, total_bytes)."""
    for p in ollama.pull(tag, stream=True):
        on_progress(p.status or "", p.completed or 0, p.total or 0)
