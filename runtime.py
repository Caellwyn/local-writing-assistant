"""How the app runs a model in Ollama: text-only copies, and all layers on the GPU when they fit.

The app only works with text. Models that can also read images (Qwen3.5, many community builds) come
with an image encoder that Ollama loads onto the GPU on every load: ~1 GiB for a 9B that this app never
uses. For each such model the app makes a *text-only copy* once: a new Ollama entry made from the same
weights file without the encoder (`ollama create` reuses the file, so no disk is used). It lives under
the `text-only/` namespace and is hidden from the model lists; the app sends requests to it in place of
the original. The weights are identical; answers can differ in the last digits of rounding only because
more of the model runs on the GPU.

Ollama also leaves a safety margin when it places layers, which can stop a model that fits a couple of
layers short of the GPU (the DavidAU Qwen3.5 9B text copy: 31 of 33 layers). When the app's estimate,
read from the weights file itself, leaves at least FORCE_SPARE_GB free, it asks Ollama for every layer
(num_gpu). After the first load it checks the result; if the model didn't land fully on the GPU, or the
driver had to borrow system memory, it reloads with Ollama's own placement and stops forcing that model.

Forcing applies to the user's own Ollama (1 slot: chat and the researcher), not to the research
server's 4-slot finders (research_server.py).
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import ollama

import hardware
import models

log = logging.getLogger("app")

COPY_NAMESPACE = "text-only"
# Measured on the 8 GB dev machine: Ollama sees 6.9 of 8 GiB free on an idle GPU (Windows and the
# driver keep the rest); the CUDA context and scratch take ~0.15 GiB besides the model's own buffers.
DRIVER_RESERVED_GB = 1.1
CUDA_CONTEXT_GB = 0.15
FORCE_SPARE_GB = 0.5  # force all layers only with at least this much left over
RUNTIME_OVERHEAD_GB = 0.12  # compute buffers at 1 slot (Ollama's log: 112 MiB for a Qwen3.5 9B)
RECURRENT_GB = 0.05  # per slot, hybrid models' recurrent state (log: 50 MiB for a Qwen3.5 9B)
ALL_LAYERS = 999
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_lock = threading.Lock()
_copies: dict[str, str] = {}  # original tag -> tag to run (the copy, or the original)
_no_force: set[str] = set()  # runtime tags whose forced load didn't work out
_confirmed: set[str] = set()  # runtime tags whose forced load was checked


def is_copy(tag: str) -> bool:
    return tag.startswith(COPY_NAMESPACE + "/")


def copy_name(tag: str) -> str:
    """A valid, stable Ollama name for the text-only copy of `tag`."""
    last = tag.rsplit("/", 1)[-1]
    name, _, version = last.partition(":")
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")[:48] or "model"
    version = re.sub(r"[^a-z0-9._-]+", "-", (version or "latest").lower())[:40]
    digest = hashlib.sha1(tag.encode()).hexdigest()[:6]
    return f"{COPY_NAMESPACE}/{slug}-{digest}:{version}"


# -- Reading GGUF headers (local weights files) -------------------------------------------------

# ggml tensor type -> (values per block, bytes per block)
_GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2)}
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@lru_cache(maxsize=32)
def gguf_summary(path: str) -> dict | None:
    """{'arch', 'bytes', 'cpu_bytes'} from a GGUF file's header: total tensor bytes, and the bytes
    llama.cpp keeps in system memory however many layers are offloaded (the input token embeddings).
    None if the file can't be read."""
    try:
        with open(path, "rb") as f:
            def read(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

            def string():
                return f.read(read("Q")).decode("utf-8", "replace")

            if f.read(4) != b"GGUF":
                return None
            read("I")
            n_tensors, n_kv = read("Q"), read("Q")
            arch = None
            for _ in range(n_kv):
                key, vtype = string(), read("I")
                if vtype == 8:
                    value = string()
                    if key == "general.architecture":
                        arch = value
                elif vtype == 9:
                    etype, n = read("I"), read("Q")
                    if etype == 8:
                        for _ in range(n):
                            f.seek(read("Q"), 1)
                    else:
                        f.seek(struct.calcsize("<" + _SCALAR[etype]) * n, 1)
                else:
                    read(_SCALAR[vtype])
            total = cpu = 0
            for _ in range(n_tensors):
                name = string()
                dims = [read("Q") for _ in range(read("I"))]
                ttype = read("I")
                read("Q")
                per_block, block_bytes = _GGML.get(ttype, (1, 2))
                count = 1
                for d in dims:
                    count *= d
                size = count // per_block * block_bytes
                total += size
                if name.startswith("token_embd"):
                    cpu += size
            return {"arch": arch, "bytes": total, "cpu_bytes": cpu}
    except (OSError, struct.error, KeyError, ValueError):
        return None


def _from_paths(modelfile: str) -> list[tuple[str, str]]:
    """(line, path) for each FROM line that names a local file."""
    out = []
    for line in modelfile.splitlines():
        if line.startswith("FROM "):
            path = line[5:].strip().strip('"')
            if Path(path).is_file():
                out.append((line, path))
    return out


# -- Text-only copies --------------------------------------------------------------------------

def text_only(tag: str) -> str:
    """The tag to run for `tag`: its text-only copy (made on first use) when it has an image encoder
    in a separate file, otherwise `tag` itself. Never raises: any problem means running `tag`."""
    if is_copy(tag):
        return tag
    with _lock:
        if tag in _copies:
            return _copies[tag]
        run = tag
        try:
            run = _ensure_copy(tag)
        except Exception as e:
            log.warning("text-only copy of %s not made (%s); using the original", tag, e)
        _copies[tag] = run
        return run


def _ensure_copy(tag: str) -> str:
    info = ollama.show(tag)
    if "vision" not in (info.capabilities or []):
        return tag
    modelfile = info.modelfile or ""
    files = _from_paths(modelfile)
    encoders = [line for line, path in files if (gguf_summary(path) or {}).get("arch") == "clip"]
    weights = [path for line, path in files if line not in encoders]
    if not encoders or len(weights) != 1:  # encoder inside the weights file, or a remote Ollama
        return tag
    name = copy_name(tag)
    try:  # an existing copy is reused if it was made from the same weights file
        if [p for _, p in _from_paths(ollama.show(name).modelfile or "")] == weights:
            return name
    except ollama.ResponseError:
        pass
    text = "\n".join(line for line in modelfile.splitlines() if line not in encoders) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(text, encoding="utf-8")
        exe = _ollama_exe()
        if not exe:
            raise RuntimeError("ollama executable not found")
        r = subprocess.run([exe, "create", name, "-f", str(path)], capture_output=True, encoding="utf-8",
                           errors="replace", timeout=300, creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    log.info("made text-only copy %s of %s", name, tag)
    return name


def _ollama_exe() -> str | None:
    import research_server  # local import: research_server imports models
    return research_server.ollama_exe()


def running_tag(tag: str) -> str:
    """The tag requests for `tag` go to, if already known (never makes a copy)."""
    return _copies.get(tag, tag)


def delete_copy(tag: str) -> None:
    """Remove the text-only copy of `tag`, if any (when the original is deleted)."""
    with _lock:
        _copies.pop(tag, None)
    try:
        ollama.delete(copy_name(tag))
    except Exception:
        pass


# -- All layers on the GPU ---------------------------------------------------------------------

def gpu_need_gb(run_tag: str, num_ctx: int = models.DEFAULT_CTX, parallel: int = 1) -> float | None:
    """GiB a fully offloaded model needs on the GPU, as the app runs it (text-only): its weights except
    the token embeddings, the context cache (from the model's architecture), recurrent state, and
    compute buffers. Works for an original or its copy: image-encoder files are left out."""
    try:
        info = ollama.show(run_tag)
    except Exception:
        return None
    files = _from_paths(info.modelfile or "")
    sums = [s for s in (gguf_summary(p) for _, p in files) if s and s["arch"] != "clip"]
    if not sums:
        return None
    weights = sum(s["bytes"] - s["cpu_bytes"] for s in sums) / 1024**3
    modelinfo = dict(info.modelinfo or {})
    kv = models.kv_elems_from_info(modelinfo) or 0
    cache = kv * models.KV_BYTES[models.kv_cache_type()] * num_ctx * parallel / 1024**3
    arch = modelinfo.get("general.architecture", "")
    recurrent = RECURRENT_GB * parallel if modelinfo.get(f"{arch}.full_attention_interval") else 0.0
    return round(weights + cache + recurrent + RUNTIME_OVERHEAD_GB, 2)


def spare_gb(need: float, vram_gb: float) -> float:
    return round(vram_gb - DRIVER_RESERVED_GB - CUDA_CONTEXT_GB - need, 2)


@lru_cache(maxsize=32)
def _force(run_tag: str, num_ctx: int, vram_gb: float) -> bool:
    need = gpu_need_gb(run_tag, num_ctx)
    ok = need is not None and spare_gb(need, vram_gb) >= FORCE_SPARE_GB
    log.info("%s: needs ~%s GiB on the GPU; all layers forced: %s", run_tag, need, ok)
    return ok


@lru_cache(maxsize=64)
def fits(tag: str, vram_gb: float, parallel: int = 1, num_ctx: int = models.DEFAULT_CTX) -> bool | None:
    """Will the app run this downloaded model fully on the GPU (text-only, all layers forced)?
    None when that can't be worked out from its weights file (then use models.fits)."""
    gpu = hardware.detect_gpu()
    if gpu.get("backend") != "CUDA":
        return None
    need = gpu_need_gb(tag, num_ctx, parallel)
    return None if need is None else spare_gb(need, vram_gb) >= FORCE_SPARE_GB


def resolve(tag: str, num_ctx: int, force_ok: bool = True) -> tuple[str, dict]:
    """(tag to send, extra Ollama options) for a request to `tag`. Use the same answer for every
    request to a model: Ollama reloads a model whenever num_gpu or num_ctx changes."""
    run = text_only(tag)
    if not force_ok or run in _no_force:
        return run, {}
    gpu = hardware.detect_gpu()
    vram = gpu.get("vram_gb") or 0.0
    if gpu.get("backend") != "CUDA" or not vram:
        return run, {}  # the driver-reserve figures were measured with NVIDIA on Windows
    return (run, {"num_gpu": ALL_LAYERS}) if _force(run, num_ctx, vram) else (run, {})


def confirm(run_tag: str, options: dict) -> bool:
    """After the first forced load of a model: is it fully on the GPU, without the driver borrowing
    system memory? If not, unload it and stop forcing it (the next request uses Ollama's placement).
    Returns False when it fell back."""
    if options.get("num_gpu") != ALL_LAYERS or run_tag in _confirmed:
        return True
    _confirmed.add(run_tag)
    fit = hardware.fit_check(run_tag)
    used = hardware.gpu_used_gb()
    vram = hardware.detect_gpu().get("vram_gb") or 0.0
    borrowed = used is not None and vram and used > vram - DRIVER_RESERVED_GB + 0.05
    if fit and fit.get("gpu_pct", 0) >= 99 and not borrowed:
        return True
    log.warning("%s didn't fit with all layers forced (fit %s, GPU used %s GiB); using Ollama's placement",
                run_tag, fit, used)
    _no_force.add(run_tag)
    models.unload(None, only_used=False, tags=[run_tag])
    return False
