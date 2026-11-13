"""Detect the GPU Ollama can use, and check how a loaded model actually fits.

Detection gives an *estimate* for picking a model. The ground truth comes
from Ollama itself after loading: /api/ps reports `size` (total memory) and
`size_vram` (the part on the GPU). If they differ, the model spilled to CPU.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import lru_cache

import httpx

OLLAMA_URL = "http://localhost:11434"
# Windows display-adapter device class
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _nvidia() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, _, mib = line.rpartition(",")
        try:
            gpus.append({"name": name.strip(), "vram_gb": round(int(mib) / 1024, 1), "backend": "CUDA"})
        except ValueError:
            continue
    return max(gpus, key=lambda g: g["vram_gb"]) if gpus else None


def gpu_clocks() -> dict | None:
    """Current NVIDIA core clock, its maximum, power draw and utilization (for benchmarks: a laptop
    GPU can be power-capped far below its normal clock, which makes timings meaningless)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=clocks.sm,clocks.max.sm,power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW,
        ).stdout
        sm, max_sm, watts, util = (float(x) for x in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"sm_mhz": sm, "max_sm_mhz": max_sm, "watts": watts, "util": util}


def gpu_used_gb() -> float | None:
    """GPU memory in use right now (NVIDIA), in GiB."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, creationflags=_NO_WINDOW).stdout
        return round(float(out.strip().splitlines()[0]) / 1024, 2)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _is_discrete_amd(name: str) -> bool:
    # Integrated Ryzen graphics report shared system RAM; skip them.
    n = name.upper()
    return ("AMD" in n or "RADEON" in n) and any(k in n for k in (" RX ", "RX ", "PRO W", "PRO V", "INSTINCT", "AI PRO"))


def _amd_from_registry() -> dict | None:
    if sys.platform != "win32":
        return None
    import winreg

    best = None
    try:
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return None
    with cls:
        for i in range(64):
            try:
                sub = winreg.EnumKey(cls, i)
            except OSError:
                break
            try:
                with winreg.OpenKey(cls, sub) as k:
                    name = winreg.QueryValueEx(k, "DriverDesc")[0]
                    raw = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
            except OSError:
                continue
            if not _is_discrete_amd(name):
                continue
            size = int.from_bytes(raw, "little") if isinstance(raw, bytes) else int(raw)
            gpu = {"name": name, "vram_gb": round(size / 1024**3, 1), "backend": "ROCm"}
            if not best or gpu["vram_gb"] > best["vram_gb"]:
                best = gpu
    return best


@lru_cache(maxsize=1)
def detect_gpu() -> dict:
    """Best GPU for Ollama: {name, vram_gb, backend}. backend 'CPU' if none.

    Intel integrated graphics is ignored on purpose: it reports shared system
    memory (18 GB on the dev machine) and isn't a real dedicated-VRAM GPU.
    """
    return _nvidia() or _amd_from_registry() or {"name": "No supported GPU", "vram_gb": 0.0, "backend": "CPU"}


def fit_check(model: str, base_url: str | None = None) -> dict | None:
    """How a loaded model is split between GPU and CPU, per Ollama's /api/ps (base_url: another server,
    e.g. the research server)."""
    try:
        data = httpx.get(f"{base_url or OLLAMA_URL}/api/ps", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return None
    for m in data.get("models", []):
        if model in (m.get("name"), m.get("model")):
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            gpu_pct = round(100 * vram / size) if size else 0
            label = "100% GPU" if gpu_pct >= 99 else ("CPU only" if gpu_pct == 0 else f"{100 - gpu_pct}% on CPU")
            return {
                "size_gb": round(size / 1024**3, 1),
                "gpu_pct": gpu_pct,
                "context": m.get("context_length"),
                "label": label,
            }
    return None
