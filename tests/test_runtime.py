"""Tests for text-only copies and forced GPU placement (runtime.py). No Ollama, no GPU."""

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa: E402
import runtime  # noqa: E402

pytestmark = pytest.mark.real_runtime


def _gguf(path: Path, arch: str, tensors: list[tuple[str, list[int], int]]) -> str:
    """A minimal GGUF v3 file: general.architecture, a string array, tensor infos (no tensor data)."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    out += s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("bb")
    for name, dims, ttype in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ttype, 0)
    path.write_bytes(out)
    return str(path)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(runtime, "_copies", {})
    monkeypatch.setattr(runtime, "_no_force", set())
    monkeypatch.setattr(runtime, "_confirmed", set())
    runtime.gguf_summary.cache_clear()
    runtime._force.cache_clear()


def test_copy_names_are_valid_stable_and_distinct():
    a = runtime.copy_name("hf.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-GGUF:Q4_K_M")
    assert a == runtime.copy_name("hf.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-GGUF:Q4_K_M")
    assert a.startswith("text-only/qwen3.5-9b-the-defiant-fable-gguf-") and a.endswith(":q4_k_m")
    assert runtime.copy_name("qwen3.5:9b") != runtime.copy_name("other/qwen3.5:9b")
    assert runtime.copy_name("mymodel").endswith(":latest") and runtime.is_copy(a)


def test_gguf_summary_counts_weights_and_the_cpu_embeddings(tmp_path):
    path = _gguf(tmp_path / "m.gguf", "qwen35", [
        ("token_embd.weight", [256, 4], 12),  # Q4_K: 4 blocks x 144 bytes, kept in system memory
        ("blk.0.attn_q.weight", [256, 8], 14),  # Q6_K: 8 blocks x 210 bytes
        ("output.weight", [256, 2], 30),  # BF16: 512 values x 2 bytes
    ])
    assert runtime.gguf_summary(path) == {"arch": "qwen35", "bytes": 576 + 1680 + 1024, "cpu_bytes": 576}
    assert runtime.gguf_summary(str(tmp_path / "missing.gguf")) is None


@pytest.fixture
def vision_model(tmp_path, monkeypatch):
    weights = _gguf(tmp_path / "weights.gguf", "qwen35", [("blk.0.w", [256], 12)])
    encoder = _gguf(tmp_path / "encoder.gguf", "clip", [("v.blk.0.w", [256], 1)])
    shows = {
        "vis:9b": SimpleNamespace(capabilities=["completion", "vision", "tools"],
                                  modelfile=f"FROM {weights}\nFROM {encoder}\nTEMPLATE {{{{ .Prompt }}}}\n"
                                            "PARAMETER temperature 1\n"),
        "text:4b": SimpleNamespace(capabilities=["completion", "tools"], modelfile=f"FROM {weights}\n"),
    }
    created = []

    def show(tag):
        if tag in shows:
            return shows[tag]
        raise runtime.ollama.ResponseError("not found", 404)

    def run(cmd, **kw):
        modelfile = Path(cmd[cmd.index("-f") + 1]).read_text(encoding="utf-8")
        created.append((cmd[2], modelfile))
        shows[cmd[2]] = SimpleNamespace(capabilities=["completion", "tools"], modelfile=modelfile)
        return SimpleNamespace(returncode=0, stdout="success", stderr="")

    monkeypatch.setattr(runtime.ollama, "show", show)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime, "_ollama_exe", lambda: "ollama")
    return SimpleNamespace(weights=weights, encoder=encoder, created=created, shows=shows)


def test_vision_models_get_a_text_only_copy_once(vision_model):
    name = runtime.text_only("vis:9b")
    assert name == runtime.copy_name("vis:9b")
    (_, modelfile), = vision_model.created
    assert f"FROM {vision_model.weights}" in modelfile and vision_model.encoder not in modelfile
    assert "TEMPLATE {{ .Prompt }}" in modelfile and "PARAMETER temperature 1" in modelfile
    runtime._copies.clear()  # a new session finds the copy and reuses it
    assert runtime.text_only("vis:9b") == name and len(vision_model.created) == 1
    assert runtime.text_only("text:4b") == "text:4b"  # no image encoder: run as is


def test_a_copy_of_older_weights_is_made_again(vision_model):
    name = runtime.copy_name("vis:9b")
    vision_model.shows[name] = SimpleNamespace(capabilities=[], modelfile=f"FROM {vision_model.encoder}\n")
    assert runtime.text_only("vis:9b") == name and len(vision_model.created) == 1


def test_no_copy_when_it_cant_be_made(vision_model, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="disk full"))
    assert runtime.text_only("vis:9b") == "vis:9b"  # falls back to the original
    vision_model.shows["single:1b"] = SimpleNamespace(capabilities=["vision"],
                                                      modelfile=f"FROM {vision_model.weights}\n")
    assert runtime.text_only("single:1b") == "single:1b"  # encoder inside the weights file
    vision_model.shows["remote:1b"] = SimpleNamespace(capabilities=["vision"], modelfile="FROM /not/here.gguf\n")
    assert runtime.text_only("remote:1b") == "remote:1b"


@pytest.fixture
def gpu(monkeypatch):
    state = {"gpu": {"name": "RTX", "vram_gb": 8.0, "backend": "CUDA"}, "need": 6.11}
    monkeypatch.setattr(runtime.hardware, "detect_gpu", lambda: state["gpu"])
    monkeypatch.setattr(runtime, "gpu_need_gb", lambda tag, num_ctx=8192, parallel=1: state["need"])
    monkeypatch.setattr(runtime, "text_only", lambda tag: tag)
    return state


def test_all_layers_forced_only_with_room_to_spare(gpu):
    # measured: the DavidAU 9B text copy needs ~6.11 GiB and ran 100% on the GPU when forced
    assert runtime.resolve("big", 8192) == ("big", {"num_gpu": runtime.ALL_LAYERS})
    assert runtime.resolve("big", 8192, force_ok=False) == ("big", {})  # e.g. the 4-slot research server
    gpu["need"] = 6.4  # less than FORCE_SPARE_GB left over
    assert runtime.resolve("bigger", 8192) == ("bigger", {})
    gpu["gpu"] = {"name": "Radeon", "vram_gb": 8.0, "backend": "ROCm"}
    gpu["need"] = 3.0
    assert runtime.resolve("small", 8192) == ("small", {})  # reserve figures were measured on NVIDIA


@pytest.mark.parametrize("fit,used,keeps_forcing", [
    ({"gpu_pct": 100}, 6.25, True),  # fully on the GPU
    ({"gpu_pct": 87}, 6.0, False),  # Ollama still left layers on the CPU
    ({"gpu_pct": 100}, 7.2, False),  # the driver borrowed system memory
])
def test_confirm_falls_back_to_ollamas_placement(gpu, monkeypatch, fit, used, keeps_forcing):
    unloaded = []
    monkeypatch.setattr(runtime.hardware, "fit_check", lambda tag: fit)
    monkeypatch.setattr(runtime.hardware, "gpu_used_gb", lambda: used)
    monkeypatch.setattr(runtime.models, "unload", lambda host=None, **kw: unloaded.append(kw.get("tags")))
    run, opts = runtime.resolve("big", 8192)
    assert runtime.confirm(run, opts) is keeps_forcing
    assert runtime.resolve("big", 8192)[1] == ({"num_gpu": runtime.ALL_LAYERS} if keeps_forcing else {})
    assert unloaded == ([] if keeps_forcing else [["big"]])
    assert runtime.confirm(run, opts)  # checked once per session


def test_model_lists_hide_text_only_copies(monkeypatch):
    listing = SimpleNamespace(models=[SimpleNamespace(model="qwen3.5:9b", size=6 * 2**30),
                                      SimpleNamespace(model="text-only/qwen3.5-abc123:9b", size=5 * 2**30)])
    monkeypatch.setattr(models.ollama, "list", lambda: listing)
    assert set(models.installed_sizes()) == {"qwen3.5:9b"}


def test_fits_reads_the_weights_file_without_the_image_encoder(gpu, monkeypatch):
    monkeypatch.setattr(runtime, "gpu_need_gb", lambda tag, num_ctx=8192, parallel=1: {1: 6.12, 4: 6.8}[parallel])
    runtime.fits.cache_clear()
    assert runtime.fits("davidau", 8.0, 1) is True and runtime.fits("davidau", 8.0, 4) is False
    gpu["gpu"] = {"name": "Radeon", "vram_gb": 8.0, "backend": "ROCm"}
    runtime.fits.cache_clear()
    assert runtime.fits("davidau", 8.0, 1) is None  # unknown: the app's own estimate decides


def test_gpu_need_leaves_out_image_encoder_files(tmp_path, monkeypatch):
    weights = _gguf(tmp_path / "w.gguf", "qwen35", [("token_embd.weight", [256, 4], 12), ("blk.0.w", [256, 1024], 12)])
    encoder = _gguf(tmp_path / "e.gguf", "clip", [("v.w", [256, 4096], 1)])
    info = SimpleNamespace(modelfile=f"FROM {weights}\nFROM {encoder}\n", modelinfo={
        "general.architecture": "qwen35", "qwen35.block_count": 32, "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
        "qwen35.full_attention_interval": 4})
    monkeypatch.setattr(runtime.ollama, "show", lambda tag: info)
    monkeypatch.setattr(runtime.models, "kv_cache_type", lambda: "q8_0")
    need = runtime.gpu_need_gb("vis", 8192, 1)
    expected = 1024 * 144 / 2**30 + 16384 * 34 / 32 * 8192 / 2**30 + runtime.RECURRENT_GB + runtime.RUNTIME_OVERHEAD_GB
    assert need == round(expected, 2)  # no encoder, no CPU-side token embeddings
