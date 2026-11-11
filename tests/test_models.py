"""Tests for model catalog logic and Hugging Face reference validation (no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa: E402


def _server(monkeypatch, parallel, kv_type):
    monkeypatch.setattr(models, "server_parallel", lambda: parallel)
    monkeypatch.setattr(models, "kv_cache_type", lambda: kv_type)


@pytest.mark.parametrize("vram,expected", [
    (0, "llama3.2:3b"), (4, "llama3.2:3b"), (6, "qwen3:4b-instruct"),
    (8, "qwen3:4b-instruct"), (12, "qwen3:8b"), (16, "qwen3:14b"),
])
def test_recommend_by_vram_default_server(monkeypatch, vram, expected):
    _server(monkeypatch, 1, "f16")  # Ollama defaults
    assert models.recommend(vram) == expected


@pytest.mark.parametrize("vram,expected", [
    (6, "llama3.2:3b"),  # 4 slots no longer leave room for the 4B on a 6 GB card
    (8, "qwen3:4b-instruct"), (12, "qwen3:8b"), (16, "qwen3:14b"),
])
def test_recommend_by_vram_four_slots_q8(monkeypatch, vram, expected):
    _server(monkeypatch, 4, "q8_0")
    assert models.recommend(vram) == expected


# Measured with `ollama ps` on an RTX 4070 Laptop (8 GB) at 8K context.
@pytest.mark.parametrize("tag,parallel,kv_type,measured_gb", [
    ("llama3.1:8b", 1, "f16", 5.8),
    ("llama3.2:3b", 4, "q8_0", 3.90),
    ("qwen3:4b-instruct", 4, "q8_0", 4.88),
    ("llama3.1:8b", 4, "q8_0", 7.00),
])
def test_estimate_matches_measurements(monkeypatch, tag, parallel, kv_type, measured_gb):
    _server(monkeypatch, parallel, kv_type)
    assert abs(models.model_fit_gb(models.by_tag(tag)) - measured_gb) < 0.25


def test_kv_elems_from_info_variants():
    qwen35_hybrid = {  # only every 4th layer keeps a KV cache
        "general.architecture": "qwen35", "qwen35.block_count": 32, "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
        "qwen35.full_attention_interval": 4,
    }
    assert models.kv_elems_from_info(qwen35_hybrid) == 8 * 4 * 512
    llama_no_key_length = {  # head dim derived from embedding / heads
        "general.architecture": "llama", "llama.block_count": 32, "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096,
    }
    assert models.kv_elems_from_info(llama_no_key_length) == 32 * 8 * 256
    per_layer_list = {**llama_no_key_length, "llama.attention.head_count_kv": [8, 8, 4]}
    assert models.kv_elems_from_info(per_layer_list) == 32 * 8 * 256
    assert models.kv_elems_from_info({}) is None
    assert models.kv_elems_from_info({"general.architecture": "x"}) is None


@pytest.mark.parametrize("ref,expected", [
    ("hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"),
    ("hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"),
    ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF", "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
    ("  huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/  ", "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
])
def test_normalize_hf_accepts(ref, expected):
    assert models.normalize_hf(ref) == expected


@pytest.mark.parametrize("ref", [
    "", "llama3.1:8b", "hf.co/user", "hf.co/user/repo/extra/path",
    "https://evil.example.com/user/repo", "hf.co/user/repo; rm -rf /", "hf.co/../../etc",
    "hf.co/user/repo:Q4 K M", "hf.co/-user/repo",
])
def test_normalize_hf_rejects(ref):
    assert models.normalize_hf(ref) is None


def test_hf_label():
    assert models.hf_label("hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M") == "Llama-3.2-3B-Instruct (Q4_K_M)"
    assert models.hf_label("hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF") == "Qwen2.5-0.5B-Instruct"


# -- chat template support ----------------------------------------------------

GO_LLAMA = '{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>{{ .System }}{{ end }}'
JINJA_QWEN = "{%- if messages[0].role == 'system' %}{{- messages[0].content }}{%- endif %}"
GO_GEMMA_FOLD = '{{- range .Messages }}{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user{{ end }}{{ end }}'
JINJA_REJECTS = "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
GO_NO_SYSTEM = '{{- range .Messages }}{{- if eq .Role "user" }}U: {{ .Content }}{{ else }}A: {{ .Content }}{{ end }}{{ end }}'
LEGACY_PROMPT_ONLY = "{{ .Prompt }}"
# Real-world (Qwen3.5 GGUF): supports system prompts, but raises for narrower cases.
JINJA_SUPPORTS_WITH_LIMITS = (
    "{%- if is_system_content %}{{- raise_exception('System message cannot contain images.') }}{%- endif %}"
    "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + content }}{%- endif %}"
    "{%- if message.role == \"system\" %}{%- if not loop.first %}"
    "{{- raise_exception('System message must be at the beginning.') }}{%- endif %}{%- endif %}"
)


@pytest.mark.parametrize("template,expected", [
    (GO_LLAMA, True), (JINJA_QWEN, True), (GO_GEMMA_FOLD, True), (JINJA_SUPPORTS_WITH_LIMITS, True),
    (JINJA_REJECTS, False), (GO_NO_SYSTEM, False), (LEGACY_PROMPT_ONLY, False), ("", False),
])
def test_template_supports_system(template, expected):
    assert models.template_supports_system(template) is expected


@pytest.mark.parametrize("modelfile, template, expected", [
    ("FROM x\nTEMPLATE {{ .Prompt }}\nRENDERER qwen3.5\nPARSER qwen3.5", "{{ .Prompt }}", True),
    ("FROM x\nTEMPLATE {{ .Prompt }}", "{{ .Prompt }}", False),
])
def test_template_info_renderer(monkeypatch, modelfile, template, expected):
    class Info:
        pass
    info = Info()
    info.modelfile, info.template, info.capabilities = modelfile, template, ["completion"]
    monkeypatch.setattr(models.ollama, "show", lambda tag: info)
    models.template_info.cache_clear()
    assert models.template_info("m")["system"] is expected
