"""Tests for model catalog logic and Hugging Face reference validation (no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa: E402


@pytest.mark.parametrize("vram,expected", [
    (0, "llama3.2:3b"), (4, "llama3.2:3b"), (6, "qwen3:4b-instruct"),
    (8, "qwen3:4b-instruct"), (12, "qwen3:8b"), (16, "qwen3:14b"),
])
def test_recommend_by_vram(vram, expected):
    assert models.recommend(vram) == expected


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
