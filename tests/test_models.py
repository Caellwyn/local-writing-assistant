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
