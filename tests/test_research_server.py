"""Tests for the private research server's setup (no processes started)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_server as rs  # noqa: E402


def test_server_env_uses_the_benchmarked_settings_on_its_own_port(monkeypatch):
    monkeypatch.setattr(rs.models, "_ollama_setting", lambda name: "D:\models" if name == "OLLAMA_MODELS" else "")
    env = rs.server_env(50123, {"PATH": "x", "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_HOST": "0.0.0.0"})
    assert env["OLLAMA_HOST"] == "127.0.0.1:50123"
    assert env["OLLAMA_NUM_PARALLEL"] == "4" and env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert env["OLLAMA_MODELS"] == "D:\models"  # same downloads as the user's Ollama
    assert env["PATH"] == "x"


@pytest.mark.parametrize("host,local", [
    ("", True), ("127.0.0.1:11434", True), ("http://localhost:11434", True), ("0.0.0.0", True),
    ("http://192.168.1.20:11434", False), ("gpu-box:11434", False),
])
def test_main_server_is_local(host, local):
    assert rs.main_server_is_local(host) is local


def test_falls_back_to_the_main_ollama_without_an_executable(monkeypatch):
    monkeypatch.setattr(rs, "ollama_exe", lambda: None)
    s = rs.ResearchServer()
    started = []
    assert s.ensure(on_start=lambda: started.append(1)) is None
    assert s.failed and not started and s.host is None
    assert s.unload_all() == []  # nothing to free
    s.stop()  # harmless


def test_falls_back_when_ollama_runs_on_another_machine(monkeypatch):
    monkeypatch.setattr(rs, "ollama_exe", lambda: "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert rs.ResearchServer().ensure() is None


def test_a_failed_start_falls_back_once(monkeypatch):
    monkeypatch.setattr(rs, "ollama_exe", lambda: "ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    tries = []

    def boom(self, exe):
        tries.append(exe)
        raise RuntimeError("port in use")

    monkeypatch.setattr(rs.ResearchServer, "_start", boom)
    s = rs.ResearchServer()
    assert s.ensure() is None and s.ensure() is None
    assert tries == ["ollama"] and "port in use" in s.failed  # not retried on every question
