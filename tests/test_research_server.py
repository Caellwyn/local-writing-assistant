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


# -- ModelClient: finders on the research server, the researcher on the user's Ollama --------------

import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from pydantic import BaseModel  # noqa: E402

import models  # noqa: E402
import research  # noqa: E402


class _Answer(BaseModel):
    ok: bool


class _FakeServer:
    """One Ollama server: which models are loaded (tag -> bytes on the GPU) and what was asked."""

    def __init__(self, name, loaded=None):
        self.name, self.loaded, self.calls, self.unloads = name, dict(loaded or {}), [], []
        self.lock = threading.Lock()

    def chat(self, model, **kw):
        with self.lock:
            self.calls.append(model)
            self.loaded[model] = 1
        return SimpleNamespace(prompt_eval_count=1, eval_count=1, load_duration=0, prompt_eval_duration=0,
                               done_reason="stop", message=SimpleNamespace(content='{"ok": true}'))

    def embed(self, model, input, options=None):
        self.calls.append(model)
        self.loaded[model] = 0  # the embedder runs on the CPU (research.EMBED_OPTIONS)
        return SimpleNamespace(prompt_eval_count=1, load_duration=0, embeddings=[[1.0] for _ in input])

    def ps(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m, size_vram=v) for m, v in self.loaded.items()])

    def generate(self, model, keep_alive=None, **kw):
        assert keep_alive == 0
        self.unloads.append(model)
        self.loaded.pop(model, None)


@pytest.fixture
def servers(monkeypatch):
    main = _FakeServer("main", {"chat:latest": 1, "other-app:latest": 1})
    research_srv = _FakeServer("research")

    def client(host=None, timeout=None):
        return research_srv if host == "http://127.0.0.1:5555" else main

    monkeypatch.setattr(research.ollama, "Client", client)
    monkeypatch.setattr(models.ollama, "Client", client)
    monkeypatch.setattr(models, "_used", {"chat"})  # the app's chat model; other-app isn't the app's
    monkeypatch.setattr(models, "capabilities", lambda tag: [])
    monkeypatch.setattr(models, "template_info", lambda tag: {"system": True, "tools": False})
    return main, research_srv


def test_without_a_research_server_every_role_uses_one_ollama(servers):
    main, research_srv = servers
    client = research.ModelClient()
    for role in ("finder", "map", "synthesis", "plan"):
        assert client.json(role, "m", "sys", "user", _Answer).ok
    client.embed("embedder", ["a"])
    assert main.calls == ["m"] * 4 + ["embedder"] and not research_srv.calls
    assert not main.unloads  # one server swaps models by itself


def test_finders_and_researcher_on_separate_servers_never_share_the_gpu(servers):
    main, research_srv = servers
    client = research.ModelClient(parallel_host="http://127.0.0.1:5555")

    client.json("finder", "finder-m", "sys", "user", _Answer)
    assert research_srv.calls == ["finder-m"]
    assert main.unloads == ["chat"]  # the app's chat model made room...
    assert "other-app:latest" in main.loaded  # ...another program's model was left alone

    client.embed("embedder", ["a", "b"])  # next to the finder, on the CPU: no swap
    client.json("synthesis", "researcher-m", "sys", "user", _Answer)
    assert main.calls == ["researcher-m"]
    assert research_srv.unloads == ["finder-m"]  # the finder made room; the CPU embedder stays
    assert "embedder" in research_srv.loaded

    client.json("plan", "researcher-m", "sys", "user", _Answer)  # same server: nothing to swap
    assert research_srv.unloads == ["finder-m"] and main.unloads == ["chat"]


def test_parallel_finder_calls_swap_once(servers):
    main, research_srv = servers
    client = research.ModelClient(parallel_host="http://127.0.0.1:5555")
    threads = [threading.Thread(target=client.json, args=("finder", "finder-m", "s", "u", _Answer))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(research_srv.calls) == 8 and main.unloads == ["chat"]
