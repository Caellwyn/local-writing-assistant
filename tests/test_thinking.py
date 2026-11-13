"""Thinking: the Thinking setting reaches Ollama, reasoning streams to the page and is kept with the reply,
and it is never sent back to the model. No Ollama needed."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import app  # noqa: E402
import models  # noqa: E402
from library import Library  # noqa: E402


def _stream(*parts):
    """Ollama stream chunks: ("think", text) or ("say", text)."""
    for kind, text in parts:
        yield SimpleNamespace(message=SimpleNamespace(thinking=text if kind == "think" else None,
                                                      content=text if kind == "say" else "", tool_calls=None))


@pytest.fixture
def fake_chat(monkeypatch):
    seen = {}

    def chat(**kw):
        seen.update(kw)
        return _stream(("think", "The user wants "), ("think", "a greeting."), ("say", "Hello"), ("say", " there."))

    monkeypatch.setattr(agent.ollama, "chat", chat)
    monkeypatch.setattr(app.ollama, "chat", chat)
    monkeypatch.setattr(models, "template_info", lambda tag: {"system": True, "tools": True})
    return seen


def test_chat_options_follow_the_thinking_setting(monkeypatch):
    assert models.chat_options("qwen3.5:9b") == {"think": False}  # off unless the user turns it on
    assert models.chat_options("qwen3.5:9b", think=True) == {"think": True}
    assert models.chat_options("llama3.1:8b", think=True) == {}  # can't reason: no think argument
    monkeypatch.setattr(models, "capabilities", lambda tag: ("completion", "thinking"))
    assert models.can_think("hf.co/someone/Qwen3.5-9B-GGUF:Q4_K_M")
    assert models.chat_options("hf.co/someone/Qwen3.5-9B-GGUF:Q4_K_M") == {"think": False}
    assert models.thinking_only("qwen3:4b-thinking-2507")
    assert models.chat_options("qwen3:4b-thinking-2507", think=False) == {"think": True}  # always reasons


def test_agent_streams_and_keeps_the_reasoning(tmp_path, fake_chat):
    lib = Library(tmp_path / "lib")
    messages = [{"role": "user", "content": "Say hello."}]
    thoughts, texts = [], []
    status = agent.run_turn(lib, messages, "qwen3.5:9b", 8192, texts.append, lambda label: None, lambda: False,
                            think=True, on_thinking=thoughts.append)
    assert status == "done" and fake_chat["think"] is True
    assert thoughts[-1] == "The user wants a greeting." and texts[-1] == "Hello there."
    assert messages[-1] == {"role": "assistant", "content": "Hello there.", "thinking": "The user wants a greeting."}
    assert all("thinking" not in m for m in agent.to_ollama(messages))  # never sent back to the model


def test_plain_chat_streams_and_keeps_the_reasoning(fake_chat):
    api = app.Api.__new__(app.Api)
    api._num_ctx, api._stop = 8192, threading.Event()
    thread = {"messages": [{"role": "user", "content": "Say hello."}], "instructions": ""}
    thoughts, texts = [], []
    assert api._plain_chat(thread, "qwen3.5:9b", texts.append, True, thoughts.append) == "done"
    assert fake_chat["think"] is True and thoughts[-1] == "The user wants a greeting."
    assert thread["messages"][-1]["thinking"] == "The user wants a greeting."
    assert thread["messages"][-1]["content"] == "Hello there."

    api._plain_chat(thread, "qwen3.5:9b", texts.append, False, thoughts.append)
    assert fake_chat["think"] is False  # the setting off: reasoning off


def test_reopened_chats_show_the_reasoning():
    api = app.Api.__new__(app.Api)
    thread = {"id": "t", "title": "x", "tools": True, "messages": [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "", "thinking": "I should search.",
         "tool_calls": [{"function": {"name": "search_library", "arguments": {"query": "x"}}}]},
        {"role": "tool", "tool_name": "search_library", "content": "...", "activity": "🔎 Searched"},
        {"role": "assistant", "content": "Found it.", "thinking": "That answers it."},
    ]}
    view = [m for m in api._view(thread)["messages"] if m["role"] == "assistant"]
    assert [(m["content"], m["thinking"], m["index"]) for m in view] == [
        ("", "I should search.", None),  # reasoning before a tool call: shown, nothing to save
        ("Found it.", "That answers it.", 3),
    ]
