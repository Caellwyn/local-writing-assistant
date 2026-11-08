"""Tests for the agent's tool layer and guardrails (no model needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from agent import ToolBox, text_tool_call, to_ollama  # noqa: E402
from library import Library  # noqa: E402


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "lib")
    src = tmp_path / "chapter1.txt"
    src.write_text("The fog came in at dusk.", encoding="utf-8")
    lib.import_files([str(src)])
    return lib


def src_id(lib):
    return next(d.id for d in lib.documents() if d.kind == "source")


def test_unknown_tool(lib):
    result, _ = ToolBox(lib).call("run_shell", {"cmd": "del C:\\"})
    assert result.startswith("Error") and "no tool" in result


def test_extra_arguments_are_dropped(lib):
    result, _ = ToolBox(lib).call("read_document", {"doc_id": src_id(lib), "path": "C:\\Windows\\win.ini"})
    assert "fog" in result


def test_argument_aliases(lib):
    result, _ = ToolBox(lib).call("create_draft", {"draft_title": "Notes", "text": "Some text."})
    assert "Saved new draft" in result


def test_missing_content_explains_itself(lib):
    result, label = ToolBox(lib).call("create_draft", {"title": "Notes"})
    assert "needs title and content" in result and "missing content" in label


def test_placeholder_rejected(lib):
    result, _ = ToolBox(lib).call("create_draft", {"title": "S", "content": "Summary: <insert summary here>"})
    assert result.startswith("Error") and "placeholder" in result


def test_must_read_before_writing(lib):
    sid = src_id(lib)
    box = ToolBox(lib, must_read={sid})
    result, _ = box.call("create_draft", {"title": "Summary", "content": "A summary."})
    assert result.startswith("Error") and sid in result
    box.call("read_document", {"doc_id": sid})
    result, _ = box.call("create_draft", {"title": "Summary", "content": "A summary."})
    assert "Saved new draft" in result


def test_writes_need_user_intent(lib):
    result, _ = ToolBox(lib, allow_write=False).call("create_draft", {"title": "Hello", "content": "Hi there."})
    assert result.startswith("Error") and "didn't ask" in result
    assert not any(d.kind == "draft" for d in lib.documents())


@pytest.mark.parametrize("text,expected", [
    ("Say hi in five words.", False),
    ("Summarize chapter1 and save it", True),
    ("Polish the Summary draft", True),
    ("What is chapter1 about?", False),
])
def test_write_intent_detection(text, expected):
    assert bool(agent._WANTS_WRITE.search(text)) is expected


def test_sources_cannot_be_revised_through_tools(lib):
    result, _ = ToolBox(lib).call("revise_draft", {"doc_id": src_id(lib), "content": "hacked"})
    assert result.startswith("Error") and "read-only" in result


def test_mentioned_documents(lib):
    assert agent.mentioned_documents(lib, "Please summarize chapter1 for me") == {src_id(lib)}
    assert agent.mentioned_documents(lib, "Tell me a joke") == set()


def test_text_tool_call_parsing():
    name, args, rest = text_tool_call('Sure.\n{"name": "create_draft", "parameters": {"title": "A", "content": "B"}}')
    assert (name, args, rest) == ("create_draft", {"title": "A", "content": "B"}, "Sure.")
    assert text_tool_call('{"name": "delete_everything", "parameters": {}}') is None  # unknown tool
    assert text_tool_call('{"name": "create_draft", "parameters": {}} and more text') is None
    assert text_tool_call("no json here") is None


def test_old_tool_results_are_stubbed():
    msgs = [
        {"role": "user", "content": "read it"},
        {"role": "tool", "tool_name": "read_document", "content": "LONG PAGE", "activity": "x"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "now polish"},
        {"role": "tool", "tool_name": "read_document", "content": "CURRENT PAGE"},
    ]
    out = to_ollama(msgs)
    assert out[1]["content"] == agent.OLD_RESULT_STUB
    assert out[4]["content"] == "CURRENT PAGE"
    assert all("activity" not in m for m in out)  # UI-only fields never reach the model
