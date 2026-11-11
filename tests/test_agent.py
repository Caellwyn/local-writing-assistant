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


# -- research tools ------------------------------------------------------------

def test_research_tool_wiring(lib, monkeypatch):
    import research

    sid = src_id(lib)
    fake = research.Report("facts", "When did the fog come?", answer="At dusk [E1].", found=True,
                           evidence=[research.Evidence(1, sid, "chapter1", 1, 4, 24, "fog came in at dusk")],
                           cited=[1], stats={"documents": 1, "seconds": 2.0})
    seen = {}

    def fake_find_facts(lib_, question, **kw):
        seen.update(question=question, **kw)
        return fake

    monkeypatch.setattr(research, "find_facts", fake_find_facts)
    notes = []
    box = ToolBox(lib, must_read={sid}, research={
        "finder_model": "small", "researcher_model": "big", "workers": 3,
        "on_report": lambda r: notes.append(r) or "Research - fog",
    })
    result, label = box.call("research_library", {"query": "When did the fog come?"})  # alias for question
    assert seen["question"] == "When did the fog come?" and seen["finder_model"] == "small" and seen["workers"] == 3
    assert "shown to the user" in result and "Research - fog" in result
    assert label.startswith("🔬") and "1 passage" in label
    assert box.last_report["note"] == "Research - fog" and box.last_report["scanned"] == [sid]
    assert sid in box.read  # documents the pipeline read count as read for later drafts
    tool_msg = {"role": "tool", "tool_name": "research_library", "content": result, "report": box.last_report}
    assert sid in agent.previously_read([tool_msg])


def test_research_tool_scope_by_title(lib, monkeypatch):
    import research

    seen = {}
    monkeypatch.setattr(research, "find_facts",
                        lambda lib_, q, **kw: seen.update(kw) or research.Report("facts", q, stats={}))
    box = ToolBox(lib, research={"finder_model": "m", "researcher_model": "m"})
    box.call("research_library", {"question": "q", "documents": "chapter1"})
    assert seen["doc_ids"] == [src_id(lib)]
    result, _ = box.call("research_library", {"question": "q", "documents": "no-such-doc"})
    assert result.startswith("Error")


def test_research_unavailable_without_context(lib):
    result, _ = ToolBox(lib).call("research_library", {"question": "q"})
    assert result.startswith("Error") and "research" in result


# -- system prompt handling ---------------------------------------------------

def test_with_system_native():
    out = agent.with_system("Be brief.", [{"role": "user", "content": "hi"}], supported=True)
    assert out[0] == {"role": "system", "content": "Be brief."}


def test_mid_conversation_system_notes_become_user_notes():
    # Many templates raise "System message must be at the beginning".
    history = [{"role": "user", "content": "hi"}, {"role": "system", "content": "save it now"}]
    out = agent.with_system("Be brief.", history, supported=True)
    assert [m["role"] for m in out] == ["system", "user", "user"]
    assert out[2]["content"] == "[Note from the app] save it now"


def test_with_system_folds_into_first_user_message():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "save it now"},
        {"role": "user", "content": "again"},
    ]
    out = agent.with_system("Be brief.", history, supported=False)
    assert all(m["role"] != "system" for m in out)
    assert out[0]["content"].startswith("[Instructions") and "Be brief." in out[0]["content"]
    assert out[2]["role"] == "user" and out[2]["content"].startswith("[Note from the app]")
    assert out[3]["content"] == "again"  # only the first user message carries the instructions
    assert history[0]["content"] == "hi"  # stored history is never modified


def test_closing_the_app_stops_work_and_unloads_models(monkeypatch):
    import app
    import models
    monkeypatch.setattr(models, "unload_used", lambda: ["qwen3.5:9b"])
    api = app.Api.__new__(app.Api)  # no window, library or settings needed
    api._stop, api._busy = __import__("threading").Event(), True
    stopped = []
    api._research_server = type("S", (), {"stop": lambda self: stopped.append(True)})()
    assert api.shutdown(grace=0.2) == ["qwen3.5:9b"]
    assert api._stop.is_set()  # a running reply or research run is told to stop
    assert stopped  # the private research server (and its model runners) ends too


def test_research_runs_on_the_research_server_and_frees_it_after(lib, monkeypatch):
    import research
    seen, calls = {}, []

    def find_facts(lib_, q, **kw):
        seen.update(kw)
        raise RuntimeError("model crashed")  # finish() must run even when research fails

    monkeypatch.setattr(research, "find_facts", find_facts)
    box = ToolBox(lib, research={
        "finder_model": "m", "researcher_model": "m", "workers": 1,
        "prepare": lambda: calls.append("prepare") or {"host": "http://127.0.0.1:5555", "workers": 4},
        "finish": lambda: calls.append("finish"),
    })
    with pytest.raises(RuntimeError):  # the app reports it; the tool doesn't swallow it
        box.call("research_library", {"question": "q"})
    assert calls == ["prepare", "finish"]
    assert seen["workers"] == 4 and seen["client"]._client._client.base_url.port == 5555


def test_research_without_a_research_server_uses_the_main_ollama(lib, monkeypatch):
    import research
    seen = {}
    monkeypatch.setattr(research, "find_facts",
                        lambda lib_, q, **kw: seen.update(kw) or research.Report("facts", q, stats={}))
    box = ToolBox(lib, research={"finder_model": "m", "researcher_model": "m", "workers": 1,
                                 "prepare": lambda: {"host": None, "workers": 1}})
    box.call("research_library", {"question": "q"})
    assert seen["client"] is None and seen["workers"] == 1  # find_facts makes its default client
