"""Tests for the research pipeline (no model needed: a stub client stands in for Ollama)."""

import re
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research  # noqa: E402
from library import Library  # noqa: E402
from research import (  # noqa: E402
    Brief, CallStat, Case, CaseSynthesis, Chunk, ChunkSummary, FinderPointers, Part, PartResult, PartsSynthesis, Plan,
    RoutedPlan, SummaryText, Synthesis,
    check_citations, chunk_document, context_for, pack_units, pointer_spans, render_numbered, split_sentences,
)

# -- sentences ------------------------------------------------------------------


def units_text(text):
    return [text[a:b] for a, b in split_sentences(text)]


def assert_clean_cover(text):
    """Units are in order, don't overlap, and only whitespace lies between them."""
    units = split_sentences(text)
    assert all(a < b for a, b in units)
    for (a, b), (c, d) in zip(units, units[1:]):
        assert b <= c and not text[b:c].strip()
    if text.strip():
        assert not text[:units[0][0]].strip() and not text[units[-1][1]:].strip()


@pytest.mark.parametrize("text,expected", [
    ("Mr. Sherlock Holmes was seated at the table. Dr. Watson came in with the morning paper.",
     ["Mr. Sherlock Holmes was seated at the table.", "Dr. Watson came in with the morning paper."]),
    ("J. H. Watson wrote the whole account of it.", ["J. H. Watson wrote the whole account of it."]),
    ("“I shall be delighted,” he said warmly. Then he left the room very quietly.",
     ["“I shall be delighted,” he said warmly.", "Then he left the room very quietly."]),
    ("A paragraph without any final stop\n\nA second paragraph follows it here.",
     ["A paragraph without any final stop", "A second paragraph follows it here."]),
    ("“Help!” she cried out loud to everyone nearby.", ["“Help!” she cried out loud to everyone nearby."]),
    ("Was it real? I could not tell, and nobody would say.", ["Was it real? I could not tell, and nobody would say."]),
])
def test_split_sentences(text, expected):
    assert units_text(text) == expected
    assert_clean_cover(text)


def test_long_sentences_are_split():
    long_one = " ".join(["word"] * 40) + "; " + " ".join(["more"] * 40) + "."
    parts = units_text(long_one)
    assert len(parts) == 2 and parts[0].endswith(";") and parts[1].endswith("more.")
    blob = " ".join(f"w{i}" for i in range(300))  # no punctuation at all
    parts = units_text(blob)
    assert all(len(p.split()) <= research.MAX_UNIT_WORDS for p in parts) and " ".join(parts) == blob
    assert_clean_cover(long_one)
    assert_clean_cover(blob)


def test_split_empty_and_whitespace():
    assert split_sentences("") == [] and split_sentences("  \n\n  ") == []


# -- chunks -----------------------------------------------------------------------


def test_pack_units_overlap_and_coverage():
    windows = pack_units([5] * 10, words=12, overlap=5)
    assert windows[0] == (0, 2) and windows[-1][1] == 10
    assert all(b > a for a, b in windows)
    assert all(nxt[0] < cur[1] and nxt[0] > cur[0] for cur, nxt in zip(windows, windows[1:]))  # overlap, progress
    assert pack_units([5] * 4, words=10, overlap=0) == [(0, 2), (2, 4)]
    assert pack_units([50, 5], words=10, overlap=0) == [(0, 1), (1, 2)]  # an oversized unit gets its own chunk
    assert pack_units([], 10, 2) == []


def test_chunk_document_whole_sentences_and_pages(monkeypatch):
    monkeypatch.setattr(research.library, "WORDS_PER_PAGE", 10)
    text = " ".join(f"Sentence {i} has exactly six words." for i in range(10))
    chunks = chunk_document("src-1", "Doc", text, words=18, overlap=6)
    assert all(len(c.units) == 3 for c in chunks[:-1])
    for c in chunks:
        assert c.text == text[c.start:c.end] and c.text.endswith("words.")
        assert c.units[0][0] == c.start and c.units[-1][1] == c.end
    assert chunks[1].units[0] == chunks[0].units[-1]  # one whole sentence of overlap
    assert chunks[0].page == 1 and chunks[1].page == 2  # 12 words before chunk 2
    assert chunks[0].count == len(chunks)


def test_render_numbered_and_pointer_spans():
    text = "First one is here.  Second one\nis here. Third one is here. Fourth one is here."
    ch = chunk_document("d", "t", text, words=100, overlap=0)[0]
    lines = render_numbered(ch).splitlines()
    assert lines == ["[1] First one is here.", "[2] Second one is here.", "[3] Third one is here.",
                     "[4] Fourth one is here."]  # one line each, inner whitespace collapsed
    spans, invalid = pointer_spans(ch, [2, 3, 3, 9, 0])
    assert [text[a:b] for a, b in spans] == ["Second one\nis here. Third one is here."]  # merged run
    assert invalid == 2
    spans, _ = pointer_spans(ch, [4, 1])
    assert [text[a:b] for a, b in spans] == ["First one is here.", "Fourth one is here."]


def test_context_for():
    assert context_for(10_600, 256) == 4096  # a 1,500-word chunk with instructions
    assert context_for(14_000, 256) == 6144  # 2,000 words
    assert context_for(21_000, 256) == 8192
    assert context_for(40_000, 256) == research.NUM_CTX  # capped


def test_check_citations():
    text, cited, flags = check_citations("A [E1]. B [E3][E1]. C [E9].", 3)
    assert cited == [1, 3]
    assert "[E9]" not in text and "[?]" in text
    assert flags and "E9" in flags[0]


# -- stub model client ------------------------------------------------------------

SOURCE = (
    "Mara Vell kept the Saltmere lighthouse for eleven years.\n"
    "She wrote each ship in a green ledger: name, flag, hour, weather.\n\n"
    "On the night the fog came, a ship passed that had no name. Its lamps burned blue."
)
KEEPER = "Mara Vell kept the Saltmere lighthouse for eleven years."


def numbered_lines(user):
    excerpt = user.split("<<<", 1)[-1]  # only the excerpt, not the brief's search terms
    return {int(m[1]): m[2] for m in re.finditer(r"^\[(\d+)\] (.*)$", excerpt, re.M)}


def sentence_with(user, phrase):
    return [n for n, s in numbered_lines(user).items() if phrase in s]


class StubClient:
    """Behaves like research.ModelClient; answers per role with simple rules."""

    def __init__(self, finder=None, synth_answer="Mara kept the light [E1]. Also [E9].", delay=0.0,
                 synth_citations=(), plan=None, parts_synth=None, embed=None, kind="fact", cases=None):
        self.stats, self.calls, self.ctx, self.prompts = [], [], [], []
        self._lock = threading.Lock()
        self.active = self.max_active = 0
        self.finder = finder or self.default_finder
        self.synth_answer = synth_answer
        self.synth_citations = list(synth_citations)
        self.delay = delay
        self.plan = plan or [("Who kept the lighthouse?", ["keeper"])]  # (ask, terms) per part
        self.parts_synth = parts_synth  # user prompt -> PartsSynthesis
        self.embedder = embed  # text -> vector
        self.embedded = []  # texts sent for embedding
        self.kind = kind  # what a routed plan (strategy auto) says the question is
        self.cases = cases  # user prompt -> CaseSynthesis (judgement and collect questions)
        self.schemas = []  # (role, schema) per call

    def embed(self, model, texts):
        with self._lock:
            self.embedded += list(texts)
            self.calls.append("embed")
        return [self.embedder(t) for t in texts]

    @staticmethod
    def default_finder(user):
        hits = sentence_with(user, "Saltmere lighthouse")
        return FinderPointers(sentences=hits + [999] if hits else [])  # 999: a number that doesn't exist

    def json(self, role, model, system, user, schema, temperature=0.0, num_ctx=None):
        with self._lock:
            self.calls.append(role)
            self.ctx.append((role, num_ctx))
            self.prompts.append((role, user))
            self.schemas.append((role, schema))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        try:
            if schema is Brief:
                return Brief(looking_for=["keeper"], terms=["lighthouse"], relevance="who kept it")
            if schema is RoutedPlan:
                return RoutedPlan(parts=[Part(ask=a, terms=t) for a, t in self.plan], kind=self.kind)
            if schema is Plan:
                return Plan(parts=[Part(ask=a, terms=t) for a, t in self.plan])
            if schema is Part:  # a rewritten search (deep research)
                return Part(ask="Who tended the light?", terms=["tended"])
            if schema is PartsSynthesis:
                return self.parts_synth(user)
            if schema is FinderPointers:
                return self.finder(user)
            if schema is CaseSynthesis:
                return self.cases(user) if self.cases else CaseSynthesis(
                    found=True, conclusion="Mostly at odds.", cases=[Case(point="The first case", citations=[1])])
            if schema is Synthesis:
                return Synthesis(found=True, answer=self.synth_answer, citations=self.synth_citations)
            if schema is ChunkSummary:
                return ChunkSummary(points=["Mara keeps the light."], support=sentence_with(user, "Saltmere") or [1])
            if schema is SummaryText:
                return SummaryText(summary="Mara keeps a lighthouse [E1].")
        finally:
            with self._lock:
                self.active -= 1
                self.stats.append(CallStat(role, model, self.delay))


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "lib")
    a = tmp_path / "keeper.txt"
    a.write_text(SOURCE + "\n\n" + " ".join(f"filler{i}" for i in range(400)), encoding="utf-8")
    b = tmp_path / "other.txt"
    b.write_text(" ".join(f"unrelated{i}" for i in range(400)), encoding="utf-8")
    lib.import_files([str(a), str(b)])
    return lib


def run_facts(lib, client, **kw):
    kw.setdefault("finder_model", "f")
    kw.setdefault("researcher_model", "r")
    return research.find_facts(lib, "Who kept the lighthouse?", client=client, chunk_words=100, overlap=10, **kw)


# -- find_facts -----------------------------------------------------------------


def test_find_facts_extracts_pointed_sentences_and_cites(lib):
    client = StubClient()
    report = run_facts(lib, client)
    assert report.status == "done" and report.found
    assert len(report.evidence) == 1
    ev = report.evidence[0]
    assert ev.quote == KEEPER and ev.title == "keeper"  # exact source text, extracted by the app
    assert report.cited == [1]
    assert any("don't exist" in f for f in report.flags)  # the out-of-range sentence number
    assert any("E9" in f for f in report.flags)  # citation to evidence that doesn't exist
    assert report.stats["chunks"] > 2 and report.stats["invalid_pointers"] >= 1


def test_listed_citations_reach_the_answer_even_without_inline_markers(lib):
    """Small models answer "Seven" with no [E#]; the structured citations field still cites."""
    report = run_facts(lib, StubClient(synth_answer="Mara Vell.", synth_citations=[1, 1, 7]))
    assert report.cited == [1]
    assert report.answer.endswith("Sources: [E1][?]")  # E7 doesn't exist: removed and flagged
    assert any("E7" in f for f in report.flags)
    assert not any("cites no evidence" in f for f in report.flags)


def test_listed_citations_already_inline_add_nothing():
    s = Synthesis(found=True, answer="Mara [E1].", citations=[1])
    assert research.with_sources(s) == "Mara [E1]."


def test_thinking_only_for_researcher_roles(monkeypatch):
    monkeypatch.setattr(research.models, "capabilities", lambda m: ["completion", "thinking"])
    off, on = research.ModelClient(), research.ModelClient(think=True)
    assert not off.thinks("synthesis", "qwen3:4b-instruct")  # not asked
    assert on.thinks("synthesis", "qwen3:4b-instruct") and on.thinks("brief", "x")
    assert not on.thinks("finder", "x") and not on.thinks("map", "x")  # per-chunk roles never think
    assert off.thinks("synthesis", "qwen3:4b-thinking")  # thinking-only build: always for its roles
    assert not off.thinks("finder", "qwen3:4b-thinking")
    monkeypatch.setattr(research.models, "capabilities", lambda m: ["completion"])
    assert not on.thinks("synthesis", "llama3.2:3b")  # can't think


def test_thinking_call_gets_a_reasoning_budget(monkeypatch):
    from types import SimpleNamespace

    sent = []

    class FakeOllama:
        def chat(self, **kw):
            sent.append(kw)
            return SimpleNamespace(message=SimpleNamespace(content='{"found": true, "answer": "x [E1]", "citations": [1]}'),
                                   done_reason="stop", prompt_eval_count=10, eval_count=900, load_duration=0)

    monkeypatch.setattr(research.models, "capabilities", lambda m: ["thinking"])
    monkeypatch.setattr(research.models, "template_info", lambda m: {"system": True})
    client = research.ModelClient(think=True)
    client._client = FakeOllama()
    assert client.json("synthesis", "m", "sys", "user", Synthesis).citations == [1]
    assert sent[0]["think"] is True
    assert sent[0]["options"]["num_predict"] == research.MAX_TOKENS["synthesis"] + research.THINK_BUDGET
    assert client.stats[0].thought
    client.json("finder", "m", "sys", "user", Synthesis)
    assert sent[1]["think"] is False and sent[1]["options"]["num_predict"] == research.MAX_TOKENS["finder"]


def test_overlapping_chunks_dont_duplicate_evidence(lib):
    def finder(user):  # every chunk points at the same sentence when it's visible
        return FinderPointers(sentences=sentence_with(user, "Saltmere"))
    report = research.find_facts(lib, "q", finder_model="f", researcher_model="r", client=StubClient(finder),
                                 chunk_words=12, overlap=11)  # heavy overlap: many chunks see it
    assert len(report.evidence) == 1 and report.evidence[0].quote == KEEPER


def test_neighbouring_sentences_merge_into_one_passage(lib):
    def finder(user):
        return FinderPointers(sentences=sentence_with(user, "Saltmere") + sentence_with(user, "green ledger"))
    report = run_facts(lib, StubClient(finder))
    assert len(report.evidence) == 1
    assert report.evidence[0].quote.startswith(KEEPER) and report.evidence[0].quote.endswith("weather.")


def test_no_evidence_abstains_without_synthesis_and_retries_broad(lib):
    client = StubClient(finder=lambda user: FinderPointers(sentences=[]))
    report = run_facts(lib, client)
    assert not report.found and "couldn't find" in report.answer
    assert "synthesis" not in client.calls
    assert report.stats["scans"] == 2  # the first pass, then a broader one


def test_finder_context_follows_the_model(lib):
    client = StubClient()
    run_facts(lib, client, finder_model="small", researcher_model="big")
    assert {c for r, c in client.ctx if r == "finder"} == {4096}  # swapped in anyway: sized to the prompt
    assert {c for r, c in client.ctx if r != "finder"} == {None}  # researcher keeps the default
    client = StubClient()
    run_facts(lib, client, finder_model="same", researcher_model="same")
    assert {c for _, c in client.ctx} == {None}  # one model: no context change, no reload


def test_worker_limit_is_respected(lib):
    client = StubClient(delay=0.05)
    run_facts(lib, client, workers=3)
    assert 1 < client.max_active <= 3


def test_stop_mid_scan(lib):
    client = StubClient(delay=0.02)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2

    report = run_facts(lib, client, workers=1, should_stop=should_stop)
    assert report.status == "stopped"
    assert client.calls.count("finder") < report.stats["chunks"]


def test_research_notes_are_excluded_from_scope(lib):
    report = run_facts(lib, StubClient())
    research.save_note(lib, report)
    titles = [d.title for d in research._scope(lib, None)]
    assert not any(t.startswith(research.RESEARCH_NOTE_PREFIX) for t in titles)
    note = next(d for d in lib.documents() if d.title.startswith(research.RESEARCH_NOTE_PREFIX))
    body = note.path.read_text(encoding="utf-8")
    assert "[E1]" in body and KEEPER in body


# -- summarize ------------------------------------------------------------------


def test_summary_cache_hit_and_invalidation(lib, tmp_path):
    keeper = next(d for d in lib.documents() if d.title == "keeper")
    client = StubClient()
    r1 = research.summarize(lib, [keeper.id], finder_model="f", researcher_model="r", client=client, chunk_words=100)
    assert r1.found and r1.cited == [1] and r1.stats["cache_hits"] == 0
    assert r1.evidence[0].quote == KEEPER  # the supporting sentence, extracted from the source
    maps_first = client.calls.count("map")

    client2 = StubClient()
    r2 = research.summarize(lib, [keeper.id], finder_model="f", researcher_model="r", client=client2, chunk_words=100)
    assert r2.stats["cache_hits"] == 1 and "map" not in client2.calls  # served from cache
    assert r2.answer == r1.answer

    keeper.path.write_text(keeper.path.read_text(encoding="utf-8") + "\n\nA new paragraph.", encoding="utf-8")
    client3 = StubClient()
    research.summarize(lib, [keeper.id], finder_model="f", researcher_model="r", client=client3, chunk_words=100)
    assert client3.calls.count("map") >= maps_first  # edited text -> cache miss

    cache_files = list((lib.root / ".cache" / "summaries").iterdir())
    assert cache_files and all(p.resolve().is_relative_to(lib.root) for p in cache_files)


def test_multi_document_summary_renumbers_evidence(lib):
    client = StubClient()
    report = research.summarize(lib, None, finder_model="f", researcher_model="r", client=client, chunk_words=100)
    assert "combine" in client.calls
    assert report.stats["documents"] == 2
    assert all(1 <= n <= len(report.evidence) for n in report.cited)
    assert {c for r, c in client.ctx if r == "map"} == {4096}  # finder != researcher: one size for all documents


# -- model client ------------------------------------------------------------------

def test_model_client_caps_output_and_rejects_runaway(monkeypatch):
    """A model stuck in a loop hits num_predict; the client counts it as failed, not as an answer."""
    from types import SimpleNamespace

    sent = []

    class FakeOllama:
        def chat(self, **kw):
            sent.append(kw)
            content = '{"sentences": [1, 2, 3, ' + "4, " * 80  # truncated JSON
            return SimpleNamespace(message=SimpleNamespace(content=content), done_reason="length",
                                   prompt_eval_count=10, eval_count=256, load_duration=0)

    monkeypatch.setattr(research.models, "capabilities", lambda m: [])
    monkeypatch.setattr(research.models, "template_info", lambda m: {"system": True})
    client = research.ModelClient()
    client._client = FakeOllama()
    assert client.json("finder", "m", "sys", "user", FinderPointers, num_ctx=4096) is None
    assert all(kw["options"]["num_predict"] == research.MAX_TOKENS["finder"] for kw in sent)
    assert all(kw["options"]["num_ctx"] == 4096 for kw in sent)
    assert len(client.stats) == 2 and all(s.capped and not s.ok for s in client.stats)


def test_parallel_requests_waiting_on_one_load_count_once():
    S = research.CallStat
    calls = [
        S("brief", "big", 4.0, load_seconds=3.0, started=0.0),          # load 1: researcher
        *[S("finder", "small", 9.0, load_seconds=2.5, started=4.01 + i / 1000) for i in range(4)],  # load 2
        S("finder", "small", 3.0, started=8.0),                         # already loaded
        S("synthesis", "big", 5.0, load_seconds=3.1, started=20.0),     # load 3: researcher back
    ]
    loads = research.model_loads(calls)
    assert [(c.role, c.model) for c in loads] == [("brief", "big"), ("finder", "small"), ("synthesis", "big")]


def test_chunk_is_importable_for_typing():
    assert Chunk.__dataclass_fields__["units"]
