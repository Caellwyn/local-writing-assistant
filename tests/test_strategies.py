"""Research strategies (focused, loop, rag), passage size, batched reading. No model needed."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research  # noqa: E402
from library import Library  # noqa: E402
from research import FinderPointers, Part, PartResult, PartsSynthesis, Plan  # noqa: E402
from test_research import KEEPER, StubClient, lib, run_facts, sentence_with  # noqa: E402,F401

LAMP = "The harbour lamp was lit each night by Tobias Wren."
MAP = "Tobias Wren later drew the map of the northern coast."
CHAIN = f"{LAMP}\n\nThe weather turned cold in the autumn of that year.\n\n{MAP}\n\n"
QUESTION = "Who lit the harbour lamp, and what did that person draw?"
PLAN = [("Who lit the harbour lamp?", ["lamp"]), ("What did the lamp-lighter draw?", ["drew", "map"])]


@pytest.fixture
def chain_lib(tmp_path):
    lib = Library(tmp_path / "lib")
    f = tmp_path / "harbour.txt"
    f.write_text(CHAIN + " ".join(f"filler{i}" for i in range(300)), encoding="utf-8")
    lib.import_files([str(f)])
    return lib


def head_of(user):
    """The search part of a finder prompt, without the excerpt (which mentions Tobias Wren anyway)."""
    return re.sub(r"<<<.*?>>>", "", user, flags=re.S)


def chain_finder(user, find_map_by_rewrite=False):
    head = head_of(user)
    if "This search: Who lit the harbour lamp?" in head:
        return FinderPointers(sentences=sentence_with(user, "harbour lamp was lit"))
    if "This search: What did the lamp-lighter draw?" in head and "Tobias Wren" in head:
        return FinderPointers(sentences=[] if find_map_by_rewrite else sentence_with(user, "drew the map"))
    if "This search: Who tended the light?" in head:  # the rewritten search
        return FinderPointers(sentences=sentence_with(user, "drew the map"))
    return FinderPointers(sentences=[])


def evidence_number(user, phrase):
    m = re.search(rf"\[E(\d+)\][^\n]*{re.escape(phrase)}", user.split("Evidence:", 1)[-1])
    return int(m.group(1)) if m else None


def chain_synth(user, leave_out_part_2=False):
    lamp, map_ = evidence_number(user, "harbour lamp was lit"), evidence_number(user, "drew the map")
    parts = [PartResult(part=1, found=bool(lamp), answer="Tobias Wren" if lamp else "", citations=[lamp] if lamp else []),
             PartResult(part=2, found=bool(map_), answer="the map of the northern coast" if map_ else "",
                        citations=[map_] if map_ else [])]
    if lamp and map_ and not leave_out_part_2:
        answer = f"Tobias Wren [E{lamp}] drew the map of the northern coast [E{map_}]."
    elif lamp:
        answer = f"Tobias Wren lit it [E{lamp}]." + ("" if map_ else " What he drew isn't in the evidence.")
    else:
        answer = "Not in the evidence."
    cites = [n for n in (lamp, map_) if n]
    return PartsSynthesis(parts=parts, found=bool(lamp and map_), answer=answer, citations=cites)


def run(lib, client, **kw):
    kw.setdefault("finder_model", "f")
    kw.setdefault("researcher_model", "r")
    return research.find_facts(lib, QUESTION, client=client, chunk_words=100, overlap=10, **kw)


# -- focused and loop -----------------------------------------------------------------


def test_focused_searches_each_part_once(chain_lib):
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="focused")
    assert r.stats["scans"] == 3 and r.stats["rounds"] == 1  # the question itself, then each part; no follow-up
    assert [e.quote for e in r.evidence] == [LAMP]  # part 2 can't be recognised without the name
    assert not r.found
    finder = [head_of(u) for role, u in client.prompts if role == "finder"]
    heads = {h.split("This search: ")[1].splitlines()[0] for h in finder if "This search: " in h}
    assert heads == {"Who lit the harbour lamp?", "What did the lamp-lighter draw?"}
    assert any(h.startswith(f"Research question: {QUESTION}") for h in finder)  # the whole question, as asked


def test_loop_follows_a_chain_with_what_it_found(chain_lib):
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop")
    assert r.found and "map of the northern coast" in r.answer
    assert [e.quote for e in r.evidence] == [LAMP, MAP]
    assert r.stats["rounds"] == 2 and r.stats["scans"] == 4
    follow = [s for s in r.stats["searches"] if s["round"] == 2]
    assert follow == [{"round": 2, "part": 2, "ask": "What did the lamp-lighter draw?", "known": ["Tobias Wren"]}]
    assert client.calls.count("synthesis") == 2  # once per round that found something new


def test_loop_never_repeats_a_search(chain_lib):
    client = StubClient(finder=lambda u: chain_finder(u, find_map_by_rewrite=True), plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop")
    assert not r.found
    assert r.stats["rounds"] == 2 and r.stats["scans"] == 4  # one follow-up, found nothing new: stop
    assert client.calls.count("synthesis") == 1


def test_deep_research_rewrites_a_dead_end(chain_lib):
    client = StubClient(finder=lambda u: chain_finder(u, find_map_by_rewrite=True), plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop", deep=True)
    assert "rewrite" in client.calls
    assert r.found and r.stats["rounds"] == 3
    assert r.stats["searches"][-1]["ask"] == "Who tended the light?"


def test_loop_respects_the_scan_limit(chain_lib, monkeypatch):
    monkeypatch.setattr(research, "MAX_SCANS", 3)
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop")
    assert r.stats["scans"] == 3 and r.stats["rounds"] == 1
    assert any("limit" in f for f in r.flags)


def test_loop_abstains_without_follow_ups_when_nothing_is_found(chain_lib):
    client = StubClient(finder=lambda u: FinderPointers(sentences=[]), plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop")
    assert not r.found and "couldn't find" in r.answer
    assert r.stats["scans"] == 6  # the question and two parts, then all again broader; no researcher call
    assert "synthesis" not in client.calls


def test_a_part_left_out_of_the_answer_is_added(chain_lib):
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=lambda u: chain_synth(u, leave_out_part_2=True))
    r = run(chain_lib, client, strategy="loop")
    assert "**Also found:**" in r.answer and "map of the northern coast [E2]" in r.answer
    assert r.cited == [1, 2] and r.stats["parts_added"] == 1
    assert any("left out part(s) 2" in f for f in r.flags)


def test_chunk_first_asks_each_chunk_all_its_searches_in_a_row(chain_lib):
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="focused", layout="chunk-first", workers=1)
    finder = [u for role, u in client.prompts if role == "finder"]
    assert all(u.startswith("Excerpt from") for u in finder)  # the chunk first: a cacheable prefix
    excerpts = [u.split(">>>")[0] for u in finder]
    assert all(excerpts[i] == excerpts[i + 1] == excerpts[i + 2]  # same chunk: the question, both parts
               for i in range(0, len(excerpts), 3))
    assert [e.quote for e in r.evidence] == [LAMP]  # same evidence as question-first


def test_unknown_strategy_is_refused(chain_lib):
    with pytest.raises(ValueError):
        run(chain_lib, StubClient(), strategy="magic")


# -- passages and batched reading ----------------------------------------------------


def test_split_passage_cuts_between_sentences():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))
    units = research.split_sentences(text)
    pieces = research.split_passage(units, text, 0, len(text), max_words=50)
    assert all(len(text[a:b].split()) <= 50 for a, b in pieces)
    assert pieces[0][0] == 0 and pieces[-1][1] == len(text)
    assert all(text[a:b].endswith("words.") for a, b in pieces)  # whole sentences
    assert all(b <= c for (_, b), (c, _) in zip(pieces, pieces[1:]))
    assert research.split_passage(units, text, 0, 40, max_words=50) == [(0, 40)]  # short: unchanged


def test_long_runs_of_pointed_sentences_become_several_passages(lib):
    def everything(user):
        return FinderPointers(sentences=list(range(1, len(re.findall(r"^\[\d+\]", user, re.M)) + 1)))
    r = run_facts(lib, StubClient(everything))
    assert len(r.evidence) > 2
    assert all(len(e.quote.split()) <= research.MAX_PASSAGE_WORDS for e in r.evidence)


def test_evidence_too_long_for_one_reading_is_read_in_batches(lib, monkeypatch):
    monkeypatch.setattr(research, "MAX_EVIDENCE_WORDS", 150)

    def everything(user):
        return FinderPointers(sentences=list(range(1, len(re.findall(r"^\[\d+\]", user, re.M)) + 1)))
    client = StubClient(everything, synth_answer="Mara [E1].")
    r = run_facts(lib, client)
    batches = r.stats["synthesis_batches"]
    assert batches > 1
    assert client.calls.count("synthesis") == batches + 1  # each batch, then one call combining them
    assert r.stats["evidence_delivered"] == len(r.evidence)  # nothing cut
    assert any("batches" in f for f in r.flags)


def test_batches_pack_in_order():
    assert research.batches(["a b", "c d", "e f g"], 4) == [["a b", "c d"], ["e f g"]]
    assert research.batches(["one two three four five"], 2) == [["one two three four five"]]


def test_clean_parts_and_clip_known():
    plan = Plan(parts=[Part(ask=" Who? ", terms=["a", " "]), Part(ask="who?", terms=[]), Part(ask="", terms=[])]
                + [Part(ask=f"Q{i}", terms=[]) for i in range(6)])
    parts = research.clean_parts(plan, "question")
    assert [p.ask for p in parts] == ["Who?", "Q0", "Q1", "Q2"] and parts[0].terms == ["a"]
    assert [p.ask for p in research.clean_parts(None, "question")] == ["question"]
    assert research.clip_known("Tobias Wren [E3]") == "Tobias Wren"
    assert len(research.clip_known("word " * 50).split()) == research.MAX_KNOWN_WORDS


def test_missing_parts_judged_by_their_words():
    results = {0: PartResult(part=1, found=True, answer="1814", citations=[1]),
               1: PartResult(part=2, found=True, answer="stranded three days at Dover", citations=[2]),
               2: PartResult(part=3, found=False, answer="", citations=[])}
    assert research.missing_parts("The great fog of 1814 [E1].", results) == [1]
    assert research.missing_parts("1814; Clerval was stranded at Dover for three days.", results) == []


# -- the RAG control -------------------------------------------------------------------


def lighthouse_vector(text):
    return [text.lower().count("lighthouse"), text.lower().count("ledger"), 0.001]


def test_rag_retrieves_top_sentences_and_caches_the_index(lib):
    client = StubClient(embed=lighthouse_vector)
    r = run_facts(lib, client, strategy="rag", rag_k=1)
    assert [e.quote for e in r.evidence] == [KEEPER]
    assert "finder" not in client.calls and "brief" not in client.calls
    assert r.stats["retrieved"] == 1 and r.stats["rag_k"] == 1
    first = len(client.embedded)
    client2 = StubClient(embed=lighthouse_vector)
    run_facts(lib, client2, strategy="rag", rag_k=1)
    assert len(client2.embedded) == 1 < first  # only the question: the sentences came from the cache
    assert client2.embedded[0].endswith("Query: Who kept the lighthouse?")


# -- the planner's guard rails ----------------------------------------------------------


def test_a_one_part_plan_is_answered_as_asked(chain_lib):
    client = StubClient(finder=lambda u: FinderPointers(sentences=sentence_with(u, "harbour lamp was lit")),
                        plan=[("Who lit the harbour lamp?", ["lamp", "lit"])], parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="loop")
    assert r.stats["parts"] == [] and r.stats["scans"] == 1 and r.stats["rounds"] == 1
    finder = [head_of(u) for role, u in client.prompts if role == "finder"]
    assert all(h.startswith(f"Research question: {QUESTION}") and "This search" not in h for h in finder)
    assert "lamp, lit" in finder[0]  # the plan's terms still help
    assert client.calls.count("synthesis") == 1 and r.answer.startswith("Mara kept the light")  # plain Synthesis


@pytest.mark.parametrize("blank_answer", ["", "false", "  True. "])
def test_a_blank_answer_is_rebuilt_from_the_part_notes(chain_lib, blank_answer):
    def synth(u):
        s = chain_synth(u)
        return s.model_copy(update={"answer": blank_answer, "found": False})
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=synth)
    r = run(chain_lib, client, strategy="loop")
    assert r.found and r.answer.startswith("Tobias Wren [E1]; the map of the northern coast [E2]")
    assert any("answer was empty" in f for f in r.flags)


def test_a_blank_answer_with_no_part_notes_abstains(chain_lib):
    def synth(u):
        return PartsSynthesis(parts=[], found=True, answer="", citations=[])
    client = StubClient(finder=chain_finder, plan=PLAN, parts_synth=synth)
    r = run(chain_lib, client, strategy="focused")
    assert not r.found and "couldn't find" in r.answer


def test_blank():
    assert all(research.blank(a) for a in ("", None, "false", "True.", " n/a "))
    assert not research.blank("Seven [E1]")


# -- hybrid: retrieval, then finders on the retrieved windows --------------------------


def test_retrieved_chunks_widen_merge_and_mark_gaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    units = {"d": research.split_sentences(text)}
    doc = type("Doc", (), {"id": "d", "title": "T"})()
    hit = lambda i: ("d", "T", units["d"][i][0], units["d"][i][1])  # noqa: E731
    chunks = research.retrieved_chunks({hit(5), hit(7), hit(20)}, [doc], {"d": text}, units, words=1000, window=1)
    assert len(chunks) == 1  # both runs fit one excerpt
    picked = [units["d"].index(u) for u in chunks[0].units]
    assert picked == [4, 5, 6, 7, 8, 19, 20, 21]  # 5 and 7 widened and merged; 20 on its own
    lines = research.render_numbered(chunks[0]).splitlines()
    assert lines[5] == "[…]" and lines[6].startswith("[6] Sentence number 19")
    small = research.retrieved_chunks({hit(5), hit(20)}, [doc], {"d": text}, units, words=10, window=1)
    assert len(small) == 2 and (small[0].index, small[0].count) == (0, 2)


def test_render_numbered_has_no_gap_marks_in_a_whole_chunk(lib):
    d = lib.documents()[0]
    text = research.library.read_text(d.path)
    for ch in research.chunk_document(d.id, d.title, text, 50, 10):
        assert "[…]" not in research.render_numbered(ch)


def test_hybrid_finders_read_only_the_retrieved_windows(lib):
    client = StubClient(embed=lighthouse_vector)
    r = run_facts(lib, client, strategy="hybrid", rag_k=1)
    assert [e.quote for e in r.evidence] == [KEEPER]
    assert "brief" not in client.calls and "plan" not in client.calls  # no researcher call before the finders
    finder = [u for role, u in client.prompts if role == "finder"]
    assert len(finder) == 1 and "passages found by a search" in finder[0]
    assert r.stats["excerpts"] == 1 and r.stats["chunks"] > 1


def test_combined_answer_keeps_only_the_partial_answers_citations(lib, monkeypatch):
    """The combining call sees partial answers, not passages: it gets each one's citations, and any
    evidence number they didn't cite (e.g. a partial answer's own index) is dropped."""
    monkeypatch.setattr(research, "MAX_EVIDENCE_WORDS", 150)

    def everything(user):
        return FinderPointers(sentences=list(range(1, len(re.findall(r"^\[\d+\]", user, re.M)) + 1)))

    class Combining(StubClient):
        def json(self, role, model, system, user, schema, temperature=0.0, num_ctx=None):
            out = super().json(role, model, system, user, schema, temperature, num_ctx)
            if schema is research.Synthesis and "Partial answer" in user:  # the combining call
                return research.Synthesis(found=True, answer="Mara [E1][E2][E3].", citations=[1, 2, 3])
            if schema is research.Synthesis:  # a batch: cite its last passage
                last = int(re.findall(r"^\[E(\d+)\]", user, re.M)[-1])
                return research.Synthesis(found=True, answer="Mara.", citations=[last])
            return out

    client = Combining(everything)
    r = run_facts(lib, client)
    combine = [u for role, u in client.prompts if role == "synthesis" and "Partial answer" in u][0]
    partial = [int(n) for n in re.findall(r"Evidence it relies on: \[E(\d+)\]", combine)]
    assert "Partial answer A" in combine and "Partial answer B" in combine
    assert len(partial) == r.stats["synthesis_batches"]
    assert r.cited and set(r.cited) <= set(partial)  # [E1]-[E3] survive only if a partial answer cited them


def test_keep_partial_citations_falls_back_to_all_partial_citations():
    partials = [research.Synthesis(found=True, answer="a [E4]", citations=[4]),
                research.Synthesis(found=True, answer="b", citations=[9])]
    out = research.keep_partial_citations(
        research.Synthesis(found=True, answer="Both [E1][E2].", citations=[1, 2]), partials)
    assert out.answer == "Both ." and out.citations == [4, 9]
    kept = research.keep_partial_citations(
        research.Synthesis(found=True, answer="A [E4] and [E7].", citations=[4, 7]), partials)
    assert kept.answer == "A [E4] and ." and kept.citations == [4]


def test_keep_partial_citations_ignores_batches_that_found_nothing():
    """b1 in the 2026-11-10 library round: a batch that found nothing cited 30 passages to dismiss them."""
    partials = [research.Synthesis(found=True, answer="Crane mended it [E2]", citations=[2]),
                research.Synthesis(found=False, answer="Not here [E4][E5]", citations=[4, 5, 6])]
    out = research.keep_partial_citations(
        research.Synthesis(found=True, answer="Crane [E2]; others [E4][E5].", citations=[2, 4, 5, 6]), partials)
    assert out.citations == [2] and "[E4]" not in out.answer
    # nobody found it: every partial's citations may stand, but an empty answer gets no fallback
    none = [research.Synthesis(found=False, answer="no", citations=[4]),
            research.Synthesis(found=False, answer="no", citations=[5])]
    kept = research.keep_partial_citations(research.Synthesis(found=False, answer="Not found [E4].", citations=[4]), none)
    assert kept.citations == [4]
    bare = research.keep_partial_citations(research.Synthesis(found=False, answer="Not found.", citations=[]), none)
    assert bare.citations == []


def test_batch_labels():
    assert [research.batch_label(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]
