"""Retrieval upgrades: keyword (BM25) search, fusion, context windows, and the rag-loop strategy.
No model needed: keyword search embeds nothing, and dense search uses a stub embedder."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research  # noqa: E402
import retrieval  # noqa: E402
from test_research import KEEPER, StubClient, lib, run_facts, sentence_with  # noqa: E402,F401
from test_strategies import LAMP, MAP, chain_lib, chain_synth, run  # noqa: E402,F401


def lighthouse_vector(text):
    return [text.lower().count("lighthouse"), text.lower().count("ledger"), 0.001]


def test_tokens_fold_possessives_plurals_and_drop_question_words():
    assert retrieval.tokens("What were Holmes's habits?") == ["holme", "habit"]
    assert retrieval.tokens("The Gryphon’s song") == ["gryphon", "song"]
    assert retrieval.tokens("glass grass status") == ["glass", "grass", "status"]  # not plurals


def test_keyword_index_ranks_the_rare_name_first():
    texts = ["The man walked to the town.", "Silas Crane mended the shell.", "The town was quiet."]
    s = retrieval.KeywordIndex(texts).scores("Who mended the shell, and did Crane walk to town?")
    assert max(s, key=s.get) == 1


def test_fuse_rewards_items_both_rankings_like():
    assert retrieval.fuse([[1, 2], [3, 2]])[0] == 2  # second in both beats first in one
    assert retrieval.fuse([[5, 9], [9, 7]])[0] == 9  # 9 is in both lists


def test_unit_text_and_widen_add_neighbours():
    text = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here."
    units = research.split_sentences(text)
    assert len(units) == 4
    assert retrieval.unit_text(text, units, 1, window=1) ==         "Sentence one is here. Sentence two is here. Sentence three is here."
    hit = ("d", "T", units[2][0], units[2][1])
    wide = retrieval.widen({hit}, {"d": units}, 1)
    assert sorted(s for _, _, s, _ in wide) == [units[1][0], units[2][0], units[3][0]]
    assert retrieval.widen({hit}, {"d": units}, 0) == {hit}


def test_keyword_rag_embeds_nothing(lib):
    client = StubClient()
    r = run_facts(lib, client, strategy="rag", rag_k=1, rag_mode="keyword")
    assert [e.quote for e in r.evidence] == [KEEPER]
    assert client.embedded == [] and r.stats["rag_mode"] == "keyword"


def test_fused_rag_finds_the_sentence(lib):
    client = StubClient(embed=lighthouse_vector)
    r = run_facts(lib, client, strategy="rag", rag_k=1, rag_mode="fused")
    assert [e.quote for e in r.evidence] == [KEEPER]


def test_rag_window_delivers_the_hit_in_context(lib):
    client = StubClient(embed=lighthouse_vector)
    r = run_facts(lib, client, strategy="rag", rag_k=1, rag_window=1)
    assert len(r.evidence) == 1 and r.evidence[0].quote.startswith(KEEPER)
    assert "green ledger" in r.evidence[0].quote  # the next sentence came with it
    assert r.stats["retrieved"] == 1 and r.stats["rag_window"] == 1


def test_embed_window_indexes_sentences_with_their_neighbours(lib):
    client = StubClient(embed=lighthouse_vector)
    run_facts(lib, client, strategy="rag", rag_k=1, embed_window=1)
    assert any("Saltmere lighthouse" in t and "green ledger" in t for t in client.embedded)
    plain = StubClient(embed=lighthouse_vector)
    run_facts(lib, plain, strategy="rag", rag_k=1)  # window 0 is a separate cache, not reused
    assert any(t.startswith("She wrote each ship") for t in plain.embedded)


def test_rag_loop_follows_a_chain_by_retrieval(chain_lib):
    """Part 2 has no words in common with the answer until part 1's answer (the name) is known."""
    plan = [("Who lit the harbour lamp?", ["lamp"]), ("What did that person make?", [])]
    client = StubClient(plan=plan, parts_synth=chain_synth)
    r = run(chain_lib, client, strategy="rag-loop", rag_k=2, rag_mode="keyword", rag_window=0)
    assert "finder" not in client.calls and client.embedded == []  # no finders, no embeddings
    assert [e.quote for e in r.evidence] == [LAMP, MAP]
    assert r.found and "map of the northern coast" in r.answer
    assert r.stats["rounds"] == 2 and r.stats["scans"] == 4  # the question, 2 parts, then 1 follow-up
    follow = [s for s in r.stats["searches"] if s["round"] == 2]
    assert follow == [{"round": 2, "part": 2, "ask": "What did that person make?", "known": ["Tobias Wren"]}]


def test_follow_up_passages_are_named_and_read_first(chain_lib):
    """b1 and m3 in the 2026-11-10 round: a follow-up found the fact, and the re-reading of ~50
    passages passed over it. The new passages now come first, named in the prompt."""
    plan = [("Who lit the harbour lamp?", ["lamp"]), ("What did that person make?", [])]
    client = StubClient(plan=plan, parts_synth=chain_synth)
    run(chain_lib, client, strategy="rag-loop", rag_k=2, rag_mode="keyword", rag_window=0)
    first, second = [u for role, u in client.prompts if role == "synthesis"]
    assert "follow-up search" not in first
    assert "A follow-up search (part 2, What did that person make?, knowing: Tobias Wren) found new passages: [E2]" in second
    evidence = second.split("Evidence:\n", 1)[1].splitlines()
    assert evidence[0].startswith("[E2]") and "drew the map" in evidence[0]


def test_rag_loop_defaults_to_fused_with_one_sentence_either_side(chain_lib):
    client = StubClient(plan=[("Who lit the harbour lamp?", ["lamp"])], embed=lighthouse_vector)
    r = run(chain_lib, client, strategy="rag-loop", rag_k=1)
    assert r.stats["rag_mode"] == "fused" and r.stats["rag_window"] == 1


def test_auto_routes_by_the_question_kind(chain_lib):
    """auto: the plan's kind (from the question's wording) picks the strategy; nothing a search found."""
    plan = [("Who lit the harbour lamp?", ["lamp"]), ("What did that person make?", [])]
    chain = StubClient(plan=plan, parts_synth=chain_synth, kind="chain")
    r = run(chain_lib, chain, strategy="auto", rag_k=2, rag_mode="keyword", rag_window=0)
    assert r.stats["strategy"] == "auto" and r.stats["routed"] == "rag-loop" and r.stats["question_kind"] == "chain"
    assert "finder" not in chain.calls and chain.calls.count("plan") == 1  # the routing plan is the plan
    assert r.found and "map of the northern coast" in r.answer

    judge = StubClient(kind="judgement", synth_answer="Tobias Wren [E1].")
    r = run(chain_lib, judge, strategy="auto")
    assert r.stats["routed"] == "single" and r.stats["question_kind"] == "judgement"
    assert "finder" in judge.calls and "brief" in judge.calls and judge.embedded == []


def test_auto_falls_back_to_rag_loop_when_the_plan_fails(chain_lib):
    class NoRoute(StubClient):
        def json(self, role, model, system, user, schema, temperature=0.0, num_ctx=None):
            out = super().json(role, model, system, user, schema, temperature, num_ctx)
            return None if schema is research.RoutedPlan else out

    client = NoRoute(plan=[("Who lit the harbour lamp?", ["lamp"])])
    r = run(chain_lib, client, strategy="auto", rag_k=2, rag_mode="keyword")
    assert r.stats["routed"] == "rag-loop" and r.stats["question_kind"] is None
    assert client.calls.count("plan") == 2  # the routed plan failed; rag-loop plans again


# -- judgement and collect-all answers: conclusion + cases, written out by the app ---------------


def wren_finder(user):
    return research.FinderPointers(sentences=sentence_with(user, "Tobias Wren"))


def cited_cases(user):
    """A partial or combined answer: one case per evidence number it was shown."""
    ns = sorted({int(n) for n in re.findall(r"\[E(\d+)\]", user)})
    return research.CaseSynthesis(found=True, conclusion="Wren is busy.",
                                  cases=[research.Case(point=f"Case {n}", citations=[n]) for n in ns])


def test_render_cases_puts_every_citation_inline():
    cs = research.CaseSynthesis(found=True, conclusion="Mostly at odds.",
                                cases=[research.Case(point="The Queen shouts [E9]", citations=[3, 3]),
                                       research.Case(point="  ", citations=[8]),
                                       research.Case(point="The Pigeon scolds", citations=[5])],
                                against=[research.Case(point="The Dodo gives a prize", citations=[7])])
    s = research.render_cases(cs)
    assert s.answer == ("Mostly at odds.\n\n- The Queen shouts [E3]\n- The Pigeon scolds [E5]"
                        "\n\n**Against:**\n- The Dodo gives a prize [E7]")
    assert s.citations == [3, 5, 7] and s.found  # an empty case and its number are dropped
    assert research.with_sources(s) == s.answer  # nothing left over for a Sources line
    none = research.render_cases(research.CaseSynthesis(found=False, conclusion="Not in the evidence.", cases=[]))
    assert not none.found and none.answer == "Not in the evidence." and none.citations == []


def test_auto_judgement_is_answered_as_cases(chain_lib):
    client = StubClient(finder=wren_finder, kind="judgement", cases=cited_cases)
    r = run(chain_lib, client, strategy="auto")
    assert [s for role, s in client.schemas if role == "synthesis"] == [research.CaseSynthesis]
    assert r.found and r.answer == "Wren is busy.\n\n- Case 1 [E1]\n- Case 2 [E2]" and r.cited == [1, 2]


def test_batched_judgement_keeps_cases_and_citations_through_the_combine(chain_lib, monkeypatch):
    monkeypatch.setattr(research, "MAX_EVIDENCE_WORDS", 5)  # one passage per batch: 2 readings + a combine
    client = StubClient(finder=wren_finder, kind="judgement", cases=cited_cases)
    r = run(chain_lib, client, strategy="auto")
    synth = [(role, s) for role, s in client.schemas if role == "synthesis"]
    assert synth == [("synthesis", research.CaseSynthesis)] * 3
    combine = [u for role, u in client.prompts if role == "synthesis"][-1]
    assert "Partial answer A" in combine and "- Case 1 [E1]" in combine  # partials arrive written out
    assert r.answer == "Wren is busy.\n\n- Case 1 [E1]\n- Case 2 [E2]" and "Sources" not in r.answer


def test_auto_chain_keeps_the_parts_answer(chain_lib):
    plan = [("Who lit the harbour lamp?", ["lamp"]), ("What did that person make?", [])]
    client = StubClient(plan=plan, parts_synth=chain_synth, kind="chain")
    run(chain_lib, client, strategy="auto", rag_k=2, rag_mode="keyword", rag_window=0)
    assert {s for role, s in client.schemas if role == "synthesis"} == {research.PartsSynthesis}


# -- the app's defaults -------------------------------------------------------------------------


def test_app_defaults_to_auto_and_the_benchmarked_researcher():
    import settings
    assert settings.DEFAULTS["research_strategy"] == research.DEFAULT_STRATEGY == "auto"
    assert research.DEFAULT_RESEARCHER == "qwen3.5:9b"


def test_app_search_options():
    have = {retrieval.EMBED_MODEL, "qwen3.5:9b"}
    assert research.app_search_options("auto", have) == {"strategy": "auto", "rag_mode": None, "flag": None}
    assert research.app_search_options("rag", have)["strategy"] == "auto"  # benchmark controls aren't offered
    assert research.app_search_options(None, have)["strategy"] == "auto"
    assert research.app_search_options("single", set()) == {"strategy": "single", "rag_mode": None, "flag": None}
    no_embedder = research.app_search_options("auto", {"qwen3.5:9b"})
    assert no_embedder["rag_mode"] == "keyword" and retrieval.EMBED_MODEL in no_embedder["flag"]


def test_first_dense_search_says_it_is_indexing(chain_lib):
    stages = []
    client = StubClient(plan=[("Who lit the harbour lamp?", ["lamp"])], embed=lighthouse_vector)
    run(chain_lib, client, strategy="rag-loop", rag_k=1, on_progress=lambda s, d, t: stages.append(s))
    assert stages.count("index") == 1 and "retrieve" in stages
    stages.clear()
    run(chain_lib, StubClient(plan=[("Who lit the harbour lamp?", ["lamp"])], embed=lighthouse_vector),
        strategy="rag-loop", rag_k=1, on_progress=lambda s, d, t: stages.append(s))
    assert "index" not in stages  # cached in the library now
    stages.clear()
    run(chain_lib, StubClient(plan=[("Who lit the harbour lamp?", ["lamp"])]), strategy="rag-loop", rag_k=1,
        rag_mode="keyword", on_progress=lambda s, d, t: stages.append(s))
    assert "index" not in stages
