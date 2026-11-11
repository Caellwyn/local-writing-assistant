"""Retrieval for the research pipeline: sentence-level search by embeddings, keywords, or both.

Every sentence of the library (research.split_sentences: the same units the finders point at) is a
search unit. Three ways to rank them for a question (search(mode=)):

  dense    cosine similarity of embeddings (Qwen3-Embedding), cached per document in the library
  keyword  BM25 over the words of each unit: exact names and rare words that embeddings blur
  fused    both, combined by reciprocal rank fusion (each list votes 1/(60 + rank))

Context options:
  embed_window  index each sentence together with its neighbours (embedding and keywords), so a
                sentence that says only "She paid four guineas" is found by the name in the sentence
                before it. The unit returned is still the one sentence, so citations stay precise.
  widen()       deliver each hit with its neighbouring sentences, so the researcher reads it in context.

Used by research.find_facts (strategies rag, hybrid and rag-loop). The app itself doesn't use it yet.
"""

from __future__ import annotations

import array
import base64
import hashlib
import heapq
import json
import math
import operator
import re
from collections import Counter

from library import Library, safe_path

EMBED_MODEL = "qwen3-embedding:0.6b"  # small and strong for its size; runs on the CPU (research.EMBED_OPTIONS)
RAG_K = 25  # sentences per question; the finders delivered 23 on average (median 7, max ~110)
INDEX_VERSION = "sentences-v1"  # part of the cache key
MODES = ("dense", "keyword", "fused")
RRF_K = 60  # reciprocal rank fusion constant (the usual value; it damps the top ranks' dominance)
BM25_K1, BM25_B = 1.2, 0.75
# Qwen3-Embedding is trained with an instruction on the query side only
QUERY_PREFIX = "Instruct: Given a question, retrieve sentences from books that help answer it\nQuery: "

Span = tuple[str, str, int, int]  # (doc_id, title, start, end), as research.Span

# Question words and function words: they match everything, so they only add noise to keyword search.
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours yourself yourselves
also ever every many much one ones upon shall may might must said says say tell told according book books
""".split())
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def unit_text(text: str, units: list[tuple[int, int]], i: int, window: int = 0) -> str:
    """Unit i's text, with `window` sentences either side when window > 0 (one line, spaces collapsed)."""
    a = units[max(0, i - window)][0]
    b = units[min(len(units) - 1, i + window)][1]
    return " ".join(text[a:b].split())


def tokens(text: str) -> list[str]:
    """Lower-case content words, possessives and plain plurals folded ("Holmes's" → "holmes",
    "habits" → "habit"), so a question's words meet the text's other forms of them."""
    out = []
    for w in _WORD.findall(text.lower().replace("’", "'")):
        w = w.removesuffix("'s").removesuffix("'")
        if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        if len(w) > 1 and w not in STOPWORDS:
            out.append(w)
    return out


def _cache_path(lib: Library, doc_id: str, text: str, model: str, window: int = 0):
    version = INDEX_VERSION if not window else f"{INDEX_VERSION}-window{window}"
    digest = hashlib.sha256(f"{text}\0{model}\0{version}".encode("utf-8")).hexdigest()[:20]
    path = safe_path(lib.root, f".cache/embeddings/{doc_id}-{digest}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def indexed(lib: Library, docs: list, texts: dict[str, str], model: str = EMBED_MODEL, window: int = 0) -> bool:
    """Whether every document's embeddings are already cached (if not, the next dense search embeds
    them, which takes minutes on the CPU for a big library: worth telling the user)."""
    return all(_cache_path(lib, d.id, texts[d.id], model, window).exists() for d in docs)


_VEC_CACHE: dict[str, list] = {}  # cache file -> vectors, so each index is read from disk once per process


def doc_vectors(client, lib: Library, doc_id: str, text: str, units: list[tuple[int, int]],
                model: str, window: int = 0) -> list:
    """Unit-length vectors (compact float arrays) for a document's sentences, each with `window`
    neighbours either side: embedded once, cached in the library, and kept in memory once read."""
    path = _cache_path(lib, doc_id, text, model, window)
    key = str(path)
    if key in _VEC_CACHE and len(_VEC_CACHE[key]) == len(units):
        return _VEC_CACHE[key]
    if len(_VEC_CACHE) > 12:
        _VEC_CACHE.clear()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["count"] == len(units):
            flat = array.array("f")
            flat.frombytes(base64.b64decode(data["vectors"]))
            dim = data["dim"]
            _VEC_CACHE[key] = [flat[i * dim:(i + 1) * dim] for i in range(data["count"])]
            return _VEC_CACHE[key]
    vecs = [_normalize(v) for v in client.embed(model, [unit_text(text, units, i, window) for i in range(len(units))])]
    dim = len(vecs[0]) if vecs else 0
    flat = array.array("f", [x for v in vecs for x in v])
    path.write_text(json.dumps({"model": model, "count": len(vecs), "dim": dim,
                                "vectors": base64.b64encode(flat.tobytes()).decode("ascii")}), encoding="utf-8")
    _VEC_CACHE[key] = [flat[i * dim:(i + 1) * dim] for i in range(len(vecs))]
    return _VEC_CACHE[key]


class KeywordIndex:
    """BM25 over a list of unit texts, with an inverted index so a query touches only matching units."""

    def __init__(self, texts: list[str]):
        self.tf = [Counter(tokens(t)) for t in texts]
        self.length = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.length) / len(self.length)) if self.length else 0.0
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for w in c:
                self.postings.setdefault(w, []).append(i)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for w in set(tokens(query)):
            for i in self.postings.get(w, ()):
                f = self.tf[i][w]
                norm = f + BM25_K1 * (1 - BM25_B + BM25_B * self.length[i] / (self.avg or 1))
                out[i] = out.get(i, 0.0) + self.idf[w] * f * (BM25_K1 + 1) / norm
        return out


_KEYWORD_CACHE: dict[tuple, KeywordIndex] = {}


def _keyword_index(docs: list, texts: dict[str, str], units: dict[str, list[tuple[int, int]]],
                   window: int) -> KeywordIndex:
    key = tuple((d.id, hashlib.sha256(texts[d.id].encode("utf-8")).hexdigest()[:16]) for d in docs) + (window,)
    if key not in _KEYWORD_CACHE:
        if len(_KEYWORD_CACHE) > 8:
            _KEYWORD_CACHE.clear()
        _KEYWORD_CACHE[key] = KeywordIndex([unit_text(texts[d.id], units[d.id], i, window)
                                            for d in docs for i in range(len(units[d.id]))])
    return _KEYWORD_CACHE[key]


def fuse(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: each ranking gives an item 1/(k + rank); higher total first."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: (-score[i], i))


def search(client, lib: Library, docs: list, texts: dict[str, str], units: dict[str, list[tuple[int, int]]],
           question: str, k: int = RAG_K, model: str = EMBED_MODEL, mode: str = "dense",
           embed_window: int = 0) -> set[Span]:
    """The k sentences that best match the question, across the documents (see MODES)."""
    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode: {mode}")
    cands: list[Span] = [(d.id, d.title, a, b) for d in docs for a, b in units[d.id]]
    if not cands:
        return set()
    rankings = []
    if mode in ("dense", "fused"):
        vecs = [v for d in docs for v in doc_vectors(client, lib, d.id, texts[d.id], units[d.id], model, embed_window)]
        q = _normalize(client.embed(model, [QUERY_PREFIX + question])[0])
        sims = [sum(map(operator.mul, q, v)) for v in vecs]
        n = len(cands) if mode == "dense" else min(len(cands), max(4 * k, 200))
        rankings.append(heapq.nlargest(n, range(len(cands)), key=lambda i: (sims[i], -i)))
    if mode in ("keyword", "fused"):
        s = _keyword_index(docs, texts, units, embed_window).scores(question)
        rankings.append(sorted(s, key=lambda i: (-s[i], i)))
    order = rankings[0] if len(rankings) == 1 else fuse(rankings)
    return {cands[i] for i in order[:k]}


def widen(hits: set[Span], units: dict[str, list[tuple[int, int]]], window: int) -> set[Span]:
    """Each hit plus `window` sentences either side, in the same document."""
    if window <= 0:
        return set(hits)
    out = set(hits)
    indexes: dict[str, dict[int, int]] = {}
    for doc_id, title, a, _ in hits:
        us = units.get(doc_id, [])
        index = indexes.setdefault(doc_id, {u[0]: i for i, u in enumerate(us)})
        if a not in index:
            continue
        i = index[a]
        out |= {(doc_id, title, s, e) for s, e in us[max(0, i - window):i + window + 1]}
    return out
