"""Build the evaluation corpus: public-domain books with planted "needle" facts.

Downloads three Project Gutenberg texts, strips the Gutenberg header/footer,
keeps a slice of each, and inserts the invented facts from questions.json at
paragraph boundaries. Because the facts are invented, a correct answer can
only come from reading the documents, never from a model's memory.

Output (gitignored): evals/.corpus/<name>.txt and evals/.corpus/needles.json

    .venv\\Scripts\\python evals\\build_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CORPUS = HERE / ".corpus"
RAW = CORPUS / "raw"
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def strip_gutenberg(text: str) -> str:
    text = text.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text, re.I)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    if not start or not end:
        raise ValueError("Gutenberg markers not found")
    return text[start.end():end.start()].strip()


def word_slice(text: str, skip: int, words: int) -> str:
    """Cut text from word `skip` for about `words` words, snapped to paragraph breaks."""
    spans = [m.start() for m in re.finditer(r"\S+", text)]
    begin = spans[min(skip, len(spans) - 1)]
    stop = spans[min(skip + words, len(spans) - 1)]
    begin = text.rfind("\n\n", 0, begin) + 2 if text.rfind("\n\n", 0, begin) != -1 else begin
    nxt = text.find("\n\n", stop)
    return text[begin: nxt if nxt != -1 else len(text)].strip()


def plant(text: str, needles: list[tuple[float, str]]) -> str:
    """Insert each needle as its own paragraph at the first paragraph break after `at`."""
    for at, needle in sorted(needles, reverse=True):  # back to front keeps positions valid
        pos = text.find("\n\n", int(len(text) * at))
        pos = len(text) if pos == -1 else pos
        text = text[:pos] + "\n\n" + needle + text[pos:]
    return text


def cases(lib_spec: dict, texts: dict[str, str]) -> dict:
    """The labelled cases of each aggregate question (evals/aggregate/*.json), located like needles.
    Each quote is one case; "group" is what it's a case of (a habit, or one encounter)."""
    out = {}
    for agg, a in lib_spec["aggregates"].items():
        files = a.get("files") or [a["file"]]
        text = texts[a["doc"]]
        rows = []
        for f in files:
            data = json.loads((HERE / f).read_text(encoding="utf-8"))
            if "encounters" in data:  # one group per encounter, several quotes each
                for i, e in enumerate(data["encounters"]):
                    group = f"ch{e['chapter']} {e['character']} ({e['stance']})"
                    rows += [(group, q) for q in e["quotes"]]
            else:
                rows += [(inst["habit"], inst["quote"]) for inst in data["instances"]]
        for i, (group, quote) in enumerate(rows):
            start = text.find(quote)
            assert start != -1 and text.find(quote, start + 1) == -1, (agg, quote)
            out[f"{agg}:{i}"] = {"doc": a["doc"], "start": start, "end": start + len(quote), "case_of": agg,
                                 "group": group, "distractor": False, "injection": False, "canary": None}
        print(f"{agg}: {len(rows)} cases in {len({g for g, _ in rows})} groups")
    return out


def main() -> int:
    full = "--full" in sys.argv  # the library round: whole books, plus evals/library.json
    spec = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    lib_spec = json.loads((HERE / "library.json").read_text(encoding="utf-8")) if full else None
    out_dir = HERE / ".corpus-full" if full else CORPUS
    out_dir.mkdir(parents=True, exist_ok=True)
    if full:
        spec["needles"] = {**spec["needles"], **lib_spec["needles"]}
    RAW.mkdir(parents=True, exist_ok=True)
    located = {}
    texts = {}
    for name, d in spec["documents"].items():
        raw = RAW / f"pg{d['gutenberg_id']}.txt"
        if not raw.exists():
            print(f"downloading {name} (Gutenberg #{d['gutenberg_id']})…")
            r = httpx.get(URL.format(id=d["gutenberg_id"]), timeout=60, follow_redirects=True,
                          headers={"User-Agent": "local-writing-assistant-evals"})
            r.raise_for_status()
            raw.write_bytes(r.content)  # bytes as served; text mode would double the \r on Windows
        body = strip_gutenberg(raw.read_bytes().decode("utf-8-sig"))
        if not full:
            body = word_slice(body, d["skip_words"], d["words"])
        # a "lead" sentence goes in the same paragraph, before the fact (context needles)
        mine = [(n["at"], (n["lead"] + " " if n.get("lead") else "") + n["text"])
                for n in spec["needles"].values() if n["doc"] == name]
        text = plant(body, mine)
        texts[name] = text
        # newline="\n": needle offsets must match the bytes the library will read
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
        for nid, n in spec["needles"].items():
            if n["doc"] == name:
                start = text.find(n["text"])
                assert start != -1, nid
                located[nid] = {"doc": name, "start": start, "end": start + len(n["text"]),
                                "distractor": n.get("distractor", False),
                                "injection": n.get("injection", False), "canary": n.get("canary")}
        print(f"{name}: {len(text.split()):,} words, {len(mine)} needles")
    if full:
        located.update(cases(lib_spec, texts))
    (out_dir / "needles.json").write_text(json.dumps(located, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
