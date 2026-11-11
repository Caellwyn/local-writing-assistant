"""Retrieval on its own: how much of the needed evidence each retrieval setup delivers, and at what
cost in words, without running any researcher. Fast (seconds per setup once the index exists), so
retrieval ideas can be screened before they get GPU time in a full round.

    .venv\\Scripts\\python -I -X utf8 evals\\retrieval_check.py [--embed-windows 0 1 2] [--modes ...]

Uses the library round's corpus (whole books) and keeps its library and embedding cache between runs.
Dense and fused search need an embedding index per context window. One that isn't cached is skipped
unless --build (on the GPU, a few minutes; with --cpu, slowly on the CPU). Keyword search needs no index. Writes
evals/runs/checks/retrieval-<date-time>.json and prints a table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import research  # noqa: E402
import retrieval  # noqa: E402
import run_eval  # noqa: E402

# Setups compared at roughly equal cost: k sentences, each delivered with ±window neighbours.
BUDGETS = [(100, 0), (50, 1), (30, 2), (200, 0), (100, 1), (60, 2)]


def hit(span, n) -> bool:
    doc_id, _, a, b = span
    return doc_id == n["doc_id"] and a < n["end"] and b > n["start"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=list(retrieval.MODES))
    p.add_argument("--embed-windows", nargs="+", type=int, default=[0, 1])
    p.add_argument("--build", action="store_true", help="build missing embedding indexes (GPU)")
    p.add_argument("--cpu", action="store_true", help="with --build: on the CPU (slow, leaves the GPU alone)")
    args = p.parse_args()

    lib, spec, needles, questions = run_eval.corpus_library("library", full=True, reuse=True)
    docs = research._scope(lib, None)
    texts = {d.id: research.library.read_text(d.path) for d in docs}
    units = {d.id: research.split_sentences(texts[d.id]) for d in docs}
    client = research.ModelClient()
    qs = [q for q in questions if q.get("needles") and q.get("kind") not in ("unanswerable", "bait")]
    rows = []
    for ew in args.embed_windows:
        cached = all(retrieval._cache_path(lib, d.id, texts[d.id], retrieval.EMBED_MODEL, ew).exists() for d in docs)
        modes = [m for m in args.modes if m == "keyword" or cached or args.build]
        if len(modes) < len(args.modes):
            print(f"index ±{ew} isn't built: skipping dense and fused (--build to build it)")
        if any(m != "keyword" for m in modes):
            client.embed_gpu = not args.cpu
            t = time.time()
            for d in docs:
                retrieval.doc_vectors(client, lib, d.id, texts[d.id], units[d.id], retrieval.EMBED_MODEL, ew)
            client.embed_gpu = False
            print(f"index ±{ew}: {time.time() - t:.0f} s", flush=True)
        for mode in modes:
            for k, w in BUDGETS:
                per_kind = defaultdict(lambda: [0, 0])  # kind -> [needles delivered, needles]
                full = defaultdict(lambda: [0, 0])  # kind -> [questions with every needle, questions]
                words = []
                cases = {}
                t = time.time()
                for q in qs:
                    hits = retrieval.search(client, lib, docs, texts, units, q["question"], k,
                                            retrieval.EMBED_MODEL, mode, ew)
                    pool = retrieval.widen(hits, units, w)
                    words.append(sum(len(texts[d][a:b].split()) for d, _, a, b in pool))
                    got = [any(hit(s, needles[nid]) for s in pool) for nid in q["needles"]]
                    kind = q["kind"]
                    if kind == "aggregate":
                        cases[q["id"]] = f"{sum(got)}/{len(got)}"
                        continue
                    kind = "chain" if kind.startswith("chain") else kind
                    per_kind[kind][0] += sum(got)
                    per_kind[kind][1] += len(got)
                    full[kind][0] += all(got)
                    full[kind][1] += 1
                facts = sum(v[0] for v in per_kind.values()) / sum(v[1] for v in per_kind.values())
                answerable = sum(v[0] for v in full.values()) / sum(v[1] for v in full.values())
                rows.append({"mode": mode, "embed_window": ew, "k": k, "window": w,
                             "facts_delivered": round(facts, 3), "questions_complete": round(answerable, 3),
                             "by_kind": {kd: f"{v[0]}/{v[1]}" for kd, v in sorted(full.items())},
                             "cases": cases, "words_median": sorted(words)[len(words) // 2],
                             "seconds_per_q": round((time.time() - t) / len(qs), 2)})
                r = rows[-1]
                print(f"{mode:8} index±{ew} k={k:<3} delivered±{w}  facts {facts:5.0%}  complete {answerable:5.0%}  "
                      f"{' '.join(f'{kd} {v}' for kd, v in r['by_kind'].items())}  "
                      f"cases {' '.join(f'{a} {c}' for a, c in cases.items())}  words {r['words_median']}", flush=True)
    out = run_eval.RUNS / "checks" / f"retrieval-{dt.datetime.now():%Y-%m-%d-%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"questions": [q["id"] for q in qs], "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
