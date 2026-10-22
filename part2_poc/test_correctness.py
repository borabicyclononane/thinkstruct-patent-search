"""Correctness of the sharded search against the unsharded one (S=1) and against brute force.

  .venv/bin/python3.14 part2_poc/test_correctness.py      (after build.py)

Queries: 100 synthetic test-split queries (Part 3, embeddings in training/bundles/full.npz), 25 per type:
no filter, CPC prefix (the source patent's subclass or main group), keywords (1-2 terms from the source patent's
title, "or" / "and"), and both. Checks, for range and hash sharding at S=2 and 4:
  1. results equal S=1 (same patents, claims and order; scores to 1e-9)
  2. per-node match counts summed over shards equal S=1, and equal brute force over the raw token sets
  3. every returned patent satisfies all filters (brute force: CPC path prefix, terms in title/abstract)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build import BUNDLE, STOPWORDS, load, load_built  # noqa: E402
from coordinator import Coordinator  # noqa: E402
from search.filters import index_terms, query_terms  # noqa: E402

TYPES = ["no filter", "CPC prefix", "keyword", "both"]
K, B = 5, 10


def make_queries(pats, n: int = 100, seed: int = 0) -> list[dict]:
    """[{qi, type, cpc_prefix, keywords, mode}] over test-split synthetic queries; filters drawn from the source patent."""
    b = np.load(BUNDLE)
    by_doc = {p["doc"]: p for p in pats}
    rng = np.random.default_rng(seed)
    qis = rng.choice(np.flatnonzero(b["query_split"] == "test"), n, replace=False)
    out = []
    for i, qi in enumerate(qis):
        src = by_doc[str(b["claim_ids"][b["answers"][qi][0]]).split("#")[0]]
        typ = TYPES[i % 4]
        q = {"qi": int(qi), "type": typ, "cpc_prefix": None, "keywords": None, "mode": "or"}
        if typ in ("CPC prefix", "both"):
            q["cpc_prefix"] = src["path"][2 + (i // 4) % 2]  # subclass or main group
        if typ in ("keyword", "both"):
            words = sorted(index_terms(src["title"]) - STOPWORDS)
            q["mode"] = ["or", "and"][(i // 8) % 2]
            q["keywords"] = list(rng.choice(words, min(len(words), 1 + (i // 8) % 2 + (q["mode"] == "and")), replace=False))
        out.append(q)
    return out


def satisfies(p: dict, q: dict) -> bool:
    if q["cpc_prefix"] and q["cpc_prefix"] not in p["path"]:
        return False
    if q["keywords"]:
        has = [t in p["terms"] for kw in q["keywords"] for t in query_terms(kw)]
        return all(has) if q["mode"] == "and" else any(has)
    return True


def main():
    pats, _ = load()
    by_doc = {p["doc"]: p for p in pats}
    queries = make_queries(pats)
    qv = np.load(BUNDLE)["query_vecs"]
    base = Coordinator(load_built("range", 1))
    all_nodes = np.arange(len(base.name))
    base_res = [base.search(qv[q["qi"]], K, B, q["cpc_prefix"], q["keywords"], q["mode"])[0] for q in queries]
    kw = [q for q in queries if q["keywords"]]
    base_counts = [base.counts(all_nodes, base.terms(q["keywords"]), q["mode"]) for q in kw]
    brute = [np.array([sum(satisfies(pats[i], {**q, "cpc_prefix": None}) for i in range(s, e))
                       for s, e in zip(base.start, base.end)]) for q in kw]
    print(f"{len(queries)} queries ({', '.join(f'{sum(q['type'] == t for q in queries)} {t}' for t in TYPES)}), "
          f"k={K}, B={B}; {sum(len(r) for r in base_res)} results at S=1")

    ok_all = True
    def report(name, bad, total):
        nonlocal ok_all
        ok_all &= bad == 0
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {name}: {total - bad}/{total}")

    print("S=1 counts vs brute force")
    report("per-node counts equal brute force", sum(not np.array_equal(c, g) for c, g in zip(base_counts, brute)), len(kw))
    report("returned patents satisfy filters", sum(not satisfies(by_doc[r["doc"]], q)
                                                   for q, res in zip(queries, base_res) for r in res),
           sum(len(r) for r in base_res))
    for scheme in ("range", "hash"):
        for S in (2, 4):
            c = Coordinator(load_built(scheme, S))
            res = [c.search(qv[q["qi"]], K, B, q["cpc_prefix"], q["keywords"], q["mode"])[0] for q in queries]
            same = lambda a, b: [(x["doc"], x["claim"]) for x in a] == [(x["doc"], x["claim"]) for x in b] and \
                np.allclose([x["score"] for x in a], [x["score"] for x in b], rtol=0, atol=1e-9)
            print(f"{scheme} S={S}")
            report("results equal S=1", sum(not same(a, b) for a, b in zip(res, base_res)), len(queries))
            report("per-node counts summed over shards equal S=1",
                   sum(not np.array_equal(c.counts(all_nodes, c.terms(q["keywords"]), q["mode"]), bc)
                       for q, bc in zip(kw, base_counts)), len(kw))
            report("returned patents satisfy filters", sum(not satisfies(by_doc[r["doc"]], q)
                                                           for q, rs in zip(queries, res) for r in rs),
                   sum(len(r) for r in res))
    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
