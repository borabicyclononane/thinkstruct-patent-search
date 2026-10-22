"""Hybrid search: latency with and without filters, and correctness checks.

Queries: the first live claim of every patent, against an index of all patents. For each filter scenario:
  flat post-filter   score every claim, then drop non-candidates (same output as pre-filter, more work)
  flat pre-filter    score only the candidates' claims
  beam pruned        start at the CPC nodes, score only children that hold a candidate
plus unfiltered flat and beam as references. Filtered timings include resolving the filter.

Checks (raise on failure):
  pre-filter == post-filter (full ranking, claims and scores, up to float ties)
  every patent beam search returns satisfies the filter, re-checked from the raw text, not the inverted index
  pruned beam with unlimited B == flat pre-filter (full ranking)
"""
import time

import numpy as np
import pandas as pd

from .index import l2n
from .filters import Filter, TextIndex, hybrid_search, index_terms, normalize_cpc, normalize_title, query_terms

SCENARIOS = [
    ("CPC subclass B60C", Filter(cpc=("B60C",))),
    ("CPC subgroup B60B 27/00", Filter(cpc=("B60B 27/00",))),
    ("CPC B60B 33 or B60D", Filter(cpc=("B60B 33", "B60D"))),
    ("keyword: tire", Filter(all_of=("tire",))),
    ("keyword: hub", Filter(all_of=("hub",))),
    ("keyword: non-pneumatic", Filter(all_of=("non-pneumatic",))),
    ("keyword: hitch", Filter(all_of=("hitch",))),
    ("tire AND pressure", Filter(all_of=("tire", "pressure"))),
    ("stud OR hitch", Filter(any_of=("stud", "hitch"))),
    ("title = 'pneumatic tire'", Filter(title="pneumatic tire")),
    ("B60C 11 + tread AND noise", Filter(cpc=("B60C 11",), all_of=("tread", "noise"))),
]


def matches(p: dict, f: Filter) -> bool:
    """The filter, evaluated directly on a patent's text (independent of the inverted index)."""
    path = set(p["path"])
    if f.cpc and not any(normalize_cpc(c) in path for c in f.cpc):
        return False
    terms = {fld: index_terms(p[fld]) for fld in f.fields}
    has = lambda kw: any(set(query_terms(kw)) <= terms[fld] for fld in f.fields)
    if not all(map(has, f.all_of)) or (f.any_of and not any(map(has, f.any_of))):
        return False
    return f.title is None or normalize_title(p["title"]) == normalize_title(f.title)


def flat_post_filter(index, text, q, f, k):
    starts, cand = text.resolve(f)
    if cand is None:
        cand = text.cpc_positions(starts)
    scores = index.score(l2n(index.project(q[None]))[0], index.leaf_K)  # every claim
    keep = np.zeros(len(index.tree.order), dtype=bool)
    keep[cand] = True
    m = keep[index.tree.start[index.tree.leaves]]
    return index._result(index.tree.leaves[m], scores[m], k)


def same(a, b, tol=1e-6) -> bool:
    """Same ranking up to float ties: same claims, each with the same score, and the same score at every rank.
    (float32 matmul over a subset can differ from the full matrix in the last bit, swapping exact ties.)"""
    sa, sb = dict(zip(a.claims.tolist(), a.claim_scores)), dict(zip(b.claims.tolist(), b.claim_scores))
    return (sa.keys() == sb.keys() and all(abs(sa[c] - sb[c]) <= tol for c in sa)
            and np.allclose(a.claim_scores, b.claim_scores, atol=tol, rtol=0))


def run_hybrid(index, queries: np.ndarray, B: int = 10, k: int = 10, repeats: int = 3, scenarios=SCENARIOS, extra_Bs=(1, 5)):
    """-> (timing table, check summary). `extra_Bs`: beam widths timed in addition to B (methods 'beam B=<b>')."""
    text = TextIndex(index)
    tree = index.tree

    def latency(fn) -> dict:
        fn(queries[0])
        ms = []
        for _ in range(repeats):
            for q in queries:
                t0 = time.perf_counter()
                fn(q)
                ms.append(1000 * (time.perf_counter() - t0))
        return {"ms_median": np.median(ms), "ms_p95": np.percentile(ms, 95)}

    rows = [{"scenario": "no filter", "method": "flat", "n_candidates": len(tree.order),
             **latency(lambda q: index.flat_search(q, k))},
            {"scenario": "no filter", "method": "beam", "n_candidates": len(tree.order),
             **latency(lambda q: index.beam_search(q, B))}]
    rows += [{"scenario": "no filter", "method": f"beam B={b}", "n_candidates": len(tree.order),
              **latency(lambda q, b=b: index.beam_search(q, b))} for b in extra_Bs]
    checks = []
    for name, f in scenarios:
        _, n = hybrid_search(index, text, queries[0], f, k, B)
        rows += [{"scenario": name, "method": "resolve filter only", "n_candidates": n, **latency(lambda q: text.resolve(f))},
                 {"scenario": name, "method": "flat post-filter", "n_candidates": n,
                  **latency(lambda q: flat_post_filter(index, text, q, f, k))},
                 {"scenario": name, "method": "flat pre-filter", "n_candidates": n,
                  **latency(lambda q: hybrid_search(index, text, q, f, k, B, flat=True))},
                 {"scenario": name, "method": "beam pruned", "n_candidates": n,
                  **latency(lambda q: hybrid_search(index, text, q, f, k, B))}]
        rows += [{"scenario": name, "method": f"beam pruned B={b}", "n_candidates": n,
                  **latency(lambda q, b=b: hybrid_search(index, text, q, f, k, b))} for b in extra_Bs]
        bad = mismatch = overlap = 0
        for q in queries:
            pre, _ = hybrid_search(index, text, q, f, None, flat=True)
            assert same(pre, flat_post_filter(index, text, q, f, None)), f"{name}: pre-filter != post-filter"
            beam, _ = hybrid_search(index, text, q, f, k, B)
            bad += sum(not matches(tree.patent[p], f) for p in beam.patents)
            mismatch += not same(hybrid_search(index, text, q, f, None, 10 ** 9)[0], pre)
            overlap += len(set(beam.patents[:k]) & set(pre.patents[:k])) / max(1, min(k, n))
        assert bad == 0, f"{name}: {bad} beam results violate the filter"
        assert mismatch == 0, f"{name}: unlimited beam != pre-filter on {mismatch} queries"
        assert n == sum(matches(p, f) for p in tree.patent.values()), f"{name}: candidate count != direct count"
        checks.append({"scenario": name, "n_candidates": n, "violations": bad, "unlimited_beam_mismatches": mismatch,
                       f"beam_patent_overlap@{k}": overlap / len(queries)})
    return pd.DataFrame(rows), pd.DataFrame(checks)
