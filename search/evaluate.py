"""Held-out evaluation and timing.

K-fold cross-validation by patent: patents are shuffled into `folds` groups; for each fold, an index
is built from the other folds and each held-out patent's first live claim (claim 1 unless cancelled)
is a query against it. Every patent is a query exactly once. Metrics are computed on the patent ranking:
  agree_<level>  fraction of top-k patents sharing the query's CPC prefix down to <level>
  ndcg           nDCG@k with relevance = depth of the deepest shared CPC prefix (0-5)
  knn_<level>    majority vote of the top-k patents' label at <level> == query's label
  claim_overlap / patent_overlap   |beam top-k ∩ flat top-k| / k
Baselines: random retrieval (k random index patents) and majority class (kNN accuracy only).
"""
import time
from collections import Counter

import numpy as np
import pandas as pd

from .data import CPC_LEVELS
from .index import build_index

BS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
KS = (1, 5, 10)


def shared_depth(a: tuple, b: tuple) -> int:
    d = 0
    for x, y in zip(a, b):
        if x != y:
            break
        d += 1
    return d


def cpc_metrics(qpath: tuple, paths: list[tuple], k: int, ideal: list[int]) -> dict:
    paths = paths[:k]
    rels = [shared_depth(qpath, p) for p in paths]
    row = {f"agree_{name}": np.mean([r > lvl for r in rels]) for lvl, name in enumerate(CPC_LEVELS)}
    discount = 1 / np.log2(np.arange(2, k + 2))
    row["ndcg"] = float(np.dot(rels, discount[:len(rels)]) / np.dot(ideal[:k], discount[:len(ideal[:k])]))
    for lvl, name in enumerate(CPC_LEVELS):
        labels = [p[lvl] for p in paths]
        votes = Counter(labels)
        pred = max(labels, key=lambda l: (votes[l], -labels.index(l)))  # ties -> higher-ranked label
        row[f"knn_{name}"] = float(pred == qpath[lvl])
    return row


def run_eval(patents: list[dict], embedder, folds: int = 5, seed: int = 0, Bs=BS, ks=KS,
             n_random: int = 20, timing_repeats: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (per-query metrics, metrics averaged over queries, timing table). Timing uses the fold-0 index."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(patents), dtype=int)
    fold_of[rng.permutation(len(patents))] = np.arange(len(patents)) % folds
    rows, timing = [], None
    for fold in range(folds):
        test = [p for p, f in zip(patents, fold_of) if f == fold]
        train = [p for p, f in zip(patents, fold_of) if f != fold]
        index, build_t = build_index(train, embedder)
        query_texts = [p["claims"][0]["text"] for p in test]
        queries = embedder.embed(query_texts)
        print(f"fold {fold}: {len(train)} index patents ({len(index.tree.leaves)} independent claims), {len(test)} queries")
        rows += _eval_fold(index, train, test, queries, Bs, ks, n_random, rng, fold)
        if timing is None:
            timing = _timing(index, build_t, embedder, query_texts, queries, Bs, timing_repeats)
    per_query = pd.DataFrame(rows)
    per_query["B"] = per_query["B"].astype("Int64")
    metrics = (per_query.drop(columns=["query", "fold"]).groupby(["method", "B", "k"], dropna=False, sort=False)
               .mean().reset_index())
    metrics["B"] = metrics["B"].astype("Int64")
    return per_query, metrics, timing


def _eval_fold(index, train, test, queries, Bs, ks, n_random, rng, fold) -> list[dict]:
    tree = index.tree
    train_paths = [p["path"] for p in train]
    majority = [Counter(p[lvl] for p in train_paths).most_common(1)[0][0] for lvl in range(len(CPC_LEVELS))]
    rows = []
    for p, q in zip(test, queries):
        ideal = sorted((shared_depth(p["path"], t) for t in train_paths), reverse=True)
        flat = index.flat_search(q)
        flat_paths = [tree.patent[n]["path"] for n in flat.patents]
        beams = {B: index.beam_search(q, B) for B in Bs}
        base = {"query": p["doc_number"], "fold": fold}
        for k in ks:
            rows.append({**base, "method": "flat", "B": None, "k": k, **cpc_metrics(p["path"], flat_paths, k, ideal)})
            for B, res in beams.items():
                if k > B:
                    continue
                row = cpc_metrics(p["path"], [tree.patent[n]["path"] for n in res.patents], k, ideal)
                row["claim_overlap"] = len(set(res.claims[:k]) & set(flat.claims[:k])) / k
                row["patent_overlap"] = len(set(res.patents[:k]) & set(flat.patents[:k])) / k
                rows.append({**base, "method": "beam", "B": B, "k": k, **row})
            rand = [cpc_metrics(p["path"], [train_paths[i] for i in rng.choice(len(train), k, replace=False)], k, ideal)
                    for _ in range(n_random)]
            rows.append({**base, "method": "random", "B": None, "k": k, **pd.DataFrame(rand).mean().to_dict()})
            rows.append({**base, "method": "majority class", "B": None, "k": k,
                         **{f"knn_{name}": float(majority[lvl] == p["path"][lvl]) for lvl, name in enumerate(CPC_LEVELS)}})
    return rows


def _timing(index, build_t, embedder, query_texts, queries, Bs, repeats) -> pd.DataFrame:
    """Warm timing on one index; each call timed separately over `repeats` passes of the queries."""
    def latency(fn, inputs, repeats=repeats) -> dict:
        fn(inputs[0])  # warm up
        ms = []
        for _ in range(repeats):
            for x in inputs:
                t0 = time.perf_counter()
                fn(x)
                ms.append(1000 * (time.perf_counter() - t0))
        return {"ms_mean": np.mean(ms), "ms_median": np.median(ms), "ms_p95": np.percentile(ms, 95), "n": len(ms)}

    timing = [{"operation": "embed independent claims (fresh, all claims ever cached)", "B": None,
               "ms_mean": 1000 * embedder.meta["embed_seconds"], "n": embedder.meta["n_embedded"]},
              {"operation": "build tree", "B": None, "ms_mean": 1000 * build_t["tree_s"], "n": 1},
              {"operation": "aggregate keys", "B": None, "ms_mean": 1000 * build_t["keys_s"], "n": 1},
              {"operation": "embed query", "B": None,
               **latency(lambda t: embedder.model.encode([t], normalize_embeddings=True), query_texts, repeats=1)},
              {"operation": "flat search", "B": None, **latency(lambda q: index.flat_search(q, 10), queries)}]
    for B in Bs:
        timing.append({"operation": "beam search", "B": B, **latency(lambda q: index.beam_search(q, B), queries)})
    timing = pd.DataFrame(timing)
    timing["B"] = timing["B"].astype("Int64")
    return timing
