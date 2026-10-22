"""Part 1: beam search with width B vs flat search keeping the top B patents; latency, and accuracy on the default dataset.

  .venv/bin/python3.14 results/compare_part1.py                 # default (vehicle) dataset: accuracy + latency
  .venv/bin/python3.14 results/compare_part1.py --sample 10000  # 10^4 large-dataset sample: latency only

Default dataset: one index over all 640 patents; queries are the Part 3 test-split synthetic queries
(training/bundles/full.npz). Accuracy = share of queries whose source claim is among the claims returned: beam
search's claims, or the claims of flat search's top B patents. A combination counts only if both sources come back.
Sign test: paired, beam vs flat at the same B. Latency is timed on the same queries.
10^4 sample: held-out split as experiment-big-data/run.py (each query is a held-out patent's first independent
claim); latency only. Writes results/part1_<dataset>.csv.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiment-big-data"))
from search.data import load_corpus  # noqa: E402
from search.embed import Embedder  # noqa: E402
from search.index import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=None, help="large-dataset sample size (default: the vehicle dataset)")
ap.add_argument("--bundle", default="training/bundles/full.npz", help="default dataset only: the synthetic queries")
ap.add_argument("--queries", type=int, default=1000, help="sample only: held-out query patents")
ap.add_argument("--widths", default="1,5,10,100")
ap.add_argument("--repeats", type=int, default=3, help="timing passes over the queries")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.chdir(ROOT)
W = [int(w) for w in args.widths.split(",")]

rng = np.random.default_rng(args.seed)
DATASETS = ["all", "paraphrase", "element_reuse", "combination"]


def latency(fn, queries) -> float:
    fn(queries[0])  # warm up
    ms = []
    for _ in range(args.repeats):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            ms.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ms))


def sign_p(x, y):
    x, y = int(x), int(y)
    return min(1.0, 2 * sum(math.comb(x + y, i) for i in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


def runs(index):
    return {(s, B): (lambda q, B=B: index.flat_search(q, B)) if s == "flat" else (lambda q, B=B: index.beam_search(q, B))
            for B in W for s in ("flat", "beam")}


if args.sample:
    from sample import load_sample
    patents, _ = load_sample(args.sample)
    embedder = Embedder("data/index/embeddings_large.npz")
    test = set(rng.choice(len(patents), args.queries, replace=False).tolist())  # same split as experiment-big-data/run.py
    index, _ = build_index([p for i, p in enumerate(patents) if i not in test], embedder)
    queries = embedder.embed([patents[i]["claims"][0]["text"] for i in sorted(test)])
    print(f"{len(patents) - len(test)} index patents ({len(index.tree.leaves)} claims), {len(queries)} queries", flush=True)
    out = pd.DataFrame([{"B": B, "search": s, "ms_median": latency(run, queries)} for (s, B), run in runs(index).items()])
    name = f"sample{args.sample}"
else:
    patents, _ = load_corpus()
    index, _ = build_index(patents, Embedder())
    tree = index.tree
    b = np.load(args.bundle)
    qs = np.flatnonzero(b["query_split"] == "test")
    queries, ds = b["query_vecs"][qs], b["query_dataset"][qs]
    leaf_of = {tree.name[l]: l for l in tree.leaves}
    assert set(leaf_of) == set(b["claim_ids"]), "index claims differ from the bundle's"
    print(f"{len(patents)} index patents ({len(tree.leaves)} claims), {len(qs)} test queries", flush=True)
    found = {}
    for (s, B), run in runs(index).items():
        hits = []
        for qi, q in zip(qs, queries):
            src = [leaf_of[str(b["claim_ids"][a])] for a in b["answers"][qi] if a >= 0]
            r = run(q)
            if s == "beam":
                hits.append(set(src) <= set(r.claims.tolist()))
            else:  # every claim of the top B patents is returned
                hits.append({tree.parent[l] for l in src} <= set(r.patents[:B].tolist()))
        found[s, B] = np.array(hits)
    rows = []
    for (s, B), x in found.items():
        ms = latency(runs(index)[s, B], queries)
        for d in DATASETS:
            m = np.ones(len(qs), bool) if d == "all" else ds == d
            row = {"B": B, "search": s, "dataset": d, "n": int(m.sum()), "source_returned": x[m].mean(), "ms_median": ms}
            if s == "beam":
                f = found["flat", B][m]
                row.update(vs_flat_better=int((x[m] & ~f).sum()), vs_flat_worse=int((~x[m] & f).sum()),
                           vs_flat_p=sign_p((x[m] & ~f).sum(), (~x[m] & f).sum()))
            rows.append(row)
    out = pd.DataFrame(rows)
    name = "default"

out.to_csv(f"results/part1_{name}.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
    print(out.to_string(index=False))
print(f"written to results/part1_{name}.csv")
