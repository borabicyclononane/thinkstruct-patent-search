"""Part 1: beam search with width B vs flat search keeping the top B patents; latency and CPC-agreement accuracy.

  .venv/bin/python3.14 results/compare_part1.py                 # default (vehicle) dataset, 5-fold as `python -m search eval`
  .venv/bin/python3.14 results/compare_part1.py --sample 10000  # 10^4 large-dataset sample, held-out split as experiment-big-data/run.py

Queries: each held-out patent's first independent claim. Accuracy on the B returned patents (search/evaluate.py):
agree_<level> = share sharing the query's CPC prefix to <level>, knn_<level> = their majority label is the query's,
ndcg = nDCG@B with relevance = depth of the shared CPC prefix. Writes results/part1_<dataset>.csv.
"""
import argparse
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
from search.evaluate import cpc_metrics, shared_depth  # noqa: E402
from search.index import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=None, help="large-dataset sample size (default: the vehicle dataset)")
ap.add_argument("--folds", type=int, default=5, help="default dataset only")
ap.add_argument("--queries", type=int, default=1000, help="sample only: held-out query patents")
ap.add_argument("--widths", default="1,5,10,100")
ap.add_argument("--repeats", type=int, default=3, help="timing passes over the queries")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.chdir(ROOT)
W = [int(w) for w in args.widths.split(",")]

rng = np.random.default_rng(args.seed)
if args.sample:
    from sample import load_sample
    patents, _ = load_sample(args.sample)
    embedder = Embedder("data/index/embeddings_large.npz")
    fold_of = np.ones(len(patents), dtype=int)  # same split as experiment-big-data/run.py
    fold_of[rng.choice(len(patents), args.queries, replace=False)] = 0
    folds, name = [0], f"sample{args.sample}"
else:
    patents, _ = load_corpus()
    embedder = Embedder()
    fold_of = np.empty(len(patents), dtype=int)  # same folds as `python -m search eval`
    fold_of[rng.permutation(len(patents))] = np.arange(len(patents)) % args.folds
    folds, name = range(args.folds), "default"


def latency(fn, queries) -> float:
    fn(queries[0])  # warm up
    ms = []
    for _ in range(args.repeats):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            ms.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ms))


rows, ms = [], {}
for fold in folds:
    test = [p for p, f in zip(patents, fold_of) if f == fold]
    train = [p for p, f in zip(patents, fold_of) if f != fold]
    index, _ = build_index(train, embedder)
    tree = index.tree
    queries = embedder.embed([p["claims"][0]["text"] for p in test])
    print(f"fold {fold}: {len(train)} index patents ({len(tree.leaves)} claims), {len(test)} queries", flush=True)
    runs = {(s, B): (lambda q, B=B: index.flat_search(q, B)) if s == "flat" else (lambda q, B=B: index.beam_search(q, B))
            for B in W for s in ("flat", "beam")}
    for p, q in zip(test, queries):
        ideal = sorted((shared_depth(p["path"], t["path"]) for t in train), reverse=True)
        for (s, B), run in runs.items():
            paths = [tree.patent[n]["path"] for n in run(q).patents[:B]]
            rows.append({"search": s, "B": B, **cpc_metrics(p["path"], paths, B, ideal)})
    if fold == folds[0]:
        ms = {key: latency(run, queries) for key, run in runs.items()}

out = pd.DataFrame(rows).groupby(["B", "search"], sort=False).mean().reset_index()
out["ms_median"] = [ms[s, B] for s, B in zip(out["search"], out["B"])]
out.to_csv(f"results/part1_{name}.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
    print(out.to_string(index=False))
print(f"written to results/part1_{name}.csv")
