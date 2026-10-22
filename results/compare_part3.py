"""Part 3: does the source claim of a synthetic query come back from beam search? Raw 768-d, PCA-64 and trained
towers; per synthetic dataset; val and test; plus query latency.

  .venv/bin/python3.14 results/compare_part3.py [--run training/runs/final]

Towers, each the configuration of the run's grid with the highest val share returned at B=5 (search_eval.csv):
shared towers without the routing loss, shared towers (any routing weight), per-level query towers. Val picked them,
so test is the unbiased number. Source returned = the source claim is among the returned claims (combination: both sources).
Reference: flat search with raw 768, keeping all claims of its top B patents.
Sign test: paired, against raw 768 beam search at the same width. Writes results/part3.csv.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from search.index import Index  # noqa: E402
from towers import load_maps  # noqa: E402
from train_towers import claim_tree  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="training/runs/final")
ap.add_argument("--bundle", default="training/bundles/full.npz")
ap.add_argument("--select-width", type=int, default=5)
ap.add_argument("--widths", default="5,30")
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()
os.chdir(ROOT)
W = [int(w) for w in args.widths.split(",")]
torch.set_num_threads(1)

b = dict(np.load(args.bundle))
maps, best = load_maps(args.run, b, args.select_width)
print("best on val at B=%d:" % args.select_width, best)

tree, claim_of_leaf = claim_tree(b)
leaf_of_claim = {c: l for l, c in claim_of_leaf.items()}
leaf_vecs = b["claim_vecs"][[claim_of_leaf[l] for l in tree.leaves]]
indexes = {name: Index(tree, leaf_vecs, project=kf, project_query=qf) for name, (qf, kf) in maps.items()}
FLAT = "flat, raw 768"
METHODS = list(indexes) + [FLAT]


def run(method, q, B):
    """Returned claim leaves: beam search, or all claims of flat search's top B patents."""
    if method == FLAT:
        return {l for p in indexes["raw 768"].flat_search(q, B).patents[:B] for l in tree.children[p]}
    return set(indexes[method].beam_search(q, B).claims.tolist())


def sign_p(x, y):
    x, y = int(x), int(y)
    return min(1.0, 2 * sum(math.comb(x + y, i) for i in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


hits, rows = {}, []
for split in ("val", "test"):
    qs = np.flatnonzero(b["query_split"] == split)
    ds = b["query_dataset"][qs]
    for method in METHODS:
        for B in W:
            found = []
            for qi in qs:
                got = run(method, b["query_vecs"][qi], B)
                src = [leaf_of_claim[a] for a in b["answers"][qi] if a >= 0]
                found.append(all(s in got for s in src))
            found = np.array(found)
            hits[split, method, B] = found
            for d in ["all", "paraphrase", "element_reuse", "combination"]:
                m = np.ones(len(qs), bool) if d == "all" else ds == d
                x, y = found[m], hits[split, "raw 768", B][m]  # raw 768 runs first
                rows.append({"split": split, "method": method, "config": best.get(method, ""), "B": B, "dataset": d,
                             "n": int(m.sum()), "source_returned": x.mean(), "vs_raw_better": int((x & ~y).sum()),
                             "vs_raw_worse": int((~x & y).sum()), "vs_raw_p": sign_p((x & ~y).sum(), (~x & y).sum())})
    print(split, "done", flush=True)


def latency(fn, queries):
    fn(queries[0])
    ms = []
    for _ in range(args.repeats):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            ms.append(1000 * (time.perf_counter() - t0))
    return np.median(ms)


queries = b["query_vecs"][np.isin(b["query_split"], ["val", "test"])]
res = pd.DataFrame(rows)
for method in METHODS:
    for B in W:
        res.loc[(res.method == method) & (res.B == B), "ms_median"] = latency(lambda q: run(method, q, B), queries)
res.to_csv("results/part3.csv", index=False)
with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
    print(res[res.dataset == "all"].to_string(index=False))
    print(res[res.split == "test"].pivot_table(index="method", columns=["B", "dataset"], values="source_returned", sort=False).to_string())
