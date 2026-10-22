"""Part 3: does the source claim of a synthetic query come back from beam search? Raw 768-d, PCA-64 and trained
towers; per synthetic dataset; val and test; plus query latency.

  .venv/bin/python3.14 results/compare_part3.py [--run training/runs/20261021-203151]

Towers, each the configuration of the run's grid with the highest val share returned at B=5 (search_eval.csv):
shared towers without the routing loss, shared towers (any routing weight), per-level query towers. Val picked them,
so test is the unbiased number. Source returned = a source claim is among the returned claims (combination: either).
Sign test: paired, against raw 768 at the same width. Writes results/part3.csv.
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
from train_towers import TOWER_OF_LEVEL, Tower, claim_tree  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="training/runs/20261021-203151")
ap.add_argument("--bundle", default="training/bundles/full.npz")
ap.add_argument("--select-width", type=int, default=5)
ap.add_argument("--widths", default="5")
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()
os.chdir(ROOT)
W = [int(w) for w in args.widths.split(",")]
torch.set_num_threads(1)

b = dict(np.load(args.bundle))
ck = torch.load(os.path.join(args.run, "towers.pt"), weights_only=False)
cfg = ck["config"]
ev = pd.read_csv(os.path.join(args.run, "search_eval.csv"))
ev = ev[(ev.dataset == "all") & (ev.search == f"beam B={args.select_width}") & ev.method.str.startswith("towers routing")]
shared, per_level = ev[~ev.method.str.endswith("per-level q")], ev[ev.method.str.endswith("per-level q")]
best = {fam: rows.sort_values("source_returned").method.iloc[-1] for fam, rows in [
    ("shared towers, no routing loss", shared[shared.method.str.startswith("towers routing=0 ")]),
    ("shared towers", shared), ("per-level towers", per_level)]}
print("best on val at B=%d:" % args.select_width, best)


def tower(sd):
    t = Tower(ck["W0"], cfg["hidden"], cfg["dropout"])
    t.load_state_dict(sd)
    return t.eval()


def as_np(t):
    def f(x):
        with torch.no_grad():
            return t(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()
    return f


def tower_maps(label):
    s = ck[label]
    names = sorted({k.split(".")[0] for k in s["query_tower"]})
    q = {n: as_np(tower({k[len(n) + 1:]: v for k, v in s["query_tower"].items() if k.startswith(n + ".")})) for n in names}
    qf = q["claim"] if len(q) == 1 else lambda x: np.stack([q[n](x) for n in TOWER_OF_LEVEL], axis=1)
    return qf, as_np(tower(s["key_tower"]))


docs = np.array([c.split("#")[0] for c in b["claim_ids"]])  # PCA fit exactly as in training/train_towers.py
train_docs = sorted(set(docs[b["claim_split"] == "train"]))
dev_docs = set(np.random.default_rng(cfg["seed"]).permutation(train_docs)[:round(cfg["dev_frac"] * len(train_docs))])
fit_c = np.flatnonzero((b["claim_split"] == "train") & ~np.isin(docs, list(dev_docs)))
mu = b["claim_vecs"][fit_c].mean(0)
comps = np.linalg.svd(b["claim_vecs"][fit_c] - mu, full_matrices=False)[2][:cfg["dim"]]
pca = lambda x: (np.asarray(x) - mu) @ comps.T

tree, claim_of_leaf = claim_tree(b)
leaf_of_claim = {c: l for l, c in claim_of_leaf.items()}
leaf_vecs = b["claim_vecs"][[claim_of_leaf[l] for l in tree.leaves]]
indexes = {name: Index(tree, leaf_vecs, project=kf, project_query=qf) for name, (qf, kf) in {
    "raw 768": (lambda x: x, lambda x: x), "PCA-64": (pca, pca), **{fam: tower_maps(l) for fam, l in best.items()}}.items()}
METHODS = list(indexes)
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
                got = set(indexes[method].beam_search(b["query_vecs"][qi], B).claims.tolist())
                src = [leaf_of_claim[a] for a in b["answers"][qi] if a >= 0]
                found.append(any(s in got for s in src))
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
        res.loc[(res.method == method) & (res.B == B), "ms_median"] = latency(lambda q: indexes[method].beam_search(q, B), queries)
res.to_csv("results/part3.csv", index=False)
with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
    print(res[res.dataset == "all"].to_string(index=False))
    print(res[res.split == "test"].pivot_table(index="method", columns=["B", "dataset"], values="source_returned", sort=False).to_string())
