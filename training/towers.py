"""The Part 3 methods of a training run as (query map, key map) pairs of numpy functions, for search.index.Index.

  raw 768                         identity
  PCA-64                          PCA fit on the training claims (dev patents excluded), as in train_towers.py
  shared towers, no routing loss  \\
  shared towers                    > the grid configuration of each family with the highest val share of sources
  per-level towers                /  returned by beam search at `select_width` (run's search_eval.csv)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_towers import TOWER_OF_LEVEL, Tower  # noqa: E402


def best_configs(run: str, select_width: int = 5) -> dict[str, str]:
    """Family -> label of its best configuration on val at B=select_width."""
    ev = pd.read_csv(os.path.join(run, "search_eval.csv"))
    ev = ev[(ev.dataset == "all") & (ev.search == f"beam B={select_width}") & ev.method.str.startswith("towers routing")]
    shared, per_level = ev[~ev.method.str.endswith("per-level q")], ev[ev.method.str.endswith("per-level q")]
    return {fam: rows.sort_values("source_returned").method.iloc[-1] for fam, rows in [
        ("shared towers, no routing loss", shared[shared.method.str.startswith("towers routing=0 ")]),
        ("shared towers", shared), ("per-level towers", per_level)]}


def load_maps(run: str, b: dict, select_width: int = 5) -> tuple[dict, dict]:
    """-> ({method: (query map, key map)}, {tower family: configuration label}). `b` is the training bundle."""
    ck = torch.load(os.path.join(run, "towers.pt"), weights_only=False)
    cfg = ck["config"]

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

    docs = np.array([c.split("#")[0] for c in b["claim_ids"]])
    train_docs = sorted(set(docs[b["claim_split"] == "train"]))
    dev_docs = set(np.random.default_rng(cfg["seed"]).permutation(train_docs)[:round(cfg["dev_frac"] * len(train_docs))])
    fit_c = np.flatnonzero((b["claim_split"] == "train") & ~np.isin(docs, list(dev_docs)))
    mu = b["claim_vecs"][fit_c].mean(0)
    comps = np.linalg.svd(b["claim_vecs"][fit_c] - mu, full_matrices=False)[2][:cfg["dim"]]
    pca = lambda x: (np.asarray(x) - mu) @ comps.T

    best = best_configs(run, select_width)
    maps = {"raw 768": (lambda x: x, lambda x: x), "PCA-64": (pca, pca), **{fam: tower_maps(l) for fam, l in best.items()}}
    return maps, best
