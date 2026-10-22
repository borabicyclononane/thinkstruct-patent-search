"""Embed a subsample of the large dataset and time hybrid search on it.

  .venv/bin/python3.14 experiment-big-data/sample.py --n 100000     # once: zip -> data/large/sample_100000.jsonl
  .venv/bin/python3.14 experiment-big-data/run.py --n 100000

Differences from `python -m search hybrid`: one held-out split (`--queries` random patents are queries, their first
independent claim; the index is built from all the others), the first `--hybrid-queries` of those queries, and filter
scenarios for a corpus spanning all CPC sections. Flat vs beam timing on the same split: results/compare_part1.py.
Embeddings are computed in chunks and cached (data/index/embeddings_large.npz), so an interrupted run resumes.
Writes experiment-big-data/results_<n>/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sample import load_sample  # noqa: E402
from search.embed import Embedder  # noqa: E402
from search.filters import Filter  # noqa: E402
from search.hybrid_eval import run_hybrid  # noqa: E402
from search.index import build_index  # noqa: E402
from search.plots import hybrid_latency  # noqa: E402

CACHE = "data/index/embeddings_large.npz"
SCENARIOS = [
    ("CPC section H", Filter(cpc=("H",))),
    ("CPC subclass G06F", Filter(cpc=("G06F",))),
    ("CPC main group G06F 3", Filter(cpc=("G06F 3",))),
    ("CPC subgroup G02B 27/0172", Filter(cpc=("G02B 27/0172",))),
    ("CPC A61K or C07D", Filter(cpc=("A61K", "C07D"))),
    ("keyword: method", Filter(all_of=("method",))),
    ("keyword: vehicle", Filter(all_of=("vehicle",))),
    ("keyword: antibody", Filter(all_of=("antibody",))),
    ("keyword: crispr", Filter(all_of=("crispr",))),
    ("wireless AND battery", Filter(all_of=("wireless", "battery"))),
    ("lidar OR radar", Filter(any_of=("lidar", "radar"))),
    ("title = 'display device'", Filter(title="display device")),
    ("G06F + neural AND network", Filter(cpc=("G06F",), all_of=("neural", "network"))),
]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100_000, help="sample size (run sample.py --n first)")
ap.add_argument("--queries", type=int, default=1000, help="held-out query patents")
ap.add_argument("--hybrid-queries", type=int, default=300)
ap.add_argument("--chunk", type=int, default=20_000, help="claims embedded between cache saves")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = os.path.join(os.path.dirname(__file__), f"results_{args.n}")
os.makedirs(os.path.join(out, "plots"), exist_ok=True)

patents, report = load_sample(args.n)
print(f"{len(patents)} patents, {sum(len(p['claims']) for p in patents)} independent claims; sample report: {report}")

# 1. Embed every independent claim, in resumable chunks
embedder = Embedder(CACHE)
texts = [c["text"] for p in patents for c in p["claims"]]
t0 = time.perf_counter()
for i in range(0, len(texts), args.chunk):
    embedder.embed(texts[i:i + args.chunk], progress=True)
    done = min(i + args.chunk, len(texts))
    rate = embedder.meta["n_embedded"] / max(embedder.meta["embed_seconds"], 1e-9)
    print(f"embedded {done}/{len(texts)} ({time.perf_counter() - t0:.0f} s this run; "
          f"{rate:.1f} claims/s fresh; ~{(len(texts) - done) / rate / 60:.0f} min left at most)", flush=True)

# 2. Held-out split, index the rest
rng = np.random.default_rng(args.seed)
is_query = np.zeros(len(patents), dtype=bool)
is_query[rng.choice(len(patents), args.queries, replace=False)] = True
test = [p for p, q in zip(patents, is_query) if q]
train = [p for p, q in zip(patents, is_query) if not q]
index, build_t = build_index(train, embedder)
index.tree.print_report()
print("build:", {k: f"{v:.2f} s" for k, v in build_t.items()})
queries = embedder.embed([p["claims"][0]["text"] for p in test])

# 3. Hybrid search: latency per filter + correctness checks (raise on failure)
t0 = time.perf_counter()
h_timing, checks = run_hybrid(index, queries[:args.hybrid_queries], scenarios=SCENARIOS)
print(f"hybrid: {time.perf_counter() - t0:.0f} s")
h_timing.to_csv(os.path.join(out, "hybrid_timing.csv"), index=False)
checks.to_csv(os.path.join(out, "hybrid_checks.csv"), index=False)
hybrid_latency(h_timing, os.path.join(out, "plots", "hybrid_latency.png"))
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
    print(h_timing.pivot(index=["scenario", "n_candidates"], columns="method", values="ms_median")
          .sort_values("n_candidates", ascending=False).to_string(), "\n")
    print(checks.to_string(index=False))

json.dump({"n_patents": len(patents), "n_index_patents": len(train), "n_queries": len(test),
           "n_index_claims": int(len(index.tree.leaves)), "n_tree_nodes": len(index.tree.name),
           "build_seconds": build_t, "sample_report": report, "embedding_meta": embedder.meta},
          open(os.path.join(out, "build.json"), "w"), indent=1)
print(f"\nall checks passed; written to {out}/")
