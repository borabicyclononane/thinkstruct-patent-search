"""Bundle frozen PatentSBERTa embeddings (768-d) of every independent claim and of the synthetic queries, for
training the query / key towers (training/train_towers.py) without touching the backbone.

  .venv/bin/python3.14 training/prepare.py --synthetic outputs/synthetic/small_100

Claim embeddings come from the Part 1 cache (data/index/embeddings.npz); query embeddings are added to it.
The bundle holds no text: claim ids, splits, query datasets, the answer claim indices of each query, and vectors.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "synthetic"))
from generate_synthetic import DATASETS, load_claims, load_split  # noqa: E402
from search.embed import Embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="outputs/synthetic/small_100", help="folder with <dataset>.jsonl")
    ap.add_argument("--out", default=None, help="default: training/bundles/<folder name>.npz")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "training", "bundles", os.path.basename(os.path.normpath(args.synthetic)) + ".npz")

    claims = load_claims()
    split_of = load_split([c["doc_number"] for c in claims])
    index = {c["claim_id"]: i for i, c in enumerate(claims)}
    rows = [json.loads(line) for d in DATASETS for line in open(os.path.join(ROOT, args.synthetic, f"{d}.jsonl"))]
    answers = np.full((len(rows), 2), -1, dtype=np.int64)  # one or two source claims per query
    for i, r in enumerate(rows):
        answers[i, :len(r["answer"])] = [index[a["claim_id"]] for a in r["answer"]]

    emb = Embedder()  # same model and cache as Part 1
    claim_vecs = emb.embed([c["text"] for c in claims], progress=True)
    query_vecs = emb.embed([r["query_text"] for r in rows], progress=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, claim_ids=np.array([c["claim_id"] for c in claims]), claim_split=np.array([split_of[c["doc_number"]] for c in claims]),
        claim_vecs=claim_vecs.astype(np.float32), query_ids=np.array([r["query_id"] for r in rows]),
        query_dataset=np.array([r["dataset"] for r in rows]), query_split=np.array([r["split"] for r in rows]),
        query_vecs=query_vecs.astype(np.float32), answers=answers,
        claim_paths=np.array([c["cpc_path"] for c in claims]),  # CPC tree for the beam-search eval, rebuilt remotely
        meta=json.dumps({"model": emb.model_name, "synthetic": args.synthetic, "n_claims": len(claims), "n_queries": len(rows)}))
    split_counts = {s: int((np.array([r["split"] for r in rows]) == s).sum()) for s in ("train", "val", "test")}
    print(f"{len(claims)} claims, {len(rows)} queries {split_counts} -> {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
