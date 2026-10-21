"""Embed a sample's independent claims on a remote GPU (agave), then merge them into the local embedding cache.

  local:  .venv/bin/python3.14 experiment-big-data/gpu_embed.py export --n 100000
          -> experiment-big-data/gpu_data/claims_100000.jsonl.gz (claim texts only; committed on the gpu-data branch)
  remote: agave run experiment-big-data/gpu_embed.py --on colab --gpu A100 --ref gpu-data --timeout 3600
          (no arguments: embeds every gpu_data/claims_*.jsonl.gz, writes $OUTPUT_DIR/embeddings_<n>.npz)
  local:  .venv/bin/python3.14 experiment-big-data/gpu_embed.py merge outputs/latest/outputs/embeddings_100000.npz
          -> adds them to data/index/embeddings_large.npz, which run.py then reads (nothing left to embed)

Keys are sha1 of the text, as in search/embed.py. Remote embeddings are computed in fp16 and stored as fp16.
Don't merge while run.py is running: it rewrites the cache from its own copy.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_DATA = os.path.join(HERE, "gpu_data")
CACHE = "data/index/embeddings_large.npz"  # same as run.py
sys.path.insert(0, os.path.join(HERE, ".."))


def export(n: int) -> None:
    from sample import load_sample
    patents, _ = load_sample(n)
    texts = list(dict.fromkeys(c["text"] for p in patents for c in p["claims"]))  # unique, in order
    os.makedirs(GPU_DATA, exist_ok=True)
    path = os.path.join(GPU_DATA, f"claims_{n}.jsonl.gz")
    with gzip.open(path, "wt") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    print(f"{len(texts)} unique claim texts -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def remote() -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    from search.embed import MODEL
    out = os.environ.get("OUTPUT_DIR", "outputs")
    os.makedirs(out, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
    model = SentenceTransformer(MODEL, device="cuda")
    model.half()
    for path in sorted(glob.glob(os.path.join(GPU_DATA, "claims_*.jsonl.gz"))):
        texts = [json.loads(line) for line in gzip.open(path, "rt")]
        t0 = time.perf_counter()
        vecs = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        dt = time.perf_counter() - t0
        name = os.path.basename(path).replace("claims_", "embeddings_").replace(".jsonl.gz", ".npz")
        np.savez(os.path.join(out, name), keys=np.array([hashlib.sha1(t.encode()).hexdigest() for t in texts]),
                 vecs=vecs.astype(np.float16), model=MODEL,
                 meta=json.dumps({"seconds": dt, "n": len(texts), "gpu": torch.cuda.get_device_name(0), "dtype": "fp16"}))
        print(f"{path}: {len(texts)} claims in {dt:.0f} s ({len(texts) / dt:.0f} claims/s) -> {name}", flush=True)


def merge(path: str) -> None:
    from search.embed import Embedder
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    emb = Embedder(CACHE)
    assert str(z["model"]) == emb.model_name, (str(z["model"]), emb.model_name)
    before = len(emb.cache)
    emb.cache.update(zip(z["keys"].tolist(), z["vecs"].astype(np.float32)))
    emb.meta["gpu_embed_seconds"] = emb.meta.get("gpu_embed_seconds", 0.0) + meta["seconds"]
    emb.meta["n_gpu_embedded"] = emb.meta.get("n_gpu_embedded", 0) + meta["n"]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, keys=np.array(list(emb.cache)), vecs=np.stack(list(emb.cache.values())), model=emb.model_name,
             meta=json.dumps(emb.meta))
    print(f"{meta['n']} embeddings from {meta['gpu']} ({meta['seconds']:.0f} s); cache {before} -> {len(emb.cache)}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        remote()
    else:
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd", required=True)
        sub.add_parser("export").add_argument("--n", type=int, default=100_000)
        sub.add_parser("merge").add_argument("path")
        args = ap.parse_args()
        export(args.n) if args.cmd == "export" else merge(args.path)
