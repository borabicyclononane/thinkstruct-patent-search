"""Build the coordinator (CPC tree, node keys as sum + count, node ranges) and S shards, range- and hash-partitioned.

  .venv/bin/python3.14 part2_poc/build.py [--shards 1,2,4]

Reads (never writes) the default dataset via search.data and the cached PatentSBERTa claim embeddings in
training/bundles/full.npz. Writes part2_poc/build/{range,hash}_S{S}.pkl.
Positions: patents sorted by (CPC path, doc number), i.e. depth-first, so every node covers [start, end).
Terms: title + abstract tokens as in Part 1 (search.filters.index_terms: lowercase, plurals folded), minus stopwords.
"""
import argparse
import os
import pickle
import sys
import zlib

import numpy as np
from pyroaring import BitMap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from search.data import load_corpus  # noqa: E402
from search.filters import index_terms  # noqa: E402
from shard import Shard  # noqa: E402

BUNDLE = os.path.join(ROOT, "training", "bundles", "full.npz")
OUT = os.path.join(HERE, "build")
LEVELS = ["root", "section", "class", "subclass", "main group", "subgroup", "patent"]
STOPWORDS = set("a an and are as at be been being by can each for from has have in into is it its may more of on one "
                "or other such than that the their then there these this to two which with wherein whereby".split())


def load() -> tuple[list[dict], np.ndarray]:
    """-> (patents in depth-first CPC order, claim embeddings). Each patent: doc, title, abstract, path, terms,
    claim_ids, claim_rows (rows of the embedding matrix)."""
    cwd = os.getcwd()
    os.chdir(ROOT)  # load_corpus uses paths relative to the repo
    corpus, _ = load_corpus()
    os.chdir(cwd)
    b = np.load(BUNDLE)
    docs = np.array([c.split("#")[0] for c in b["claim_ids"]])
    pats = []
    for p in corpus:
        rows = np.flatnonzero(docs == p["doc_number"])
        assert len(rows) and tuple(b["claim_paths"][rows[0]]) == p["path"], p["doc_number"]
        pats.append({"doc": p["doc_number"], "title": p["title"], "abstract": p["abstract"], "path": p["path"],
                     "terms": (index_terms(p["title"]) | index_terms(p["abstract"])) - STOPWORDS,
                     "claim_ids": b["claim_ids"][rows].tolist(), "claim_rows": rows})
    pats.sort(key=lambda p: (p["path"], p["doc"]))
    return pats, b["claim_vecs"]


def coordinator_data(pats: list[dict], vecs: np.ndarray) -> dict:
    """The CPC tree in depth-first order: name, level, parent, children, [start, end), key sum and count."""
    name, level, parent, start, end, ids = [], [], [], [], [], {}
    for pos, p in enumerate(pats):
        key = ()
        for lvl, part in enumerate(("root", *p["path"], p["doc"])):
            key += (part,)
            if key not in ids:
                ids[key] = len(name)
                name.append(part), level.append(lvl), parent.append(ids.get(key[:-1], -1)), start.append(pos), end.append(0)
            end[ids[key]] = pos + 1
    n = len(name)
    children = [[] for _ in range(n)]
    for i, par in enumerate(parent):
        if par >= 0:
            children[par].append(i)
    key_sum, key_count = np.zeros((n, vecs.shape[1])), np.zeros(n, dtype=np.int64)
    patent_node = [ids[("root", *p["path"], p["doc"])] for p in pats]
    for pos, p in enumerate(pats):  # each patent's claims added to it and all its ancestors
        node = patent_node[pos]
        while node >= 0:
            key_sum[node] += vecs[p["claim_rows"]].sum(0)
            key_count[node] += len(p["claim_rows"])
            node = parent[node]
    vocab = {t: i for i, t in enumerate(sorted(set().union(*(p["terms"] for p in pats))))}
    return {"name": name, "level": np.array(level), "parent": np.array(parent), "start": np.array(start),
            "end": np.array(end), "children": [np.array(c, dtype=np.int64) for c in children],
            "key_sum": key_sum, "key_count": key_count, "patent_node": np.array(patent_node),
            "docs": [p["doc"] for p in pats], "vocab": vocab}


def postings(pats: list[dict], vocab: dict) -> dict[int, BitMap]:
    bm = {i: BitMap() for i in vocab.values()}
    for pos, p in enumerate(pats):
        for t in p["terms"]:
            bm[vocab[t]].add(pos)
    for m in bm.values():
        m.run_optimize()
    return bm


def owners(pats: list[dict], S: int, scheme: str) -> tuple[np.ndarray, np.ndarray | None]:
    """-> (shard of each position, range boundaries or None). Range: S equal contiguous blocks; hash: crc32(doc) % S."""
    n = len(pats)
    if scheme == "range":
        bounds = np.linspace(0, n, S + 1).round().astype(int)
        return np.searchsorted(bounds, np.arange(n), side="right") - 1, bounds
    return np.array([zlib.crc32(p["doc"].encode()) % S for p in pats]), None


def build(pats, vecs, coord, S: int, scheme: str) -> dict:
    owner, bounds = owners(pats, S, scheme)
    full = postings(pats, coord["vocab"])
    shards = []
    for sid in range(S):
        mine = BitMap(np.flatnonzero(owner == sid).tolist())
        bitmaps = {t: m & mine for t, m in full.items()}
        bitmaps = {t: m for t, m in bitmaps.items() if m}
        for m in bitmaps.values():
            m.run_optimize()
        claim_pos = [pos for pos in mine for _ in pats[pos]["claim_rows"]]
        rows = np.concatenate([pats[pos]["claim_rows"] for pos in mine])
        shards.append(Shard(sid, mine, bitmaps, claim_pos, [c for pos in mine for c in pats[pos]["claim_ids"]], vecs[rows]))
    return {"scheme": scheme, "S": S, "owner": owner, "bounds": bounds, "coord": coord, "shards": shards}


def path(scheme: str, S: int) -> str:
    return os.path.join(OUT, f"{scheme}_S{S}.pkl")


def load_built(scheme: str, S: int) -> dict:
    return pickle.load(open(path(scheme, S), "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1,2,4")
    args = ap.parse_args()
    pats, vecs = load()
    coord = coordinator_data(pats, vecs)
    full = postings(pats, coord["vocab"])
    lvl = coord["level"]
    print(f"{len(pats)} patents, {len(vecs)} independent claims, {len(coord['name'])} tree nodes "
          f"({', '.join(f'{(lvl == i).sum()} {l}' for i, l in enumerate(LEVELS))})")
    print(f"{len(full)} terms; {sum(len(m) for m in full.values())} postings; "
          f"Roaring {sum(len(m.serialize()) for m in full.values()):,} bytes serialized vs raw int32 "
          f"{4 * sum(len(m) for m in full.values()):,} bytes")

    os.makedirs(OUT, exist_ok=True)
    subclasses = np.flatnonzero(lvl == LEVELS.index("subclass"))
    for S in [int(s) for s in args.shards.split(",")]:
        for scheme in ("range", "hash"):
            built = build(pats, vecs, coord, S, scheme)
            pickle.dump(built, open(path(scheme, S), "wb"))
            print(f"\n{scheme} S={S}")
            for sh in built["shards"]:
                nbytes = sum(len(m.serialize()) for m in sh.bitmaps.values())
                line = f"  shard {sh.sid}: {len(sh.positions)} patents, {len(sh.claim_ids)} claims, bitmaps {nbytes:,} B"
                if scheme == "range":
                    s, e = built["bounds"][sh.sid], built["bounds"][sh.sid + 1]
                    cover = [coord["name"][n] + ("" if s <= coord["start"][n] and coord["end"][n] <= e else " (part)")
                             for n in subclasses if coord["start"][n] < e and coord["end"][n] > s]
                    line += f", positions [{s}, {e}): " + ", ".join(cover)
                print(line)
    print(f"\nwritten to {OUT}/")


if __name__ == "__main__":
    main()
