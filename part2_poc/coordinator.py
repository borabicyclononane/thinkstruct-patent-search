"""The coordinator: holds the CPC tree (node keys as sum + count, node ranges), routes the beam, merges shard results.

Range-partitioned shards: the coordinator knows the boundaries, so it contacts only shards overlapping a node range.
Hash-partitioned baseline: any shard may hold any position, so routing asks every shard; scoring asks the owners.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from build import LEVELS  # noqa: E402
from search.filters import query_terms  # noqa: E402
from search.index import l2n, topk  # noqa: E402

PATENT = LEVELS.index("patent")


class Coordinator:
    def __init__(self, built: dict):
        c = built["coord"]
        self.scheme, self.S, self.shards = built["scheme"], built["S"], built["shards"]
        self.owner, self.bounds = built["owner"], built["bounds"]
        self.name, self.level, self.start, self.end = c["name"], c["level"], c["start"], c["end"]
        self.children, self.docs, self.vocab = c["children"], c["docs"], c["vocab"]
        self.key = l2n(c["key_sum"] / c["key_count"][:, None])
        self.by_name = {n: i for i, n in enumerate(self.name) if self.level[i] < PATENT}

    def terms(self, keywords) -> list[int]:
        """Term ids of the keywords; an unknown term is -1 (matches nothing)."""
        return [self.vocab.get(t, -1) for kw in keywords or [] for t in query_terms(kw)]

    def shards_for(self, ranges) -> list[int]:
        if self.scheme == "hash":
            return list(range(self.S))
        return [s for s in range(self.S) if any(a < self.bounds[s + 1] and b > self.bounds[s] for a, b in ranges)]

    def counts(self, nodes, terms, mode, contacted: set | None = None) -> np.ndarray:
        """Matching patents under each node: one batched call per relevant shard, summed."""
        ranges = list(zip(self.start[nodes].tolist(), self.end[nodes].tolist()))
        total = np.zeros(len(nodes), dtype=np.int64)
        for s in self.shards_for(ranges):
            total += self.shards[s].count_in_ranges(terms, mode, ranges)
            if contacted is not None:
                contacted.add(s)
        return total

    def search(self, query_vec, k: int = 5, B: int = 10, cpc_prefix=None, keywords=None, mode: str = "or"):
        """-> (results [{doc, claim, score}] best first, trace {levels, routing, scoring})."""
        q = l2n(np.asarray(query_vec, dtype=np.float64))
        terms = self.terms(keywords)
        trace = {"levels": [], "routing": set(), "scoring": set()}
        if cpc_prefix and cpc_prefix not in self.by_name:
            return [], trace
        beam = np.array([self.by_name[cpc_prefix] if cpc_prefix else 0])
        while self.level[beam[0]] < PATENT:
            kids = np.concatenate([self.children[n] for n in beam])
            valid = kids[self.counts(kids, terms, mode, trace["routing"]) > 0] if terms else kids
            trace["levels"].append((LEVELS[self.level[kids[0]]], len(valid), len(kids)))
            if not len(valid):
                return [], trace
            beam = valid[topk(self.key[valid] @ q, B)]

        positions = self.start[beam]  # a patent node's range is its own position
        hits = []
        for s in sorted(set(self.owner[positions].tolist())):
            mine = positions[self.owner[positions] == s]
            trace["scoring"].add(s)
            if terms:
                mine = mine[self.shards[s].exact_match(mine, terms, mode)]
            if len(mine):
                hits += self.shards[s].score(q, mine)
        best = {}
        for pos, claim, score in hits:  # dedupe by patent: its best claim
            if pos not in best or score > best[pos][1]:
                best[pos] = (claim, score)
        # ties (duplicate claims exist) broken by position, so the order doesn't depend on the partitioning;
        # rounded because a shard's matrix product can differ from another's in the last bit
        ranked = sorted(best.items(), key=lambda x: (-round(x[1][1], 12), x[0]))[:k]
        return [{"doc": self.docs[p], "claim": c, "score": s} for p, (c, s) in ranked], trace
