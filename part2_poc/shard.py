"""A shard: the patents at some positions (a contiguous range, or hashed), their claim embeddings, and one Roaring
bitmap per term over those positions. Every call is counted in `calls`."""
from collections import Counter

import numpy as np
from pyroaring import BitMap


class Shard:
    def __init__(self, sid: int, positions, bitmaps: dict[int, BitMap], claim_pos, claim_ids, claim_vecs):
        self.sid = sid
        self.positions = BitMap(positions)
        self.lo, self.hi = self.positions.min(), self.positions.max() + 1  # range shards: exactly their block
        self.bitmaps = bitmaps  # term id -> positions of this shard whose title/abstract has the term
        self.claim_pos, self.claim_ids, self.claim_vecs = np.asarray(claim_pos), list(claim_ids), claim_vecs
        self.calls = Counter()

    def _match(self, term_ids: list[int], mode: str) -> BitMap:
        bms = [self.bitmaps.get(t, BitMap()) for t in term_ids]
        return BitMap.intersection(*bms) if mode == "and" else BitMap.union(*bms)

    def count_in_ranges(self, term_ids: list[int], mode: str, ranges: list[tuple[int, int]]) -> np.ndarray:
        """Matching patents in each [start, end), clipped to this shard. mode "and": patents with every term."""
        self.calls["count_in_ranges"] += 1
        m = self._match(term_ids, mode)
        return np.array([m.range_cardinality(max(s, self.lo), min(e, self.hi)) if s < self.hi and e > self.lo else 0
                         for s, e in ranges], dtype=np.int64)

    def exact_match(self, patent_positions, term_ids: list[int], mode: str) -> np.ndarray:
        self.calls["exact_match"] += 1
        m = self._match(term_ids, mode)
        return np.array([int(p) in m for p in patent_positions], dtype=bool)

    def score(self, query_vec: np.ndarray, patent_positions) -> list[tuple[int, str, float]]:
        """(position, claim id, cosine) for every claim of these patents. query_vec is L2-normalised."""
        self.calls["score"] += 1
        rows = np.flatnonzero(np.isin(self.claim_pos, list(patent_positions)))
        s = self.claim_vecs[rows] @ query_vec
        return [(int(self.claim_pos[r]), self.claim_ids[r], float(x)) for r, x in zip(rows, s)]
