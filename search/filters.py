"""Hybrid-search filters: CPC prefixes, keywords in title/abstract, exact title.

Filters resolve to (start nodes, candidate positions):
  CPC prefixes -> tree nodes; beam search starts from them (several prefixes = a union of start nodes)
  keywords / title -> one sorted array of patent positions (tree.order), from an inverted index
Different filter types are ANDed; `all_of` keywords are ANDed, `any_of` keywords are ORed.

Tokens: lowercased runs of letters/digits; a hyphenated term is indexed whole AND as its parts
("non-pneumatic" -> "non-pneumatic", "non", "pneumatic"), so the query "non-pneumatic" matches only the
whole term while "pneumatic" also matches it. Plurals are folded (tires -> tire, assemblies -> assembly,
branches -> branch). A multi-word keyword ("tire pressure") needs all its words in the same field (not a phrase).
Title: exact match after normalization (lowercase, punctuation and repeated whitespace removed).
"""
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .index import Index

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+")
FIELDS = ("title", "abstract")
EMPTY = np.array([], dtype=np.int64)


def fold(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def index_terms(text: str) -> set[str]:
    out = set()
    for t in TOKEN.findall((text or "").lower()):
        out.add(fold(t))
        if "-" in t:
            out.update(fold(part) for part in t.split("-"))
    return out


def query_terms(keyword: str) -> list[str]:
    return [fold(t) for t in TOKEN.findall(keyword.lower())]


def normalize_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))


@dataclass
class Filter:
    cpc: tuple[str, ...] = ()  # CPC prefixes as in the tree, e.g. "B60B", "B60B 19", "B60B 19/00"; ORed
    all_of: tuple[str, ...] = ()  # keywords that must all occur
    any_of: tuple[str, ...] = ()  # at least one must occur
    fields: tuple[str, ...] = FIELDS  # where keywords are searched
    title: str | None = None  # exact (normalized) title

    def __bool__(self):
        return bool(self.cpc or self.all_of or self.any_of or self.title)


class TextIndex:
    """Inverted index over the patents of `index`: field -> term -> sorted positions; normalized title -> positions."""

    def __init__(self, index: Index):
        tree = index.tree
        self.index = index
        post = {f: defaultdict(list) for f in FIELDS}
        titles = defaultdict(list)
        for pos, n in enumerate(tree.order):  # positions ascend, so every posting list comes out sorted
            p = tree.patent[n]
            for f in FIELDS:
                for t in index_terms(p[f]):
                    post[f][t].append(pos)
            titles[normalize_title(p["title"])].append(pos)
        self.post = {f: {t: np.array(v, dtype=np.int64) for t, v in d.items()} for f, d in post.items()}
        self.titles = {t: np.array(v, dtype=np.int64) for t, v in titles.items()}

    def keyword(self, keyword: str, fields=FIELDS) -> np.ndarray:
        """Positions where every word of `keyword` occurs in one of `fields`."""
        hits = [_intersect([self.post[f].get(t, EMPTY) for t in query_terms(keyword)]) for f in fields]
        return _union(hits, len(self.index.tree.order))

    def resolve(self, f: Filter) -> tuple[np.ndarray, np.ndarray | None]:
        """-> (beam start nodes, sorted candidate positions or None if there is no keyword/title filter).
        Candidates are already restricted to the CPC prefixes, so flat search can use them directly."""
        tree = self.index.tree
        starts = np.array([0], dtype=np.int64)
        if f.cpc:
            nodes = {tree.cpc_node[c] for c in map(normalize_cpc, f.cpc) if c in tree.cpc_node}
            starts = np.array(sorted(n for n in nodes if not _has_ancestor(tree, n, nodes)), dtype=np.int64)
        lists = [self.keyword(k, f.fields) for k in f.all_of]
        if f.any_of:
            lists.append(_union([self.keyword(k, f.fields) for k in f.any_of], len(tree.order)))
        if f.title is not None:
            lists.append(self.titles.get(normalize_title(f.title), EMPTY))
        if not lists:
            return starts, None
        cand = _intersect(lists)
        if f.cpc and not len(starts):
            cand = EMPTY
        elif f.cpc:  # keep candidates inside one of the (disjoint) start-node ranges
            lo, hi = tree.start[starts], tree.end[starts]
            o = np.argsort(lo)
            i = np.searchsorted(lo[o], cand, side="right") - 1
            cand = cand[(i >= 0) & (cand < hi[o][np.maximum(i, 0)])]
        return starts, cand

    def cpc_positions(self, starts: np.ndarray) -> np.ndarray:
        tree = self.index.tree
        return np.concatenate([np.arange(tree.start[n], tree.end[n]) for n in starts]) if len(starts) else EMPTY



def normalize_cpc(code: str) -> str:
    return " ".join(code.split())


def _has_ancestor(tree, n: int, nodes: set) -> bool:
    n = tree.parent[n]
    while n >= 0:
        if n in nodes:
            return True
        n = tree.parent[n]
    return False


def _intersect(lists: list[np.ndarray]) -> np.ndarray:
    """AND of sorted position arrays, smallest first (so every step is at most as big as the rarest term)."""
    if not lists:
        return EMPTY
    lists = sorted(lists, key=len)
    out = lists[0]
    for other in lists[1:]:
        if not len(out):
            break
        out = out[np.isin(out, other, assume_unique=True)]
    return out


def _union(lists: list[np.ndarray], n: int) -> np.ndarray:
    """OR of sorted position arrays (positions < n): mark them in a length-n boolean array, read back in order.
    O(n + total length), no sort; np.unique (a sort) was 40x slower for two ~3.4k-long lists."""
    mask = np.zeros(n, dtype=bool)
    for positions in lists:
        mask[positions] = True
    return np.flatnonzero(mask)


def hybrid_search(index: Index, text: TextIndex, query: np.ndarray, f: Filter, k: int | None = 10, B: int = 10,
                  flat: bool = False, valid: np.ndarray | None = None):
    """-> (Result, number of candidate patents or None if unfiltered). Beam: start at the CPC nodes, prune by
    candidates. Flat: pre-filter, i.e. score only the candidates' claims."""
    starts, cand = text.resolve(f)
    if flat:
        if cand is None and f.cpc:
            cand = text.cpc_positions(starts)
        res = index.flat_search(query, k, valid, cand)
    else:
        res = index.beam_search(query, B, starts, valid, cand)
    if cand is not None:
        return res, len(cand)
    return res, int((index.tree.end[starts] - index.tree.start[starts]).sum()) if f else None
