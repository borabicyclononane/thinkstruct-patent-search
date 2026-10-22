"""Node keys + flat and beam search over the CPC tree.

Swappable pieces (all plain functions passed to Index):
  score(q, K) -> scores        vectorised similarity of one query to a key matrix (default cosine)
  aggregate(child_keys) -> key how an internal node's key is built from its children (default mean)
  project(X) -> X'             applied to leaf embeddings AND queries before keys are aggregated,
  project_query(X) -> X'       optional separate map for queries (two-tower models); defaults to `project`.
                               May return one vector per tree level, shape (n, len(LEVELS), d): each node is
                               then scored with its level's query vector (per-level query models)
                               e.g. a learned linear map / small MLP (default identity)
  combine(parent, child)       how a child's routing score uses its parent's (default: child's own score;
                               e.g. `lambda p, c: p + c` for cumulative path scores)
"""
import time
from dataclasses import dataclass

import numpy as np

from .tree import CLAIM, PATENT, Tree


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def cosine(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    return K @ q  # keys and queries are L2-normalised


def mean(child_keys: np.ndarray) -> np.ndarray:
    return child_keys.mean(0)


def identity(x: np.ndarray) -> np.ndarray:
    return x


def own_score(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    return child


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first."""
    if k < len(scores):
        idx = np.argpartition(-scores, k)[:k]
        return idx[np.argsort(-scores[idx])]
    return np.argsort(-scores)


@dataclass
class Result:
    claims: np.ndarray  # claim leaf ids, best first
    claim_scores: np.ndarray
    patents: np.ndarray  # patent node ids, ranked by their best claim
    patent_scores: np.ndarray


class Index:
    def __init__(self, tree: Tree, leaf_vecs: np.ndarray, score=cosine, aggregate=mean, project=identity,
                 combine=own_score, project_query=None):
        self.tree, self.leaf_vecs = tree, leaf_vecs  # leaf_vecs rows follow tree.leaves
        self.score, self.aggregate, self.project, self.combine = score, aggregate, project, combine
        self.project_query = project_query or project
        self.rebuild_keys()

    def rebuild_keys(self) -> None:
        """Leaf keys = projected claim embeddings; internal keys aggregated bottom-up. Rerun after changing project/aggregate."""
        leaf = l2n(self.project(self.leaf_vecs))
        K = np.zeros((len(self.tree.name), leaf.shape[1]), dtype=leaf.dtype)
        K[self.tree.leaves] = leaf
        for lvl in range(CLAIM - 1, -1, -1):
            for n in np.flatnonzero(self.tree.level == lvl):
                K[n] = l2n(self.aggregate(K[self.tree.children[n]]))
        self.K = K
        self.leaf_K = K[self.tree.leaves]  # contiguous copy: flat search is then one matmul, no per-query gather

    def children(self, node: int, valid: np.ndarray | None = None) -> np.ndarray:
        """Children of `node`; `valid` is an optional boolean mask over all nodes (hybrid-search filters)."""
        c = self.tree.children[node]
        return c if valid is None else c[valid[c]]

    def exclude_patents(self, doc_numbers) -> np.ndarray:
        """Validity mask that removes the given patents (and their claims) from search."""
        valid = np.ones(len(self.tree.name), dtype=bool)
        for d in doc_numbers:
            n = self.tree.patent_node.get(d)
            if n is not None:
                valid[n] = False
                valid[self.tree.children[n]] = False
        return valid

    def node_scores(self, q: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Score `nodes`; a 2-D `q` holds one query vector per tree level."""
        if q.ndim == 1:
            return self.score(q, self.K[nodes])
        out, lvl = np.empty(len(nodes)), self.tree.level[nodes]
        for l in np.unique(lvl):
            out[lvl == l] = self.score(q[l], self.K[nodes[lvl == l]])
        return out

    def has_candidate(self, nodes: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """For each node: does the sorted candidate-position array hit the node's range [start, end)?"""
        return np.searchsorted(cand, self.tree.start[nodes]) < np.searchsorted(cand, self.tree.end[nodes])

    def _result(self, claims: np.ndarray, scores: np.ndarray, k: int | None) -> Result:
        if k is not None:  # partial top-k: only claims scoring at least the m-th best score get sorted
            m = k
            while m < len(scores):
                keep = scores >= np.partition(scores, -m)[-m]  # ties at the threshold kept: an exact ranking prefix
                if len(np.unique(self.tree.parent[claims[keep]])) >= k:  # enough distinct patents for top-k patents
                    claims, scores = claims[keep], scores[keep]
                    break
                m *= 4
        order = np.lexsort((claims, -scores))  # ties (identical claim texts are common) broken by node id
        claims, scores = claims[order], scores[order]
        pats = self.tree.parent[claims]
        first = np.sort(np.unique(pats, return_index=True)[1])  # each patent's best claim, in rank order
        return Result(claims[:k], scores[:k], pats[first][:k], scores[first][:k])

    def flat_search(self, query: np.ndarray, k: int | None = None, valid: np.ndarray | None = None,
                    cand: np.ndarray | None = None) -> Result:
        """Score every independent claim; top-k claims and top-k patents (by best claim).
        `cand` (sorted patent positions) pre-filters: only the candidates' claims are gathered and scored."""
        q = l2n(self.project_query(query[None]))[0]
        q = q[CLAIM] if q.ndim == 2 else q
        if valid is None and cand is None:
            return self._result(self.tree.leaves, self.score(q, self.leaf_K), k)
        if cand is None:
            idx = np.arange(len(self.tree.leaves))
        else:  # concatenated slices leaf_offset[c]:leaf_offset[c + 1] for each candidate c
            lo, n = self.tree.leaf_offset[cand], np.diff(self.tree.leaf_offset)[cand]
            idx = np.repeat(lo - np.cumsum(n) + n, n) + np.arange(n.sum())
        if valid is not None:
            idx = idx[valid[self.tree.leaves[idx]]]
        return self._result(self.tree.leaves[idx], self.score(q, self.leaf_K[idx]), k)

    def beam_search(self, query: np.ndarray, B: int = 10, start_node=0, valid: np.ndarray | None = None,
                    cand: np.ndarray | None = None) -> Result:
        """Route down the tree keeping at most B nodes per level, until the beam holds (up to) B patents;
        then score all independent claims of those patents. Returns every claim of the B patents, ranked.

        Nodes that reach patent level early (a shallower CPC path) are carried along and compete with
        the next level's candidates on their own score.

        Hybrid search: `start_node` may be several nodes (a union of CPC prefixes), and `cand` (sorted patent
        positions passing keyword/title filters) prunes the tree: only children whose range holds a
        candidate are scored.
        """
        q = l2n(self.project_query(query[None]))[0]
        beam = np.atleast_1d(np.asarray(start_node, dtype=np.int64))
        beam_scores = np.zeros(len(beam))
        while True:
            done = self.tree.level[beam] >= PATENT
            if done.all():
                break
            kids = [self.children(n, valid) for n in beam[~done]]
            parent_scores = np.repeat(beam_scores[~done], [len(c) for c in kids])
            kids = np.concatenate(kids)
            if cand is not None:
                ok = self.has_candidate(kids, cand)
                kids, parent_scores = kids[ok], parent_scores[ok]
            nodes = np.concatenate([beam[done], kids])
            node_scores = np.concatenate([beam_scores[done], self.combine(parent_scores, self.node_scores(q, kids))])
            keep = topk(node_scores, B)
            beam, beam_scores = nodes[keep], node_scores[keep]
        claims = np.concatenate([self.children(p, valid) for p in beam]) if len(beam) else np.array([], dtype=np.int64)
        return self._result(claims, self.node_scores(q, claims), None)


def build_index(patents: list[dict], embedder, progress: bool = False, **kwargs) -> tuple[Index, dict]:
    """-> (index, timings in seconds)."""
    t0 = time.perf_counter()
    tree = Tree(patents)
    t1 = time.perf_counter()
    vecs = embedder.embed([tree.claim[leaf]["text"] for leaf in tree.leaves], progress=progress)
    t2 = time.perf_counter()
    index = Index(tree, vecs, **kwargs)
    t3 = time.perf_counter()
    return index, {"tree_s": t1 - t0, "embed_s": t2 - t1, "keys_s": t3 - t2}
