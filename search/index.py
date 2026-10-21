"""Node keys + flat and beam search over the CPC tree.

Swappable pieces (all plain functions passed to Index):
  score(q, K) -> scores        vectorised similarity of one query to a key matrix (default cosine)
  aggregate(child_keys) -> key how an internal node's key is built from its children (default mean)
  project(X) -> X'             applied to leaf embeddings AND queries before keys are aggregated,
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
                 combine=own_score):
        self.tree, self.leaf_vecs = tree, leaf_vecs  # leaf_vecs rows follow tree.leaves
        self.score, self.aggregate, self.project, self.combine = score, aggregate, project, combine
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

    def _result(self, claims: np.ndarray, scores: np.ndarray, k: int | None) -> Result:
        order = np.lexsort((claims, -scores))  # ties (identical claim texts are common) broken by node id
        claims, scores = claims[order], scores[order]
        pats = self.tree.parent[claims]
        first = np.sort(np.unique(pats, return_index=True)[1])  # each patent's best claim, in rank order
        return Result(claims[:k], scores[:k], pats[first][:k], scores[first][:k])

    def flat_search(self, query: np.ndarray, k: int | None = None, valid: np.ndarray | None = None) -> Result:
        """Score every independent claim; top-k claims and top-k patents (by best claim)."""
        q = l2n(self.project(query[None]))[0]
        if valid is None:
            return self._result(self.tree.leaves, self.score(q, self.leaf_K), k)
        keep = valid[self.tree.leaves]
        return self._result(self.tree.leaves[keep], self.score(q, self.leaf_K[keep]), k)

    def beam_search(self, query: np.ndarray, B: int = 10, start_node: int = 0,
                    valid: np.ndarray | None = None) -> Result:
        """Route down the tree keeping at most B nodes per level, until the beam holds (up to) B patents;
        then score all independent claims of those patents. Returns every claim of the B patents, ranked.

        Nodes that reach patent level early (a shallower CPC path) are carried along and compete with
        the next level's candidates on their own score.
        """
        q = l2n(self.project(query[None]))[0]
        beam, beam_scores = np.array([start_node]), np.array([0.0])
        while True:
            done = self.tree.level[beam] >= PATENT
            if done.all():
                break
            kids = [self.children(n, valid) for n in beam[~done]]
            parent_scores = np.repeat(beam_scores[~done], [len(c) for c in kids])
            kids = np.concatenate(kids)
            cand = np.concatenate([beam[done], kids])
            cand_scores = np.concatenate([beam_scores[done], self.combine(parent_scores, self.score(q, self.K[kids]))])
            keep = topk(cand_scores, B)
            beam, beam_scores = cand[keep], cand_scores[keep]
        claims = np.concatenate([self.children(p, valid) for p in beam]) if len(beam) else np.array([], dtype=np.int64)
        return self._result(claims, self.score(q, self.K[claims]), None)


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
