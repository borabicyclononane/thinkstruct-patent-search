"""CPC tree: root -> section -> class -> subclass -> main group -> subgroup -> patent -> independent claim.

Nodes are integer ids. `children[n]` is an int array, so all children of a node
(or of a whole beam, concatenated) can be scored with one matrix operation.
"""
import numpy as np

from .data import CPC_LEVELS

LEVELS = ["root", *CPC_LEVELS, "patent", "claim"]
PATENT, CLAIM = LEVELS.index("patent"), LEVELS.index("claim")


class Tree:
    def __init__(self, patents: list[dict]):
        self.name: list[str] = []
        level, parent, kids, ids = [], [], [], {}

        def node(key, name, lvl, par):
            if key not in ids:
                ids[key] = len(self.name)
                self.name.append(name)
                level.append(lvl)
                parent.append(par)
                kids.append([])
                if par >= 0:
                    kids[par].append(ids[key])
            return ids[key]

        root = node((), "root", 0, -1)
        self.patent: dict[int, dict] = {}  # patent node -> patent
        self.claim: dict[int, dict] = {}  # claim leaf -> claim
        for p in patents:
            n = root
            for depth in range(1, len(p["path"]) + 1):
                n = node(p["path"][:depth], p["path"][depth - 1], depth, n)
            n = node(("patent", p["doc_number"]), p["doc_number"], PATENT, n)
            self.patent[n] = p
            for c in p["claims"]:
                if c["independent"]:
                    leaf = node(("claim", p["doc_number"], c["num"]), f"{p['doc_number']}#{c['num']}", CLAIM, n)
                    self.claim[leaf] = c
        self.level = np.array(level)
        self.parent = np.array(parent)
        self.children = [np.array(k, dtype=np.int64) for k in kids]

        # Depth-first order of patents ("positions"); each node covers the contiguous positions [start, end)
        # (a claim covers its patent's position). Leaves are stored in the same order, so the independent
        # claims of position i are leaves[leaf_offset[i]:leaf_offset[i + 1]].
        order, self.start, self.end = [], np.zeros(len(self.name), np.int64), np.zeros(len(self.name), np.int64)

        def visit(n):
            self.start[n] = len(order)
            if level[n] == PATENT:
                order.append(n)
                self.start[kids[n]], self.end[kids[n]] = self.start[n], self.start[n] + 1
            else:
                for c in kids[n]:
                    visit(c)
            self.end[n] = len(order)

        visit(root)
        self.order = np.array(order, dtype=np.int64)  # position -> patent node
        self.leaves = np.concatenate([self.children[n] for n in self.order])
        self.leaf_offset = np.concatenate([[0], np.cumsum([len(self.children[n]) for n in self.order])])
        self.cpc_node = {self.name[n]: n for n in range(len(self.name)) if 0 < level[n] < PATENT}  # "B60B 19" -> node
        self.patent_node = {p["doc_number"]: n for n, p in self.patent.items()}

    def print_report(self) -> None:
        """Nodes per level, and how many patents sit under each node."""
        under = np.zeros(len(self.name), dtype=int)
        for n in self.patent:
            while n >= 0:
                under[n] += 1
                n = self.parent[n]
        print("tree level    nodes  patents per node (min / median / max)")
        for lvl in range(1, PATENT):
            u = under[self.level == lvl]
            print(f"  {LEVELS[lvl]:11s} {len(u):5d}  {u.min()} / {np.median(u):.0f} / {u.max()}")
        print(f"  {'patent':11s} {len(self.patent):5d}")
        print(f"  {'claim':11s} {len(self.leaves):5d}  (independent claims, leaves)")
