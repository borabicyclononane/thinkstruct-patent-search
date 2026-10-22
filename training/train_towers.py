"""Train a query tower and a key tower (768 -> 64) on frozen PatentSBERTa embeddings; evaluate claim retrieval.

  local:  .venv/bin/python3.14 training/train_towers.py [--bundle training/bundles/small_100.npz]
  remote: agave run training/train_towers.py --on colab   (defaults only; results go to $OUTPUT_DIR)

Needs numpy, torch and search/ (the Part 1 index, for the search eval). The backbone is never touched: the bundle
(training/prepare.py) holds its 768-d embeddings of every independent claim (keys) and of the synthetic queries.

Towers: tower(x) = normalize(W x + MLP(x)), W a 768 -> 64 linear map, MLP 768 -> hidden -> 64 with its last layer
at zero. Both towers start from the SAME random orthonormal W, so at initialisation u.v ~ raw cosine (a random
projection roughly preserves angles); they are free to diverge from the first step.

Training pairs: (synthetic query, a source claim) from train-split queries; a combination gives two pairs.
Claim loss: InfoNCE over cosines in 64-d. Each query is scored against every key of its step: the batch's positives,
one mined hard negative per pair (nearest non-source train-split claim by raw 768-d cosine) and a fresh random
sample of train-split claims. Keys that are true sources of the query (e.g. claim B when the target is A) are masked.
Routing loss (--routing-weight): every step, 64-d keys for all CPC-tree nodes are built from the train-split claims'
keys exactly as the index builds them (patent = normalised mean of its claims' keys, subgroup = of its patents', ...);
at each level (patent, subgroup, main group, subclass) a cross-entropy asks the query to score its source's ancestor
above every other node of that level; the other source's ancestor of a combination is masked. Only train-split claims
are ever keys in training. Loss = claim loss + routing_weight x mean over levels.
Per-level query towers (--query-towers per_level): a separate query tower for each level the beam scores (claim,
patent, subgroup, main group, subclass; section and class have a single node and use the subclass tower), all from the
same W. The claim loss trains the claim tower, each level's routing loss trains that level's tower; one key tower.

Evaluation 1, ranking: each eval query ranks all claims. Recall@k = every source in the top k (a combination needs
both); MRR uses the rank of the last source.
Evaluation 2, search (the one that matters): the Part 1 index (search/index.py) over all claims, with the key map as
`project` and the query map as `project_query`; each eval query goes through flat search and beam search at each
width: does its source claim come back (a combination: both sources), and at what rank (R@1, R@10); does at least
one source of a combination come back.
Methods: raw 768-d cosine; PCA-64 (fit on train-split claims, applied to queries and claims); towers at init;
towers trained with the claim loss only; towers trained with claim + routing loss.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from search.index import Index  # noqa: E402
from search.tree import CLAIM, Tree  # noqa: E402
from search.tree import LEVELS as TREE_LEVELS  # noqa: E402

KS = (1, 10, 100)
LEVELS = ["patent", "subgroup", "main_group", "subclass"]  # bottom-up, as the index aggregates
TOWER_OF_LEVEL = [{"root": "subclass", "section": "subclass", "class": "subclass"}.get(l, l) for l in TREE_LEVELS]


class Tower(nn.Module):
    def __init__(self, W0: torch.Tensor, hidden: int, dropout: float):
        super().__init__()
        dim_out, dim_in = W0.shape
        self.proj = nn.Linear(dim_in, dim_out, bias=False)
        self.proj.weight.data.copy_(W0)
        self.mlp = nn.Sequential(nn.Linear(dim_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return F.normalize(self.proj(x) + self.mlp(x), dim=-1)


def hard_negatives(qv: np.ndarray, kv: np.ndarray, answers: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Per query: the most similar claim (raw cosine) among `allowed` claims that is not one of its sources."""
    s = qv @ kv.T
    s[:, ~allowed] = -np.inf
    rows = np.arange(len(qv))
    for col in range(answers.shape[1]):
        ok = answers[:, col] >= 0
        s[rows[ok], answers[ok, col]] = -np.inf
    return s.argmax(1)


def hierarchy(b, claims: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """For `claims`: per level, child -> parent index (claims -> patents -> subgroups -> main groups -> subclasses),
    and per level each claim's ancestor index."""
    docs = np.array([c.split("#")[0] for c in b["claim_ids"][claims]])
    paths = b["claim_paths"][claims]
    names = [docs, paths[:, 4], paths[:, 3], paths[:, 2]]
    parents, ancestors, child = [], [], None
    for lvl, name in enumerate(names):
        ids, anc = np.unique(name, return_inverse=True)
        ancestors.append(anc)
        if lvl == 0:
            parents.append(anc)  # claim -> patent
        else:  # previous level's node -> this level's node (via any of its claims)
            parents.append(np.array([anc[np.flatnonzero(child == i)[0]] for i in range(child.max() + 1)]))
        child = anc
    return parents, ancestors


def evaluate(zq: np.ndarray, zk: np.ndarray, answers: np.ndarray, datasets: np.ndarray) -> dict:
    s = zq @ zk.T
    order = np.argsort(-s, axis=1)
    rank = np.zeros(answers.shape)  # 0 = no source in this column, never the max
    for col in range(answers.shape[1]):
        ok = answers[:, col] >= 0
        rank[ok, col] = np.argmax(order[ok] == answers[ok, col][:, None], axis=1) + 1
    last = rank.max(1)  # a combination counts once both sources are in
    out = {}
    for name in ["all", *sorted(set(datasets))]:
        m = np.ones(len(s), bool) if name == "all" else datasets == name
        row = {"n": int(m.sum()), "mrr": float((1 / last[m]).mean()), **{f"recall@{k}": float((last[m] <= k).mean()) for k in KS}}
        rng = np.random.default_rng(0)  # bootstrap 95% CI of the MRR
        boot = [(1 / last[m][rng.integers(0, m.sum(), m.sum())]).mean() for _ in range(1000)]
        row["mrr_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out[name] = row
    return out


_TREES = {}


def claim_tree(b) -> tuple[Tree, dict]:
    """The Part 1 CPC tree over all claims of the bundle (built once), and leaf node -> claim index."""
    if id(b) not in _TREES:
        patents = {}
        for i, (cid, path) in enumerate(zip(b["claim_ids"], b["claim_paths"])):
            doc, num = cid.split("#")
            patents.setdefault(doc, {"doc_number": doc, "path": tuple(path), "claims": []})["claims"].append(
                {"num": int(num), "independent": True, "idx": i})
        tree = Tree(list(patents.values()))
        _TREES[id(b)] = (tree, {leaf: tree.claim[leaf]["idx"] for leaf in tree.leaves})
    return _TREES[id(b)]


def search_eval(b, qproj, kproj, eval_q: np.ndarray, widths: list[int], flat: bool = True) -> list[dict]:
    """Eval queries through the Part 1 index (flat search, beam search per width); one row per (search, dataset)."""
    tree, claim_of_leaf = claim_tree(b)
    index = Index(tree, b["claim_vecs"][[claim_of_leaf[l] for l in tree.leaves]], project=kproj, project_query=qproj)
    rows = []
    for name, run in [("flat", lambda q: index.flat_search(q))][:int(flat)] + [
            (f"beam B={w}", lambda q, w=w: index.beam_search(q, w)) for w in widths]:
        per = []
        for qi in eval_q:
            got = [claim_of_leaf[l] for l in run(b["query_vecs"][qi]).claims]
            src = [a for a in b["answers"][qi] if a >= 0]
            pos = [got.index(a) + 1 for a in src if a in got]
            found = len(pos) == len(src)  # a combination needs both sources
            per.append({"dataset": b["query_dataset"][qi], "returned": len(got), "found": found,
                        "found_any": bool(pos), "r@1": found and max(pos) <= 1, "r@10": found and max(pos) <= 10})
        for ds in ["all", *sorted({p["dataset"] for p in per})]:
            sel = [p for p in per if ds == "all" or p["dataset"] == ds]
            two = [p for p in sel if p["dataset"] == "combination"]
            rows.append({"search": name, "dataset": ds, "n": len(sel), "claims_returned": float(np.mean([p["returned"] for p in sel])),
                         "source_returned": float(np.mean([p["found"] for p in sel])),
                         "r@1": float(np.mean([p["r@1"] for p in sel])), "r@10": float(np.mean([p["r@10"] for p in sel])),
                         "any_source_returned": float(np.mean([p["found_any"] for p in two])) if two else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "bundles", "small_100.npz"))
    ap.add_argument("--eval-split", default="val", choices=["val", "test"], help="test stays held out until the end")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lrs", default="1e-3,3e-4,1e-4", help="learning rates to try")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--random-negatives", type=int, default=256, help="random training claims added as keys each step")
    ap.add_argument("--routing-weights", default="0,1,3", help="weights of the CPC-tree routing loss to try")
    ap.add_argument("--query-towers", default="shared,per_level",
                    help="query models to try: shared (one for all levels) and/or per_level (one per tree level)")
    ap.add_argument("--dev-frac", type=float, default=0.1, help="share of train patents held out as dev for model selection")
    ap.add_argument("--select-widths", default="5,10,30", help="dev score = mean share of sources returned at these beam widths")
    ap.add_argument("--beam-widths", default="1,5,10,30,100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(HERE, "runs", time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    t_start = time.time()

    b = dict(np.load(args.bundle))
    kv, qv, answers = b["claim_vecs"], b["query_vecs"], b["answers"]
    docs = np.array([c.split("#")[0] for c in b["claim_ids"]])
    train_docs = sorted(set(docs[b["claim_split"] == "train"]))
    dev_docs = set(np.random.default_rng(args.seed).permutation(train_docs)[:round(args.dev_frac * len(train_docs))])
    fit_c = np.flatnonzero((b["claim_split"] == "train") & ~np.isin(docs, list(dev_docs)))  # the only keys in training
    in_dev = lambda q: any(docs[a] in dev_docs for a in answers[q] if a >= 0)
    train_all = np.flatnonzero(b["query_split"] == "train")
    dev_q = np.array([q for q in train_all if in_dev(q)])
    fit_q = np.array([q for q in train_all if not in_dev(q)])
    eval_q = np.flatnonzero(b["query_split"] == args.eval_split)
    print(f"{len(kv)} claims ({len(fit_c)} used as training keys); queries: {len(fit_q)} train, {len(dev_q)} dev "
          f"({len(dev_docs)} train patents held out for model selection), {len(eval_q)} {args.eval_split}", flush=True)

    allowed = np.zeros(len(kv), bool)
    allowed[fit_c] = True
    hard = hard_negatives(qv[fit_q], kv, answers[fit_q], allowed)
    pairs = [(q, a, h) for q, h in zip(fit_q, hard) for a in answers[q] if a >= 0]  # (query, source, hard negative)
    parents, ancestors = hierarchy(b, fit_c)
    local = np.full(len(kv), -1)
    local[fit_c] = np.arange(len(fit_c))  # claim index -> position among training claims

    Kt, Qt, A = torch.tensor(kv, device=device), torch.tensor(qv, device=device), torch.tensor(answers, device=device)
    P = torch.tensor(pairs, device=device)
    TC = torch.tensor(fit_c, device=device)
    LOCAL = torch.tensor(local, device=device)
    PAR = [torch.tensor(p, device=device) for p in parents]
    ANC = [torch.tensor(np.append(a, -1), device=device) for a in ancestors]  # index -1 (no source) -> -1
    gen = torch.Generator().manual_seed(args.seed)
    W0 = torch.linalg.qr(torch.randn(kv.shape[1], args.dim, generator=gen))[0].T.contiguous()  # orthonormal rows
    select_widths = [int(w) for w in args.select_widths.split(",")]

    def node_keys(kz):
        """64-d keys of every CPC-tree node from the training claims' keys, as the index aggregates (normalised means)."""
        keys, cur = [], kz
        for par in PAR:
            cur = F.normalize(torch.zeros(int(par.max()) + 1, kz.shape[1], device=device).index_add(0, par, cur), dim=-1)
            keys.append(cur)
        return keys

    def fn(tower):
        def f(x):
            tower.eval()
            with torch.no_grad():
                return tower(torch.tensor(np.asarray(x, dtype=np.float32), device=device)).cpu().numpy()
        return f

    def qfn(qts):
        """Query map for the index: (n, d) for one shared tower, else (n, tree levels, d), each level by its tower."""
        if len(qts) == 1:
            return fn(qts["claim"])
        def f(x):
            z = {name: fn(t)(x) for name, t in qts.items()}
            return np.stack([z[name] for name in TOWER_OF_LEVEL], axis=1)
        return f

    def dev_score(qts, kt) -> float:
        rows = search_eval(b, qfn(qts), fn(kt), dev_q, select_widths, flat=False)
        return float(np.mean([r["source_returned"] for r in rows if r["dataset"] == "all"]))

    def train(routing_weight: float, lr: float, per_level: bool, label: str):
        torch.manual_seed(args.seed)
        qt = nn.ModuleDict({name: Tower(W0, args.hidden, args.dropout) for name in ["claim", *LEVELS][:5 if per_level else 1]}).to(device)
        kt = Tower(W0, args.hidden, args.dropout).to(device)
        opt = torch.optim.AdamW([*qt.parameters(), *kt.parameters()], lr=lr, weight_decay=args.weight_decay)
        best = (dev_score(qt, kt), 0, None)  # epoch 0 = the untrained towers
        curve = [{"run": label, "epoch": 0, "claim_loss": None, "routing_loss": None, "dev_score": best[0]}]
        for epoch in range(1, args.epochs + 1):
            qt.train(), kt.train()
            sums = np.zeros(2)
            for idx in torch.randperm(len(P), device=device).split(args.batch):
                q, pos, neg = P[idx, 0], P[idx, 1], P[idx, 2]
                rand = TC[torch.randint(len(TC), (args.random_negatives,), device=device)]
                keys = torch.cat([pos, neg, rand])  # B positives, B hard negatives, random training claims
                u = qt["claim"](Qt[q])
                logits = u @ kt(Kt[keys]).T / args.temperature
                is_source = (keys[None, :, None] == A[q][:, None, :]).any(-1)  # other sources of the query: not negatives
                target = torch.arange(len(q), device=device)
                is_source[target, target] = False
                claim_loss = F.cross_entropy(logits.masked_fill(is_source, float("-inf")), target)
                route_loss = torch.zeros((), device=device)
                if routing_weight:
                    src_local, tgt_local = LOCAL[A[q]], LOCAL[pos]  # positions among training claims, -1 if none
                    for lvl, nk in enumerate(node_keys(kt(Kt[TC]))):
                        tgt, other = ANC[lvl][tgt_local], ANC[lvl][src_local]
                        u_lvl = qt[LEVELS[lvl]](Qt[q]) if per_level else u
                        lg = u_lvl @ nk.T / args.temperature
                        mask = torch.zeros_like(lg, dtype=torch.bool)
                        for col in range(other.shape[1]):  # the query's other source's ancestor is not a negative
                            ok = (other[:, col] >= 0) & (other[:, col] != tgt)
                            mask[ok.nonzero()[:, 0], other[ok, col]] = True
                        route_loss = route_loss + F.cross_entropy(lg.masked_fill(mask, float("-inf")), tgt) / len(LEVELS)
                loss = claim_loss + routing_weight * route_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                sums += [claim_loss.item(), route_loss.item()]
            score = dev_score(qt, kt)
            n_steps = -(-len(P) // args.batch)
            curve.append({"run": label, "epoch": epoch, "claim_loss": sums[0] / n_steps, "routing_loss": sums[1] / n_steps,
                          "dev_score": score})
            if score > best[0]:
                best = (score, epoch, ({k: v.clone() for k, v in qt.state_dict().items()},
                                       {k: v.clone() for k, v in kt.state_dict().items()}))
        if best[2]:
            qt.load_state_dict(best[2][0]), kt.load_state_dict(best[2][1])
        print(f"  {label:40s} best dev score {best[0]:.3f} at epoch {best[1]:2d} (untrained {curve[0]['dev_score']:.3f}); "
              f"{time.time() - t_start:.0f} s so far", flush=True)
        return qt, kt, best[0], best[1], curve

    mu = kv[fit_c].mean(0)
    comps = np.linalg.svd(kv[fit_c] - mu, full_matrices=False)[2][:args.dim]
    pca = lambda x: F.normalize(torch.tensor((np.asarray(x) - mu) @ comps.T), dim=-1).numpy()
    q0, k0 = Tower(W0, args.hidden, args.dropout).to(device), Tower(W0, args.hidden, args.dropout).to(device)
    methods = {"raw 768-d cosine": (lambda x: x, lambda x: x), f"PCA-{args.dim}": (pca, pca), "towers untrained": (fn(q0), fn(k0))}
    per_level_opts = [{"shared": False, "per_level": True}[x] for x in args.query_towers.split(",")]
    dev = {name: float(np.mean([r["source_returned"] for r in search_eval(b, *m, dev_q, select_widths, flat=False)
                                if r["dataset"] == "all"])) for name, m in methods.items()}
    curves, saved = [], {}
    print(f"training on {len(pairs)} pairs, {args.epochs} epochs per configuration; dev score = mean share of dev "
          f"sources returned by beam search at B={select_widths}", flush=True)
    grid = [(rw, per_level, lr) for rw in [float(x) for x in args.routing_weights.split(",")] for per_level in per_level_opts
            for lr in [float(x) for x in args.lrs.split(",")] if rw or not per_level]  # level towers need the routing loss
    for rw, per_level, lr in grid:
        label = f"towers routing={rw:g} lr={lr:g}" + (" per-level q" if per_level else "")
        qt, kt, score, epoch, curve = train(rw, lr, per_level, label)
        methods[label], dev[label] = (qfn(qt), fn(kt)), score
        saved[label] = {"best_epoch": epoch, "query_tower": qt.state_dict(), "key_tower": kt.state_dict()}
        curves += curve
    chosen = max((m for m in methods if m.startswith("towers routing")), key=lambda m: dev[m])

    widths = [int(w) for w in args.beam_widths.split(",")]
    ranking, search = {}, []
    for name, (qf, kf) in methods.items():
        zq = qf(qv[eval_q])
        ranking[name] = evaluate(zq[:, CLAIM] if zq.ndim == 3 else zq, kf(kv), answers[eval_q], b["query_dataset"][eval_q])
        search += [{"method": name, "dev_score": dev[name], **r} for r in search_eval(b, qf, kf, eval_q, widths)]

    json.dump({"config": vars(args), "n_train_pairs": len(pairs), "n_dev_queries": len(dev_q), "eval_split": args.eval_split,
               "chosen_on_dev": chosen, "dev_score": dev, "ranking": ranking},
              open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    for fname, rows in (("search_eval.csv", search), ("curve.csv", curves)):
        with open(os.path.join(out_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    torch.save({"config": vars(args), "W0": W0, "chosen": chosen, **saved}, os.path.join(out_dir, "towers.pt"))

    searches = list(dict.fromkeys(r["search"] for r in search))
    for ds in ["all", *sorted(set(b["query_dataset"][eval_q]))]:
        n = next(r["n"] for r in search if r["dataset"] == ds)
        print(f"\nHEADLINE ({ds}, {n} {args.eval_split} queries): share whose source claim comes back from the search")
        print(f"{'':42s} {'dev':>5s}  " + " ".join(f"{s:>10s}" for s in searches))
        for name in methods:
            cells = [next(r for r in search if r["method"] == name and r["search"] == s and r["dataset"] == ds) for s in searches]
            mark = "  <- chosen on dev" if name == chosen else ""
            print(f"{name:42s} {dev[name]:5.3f}  " + " ".join(f"{c['source_returned']:10.3f}" for c in cells) + mark)
    print(f"\nRANKING ({len(eval_q)} {args.eval_split} queries, all {len(kv)} claims)")
    for name, res in ranking.items():
        r = res["all"]
        print(f"{name:42s} MRR {r['mrr']:.3f} [{r['mrr_ci95'][0]:.2f}, {r['mrr_ci95'][1]:.2f}]  R@1 {r['recall@1']:.2f}  "
              f"R@10 {r['recall@10']:.2f}  R@100 {r['recall@100']:.2f}")
    print(f"\n{time.time() - t_start:.0f} s; written to {out_dir}/ (metrics.json, search_eval.csv, curve.csv, towers.pt)")


if __name__ == "__main__":
    main()
