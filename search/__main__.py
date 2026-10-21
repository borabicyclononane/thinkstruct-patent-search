"""Hierarchical patent claim search.

  python -m search build                          data report, embed independent claims, build the tree
  python -m search query "a wheel hub with ..."   free-text query
  python -m search query --doc 20240051333 --claim 1
  python -m search eval                           held-out eval + timing, tables and plots written to results/
  python -m search plots                          redraw the plots from results/
"""
import argparse
import os
import textwrap

import pandas as pd

from .data import DATA_DIR, SCRAPED, load_corpus, print_report, clean
from .embed import CACHE, Embedder
from .evaluate import run_eval
from .index import build_index
from .plots import make_plots

ap = argparse.ArgumentParser(prog="python -m search")
ap.add_argument("--data-dir", default=DATA_DIR)
ap.add_argument("--scraped", default=SCRAPED, help="output of python -m scraper")
ap.add_argument("--cache", default=CACHE, help="embedding cache")
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("build")
q = sub.add_parser("query")
q.add_argument("text", nargs="?", help="free-text claim")
q.add_argument("--doc", help="query with a claim from the corpus: doc number")
q.add_argument("--claim", type=int, default=1, help="claim number for --doc")
q.add_argument("--k", type=int, default=10)
q.add_argument("--B", type=int, default=10, help="beam width")
q.add_argument("--flat", action="store_true", help="flat search instead of beam search")
q.add_argument("--include-self", action="store_true", help="with --doc, keep the query's own patent in results")
e = sub.add_parser("eval")
e.add_argument("--folds", type=int, default=5, help="cross-validation folds; every patent is a query once")
e.add_argument("--seed", type=int, default=0)
e.add_argument("--out", default="results")
pl = sub.add_parser("plots", help="redraw plots from the tables written by eval")
pl.add_argument("--out", default="results")
args = ap.parse_args()

patents, report = load_corpus(args.data_dir, args.scraped)
embedder = Embedder(args.cache)

if args.cmd == "build":
    print_report(patents, report)
    print()
    index, t = build_index(patents, embedder, progress=True)
    index.tree.print_report()
    texts = [index.tree.claim[leaf]["text"] for leaf in index.tree.leaves]
    n_trunc, max_len, longest = embedder.truncation(texts)
    print(f"\nmodel {embedder.model_name}: dim {index.K.shape[1]}, max_seq_length {max_len}; "
          f"{n_trunc}/{len(texts)} independent claims truncated (longest {longest} tokens)")
    print(f"build: tree {t['tree_s'] * 1000:.1f} ms, embeddings {t['embed_s']:.1f} s "
          f"(fresh embedding cost so far: {embedder.meta['embed_seconds']:.1f} s for {embedder.meta['n_embedded']} texts), "
          f"keys {t['keys_s'] * 1000:.1f} ms")

elif args.cmd == "query":
    index, _ = build_index(patents, embedder)
    tree = index.tree
    if args.doc:
        p = next((p for p in patents if p["doc_number"] == args.doc), None)
        claim = p and next((c for c in p["claims"] if c["num"] == args.claim), None)
        if claim is None:
            ap.error(f"no claim {args.claim} for patent {args.doc} in the index")
        text = claim["text"]
    elif args.text:
        text = clean(args.text)
    else:
        ap.error("give a query text or --doc")
    valid = None if args.include_self or not args.doc else index.exclude_patents([args.doc])
    qv = embedder.embed([text])[0]
    res = index.flat_search(qv, args.k, valid) if args.flat else index.beam_search(qv, args.B, valid=valid)
    print("query:", textwrap.shorten(text, 200), "\n")
    print(f"top {args.k} independent claims ({'flat' if args.flat else f'beam, B={args.B}'}):")
    for rank, (leaf, s) in enumerate(zip(res.claims[:args.k], res.claim_scores), 1):
        p = tree.patent[tree.parent[leaf]]
        print(f"{rank:3d}. {s:.3f}  {p['doc_number']} claim {tree.claim[leaf]['num']:<3} {p['path'][-1]:12s} {p['title'][:60]}")
        print(textwrap.indent(textwrap.shorten(tree.claim[leaf]["text"], 220), " " * 12))
    print("\npatents (by best claim):")
    for rank, (n, s) in enumerate(zip(res.patents[:args.k], res.patent_scores), 1):
        p = tree.patent[n]
        print(f"{rank:3d}. {s:.3f}  {p['doc_number']}  {p['path'][-1]:12s} {p['title'][:70]}")

elif args.cmd == "eval":
    per_query, metrics, timing = run_eval(patents, embedder, args.folds, args.seed)
    os.makedirs(args.out, exist_ok=True)
    per_query.to_csv(os.path.join(args.out, "eval_per_query.csv"), index=False)
    metrics.to_csv(os.path.join(args.out, "eval.csv"), index=False)
    timing.to_csv(os.path.join(args.out, "timing.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(metrics.to_string(index=False), "\n")
        print(timing.to_string(index=False))
    plots = make_plots(per_query, timing, os.path.join(args.out, "plots"))
    print(f"\nwritten to {args.out}/ (eval.csv, eval_per_query.csv, timing.csv) and {len(plots)} plots in {args.out}/plots/")

elif args.cmd == "plots":
    plots = make_plots(pd.read_csv(os.path.join(args.out, "eval_per_query.csv")),
                       pd.read_csv(os.path.join(args.out, "timing.csv")), os.path.join(args.out, "plots"))
    print("\n".join(plots))
