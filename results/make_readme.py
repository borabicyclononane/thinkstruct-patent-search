"""Rebuild results/README.md from the CSVs in results/ (and experiment-big-data/results_10000/).

  .venv/bin/python3.14 results/make_readme.py      (or: make readme)

Tables are padded so the raw markdown is aligned too.
"""
import os

import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def table(header, rows, left=1):
    """Markdown table padded so the raw text is aligned; first `left` columns left-aligned, the rest right-aligned."""
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w[i]) if i < left else c.rjust(w[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w[i] if i < left else "-" * (w[i] - 1) + ":" for i in range(len(w))) + " |"
    return "\n".join([fmt(header), sep] + [fmt(r) for r in rows])

f3 = lambda x: f"{x:.3f}"
def pval(p): return "p<0.001" if p < 0.001 else f"p={p:.3f}"
d = {n: pd.read_csv(f"results/part1_{n}.csv") for n in ["default", "sample10000"]}

speed = []
for B in [1, 5, 10, 100]:
    r = []
    for n in d:
        x = d[n].drop_duplicates(["B", "search"]).set_index(["B", "search"])
        fl, be = x.loc[(B, "flat"), "ms_median"], x.loc[(B, "beam"), "ms_median"]
        r += [f3(fl), f3(be), f"{fl / be:.1f}x"]
    speed.append([B] + r)
speed_t = table(["B", "default: flat", "default: beam", "flat ÷ beam", "10^4: flat", "10^4: beam", "flat ÷ beam"], speed)

x = d["default"].set_index(["B", "search", "dataset"])
acc = []
for B in [1, 5, 10, 100]:
    for sch in ("flat", "beam"):
        y = x.loc[(B, sch, "all")]
        sig = "—" if sch == "flat" else f"+{int(y.vs_flat_better)} / −{int(y.vs_flat_worse)}, {pval(y.vs_flat_p)}"
        acc.append([B if sch == "flat" else "", sch] + [f3(x.loc[(B, sch, ds), "source_returned"])
                    for ds in ("paraphrase", "element_reuse", "combination", "all")] + [sig])
acc_t = table(["B", "search", "paraphrase", "element reuse", "combination", "all", "all: beam vs flat"], acc, left=2)

def hybrid(tpath):
    t = pd.read_csv(tpath).pivot_table(index=["scenario", "n_candidates"], columns="method", values="ms_median", sort=False)
    rows = []
    for (s, n), r in t.iterrows():
        if s == "no filter":
            rows.append([s, n, f3(r["flat"]), "", f3(r["beam B=1"]), f3(r["beam B=5"]), f3(r["beam"])])
        else:
            rows.append([s, n, f3(r["flat post-filter"]), f3(r["flat pre-filter"]), f3(r["beam pruned B=1"]),
                         f3(r["beam pruned B=5"]), f3(r["beam pruned"])])
    return table(["filter", "candidate patents", "flat post-filter", "flat pre-filter", "beam B=1", "beam B=5", "beam B=10"], rows)

p3 = pd.read_csv("results/part3.csv")
train_s = {"shared towers, no routing loss": "8 s", "shared towers": "14 s", "per-level towers": "26 s"}
label = {"flat, raw 768": "flat search, raw 768-d", "raw 768": "raw 768-d", "PCA-64": "PCA-64", "shared towers, no routing loss": "shared towers, no routing loss",
         "shared towers": "shared towers", "per-level towers": "per-level towers"}
def part3(B):
    """Test table at beam width B."""
    rows = []
    for m in label:
        t = p3[(p3.split == "test") & (p3.method == m) & (p3.B == B)].set_index("dataset")
        v = p3[(p3.split == "val") & (p3.method == m) & (p3.dataset == "all") & (p3.B == B)].iloc[0]
        cfg = t.loc["all", "config"] if isinstance(t.loc["all", "config"], str) else "—"
        cfg = cfg.replace("towers ", "").replace(" per-level q", "").replace("lr=0.0001", "lr=1e-4").replace("lr=0.0003", "lr=3e-4").replace("lr=0.001", "lr=1e-3")
        sig = "—" if m == "raw 768" else f"+{t.loc['all', 'vs_raw_better']} / −{t.loc['all', 'vs_raw_worse']}, {pval(t.loc['all', 'vs_raw_p'])}"
        rows.append([label[m], cfg, f3(t.loc["paraphrase", "source_returned"]), f3(t.loc["element_reuse", "source_returned"]),
                     f3(t.loc["combination", "source_returned"]), f3(t.loc["all", "source_returned"]), sig, f3(v.source_returned),
                     f3(t.loc["all", "ms_median"]), train_s.get(m, "—")])
    return table(["method", "config", "paraphrase", "element reuse", "combination", "all", "all vs raw 768", "val all",
                  "ms / query", "training"], rows, left=2)


readme = f"""# Results

All numbers come from the commands below, run on one Apple M3 CPU (numpy 2.5, torch 2.14).
Latency is the median per query, measured from the query's 768-d embedding.
Embedding the query text with PatentSBERTa adds about 36 ms, the same for every method.

```
make part1   # default dataset: Part 1 + 1b
make large   # 10^4 sample: Part 1 + 1b
make part3   # train and evaluate the towers
make readme  # this file
```

## Part 1: is beam search better than flat search?

**What is compared.** Beam search with width B returns all claims of B patents. Flat search scores every claim and keeps all claims of its top B patents.

**Accuracy** (default dataset only). The queries are the 357 Part 3 test-split synthetic queries (119 per task), searched against one index over all 640 patents. The metric is the share of queries whose source claim is among the returned claims; a combination counts only if **both** of its source claims come back. "beam vs flat" gives the queries only beam finds and the queries only flat finds, with an exact two-sided sign test. Beam B=5 equals the raw 768-d row of Part 3.

**Speed.**
- **Default dataset:** the same 357 queries.
- **10^4 sample:** 1,000 held-out patents' first independent claims, searched against the other 9,000 patents (all CPC sections). Accuracy isn't measured here, because there are no synthetic queries for this sample.

### Accuracy: default dataset

{acc_t}

### Speed (ms per query)

"flat ÷ beam" above 1 means beam search is faster.

{speed_t}

## Part 1b: hybrid search speed

**Setup.**
- **Default dataset:** every patent's first claim is a query against all 640 patents, its own included.
- **10^4 sample:** 300 held-out queries against the 9,000 indexed patents.
- Flat search keeps the top 10 patents. Beam search is timed at B=1, 5 and 10.

**Methods compared** (ms per query; resolving the filter is included):
- **flat post-filter:** score every claim, then drop the non-candidates.
- **flat pre-filter:** score only the candidates' claims.
- **beam:** start at the filter's CPC nodes and score only children that contain a candidate.

**Checks** (the script stops if any fails):
- pre- and post-filtering return the same ranking;
- beam search returns only patents that pass the filter;
- beam search with unlimited width returns exactly the flat result.

The "no filter" row shows plain flat search and plain beam search.

### Default dataset (640 patents)

{hybrid("results/hybrid_timing.csv")}

### 10^4 sample (9,000 patents)

{hybrid("experiment-big-data/results_10000/hybrid_timing.csv")}

## Part 3: training and evaluation

**Queries.** 3,573 synthetic queries made with Claude Haiku 5.5 from the 1,191 independent claims of the default dataset (`synthetic/`):
- **paraphrase:** the claim reworded.
- **element reuse:** a new claim that reuses 1–2 parts of it.
- **combination:** a new claim mixing two claims.

**Split.** Train / val / test by patent, 80/10/10. Test has 119 queries per dataset.

**Metric.** The share of queries whose source claim is among the claims the search returns at **B=5** and **B=30**, over a tree of all 1,191 claims. A combination counts only if **both** of its source claims come back.

**Methods.** All are on frozen PatentSBERTa embeddings.
- **flat search, raw 768-d:** a reference, not beam search. It scores every claim and keeps all claims of its top B patents, as in Part 1.
- **raw 768-d** and **PCA-64:** PCA is fit on the training claims.
- **Towers:** a 768 → 64 query tower and key tower, trained with InfoNCE. "Routing loss" adds a cross-entropy at each CPC level, so the query scores its source's ancestors above the other nodes. **Per-level towers** use a separate query tower for each level of the tree.
- **Tower configuration:** each tower row uses the grid configuration with the best **val** score at B=5. The tables therefore report **test**, which was not used for any choice.
- **all vs raw 768:** test queries where the method finds the source and raw 768-d beam search doesn't, then the reverse, with an exact two-sided sign test.
- **Training:** wall time for 20 epochs on CPU, including a beam-search evaluation on the dev set after every epoch.

**B=5**

{part3(5)}

**B=30** (same configurations, chosen at B=5)

{part3(30)}

Val results and per-dataset sign tests are in `part3.csv`. The full training grid is in `../training/runs/final/`.
"""
open("results/README.md", "w").write(readme)
