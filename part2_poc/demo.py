"""Example queries through the range-sharded search (S=4), then mean shards touched per query, range vs hash sharding.

  .venv/bin/python3.14 part2_poc/demo.py      (after build.py); writes part2_poc/results.md
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build import BUNDLE, ROOT, load, load_built  # noqa: E402
from coordinator import Coordinator  # noqa: E402
from test_correctness import B, K, TYPES, make_queries  # noqa: E402


def table(header, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w[i]) if i < 2 else str(c).rjust(w[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(header), "| " + " | ".join("-" * w[i] if i < 2 else "-" * (w[i] - 1) + ":" for i in range(len(w))) + " |"]
                     + [line(r) for r in rows])


def main():
    pats, _ = load()
    by_doc = {p["doc"]: p for p in pats}
    b = np.load(BUNDLE)
    text = {}
    for ds in ("paraphrase", "element_reuse", "combination"):
        for line in open(os.path.join(ROOT, "outputs", "synthetic", f"{ds}.jsonl")):
            r = json.loads(line)
            text[r["query_id"]] = r["query_text"]

    # one test-split paraphrase whose source is a B60B patent, with three filters
    qi = next(i for i in np.flatnonzero((b["query_split"] == "test") & (b["query_dataset"] == "paraphrase"))
              if by_doc[str(b["claim_ids"][b["answers"][i][0]]).split("#")[0]]["path"][2] == "B60B")
    print(f"query {b['query_ids'][qi]}: {text[b['query_ids'][qi]][:200]}...")
    print(f"source claim: {b['claim_ids'][b['answers'][qi][0]]}\n")
    rng, hsh = Coordinator(load_built("range", 4)), Coordinator(load_built("hash", 4))
    for desc, f in [("CPC prefix B60B", {"cpc_prefix": "B60B"}),
                    ("keyword 'wheel'", {"keywords": ["wheel"]}),
                    ("keywords 'wheel' AND 'motor'", {"keywords": ["wheel", "motor"], "mode": "and"}),
                    ("CPC prefix B60C + keyword 'pressure'", {"cpc_prefix": "B60C", "keywords": ["pressure"]})]:
        res, tr = rng.search(b["query_vecs"][qi], K, B, **f)
        _, th = hsh.search(b["query_vecs"][qi], K, B, **f)
        print(f"== {desc}  (k={K}, B={B}, S=4)")
        for i, r in enumerate(res, 1):
            p = by_doc[r["doc"]]
            print(f"  {i}. {r['claim']:16s} {r['score']:.3f}  {p['path'][-1]:12s} {p['title'][:60]}")
        if not res:
            print("  (no results)")
        print("  valid children kept / total, per level: " + ", ".join(f"{l} {v}/{t}" for l, v, t in tr["levels"]))
        for name, t in (("range", tr), ("hash ", th)):
            print(f"  {name} shards contacted: routing {sorted(t['routing'])}, scoring {sorted(t['scoring'])}")
        print()

    queries = make_queries(pats)
    rows = []
    for typ in TYPES:
        qs = [q for q in queries if q["type"] == typ]
        for S in (2, 4):
            row = [typ, S]
            for scheme in ("range", "hash"):
                c = Coordinator(load_built(scheme, S))
                tr = [c.search(b["query_vecs"][q["qi"]], K, B, q["cpc_prefix"], q["keywords"], q["mode"])[1] for q in qs]
                row += [f"{np.mean([len(t[p]) for t in tr]):.2f}" for p in ("routing", "scoring")]
                row.append(f"{np.mean([len(t['routing'] | t['scoring']) for t in tr]):.2f}")
            rows.append(row)
    t = table(["query type", "S", "range routing", "range scoring", "range total", "hash routing", "hash scoring",
               "hash total"], rows)
    md = (f"# Part 2 PoC: shards touched per query\n\n"
          f"Mean shards contacted per query over {len(queries)} synthetic test-split queries ({len(queries) // 4} per "
          f"type), k={K}, B={B}, on the 640-patent default dataset. Routing: `count_in_ranges` calls during beam "
          f"search (only with keywords). Scoring: `exact_match` / `score` on the owners of the final B patents. "
          f"Total: distinct shards.\n\n{t}\n")
    open(os.path.join(HERE, "results.md"), "w").write(md)
    print(t)
    print(f"\nwritten to {os.path.join(HERE, 'results.md')}")


if __name__ == "__main__":
    main()
