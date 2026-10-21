"""Subsample the large dataset straight from the zip into the corpus format of search/data.py:load_corpus.

  .venv/bin/python3.14 experiment-big-data/sample.py --n 100000

Batch files (1,000 patents each, all CPC sections, years mixed) are read in a seeded random order until
n patents are kept; a random set of batches stands in for a random sample of patents. Each batch is
parsed from the zip in memory, nothing is extracted.

Claims here are clean (numbered, references intact), so instead of scraping, independence uses the
scraper's text rule (scraper/google_patents.py): a claim is dependent iff it cites another claim. Only
independent claims are kept (that is all the index uses); claims[0] is the first one, the eval query.
Claims dropped: cancelled stubs, and empty claims (a number only).
Patents dropped: classification "N/A" or unparseable, no claims, no live independent claim, duplicate doc_number.
"""
import argparse
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scraper.google_patents import BROKEN_REF, TEXT_REF  # noqa: E402
from search.data import clean, cpc_path  # noqa: E402

ZIP = "coding_challenge_large_data.zip"
OUT_DIR = "data/large"
CPC = re.compile(r"^[A-H]\d\d[A-Z]\s*\d+/\d+$")
NUM = re.compile(r"^\s*(\d+)\s*\.")
CANCELLED = re.compile(r"cancel|delet", re.I)  # only checked on short claims: "1-15. (canceled)", "10 and 11. (canceled)"


def independent_claims(claims: list[str], report: Counter) -> list[dict]:
    out = []
    for i, raw in enumerate(claims):
        m = NUM.match(raw)
        num = int(m.group(1)) if m else i + 1
        if len(raw) < 60 and CANCELLED.search(raw):
            report["claims cancelled"] += 1
            continue
        if not re.search(r"[A-Za-z]{2}", raw):  # number only, e.g. "4." or "12-15."
            report["claims empty"] += 1
            continue
        if TEXT_REF.search(raw) or BROKEN_REF.search(raw):
            report["claims dependent"] += 1
            continue
        report["claims independent"] += 1
        out.append({"num": num, "text": clean(raw), "independent": True, "depends_on": []})
    return out


def sample(n: int, seed: int = 0, zip_path: str = ZIP) -> tuple[list[dict], Counter]:
    report, patents, seen = Counter(), [], set()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(x for x in zf.namelist() if re.search(r"patents_batch_\d+\.json$", x))
        random.Random(seed).shuffle(names)
        for name in names:
            report["batch files read"] += 1
            for p in json.load(zf.open(name)):
                report["patents read"] += 1
                code = (p.get("classification") or "").strip()
                if not CPC.match(code):
                    report["dropped: classification N/A" if code == "N/A" else "dropped: classification unparseable"] += 1
                    continue
                if p["doc_number"] in seen:
                    report["dropped: duplicate doc_number"] += 1
                    continue
                if not p.get("claims"):
                    report["dropped: no claims"] += 1
                    continue
                claims = independent_claims(p["claims"], report)
                if not claims:
                    report["dropped: no live independent claim"] += 1
                    continue
                seen.add(p["doc_number"])
                patents.append({"doc_number": p["doc_number"], "title": p.get("title") or "",
                                "abstract": p.get("abstract") or "", "cpc": code, "path": cpc_path(code),
                                "claims": claims})
                if len(patents) == n:
                    return patents, report
            print(f"{report['batch files read']} batch files, {len(patents)} patents kept", flush=True)
    return patents, report


def path_for(n: int) -> str:
    return os.path.join(OUT_DIR, f"sample_{n}.jsonl")


def load_sample(n: int) -> tuple[list[dict], dict]:
    patents = []
    for line in open(path_for(n)):
        p = json.loads(line)
        p["path"] = tuple(p["path"])
        patents.append(p)
    return patents, json.load(open(path_for(n).replace(".jsonl", "_report.json")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    patents, report = sample(args.n, args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path_for(args.n), "w") as f:
        for p in patents:
            f.write(json.dumps(p) + "\n")
    json.dump(dict(report), open(path_for(args.n).replace(".jsonl", "_report.json"), "w"), indent=1)
    print(json.dumps(dict(report), indent=1))
    print(f"{len(patents)} patents -> {path_for(args.n)}")
