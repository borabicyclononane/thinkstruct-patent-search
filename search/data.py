"""Load patents + scraped claims, clean claim text, parse CPC paths, and report on the data.

Missing-field policy: a patent needs claims (from the scraper output) and a
classification code to be indexed; anything else may be missing. Dropped
patents are counted in the report. If a patent has several CPC codes, only the
first (the FI / main code) is used.
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np

DATA_DIR = "data/patent_data_small"
SCRAPED = "data/scraped/claims.jsonl"
FIELDS = ["title", "doc_number", "abstract", "detailed_description", "claims", "bibtex", "classification"]
CPC_LEVELS = ["section", "class", "subclass", "main_group", "subgroup"]

# Same heuristic as viewer/app.py:split_cpc (the main group / subgroup separator is missing from the data).
CPC_MAIN_GROUP_OVERRIDES = {"B60D1167": "1"}  # B60D 1/167, not 11/67

CLEANERS = [  # (report name, pattern, replacement), applied in order
    ("duplicate numbering", re.compile(r"^\s*(\d+)\s*\.\s*(?=\1\s*\.)"), ""),  # "4. 4 . The" -> "4 . The"
    ("leading claim number", re.compile(r"^\s*\d+\s*\.\s+(?=[A-Z])"), ""),  # "4 . The" -> "The"
    ("glued FIG.", re.compile(r"([a-z])(FIGS?\.)"), r"\1 \2"),  # "inFIG.5" -> "in FIG.5"
    ("glued FIG. number", re.compile(r"(FIGS?\.)(\d)"), r"\1 \2"),  # "FIG.5" -> "FIG. 5"
    ("glued reference numeral", re.compile(r"\b([a-z]{3,})(\d+[a-z]?)\b"), r"\1 \2"),  # "unit100" -> "unit 100"
]


def clean(text: str, stats: Counter | None = None) -> str:
    for name, pattern, repl in CLEANERS:
        text, n = pattern.subn(repl, text)
        if stats is not None and n:
            stats[name] += n
    return re.sub(r"\s+", " ", text).strip()


def cpc_path(code: str) -> tuple[str, ...]:
    """'B60B1900FI' -> ('B', 'B60', 'B60B', 'B60B 19', 'B60B 19/00')."""
    subclass, digits = re.match(r"([A-H]\d\d[A-Z])(\d+)", code).groups()
    main = CPC_MAIN_GROUP_OVERRIDES.get(subclass + digits) or (digits[:1] if len(digits) == 3 else digits[:2])
    return code[0], code[:3], subclass, f"{subclass} {main}", f"{subclass} {main}/{digits[len(main):]}"


def cpc_codes(value) -> list[str]:
    if isinstance(value, list):
        return [c for c in value if c]
    return re.split(r"[,;\s]+", value.strip()) if value else []


def load_corpus(data_dir: str = DATA_DIR, scraped_path: str = SCRAPED) -> tuple[list[dict], dict]:
    """-> (patents, report). Each patent: doc_number, title, abstract, cpc, path, claims.

    `claims` are the scraped claims ({num, text, independent, depends_on}) with cleaned text; the
    `independent` labels come from the scraper and are used as ground truth.
    """
    raw = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "patents_ipa*.json"))):
        for p in json.load(open(path)):
            raw.setdefault(p["doc_number"], p)
    scraped = {}
    if os.path.exists(scraped_path):
        for line in open(scraped_path):
            r = json.loads(line)
            if "error" not in r:
                scraped[r["doc_number"]] = r

    report = {"n_raw": len(raw), "missing": Counter(), "dropped": Counter(), "multi_cpc": 0, "cleaned": Counter()}
    patents = []
    for doc, p in raw.items():
        for f in FIELDS:
            if not p.get(f):
                report["missing"][f] += 1
        codes = cpc_codes(p.get("classification"))
        report["multi_cpc"] += len(codes) > 1
        claims = scraped.get(doc, {}).get("claims")
        if not codes:
            report["dropped"]["no classification"] += 1
            continue
        if not claims:
            report["dropped"]["no claims (missing, or not in scraper output)"] += 1
            continue
        claims = [{**c, "text": clean(c["text"], report["cleaned"])} for c in claims]
        patents.append({
            "doc_number": doc, "title": p.get("title", ""), "abstract": p.get("abstract", ""),
            "cpc": codes[0], "path": cpc_path(codes[0]), "claims": claims,
        })
    return patents, report


def print_report(patents: list[dict], report: dict) -> None:
    print(f"patents: {report['n_raw']} loaded, {len(patents)} indexed")
    for reason, n in report["dropped"].items():
        print(f"  dropped {n}: {reason}")
    print("missing/empty fields:", dict(report["missing"]) or "none")
    print(f"patents with more than one CPC code: {report['multi_cpc']}")
    print("claim cleaning (scraped claims):", dict(report["cleaned"]) or "nothing to fix")
    n_claims = np.array([len(p["claims"]) for p in patents])
    n_indep = np.array([sum(c["independent"] for c in p["claims"]) for p in patents])
    print(f"claims per patent: mean {n_claims.mean():.1f}, median {np.median(n_claims):.0f}, "
          f"min {n_claims.min()}, max {n_claims.max()} ({n_claims.sum()} total)")
    print(f"independent claims per patent: mean {n_indep.mean():.2f}, distribution "
          f"{dict(sorted(Counter(n_indep.tolist()).items()))} ({n_indep.sum()} total)")
