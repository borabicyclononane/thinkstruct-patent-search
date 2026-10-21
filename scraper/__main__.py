"""Scrape full claims (with exact independent/dependent labels) for every patent in data/.

Run:  python -m scraper   (see `python -m scraper --help`)

Writes one JSON line per patent to --out:
  {"doc_number", "publication_number", "n_cancelled", "claims": [{"num", "text", "depends_on", "independent"}]}
Patents that fail get {"doc_number", "publication_number", "error"} instead; rerun to retry them
(pages already fetched are read from the HTML cache).
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .google_patents import fetch_html, parse_claims

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/patent_data_small")
ap.add_argument("--out", default="data/scraped/claims.jsonl")
ap.add_argument("--cache-dir", default="data/scraped/html")
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
os.makedirs(args.cache_dir, exist_ok=True)

# filename looks like US20240051333A1-20240215.XML -> publication number US20240051333A1
pubs = {}
for path in sorted(glob.glob(os.path.join(args.data_dir, "patents_ipa*.json"))):
    for p in json.load(open(path)):
        pubs[p["doc_number"]] = p["filename"].split("-")[0]


def scrape(item):
    doc, pub = item
    try:
        claims, n_cancelled = parse_claims(fetch_html(pub, args.cache_dir))
        return {"doc_number": doc, "publication_number": pub, "n_cancelled": n_cancelled, "claims": claims}
    except Exception as e:
        return {"doc_number": doc, "publication_number": pub, "error": repr(e)}


n_err = 0
with ThreadPoolExecutor(args.workers) as pool, open(args.out, "w") as f:
    for i, rec in enumerate(pool.map(scrape, pubs.items()), 1):
        f.write(json.dumps(rec) + "\n")
        n_err += "error" in rec
        if i % 50 == 0 or i == len(pubs):
            print(f"{i}/{len(pubs)} done, {n_err} errors")
