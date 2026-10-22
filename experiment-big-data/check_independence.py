"""Check the text rule that labels independent claims in the large-dataset sample against Google Patents.

  .venv/bin/python3.14 experiment-big-data/check_independence.py --n 10000 --check 200

sample.py labels a claim independent iff its text cites no other claim (scraper/google_patents.py: TEXT_REF,
BROKEN_REF), applied to the dataset's own claim text. Here `--check` random patents of the sample are re-read
from the zip (all claims, cancelled stubs dropped as in sample.py) and scraped from Google Patents, and every
claim number present in both is compared with three references:
  scraper     the scraper's label: <claim-ref> tags OR the same text rule on Google's text (the ground truth
              used for the small dataset)
  claim-ref   <claim-ref> tags only: structural, independent of any text rule
  google      Google's own div.claim-dependent marking (known to misfire, e.g. "I claim a scissor lift")

End to end: the sample's stored independent claims (sample_<n>.jsonl, the text the index embeds, after cleaning
and dropping stubs) are matched to Google's independent claims by text, not by number, so numbering problems
can't hide a miss. Within each patent, claims are paired one-to-one, greedily by full-text similarity (after
dropping the number prefix, HTML entities and all whitespace); a pair counts as a match at similarity >= 0.5,
and the similarity itself shows how much the text differs.
Writes results_<n>/independence_check.json (summary), independence_disagreements.csv (label comparison) and
independence_end_to_end.csv (every Google independent claim and its match in the sample).
"""
import difflib
import html as htmllib
import argparse
import json
import os
import random
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sample import CANCELLED, NUM, ZIP, load_sample  # noqa: E402
from scraper.google_patents import BROKEN_REF, TEXT_REF, fetch_html, parse_claims  # noqa: E402

CACHE_DIR = "data/scraped/html_large"


def raw_labels(claims: list[str]) -> dict[int, tuple[bool, str]]:
    """claim number -> (independent by the text rule, text), exactly as sample.py decides."""
    out = {}
    for i, raw in enumerate(claims):
        m = NUM.match(raw)
        num = int(m.group(1)) if m else i + 1
        if (len(raw) < 60 and CANCELLED.search(raw)) or not re.search(r"[A-Za-z]{2}", raw):
            continue
        out[num] = (not (TEXT_REF.search(raw) or BROKEN_REF.search(raw)), raw)
    return out


def google_labels(html: str) -> dict[int, dict]:
    """claim number -> {scraper, claim_ref, google} independence labels, from the page."""
    scraped = {c["num"]: c["independent"] for c in parse_claims(html)[0]}
    section = BeautifulSoup(html, "html.parser").find("section", itemprop="claims")
    out = {}
    for div in section.select("div.claim[num]"):
        if not div["num"].isdigit() or int(div["num"]) not in scraped:
            continue
        num = int(div["num"])
        out[num] = {"scraper": scraped[num], "claim_ref": not div.find_all("claim-ref", idref=True),
                    "google": div.find_parent("div", class_="claim-dependent") is None}
    return out


def norm(text: str) -> str:
    text = htmllib.unescape(re.sub(r"^\s*(\d+\s*)?[.)]\s*", "", text))  # "12." / "45)" / a stray ")", "&lt;"
    return re.sub(r"\s+", "", text).lower()  # whitespace differs a lot: "( 16 )" vs "(16)", "cr 1 r 2" vs "cr1r2"


def end_to_end(sample: dict, google: dict[str, list[dict]]) -> tuple[pd.DataFrame, int]:
    """-> (one row per Google independent claim with its matching sample claim, sample claims Google doesn't have)."""
    rows, extra = [], 0
    for doc, g in google.items():
        ours = {i: norm(c["text"]) for i, c in enumerate(sample[doc]["claims"])}  # by position: numbers can repeat
        theirs = {c["num"]: c for c in g if c["independent"] and c["text"]}
        sims = sorted(((difflib.SequenceMatcher(None, ours[a], norm(theirs[b]["text"]), autojunk=False).ratio(), a, b)
                       for a in ours for b in theirs), reverse=True)
        pair = {}  # google num -> (sample num, similarity)
        for sim, a, b in sims:
            if sim >= 0.5 and b not in pair and a not in {x for x, _ in pair.values()}:
                pair[b] = (a, sim)
        for b, c in theirs.items():
            a, sim = pair.get(b, (None, None))
            rows.append({"doc_number": doc, "google_num": b,
                         "sample_num": None if a is None else sample[doc]["claims"][a]["num"],
                         "similarity": None if sim is None else round(sim, 4), "google_text": c["text"][:200]})
        extra += len(ours) - len(pair)
    return pd.DataFrame(rows), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000, help="which sample")
    ap.add_argument("--check", type=int, default=200, help="random patents to scrape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    patents, _ = load_sample(args.n)
    chosen = {p["doc_number"] for p in random.Random(args.seed).sample(patents, args.check)}

    raw = {}  # doc -> (publication number, claims), first occurrence in the same batch order as sample.py
    with zipfile.ZipFile(ZIP) as zf:
        names = sorted(x for x in zf.namelist() if re.search(r"patents_batch_\d+\.json$", x))
        random.Random(0).shuffle(names)  # sample.py's order (seed 0)
        for name in names:
            for p in json.load(zf.open(name)):
                if p["doc_number"] in chosen and p["doc_number"] not in raw:
                    raw[p["doc_number"]] = (p["filename"].split("-")[0].removesuffix(".XML"), p["claims"])
            if len(raw) == len(chosen):
                break
    print(f"{len(raw)} patents re-read from the zip", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)

    def scrape(doc):
        pub = raw[doc][0]
        try:
            page = fetch_html(pub, CACHE_DIR)
            return doc, pub, google_labels(page), parse_claims(page)[0], None
        except Exception as e:
            return doc, pub, None, None, repr(e)

    rows, errors, google = [], {}, {}
    with ThreadPoolExecutor(args.workers) as pool:
        for doc, pub, labels, claims, err in pool.map(scrape, sorted(raw)):
            if err:
                errors[doc] = err
                continue
            google[doc] = claims
            ours = raw_labels(raw[doc][1])
            for num in sorted(set(ours) | set(labels)):
                rows.append({"doc_number": doc, "publication_number": pub, "num": num,
                             "text_rule": ours[num][0] if num in ours else None,
                             **(labels.get(num) or {"scraper": None, "claim_ref": None, "google": None}),
                             "text": ours[num][1][:300] if num in ours else ""})
    df = pd.DataFrame(rows)
    both = df.dropna(subset=["text_rule", "scraper"])
    summary = {"sample": args.n, "patents_checked": len(raw) - len(errors), "scrape_errors": errors,
               "claims_compared": len(both),
               "claims_only_in_dataset": int(df["scraper"].isna().sum()),
               "claims_only_on_google": int(df["text_rule"].isna().sum()),
               "independent_by_text_rule": int(both["text_rule"].sum())}
    for ref in ("scraper", "claim_ref", "google"):
        t, r = both["text_rule"].astype(bool), both[ref].astype(bool)
        summary[f"vs_{ref}"] = {
            "agreement": round(float((t == r).mean()), 5),
            "text_rule_independent_but_ref_dependent": int((t & ~r).sum()),  # would wrongly enter the index
            "text_rule_dependent_but_ref_independent": int((~t & r).sum()),  # would be wrongly left out
            "patents_with_identical_independent_sets": int(
                both.assign(t=t, r=r).groupby("doc_number").apply(lambda g: (g["t"] == g["r"]).all()).sum())}
    e2e, extra = end_to_end({p["doc_number"]: p for p in patents if p["doc_number"] in google}, google)
    found = e2e["sample_num"].notna()
    sim = e2e.loc[found, "similarity"]
    summary["end_to_end"] = {
        "google_independent_claims": len(e2e), "found_in_sample": int(found.sum()),
        "recall": round(float(found.mean()), 5),
        "sample_claims": int(sum(len(p["claims"]) for p in patents if p["doc_number"] in google)),
        "sample_claims_not_independent_on_google": extra,
        "text_similarity": {"min": float(sim.min()), "p1": round(float(sim.quantile(0.01)), 4),
                            "median": float(sim.median()), "share_identical": round(float((sim == 1).mean()), 4),
                            "share_below_0.95": round(float((sim < 0.95).mean()), 4)}}
    out_dir = os.path.join(os.path.dirname(__file__), f"results_{args.n}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, "independence_check.json"), "w"), indent=1)
    bad = df[(df["text_rule"] != df["scraper"]) | (df["text_rule"] != df["google"]) | df["text_rule"].isna() | df["scraper"].isna()]
    bad.to_csv(os.path.join(out_dir, "independence_disagreements.csv"), index=False)
    e2e.to_csv(os.path.join(out_dir, "independence_end_to_end.csv"), index=False)
    print(json.dumps(summary, indent=1))
    print(f"{len(bad)} rows that differ in any way -> {out_dir}/independence_disagreements.csv")


if __name__ == "__main__":
    main()
