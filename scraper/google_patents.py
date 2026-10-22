"""Fetch a patent's claims from Google Patents and parse them with exact dependency info.

Google Patents serves the USPTO full-text XML as HTML: each claim is a
`div.claim` (independent) or `div.claim-dependent`, and dependencies are
explicit `<claim-ref idref="CLM-00001">` tags. Unlike data/, the claim text
keeps its preamble and the "claim N" reference text.
"""
import os
import re
import time

import requests
from bs4 import BeautifulSoup

URL = "https://patents.google.com/patent/{pub}/en"
HEADERS = {"User-Agent": "Mozilla/5.0 (patent claims research)"}
CANCELLED = re.compile(r"^\(\s*cancell?ed\s*\)\.?$", re.I)
# Not every source XML tags its references, so also read them from the text: "claim 3", "claims 1-4", "claims 1 or 2".
TEXT_REF = re.compile(r"\bclaims?\s+(\d+)(?:\s*(?:-|–|to|through)\s*(\d+))?((?:\s*(?:,|or|and)\s*\d+)*)", re.I)
BROKEN_REF = re.compile(
    r"\bclaim\s+Error!"  # Word cross-reference that never resolved
    r"|\b(?:any|one)\s+of\s+(?:the\s+)?(?:preceding|previous|above)\s+claims\b"
    r"|\b(?:of|to|in)\s+claims?\b(?!\s*\d)",  # "according to claim ," with the number missing
    re.I,
)


def fetch_html(pub: str, cache_dir: str, retries: int = 4) -> str:
    """Return the page for publication number `pub` (e.g. US20240051333A1), cached on disk."""
    path = os.path.join(cache_dir, f"{pub}.html")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    for attempt in range(retries):
        r = requests.get(URL.format(pub=pub), headers=HEADERS, timeout=30)
        if r.status_code == 200:
            r.encoding = "utf-8"  # the pages are UTF-8, but without a charset header requests decodes them as Latin-1
            open(path, "w", encoding="utf-8").write(r.text)
            return r.text
        if r.status_code == 404:
            break
        time.sleep(2 ** attempt * 5)  # 429 / 5xx: back off and retry
    r.raise_for_status()


def parse_claims(html: str) -> tuple[list[dict], int]:
    """-> (claims, n_cancelled). Each claim: num, text, depends_on, independent.

    `independent` follows the USPTO definition: the claim cites no other claim.
    A claim like "A vehicle wheel comprising the accessory of claim 1" therefore
    counts as dependent, even though it starts a new claim category.

    A claim is dependent if it has a <claim-ref> tag or its text cites a claim
    (Google's claim-dependent class is not used: it misfires on e.g. "I claim a
    scissor lift"). `depends_on` lists the cited earlier claims; it can be empty
    for a dependent claim whose reference is broken in the source (e.g. "claim
    Error! Bookmark not defined", or a claim citing itself).
    """
    section = BeautifulSoup(html, "html.parser").find("section", itemprop="claims")
    if section is None:
        raise ValueError("no claims section on page")
    claims, n_cancelled = [], 0
    for div in section.select("div.claim[num]"):
        num = div["num"]
        text = re.sub(r"\s+", " ", div.get_text(" ")).strip()
        text = re.sub(r"^\d+\s*[.:]?\s*(?:[-–]\s*\d+\s*[.:]?\s*)?", "", text)  # leading "12 ." / "1 .- 15 ." / "1 - 15 :"
        text = re.sub(r"\s+([,.;:])", r"\1", text)  # get_text(" ") pads punctuation after tags
        if CANCELLED.match(text):
            a, _, b = num.partition("-")
            n_cancelled += int(b or a) - int(a) + 1
            continue
        num = int(num)
        refs = {int(re.search(r"\d+", ref["idref"]).group()) for ref in div.find_all("claim-ref", idref=True)}
        for m in TEXT_REF.finditer(text):
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            refs |= set(range(a, b + 1)) | {int(n) for n in re.findall(r"\d+", m.group(3))}
        dependent = bool(refs) or bool(BROKEN_REF.search(text))
        claims.append({"num": num, "text": text, "depends_on": sorted(r for r in refs if r < num), "independent": not dependent})
    return claims, n_cancelled
