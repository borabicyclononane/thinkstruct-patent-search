"""Browse the synthetic query datasets: pick a real independent claim, see every query derived from it.

  .venv/bin/python3.14 -m streamlit run synthetic/viewer.py -- [--dir outputs/synthetic | outputs/synthetic/pilot]
"""
import argparse
import glob
import json
import os
import random
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic import DATASETS, ROOT, SYN, load_claims, load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=None, help="folder with <dataset>.jsonl and rejected.jsonl (default: full run, else pilot)")
args = ap.parse_args()
DIRS = [d for d in [SYN] + sorted(glob.glob(os.path.join(SYN, "*/"))) if os.path.exists(os.path.join(d, "paraphrase.jsonl"))]


@st.cache_data
def load_sources() -> tuple[dict, dict]:
    claims = {c["claim_id"]: c for c in load_claims()}
    return claims, load_split([c["doc_number"] for c in claims.values()])


@st.cache_data
def load_derived(folder: str) -> tuple[dict, dict]:
    """claim_id -> queries / failed calls derived from it (a combination is listed under both claims)."""
    kept, rejected = {}, {}
    for d in DATASETS:
        path = os.path.join(folder, f"{d}.jsonl")
        for row in (json.loads(line) for line in open(path)) if os.path.exists(path) else []:
            for a in row["answer"]:
                kept.setdefault(a["claim_id"], []).append(row)
    path = os.path.join(folder, "failed.jsonl")
    for row in (json.loads(line) for line in open(path)) if os.path.exists(path) else []:
        for cid in row["claim_ids"]:
            rejected.setdefault(cid, []).append(row)
    return kept, rejected


def elements_table(elements: list[dict], used: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"id": e["id"], "used": "✔" if e["id"] in used else "", "core": "core" if e["core"] else "conventional",
                          "element": e["text"]} for e in elements]).set_index("id")


st.set_page_config(page_title="Synthetic Queries", layout="wide")
st.title("Synthetic query datasets")
if not DIRS and not args.dir:
    st.error("No synthetic datasets yet. Run `synthetic/generate_synthetic.py --pilot` (or `--full`) first.")
    st.stop()
folder = args.dir or st.sidebar.radio("Run", DIRS, format_func=lambda d: os.path.relpath(d, ROOT))
claims, split_of = load_sources()
kept, rejected = load_derived(folder)
show_rejected = st.sidebar.checkbox("Show failed calls", value=True)
only_flagged = st.sidebar.checkbox("Only claims with flagged queries")

sources = sorted(set(kept) | (set(rejected) if show_rejected else set()), key=lambda c: (-len(kept.get(c, [])), c))
splits = st.sidebar.multiselect("Split", ["train", "val", "test"], default=["train", "val", "test"])
sources = [c for c in sources if split_of[claims[c]["doc_number"]] in splits
           and (not only_flagged or any(r["meta"].get("flags") for r in kept.get(c, [])))]
st.caption(f"`{os.path.relpath(folder, ROOT)}` · {sum(len(v) for v in kept.values())} derived queries "
           f"(combinations counted under both sources) · {len(sources)} source claims shown")
if not sources:
    st.info("No source claims match.")
    st.stop()


def pick_random():
    st.session_state.claim = random.choice(sources)


if st.session_state.get("claim") not in sources:
    st.session_state.claim = sources[0]
c1, c2 = st.columns([1, 4])
c1.button("🎲 Random claim", on_click=pick_random)
c2.selectbox("Source claim", sources, key="claim",
             format_func=lambda c: f"{c} — {claims[c]['title'][:70]}  ({len(kept.get(c, []))} queries"
                                   + (f", {len(rejected.get(c, []))} failed)" if rejected.get(c) else ")"))
cid = st.session_state.claim
src = claims[cid]

st.header(src["title"])
st.markdown(f"`{cid}` · split **{split_of[src['doc_number']]}** · " + " → ".join(f"`{n}`" for n in src["cpc_path"]))
st.subheader("Source claim")
st.write(src["text"])
with st.expander("Abstract (given to the model as context)"):
    st.write(src["abstract"])

rows = kept.get(cid, [])
tabs = st.tabs([f"{d.replace('_', ' ')} ({sum(r['dataset'] == d for r in rows)})" for d in DATASETS])
for tab, dataset in zip(tabs, DATASETS):
    with tab:
        mine = [r for r in rows if r["dataset"] == dataset]
        if not mine:
            st.caption("No kept query from this claim.")
        for r in mine:
            m = r["meta"]
            st.markdown(f"**{r['query_id']}** · {r['split']} · length {m['length_ratio']}× the source"
                        + (f" · token Jaccard {m['jaccard']} · ROUGE-L {m['rouge_l']}" if dataset == "paraphrase" else "")
                        + (f" · mix: **{m['mix'].replace('_', ' ')}**" if "mix" in m else "")
                        + (f" · stripped: {', '.join(m['stripped'])}" if m.get("stripped") else ""))
            for flag in m.get("flags", []):
                st.warning(f"flag: {flag}")
            st.info(r["query_text"])
            if dataset == "element_reuse":
                st.caption("Elements of the source claim; ✔ = the part(s) the new claim reuses")
                st.dataframe(elements_table(m["elements"], m["chosen_elements"]), width="stretch")
            elif dataset == "combination":
                ids = [a["claim_id"] for a in r["answer"]]
                me, other = ("A", "B") if ids[0] == cid else ("B", "A")
                partner = claims[ids[1] if me == "A" else ids[0]]
                st.markdown(f"This claim is **claim {me}**. Combined with claim {other}: `{partner['claim_id']}` — "
                            f"{partner['title']} (`{partner['cpc']}`)")
                with st.expander(f"Claim {other} text"):
                    st.write(partner["text"])
                for side in ("a", "b"):
                    label = "this claim" if side.upper() == me else "the partner"
                    st.caption(f"Elements of claim {side.upper()} ({label}); ✔ = used in the new claim")
                    st.dataframe(elements_table(m[f"elements_{side}"], m[f"used_from_{side}"]), width="stretch")
            st.divider()

if show_rejected and rejected.get(cid):
    st.subheader(f"Failed calls ({len(rejected[cid])})")
    for x in rejected[cid]:
        resp = x["response"]
        text = resp
        st.warning(f"**{x['dataset']}** {' + '.join(x['claim_ids'])}: {x['reason']}")
        if text:
            with st.expander("Raw response"):
                st.code(text)
