"""Patent viewer: browse random patents and plot any features attached to them.

Run:  python -m viewer   (see `python -m viewer --help`)

Features: drop any .csv or .parquet into features/ with a `doc_number` column.
Every other column is joined onto the patents and becomes available for
display, grouping and plotting. Non-scalar columns (e.g. embedding vectors)
are ignored by the plots.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from search.data import CPC_LEVELS, load_corpus  # noqa: E402
from search.embed import MODEL, Embedder  # noqa: E402
from search.filters import FIELDS, TOKEN, Filter, TextIndex, fold, hybrid_search, query_terms  # noqa: E402
from search.index import build_index, l2n  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/patent_data_small")
ap.add_argument("--features-dir", default="features")
ap.add_argument("--sample", help="a large-dataset sample (experiment-big-data/sample.py) instead of --data-dir")
args = ap.parse_args()
DATA_DIR = args.data_dir
FEATURES_DIR = args.features_dir
SAMPLE = args.sample
SAMPLE_CACHE = "data/index/embeddings_large.npz"  # filled by experiment-big-data/run.py (or gpu_embed.py merge)
DATE = "year" if SAMPLE else "week"  # the large dataset has no filing week; year comes from the doc number


# Codes where the digit heuristic below picks the wrong split (checked against the USPTO CPC scheme pages).
CPC_MAIN_GROUP_OVERRIDES = {"B60D1167": "1"}  # B60D 1/167, not 11/67


def split_cpc(code: str) -> tuple[str, str]:
    """'B60C1103FI' -> ('B60C 11', 'B60C 11/03').

    The separator between main group and subgroup is missing from the data, so
    this is a heuristic: 3 digits -> 1/23, 4 -> 12/34, 5 -> 12/345. On the small
    dataset, every code but the overrides above maps to a symbol in the official scheme.
    """
    subclass, digits = re.match(r"([A-H]\d\d[A-Z])(\d+)", code).groups()
    main = CPC_MAIN_GROUP_OVERRIDES.get(subclass + digits) or (digits[:1] if len(digits) == 3 else digits[:2])
    return f"{subclass} {main}", f"{subclass} {main}/{digits[len(main):]}"


@st.cache_data
def load_patents(data_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "patents_ipa*.json"))):
        week = pd.to_datetime(re.search(r"ipa(\d{6})", path).group(1), format="%y%m%d")
        for p in json.load(open(path)):
            rows.append({**p, "week": week})
    df = pd.DataFrame(rows).drop_duplicates("doc_number").set_index("doc_number")
    df["subclass"] = df["classification"].str[:4]
    df["main_group"], df["subgroup"] = zip(*df["classification"].map(split_cpc))
    df["n_claims"] = df["claims"].str.len()
    df["n_paragraphs"] = df["detailed_description"].str.len()
    df["abstract_words"] = df["abstract"].str.split().str.len()
    return df


def read_sample(path: str) -> list[dict]:
    """Patents as in search/data.py:load_corpus (independent claims only), from experiment-big-data/sample.py."""
    patents = [json.loads(line) for line in open(path)]
    for p in patents:
        p["path"] = tuple(p["path"])
    return patents


@st.cache_data
def load_sample_patents(path: str) -> pd.DataFrame:
    """Same columns as load_patents where the sample has them: no description, BibTeX or filing week."""
    patents = read_sample(path)
    df = pd.DataFrame(patents).set_index("doc_number").rename(columns={"cpc": "classification"})
    levels = pd.DataFrame(df.pop("path").tolist(), index=df.index, columns=CPC_LEVELS)
    df[["subclass", "main_group", "subgroup"]] = levels[["subclass", "main_group", "subgroup"]]
    df["claim_nums"] = df["claims"].map(lambda cs: [c["num"] for c in cs])
    df["claims"] = df["claims"].map(lambda cs: [c["text"] for c in cs])
    df["year"] = df.index.str[:4].astype(int)
    df["n_claims"] = df["claims"].str.len()
    df["abstract_words"] = df["abstract"].str.split().str.len()
    return df


@st.cache_resource
def load_search(data_dir: str, sample: str | None):
    """Search index over independent claims (scraped, or the sample's) + inverted index; embeddings come from the cache."""
    if sample:
        index, _ = build_index(read_sample(sample), Embedder(SAMPLE_CACHE))
    else:
        corpus, _ = load_corpus(data_dir)
        index, _ = build_index(corpus, Embedder())
    return index, TextIndex(index)


@st.cache_resource
def load_query_model():
    """On CPU on purpose: Streamlit runs every rerun in a new thread, and torch's MPS (Apple GPU) backend aborts the
    whole process (Metal assertion) or hangs when two threads use it at once, e.g. typing while a query embeds.
    One query takes ~35 ms on CPU. The lock makes overlapping reruns take turns."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device="cpu"), threading.Lock()


def embed_query(text: str) -> np.ndarray:
    model, lock = load_query_model()
    with lock:
        return model.encode([text], normalize_embeddings=True)[0]


def highlight(text: str, terms: set[str]) -> str:
    """Bold every word (or part of a hyphenated word) that matches a keyword."""
    def bold(m):
        w = m.group().lower()
        return f"**{m.group()}**" if fold(w) in terms or any(fold(x) in terms for x in w.split("-")) else m.group()
    return re.sub(TOKEN.pattern, bold, text, flags=re.I)


def load_features(features_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(features_dir, "*"))):
        if path.endswith(".csv"):
            f = pd.read_csv(path, dtype={"doc_number": str})
        elif path.endswith(".parquet"):
            f = pd.read_parquet(path)
            f["doc_number"] = f["doc_number"].astype(str)
        else:
            continue
        frames.append(f.set_index("doc_number"))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


st.set_page_config(page_title="Patent Viewer", layout="wide")
patents = load_sample_patents(SAMPLE) if SAMPLE else load_patents(DATA_DIR)
features = load_features(FEATURES_DIR)
df = patents.join(features, how="left")

TEXT_COLS = ["title", "abstract", "claims", "claim_nums", "detailed_description", "bibtex", "filename"]
table_cols = [c for c in df.columns if c not in TEXT_COLS]
scalar_cols = [c for c in table_cols if df[c].map(lambda v: not isinstance(v, (list, dict))).all()]
numeric_cols = [c for c in scalar_cols if pd.api.types.is_numeric_dtype(df[c])]

st.title("Patent Viewer")
st.caption(f"{len(df)} patents{f' from `{SAMPLE}`' if SAMPLE else ''} · {len(features.columns)} feature columns loaded from `{FEATURES_DIR}/`")

browse, search, tree, explore = st.tabs(["Browse", "Search", "Classification tree", "Explore features"])


def set_doc(d):
    st.session_state.doc = d
    st.session_state.pick = None


def open_selected_row(key, rows):
    sel = st.session_state[key].selection.rows
    if sel:
        set_doc(rows.index[sel[0]])


with browse:
    if "doc" not in st.session_state:
        st.session_state.doc = df.sample(1).index[0]
    c1, c2 = st.columns([1, 3])
    c1.button("🎲 Random patent", on_click=lambda: set_doc(df.sample(1).index[0]))
    c2.selectbox(
        "Or pick one", df.index, index=None, key="pick", placeholder="Search by doc number or title",
        format_func=lambda d: f"{d} — {df.at[d, 'title']}",
        on_change=lambda: st.session_state.pick and set_doc(st.session_state.pick),
    )
    doc = st.session_state.doc
    p = df.loc[doc]

    st.header(p["title"])
    path = [p["classification"][:1], p["classification"][:3], p["subclass"], p["main_group"], p["subgroup"]]
    st.markdown(" → ".join(f"`{n}`" for n in path))
    st.dataframe(p[table_cols].astype(str).to_frame("value"), width="stretch")

    st.subheader("Nearby in the classification tree")
    level = st.radio(
        "Other patents in the same", ["subgroup", "main_group", "subclass"], horizontal=True,
        format_func=lambda lvl: f"{lvl.replace('_', ' ')} ({p[lvl]})",
    )
    rel = df[(df[level] == p[level]) & (df.index != doc)].sort_values(["subgroup", DATE])
    if rel.empty:
        st.info(f"No other patents in {p[level]}. Try a broader level.")
    else:
        st.caption(f"{len(rel)} other patents · click a row to open it")
        key = f"rel_{doc}_{level}"  # fresh widget per patent/level so old selections don't carry over
        st.dataframe(
            rel[["title", "subgroup", DATE]], key=key, width="stretch",
            on_select=lambda: open_selected_row(key, rel), selection_mode="single-row",
        )

    st.subheader("Abstract")
    st.write(p["abstract"])
    nums = p["claim_nums"] if SAMPLE else range(1, len(p["claims"]) + 1)
    with st.expander(f"{'Independent claims' if SAMPLE else 'Claims'} ({len(p['claims'])})"):
        for i, c in zip(nums, p["claims"]):
            st.markdown(f"**{i}.** {c}")
    if not SAMPLE:  # the sample keeps neither
        with st.expander(f"Detailed description ({len(p['detailed_description'])} paragraphs)"):
            for i, para in enumerate(p["detailed_description"], 1):
                st.markdown(f"`[{i:04d}]` {para}")
        with st.expander("BibTeX"):
            st.code(p["bibtex"], language="bibtex")

with search:
    index, text = load_search(DATA_DIR, SAMPLE)
    st.caption("Keywords match whole words in the title and/or abstract, ignoring case and plurals (tires = tire). "
               "A hyphenated keyword matches only the whole term (non-pneumatic); its parts match it too.")
    k1, k2, k3 = st.columns(3)
    all_of = k1.text_input("All of these keywords", placeholder="tread noise")
    any_of = k2.text_input("Any of these keywords", placeholder="stud hitch")
    fields = k3.multiselect("Search keywords in", FIELDS, default=list(FIELDS))
    c1, c2 = st.columns(2)
    cpc = c1.text_input("CPC prefixes (comma-separated, any of)", placeholder="B60B, B60C 11")
    title = c2.text_input("Exact title (case and punctuation ignored)")
    query = st.text_area("Rank by similarity to (optional): a claim or a description of the invention")
    r1, r2, r3 = st.columns(3)
    method = r1.radio("Ranking", ["beam", "flat"], horizontal=True, help="beam: walk the CPC tree; flat: score every claim")
    B = r2.number_input("Beam width B", 1, 500, 10)
    k = r3.number_input("Results", 1, 500, 20)

    words = lambda s: tuple(w for w in re.split(r"[\s,]+", s.strip()) if w)
    f = Filter(cpc=tuple(c.strip() for c in cpc.split(",") if c.strip()), all_of=words(all_of), any_of=words(any_of),
               fields=tuple(fields), title=title.strip() or None)
    if (f.all_of or f.any_of) and not fields:
        st.warning("Pick at least one field to search keywords in.")
    elif not f and not query.strip():
        st.info("Enter keywords, a CPC prefix, a title or a query.")
    else:
        ctree = index.tree
        t0 = time.perf_counter()
        starts, cand = text.resolve(f)
        if cand is None:
            cand = text.cpc_positions(starts) if f.cpc else None
        t_filter = time.perf_counter() - t0
        n = len(ctree.order) if cand is None else len(cand)
        qv = None
        if query.strip():
            with st.spinner("Embedding the query (the model loads on first use, ~10 s)..."):
                t0 = time.perf_counter()
                try:
                    qv = embed_query(query.strip())
                except Exception as e:  # show it, and fall back to the unranked list below
                    st.error(f"Couldn't embed the query, so the matches below are not ranked: {e!r}")
                t_embed = time.perf_counter() - t0
        if qv is not None:
            t0 = time.perf_counter()
            res, _ = hybrid_search(index, text, qv, f, k, B, flat=method == "flat")
            t_search = time.perf_counter() - t0
            pats, scores = res.patents[:k], res.patent_scores[:k]
            qn = l2n(index.project(qv[None]))[0]
            best = [ctree.children[p][np.argmax(index.score(qn, index.K[ctree.children[p]]))] for p in pats]
            hits = pd.DataFrame({"score": scores.round(3), "best claim": [ctree.claim[c]["num"] for c in best]},
                                index=pd.Index([ctree.patent[p]["doc_number"] for p in pats], name="doc_number"))
            timing = (f"filter {1000 * t_filter:.2f} ms · {method} search {1000 * t_search:.2f} ms · "
                      f"query embedding {1000 * t_embed:.0f} ms")
        else:
            positions = range(len(ctree.order)) if cand is None else cand
            hits = pd.DataFrame(index=pd.Index([ctree.patent[ctree.order[i]]["doc_number"] for i in positions],
                                               name="doc_number"))
            timing = f"filter {1000 * t_filter:.2f} ms"
        hits = hits.join(df[["title", "subgroup", DATE]])
        if qv is not None and method == "beam" and k > B and n > B:
            st.caption(f"Beam search returns at most B = {B} patents; raise B to see up to {k}.")
        st.caption(f"{n} of {len(ctree.order)} patents match the filters · showing {len(hits)} · {timing} · "
                   "click a row to preview it")
        key = f"hits_{hash((repr(f), query, method, B, k))}"  # fresh widget per search so old selections don't carry over
        st.dataframe(hits, key=key, width="stretch", on_select="rerun", selection_mode="single-row")
        sel = st.session_state[key].selection.rows
        if sel:
            d = hits.index[sel[0]]
            terms = {t for kw in f.all_of + f.any_of for t in query_terms(kw)}
            st.button(f"Open {d} in the Browse tab", on_click=set_doc, args=(d,))
            st.markdown(f"#### {highlight(df.at[d, 'title'], terms)}")
            st.markdown(highlight(df.at[d, "abstract"], terms))
            if "best claim" in hits:
                num = hits.at[d, "best claim"]
                claim = next(c for c in ctree.patent[ctree.patent_node[d]]["claims"] if c["num"] == num)
                st.markdown(f"**Best-matching claim ({num}):** {claim['text']}")

with tree:
    st.caption(
        "CPC hierarchy: subclass → main group → subgroup, sized by number of patents. Click a box to zoom in. "
        "Subgroups are shown flat under their main group (the true dot-level nesting isn't in the data)."
    )
    counts = df.groupby(["subclass", "main_group", "subgroup"]).size().reset_index(name="patents")
    fig = px.treemap(counts, path=[px.Constant("all classes" if SAMPLE else "B60"), "subclass", "main_group", "subgroup"], values="patents")
    fig.update_traces(root_color="lightgrey", textinfo="label+value")
    fig.update_layout(height=750, margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig, width="stretch")

with explore:
    st.subheader("Table")
    st.dataframe(df[["title"] + scalar_cols], width="stretch")

    st.subheader("Group by")
    g1, g2 = st.columns(2)
    by = g1.selectbox("Group by", scalar_cols, index=scalar_cols.index("subclass"))
    aggs = g2.multiselect("Aggregate (mean)", numeric_cols)
    grouped = df.groupby(by).agg(count=("title", "size"), **{f"mean_{c}": (c, "mean") for c in aggs})
    st.dataframe(grouped.sort_values("count", ascending=False), width="stretch")

    st.subheader("Plot")
    kind = st.radio("Chart", ["Histogram", "Scatter", "Bar (count)", "Box"], horizontal=True)
    none = [None]
    p1, p2, p3 = st.columns(3)
    if kind == "Histogram":
        x = p1.selectbox("x", scalar_cols, key="hx")
        color = p2.selectbox("color", none + scalar_cols, key="hc")
        fig = px.histogram(df, x=x, color=color)
    elif kind == "Scatter":
        x = p1.selectbox("x", numeric_cols, key="sx")
        y = p2.selectbox("y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sy")
        color = p3.selectbox("color", none + scalar_cols, key="sc")
        fig = px.scatter(df.reset_index(), x=x, y=y, color=color, hover_data=["doc_number", "title"])
    elif kind == "Bar (count)":
        x = p1.selectbox("x", scalar_cols, index=scalar_cols.index("subclass"), key="bx")
        color = p2.selectbox("color", none + scalar_cols, key="bc")
        fig = px.histogram(df, x=x, color=color).update_xaxes(categoryorder="total descending")
    else:
        x = p1.selectbox("group (x)", scalar_cols, index=scalar_cols.index("subclass"), key="ox")
        y = p2.selectbox("value (y)", numeric_cols, key="oy")
        fig = px.box(df.reset_index(), x=x, y=y, points="all", hover_data=["doc_number", "title"])
    st.plotly_chart(fig, width="stretch")
