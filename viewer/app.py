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

import pandas as pd
import plotly.express as px
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/patent_data_small")
ap.add_argument("--features-dir", default="features")
args = ap.parse_args()
DATA_DIR = args.data_dir
FEATURES_DIR = args.features_dir


@st.cache_data
def load_patents(data_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "patents_ipa*.json"))):
        week = pd.to_datetime(re.search(r"ipa(\d{6})", path).group(1), format="%y%m%d")
        for p in json.load(open(path)):
            rows.append({**p, "week": week})
    df = pd.DataFrame(rows).drop_duplicates("doc_number").set_index("doc_number")
    df["subclass"] = df["classification"].str[:4]
    df["n_claims"] = df["claims"].str.len()
    df["n_paragraphs"] = df["detailed_description"].str.len()
    df["abstract_words"] = df["abstract"].str.split().str.len()
    return df


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
patents = load_patents(DATA_DIR)
features = load_features(FEATURES_DIR)
df = patents.join(features, how="left")

TEXT_COLS = ["title", "abstract", "claims", "detailed_description", "bibtex", "filename"]
table_cols = [c for c in df.columns if c not in TEXT_COLS]
scalar_cols = [c for c in table_cols if df[c].map(lambda v: not isinstance(v, (list, dict))).all()]
numeric_cols = [c for c in scalar_cols if pd.api.types.is_numeric_dtype(df[c])]

st.title("Patent Viewer")
st.caption(f"{len(df)} patents · {len(features.columns)} feature columns loaded from `{FEATURES_DIR}/`")

browse, explore = st.tabs(["Browse", "Explore features"])

with browse:
    if "doc" not in st.session_state:
        st.session_state.doc = df.sample(1).index[0]
    c1, c2 = st.columns([1, 3])
    if c1.button("🎲 Random patent"):
        st.session_state.doc = df.sample(1).index[0]
    picked = c2.selectbox(
        "Or pick one", df.index, index=None, placeholder="Search by doc number",
        format_func=lambda d: f"{d} — {df.at[d, 'title']}",
    )
    doc = picked or st.session_state.doc
    p = df.loc[doc]

    st.header(p["title"])
    st.dataframe(p[table_cols].astype(str).to_frame("value"), width="stretch")
    st.subheader("Abstract")
    st.write(p["abstract"])
    with st.expander(f"Claims ({len(p['claims'])})"):
        for i, c in enumerate(p["claims"], 1):
            st.markdown(f"**{i}.** {c}")
    with st.expander(f"Detailed description ({len(p['detailed_description'])} paragraphs)"):
        for i, para in enumerate(p["detailed_description"], 1):
            st.markdown(f"`[{i:04d}]` {para}")
    with st.expander("BibTeX"):
        st.code(p["bibtex"], language="bibtex")

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
