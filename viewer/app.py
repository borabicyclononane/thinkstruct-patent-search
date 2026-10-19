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

browse, tree, explore = st.tabs(["Browse", "Classification tree", "Explore features"])


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
    rel = df[(df[level] == p[level]) & (df.index != doc)].sort_values(["subgroup", "week"])
    if rel.empty:
        st.info(f"No other patents in {p[level]}. Try a broader level.")
    else:
        st.caption(f"{len(rel)} other patents · click a row to open it")
        key = f"rel_{doc}_{level}"  # fresh widget per patent/level so old selections don't carry over
        st.dataframe(
            rel[["title", "subgroup", "week"]], key=key, width="stretch",
            on_select=lambda: open_selected_row(key, rel), selection_mode="single-row",
        )

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

with tree:
    st.caption(
        "CPC hierarchy: subclass → main group → subgroup, sized by number of patents. Click a box to zoom in. "
        "Subgroups are shown flat under their main group (the true dot-level nesting isn't in the data)."
    )
    counts = df.groupby(["subclass", "main_group", "subgroup"]).size().reset_index(name="patents")
    fig = px.treemap(counts, path=[px.Constant("B60"), "subclass", "main_group", "subgroup"], values="patents")
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
