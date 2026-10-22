"""Patent search viewer: search with any method, browse patents, inspect the synthetic queries, read the results.

Run:  python -m viewer   (or: make viewer)

Pages (sidebar): Search, Browse, Synthetic queries, Results, Explore. The dataset switch covers Search, Browse and
Explore: the vehicle dataset, or the 10^4 large-dataset sample once `make large` has built and embedded it.
Embeddings in Search: raw 768-d and PCA-64 always; the trained towers (vehicle dataset) once `make part3` has run.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "synthetic")):
    sys.path.insert(0, p)
from generate_synthetic import DATASETS as SYN_DATASETS, SYN, load_claims  # noqa: E402
from search.data import CPC_LEVELS, load_corpus  # noqa: E402
from search.embed import MODEL, Embedder  # noqa: E402
from search.filters import FIELDS, TOKEN, Filter, TextIndex, fold, hybrid_search, query_terms  # noqa: E402
from search.index import Index, build_index, l2n  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/patent_data_small")
ap.add_argument("--features-dir", default="features")
args = ap.parse_args()
os.chdir(ROOT)
SAMPLE = "data/large/sample_10000.jsonl"
SAMPLE_CACHE = "data/index/embeddings_large.npz"  # filled by experiment-big-data/run.py
RUN, BUNDLE = "training/runs/final", "training/bundles/full.npz"
HAS_SAMPLE = os.path.exists(SAMPLE) and os.path.exists(SAMPLE_CACHE)
HAS_TOWERS = os.path.exists(os.path.join(RUN, "towers.pt"))

# Codes where the digit heuristic below picks the wrong split (checked against the USPTO CPC scheme pages).
CPC_MAIN_GROUP_OVERRIDES = {"B60D1167": "1"}  # B60D 1/167, not 11/67


def split_cpc(code: str) -> tuple[str, str]:
    """'B60C1103FI' -> ('B60C 11', 'B60C 11/03'). The separator is missing from the data: 3 digits -> 1/23,
    4 -> 12/34, 5 -> 12/345; on the small dataset every code but the overrides maps to an official symbol."""
    subclass, digits = re.match(r"([A-H]\d\d[A-Z])(\d+)", code).groups()
    main = CPC_MAIN_GROUP_OVERRIDES.get(subclass + digits) or (digits[:1] if len(digits) == 3 else digits[:2])
    return f"{subclass} {main}", f"{subclass} {main}/{digits[len(main):]}"


# ---------------------------------------------------------------- data

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
    patents = [json.loads(line) for line in open(path)]
    for p in patents:
        p["path"] = tuple(p["path"])
    return patents


@st.cache_data
def load_sample_patents(path: str) -> pd.DataFrame:
    """Same columns as load_patents where the sample has them: independent claims only, no description, BibTeX or week."""
    df = pd.DataFrame(read_sample(path)).set_index("doc_number").rename(columns={"cpc": "classification"})
    levels = pd.DataFrame(df.pop("path").tolist(), index=df.index, columns=CPC_LEVELS)
    df[["subclass", "main_group", "subgroup"]] = levels[["subclass", "main_group", "subgroup"]]
    df["claim_nums"] = df["claims"].map(lambda cs: [c["num"] for c in cs])
    df["claims"] = df["claims"].map(lambda cs: [c["text"] for c in cs])
    df["year"] = df.index.str[:4].astype(int)
    df["n_claims"] = df["claims"].str.len()
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


@st.cache_resource
def load_bundle() -> dict:
    return dict(np.load(BUNDLE))


@st.cache_resource
def search_methods(sample: bool) -> tuple[dict, TextIndex]:
    """Embedding name -> Index over the dataset's independent claims (all share one tree), and the keyword index.
    Vehicle dataset with a trained run: the Part 3 maps (training/towers.py). Otherwise raw and PCA-64 fit on all claims."""
    if sample:
        raw, _ = build_index(read_sample(SAMPLE), Embedder(SAMPLE_CACHE))
    else:
        raw, _ = build_index(load_corpus(args.data_dir)[0], Embedder())
    if not sample and HAS_TOWERS:
        from towers import load_maps
        maps = load_maps(RUN, load_bundle())[0]
    else:
        mu = raw.leaf_vecs.mean(0)
        comps = np.linalg.svd(raw.leaf_vecs - mu, full_matrices=False)[2][:64]
        pca = lambda x: (np.asarray(x) - mu) @ comps.T
        maps = {"raw 768": None, "PCA-64": (pca, pca)}
    return {name: raw if m is None or name == "raw 768" else Index(raw.tree, raw.leaf_vecs, project=m[1], project_query=m[0])
            for name, m in maps.items()}, TextIndex(raw)


@st.cache_resource
def load_query_model():
    """On CPU on purpose: Streamlit runs every rerun in a new thread, and torch's MPS backend aborts the process when
    two threads use it at once. One query takes ~35 ms on CPU; the lock makes overlapping reruns take turns."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device="cpu"), threading.Lock()


def embed_query(text: str) -> np.ndarray:
    model, lock = load_query_model()
    with lock:
        return model.encode([text], normalize_embeddings=True)[0]


@st.cache_data
def synthetic_queries() -> pd.DataFrame:
    rows = [json.loads(line) for d in SYN_DATASETS for line in open(os.path.join(SYN, f"{d}.jsonl"))]
    return pd.DataFrame(rows).set_index("query_id")


@st.cache_resource
def source_claims() -> dict:
    return {c["claim_id"]: c for c in load_claims()}


@st.cache_resource
def part3_indexes():
    """The Part 3 evaluation setup (results/compare_part3.py): every method's Index over all 1,191 claims."""
    from towers import load_maps
    from train_towers import claim_tree
    b = load_bundle()
    maps, best = load_maps(RUN, b)
    tree, claim_of_leaf = claim_tree(b)
    leaf_vecs = b["claim_vecs"][[claim_of_leaf[l] for l in tree.leaves]]
    indexes = {name: Index(tree, leaf_vecs, project=kf, project_query=qf) for name, (qf, kf) in maps.items()}
    row_of = {q: i for i, q in enumerate(b["query_ids"])}
    claim_ids = {l: b["claim_ids"][c] for l, c in claim_of_leaf.items()}
    return indexes, best, row_of, claim_ids


def highlight(text: str, terms: set[str]) -> str:
    def bold(m):
        w = m.group().lower()
        return f"**{m.group()}**" if fold(w) in terms or any(fold(x) in terms for x in w.split("-")) else m.group()
    return re.sub(TOKEN.pattern, bold, text, flags=re.I)


# ---------------------------------------------------------------- layout

st.set_page_config(page_title="Patent Search", page_icon="🔎", layout="wide")
with st.sidebar:
    st.title("Patent search")
    page = st.radio("Page", ["Search", "Browse", "Synthetic queries", "Results", "Explore"], key="page",
                    label_visibility="collapsed")
    st.divider()
    use_sample = st.radio("Dataset", ["Vehicles (B60)", "Large sample (10^4)"], disabled=not HAS_SAMPLE,
                          help=None if HAS_SAMPLE else "Run `make large` to build the 10^4 sample.") != "Vehicles (B60)"
DATE = "year" if use_sample else "week"
patents = load_sample_patents(SAMPLE) if use_sample else load_patents(args.data_dir)
df = patents if use_sample else patents.join(load_features(args.features_dir), how="left")
st.sidebar.caption(f"{len(df):,} patents")


def set_doc(d):
    st.session_state.doc = d
    st.session_state.pick = None


def search_page():
    st.header("Search")
    methods, text = search_methods(use_sample)
    query = st.text_area("Query", placeholder="A claim, or a description of the invention", height=110)
    c1, c2, c3, c4 = st.columns([1.2, 1, 2, 1])
    mode = c1.segmented_control("Search", ["Beam", "Flat"], default="Beam",
                                help="Beam: walk the CPC tree. Flat: score every claim.") or "Beam"
    B = c2.number_input("Beam width B", 1, 500, 5, disabled=mode == "Flat")
    emb = c3.selectbox("Embedding", list(methods), help=None if HAS_TOWERS or use_sample else "Run `make part3` to add the trained towers.")
    k = c4.number_input("Results", 1, 200, 10)

    with st.expander("Filters"):
        st.caption("Keywords match whole words in the title and/or abstract, ignoring case and plurals.")
        f1, f2, f3 = st.columns(3)
        all_of = f1.text_input("All of", placeholder="tread noise")
        any_of = f2.text_input("Any of", placeholder="stud hitch")
        fields = f3.multiselect("In", FIELDS, default=list(FIELDS))
        f4, f5 = st.columns(2)
        cpc = f4.text_input("CPC prefixes (any of)", placeholder="B60B, B60C 11")
        title = f5.text_input("Exact title")
    words = lambda s: tuple(w for w in re.split(r"[\s,]+", s.strip()) if w)
    f = Filter(cpc=tuple(c.strip() for c in cpc.split(",") if c.strip()), all_of=words(all_of), any_of=words(any_of),
               fields=tuple(fields), title=title.strip() or None)
    if (f.all_of or f.any_of) and not fields:
        return st.warning("Pick at least one field to search keywords in.")
    if not f and not query.strip():
        return st.caption("Type a query, or set a filter to list matching patents.")

    index = methods[emb]
    tree = index.tree
    t0 = time.perf_counter()
    starts, cand = text.resolve(f)
    if cand is None and f.cpc:
        cand = text.cpc_positions(starts)
    t_filter = time.perf_counter() - t0
    n = len(tree.order) if cand is None else len(cand)
    if query.strip():
        with st.spinner("Embedding the query (the model loads on first use, ~10 s)..."):
            t0 = time.perf_counter()
            qv = embed_query(query.strip())
            t_embed = time.perf_counter() - t0
        t0 = time.perf_counter()
        res, _ = hybrid_search(index, text, qv, f, k, B, flat=mode == "Flat")
        t_search = time.perf_counter() - t0
        pats = res.patents[:k]
        qn = l2n(index.project_query(qv[None]))[0]
        best = [tree.children[p][np.argmax(index.node_scores(qn, tree.children[p]))] for p in pats]
        hits = pd.DataFrame({"score": res.patent_scores[:k].round(3), "best claim": [tree.claim[c]["num"] for c in best]},
                            index=pd.Index([tree.patent[p]["doc_number"] for p in pats], name="doc_number"))
        timing = f"search {1000 * t_search:.2f} ms · filter {1000 * t_filter:.2f} ms · query embedding {1000 * t_embed:.0f} ms"
    else:
        positions = range(len(tree.order)) if cand is None else cand
        hits = pd.DataFrame(index=pd.Index([tree.patent[tree.order[i]]["doc_number"] for i in positions], name="doc_number"))
        timing = f"filter {1000 * t_filter:.2f} ms"
    hits = hits.join(df[["title", "subgroup", DATE]])
    if DATE == "week":
        hits["week"] = hits["week"].dt.date
    note = f" · beam returns at most B = {B} patents" if query.strip() and mode == "Beam" and k > B and n > B else ""
    st.caption(f"{n:,} of {len(tree.order):,} patents pass the filters · {timing}{note}")
    key = f"hits_{hash((repr(f), query, mode, B, k, emb, use_sample))}"
    st.dataframe(hits, key=key, width="stretch", on_select="rerun", selection_mode="single-row")
    sel = st.session_state[key].selection.rows
    if sel:
        d = hits.index[sel[0]]
        terms = {t for kw in f.all_of + f.any_of for t in query_terms(kw)}
        with st.container(border=True):
            st.markdown(f"#### {highlight(df.at[d, 'title'], terms)}")
            st.caption(f"{d} · {df.at[d, 'subgroup']}")
            st.markdown(highlight(df.at[d, "abstract"], terms))
            if "best claim" in hits:
                num = hits.at[d, "best claim"]
                claim = next(c for c in tree.patent[tree.patent_node[d]]["claims"] if c["num"] == num)
                st.markdown(f"**Best-matching claim ({num}).** {claim['text']}")
            st.button("Open in Browse", on_click=lambda: (set_doc(d), st.session_state.update(page="Browse")))


def browse_page():
    if st.session_state.get("doc") not in df.index:
        st.session_state.doc = df.sample(1).index[0]
    c1, c2 = st.columns([1, 4])
    c1.button("🎲 Random patent", on_click=lambda: set_doc(df.sample(1).index[0]))
    c2.selectbox("Pick one", df.index, index=None, key="pick", placeholder="Search by doc number or title",
                 format_func=lambda d: f"{d} — {df.at[d, 'title']}", label_visibility="collapsed",
                 on_change=lambda: st.session_state.pick and set_doc(st.session_state.pick))
    doc = st.session_state.doc
    p = df.loc[doc]
    st.header(p["title"])
    st.markdown(f"`{doc}` · " + " → ".join(f"`{n}`" for n in
                [p["classification"][:1], p["classification"][:3], p["subclass"], p["main_group"], p["subgroup"]]))
    st.write(p["abstract"])
    nums = p["claim_nums"] if use_sample else range(1, len(p["claims"]) + 1)
    with st.expander(f"{'Independent claims' if use_sample else 'Claims'} ({len(p['claims'])})"):
        for i, c in zip(nums, p["claims"]):
            st.markdown(f"**{i}.** {c}")
    if not use_sample:
        with st.expander(f"Detailed description ({len(p['detailed_description'])} paragraphs)"):
            for i, para in enumerate(p["detailed_description"], 1):
                st.markdown(f"`[{i:04d}]` {para}")
    with st.expander("Nearby in the classification tree"):
        level = st.segmented_control("Same", ["subgroup", "main_group", "subclass"], default="subgroup",
                                     format_func=lambda lvl: f"{lvl.replace('_', ' ')} {p[lvl]}") or "subgroup"
        rel = df[(df[level] == p[level]) & (df.index != doc)].sort_values(["subgroup", DATE])
        if rel.empty:
            st.caption(f"No other patents in {p[level]}.")
        else:
            key = f"rel_{doc}_{level}"
            st.dataframe(rel[["title", "subgroup", DATE]].assign(**{DATE: rel[DATE].dt.date} if DATE == "week" else {}), key=key, width="stretch", selection_mode="single-row",
                         on_select=lambda: st.session_state[key].selection.rows
                         and set_doc(rel.index[st.session_state[key].selection.rows[0]]))


def elements_table(elements: list[dict], used: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"used": "✔" if e["id"] in used else "", "element": e["text"]} for e in elements])


def synthetic_page():
    st.header("Synthetic queries")
    if not HAS_TOWERS:
        st.info("Run `make part3` to train the towers; this page compares every method on each query.")
        return
    q_df, claims = synthetic_queries(), source_claims()
    c1, c2, c3 = st.columns([2, 1, 1])
    dataset = c1.segmented_control("Dataset", SYN_DATASETS, default="paraphrase", format_func=lambda d: d.replace("_", " "))
    split = c2.selectbox("Split", ["test", "val", "train"])
    pool = q_df[(q_df.dataset == (dataset or "paraphrase")) & (q_df.split == split)].index.tolist()
    if st.session_state.get("query") not in pool:
        st.session_state.query = pool[0]
    c3.button("🎲 Random query", on_click=lambda: st.session_state.update(query=np.random.choice(pool)))
    st.selectbox("Query", pool, key="query", label_visibility="collapsed",
                 format_func=lambda q: f"{q} — {claims[q_df.at[q, 'answer'][0]['claim_id']]['title'][:80]}")
    r = q_df.loc[st.session_state.query]
    with st.container(border=True):
        st.caption("Query")
        st.write(r["query_text"])
    src = [claims[a["claim_id"]] for a in r["answer"]]
    for col, c, name in zip(st.columns(len(src)), src, ["Source A", "Source B"] if len(src) == 2 else ["Source"]):
        with col.container(border=True):
            st.caption(f"{name} · {c['claim_id']} · {c['cpc']} · {c['title']}")
            st.write(c["text"])
    m = r["meta"]
    with st.expander("How it was made"):
        if "mix" in m:
            st.caption(f"mix: {m['mix'].replace('_', ' ')}")
            a, b = st.columns(2)
            a.dataframe(elements_table(m["elements_a"], m["used_from_a"]), hide_index=True, width="stretch")
            b.dataframe(elements_table(m["elements_b"], m["used_from_b"]), hide_index=True, width="stretch")
        elif "elements" in m:
            st.dataframe(elements_table(m["elements"], m["chosen_elements"]), hide_index=True, width="stretch")
        else:
            st.caption(f"token Jaccard {m.get('jaccard')} · ROUGE-L {m.get('rouge_l')}")
        for flag in m.get("flags", []):
            st.caption(f"flag: {flag}")

    st.subheader("Where does each method find the source?")
    indexes, best, row_of, claim_ids = part3_indexes()
    B = st.segmented_control("Beam width B", [1, 5, 10, 30, 100], default=5) or 5
    qv = load_bundle()["query_vecs"][row_of[st.session_state.query]]
    want = {a["claim_id"] for a in r["answer"]}
    rows = []
    for name, index in indexes.items():
        found = {claim_ids[l] for l in index.beam_search(qv, B).claims} & want
        ranking = [claim_ids[l] for l in index.flat_search(qv).claims]
        rows.append({"method": name, "config": best.get(name, "").replace("towers ", "").replace(" per-level q", ""),
                     f"beam B={B}": "✔" if found == want else "✗", "rank in flat search": max(ranking.index(w) for w in want) + 1})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Found: every source came back (a combination needs both). Rank: position of the (last) source among "
               "all 1,191 claims when every claim is scored.")


def results_page():
    text = open(os.path.join(ROOT, "results", "README.md")).read()
    st.markdown(re.sub(r"```.*?```\n", "", text, flags=re.S))  # drop the command block


def explore_page():
    st.header("Explore")
    view = st.segmented_control("View", ["Classification tree", "Features"], default="Classification tree",
                                label_visibility="collapsed") or "Classification tree"
    if view == "Features":
        text_cols = ["title", "abstract", "claims", "claim_nums", "detailed_description", "bibtex", "filename"]
        cols = [c for c in df.columns if c not in text_cols and df[c].map(lambda v: not isinstance(v, (list, dict))).all()]
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        p1, p2, p3 = st.columns(3)
        x = p1.selectbox("x", numeric)
        y = p2.selectbox("y", [None] + numeric)
        color = p3.selectbox("color", [None] + cols, index=1 + cols.index("subclass"))
        data = df.reset_index()
        fig = px.histogram(data, x=x, color=color) if y is None else \
            px.scatter(data, x=x, y=y, color=color, hover_data=["doc_number", "title"])
        st.plotly_chart(fig, width="stretch")
        with st.expander("Table"):
            st.dataframe(df[["title"] + cols], width="stretch")
    else:
        st.caption("Subclass → main group → subgroup, sized by number of patents. Click a box to zoom in.")
        counts = df.groupby(["subclass", "main_group", "subgroup"]).size().reset_index(name="patents")
        fig = px.treemap(counts, path=[px.Constant("all" if use_sample else "B60"), "subclass", "main_group", "subgroup"],
                         values="patents")
        fig.update_traces(root_color="lightgrey", textinfo="label+value")
        fig.update_layout(height=700, margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")


{"Search": search_page, "Browse": browse_page, "Synthetic queries": synthetic_page,
 "Results": results_page, "Explore": explore_page}[page]()
