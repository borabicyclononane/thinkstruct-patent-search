"""First pass: embed claim 1 of every patent with PatentSBERTa, look at the latent space.

Run:  .venv/bin/python3.14 explore/claim1_umap.py

Writes plots + metrics to explore/out/, and UMAP/PCA coords to
features/claim1_umap.parquet so they show up in the viewer.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "data/patent_data_small"
OUT_DIR = "explore/out"
MODEL = "AI-Growth-Lab/PatentSBERTa"
SEED = 0

# Default categorical palette, fixed order; anything past slot 8 folds into grey "other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#c3c2b7"

# Same heuristic as viewer/app.py:split_cpc (main group / subgroup separator is missing from the data).
CPC_MAIN_GROUP_OVERRIDES = {"B60D1167": "1"}

# Placeholder claims like "1 - 15 . (canceled)" left over from continuation filings.
CANCELLED = re.compile(r"^[\d\s.,:\-–]*\(\s*cancell?ed\s*\)\s*\.?\s*$", re.I)


def main_group(code: str) -> str:
    subclass, digits = re.match(r"([A-H]\d\d[A-Z])(\d+)", code).groups()
    main = CPC_MAIN_GROUP_OVERRIDES.get(subclass + digits) or (digits[:1] if len(digits) == 3 else digits[:2])
    return f"{subclass} {main}"


def load() -> pd.DataFrame:
    rows = [p for f in sorted(glob.glob(os.path.join(DATA_DIR, "patents_ipa*.json"))) for p in json.load(open(f))]
    df = pd.DataFrame(rows).drop_duplicates("doc_number").reset_index(drop=True)
    df["subclass"] = df["classification"].str[:4]
    df["main_group"] = df["classification"].map(main_group)
    df["claims"] = df["claims"].map(lambda cs: [c for c in cs if not CANCELLED.match(c)])
    df["claim1"] = df["claims"].str[0]  # first live claim
    return df


def embed(texts: list[str]) -> np.ndarray:
    cache = os.path.join(OUT_DIR, "claim1_emb.npy")
    if os.path.exists(cache):
        return np.load(cache)
    model = SentenceTransformer(MODEL)
    print(f"max_seq_length={model.max_seq_length}, dim={model.get_sentence_embedding_dimension()}")
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def color_map(labels: pd.Series, top: int = len(PALETTE)) -> dict:
    """Most frequent labels get palette slots in order; the rest map to OTHER."""
    order = labels.value_counts().index[:top]
    return {lab: PALETTE[i] for i, lab in enumerate(order)}


def scatter(ax, xy, labels: pd.Series, cmap: dict, title: str):
    other = ~labels.isin(cmap)
    ax.scatter(*xy[other.values].T, s=14, c=OTHER, edgecolors="white", linewidths=0.4, label=f"other ({other.sum()})")
    for lab, col in cmap.items():
        m = (labels == lab).values
        ax.scatter(*xy[m].T, s=18, c=col, edgecolors="white", linewidths=0.4, label=f"{lab} ({m.sum()})")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d9d8d4")
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))


def fit_umap(emb: np.ndarray) -> np.ndarray:
    pcs = PCA(n_components=min(50, len(emb) - 1), random_state=SEED).fit_transform(emb)
    return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(pcs)


def knn_purity(emb: np.ndarray, labels: pd.Series, k: int = 10) -> tuple[float, float]:
    """Fraction of each point's k nearest neighbours (cosine, full 768-d) sharing its label, vs. chance."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
    y = labels.to_numpy(dtype=object)
    purity = (y[idx] == y[:, None]).mean()
    p = labels.value_counts(normalize=True)
    chance = (p ** 2).sum()  # expected agreement if neighbours were random
    return purity, chance


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    words = df["claim1"].str.split().str.len()
    print(f"{len(df)} patents; claim-1 words: median {words.median():.0f}, p95 {words.quantile(.95):.0f}, "
          f"<20 words: {(words < 20).sum()}")

    emb = embed(df["claim1"].tolist())
    pca = PCA(n_components=50, random_state=SEED).fit(emb)
    pc = pca.transform(emb)
    xy = fit_umap(emb)

    # --- metrics ---
    lines = [f"PCA explained variance: PC1 {pca.explained_variance_ratio_[0]:.3f}, PC2 {pca.explained_variance_ratio_[1]:.3f}, "
             f"top-10 {pca.explained_variance_ratio_[:10].sum():.3f}, top-50 {pca.explained_variance_ratio_.sum():.3f}"]
    for name, lab, mask in [
        ("subclass (all)", df["subclass"], np.ones(len(df), bool)),
        ("subclass (B60B vs B60C)", df["subclass"], df["subclass"].isin(["B60B", "B60C"]).values),
        ("main group (all)", df["main_group"], np.ones(len(df), bool)),
        ("main group within B60B", df["main_group"], (df["subclass"] == "B60B").values),
        ("main group within B60C", df["main_group"], (df["subclass"] == "B60C").values),
    ]:
        purity, chance = knn_purity(emb[mask], lab[mask])
        sil = silhouette_score(emb[mask], lab[mask], metric="cosine") if lab[mask].nunique() > 1 else float("nan")
        lines.append(f"{name:28s} n={mask.sum():4d}  kNN@10 purity {purity:.3f} (chance {chance:.3f})  silhouette {sil:+.3f}")
    report = "\n".join(lines)
    print(report)
    open(os.path.join(OUT_DIR, "metrics.txt"), "w").write(report + "\n")

    # --- plots ---
    sub_cmap = color_map(df["subclass"], top=3)  # B60B / B60C / B60D; the 4 singletons go to other
    mg_cmap = color_map(df["main_group"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scatter(axes[0], pc[:, :2], df["subclass"], sub_cmap,
            f"PCA (PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}) by subclass")
    scatter(axes[1], xy, df["subclass"], sub_cmap, "UMAP (PCA-50 → cosine UMAP) by subclass")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "1_subclass_pca_vs_umap.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter(ax, xy, df["main_group"], mg_cmap, "UMAP by main group (top 8)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "2_umap_main_group.png"), dpi=150)

    # Re-fit UMAP within each big subclass: does the space separate main groups locally?
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, sc in zip(axes, ["B60B", "B60C"]):
        m = (df["subclass"] == sc).values
        sub = df.loc[m, "main_group"]
        scatter(ax, fit_umap(emb[m]), sub, color_map(sub), f"{sc} only: UMAP by main group (top 8)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "3_within_subclass_umap.png"), dpi=150)

    # Same UMAP, coloured by claim length, to check whether length (not topic) drives structure.
    fig, ax = plt.subplots(figsize=(8, 6))
    s = ax.scatter(*xy.T, s=16, c=np.log10(words), cmap="Blues", edgecolors="#52514e", linewidths=0.2)
    fig.colorbar(s, ax=ax, label="log10(claim 1 words)")
    ax.set_title("UMAP by claim-1 length", fontsize=11, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "4_umap_claim_length.png"), dpi=150)

    os.makedirs("features", exist_ok=True)
    pd.DataFrame({"doc_number": df["doc_number"], "c1_umap_x": xy[:, 0], "c1_umap_y": xy[:, 1],
                  "c1_pc1": pc[:, 0], "c1_pc2": pc[:, 1], "claim1_words": words}).to_parquet("features/claim1_umap.parquet")


if __name__ == "__main__":
    main()
