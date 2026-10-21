"""Vocabulary of titles + abstracts: how many distinct words, and how often they occur.

Run:  .venv/bin/python3.14 explore/vocab.py

Tokens: lowercased, split on anything that isn't a letter or digit ("non-pneumatic" -> "non", "pneumatic").
Writes explore/out/vocab_*.png.
"""
import glob
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/patent_data_small"
OUT_DIR = "explore/out"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TF_BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 200), (201, 1000), (1001, None)]
DF_BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 32), (33, 64), (65, 320), (321, None)]

plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def bucket_label(lo, hi):
    return f">{lo - 1}" if hi is None else (str(lo) if lo == hi else f"{lo}–{hi}")


def in_bucket(c, lo, hi):
    return c >= lo and (hi is None or c <= hi)


def bars(ax, labels, values, color, fmt, title, xlabel, ylabel):
    b = ax.bar(labels, values, color=color, width=0.7)
    ax.bar_label(b, [fmt(v) for v in values], padding=2, fontsize=8, color=INK)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patents = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "patents_ipa*.json"))):
        for p in json.load(open(path)):
            patents.setdefault(p["doc_number"], p)
    patents = list(patents.values())
    titles = [tokens(p["title"]) for p in patents]
    abstracts = [tokens(p["abstract"]) for p in patents]
    both = [t + a for t, a in zip(titles, abstracts)]
    tf = Counter(w for d in both for w in d)
    df = Counter(w for d in both for w in set(d))
    n_tokens, n_patents = sum(tf.values()), len(patents)

    # 1. Corpus size: titles vs abstracts vs both
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    names = ["titles", "abstracts", "titles + abstracts"]
    groups = [titles, abstracts, both]
    bars(a1, names, [sum(map(len, g)) for g in groups], BLUE, lambda v: f"{v:,}",
         "Word occurrences", "", "occurrences")
    bars(a2, names, [len({w for d in g for w in d}) for g in groups], AQUA, lambda v: f"{v:,}",
         f"Distinct words ({n_patents} patents)", "", "distinct words")
    save(fig, "vocab_1_size.png")

    # 2. Occurrences per word: share of the vocabulary vs share of all occurrences
    labels = [bucket_label(lo, hi) for lo, hi in TF_BUCKETS]
    words = [[w for w, c in tf.items() if in_bucket(c, lo, hi)] for lo, hi in TF_BUCKETS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    bars(a1, labels, [100 * len(ws) / len(tf) for ws in words], BLUE, lambda v: f"{v:.0f}%",
         f"Share of the vocabulary ({len(tf):,} distinct words)", "times the word occurs", "% of distinct words")
    bars(a2, labels, [100 * sum(tf[w] for w in ws) / n_tokens for ws in words], ORANGE, lambda v: f"{v:.1f}%",
         f"Share of all occurrences ({n_tokens:,})", "times the word occurs", "% of all word occurrences")
    fig.suptitle("Most distinct words are rare; most text is a few very common words", x=0.01, ha="left",
                 fontsize=11, color=INK)
    save(fig, "vocab_2_occurrences.png")

    # 3. Document frequency: how many patents contain each word
    labels = [bucket_label(lo, hi) + (f"\n{100 * (lo - 1) / n_patents:.0f}–{100 * (hi or n_patents) / n_patents:.0f}%"
                                      if lo > 10 else "") for lo, hi in DF_BUCKETS]
    counts = [sum(in_bucket(c, lo, hi) for c in df.values()) for lo, hi in DF_BUCKETS]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars(ax, labels, counts, AQUA, lambda v: f"{v:,}\n({100 * v / len(df):.0f}%)",
         "How many patents contain each word (document frequency)",
         f"patents containing the word (second line: as % of all {n_patents} patents)", "distinct words")
    ax.set_ylim(top=max(counts) * 1.2)
    save(fig, "vocab_3_document_frequency.png")

    # 4. Most common words, coloured by whether they appear in more than half of all patents
    top = tf.most_common(25)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [GREY if df[w] > n_patents / 2 else BLUE for w, _ in top]
    b = ax.barh([w for w, _ in top], [c for _, c in top], color=colors, height=0.7)
    ax.bar_label(b, [f"{c:,}  (in {df[w]} patents)" for w, c in top], padding=3, fontsize=7.5, color=MUTED)
    ax.set_xlim(right=max(c for _, c in top) * 1.35)
    ax.set_title("25 most frequent words (grey: in more than half of all patents)")
    ax.set_xlabel("occurrences in titles + abstracts")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "vocab_4_top_words.png")

    # 5. Rank vs occurrences (Zipf) and words reaching each document frequency
    rank = np.arange(1, len(tf) + 1)
    occ = np.array(sorted(tf.values(), reverse=True))
    fit = (rank >= 10) & (rank <= 1000)
    slope = np.polyfit(np.log(rank[fit]), np.log(occ[fit]), 1)[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.loglog(rank, occ, color=BLUE, lw=2)
    a1.set_title(f"Word rank vs occurrences (Zipf slope {slope:.2f} over ranks 10–1000)")
    a1.set_xlabel("word rank")
    a1.set_ylabel("occurrences")
    d = np.array(sorted(df.values()))
    x = np.unique(d)
    a2.loglog(x, [(d >= v).sum() for v in x], color=AQUA, lw=2, drawstyle="steps-post")
    a2.set_title("Words appearing in at least x patents")
    a2.set_xlabel("patents containing the word")
    a2.set_ylabel("distinct words")
    for ax in (a1, a2):
        ax.grid(True, which="major", color=GRID)
    save(fig, "vocab_5_rank_and_df_curves.png")


if __name__ == "__main__":
    main()
