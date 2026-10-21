"""Plots of the beam-vs-flat tradeoff, from the tables written by `python -m search eval`."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLAT, BEAM, RANDOM = "#eb6834", "#2a78d6", "#8a8984"
BEAM_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # light -> dark = small -> large B
K_COLORS = {1: "#2a78d6", 5: "#1baf7a", 10: "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = [("agree_subclass", "Agreement: subclass"), ("agree_main_group", "Agreement: main group"),
           ("agree_subgroup", "Agreement: subgroup"), ("knn_main_group", "kNN accuracy: main group"),
           ("knn_subgroup", "kNN accuracy: subgroup"), ("ndcg", "nDCG")]
LABEL_B = {1, 10, 30, 100}  # annotated points
TICK_B = {1, 2, 5, 10, 20, 50, 100}  # labelled x ticks

plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left", "font.size": 9,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def _stats(per_query: pd.DataFrame, k: int) -> pd.DataFrame:
    d = per_query[per_query["k"] == k]
    cols = [c for c in d.columns if c.startswith(("agree_", "knn_")) or c == "ndcg"]
    g = d.groupby(["method", "B"], dropna=False)[cols]
    return g.mean().join(g.sem(), rsuffix="_se").reset_index()


def speed_vs_accuracy(per_query, timing, k, path):
    s = _stats(per_query, k)
    beam = s[s["method"] == "beam"].merge(timing[timing["operation"] == "beam search"][["B", "ms_median"]], on="B")
    flat = s[s["method"] == "flat"].iloc[0]
    flat_ms = timing.loc[timing["operation"] == "flat search", "ms_median"].iloc[0]
    rand = s[s["method"] == "random"].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (m, title) in zip(axes.flat, METRICS):
        ax.errorbar(beam["ms_median"], beam[m], yerr=beam[m + "_se"], color=BEAM, lw=2, marker="o", ms=6,
                    mec="white", mew=1, capsize=0, elinewidth=1, label="beam search")
        for _, r in beam[beam["B"].isin(LABEL_B)].iterrows():
            ax.annotate(f"B={r['B']}", (r["ms_median"], r[m]), textcoords="offset points", xytext=(5, 5),
                        fontsize=7.5, color=MUTED)
        ax.errorbar([flat_ms], [flat[m]], yerr=[flat[m + "_se"]], color=FLAT, marker="D", ms=8, mec="white", mew=1,
                    elinewidth=1, ls="none", label="flat search")
        ax.axhline(flat[m], color=FLAT, lw=1, ls="--", alpha=0.6)
        ax.axhline(rand[m], color=RANDOM, lw=1, ls=":", label="random")
        ax.set_title(f"{title}@{k}")
        ax.set_xlabel("median search latency (ms / query)")
        ax.set_xlim(left=0)
    axes.flat[0].legend(loc="lower right")
    fig.suptitle(f"Speed vs accuracy, k={k}  ({per_query['query'].nunique()} queries; bars = ±1 standard error; latency excludes the "
                 f"~40 ms query embedding, same for both)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paired_difference(per_query, k, path):
    flat = per_query[(per_query["method"] == "flat") & (per_query["k"] == k)].set_index("query")
    beam = per_query[(per_query["method"] == "beam") & (per_query["k"] == k)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (m, title) in zip(axes.flat, METRICS):
        rows = []
        for B, d in beam.groupby("B"):
            diff = d.set_index("query")[m] - flat.loc[d["query"], m].values
            rows.append((B, diff.mean(), 1.96 * diff.sem()))
        B, mean, ci = map(np.array, zip(*rows))
        ax.axhline(0, color=FLAT, lw=1.2, ls="--")
        ax.fill_between(B, mean - ci, mean + ci, color=BEAM, alpha=0.15, lw=0)
        ax.plot(B, mean, color=BEAM, lw=2, marker="o", ms=6, mec="white", mew=1)
        ax.set_xscale("log")
        _b_ticks(ax, B)
        ax.set_title(f"{title}@{k}: beam − flat")
        ax.set_xlabel("beam width B")
    fig.suptitle(f"Beam minus flat, paired per query, k={k}  (band = 95% CI; above the dashed line = beam better)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overlap_and_latency(per_query, timing, path):
    beam = per_query[per_query["method"] == "beam"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, d in beam.groupby("k"):
        g = d.groupby("B")
        ax1.plot(g["claim_overlap"].mean(), color=K_COLORS[k], lw=2, marker="o", ms=6, mec="white", mew=1,
                 label=f"claims, k={k}")
        ax1.plot(g["patent_overlap"].mean(), color=K_COLORS[k], lw=1.2, ls="--", label=f"patents, k={k}")
    ax1.set_xscale("log")
    Bs = sorted(beam["B"].unique())
    _b_ticks(ax1, Bs)
    ax1.set_ylim(0, 1.03)
    ax1.set_xlabel("beam width B")
    ax1.set_title("Share of flat search's top-k that beam search also returns")
    ax1.legend(ncol=2, loc="lower right")
    t = timing[timing["operation"] == "beam search"]
    ax2.plot(t["B"], t["ms_median"], color=BEAM, lw=2, marker="o", ms=6, mec="white", mew=1, label="beam, median")
    ax2.fill_between(t["B"], t["ms_median"], t["ms_p95"], color=BEAM, alpha=0.15, lw=0, label="beam, median–p95")
    f = timing[timing["operation"] == "flat search"].iloc[0]
    ax2.axhline(f["ms_median"], color=FLAT, lw=2, ls="--", label=f"flat, median ({f['ms_median']:.2f} ms)")
    ax2.axhspan(f["ms_median"], f["ms_p95"], color=FLAT, alpha=0.08, lw=0)
    ax2.set_xscale("log")
    _b_ticks(ax2, Bs)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("beam width B")
    ax2.set_ylabel("ms per query")
    q = timing.loc[timing["operation"] == "embed query", "ms_mean"].iloc[0]
    ax2.set_title(f"Search latency (query embedding adds ~{q:.0f} ms to both)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_level(per_query, k, path, show_B=(10, 20, 50)):
    levels = ["subclass", "main_group", "subgroup"]
    s = _stats(per_query, k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, prefix, title in [(axes[0], "agree", "Agreement@k by CPC level"), (axes[1], "knn", "kNN accuracy by CPC level")]:
        cols = [f"{prefix}_{l}" for l in levels]
        x = np.arange(len(levels))
        def line(row, color, label, **kw):
            ax.errorbar(x, row[cols].astype(float), yerr=row[[c + "_se" for c in cols]].astype(float), color=color,
                        lw=2, marker="o", ms=6, mec="white", mew=1, elinewidth=1, label=label, **kw)
        line(s[s["method"] == "flat"].iloc[0], FLAT, "flat", ls="--", zorder=5)  # drawn on top: large B matches it
        for B, color in zip(show_B, [BEAM_RAMP[1], BEAM_RAMP[2], BEAM_RAMP[4]]):
            r = s[(s["method"] == "beam") & (s["B"] == B)]
            if len(r):
                line(r.iloc[0], color, f"beam B={B}")
        base = "random" if prefix == "agree" else "majority class"
        line(s[s["method"] == base].iloc[0], RANDOM, base, ls=":")
        ax.set_xticks(x, [l.replace("_", " ") for l in levels])
        ax.set_title(f"{title}, k={k}")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hybrid_latency(timing: pd.DataFrame, path: str):
    """Median latency per filter scenario (dots; line to p95), with the unfiltered searches as reference lines."""
    methods = [("flat post-filter", RANDOM, "o"), ("flat pre-filter", FLAT, "D"), ("beam pruned", BEAM, "s")]
    t = timing[timing["scenario"] != "no filter"]
    order = t.drop_duplicates("scenario").sort_values("n_candidates")
    y = {s: i for i, s in enumerate(order["scenario"])}
    fig, ax = plt.subplots(figsize=(10, 6))
    for (m, color, marker), dy in zip(methods, (0.2, 0, -0.2)):
        d = t[t["method"] == m]
        ys = d["scenario"].map(y) + dy
        ax.hlines(ys, d["ms_median"], d["ms_p95"], color=color, lw=1, alpha=0.5)
        ax.plot(d["ms_median"], ys, ls="none", marker=marker, ms=8, color=color, mec="white", mew=1, label=m)
    for m, color in (("flat", FLAT), ("beam", BEAM)):
        ms = timing.loc[(timing["scenario"] == "no filter") & (timing["method"] == m), "ms_median"].iloc[0]
        ax.axvline(ms, color=color, lw=1.2, ls="--", label=f"{m}, no filter ({ms:.3f} ms)")
    ax.set_yticks(list(y.values()), [f"{s}   ({n} patents)" for s, n in zip(order["scenario"], order["n_candidates"])])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=0)
    ax.set_xlabel("ms per query (dot = median, line to p95); filtered timings include resolving the filter")
    ax.set_title(f"Hybrid search latency by filter ({timing['n_candidates'].max()} patents in the index; "
                 f"query embedding ~40 ms not included)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _b_ticks(ax, Bs):
    ax.set_xticks(list(Bs), [str(b) if b in TICK_B else "" for b in Bs])
    ax.minorticks_off()


def make_plots(per_query: pd.DataFrame, timing: pd.DataFrame, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    per_query, timing = per_query.copy(), timing.copy()
    per_query["B"], timing["B"] = per_query["B"].astype("Int64"), timing["B"].astype("Int64")  # NaN-able ints from CSV
    paths = []
    for k in sorted(per_query["k"].unique()):
        paths.append(os.path.join(out_dir, f"speed_vs_accuracy_k{k}.png"))
        speed_vs_accuracy(per_query, timing, k, paths[-1])
        paths.append(os.path.join(out_dir, f"beam_minus_flat_k{k}.png"))
        paired_difference(per_query, k, paths[-1])
    paths.append(os.path.join(out_dir, "overlap_and_latency_vs_B.png"))
    overlap_and_latency(per_query, timing, paths[-1])
    paths.append(os.path.join(out_dir, "cpc_levels_k10.png"))
    by_level(per_query, 10, paths[-1])
    return paths
