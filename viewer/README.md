# Patent viewer

A small web UI for browsing the patents and plotting any features you compute for them.

## Run

From the repo root:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m viewer
```

This opens http://localhost:8501. Options (`python -m viewer --help`):

| flag | default | |
|---|---|---|
| `--data-dir` | `data/patent_data_small` | folder of `patents_ipa*.json` files |
| `--features-dir` | `features` | folder of feature files (see below) |
| `--port` | `8501` | |
| `--no-browser` | off | don't open a browser tab |

## Tabs

- **Browse**: 🎲 shows a random patent, or search by doc number or title. Shows metadata, abstract, claims, description and BibTeX.
  The breadcrumb shows where the patent sits in the CPC tree (e.g. `B` → `B60` → `B60C` → `B60C 11` → `B60C 11/03`). Under "Nearby in the classification tree", pick subgroup / main group / subclass to list other patents in that subtree, and click a row to open one.
- **Classification tree**: treemap of subclass → main group → subgroup, sized by patent count. Click a box to zoom in.
- **Explore features**: a table of all patents, group-by with counts and means, and histogram, scatter, bar and box plots.

## Adding features

Save a `.csv` or `.parquet` with a `doc_number` column into `features/`, then refresh the page:

```python
df[["doc_number", "my_score", "my_cluster"]].to_parquet("features/my_feats.parquet")
```

Every column is joined onto the patents and shows up in the table, group-by and plot dropdowns. List or vector columns (e.g. embeddings) are loaded but left out of the plots, so reduce them to 2D columns first (e.g. `pca_x`, `pca_y`).

Built-in columns: `week` (taken from the filename), `subclass` (first 4 characters of the classification), `main_group` and `subgroup` (parsed from the classification; see `split_cpc` in `app.py`), `n_claims`, `n_paragraphs`, `abstract_words`.
