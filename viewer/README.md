# Viewer

```
make viewer        # or: .venv/bin/python3.14 -m viewer [--port 8501] [--no-browser]
```

Opens http://localhost:8501. The sidebar picks a page and a dataset (the vehicle patents, or the 10^4 large-dataset sample once `make large` has run).

- **Search**: a query ranked by beam or flat search, with any embedding: raw 768-d, PCA-64, or the trained towers (after `make part3`). Keyword, CPC and title filters sit under "Filters". Click a result to preview it.
- **Browse**: one patent at a time: CPC path, abstract, claims, description, and its neighbours in the CPC tree.
- **Synthetic queries**: a Part 3 query next to the claim(s) it came from, how it was made, and whether each method finds the source at a chosen beam width.
- **Results**: `results/README.md`.
- **Explore**: the CPC tree as a treemap, and plots of any column, including feature files in `features/` (a `.csv` or `.parquet` with a `doc_number` column).
