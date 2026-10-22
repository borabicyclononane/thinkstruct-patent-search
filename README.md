# Patent claim search

Search over patent claims that walks the CPC classification tree (beam search) instead of scoring every claim. It has keyword, CPC and title filters, and a small study on training embeddings for this kind of search.

All results and tables are in [results/README.md](results/README.md).

## Setup

You need Python 3.14 and `make`.

```
make setup
```

The task data isn't in git. Put it here:

- `data/patent_data_small/`: the `patents_ipa*.json` files of the small dataset
- `coding_challenge_large_data.zip` in the repo root, only for `make large`

## Run

```
make part1    # search index, beam vs flat, hybrid search        ~2 min
make large    # the same on a 10^4 sample of the large dataset   ~20 min (mostly embedding all the samples)
make part3    # embed synthetic queries, train towers, evaluate  ~6 min
make readme   # rebuild the tables in results/README.md
make viewer   # explore everything at http://localhost:8501
```

The first run downloads PatentSBERTa (~0.5 GB). Embeddings are cached in `data/index/`, so reruns are fast. Numbers can differ from ours in the last digit on other hardware.

## What's where

- `search/`: the index, beam and flat search, filters, evaluation (`python -m search --help`)
- `scraper/`: claims from Google Patents, with exact independent/dependent labels. The result is committed as `data/scraped/claims.jsonl`, so you don't need to rerun it (`python -m scraper`).
- `experiment-big-data/`: sampling and running everything on the large dataset
- `synthetic/`: synthetic queries made with Claude, committed in `outputs/synthetic/`. `make synthetic` regenerates them; it needs `ANTHROPIC_API_KEY` and costs about $1.50.
- `training/`: query and key towers on frozen PatentSBERTa embeddings
- `results/`: the write-up and the scripts behind its tables
- `viewer/`: the web UI

## Data choices

- A patent is indexed only if it has claims and a classification code. Everything else may be missing.
- Only the first classification code is used.
- Claims come from Google Patents rather than the provided JSON, because the provided claim text is often damaged (merged numbering, missing references).
- The large dataset's claims are clean, so independence there is decided by the same text rule the scraper uses: a claim is dependent if it cites another claim.
