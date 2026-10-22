# Part 2 proof of concept: range-sharded hybrid search with Roaring bitmaps

This builds the sharded keyword lookup from `../part_2_system_design.md` on the 640-patent default dataset.

- **Coordinator:** holds the CPC tree. Node keys are stored as sum + count of claim embeddings, and each node covers a range [start, end) of patent positions in depth-first order.
- **Shards:** positions are split into S contiguous blocks. Each shard holds the claim embeddings of its block and one Roaring bitmap per title/abstract term.
- **Query:**
  - Beam search starts at the root or at the CPC filter node.
  - With keywords, each level sends one batched `count_in_ranges` call to each shard overlapping the beam's children, sums the counts, and keeps only children that contain a match ("and" counts patents that have every term).
  - The final B patents go through `exact_match` and `score` on the shards that own them; the results are merged and deduplicated by patent into the top k.
- **Hash baseline:** patents are assigned to shards by `crc32(doc number) % S`. It answers the same queries, so shards touched can be compared.

`test_correctness.py` checks, on 100 queries, that sharded results equal unsharded ones and that the counts summed over shards equal both the unsharded counts and brute force. `demo.py` shows example queries and counts the shards touched.

This is a small-scale demonstration of correctness and fan-out. Scaling behaviour is argued in the design doc, not measured here.

## Run

From the repo root, after the main setup (`make setup`):

```
.venv/bin/python3.14 -m pip install -r part2_poc/requirements.txt
.venv/bin/python3.14 part2_poc/build.py               # index + shards for S = 1, 2, 4 -> part2_poc/build/
.venv/bin/python3.14 part2_poc/test_correctness.py    # PASS/FAIL per check
.venv/bin/python3.14 part2_poc/demo.py                # examples + shards-touched table -> part2_poc/results.md
```

## Inputs

All inputs are read-only:
- the default dataset, via `search.data.load_corpus`
- the cached PatentSBERTa claim embeddings and the synthetic test queries in `training/bundles/full.npz`
- the query texts in `outputs/synthetic/`

It imports `search.filters` (tokenizer) and `search.index` (`l2n`, `topk`). Nothing outside `part2_poc/` is written.

## Choices

- **Queries:** the 100 queries are Part 3 synthetic test-split queries, 25 each with no filter, a CPC prefix, keywords, or both. The filters are drawn from the source patent: its subclass or main group, and 1–2 words of its title.
- **Ties:** results are ordered by score, then by patent position. The dataset has duplicate claims across patents, and without this tie-break the merge order would depend on the sharding.
- **Defaults:** k=5, B=10.
