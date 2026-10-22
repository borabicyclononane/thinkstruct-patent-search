# Patent claim search

Search over patent claims that walks the CPC classification tree (beam search) instead of scoring every claim. It has keyword, CPC and title filters, and a small study on training embeddings for this kind of search.

All results and tables are in [results/README.md](results/README.md).

The written side of the part 2 system design question is in part_2_system_design.md. 

AI Disclaimer: AI was used quite liberally for all parts of the project (no lines of code are hand-written). Core design choices and experiments are human-made (with AI occasionally used to brainstorm ideas) and all markdown files are human-reviewed and edited. I take responsibility for the contents of this repo.

## Design choices

### Problem statement

- **Input:** an independent claim, which can be a real claim from the corpus or any claim-like text, plus optional filters: CPC prefixes, keywords in the title or abstract, and an exact title.
- **Output:** the most similar independent claims in the corpus, grouped by patent.
- **Why a tree:** CPC codes already organise patents into a hierarchy: section → class → subclass → main group → subgroup. The search uses this hierarchy as an index, instead of scoring every claim (flat search). The question throughout is whether walking the tree is faster than flat search, and what it costs in accuracy.

### Data (`search/data.py`, `scraper/`)

- **The claims in the provided JSON are damaged.**
  - About 3,200 of 10.4k claims lost their first chunk (number and preamble, and for dependents the "of claim N" reference). Almost every independent claim is affected.
  - Cancelled placeholders ("1 - 15 . (canceled)") cover several claim slots.
  - So independence can't be read reliably from the provided text.
- **Claims are scraped from Google Patents instead** (`python -m scraper`):
  - Pages are the USPTO full-text XML rendered as HTML. Each claim's dependencies come from explicit `<claim-ref>` tags.
  - Some source XML lacks those tags, so references are also read from the text ("claim 3", "claims 1-4", "claims 1 or 2"). Broken references count too: "claim Error!", "any of the preceding claims", "according to claim ," with the number missing.
  - **Independent = cites no other claim** (the USPTO definition). "A vehicle wheel comprising the accessory of claim 1" is therefore dependent, even though it starts a new claim category.
  - The result is committed (`data/scraped/claims.jsonl`), so nobody has to scrape again. Pages are cached on disk and failures are retried with backoff. Pages are decoded as UTF-8 explicitly; without a charset header, requests decoded them as Latin-1.
- **Claim cleaning:** duplicated numbering, a leading claim number, and glued text ("inFIG.5", "unit100") are fixed with regexes. On the scraped claims there turned out to be nothing to fix.
- **Missing fields:** a patent is indexed only if it has claims and a classification code. Anything else may be missing. Dropped patents are counted in a report. In the default dataset all 640 patents qualify, with 1,191 independent claims (1.86 per patent) out of 10,411 claims in total.
- **CPC codes:**
  - Only the first classification code is used. No patent in the default dataset has more than one.
  - The default dataset's codes have no main group / subgroup separator ("B60B1900FI"). A heuristic splits them: 3 digits means a 1-digit main group, otherwise a 2-digit one. One exception is overridden by hand (B60D 1/167, not 11/67).
  - Large-dataset codes have the separator, so they are parsed directly.

### Embeddings (`search/embed.py`)

- **Model:** PatentSBERTa (`AI-Growth-Lab/PatentSBERTa`), a sentence-BERT model fine-tuned on patents, giving 768-d L2-normalised embeddings. It was chosen because it is trained on patent text and the CPC hierarchy, and it is small enough to run on my mac.
- **Only independent claims are embedded.** Abstracts and descriptions are not, which is a known limitation (see `part_2_system_design.md`).
- **Cache:** embeddings are stored on disk, keyed by the SHA-1 of the text, so reruns, the evaluation scripts and the viewer never re-embed. The large dataset has its own cache file.
- Embedding a query costs about 36 ms on CPU, the same for every search method, so all reported latencies start from the query's embedding.

### Index and search (`search/tree.py`, `search/index.py`)

- **Tree:** root → section → class → subclass → main group → subgroup → patent → independent claim.
  - Nodes are integer ids, and children are int arrays, so a whole beam's children are scored with one matrix product.
  - Patents get **positions** in depth-first order, so every node covers a contiguous range [start, end) of positions. This is what makes filtering cheap (below).
- **Node keys:** a claim's key is its embedding. Every internal node's key is the L2-normalised mean of its children's keys, built bottom-up. Similarity is cosine (a dot product of normalised vectors).
- **Beam search:**
  - Start at the root, or at the CPC filter's nodes. At each level, score the children of every node in the beam and keep the B best.
  - A node is ranked on its own score; its parent's score is not added.
  - Patents whose CPC path is shallower reach patent level early. They are carried along and compete with the next level's nodes on their own score.
  - When the beam holds only patents, score all independent claims of those B patents and return them ranked.
  - Cost grows with depth × B × fan-out, not with corpus size.
- **Flat search** (the baseline): score every claim with one matrix product, then rank.
  - It uses a partial top-k (`argpartition`), so only claims that can reach the top-k patents get sorted.
  - It is "flat top-B patents" when compared with beam search: it keeps all claims of its B best patents (ranked by best claim). Both methods then return the same number of patents.
- **Swappable pieces:** the index takes `score`, `aggregate`, `project`, `project_query` and `combine` as plain functions. Part 3's towers plug in as `project` (keys) and `project_query` (queries), with no other change. A query map may return one vector per tree level (per-level towers); each node is then scored with its level's vector.

### Hybrid search (`search/filters.py`, `search/hybrid_eval.py`)

- **Filters:**
  - CPC prefixes become start nodes for beam search. Several prefixes are a union of start nodes, minus any node already under another.
  - Keywords and titles become a sorted array of candidate patent positions, from an inverted index (field → term → sorted positions; normalised title → positions).
  - Different filter types are ANDed. `all_of` keywords are ANDed and `any_of` keywords are ORed.
- **Tokens:**
  - Lowercased runs of letters and digits. Plurals are folded (tires → tire, assemblies → assembly, branches → branch).
  - A hyphenated term is indexed whole and as its parts. "non-pneumatic" then matches only the whole term, while "pneumatic" also matches it.
  - A multi-word keyword needs all its words in the same field; it is not a phrase search.
  - A title match is exact after normalisation (case, punctuation and whitespace ignored).
- **Set operations on sorted arrays:**
  - AND intersects the smallest list first, so no intermediate result is larger than the rarest term's list.
  - OR marks positions in a boolean array and reads them back in order. This was about 40x faster than `np.unique` (a sort) on two lists of ~3.4k positions.
- **Beam pruning:** a child is scored only if its position range contains a candidate. That is two binary searches into the sorted candidate array (`searchsorted`), so pruning costs O(log n) per node, whatever the filter.
- **Flat search with filters,** two versions:
  - **post-filter:** score every claim, then drop non-candidates.
  - **pre-filter:** gather only the candidates' claims (contiguous slices, because leaves follow positions) and score those.
- **Correctness checks,** which stop the script if any fails:
  - pre-filter = post-filter (full ranking);
  - every patent beam search returns satisfies the filter, re-checked from the raw text rather than the inverted index;
  - beam search with unlimited width = flat pre-filter.

### Large dataset (`experiment-big-data/`)

- **Sampling:** batch files (1,000 patents each, all CPC sections) are read from the zip in a seeded random order until 10^4 patents are kept. A random set of batches stands in for a random sample of patents. Nothing is extracted to disk.
- **Claims there are clean,** so they are not scraped. Independence uses the scraper's text rule on the dataset's own text. This was checked against Google Patents on 200 random patents (3,394 claims): 100% agreement, and all 456 of Google's independent claims were found in the sample (`results_10000/independence_check.md`).
  - Known unfixed bug: claims numbered "45)" instead of "45." get positional numbers and keep the prefix. This affects 26 of 24,466 independent claims.
- **Dropped:** classification "N/A" (about 40% of patents read), no live independent claim, duplicate doc numbers, cancelled stubs and empty claims.
- **Split:** 1,000 random patents are held out as queries (their first independent claim), against an index of the other 9,000 (22,036 claims). Hybrid timing uses 300 of those queries, with filters chosen for a corpus that spans all sections.
- **Embedding** the 10^4 sample took about 15 minutes on an M3 CPU. For 10^5, a remote-GPU path exists (`gpu_embed.py`): export the texts, embed on Colab, and merge into the local cache.
- **Only speed is measured here.** There are no synthetic queries for this sample, so accuracy is evaluated on the default dataset only.

### Part 1 evaluation (`results/compare_part1.py`)

- **Queries:** the 357 Part 3 test-split synthetic queries (119 per task), against one index over all 640 patents. A query's source patent must be in the index, since finding it is the test.
- **Metric:** the share of queries whose source claim is among the returned claims. A combination counts only if both of its source claims come back. It was chosen over CPC-agreement metrics (share of results in the query's subclass, kNN label, nDCG over CPC depth) originally used in my first iteration as it more directly addresses my stated problem.
- **Compared at equal budget:** beam with width B against flat top-B patents, for B = 1, 5, 10, 100.
- **Significance:** an exact two-sided sign test on the paired per-query outcomes (queries only beam finds against queries only flat finds).
- **Latency:** the median per query over 3 passes, after a warm-up query. It is timed on the same queries for the default dataset, and on the held-out first claims for 10^4.

### Part 2: system design and proof of concept (`part_2_system_design.md`, `part2_poc/`)

- **Design:**
  - A coordinator holds the CPC tree (small) and routes the beam. Shards hold claim embeddings and Roaring bitmaps (one per keyword) for a contiguous range of positions.
  - Because node ranges are contiguous, a CPC filter maps to the few shards overlapping it. The final B patents live on at most B shards.
- **Proof of concept:** the same design in-process on the 640 patents.
  - The coordinator stores node keys as sum + count of claim embeddings, so keys can be updated incrementally.
  - Shards answer one batched `count_in_ranges` call per tree level: matching patents per node range, clipped to the shard, from `range_cardinality` on the bitmap. "and" counts patents that have every term. The looser "every term somewhere in the range" can return fewer than k results.
  - Shards also provide `exact_match` and `score`, and count their calls.
- **Checked on 100 synthetic test queries,** across no filter, CPC, keyword and both:
  - results at S = 2 and 4 equal S = 1;
  - per-node counts summed over shards equal S = 1 and brute force;
  - every result satisfies its filters.
- **Hash-partitioned baseline:** the same queries, for comparing the shards touched.
- **Ties** are broken by position, so the merge order doesn't depend on the partitioning. Two patents in the dataset share an identical claim.
- **Small-scale only:** this shows correctness and fan-out on 640 patents. Scaling is argued in the design doc, not measured.

### Part 3: synthetic queries (`synthetic/generate_synthetic.py`)

- **Why synthetic:** there are no labelled query → relevant-claim pairs. Claude generates queries from real claims, so each query's answer is known: the claim(s) it came from. Synthetic text is only ever a query; the index holds only real claims.
- **Three tasks,** from easy to hard:
  - **paraphrase:** the same claim scope in different words. One per independent claim.
  - **element reuse:** a new invention built around 1–2 key parts of the claim, like an infringement-style overlap. One per claim.
  - **combination:** 1–2 key parts from each of two claims from different patents in the same split, combined into one claim. As many per split as the split has claims.
- **Model and settings:**
  - Claude Haiku 5.5 through Message Batches (half price), with structured JSON outputs (schemas for elements, chosen ids and mix).
  - Paraphrase runs without thinking. In the pilot, thinking used up all of `max_tokens` and returned empty text. The other two tasks use adaptive thinking at medium effort.
  - `max_tokens` scales with the source length.
- **Cost control:**
  - A worst-case cost is computed before every run, and the run aborts if it exceeds what's left of a $15 budget. Actual spend is logged.
  - Responses are cached, keyed by dataset, claim ids, prompt version, model, attempt, thinking settings and the full prompt, so nothing is requested twice.
  - The full run cost about $1.50.
- **Quality:**
  - Prompts are versioned and were iterated on pilots (paraphrase v1, element reuse v3, combination v4, shared rules v2). Every query must be a complete claim (preamble, "comprising:", elements, period).
  - No query is rejected for its content. Checks are recorded as flags only: claim form, element-id consistency, length ratio, ROUGE-L / Jaccard against the source for paraphrases, and near-duplicates (token Jaccard ≥ 0.9). Only calls that produced no usable claim are failures.
  - A threshold for "paraphrase too close to the source" was never chosen.
- **Split:** by patent, 80/10/10, seed 0 (`outputs/split.json`), so a test query's source patent is never seen in training. Result: 3,573 queries (1,191 per task); test has 119 per task.
- **Committed:** the queries (`outputs/synthetic/`), the split, and the embedding bundle (`training/bundles/full.npz`: vectors, ids, splits and answers, no text).

### Part 3: training (`training/`)

- **What is trained:** a query tower and a key tower (768 → 64) on top of **frozen** PatentSBERTa embeddings. The backbone is never touched, so training takes seconds on a CPU and the 768-d cache is reused.
  - Each tower computes `normalize(W x + MLP(x))`. W is a linear map, and the MLP is 768 → 256 → 64 with GELU and dropout 0.1.
  - Both towers start from the same random orthonormal W, with the MLP's last layer at zero. At initialisation, query·key ≈ the raw cosine (a random projection roughly preserves angles), so training starts from a sensible point. The towers are free to diverge from the first step.
  - 64-d keys are 12x smaller than 768-d ones, which matters for memory at 10^7 patents.
- **Claim loss:** InfoNCE over 64-d cosines (temperature 0.05). Each query is scored against:
  - the batch's positives;
  - one mined hard negative per pair (the nearest non-source training claim by raw cosine);
  - 256 random training claims.
  - A query's other true source (combinations) is masked, not treated as a negative.
- **Routing loss** (`--routing-weights 0,1,3`): beam search fails when the query takes a wrong turn at a high level, which a claim-level loss doesn't directly train. Every step, 64-d keys for all tree nodes are rebuilt from the training claims' keys exactly as the index builds them. At each level (patent, subgroup, main group, subclass), a cross-entropy asks the query to score its source's ancestor above every other node of that level. The other source's ancestor in a combination is masked.
- **Per-level query towers:** a separate query tower for each level the beam scores (claim, patent, subgroup, main group, subclass). Section and class have a single node and use the subclass tower. They need the routing loss, so they are trained only with routing weight > 0.
- **No leakage:**
  - Only train-split claims are keys during training.
  - 10% of train patents are held out as a dev set for picking the best epoch (dev score = mean share of sources returned by beam search at B = 5, 10, 30).
  - PCA-64 is fit on the same training claims.
- **Grid and selection:**
  - 3 learning rates × routing weight {0, 1, 3} × {shared, per-level}, giving 15 configurations of 20 epochs each, AdamW (weight decay 1e-4), batch 32.
  - `results/compare_part3.py` takes the best configuration of each family on **val** at B=5: shared towers without routing loss, shared towers, and per-level towers. It then reports **test**, which no choice used.
- **Baselines:** raw 768-d, PCA-64, and flat search with raw 768-d as a reference.
- **Evaluation:** the same metric as Part 1 (source returned, combinations need both), on a tree of all 1,191 claims, at B = 5 and B = 30. It reports per-task numbers, a paired sign test against raw 768-d beam search, latency and training time.

### Reproducibility

- **Commands:** `make setup`, `make part1`, `make large`, `make part3` and `make readme` regenerate every number in `results/README.md`. Each target runs only what those tables need.
- **Pinned environment:** Python 3.14, with versions pinned in `requirements.txt`. Seeds are fixed for splits, pairing, sampling and training.
- **Committed so reruns are cheap:** scraped claims, synthetic queries, the split, the embedding bundle, and the training runs' metrics.
- **Not committed:** model checkpoints (`towers.pt`), which `make part3` retrains in about 6 minutes; the task data itself; and API keys (`.env` is gitignored).

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
make part1    # beam vs flat, hybrid search                      ~1 min
make large    # speed on a 10^4 sample of the large dataset      ~20 min (mostly embedding all the samples)
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

## Viewer

Run `make viewer` and open http://localhost:8501 to search, browse patents, inspect the synthetic queries and read the results; see `viewer/README.md` for details.
