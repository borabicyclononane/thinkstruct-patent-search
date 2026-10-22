# Results

All numbers come from the commands below, run on one Apple M3 CPU (numpy 2.5, torch 2.14).
Latency is the median per query, measured from the query's 768-d embedding.
Embedding the query text with PatentSBERTa adds about 36 ms, the same for every method.

```
make part1   # default dataset: Part 1 + 1b
make large   # 10^4 sample: Part 1 + 1b
make part3   # train and evaluate the towers
make readme  # this file
```

## Part 1: is beam search better than flat search?

**Queries.** Each query is a held-out patent's first independent claim.
- **Default dataset:** 640 vehicle patents, 5-fold, so every patent is a query once against the other 512 patents.
- **10^4 sample:** 1,000 patents from the large dataset, each queried against the other 9,000 (all CPC sections).

**What is compared.** Beam search with width B returns B patents. Flat search scores every claim and keeps its top B patents.

**Accuracy metrics,** computed on those B patents:
- **agree (subclass / main group):** share of the B patents with the query's CPC code down to that level.
- **kNN (subclass / main group):** the B patents' majority label at that level equals the query's.
- **nDCG:** nDCG@B, where a patent's relevance is the depth of the CPC prefix it shares with the query.

### Speed (ms per query)

"flat ÷ beam" above 1 means beam search is faster.

| B   | default: flat | default: beam | flat ÷ beam | 10^4: flat | 10^4: beam | flat ÷ beam |
| --- | ------------: | ------------: | ----------: | ---------: | ---------: | ----------: |
| 1   |         0.026 |         0.065 |        0.4x |      1.054 |      0.071 |       14.9x |
| 5   |         0.038 |         0.087 |        0.4x |      1.206 |      0.133 |        9.0x |
| 10  |         0.046 |         0.096 |        0.5x |      1.265 |      0.200 |        6.3x |
| 100 |         0.109 |         0.245 |        0.4x |      1.345 |      1.227 |        1.1x |

### Accuracy: default dataset

| B   | search      | agree subclass | agree main group | kNN subclass | kNN main group |   nDCG |
| --- | ----------- | -------------: | ---------------: | -----------: | -------------: | -----: |
| 1   | flat        |          0.941 |            0.620 |        0.941 |          0.620 |  0.805 |
|     | beam        |          0.914 |            0.569 |        0.914 |          0.569 |  0.780 |
|     | beam − flat |         -0.027 |           -0.052 |       -0.027 |         -0.052 | -0.024 |
| 5   | flat        |          0.927 |            0.506 |        0.942 |          0.617 |  0.789 |
|     | beam        |          0.945 |            0.560 |        0.950 |          0.634 |  0.806 |
|     | beam − flat |         +0.018 |           +0.054 |       +0.008 |         +0.017 | +0.017 |
| 10  | flat        |          0.911 |            0.448 |        0.945 |          0.617 |  0.789 |
|     | beam        |          0.929 |            0.480 |        0.942 |          0.616 |  0.800 |
|     | beam − flat |         +0.018 |           +0.032 |       -0.003 |         -0.002 | +0.011 |
| 100 | flat        |          0.832 |            0.203 |        0.903 |          0.384 |  0.887 |
|     | beam        |          0.834 |            0.203 |        0.903 |          0.388 |  0.888 |
|     | beam − flat |         +0.003 |           +0.000 |       +0.000 |         +0.003 | +0.001 |

### Accuracy: 10^4 sample

| B   | search      | agree subclass | agree main group | kNN subclass | kNN main group |   nDCG |
| --- | ----------- | -------------: | ---------------: | -----------: | -------------: | -----: |
| 1   | flat        |          0.371 |            0.197 |        0.371 |          0.197 |  0.402 |
|     | beam        |          0.178 |            0.075 |        0.178 |          0.075 |  0.242 |
|     | beam − flat |         -0.193 |           -0.122 |       -0.193 |         -0.122 | -0.160 |
| 5   | flat        |          0.334 |            0.156 |        0.431 |          0.226 |  0.399 |
|     | beam        |          0.362 |            0.173 |        0.404 |          0.216 |  0.434 |
|     | beam − flat |         +0.028 |           +0.017 |       -0.027 |         -0.010 | +0.035 |
| 10  | flat        |          0.297 |            0.132 |        0.453 |          0.240 |  0.387 |
|     | beam        |          0.346 |            0.156 |        0.428 |          0.227 |  0.437 |
|     | beam − flat |         +0.049 |           +0.025 |       -0.025 |         -0.013 | +0.049 |
| 100 | flat        |          0.181 |            0.059 |        0.352 |          0.200 |  0.361 |
|     | beam        |          0.197 |            0.065 |        0.344 |          0.191 |  0.383 |
|     | beam − flat |         +0.016 |           +0.006 |       -0.008 |         -0.009 | +0.022 |

Section- and class-level metrics are in the CSVs. They are always 1.000 on the default dataset, because every patent there is B60.

## Part 1b: hybrid search speed

**Setup.**
- **Default dataset:** every patent's first claim is a query against all 640 patents, its own included.
- **10^4 sample:** 300 held-out queries against the 9,000 indexed patents.
- Flat search keeps the top 10 patents. Beam search is timed at B=1, 5 and 10.

**Methods compared** (ms per query; resolving the filter is included):
- **flat post-filter:** score every claim, then drop the non-candidates.
- **flat pre-filter:** score only the candidates' claims.
- **beam:** start at the filter's CPC nodes and score only children that contain a candidate.

**Checks** (the script stops if any fails):
- pre- and post-filtering return the same ranking;
- beam search returns only patents that pass the filter;
- beam search with unlimited width returns exactly the flat result.

The "no filter" row shows plain flat search and plain beam search.

### Default dataset (640 patents)

| filter                    | candidate patents | flat post-filter | flat pre-filter | beam B=1 | beam B=5 | beam B=10 |
| ------------------------- | ----------------: | ---------------: | --------------: | -------: | -------: | --------: |
| no filter                 |               640 |            0.052 |                 |    0.066 |    0.087 |     0.099 |
| CPC subclass B60C         |               318 |            0.054 |           0.130 |    0.046 |    0.068 |     0.083 |
| CPC subgroup B60B 27/00   |                34 |            0.046 |           0.041 |    0.031 |    0.036 |     0.038 |
| CPC B60B 33 or B60D       |                59 |            0.049 |           0.049 |    0.038 |    0.048 |     0.061 |
| keyword: tire             |               318 |            0.052 |           0.130 |    0.082 |    0.103 |     0.117 |
| keyword: hub              |               138 |            0.051 |           0.064 |    0.079 |    0.091 |     0.110 |
| keyword: non-pneumatic    |                32 |            0.045 |           0.043 |    0.074 |    0.088 |     0.093 |
| keyword: hitch            |                 8 |            0.037 |           0.026 |    0.074 |    0.077 |     0.077 |
| tire AND pressure         |                28 |            0.057 |           0.054 |    0.088 |    0.099 |     0.104 |
| stud OR hitch             |                12 |            0.041 |           0.031 |    0.077 |    0.088 |     0.090 |
| title = 'pneumatic tire'  |                33 |            0.036 |           0.030 |    0.073 |    0.083 |     0.088 |
| B60C 11 + tread AND noise |                 3 |            0.050 |           0.038 |    0.054 |    0.056 |     0.057 |

### 10^4 sample (9,000 patents)

| filter                    | candidate patents | flat post-filter | flat pre-filter | beam B=1 | beam B=5 | beam B=10 |
| ------------------------- | ----------------: | ---------------: | --------------: | -------: | -------: | --------: |
| no filter                 |              9000 |            1.243 |                 |    0.073 |    0.162 |     0.228 |
| CPC section H             |              2465 |            1.214 |           2.277 |    0.064 |    0.102 |     0.137 |
| CPC subclass G06F         |               804 |            1.145 |           0.495 |    0.060 |    0.112 |     0.101 |
| CPC main group G06F 3     |               206 |            1.114 |           0.181 |    0.048 |    0.066 |     0.076 |
| CPC subgroup G02B 27/0172 |                12 |            1.096 |           0.028 |    0.027 |    0.030 |     0.031 |
| CPC A61K or C07D          |               413 |            1.117 |           0.248 |    0.057 |    0.099 |     0.124 |
| keyword: method           |              4694 |            1.370 |           4.730 |    0.098 |    0.151 |     0.193 |
| keyword: vehicle          |               485 |            1.118 |           0.249 |    0.084 |    0.114 |     0.133 |
| keyword: antibody         |                67 |            1.098 |           0.057 |    0.079 |    0.096 |     0.106 |
| keyword: crispr           |                 5 |            1.093 |           0.029 |    0.075 |    0.084 |     0.084 |
| wireless AND battery      |                 9 |            1.166 |           0.046 |    0.094 |    0.107 |     0.111 |
| lidar OR radar            |                34 |            1.114 |           0.055 |    0.084 |    0.102 |     0.112 |
| title = 'display device'  |                24 |            1.100 |           0.038 |    0.074 |    0.091 |     0.098 |
| G06F + neural AND network |                 4 |            1.165 |           0.053 |    0.072 |    0.077 |     0.077 |

## Part 3: training and evaluation

**Queries.** 3,573 synthetic queries made with Claude Haiku 5.5 from the 1,191 independent claims of the default dataset (`synthetic/`):
- **paraphrase:** the claim reworded.
- **element reuse:** a new claim that reuses 1–2 parts of it.
- **combination:** a new claim mixing two claims.

**Split.** Train / val / test by patent, 80/10/10. Test has 119 queries per dataset.

**Metric.** The share of queries whose source claim is among the claims beam search returns at **B=5**, over a tree of all 1,191 claims. For a combination, either source counts.

**Methods.** All are on frozen PatentSBERTa embeddings.
- **raw 768-d** and **PCA-64:** PCA is fit on the training claims.
- **Towers:** a 768 → 64 query tower and key tower, trained with InfoNCE. "Routing loss" adds a cross-entropy at each CPC level, so the query scores its source's ancestors above the other nodes. **Per-level towers** use a separate query tower for each level of the tree.
- **Tower configuration:** each tower row uses the grid configuration with the best **val** score at B=5. The table therefore reports **test**, which was not used for any choice.
- **all vs raw 768:** test queries where the method finds the source and raw 768-d doesn't, then the reverse, with an exact two-sided sign test.
- **Training:** wall time for 20 epochs on CPU, including a beam-search evaluation on the dev set after every epoch.

| method                         | config            | paraphrase | element reuse | combination |   all |     all vs raw 768 | val all | ms / query | training |
| ------------------------------ | ----------------- | ---------: | ------------: | ----------: | ----: | -----------------: | ------: | ---------: | -------: |
| raw 768-d                      | —                 |      0.622 |         0.311 |       0.924 | 0.619 |                  — |   0.595 |      0.094 |        — |
| PCA-64                         | —                 |      0.496 |         0.134 |       0.748 | 0.459 |  +7 / −64, p<0.001 |   0.468 |      0.074 |        — |
| shared towers, no routing loss | routing=0 lr=1e-4 |      0.479 |         0.412 |       0.681 | 0.524 | +36 / −70, p=0.001 |   0.532 |      0.101 |      8 s |
| shared towers                  | routing=3 lr=1e-3 |      0.714 |         0.412 |       0.815 | 0.647 | +51 / −41, p=0.348 |   0.625 |      0.102 |     14 s |
| per-level towers               | routing=1 lr=3e-4 |      0.605 |         0.445 |       0.807 | 0.619 | +48 / −48, p=1.000 |   0.653 |      0.358 |     26 s |

Val results and per-dataset sign tests are in `part3.csv`. The full training grid is in `../training/runs/final/`.
