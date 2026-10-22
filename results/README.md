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

**What is compared.** Beam search with width B returns all claims of B patents. Flat search scores every claim and keeps all claims of its top B patents.

**Accuracy** (default dataset only). The queries are the 357 Part 3 test-split synthetic queries (119 per task), searched against one index over all 640 patents. The metric is the share of queries whose source claim is among the returned claims; a combination counts only if **both** of its source claims come back. "beam vs flat" gives the queries only beam finds and the queries only flat finds, with an exact two-sided sign test. Beam B=5 equals the raw 768-d row of Part 3.

**Speed.**
- **Default dataset:** the same 357 queries.
- **10^4 sample:** 1,000 held-out patents' first independent claims, searched against the other 9,000 patents (all CPC sections). Accuracy isn't measured here, because there are no synthetic queries for this sample.

### Accuracy: default dataset

| B   | search | paraphrase | element reuse | combination |   all |  all: beam vs flat |
| --- | ------ | ---------: | ------------: | ----------: | ----: | -----------------: |
| 1   | flat   |      0.437 |         0.244 |       0.000 | 0.227 |                  — |
|     | beam   |      0.244 |         0.084 |       0.000 | 0.109 |  +8 / −50, p<0.001 |
| 5   | flat   |      0.731 |         0.445 |       0.361 | 0.513 |                  — |
|     | beam   |      0.622 |         0.311 |       0.227 | 0.387 | +10 / −55, p<0.001 |
| 10  | flat   |      0.782 |         0.529 |       0.429 | 0.580 |                  — |
|     | beam   |      0.723 |         0.454 |       0.328 | 0.501 | +10 / −38, p<0.001 |
| 100 | flat   |      0.975 |         0.866 |       0.824 | 0.888 |                  — |
|     | beam   |      0.966 |         0.874 |       0.798 | 0.880 |   +2 / −5, p=0.453 |

### Speed (ms per query)

"flat ÷ beam" above 1 means beam search is faster.

| B   | default: flat | default: beam | flat ÷ beam | 10^4: flat | 10^4: beam | flat ÷ beam |
| --- | ------------: | ------------: | ----------: | ---------: | ---------: | ----------: |
| 1   |         0.029 |         0.065 |        0.4x |      1.045 |      0.071 |       14.7x |
| 5   |         0.043 |         0.094 |        0.5x |      1.209 |      0.127 |        9.5x |
| 10  |         0.048 |         0.100 |        0.5x |      1.300 |      0.170 |        7.6x |
| 100 |         0.117 |         0.278 |        0.4x |      1.339 |      1.172 |        1.1x |

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
| no filter                 |               640 |            0.052 |                 |    0.066 |    0.086 |     0.097 |
| CPC subclass B60C         |               318 |            0.054 |           0.124 |    0.045 |    0.061 |     0.072 |
| CPC subgroup B60B 27/00   |                34 |            0.044 |           0.042 |    0.031 |    0.036 |     0.038 |
| CPC B60B 33 or B60D       |                59 |            0.047 |           0.049 |    0.039 |    0.049 |     0.060 |
| keyword: tire             |               318 |            0.053 |           0.087 |    0.082 |    0.095 |     0.107 |
| keyword: hub              |               138 |            0.050 |           0.063 |    0.077 |    0.091 |     0.102 |
| keyword: non-pneumatic    |                32 |            0.045 |           0.043 |    0.075 |    0.090 |     0.094 |
| keyword: hitch            |                 8 |            0.037 |           0.026 |    0.074 |    0.077 |     0.077 |
| tire AND pressure         |                28 |            0.057 |           0.054 |    0.087 |    0.099 |     0.105 |
| stud OR hitch             |                12 |            0.042 |           0.031 |    0.078 |    0.086 |     0.091 |
| title = 'pneumatic tire'  |                33 |            0.036 |           0.029 |    0.072 |    0.082 |     0.087 |
| B60C 11 + tread AND noise |                 3 |            0.051 |           0.038 |    0.055 |    0.057 |     0.057 |

### 10^4 sample (9,000 patents)

| filter                    | candidate patents | flat post-filter | flat pre-filter | beam B=1 | beam B=5 | beam B=10 |
| ------------------------- | ----------------: | ---------------: | --------------: | -------: | -------: | --------: |
| no filter                 |              9000 |            1.212 |                 |    0.071 |    0.123 |     0.173 |
| CPC section H             |              2465 |            1.127 |           1.964 |    0.064 |    0.106 |     0.142 |
| CPC subclass G06F         |               804 |            1.112 |           0.431 |    0.049 |    0.075 |     0.101 |
| CPC main group G06F 3     |               206 |            1.085 |           0.154 |    0.047 |    0.054 |     0.060 |
| CPC subgroup G02B 27/0172 |                12 |            1.070 |           0.028 |    0.028 |    0.030 |     0.032 |
| CPC A61K or C07D          |               413 |            1.091 |           0.236 |    0.054 |    0.076 |     0.120 |
| keyword: method           |              4694 |            1.324 |           4.416 |    0.095 |    0.143 |     0.179 |
| keyword: vehicle          |               485 |            1.098 |           0.257 |    0.083 |    0.120 |     0.133 |
| keyword: antibody         |                67 |            1.083 |           0.083 |    0.079 |    0.094 |     0.103 |
| keyword: crispr           |                 5 |            1.072 |           0.028 |    0.074 |    0.082 |     0.082 |
| wireless AND battery      |                 9 |            1.093 |           0.046 |    0.093 |    0.107 |     0.111 |
| lidar OR radar            |                34 |            1.098 |           0.071 |    0.085 |    0.105 |     0.136 |
| title = 'display device'  |                24 |            1.083 |           0.042 |    0.072 |    0.090 |     0.097 |
| G06F + neural AND network |                 4 |            1.100 |           0.051 |    0.073 |    0.077 |     0.077 |

## Part 3: training and evaluation

**Queries.** 3,573 synthetic queries made with Claude Haiku 5.5 from the 1,191 independent claims of the default dataset (`synthetic/`):
- **paraphrase:** the claim reworded.
- **element reuse:** a new claim that reuses 1–2 parts of it.
- **combination:** a new claim mixing two claims.

**Split.** Train / val / test by patent, 80/10/10. Test has 119 queries per dataset.

**Metric.** The share of queries whose source claim is among the claims the search returns at **B=5** and **B=30**, over a tree of all 1,191 claims. A combination counts only if **both** of its source claims come back.

**Methods.** All are on frozen PatentSBERTa embeddings.
- **flat search, raw 768-d:** a reference, not beam search. It scores every claim and keeps all claims of its top B patents, as in Part 1.
- **raw 768-d** and **PCA-64:** PCA is fit on the training claims.
- **Towers:** a 768 → 64 query tower and key tower, trained with InfoNCE. "Routing loss" adds a cross-entropy at each CPC level, so the query scores its source's ancestors above the other nodes. **Per-level towers** use a separate query tower for each level of the tree.
- **Tower configuration:** each tower row uses the grid configuration with the best **val** score at B=5. The tables therefore report **test**, which was not used for any choice.
- **all vs raw 768:** test queries where the method finds the source and raw 768-d beam search doesn't, then the reverse, with an exact two-sided sign test.
- **Training:** wall time for 20 epochs on CPU, including a beam-search evaluation on the dev set after every epoch.

**B=5**

| method                         | config            | paraphrase | element reuse | combination |   all |     all vs raw 768 | val all | ms / query | training |
| ------------------------------ | ----------------- | ---------: | ------------: | ----------: | ----: | -----------------: | ------: | ---------: | -------: |
| flat search, raw 768-d         | —                 |      0.731 |         0.445 |       0.361 | 0.513 | +55 / −10, p<0.001 |   0.479 |      0.050 |        — |
| raw 768-d                      | —                 |      0.622 |         0.311 |       0.227 | 0.387 |                  — |   0.355 |      0.090 |        — |
| PCA-64                         | —                 |      0.496 |         0.134 |       0.076 | 0.235 |  +6 / −60, p<0.001 |   0.242 |      0.077 |        — |
| shared towers, no routing loss | routing=0 lr=1e-4 |      0.513 |         0.378 |       0.101 | 0.331 | +47 / −67, p=0.075 |   0.303 |      0.104 |      8 s |
| shared towers                  | routing=1 lr=1e-4 |      0.647 |         0.479 |       0.160 | 0.429 | +62 / −47, p=0.180 |   0.391 |      0.105 |     14 s |
| per-level towers               | routing=1 lr=3e-4 |      0.613 |         0.471 |       0.202 | 0.429 | +58 / −43, p=0.163 |   0.419 |      0.361 |     26 s |

**B=30** (same configurations, chosen at B=5)

| method                         | config            | paraphrase | element reuse | combination |   all |     all vs raw 768 | val all | ms / query | training |
| ------------------------------ | ----------------- | ---------: | ------------: | ----------: | ----: | -----------------: | ------: | ---------: | -------: |
| flat search, raw 768-d         | —                 |      0.874 |         0.714 |       0.613 | 0.734 | +14 / −12, p=0.845 |   0.747 |      0.079 |        — |
| raw 768-d                      | —                 |      0.891 |         0.681 |       0.613 | 0.728 |                  — |   0.727 |      0.150 |        — |
| PCA-64                         | —                 |      0.857 |         0.504 |       0.420 | 0.594 | +11 / −59, p<0.001 |   0.603 |      0.098 |        — |
| shared towers, no routing loss | routing=0 lr=1e-4 |      0.849 |         0.714 |       0.521 | 0.695 | +40 / −52, p=0.251 |   0.658 |      0.125 |      8 s |
| shared towers                  | routing=1 lr=1e-4 |      0.899 |         0.765 |       0.605 | 0.756 | +46 / −36, p=0.320 |   0.730 |      0.127 |     14 s |
| per-level towers               | routing=1 lr=3e-4 |      0.874 |         0.756 |       0.605 | 0.745 | +40 / −34, p=0.561 |   0.760 |      0.389 |     26 s |

Val results and per-dataset sign tests are in `part3.csv`. The full training grid is in `../training/runs/final/`.
