# Part 2 PoC: shards touched per query

Mean shards contacted per query over 100 synthetic test-split queries (25 per type), k=5, B=10, on the 640-patent default dataset. Routing: `count_in_ranges` calls during beam search (only with keywords). Scoring: `exact_match` / `score` on the owners of the final B patents. Total: distinct shards.

| query type | S | range routing | range scoring | range total | hash routing | hash scoring | hash total |
| ---------- | - | ------------: | ------------: | ----------: | -----------: | -----------: | ---------: |
| no filter  | 2 |          0.00 |          1.48 |        1.48 |         0.00 |         1.96 |       1.96 |
| no filter  | 4 |          0.00 |          2.32 |        2.32 |         0.00 |         3.68 |       3.68 |
| CPC prefix | 2 |          0.00 |          1.00 |        1.00 |         0.00 |         2.00 |       2.00 |
| CPC prefix | 4 |          0.00 |          1.44 |        1.44 |         0.00 |         3.68 |       3.68 |
| keyword    | 2 |          2.00 |          1.52 |        2.00 |         2.00 |         1.76 |       2.00 |
| keyword    | 4 |          4.00 |          2.24 |        4.00 |         4.00 |         3.08 |       4.00 |
| both       | 2 |          1.24 |          1.12 |        1.24 |         2.00 |         1.60 |       2.00 |
| both       | 4 |          1.80 |          1.48 |        1.80 |         4.00 |         2.64 |       4.00 |
