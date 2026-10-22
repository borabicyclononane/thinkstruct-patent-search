# System Design!




## General:

### Goal:

**Input:** Independent Claim (potentially synthetic claim), optional filters (CPC prefix, keywords in title/abstract, title)

**Output:** k potentially similar independent claims from our database

## System Components:

### Embedder:

- **Layer 1:**
  - Input: Natural Language Claim
  - Output: PatentSBERTa embedding of claim
- **(optional) Layer 2:**
  - Input: PatentSBERTa embedding
  - Output: lower dimensional key/query embedding
  - Separate query embedder and key embedder.
- Used offline (embed all claims in database) and online (embed each query)

### Coordinator (Tree Structure):

- **Nodes:** One for each point along the CPC hierarchy (e.g., B60B, B60C 3)
  - section -> class -> subclass -> main group -> subgroups -> patent -> independent claims (leaves)
  - Currently, we don’t embed the patent abstracts at all - this is a potential area of improvement.
- **Nodes store:**
  - Node key: mean of children's keys
  - Node range: claims numbered in depth-first order, so each node covers a contiguous range [start, end)
- **Key function of node: search!**
- **Search:**
  - Input: query embedding, beam width B, start node (root, or CPC filter node), list of valid children (if doing hybrid lookup)
  - Output: claims under the final B patents, ranked by our computed (query, key) similarity

### Hybrid Keyword Lookup Database:

- **Contents:** keyword -> sorted list of patent positions (inverted index), title -> positions
  - For ALL words present in any document abstract/title
- **Validity check:** Input: tree node + keyword filter, Output: whether the node's subtree contains a match (two binary searches)
- **Implementation:** Roaring bitmaps (one per keyword, over positions): compressed, fast AND/OR, and rank queries give range counts.

## Major Pipelines:

### Query:

- Input: claim + filters, Output: k similar independent claims
- Embed query -> set start node(s) -> beam search over valid children -> score claims in final B patents -> top k

### Update:

- Input: new patents to add
- Embed patents (as keys), move up through the tree, updating each of the nodes’ keys
- Retrain every n updates as performance degrades.

## Error/Status Handling:

- Status table per patent: stage (fetched / parsed / embedded / indexed), status (ok / failed / dead), last error, attempts
- Queries are idempotent, so retries are always safe
- Quality checks: no independent claims found, unparseable CPC code, NaN/zero embedding, duplicate patent
- Monitoring: patents per stage, max node fanout, query latency (p50/p95), shards touched per query, max leaf size

## Scaling:

### Sharding (for hybrid search database):

- Split positions into equal-size contiguous ranges, one per shard
- Each shard holds: claim embeddings (leaves) + Roaring bitmaps for its range
- Boundaries may cut through a subtree; each shard clips node ranges to its own block

### Coordinator(s):

- Holds the tree (~250k nodes, small) + node keys, runs beam search routing, merges shard results.
- Replicated, since its state is small and read-only between updates.

### Fan-out:

- Final B leaves live on at most B shards, so shards touched per query is bounded by B, not corpus size
- CPC filters map to a position range, so only overlapping shards are queried

### Throughput:

- Replicate shards + coordinator; hot shards (popular CPC areas) get more replicas.

### Offline:

- Embedding is embarrassingly parallel
- Rebuilds are per shard; a shard that grows too big is split in two (new boundary)

### Size:

- ~2x10^7 independent claims in 10^7 patents
- ~60 GB embeddings if 768 dim embeddings cached, (768-d fp32), 5 GB embeddings if 64d used.
- ~4 GB keyword index (~100 distinct words per patent abstract *10^7, fp32)

### Cost:

Serving (a few shards x replicas + coordinators) is the dominant cost

**Rough estimate (using AWS prices):**

- Method: size each component by memory -> cheapest instance that fits -> instances x $/hr x 730
- Shards: 4 shards x 2 replicas on r7g.large (16 GB), $625/mo
- Coordinators: 2 x r7g.large -> ~$156/mo
- Query embedder: 1 x g5.xlarge (A10G), 734/mo
- General Storage: ~1 TB S3 ~23/mo; Postgres ~$78/mo (general storage of detailed patents + metadata, not on the actual query path)

**Total:** ~$1600/mo; allows for high throughput and latency (replicas, GPU for embedding)

## Strengths, Weaknesses, and Scaling Challenges:

### Strengths:

- Beam search yields sublinear search time (proportional to depth * B * fanout)
- Cheap small updates (just add an extra leaf node and update keys)

### Weaknesses:

- Doesn’t encode patent abstract or in-depth information from the patent (best solved with rerankers and wide beam search)
- Cross-domain claim similarity is expected to be harder to track as patent diversity increases (note decent performance on ‘combination’ task)
- Only allows for exact hybrid search, not fuzzy search (would require a lot of merges of index lists)

### Challenges:

- Fanout is not necessarily even across all nodes
- CPC classification leaves in particular could have very large size (lots of patents in a similar area), which would greatly reduce the hierarchical benefit of the tree