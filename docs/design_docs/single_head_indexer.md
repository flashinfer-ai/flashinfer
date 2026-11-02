# Single-head paged MQA indexer

DeepSeek Sparse Attention (DSA) attends each query token to its Top-K keys. An
indexer gives the scores that pick these keys. The DSA indexer has many heads
(`fp8_paged_mqa_logits`, `fp4_paged_mqa_logits`). The kernels in
`flashinfer.attn_scores.single_head` serve an indexer with one query head and no
head weights: a score is the dot product of a query row and a key. The module
has four parts:

- decode scores: `single_head_paged_mqa_logits` scores 1 to 8 query rows of a
  request against all its keys;
- prefill selection: `single_head_paged_mqa_threshold_select` writes only the
  keys that score at or above a threshold for each query row;
- thresholds from a key sample: `gather_key_sample` and `thresholds_from_sample`;
- NVFP4 keys and queries: `quantize_write_index_qk_nvfp4` writes the keys to the
  paged cache and the queries to rows, in one launch.

A Top-K kernel then takes the final Top-K from the scores, or from the selected
keys. The selected keys feed token-sparse attention
([token_sparse_attention.md](token_sparse_attention.md)).

## Public API

All functions are in `flashinfer.attn_scores`.

| API | Purpose |
| --- | --- |
| `single_head_paged_mqa_logits(query, key_cache, block_table, context_lengths, *, requests, scores, query_scales=None, key_scales=None)` | Write the FP32 score of each query row with each key of its request. |
| `single_head_paged_mqa_threshold_select(query, key_cache, row_ends, tile_requests, block_table, *, thresholds, counters, candidates, max_context, top_k, query_scales=None, key_scales=None, arrivals=None, ranges=None)` | Keep the keys at or above the threshold of each query row, as (key, score) pairs. |
| `gather_key_sample(key_cache, block_table, offsets, sample_ends, *, sample_keys, samples_per_run, pages_per_run, key_scales=None, sample_scales=None)` | Copy a strided random sample of the keys of each request into dense rows. |
| `thresholds_from_sample(sample_scores, sample_top, *, thresholds, counters)` | Set the threshold of each query row from its sampled Top scores, and zero its counters. |
| `quantize_write_index_qk_nvfp4(key, slots, query, query_rows, *, key_cache, key_scales, packed_query, query_scales)` | Quantize BF16 or FP16 keys and queries to NVFP4: the keys at their cache slots, the queries at their rows. |

## Layout

A key has 128 values. The keys are in pages of 64 keys.

```text
BF16 query:        [rows, 128]
BF16 key_cache:    [pages, 64, 128]
NVFP4 query:       [rows, 64] uint8    E2M1 values, two in a byte
NVFP4 query_scales:[rows, 8] E4M3      a scale for each 16 values
NVFP4 key_cache:   [pages, 64, 64] uint8
NVFP4 key_scales:  [pages, 64, 8] E4M3
block_table:       [requests, blocks] int32   the pages of each request
```

The pages can be any multiple of 16 bytes apart. So `key_cache` and
`key_scales` can be views of one cache that holds the keys of a page and then
their scales.

### Decode scores

The query rows come in groups of 1 to 8 rows, such as a decode token and its
draft tokens. All rows of a group see the same keys.

```text
context_lengths:  [groups] int32            the number of keys of each group
requests:         [groups] int32            the block_table row of each group
scores:           [rows, max_context] fp32  output: scores[q, k] for k below the group length
```

The scores at or past the length of a group stay as they are.
`max_context` sizes the grid, and the block table must cover it.

### Selection

The query rows come in tiles of 128 rows. Each row has its own number of keys.

```text
row_ends:       [rows] int32                       row r sees the keys below row_ends[r]
tile_requests:  [tiles] int32                      the block_table row of each tile
thresholds:     [rows] fp32                        the threshold of each row
counters:       [rows, 2] int32                    candidates and ties of each row, zeroed first
candidates:     [rows, top_k + capacity, 2] int32  output: (key, score bits) pairs
arrivals:       [tiles] int32                      optional: zeroed once
ranges:         [2, rows] int32                    optional output: the slots of each row
```

A key that scores above the threshold of its row is a candidate. A key that
scores equal to the threshold, or NaN, is a tie. Candidate `n` goes to slot
`top_k + n`, and tie `n` goes to slot `top_k - 1 - n`. Each slot holds the key
and a score value that keeps the order of the scores. For example, with
`top_k = 4`, a threshold of 2.0, and a row whose 5 keys score
`[3.5, 1.0, 2.0, 5.0, 2.0]`:

```text
slots 4 and 5 (candidates): keys 0 and 3
slots 2 and 3 (ties):       keys 2 and 4
counters of the row:        [2, 2]   2 candidates, 2 ties
```

The order of the keys in their slots is not specified. A row fits when its
candidates fit the `capacity` slots, and its candidates and ties reach
`min(row_ends[r], top_k)`.
Then the exact Top-`top_k` keys of the row are among its slots. Here, 2 + 2
reaches 4, so the Top-4 keys are keys 3 and 0 and the two ties.

## How it works

### Decode scores

On SM10x, a CuTe-DSL kernel scores the keys. Each CTA streams a run of 128-key
tiles of one group:

- one warp loads the keys with TMA, through an L2 evict-first policy, because
  each key is read once;
- one warp issues the tcgen05 MMA, with the keys on its M (128) and the query
  rows on its N (8);
- four warps read the results from TMEM and store the scores.

The grid size comes from `max_context` and not from the lengths, so the call
runs without host synchronization. CTAs after the length of their group exit
at once. The groups share two CTAs for each SM. A group gets more CTAs when it
needs them, so that the pages of each CTA run fit in its shared memory.

On other GPUs, such as Hopper, a Triton kernel scores BF16 keys. NVFP4 keys need
SM10x.

### Selection

The same kernel family selects. The query rows are on the MMA's M (128), and
the keys on its N (128). Each CTA scores a run of key tiles for one query tile,
and appends the candidates and ties of each row to its slots.

With `arrivals` and `ranges`, and at least `top_k - 1` candidate slots, every
row fits. The last CTA of a query tile to finish repairs each row that does not
fit: it scores the keys of the tile again, and selects again at the
`min(row_ends[r], top_k)`-th largest score of the row, which a radix select
finds. So any threshold gives the same exact result. The threshold only sets
how often the repair runs.

### Thresholds from a sample

A good threshold makes the repair rare:

1. `gather_key_sample` copies a strided random sample of the keys of each
   request.
2. `single_head_paged_mqa_logits` scores the sample, as pages of 64 keys.
3. A Top-K kernel ranks the sampled scores of each row.
4. `thresholds_from_sample` sets the threshold of each row to the least of its
   sampled Top scores, and zeroes its counters. A -1 position or a NaN score
   counts as -inf.

### NVFP4 keys and queries

`quantize_write_index_qk_nvfp4` gives each 16 values of a key or query an E4M3
scale: their largest magnitude over 6, with a global scale of 1. It writes the
64 packed bytes and the 8 scales of a key at its slot `s`: page `s // 64`, row
`s % 64`, and those of a query at its row `r` of `packed_query` and
`query_scales`. It skips negative slots and rows. Keys and queries are
quantized the same way, so their scores stay comparable.

## Limits

- A key has 128 values, and a page has 64 keys.
- Decode takes 1 to 8 query rows for each request. Selection takes tiles of 128
  query rows.
- The block table must cover `max_context`.
- The CuTe-DSL kernels need CuTe DSL 4.7 or newer. The scorer and the select
  need SM10x. Off SM10x, `single_head_paged_mqa_logits` runs the Triton kernel,
  which takes only BF16 queries and keys, with contiguous rows.
- `gather_key_sample` takes sample rows of whole 128-key tiles.

## CUDA graphs

No kernel synchronizes with the host. Run each one once before capture, because
the CuTe-DSL kernels compile on first use. Queries, keys, pages, lengths and
thresholds can change between replays. Zero the `arrivals` of the select once:
every call leaves them at zero.

## Tracing

Each API records a trace template of the same name, with a reference.

## Validation

`tests/attn_scores/test_single_head.py` covers:

- decode scores: BF16, BF16 through the Triton kernel, NVFP4, and NVFP4 keys and
  scales in one cache; 1, 3 and 8 rows for each request; lengths that end on a
  page, in a key tile, at 0, and in runs of several CTAs. The values are exact in
  FP32, so the scores must equal an FP64 reference bit for bit. The pages are
  shuffled, block-table entries past a length are invalid, and the scores past a
  length must stay as they are. A CUDA graph replay then uses new lengths.
- selection: BF16 and NVFP4; `top_k` 64, 512 and 2048; with and without the
  repair; query and key lengths that end on a page or in a key tile, and tiles
  with runs of several CTAs, some short and some empty; a CUDA graph replay.
- `gather_key_sample`: BF16 and NVFP4, with pages in the first 2**31 bytes of
  the cache and after them. Slots after the sample end stay as they are.
- `thresholds_from_sample`: -1 positions and NaN scores.
- `quantize_write_index_qk_nvfp4`: 0 to 300 keys and 0 to 300 queries, at
  shuffled slots and rows, with negative ones skipped.
- the checks that reject bad shapes, bad layouts off SM10x, bad counters, and
  CuTe DSL older than 4.7.
