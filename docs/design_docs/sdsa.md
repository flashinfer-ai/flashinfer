# SDSA (Simplified DSA)

SDSA attends each query token to its Top-K keys. A one-head indexer picks the
keys, and token-sparse attention attends to them.

## Indexer

DeepSeek Sparse Attention (DSA) attends each query token to its Top-K keys. An
indexer gives the scores that pick these keys. The DSA indexer has many heads
(`fp8_paged_mqa_logits`, `fp4_paged_mqa_logits`). The SDSA (Simplified DSA) indexer
has one query head and no head weights: a score is the dot product of a
query row and a key. The APIs in `flashinfer.sdsa` serve this indexer. The
module has five parts:

- decode scores: `indexer_scores` scores 1 to 8 query rows of a request against
  all its keys;
- prefill selection: `indexer_threshold_select` writes only the keys that score
  at or above a threshold for each query row;
- Top-K from the selection: `top_k_from_candidates` writes the Top-K keys of
  each query row from its selected keys;
- thresholds from a key sample: `gather_key_sample` and `thresholds_from_sample`;
- NVFP4 keys and queries: `quantize_write_index_qk_nvfp4` writes the keys to the
  paged cache and the queries to rows, in one launch.

For decode, a Top-K kernel takes the Top-K from the scores. The Top-K keys feed
token-sparse attention (below).

### Public API

All functions are in `flashinfer.sdsa`.

| API | Purpose |
| --- | --- |
| `indexer_scores(query, key_cache, block_table, context_lengths, *, requests, scores, query_scales=None, key_scales=None)` | Write the FP32 score of each query row with each key of its request. |
| `indexer_threshold_select(query, key_cache, row_ends, tile_requests, block_table, *, thresholds, counters, candidates, max_context, top_k, query_scales=None, key_scales=None, arrivals=None, ranges=None)` | Keep the keys at or above the threshold of each query row, as (key, score) pairs. |
| `top_k_from_candidates(candidates, ranges, *, top_keys)` | Write the Top-K keys of each query row from the pairs in its range. |
| `gather_key_sample(key_cache, block_table, offsets, sample_ends, *, sample_keys, samples_per_run, pages_per_run, key_scales=None, sample_scales=None)` | Copy a strided random sample of the keys of each request into dense rows. |
| `thresholds_from_sample(sample_scores, sample_top, *, thresholds, counters)` | Set the threshold of each query row from its sampled Top scores, and zero its counters. |
| `quantize_write_index_qk_nvfp4(key, slots, query, query_rows, *, key_cache, key_scales, packed_query, query_scales)` | Quantize BF16 or FP16 keys and queries to NVFP4: the keys at their cache slots, the queries at their rows. |

### Layout

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

#### Decode scores

The query rows come in groups of 1 to 8 rows, such as a decode token and its
draft tokens. All rows of a group see the same keys.

```text
context_lengths:  [groups] int32            the number of keys of each group
requests:         [groups] int32            the block_table row of each group
scores:           [rows, max_context] fp32  output: scores[q, k] for k below the group length
```

The scores at or past the length of a group stay as they are.
`max_context` sizes the grid, and the block table must cover it.

#### Selection

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

#### Top-K from the selection

```text
candidates:  [rows, slots, 2] int32  the select's (key, score bits) pairs
ranges:      [2, rows] int32         the select's ranges
top_keys:    [rows, top_k] int32     output: the Top-K keys of each row
```

`top_k_from_candidates` reads the pairs of row r in slots `ranges[0, r]` to
`ranges[1, r] - 1`, and writes the keys of the `top_k` largest scores to
`top_keys[r]`, in no set order. If the scores at the cutoff are equal, it takes
any of them. A row of at most `top_k` pairs gets all its keys, then -1. In the
example, the row's 4 pairs give the keys 0, 2, 3 and 4.

### How it works

#### Decode scores

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

#### Selection

The same kernel family selects. The query rows are on the MMA's M (128), and
the keys on its N (128). Each CTA scores a run of key tiles for one query tile,
and appends the candidates and ties of each row to its slots.

With `arrivals` and `ranges`, and at least `top_k - 1` candidate slots, every
row fits. The last CTA of a query tile to finish repairs each row that does not
fit: it scores the keys of the tile again, and selects again at the
`min(row_ends[r], top_k)`-th largest score of the row, which a radix select
finds. So any threshold gives the same exact result. The threshold only sets
how often the repair runs.

#### Top-K from the selection

The select stores scores that keep the order of the row's scores, so the Top-K
of the stored scores is the Top-K of the scores. A CTA of 256 threads takes a
row. It finds the row's `top_k`-th largest score, the threshold, on the score
bits flipped so that their unsigned order is float order, one digit at a time:

1. It counts the top 11 bits of each score in 2048 bins in shared memory. A scan
   of the bins finds the bin that holds the threshold.
2. It reads the scores again, from L2, and counts the next 11 bits of the scores
   in that bin. A scan finds the bin that holds the threshold, and so on for the
   last 10 bits. It stops when it needs all the scores of a bin.
3. It reads the pairs again. The keys above the threshold go to the first
   positions of the row. The keys equal to the threshold go to the last
   positions, until the row has `top_k` keys.

For each 64 pairs, a warp gets its positions from ballots, and reserves them
with one shared-memory atomic. Thus its stores are coalesced. Many equal scores
need no other path: step 2 then counts all the digits.

#### Thresholds from a sample

A good threshold makes the repair rare:

1. `gather_key_sample` copies a strided random sample of the keys of each
   request.
2. `indexer_scores` scores the sample, as pages of 64 keys.
3. A Top-K kernel ranks the sampled scores of each row.
4. `thresholds_from_sample` sets the threshold of each row to the least of its
   sampled Top scores, and zeroes its counters. A -1 position or a NaN score
   counts as -inf.

#### NVFP4 keys and queries

`quantize_write_index_qk_nvfp4` gives each 16 values of a key or query an E4M3
scale: their largest magnitude over 6, with a global scale of 1. It writes the
64 packed bytes and the 8 scales of a key at its slot `s`: page `s // 64`, row
`s % 64`, and those of a query at its row `r` of `packed_query` and
`query_scales`. It skips negative slots and rows. Keys and queries are
quantized the same way, so their scores stay comparable.

### Limits

- A key has 128 values, and a page has 64 keys.
- Decode takes 1 to 8 query rows for each request. Selection takes tiles of 128
  query rows.
- The block table must cover `max_context`.
- The CuTe-DSL kernels need CuTe DSL 4.7 or newer. The scorer and the select
  need SM10x. Off SM10x, `indexer_scores` runs the Triton kernel,
  which takes only BF16 queries and keys, with contiguous rows.
- `gather_key_sample` takes sample rows of whole 128-key tiles.
- `top_k_from_candidates` takes `top_k` below 65536.

### CUDA graphs

No kernel synchronizes with the host. Run each one once before capture, because
the CuTe-DSL kernels compile on first use. Queries, keys, pages, lengths,
thresholds and ranges can change between replays. Zero the `arrivals` of the
select once: every call leaves them at zero.

### Tracing

Each API records a trace template, with the op type `sdsa_<api>`.

### Validation

`tests/sdsa/test_indexer.py` covers:

- decode scores: BF16, BF16 through the Triton kernel, NVFP4, and NVFP4 keys and
  scales in one cache; 1, 3 and 8 rows for each request; lengths that end on a
  page, in a key tile, at 0, and in runs of several CTAs. The values are exact in
  FP32, so the scores must equal an FP64 reference bit for bit. The pages are
  shuffled, block-table entries past a length are invalid, and the scores past a
  length must stay as they are. A CUDA graph replay then uses new lengths.
- selection: BF16 and NVFP4; `top_k` 64, 512 and 2048; with and without the
  repair; query and key lengths that end on a page or in a key tile, and tiles
  with runs of several CTAs, some short and some empty; a CUDA graph replay.
  After the repair, `top_k_from_candidates` must give the Top-K keys of each
  row's scores.
- `top_k_from_candidates`: `top_k` 512, 2048 and 65535; random ranges, empty,
  shorter than `top_k`, one pair longer, and of all the slots; normal scores, a
  few values, all equal, and zeros of both signs, infinities and NaNs; a CUDA
  graph replay with new ranges. The selected scores must be the largest of the
  row.
- `gather_key_sample`: BF16 and NVFP4, with pages in the first 2**31 bytes of
  the cache and after them. Slots after the sample end stay as they are.
- `thresholds_from_sample`: -1 positions and NaN scores.
- `quantize_write_index_qk_nvfp4`: 0 to 300 keys and 0 to 300 queries, at
  shuffled slots and rows, with negative ones skipped.
- the checks that reject bad shapes, bad layouts off SM10x, bad counters, and
  CuTe DSL older than 4.7.

## Token-sparse attention

In token-sparse attention, each query token attends only to its own list of
keys, such as the 2048 keys that the DSA indexer selects for it. FlashInfer runs
it in two ways:

- `trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K, uses_shared_paged_kv_idx=False)`
  runs MQA decode on
  the token-sparse kernels of trtllm-gen. `token_sparse_kv_indices` makes its
  index lists from the key positions of each token.
- `group_union_masks` lets a dense attention kernel that takes a mask for each
  query run it: a group of tokens attends to the union of their keys, and a mask
  limits each token to its own keys.

### Public API

| API | Purpose |
| --- | --- |
| `trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K, uses_shared_paged_kv_idx=False)` | Each query token attends to the K and V rows in its own lists. |
| `flashinfer.sdsa.token_sparse.token_sparse_kv_indices(positions, requests, block_table, *, block_tables, seq_lens, requests_per_token)` | Make `block_tables` and `seq_lens` from the key positions of each token, in one launch. |
| `flashinfer.sdsa.token_sparse.group_union_masks(positions, query_rows, group_unions, requests, block_table, *, union, union_sizes, mask, num_keys)` | Make the union of keys of each group of tokens, and the mask of each token over that union, in one launch. |
| `flashinfer.sdsa.token_sparse.union_key_limit(group, top_k, device=None)` | The most keys (`num_keys`) that `group_union_masks` takes for groups of `group` tokens. |

### Layout

#### Token-sparse decode

Each query token is its own request, over cache pages of one token. MTP draft
tokens are requests too.

```text
query:         [batch, num_qo_heads, head_dim]
kv_cache:      [rows, 2, 1, 1, head_dim], or K and V as two [rows, 1, 1, head_dim]
block_tables:  [batch, 2, K] int32   the K rows, then the V rows, of each token
seq_lens:      [batch] int32         the entries of each list that the token uses
output:        [batch, num_qo_heads, head_dim]
```

`seq_lens[b]` is at most `K`, and the kernels read only the first `seq_lens[b]`
entries. For example, with 2 query tokens and `sparse_top_k=4`:

```python
block_tables = [[[7, 2, 9, -1], [7, 2, 9, -1]],  # token 0: K rows, V rows
                [[5, 1, 8,  3], [5, 1, 8,  3]]]  # token 1: K rows, V rows
seq_lens = [3, 4]  # token 0 uses 3 entries, token 1 uses 4
```

Token 0 attends to the keys and values in rows 7, 2 and 9. The K list and the
V list are the same when the K and V of a key are in the same row.

#### Index lists

`token_sparse_kv_indices` reads and writes:

```text
positions:     [tokens, K] int32         the key positions of each token, valid ones first, then -1
requests:      [tokens] int32            the block_table row of each token
block_table:   [requests, blocks] int32  pages of 64 keys
block_tables:  [tokens * requests_per_token, 2, K] int32, output
seq_lens:      [tokens * requests_per_token] int32, output
```

Key `k` of a request is in row `block_table[request, k // 64] * 64 + k % 64`.

#### Group unions and masks

`group_union_masks` reads and writes:

```text
positions:     [tokens, K] int32    as for token_sparse_kv_indices
query_rows:    [tokens] int32       the mask row of each token, -1 for none
group_unions:  [groups] int32       the union row of each group, -1 for none
requests:      [groups] int32       the block_table row of each group
block_table:   [requests, blocks] int32
union:         [unions, columns] int32, output   the union of each group, as cache rows in key order
union_sizes:   [unions] int32, output            the number of keys in each union
mask:          [masks, words] int32, output      bit j of word w: the token uses union column 32 * w + j
```

The tokens make `len(requests)` groups of equal size, which the caller chooses.
For example, take one group of 2 tokens, where token 0 uses keys 3 and 9, and
token 1 uses keys 9 and 12. These keys are on the first page of the request,
which is cache page 5 (rows 320 to 383):

```python
union[0][:3] = [323, 329, 332]  # the cache rows of keys 3, 9 and 12
union_sizes = [3]
mask[0][0] = 0b011  # token 0: union columns 0 and 1
mask[1][0] = 0b110  # token 1: union columns 1 and 2
```

The group then runs as one varlen entry of the dense kernel, with the union rows
as its K and V. This pays when the tokens of a group share many keys, as the
neighboring prefill tokens of a request do.

### Limits

The decode call rejects calls outside these limits:

- `K` is a multiple of 4, because the kernels copy indices 16 bytes at a time.
- `batch` is at most 65535, because of the launch grid. There are at most 128
  query heads.
- The KV cache has one KV head, because the kernels have no KV-head stride.
- Each request has one query token: `q_len_per_req` is 1, without
  `cum_seq_lens_q`.
- There is no sliding window and there are no attention sinks.
- The head dim is not 512. Head dim 512 is MLA, which
  `trtllm_batch_decode_with_kv_cache_mla` runs.
- `uses_shared_paged_kv_idx` is `False`, and K and V have the same page stride.
  They can be two halves of one `[K | V]` cache row.
- The backend is `trtllm-gen` (SM100 and SM103). K and V are BF16 or FP8.

`token_sparse_kv_indices` and `group_union_masks` take `K` as a multiple of
512, and contiguous int32 tensors on one GPU. `group_union_masks` also needs:

- groups of equal size that cover the tokens, with an entry in `query_rows`
  for each token;
- distinct output rows: other than -1, no two tokens share a `query_rows` entry,
  and no two groups share a `group_unions` entry;
- a block table that covers `num_keys`;
- union rows of at least `min(group * K, 128 * ceil(num_keys / 128))` columns,
  and mask rows of at least as many bits;
- `num_keys` at most `union_key_limit(group, K, positions.device)`, so that a key bitmap with a bit
  for each key below `num_keys`, and the mask rows, fit in shared memory. On
  SM100 with `K` 2048, that is 1,441,792 keys for groups of up to 4 tokens and
  1,179,648 for 32.

### How the trtllm-gen dispatch runs it

- The call uses the static token-sparse mode of trtllm-gen: one list length `K`
  for all tokens, and `seq_lens` for the entries that each token uses.
- `TllmGenFmhaRunnerParams::isMla()` names the MLA shapes. A token-sparse call
  with other shapes is MQA (`isSparseMqa()`).
- The call runs the generation kernels with a dense mask. One CTA holds all
  query heads of its token, because the index lists are per KV head.
- K and V each have their own base pointer and row stride.
- When several CTAs share one list, they combine their results in shared memory
  (the CGA reduction).

### From key positions to kernel inputs

An indexer gives the key positions of each token, and the kernels read cache
rows. Both helpers change the positions to cache rows through the block table of
the request, in one launch, without host synchronization.

`token_sparse_kv_indices` writes the lists of each token. A token can also
become several attention requests (`requests_per_token`), for example one for
each group of its query heads. Each request gets the same lists.

`group_union_masks` writes the union of each group of tokens and the mask of
each token. Positions at or past `num_keys` count as -1.

### CUDA graphs

Run each kernel once before capture, because the CuTe-DSL helpers compile on
first use. The lengths, positions and block tables can change in place between
replays. The decode kernels leave their split-KV semaphores
(`multi_ctas_kv_counter_buffer`) at zero, so one zeroed buffer serves every call
and every replay.

### Tracing

- A decode call with `sparse_top_k` records the `trtllm_batch_decode_token_sparse`
  template: the `[batch, 2, K]` `block_tables`, `sparse_top_k`,
  `uses_shared_paged_kv_idx`, and a token-sparse reference.
- `token_sparse_kv_indices` and `group_union_masks` record the
  `sdsa_token_sparse_kv_indices` and `sdsa_group_union_masks` templates.

### Validation

- `tests/attention/test_trtllm_gen_token_sparse.py` compares the decode call
  with an FP32 reference:
  - `K` 512 and 2048, 7 and 300 query tokens, 2 to 96 query heads, BF16 and FP8;
  - entries per token 1, 5, 127, 128, 129, `K` - 1 and `K`;
  - K and V as two halves of one cache row;
  - the rows outside the lists hold 300, so a wrong read changes the output;
  - a second run with new lists reuses the split-KV semaphores.

  It also checks that the call rejects several KV heads, a sliding window,
  attention sinks and head dim 512.
- `tests/sdsa/test_token_sparse.py` compares both helpers with PyTorch
  references, without tolerance, eagerly and in a CUDA graph replay:
  - `token_sparse_kv_indices`: `K` 512 and 2048, and 1 to 77 tokens with 1 to 4
    requests per token. Rows after the output stay as they are.
  - `group_union_masks`: `K` 512 and 2048, and groups of 1, 4 and 16 tokens of
    two requests, one with 5000 keys and one with 600000 or `union_key_limit`
    keys. It covers short lists,
    tokens without a mask row, a group without a union row, and rows in reverse
    order. Nothing is written after a union or after the words of a mask row.
    The replay uses new keys.
- `tests/trace/test_trtllm_batch_decode_reference_correctness.py` compares the
  token-sparse trace reference with the kernel, for two shapes.
