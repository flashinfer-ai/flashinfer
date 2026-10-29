# Token-sparse attention for MQA

## Scope

In token-sparse attention, each query token attends only to its own list of
keys. For example, the DSA indexer selects 2048 keys for each token.

This design covers:

- token-sparse MQA decode on the token-sparse kernels of trtllm-gen, through
  `trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K)`;
- `token_sparse_kv_indices`, which makes the index lists of that call from the
  key positions of each token;
- `group_union_masks`, which lets dense attention kernels that take a mask for
  each query run token-sparse attention: a group of tokens attends to the union
  of their keys, and a mask limits each token to its own keys;
- the trace templates of these calls.

It does not select the keys: an indexer does that. It does not add attention
kernels, and it does not change MLA or dense attention. For token-sparse MLA,
use `trtllm_batch_decode_with_kv_cache_mla`.

## Public API

| API | Purpose |
| --- | --- |
| `trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K, uses_shared_paged_kv_idx=False)` | Each query token attends to the K and V rows in its own lists. |
| `flashinfer.token_sparse.token_sparse_kv_indices(positions, requests, block_table, *, block_tables, seq_lens, requests_per_token)` | Make `block_tables` and `seq_lens` from the key positions of each token, in one launch. |
| `flashinfer.token_sparse.group_union_masks(positions, query_rows, group_unions, requests, block_table, *, union, union_sizes, mask, num_keys)` | Make the union of keys of each group of tokens, and the mask of each token over that union, in one launch. |

## Layout

### Token-sparse decode

Each query token is its own request, over cache pages of one token. MTP draft
tokens are requests too.

```text
query:         [batch, num_qo_heads, head_dim]
kv_cache:      [rows, 2, 1, 1, head_dim], or K and V as two [rows, 1, 1, head_dim]
block_tables:  [batch, 2, K] int32   the K rows, then the V rows, of each token
seq_lens:      [batch] int32         the entries of each list that the token uses
output:        [batch, num_qo_heads, head_dim]
```

`seq_lens[b]` is at most `K`. The kernels do not read the other entries. For
example, with 2 query tokens and `sparse_top_k=4`:

```python
block_tables = [[[7, 2, 9, -1], [7, 2, 9, -1]],  # token 0: K rows, V rows
                [[5, 1, 8,  3], [5, 1, 8,  3]]]  # token 1: K rows, V rows
seq_lens = [3, 4]  # token 0 uses 3 entries, token 1 uses 4
```

Token 0 attends to the keys and values in rows 7, 2 and 9. The K list and the
V list are the same when the K and V of a key are in the same row.

### Index lists

`token_sparse_kv_indices` reads and writes:

```text
positions:     [tokens, K] int32         the key positions of each token, valid ones first, then -1
requests:      [tokens] int32            the block_table row of each token
block_table:   [requests, blocks] int32  pages of 64 keys
block_tables:  [tokens * requests_per_token, 2, K] int32, output
seq_lens:      [tokens * requests_per_token] int32, output
```

Key `k` of a request is in row `block_table[request, k // 64] * 64 + k % 64`.

### Group unions and masks

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

The tokens make `len(requests)` groups of equal size. For example, take one
group of 2 tokens, where token 0 uses keys 3 and 9, and token 1 uses keys 9 and
12. These keys are on the first page of the request, which is cache page 5
(rows 320 to 383):

```python
union = [[323, 329, 332]]  # the cache rows of keys 3, 9 and 12
union_sizes = [3]
mask = [[0b011],   # token 0: union columns 0 and 1
        [0b110]]   # token 1: union columns 1 and 2
```

The group then runs as one varlen entry of the dense kernel, with the union rows
as its K and V. This pays when the tokens of a group share many keys, as the
neighboring prefill tokens of a request do.

## Limits

The decode call rejects calls outside these limits:

- `K` is a multiple of 4, because the kernels copy indices 16 bytes at a time.
- `batch` is at most 65535, because of the launch grid. There are at most 128
  query heads.
- The KV cache has one KV head, because the kernels cannot select a KV head.
- Each request has one query token: `q_len_per_req` is 1, without
  `cum_seq_lens_q`.
- There is no sliding window and there are no attention sinks.
- The head dim is not 512. Head dim 512 selects the MLA kernels.
- `uses_shared_paged_kv_idx` is `False`, and K and V have the same page stride.
  They can be two halves of one `[K | V]` cache row.
- The backend is `trtllm-gen` (SM100 and SM103). K and V are BF16 or FP8.

`token_sparse_kv_indices` and `group_union_masks` take `K` as a multiple of
512, and contiguous int32 tensors on one GPU. `group_union_masks` also needs:

- groups of equal size that cover the tokens, with an entry in `query_rows`
  for each token;
- a block table that covers `num_keys`;
- union and mask rows that hold the union of every group;
- a key bitmap and mask rows that fit in shared memory. The bitmap has a bit for
  each key below `num_keys`.

## How the trtllm-gen dispatch runs it

- The call uses the static token-sparse mode of trtllm-gen: one list length
  `K` for all tokens, and `seq_lens` for the entries that each token uses. The
  dynamic mode, with a length for each query token of a request, is for MLA.
  MQA does not need it, because each request has one query token.
- `TllmGenFmhaRunnerParams::isMla()` names the MLA shapes. A token-sparse call
  with other shapes is MQA (`isSparseMqa()`).
- The call runs the generation kernels with a dense mask. One CTA holds all
  query heads of its token, because the index lists are per KV head.
- K and V each have their own base pointer and row stride.
- When several CTAs share one list, they combine their results in shared memory
  (the CGA reduction).

## From key positions to kernel inputs

An indexer gives the key positions of each token, but the kernels read cache
rows. Both helpers change the positions to cache rows through the block table of
the request, in one launch, without host synchronization.

`token_sparse_kv_indices` writes the lists of each token. A token can also
become several attention requests (`requests_per_token`), for example one for
each group of its query heads. Each request gets the same lists.

`group_union_masks` writes the union of each group of tokens and the mask of
each token. Positions at or past `num_keys` count as -1.

## CUDA graphs

Run each kernel once before capture, because the CuTe-DSL helpers compile on
first use. The lengths, positions and block tables can change in place between
replays. The decode kernels leave their split-KV semaphores
(`multi_ctas_kv_counter_buffer`) at zero, so one zeroed buffer serves every call
and every replay.

## Tracing

- A decode call with `sparse_top_k` records the `trtllm_batch_decode_token_sparse`
  template: the `[batch, 2, K]` `block_tables`, `sparse_top_k`,
  `uses_shared_paged_kv_idx`, and a token-sparse reference.
- `token_sparse_kv_indices` and `group_union_masks` record templates of the same
  names.

## Validation

- `tests/attention/test_trtllm_gen_token_sparse.py` compares the decode call
  with an FP32 reference:
  - `K` 512 and 2048, 7 and 300 query tokens, 2 to 96 query heads, BF16 and FP8;
  - entries per token 1, 5, 127, 128, 129, `K` - 1 and `K`;
  - K and V as two halves of one cache row;
  - the rows outside the lists hold 300, so a wrong read changes the output;
  - a second run with new lists reuses the split-KV semaphores.

  It also checks that the call rejects several KV heads, a sliding window,
  attention sinks and head dim 512.
- `tests/attention/test_token_sparse.py` compares both helpers with PyTorch
  references, without tolerance, eagerly and in a CUDA graph replay:
  - `token_sparse_kv_indices`: `K` 512 and 2048, and 1 to 77 tokens with 1 to 4
    requests per token. Rows after the output stay as they are.
  - `group_union_masks`: `K` 512 and 2048, and groups of 1, 4 and 16 tokens of
    two requests, one with 5000 keys and one with 600000. It covers short lists,
    tokens without a mask row, a group without a union row, and rows in reverse
    order. Nothing is written after a union or after the words of a mask row.
    The replay uses new keys.
- `tests/trace/test_trtllm_batch_decode_reference_correctness.py` compares the
  token-sparse trace reference with the kernel, for two shapes.
- The trtllm-gen decode, MLA, sparse MLA and block-sparse tests cover the paths
  that this design does not change.

## Non-goals and follow-up work

- Several KV heads (GQA). The kernels would need a KV-head stride.
- One index list for both K and V. The two lists are the same in practice, so
  one list would read half as many indices.
- A sliding window and attention sinks.
- Group sizes for `group_union_masks`: the caller chooses them.
