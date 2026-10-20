# Single-head paged MQA indexer

Sparse attention in the style of DeepSeek Sparse Attention (DSA) picks each
query token's Top-K keys by indexer scores, then attends to those keys. DSA's
indexer scores with many heads (`fp8_paged_mqa_logits`, `fp4_paged_mqa_logits`).
The kernels in `flashinfer.attn_scores.single_head` serve an indexer that scores
with one query head and no head weights, so a score is the dot product of a
query row and a key, both BF16 or both NVFP4. Keys are paged 64 to a page, and
`quantize_write_index_k_nvfp4` writes NVFP4 keys at their slots.

## Decode

`single_head_paged_mqa_logits` scores each request's 1 to 8 query rows (a decode
token and its draft tokens) against its keys, on a static grid sized from the
context capacity. A Top-K over the scores then picks each row's keys.

## Prefill

A long prefill would write a score for every query and key. Instead,
`single_head_paged_mqa_threshold_select` keeps only the keys at or above a
per-row threshold, as (key, score) candidate and tie slots, in one pass over the
keys. A Top-K over the row's slots then picks its keys. Given `arrivals` and
`ranges`, the result is exact for any threshold. A row whose threshold misses
(too many candidates, or too few with its ties) is repaired in the same launch,
and `ranges` gives each row's slots.

A good threshold makes repairs rare. `gather_key_sample` copies a strided random
sample of each request's keys, and `single_head_paged_mqa_logits` scores it. A
Top-K over the sample's scores ranks it, and `thresholds_from_sample` takes each
row's threshold from its sampled Top scores, and zeroes the select's counters.

## Attention over the selection

The selected keys feed token-sparse attention
([token_sparse_attention.md](token_sparse_attention.md)): either per-token K/V
lists for trtllm-gen (`token_sparse_kv_indices`), or each group of tokens' union
with per-token masks for masked dense kernels (`group_union_masks`).

## CUDA graphs

No kernel synchronizes with the host. Run each one once before capture, since
the CuTe-DSL kernels compile on first use. Queries, keys, pages, lengths and
thresholds can change between replays. The select's arrival counts are zeroed
once, and every call leaves them zeroed.
