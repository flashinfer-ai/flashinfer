# Token-sparse attention for MQA/GQA

In token-sparse attention, each query token attends to its own list of keys,
such as the Top-K keys an indexer selects for it (DeepSeek Sparse Attention,
DSA, selects 2048 per token). trtllm-gen ships static token-sparse kernels, but
FlashInfer only dispatched them for MLA, DSA's core attention.
`trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K)` now dispatches them for
MQA/GQA: each query token is its own request over one-token pages, attending to
its own list of at most `K` keys.

- `block_tables` is `[batch, 2, K]`: each token's K and V slot indices, with
  `uses_shared_paged_kv_idx=False`. `seq_lens` counts each list's valid entries.
- `K` is a multiple of 4, since indices are copied 16 bytes at a time. `batch`
  is at most 65535 (the launch grid), and a KV head at most 128 query heads.
- K and V can be views of one `[K | V]` cache row, in BF16 or FP8.

The dispatch changes are small. `TllmGenFmhaRunnerParams::isMla()` names the MLA
shapes, so token-sparse no longer implies MLA. One CTA holds all query heads of
its token, since the index table is per KV head. K and V are flattened to token
rows, each with its own base pointer and row stride. When one CTA covers a
token's whole list, the static tile scheduler replaces the persistent one, and
split lists can use the CGA shared-memory reduction. MLA and dense attention
take the same paths as before.

## Index lists from key positions

A selection usually gives each token logical key positions, while the kernels
read cache slots. `flashinfer.token_sparse.token_sparse_kv_indices` maps each
token's positions to the K and V slot lists and lengths above, through its
request's block table (64-key pages), in one launch. A token can be split into
several attention requests (`requests_per_token`), for example one per group of
its query heads. Each of them gets the same lists.

## Group unions and masks

Dense attention kernels that take a per-query mask can serve token-sparse
attention too. `flashinfer.token_sparse.group_union_masks` collects the union of
keys of a group of consecutive tokens, as cache slots in ascending key order,
and each token's bitmask over the union's columns, in one launch. The group
then runs as one varlen entry whose K/V are the union's slots. This pays off
when the tokens of a group share many keys, as neighboring prefill tokens of a
request do. Per-token lists do not depend on that.

## CUDA graphs

Run each kernel once before capture (the CuTe-DSL helpers compile on first use).
Lengths, positions and block tables can change in place between replays. The
token-sparse core leaves its split-KV semaphores (`multi_ctas_kv_counter_buffer`)
zeroed, so one zeroed buffer serves every call and every replay.
