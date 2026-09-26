# Sparse MXFP4/MXFP8 MQA logits indexer

`from flashinfer.sparse_mqa import prepare_sparse_mqa_metadata, prepare_sparse_mqa_logits`
prepares the compressed sparse MQA indexer (metadata generation followed by the
block-scaled logits kernel) on SM100a (148 SMs) and SM103a (152 SMs). Each prepared
call returns a plan; `plan.run()` submits on the current PyTorch stream.

The exported routes use 32 heads, D=128, a sparse capacity of 2048 blocks of 8 KV
tokens, 64-token pages and aligned windows, for MXFP4 and MXFP8 inputs in both the
contiguous and the paged KV layout (four physical routes per architecture, each a
metadata program plus a logits program).

`prepare_sparse_mqa_metadata(sparse_indices, ...)` binds sorted int32 `[Q, capacity]`
indices with duplicate padding in unused slots. Contiguous inputs pass `starts`, `ends`
(int32 `[Q]`) and `num_kv_tokens`; paged inputs pass `context_lens`, `request_indices`
(int32 `[Q]`) and a per-query `block_table` (int32 `[Q, pages]`). The packed metadata is
uint8 with the extent given by `metadata_size_bytes()`; the int32 workspace has
`metadata_workspace_words()` entries and must initially be zero. Its three counters are
restored by the kernel after every submission. Split allocation order is not a canonical
serialization; consume metadata through this ABI rather than comparing raw buffers.

`prepare_sparse_mqa_logits(q, sf_q, kv, sf_kv, weights, metadata_plan, output=None)`
binds packed E2M1 `[Q, 32, 64]` or E4M3 `[Q, 32, 128]` queries, int32 `[Q, 32]` packed
UE8M0 query scales and BF16 `[Q, 32]` head weights. Contiguous KV is packed rows
`[K, 64|128]` with int32 `[K]` scales; paged KV is uint8 `[pages, stride]` with each
page's row bytes followed by its scale bytes, padded to a 512-byte stride (`sf_kv` is
unused). The output is BF16 `[Q, capacity * 8]`; only selected valid slots are written and
other slots keep their contents. `plan.run()` submits the complete metadata + logits
pipeline with launch overhead and without allocation. Operand values may change in place
between runs; prepare a new plan when shapes, layouts or the window vectors' identity
change. Do not use one plan's mutable buffers concurrently on different streams.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_sparse_mqa/generated`
(one directory per architecture). The runtime and the per-architecture route catalog are in
this directory. The public API is `flashinfer/sparse_mqa.py`; see
`examples/experimental/sparse_mqa.py`. Build/install FlashInfer with its supported CUDA
toolchain before running:

```bash
python examples/experimental/sparse_mqa.py
pytest tests/experimental/test_sparse_mqa_generated.py -q
```

The test covers all four routes of the present architecture with exact analytical
values, native metadata consumption, a non-default stream, changed-input graph replay and
invalid-slot retention; it skips on devices without catalogued programs or SM counts.
Performance qualification measures the complete prepared pipeline with prepacked inputs
over the twenty model rows (contiguous Q=8192 and paged Q=512, KV lengths 4096 to
1048576, MXFP4 and MXFP8). Per-row exported execution must remain within 3% of its source
route. Performance results accompany the generated bundle; preparation is excluded from
both timed arms.
