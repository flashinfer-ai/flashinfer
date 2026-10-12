# Sparse MXFP4/MXFP8 MQA logits indexer

`from flashinfer.sparse_mqa import prepare_sparse_mqa_metadata, prepare_sparse_mqa_logits`
prepares the compressed sparse MQA indexer (metadata generation followed by the
block-scaled logits kernel) on SM100a and SM103a. Each prepared call returns a
plan; `plan.run()` submits on the current PyTorch stream.

Ten generated programs serve both architectures (the loader compiles each
program's source with the exact flag set of the attached device):

- two runtime-geometry metadata programs, one per KV layout (contiguous,
  paged). The SM count, the sparse index capacity (a multiple of 4 up to
  `LIMITS["max_sparse_blocks"]`), the split width (640 MXFP4 / 512 MXFP8 KV
  tokens over the block size), the sparse block size (8 or 16) and the page
  size (any multiple of the block size) are runtime arguments;
- four exact-geometry metadata programs: the production geometry (capacity
  `LIMITS["exact_capacity"]`, 8-token blocks, 64-token pages) per layout,
  compiled with the device SM count folded into the schedule layout, one per
  (architecture, SM count) pair of the measured devices (`ROUTES` keys
  `metadata:<layout>:exact:<sms>`; `exported_exact_sms(arch, paged=...)` lists
  the SM counts exported for an architecture, `LIMITS["exact_num_sms"]` their
  union). `metadata_route_key()` selects an exact program only for an
  (architecture, SM count) pair it was exported for and the runtime program
  otherwise (e.g. 152 SMs on SM100a), never a program compiled for another SM
  count;
- four logits programs, one per format x layout (MXFP4 runs five math
  warpgroups over 640-token splits, MXFP8 four over 512-token splits), with
  32 heads, D=128, 8-token blocks, 64-token pages and aligned windows.

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
other slots keep their contents. A metadata plan prepared with a block or page size the
logits programs do not implement raises `NotImplementedError` here. `plan.run()` submits
the complete metadata + logits pipeline with launch overhead and without allocation.
Operand values may change in place between runs; prepare a new plan when shapes,
layouts or the window vectors' identity change. Do not use one plan's mutable buffers
concurrently on different streams.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_sparse_mqa/generated`
(one device/binding pair per program). The runtime, program registry and route table
are in `sparse_mqa.py` in this directory. The public API is `flashinfer/sparse_mqa.py`;
see `examples/experimental/sparse_mqa.py`. Build/install FlashInfer with its supported
CUDA toolchain before running:

```bash
python examples/experimental/sparse_mqa.py
pytest tests/experimental/test_sparse_mqa_generated.py -q
```

The test covers all four routes of the present architecture with exact analytical
values at several sparse capacities, metadata decoding at runtime page sizes, block
sizes and split widths, a non-default stream, changed-input graph replay and
invalid-slot retention; the native-consumer cross-check runs when DeepGEMM is
installed. Performance qualification measures the complete prepared pipeline with
prepacked inputs over the twenty model rows (contiguous Q=8192 and paged Q=512, KV
lengths 4096 to 1048576, MXFP4 and MXFP8) against the generating source route;
preparation is excluded from both timed arms.
