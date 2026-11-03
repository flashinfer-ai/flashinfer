# Experimental QSA NVFP4 sparse paged attention

`flashinfer.sparse.qsa_sparse_paged_attention` runs the main attention of the Qwen
QSA layer over an NVFP4, FP8-e4m3 or BF16 KV cache on SM80 through SM90. It implements
the contract of vLLM's Triton `qsa_sparse_paged_attention` (same arguments, same order)
and adds the NVFP4 block scales `k_sf` / `v_sf`.

It is reached only through that experimental API: with `backend="default"`, with the
entry's default `backend="auto"` for every cache on SM80 through SM90 and for FP8 and
BF16 caches anywhere, and for NVFP4 on a device the Cake package has no library for,
where it reports the unsupported device (on the capabilities above SM90 that Cake
exports, `"auto"` runs NVFP4 on the Cake kernels of
`flashinfer.experimental.cake_qsa_nvfp4`).
No AOT packaging. The experimental
lifecycle is owned by `@lesj0610`; the tracking issue is linked from the pull request
that adds the package.

## Contract

- `q`: BF16 `[rows, num_qo_heads, 256]`; `num_qo_heads / num_kv_heads` is 1 to 16.
- `k_cache`, `v_cache`: packed E2M1 uint8 `[num_blocks, page_size, num_kv_heads, 128]`
  with `k_sf`, `v_sf`: E4M3 `[num_blocks, page_size, num_kv_heads, 16]`, the layout
  written by `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`; or float8_e4m3fn
  or bfloat16 `[num_blocks, page_size, num_kv_heads, 256]` without block scales. Strides
  are free as long as the last dim is contiguous and 16-byte aligned, so an HND cache is
  passed as its `transpose(1, 2)` view.
- `logical_indices`: int32 `[rows, selection_width + 1]` logical token indices; the
  last column is the row's valid-entry count. `block_table` maps logical pages to cache
  blocks per request and `token_to_req` gives each row's request. Entries that are
  negative, past the table, unmapped or past the cache are skipped (never read).
- `k_scale`, `v_scale`: NVFP4 global or FP8 per-tensor scales, ignored for BF16.
  `output_gate`: BF16 like `q`.
- Output: `bf16(attention) * sigmoid(output_gate)`, the attention output rounded to BF16
  before the FP32 gate, as in vLLM. Rows with no valid entry are zero.

## Implementation

`csrc/qsa_nvfp4_attention.cuh` holds two kernels: a split-K kernel with one CTA per
(row, KV head, split) over 64-token tiles (32 for BF16), templated on the cache format,
and an LSE merge that also applies the gate when the selection is split. It uses SM80
instructions only (cp.async, mma.sync m16n8k16), so one JIT build serves SM80 through
SM90. K is decoded exactly to BF16 straight into MMA fragments; PV takes BF16 operands
for NVFP4 and BF16 and FP16 operands for FP8, like the Triton kernel; both accumulate in
FP32, with the global scales folded into the softmax scale and the output scale.

The split count fills one wave of resident CTAs (occupancy times the SM count,
`backend.py`) for FP8 and BF16, and two CTAs per SM for NVFP4.
