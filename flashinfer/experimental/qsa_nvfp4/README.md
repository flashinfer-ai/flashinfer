# Experimental QSA NVFP4 sparse paged attention

`flashinfer.sparse.qsa_sparse_paged_attention` runs the main attention of the Qwen
QSA layer over an NVFP4 KV cache on SM80 through SM90. It implements the contract of
vLLM's Triton `qsa_sparse_paged_attention` (same arguments, same order) and adds the
NVFP4 block scales `k_sf` / `v_sf`. vLLM keeps its Triton kernel for BF16 and FP8
caches; this backend covers the NVFP4 case it does not.

It is reached only through that experimental API: with `backend="default"`, or with
the entry's default `backend="auto"` on a device outside SM80 through SM90, where it
reports the unsupported device (on SM80 through SM90, `"auto"` runs the Cake kernels
of `flashinfer.experimental.cake_qsa_nvfp4`). No AOT packaging. The experimental
lifecycle is owned by `@lesj0610`; the tracking issue is linked from the pull request
that adds the package.

## Contract

- `q`: BF16 `[rows, num_qo_heads, 256]`; `num_qo_heads / num_kv_heads` is 1 to 16.
- `k_cache`, `v_cache`: packed E2M1 `[num_blocks, page_size, num_kv_heads, 128]`,
  `k_sf`, `v_sf`: E4M3 `[num_blocks, page_size, num_kv_heads, 16]`, the layout written
  by `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`. Strides are free as long
  as the last dim is contiguous and 16-byte aligned, so an HND cache is passed as its
  `transpose(1, 2)` view.
- `logical_indices`: int32 `[rows, selection_width + 1]` logical token indices; the
  last column is the row's valid-entry count. `block_table` maps logical pages to cache
  blocks per request and `token_to_req` gives each row's request. Entries that are
  negative, past the table, unmapped or past the cache are skipped (never read).
- `k_scale`, `v_scale`: NVFP4 global scales. `output_gate`: BF16 like `q`.
- Output: `bf16(attention) * sigmoid(output_gate)`, the attention output rounded to BF16
  before the FP32 gate, as in vLLM. Rows with no valid entry are zero.

## Implementation

`csrc/qsa_nvfp4_attention.cuh` holds two kernels: a split-K kernel with one CTA per
(row, KV head, split) over 64-token tiles, and an LSE merge that also applies the gate
when the selection is split. It uses SM80 instructions only (cp.async, mma.sync
m16n8k16), so one JIT build serves SM80 through SM90. NVFP4 is decoded directly into
MMA fragments: QK and PV both take BF16 operands (E2M1 times an E4M3 block scale is
exact in BF16) and accumulate in FP32, with the global scales folded into the softmax
scale and the output scale.

The split count targets about 128 CTAs per launch (`backend.py`), one value per
architecture family.
