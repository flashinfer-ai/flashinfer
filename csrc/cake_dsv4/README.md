# Cake DeepSeek V4 sparse MLA

Generated device kernels and TVM-FFI launchers for SM100 / SM103 behind
`trtllm_batch_decode_sparse_mla_dsv4(..., backend="cake")` for BF16 and FP8 E4M3
inputs with head dimension 512 (BF16 output). Select the backend explicitly;
`backend="auto"` keeps the architecture-based selection. Programmatic dependent
launch and the TRTLLM-GEN RopeQuant epilogue are not supported.

```python
from flashinfer.mla import (
    cake_dsv4_workspace_reset,
    get_cake_dsv4_workspace_bytes,
    trtllm_batch_decode_sparse_mla_dsv4,
)

workspace = torch.empty(
    get_cake_dsv4_workspace_bytes(num_query_tokens, num_heads, sparse_topk, dtype),
    dtype=torch.uint8, device="cuda",
)
cake_dsv4_workspace_reset(workspace)  # zero the split-merge counters once
out = trtllm_batch_decode_sparse_mla_dsv4(
    query=query, swa_kv_cache=swa_kv_cache, compressed_kv_cache=compressed_kv_cache,
    workspace_buffer=workspace, sparse_indices=sparse_indices,
    sparse_topk_lens=sparse_topk_lens, seq_lens=seq_lens, backend="cake",
)
```

## Layout

| path | contents |
| --- | --- |
| `sm_100a/`, `sm_103a/` | one `*_kernel.cu` (device code) and one `*_binding.cu` (launcher) per variant and architecture |
| `common/` | kernels whose source is identical on both architectures (`split_reduce`, `bf16_h64_compressed_reduce`) |
| `cake_dsv4_host_shim.h` | the launcher helpers shared by every binding (device guard, tensor checks, SM103 descriptor-storage writes) |

`flashinfer/jit/cake_dsv4.py` registers each variant per architecture: its
sources, nvcc flags, an identity that names the JIT module, and the `arg_plan`
the host binds by name. `flashinfer/mla/cake_dsv4.py` resolves the sparse
metadata, carves the caller-owned workspace, picks the route (`_route`) and
launches the variant kernels directly, reducers included; shapes without an
exported kernel are rejected before any launch. Bindings perform no device
allocation.

## Host contract

* Metadata: every kernel takes `swa_indices`, `compressed_indices`,
  `sparse_topk_lens`, `swa_index_stride`, `compressed_index_stride`,
  `sparse_topk_lens_offset`, `sparse_topk`, `num_query_tokens`. Combined
  (`sparse_indices [T, sparse_topk]`) and separate (`sparse_indices [T, 128]` +
  `extra_sparse_indices [T, topk_c]`, lengths via `sparse_topk_lens` or
  `extra_sparse_topk_lens`) tables are resolved without copies; int32, unit column
  stride, free row stride.
* Padded rows: `num_query_tokens = T` is the metadata row count; `query` / `out`
  may carry more rows, rows `>= T` are neither read nor written.
* Workspace (contiguous, 128-byte aligned): `[0, 1024)` reserved,
  `[1024, 1024 + 256 KiB)` split-merge counters (zero at first use, left zero by
  every launch), then `partial_O` (BF16) and `partial_lse` (FP32) sized by
  `get_cake_dsv4_workspace_bytes`. CUDA-graph replays are self-contained once the
  counters were zeroed eagerly.
* SM103 descriptor storage: the bindings with `tma_workspace_bytes` read their TMA
  descriptors from a private device tensor the host passes in; the binding writes
  the descriptors of the call into it when they differ from what it holds (in
  stream order, never inside graph capture). The host keeps a bounded pool of
  such tensors per (variant, device), keyed by the TMA source geometry: recent
  geometries reuse their storage, new ones take a free or the least recently
  used storage, geometries launched under graph capture keep theirs for the
  process lifetime, and a capture that reaches a geometry never launched
  eagerly raises.
* Query / output / KV pools must be densely packed; the host makes no copies.

## Tests

```bash
pytest tests/mla/test_cake_dsv4.py -q              # routing, binding, workspace, metadata (CPU)
pytest tests/mla/test_cake_dsv4_hardening.py -q    # numerical and graph-replay checks (SM100/SM103 GPU)
```

Tolerances: `atol=rtol=0.01` (BF16), `atol=rtol=0.1` (FP8).

## SM120 / SM121: DeepSeek-V4 NVFP4 sparse-MLA decode

`sm_120a/` holds the Cake-generated SM120 (GB202: RTX 5090, RTX PRO 6000
Blackwell) DeepSeek-V4 NVFP4 sparse-MLA decode family: one translation unit per
query head count (8, 16, 32, 48, 64, 80, 96, 112, 128) with the single-cache
decode, dual-cache decode and split-merge kernels (head counts divisible by 32
also carry two-tile variants that process 32 heads per CTA over one shared
candidate gather), the TVM-FFI binding
`cake_sparse_mla_dsv4_nvfp4_binding.cu`, and
`cake_sparse_mla_dsv4_nvfp4_manifest.json` (provenance, geometry constants
and the source list the JIT spec compiles). Regenerate the whole directory
from the Cake kernel schedules with one command; do not edit the generated
files.

Select it through the existing SM120 entry points:

```python
from flashinfer.mla import SparseMLASm120Wrapper, trtllm_batch_decode_sparse_mla_dsv4

out = trtllm_batch_decode_sparse_mla_dsv4(
    query=q, swa_kv_cache=nvfp4_cache, workspace_buffer=workspace,
    sparse_indices=indices, swa_topk_lens=lengths, bmm1_scale=sm_scale,
    backend="cake", kv_cache_format="nvfp4",
)
runner = SparseMLASm120Wrapper(kv_cache_format="nvfp4", backend="cake")
```

The route accepts the packed NVFP4 cache (`page_size * 352` data bytes followed
by `page_size * 32` scale bytes per page) as `[P, page, 384]`, HND or NHD views
with any positive page size and a 16-byte multiple page stride (padded pools),
an optional second cache (`compressed_kv_cache` / `extra_kv_cache`, any page
size), per-token lengths, `-1` masking, attention sinks and `lse_scale`.
Split-K scratch (`mid_out` / `mid_lse`) is caller-owned or carved from the
public `workspace_buffer`; size it with
`flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes`. The Python
planners `cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles` (1 or 2 head tiles
per CTA) and `cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits` (split count and
chunks per CTA, at most 16 chunks of 64 candidates per CTA, so more than 1024
candidates always split) mirror the generating kernel module's launcher;
`head_tiles` / `num_splits` on the low-level decode entry override them.
`backend="auto"` keeps the hand-written SM120 kernels.
