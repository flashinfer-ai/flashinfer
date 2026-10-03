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
| `cake_dsv4_launch_sequence.cc` | host-only `run_sequence`: issues the producer and reducer launches of a two-stage route from one FFI call |

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

## SM120 / SM121: DeepSeek-V4 NVFP4 sparse-MLA decode and prefill

`sm_120a/` holds the Cake-generated SM120 (GB202: RTX 5090, RTX PRO 6000
Blackwell) DeepSeek-V4 NVFP4 sparse-MLA families (DeepSeek-V4 sparse MLA SM120
tracker: flashinfer#4254):

* **decode** -- one translation unit per query head count (8, 16, 32, 48, 64,
  80, 96, 112, 128), `cake_sparse_mla_dsv4_nvfp4_h<N>.cu`, with the
  single-cache decode, dual-cache decode and split-merge kernels (head counts
  divisible by 32 also carry two-tile variants that process 32 heads per CTA
  over one shared candidate gather);
* **prefill** -- one translation unit per head count 16 .. 128,
  `cake_sparse_mla_dsv4_nvfp4_prefill_h<N>.cu`, with one kernel per (head tiles
  x single / dual cache x one-item / persistent CTAs): a prefill CTA holds
  `16 * head_tiles` heads of one token (1 tile always, 2 for head counts
  divisible by 32, 4 for head counts divisible by 64) and runs *all* of the
  token's 64-candidate chunks (at most 16, so `topk + extra_topk <= 1024`)
  with a direct epilogue -- no split scratch, no merge launch. The persistent
  form launches `min(items, SMs)` CTAs that walk the (token, head block) items
  with a grid stride while the IO warps prefetch and quantize the next item's
  Q.

Both share the TVM-FFI binding `cake_sparse_mla_dsv4_nvfp4_binding.cu`
(entries `cake_sparse_mla_sm120_dsv4_nvfp4_decode` and
`cake_sparse_mla_sm120_dsv4_nvfp4_prefill`), the kernel ABI header and
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

For several query tokens the same entry points pick between the decode and
the prefill kernel through
`flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel` (one token,
head counts without a prefill instance, more than 16 chunks and candidate
counts that are not multiples of 64 always decode; wide head counts (>= 64)
decode up to 16 tokens, or up to 32 tokens with at most 128 candidates;
narrower head counts decode up to 64 tokens with at least 512 candidates). The
prefill planner `cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill` returns
`(head_tiles, persistent)`: two tiles below 128 tokens for head counts >= 64
that are divisible by 32, otherwise the largest instance (one tile for 16 / 32
/ 48 heads), persistent CTAs once the one-item grid covers two waves of SMs.
These thresholds are provisional until the paired decode / prefill sweeps on
RTX PRO 6000 Blackwell and RTX 5090 land. The low-level entry
`flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_prefill` runs the prefill
directly (caller-owned `output` / `out_lse`, `head_tiles` / `persistent`
overrides, no scratch):

```python
from flashinfer.mla import cake_sparse_mla_sm120_dsv4_nvfp4_prefill

plan = cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
    q, nvfp4_cache, indices, output, out_lse, sm_scale,
    topk_length=lengths, attn_sink=sink,
    extra_kv_cache=compressed_cache, extra_indices=extra_indices,
)
# plan == {"head_tiles": 4, "persistent": 1, "num_ctas": <SMs>} for 128 heads x 2048 tokens
```

```bash
pytest tests/attention/test_cake_sparse_mla_sm120_dsv4_nvfp4.py -q          # decode (needs SM120/SM121)
pytest tests/attention/test_cake_sparse_mla_sm120_dsv4_nvfp4_prefill.py -q  # prefill + CPU planner tests
python benchmarks/bench_cake_sparse_mla_sm120_dsv4_nvfp4_prefill.py         # paired sparse / cake prefill rows
```

## SM120 / SM121: DeepSeek-V4.1 mixed-cache sparse-MLA decode

`sm_120a/` also holds the Cake-generated DeepSeek-V4.1 mixed-cache decode
family `cake_sparse_mla_dsv41_mixed_*`: a 528-byte FP8 + UE8M0 group-32 main
(SWA) cache, a 288-byte V41_FP4 extra (compressed) cache (512 E2M1 dims
including RoPE, 256-byte payload + 32-byte E4M3 group-16 footer scales; never
read as the 384-byte NVFP4 layout) and a BF16 query. QK runs in BF16
(`mma.sync m16n8k16`) on exactly dequantized K (E4M3 x 2^e, E2M1 x E4M3), P and
V are BF16, accumulation / softmax / split merge are fp32; output BF16, LSE
base-2 with the public sink and `lse_scale` semantics. One translation unit per
query head count (8, 16, 32, 48, 64, 80, 96, 112, 128: single-cache decode,
dual-cache decode and the split merge kernel; 32 heads and up run two 16-head
tiles per CTA sharing one gather), the TVM-FFI binding
`cake_sparse_mla_dsv41_mixed_binding.cu` and
`cake_sparse_mla_dsv41_mixed_manifest.json` (identity, kernel commit, ABI,
planner geometry). The split merge runs one 64-thread CTA per (token, head) and
is launched as a programmatic dependent of the decode grid when `enable_pdl`
is set (`None` = device default; the merge waits for the decode's memory before
its first load, so both settings are bitwise identical). The names are distinct
from the NVFP4 family above so the two generated kernel sets never share a
translation unit, header, manifest or JIT module name. Without the manifest,
`flashinfer.mla.cake_sparse_mla_sm120_dsv41_mixed_format_info()["kernels_available"]`
is `False`, the planners run on the provisional geometry and the first launch
raises `FileNotFoundError` naming the expected manifest location.

Select it through the existing SM120 entry points:

```python
from flashinfer.mla import (
    SparseMLASm120Wrapper,
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    dsv41_fp8_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)

main_cache = dsv41_fp8_quantize_pack_sparse_mla_cache(latent_pages)   # [P, 1, page, 528]
extra_cache = dsv41_fp4_quantize_pack_sparse_mla_cache(extra_pages)   # [P', 1, page', 288]
out = trtllm_batch_decode_sparse_mla_dsv4(
    query=q, swa_kv_cache=main_cache, workspace_buffer=workspace,
    sparse_indices=indices, swa_topk_lens=lengths,
    compressed_kv_cache=extra_cache, extra_sparse_indices=extra_indices,
    extra_sparse_topk_lens=extra_lengths, bmm1_scale=sm_scale,
    backend="cake", kv_cache_format="fp8_dsv41_fp4_ca",
    enable_pdl=None,  # or True / False: merge launch attribute only
)
runner = SparseMLASm120Wrapper(
    backend="cake", kv_cache_format="fp8", kv_scale_format="ue8m0_g32",
    extra_kv_fp4=True, compute_precision="bf16",  # or "fp8" once exported
)
```

Both caches keep their own positive page size and 16-byte-multiple page stride
(`[P, page_bytes]`, `[P, page, bytes]`, HND or NHD views; rows stay packed
inside a page because the footer layout stores `page * data` bytes followed by
`page * scale` bytes). `compute_precision` is explicit: `"default"` and
`"bf16"` select the BF16 route (Q unquantized, both caches dequantized exactly
on chip), `"fp8"` the FP8 route when the family exports it; `"nvfp4"` is
rejected. Split-K scratch is caller-owned or carved from `workspace_buffer`;
size it with `cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes`. The planners
`cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles` /
`cake_sparse_mla_sm120_dsv41_mixed_plan_splits` are pure functions of
`(num_tokens, num_heads, topk, extra_topk, num_sms)` and the manifest
geometry. `backend="auto"` and `backend="sparse"` keep the hand-written SM120
kernels; the main-cache writers `dsv41_fp8_quantize_pack_sparse_mla_cache` /
`dsv41_fp8_quantize_append_sparse_mla_cache` are bit-identical to the torch
reference quantizer used by the FlashInfer tests.
