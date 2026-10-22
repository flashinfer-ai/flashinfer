# Cake QSA sparse paged GQA decode over an NVFP4 KV cache (SM80 through SM90)

Experimental backend of `flashinfer.sparse.qsa_sparse_paged_attention`: gated
sparse paged GQA decode attention (Qwen QSA: one query token per row, 256-dim
heads, a per-row selection of logical tokens, `bf16(attention) *
sigmoid(output_gate)`) over the packed E2M1 KV cache with E4M3 block scales
that `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` writes, on
compute capability 8.x (the `sm_80` libraries) and 9.0 (`sm_90a`).

Both the API and the backend are experimental: no compatibility guarantee.
The feature does not participate in automatic backend selection, autotuning
or trace-apply. Its libraries are loaded prebuilt from `aot/` when the package
ships them and JIT-built by FlashInfer from `build.json` otherwise (see
"Layout of this package"). Toolkit requirement: CUDA 12.8 or newer
(`build.json` `min_cuda`; FlashInfer's supported toolkits apply).

## Public entry point (`flashinfer/sparse.py`)

```python
import torch
from flashinfer.sparse import qsa_sparse_paged_attention

out = qsa_sparse_paged_attention(
    q, k_cache, v_cache, logical_indices, block_table, token_to_req, False,
    out, k_scale, v_scale, output_gate=gate, k_sf=k_sf, v_sf=v_sf,
    backend="cake",
)
```

`backend="default"` (the default) runs the existing implementation; `"cake"`
runs this package; any other value raises `ValueError`. The arguments, their
checks and the result contract are the public entry's (its docstring). The
package's own entry, `flashinfer.experimental.cake_qsa_nvfp4.qsa_sparse_paged_attention`,
takes the same arguments plus the keyword `workspace_buffer` (below).

### Contract

| Tensor | Shape | dtype |
| --- | --- | --- |
| `q` | `[rows, num_qo_heads, 256]` | bfloat16 (8-byte aligned, contiguous last dim) |
| `k_cache`, `v_cache` | `[num_blocks, page_size, num_kv_heads, 128]` | uint8 (packed E2M1; any strides with a contiguous last dim, 16-byte aligned; NHD and HND views) |
| `k_sf`, `v_sf` | `[num_blocks, page_size, num_kv_heads, 16]` | float8_e4m3fn (same stride rule) |
| `logical_indices` | `[rows, selection_width + 1]` | int32 (the last column is the row's valid-entry count; `-1` entries and unmapped pages attend nothing) |
| `block_table` | `[num_requests, max_blocks]` | int32 |
| `token_to_req` | `[rows]` | int32 (contiguous) |
| `output_gate` | `[rows, num_qo_heads, 256]` | bfloat16 (contiguous) |
| `out` | `[rows, num_qo_heads, 256]` | bfloat16 (optional; 16-byte aligned, contiguous last dim) |

`num_qo_heads` is a multiple of `num_kv_heads`, at most 16 times it.
`k_scale` and `v_scale` are the NVFP4 global scales of K and V.

## Host plan (`cake_backend.py`)

The selection is cut into tiles of 64 tokens and the tiles into `splits`
splits; one decode CTA (128 threads) serves one (row, KV head, split).

**Split rule.** `tiles = ceil(selection_width / 64)`;
`splits = max(1, min(tiles, 2 * sm_count // (rows * num_kv_heads)))` (two
resident CTAs per SM); then the measured table `SPLIT_OVERRIDES` of the
device's SM count replaces the formula where it names `(sm_count, rows)`
(`(132, 4) -> 16`, `(132, 8) -> 8`, `(108, 64) -> 4`, `(108, 128) -> 2`);
then `splits <= min(tiles, 128)`. The rule reads the SM count alone.

**Form rule.** With one split the plain decode writes `out` itself. Above
one split, on compute capability 9.0 the call takes the *fused* form (the
cooperative build of the decode, whose CTAs merge the splits in the same
launch through a counter per (row, KV head)) when `rows * num_kv_heads *
splits` does not exceed the co-residency the decode's launch bound guarantees
(two CTAs per SM times the SM count; 264 on an H100); otherwise, and
always on compute capability 8.x, the *split* form: the plain decode, then
the separate merge (`(rows, ceil(num_qo_heads / heads_per_cta))` CTAs, one
head per CTA above four splits, four at or below).

**Workspace.** `partial_out` FP32 `[splits, rows, num_qo_heads, 256]`,
`partial_lse` FP32 `[splits, rows, num_qo_heads]` and the fused form's
counters uint32 `[rows * num_kv_heads]`. With `workspace_buffer` (a
contiguous uint8 tensor on the device of `q`, of at least
`qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)` bytes,
256-byte aligned; checked on every call that passes one) they are carved
from the caller's buffer: the call allocates nothing and is CUDA-graph
capturable with a replay-stable layout. The counters are zeroed once, the
first time a buffer's storage is seen for a `rows * num_kv_heads` (a new
buffer at a freed buffer's address is a new storage and is zeroed again), and
reset themselves after every launch; `reset_workspace(buffer)` forgets a
buffer whose bytes another use overwrote. Without a workspace the partials and
counters are allocated per call, as the default implementation does; that
path is outside the zero-allocation guarantee (a CUDA graph still captures
it through its private memory pool).

Nothing on the host reads tensor contents; every dispatch is on host scalars
and tensor metadata.

## Kernel structure of one call

- `decode` (plain build): grid `(rows, num_kv_heads, splits)`, 128 threads, a
  49,792-byte shared-memory pool passed as dynamic shared memory (opt-in above
  48 KB), `__launch_bounds__(128, 2)`. Each CTA gathers its
  tiles' K / V rows through a two-stage `cp.async` ring, computes the scores
  and the normalised partial output of its group's heads in FP32 and, above
  one split, writes `partial_out` / `partial_lse`; with one split it writes
  the gated BF16 output.
- `decode_coop` (cooperative build): the same decode launched cooperatively;
  after its partials each CTA takes a slice of its (row, KV head) group's
  output, waits on the group's counter and merges every split's partial for
  that slice, writing the gated BF16 output.
- `merge`: `out[row, head] = bf16(bf16(merged) * sigmoid(gate))` where
  `merged` is the LSE-weighted mean of the splits' partials; split-parallel
  over four warps above four splits, one warp per head at or below.

Every kernel is compiled with `--use_fast_math` (recorded on every
`build.json` entry); `--fmad=true` is nvcc's default.

## Layout of this package

| File | Role |
| --- | --- |
| `cake_jit.py` | registry (`MODULES`: one record per program, host plan + derived leaves), the rules as data (`select_records`, `launch_grid`), `run_module` |
| `cake_backend.py` | the host plan: checks, split rule, form rule, workspace rule, launch sequence |
| `build.json` | the manifest: one library per (program, architecture), sources with digests, nvcc flags |
| `cake_build.py` | the loader: `aot/` prebuilt libraries first, else FlashInfer's JIT from `build.json` |
| `csrc/` | per program: `<library>_kernel.cu` (device unit), `<library>_binding.cu` (tvm-ffi `run`); `cake_ffi_checks.h` |
| `aot/` | optional prebuilt `<library>_<arch>.so` + `build_info.json` (untracked) |

The registry's derived leaves and everything under `csrc/`, `build.json` and
`cake_build.py` are generated; change their producer, not the files.

## Status

Experimental. Validated against the default implementation's test suite with
`backend="cake"` on compute capability 9.0 (`sm_90a`, H100). The `sm_80`
libraries build from `build.json` but have not been run on a compute
capability 8.x device.
