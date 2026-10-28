# Cake dense NVFP4 MLA decode on Rubin (SM107)

Experimental backend for absorbed multi-head latent attention decode
(Kimi-K3 / DeepSeek-V3 geometry: 512 latent + 64 rope channels per key,
512-wide value) over the NVFP4 paged latent cache written by
flashinfer-ai/flashinfer#4676, on R200 (compute capability 10.7). It serves
dense decode (`q_len = 1`), packed variable-length / MTP queries, optional
natural-log LSE and decode context parallelism, with an NVFP4 query produced
by its own quantizer. Tracking: flashinfer-ai/flashinfer#4644 (Cake tracker
#4254).

Both the API and the backend are experimental: no compatibility guarantee.
The feature does not participate in automatic backend selection, autotuning
or trace-apply. Its libraries are loaded prebuilt from `aot/` when the package
ships them and JIT-built by FlashInfer from `build.json` otherwise (see
"Layout of this package"). **Toolkit requirement: CUDA 13.4 or newer** (the
decode runs in the oversized shared-memory mode and spells Blackwell
instructions of PTX ISA 9.4).

## Public entry points (`flashinfer/mla/_core.py`)

The three public functions dispatch on the device's compute capability:
10.7 comes here, 10.0 / 10.3 to `flashinfer.experimental.cake_mla_nvfp4_paged_decode`.

```python
import torch
from flashinfer.mla import (
    cake_mla_nvfp4_paged_decode,
    prepare_cake_mla_nvfp4_paged_decode,
    quantize_mla_nvfp4_query,
)
from flashinfer.experimental.cake_rubin_mla_nvfp4_decode import (
    max_workspace_bytes, mla_nvfp4_query_buffers, query_workspace_bytes,
)

batch, heads = 8, 96
q = torch.randn(batch, 1, heads, 576, device="cuda", dtype=torch.bfloat16)
query_workspace = torch.empty(query_workspace_bytes(batch * heads), dtype=torch.uint8, device="cuda")
q_nope, q_sf, q_rope, q_scale = quantize_mla_nvfp4_query(
    q, ckv_scale, kpe_scale, out=mla_nvfp4_query_buffers(query_workspace, (batch, 1, heads))
)
workspace = torch.empty(max_workspace_bytes(batch * heads, batch), dtype=torch.uint8, device="cuda")
out = torch.empty(batch, 1, heads, 512, dtype=torch.bfloat16, device="cuda")
runner = prepare_cake_mla_nvfp4_paged_decode(
    q_nope, q_sf, q_rope, q_scale, ckv_cache, ckv_sf_cache, kpe_cache, block_tables, seq_lens,
    out, workspace, sm_scale=1.0 / 192 ** 0.5, ckv_scale=ckv_scale, seq_lens_host=seq_lens_host,
)
runner.launch()   # allocates nothing, synchronizes nothing: capture it in a CUDA graph
```

A captured graph replays through the runner's workspace (the partials and the plan
block): keep the runner, `workspace` and the bound tensors alive while the graph is
replayed.

### Contract

| Tensor | Shape | dtype |
| --- | --- | --- |
| `q_nope` | `[batch, q_len, num_heads, 256]` or `[total_q, num_heads, 256]` | uint8 (packed E2M1) |
| `q_sf` | `[.., 32]` | float8_e4m3fn (block-16 scales) |
| `q_rope` | `[.., 64]` | float8_e4m3fn |
| `q_scale` | `[..]` | float32 (per-row scale) |
| `ckv_cache` | `[num_pages, 64, 256]` | uint8 (packed E2M1) |
| `ckv_sf_cache` | `[num_pages, 64, 32]` | float8_e4m3fn (linear block-16 scales) |
| `kpe_cache` | `[num_pages, 64, 64]` | float8_e4m3fn (rope at `kpe_scale`) |
| `block_tables` | `[batch, max_pages_per_seq]` | int32 |
| `seq_lens` | `[batch]` | int32 (this rank's local KV lengths) |
| `out` | `q.shape[:-1] + (512,)` | bfloat16 |
| `lse` (optional) | `q.shape[:-1]` | float32 (natural log) |

* A decoded key is `e2m1 * sf * ckv_scale | fp8 * kpe_scale`; the logits are
  `sm_scale * ckv_scale * q_scale * (QN . KN + QR . KR)`; `out` is the
  softmax-weighted latent times `o_scale`. The three cache tensors may be
  strided views of one allocation (the 352-byte-per-token
  `[.., 256 | 64 | 32]` buffer): the last axis contiguous, the token stride,
  the page stride (64 token strides) and the base 16-byte multiples. The
  page size is 64 (`ValueError` otherwise). `block_tables[b, 0]` is read even
  for a request without local keys: keep it a valid page; entries past a
  request's live pages are never read.
* Rows are `token * num_heads + head` per request, token-major. Every
  request's `q_len * num_heads` is a multiple of 16 (the epilogue's 16-row
  bulk stores; a head count that is a multiple of 16 covers every query
  length). A packed query gives `cum_seq_lens_q` (`[batch + 1]`); its tokens
  are bottom-right causal against the request's last keys.
* Decode context parallelism: `cp_world` ranks, this one `cp_rank`, local
  key `k` of a request at global position `((k // I) W + r) I + k % I` with
  `I = cp_interleave` (1 by default), and `kv_len_global` the requests'
  global key counts (the causal bound). A row no key is visible to on this
  rank gives `out = 0`, `lse = -inf`; the ranks merge by LSE.
* The host plan reads **host** lengths: `seq_lens_host` (and
  `cum_seq_lens_q_host`, `kv_len_global_host`) as Python sequences. When a
  host list is omitted the call copies the device tensor to the host, a
  synchronization; the prepared runner is the graph-safe path (prepare
  outside capture, `launch()` inside). The plan also takes the device's
  cluster counts `units` (clusters of four it holds at once) and `pairs`
  (pair clusters on the SMs the fours leave), read once per device through
  the occupancy query (`units = MAC(4)`, `pairs = MAC(2) - 2 MAC(4)`) or a
  table by SM count (212 SMs: 46 / 14), both overridable by keyword; they
  shape the cost model and the item order only.
* `workspace_buffer` is a caller-owned uint8 buffer of
  `cake_backend.workspace_bytes(plan)` bytes (exact) or
  `max_workspace_bytes(rows, batch)` (a bound over every plan of a row and
  request count, 64 partial slots per row). It holds the split partials, an
  LSE scratch and the int32 plan block (work-item table, query prefix sums,
  per-row key limits), uploaded once when the runner is prepared. Never share
  one workspace between two live runners. Every buffer the calls write is
  caller-owned: `out` (and `lse` with `return_lse=True`) must be given, and
  the query buffers `quantize_mla_nvfp4_query(..., out=...)` fills are views
  of a uint8 workspace of `query_workspace_bytes(rows)` bytes
  (`mla_nvfp4_query_buffers(workspace, lead)`).
* Repeated calls on the same inputs give bitwise the same outputs: the
  kernels reduce in a fixed order.

## Kernel structure of one call

Three programs share one contract. The **query quantizer** (one warp per
BF16 `[.., 576]` row) forms `q_scale = max(amax(nope) / (6 * 448),
amax(rope) * kpe_scale / (448 * ckv_scale))`, the E4M3 block-16 scales
`amax_block / (6 q_scale)`, the E2M1 codes `x / (sf q_scale)` and the FP8
rope `x * kpe_scale / (q_scale * ckv_scale)`, bit for bit the torch reference.

The **decode** is a persistent mixed-cluster kernel in the shape of a
two-pair cluster of four CTAs: ranks 0-1 hold 128 query rows each in tensor
memory and compute S^T over 128-key tiles (two pages) with two
`kind::mxf4nvf4` K=128 block-scaled MMAs (the cache block scales as the A
scale factors, the query block scales as B's) plus one `kind::f8f6f4` rope
MMA, run the softmax with a per-row anchor and push P (E4M3, with a per-row
UE8M0 scale) into ranks 2-3, which convert V on chip to E4M3 and accumulate
O with `kind::mxf8f6f4` K=64 M256 N256 MMAs under the row's anchor. The launch
requires a cluster of two and prefers one of four: the GPU fills its GPCs
with fours and the SMs they leave hold pairs, every CTA reading its cluster's
size and running the body of that size; the clusters the first wave does not
hold are handed out by cluster launch control. The host plan cuts every
(request, 256-row tile) into key ranges -- the pair of split counts (full
tiles, half tiles whose rows past the first 128 are padding) with the least
modelled makespan over the device's clusters and pairs -- and emits the work
items (full-tile items first). One wave of nearly as many full tiles as
clusters over 1024 key tiles or more also prices a shaved plan -- every tile
a long range and a short one of 48 tiles, the long ranges emitted first and
the short ones left to the clusters the wave does not fill and the pairs as
they free -- and takes it when the model puts it under the equal-length plan
by more than the slack. A plan without half tiles launches the
`decode_full` build (the same kernel without the double-item test at each
item's end); otherwise `decode`. The one plan whose scalars (rows, full-tile
items, items, splits) lie in the `decode_full_pairpipe` record's box launches
that build instead: the full kernel with the pair body's converter pipeline.
An item holding all of its rows' keys writes
`out` / `lse` directly; split items write normalised BF16 partials and
log2-domain partial LSEs into the workspace.

The **split-KV merge** (one program, three warps-per-row instantiations
chosen by the row count: four up to 1024 rows, two up to 2048, one beyond;
up to 64 splits per row) folds the partials and emits the natural-log LSE,
skipping the rows the decode wrote directly; it runs behind the decode when
the plan has more than one split.

Programmatic dependent launch is a per-plan switch (`cake_backend.pdl_for`,
reported as `runner.route_metadata["pdl"]`): when the plan has a split merge
and its decode is one hardware wave (`items <= units + pairs`, the clusters
of four plus the pairs), the decode and the merge launch with the
programmatic attribute (the `enable_pdl` argument of
their `run` entries), so each prologue overlaps its predecessor's tail; the
quantizer triggers its dependents once its rows are stored and the decode once
a role runs out of items, and each kernel syncs on its dependency before
reading the predecessor's output. A plan without a merge, or with more items
than resident clusters (an early-launched merge would take SMs from the
decode's second wave), launches all three plain.

The selection and grid rules are data in `cake_jit.MODULES` (`select_rule`,
`grid_rule`), evaluated by `cake_jit.select_record` / `launch_grid`;
`cake_backend.plan(...)` resolves them once per distinct problem.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program: its
  `library` stem in `build.json`, the stage it fills and its variant, the
  selection and grid rules, the kernel-parameter -> logical-argument map and
  the argument plan of its `run` entry; the derived leaves `kernel_symbol`,
  `launch`, `arg_plan` and `logical` are rewritten by the export), the stage
  order, the logical vocabulary, the loader glue (`library_name`, `load`,
  `specs_for_arch`) and `run_module(name, values, grid=, device_index=,
  enable_pdl=)` (the switch is taken by the decode and merge records, whose
  `launch.use_pdl` is `"runtime"`).
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`): one entry per `<library>_sm_107a` (8 programs) with its
  two translation units, digests, nvcc flags, closure digest and
  `aot/<name>.so` path. `cake_build.load_library(name)` loads the prebuilt
  library when `aot/` holds it and otherwise JIT-builds the entry through
  FlashInfer (`gen_jit_spec`, sources digest-verified first). `aot/` is never
  tracked (`.gitignore`).
* `cake_backend.py` -- operand checks, the host plan (`mla2_plan`, the
  per-row key limits, the device's cluster counts), the workspace layout, the
  prepared decode runner, the query quantizer and the public-API functions
  the routing in `flashinfer/mla/_core.py` calls.
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include,
  shipped by the export so the package compiles without FlashInfer's `csrc/`.
* `csrc/cake_rubin_mla_nvfp4_decode_<digest>_{kernel,binding}.cu` -- the
  generated device and binding translation units of the 6 programs. The
  sources are identity-checked by the digests in `build.json`.

## Status

The registry holds the 6 programs for `sm_107a`, exported from the kernel
snapshot named in the pull request.

Tests: `tests/experimental/test_cake_rubin_mla_nvfp4_decode.py` (the backend
suite: imports only this package, `flashinfer.experimental.cake_rubin_mla_nvfp4_decode`;
the plan and registry tests run everywhere, the kernel tests skip without a
compute capability 10.7 device or before the package is exported) against an
FP32 reference over the decoded NVFP4 operands, and
`tests/experimental/test_cake_rubin_mla_nvfp4_decode_api.py` (the public
routing).
