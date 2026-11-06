# Cake QSA sparse paged GQA decode over an NVFP4 KV cache (SM80 through SM120)

Experimental backend of `flashinfer.sparse.qsa_sparse_paged_attention`: gated
sparse paged GQA decode attention (Qwen QSA: one query token per row, 256-dim
heads, a per-row selection of logical tokens, `bf16(attention) *
sigmoid(output_gate)`) over the packed E2M1 KV cache with E4M3 block scales
that `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` writes, on
compute capability 8.x (the `sm_80` libraries), 9.0 (`sm_90a`), 10.0
(`sm_100a`: B200 / GB200), 10.3 (`sm_103a`: B300 / GB300), 10.7 (`sm_107a`:
R200) and 12.0 (`sm_120a`: RTX PRO 6000 Blackwell). Two decode programs
partition them: the `mma.sync` decode of 8.x / 9.0 / 12.0 and the tcgen05
decode of 10.0 / 10.3 / 10.7 (tensor memory, one CTA per SM); the merge serves
all. Compute capability 12.0 has no tcgen05 instructions and no tensor memory,
so the SM80 `mma.sync` program is built for `sm_120a`: the one Blackwell
architecture that program serves.

Both the API and the backend are experimental: no compatibility guarantee.
The public entry's `backend="auto"` (its default) selects this package on
every compute capability above 9.0 it exports a library for (capability-exact:
12.1 is not included); the feature does not participate in autotuning or
trace-apply. Its libraries are loaded prebuilt from `aot/` when the package
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
)
```

`backend="auto"` (the default) runs this package on the exported compute
capabilities above 9.0 (10.0, 10.3, 10.7, 12.0) and the existing implementation on 8.x
and 9.0 and on any other device, which it then reports as unsupported; `"cake"` selects this package and `"default"` the existing
implementation explicitly; any other value raises `ValueError`. This package's
results agree with the existing implementation within the test suites'
tolerances but are not bit-identical to it: callers that need the existing
numerics pass `backend="default"`. The arguments, their
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
splits; one decode CTA serves one (row, KV head, split): 128 threads on the
`mma.sync` decode (8.x / 9.0 / 12.0), 512 on the tcgen05 decode (10.0 / 10.3 / 10.7).

**Split rule.** `tiles = ceil(selection_width / 64)`;
`splits = max(1, min(tiles, CTAS_PER_SM[arch] * sm_count // (rows *
num_kv_heads)))` (the resident CTAs per SM of the architecture's plain decode,
its launch bound: two for the `mma.sync` decode, one for the tcgen05 decode,
whose tensor memory and shared-memory pool hold the SM); then the measured
table `SPLIT_OVERRIDES` of the device's SM count replaces the formula where it
names `(sm_count, rows)` (`(132, 4) -> 16`, `(132, 8) -> 8`, `(108, 64) -> 4`,
`(108, 128) -> 2`, `(188, 4) -> 16`, `(188, 8) -> 11`, `(188, 16) -> 5`,
`(188, 32) -> 3`, `(188, 64) -> 1`; no entry for a tcgen05 device's SM count); then
`splits <=
min(tiles, 128)`. The rule reads the architecture and the SM count alone. On a
B200 (148 SMs) at the owner's 24 x 2 heads and 2051 selections the formula
gives 33, 33, 18, 9, 4, 2, 1, 1 splits for rows 1, 2, 4, 8, 16, 32, 64, 128;
on an RTX PRO 6000 (188 SMs, two CTAs per SM) it gives 33, 33, 33, 23, 11, 5,
2, 1 and the table replaces rows 4, 8, 16 and 32 with 16, 11, 5 and 3, and rows
64 with 1 (one whole-tile CTA of 33 tiles per (row, KV head), no merge: +5 to
+13 % whole call over two splits, measured on three units, both layouts, both L2
flushes).
Where the persistent walk serves the call (compute capability 10.3 and 10.7,
program rule below) the split count is instead the walk's partial slots per
(row, KV head) -- 2 for rows 128 on a GB300, 2 for rows 128 and 256 on an
R200 -- and the merge reads that many.

**Form rule.** With one split the plain decode writes `out` itself. Above
one split the *fused* form (the cooperative build of the decode, whose CTAs
merge the splits in the same launch through a counter per (row, KV head))
needs `rows * num_kv_heads * splits` within the co-residency the decode's
launch bound guarantees (`FUSED_CTAS_PER_SM[arch]` times the SM count: two per
SM on compute capability 9.0 and 12.0, 264 on an H100, 376 on an RTX PRO 6000;
one per SM on 10.0 / 10.3 / 10.7, 148 on a B200, 152 on a GB300, 212 on an R200
board of 212 SMs). Within it, compute capability 9.0 always takes the fused
form, and 12.0 does except on a *wide* call (the program rule below: the
whole-tile program ships in the split form alone, so a wide call takes the
split form there). Compute capability 10.0 and 10.3 take it under the eager host rule
alone: on a host of `FUSED_HOSTS`, at `rows <= 4` (`FUSED_MAX_ROWS`) and when
the current stream is not capturing -- and `FUSED_HOSTS` lists no host, so
they keep the split form everywhere (below). Compute capability 10.7 keeps
the split form on every host (`FUSED_SPLIT_ONLY_ARCHES`): on an R200 system the
split route's eager whole call is the shorter one at every row. Everywhere else -- compute
capability 8.x, a capturing stream, a call on a
device other than the current one -- the *split* form: the plain decode, then
the separate merge (`(rows, ceil(num_qo_heads / heads_per_cta))` CTAs, one
head per CTA above four splits, four at or below).

Why `FUSED_HOSTS` is empty: the rule once named the `aarch64` host of a GB200 /
GB300, where an event-timed measurement (CUDA events around each call, an L2
flush between samples) had the fused form 1.1-1.6x faster whole-call than the
split form at rows 1-4. That measurement timed the gap the split form's
second launch leaves on the GPU but not the fused form's longer host path
before its launch. Timed host-inclusively on a GB300 -- calls back to back,
calls behind an L2 flush enqueued in the stream, and lone synchronous calls --
the split form's whole call is the shorter one at rows 1-4 (1.07-1.14x,
1.21-1.40x and 1.08-1.14x) and the two forms tie at rows 8-128, so 10.0 and
10.3 take the split form on every host. CUDA-graph replay is unchanged (a
captured call takes the split form either way), and the cooperative build
and the rule stay for a host that measures otherwise. The two forms agree
bit for bit: the graph tests compare an eager call with its captured replay
at `atol=0`.

**Program rule.** The records are filtered by the device's architecture
first: the `mma.sync` decode's records on 8.x / 9.0 / 12.0, the tcgen05
decode's on 10.0 / 10.3 / 10.7. On compute capability 8.x the plain decode has
two programs: a
split-form call is *wide* when both hold: its decode grid `rows *
num_kv_heads * splits` exceeds one resident wave (two CTAs per SM times the
SM count; 216 on an A100) and each of its CTAs walks a long tile stream,
`ceil(tiles / splits) >= 16` (1024 selected tokens). A wide call launches
`decode_whole_tile`, the previous release's `sm_80` program (whole-tile
gather, no L2 cache hint, shift selector); every other call launches the
K-first program `decode`, and compute capability 9.0 always does. On compute
capability 12.0 a split-form call is *wide* when each of its CTAs walks at
least 11 tiles (704 selected tokens), whatever the grid (an RTX PRO 6000's 376
resident CTAs hold every owner grid within one wave): it launches the
`sm_120a` build of `decode_whole_tile`, whose copies carry the L2 evict-first
policy (the form rule takes the split form on a wide call); below that stream
the K-first program. On an
A100 (108 SMs) at the owner's 24 x 2 heads and 2051 selections (33 tiles):
rows 1-32 run at most one wave and take the K-first program; rows 64 (4
splits, 512 CTAs, 9 tiles per CTA) take the K-first program; rows 128 (2
splits, 512 CTAs, 17 tiles per CTA) and the 2048-row single-split call (33
tiles per CTA) take the whole-tile program. Measured paired on two A100s
(80 GB and 40 GB PCIe) under a memset flush, the K-first program is 1-7 %
faster at rows 64 and the whole-tile program 6-8 % faster at rows 128, on
both units. On an RTX PRO 6000 (188 SMs) at the same heads and selections,
rows 32 (3 splits, 11 tiles per CTA), 64 and 128 (1 split, 33 tiles) are wide
and rows 1-16 (at most 7 tiles per CTA) are not; measured paired
there under both L2 flushes, the whole-tile program is 2-9 % faster whole call
at rows 32 / 64 / 128 and within noise or slower at rows 16 and below. The
rule reads the architecture, the SM count and the call's tile and split
counts alone. 10.0,
10.3 and 10.7 launch `decode_tcgen05` in the split form and `decode_tcgen05_coop`
in the fused form, by the form rule alone -- except that 10.3 and 10.7 launch
the *persistent walk* `decode_tcgen05_persistent` on a split-form grid whose
plain cut runs more than one resident wave within the architecture's measured
range (`1 < ceil(rows * num_kv_heads * splits / sm_count) <=
WALK_MAX_WAVES[arch]`: 2 on 10.3, 3 on 10.7) and whose longest walked CTA is
shorter than the plain program's longest: one CTA per SM walking a balanced,
contiguous share of the (row, KV head) x tile slots. On a GB300 (152 SMs) at
the owner's shape that is rows 128 alone: 256 plain CTAs of 33 tiles in two
waves become 152 CTAs of 56-57 tiles with 2 partial slots per pair, 5.6-7.1 %
faster whole call in every mode; rows 1-64 (one wave) and rows 256 and above
(more than two waves) keep the plain cut. On an R200 (212 SMs) it is rows 128
(256 plain CTAs in two waves, the second 44 wide, become 212 CTAs of 39-40
tiles with 2 slots per pair: 19-21 % faster whole call in every mode) and rows
256 (512 plain CTAs in three waves become 212 CTAs of 79-80 tiles with 2
slots: 8-9 % faster); rows 1-64 (one wave) and rows 318 and above (more than
three waves) keep the plain cut. So does 10.0 everywhere, where the same
two-wave grids tie on a B200. On an
A100 (108 SMs) at the owner's 24 x 2 heads and 2051 selections (33 tiles):
rows 1-32 run at most one wave and take the K-first program; rows 64 (4
splits, 512 CTAs, 9 tiles per CTA) take the K-first program; rows 128 (2
splits, 512 CTAs, 17 tiles per CTA) and the 2048-row single-split call (33
tiles per CTA) take the whole-tile program. Measured paired on two A100s
(80 GB and 40 GB PCIe) under a memset flush, the K-first program is 1-7 %
faster at rows 64 and the whole-tile program 6-8 % faster at rows 128, on
both units. The rule reads the architecture, the SM count and the call's
tile and split counts alone.

**Workspace.** `partial_out` FP32 `[splits, rows, num_qo_heads, 256]`,
`partial_lse` FP32 `[splits, rows, num_qo_heads]` and the fused form's
counters uint32 `[rows * num_kv_heads]`. With `workspace_buffer` (a
contiguous uint8 tensor on the device of `q`, of at least
`qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)` bytes,
256-byte aligned) they are carved from the caller's buffer -- checked (and,
above one split, carved) on a tensor's first call, and from its second call
on once per call shape into a plan that every later call passing the same
tensor reuses (a tensor that dies or is rebound to other bytes, or
`reset_workspace`, drops the plan): the call
allocates nothing and is CUDA-graph capturable with a replay-stable layout. The counters are zeroed once, the
first time a buffer's storage is seen for a `rows * num_kv_heads` (a new
buffer at a freed buffer's address is a new storage and is zeroed again), and
reset themselves after every launch; `reset_workspace(buffer)` forgets a
buffer whose bytes another use overwrote. Without a workspace the partials
come from a module scratch per (device, stream) -- one float32 buffer, grown
when a call needs more, its views kept per call shape -- and the fused form's
counters from a module buffer per (device, stream): zeroed when it is
allocated, left idle by every launch, grown on demand. Calls on one stream are
ordered, so no two launches share live partials or counters; a module lock
orders two Python threads enqueuing on one stream from the carve through the
call's last launch. A capturing stream allocates its partials and counters per
call instead (the graph's private pool) and leaves both caches untouched. That
path is outside the zero-allocation guarantee.

Nothing on the host reads tensor contents; every dispatch is on host scalars
and tensor metadata. The launch plan of a call shape -- split count, form,
program, loaded FFI entries, argument order and grids -- is resolved once per
shape and device (`cake_jit.resolved_launches`); a call checks its arguments,
binds them in one flat sequence and issues its launches under one
environment-stream context.

## Kernel structure of one call

- `decode` (plain build): grid `(rows, num_kv_heads, splits)`, 128 threads, a
  49,792-byte shared-memory pool passed as dynamic shared memory (opt-in above
  48 KB), `__launch_bounds__(128, 2)`. Each CTA gathers its
  tiles' K / V rows through a two-stage `cp.async` ring, computes the scores
  and the normalised partial output of its group's heads in FP32 and, above
  one split, writes `partial_out` / `partial_lse`; with one split it writes
  the gated BF16 output.
- `decode_whole_tile` (plain build, `sm_80` and `sm_120a`): the previous
  release's decode program with the same parameters, grid, launch bound and
  shared-memory pool; it gathers each tile's K / V rows whole and selects the
  E2M1 nibbles by shift; the `sm_80` build issues its copies without an L2
  cache hint, the `sm_120a` build with the evict-first policy. The program
  rule launches it on the wide calls: on `sm_80` above one resident wave with
  at least 16 tiles (1024 selected tokens) per CTA, on `sm_120a` at least 11
  tiles (704 selected tokens) per CTA whatever the grid.
- `decode_coop` (cooperative build): the same decode launched cooperatively;
  after its partials each CTA takes a slice of its (row, KV head) group's
  output, waits on the group's counter and merges every split's partial for
  that slice, writing the gated BF16 output.
- `decode_tcgen05` (plain build, `sm_100a` / `sm_103a` / `sm_107a`): the
  tcgen05 decode with the same parameters, grid and outputs; 512 threads in six roles
  (softmax, K transform, V transform, gather, two MMA issuers), a four-stage
  `cp.async` raw ring, K dequantized into tensor memory and V into shared
  memory as BF16, `S^T = K Q^T` and `O^T = V^T P^T` on `tcgen05.mma`, the same
  64-token online softmax and rounding points as the `mma.sync` decode; its
  shared-memory pool as dynamic shared memory, `__launch_bounds__(512, 1)`:
  one CTA per SM.
- `decode_tcgen05_coop` (cooperative build, `sm_100a` / `sm_103a` /
  `sm_107a`): the tcgen05 decode launched cooperatively, its softmax warps running the same
  counter and slice merge as `decode_coop`; one CTA per SM. The host selects
  it under the form rule's eager rule alone (10.0 / 10.3 on a host of `FUSED_HOSTS`,
  `rows <= 4`, no capture; never on 10.7), and that list is empty: every 10.0 / 10.3 /
  10.7 call takes `decode_tcgen05` (the build ships for a host that measures otherwise).
- `decode_tcgen05_persistent` (persistent build, `sm_103a` / `sm_107a`): the tcgen05
  decode's body on a grid of `walk_ctas = min(sm_count, rows * num_kv_heads *
  tiles)` CTAs, each walking a contiguous range of the (row, KV head) x tile
  slots and writing one partial slot per pair segment (`num_splits` is the
  most segments any pair has, `walk_splits`); Q^T double-buffered across
  pairs, the raw ring live across the walk, a 183,680-byte pool (192,896 B of
  dynamic shared memory as launched). The host
  selects it on 10.3 for two-wave and on 10.7 for two- and three-wave
  split-form grids (the program rule).
- `merge`: `out[row, head] = bf16(bf16(merged) * sigmoid(gate))` where
  `merged` is the LSE-weighted mean of the splits' partials; split-parallel
  over four warps above four splits, one warp per head at or below. On
  `sm_100a` / `sm_103a` / `sm_107a` the split-parallel form prefetches a wider
  first batch.

Every kernel is compiled with `--use_fast_math` (recorded on every
`build.json` entry); `--fmad=true` is nvcc's default.

## Layout of this package

| File | Role |
| --- | --- |
| `cake_jit.py` | registry (`MODULES`: one record per program, host plan + derived leaves; a record lists the architectures it ships for), the rules as data (`select_records`, `launch_grid`), the launch plan resolved once per shape (`resolved_launches`), `run_module` |
| `cake_backend.py` | the host plan: checks, split rule, form rule, program rule, workspace rule, launch sequence |
| `build.json` | the manifest: one library per (program, architecture), sources with digests, nvcc flags |
| `cake_build.py` | the loader: `aot/` prebuilt libraries first, else FlashInfer's JIT from `build.json` |
| `csrc/` | per program: `<library>_kernel.cu` (device unit), `<library>_binding.cu` (tvm-ffi `run`); `cake_ffi_checks.h` |
| `aot/` | optional prebuilt `<library>_<arch>.so` + `build_info.json` (untracked) |

The registry's derived leaves and everything under `csrc/`, `build.json` and
`cake_build.py` are generated; change their producer, not the files.

## Status

Experimental. Validated against the default implementation's test suite with
`backend="cake"` on compute capability 9.0 (`sm_90a`, H100) and 8.0 (`sm_80`,
A100 80 GB and 40 GB PCIe); the tcgen05 programs (`sm_100a`, `sm_103a`,
`sm_107a`) against the same suite on a B200 (10.0), a GB300 (10.3) and an R200
(10.7); the `sm_120a` build of the `mma.sync` programs against the same suite
on an RTX PRO 6000 Blackwell Server Edition (12.0).
