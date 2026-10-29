# Experimental ragged BF16 MoE grouped GEMM (Cake backend)

Both this API and its Cake backend are experimental and may change or be
removed without compatibility guarantees. The entry points are the core
functions `flashinfer.grouped_mm.grouped_gemm_fwd`, `grouped_gemm_dgrad`,
`grouped_gemm_wgrad` and `cake_grouped_mm`, and the stable API
`flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`; calling one of
them is the explicit opt-in and emits FlashInfer's `ExperimentalWarning` once.
There is no automatic backend selection. Tracking: FlashInfer issue #5678.

Three operations over `E` groups of rows, described by an int32 **device**
tensor `offs[E]` of cumulative end offsets (`offs[-1] == sum_m`; groups may be
imbalanced, empty, one row long, or any size that is not a tile multiple) or,
equivalently, an `m_indptr[E + 1]` with a leading 0 (`torch._grouped_mm` /
`grouped_mm_bf16` form; `grouped_gemm_wgrad` needs `num_groups=E` to tell the
two forms apart). The kernels read the offsets on the device; the host never
synchronises on them.

| helper | computation | inputs | output |
| --- | --- | --- | --- |
| `grouped_gemm_fwd(x, w, offs, out=None)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `x [sum_m, K]`, `w [E, N, K]` bf16 | `Y [sum_m, N]` bf16 |
| `grouped_gemm_dgrad(g, w, offs, out=None)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `g [sum_m, N]`, `w [E, N, K]` bf16 | `dX [sum_m, K]` bf16 |
| `grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]` | `g [sum_m, N]`, `x [sum_m, K]` bf16 | `dW [E, N, K]` bf16 or fp32 |

Alignment: `fwd` needs `N % 256 == 0` and `K % 64 == 0`; `dgrad` needs
`K % 256 == 0` and `N % 64 == 0`; `wgrad` needs `N % 256 == 0` and `K` a
multiple of its k tile (256, or 512 when selected, see below). Outputs may be
row-padded (any leading strides, unit stride along the last dimension); the
padding is never written.

Numerics: fp32 tensor-core accumulation with one bf16 rounding at the store
(the fp32 weight-gradient variant stores the accumulator); bitwise
deterministic (no atomics; the split-K tail sums fp32 partials in a fixed
order); `dW[e] == 0` exactly for empty groups.

## Usage

```python
import torch
from flashinfer.grouped_mm import (
    cake_grouped_mm,
    grouped_gemm_dgrad,
    grouped_gemm_fwd,
    grouped_gemm_wgrad,
)
from flashinfer.experimental.cake_moe_grouped_gemm import prepare_grouped_gemm_wgrad

E, N, K = 4, 4096, 6144
sizes = [28128, 28128, 28128, 28192]
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
x = torch.randn(sum_m, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
g = torch.randn(sum_m, N, dtype=torch.bfloat16, device="cuda")

y = grouped_gemm_fwd(x, w, offs)  # one-shot: prepares and launches
dx = grouped_gemm_dgrad(g, w, offs)
dw = grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32)
y = cake_grouped_mm(x, w, offs)  # autograd wrapper: dX and dW in backward

# Prepare once, launch many (CUDA-graph capturable; new offs / values may be
# written into the bound tensors between launches):
dw_launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32)
dw = dw_launch.launch()
print(dw_launch.record_name, dw_launch.plan)
```

The one-shot functions and `cake_grouped_mm` are imported from
`flashinfer.grouped_mm`: those core entry points carry the experimental-API
marking and the once-only `ExperimentalWarning` and hand off to the plain
functions of the same names in this package (`grouped_gemm_fwd`,
`grouped_gemm_dgrad` and `grouped_gemm_wgrad` in the package root,
`cake_grouped_mm` in `cake_autograd`). A build that vendors the package calls
those plain functions directly; they emit no warning. The prepared-launch
surface below is reached from the package itself.

`prepare_grouped_gemm_*` validates, allocates the output (when not given) and
the weight-gradient workspaces, resolves the route and binds the generated
programs, and returns a `GroupedGemmLaunch`. `launch()` passes the operands'
TMA descriptors by value (the binding encodes them on the host from the bound
tensors' metadata at every launch), performs no allocation and no host
synchronisation. Prepare a new launch when a shape, dtype or tensor binding
changes; the one-shot helpers prepare on every call.

## Stable API opt-in and autograd

The forward projection is also reachable from the stable grouped GEMM API:
`flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")`
(bfloat16 outputs only, no tactic index, `N % 256 == 0`, `K % 64 == 0`; rows
of `a` past `m_indptr[-1]` are left untouched). Naming `backend="cake"` is the
opt-in and emits FlashInfer's `ExperimentalWarning` once, from the core API;
it then hands off to `cake_backend.grouped_mm_bf16_cake`, which rejects the
options the backend does not take and emits no warning itself.

`cake_grouped_mm(x, w, offs, deterministic=True)` is a `torch.autograd.Function`
wrapper (`CakeGroupedMm`): `backward` produces `x.grad` with the `dgrad`
program and `w.grad` with the `wgrad` program in the dtype of `w`. The
programs always reduce in a fixed order, so `deterministic=False` currently
runs the same deterministic programs; `grouped_gemm_wgrad(..., out_dtype=torch.float32)`
gives an fp32 weight gradient. `CakeGroupedMm` calls the three programs as
plain functions; no torch custom op is registered.

## Host plan

Every decision is taken from host-known scalars (shapes, SM count,
architecture), never from `offs`, so one prepared launch serves any group
distribution. `cake_backend.py` reproduces the production planner from the
constants delivered with the generated programs (`HOST_PLAN_CONSTANTS`):

- Grid: one 2-CTA cluster per SM pair, capped by the tile count
  (`persistent_grid`); for `fwd` / `dgrad` the tile count bound lets every
  group add one partial 256-row tile (`max_cluster_tiles_upper_bound`).
- `fwd` / `dgrad` output tile (`fwd_dgrad_tile_n`): 512 columns
  (`wide_block_n`) when the output columns (`N` for `fwd`, `K` for `dgrad`)
  are a multiple of 512 and the architecture is listed in `wide_fwd_archs`
  (`fwd`) or `wide_dgrad_archs` (`dgrad`), 256 (`block_n`) otherwise. The
  512-column programs are the routes' `main_wide` stage; every output element
  accumulates its k steps in the same order in both tiles, so both give the
  same bits. The `sm_107a` `dgrad` program of that stage keeps each G tile in
  the tensor core's A collector for both 256-column halves of the tile.
- `dgrad` L2 evict-first twin (`dgrad_evict_first`): on the architectures
  listed in `dgrad_evict_first_archs`, when the groups average at least
  `dgrad_evict_first_min_rows` rows (`sum_m // E`) and `dX` has at least
  `dgrad_evict_first_min_k` columns (`K`), the wide tile is the route's
  `main_wide_ef` stage: the same program with G loaded and `dX` stored under
  the L2 evict-first policy, so a group's weights stay resident in L2 while
  its row blocks stream past them, and with two head and two tail k steps of
  an item split around the release of the accumulator's two halves instead of
  four and three, so fewer of an item's k steps pay the split. Every element
  still accumulates its k steps in order, so its output is bitwise that of
  `main_wide`.
- `fwd` last-wave-halves evict-first twin (`fwd_lwh_ef`): on the architectures
  listed in `fwd_lwh_ef_archs`, when the reduction `K` (the columns of `x`) is
  at most `fwd_lwh_ef_max_k`, as in the down projection, the wide tile is the
  route's `main_wide_ef` stage: the A-collector program that releases the last
  wave of tiles in two halves, stores Y under the L2 evict-first policy and
  splits two head and two tail k steps of an item around the release of the
  accumulator's two halves instead of four and three. Its output is bitwise
  that of `main_wide`.
- `fwd` / `dgrad` tile order on `sm_100a` / `sm_103a`: the column tiles of
  every 256-row block are split between two claim queues, one per locality
  domain of the device, in proportion to the clusters that claim each queue
  first (`c0 = clusters // 2`, `c1 = clusters - c0`, `clusters = grid_x // 2`);
  a cluster finds its domain on the device and claims from the other queue
  when its own is empty, so every tile is computed exactly once whatever the
  split. The `sm_107a` programs stride the tiles over the clusters in the same
  row-block-major order and ignore the queues.
- Weight-gradient k tile (`select_wgrad_tile`): 512 when `K % 512 == 0`, the
  architecture is listed in `k512_archs` and the average rows per group reach
  the architecture's `k512_min_rows_per_group`; 256 otherwise. The two tiles
  are separate registered programs; the fp32 output is the compile-line
  specialization `-DWGRAD_OUT=32` of the bf16 program.
- Weight-gradient tail: the tiles of the last partial wave are split into
  `tail_splits` k-step chunks over all clusters (`wgrad_tail_tiles`,
  `wgrad_tail_plan`, a closed-form cost model of rounds, partial traffic and one
  launch); when `tail_splits > 1` an ordered fp32 tail-reduce kernel
  (`wgrad_reduce_grid` CTAs) sums the chunks, otherwise only the main kernel
  runs. The tile raster inside a group follows the per-wave operand footprint
  (`wgrad_raster_rows`).
- Weight-gradient wave sync: when `K` reaches `wave_sync_min_k` (the
  gate-up-shaped rows), `wave_sync_archs` lists the architecture for the
  selected k tile and the route has a `main_wave` stage on it, the main kernel
  is the wave-synchronized build, launched with the architecture's
  `wave_sync_mode` as `wave_mode`; otherwise the plain build (`wave_mode = 0`).
  Both builds give the same output bits for a given route.
- Weight-gradient accumulator release: with the 512-wide k tile on the
  architectures listed in `k512_release_archs`, the main kernel is the build
  that hands the tensor-memory accumulator to the epilogue in two halves
  (stage `main_halves`, or `main_wave_halves` for the wave-synchronized build),
  so the next tile's MMAs start on the first half while the epilogue drains the
  second. Its output is bitwise that of the build that drains the whole
  accumulator at once. The `main_halves` program also keeps each A tile in the
  tensor core's A collector; on `sm_100a` and `sm_103a` it also walks the tiles
  in an empty-aware order that the kernel plans itself (no host input). On
  `sm_100a` and `sm_103a` the `main_wave_halves` program keeps the A tile in
  the collector and synchronizes the waves every eight waves. None of these
  changes the output bits.
- Weight-gradient k256 tail (`wgrad_k256_tail`): on the architectures listed
  in `k256_tail_archs`, when `K` (the columns of `dW[e]` and of `x`) is at
  most `k256_tail_max_k` and the groups average at least `k256_tail_min_rows`
  rows (`sum_m // E`), as in the down projection over few groups, the main
  kernel is the route's `main_halves_k256t` stage instead of `main_halves`:
  the A-collector half-release kernel runs only the full waves of the 512-wide
  tile, then the tiles left as the 256-column half tiles of the 256-wide
  tiling, walked n-major (`raster_rows` 0). The last partial wave of that
  tiling is split as the 256-wide k tile splits its tail (`tail_tile_k` 256 in
  `wgrad_tail_tiles` / `wgrad_tail_plan`): each half tile runs as
  `tail_splits` k-step chunks whose fp32 partials, 256 x 256 slots
  `j * tail_splits + s`, the tail-reduce stage of the 256-wide tile's route
  sums (`wgrad_reduce_grid` CTAs). Every element accumulates its rows in the
  order of the 256-wide k tile, so where the two plans split the same tiles
  the same way the output bits are that tile's.
- Workspaces (allocated at preparation, retained by the launch object): the
  per-CTA TMA descriptor slots of the weight gradient (a ring of nine
  descriptor pairs per CTA: a group switch edits the next pair and warms the
  one after it, never a pair an in-flight load may still read) and its fp32
  tail partials. The operand
  descriptors are launch arguments; there is no descriptor workspace and no
  descriptor upload kernel.
- Stream workspaces (kept per device and CUDA stream, reused by every launch
  on that stream): the `fwd` / `dgrad` scheduler words (`sched`,
  `sched_words` u32 words whose first three rest at `sched_rest`) and the
  wave-sync counters (`wave_sync`, `wave_sync_words` u32 words resting at
  zero). The first launch on a stream allocates a buffer at its rest state;
  every kernel leaves it at its rest state, so launches on one stream share it
  and launches on two streams never do. A launch under CUDA graph capture, or
  for a device that is not the current one, takes buffers of its own, set to
  their rest state at the launch. `stream_workspace(kind, device, stream=None)`
  returns the buffer kept for a stream, or `None` before a launch on it needed
  one.

`GroupedGemmLaunch.plan` exposes `grid`, `tile_k`, `tail_splits`,
`raster_rows`, `launches` and, for `fwd` / `dgrad`, `tile_n`, `clusters`, `c0`
and `c1`, for `wgrad` `tail_tile_k` (the k width of the split tail tiles) and
`wave_mode`, for inspection and tests; `GroupedGemmLaunch.stages`
names the bound stages in launch order.

## Build targets and layout

The programs are exact-architecture tcgen05 payloads built for the exact
targets `sm_100a`, `sm_103a` and `sm_107a` (`build.json` `arch_flags`).
`cake_jit.PROGRAMS` lists twenty libraries, one per kernel and build, each a
kernel and a host-binding translation unit under `csrc/cake_moe_grouped_gemm/`
shared by the architectures it is built for (an architecture-dependent line is
folded under `__CUDA_ARCH__`):

- `fwd` and `dgrad` in five builds each: the domain-scheduled build for
  `sm_100a` and `sm_103a` and the static-schedule build for `sm_107a`, each
  with the 256-column output tile (the routes' `main` stage) and with the
  512-column one (`main_wide`), and the `sm_107a` evict-first twin of the
  512-column build (`main_wide_ef`). A route spells the two builds of a stage as
  two stages of one name on disjoint architectures, and
  `cake_jit.stage_binding(name, stage, arch)` returns the one built for the
  architecture.
- the wave-synchronized weight-gradient main kernel with a 256- or 512-wide
  k tile, for `sm_100a`, `sm_103a` and `sm_107a` (the routes' `main_wave`
  stage).
- the 512-wide k tile's A-collector main kernel that releases the
  accumulator in two halves (`main_halves`): with the planned tile walk for
  `sm_100a` and `sm_103a`, without it for `sm_107a`; for `sm_107a` its twin
  that runs the tiles left after the full waves as 256-column half tiles
  (`main_halves_k256t`); the wave-synchronized A-collector kernel that
  releases it in two halves (`main_wave_halves`), for `sm_100a` and `sm_103a`.
- the 512-wide k tile's main kernel for `sm_107a`, a second `main` stage
  beside the one built for `sm_100a` and `sm_103a`.
- the weight-gradient main kernel with a 256- or 512-wide k tile and the
  weight-gradient tail-reduce, as the previous delivery shipped them.

The host launches a stage only where the constants above list the
architecture, so a library may be built for an architecture its stage is not
launched on.

A specialization is a compile-line definition on its `build.json` entries:
the tail-reduce k tile (`-DWGRAD_TILE=256|512`) and the fp32 weight-gradient
output (`-DWGRAD_OUT=32` on the kernels of the last list item,
`-DCAKE_WGRAD_OUT=32` on the other weight-gradient kernels; the bf16 output is
the default). `cake_jit.ROUTES` maps each public route (operation, output dtype, k tile) to its stage programs
and specializations. `build.json` (schema `cake-build/1`, written by the
export) is the only record of what to compile: one entry per build
`<library>[<specialization tag>]_<arch>` -- the tag is `_wgrad_tile256` /
`_wgrad_tile512` for the specialized tail-reduce builds, `_wgrad_out32` for the
fp32 weight-gradient builds (composed on the fp32 tail-reduce:
`_wgrad_out32_wgrad_tile256` / `_wgrad_out32_wgrad_tile512`), empty otherwise
-- with its translation units, digests, nvcc flags (the `-D` defines
included), per-architecture closure digest and `aot/<name>.so` path.
`cake_jit.load_module(program, arch, specializations)` loads that entry
through the shared `cake_build.py` loader: the prebuilt library when `aot/`
holds it, else FlashInfer's JIT build (the cache name carries the exact target
and the closure digest; FlashInfer's build-target rules --
`FLASHINFER_CUDA_ARCH_LIST` or the visible devices -- apply to that path
only). The bindings include the package's `csrc/cake_ffi_checks.h`, no
FlashInfer header.

The kernel translation units of the first four list items are
self-contained. Programs whose bindings differ only in the kernel they launch,
its block size and its dynamic shared memory share one binding, which a
member's entries compile under `-DCAKE_KERNEL_SYMBOL`, `-DCAKE_PROGRAM_SHIM`
and, where they differ from the owner's, `-DCAKE_BINDING_THREADS` /
`-DCAKE_BINDING_SMEM`. The previous weight-gradient and tail-reduce kernels
share one device header, `cake_moe_grouped_gemm_device.cuh` (their common preamble:
typedefs, tensor-map ABI, math constants and the tcgen05 helpers behind
need-guards), and their bindings share the dynamic shared-memory opt-in and
TMA descriptor encoders through the fragment headers
`cake_moe_grouped_gemm_binding_*.h` included inside each host-shim namespace.
These headers are hand folds of exported text that no `build.json` entry
lists, so the build's dependency tracking, not a manifest digest, covers them.

Measured on B200 (`sm_100a`), GB300 (`sm_103a`) and R200 (`sm_107a`); the
performance summary is in the delivery pull requests.

## Standalone import

The package imports only itself, `torch` and `tvm_ffi`; it imports nothing
from the rest of FlashInfer, so a copy of it loads without a FlashInfer
installation (under another top-level name, or with
`sys.modules["flashinfer"] = None` as a vendoring gate sets it). The only
`flashinfer` import in the package is the loader's JIT fallback in
`cake_build.py`, inside `try/except ImportError`, reached after import when a
library is not prebuilt under `aot/`. The experimental warnings and the API
logging belong to the core entry points (`flashinfer.grouped_mm`), so a direct
call of the package's functions emits none of them.

## Tests and benchmark

```bash
pytest tests/experimental/test_cake_moe_grouped_gemm.py tests/experimental/test_cake_moe_grouped_gemm_api.py tests/experimental/test_cake_moe_grouped_gemm_standalone.py
python benchmarks/bench_cake_moe_grouped_gemm.py --help
```

`test_cake_moe_grouped_gemm.py` (the backend suite, which imports the package
through `tests.test_helpers.cake_package`) compares against a per-group fp64
reference (including an empty group, a one-row group, padded outputs,
determinism and the fp32 output) and skips when no supported device or no
registered program is present. `test_cake_moe_grouped_gemm_api.py` covers the
core entry points of `flashinfer.grouped_mm` (the experimental marking, the
once-only warnings and the `backend="cake"` route of `grouped_mm_bf16`).
`test_cake_moe_grouped_gemm_standalone.py` copies the package under another
name into a CPU subprocess with `flashinfer` blocked and checks that it
imports, raises its host-side errors without any warning and has its only
`flashinfer` import in the JIT fallback. The benchmark times the prepared
launches against `torch._grouped_mm`, a per-group cuBLAS loop and
`flashinfer.grouped_mm.grouped_mm_bf16` on the MoE projection rows of the
design (E = 4 ragged distributions, E = 8 / 16 / 32 balanced, fp32 weight
gradients).

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N`, `K` and the output span must fit
  32-bit element addressing.
- The alignment rules above; no transposed operand layouts other than the
  documented in-place `W` read of `dgrad`.
- Compute capability 10.0, 10.3 or 10.7 only; other devices raise.
