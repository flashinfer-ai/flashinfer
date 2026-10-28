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
- Weight-gradient k tile (`select_wgrad_tile`): 512 when `K % 512 == 0`, the
  architecture is listed for it and the average rows per group reach the
  delivered threshold; 256 otherwise. The two tiles are separate registered
  programs; the fp32 output is the compile-line specialization
  `-DWGRAD_OUT=32` of the bf16 program.
- Weight-gradient tail: the tiles of the last partial wave are split into
  `tail_splits` k-step chunks over all clusters (`wgrad_tail_tiles`,
  `wgrad_tail_plan`, a closed-form cost model of rounds, partial traffic and one
  launch); when `tail_splits > 1` an ordered fp32 tail-reduce kernel
  (`wgrad_reduce_grid` CTAs) sums the chunks, otherwise only the main kernel
  runs. The tile raster inside a group follows the per-wave operand footprint
  (`wgrad_raster_rows`).
- Workspaces (allocated at preparation, retained by the launch object): the
  per-CTA TMA descriptor slots of the weight gradient (a ring of nine
  descriptor pairs per CTA: a group switch edits the next pair and warms the
  one after it, never a pair an in-flight load may still read) and its fp32
  tail partials. The operand
  descriptors are launch arguments; there is no descriptor workspace and no
  descriptor upload kernel.

`GroupedGemmLaunch.plan` exposes `grid`, `tile_k`, `tail_splits`,
`raster_rows` and `launches` for inspection and tests.

## Build targets and layout

The programs are exact-architecture tcgen05 payloads built for the exact
targets `sm_100a`, `sm_103a` and `sm_107a` (`build.json` `arch_flags`). Each
of the five physical programs (`fwd`, `dgrad`, the weight-gradient main
kernel with a 256- or 512-wide k tile, and the weight-gradient tail-reduce)
has one kernel and one host-binding translation unit under
`csrc/cake_moe_grouped_gemm/`, shared by the three architectures: the single
architecture-dependent line is folded under `__CUDA_ARCH__`, the tail-reduce
k tile is a compile-line specialization (`-DWGRAD_TILE=256|512`), and so is
the fp32 weight-gradient output (`-DWGRAD_OUT=32`; `WGRAD_OUT=16`, the bf16
output, is the default). The shared weight-gradient sources, their registry
rows and their manifest entries are a hand fold of the exported files; a
later export regenerates them. `cake_jit.PROGRAMS` lists each program once
(its architecture-free `library` stem, FFI entry, argument plan and launch
geometry); `cake_jit.ROUTES` maps each public route (operation, output dtype,
k tile) to its stage programs and specializations. `build.json` (schema
`cake-build/1`, written by the export) is the only record of what to compile:
one entry per build `<library>[<specialization tag>]_<arch>` -- the tag is
`_wgrad_tile256` / `_wgrad_tile512` for the specialized tail-reduce builds,
`_wgrad_out32` for the fp32 weight-gradient builds (composed on the fp32
tail-reduce: `_wgrad_out32_wgrad_tile256` / `_wgrad_out32_wgrad_tile512`),
empty otherwise -- with its translation units, digests, nvcc flags (the
`-D` defines included), per-architecture closure digest and `aot/<name>.so`
path. `cake_jit.load_module(program, arch, specializations)` loads that entry
through the shared `cake_build.py` loader: the prebuilt library when `aot/`
holds it, else FlashInfer's JIT build (the cache name carries the exact target
and the closure digest; FlashInfer's build-target rules --
`FLASHINFER_CUDA_ARCH_LIST` or the visible devices -- apply to that path
only). The bindings include the package's `csrc/cake_ffi_checks.h`, no
FlashInfer header. The 512-wide weight-gradient tile is registered for
`sm_100a` and `sm_103a`.
The five kernel translation units share one device header,
`cake_moe_grouped_gemm_device.cuh`, carrying their common preamble (typedefs,
tensor-map ABI, math constants and the tcgen05 helpers behind need-guards): a
hand fold of the exported preambles that no `build.json` entry lists, so the
build's dependency tracking, not a manifest digest, covers it.
The four main-kernel bindings share their dynamic shared-memory opt-in and
TMA descriptor encoders the same way, through the fragment headers
`cake_moe_grouped_gemm_binding_*.h` included inside each host-shim namespace
(hand folds of exported text that no `build.json` entry lists).

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
