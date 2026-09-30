# Experimental ragged BF16 MoE grouped GEMM (Cake backend)

Both this API and its Cake backend are experimental and may change or be
removed without compatibility guarantees. Calling one of the helpers is the
explicit opt-in and emits FlashInfer's experimental API warning once. There is no automatic
backend selection. Tracking: a FlashInfer issue is linked when the pull
request opens.

Three operations over `E` groups of rows, described by an int32 **device**
tensor `offs[E]` of cumulative end offsets (`offs[-1] == sum_m`; groups may be
imbalanced, empty, one row long, or any size that is not a tile multiple).
The kernels read `offs` on the device; the host never synchronises on it.

| helper | computation | inputs | output |
| --- | --- | --- | --- |
| `grouped_gemm_fwd(x, w, offs, out=None)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `x [sum_m, K]`, `w [E, N, K]` bf16 | `Y [sum_m, N]` bf16 |
| `grouped_gemm_dgrad(g, w, offs, out=None)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `g [sum_m, N]`, `w [E, N, K]` bf16 | `dX [sum_m, K]` bf16 |
| `grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]` | `g [sum_m, N]`, `x [sum_m, K]` bf16 | `dW [E, N, K]` bf16 or fp32 |

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
from flashinfer.experimental.cake_moe_grouped_gemm import (
    grouped_gemm_fwd,
    prepare_grouped_gemm_wgrad,
)

E, N, K = 4, 4096, 6144
sizes = [28128, 28128, 28128, 28192]
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
x = torch.randn(sum_m, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
g = torch.randn(sum_m, N, dtype=torch.bfloat16, device="cuda")

y = grouped_gemm_fwd(x, w, offs)  # one-shot: prepares and launches

# Prepare once, launch many (CUDA-graph capturable; new offs / values may be
# written into the bound tensors between launches):
dw_launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32)
dw = dw_launch.launch()
print(dw_launch.record_name, dw_launch.plan)
```

`prepare_grouped_gemm_*` validates, allocates the output (when not given) and
the caller-owned workspaces, binds the generated program, encodes its TMA
descriptors once (outside CUDA Graph capture) and returns a
`GroupedGemmLaunch`. `launch()` performs no allocation and no host
synchronisation. Prepare a new launch when a shape, dtype or tensor binding
changes; the one-shot helpers prepare on every call.

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
  delivered threshold; 256 otherwise. The bf16 and fp32 outputs and the two
  tiles are separate registered programs.
- Weight-gradient tail: the tiles of the last partial wave are split into
  `tail_splits` k-step chunks over all clusters (`wgrad_tail_tiles`,
  `wgrad_tail_plan`, a closed-form cost model of rounds, partial traffic and one
  launch); when `tail_splits > 1` an ordered fp32 tail-reduce kernel
  (`wgrad_reduce_grid` CTAs) sums the chunks, otherwise only the main kernel
  runs. The tile raster inside a group follows the per-wave operand footprint
  (`wgrad_raster_rows`).
- Workspaces (allocated at preparation, retained by the launch object): the
  per-CTA TMA descriptor slots of the weight gradient, its fp32 tail partials,
  and one caller-owned descriptor workspace per pointer-ABI stage.

`GroupedGemmLaunch.plan` exposes `grid`, `tile_k`, `tail_splits`,
`raster_rows` and `launches` for inspection and tests.

## Build targets and layout

The programs are exact-architecture tcgen05 payloads: one registered module
per architecture (`sm_100a`, `sm_103a`, `sm_107a`), compiled with FlashInfer's
exact flag sets. Which of them can be built follows
`FLASHINFER_CUDA_ARCH_LIST` (or the visible devices) through
`flashinfer.compilation_context`; the loader never probes `nvcc` or the device.
Generated sources live under `csrc/cake_moe_grouped_gemm/<arch>/` in this
package (device and host-binding translation units per stage) and are
registered in `cake_jit.MODULES`; the JIT cache name carries each stage's
sealed closure identity. Until the export is delivered the registry is empty
and every helper raises `NotImplementedError` naming the missing program.

Measured architectures and the public performance summary are added with the
generated programs.

## Tests and benchmark

```bash
pytest tests/experimental/test_cake_moe_grouped_gemm.py
python benchmarks/bench_cake_moe_grouped_gemm.py --help
```

The tests compare against a per-group fp64 reference (including an empty
group, a one-row group, padded outputs, determinism and the fp32 output) and
skip when no supported device or no registered program is present. The
benchmark times the prepared launches against `torch._grouped_mm`, a
per-group cuBLAS loop and `flashinfer.grouped_mm.grouped_mm_bf16` on the
MoE projection rows of the design (E = 4 ragged distributions, E = 8 / 16 /
32 balanced, fp32 weight gradients).

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N`, `K` and the output span must fit
  32-bit element addressing.
- The alignment rules above; no transposed operand layouts other than the
  documented in-place `W` read of `dgrad`.
- Compute capability 10.0, 10.3 or 10.7 only; other devices raise.
