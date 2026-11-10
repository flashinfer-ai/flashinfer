# Non-deterministic ragged BF16 MoE grouped GEMM (Cake backend)

This package is the backend of the module
`flashinfer.cake_moe_grouped_gemm_nondet_train`, the non-deterministic mode of the
MoE grouped GEMM; the deterministic mode is the module
`flashinfer.cake_moe_grouped_gemm_det_train` (backend
`flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train`). The module's
functions are stable FlashInfer APIs: the one-shot functions
`cake_moe_grouped_gemm_nondet_train_forward`, `_dgrad` and `_wgrad`, the prepared
launches `cake_moe_grouped_gemm_nondet_train_prepare_forward` / `_prepare_dgrad` /
`_prepare_wgrad`, the autograd entry `cake_moe_grouped_gemm_nondet` and the support
query `is_cake_moe_grouped_gemm_nondet_train_supported`. There is no automatic
backend selection. Tracking: FlashInfer issue #8070
(source requirements: #5678, #6570, #6553, #7047).

The three operations, their arguments, alignment rules and outputs are the
deterministic package's (see its README): over `E` groups of rows described by
an int32 **device** tensor `offs[E]` of cumulative end offsets, or
`m_indptr[E + 1]`, read by the kernels only.

The slice lower bound is `0` for `e == 0` and `offs[e-1]` for `e > 0`.
For `m_indptr` input, `offs` in these formulas means `m_indptr[1:]`.

| helper | computation | output |
| --- | --- | --- |
| `cake_moe_grouped_gemm_nondet_train_forward(x, w, offs, out=None)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `Y [sum_m, N]` bf16 |
| `cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs, out=None)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `dX [sum_m, K]` bf16 |
| `cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`; with `accumulate=True`, `out[e] += dW[e]` | `dW [E, N, K]` bf16 or fp32 |

## Mode rules

The module is the mode: this package and `flashinfer.cake_moe_grouped_gemm_nondet_train`
are the non-deterministic mode only, every public name carries `nondet`, and
nothing here routes to the deterministic module:

| call | result |
| --- | --- |
| any entry point of this package or of `flashinfer.cake_moe_grouped_gemm_nondet_train` | this package |
| a `deterministic` argument (`True` or `False`) | `TypeError` before anything runs: no entry takes one |
| a fourth argument of `CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)` other than `None` (an internal slot; the fifth is `fuse_wgrad_accumulation`) | `TypeError` |
| a call under `torch.use_deterministic_algorithms(True)` | the same launches and bits as without the mode: every launch of this release runs in a fixed order, so none raises PyTorch's non-determinism alert; never a deterministic fallback |

Every launch of this release runs in a fixed order, so
`CakeMoEGroupedGemmNondetLaunch.order_varying` is `False` for every launch. The
alert stays for an order-varying launch: a weight gradient planned with
`plan["unordered_tail"]` and a split tail (`plan["tail_splits"] > 1`), which
only an architecture listed in `HOST_PLAN_CONSTANTS["unordered_tail_archs"]`
plans. That list is empty. For such a launch,
`cake_moe_grouped_gemm_nondet_train_wgrad`, `CakeMoEGroupedGemmNondetLaunch.launch()`
and the backward of `CakeMoEGroupedGemmNondetFunction` would raise PyTorch's
non-determinism alert immediately before it runs (`RuntimeError`, nothing
launches; `warn_only=True`: a `UserWarning`, then the launch runs). The alert
names the counterpart in `flashinfer.cake_moe_grouped_gemm_det_train`, and the
caller's line for a direct call or a frame of PyTorch's autograd for the
backward. A prepared launch checks at every `launch()`; under CUDA graph
capture that check runs when the launch is captured, not at replay. The
preparation never alerts. No tape, saved tensor or workspace crosses between
this package and the deterministic one.

The former names were removed with no alias (`ModuleNotFoundError`,
`ImportError` or `AttributeError`). The table lists the calls that ran the
non-deterministic package, in the v0.7.4rc1 prerelease and on `main` after it
(v0.7.3 had no non-deterministic package). Every entry of the package
`flashinfer.experimental.cake.cake_moe_grouped_gemm` ran only with
`deterministic=False`; any other value raised `ValueError`. The module
`flashinfer.experimental.cake.grouped_mm` selected the mode with its
`deterministic` argument: in v0.7.4rc1 the default (`True`) ran the
deterministic package, so a call without `deterministic=False` moves to
`flashinfer.cake_moe_grouped_gemm_det_train`, with every former deterministic
name (see its package README); on `main` only `deterministic=False` ran.

| removed | use instead |
| --- | --- |
| the package `flashinfer.experimental.cake.cake_moe_grouped_gemm` with its submodules | `flashinfer.cake_moe_grouped_gemm_nondet_train` (this package: `flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`, same submodule names) |
| the module `flashinfer.experimental.cake.grouped_mm` | `flashinfer.cake_moe_grouped_gemm_nondet_train` for its calls with `deterministic=False`; `flashinfer.cake_moe_grouped_gemm_det_train` for its other calls (v0.7.4rc1) |
| `grouped_gemm_fwd` / `_dgrad` / `_wgrad` with `deterministic=False`, from `flashinfer.experimental.cake.grouped_mm` or the package above | `cake_moe_grouped_gemm_nondet_train_forward` / `_dgrad` / `_wgrad` |
| `prepare_grouped_gemm_fwd` / `_dgrad` / `_wgrad` with `deterministic=False`, from the package above | `cake_moe_grouped_gemm_nondet_train_prepare_forward` / `_prepare_dgrad` / `_prepare_wgrad` |
| `GroupedGemmLaunch`, from the package above | `CakeMoEGroupedGemmNondetLaunch` |
| `cake_grouped_mm` with `deterministic=False`, from `flashinfer.experimental.cake.grouped_mm` or the package above | `cake_moe_grouped_gemm_nondet` |
| `CakeGroupedMm`, from the package above | `CakeMoEGroupedGemmNondetFunction` |
| the `deterministic` argument of every entry | the module of the mode: `flashinfer.cake_moe_grouped_gemm_det_train` or `flashinfer.cake_moe_grouped_gemm_nondet_train`; passing the argument raises `TypeError` |

Under `torch.use_deterministic_algorithms(True)` this package runs the same
launches as without the mode and gives the same bits: no launch of this release
alerts (above). It never falls back to the deterministic package: use
`flashinfer.cake_moe_grouped_gemm_det_train` where the deterministic package's
bits are required. This differs from v0.7.4rc1, where a call of
`flashinfer.experimental.cake.grouped_mm` with `deterministic=False` ran the
deterministic package with a one-time `UserWarning`, and the package
`flashinfer.experimental.cake.cake_moe_grouped_gemm` alerted at every
weight-gradient call, its preparation included. On `main` after v0.7.4rc1 both
alerted only right before an order-varying weight-gradient launch, as this
package did before its split tail became ordered.

A source checkout that had the old package directory keeps its untracked
`__pycache__` (and ignored `aot/`) there, which Python then imports as an empty
namespace package; remove them with
`git clean -fdX flashinfer/experimental/cake/cake_moe_grouped_gemm`.

What differs from the deterministic module:

- `fwd`, `dgrad` and the weight gradient with the 256-wide k tile run this
  package's copies of the deterministic programs and host plan, the L2
  evict-first twins of the wide tile included: bitwise the deterministic
  outputs.
  The 256-wide k tile's `main_wave` program is the same kernel rendered from a
  newer source: it takes two more buffer arguments (`A_rows`, `B_rows`) that
  its sm_100a / sm_103a builds never read, and it has no sm_107a build.
- The weight gradient with the 512-wide k tile runs builds of the
  deterministic package's k512 forms (`main`, `main_wave`, `main_halves`,
  `main_wave_halves`, `main_halves_k256t`) that finish their split tail in
  the same launch (the ordered one-launch tail). The split tail is the tiles
  of the last partial wave, cut into `tail_splits` k-step units over the
  clusters. Unit `s` of split tail tile `j` stores its fp32 partial to slot
  `j * tail_splits + s` of `tail_sums` and counts its arrival on the tile's
  tickets (`tail_tickets`). The tile's last unit to arrive sums the tile's
  slots from 0.0 in slot order, rounds the sum once and stores it (an `_acc`
  build adds the sum into C once). The slot order does not depend on the
  order in which the units arrive. The deterministic package writes the
  units' fp32 partials and sums them in slot order in a second launch; this
  package launches one kernel and keeps no per-preparation partials.

Numerical contract: every output (`Y`, `dX`, `dW` in bf16 or fp32, and the
accumulated `out` or `w.main_grad`) repeats bit for bit for the same inputs,
shapes, device SM count and configuration. The SM count and the host plan
constants fix the tail plan (`tail_splits`). `Y` and `dX` are bitwise the
deterministic package's, and so is `dW` wherever the 256-wide k tile runs or
the 512-wide k tile does not split its tail. Every whole tile is reduced by
one CTA pair over the group's rows in a fixed order, and a split tail tile
sums its units in slot order. The bits of the 512-wide k tile's `dW` depend
on the SM count and on `tail_splits`. They can differ from the deterministic
package's bits, whose tail plan and group order differ, within that package's
error budgets. bf16 operands, fp32 accumulation, one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the offsets contract (no
host synchronisation, no caller padding), the exact zeros of empty groups in
overwrite mode and the deterministic package's error budgets are unchanged.
The descriptor-ring and short-group fixes of #6570 live in the shared kernel
body and host plan. The kernels use no cooperative launch, and the autograd
entry saves only the tensors that backward reads; the latency under resident
kernels and the shared-memory carveout wait of #7038 and #7548 are not
addressed by this package. The slot sum of a split tail tile uses fp32 adds
without flush to zero, as the deterministic tail reduce does. Whether the
tensor-core fp32 accumulation keeps subnormal values is not yet measured
(see [Accuracy](#accuracy)).

This is a local operator: on a context-parallel rank it runs on that rank's
local rows; it adds no collective and no KCP form, and a parameter gradient's
reduction across ranks stays with the caller.

## Weight-gradient accumulation

`cake_moe_grouped_gemm_nondet_train_wgrad(..., out=out, accumulate=True)` (and
`cake_moe_grouped_gemm_nondet_train_prepare_wgrad(..., accumulate=True)`) adds the weight gradient
into the caller's `out` instead of overwriting it: `out[e] += G[e].T @ X[e]`,
Megatron's `gradient_accumulation_fusion` into an fp32 or bf16 `main_grad`.
`cake_moe_grouped_gemm_nondet(..., fuse_wgrad_accumulation=True)` does the same in its
backward into `w.main_grad` (below). Both default to off; with the defaults
the routes, programs, plans and bits are unchanged, and `plan["accumulate"]`
is `"off"`.

The fused accumulate builds of this release add into an **fp32** `out` only:
the 512-wide k tile's half-release forms have fp32 accumulate builds (feature
matrix below). A **bf16** `out` always takes the composed route: the bf16
overwrite into a scratch, then the same-dtype `out.add_`, which is
autograd's AccumulateGrad (the same bits and the same cost as the default
bf16 `w.grad` accumulation).

The host plans an accumulating launch exactly as the overwrite of `out.dtype`
(route, k tile, stage, grid, tail split, raster, wave mode) and records how it
adds in `plan["accumulate"]`:

| `plan["accumulate"]` | when | what runs |
| --- | --- | --- |
| `"fused"` | the 512-wide k tile with its ordered one-launch split tail (`plan["ordered_tail"]`), the route serves the selected stage's accumulate build (`stages == (stage + "_acc",)`) on the architecture, that build's library can load (below), and `out` is 32-byte aligned (base, row pitch, expert pitch) | one launch whose epilogue reads C and adds the tile into it; no new workspace, nothing allocated after the first launch on a stream |
| `"composed"` | otherwise; `plan["accumulate_reason"]` is the first condition that fails: `"tile_k"` (the 256-wide k tile), `"tail"` (a two-launch tail: a `tail_reduce` stage after the main launch, which this release never plans for the 512-wide k tile), `"stage"` (no accumulate build of the stage for the dtype and architecture: every bf16 `out`, and the `main` / `main_wave` forms), `"min_cuda"` (the build cannot load) or `"alignment"`; the last two warn once per process | the overwrite into a contiguous scratch of `out.dtype`, then `out.add_(scratch)` on the same stream |

The route is chosen by a static rule (nothing is compiled, loaded or
test-built to decide it): an accumulate build is bound when its prebuilt
library exists under `aot/`, or else when the version of FlashInfer's JIT
toolkit reaches the build's `build.json` `min_cuda`, with `nvcc` under the CUDA
path and `FLASHINFER_DISABLE_JIT` unset (the loader's own rule). The fp32 accumulate builds use the PTX ISA 9.4
`red.add.noftz.f32` form, so their `min_cuda` is 13.4 on every architecture;
an ahead-of-time build with an older toolkit ships no library for them, and
with a JIT toolkit below 13.4 the call composes (`"min_cuda"`) instead of
failing.

`out` is required (`out_dtype` None or `out.dtype`) and must not overlap
itself (row stride at least `K`, expert stride at least `N` row strides: an
expanded `out` is refused) nor share memory with `g`, `x` or `offs`; both
routes check this before running anything (`ValueError`). Row and expert
padding is allowed and never read or written. `launch()` and every CUDA graph
replay add again: a prepared accumulate launch is not idempotent. A fused
launch bumps `out._version` when it is launched or captured (the composed
`add_` bumps it itself). `sum_m == 0` launches nothing and leaves `out`
unchanged. The composed route's scratch (`E * N * K` elements of `out.dtype`,
the size of the weight gradient) is owned by the prepared launch, as its
descriptor slots are: keep the launch alive while any CUDA graph that captured
it may replay (the graph does not keep the scratch alive), and do not have one
prepared composed launch in flight on two streams at once. Inside a captured
backward, the scratch is allocated from, and returned to, the graph's pool
within that backward. As for any in-place operation, two launches into the
same `out` on different streams must be ordered by the caller.

Numerical contract:

- Fused (fp32 `out`): `out <- out + dW` with one round-to-nearest fp32 add of
  the tile's fp32 sum into C. Every whole tile is bitwise `out0.add_(dW32)`,
  where `dW32` is the fp32 weight gradient of the same plan; that add keeps
  subnormals (no flush to zero).
- A split tail tile (512-wide k tile, `tail_splits > 1`) adds `C + S` once,
  where `S` is the tile's fp32 sum of its slots in slot order: the value the
  fp32 overwrite of the same plan stores for that tile. So the tile is also
  `out0.add_(dW32)`, and it repeats bit for bit as the overwrite does. The
  slot sum adds without flush to zero (above).
- Composed: the overwrite's `dW` in `out.dtype` added by `out.add_`. For a
  bf16 `out` that is `RN_bf16(out + dW_bf16)`, AccumulateGrad's bits.
- An empty group's `out[e]` is not read or written on the fused route (its
  bits, `-0.0` and NaN payloads included, stay). The composed route has the
  semantics of `out.add_`: the values are kept, but `-0.0` may become `+0.0`
  and a signaling NaN is quieted. The published contract is the composed one;
  bitwise-untouched empty groups are a property of the fused route.

`cake_moe_grouped_gemm_nondet(x, w, offs, fuse_wgrad_accumulation=True)`
follows Megatron and Transformer Engine: `w.main_grad` must be an fp32 or bf16
tensor of `w`'s shape on its device with unit stride along the last dimension
(checked at the forward call when autograd records it, that is when `w`
requires grad and grad mode is on, so not under `torch.no_grad()`; and again
at the backward, which reads `w.main_grad` then, so a re-pointed buffer is
honoured).
The backward adds into it (fused for an fp32 `main_grad` where the route
serves it, composed for a bf16 one) and returns for `w` a fresh `torch.empty`
of `w`'s shape and dtype when `w` has the attribute `grad_added_to_main_grad`
(it is set to `True`; `torch.zeros` when `w.zero_out_wgrad` is true), else
`None` (`w.grad` stays untouched and no gradient hook of `w` fires). `x.grad`
is unchanged, bitwise. Every backward adds: `retain_graph` twice adds twice,
and a `torch.autograd.grad(..., inputs=...)` that excludes `w` still adds into
`w.main_grad`, so zero-bubble (split backward-input / backward-weight)
schedules are not supported. A `w` that does not require grad is never added
to. The flag raises `NotImplementedError` under `torch.jit.trace`,
`torch.compile` and in a `create_graph` backward; under a reentrant checkpoint
`w` arrives detached without `main_grad` (`ValueError`; use
`use_reentrant=False`).

Peak memory of one weight gradient into an fp32 `main_grad` (`S = E * N * K`
elements): against an fp32 `dW` followed by `main_grad.add_` (a `4 * S`-byte
temporary), the fused route with `grad_added_to_main_grad` saves `2 * S` bytes
(its bf16 placeholder of `2 * S` bytes remains; without the attribute it saves
`4 * S`); against Megatron's unfused path (a bf16 `w.grad` of `2 * S` bytes,
then the fp32 add) the peak is equal. The benchmark's accumulation arms report
the measured peak.

The module `flashinfer.cake_moe_grouped_gemm_nondet_train` takes the same keywords
and hands them to this package; it has no deterministic route.

```python
from flashinfer.cake_moe_grouped_gemm_nondet_train import (
    cake_moe_grouped_gemm_nondet,
    cake_moe_grouped_gemm_nondet_train_prepare_wgrad,
)

# Megatron-style fused accumulation into an fp32 main_grad
w = torch.nn.Parameter(torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda"))
w.main_grad = torch.zeros(E, N, K, dtype=torch.float32, device="cuda")
# (Megatron's DDP also sets w.grad_added_to_main_grad; the backward then returns a
# placeholder gradient for its hooks.  Without the attribute w.grad stays None.)
for x, g in micro_batches:
    y = cake_moe_grouped_gemm_nondet(x, w, offs, fuse_wgrad_accumulation=True)
    y.backward(g)  # w.main_grad[e] += G[e].T @ X[e]; x.grad as usual

# Low level: add into any fp32 / bf16 buffer
launch = cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs, out=main_grad, accumulate=True)
launch.launch()  # main_grad += dW; launch.plan["accumulate"] is "fused" or "composed"
```

## Usage

```python
import torch
from flashinfer.cake_moe_grouped_gemm_nondet_train import (
    cake_moe_grouped_gemm_nondet,
    cake_moe_grouped_gemm_nondet_train_prepare_wgrad,
    cake_moe_grouped_gemm_nondet_train_wgrad,
)

E, N, K = 4, 4096, 6144
sizes = [28128, 28128, 28128, 28192]
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
x = torch.randn(sum_m, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
g = torch.randn(sum_m, N, dtype=torch.bfloat16, device="cuda")

dw = cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out_dtype=torch.float32)
y = cake_moe_grouped_gemm_nondet(x, w, offs)  # dX and dW in backward

# Prepare once, launch many (CUDA-graph capturable):
dw_launch = cake_moe_grouped_gemm_nondet_train_prepare_wgrad(g, x, offs)
dw = dw_launch.launch()
print(
    dw_launch.record_name,
    dw_launch.stages,
    dw_launch.plan["ordered_tail"],
    dw_launch.plan["unordered_tail"],
    dw_launch.plan["tail_splits"],
)
```

The module's functions hand off to the plain functions of the same names in
this package (`cake_moe_grouped_gemm_nondet_train_forward`, `_dgrad` and `_wgrad`
in the package root, `cake_moe_grouped_gemm_nondet_train_prepare_*` and
`CakeMoEGroupedGemmNondetLaunch` in `cake_backend`, `cake_moe_grouped_gemm_nondet`
and `CakeMoEGroupedGemmNondetFunction` in `cake_autograd`), which a vendoring
build calls directly. Importing the module loads none of this package.
`cake_moe_grouped_gemm_nondet_train_prepare_*` validates, allocates the output (when not given) and
the weight-gradient descriptor slots, resolves the route and binds the
generated programs, and returns a `CakeMoEGroupedGemmNondetLaunch` whose `launch()` passes
the operands' TMA descriptors by value and performs no host synchronisation.
There is no stable-API handoff: `flashinfer.grouped_mm.grouped_mm_bf16(...,
backend="cake")` stays on the deterministic module.

`cake_moe_grouped_gemm_nondet(x, w, offs)` is a
`torch.autograd.Function` wrapper (`CakeMoEGroupedGemmNondetFunction`) that saves `x`, `w` and
`offs` only. Its forward is the deterministic forward (bitwise), so a
non-reentrant activation-checkpoint replay recomputes the same `Y`; its
backward produces `x.grad` with the `dgrad` programs and `w.grad` with the
`wgrad` programs in the dtype of `w`, accumulated into an existing `.grad` by
autograd as usual; forward and backward always run this package.
The wrapper launches the forward GEMM after `CakeMoEGroupedGemmNondetFunction.apply` returns,
that is after autograd saved `x`, `w` and `offs`, as for a built-in op. Under
`torch.utils.checkpoint(..., use_reentrant=False)` with early stop on (the
default), the recomputation therefore stops before this GEMM when these are
the last saves of the innermost checkpointed region; a later save in that
region (of its output or of any other tensor) makes it run, with the same
bits. A direct `CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)` launches inside
`forward` and is always recomputed, as is the wrapper under `torch.jit.trace`
and `torch.compile`.

## Host plan and workspaces

Every decision is the deterministic package's (`cake_backend.py` carries the
same planner, constants and stage rules: grid, output tile, evict-first and
3 / 3 head / tail twins, probed queue split, k tile, wave sync, accumulator
release, k256 tail); see that package's README. The L2 evict-first twins
(`main_wide_ef`) follow the deterministic rules and constants: the data
gradient's on the architectures of `dgrad_evict_first_archs` (sm_100a, the
domain-scheduled twin; sm_107a, the A-collector twin) from the architecture's
`dgrad_evict_first_min_rows` rows per group and `K >= dgrad_evict_first_min_k`
(`dgrad_evict_first`), the forward pass's on those of `fwd_lwh_ef_archs`
(sm_107a) for `K <= fwd_lwh_ef_max_k` (`fwd_lwh_ef`) and on those of
`fwd_evict_first_archs` (sm_100a, sm_103a) for `K >= fwd_evict_first_min_k`
(`fwd_evict_first`). The eviction priority changes no arithmetic. Differences:

- Ordered one-launch tail (`plan_wgrad(..., ordered_tail=True)`): the
  512-wide k tile on the architectures of
  `HOST_PLAN_CONSTANTS["ordered_tail_archs"]` (`wgrad_ordered_tail`; sm_100a,
  sm_103a and sm_107a, every architecture of the k512 routes) binds the builds
  that take `tail_sums` and `tail_tickets`.
  `HOST_PLAN_CONSTANTS["unordered_tail_archs"]` (`wgrad_unordered_tail`) is
  empty: the registry holds no build that adds its split tail with unordered
  reductions. Both lists are extension points. The preparation raises
  `ValueError` for an architecture listed in both. An architecture listed in
  neither needs the deterministic k512 builds and a `tail_reduce` stage on
  those routes (a new render); without them the preparation raises
  `ValueError`. The one-launch tail plans its split from
  `HOST_PLAN_CONSTANTS["atomic_tail_plan"]` (`wgrad_atomic_tail_plan`, the
  cost model of `wgrad_tail_plan` over its own constants) and has no reduce
  launch (`tail_reduce_grid` None, `launches` 1). `plan["ordered_tail"]`
  (`True` on the 512-wide k tile), `plan["unordered_tail"]` (`False`),
  `plan["tail_splits"]`, `plan["tail_sum_elems"]` and
  `plan["tail_ticket_words"]` show the form of the tail and the workspace it
  needs. The 256-wide k tile keeps `tail_plan` and the two-launch tail reduce.
- Tail workspaces: `tail_sums` (fp32, `num_tail * tail_splits * cluster_m *
  tail_tile_k` elements: one `cluster_m x tail_tile_k` slot per unit) and
  `tail_tickets` (u32, `num_tail * cta_group * TAIL_TICKETS_PER_CTA` words).
  Only the tickets rest at zero: a launch needs them zero at its start and
  leaves them zero (the tickets wrap). The slots have no rest state: every
  unit of a tile stores its slot before the tile's last unit reads the slots,
  so `tail_sums` is allocated with `torch.empty` and never cleared. Both are
  stream workspaces like `sched` and `wave_sync`: kept per (device, CUDA
  stream), allocated by the first launch on a stream that splits its tail
  (the tickets zeroed), replaced by a larger buffer when a later launch needs
  more, never shared by two streams. A launch under CUDA graph capture, or
  for a device that is not the current one, takes buffers of its own: the
  zero fill of the tickets is recorded with the launch, and no fill of
  `tail_sums` is recorded. A launch whose tail is not split passes
  one-element placeholders.
  `stream_workspace("tail_sums" | "tail_tickets", device, stream=None)`
  returns the buffer kept for a stream.
- Queue walk (`wgrad_queue_walk`, this package's own): on sm_100a and
  sm_103a the 512-wide k tile routes also serve `main_halves_queue` (and
  `main_halves_queue_acc` on the fp32 route), the half-release build whose
  tiles before the split tail are claimed from one global counter in tile
  order instead of strided over the clusters, so the tiles in flight are
  consecutive and tiles that read the same G or X strip run at nearly the same
  time. Each tile keeps its k-step order, so the build gives the bits of the
  same build on the strided walk. The host binds it in place of the stage it
  selects where a rule of `HOST_PLAN_CONSTANTS["queue_walk"]` names the
  device's architecture and SM count, the selected stage (`main_halves` /
  `main_wave_halves`), the output (`"bfloat16"` / `"float32"` overwrite,
  `"accumulate"` for the fused fp32 accumulate) and a range holding the group
  count (both ends included); `launch.stages` names the bound stage, and the
  plan is the selected stage's. The rules: sm_100a with 148 SMs (B200): `main_wave_halves`, E4 to
  E32, and `main_halves`, E8 to E32, for bf16 and the fused accumulate;
  sm_100a with 152 SMs (GB200): `main_wave_halves`, E4 to E32, for the fused
  accumulate, and `main_halves`, E16 to E32, for bf16; sm_103a with 152 SMs
  (GB300): `main_wave_halves`, E4 to E32, for bf16 and the fused accumulate,
  and `main_halves`, E8 to E16, for the fused accumulate. Every other call
  (fp32 overwrite, other group counts, other SM counts, sm_107a) keeps its
  stage. The queue build places its split tail as the host plans it, so on
  skewed groups its split tail tiles can differ in bits from `main_halves`'s
  slack placement (both ordered, both repeat bit for bit). It takes one more
  workspace, `sched`, which the host binds to the stream workspace
  `queue_sched`: two u32 words, zero before and after a launch (the last
  cluster to retire sets them back to zero), kept per (device, CUDA stream)
  like the other stream workspaces, so launches on one stream reuse them with
  no fill per launch; a capture or a launch for another device takes words of
  its own. `stream_workspace("queue_sched", device, stream=None)` returns the
  words kept for a stream.
- Slot memory: the kept slot buffer holds `tail_splits` slots per tail tile,
  so it is `tail_splits` times larger than one fp32 sum per tile. For each
  stream, the package keeps the largest buffer that a launch needed, for the
  life of the process. A launch needs `plan["tail_sum_elems"]` floats, at
  most `(clusters - 1) * 16 * 256 * 512`: at most `clusters - 1` tail tiles
  (one partial wave), each cut into at most `atomic_tail_plan["max_splits"]`
  = 16 units, each with one slot of 256 x 512 floats (256 x 256 for the
  k256-tail form). That bound is about 600 MiB per
  stream at 76 clusters (GB300) and about 840 MiB at 106 clusters (R200, 212
  SMs). Example: the gate/up projection with one group of 112,576 rows on
  GB300 splits 40 tail tiles into 9 units each, which takes about 180 MiB
  (about 20 MiB with one sum per tile).

## Build targets and layout

The programs are exact-architecture tcgen05 payloads for `sm_100a`, `sm_103a`
and `sm_107a`. `cake_jit.PROGRAMS` / `ROUTES`, `build.json` and `csrc/` are
written by the Cake export; `cake_build.py` is the shared loader, byte-identical
in every Cake package. The fwd, dgrad, 256-wide k tile weight-gradient and
tail-reduce programs are copies of the deterministic package's under this
package's prefix (`cake_moe_grouped_gemm_<digest>`, the same digests), except
that the 256-wide k tile's `main_wave` program is rendered from a newer source
of the same kernel (two more, unused buffer arguments; sm_100a / sm_103a
only); the 512-wide k tile's routes bind the ordered one-launch tail builds.

| feature | sm_100a | sm_103a | sm_107a |
| --- | --- | --- | --- |
| `fwd` / `dgrad` `main`, `main_wide` | domain schedule | domain schedule | static schedule (A collector on `main_wide`) |
| `fwd` `main_wide_ef` / `main_wide_ts33` | yes / - | yes / yes | yes / - |
| `dgrad` `main_wide_ef` / `main_wide_ts33` | yes / - | - / yes | yes / - |
| `dgrad` probed queue split | yes | yes | - |
| `wgrad` k256 `main` + two-launch `tail_reduce` | yes | yes | yes |
| `wgrad` k256 `main_wave` | yes | yes | - |
| `wgrad` k512 ordered one-launch tail: `main`, `main_halves` | yes (planned walk, slack tail) | yes (planned walk, slack tail) | yes (planned walk, slack tail) |
| `wgrad` k512 ordered one-launch tail: `main_wave`, `main_wave_halves` (wp8) | yes | yes | - |
| `wgrad` k512 ordered one-launch tail: `main_halves_k256t` | - | - | yes (slack tail) |
| `wgrad` k512 ordered one-launch tail: `main_halves_queue` (queue walk) | yes (planned tail) | yes (planned tail) | - |
| `wgrad` output dtypes | bf16, fp32 | bf16, fp32 | bf16, fp32 |
| `wgrad` accumulate, fused (`_acc` builds, fp32 `out` only) | `main_halves`, `main_wave_halves`, `main_halves_queue` | `main_halves`, `main_wave_halves`, `main_halves_queue` | `main_halves`, `main_halves_k256t` |
| `wgrad` accumulate, composed | bf16 `out`; k256 tile; `main` / `main_wave`; misaligned `out`; `_acc` library not loadable | bf16 `out`; k256 tile; `main` / `main_wave`; misaligned `out`; `_acc` library not loadable | bf16 `out`; k256 tile; `main`; misaligned `out`; `_acc` library not loadable |
| CUDA toolkit floor (`min_cuda`) | 12.8 | 12.9 | 13.4 |
| CUDA toolkit floor of the fp32 `_acc` builds | 13.4 | 13.4 | 13.4 |

The slack tail placement runs a launch's split tail whole on the clusters the
full waves leave least loaded when every tail tile fits there; the kernel
decides per launch on the device.

A stage runs where the host plan constants select it; validation and
performance results per architecture are reported in the delivery pull
request.

## Standalone import

The package imports only itself, `torch` and `tvm_ffi`; the only `flashinfer`
import is the loader's JIT fallback in `cake_build.py`, inside
`try/except ImportError`. A copy of the package under another top-level name
loads without a FlashInfer installation.

## Accuracy

Every output repeats bit for bit for the same inputs, shapes, device SM count
and configuration. The reference package is the deterministic package at
FlashInfer commit `2e989556` (`flashinfer.experimental.cake_moe_grouped_det_gemm`).
rel-L2 is the relative L2 error `||out - ref|| / ||ref||`; worst / max |ref|
is the largest element error divided by the largest reference magnitude.

### Tolerances

The tests and the accuracy measurement hold the outputs to these tolerances:

- G1, error against FP64: for each output, rel-L2 against an FP64 reference
  is at most 1.1 times the reference package's median error over 5 repeats
  (#8946), and worst / max |ref| is at most 1.5 times the reference
  package's median. The worst element of a split weight-gradient tile
  depends on which tiles take the split tail; the reference package's own
  worst / max |ref| varies by about 1.2x from seed to seed. The table below
  reports both.
- G2, bf16 outputs (`Y`, `dX`, bf16 `dW`): every element is within
  `1e-2 + 1e-2 |ref|` of the fp64 reference
  (`test_cake_moe_grouped_gemm_nondet_train.py`,
  `test_cake_moe_grouped_gemm_nondet_train_reproducibility.py`).
- G3, fp32 `dW`: rel-L2 is at most `1e-4` against the fp64 reference (the
  same two files). The one-group E1 rows of 112,576 and 136,064 rows are the
  exception: there the in-tree deterministic package's own fp32 `dW` has
  rel-L2 `1.17e-4` / `1.41e-4` on GB300, so rel-L2 and worst / max |ref| are
  each at most 1.1 times that package's error at the same inputs
  (`test_e1_fp32_wgrad_error_within_det` in the reproducibility test).
- G4, repeats: the backend suite checks the first launch against the fp64
  reference and requires every repeat (the output refilled, the split-tail
  slots filled with garbage bytes) to give its bits. The reproducibility test
  also requires every repeat to be bitwise the first, and the weight gradient
  to stay within the deterministic budget of the deterministic package's
  output.
- G5, accumulation (`test_cake_moe_grouped_gemm_nondet_train_acc.py`): every
  element of a bf16 `out` is within `1e-2 + 1e-2 (|out0| + n |dW|)` of the
  fp64 reference, where `out0` is the start value and `n` the number of
  accumulating launches. An fp32 `out` has rel-L2 at most `1e-4`.
- G6, no lost or doubled tile update (the same file): in every 128 x 128
  block of an fp32 `out`, `|(out - out0) - dW|` is at most
  `2^-6 max |dW| + 1e-6`.
- G7, exact results (the backend suite and the accumulate test): empty groups
  give zeros in overwrite mode and stay untouched on the fused route; padding
  is never written; fused tiles are bitwise `out0.add_(dW32)`; a subnormal
  value of `out0` is kept.

### Error against FP64 per output

Each row gives the largest error of this package's output against an FP64
reference over 40 shapes: the `gate_up` and `down` projections with 1, 4, 8,
16 and 32 experts and 112,576, 131,072 and 136,064 rows, balanced loads, and
skewed and one-empty loads at 4 experts. Next to it is the reference
package's error at the same inputs (median of 5 repeats). For the accumulated
outputs the reference adds its own `dW` with `out.add_`. The maximum is taken
over four GPUs: B200 (148 SMs), GB200 and GB300 (152 SMs) and VR200 NVL72
(212 SMs), with CUDA 13.4.

| id | output | run to run | rel-L2 | reference rel-L2 | worst / max \|ref\| | reference worst / max \|ref\| |
| --- | --- | --- | --- | --- | --- | --- |
| O1 | `Y` bf16 (forward) | repeats bit for bit | 1.66e-03 | 1.66e-03 | 3.58e-03 | 3.58e-03 |
| O2 | `dX` bf16 (data gradient) | repeats bit for bit | 1.66e-03 | 1.66e-03 | 3.24e-03 | 3.24e-03 |
| O3 | `dW` bf16 (weight gradient) | repeats bit for bit | 1.67e-03 | 1.67e-03 | 3.64e-03 | 3.64e-03 |
| O4 | `dW` fp32 (weight gradient) | repeats bit for bit | 1.58e-04 | 1.58e-04 | 1.81e-04 | 1.81e-04 |
| O5 | fp32 `out += dW`, fused, 1 / 4 launches | repeats bit for bit | 1.58e-04 / 1.58e-04 | 1.58e-04 / 1.58e-04 | 1.81e-04 / 1.81e-04 | 1.81e-04 / 1.81e-04 |
| O6 | bf16 `out += dW`, composed, 1 / 4 launches | repeats bit for bit | 1.92e-03 / 1.74e-03 | 1.92e-03 / 1.74e-03 | 3.68e-03 / 3.63e-03 | 3.68e-03 / 3.63e-03 |
| O7 | autograd `y` / `x.grad` / `w.grad` | repeats bit for bit | 1.66e-03 / 1.66e-03 / 1.67e-03 | 1.66e-03 / 1.66e-03 / 1.67e-03 | 3.58e-03 / 3.24e-03 / 3.64e-03 | 3.58e-03 / 3.24e-03 / 3.64e-03 |
| O8 | `w.main_grad` fp32 (`fuse_wgrad_accumulation=True`) | repeats bit for bit | 1.58e-04 | 1.58e-04 (O5's reference) | 1.81e-04 | 1.81e-04 (O5's reference) |
| O9 | gradients of a checkpointed step, early stop on / off | repeats bit for bit | 1.67e-03 / 1.67e-03 | 1.67e-03 / 1.67e-03 | 3.64e-03 / 3.64e-03 | 3.64e-03 / 3.64e-03 |

Over every case of the accuracy suite (the 40 shapes, short and edge shapes
and adversarial inputs: 240 cases, 384 on B200), the largest ratio to the
reference is 1.0178 (B200), 1.0572 (GB200, GB300) and 1.0502 (VR200) for
rel-L2, and 1.0499, 1.0085, 1.0085 and 1.0110 for worst / max |ref|. With
four more input seeds of 44 to 59 cases per GPU, worst / max |ref| reaches
1.2564 (GB200 and GB300, `down`, 32 experts, 136,064 rows) at a rel-L2 ratio
of 1.0032. Every row passes G1.

Subnormal values: the adversarial cases scale the inputs so that the products
straddle the smallest normal fp32 (2^-126). Every output passes G1 on every
GPU (1,152 output rows per GPU, 1,728 on B200), and 992 to 1,328 of them are
bitwise equal to the reference package's output. The slot sum of a split tail
tile and the fused add into `out` use fp32 adds without flush to zero.

## Tests and benchmark

```bash
pytest tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_api.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_names.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_standalone.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py
python benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py --help
```

`test_cake_moe_grouped_gemm_nondet_train.py` (the backend suite, through
`tests.cake.test_helpers.cake_package("cake_moe_grouped_gemm_nondet_train")`) checks every
op, output dtype and stage form against a per-group fp64 reference (worst case
over repeats), empty groups, the #6570 shape relaunched and on concurrent
streams, CUDA graph replay with changed offsets, checkpointing, gradient
accumulation, changed shapes with reused and garbage-filled scratch, and the
tail tickets at rest. `test_cake_moe_grouped_gemm_nondet_train_api.py` covers the
module (stable APIs with the package functions' signatures, no warning, the
support query, the handoff with no deterministic route, the `TypeError` of a
`deterministic` argument). It also checks that no launch alerts under
`torch.use_deterministic_algorithms(True)` and that the bits equal those of a
run with the mode off (on a GPU), and the dormant alert of a recorded
order-varying launch (on the CPU).
`test_cake_moe_grouped_gemm_nondet_train_names.py` (CPU) checks the names: every
public name carries `nondet`, no callable takes a `deterministic` argument, and
every removed path and name fails at import.
`test_cake_moe_grouped_gemm_nondet_train_reproducibility.py` checks that `fwd` /
`dgrad` give the deterministic package's bits (also through the module). It
also checks that every weight gradient repeats bit for bit across launches and
stays within the deterministic budget of the deterministic package's output.
`test_cake_moe_grouped_gemm_nondet_train_determinism.py` checks that every
output repeats bit for bit: repeated launches into outputs prefilled with
`0x00` and `0x5A` bytes, digests from fresh processes, fused and composed
accumulation over 1 and 4 launches, the autograd entry with and without
activation checkpointing, CUDA graph replay, and two CUDA streams.
`test_cake_moe_grouped_gemm_nondet_train_standalone.py` imports a copy with `flashinfer`
blocked. `test_cake_moe_grouped_gemm_nondet_train_acc.py` covers the accumulate mode: the
`_acc` stages against the shipped set and their twins, the plans, the routes
and the checks on the CPU; the fused and composed results against the fp64
reference and the bitwise oracles (also on every benchmark row as planned),
empty groups, padding, short groups, streams, graph replay, allocations and
the version counter on a GPU; and the autograd flag. The benchmark's `acc`
(checkpointed micro-step), `bwd_acc` (its backward) and `wgrad_acc` (one
weight gradient) rows time the accumulation arms (`ovw`, `acc`, `mgB`, `mgF`,
`mgO`, `mgX`, `mgXn`, `mgX_zero`, `accX`, and the deterministic package's
`det_acc` and `det_mgB`) with peak memory. The benchmark times the deterministic and non-deterministic packages
(fwd, dgrad, wgrad and an autograd `2 x fwd + bwd` step) with peak memory,
against `torch._grouped_mm` and a per-group cuBLAS loop. Its rows include
`E = 1` and `E = 32` at 112,576 and 136,064 rows, and the note of a Cake
weight-gradient row records its plan (`tail_splits`, `ordered_tail`,
`unordered_tail`).

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N`, `K` and the output span must fit
  32-bit element addressing.
- The deterministic package's alignment rules; compute capability 10.0, 10.3
  or 10.7 only.
- The optional fused SwiGLU and routing-scale epilogues of #5678 are not part
  of this family.
