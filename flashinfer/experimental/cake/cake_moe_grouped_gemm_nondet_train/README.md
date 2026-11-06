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

| helper | computation | output |
| --- | --- | --- |
| `cake_moe_grouped_gemm_nondet_train_forward(x, w, offs, out=None)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `Y [sum_m, N]` bf16 |
| `cake_moe_grouped_gemm_nondet_train_dgrad(g, w, offs, out=None)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `dX [sum_m, K]` bf16 |
| `cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *, accumulate=False)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`; with `accumulate=True`, `out[e] += dW[e]` | `dW [E, N, K]` bf16 or fp32 |

## Mode rules

The module is the mode: this package and `flashinfer.cake_moe_grouped_gemm_nondet_train`
are non-deterministic only, every public name carries `nondet`, and nothing here
routes to the deterministic module:

| call | result |
| --- | --- |
| any entry point of this package or of `flashinfer.cake_moe_grouped_gemm_nondet_train` | this package |
| a `deterministic` argument (`True` or `False`) | `TypeError` before anything runs: no entry takes one |
| a fourth argument of `CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)` other than `None` (an internal slot; the fifth is `fuse_wgrad_accumulation`) | `TypeError` |
| a call under `torch.use_deterministic_algorithms(True)` | PyTorch's non-determinism alert, naming the counterpart in `flashinfer.cake_moe_grouped_gemm_det_train`, immediately before an order-varying launch (`RuntimeError`, nothing launches; `warn_only=True`: a `UserWarning`, then the non-deterministic launch runs; it names the caller's line for a direct call of `cake_moe_grouped_gemm_nondet_train_wgrad` or `launch()`, a frame of PyTorch's autograd for the backward); never a deterministic fallback |

An order-varying launch (`CakeMoEGroupedGemmNondetLaunch.order_varying`) is the weight
gradient whose 512-wide k tile splits its tail (`plan["unordered_tail"]` and
`plan["tail_splits"] > 1`), in overwrite or accumulate mode: `cake_moe_grouped_gemm_nondet_train_wgrad`,
`CakeMoEGroupedGemmNondetLaunch.launch()` and the backward of `CakeMoEGroupedGemmNondetFunction` when it computes
`dW` alert there. A prepared launch checks at every `launch()`, so switching the
mode on after the preparation is not missed; under CUDA graph capture that check
runs when the launch is captured, not at replay. The preparation, the forward,
the data gradient, the 256-wide k tile's weight gradient (ordered tail reduce)
and an unsplit tail run in a fixed order and never alert. No tape, saved tensor
or workspace crosses between this package and the deterministic one.

The former names were removed with no alias (`ModuleNotFoundError`,
`ImportError` or `AttributeError`). Only the calls that passed
`deterministic=False` move here. The calls of
`flashinfer.experimental.cake.grouped_mm` without `deterministic=False` (the
default `deterministic=True`) ran the deterministic package: they move to
`flashinfer.cake_moe_grouped_gemm_det_train` (see its package README), as do the
names without `det` of `flashinfer.grouped_mm`, `flashinfer.grouped_mm.cake`,
`flashinfer.experimental.cake_moe_grouped_det_gemm` and
`flashinfer.experimental.cake_moe_grouped_gemm`, which were aliases of the
deterministic objects.

| removed | use instead |
| --- | --- |
| the package `flashinfer.experimental.cake.cake_moe_grouped_gemm` with its submodules (every entry required `deterministic=False`) | `flashinfer.cake_moe_grouped_gemm_nondet_train` (this package: `flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`, same submodule names) |
| `grouped_gemm_fwd` / `_dgrad` / `_wgrad` with `deterministic=False`, from `flashinfer.experimental.cake.grouped_mm` or the package above | `cake_moe_grouped_gemm_nondet_train_forward` / `_dgrad` / `_wgrad` |
| `prepare_grouped_gemm_fwd` / `_dgrad` / `_wgrad` with `deterministic=False`, from the package above | `cake_moe_grouped_gemm_nondet_train_prepare_forward` / `_prepare_dgrad` / `_prepare_wgrad` |
| `GroupedGemmLaunch`, from the package above | `CakeMoEGroupedGemmNondetLaunch` |
| `cake_grouped_mm` with `deterministic=False`, from `flashinfer.experimental.cake.grouped_mm` or the package above | `cake_moe_grouped_gemm_nondet` |
| `CakeGroupedMm`, from the package above | `CakeMoEGroupedGemmNondetFunction` |
| the `deterministic` argument of every entry | the module of the mode: `flashinfer.cake_moe_grouped_gemm_det_train` or `flashinfer.cake_moe_grouped_gemm_nondet_train`; passing the argument raises `TypeError` |

One behavior changes. Under `torch.use_deterministic_algorithms(True)`, a call of
`flashinfer.experimental.cake.grouped_mm` with `deterministic=False` ran the
deterministic package and warned once. This package never falls back: it raises
PyTorch's non-determinism alert (`RuntimeError`) right before an order-varying
weight-gradient launch, and with `warn_only=True` it warns (`UserWarning`) and
runs the non-deterministic launch. Use
`flashinfer.cake_moe_grouped_gemm_det_train` where the deterministic mode must
hold.

A source checkout that had the old package directory keeps its untracked
`__pycache__` (and ignored `aot/`) there, which Python then imports as an empty
namespace package; remove them with
`git clean -fdX flashinfer/experimental/cake/cake_moe_grouped_gemm`.

What differs from the deterministic module:

- `fwd`, `dgrad` and the weight gradient with the 256-wide k tile run this
  package's copies of the deterministic programs and host plan (but for the
  sm_100a `dgrad` evict-first twin below): bitwise the deterministic outputs.
  The 256-wide k tile's `main_wave` program is the same kernel rendered from a
  newer source: it takes two more buffer arguments (`A_rows`, `B_rows`) that
  its sm_100a / sm_103a builds never read, and it has no sm_107a build.
- On sm_100a the data gradient also has an L2 evict-first twin of its wide
  tile (`main_wide_ef`), this package's own build: the deterministic
  package's sm_100a wide domain-scheduled program with the G loads and the
  dX stores marked evict-first, so that each group's weights stay in L2. The
  host takes it under the deterministic package's evict-first rule
  (`dgrad_evict_first`: at least `dgrad_evict_first_min_rows` rows per group
  on average and `K >= dgrad_evict_first_min_k`), with sm_100a added to
  `dgrad_evict_first_archs`. The eviction priority changes no arithmetic: dX
  is bitwise the deterministic package's.
- The weight gradient with the 512-wide k tile runs builds of the
  deterministic package's k512 forms (`main`, `main_wave`, `main_halves`,
  `main_wave_halves`, `main_halves_k256t`) whose split tail -- the tiles of
  the last partial wave, cut into `tail_splits` k-step units over the
  clusters -- is added in place. Each unit adds its fp32 accumulator into the
  tile's fp32 sum (`tail_sums`) with unordered reductions and counts its
  arrival on the tile's tickets (`tail_tickets`); the last unit to arrive
  reads the sum, clears it, rounds it once and stores it. The deterministic
  package writes the units' fp32 partials and sums them in a fixed order in a
  second launch; this package launches one kernel and keeps no partials.

Numerical contract: only the run-to-run bitwise reproducibility of `dW` is
relaxed, and only for split tail tiles of the 512-wide k tile. Every whole
tile is reduced by one CTA pair over the group's rows in a fixed order (the
deterministic bits). bf16 operands, fp32 accumulation, one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the offsets contract (no
host synchronisation, no caller padding), the exact zeros of empty groups in
overwrite mode and the deterministic package's error budgets are unchanged.
The descriptor-ring and short-group fixes of #6570 live in the shared kernel
body and host plan. The kernels use no cooperative launch, and the autograd
entry saves only the tensors that backward reads; the latency under resident
kernels and the shared-memory carveout wait of #7038 and #7548 are not
addressed by this package. A floating-point
sum below the smallest normal fp32 flushes to zero in the reduction (the
deterministic tail reduce keeps it), well inside the error budget.

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
| `"fused"` | the 512-wide k tile with its unordered split tail, the route serves the selected stage's accumulate build (`stages == (stage + "_acc",)`) on the architecture, that build's library can load (below), and `out` is 32-byte aligned (base, row pitch, expert pitch) | one launch whose epilogue reads C and adds the tile into it; no new workspace, nothing allocated after the first launch on a stream |
| `"composed"` | otherwise; `plan["accumulate_reason"]` is the first condition that fails: `"tile_k"` (the 256-wide k tile), `"tail"` (an ordered tail), `"stage"` (no accumulate build of the stage for the dtype and architecture: every bf16 `out`, and the `main` / `main_wave` forms), `"min_cuda"` (the build cannot load) or `"alignment"`; the last two warn once per process | the overwrite into a contiguous scratch of `out.dtype`, then `out.add_(scratch)` on the same stream |

The route is chosen by a static rule (nothing is compiled, loaded or
test-built to decide it): an accumulate build is bound when its prebuilt
library exists under `aot/`, or else when the version of FlashInfer's JIT
toolkit reaches the build's `build.json` `min_cuda` (the loader's own rule). The fp32 accumulate builds use the PTX ISA 9.4
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
  where `S` is the tile's unordered fp32 sum of its split units, so it varies
  run to run as the overwrite does; the nondeterminism scope is unchanged.
  `S` is the overwrite's partial sum, whose unit reductions flush subnormal
  values to zero (above); subnormals are kept only in the `C + S` add.
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
print(dw_launch.record_name, dw_launch.stages, dw_launch.plan["unordered_tail"])
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
release, k256 tail); see that package's README. Differences:

- Data-gradient evict-first twin: `HOST_PLAN_CONSTANTS["dgrad_evict_first_archs"]`
  lists sm_100a and sm_107a (the deterministic package: sm_107a). On sm_100a
  `main_wide_ef` is this package's own build of the domain-scheduled wide
  tile (same domain schedule and probed queue split as `main_wide`); on
  sm_107a it is the copy of the deterministic A-collector twin.

- Unordered tail (`plan_wgrad(..., unordered_tail=True)`): the 512-wide k
  tile on the architectures of `HOST_PLAN_CONSTANTS["unordered_tail_archs"]`
  (`wgrad_unordered_tail`; sm_100a, sm_103a and sm_107a) binds the builds that
  take `tail_sums`. The list must name every architecture of the k512 routes:
  the registry has no ordered k512 build. Leaving an architecture out would
  need the deterministic k512 builds and a `tail_reduce` stage on those routes
  (a new render). The unordered tail plans its split from
  `HOST_PLAN_CONSTANTS["atomic_tail_plan"]` (`wgrad_atomic_tail_plan`, the
  cost model of `wgrad_tail_plan` over its own constants) and has no reduce
  launch (`tail_reduce_grid` None, `launches` 1). `plan["unordered_tail"]`,
  `plan["tail_splits"]`, `plan["tail_sum_elems"]` and
  `plan["tail_ticket_words"]` show whether a launch adds a split tail in place
  and the workspace it needs. The 256-wide k tile keeps `tail_plan` and the
  ordered tail reduce.
- Tail workspaces: `tail_sums` (fp32, `num_tail * cluster_m * tail_tile_k`
  elements) and `tail_tickets` (u32, `num_tail * cta_group *
  TAIL_TICKETS_PER_CTA` words) rest at zero; a launch needs them zero at its
  start and leaves them zero. They are stream workspaces like `sched` and
  `wave_sync`: kept per (device, CUDA stream), allocated zeroed by the first
  launch on a stream that splits its tail, replaced by a larger zeroed buffer
  when a later launch needs more, never shared by two streams; a launch under
  CUDA graph capture, or for a device that is not the current one, takes
  zeroed buffers of its own (the zero fill is recorded with the launch). A
  launch whose tail is not split passes one-element placeholders.
  `stream_workspace("tail_sums" | "tail_tickets", device, stream=None)`
  returns the buffer kept for a stream.

## Build targets and layout

The programs are exact-architecture tcgen05 payloads for `sm_100a`, `sm_103a`
and `sm_107a`. `cake_jit.PROGRAMS` / `ROUTES`, `build.json` and `csrc/` are
written by the Cake export; `cake_build.py` is the shared loader, byte-identical
in every Cake package. The fwd, dgrad, 256-wide k tile weight-gradient and
tail-reduce programs are copies of the deterministic package's under this
package's prefix (`cake_moe_grouped_gemm_<digest>`, the same digests), except
that the 256-wide k tile's `main_wave` program is rendered from a newer source
of the same kernel (two more, unused buffer arguments; sm_100a / sm_103a
only) and that the data gradient's sm_100a `main_wide_ef` program is this
package's own evict-first build of the deterministic wide domain-scheduled
kernel; the 512-wide k tile's routes bind the unordered-tail builds.

| feature | sm_100a | sm_103a | sm_107a |
| --- | --- | --- | --- |
| `fwd` / `dgrad` `main`, `main_wide` | domain schedule | domain schedule | static schedule (A collector on `main_wide`) |
| `fwd` `main_wide_ef` / `main_wide_ts33` | - / - | - / yes | yes / - |
| `dgrad` `main_wide_ef` / `main_wide_ts33` | yes (this package's build) / - | - / yes | yes / - |
| `dgrad` probed queue split | yes | yes | - |
| `wgrad` k256 `main` + ordered `tail_reduce` | yes | yes | yes |
| `wgrad` k256 `main_wave` | yes | yes | - |
| `wgrad` k512 unordered tail: `main`, `main_halves` | yes (planned walk, slack tail) | yes (planned walk, slack tail) | yes (planned walk, slack tail) |
| `wgrad` k512 unordered tail: `main_wave`, `main_wave_halves` (wp8) | yes | yes | - |
| `wgrad` k512 unordered tail: `main_halves_k256t` | - | - | yes (slack tail) |
| `wgrad` output dtypes | bf16, fp32 | bf16, fp32 | bf16, fp32 |
| `wgrad` accumulate, fused (`_acc` builds, fp32 `out` only) | `main_halves`, `main_wave_halves` | `main_halves`, `main_wave_halves` | `main_halves`, `main_halves_k256t` |
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

## Tests and benchmark

```bash
pytest tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_api.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_names.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_standalone.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py
python benchmarks/bench_cake_moe_grouped_gemm_nondet_train.py --help
```

`test_cake_moe_grouped_gemm_nondet_train.py` (the backend suite, through
`tests.test_helpers.cake_package("cake_moe_grouped_gemm_nondet_train")`) checks every
op, output dtype and stage form against a per-group fp64 reference (worst case
over repeats), empty groups, the #6570 shape relaunched and on concurrent
streams, CUDA graph replay with changed offsets, checkpointing, gradient
accumulation, changed shapes with reused and garbage-filled scratch, and the
tail workspaces at rest. `test_cake_moe_grouped_gemm_nondet_train_api.py` covers the
module (stable APIs with the package functions' signatures, no warning, the
support query, the handoff with no deterministic route, the `TypeError` of a
`deterministic` argument) and the alert before an order-varying launch only.
`test_cake_moe_grouped_gemm_nondet_train_names.py` (CPU) checks the names: every
public name carries `nondet`, no callable takes a `deterministic` argument, and
every removed path and name fails at import.
`test_cake_moe_grouped_gemm_nondet_train_reproducibility.py` checks that `fwd` /
`dgrad` give the deterministic package's bits (also through the module), and
reports the run-to-run variation of the unordered weight
gradient.
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
against `torch._grouped_mm` and a per-group cuBLAS loop.

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N`, `K` and the output span must fit
  32-bit element addressing.
- The deterministic package's alignment rules; compute capability 10.0, 10.3
  or 10.7 only.
- The optional fused SwiGLU and routing-scale epilogues of #5678 are not part
  of this family.
