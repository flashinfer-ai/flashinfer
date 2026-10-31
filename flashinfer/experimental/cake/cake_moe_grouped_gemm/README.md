# Experimental non-deterministic ragged BF16 MoE grouped GEMM (Cake backend)

Both this API and its Cake backend are experimental and may change or be
removed without compatibility guarantees. This package is the
`deterministic=False` counterpart of the deterministic package
`flashinfer.experimental.cake_moe_grouped_det_gemm`. The entry points are the
facade functions `flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd`,
`grouped_gemm_dgrad`, `grouped_gemm_wgrad` and `cake_grouped_mm`; calling one of
them is the explicit opt-in and emits FlashInfer's `ExperimentalWarning` once.
There is no automatic backend selection. Tracking: FlashInfer issue #8070
(source requirements: #5678, #6570, #6553, #7047).

The three operations, their arguments, alignment rules and outputs are the
deterministic package's (see its README): over `E` groups of rows described by
an int32 **device** tensor `offs[E]` of cumulative end offsets, or
`m_indptr[E + 1]`, read by the kernels only.

| helper | computation | output |
| --- | --- | --- |
| `grouped_gemm_fwd(x, w, offs, out=None, *, deterministic=True)` | `Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T` | `Y [sum_m, N]` bf16 |
| `grouped_gemm_dgrad(g, w, offs, out=None, *, deterministic=True)` | `dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]` (W read in place) | `dX [sum_m, K]` bf16 |
| `grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *, deterministic=True)` | `dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]` | `dW [E, N, K]` bf16 or fp32 |

## Numerical mode and dispatch

Every function of this package (`grouped_gemm_*`, `prepare_grouped_gemm_*`,
`cake_grouped_mm` / `CakeGroupedMm`) takes `deterministic: bool = True` and
raises `ValueError` unless it is `False`: the package has no deterministic
programs of its own. The facade `flashinfer.experimental.cake.grouped_mm`
dispatches:

| `deterministic` | `torch.are_deterministic_algorithms_enabled()` | runs |
| --- | --- | --- |
| `True` (default) | any | the deterministic package, unchanged bits |
| `False` | `False` | this package |
| `False` | `True` | the deterministic package, one `UserWarning` per entry point |

Called directly under `torch.use_deterministic_algorithms(True)`, the entry
points that can launch the unordered weight gradient (`prepare_grouped_gemm_wgrad`,
`grouped_gemm_wgrad` and the backward of `CakeGroupedMm` when it computes `dW`)
raise PyTorch's non-determinism alert (a `RuntimeError`, or a `UserWarning`
under `warn_only=True`, naming the deterministic package) and never fall back.
A prepared weight gradient checks again at every `GroupedGemmLaunch.launch()`,
so switching the mode on after the preparation is not missed; under CUDA graph
capture that check runs when the launch is captured, not at replay. The forward and data-gradient entry
points do not alert: they give the deterministic bits.

What changes with `deterministic=False`:

- `fwd`, `dgrad` and the weight gradient with the 256-wide k tile run this
  package's copies of the deterministic programs, with the same host plan:
  bitwise the deterministic outputs. The 256-wide k tile's `main_wave`
  program is the same kernel rendered from a newer source: it takes two more
  buffer arguments (`A_rows`, `B_rows`) that its sm_100a / sm_103a builds
  never read, and it has no sm_107a build.
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

## Usage

```python
import torch
from flashinfer.experimental.cake.grouped_mm import (
    cake_grouped_mm,
    grouped_gemm_wgrad,
)
from flashinfer.experimental.cake.cake_moe_grouped_gemm import (
    prepare_grouped_gemm_wgrad,
)

E, N, K = 4, 4096, 6144
sizes = [28128, 28128, 28128, 28192]
sum_m = sum(sizes)
offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
x = torch.randn(sum_m, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
g = torch.randn(sum_m, N, dtype=torch.bfloat16, device="cuda")

dw = grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32, deterministic=False)
y = cake_grouped_mm(x, w, offs, deterministic=False)  # dX and dW in backward

# Prepare once, launch many (CUDA-graph capturable):
dw_launch = prepare_grouped_gemm_wgrad(g, x, offs, deterministic=False)
dw = dw_launch.launch()
print(dw_launch.record_name, dw_launch.stages, dw_launch.plan["unordered_tail"])
```

The facade functions carry the experimental-API marking and the once-only
warnings and hand off to the plain functions of the same names in this
package (`grouped_gemm_*` in the package root, `cake_grouped_mm` in
`cake_autograd`), which a vendoring build calls directly without any warning.
`prepare_grouped_gemm_*` validates, allocates the output (when not given) and
the weight-gradient descriptor slots, resolves the route and binds the
generated programs, and returns a `GroupedGemmLaunch` whose `launch()` passes
the operands' TMA descriptors by value and performs no host synchronisation.
There is no stable-API handoff: `flashinfer.grouped_mm.grouped_mm_bf16(...,
backend="cake")` stays on the deterministic package.

`cake_grouped_mm(x, w, offs, deterministic=False)` is a
`torch.autograd.Function` wrapper (`CakeGroupedMm`) that saves `x`, `w` and
`offs` only. Its forward is the deterministic forward (bitwise), so a
non-reentrant activation-checkpoint replay recomputes the same `Y`; its
backward produces `x.grad` with the `dgrad` programs and `w.grad` with the
`wgrad` programs in the dtype of `w`, accumulated into an existing `.grad` by
autograd as usual. The facade fixes the mode at the forward call.
The wrapper launches the forward GEMM after `CakeGroupedMm.apply` returns,
that is after autograd saved `x`, `w` and `offs`, as for a built-in op. Under
`torch.utils.checkpoint(..., use_reentrant=False)` with early stop on (the
default), the recomputation therefore stops before this GEMM when these are
the last saves of the innermost checkpointed region; a later save in that
region (of its output or of any other tensor) makes it run, with the same
bits. A direct `CakeGroupedMm.apply(x, w, offs, False)` launches inside
`forward` and is always recomputed, as is the wrapper under `torch.jit.trace`
and `torch.compile`.

## Host plan and workspaces

Every decision is the deterministic package's (`cake_backend.py` carries the
same planner, constants and stage rules: grid, output tile, evict-first and
3 / 3 head / tail twins, probed queue split, k tile, wave sync, accumulator
release, k256 tail); see that package's README. Differences:

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
only); the 512-wide k tile's routes bind the unordered-tail builds.

| feature | sm_100a | sm_103a | sm_107a |
| --- | --- | --- | --- |
| `fwd` / `dgrad` `main`, `main_wide` | domain schedule | domain schedule | static schedule (A collector on `main_wide`) |
| `fwd` `main_wide_ef` / `main_wide_ts33` | - / - | - / yes | yes / - |
| `dgrad` `main_wide_ef` / `main_wide_ts33` | - / - | - / yes | yes / - |
| `dgrad` probed queue split | yes | yes | - |
| `wgrad` k256 `main` + ordered `tail_reduce` | yes | yes | yes |
| `wgrad` k256 `main_wave` | yes | yes | - |
| `wgrad` k512 unordered tail: `main`, `main_halves` | yes (planned walk) | yes (planned walk) | yes |
| `wgrad` k512 unordered tail: `main_wave`, `main_wave_halves` (wp8) | yes | yes | - |
| `wgrad` k512 unordered tail: `main_halves_k256t` | - | - | yes |
| `wgrad` output dtypes | bf16, fp32 | bf16, fp32 | bf16, fp32 |
| CUDA toolkit floor (`min_cuda`) | 12.8 | 12.9 | 13.4 |

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
pytest tests/experimental/test_cake_moe_grouped_gemm.py tests/experimental/test_cake_moe_grouped_gemm_api.py tests/experimental/test_cake_moe_grouped_gemm_standalone.py tests/experimental/test_cake_moe_grouped_gemm_reproducibility.py
python benchmarks/bench_cake_moe_grouped_gemm.py --help
```

`test_cake_moe_grouped_gemm.py` (the backend suite, through
`tests.test_helpers.cake_package("cake.cake_moe_grouped_gemm")`) checks every
op, output dtype and stage form against a per-group fp64 reference (worst case
over repeats), empty groups, the #6570 shape relaunched and on concurrent
streams, CUDA graph replay with changed offsets, checkpointing, gradient
accumulation, changed shapes with reused and garbage-filled scratch, and the
tail workspaces at rest. `test_cake_moe_grouped_gemm_api.py` covers the facade
(marking, signatures, dispatch, the torch-deterministic fallback) and the
package's mode checks and alert. `test_cake_moe_grouped_gemm_reproducibility.py`
checks that `deterministic=True` gives the deterministic package's bits, that
`fwd` / `dgrad` give them with `deterministic=False`, and reports the
run-to-run variation of the unordered weight gradient.
`test_cake_moe_grouped_gemm_standalone.py` imports a copy with `flashinfer`
blocked. The benchmark times the deterministic and non-deterministic packages
(fwd, dgrad, wgrad and an autograd `2 x fwd + bwd` step) with peak memory,
against `torch._grouped_mm` and a per-group cuBLAS loop.

## Limitations

- bf16 inputs only; `E`, `sum_m`, `N`, `K` and the output span must fit
  32-bit element addressing.
- The deterministic package's alignment rules; compute capability 10.0, 10.3
  or 10.7 only.
- The optional fused SwiGLU and routing-scale epilogues of #5678 are not part
  of this family.
