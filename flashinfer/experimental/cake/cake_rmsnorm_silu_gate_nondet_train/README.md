# `cake_rmsnorm_silu_gate_nondet_train`: Cake SiLU-gated RMSNorm training backend, non-deterministic family (SM100 / SM103 / SM107)

Experimental backend of `RMSNorm(x) * weight * SiLU(z)` for training, on BF16
rows of 128 (`[T, H, 128]` or `[T * H, 128]`) with a weight shared by every
token and head.  It is the backend of the public module
`flashinfer.cake_rmsnorm_silu_gate_nondet_train`
(`flashinfer/cake_rmsnorm_silu_gate_nondet_train.py`); the deterministic
counterpart is `flashinfer.cake_rmsnorm_silu_gate_det_train`, with the backend
`flashinfer.experimental.cake.cake_rmsnorm_silu_gate_det_train`.

The public module is a stable API; this backend package is experimental (its
own names carry no compatibility guarantee).  It ships programs for `sm_100a`
(B200 / GB200), `sm_103a` (B300 / GB300) and `sm_107a` (R200).  The feature
does not take part in automatic backend selection, autotuning or trace-apply.
Its libraries are JIT-built by FlashInfer from `build.json`.

## What each output promises

Every output repeats bit for bit: the same inputs on the same device give the
same bits on every call, whatever the grid, the order in which CTAs run and the
work beside the call on other streams.  `y` is not the deterministic
package's bits; `dx` and `dz` are, and `dw` is except where the exact sum lies
within the deterministic FP64 sum's error of a rounding boundary.

| output | how it is computed | run to run | error against an FP64 reference |
| --- | --- | --- | --- |
| `y` | `rstd` with `rsqrt.approx`, the sigmoid with `ex2.approx` and `rcp.approx`, FP32 products, one rounding to BF16 | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude; at most 1.1 times the deterministic package's |
| `dx` | the deterministic backward's row arithmetic (its bits) | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude (cancellation); at most 1.1 times the deterministic package's |
| `dz` | the deterministic backward's row arithmetic (its bits) | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude; at most 1.1 times the deterministic package's |
| `dw` (BF16 or FP32) | the deterministic backward's FP32 chunk partials, summed exactly in 64-bit integer windows (in the small-row bands: on chip in the deterministic pair's order in FP64), rounded to FP32 from the FP64 value of that sum, then to BF16 for a BF16 `dw` | bit for bit | within the deterministic budget `abs(dw64) * 2^-7 + sum(abs(terms)) * 2^-20`; at most 1.1 times the deterministic package's; in every band a column with an infinite or NaN term has the FP64 reference's class (`+inf`, `-inf` or `NaN`), also when the row count is not a multiple of the row block |

"At most 1.1 times the deterministic package's" holds for both error measures
the tests compute on the same inputs: the relative L2 error and the largest
absolute error relative to the largest reference value.  The tests also check
every output against the explicit eager operation (FP32 arithmetic, one final
cast to BF16, PyTorch autograd for the gradients) with the deterministic
suite's bounds: `y`, `dx` and `dz` within one BF16 ulp of the eager output plus
1/64 BF16 ulp of the FP64 reference's largest magnitude, `dw` within
`abs(dw_eager) * 2^-7 + sum(abs(terms)) * 2^-20`.

Measured errors against an FP64 reference computed from the BF16 inputs, on
normal inputs at three training shapes `[T, H, 128]`: `x`, `z` and `dy`
standard normal, the weight 1 + 0.5 times standard normal, all rounded to
BF16, `eps` = 1e-6, seed 0.  Each cell gives this package / the deterministic
package at FlashInfer commit 2e989556.  This package's value is the largest
over 10 runs, which gave the same bits every time; the deterministic package's
value is the median over 5 runs with equal bits.  `y`, `dx` and `dz` come from
the calls with a BF16 weight gradient; the calls with an FP32 weight gradient
give the same values.  These tables do not cover R200 (`sm_107a`).

Relative L2 error (this package / the deterministic package):

| GPU | T x H | `y` | `dx` | `dz` | `dw` BF16 | `dw` FP32 |
| --- | --- | --- | --- | --- | --- | --- |
| B200 (`sm_100a`) | 30193 x 48 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.699e-03 / 1.699e-03 | 1.305e-07 / 1.305e-07 |
| B200 (`sm_100a`) | 32388 x 48 | 1.658e-03 / 1.658e-03 | 1.659e-03 / 1.659e-03 | 1.658e-03 / 1.658e-03 | 1.650e-03 / 1.650e-03 | 1.008e-07 / 1.008e-07 |
| B200 (`sm_100a`) | 32737 x 48 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.718e-03 / 1.718e-03 | 1.259e-07 / 1.259e-07 |
| GB300 (`sm_103a`) | 30193 x 48 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.658e-03 / 1.658e-03 | 1.741e-03 / 1.741e-03 | 1.171e-07 / 1.171e-07 |
| GB300 (`sm_103a`) | 32388 x 48 | 1.658e-03 / 1.658e-03 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.545e-03 / 1.545e-03 | 1.152e-07 / 1.152e-07 |
| GB300 (`sm_103a`) | 32737 x 48 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.659e-03 / 1.659e-03 | 1.677e-03 / 1.677e-03 | 1.410e-07 / 1.410e-07 |

Largest error relative to the largest reference value (this package / the deterministic package):

| GPU | T x H | `y` | `dx` | `dz` | `dw` BF16 | `dw` FP32 |
| --- | --- | --- | --- | --- | --- | --- |
| B200 (`sm_100a`) | 30193 x 48 | 2.019e-03 / 2.019e-03 | 2.232e-03 / 2.232e-03 | 2.168e-03 / 2.168e-03 | 2.432e-03 / 2.432e-03 | 1.327e-07 / 1.327e-07 |
| B200 (`sm_100a`) | 32388 x 48 | 2.457e-03 / 2.457e-03 | 2.245e-03 / 2.245e-03 | 2.129e-03 / 2.129e-03 | 2.505e-03 / 2.505e-03 | 1.218e-07 / 1.218e-07 |
| B200 (`sm_100a`) | 32737 x 48 | 2.307e-03 / 2.307e-03 | 2.097e-03 / 2.097e-03 | 2.183e-03 / 2.183e-03 | 2.026e-03 / 2.026e-03 | 1.590e-07 / 1.590e-07 |
| GB300 (`sm_103a`) | 30193 x 48 | 2.222e-03 / 2.222e-03 | 2.317e-03 / 2.317e-03 | 2.386e-03 / 2.386e-03 | 1.922e-03 / 1.922e-03 | 9.621e-08 / 9.621e-08 |
| GB300 (`sm_103a`) | 32388 x 48 | 2.335e-03 / 2.335e-03 | 2.054e-03 / 2.054e-03 | 1.965e-03 / 1.965e-03 | 1.961e-03 / 1.961e-03 | 1.331e-07 / 1.331e-07 |
| GB300 (`sm_103a`) | 32737 x 48 | 2.106e-03 / 2.106e-03 | 1.836e-03 / 1.836e-03 | 2.465e-03 / 2.465e-03 | 1.915e-03 / 1.915e-03 | 1.354e-07 / 1.354e-07 |

Under `torch.use_deterministic_algorithms(True)` (strict or `warn_only`) no
entry point alerts or warns, also during CUDA Graph capture.

## Public entry points (`flashinfer/cake_rmsnorm_silu_gate_nondet_train.py`)

```python
from flashinfer.cake_rmsnorm_silu_gate_nondet_train import (
    cake_rmsnorm_silu_gate_nondet,                                  # autograd
    cake_rmsnorm_silu_gate_nondet_train_forward,
    cake_rmsnorm_silu_gate_nondet_train_backward,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace,
)

y = cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)
dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, weight, eps)
ws = cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, x.device)
cake_rmsnorm_silu_gate_nondet_train_backward(
    dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
    workspace=ws,
)
```

The mode is part of every name: no entry point takes a `deterministic`
keyword (passing one is a `TypeError`), and no call runs the deterministic
package.  This package exports the same names.

A call with no rows (`T * H == 0`) checks the last dimension (128), `weight`,
the given outputs and, in the backward, `grad_weight` and the refusal of an
FP32 workspace, then returns without a launch; it does not check the dtype,
device or layout of `x`, `z` and the upstream gradient or the size of a
`torch.uint8` workspace (the deterministic package checks a call with no
rows the same way).

## Programs and routes

`cake_jit.py` routes each call by its row count `rows = T * H`, the device's
SM count and, for the forward, whether `x` and `z` hold packed rows.  Each
program's arithmetic is fixed.  The backward
programs write `dx` and `dz` with the deterministic backward's row arithmetic.
The forward programs associate a row's sum of squares differently: the tiny
and the short forward add 8-term FMA chains of 16 lanes folded by XOR shuffles 1, 2, 4, 8,
the other forward programs 16-term chains of 8 lanes folded by XOR shuffles
1, 2, 4.  So `y`'s bits can differ between the programs, and so between the
forward bands of one architecture; each program repeats `y` bit for bit for
the same inputs on the same device.

| role | program | runs |
| --- | --- | --- |
| `fwd_tiny` | forward of packed rows, 128 threads, 16 lanes a row, one CTA per 8-row block (not persistent), the short forward's arithmetic and bits | its band of `FORWARD_BANDS` (packed rows) |
| `fwd_short` | forward, 256 threads, 16 lanes a row, one CTA per 16-row block (not persistent) | its band of `STRIDED_FORWARD_BANDS` (other layouts) |
| `fwd` | forward, 64 threads, 8 lanes a row, one CTA per 8-row block | the call role; no forward band routes it (the support query needs every role routed) |
| `fwd_mid` | whole waves of 6 CTAs per SM (as many waves as the rows need), each CTA at most 2 blocks of 16 rows | its forward bands |
| `fwd_long` | whole waves of 3 CTAs per SM (as many waves as the rows need), each CTA at most 6 blocks of 32 rows | its forward band (`sm_100a`) |
| `fwd_xlong` | the long forward's grid, also prefetching the next block with one bulk prefetch | its forward bands |
| `bwd_bf16_t512`, `bwd_f32_t512` | exact tiny backward: one cluster of 16 CTAs of 4 teams of 128 threads, the deterministic backward's chunk partials summed on chip in its reduction order, no workspace | the rows of its band (at most 512) |
| `bwd_bf16`, `bwd_f32` | static backward: CTA `b` runs chunks `b`, `b + grid`, ... | past the bands, while the chunks fit 8 CTAs per SM (up to 64 rows per SM) |
| `bwd_bf16_mid`, `bwd_f32_mid` | claimed backward: CTAs claim runs of chunks at run time | past 8 chunks per SM |

The forward bands (rows per SM: a call of `rows` rows on a device of `SMs`
SMs runs the first band with `rows <= last * SMs`): `FORWARD_BANDS` where `x`
and `z` hold packed rows (row `n` at element `128 n`, as contiguous
`[T, H, 128]` and `[T * H, 128]` inputs do), `STRIDED_FORWARD_BANDS` for
other layouts:

| architecture | packed rows (rows per SM) | other layouts (rows per SM) |
| --- | --- | --- |
| `sm_100a` | 1-140 `tiny`, 141-400 `long`, 401-1000 `xlong`, past 1000 `mid` | 1-12 `short`, 13-140 `mid`, 141-400 `long`, 401-1000 `xlong`, past 1000 `mid` |
| `sm_103a` | 1-120 `tiny`, 121-250 `mid`, 251-550 `xlong`, past 550 `mid` | 1-12 `short`, 13-250 `mid`, 251-550 `xlong`, past 550 `mid` |
| `sm_107a` | 1-65 `tiny`, 66-200 `mid`, 201-520 `xlong`, past 520 `mid` | 1-59 `short`, 60-200 `mid`, 201-520 `xlong`, past 520 `mid` |

The small-row backward bands (`BACKWARD_BANDS`, rows per call; a call with
`compute_grad_weight=False` takes the BF16 bands):

| architecture | BF16 weight gradient | FP32 weight gradient |
| --- | --- | --- |
| `sm_100a` | 1-512 `t512` | 1-512 `t512` |
| `sm_103a` | 1-512 `t512` | 1-512 `t512` |
| `sm_107a` | 1-512 `t512` | 1-512 `t512` |

The backward cuts the rows into 8-row blocks and the blocks into the
deterministic backward's chunks: `blocks_per_chunk = max(1, ceil(blocks /
8192))`, `chunks = ceil(blocks / blocks_per_chunk)`, and launches `min(chunks,
8 * SMs)` CTAs of 128 threads (15616 bytes of dynamic shared memory each).
The claimed backward lets a CTA slowed by a kernel on another stream take
fewer chunks; the bits do not depend on which CTA runs a chunk.

## Saved state

The autograd function (`cake_rmsnorm_silu_gate_nondet`) saves `x`, `z` and
`weight` by reference through `save_for_backward`, and `eps` as a Python
float.  It saves no `rstd` and no other intermediate, so beyond its inputs it
keeps 0 bytes, and a non-reentrant checkpointed forward keeps nothing but its
output (FlashInfer issue #7039).  The saved tensors are the caller's own: `x`
and `z` hold `T * H * 128` BF16 values (`T * H * 256` bytes each) and
`weight` 128 BF16 values (256 bytes).  A backward's workspace (below) is not
saved state: no tensor of it is kept by the autograd node.

## Workspace

The static and the claimed backward need a workspace, also with
`compute_grad_weight=False`; the small-row backwards read none, so a call in a
small-row band uses no workspace.  A workspace is a zero-filled, contiguous `torch.uint8` CUDA tensor of
29192 bytes whatever `T * H`, starting at a 16-byte aligned address:

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `dw_win` (integer windows: 4 slots x 7 windows x 128 columns) | u64 | 3584 | 0 |
| `dw_spec` (class words of infinite and NaN partials) | FP32 | 128 | 28672 |
| `dw_done` (arrival ticket, claim counter) | u32 | 2 | 29184 |

Every launch leaves the whole workspace zero, so one workspace serves every
later call on its stream (any shape, both weight-gradient dtypes, CUDA Graph
replays) without a reset.  Give each stream that may run a backward at the
same time its own workspace.  A buffer that did not come from
`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
else wrote into, must be zeroed with
`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` before its next
backward.  The deterministic package's FP32 workspace is refused with a
`ValueError` that names both workspaces.

Calls without a workspace use one per device, stream and backward program
that takes one (29192 bytes each; on each stream at most one per backward
program: the static and the claimed backward of each weight-gradient dtype),
kept for the process.  Eager
calls without a workspace also keep a reserve per device for CUDA Graph
captures: 64 zero-filled workspaces in one allocation (29440 bytes each with
256-byte alignment, about 1.8 MiB), refilled outside any capture, at most 1024
per device.  A captured call without a workspace takes one from the reserve
and keeps it, so a replay launches no fill kernel: one for its capture and
capture stream when cuda-python is installed (it gives the capture ids),
otherwise one per captured call.  With none ready, the graph zero-fills a
workspace of its own.

## Layout of this package

* `__init__.py` -- validation, the workspace views, the stream workspaces and
  the capture reserve, the launches, the autograd `Function` and the entry
  points.
* `cake_jit.py` -- the `MODULES` registry (one record per program; the
  exporter writes the derived fields), `ROUTES`, the row-count routing
  (`module_for_rows`), the backward's chunks and grid (`backward_chunks`,
  `backward_ctas`, `launch_grid`), the workspace layout
  (`workspace_layout`, `workspace_bytes`) and the loader glue.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package and this package's build manifest (schema `cake-build/1`).
* `csrc/` -- generated translation units (kernel and tvm-ffi binding per
  program) and `cake_ffi_checks.h`.

The package imports only itself, `torch` and `tvm_ffi`, and, when installed,
cuda-python's `cuda.bindings.runtime` for CUDA Graph capture ids (without it,
each captured call takes its own reserve workspace); its only `flashinfer`
import is the JIT fallback of `cake_build.py`.

## Tests

* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train.py` -- the package
  alone: FP64 budgets, the explicit eager operation, run-to-run bits on every
  path, workspaces, CUDA Graphs, routing and grids,
  `torch.use_deterministic_algorithms`.
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_api.py` -- the public
  module and the comparison with the deterministic package (each output's
  FP64 error at most 1.1 times the deterministic package's; `dx` and `dz` its
  bits).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_stall.py` -- the speed
  beside kernels resident on other streams.
* `benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py` -- timing.
