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
work beside the call on other streams.  The outputs are not the deterministic
package's bits.

| output | how it is computed | run to run | error against an FP64 reference |
| --- | --- | --- | --- |
| `y` | `rstd` with `rsqrt.approx`, the sigmoid with `ex2.approx` and `rcp.approx`, FP32 products, one rounding to BF16 | bit for bit | within one BF16 ulp; at most 1.1 times the deterministic package's |
| `dx` | the deterministic backward's row arithmetic | bit for bit | within one BF16 ulp; at most 1.1 times the deterministic package's |
| `dz` | the deterministic backward's row arithmetic | bit for bit | within one BF16 ulp; at most 1.1 times the deterministic package's |
| `dw` (BF16 or FP32) | the deterministic backward's FP32 chunk partials, summed exactly in 64-bit integer windows, one rounding | bit for bit | within the deterministic budget `abs(dw64) * 2^-7 + sum(abs(terms)) * 2^-20`; at most 1.1 times the deterministic package's |

"At most 1.1 times the deterministic package's" holds for both error measures
the tests compute on the same inputs: the relative L2 error and the largest
absolute error relative to the largest reference value.

Measured errors (relative L2 / largest relative), this package and the
deterministic package on the same inputs:

| GPU | shape | `y` | `dx` | `dz` | `dw` |
| --- | --- | --- | --- | --- | --- |
| (pending measurement) | | | | | |

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

## Programs and routes

`cake_jit.py` routes each call by its row count `rows = T * H` and the
device's SM count.  Every program of one role has the same row arithmetic.

| role | program | runs |
| --- | --- | --- |
| `fwd_short` | persistent forward, 16 lanes a row, one CTA per 16-row block | up to 12 rows per SM on `sm_100a` and `sm_103a`, 59 on `sm_107a` |
| `fwd` | forward, 64 threads, 8 lanes a row | between the short and the mid forward |
| `fwd_mid` | persistent forward, at most 6 CTAs per SM, each at most 2 blocks of 16 rows | from 976 rows per SM (800 on `sm_107a`) |
| `fwd_long` | persistent forward, at most 3 CTAs per SM, each at most 6 blocks of 32 rows | from 2400 rows per SM on `sm_100a` and `sm_103a` |
| `fwd_xlong` | the long forward, also prefetching the next block with one bulk prefetch | from 3000 rows per SM on `sm_103a`, 1740 on `sm_107a` |
| `bwd_bf16_short` | one cluster of 16 CTAs, the weight gradient summed on chip, no workspace | up to 256 rows, BF16 weight gradient |
| `bwd_bf16`, `bwd_f32` | static backward: CTA `b` runs chunks `b`, `b + grid`, ... | while the chunks fit 8 CTAs per SM (up to 64 rows per SM) |
| `bwd_bf16_mid`, `bwd_f32_mid` | claimed backward: CTAs claim runs of chunks at run time | past 8 chunks per SM |

The backward cuts the rows into 8-row blocks and the blocks into the
deterministic backward's chunks: `blocks_per_chunk = max(1, ceil(blocks /
8192))`, `chunks = ceil(blocks / blocks_per_chunk)`, and launches `min(chunks,
8 * SMs)` CTAs of 128 threads (15616 bytes of dynamic shared memory each).
The claimed backward lets a CTA slowed by a kernel on another stream take
fewer chunks; the bits do not depend on which CTA runs a chunk.

## Workspace

The static and the claimed backward need a workspace; the short backward reads
none.  A workspace is a zero-filled, contiguous `torch.uint8` CUDA tensor of
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

Calls without a workspace use one per device, stream and backward program,
kept for the process.  Each device also keeps a reserve for CUDA Graph
captures: 64 zero-filled workspaces in one allocation (29440 bytes each with
256-byte alignment, about 1.8 MiB), refilled outside any capture, at most 1024
per device.  A captured call without a workspace takes one from the reserve
for its capture and capture stream and keeps it, so a replay launches no fill
kernel; with none ready, the graph zero-fills a workspace of its own.

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

The package imports only itself, `torch` and `tvm_ffi`; its only `flashinfer`
import is the JIT fallback of `cake_build.py`.

## Tests

* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train.py` -- the package
  alone: FP64 budgets, run-to-run bits on every path, workspaces, CUDA
  Graphs, routing and grids, `torch.use_deterministic_algorithms`.
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_api.py` -- the public
  module and the comparison with the deterministic package (each output's
  FP64 error at most 1.1 times the deterministic package's).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_stall.py` -- the speed
  beside kernels resident on other streams.
* `benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py` -- timing.
