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

`y`, `dx`, `dz` and an FP32 `dw` repeat bit for bit everywhere: the same
inputs on the same device give the same bits on every call, whatever the
grid, the order in which CTAs run and the work beside the call on other
streams.  A BF16 `dw` repeats bit for bit too, except in the row-claims bands
(`bwd_bf16_rc`, `bwd_bf16_rc_mid`; see "Programs and routes"), where its last
bits may vary from run to run, within the family's error budget:

| architecture | row-claims bands of a BF16 `dw` |
| --- | --- |
| `sm_100a` | 513 rows .. 1329 rows per SM |
| `sm_103a` | 257 rows .. 1294 rows per SM |
| `sm_107a` | above 24 .. 68 rows per SM |

There the row-claims programs take less time than a repeatable backward of
the same rows.  As currently measured (CUPTI, cold L2), they save about
3.2 % .. 16.3 % of a checkpointed training step (2 x forward + backward with a
BF16 `dw`) on `sm_100a` (B200), about 4.0 % (an estimate) .. 18.4 %
(provisional) on `sm_103a` (GB300) and about 3.3 % .. 5.5 % on `sm_107a`
(R200).  A band runs a row-claims program only where the measured saving is
at least 3 %; the cuts are current measurements and live in one place,
`cake_jit.BACKWARD_BANDS`.

The BF16 bands up to 512 rows follow measurements of a checkpointed training
step (FlashInfer's benchmark timer, cold L2).  On GB300 (`sm_103a`) the step
with the short backward (`bwd_bf16_short`) is 7.8 % .. 14.8 % faster than
with the faster of the exact tiny backward and the row-claims backward at
48 .. 256 rows (13.1 % .. 18.7 % faster than with the exact tiny backward),
and the row-claims backward saves 6.9 % .. 13.1 % of the step against the
exact tiny backward at 336 .. 512 rows (at least 3 %), so `sm_103a` runs no
BF16 exact tiny backward and its row-claims band starts at 257 rows.  On
B200 (`sm_100a`) the steps with the short and with the exact tiny backward
are within 3 % of each other at up to 256 rows, and under FlashInfer's
benchmark timer the short backward alone is the faster of the two at every
row count up to 256.

`y` is not the deterministic package's bits.  `dx` and `dz` are, in every
band.  Outside the row-claims bands and the short backward's band (a BF16
`dw` up to 256 rows on `sm_100a` and `sm_103a`) `dw` is too, except where the
exact sum lies within the deterministic FP64 sum's error of a rounding
boundary.  In the short backward's band `dw` is that program's own
fixed-order sum: the same bits on every call, which need not be the
deterministic package's.

| output | how it is computed | run to run | error against an FP64 reference |
| --- | --- | --- | --- |
| `y` | `rstd` with `rsqrt.approx`, the sigmoid with `ex2.approx` and `rcp.approx`, FP32 products, one rounding to BF16 | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude; at most 1.1 times the deterministic package's |
| `dx` | the deterministic backward's row arithmetic (its bits) | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude (cancellation); at most 1.1 times the deterministic package's |
| `dz` | the deterministic backward's row arithmetic (its bits) | bit for bit | within one BF16 ulp plus 1/64 BF16 ulp of the tensor's largest magnitude; at most 1.1 times the deterministic package's |
| `dw` (FP32; BF16 outside the short and the row-claims bands) | the deterministic backward's FP32 chunk partials, summed exactly in 64-bit integer windows (in the exact tiny backward's bands: on chip in the deterministic pair's order in FP64), rounded to FP32 from the FP64 value of that sum, then to BF16 for a BF16 `dw` | bit for bit | within the deterministic budget `abs(dw64) * 2^-7 + sum(abs(terms)) * 2^-20`; at most 1.1 times the deterministic package's; in every band a column with an infinite or NaN term has the FP64 reference's class (`+inf`, `-inf` or `NaN`), also when the row count is not a multiple of the row block |
| `dw` (BF16 in the short backward's band) | each CTA's FP32 partial of its 16-row blocks (the row-claims backward's row arithmetic); CTA `r` adds every CTA's partials of the columns `8r` .. `8r + 7` in rank order in FP64 and rounds the sum to FP32, then to BF16 | bit for bit | the same budget, the same 1.1 times bound and the same class rule as the row above |
| `dw` (BF16 in the row-claims bands) | each thread's FP32 run of rows added to FP64 accumulators with atomics (`_rc_mid`: FP32 two-sum pairs, added once as FP64) in an order that depends on scheduling; the CTA that arrives last rounds the FP64 sum to FP32, then to BF16 | its last bits may vary | the same budget, the same 1.1 times bound and the same class rule as the row above |

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

Under `torch.use_deterministic_algorithms(True)` a backward call that
computes a BF16 `dw` in a row-claims band (through
`cake_rmsnorm_silu_gate_nondet_train_backward` or the autograd node's
backward) raises PyTorch's alert for an operation without a deterministic
implementation, which names the deterministic package's entry point: a
`RuntimeError`, or with `warn_only=True` a `UserWarning`, after which the call
runs.  The alert comes after the argument checks, immediately before the
launch; the call never falls back to another program.  Nothing else alerts:
the other bands, a call with `compute_grad_weight=False` (it takes the BF16
bands), a call with no rows, the forward, the workspace entry points and the
support query.  The check runs on the host, so a CUDA Graph checks it when it
is captured, not when it is replayed.

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
| `bwd_bf16_short` | short backward (a BF16 weight gradient; `sm_100a`, `sm_103a`): one cluster of 16 CTAs of 256 threads, 16 lanes a row, CTA `r` running the 16-row blocks `r`, `r + 16`, ... with the row-claims backward's row arithmetic; CTA `r` adds every CTA's FP32 partials of the columns `8r` .. `8r + 7` in rank order in FP64 (a fixed order), no workspace | the rows of its band (at most 256) |
| `bwd_bf16_t512` (`sm_100a`, `sm_107a`), `bwd_f32_t512` | exact tiny backward: one cluster of 16 CTAs of 4 teams of 128 threads, the deterministic backward's chunk partials summed on chip in its reduction order, no workspace | the rows of its band (at most 512) |
| `bwd_f32` (every architecture), `bwd_bf16` (`sm_107a`) | static backward: 128 threads, 8-row blocks, CTA `b` runs chunks `b`, `b + grid`, ... (the same bits at any grid); on `sm_100a` and `sm_103a` 4 copies of the weight-gradient windows and no prefetch, on `sm_107a` 4 copies plus occupancy words (the last CTA reads only the windows some CTA added into) | its backward band (the call role `bwd_bf16` has no program on `sm_100a` and `sm_103a`: no BF16 band routes a static backward there) |
| `bwd_bf16_rc` | row-claims backward (a BF16 weight gradient): 256 threads, 16 lanes a row, 16-row blocks, the CTAs past one wave claiming blocks from the claim counter; each thread's FP32 run of rows added to 4 FP64 accumulator copies with atomics, the CTA that arrives last rounding the sum | its backward band (every architecture) |
| `bwd_bf16_rc_mid` | row-claims mid backward (`sm_100a`, `sm_103a`): the row-claims backward with 5 CTAs per SM, thread 0 warming the next block of packed `dy`, `x` and `z` in L2 with one bulk prefetch per tensor; the runs and the CTA's column sums kept as FP32 two-sum pairs and added once as FP64 (a column whose pair sum is not finite takes the sum alone) | its backward band |
| `bwd_bf16_mid`, `bwd_f32_mid` | claimed backward: CTAs claim runs of chunks at run time; on `sm_100a` and `sm_103a` thread 0 warms the CTA's next block in L2 with one bulk prefetch per tensor where `dy`, `x` and `z` hold packed rows (head stride 128 and token stride `H * 128`; other layouts keep the per-thread prefetches), on `sm_107a` it warms the block two ahead | its backward band (the last) |

The programs (`ROUTES`; each name is `cake_rmsnorm_silu_gate_train_` and
the 20 hex digits shown):

| role | `sm_100a` | `sm_103a` | `sm_107a` |
| --- | --- | --- | --- |
| `fwd` | `c7919260b39b69e4e646` | `c7919260b39b69e4e646` | `c7919260b39b69e4e646` |
| `fwd_tiny` | `04bb67629537ef270bd4` | `04bb67629537ef270bd4` | `04bb67629537ef270bd4` |
| `fwd_short` | `3cb6aa31e140d9b13052` | `3cb6aa31e140d9b13052` | `3cb6aa31e140d9b13052` |
| `fwd_mid` | `cf7a87977aaad1a9822f` | `cf7a87977aaad1a9822f` | `cf7a87977aaad1a9822f` |
| `fwd_long` | `c6151f64f22ae65a48eb` | (not routed) | (not routed) |
| `fwd_xlong` | `83fe9e25ce0d01cc3ab1` | `83fe9e25ce0d01cc3ab1` | `83fe9e25ce0d01cc3ab1` |
| `bwd_bf16_short` | `cb597a0bebd6145e2024` | `cb597a0bebd6145e2024` | (none) |
| `bwd_bf16_t512` | `f6a3420d9aaa20a99e8d` | (none) | `f6a3420d9aaa20a99e8d` |
| `bwd_bf16` | (none) | (none) | `6d7cf79fb3d679d07410` |
| `bwd_bf16_rc` | `615c6fae07a6a89f3a45` | `615c6fae07a6a89f3a45` | `615c6fae07a6a89f3a45` |
| `bwd_bf16_rc_mid` | `f14663d91454a0e833f7` | `f14663d91454a0e833f7` | (none) |
| `bwd_bf16_mid` | `2a3bbf332a501a11ca31` | `2a3bbf332a501a11ca31` | `51408d05160a7056aad8` |
| `bwd_f32_t512` | `348a297be36a61e8f015` | `348a297be36a61e8f015` | `348a297be36a61e8f015` |
| `bwd_f32` | `f2915438ea54b487d93f` | `f2915438ea54b487d93f` | `a8cfbb391df971a9cb59` |
| `bwd_f32_mid` | `f2f608aa8d7a81906c9b` | `f2f608aa8d7a81906c9b` | `75a7f19a91e6ec05dffc` |

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

The backward bands (`BACKWARD_BANDS`, in order: a call of `rows` rows on a
device of `SMs` SMs runs the first band it fits, a small-row band while
`rows` is at most its count of rows (256 for `bwd_bf16_short`, 512 for the
exact tiny backwards), another band while `rows <= last * SMs` (rows per SM),
the last band always; a call with `compute_grad_weight=False` takes the BF16
bands).
A static backward runs at most 4 row blocks a CTA of its one-wave grid (256
rows per SM; on `sm_107a` one wave at most); a BF16 weight gradient runs a
row-claims backward only where it saves at least 3 % of a checkpointed
training step (see "What each output promises"); the claimed backward runs
past them:

| architecture | BF16 weight gradient | FP32 weight gradient |
| --- | --- | --- |
| `sm_100a` | 1-256 rows `bwd_bf16_short`, 257-512 rows `bwd_bf16_t512`, up to 800 rows per SM `bwd_bf16_rc`, up to 1329 rows per SM `bwd_bf16_rc_mid`, past that `bwd_bf16_mid` | 1-512 rows `bwd_f32_t512`, up to 256 rows per SM `bwd_f32`, past that `bwd_f32_mid` |
| `sm_103a` | 1-256 rows `bwd_bf16_short`, up to 700 rows per SM `bwd_bf16_rc`, up to 1294 rows per SM `bwd_bf16_rc_mid`, past that `bwd_bf16_mid` | 1-512 rows `bwd_f32_t512`, up to 256 rows per SM `bwd_f32`, past that `bwd_f32_mid` |
| `sm_107a` | 1-512 rows `bwd_bf16_t512`, up to 24 rows per SM `bwd_bf16`, up to 68 rows per SM `bwd_bf16_rc`, past that `bwd_bf16_mid` | 1-512 rows `bwd_f32_t512`, up to 64 rows per SM `bwd_f32`, past that `bwd_f32_mid` |

For example, the BF16 row-claims bands hold 513 to 196692 rows on a 148-SM
B200 (`bwd_bf16_rc` up to 118400), 257 to 196688 rows on a 152-SM GB300
(`bwd_bf16_rc` up to 106400) and 5089 to 14416 rows on a 212-SM R200.

The chunked backwards (static and claimed) cut the rows into 8-row blocks and
the blocks into the deterministic backward's chunks: `blocks_per_chunk =
max(1, ceil(blocks / 8192))`, `chunks = ceil(blocks / blocks_per_chunk)`, and
launch `min(chunks, 8 * SMs)` CTAs of 128 threads (15616 bytes of dynamic
shared memory each).  The claimed backward lets a CTA slowed by a kernel on
another stream take fewer chunks; the bits do not depend on which CTA runs a
chunk.  The row-claims backwards cut the rows into 16-row blocks and launch
`min(blocks, 4 * SMs)` CTAs (`5 * SMs` for `bwd_bf16_rc_mid`) of 256 threads
(20736 bytes of dynamic shared memory each); past one wave the CTAs claim
blocks from the claim counter, so a CTA that starts late takes fewer blocks.
A small-row backward launches one cluster of 16 CTAs whatever the rows: 512
threads each for the exact tiny backward (19584 bytes of dynamic shared
memory each), 256 threads each for the short backward (6144 bytes each).

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

The static, the row-claims and the claimed backwards need a workspace, also
with `compute_grad_weight=False`; the small-row backwards read none, so a call
in a small-row band uses no workspace.  A workspace is a zero-filled,
contiguous `torch.uint8` CUDA tensor of
`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes whatever
`T * H`: 29192 bytes on `sm_100a` and `sm_103a` and 36360 on `sm_107a` (the
largest layout among the architecture's routed backward programs), starting
at a 16-byte aligned address.  These sizes are separate from the saved state
above.  Each program lays its buffers from the start of it
(`workspace_layout`).  A chunked backward (static or claimed) lays three, with
`slots` copies of the integer windows:

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `dw_win` (integer windows: `slots` x 7 windows x 128 columns) | u64 | `slots * 896` | 0 |
| `dw_spec` (class words of infinite and NaN partials) | FP32 | 128 | `slots * 7168` |
| `dw_done` (arrival ticket, claim counter) | u32 | 2 | `slots * 7168 + 512` |

A row-claims backward lays two, 4104 bytes in all:

| buffer | dtype | elements | byte offset |
| --- | --- | --- | --- |
| `dw_sums` (FP64 accumulators: 4 copies x 128 columns) | FP64 | 512 | 0 |
| `dw_done` (arrival ticket, claim counter) | u32 | 2 | 4096 |

| program | `sm_100a`, `sm_103a` | `sm_107a` |
| --- | --- | --- |
| `bwd_bf16_short` | none | (no program) |
| `bwd_bf16_t512` | none (`sm_100a`; no program on `sm_103a`) | none |
| `bwd_f32_t512` | none | none |
| `bwd_bf16` | (no program) | 5 slots (4 window copies, then the occupancy words), 36360 bytes |
| `bwd_f32` | 4 slots, 29192 bytes | 5 slots (4 window copies, then the occupancy words), 36360 bytes |
| `bwd_bf16_rc` | `dw_sums` and `dw_done`, 4104 bytes | `dw_sums` and `dw_done`, 4104 bytes |
| `bwd_bf16_rc_mid` | `dw_sums` and `dw_done`, 4104 bytes | (no program) |
| `bwd_bf16_mid`, `bwd_f32_mid` | 4 slots, 29192 bytes | 4 slots, 29192 bytes |

Every launch leaves the whole workspace zero, so one workspace serves every
later call on its stream (any shape, both weight-gradient dtypes, every
backward program of its architecture, CUDA Graph replays) without a reset.
Give each stream that may run a backward at the same time its own workspace.
A buffer that did not come from
`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
else wrote into, must be zeroed with
`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` before its next
backward.  The deterministic package's FP32 workspace is refused with a
`ValueError` that names both workspaces.

Calls without a workspace use one per device, stream and backward program
that takes one (of the size above: 29192 bytes on `sm_100a` and `sm_103a`,
36360 on `sm_107a`; on each stream at most one per such program the
architecture routes: the static, the row-claims `_rc` and `_rc_mid` and the
claimed `_mid` backwards), kept for the process.  Eager calls without a
workspace also keep a reserve per device for CUDA Graph captures: 64
zero-filled workspaces in one allocation (with 256-byte alignment, 29440 bytes
each, about 1.8 MiB, on `sm_100a` and `sm_103a`, and 36608 bytes each, about
2.2 MiB, on `sm_107a`), refilled outside any capture, at most 1024 per device.
A captured call without a workspace takes one from the reserve and keeps it,
so a replay launches no fill kernel: one for its capture and capture stream
when cuda-python is installed (it gives the capture ids), otherwise one per
captured call.  With none ready, the graph zero-fills a workspace of its own.

## Layout of this package

* `__init__.py` -- validation, the workspace views, the stream workspaces and
  the capture reserve, the launches, the autograd `Function` and the entry
  points.
* `cake_jit.py` -- the `MODULES` registry (one record per program; the
  exporter writes the derived fields), `ROUTES`, the row-count routing
  (`FORWARD_BANDS`, `STRIDED_FORWARD_BANDS`, `BACKWARD_BANDS`,
  `module_for_rows`), the backward's chunks and grid (`backward_chunks`,
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
  alone: FP64 budgets and classes in every band, the explicit eager
  operation, run-to-run bits on every path (a BF16 `dw` in the row-claims
  bands: within its budget instead), workspaces, CUDA Graphs, routing and
  grids, `torch.use_deterministic_algorithms` (the alert in the row-claims
  bands, no alert elsewhere).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_api.py` -- the public
  module and the comparison with the deterministic package (each output's
  FP64 error at most 1.1 times the deterministic package's; `dx` and `dz` its
  bits).
* `tests/norm/test_cake_rmsnorm_silu_gate_nondet_train_stall.py` -- the speed
  beside kernels resident on other streams.
* `benchmarks/bench_cake_rmsnorm_silu_gate_nondet_train.py` -- timing.
