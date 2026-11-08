# Cake non-deterministic packed causal conv1d + SiLU training backend (SM100 / SM103 / SM107)

This package is the backend of the non-deterministic (`nondet`) packed width-4
depthwise causal convolution + SiLU training routes. It serves two routes: the single-rank route
(`flashinfer.cake_causal_conv1d_nondet_train`) and the context-parallel (KCP)
route (`flashinfer.cake_causal_conv1d_nondet_train_kcp`, module `kcp` of this
package). The facades are stable FlashInfer APIs; this package, under
`flashinfer.experimental.cake`, is their backend. Nothing in `flashinfer`
imports it; a facade imports it on first use. The package reads no
environment variable. The deterministic backend is
`flashinfer.experimental.cake.cake_causal_conv1d_det_train` (facades
`flashinfer.cake_causal_conv1d_det_train` and
`flashinfer.cake_causal_conv1d_det_train_kcp`).

## Context-parallel (KCP) route

Every rank convolves its fragments of a packed batch. A fragment can continue a document. Its first three tokens then read the three document tokens before it (history rows) from the ranks that hold them.
The backward sends the derivative of those first tokens back to the
predecessor fragments. They add it to their last rows' `dx` before the one
BF16 rounding. Tokens never move otherwise: there is no activation
concatenation and no full-sequence gather. The route keeps the deterministic
KCP route's FP32-tap contract: FP32 taps, no bias, FP32 arithmetic, one BF16
rounding per output, and the SiLU derivative taken at the FP32 preactivation.

The following parts are the deterministic KCP package's:

- the context contract
- the forward and head programs (this package's copies of the deterministic programs)
- the halo exchanges
- the recompute handle
- graph replay
- the multi-tensor calls

Of the programs, only the backward after the head program differs: it overlaps the halo exchange, and its `dweight`
sum is unordered. The forward program is the same, but its launch grid follows the architecture
(`kcp.FWD_RESERVE_SMS`). On sm_107a it is the deterministic route's grid, one SM short of the co-resident CTA count.
On sm_100a and sm_103a it is the full co-resident grid, which measured faster there. `y` does not depend on the grid.

- **Halo exchange:** one `all_to_all_single` per direction and call. The
  forward delivers the history rows into the history pool `hist`
  (`[n_hist_rows, C]` BF16, saved for the backward). The backward delivers
  the derivatives of the first tokens as `dpre_tail` (`[n_tail_rows, C]`
  FP32). With one rank no collective is issued.
- **Devices:** sm_100a (B200 / GB200), sm_103a (B300 / GB300) and sm_107a
  (R200). Another device raises `RuntimeError`.
- **Libraries:** load prebuilt from `aot/` when present; otherwise they are
  JIT-built from `build.json`.

### Entry points

```python
from flashinfer.cake_causal_conv1d_nondet_train_kcp import (
    cake_causal_conv1d_nondet_kcp,          # autograd, one tensor
    cake_causal_conv1d_nondet_kcp_multi,    # autograd, n tensors of one layer (q / k / v), one exchange per direction
    CakeCausalConv1dNondetKcpRecomputeHandle,
)

y = cake_causal_conv1d_nondet_kcp(x_local, weight, ctx)                    # overlapped backward
q, k, v = cake_causal_conv1d_nondet_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)
```

| operand | shape | dtype | notes |
|---|---|---|---|
| `x_local`, `y`, `dx` | `[T_r, C]` or `[1, T_r, C]` | BF16 | contiguous; the rank's fragments back to back in the partition's order; `C` a multiple of 8 |
| `weight`, `dweight` | `[C, 4]` | FP32 | contiguous; `weight[c, 3]` multiplies the current token; `dweight` is the rank-local partial |
| `ctx` | | | the context of the rank's KDA step (`CakeKDADetKcpContext`, from `flashinfer.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context`): fragment table, halo plan, device tables, process group |

Each name maps to the deterministic name with `det` in place of `nondet`. A
recompute handle (any object with an `entries` dict) serves either package.
Every rank of `ctx.group` calls the entry once per layer, in the same order,
with the same tensor count and channel counts (see Rank agreement).

The module `kcp` also holds two autograd functions and four explicit calls:

- autograd functions: `CakeCausalConv1dNondetKcpFunction` and `CakeCausalConv1dNondetKcpMultiFunction`
- explicit calls: `cake_causal_conv1d_nondet_train_kcp_forward`, `cake_causal_conv1d_nondet_train_kcp_backward`, `cake_causal_conv1d_nondet_train_kcp_multi_forward` and `cake_causal_conv1d_nondet_train_kcp_multi_backward`

The explicit calls take the keyword `graph="replay"`. It captures each call in a keyed CUDA graph and replays it; a replayed graph does no host work. Call
`cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()` before
`torch.distributed.destroy_process_group()`. The eager call computes its
context-derived host work once per context and reuses it. The cache holds at most 64
contexts, and each entry keeps its context alive.
`cake_causal_conv1d_nondet_train_kcp_host_cache_clear()` empties that cache.

### Mode rules

The module name chooses the package. No entry takes a mode keyword. "Torch
deterministic mode" means `torch.use_deterministic_algorithms(True)`.

| call | torch deterministic mode off | torch deterministic mode on |
|---|---|---|
| this package's forward and checkpoint recompute | run | run (fixed-order); no alert |
| this package's backward (autograd, `cake_causal_conv1d_nondet_train_kcp_backward`, `cake_causal_conv1d_nondet_train_kcp_multi_backward`) | runs; `dweight` is non-deterministic | raises `RuntimeError` before any exchange. With `warn_only=True`, a `UserWarning`, then it runs |
| `flashinfer.cake_causal_conv1d_det_train_kcp` (the deterministic package) | deterministic | deterministic; no warning |

- The backward reads torch's setting when it runs, including the autograd
  backward. For a given call, every rank of `ctx.group` runs the same entry points with
  the same torch deterministic setting.
- The package never falls back to the deterministic package. For
  deterministic results, call `flashinfer.cake_causal_conv1d_det_train_kcp`.
- A checkpoint recompute (`torch.utils.checkpoint`, reentrant or not) runs the
  same package.
- `cooperative=True` raises `ValueError`. The cooperative launch form belongs
  to the deterministic KCP package (`cake_causal_conv1d_det_kcp(...,
  cooperative=True)`).

### The backward

The head program writes the FP32 derivative of each continuing fragment's
first tokens (the halo adjoints) into the send layout. It matches the
deterministic route's head program. Then:

1. The halo exchange is issued.
2. The main program (`kcp_bwd_dyn`) runs before the exchange is waited, as if
   no fragment had tail rows. Its CTAs take units of tokens from lanes. Through
   Cluster Launch Control, they also take the lanes of CTAs that have not
   started. A kernel resident on another stream therefore leaves no work waiting
   for the SMs it holds. Its token arithmetic computes two adjacent channels
   per packed FP32 (`f32x2`) instruction, as the deterministic KCP backward does
   since FlashInfer PR #9142; `dx` and each run's FP32 `dweight` sum are bitwise
   those of the scalar (one channel per instruction) build. The FP32 `dweight` runs are added into four FP32
   accumulator copies in the order the runs end. The copies are summed once per
   channel tile.
3. After the wait, on a rank that receives tail rows, the tail-row program
   (`kcp_bwd_halo`) rewrites `dx` of the last three rows of each such
   fragment from the received rows.

Under any of these conditions, the one-program backward (`kcp_bwd`) runs after the wait instead:

- `bwd_reserve_sms >= 1`
- `C > 65536` (`kcp_jit.DYN_MAX_CHANNELS`, 32 x 2048)
- the call has no tokens

It is one ordinary launch. Its CTAs add their FP32 `dweight` rows into zeroed FP32 accumulators with unordered global reductions. The last CTA of each channel tile stores the rank-local partial and zeroes the tile's accumulators. Neither form uses a cooperative launch or a grid
barrier.

Both programs use the package's cached backward workspace for the device and
current stream. The workspace is zero before and after every launch. Inside a CUDA Graph capture the
package keeps one zeroed buffer per capture and stream. When CUDA's capture
id is not available, each captured call gets its own buffer.

### Rank agreement

Every rank of a call must pass a context built from the same partition. Each rank must also call the same entry (single or multi) with the same channel counts.
FlashInfer issue #8946 added a check of this agreement. The check runs on the first eager exchange of each (context, phase); the phase is forward or backward. It all-gathers one int64[8] fingerprint per rank over `ctx.group` and compares the fingerprints on the host. On a mismatch, every
rank raises the same `ValueError` naming the ranks and fields that differ.
This is the call's only collective beyond the halo exchange and its only host
sync, once per context and phase.

- No check is made by later calls on the same context and phase, by recompute with a handle (no exchange) or by graph replays.
- A first call inside a CUDA-graph capture raises: run one eager call first.
- With one rank there is no check.

The fingerprint fields are: protocol, mode, phase, entry and tensor count,
channels, world size and conv width, a partition hash, and launch form. The
layout is shared with the deterministic KCP package. Field 1 (mode) is 2 here
and 1 there.

### Determinism and accuracy, per output

Below, "same call" means the same inputs, device, SM count, package version,
world size, partition (CP layout) and keywords.

| Output | Repeats bit for bit | Relation to the deterministic KCP route | Accuracy vs exact FP64 |
| --- | --- | --- | --- |
| `y` | yes, for the same call | bitwise the deterministic KCP forward's | one BF16 rounding of the FP32 result |
| `hist` (history pool) | yes, for the same call | bitwise the deterministic KCP forward's | copies of other fragments' `x_local` rows |
| `dx` | yes, for the same call | bitwise the deterministic KCP backward's, on both backward forms (overlapped and one-program) | the deterministic KCP `dx`'s: within one BF16 ulp of an FP64 oracle |
| `dweight` (rank-local FP32 partial) | no: the cross-CTA sum is unordered | not bitwise | the sum of the ranks' partials is within the deterministic program's budget of an FP64 oracle, `5e-2 + 1e-4 abs(dw)` |
| checkpoint recompute (with or without a handle) | yes | | same `y` bits; the backward from it gives the same `dx` bits |

The measured `dweight` error was compared with the deterministic package's median FP64 error.
The measurement used 5 repeats, worlds 1, 2 and 4, and two shapes per rank:

- 32768 tokens x 10240 channels, one tensor
- 16384 tokens x 8192 channels, three tensors

It covered the contiguous, zigzag and zigzag-per-document layouts and five input cases: random and four adversarial cases. The rel-L2 error ratio was at most 1.016. The worst-element
error ratio (`max |err| / max |ref|`) was at most 1.29. Both bounds held on GB300 (sm_103a), GB200
(sm_100a) and R200 (sm_107a).

<!-- B200 W 1/2/4/8 numbers: pending -->

The package issues no reduction collective for `dweight`. Each rank returns
its partial, and the framework sums the partials across the ranks (for example, the
data-parallel or FSDP reduction of the weight gradient).

### Keyword controls

There are no environment variables. Every setting is a keyword argument.

| Keyword | Values | Default | Effect |
| --- | --- | --- | --- |
| `recompute_handle` | `CakeCausalConv1dNondetKcpRecomputeHandle()` or any object with an `entries` dict | `None` | The checkpoint recompute runs without the forward halo exchange (see below). |
| `cooperative` | `None`, `False` | `None` | `None` and `False` mean the same. `True` raises `ValueError`. |
| `bwd_reserve_sms` | int `K`, `0 <= K <` SM count | `None` (= 0) | `K >= 1` runs the one-program backward after the exchange with a grid `K x ctas_per_sm` CTAs short. Supported on SM100 / SM103 only; `ValueError` on SM107. A `bool` or a non-`int` raises `TypeError`. A value out of range raises `ValueError`. `y`, `dx` and the exchanges are unchanged; `dweight` stays within the same budget. Each rank may choose its own value. |
| `graph` | `"replay"` | omitted (eager) | Explicit `kcp` calls only: keyed CUDA-graph capture and replay. |

### Sharing the GPU with other streams

The overlapped backward is the default and needs no setting. Its CTAs claim
their tokens at run time. The repro from FlashInfer issues #7548 / #8365 timed the KCP
backward at world 4 with CUDA events, alone and next to a kernel on another
stream. It used two per-rank shapes:

- **Shape A:** 32768 tokens x 10240 channels, one tensor.
- **Shape B:** 16384 tokens x 8192 channels, three tensors.

A one-thread side kernel on GB200 (sm_100a, 152 SMs) barely changes the time:

| Side kernel on another stream | shape A next-to / alone | shape B next-to / alone |
| --- | --- | --- |
| one thread, 2 ms | 1.000 | 1.002 |
| one thread, 4 ms | 0.999 | 1.006 |

The next table shows a resident kernel on another stream that holds `k` SMs. `N` is the SM count. The
ratio columns give the time next to the resident kernel divided by the time alone. The `k = 0` rows give the
unloaded backward time. The previous non-deterministic backward ran as one program after the exchange.

| GPU | `k` | ideal `N / (N - k)` | shape A | shape B | previous backward, A / B |
| --- | --- | --- | --- | --- | --- |
| GB200 (sm_100a, 152 SMs) | 0 | 1 | 365.0 us | 493.1 us | 395.0 us / 560.5 us |
| GB200 | 16 | 1.118 | 1.087 (396.7 us) | 1.175 (579.5 us) | 667.5 us / 921.1 us |
| GB200 | 32 | 1.267 | 1.208 (440.8 us) | 1.286 (634.2 us) | 687.8 us / 914.1 us |
| B200 (sm_100a, 148 SMs) | 0 | 1 | 384.3 us | 519.2 us | 423.9 us / 583.7 us |
| B200 | 16 | 1.121 | 1.114 (428.0 us) | 1.190 (618.0 us) | 738.0 us / 973.8 us |
| B200 | 32 | 1.276 | 1.258 (483.3 us) | 1.295 (672.6 us) | 743.4 us / 975.9 us |

For shape A, the slowdown closely follows the share of SMs the resident kernel holds and stays at or below the ideal ratio. For
shape B it is above the ideal ratio.

`bwd_reserve_sms=K` is an opt-in alternative (SM100 / SM103 only). It runs the one-program backward
after the exchange with a grid `K x ctas_per_sm` CTAs short. A kernel holding up
to `K` SMs then does not push the top CTAs into a second wave. The unloaded
backward gives up the work of those `K` SMs. On sm_107a it raises
`ValueError`: there, the reserved grid costs up to 1.055x the default grid
unloaded (C 1024, 62585 tokens), which is above the opt-in's 1.040 limit.

### Checkpoint recompute without the halo exchange

FlashInfer issues #7815 / #7818 added this mode.
With `recompute_handle=CakeCausalConv1dNondetKcpRecomputeHandle()`, the checkpoint
recompute of a call issues no collective:

1. The original forward exchanges as always and stores `hist` in the handle (key `"conv"`). The handle retains `n_hist_rows x C x 2` bytes per tensor.
2. Torch's recompute of that forward inside the backward binds `hist` and launches the forward program on it, with no gather, no collective and no wait.
3. The backward still exchanges once.

The handle has these rules:

- One handle serves one call (single or multi) of one micro-step.
- Create it outside the outermost checkpoint region.
- Every rank passes a handle for the same call or none, at the same checkpoint nesting level. This is not checked: a mismatch hangs or corrupts data.
- A handle is refused with `graph="replay"` and inside a CUDA-graph capture.

### Multi-tensor calls

`cake_causal_conv1d_nondet_kcp_multi(xs, weights, ctx)` convolves `n >= 1`
tensors that share the context, such as a layer's q, k and v. It uses one halo exchange per
direction instead of `n`.

- Inputs: each `xs[i]` is BF16 `[T_r, C_i]` and each `weights[i]` is FP32 `[C_i, 4]`. Each `C_i` is a multiple of 8, and they may differ.
- Exchange: the tensors' send rows travel side by side in one `all_to_all_single`. After the wait, the rows are copied into each tensor's own pool.
- Programs: each tensor then runs the single call's programs. Outputs and `dx` are bitwise one single call per tensor. Each `dweight` is that tensor's own unordered partial.
- Keywords: one `bwd_reserve_sms` value applies to every tensor. One recompute handle covers the whole call; its `"conv"` entry holds the `n` history pools.
