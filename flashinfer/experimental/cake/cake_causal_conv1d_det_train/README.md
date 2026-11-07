# Cake deterministic packed causal conv1d + SiLU training backend (SM100 / SM103 / SM107)

Backend package of the deterministic (`det`) packed width-4 depthwise causal
convolution + SiLU training routes: the single-rank route
(`flashinfer.cake_causal_conv1d_det_train`) and the context-parallel (KCP)
route (`flashinfer.cake_causal_conv1d_det_train_kcp`, module `kcp` of this
package).  The facades are stable FlashInfer APIs; this package, under
`flashinfer.experimental.cake`, is their backend.

## Single-rank route

This section is maintained with the single-rank package
(`flashinfer.cake_causal_conv1d_det_train`).

## Context-parallel (KCP) route

Every rank convolves its fragments of a packed batch.  The first three tokens
of a fragment that continues a document read the three document tokens before
it (history rows) from the ranks that hold them; the backward sends the
derivative of those first tokens back to the predecessor fragments.  Tokens
never move otherwise: there is no activation concatenation and no
full-sequence gather.  The route keeps the package's FP32-tap contract: FP32
taps, no bias, FP32 arithmetic, one BF16 rounding per output, and the SiLU
derivative taken at the FP32 preactivation.

### Entry points

```python
from flashinfer.cake_causal_conv1d_det_train_kcp import (
    cake_causal_conv1d_det_kcp,          # autograd, one tensor
    cake_causal_conv1d_det_kcp_multi,    # autograd, n tensors of one layer (q / k / v), one exchange per direction
    CakeCausalConv1dDetKcpRecomputeHandle,
)

y = cake_causal_conv1d_det_kcp(x_local, weight, ctx)                     # cooperative=None: False
q, k, v = cake_causal_conv1d_det_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)
```

| operand | shape | dtype | notes |
|---|---|---|---|
| `x_local`, `y`, `dx` | `[T_r, C]` or `[1, T_r, C]` | BF16 | contiguous; the rank's fragments back to back in the order of `ctx.cu_seqlens_local`; `C` a multiple of 8; `T_r` may be 0 |
| `weight`, `dweight` | `[C, 4]` | FP32 | `weight[c, 3]` multiplies the current token; `dweight` is the rank-local partial |
| `ctx` | | | the rank's KCP context, built by the calling family's context builder (for example `cake_kda_det_train_kcp_build_context(..., conv_kernel_size=4)`): fragment table, halo plan, device tables, process group |

Every rank of `ctx.group` calls the entry once per layer, in the same order,
with the same tensor count and channel counts; the ranks are not compared (a
mismatch hangs the exchange or corrupts the history rows).  The module `kcp`
also holds the explicit calls: `cake_causal_conv1d_det_train_kcp_forward`
(returns `y` and the history pool `hist`), `cake_causal_conv1d_det_train_kcp_backward`
(returns `dx` and the `dweight` partial) and their `_multi_` forms, with the
keyword `graph="replay"` (keyed CUDA-graph capture and replay, bitwise the
eager result; call `cake_causal_conv1d_det_train_kcp_graph_cache_clear()`
before `torch.distributed.destroy_process_group()`).  Every switch is a keyword
argument; the backend reads no environment variable.

### Launch forms (`cooperative`)

The keyword selects the launch form of the backward's main program; the
forward and the backward's head program are the same in both forms.

| `cooperative` | program | behavior |
|---|---|---|
| `False` (default; `None` = `kcp.COOPERATIVE_DEFAULT`) | `kcp_bwd_split` + its `reduce` program | an ordinary launch one SM short of the co-resident CTA count writes per-CTA partial rows; a programmatically dependent reduce program on the same stream sums them in a fixed order.  It starts beside a small kernel resident on another stream (for example a communication kernel); more such kernels delay some CTAs, never the whole launch |
| `True` | `kcp_bwd` | one cooperative launch whose grid barrier precedes the fixed-order `dweight` reduction.  It starts only when all its CTAs fit at once, so it waits for every kernel of another stream that occupies an SM |

Every rank of a group passes the same `cooperative` value.  The form changes
no collective, so a mismatch does not hang, but the ranks' `dweight` partials
would then mix two reduction orders.

### Halo exchange

One `all_to_all_single` per direction and call:

- Forward: the history rows (up to three per continuing fragment) travel to
  the ranks of the fragments that read them, into the history pool `hist`
  (`[n_hist_rows, C]` BF16).  `hist` is saved for the backward; the backward
  does not exchange it again.
- Backward: the head program writes the FP32 derivative of each continuing
  fragment's first tokens (the halo adjoints; a token feeds at most three
  predecessor fragments) into the send layout; the exchange delivers them to
  the predecessors as `dpre_tail` (`[n_tail_rows, C]` FP32), and the main
  program adds them to the last rows' `dx` before the one BF16 rounding.

The multi entry exchanges the rows of its `n` tensors side by side in one
collective per direction and copies each tensor's rows into its own pool after
the wait; each tensor then runs the single call's programs, so its outputs and
gradients are bitwise those of one single call per tensor with the same
`cooperative`.  With one rank no collective is issued.

### Recompute handle

`recompute_handle=CakeCausalConv1dDetKcpRecomputeHandle()` makes the checkpoint
recompute of a call issue no collective: the original forward exchanges and
stores `hist` in the handle (key `"conv"`; `n_hist_rows x C x 2` bytes retained
per tensor), and torch's recompute of that forward inside the backward binds it
and launches the forward program on it, with no gather, no collective and no
wait.  The backward still exchanges once.  One handle serves one call (single
or multi) of one micro-step and is created outside the outermost checkpoint
region; every rank passes a handle for the same call or none.  A handle is
refused with `graph="replay"` and inside a CUDA-graph capture.

### Determinism

"Repeats bit for bit" means: the same bits on every call with identical inputs,
the same device type with the same SM count, the same configuration (shapes and
channel counts; graph replay and the eager path give the same bits), the same
CP layout (the context's partition) and the same `cooperative` value on every
rank.  Bits across CP layouts or SM counts are not guaranteed.
`torch.use_deterministic_algorithms(True)` raises no alert on either form.

| output | repeats bit for bit | the two `cooperative` forms | notes |
|---|---|---|---|
| `y` | yes | bitwise equal | within one BF16 ulp of an FP64 oracle; this revision's `y` is also bitwise the single-rank `y` of the same documents (the forward runs the single-rank forward's instructions with history rows), a property of this revision, not a guarantee |
| `hist` (history pool, the saved set's tensor) | yes | bitwise equal | copies of `x_local` rows of other fragments |
| halo adjoints (`dpre` rows sent, `dpre_tail` received) | yes | bitwise equal | written by the head program, which is the same in both forms |
| `dx` | yes | bitwise equal | within one BF16 ulp of an FP64 oracle; not bitwise the single-rank `dx`, which differentiates the BF16-rounded preactivation (within `1e-2 + 1e-2 abs(dx_single)` of it) |
| `dweight` (rank-local FP32 partial) | yes, within one form | bits differ | a fixed-order sum of the rank's partial rows; the two forms sum different partial rows.  The sum of the ranks' partials is within `5e-2 + 1e-4 abs(dw)` of an FP64 oracle |
| checkpoint replay | yes | | with a recompute handle `y` and the saved set `(x_local, weight, hist)` are bitwise those of the exchanging recompute, and the gradients bitwise those of the saved-set path (no checkpoint) |

The package issues no reduction collective for `dweight`: each rank returns
its partial, and the sum over the ranks is the framework's (for example the
data-parallel or FSDP reduction of the weight gradient).  Whether that sum
repeats bit for bit depends on the framework's reduction; a reduction in a
fixed rank order keeps it.
