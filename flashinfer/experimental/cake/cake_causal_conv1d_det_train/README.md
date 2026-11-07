# Cake deterministic packed causal conv1d + SiLU training backend (SM100 / SM103 / SM107)

Backend package of the deterministic (`det`) packed width-4 depthwise causal
convolution + SiLU training routes: the single-rank route
(`flashinfer.cake_causal_conv1d_det_train`) and the context-parallel (KCP)
route (`flashinfer.cake_causal_conv1d_det_train_kcp`, module `kcp` of this
package).  The facades are stable FlashInfer APIs; this package, under
`flashinfer.experimental.cake`, is their backend.

## Single-rank route

The stable entry points live in `flashinfer/cake_causal_conv1d_det_train.py`; that module imports this package on first use, and FlashInfer does not select it automatically (check `is_cake_causal_conv1d_det_train_supported(device)` and call it).

### Entry points (`flashinfer/cake_causal_conv1d_det_train.py`)

```python
y = cake_causal_conv1d_det_train_forward(x, weight, cu_seqlens, bias=None, out=None, workspace=None)
dx, dweight = cake_causal_conv1d_det_train_backward(x, weight, cu_seqlens, dy)                 # no bias
dx, dweight, dbias = cake_causal_conv1d_det_train_backward(x, weight, cu_seqlens, dy, bias=bias)
y = cake_causal_conv1d_det(x, weight, cu_seqlens, bias,                                        # autograd
                           silu_grad_preactivation="bf16", bwd_grid_mult=1)
fws = cake_causal_conv1d_det_train_forward_workspace(S, C, device)
ws = cake_causal_conv1d_det_train_backward_workspace(S, C, device, bwd_grid_mult=1)
is_cake_causal_conv1d_det_train_supported(device)
```

`x` is one packed BF16 `[S, C]` (or `[1, S, C]`) activation, and `C` must be a multiple of 8. `cu_seqlens` holds the document boundaries as int32 `[num_docs + 1]` prefix sums on the device. Each document starts with zero history. The taps `weight[C, 4]` and the optional `bias[C]` can each be FP32 or BF16, independently of each other. Arithmetic is FP32, and the bias is added last, in the same order as flash-linear-attention. `y` and `dx` are BF16. `dweight` and `dbias` are FP32 or BF16: they take the dtype of their parameter unless you pass preallocated outputs.

- **Forward:** one ordinary launch. Co-resident CTAs take token chunks from int32 counters in the forward workspace. The workspace must be zero before the first launch, and every launch leaves it zero.
- **Backward:** two ordinary (non-cooperative) launches on one stream. A persistent main program writes `dx` and one FP32 partial row of `dweight` (and `dbias`) per CTA. A reduce program then sums those rows in a fixed order. No floating-point atomics are used.
- **Autograd:** saves only `x`, `weight`, `cu_seqlens` and `bias`. The backward recomputes the preactivation.
- **Libraries:** load prebuilt from `aot/` when present; otherwise they are JIT-built from `build.json`. Once built, the calls can be captured in a CUDA Graph. Build every library for the arch before capture (see `cake_jit.specs_for_arch`).
- The host never reads `cu_seqlens`, so the calls do not sync with the host. `S == 0` returns empty outputs without launching.

### Determinism, per output

Below, "same call" means the same inputs, device architecture, package version and keyword settings. The keyword settings are the taps, bias and output dtypes, `silu_grad_preactivation`, and `bwd_grid_mult`. Every output was bitwise identical over 5 repeated runs on every measured shape on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

| Output | Repeats bit for bit when | What changes its bits | Accuracy vs exact FP64 |
| --- | --- | --- | --- |
| `y` | same call, whichever CTA computes a chunk | Not `bwd_grid_mult`. May change between package versions (unchanged in this version) | one BF16 rounding of the FP32 result |
| `dx` | same call | Not `bwd_grid_mult` or the backward tile plan. May change between package versions (unchanged in this version) | within one BF16 ulp under both `silu_grad_preactivation` policies |
| `dweight` | same call, including `bwd_grid_mult` | Summation order follows the backward tile plan and `bwd_grid_mult`, so different settings differ in the last bits. Package version | FP32 output: about 3e-5 rel-L2 with `"bf16"`, about 2e-7 with `"fp32"` (measured worst 1.39e-7). BF16 output: limited by BF16 storage (measured 1.6e-3 to 1.8e-3) |
| `dbias` | same as `dweight` | same as `dweight` | FP32 output with `"fp32"`: measured worst 1.49e-7 rel-L2 |

- With `silu_grad_preactivation="bf16"` (the default), SiLU' is evaluated at the BF16-rounded preactivation, the same boundary flash-linear-attention uses. With `"fp32"`, it is evaluated at the unrounded FP32 preactivation and the final reduction runs in FP64. The `"fp32"` program is slower.
- This version runs 32-channel-vector tiles at C 224 and C 520-1016 on every backward route, and at C 576-768 on the BF16-tap forward. At those backward widths, `dweight` and `dbias` bits therefore differ from the previous version. `y` and `dx` keep their bits, and the accuracy bounds are unchanged.
- Because the backward recomputes the preactivation, a checkpointed (recomputed) forward gives the same `y` bits. The backward from it gives the same gradient bits.

### Keyword controls

There are no environment variables. Every setting is a keyword argument.

| Keyword | Values | Default | Effect |
| --- | --- | --- | --- |
| `silu_grad_preactivation` | `"bf16"`, `"fp32"` | `"bf16"` | Sets where the backward evaluates SiLU'. `"fp32"` is more accurate and slower. |
| `bwd_grid_mult` | `1` to `4` | `1` | Splits the backward work more finely. Changes the last bits of `dweight` / `dbias`, never `y` / `dx`. Size the backward workspace with the same or a larger value. |
| `bias` | `None`, or `[C]` FP32 / BF16 | `None` | Adds the bias. The backward then also returns `dbias`. |
| `out` | BF16 tensor | `None` | Preallocated forward output `y`. |
| `dx`, `dweight`, `dbias` | tensors | `None` | Preallocated backward outputs. `dweight` / `dbias` can be FP32 or BF16. |
| `workspace` | from `cake_causal_conv1d_det_train_forward_workspace` / `cake_causal_conv1d_det_train_backward_workspace` | `None` | Workspace you allocate yourself for the forward or backward. |

### Sharing the GPU with other streams

The backward's main program is persistent, so a kernel that stays resident on another stream slows it down. The repro from FlashInfer issue #7548 timed the backward alone and next to a resident kernel on another stream, using CUDA events over 18 shapes each. With `bwd_grid_mult=1`, the median ratio of time next to the side kernel over time alone was 1.43-1.45 on GB300 (sm_103a, 152 SMs) and 1.42-1.48 on R200 (sm_107a, 212 SMs). That range covers three backward routes (default, bias, `"fp32"` SiLU') and two side kernels (one-thread and NCCL-like). A finer split costs some time when the backward runs alone but cuts the slowdown. The table below shows medians for the NCCL-like side kernel on the default route:

| `bwd_grid_mult` | GB300 next-to / alone | GB300 alone (us) | R200 next-to / alone | R200 alone (us) |
| --- | --- | --- | --- | --- |
| 1 | 1.45 | 142.0 | 1.42 | 94.1 |
| 2 | 1.20 | 153.0 | 1.17 | 105.6 |
| 3 | 1.13 | 162.2 | 1.10 | 114.5 |
| 4 | 1.09 | 168.8 | 1.06 | 122.8 |

The default stays 1 because it is fastest alone. Set 2-4 when the backward usually overlaps collectives, and size the backward workspace with the same or a larger `bwd_grid_mult`.

## Context-parallel (KCP) route

This section is maintained with the context-parallel package
(`flashinfer.cake_causal_conv1d_det_train_kcp`).
