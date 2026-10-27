# Experimental mHC pre big-fuse with fused RMSNorm

`flashinfer.mhc.mhc_pre_big_fuse_with_norm` is an **experimental API** with an
experimental implementation. It is the last step of a manifold-constrained
hyper-connection (mHC) boundary with `HC = 4` residual streams, in the decode
regime of models that use mHC. The op consumes the split-K partials of the mHC
mix projection, so it runs right after the kernel that produces them. It
returns the mixing coefficients for the next boundary and the RMSNorm'd input
of the next sublayer:

```
rms         = rsqrt(sum_s gemm_out_sqrsum[s] / (4 H) + rms_eps)
mixes       = (sum_s gemm_out_mul[s]) * rms
pre         = sigmoid(mixes[0:4] * hc_scale[0] + hc_base[0:4]) + hc_pre_eps
post_mix    = sigmoid(mixes[4:8] * hc_scale[1] + hc_base[4:8]) * hc_post_mult_value
comb_mix    = sinkhorn_repeat-iteration Sinkhorn of
              softmax(mixes[8:24].view(4, 4) * hc_scale[2] + hc_base[8:24]) + hc_sinkhorn_eps
ol          = sum_j pre[j] * residual[:, j]                                   (fp32)
layer_input = bf16(float(bf16(ol)) * rsqrt(mean(ol^2) + norm_eps) * norm_weight)
```

It differs from the stable `mhc_pre_big_fuse` in its contract: the output
RMSNorm (`norm_weight`, `norm_eps`) is fused into `layer_input`, and the split
count is read from the leading dimension of the partials.

The experimental lifecycle is owned by `@nv-yunzheq` and tracked in issue #TODO.

## Inputs and outputs

| Tensor | dtype | Shape |
|---|---|---|
| `gemm_out_mul` | fp32 | `[NS, T, 24]` |
| `gemm_out_sqrsum` | fp32 | `[NS, T]` |
| `hc_scale` / `hc_base` | fp32 | `[3]` / `[24]` |
| `residual` | bf16 | `[T, 4, H]` |
| `norm_weight` | bf16 | `[H]` |
| `post_mix` (out) | fp32 | `[T, 4]` |
| `comb_mix` (out) | fp32 | `[T, 16]` (or a caller-owned `[T, 4, 4]`) |
| `layer_input` (out) | bf16 | `[T, H]` |

The outputs are allocated when omitted. The model constants have no defaults,
because they are part of the numerical contract.

## Implementations

- **Cluster kernel** (`csrc/mhc_pre_big_fuse_with_norm_cluster.cu`, SM100 /
  SM103). It covers `H = 4096`, `1 <= T <= 32` and `NS` in `{8, 64}`. Its
  compiled-in constants are `rms_eps = norm_eps = 1e-5`,
  `hc_pre_eps = hc_sinkhorn_eps = 1e-6`, `hc_post_mult_value = 2` and
  `sinkhorn_repeat = 20`; the support check compares these as fp32. Inputs and
  outputs must be dense, contiguous and 16-byte aligned.
  - Each token runs on one thread-block cluster: 4 CTAs, or 2 CTAs when
    `T > 24`. Each CTA owns 1024 channels.
  - Two warps per CTA collapse the four bf16 streams. They all-gather the
    `sum(ol^2)` partials over distributed shared memory (`st.async` +
    `mbarrier`), then apply the RMSNorm.
  - Warp 0 of CTA rank 1 computes the softmax and a division-free 20-iteration
    Sinkhorn. Warp 0 of the post rank writes `post_mix`.
  - `enable_pdl=True` launches with programmatic dependent launch. The kernel
    reads no input before `griddepcontrol.wait`.
- **Composable implementation** (`backend.composable`). An fp32 PyTorch
  implementation of the contract for any `NS`, `T` and `H`. It serves every
  call outside the kernel's envelope, so the API always returns a result.

`mhc_pre_big_fuse_with_norm_supported(...)` reports which path a call takes.
Callers that have their own fast path for the op should keep it whenever this
returns `False`.

### Numerics

The module is compiled without `-use_fast_math`:

- `expf` is libdevice, and the post sigmoid uses IEEE division.
- The pre sigmoids and the final Sinkhorn `1/G` use Newton-refined reciprocals.
- `rsqrt.approx` is applied only to arguments `>= 1e-5`.
- fp32 add, mul and fma never flush to zero.

The Sinkhorn is kept in diagonal-scaling form `c = r K s / G`:

- Cofactor products replace the per-step divisions.
- The common denominator is renormalised by an exact power of two.
- An exact power-of-two rebalance of `r` and `s` after every column half-step
  keeps both within `(2^-64, 2^50)` for every finite input.

`comb_mix` is therefore finite for all finite logits, including the extreme
patterns in the tests.

Against the composable implementation, the kernel holds these tolerances:

| Output | atol | rtol / bound |
|---|---|---|
| `post_mix` | `2e-6` | rtol `2e-6` |
| `comb_mix` | `1e-7` | rtol `1e-5` |
| `layer_input` | — | within `2^-5` relative, and at least 99.5% bit-identical |

## Opt-in, gating and routing

- Calling `mhc_pre_big_fuse_with_norm`, `mhc_pre_big_fuse_with_norm_supported`
  or `mhc_pre_big_fuse_with_norm_warmup` is the opt-in. Each warns once with an
  `ExperimentalWarning`; no environment variable is needed.
- The op does not take part in any automatic routing: no stable API dispatches
  to it, and it has no autotuning or trace-apply support.
  `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` has no effect on it.
- `FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`, read at call time, forces the
  composable implementation.
- The kernel is JIT-only. It is not part of the `flashinfer-jit-cache` /
  `flashinfer-cubin` packages.

## CUDA graphs

The kernel module is never built during CUDA-graph capture. A capturing call
made before the module is loaded captures the composable implementation.

Call `mhc_pre_big_fuse_with_norm_warmup()` before capture to build and load the
kernel. Pass caller-owned `post_mix`, `comb_mix` and `layer_input` buffers to
captured calls.

## Limitations

- The kernel targets SM100 / SM103 only. It has been validated on GB300
  (SM103); SM100 is compiled but not separately validated.
- The kernel covers `H = 4096` and `HC = 4` only, with the constants listed
  above. Prefill-sized batches (`T > 32`) and other models' constants run the
  composable implementation, which is much slower.
- The kernel supports `NS` in `{8, 64}`.

## Example and tests

```bash
python examples/experimental/mhc_pre_big_fuse_with_norm.py --tokens 4 --num-splits 8
pytest tests/experimental/test_mhc_pre_big_fuse_with_norm.py
```

## Graduation

Graduation requires:

- an agreed placement next to the stable `mhc_pre_big_fuse` APIs (a variant
  flag versus a separate function);
- runtime constants in place of the compiled-in model constants, or a
  per-constant-set specialization;
- validation on SM100;
- a trace template.
