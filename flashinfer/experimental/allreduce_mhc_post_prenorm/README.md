# MNNVL all-reduce fused with the mHC post-map and pre-norm partials (SM100/SM103, experimental)

Status: experimental API and experimental backend, JIT-only. Not part of automatic routing.

`flashinfer.comm.allreduce_mhc_post_prenorm` serves the decode-time boundary between two mHC
(manifold-constrained hyper-connection) sublayers of a tensor-parallel model such as GLM-5.3-Flash.
It all-reduces the sublayer output and, in the same kernel, applies the mHC post-map and the
split-K partials of the next sublayer's pre-norm projection:

```
x            = sum over ranks of input                                         (bf16)
new_r[j]     = post_layer_mix[j] * x + sum_k comb_res_mix[k, j] * residual[k]  (fp32)
residual_out = bf16(new_r)
yp[s, m, n]  = sum_{j, h in split s} fn[n, j * H + h] * new_r[m, j, h]          (fp32)
rp[s, m]     = sum_{j, h in split s} new_r[m, j, h] ** 2                        (fp32)
```

`yp` and `rp` come from the unrounded `new_r`, over 8 contiguous hidden splits of `H / 8`; the
caller's pre-norm finalize sums them.

## Why

Without it, every mHC boundary of a decode step runs FlashInfer's MNNVL one-shot all-reduce and
then a separate post-map / pre-norm kernel that reads `x` and the residual streams back from
global memory. At decode token counts both kernels are latency-bound, and GLM-5.3-Flash has about
90 such boundaries per step. Here the epilogue runs on the reduced values while they are in
registers, in the CTAs that polled them, so the boundary costs one launch and one pass over its
operands.

It is experimental because it is a new fused operator for one model family with a narrow
supported surface (one world size, hidden size and stream count), and its API contract (the
split-K partial layout handed to the caller's finalize) is not settled.

## Usage

Calling the API is the opt-in: it warns once with `ExperimentalWarning` and needs no environment
variable. Every rank of the tensor-parallel group makes the same call.

```python
import torch
import flashinfer.comm as comm

# workspace: the MNNVLAllReduceFusionWorkspace of the 8-rank group (shared with
# trtllm_mnnvl_allreduce). Build the module once, outside CUDA-graph capture:
comm.allreduce_mhc_post_prenorm_prepare()

num_tokens, hc_mult, hidden = residual.shape  # bf16 [T, 4, 4096]
if comm.allreduce_mhc_post_prenorm_supported(
    workspace, num_tokens, hidden, hc_mult, torch.bfloat16
):
    yp, rp, residual_out = comm.allreduce_mhc_post_prenorm(
        partial,         # bf16 [T, 4096], this rank's un-reduced sublayer output
        residual,        # bf16 [T, 4, 4096]
        post_layer_mix,  # fp32 [T, 4]
        comb_res_mix,    # fp32 [T, 4, 4]
        fn,              # fp32 [24, 4 * 4096]
        workspace,
    )
    # yp: fp32 [8, T, 24], rp: fp32 [8, T], residual_out: bf16 [T, 4, 4096]
else:
    ...  # the caller's unfused path
```

`examples/experimental/allreduce_mhc_post_prenorm.py` runs the call on 8 ranks and checks it
against the stock all-reduce plus a PyTorch epilogue.

## Supported surface and limits

The fused kernel runs when all of these hold; `allreduce_mhc_post_prenorm_supported` checks them:

- an `MNNVLAllReduceFusionWorkspace` of world size 8 (TP8 in one MNNVL domain, e.g. two 4-GPU
  GB200 / GB300 trays) with one-shot capacity for the call's tokens;
- hidden size 4096, 4 residual streams (`hc_mult`), `mix = 24` pre-norm rows, bf16 `input` and
  `residual`, fp32 `post_layer_mix`, `comb_res_mix` and `fn`;
- 1 to 32 tokens;
- compute capability 10.x (SM100 / SM103); the module is built for the SM10x targets of the JIT
  compilation context (`sm_100a`, `sm_103a`);
- during CUDA-graph capture, the module is already loaded (`allreduce_mhc_post_prenorm_prepare`).

Anywhere else `allreduce_mhc_post_prenorm` takes a composable path with the same outputs: the
stock MNNVL all-reduce, then the epilogue in PyTorch (fp32 post-map, fp64 partials). It is a
correctness fallback, not a fast path; callers that have their own unfused path should take it
when the support query returns False. Each rank chooses its path on its own, so inside the
surface (e.g. a rank whose module failed to build) the composable path all-reduces with the stock
one-shot kernel, the fused kernel's protocol: a group whose ranks chose differently still
completes the call, the ranks' epilogue results then differing in rounding only.

Inputs and preallocated outputs may have any strides on both paths, and `residual_out` may be
`residual` (an in-place update). With programmatic dependent launch the kernel loads `fn` before
it waits on the preceding kernel, as it would a weight, so `fn` must not be written by the kernel
launched right before the call; a non-contiguous `fn` is copied and that call launches without
programmatic dependent launch.

The API takes no part in automatic routing: there is no `backend="auto"` selection, autotuning
or trace-apply for it, and `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` does not affect it. It
is never registered for AOT builds and has no trace template yet.

## Design

All-reduce: the one-shot Lamport protocol of `include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh`
on the same workspace (buffer layout `[token][rank][hidden]`, flags, triple-buffer rotation,
dirty-buffer clearing, arrival counter, sanitized multicast broadcast), so these kernels
interleave freely with the stock one-shot and two-shot kernels on one workspace. The reduction is
the stock deterministic one, fp32 in rank order 0..7 and one bf16 round-to-nearest-even: `x` is
bit-identical to the stock one-shot kernel's output.

Seven kernel instantiations in one JIT module, selected by token count, every grid small enough
to be co-resident (at most 128 CTAs x 256 threads or 192 x 128), so Lamport polling never waits on
an unscheduled CTA:

| tokens | kernel | geometry |
|---|---|---|
| 1, 2 | `arMhcPostRowSplitKernel` | each CTA reduces a whole 512-wide split slice and projects 3 `fn` rows held in registers; no cross-CTA combine |
| 3 | `arMhcPostLogitSplitKernel` | 2-CTA clusters; partials sent with `st.async` to cluster rank 0 and summed in fixed order |
| 4 | `arMhcPostRowGroupKernel` | the row split with 6 `fn` rows per CTA |
| 5-8, 9-16, 17-32 | `arMhcPostPrenormKernel` | 4-CTA hidden-split clusters with 2-, 4- and 8-token tiles, combined through distributed shared memory in fixed order |

The post-map runs as packed `mul.rn.f32x2` / `fma.rn.f32x2` with the scalar operation order. With
programmatic dependent launch only `fn` (a weight) is read before `griddepcontrol.wait`. In the
1- to 4-token kernels several CTAs read the residual slice that one of them overwrites, so an
in-place call passes those kernels a separate `residual_out` buffer and copies it back; from 5
tokens every residual element is read and written by the same thread and the call stays in place.

Built with CUDA 13.0 for `sm_103a` (`ptxas`): 72, 72, 96, 100, 106, 110 and 169 registers for
1, 2, 3, 4, 5-8, 9-16 and 17-32 tokens, no spills, at most 17,416 bytes of static shared memory.

## Numerics

The module is built without fast math (`gen_jit_spec(..., use_fast_math=False)`): no `.ftz` in
the PTX, and `FMUL2` / `FFMA2` without `.FTZ` in the SASS. `residual_out` follows the same
per-element operation order in every geometry and is within one bf16 ulp of a PyTorch fp32
post-map (at least 99.5 % bit-identical; the kernel's fused multiply-adds round once). `yp` and
`rp` differ from an fp64 reference by fp32 summation order only.

## Measurements

FlashInfer main (0.7.2) with these seven kernels on 8 GPUs in one MNNVL domain (two 4-GPU trays):
NVIDIA GB300 (SM103) and NVIDIA GB200 (SM100), CUDA 13.0, torch 2.13.0+cu130. The unfused pair
is FlashInfer main's MNNVL one-shot all-reduce followed by the serving stack's TileLang mHC
post-map / pre-norm kernel. Each variant is a CUDA graph of 90 back-to-back boundaries with
distinct operand sets (larger than L2); microseconds per boundary, maximum over the 8 ranks, with
the two graphs replayed alternately in one process (per-rank median of 120 alternating pairs).

| tokens | GB300 unfused (us) | GB300 fused (us) | GB300 speedup | GB200 unfused (us) | GB200 fused (us) | GB200 speedup |
|---|---|---|---|---|---|---|
| 1 | 11.75 | 5.90 | 1.99x | 11.78 | 5.83 | 2.02x |
| 2 | 12.18 | 6.05 | 2.01x | 12.15 | 6.01 | 2.02x |
| 4 | 11.64 | 5.65 | 2.06x | 12.13 | 5.91 | 2.05x |
| 8 | 12.07 | 6.80 | 1.77x | 12.42 | 7.10 | 1.75x |
| 16 | 12.91 | 7.82 | 1.65x | 13.29 | 8.02 | 1.66x |
| 24 | 13.76 | 8.95 | 1.54x | 14.02 | 8.99 | 1.56x |
| 32 | 13.96 | 9.20 | 1.52x | 14.46 | 9.61 | 1.50x |

The fused kernel is faster at every token count from 1 to 32 on both: 1.52x to 2.06x on GB300 and
1.50x to 2.05x on GB200, 4.7 to 8.0 us saved per boundary; 1.88x (GB300) and 1.89x (GB200)
weighted by GLM-5.3-Flash's decode token-count mix. Timing each graph on its own gives 1.45x to
2.08x.

End to end, GLM-5.3-Flash NVFP4 on vLLM with TP8 across two GB300 trays (multi-turn replay),
against vLLM with stock FlashInfer main: mean inter-token latency -9.3 % at concurrency 1 (4.386
to 3.977 ms) and -7.4 % at concurrency 8 (5.636 to 5.221 ms), output throughput +4.7 % and +0.1 %.
The median decode step is 0.31 to 0.51 ms shorter at every decode batch size from 1 to 8 tokens.

Accuracy, against the same build with the unfused path (measured on an earlier revision of this
change whose seven kernels compile to the same SASS): GSM8K (all 1319 test questions, greedy)
1288 vs 1287 correct (McNemar p = 1.00). The fused run reached the 8192-token cap on 4 questions,
the unfused run on none (Fisher p = 0.12). Teacher-forced perplexity is identical, but prefill
does not use this kernel.

## Validation status

- Builds with CUDA 13.0 and 13.1 for `sm_100a` and `sm_103a`.
- GB300 (SM103) and GB200 (SM100), two 4-GPU trays each: every test in
  `tests/experimental/test_allreduce_mhc_post_prenorm.py` passes on all 8 ranks, including
  CUDA-graph capture and replay, interleaving with the stock all-reduce, in-place residual
  updates, strided operands, and a rank taking the composable path while the others run the
  fused kernel.

## Graduation

Graduating needs a settled API contract (the partial layout and the name), a trace template, a
docs page, AOT registration if it should ship pre-compiled, and the tests moved to `tests/comm/`.
Widening the surface (other world sizes, hidden sizes or stream counts) is separate work.
