# GLM5 low-latency MoE

This backend is a shape-specialized Blackwell decode path for the GLM5 MoE
layer. It fuses no-aux sigmoid top-k routing, activation quantization, the
gate/up projections, SwiGLU, the down projections, and routed/shared reduction
into two CUDA launches. It is intended for tensor-parallel decode with one to
four tokens per rank.

The CUDA implementation is adapted from the `fuse_more_kernels` branch of
[`yijingl-nvidia/TensorRT-LLM`](https://github.com/yijingl-nvidia/TensorRT-LLM/tree/fuse_more_kernels),
at source revision `73dd714d5`. The port replaces the Torch custom-op boundary
with TVM FFI, uses caller-owned outputs and reusable workspaces, and integrates
with FlashInfer's JIT, AOT, trace, test, and benchmark infrastructure.

## Supported contract

| Property | Supported value |
| --- | --- |
| Architecture | SM100 and SM103 |
| Tokens per call | 1 through 4 |
| Routed/shared experts | 256 routed plus 1 shared |
| Routing | sigmoid, no-aux bias, top-k 8, normalized then scaled |
| Hidden size | 6144 |
| Global intermediate size | 2048 |
| Local intermediate size | 256 (TP8) or 512 (TP4) |
| Activations/output | BF16 input, FP16 expert handoff, BF16 output |
| Weights/scales | FP8 E4M3 with FP32 128x128 block scales |

`glm5_low_latency_moe` returns the local TP contribution. The serving framework must
all-reduce that tensor across TP ranks before applying the residual connection.
The router GEMM that produces `router_logits` is also outside this operator.

## Weight preparation

Call `prepare_glm5_low_latency_moe_weights` once during model loading. The raw GLM5
checkpoint conventions are:

- shared gate/up: `[gate, up]`, shape `[2 * I, 6144]`;
- routed gate/up: `[up, gate]`, shape `[256, 2 * I, 6144]`;
- routed down: row-major `[256, 6144, I]`;
- shared down: row-major `[6144, I]`.

The up-projection helper normalizes the half ordering and packs the weight into
`[257, I / 64, 8, 98304]`, the lane layout consumed by the FP8 MMA kernel. Down
weights remain row-major and are staged with TMA at runtime. Preparation is not
part of decode latency.

For repeated calls, allocate `Glm5LowLatencyMoeWorkspace` once and pass both it and
an `out` tensor. This avoids allocator work in the serving loop and keeps tensor
addresses stable for CUDA graph capture.

## Kernel pipeline

The expert-up launch selects top-8 routed experts from biased sigmoid scores,
normalizes the corresponding unbiased scores, quantizes each 128-column input
block, and computes the shared and routed gate/up projections. It writes nine
FP16 SwiGLU slots per token: one shared slot followed by eight routed slots.

The expert-down launch buckets repeated expert IDs, stages the selected down
weights, computes all nine down projections, applies routing weights to the
routed slots, and writes the BF16 local sum. The public wrapper supports one or
two packed up-weight stages and either TMA or cp.async loading.

## Validation and performance

The dump replay test uses the eight rank-specific tensors and saved PyTorch
reference outputs from a GLM5 TP8 serving run. On eight B200 GPUs, all ranks
passed for both stage counts and both weight loaders. With the default
two-stage TMA path, the observed per-rank maximum absolute error was
`6.87e-05` through `4.08e-04`, within the established rank-specific FP8
thresholds.

For `M=4`, 20 warmups, and 100 timed iterations, the two-launch operator
averaged `43.349 us` across eight B200 ranks; individual rank means ranged from
`41.555 us` to `46.321 us`. This timing excludes the router GEMM and TP
all-reduce, matching the operator boundary described above.

Run the replay and benchmark with:

```bash
FLASHINFER_GLM5_LOW_LATENCY_MOE_DUMP_DIR=/path/to/dumps \
  torchrun --nproc_per_node=8 -m pytest \
  tests/moe/test_glm5_low_latency_moe.py -v -m "gpu_8 and arch_blackwell"

torchrun --nproc_per_node=8 benchmarks/bench_glm5_low_latency_moe.py \
  --dump-dir /path/to/dumps --tokens 4 --warmup 20 --iterations 100
```
