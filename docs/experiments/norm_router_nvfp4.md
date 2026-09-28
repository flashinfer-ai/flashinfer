# Add-RMSNorm, padded NVFP4, and router fusion experiment

This branch builds on the corrected single-CTA NVFP4 arithmetic. It contains two
independently selectable changes. Neither is selected automatically by a model.

## Direct padded output

`flashinfer.cute_dsl.add_rmsnorm_fp4quant` accepts `output_hidden_size=K`.
Normalization still uses the logical input width H, including its RMS denominator.
The residual and optional normalized output retain H columns. Only FP4 payload
and E4M3 scale outputs use K columns, with zeros written into the tail in the
same kernel. No padding/copy kernel is needed.

For H=2880, K=3072:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Input / residual / optional normalized output | `[M, 2880]` | BF16 or FP16 |
| Packed output | `[M, 1536]` | `float4_e2m1fn_x2` |
| Row-major block scales | `[M, 192]` | `float8_e4m3fn` |

K must be at least H and divisible by 16. Padded output currently requires
single-CTA NVFP4 and row-major scales; multi-CTA, MXFP4, swizzled, and dual-scale
padded variants are rejected. Omitting K preserves the existing output shape.
The existing partial trace template declines calls specifying K instead of
recording incorrect output shapes.

```python
from flashinfer.cute_dsl import add_rmsnorm_fp4quant

fp4, scales = add_rmsnorm_fp4quant(
    x, residual, norm_weight,
    global_scale=global_scale,
    y_out=normalized,
    output_hidden_size=3072,
)
# Consumers expecting raw storage must reinterpret, not numerically cast:
fp4_bytes, scale_bytes = fp4.view(torch.uint8), scales.view(torch.uint8)
```

The regression gate is **exact equality** of residual, normalized output, live
FP4 bytes, and live scale bytes against the corrected unpadded kernel, plus zero
tails. This is not a claim that all independent RMSNorm implementations use the
same reduction order. The prior numerical regression tests remain required.

## Private one-launch router prototype

`flashinfer.cute_dsl.add_rmsnorm_router_fp4quant` contains the private function
`_experimental_add_rmsnorm_router_fp4quant`. It additionally computes
`normalized @ router_weight.T + router_bias` in the same kernel, returning
`(fp4, scales, router_logits)`. It does not implement expert selection, softmax,
expert dispatch, or expert GEMM. It can omit the global normalized-output store.

The router uses the rounded BF16/FP16 normalized values retained in shared memory,
not dequantized FP4 values. A CTA barrier precedes the router reads, including
threads for inactive rows. FP32 SIMT accumulation is followed by an input-dtype
output conversion. Bias is optional. Only contiguous, non-overlapping buffers
and the single-CTA NVFP4 path are accepted.

This is a correctness/performance prototype, **not an approved replacement for
the existing router GEMM**. Quantized outputs are checked exactly against the
non-router kernel. Router logits are checked against FP64 dot products using an
FP32 accumulation error bound plus the final BF16/FP16 rounding bound. Near-tied
expert choices are not guaranteed to match a different GEMM reduction order.
A stable public API and trace registration are deliberately deferred.

## Validation and reproduction

Run with a CUDA-enabled FlashInfer checkout and its required CuTe DSL:

```bash
pytest -v tests/norm/test_add_rmsnorm_fp4_quant_numerics.py \
  tests/norm/test_add_rmsnorm_fp4_quant_padding.py \
  tests/norm/test_add_rmsnorm_router_fp4_quant.py \
  tests/norm/test_add_rmsnorm_fp4_quant_cute_dsl.py::TestYNormOutput

python benchmarks/bench_add_rmsnorm_router_fp4quant.py \
  --trtllm-padding --profile-dir /tmp/norm-router-profiles
```

The optional `--trtllm-padding` requires the existing custom TensorRT-LLM
`pad_nvfp4_activation` op. Without it the benchmark compares direct padding plus
`torch.addmm` against the one-launch prototype. Profiling asserts one device
kernel for the prototype and exports Chrome traces. The router baseline is
`torch.addmm`, **not** TensorRT-LLM's shape-dependent router backend.

The benchmark uses preallocated buffers, warmed JIT code, CUDA graphs (10 calls
per replay), 30 warmup replays, and 100 measured replays in each of three trials.
Variant order is shuffled with a fixed seed. These are warm-L2 measurements;
they do not predict cold-cache or end-to-end serving performance. Input is zero
and residual is random so repeated in-place replay has stationary finite data;
correctness tests separately use random input and residual and change graph
inputs, scale, router weights, and bias across replays.

Initial GB300 result (2026-09-22, BF16, H=2880, K=3072, 128 experts,
PyTorch `2.14.0a0+4fdf77b940.nv26.08`; p50 microseconds):

| Tokens | Separate padding + router GEMM | Direct padding + router GEMM | One-launch SIMT prototype |
| ---: | ---: | ---: | ---: |
| 1 | 7.725 | 6.626 | 35.952 |
| 7 | 8.741 | 7.002 | 37.163 |
| 32 | 9.398 | 7.536 | 37.629 |
| 228 | 11.200 | 9.149 | 42.746 |
| 7700 | 80.110 | 69.590 | 820.779 |

Direct padding removed exactly one launch. The prototype used one launch at all
five sizes, but was substantially slower. cuBLAS selected a split-K GEMM with a
separate reduction for the four smaller cases, so measured total kernel counts
were 4/3/1 for separate-padding/direct-padding/prototype, and 3/2/1 at M=7700.
These baselines already fuse add, norm, and quantization; they are not the
original fully unfused five-kernel trace.

Recommendation: integrate and validate **direct padded output first**, retaining
the optimized router GEMM. Do not enable the SIMT router prototype in serving.
A competitive one-launch router will need a different scheduling/GEMM strategy
(for example tensor-core tiling with adequate parallelism and weight reuse),
plus routing and model-level accuracy validation. Fewer launches alone are not
a performance win.
