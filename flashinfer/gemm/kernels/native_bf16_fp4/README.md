# Native-layout W4A16 on SM12x

This CuTe DSL backend computes BF16 x NVFP4 directly from
canonical packed weights and 128x4-swizzled E4M3 block scales. It performs
no preparation of alternate weight or scale buffers, or full-matrix dequantization.

Explicitly select `backend="cute-dsl-native"` in `flashinfer.mm_bf16_fp4`.
No environment variable is required. Automatic backend selection does not
include this implementation. Existing backends retain their preparation
requirements. Calling `prepare_bf16_fp4_weights` with this backend is optional
and returns the three input objects unchanged.

```python
import torch
import flashinfer

a = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
quant_scale = 2688.0 / w.float().abs().max()
b, sf = flashinfer.nvfp4_quantize(
    w,
    quant_scale,
    sfLayout=flashinfer.SfLayout.layout_128x4,
    do_shuffle=False,
    backend="cute-dsl",
)
alpha = quant_scale.reciprocal().reshape(1)
y = flashinfer.mm_bf16_fp4(a, b, sf, alpha, backend="cute-dsl-native")
```

Support: SM120/121, contiguous BF16 activations with a positive number of
rows, contiguous uint8 weights `(N,K/2)`, positive N, K divisible by 16,
and N*K, M*K and M*N below 2**31. The scale buffer contains the padded 128x4 layout;
weight rows and output columns may have tails. Output is BF16 or FP16.
Alpha is an optional live GPU float32 scalar with shape `(1,)`.

Small-M kernels use asynchronous shared-memory staging, scalar-load MMA or
SIMD depending on shape and alignment. For BF16 output with alpha and M <= 4,
a 64-column tactic stages only the live activation rows plus one zero row.
Larger M uses tiled MMA with weight fragments reused across activation rows,
or scalar-load MMA for unaligned inputs and K tails. Accumulation is FP32.
CUDA 13.2 and newer use direct packed BF16 conversions; older compilers decode
through FP16 and FP32 without narrowing BF16 activations.

Autotuning selects staging depth, warp count, tile shape and split-K count.
Split-K allocates `splits * M * N` FP32 elements for deterministic reduction.
Warm shapes and tactics before CUDA graph capture. A caller-owned output
avoids output allocation; split-K scratch remains separate.

Dynamic W4A4/W4A16 selection is implemented separately in
[vllm-project/vllm#54614](https://github.com/vllm-project/vllm/pull/54614).

Run correctness tests with:

```bash
.venv/bin/python -m pytest tests/gemm/test_mm_bf16_fp4.py -k cute-dsl-native -v
.venv/bin/python -m pytest tests/gemm/test_native_bf16_fp4.py -v
```

Use the existing benchmark harness for comparisons. Preparation and tuning
happen before timing; the measured operation includes split-K reduction.

```bash
.venv/bin/python benchmarks/flashinfer_benchmark.py \
  --routine mm_bf16_fp4 --backends cute-dsl cute-dsl-native \
  --m 4 --n 5120 --k 17408 --input_dtype bfloat16 --out_dtype bfloat16 \
  --refcheck --autotune --enable_pdl --num_iters 30 --dry_run_iters 10
```
