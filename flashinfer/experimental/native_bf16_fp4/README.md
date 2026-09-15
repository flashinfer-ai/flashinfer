# Native-layout W4A16 on SM12x

This experimental CuTe DSL backend computes BF16 x NVFP4 directly from
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

Initial support: SM120/121, contiguous BF16 activations with 1 through 16
rows, contiguous uint8 weights `(N,K/2)`, positive N, K divisible by 16,
and both N*K and M*K below 2**31. The scale buffer contains the padded 128x4 layout;
weight rows and output columns may have tails. Output is BF16 or FP16.
Alpha is an optional live GPU float32 scalar with shape `(1,)`.

CuTe DSL kernels consume the same buffers. For 16-byte-aligned inputs and K
divisible by 64, asynchronous copies stage canonical weights, activations and
scales in shared memory. Two or three buffers overlap loading with computation.
Matrix loads feed BF16 tensor cores, while FP4 decoding and scaling happen in
registers. Scalar-load MMA and SIMD kernels handle other alignments and K tails.
All accumulate in FP32. CUDA 13.2 or newer compilers use direct packed conversions;
older compilers decode through FP16 and FP32. Activations retain BF16 range.

For BF16 output with alpha and M at most four, tuning also considers a
64-column kernel with 128- or 256-element K tiles. It stages only the live
activation rows plus one zero row and reads the same canonical weight and
scale buffers. FP16 output and calls without alpha retain the existing tactics.

Tactics vary staging depth, warp count, row reuse and split-K count. Split-K uses a temporary
FP32 buffer of `splits * M * N` elements and a deterministic reduction kernel.
It never materializes the full dequantized weight matrix. Staged tactics use up
to eight splits. The scalar-load MMA fallback also considers up to 64 for N
below 8192 and K at least 4096. Larger split counts trade more scratch space
for more parallel work. Staging uses only per-CTA shared memory.
Warm each shape/tactic outside CUDA graph capture. A
caller-owned output avoids output allocation; split-K scratch remains separate.

This initial backend does not add W4A4 kernels or select activation precision.
Keep selection explicit: the useful crossover depends on hardware and shape.
The native implementation passed 29 focused GPU tests on SM120 and SM121.
In a combined vLLM integration with the CuTe W4A4 serving implementation, a
300 W RTX PRO 6000 Max-Q benefited from native W4A16 for single-token down
projections. The same integration on DGX Spark favored W4A4 throughout.
The vLLM integration is tracked in
[vllm-project/vllm#54614](https://github.com/vllm-project/vllm/pull/54614).
Those serving measurements use a separate W4A4 source overlay and are not
performance results for this standalone FlashInfer patch alone.

The paired 64-question model screen scored 61/64 for both stock and dynamic
on Max-Q, but does not establish general quality parity. Compilation, tuning
and CUDA graph warmup remain necessary. The combined integration's first
launch also reported a larger temporary memory peak than its warm repeat;
sharing weights does not imply identical peak startup memory.
Native checkpoint scale layouts other than 128x4 are not supported.
Trace export is not yet available for this layout.

Run correctness tests with:

```bash
.venv/bin/python -m pytest tests/experimental/test_native_bf16_fp4.py -v
```

Use the existing benchmark harness for comparisons. Preparation and tuning
happen before timing; the measured operation includes split-K reduction.

```bash
.venv/bin/python benchmarks/flashinfer_benchmark.py \
  --routine mm_bf16_fp4 --backends cute-dsl cute-dsl-native \
  --m 4 --n 5120 --k 17408 --input_dtype bfloat16 --out_dtype bfloat16 \
  --refcheck --autotune --enable_pdl --num_iters 30 --dry_run_iters 10
```
