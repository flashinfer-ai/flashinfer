# W4A16 / W8A16 Kernel Integration Handoff

## Goal

Replace the W8A16 PyTorch reference with an optimized kernel and add an
optimized W4A16 backend. Preserve weight compatibility with the corresponding
fully quantized GEMMs so a framework can choose A16 or quantize the activation
and run A8/A4 without maintaining a second weight allocation.

## W8A16: `mm_bf16_fp8`

The public API and integration scaffolding are complete. Do not add a
`backend` argument.

- Public API: `flashinfer/gemm/gemm_bf16_fp8.py`
- Kernel insertion point:
  `flashinfer/gemm/kernels/dense_bf16_fp8_gemm_sm12x.py`
- Replace `mm_bf16_fp8_sm12x`'s PyTorch dequantize-plus-`torch.mm` body with
  the optimized implementation; keep its `(A, B, B_scale, out)` contract.
- `A`: contiguous BF16 `[M, K]`.
- `B`: FP8 E4M3 `[K, N]`, column-major, backed by contiguous `[N, K]`
  storage.
- `B_scale`: contiguous scalar FP32 weight-dequantization scale.
- `out`: preallocated contiguous BF16 or FP16 `[M, N]`.

The weight contract is intentional. The same `B` allocation must work in the
fully quantized path by adding a size-one batch view and passing it to
`bmm_fp8`; do not preprocess, reorder, or copy the weight. Preserve `out`
support, current-stream execution, CUDA Graph safety, and the public API's
support checks.

Correctness and weight-reuse coverage is already in
`tests/gemm/test_mm_bf16_fp8.py`, including twelve comparisons against cuDNN
`bmm_fp8`.

## W4A16: new `mm_bf16_fp4` backend

Do not add another public GEMM API. Add a backend to both existing dispatchers
in `flashinfer/gemm/gemm_bf16_fp4.py`:

1. Add the backend name and its requirement check to `mm_bf16_fp4`.
2. Add the same backend to `prepare_bf16_fp4_weights`.
3. Add internal prepare and compute functions for the optimized kernel.
4. Register the backend in benchmark argument choices and add its trace
   template/dispatch entry.

The input to `prepare_bf16_fp4_weights` is the canonical output of
`nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False)`:

- packed FP4 weight: contiguous `uint8 [N, K/2]`;
- weight scales: canonical 128x4-swizzled E4M3 scale-factor storage;
- optional scalar FP32 `alpha` carrying the global weight scale.

The new prepare path should ideally be an identity operation, or return only
zero-copy views. It must not create a persistent unswizzled, shuffled, or
backend-specific weight copy. The optimized kernel should consume the
canonical packed weight and swizzled scales directly.

This permits one weight allocation to serve both paths:

```python
# Quantize/store the weight once.
w_fp4, w_sf = flashinfer.nvfp4_quantize(
    w_bf16,
    weight_global_scale,
    sfLayout=flashinfer.SfLayout.layout_128x4,
    do_shuffle=False,
)

# W4A16: no activation quantization.
w_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(
    w_fp4, w_sf, weight_alpha, backend=NEW_BACKEND
)
y_a16 = flashinfer.mm_bf16_fp4(
    x_bf16, w_p, sf_p, alpha_p, backend=NEW_BACKEND
)

# W4A4: quantize only the activation and reuse weight storage via views.
x_fp4, x_sf = flashinfer.nvfp4_quantize(
    x_bf16,
    activation_global_scale,
    sfLayout=flashinfer.SfLayout.layout_128x4,
    do_shuffle=False,
)
y_a4 = flashinfer.mm_fp4(
    x_fp4,
    w_fp4.T,
    x_sf,
    w_sf.T,
    combined_global_scale,
    backend="auto",
)
```

Add an integration test that asserts the W4A16 and W4A4 calls share the same
weight and scale data pointers, validates each result against its appropriate
reference, and compares the outputs with a quantization-appropriate tolerance.

## Benchmarking and completion criteria

Benchmark routines already exist:

- `--routine mm_bf16_fp8` (implicitly reported as backend `auto`)
- `--routine mm_bf16_fp4 --backends <new-backend>`

Use `--refcheck`; the harness handles CUDA Graph timing and cold-L2 rotation.
Before handoff, run the focused GEMM tests, trace-template consistency tests,
pre-commit, and representative low-M benchmarks against the fully quantized
cuDNN paths. A kernel is complete only when it is numerically correct, reuses
the canonical weight allocation, is CUDA Graph safe, and replaces all PyTorch
dequantization/matmul work on the hot path.
