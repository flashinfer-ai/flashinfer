# Experimental Frost WOA

Owner: @YangXu1990uiuc. Tracking: [#5115](https://github.com/flashinfer-ai/flashinfer/issues/5115).

WOA is the first grouped attention output projection in DeepSeek V4.1.
The explicit `deepseek_v41_woa_plan(..., backend="frost")` and
`deepseek_v41_woa(...)` APIs live in `flashinfer.deepseek_v41`.
They are experimental, JIT-only, and provide no compatibility guarantees.

The current SM100 implementation accepts BF16 input `[1,8,4096]`, contiguous
E4M3 weight `[8192,4096]`, and E8M0 block scales `[256,128]`, producing BF16
output `[1,8,1024]`. Each scale covers a 32-by-32 block of the flattened weight
matrix. This is a two-dimensional checkpoint layout. It differs from MXFP8
GEMM layouts with a separate scale for every row's group of 32 elements.
Scale bytes 1 through 254 are supported; scales can use `uint8` or
`torch.float8_e8m0fnu` storage.

Weights are decoded and rounded to BF16 before multiplication by BF16 input.
Accumulation uses FP32. There is no input quantization or autograd support.
Inputs and decoded weights must be finite. Prepare and warm up outside CUDA
Graph capture, keep the original weight and scale storage immutable, and
reuse the plan across inputs. An optional `out` buffer must not overlap input,
weight, or scale storage. The plan retains weights but owns no output buffer.
Inverse RoPE and the subsequent WOB projection are separate operations.

Run `python examples/deepseek_v41_woa.py` on SM100 for a complete usage example
and a comparison with the native TGV BF16 BMM API. It uses five synthetic
weight sets with the model's single-token geometry, verifies changed-input
Graph replay, and reports component GPU time. Run correctness coverage with
`pytest tests/experimental/test_deepseek_v41_woa.py`.

Graduation follows #5115: serving integration, target-hardware performance and
correctness coverage, then API review. There is no full-model performance claim.
