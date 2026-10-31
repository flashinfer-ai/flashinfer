# ResInfer MEXT (experimental)

Owner: @ShawRong. [Tracking issue](https://github.com/flashinfer-ai/flashinfer/issues/6886).
MEXT is an experimental, explicit opt-in API on SM100.

## Source and attribution

This is a port of **ResInfer**, not an independent implementation of its
residue quantization or row-pair Fold algorithm. The donor is the public
[ResInfer commit 51b9068](https://github.com/ShawRong/resinfer/commit/51b906836a69fa43915040314df57bc9f23030ad).

Original files at that commit:

- `csrc/common/fold/sm10x/{host,kernel_sm100_fold}.py`: CuTe Fold GEMM.
- `csrc/common/fp4/nvfp4_utils.cuh`: pack16 base/residue quantization arithmetic.
- `csrc/nvfp4/cuda/nvfp4_quant_kernels.cu`: quantization kernel and launch policy.
- `csrc/nvfp4/mext.py`: output allocation and layout contract.

The CUDA helpers retain NVIDIA and Rong Shuo notices; the Fold kernel retains
its BSD-3-Clause identifier. FlashInfer integration changes are TVM-FFI/JIT
bindings, SM100 and tensor validation, a row-pair-only producer, deterministic
zero scale padding, and experimental API entry points. The producer uses
ordinary stream ordering instead of PDL. Other donor Fold specializations are
retained internally to minimize changes to the kernel; only NVFP4 row-pair TMA
Fold is exposed by this API.

## Usage

Public entry points are `flashinfer.resinfer.nvfp4_quantize_residue` and
`flashinfer.resinfer.mm_nvfp4_residue`. Both use
`@flashinfer_experimental_api`; calling them is the explicit opt-in.
The host, JIT and kernel modules under this directory are internal details,
reached through deferred imports. These experimental APIs do not promise
stable signatures yet.

```python
import torch
from flashinfer.resinfer import nvfp4_quantize_residue, mm_nvfp4_residue

x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
activation_global = torch.tensor([64.0, 512.0], device="cuda")
a, a_sf = nvfp4_quantize_residue(x, activation_global)
# w and w_sf: prequantized weight [N, K/2] and swizzled E4M3 scale bytes.
# weight_global: the scalar multiplier used by the weight quantizer.
# alpha = 1 / (activation_global * weight_global)
# y = mm_nvfp4_residue(a, w, a_sf, w_sf, alpha)
```

SM100 only; FP16/BF16 input and output; contiguous aligned tensors. The GEMM
requires K divisible by 64 and N divisible by 8. Global multipliers must be
positive finite values; inputs must be finite. Warm up before CUDA graph
capture. MEXT quantizes both the activation and its reconstruction error, uses
shared weights, and adds paired accumulators in FP32 before output conversion.
It is not a low-rank SVDQuant/LoRA correction.

Run `python examples/resinfer/bench_mext.py --m 32 --n 128 --k 256` for a
small runnable pipeline comparison on SM100. Run
`pytest tests/experimental/test_resinfer_mext.py` for correctness. Graduation
requires correctness and graph replay validation, matched performance records,
API review, and completion of the tracking issue. ResAttn and
[KEXT](../resinfer_kext/README.md) share the quantization helpers.

## Validation

On 2026-10-21, the fork's `tests/experimental/test_resinfer_mext.py` completed
with **18 passed in 13.41 seconds** on one B200 / SM100. Runtime: Torch
2.13.0+cu130, CUDA toolkit 13.0, CUTLASS DSL 4.6.2, TVM-FFI 0.1.11.

The suite covers FP16/BF16, one/two global scales, M = 1/7/65/129,
K = 64/128/256/1024 (four paired shapes), N = 128, residue reconstruction
error, Fold against decoded FP32 matrix multiplication, changed-input
quantizer CUDA Graph replay, and rejection of an invalid scale dtype.
The first JIT attempt exposed a missing `<cstdint>` include, fixed in
the integration before this passing run. Compiler deprecation warnings remain.

The subsequent source review widened the shared scale-offset tile count to
int64 before ceil-division and stride multiplication. Previously, the scale
row stride overflowed int32 at K=268435456; no allocation at that size was run.
The expanded MEXT suite has 30 tests, adding N=8/24/136 tails. All 30 passed
on SM100 with DSL 4.8 during review.

This is bounded kernel correctness evidence. Full-pipeline graph replay,
broader shapes, multi-device behavior and model E2E remain unvalidated.
`examples/resinfer/bench_mext.py --m 32 --n 128 --k 256` completed on the
same SM100/DSL 4.8 runtime as a runnable-example smoke. The script reports
separate timing boundaries; representative performance coverage remains part
of the graduation plan.
