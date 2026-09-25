# Batched FP8 projections

`from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm` prepares
`A[T,H,K] @ B[H,N,K] -> D[T,H,N]` on SM100a (148 SMs) and SM103a (152 SMs). Inputs use FP8 E4M3
values and FP32 positive power-of-two scales: A scales are `[T,H,K/128]`, and
B scales are `[H,N/128,K/128]`. The prepared call returns a plan; `plan.run()`
submits the projection on the current PyTorch stream.

The exported routes use H=8, K=4096 and N=1024. Dynamic FP8 output covers
T=1, 4, 16, 128, 512 and 4096. BF16 output and BF16 with runtime alpha cover
T=4 and 128. FP8 and alpha are separate epilogues.

BF16 plans return a tensor. FP8 plans return `(values, scales)`, where values
are E4M3 `[T,H,N]` and scales are int32 `[T,H*N/128]` containing four per-32
UE8M0 scale bytes per word. Scale storage is column-major, with T padded to4.
Callers can provide `out`, `output_scales` and descriptor workspace storage.
Preparation owns allocations and scale packing; run submits the prepared
projection with descriptor updates and launch overhead, without allocation.
Operand values may change in place between runs. Prepare a new plan if input
scales change. Do not concurrently reuse one plan's output or workspace.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_batched_gemm/generated`.
The runtime and catalog are in this directory. The public API is
`flashinfer/fp8_batched_gemm.py`; see `examples/experimental/fp8_batched_gemm.py`.
Build/install FlashInfer with its supported CUDA toolchain before running:

```bash
python examples/experimental/fp8_batched_gemm.py
pytest tests/experimental/test_fp8_batched_gemm_generated.py -q
```

The test covers all ten exported routes, FP8 output and scale bytes, BF16
values, runtime alpha, changed input values, graph replay and a non-default
stream. Performance qualification measures complete prepared calls with
prepacked inputs and retains all seven model rows plus the three semantic
rows. Per-row exported execution must remain within 3% of its source route.
Performance results accompany the generated bundle; preparation is excluded
from both timed arms.
