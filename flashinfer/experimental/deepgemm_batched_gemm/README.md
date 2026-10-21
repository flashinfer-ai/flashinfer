# Batched FP8 projections

`from flashinfer.fp8_batched_gemm import prepare_fp8_batched_gemm` prepares
`A[T,H,K] @ B[H,N,K] -> D[T,H,N]` on SM100a and SM103a. Inputs use FP8 E4M3
values and FP32 positive power-of-two scales: A scales are `[T,H,K/128]`, and
B scales are `[H,N/128,K/128]`. The prepared call returns a plan; `plan.run()`
submits the projection on the current PyTorch stream.

The exported programs cover H=8, K=4096 and N=1024. One source per schedule
compiles for both architectures with that architecture's exact flags. The
schedule is selected by `batched_gemm.route_config` from the token count, the
device SM count and the epilogue, exactly as the producing dispatcher selects
it; M, the M tile count and the grid are launch arguments, so one program
serves every token count that selects it:

| schedule | epilogue | token counts (148 and 152 SMs) |
|---|---|---|
| swap_ab BM16/BN128, 12 stages | dynamic FP8 | 1-32 |
| swap_ab BM64/BN128, 10 stages | dynamic FP8 | 97-128 |
| n256 BM128/BN256, 5 stages | dynamic FP8 | 481-512, 961-1024 and further windows that fill an even number of 128-row tiles with two-CTA clusters (`route_config` is the reference) |
| general BM128/BN128, 5 stages | BF16 with runtime alpha | every T |
| bf16_t4 BM16/BN128, 12 stages | BF16 | 4 |
| bf16_t128 BM64/BN128, 10 stages | BF16 | 128 |

A token count whose selected schedule is not exported (the general 128x128
schedule with dynamic FP8 or plain BF16 output) raises `NotImplementedError`
at preparation, naming the schedule. FP8 and alpha are separate epilogues.

BF16 plans return a tensor. FP8 plans return `(values, scales)`, where values
are E4M3 `[T,H,N]` and scales are int32 `[T,H*N/128]` containing four per-32
UE8M0 scale bytes per word. Scale storage is column-major, with T padded to4.
Callers can provide `out` and `output_scales`; `descriptor_workspace` is
accepted and ignored (tensor maps travel by value). Preparation owns allocations
and scale packing; run submits the prepared projection with launch overhead
and without allocation. Tensor maps travel by value with every launch.
Operand values may change in place between runs. Prepare a new plan if input
scales change. Do not concurrently reuse one plan's output.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_batched_gemm/generated`
(one `*_kernel.cu` / `*_binding.cu` pair per program; `PROGRAMS` and `ROUTES`
in `batched_gemm.py` are the registry). The public API is
`flashinfer/fp8_batched_gemm.py`; see `examples/experimental/fp8_batched_gemm.py`.
Build/install FlashInfer with its supported CUDA toolchain before running:

```bash
python examples/experimental/fp8_batched_gemm.py
pytest tests/experimental/test_fp8_batched_gemm_generated.py -q
```

The test covers the ten original routes and non-listed token counts of every
exported schedule: FP8 output and scale bytes, BF16 values, runtime alpha,
changed input values, graph replay, a non-default stream, and the
`NotImplementedError` of an unexported schedule.
