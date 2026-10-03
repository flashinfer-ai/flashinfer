# Mixed FP8 x FP4 GEMM

`from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm` prepares
`A[M,K] @ B[N,K].T -> D[M,N]` with FP8 E4M3 activations, packed FP4 E2M1
weights, FP32 accumulation and BF16 output on SM100a (148 SMs) and SM103a
(152 SMs). The prepared call returns a plan; `plan.run()` submits the GEMM on
the current PyTorch stream.

A is `[storage_M,K]` E4M3 (or raw uint8). B is `[N,K/2]` packed E2M1 with the
even K element in the low nibble. Scales are prepacked UE8M0 words: four bytes
per int32/uint32 word, packed-K major with the aligned MN extent contiguous
(`[words, aligned_MN]`). Model routes use per-32 A and B scales. The explicit
variants require `gran_k_a=128`: `bk128_s6` packs four distinct per-128 A bytes
per word; `bk256_s4` repeats each per-128 A byte over its four 32-K sub-blocks.
The catalog fixes the physical A/output rows and the packed scale geometry per
route: source-selected routes bind the logical M, the generic routes pad M to a
multiple of 256. `plan.storage` is the physical output; `plan.output` is the
`[m,N]` view.

The exported routes cover M=1, 16, 128, 512 and 4096 for (N,K)=(4608,5120)
and (5120,2304) with per-32 scales, the 256x256x256 smoke shape, the registered
4096x7168x4096 shape with `block_n=224, gran_k_a=128`, the `bk128_s6` variant
at 256x224x128 and 4096x7168x4096, and the `bk256_s4` variant at 256x128x256.
Any other shape, SM count or option set raises `NotImplementedError`.

Preparation owns allocations and descriptor encoding: routes with by-value
descriptors pass them at launch; routes with a descriptor workspace encode it
once during preparation when the plan owns the workspace, while a caller-provided
`descriptor_workspace` is refreshed before every launch. `run()` submits without
allocation. Changing tensor addresses or layouts requires a new plan.
Operand values may change in place between runs. Do not concurrently reuse one
plan's output or workspace.

Generated CUDA and bindings live in `csrc/experimental/deepgemm_mixed_gemm/generated/<arch>`.
The runtime and catalog are in this directory. The public API is
`flashinfer/fp8_fp4_gemm.py`; see `examples/experimental/fp8_fp4_gemm.py`.
Build/install FlashInfer with its supported CUDA toolchain before running:

```bash
python examples/experimental/fp8_fp4_gemm.py
pytest tests/experimental/test_mixed_fp8_fp4_gemm_generated.py -q
```

The test covers all fifteen exported routes with analytical values, changed
input values, graph replay and a non-default stream. Performance qualification
measures complete prepared calls with prepacked inputs and retains all ten model
rows plus the five semantic rows. Per-row exported execution must remain within
3% of its source route. Performance results accompany the generated bundle;
preparation is excluded from both timed arms.
