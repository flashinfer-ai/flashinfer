# Mixed FP8 x FP4 GEMM

`from flashinfer.fp8_fp4_gemm import prepare_fp8_fp4_gemm` prepares
`A[M,K] @ B[N,K].T -> D[M,N]` with FP8 E4M3 activations, packed FP4 E2M1
weights, FP32 accumulation and BF16 output on SM100a and SM103a. The prepared
call returns a plan; `plan.run()` submits one kernel on the current stream.

A is `[M,K]` E4M3 (or raw uint8). B is `[N,K/2]` packed E2M1 with the even K
element in the low nibble. Scales are prepacked UE8M0 words: four bytes per
int32/uint32 word, packed-K major with the 4-aligned MN extent contiguous
(`[K/(4*gran_k), align(MN,4)]`). B scales are per 32 K elements; A scales are
per 32 (default) or per 128 (`gran_k_a=128`).

Eight generated programs cover every shape (`mixed_gemm.py`):

| schedule | tiles | selected for |
| --- | --- | --- |
| normal BN128/160/224/256 | BM128 x BK128, 7/7/6/5 stages | M > 128 with an even M tile count |
| swap-AB BM16/32/48 | BN128 x BK128, 12/11/10 stages | M <= 128 (and odd M tile counts) |
| BK128 BN224 | BM128 x BK128, 6 stages, per-128 A scales | `gran_k_a=128` |

The route is chosen per problem shape and device SM count with the DeepGEMM
rule (fewest waves, then fullest last wave, then the smaller tile). Constraints:
K is a multiple of 128; M <= 128 needs an even `ceil(N/128)`; M > 128 needs an
even `ceil(M/128)` or an even `ceil(N/128)`; `gran_k_a=128` needs an even
`ceil(M/128)` and N >= 224. Other shapes raise `NotImplementedError`.

Each program is one device source and one binding under
`csrc/experimental/deepgemm_mixed_gemm`, compiled for the device's exact
architecture; tile counts and the SM count are launch arguments and tensor maps
are passed by value. Build/install FlashInfer with its supported CUDA toolchain
before running:

```bash
python examples/experimental/fp8_fp4_gemm.py
pytest tests/experimental/test_mixed_fp8_fp4_gemm_generated.py -q
```

The test covers analytical values, random-input references at the shipped and
held-out shapes (M outside {1, 16, 128, 512, 4096}, N and K outside the model
shapes), changed input values, graph replay and a non-default stream.
