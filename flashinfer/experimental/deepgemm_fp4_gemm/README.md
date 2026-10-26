# Native FP4 GEMM

`from flashinfer.fp4_gemm import prepare_fp4_gemm` prepares
`alpha * A[M,K] @ B[N,K].T -> D[M,N]` with packed FP4 E2M1 activations and
weights, FP32 accumulation and BF16 output on SM100a and SM103a. The prepared
call returns a plan; `plan.run()` submits one kernel on the current stream.

A is `[rows,K/2]` and B `[N,K/2]` packed E2M1 bytes with the even K element in
the low nibble. Scales are prepacked UE8M0 words: four bytes per int32/uint32
word, one byte per 32 K elements, packed-K major with the 4-aligned MN extent
contiguous (`[K/128, align(MN,4)]`). `m` is the logical output M (default: the
rows of A). `alpha` multiplies the FP32 accumulators before the BF16 store.

Eight generated programs cover every shape (`fp4_gemm.py`):

| schedule | tiles | selected for |
| --- | --- | --- |
| swap-AB BM16/32/48 | BN128 x BK256, 11/10/10 stages | M <= 128 (and odd M tile counts) |
| normal BN16/128/160 | BM128 x BK256, 7 stages | M > 128 with an even M tile count |
| normal BN224/256 | BM128 x BK256, 6 stages, grouped raster | large M and N |

The route is chosen per problem shape and device SM count with the DeepGEMM
rule (fewest waves, then fullest last wave, then the smaller tile). Constraints:
K is a multiple of 256; M <= 128 needs N >= 128 and an even `ceil(N/128)`;
M > 128 needs an even `ceil(M/128)` or an even `ceil(N/128)`. Other shapes raise
`NotImplementedError`. `num_stages` may name the selected schedule's stage
count; `block_n`, `epilogue_store_n` and `descriptor_workspace` accept their
defaults (tensor maps are passed by value).

Each program is one device source and one binding under
`csrc/experimental/deepgemm_fp4_gemm`, compiled for the device's exact
architecture; tile counts and the SM count are launch arguments. Build/install
FlashInfer with its supported CUDA toolchain before running:

```bash
python examples/experimental/fp4_gemm.py
pytest tests/experimental/test_native_fp4_gemm_generated.py -q
```

The test covers analytical values with runtime alpha, random-input references
at the shipped and held-out shapes (M outside {16, 128, 512, 4096}, N and K
outside the model shapes), changed input values, graph replay and a
non-default stream.
