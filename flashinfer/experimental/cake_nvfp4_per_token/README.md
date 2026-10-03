# Experimental per-token NVFP4 quantizer and GEMM (Cake backend)

The backend is experimental and may change without backward compatibility.
Naming it (`backend="cake"`) is the explicit opt-in and emits FlashInfer's
experimental backend warning once; `backend="auto"` never selects it.

It serves FlashInfer's per-token NVFP4 quantize + GEMM path on SM100 (B200) and SM103
(B300 / GB300):

```python
from flashinfer import mm_fp4, nvfp4_quantize

fp4, sf, scale = nvfp4_quantize(x, 1.0 / (448 * 6), per_token_activation=True, backend="cake")
out = mm_fp4(fp4, w_fp4.T, sf, w_sf.T, scale * w_scale, torch.bfloat16, backend="cake")
```

| Entry point | Program | Contract |
| --- | --- | --- |
| `nvfp4_quantize(x, global_scale_inv, per_token_activation=True, backend="cake", out_scale=...)` | `quant:...` (one CTA per token row, register-resident row) | `x [M, K]` bf16 / fp16, `K % 16 == 0`, 128x4 scale layout, no shuffle, no `expanded_idx_to_permuted_idx`; returns `fp4 [M, K/2]` uint8, `sf [round_up(M, 128), round_up(K/16, 4)]` uint8, `per_token_scale [M]` fp32 (times `out_scale` when given) -- bitwise equal to `backend="cute-dsl"` |
| `mm_fp4(a, b, a_descale, b_descale, alpha, out_dtype, out, backend="cake")` | `gemm:...` (persistent tcgen05 block-scaled GEMM, per-token alpha epilogue) | `alpha [M]` fp32 (per-token path only), `a [M, K/2]` contiguous, `b = b_fp4.T` of a contiguous `[N, K/2]` weight, 128x4 scales (`b_descale = b_sf.T`), `block_size 16`, `N % 8 == 0`, `K % 256 == 0`, bf16 / fp16 output, contiguous `out` |

The GEMM tactic is the rule of the Cake launcher (`cake_backend.default_tactic`):
`M <= 32` runs the swapped orientation (8 / 16 / 32 tokens per tile; cluster split-K
while the tile grid leaves most SMs idle - three K slices instead of two on the 8-token rows
with several token tiles when the whole cluster grid is co-resident per the part's
driver-measured cluster capacity (`cake_backend.CLUSTER_CAPACITY_BY_SM_COUNT`) - and
split-K 2 on the deep-K rows whose weight tiles fill at most half the SMs; two CTAs per SM or three mainloop stages on the rows
whose weight-tile count exceeds or fills one wave, chosen per SM count); larger `M`
runs the tile the bucket scorer picks (1-CTA 128-token tiles, or 2-CTA 256-token tiles
whose width is re-picked on multi-wave grids from the measured per-wave cost of the
128 / 192 / 256-wide tiles, with the grouped raster for narrow weights on multi-wave
grids and the cluster-launch-control tile scheduler for the largest rows), with three
single-token-tile overrides: a 2-CTA
256x64 pair for `M <= 128` over at most 40 narrow weight tiles, the 128-wide two-wave tile
on the 148-SM part when more 128-wide weight tiles than SMs exist, and no L2 promotion
when one wave of 128-wide tiles covers the row.  The quantizer's CTA
width and occupancy follow `cake_backend.cta_config`; for row sets of 512 tokens or more its fp4 and
scale stores carry the L2 evict-last policy, so the outputs the dependent GEMM reads next stay
resident instead of being written back into the quantizer's own read stream.  Every rule is a
documented pure-Python port; the generated-program export checks route and bitwise
output parity against the Cake launchers on every validated shape.

Allocation-free repeated launches (CUDA Graph capturable; prepare outside capture):

```python
from flashinfer.experimental.cake_nvfp4_per_token import cake_backend as cb

ws = cb.allocate_nvfp4_per_token_quantize_outputs(M, K, x.device)   # fp4, sf, scale
runner = cb.prepare_nvfp4_per_token_chain(x, gs_inv, w_fp4, w_sf, out, ws, out_scale=w_scale)
runner()      # quantize + GEMM on the current stream; re-reads x on device at every launch
cb.prepare_nvfp4_per_token_quantize(x, gs_inv, ws)()      # quantizer alone
cb.prepare_mm_fp4_per_token(ws.fp4, ws.sf, w_fp4, w_sf, ws.scale, out)()   # GEMM alone
```

Validated matrix (`cake_backend.validated_problems`): `(K, N)` in {(7168, 2112),
(7168, 1536), (16384, 7168), (7168, 18432), (18432, 7168), (8192, 8192),
(8192, 28672), (28672, 8192)} x `M` in {1, 8, 17, 32, 128, 130, 257, 512, 2048, 8192}
x bf16 / fp16 output, ragged `M` in {3, 1000, 4097}, fp16 activations and a folded
output scale on the first family, and the quantizer alone for `K` in {7168, 8192,
16384, 18432, 28672}.  Only the kernels those shapes select are registered
(`cake_jit.KERNELS`); a shape outside the matrix raises `NotImplementedError` naming
the missing kernel.  The dispatch depends on the SM count (148 on B200, 152 on GB300).

Correctness: quantizer outputs bitwise equal to the CuTe-DSL per-token kernel; GEMM
`|out - ref| <= 1e-2 + 1e-2 |ref| + half an output ulp` against the FP32 dequantized
operands (`tests/gemm/test_cake_mm_fp4.py`, `tests/quantization/test_cake_nvfp4_quantize.py`).

Generated sources live under `csrc/cake_nvfp4_per_token/<arch>/` and are registered in
`cake_jit.py` (`MODULES`, `KERNELS`) by the Cake generated-program export; nothing
there is hand-written.
