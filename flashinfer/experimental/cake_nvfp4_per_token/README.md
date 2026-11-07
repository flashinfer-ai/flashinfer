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
| `nvfp4_quantize(x, global_scale_inv, per_token_activation=True, backend="cake", out_scale=...)` | `quant:...` (one CTA per token row, register-resident row; short launches of long bf16 rows split each row over a cluster of four CTAs) | `x [M, K]` bf16 / fp16, `K % 16 == 0`, 128x4 scale layout, no shuffle, no `expanded_idx_to_permuted_idx`; returns `fp4 [M, K/2]` uint8, `sf [round_up(M, 128), round_up(K/16, 4)]` uint8, `per_token_scale [M]` fp32 (times `out_scale` when given) -- bitwise equal to `backend="cute-dsl"` |
| `mm_fp4(a, b, a_descale, b_descale, alpha, out_dtype, out, backend="cake")` | `gemm:...` (persistent tcgen05 block-scaled GEMM, per-token alpha epilogue) | `alpha [M]` fp32 (per-token path only), `a [M, K/2]` contiguous, `b = b_fp4.T` of a contiguous `[N, K/2]` weight, 128x4 scales (`b_descale = b_sf.T`), `block_size 16`, `N % 8 == 0`, `K % 256 == 0`, bf16 / fp16 output, contiguous `out` |

The GEMM tactic is the rule of the Cake launcher (`cake_backend.default_tactic`):
`M <= 32` runs the swapped orientation (8 / 16 / 32 tokens per tile; cluster split-K
while the tile grid leaves most SMs idle - three K slices instead of two on the 8-token rows
with several token tiles when the whole cluster grid is co-resident per the part's
driver-measured cluster capacity (`cake_backend.CLUSTER_CAPACITY_BY_SM_COUNT`), else on
the 16-token tile for the measured narrow rows of each output dtype (`cake_backend.N16_SPLITK3_M`) - and
split-K 2 on the deep-K rows whose weight tiles fill at most half the SMs; two CTAs per SM or three mainloop stages on the rows
whose weight-tile count exceeds or fills one wave, chosen per SM count, and 64-row weight tiles at one CTA per SM for the
8-token tile on the 152-SM part when the weight tiles outnumber the SMs); larger `M`
runs the tile the bucket scorer picks (1-CTA 128-token tiles, or 2-CTA 256-token tiles
whose width is re-picked on multi-wave grids from the measured per-wave cost of the
128 / 192 / 256-wide tiles -- on the 152-SM SM103 part the cost of the K = 96 mainloop
programs when the re-picked plan is one of them -- with the grouped raster for narrow weights on multi-wave
grids and the cluster-launch-control tile scheduler for the largest rows), with three
single-token-tile overrides: a 2-CTA
256x64 pair for `M <= 128` over at most 40 narrow weight tiles, the 128-wide two-wave tile
on the 148-SM part when more 128-wide weight tiles than SMs exist, and no L2 promotion
when one wave of 128-wide tiles covers the row.  The 1-CTA 64-wide tile, and on SM100 that
128-wide two-wave tile, stage their output over the start of the mainloop ring, which takes
one more stage; the epilogue hands the overlaid stages back to the CTA's next tile.  In the
quantize + GEMM chain (`prepare_nvfp4_per_token_chain`) the 1-CTA and split-K GEMM programs of the
cells of `cake_backend.CHAIN_EP_RULES` (the 8 / 16 / 32-token tiles and the single-token-tile 128-token
rows, per architecture) prefetch the weight operand and its scales of their first tile's first four K
tiles into L2 before the grid dependency wait, so the weights load while the quantizer still runs;
the GEMM alone keeps the programs above.  The quantizer's CTA
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

Validated matrix (`validated_problems` in `tests/test_helpers/cake_nvfp4_per_token.py`):
`(K, N)` in {(7168, 2112),
(7168, 1536), (16384, 7168), (7168, 18432), (18432, 7168), (8192, 8192),
(8192, 28672), (28672, 8192)} x `M` in {1, 8, 17, 32, 128, 130, 257, 512, 2048, 8192}
x bf16 / fp16 output, ragged `M` in {3, 1000, 4097}, fp16 activations and a folded
output scale on the first family, and the quantizer alone for `K` in {7168, 8192,
16384, 18432, 28672}.  Only the kernels those shapes select are registered
(`cake_jit.KERNELS`).  The dispatch depends on the SM count (148 on B200, 152 on GB300); on
GB200 (SM100, 152 SMs) and B300 (SM103, 148 SMs) a shape whose own tactic selects a kernel
the architecture does not register runs the tactic of B200 / GB300 instead.  A GEMM shape
outside the matrix whose tactic selects a kernel that is not registered runs the nearest
registered program: the same orientation, CTA group and closest tile width, with the raster
group, stream-K tail, scheduler or cache hints of that program (on GB200 / B300 the program
B200 / GB300 runs); its plan records the replaced key in `unregistered_key`.  A quantizer
shape whose instance is not registered runs the nearest registered one-CTA-per-row program
of the same activation dtype and folded scale whose CTA holds the row (a compiled-in row
width only for contiguous rows of that width), and records the replaced key the same way.
fp16 activations without such a program (row widths other than 7168, a folded output
scale, row-strided rows) raise `NotImplementedError` naming the missing kernel.  A GEMM
plan whose program's TMA boxes exceed the operands (its host binding would refuse the
shape: the 64-row output box of the half-M 2-CTA pair at 33-63 tokens, the 64-column
output box of the swapped orientation at `N` < 64) runs the nearest registered program
whose boxes fit and records the replaced key in `unfit_key`.  Every GEMM program stores
the output through a box at least 32 columns wide, so `N` in {8, 16, 24} raises
`ValueError` at prepare.

Correctness: quantizer outputs bitwise equal to the CuTe-DSL per-token kernel; GEMM
`|out - ref| <= 1e-2 + 1e-2 |ref| + half an output ulp` against the FP32 dequantized
operands (`tests/gemm/test_cake_mm_fp4.py`, `tests/quantization/test_cake_nvfp4_quantize.py`).

Generated sources live flat under `csrc/cake_nvfp4_per_token/` (architecture-neutral
`*_kernel.cu` / `*_binding.cu` sources whose per-architecture lowering is selected by
`__CUDA_ARCH__` guards, plus the shared `*_common.cuh` headers) and are registered in
`cake_jit.py` (`MODULES`, `KERNELS`) by the Cake generated-program export (see the
hand-fold note below).  The cluster-split quantizers and the SM103 K = 96 GEMM programs
are rendered by the export with their own kernel and binding sources and the
`cake_ffi_checks.h` header.

Hand-fold note: the kernel and binding sources under `csrc/` and the registry rows in
`cake_jit.py` are exporter output hand-folded here pending exporter support. A program
that differs from a sibling only by constant knobs (output dtype, TMA cache-hint
policies, raster group, quantizer CTA shape, folded output scale, store policy) or only
by its kernel symbol and host-shim namespace builds the sibling's translation unit,
selected by its own `-D` flags (`-DCAKE_PROGRAM_KERNEL` / `-DCAKE_PROGRAM_SHIM` and the
`-DCAKE_*` knob values; the `kernel` / `binding` row fields and `compile_flags` in
`cake_jit.py`), and every shared source carries `#ifndef` defaults for the program it
was exported for. The five TMA descriptor encoders every GEMM binding carried sit once
per distinct text in `cake_nvfp4_per_token_host_common.cuh`, which each GEMM binding
includes a second time from inside its namespace with `CAKE_TMA_A` / `_B` / `_SFA` /
`_SFB` / `_OUT` selecting its variants, so the preprocessed binding is the exported
text. The registry keeps the exported per-program data as sparse `_PROGRAMS` rows, the
four named argument plans of `ARG_PLANS` and one `KERNELS` table shared by both
architectures plus the per-architecture additions, and expands them into `MODULES` and
`KERNELS` at import.
