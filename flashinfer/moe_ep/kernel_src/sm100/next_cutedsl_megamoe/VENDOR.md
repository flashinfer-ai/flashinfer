# Blackwell inference sources

`src/sources/` contains standard inference and GenPhase in one provider export.
Both paths use the same Blackwell FC12 epilogue, scheduler and shared helpers.
The shim imports this tree through its full FlashInfer package name, without
registering a global `sources` package or loading Rubin kernels.

## Source

- Provider base: `228a02d`.
- Source revision: `24c773e2b13f68a7c490a7950722cdef04f1c636` plus
  [`patches/genphase_dtypes.patch`](patches/genphase_dtypes.patch).
- Dtype patch SHA-256: `cb17635d21059dbd90402e2c850df909c310170b79d7ef2511077cb28e13ad27`.
- Exporter revision: `f2d13b3cb8277398ff281fc80ae5f725cf1b2f10`.
- Exporter SHA-256: `0e578237d416970d099750606a2489c0bfe2c0fd39bc86fa7c83bc7343d87814`.
- Entry points: `BlackwellInferenceMegaMoE` and `BlackwellInferenceGenphaseMegaMoE`.
- Exported Python files: 46.

The changes pending upstream bring the Blackwell GenPhase implementation from
[FlashInfer #6565](https://github.com/flashinfer-ai/flashinfer/pull/6565) into
the provider tree. It retains native K32 MMA, Gather4 loading, the 512-column
TMEM plan and reset ordering. The shared FC12 extension gains optional expert
remapping. The shared BF16 reducer gains the route-mask interface already used
by GenPhase while preserving the provider's current accumulation arithmetic.
The exporter registers both entry points and generates lazy imports. The
counted CTA barrier helper is retained for GenPhase.

The shared reducer masks invalid routes in BF16 and quantized combine, in
both static and persistent execution. Deferred output clears invalid slots
for external consumers; quantized combine relies on the reducer mask.
The standard router checks the packed rank and top-k bounds, and NVFP4 calls
require all three global scales. FC1 scale views match AOT padding. Grid-stride
scheduling requires one compute cluster per launched cluster.

Export both entry points together from the reconciled revision. The generated
Python files must match `src/sources/` byte for byte. Source changes belong in
the provider before regeneration; do not edit or format this copy. Update the
source and exporter identifiers when refreshing the export.

## FlashInfer integration

`shim/` owns configuration, weight conversion, workspace allocation and launch.
The existing GenPhase API remains available. Standard inference is selected
with the existing NVFP4 or MXFP8 config and one of these backend names:

- `sm100_nvfp4_nvfp4_bf16_cutedsl_next`
- `sm100_mxfp8_mxfp8_bf16_cutedsl_next`

For MXFP8, keep `kernel_variant="inference"`; both E4M3 and E5M2 are supported.
Existing backend names and defaults continue to use the legacy implementation.

Fused staging still uses the single `DataPreprocess` implementation exposed
by `sm100/cutedsl_megamoe`, including Rubin's existing staging consumer. This
export contains no second `inputs_process.py`. BF16/BF16 and BF16/MXFP8 remain
in the legacy package.

The standard adapter accepts `knobs=None`, a dictionary, or `knobs="auto"`.
Automatic selection uses `autotune_v2()` for both standard and GenPhase kernels;
the offline CLI writes managed results and the MegaMoE cache file by default.
Both dtypes support
parameterized SwiGLU, bounded or unbounded SiTU, early and late routing weights,
and in-kernel BF16 reduction. NVFP4 also exposes per-token FC1 activation scales
and quantized combine. MXFP8 supports 64-aligned hidden rows with padded scales.

`FinalizeMode.DEFAULT` uses the static reducer. `PERSISTENT` borrows the caller's
CUDA int32 `(1,)` live-token count; update its contents before each forward.
`NONE` returns borrowed BF16 rows, remaining routing weights and a route map
through `forward_unfinalized()`. Consume them on the launch stream before
workspace reuse. Persistent and deferred finalization cannot be combined.
`CAKE` retains the existing NVFP4 device and shape restrictions. `AUTO` selects
CAKE when supported, then persistent reduction when a live count is supplied,
otherwise static reduction. In-kernel reduction needs no separate finalizer.

IKET capture is unavailable through the standard selectors.
Legacy scheduler controls without a standard-kernel equivalent are
rejected before allocation. GenPhase retains its existing SwiGLU and separate
BF16 finalization restrictions.

The GenPhase autotuner keeps its existing interface. Its cache revision changes
with this export, and the shim preserves its previous default FC2 staging depth.

## GenPhase dtype extension

The dtype patch lifts the E4M3-only gate for E5M2, NVFP4 and mixed MXFP4/MXFP8.
It keeps the complete reconciled dependency closure above. Apply the patch to
the provider before exporting both Blackwell entry points. The 46 Python files
were checked byte for byte against two exports; the standard kernel payload
and lazy package initializers are unchanged.

NVFP4 uses native K64 instructions and a K256 Gather4 tile, so a packed
activation row still occupies one 128-byte swizzle atom. MXFP8 E4M3/E5M2 and
mixed MXFP4/MXFP8 retain native K32 instructions and K128 Gather4 tiles.
NVFP4 admits N128/CGA4; mixed MXFP4/MXFP8 admits N128/CGA4 and N64/CGA4;
MXFP8 E4M3/E5M2 also admit N64/CGA2. All use separate BF16 finalization.
The shim's dtype-aware tuning grid uses cache revision
`sm100-genphase-block-scaled-auto-v6` and preserves FC2 staging depth.
