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

## NVFP4 batch-invariant integration patch

Pending upstream, the two kernel entry points pass an optional
`batch_invariant` problem field to the shared FC12 epilogue and BF16 reducer.
The opt-in epilogue interprets FlashInfer's ordered NVFP4 activation,
block16 requantization and FC2 conversion chains. Standard inference also
pins instruction N and the matching two-CTA load split to the static tile;
GenPhase already uses fixed N128 instructions. The opt-in reducer starts at
positive zero, retains every top-k slot in order and masks both operands of
invalid routes. These branches leave ordinary execution unchanged and must
be retained when regenerating the provider export until incorporated upstream.
The shim admits only the certified NVFP4 geometry and arithmetic modes.

## MXFP4 batch-invariant integration patch

The mixed MXFP4-weight/MXFP8-E4M3-activation path additionally supports the
opt-in batch-invariant problem field. The shim dispatches to its separate
arithmetic policy and receipt; NVFP4 retains its existing policy and receipt.
The mixed path uses native K32 MMA with a canonical `(256, 128, 128)` tile and
cluster `(2, 1)` for standard inference or `(4, 1)` for GenPhase. It has no
per-expert alpha or normalization operands. Pure MXFP4 remains outside this
Blackwell adapter's supported set.

The shared FC1 epilogue's additional integration patch fixes each block32
amax as two ordered block16 folds, merged with the logical low half first
on both participating warps. Explicit UE8M0 RP conversion and round-trip,
the existing capped/masked approximate reciprocal recipe, and RN data scaling
interpret the shared MXFP4 specification. Existing fixed-N handling, SwiGLU,
BF16 conversion and ordered masked reduction are reused. Ordinary execution
and the NVFP4 branches remain unchanged. Retain these additional
FlashInfer-owned exceptions when regenerating the export until upstreamed.

Standard inference can regroup the unchanged native instruction sequence
into a larger K stage: K256 for NVFP4, mixed MXFP4 and MXFP8. Admission requires
the H and post-activation I padding extents to match the canonical stage
exactly. M256, N128 and the two-CTA instruction remain fixed; GenPhase keeps
its Gather4 stage. This changes host admission and lowering partitions, not
the native arithmetic atom or canonical receipt.

## Activation variant integration patch

The standard FC2 epilogue also forwards its BI flag to the shared canonical
BF16-to-quantized combine encoder. The shared reducer uses one zero-seeded,
ordered masked slot fold for quantized BI wires, preserving the ordinary
quantizer and reducer branches. Reapply until upstream absorbs this patch.

The FC12 epilogue accepts an opt-in BI ordered activation recipe from
`ProblemDesc.batch_invariant_activation`. A missing recipe keeps the existing
default helper. The standard provider's BI per-token dequantization preserves
the scratch token mapping and fixes token-scale times expert-alpha products
to explicit RN multiply. Ordinary activation and scale products keep their
original paths. Reapply until upstream absorbs these integration hooks.

## MXFP8 batch-invariant integration patch

The E4M3/E5M2 W8A8 paths additionally use the opt-in batch-invariant problem
field with a separate typed quantizer and arithmetic receipt. They retain
native K32 MMA, fixed `(256, 128, 128)` tiles, cluster `(2, 1)` for standard
inference or `(4, 1)` for GenPhase, and no per-expert alpha or normalization
operands. The shared epilogue dispatches block32 requantization to the MXFP8
specification with the data format's reciprocal limit. Its ordered block16
amax folds, canonical half merge, explicit UE8M0 round-trip, capped/masked
reciprocal and RN scaling retain the next-generation arithmetic chain.
Existing fixed-N handling, SwiGLU, BF16 conversion and ordered masked
finalization are reused. NVFP4, MXFP4 and ordinary branches keep their existing
arithmetic. Retain this additional FlashInfer-owned exception until upstreamed.

## Batch-invariant compiled identity patch

Both standard and GenPhase kernel `name()` methods append the BI mode and,
when present, a hash of the ordered activation signature's effective FP32
coefficient bits. Ordinary names remain byte-for-byte unchanged. This also
separates the standard provider's name-derived AOT symbols. Reapply this host
metadata patch together with the BI arithmetic hooks after regenerating the
export until upstream absorbs it.

## Collective workspace preparation

The FlashInfer shim agrees on local allocation and preparation failures across
the selected EP process group before entering the symmetric heap. Winner
installation prepares replacement local storage before freeing existing
symmetric storage, so a failed local preparation preserves the current owner.
Trial staging and compilation finish collectively before any trial launch.
These agreements occur during preparation, outside serving and graph replay.
