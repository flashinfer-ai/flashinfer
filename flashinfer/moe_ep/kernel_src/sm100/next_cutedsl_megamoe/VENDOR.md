# Blackwell inference sources

`src/sources/` contains standard inference and GenPhase in one provider export.
Both paths use the same Blackwell FC12 epilogue, scheduler and shared helpers.
The shim imports this tree through its full FlashInfer package name, without
registering a global `sources` package or loading Rubin kernels.

## Source

- Provider base: `228a02d`.
- Reconciled source and exporter: `f2d13b3cb8277398ff281fc80ae5f725cf1b2f10`.
- Exporter SHA-256: `0e578237d416970d099750606a2489c0bfe2c0fd39bc86fa7c83bc7343d87814`.
- Entry points: `BlackwellInferenceMegaMoE` and `BlackwellInferenceGenphaseMegaMoE`.
- Exported Python files: 45.

The changes pending upstream bring the Blackwell GenPhase implementation from
[FlashInfer #6565](https://github.com/flashinfer-ai/flashinfer/pull/6565) into
the provider tree. It retains native K32 MMA, Gather4 loading, the 512-column
TMEM plan and reset ordering. The shared FC12 extension gains optional expert
remapping. The shared BF16 reducer gains the route-mask interface already used
by GenPhase while preserving the provider's current accumulation arithmetic.
The exporter registers both entry points and generates lazy imports. The
counted CTA barrier helper is retained for GenPhase.

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

The standard adapter accepts `knobs=None` or a dictionary. It supports early
and late routing weights, BF16 or quantized combine, in-kernel BF16 reduction,
and NVFP4 per-token FC1 activation scales. SiTU requires both bounds.
Parameterized SwiGLU, deferred finalization, online autotuning and IKET capture
are unavailable through these selectors. Legacy scheduler controls without an
equivalent in the standard kernel are rejected before allocation.

The GenPhase autotuner keeps its existing interface. Its cache revision changes
with this export, and the shim preserves its previous default FC2 staging depth.
