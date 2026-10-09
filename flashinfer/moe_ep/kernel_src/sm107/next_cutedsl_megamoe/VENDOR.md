# Vendoring record: next_cutedsl_megamoe

This directory contains one upstream kernel snapshot.
[SKILL.md](SKILL.md) describes updates; [TUNING.md](TUNING.md) covers measurements.

## Upstream

- **Source**: the kernel team's `cutedsl_megamoe` repository; see the shared
  [acknowledgements](../../sm100/cutedsl_megamoe/ACKNOWLEDGEMENT.md).
- **Vendored commit**:
  `1667b47a3c911ecade464ab524baf192a0bf5962`,
  committed 2026-09-16, the upstream `main` snapshot selected for this refresh.
- **Last synced**: 2026-09-16, using the upstream exporter from that same commit.
  Previous drop: `92dd334af2eeedb36087834354b58ace08e880c6` (2026-08-15), inherited
  from FlashInfer PR #4601. The older `47881ad2` reference described equivalent
  inference files in that previous drop; it is not the current source pin.
- **Exporter**: `next/export_src.py` at the vendored commit,
  SHA-256 `a1deca5d171a4ba92ee56b5ee82f064d5dc8f44cdf52b9ce49c88d46f10fb523`.
- **Selected kernels**: `RubinInferenceMegaMoE` and `RubinInferenceGenphaseMegaMoE`.
  Their dependency closure is exported to `src/sources/`: 32 implementation
  modules and nine generated package initializers, 41 files total.
- **Verification**: two independent exports at the pinned commit produced
  identical file lists and bytes. The local comment edit is listed below.
  The exporter validates internal import closure and rejects
  unresolved imports and dependencies outside its external-module allowlist.

The export command, from a checkout at the pinned SHA, is:

```bash
python next/export_src.py \
    --kernels RubinInferenceMegaMoE RubinInferenceGenphaseMegaMoE \
    --dst-dir "${FI_EXPORT_STAGE:?}/sources"
```

The destination must be absent or empty and its parent must exist. Preserve the
export manifest/log and file hashes with the integration evidence.

## Policy

- `src/` is a **verbatim copy of the pinned upstream export output**. Do not
  hand-edit, format or inject files into it. Compare it with a regenerated
  export, rather than expecting every generated file to match a raw upstream path.
- All FlashInfer adaptation lives in `shim/`, re-exported through `__init__.py`;
  backends import the package API, never raw `sources` modules.
- Device-kernel and exporter fixes go upstream first, then re-export. Any
  unavoidable local exception must be recorded below until upstream absorbs it.
  See the shared [vendoring rules](../../README.md).

## Scope of this drop

The generic inference entry point is `RubinInferenceMegaMoE`
(`BlockScaledSwapAbMegaMoeKernel`). FlashInfer's NVFP4, MXFP8 E4M3/E5M2, and MXFP4-weight/MXFP8-activation
backends expose SwiGLU and SiTU with BF16 output. BF16 combine supports separate
and in-kernel reduction; `combine_dtype="nvfp4"` and `combine_dtype="mxfp8"`
require separate reduction. They accept canonical prequantized weights; NVFP4
also supports non-unit normalization and per-expert correction tensors.
Shared helpers, schedulers, communication and workspace code come from the
same pinned snapshot.

`RubinInferenceGenphaseMegaMoE` (`BlockScaledSwapAbGenphaseMoeKernel`) is selected
with `kernel_variant="genphase"`. It requires uniform `(4, 1)` clusters,
`fc2_use_bulk=True`, and at most 1024 tokens per rank. Combine remains BF16.
Tile K is 256 for NVFP4 activations and 128 for MXFP8 activations.
SiTU requires both beta parameters, matching the exported kernel.

`combine_dtype="nvfp4"` selects the existing quantized FC2 return path.
It sends packed E2M1 values with a BF16 amax per 16 elements.
`combine_dtype="mxfp8"` sends E4M3 values with an E8M0 scale per 32 elements.
Both formats use the generic inference kernel. Training kernels and the local
fused-routing kernel are not selected; the latter supplies one dependency through an export marker.
No upstream tester, build metadata or repository scaffolding is vendored.

## Compiler contract and validation

The shim requires native `sm_107a` and `cutlass.utils.rubin_helpers`, available
in public `nvidia-cutlass-dsl==4.8.0.dev0` and compatible newer builds. Install the
`sm107` extra and set `CUTE_DSL_ARCH=sm_107a` before Python imports the DSL.
The shim checks capabilities and the target captured at import.

The older `92dd334` payload passed integration correctness at FlashInfer revision
`9a414e73c8f4246746b281f98db2217a515819cb`, including EP2/4/8. That evidence is for
the older drop. At FlashInfer `0f710df8`, this export passed 187 host/CUDA checks,
72 single-GPU cases, and 34 distributed cases per rank at EP2, EP4, and EP8,
with no skips. See the
[qualification guide](../../../../../docs/design_docs/moe_ep_sm107_qualification.md)
for coverage and the separate reference-decoder regression.
Tuning-cache entries use revision `sm107-block-scaled-1667b47a-runtime-options-v1`.
The cache identity includes the kernel variant and combine format, along with
activation, beta parameters, and clamp settings.

## Export transformations

Upstream's exporter materializes three `COPY_FROM_IMPORT` markers at the pinned commit:

| Exported file under `src/sources/kernel_src/` | Upstream source under `next/sources/kernel_src/` |
| --- | --- |
| `rubin/inference/mega/topk_reduce.py` | `blackwell/inference/mega/topk_reduce.py` |
| `rubin/custom_mix_cga_helpers.py` | `blackwell/custom_mix_cga_helpers.py` |
| `rubin/inference/mega/tma_gather.py` | `rubin/inference/local_mega/tma_gather.py` |

It also generates nine `__init__.py` files and rewrites one import statement in
`block_scaled_swap_ab_fc12_epilogue.py` from scheduler package re-exports to the
implementation modules. The generated root exposes the two selected aliases;
intermediate package initializers are empty. These are upstream exporter
transformations, not manual FlashInfer patches.

## Local differences

The comment above `pusher_cta_count` in
`src/sources/kernel_src/rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py`
describes the residency requirement without device-specific counts. Kernel
logic and configuration values are unchanged. All other exported files match
the pinned upstream export.

## Related trees

`sm100/cutedsl_megamoe` and the SM90 drop are separate snapshots. This Rubin
drop comes from upstream's `next/` generation, with relative imports and the
`api.py` component model; it does not replace the older SM100 packages.

## Consumers

- `backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp4_bf16_cutedsl/`
