# Vendoring record: next_cutedsl_megamoe

This directory contains one upstream kernel snapshot.
[SKILL.md](SKILL.md) describes updates; [TUNING.md](TUNING.md) covers measurements.

## Upstream

- **Repo**: <https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe>
  (NVIDIA-internal GitLab; see the shared
  [acknowledgements](../../sm100/cutedsl_megamoe/ACKNOWLEDGEMENT.md)).
- **Vendored commit**:
  [`1667b47a3c911ecade464ab524baf192a0bf5962`](https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/commit/1667b47a3c911ecade464ab524baf192a0bf5962),
  committed 2026-09-16, the upstream `main` snapshot selected for this refresh.
- **Last synced**: 2026-09-16, using the upstream exporter from that same commit.
  Previous drop: `92dd334af2eeedb36087834354b58ace08e880c6` (2026-08-15), inherited
  from FlashInfer PR #4601. The older `47881ad2` reference described equivalent
  inference files in that previous drop; it is not the current source pin.
- **Exporter**: [`next/export_src.py`](https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/blob/1667b47a3c911ecade464ab524baf192a0bf5962/next/export_src.py),
  SHA-256 `a1deca5d171a4ba92ee56b5ee82f064d5dc8f44cdf52b9ce49c88d46f10fb523`.
- **Selected kernels**: `RubinInferenceMegaMoE` and `RubinInferenceGenphaseMegaMoE`.
  Their dependency closure is exported to `src/sources/`: 32 implementation
  modules and nine generated package initializers, 41 files total.
- **Verification**: two independent exports at the pinned commit produced
  identical file lists and bytes; the installed `src/sources/` matches that
  output exactly. The exporter validates internal import closure and rejects
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
(`BlockScaledSwapAbMegaMoeKernel`). FlashInfer's existing NVFP4 and MXFP8 E4M3/E5M2
backends use this updated kernel with SwiGLU and BF16 combine/output, supporting
separate and in-kernel reduction. Its shared helpers, schedulers, communication
and workspace code are from the same pinned snapshot.

`RubinInferenceGenphaseMegaMoE` (`BlockScaledSwapAbGenphaseMoeKernel`) is included
in the vendored export for future integration. It is not yet selectable through
the FlashInfer backend or benchmark. Upstream SiTU activation parameters are also
not exposed by the current wrapper.

`+combine_nvfp4` and `+combine_mxfp8` remain future FlashInfer configuration,
workspace/scale handling, correctness and measurement work. Their device paths
already exist in the export. Training kernels and the local fused-routing kernel
are not selected; the latter supplies one dependency through an export marker.
No upstream tester, build metadata or repository scaffolding is vendored.

## Compiler contract and validation

The shim requires native `sm_107a` and `cutlass.utils.rubin_helpers`, available
in public `nvidia-cutlass-dsl==4.8.0.dev0` and compatible newer builds. Install the
`sm107` extra and set `CUTE_DSL_ARCH=sm_107a` before Python imports the DSL.
The shim checks capabilities and the target captured at import.

The older `92dd334` payload passed integration correctness at FlashInfer revision
`9a414e73c8f4246746b281f98db2217a515819cb`, including EP2/4/8. That evidence is for
the older drop. This export passed native single-GPU (50 cases) and EP4
(16 cases per rank) correctness at FlashInfer `5bd5aeef` on the recorded ARM
Rubin stack, followed by all 420 planned benchmark records on September 18,
2026. See the [results](../../../../../docs/design_docs/moe_ep_sm107_results.md)
and [qualification guide](../../../../../docs/design_docs/moe_ep_sm107_qualification.md).
EP2/EP8 on this export remain unmeasured. Tuning-cache entries use revision
`sm107-block-scaled-1667b47a-v3`, so the previous drop's timings are not reused.

## Export transformations

There are **no handwritten changes** to the export. Upstream's exporter
materializes three `COPY_FROM_IMPORT` markers at the pinned commit:

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

## Related trees

`sm100/cutedsl_megamoe` and the SM90 drop are separate snapshots. This Rubin
drop comes from upstream's `next/` generation, with relative imports and the
`api.py` component model; it does not replace the older SM100 packages.

## Consumers

- `backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/`
