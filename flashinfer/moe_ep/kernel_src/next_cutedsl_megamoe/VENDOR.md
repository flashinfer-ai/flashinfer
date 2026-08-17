# next_cutedsl_megamoe — vendored kernel drop

Rubin (SM107) MegaMoE kernels from the kernel team's `cutedsl_megamoe` repo,
**`next/` greenfield tree** (a different codebase generation from the
`kernel_src/cutedsl_megamoe` SM100 drop and the `kernel_src/sm90` fork — the
`next/` tree uses fully relative imports and a composable `api.py`
kernel-component model).

## Provenance

- Upstream repo: `cutedsl_megamoe` (kernel team), local mirror
  `/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/cutedsl_megamoe`
- Upstream commit: `882c83e2ce4086c3cd4211fc5a2296143c5e2aea`
  (2026-08-08, "Merge branch 'training/next_glu_rubin' into 'main'")
- Copied subtree: `next/sources/` → `src/sources/`
- DSL floor: upstream CI pins `nvidia-cutlass-dsl[cu13]==4.6.0`
  (`ci/requirements.txt`); the kernels need `cutlass.utils.rubin_helpers`.
  Validated here on 4.7.0.

## Scope of this drop (inference only)

Only the **Rubin inference block-scaled swap-AB MegaMoE kernel** closure is
vendored — `sources/kernel_src/rubin/inference/mega/`
(`BlockScaledSwapAbMegaMoeKernel` + its FC12 mainloop / gated-act epilogue /
extension / dynamic mainloop / topk reduce) plus the shared runtime it imports
(`sources/{api,quant_def}.py`, `sources/helpers/`, `sources/communication/`,
`sources/kernel_src/{schedulers,function_mapping}`).

The kernel is generic over `QuantKind` (nvfp4, mxfp4, mxfp8_e4m3, mxfp8_e5m2,
mxfp4_mxfp8); the flashinfer backends currently wire up nvfp4 and mxfp8.

Deliberately NOT vendored (future migrations extend this same directory):
`rubin/training/` (the fwd_glu fprop / bwd_dglu dgrad / traditional wgrad
training kernels — an earlier revision of this drop vendored the fwd_glu
subtree; it was removed when the flashinfer backends moved to the inference
kernel) and the whole `kernel_src/blackwell/` tree.

## Pending local diffs vs upstream

Per `kernel_src/README.md`, `src/` is verbatim except for the following
recorded diff (a consequence of the inference-only scope):

1. `src/sources/kernel_src/rubin/inference/mega/topk_reduce.py` — upstream
   ships this as a `<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` marker shim
   importing from `blackwell/inference/mega/topk_reduce.py`; upstream's
   `kernel_export` script inlines the blackwell source at export time. The
   same inline was performed at vendor time (whole-file copy — its relative
   imports resolve identically at this depth). If the blackwell tree is ever
   vendored, this can revert to the marker-shim form.

## Layout

- `src/` — the drop (`sources` becomes a top-level module via
  `shim/_paths.py`, same mechanism as the sibling trees)
- `shim/` — flashinfer-owned adaptation layer; the only importer of `src/`
