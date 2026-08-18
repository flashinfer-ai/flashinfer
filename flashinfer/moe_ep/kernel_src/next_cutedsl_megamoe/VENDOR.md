# next_cutedsl_megamoe — vendored kernel drop

Rubin (SM107) MegaMoE kernels from the kernel team's `cutedsl_megamoe` repo,
**`next/` greenfield tree** (a different codebase generation from the
`kernel_src/cutedsl_megamoe` SM100 drop and the `kernel_src/sm90` fork — the
`next/` tree uses fully relative imports and a composable `api.py`
kernel-component model).

## Provenance

- Upstream repo: `cutedsl_megamoe` (kernel team), local mirror
  `/lustre/fsw/coreai_libraries_cudnn/mhoqueanik/cutedsl_megamoe`
- Upstream commit: `92dd334` (2026-08-15, "Merge branch 'ag_dev/perf_details'
  into 'main'"; brings in `a5b4d33` "Rubin MegaMoE Perf Improvment" — the
  mixed-CGA preferred/fallback cluster launch, reworked FC12 scheduler, and
  the token-in size-copy reorder around the metadata-ready wait). Previous
  snapshot: `882c83e2` (2026-08-08). The `rubin/inference/mega` files at
  `92dd334` are identical to perf-report commit `47881ad2`
  (`ag_dev/investigate_blackwell`).
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
kernel), `rubin/inference/local_mega/` (the single-GPU fused-routing
`BlockScaledSwapAbLocalMegaMoeKernel` added upstream in `4e0498e` — no EP
token comm, out of scope for the moe_ep backends), and the whole
`kernel_src/blackwell/` tree.

Note: upstream `92dd334` moved `software_sync.py` from
`sources/communication/nvlink_domain/` to `sources/helpers/`; the vendored
tree mirrors the move.

## Pending local diffs vs upstream

Per `kernel_src/README.md`, `src/` is verbatim except for the following
recorded diffs (both a consequence of the inference-only scope — upstream
ships each file as a `<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` marker shim
importing from the un-vendored `blackwell/` tree, and upstream's
`kernel_export` script inlines the blackwell source at export time; the same
whole-file inline was performed at vendor time. If the blackwell tree is ever
vendored, these can revert to the marker-shim form):

1. `src/sources/kernel_src/rubin/inference/mega/topk_reduce.py` — inlines
   `blackwell/inference/mega/topk_reduce.py`.
2. `src/sources/kernel_src/rubin/custom_mix_cga_helpers.py` — inlines
   `blackwell/custom_mix_cga_helpers.py` (mixed-CGA TMA helpers + the
   TMA-to-UMMA mixed-cluster pipeline; no relative imports, so the copy is
   byte-identical to the blackwell source).

## Layout

- `src/` — the drop (`sources` becomes a top-level module via
  `shim/_paths.py`, same mechanism as the sibling trees)
- `shim/` — flashinfer-owned adaptation layer; the only importer of `src/`
