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

## Scope of this drop (fprop only)

Only the **Rubin training forward (fprop) mxfp8 GLU mega kernel** closure is
vendored — `sources/kernel_src/rubin/training/mega/fwd_glu/`
(`Sm107MegaMoEMxfp8GluKernel`, `Sm107Mxfp8GluFc12Kernel`) plus the shared
runtime it imports (`sources/{api,quant_def}.py`, `sources/helpers/`,
`sources/communication/`, `sources/kernel_src/{schedulers,function_mapping}`).

Deliberately NOT vendored (future migrations extend this same directory):
`rubin/inference/`, `rubin/training/mega/bwd_dglu/` (dgrad),
`rubin/training/traditional/` (wgrad), and the whole
`kernel_src/blackwell/` tree.

## Pending local diffs vs upstream

Per `kernel_src/README.md`, `src/` is verbatim except for the following
recorded diffs (all consequences of the fprop-only scope):

1. `src/sources/kernel_src/rubin/training/__init__.py` — upstream re-exports
   the `.traditional` wgrad kernels; the import is removed because that
   subtree is not vendored. Restore verbatim when wgrad migrates.
2. `src/sources/kernel_src/rubin/training/mega/topk_reduce.py` and
   `.../tmem_transpose.py` — upstream ships these as
   `<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` marker shims importing from
   `blackwell/inference/mega/`; upstream's `kernel_export` script inlines the
   blackwell source at export time. The same inline was performed at vendor
   time (topk_reduce: whole-file copy of
   `blackwell/inference/mega/topk_reduce.py`, whose relative imports resolve
   identically at this depth; tmem_transpose: verbatim extraction of
   `_TmemTranspose16x32Core` from
   `blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py` lines
   521-741). If the blackwell tree is ever vendored, these can revert to the
   marker-shim form.

## Layout

- `src/` — the drop (`sources` becomes a top-level module via
  `shim/_paths.py`, same mechanism as the sibling trees)
- `shim/` — flashinfer-owned adaptation layer; the only importer of `src/`
