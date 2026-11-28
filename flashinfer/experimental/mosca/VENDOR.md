# Vendoring record: MOSCA kernels

This file records where the upstream code under this package comes from and
how to re-sync it. `vendor/<tree>.json` pins the same facts in machine-readable
form. Two checks read it:
- `tests/experimental/mosca/test_mosca_vendor.py` checks the installed files
  against the manifest;
- `python -m flashinfer.experimental.mosca._vendor <tree> <upstream checkout>`
  checks the manifest against a checkout of the upstream commit. With `--diff`
  it also prints each derived file's diff from upstream.

## Policy (every tree)

- **Vendored files are verbatim.** `csrc/vendor/<tree>/` holds upstream files
  byte-identical to the pinned commit: no MOSCA change, no local edits, no
  formatting. `.pre-commit-config.yaml` excludes `csrc/vendor/`.
- **MOSCA changes are code of this package.** No patch series is kept.
  - A file MOSCA changes is copied out of the tree into `csrc/<family>/`. It
    starts with FlashInfer's license header and the line "Portions derived
    from <upstream>, <copyright>, <license>" (FlashInfer's convention for
    derived code), followed by the upstream file and commit and what changed.
    It keeps upstream's layout, so upstream changes merge into it cleanly, and
    formatters skip it.
  - Every change in such a file is gated by a macro, so with no MOSCA macro
    defined the kernel is the upstream kernel.
  - The manifest lists these files under `derived_sources`, with the upstream
    hash and the line regions of each macro.
- **Adaptation** lives in files this package owns:
  - `csrc/common/` (shims included before any upstream header);
  - `csrc/<family>/` (FlashInfer-side host code);
  - the Python modules.
- **License.** Each tree's upstream license ships as `LICENSE.<Upstream>` next
  to this file.

## Tree `deep_gemm`

- **Upstream.**
  [`sgl-project/DeepGEMM`](https://github.com/sgl-project/DeepGEMM).
  - Commit: `1ff3932f0f9d9187b92afb476da4e3ade10a6575` (2026-09-12), the parent
    of tag `v0.2.0` and the source of the `sgl-deep-gemm 0.2.0` wheel.
  - License: MIT (`LICENSE.DeepGEMM`).
- **Vendored subset.** `csrc/vendor/deep_gemm/` holds the include closure of
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` without that
  file: 17 headers, byte-identical to upstream. No host code, no Python, no
  other kernels. CuTe/CUTLASS come from FlashInfer's `3rdparty/cutlass`.
- **The MOSCA kernel.** `csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh` is upstream's
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` with the MOSCA
  changes: +163 / -13 lines, 40 of them the license and provenance header.
  The macros:
  - `DGX_SMALL_T`: small-T dispatch path;
  - `DGX_SPARSE_COUNT`: sparse expert counting (requires `DGX_SMALL_T`);
  - `DGX_BCAST`: in-kernel all-gather of the combined output. Every warp that
    wrote gathered rows issues `fence.acq_rel.sys` before the final NVLink
    barrier.

  The line regions are in the manifest and in `README.md`.
- **Host code reproduced here.** For the FP8 x FP4 kind without shared experts,
  `csrc/megamoe/sm100_fp8_fp4_planner.cu` and
  `csrc/megamoe/sm100_fp8_fp4_launcher.cu` reproduce the upstream host
  functions listed in their headers, from:
  - `csrc/apis/mega_moe.hpp`
  - `csrc/jit_kernels/heuristics/mega_moe.hpp`
  - `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`
  - `csrc/jit_kernels/impls/runtime_utils.hpp`

  `megamoe/weights.py` and `megamoe/staging.py` port the matching Python
  helpers (`deep_gemm/mega/__init__.py`, `deep_gemm/utils/math.py`).

  These files are re-implemented, not copied. MOSCA's host-side additions are
  this package's own code:
  - the tile rule (planner);
  - the gathered output region (planner);
  - the row offset packed into the token-count argument (launcher).

  The manifest (`ported_host_sources`) pins the upstream files' hashes at the
  commit above, and a changed hash on re-sync means the port must be
  re-diffed. The tests compare the planner with the installed `deep_gemm` for
  every per-rank token count.
- **Toolchain.** The kernel compiles against FlashInfer's CUTLASS/CuTe and CCCL
  (`3rdparty/cutlass` at `b46b16d0`), not DeepGEMM's submodules (CUTLASS
  `f3fde583` at the upstream commit). The multi-GPU tests compare outputs
  bitwise against the stable backend, which is built with DeepGEMM's own
  toolchain.

### Re-sync workflow

1. Check out the new upstream commit (`NEW`) next to the current one (`OLD`).
2. Replace `csrc/vendor/deep_gemm/` with the include closure of the kernel
   header at `NEW`, without the kernel header itself.
3. Merge upstream's change to the kernel into the MOSCA kernel:
   `git merge-file csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh <OLD kernel> <NEW kernel>`.
   Resolve conflicts inside the macro blocks so that, with no MOSCA macro
   defined, the file is still the upstream kernel.
4. Update `vendor/deep_gemm.json`:
   - the commit;
   - the file hashes;
   - the `derived_sources` record (upstream hash, line counts, macro regions).

   Update the region table in `README.md` to match. Then run
   `python -m flashinfer.experimental.mosca._vendor deep_gemm <NEW checkout> --diff`
   and review the printed diff.
5. Diff the upstream host functions listed above between `OLD` and `NEW`. Port
   every change to the planner, the launcher, `weights.py` and `staging.py`,
   and update the `ported_host_sources` hashes.
6. Run `tests/experimental/mosca/` on 8 SM100 GPUs, including the bitwise
   comparison against the stable `sm100_fp8_fp4_bf16_deepgemm` backend with the
   matching `deep_gemm` installed.
