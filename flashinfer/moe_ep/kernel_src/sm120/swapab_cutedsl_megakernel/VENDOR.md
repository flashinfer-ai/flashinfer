# Vendoring record: sm120/swapab_cutedsl_megakernel

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Repo**: https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe.git
  (fork of the NVIDIA CuTeDSL MegaMoE kernel team's repo; see the mother
  tree's `ACKNOWLEDGEMENT.md` for authors/contacts)
- **Branch**: `run/sm120-mxfp8-perf`
- **Vendored commit**: `d19d30a748f9e402b8a2a33083fdf530231cf647`
  ("Adapt SM120 runner to latest MXFP8 host utils", 2026-07-27)
- **Last synced**: 2026-08-06 (copied from the local worktree
  `/home/scratch.mhoqueanik_gpu/cutedsl_megamoe_sm120/sm120_swapab_wt`)
- **Vendored subset**: the four kernel packages only (`common/`, `src/`,
  `moe_sm120_mxfp8_swapab/`, `moe_mxfp8_glu/`) under `src/` — no repo
  scaffolding (`ci/`, `tester/`, `tests/`, `scripts/`, `pyproject.toml`, …).
  `moe_mxfp8_glu/` is included because
  `moe_sm120_mxfp8_swapab/mega_reference.py` lazily imports its torch
  reference (`mega_reference_mxfp8`).

## Policy

- `src/` is a **verbatim** copy of the upstream drop: no injected files, no
  local edits. `diff -r src/<pkg> <upstream>/<pkg>` must come back clean.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Pending local diffs vs upstream

The snapshot was taken from a **dirty worktree**: two files carried
uncommitted changes on top of `d19d30a` and are vendored as found on disk
(they are still verbatim w.r.t. the worktree, not the commit):

- `moe_sm120_mxfp8_swapab/mega_runner.py` (+~25/−~19 lines)
- `src/bootstrap.py` (+~42/−~7 lines)

Fold these into a committed upstream state on the next re-sync.

## Related trees

- `kernel_src/cutedsl_megamoe/` is the mother-repo snapshot (SM100 NVFP4 +
  MXFP8 kernels). This tree is a **separate snapshot** of a fork carrying the
  SM120 MXFP8 swap-AB kernel; it is not merged into the mother tree on
  purpose: one kernel_src dir = one upstream commit. If upstream merges the
  SM120 kernel into the mother repo, fold this tree into that one on the next
  re-sync.
- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is the analogous separate
  snapshot for the Hopper FP8 pull kernel.

## Consumers

- `backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/`
