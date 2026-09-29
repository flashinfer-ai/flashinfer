# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Vendored commit**: `4d80fece66fc5b5e8c94026a33f0854d3a01afe9`
- **Last synced**: 2026-09-29
- **Vendored subset**: the six kernel packages only (`common/`, `src/`,
  `moe_bf16_glu/`, `moe_mxfp8_glu/`, `moe_mxfp8_bf16_glu/`,
  `moe_nvfp4_swapab/`) under `src/` — no repo scaffolding (`ci/`, `tester/`,
  `tests/`, `scripts/`, `pyproject.toml`, …).

## Policy

- `src/` is a **verbatim** copy of the upstream drop: no injected files, no
  local edits. `diff -r src/<pkg> <upstream>/<pkg>` must come back clean.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Pending local diffs vs upstream

Compared with the reference upstream branch, the tracked differences are:

- **Persistent top-k reduction:** `src/moe_nvfp4_swapab/topk_reduce.py`
  carries the newer one-cursor fixed-grid scheduler from
  `89e339be770daa35b2faaa29d8c6a13606330845`. It is integrated behind
  `topk_reduce_persistent` in
  `src/moe_nvfp4_swapab/megamoe_kernel.py` and
  `src/moe_mxfp8_glu/megamoe_kernel_mxfp8.py`; both integrations honor the
  vendored `skip_topk_reduce` path.
- **Activation controls:** `src/moe_nvfp4_swapab/{epilogue_refactor,kernel_fc12,megamoe_kernel}.py`
  retain FlashInfer's custom SwiGLU alpha/beta parameters and allow
  `situ_linear_beta=None`; upstream implements standard SwiGLU and requires
  both positive SiTU beta values.
- **Vendored-subset imports:** `src/moe_mxfp8_glu/runner_col_requant.py`
  uses package-local helpers and lazily imports its checker so importing the
  runner does not require omitted test scaffolding.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
