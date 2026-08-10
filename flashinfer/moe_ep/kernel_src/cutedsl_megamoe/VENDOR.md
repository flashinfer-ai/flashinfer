# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Repo**: NVIDIA CuTeDSL MegaMoE kernel team's repo (see `ACKNOWLEDGEMENT.md`
  for authors/contacts). <!-- TODO: pin the repo URL -->
- **Vendored commit**: <!-- TODO: record the upstream SHA of the current src/ drop -->
- **Last synced**: <!-- TODO: date of the current drop -->
- **Vendored subset**: the four kernel packages only (`common/`, `src/`,
  `moe_mxfp8_glu/`, `moe_nvfp4_swapab/`) under `src/` — no repo scaffolding
  (`ci/`, `tester/`, `tests/`, `scripts/`, `pyproject.toml`, …).

## Policy

- `src/` is a **verbatim** copy of the upstream drop: no injected files, no
  local edits. `diff -r src/<pkg> <upstream>/<pkg>` must come back clean.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Pending local diffs vs upstream

- `src/src/inputs_process.py` and `src/common/host_utils.py` are synced
  **ahead** of the recorded drop, to upstream commit
  `50117315dbcd2ffb1e8c1c4dab4be9b42cad24ab`
  (<https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/tree/50117315dbcd2ffb1e8c1c4dab4be9b42cad24ab>),
  taken 2026-08-10: the kernel team's fix for the fused activation-quant
  staging breaking on CuTe-DSL 4.7 (mxfp8 path reworked so each lane owns one
  contiguous 16-byte fp8 store, lane pairs reduce the 32-element block amax
  via shuffle; plus a hidden-size row-alignment guard in `__init__`).
  `host_utils.py` rides along because the file's self-test harness imports
  its `mxfp8_quantize_per_block_32_row` reference quantizer (additive change,
  no dependents beyond the harness). Resolves at the next full re-sync once
  the tree moves past that commit.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
