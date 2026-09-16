# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Repo**: <https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe>
  (NVIDIA-internal GitLab; see `ACKNOWLEDGEMENT.md` for authors/contacts).
- **Vendored commit**: the original full `src/` drop is not recorded — it was taken
  2026-07-13, before this VENDOR.md existed (it landed in flashinfer via
  PR #3980). The next full re-sync MUST pin the upstream SHA here. Until
  then the only pinned points are the two files synced ahead of the drop
  (see pending diffs below, `50117315d`).
- **Last synced**: 2026-07-13 (full drop); 2026-08-10 partial re-sync of
  `inputs_process.py` + `host_utils.py`; 2026-09-16 addition of
  `moe_nvfp4_bf16_glu/` from dastokes commit
  `d7b3a3d3ab6d29745f9ae321cd07ff81448df1c8` plus its three shared-file
  prerequisites (see pending diffs).
- **Vendored subset**: the kernel packages only (`common/`, `src/`,
  `moe_bf16_glu/`, `moe_mxfp8_glu/`, `moe_nvfp4_bf16_glu/`,
  `moe_nvfp4_swapab/`) under `src/` — no repo scaffolding
  (`ci/`, `tester/`, `tests/`, `scripts/`, `pyproject.toml`, …).

## Policy

- `src/` is normally a verbatim copy of the recorded upstream drop. Files
  explicitly listed below may be synced from a newer branch or carry a small
  local fix while that change is being upstreamed.
- All adaptation lives in `shim/` (ours), re-exported through `__init__.py`;
  FlashInfer backends import the package `__init__` only, never `src/`.
- Local bug fixes go upstream first, then re-sync. If an emergency local edit
  is unavoidable, list it here as a pending-upstream diff until the next drop
  absorbs it.

## Pending local diffs vs upstream

- `src/moe_nvfp4_bf16_glu/` is a verbatim package copy from dastokes commit
  `d7b3a3d3ab6d29745f9ae321cd07ff81448df1c8` (2026-09-11). Its required
  shared changes were applied to `src/src/sym_buffer.py`,
  `src/moe_bf16_glu/mega_runner.py`, and
  `src/moe_bf16_glu/mega_reference_bf16.py` while preserving FlashInfer's
  newer local formatting and compatibility fixes. Reconcile those files when
  the branch lands in the kernel-team upstream.
- `src/moe_nvfp4_bf16_glu/kernel_nvfp4_bf16_glu_fc12.py` and
  `megamoe_kernel_nvfp4_bf16.py` additionally carry FlashInfer's
  singleton-expert TMA-mode preservation fix, matching the pure-NVFP4 fix in
  PR #4296. Confirm the mixed branch contains it before the next re-sync; if
  not, retain/reapply this diff after replacing the package.

- `src/src/inputs_process.py` is synced **ahead** of the recorded drop, to
  upstream commit `50117315dbcd2ffb1e8c1c4dab4be9b42cad24ab`
  (<https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/blob/50117315dbcd2ffb1e8c1c4dab4be9b42cad24ab/src/inputs_process.py>),
  taken 2026-08-10: the kernel team's fix for the fused activation-quant
  staging breaking on CuTe-DSL 4.7 (mxfp8 path reworked so each lane owns one
  contiguous 16-byte fp8 store, lane pairs reduce the 32-element block amax
  via shuffle; plus a hidden-size row-alignment guard in `__init__`).
  ONLY this one file is ahead: at that commit upstream also renamed
  `common/host_utils.py`'s `mxfp8_quantize_per_block_32` to `..._row`, and
  pulling that file forward breaks the rest of the recorded drop (shim
  `kernel_helpers`, `mega_reference*.py` — the rename ripples through
  `mega_reference.py`'s changed return signature into the runners). Known
  cost: the harness at the bottom of `inputs_process.py`
  (`python -m src.inputs_process`) fails its **mxfp8** case with an
  ImportError against the recorded-drop `host_utils` — the nvfp4 cases and
  every shim/kernel path are unaffected (the kernel code imports
  `host_utils` nowhere). The harness was validated green (3/3 cases, dsl
  4.6.1 + 4.7.0) with the newer `host_utils` before this was understood.
  Resolves at the next full re-sync once the tree moves past that commit.
- `src/moe_nvfp4_swapab/kernel_fc12.py` carries the flashinfer-upstream
  singleton-expert TMA-modes fix (flashinfer-ai/flashinfer `4fbac49f`,
  PR #4296, applied 2026-08-12 during the TOT merge): the compact expert
  mode of singleton weight tensors stays dynamic so the runtime expert
  extent remains visible in FC1/FC2 weight TMA descriptors. Confirm the
  kernel-team repo has an equivalent before the next re-sync.
- `src/moe_nvfp4_swapab/runner_common.py` carries a local
  `_check_triton_flat_index` guard (added for PR #4113 review) on the
  int32-indexed Triton helpers (`_rcp_approx_kernel`, `_swiglu_pair_kernel`);
  `_pack_fp4_kernel` is exempt because it widens its flat index to int64 for
  the > 2**31-element combine round-trip. Send upstream on the next re-sync.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
- `backends/mega/kernel/sm100/bf16_nvfp4_bf16_cutedsl/`
