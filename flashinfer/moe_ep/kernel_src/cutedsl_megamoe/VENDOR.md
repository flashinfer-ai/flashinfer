# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

- **Repo**: <https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe>
  (NVIDIA-internal GitLab; see `ACKNOWLEDGEMENT.md` for authors/contacts).
- **Vendored commit**: not recorded — the current `src/` drop was taken
  2026-07-13, before this VENDOR.md existed (it landed in flashinfer via
  PR #3980). The next full re-sync MUST pin the upstream SHA here. Until
  then the only pinned points are the two files synced ahead of the drop
  plus the reviewed ReLU2 partial sync (see pending diffs below,
  `50117315d` and `d8cbe837abf6a528ecd49ca960db846f7b8ba321`). Git
  archaeology found `8ba9c4efc5b6274b0307c289c876840d18c642ae` and its
  mainline merge `1995b17d946b7a56e39eb3a9c931b83a906e4b1b` as the
  nearest historical lineage anchors for this mixed drop; neither is asserted
  to be an exact full-tree snapshot.
- **Last synced**: 2026-07-13 (full drop); 2026-08-10 partial re-sync of
  `inputs_process.py` + `host_utils.py`; 2026-08-20 approved partial re-sync
  of the NVFP4 single-plane ReLU2 production/reference surface (see pending
  diffs).
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
- The 2026-08-20 ReLU2 update is an explicitly approved partial re-vendor from
  a pinned upstream commit, not a new full-drop claim. The five-file surface
  below is therefore the documented exception until the next compatible full
  re-sync.

## Pending local diffs vs upstream

- The native single-plane ReLU2 production/reference delta was selectively
  re-vendored from upstream commit
  `d8cbe837abf6a528ecd49ca960db846f7b8ba321` (parent
  `a23de9677cb5af40420bd74430dff5e2cd721003`) on 2026-08-20. The exact
  vendored surface is:
  `src/moe_nvfp4_swapab/activation.py` (new),
  `src/moe_nvfp4_swapab/epilogue_refactor.py`,
  `src/moe_nvfp4_swapab/kernel_fc12.py`,
  `src/moe_nvfp4_swapab/mega_reference.py`, and
  `src/moe_nvfp4_swapab/megamoe_kernel.py`. This was a reviewed semantic
  backport onto the older mixed tree, not a whole-file copy: it preserves the
  FlashInfer singleton-expert TMA-mode fix in `kernel_fc12.py` and all other
  existing drop divergences. No runner or harness file was synced.
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
