# Vendoring record: sm90/push_style_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only.

## Upstream

- **Repo**: flashinfer-ai/flashinfer — PR
  [#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069)
  (`sm90_push_fp8`: push-style FP8 whole-layer EP MoE backend for Hopper).
- **Vendored commit**: `301f8ce3dd42646bb12707251f50db619fb5c653` (PR head).
  The PR **merged** to main on 2026-08-12 as squash commit
  `f9b13ef11472d994ec02210e7a6fe62df3254636`; re-diffed 2026-08-13 — this
  tree is byte-for-byte identical to the merged SHA (no post-review deltas
  between the PR head and the merge). Future syncs diff against main.
- **Last synced**: 2026-08-12 (re-diffed vs merged SHA 2026-08-13).
- **Vendored subset**: the whole
  `flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe/` tree byte-for-byte
  (`src/{a2a,fp8_gemm}/` CUDA sources, `shim/`, `__init__.py`,
  `ACKNOWLEDGEMENT.md`). No `{$nv-internal-release}` markers present at this
  SHA (verified).

## Policy

- Taken **verbatim** from the PR head: no injected files, no local edits.
  Unlike the CuTeDSL drops, `shim/` here is part of the upstream PR and is
  vendored with it; FlashInfer-side adaptation lives in the taxonomy wrapper
  `backends/mega/kernel/sm90/fp8_fp8_bf16_push_cuda/` (ours).
- Local bug fixes go to the upstream PR first, then re-sync. If an emergency
  local edit is unavoidable, list it here as a pending-upstream diff until
  the next drop absorbs it.

## Pending local diffs vs upstream

- None.
