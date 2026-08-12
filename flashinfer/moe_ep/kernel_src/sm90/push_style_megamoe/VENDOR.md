# Vendoring record: sm90/push_style_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only.

## Upstream

- **Repo**: flashinfer-ai/flashinfer — PR
  [#4069](https://github.com/flashinfer-ai/flashinfer/pull/4069)
  (`sm90_push_fp8`: push-style FP8 whole-layer EP MoE backend for Hopper).
- **Vendored commit**: `301f8ce3dd42646bb12707251f50db619fb5c653` (PR head;
  the PR was still **open** when taken — re-diff against the merged SHA when
  it lands and absorb any post-review deltas).
- **Last synced**: 2026-08-12.
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
