# Vendoring record: cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records *provenance and sync state* only; the drop-update *workflow* (what to
replace, what to audit) lives in `SKILL.md`.

## Upstream

The explicit Blackwell GenPhase prototype lives in the independent
[`../next_cutedsl_megamoe`](../next_cutedsl_megamoe/VENDOR.md) package. This
legacy package lazily re-exports its public API; it does not import Rubin or
replace its own device snapshot.

- **Vendored commit**: `fa662f345980f75776ae0c85d2e5c188918ded26`
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

- **Sparse decode (FlashInfer PR #6447 / #6565, 2026-10-14):** re-ported onto
  this drop in four device files. `src/src/token_comm.py` gives CTA0 ownership
  of small shared reset prefixes (EP1, prefixes above 16 KiB and NCU keep the
  distributed reset), preserves the grid counter phase when omitting a join,
  and, for capacities at most 1024, uses the last valid route extent to reduce
  the dispatch CTA count (`active_dispatch_warps` / `active_dispatch_threads`)
  while preserving physical metadata strides and full-grid high-word
  completion counts. `src/moe_nvfp4_swapab/topk_reduce.py` adds optional
  BF16 route masking (static and persistent paths); the NVFP4 and MXFP8
  `megamoe_kernel*.py` launchers pass the indices and expert count for BF16
  combine only. Payloads and scores of invalid routes are ignored.
  Those launchers also allocate a local extent/ready counter for capacities
  at most 1024. `token_comm.py` uses the last valid route extent to reduce the
  dispatch CTA count while preserving physical metadata strides and full-grid
  high-word completion counts. Larger capacities retain the original dispatch.
  The histogram allocation also stages the four per-warp extent maxima;
  padding it to at least four entries fits its existing 16-byte alignment.
  Reapply
  these four local device-file patches until upstream absorbs them; they do
  not introduce a GenPhase path or change quantized/in-kernel reduction.

- `src/src/inputs_process.py` carries the routing-output cast and optional
  offline NVFP4 scale multiplier from source revision
  `381675f3fa078867cebf8d64a0e034829f356424`. This is a narrow backport onto
  the existing snapshot, not a full file sync. Replace it with the next
  staging-helper import.
- `src/src/inputs_process.py` is synced **ahead** of the recorded drop, to
  upstream commit `50117315dbcd2ffb1e8c1c4dab4be9b42cad24ab`
  (`src/inputs_process.py` in the internal repo), taken 2026-08-10: the kernel team's fix for the fused activation-quant
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
- **Activation controls:** `src/moe_nvfp4_swapab/{epilogue_refactor,kernel_fc12,megamoe_kernel}.py`
  retain FlashInfer's custom SwiGLU alpha/beta parameters and allow
  `situ_linear_beta=None`; upstream implements standard SwiGLU and requires
  both positive SiTU beta values.
- **Vendored-subset imports:** `src/moe_mxfp8_glu/runner_col_requant.py`
  uses package-local helpers and lazily imports its checker so importing the
  runner does not require omitted test scaffolding.
- `src/src/inputs_process.py` launches pass `min_blocks_per_mp=1` (FlashInfer
  launch hint); reapply until upstream absorbs it.

- FC1 per-token activation scale (`enable_fc1_activation_per_token_scale`),
  ported from the kernel-team MR `qiangf/cutedsl_megamoe!1` (branch
  `feature/nvfp4-static-per-token-scale`, head `71a70c7`, base `9d004cc`)
  into the recorded drop. Touches `src/moe_nvfp4_swapab/{kernel_fc12,
  megamoe_kernel,epilogue_refactor,custom_ext}.py` and `src/src/token_comm.py`:
  a runtime `(T,)` fp32 tensor on the symmetric heap is peer-pulled by the
  dispatch warps next to the topk weight into a new pool-ordered local region
  (`l1_fc1_activation_per_token_scale_buffer`), and the fc1 epilogue folds it
  into the dequant (`alpha * scale_t * acc`) before the clamp / gated
  activation (pre-transpose half via warp shuffle, post-transpose half one
  scale per lane -- same 3-transpose schedule). Only the per-token-scale part
  of the MR is taken: the MR's compile-time `fc1_activation_norm_const` and
  `gate_activation="quick_geglu"` are NOT ported (flashinfer keeps the runtime
  per-expert `fc1_norm_const` tensor -- fold `1/norm_const` into `fc2_alpha`
  on the host -- and already covers quick-GeGLU via `swiglu_alpha=1.702`).
  Off by default: `name()` and the compile ABI are unchanged when disabled.
  Upstream `main` has since moved this feature to the `next/` path (see the
  kernel repo's `docs/FC1_PER_TOKEN_SCALE_NEXT_DESIGN.md`); re-derive on the
  next re-sync.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
