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

- **Opt-in quantized combine arithmetic:** the NVFP4 FC2 epilogue forwards
  its BI flag to the shared canonical BF16-to-quantized wire encoder for
  both register and across-lane layouts. Legacy MXFP8 instead preserves its
  FP32-to-MXFP8 legacy quantizer through an explicit ordered helper. The
  shared top-k reducer uses a zero-seeded ordered slot fold in BI mode,
  masking payload, scale and score loads for invalid routes. Legacy launchers
  pass route indices for this quantized BI branch. Ordinary quantization and
  reduction retain their existing paths. Reapply until upstream absorbs this
  integration patch.

- **Opt-in activation variants:** the BF16, NVFP4 and MXFP8 FC12 setup forwards
  an optional `batch_invariant_activation` recipe to its epilogue. Only the
  enabled BI path interprets that recipe through the shared ordered activation
  helper; a missing recipe keeps each provider's existing default helper.
  The NVFP4 BI path also follows the existing per-token scale shuffle mapping,
  folding token scale times expert alpha with an explicit RN multiply before
  accumulator dequantization. Ordinary activation branches are unchanged.
  Reapply until upstream absorbs these integration hooks.

- **Shared FC2 epilogue compatibility:**
  `src/moe_nvfp4_swapab/epilogue_refactor.py` defaults the optional
  `batch_invariant_chain` flag to false when forwarding FC2 conversion. The
  mixed MXFP8-weight/BF16-activation epilogue also uses this adapter and has no
  NVFP4 arithmetic flag; its ordinary conversion must remain available.
  An explicitly enabled NVFP4 batch-invariant flag is preserved. Reapply until
  upstream absorbs the fix.

- **MXFP8 batch-invariant arithmetic:** the MXFP8 frontend binds a fixed
  M256/N256/K128 paired-CTA geometry and native K32 MMA fold, with routing
  weights applied before FC2. `src/moe_mxfp8_glu/kernel_mxfp8_glu_fc12.py`
  passes the opt-in arithmetic flag to `epilogue_mxfp8.py`, which interprets
  the separate legacy MXFP8 activation and block32 UE8M0 requantization
  chains, including explicit scale conversion and the existing packed
  E4M3/E5M2 data conversion. Its FC2 BF16 conversion and launcher use the
  pinned ordered masked top-k fold shared with NVFP4. The legacy multiplication
  association and quantizer remain distinct from the next-generation providers.
  Ordinary execution and the inherited BF16 epilogue retain their old branches.
  Reapply these necessary local changes until upstream absorbs them.

- **NVFP4 batch-invariant arithmetic:** the NVFP4 frontend binds a fixed
  M256/N128/K256 paired-CTA geometry, K64 MMA fold and full-N descriptor.
  `src/moe_nvfp4_swapab/kernel_fc12.py` omits dynamic short-N input shifts
  in this mode; `epilogue_refactor.py` uses the shared ordered activation,
  block16 requantization and FC2 conversion. `topk_reduce.py` uses an explicit
  +0 slot fold with early add or late FMA and masked operands.
  `src/src/inputs_process.py` binds the same pinned offline input quantizer.
  The branches are opt-in and preserve the ordinary implementations.
  Reapply these necessary local changes until upstream absorbs them.

- **Single-CTA dynamic scheduling:**
  `src/moe_nvfp4_swapab/fc1_fc2_fuse_sched.py` publishes atomic-counter work
  through a warp shuffle for one-CTA clusters, followed by the existing local
  scheduler-to-consumer pipeline. Only multi-CTA clusters use the DSMEM
  asynchronous-store publication protocol, whose PTX instruction requires at
  least two CTAs. This fixes newly admitted M128 BF16 plans; paired-CTA
  scheduling and the arithmetic trace are unchanged. Reapply until upstream
  absorbs the fix.

- **BF16 batch-invariant tile variants:**
  `src/moe_bf16_glu/kernel_bf16_glu_fc12.py` admits M128 single-CTA and M256
  paired-CTA tiles, N128/N256 and K64/K128/K256, with one instruction group
  per cluster and a guard against zero A/B pipeline stages. Its FC1 output
  SMEM has one stage per epilogue warp, independent of the N subtile count.
  `src/moe_bf16_glu/epilogue_bf16.py` uses N/2 post-SwiGLU output columns.
  The shared `src/moe_mxfp8_glu/epilogue_mxfp8.py` FC1 TMA-store view uses
  the epilogue's tile width; the MXFP8 default stays at 128 columns. Keep
  these changes together: allowing N128 alone misaddresses FC1 stores.
  Shape-dependent arithmetic-padding and SMEM admission live in the shim
  and FlashInfer policy, outside the vendored device source.

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

- **Opt-in W8A16 BI:** `src/moe_mxfp8_bf16_glu/epilogue_mxfp8_bf16.py`,
  `kernel_mxfp8_bf16_glu_fc12.py` and `megamoe_kernel_mxfp8_bf16.py` carry the
  BF16 handoff activation adapter and masked ordered finalizer. The BI path
  fixes the dense M256/N128 two-CTA atom and rejects accumulator overlap and
  floating atomics. Its weight-decode/BF16-MMA receipt is separate from native
  block-scaled W8A8. The ordinary branch retains its original activation.
- **Legacy W8A8 BI quantized combine:** the shim can select the existing
  internal MXFP8 payload/SF planes. The wire always stores E4M3, including
  E5M2 input sessions, and quantizes FC2 FP32 values directly. This has a
  separate receipt from providers that first round FC2 to BF16. Public outputs
  remain BF16; wire payloads and SFs stay inside the symmetric workspace.

## Batch-invariant compiled identity patch

The NVFP4 FC12/MegaMoE and W8A16 FC12 `name()` methods append the BI mode and,
when present, a hash of the ordered activation signature's effective FP32
coefficient bits. Their ordinary names remain byte-for-byte unchanged. Retain
this host metadata exception alongside the BI lowering hooks until upstream
absorbs it.

## Related trees

- `kernel_src/sm90/pull_style_cutedsl_megakernel/` is a **separate snapshot**
  of a fork of this repo (older common code, Hopper FP8 pull kernel). It is
  not merged into this directory on purpose: one kernel_src dir = one upstream
  commit. If upstream merges the SM90 kernel into the mother repo, fold that
  tree into this one on the next re-sync.

## Consumers

- `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/`
