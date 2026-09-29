# FlashInfer GDN non-CP general prefill and decode source

This reproducible package contains only the exact optimized GDN schedule specializations admitted by the frozen general prefill, FP32-state T=1/MTP decode, and promoted BF16 serving contracts. The manifest records every row on SM100a and SM103a. Explicit GDN non-CP requests outside the listed routes must fail closed. Generated CUDA and TVM FFI host shims are source-only.

Indexed prefill requires every state slot to be in `[0, pool_size)`. Indexed decode requires every initial/output state slot to be `-1` or in `[0, state.size(0))`. The prefill adapter validates CUDA-resident slots asynchronously on the caller stream before dispatch, including during CUDA Graph capture and replay. The decode adapters trust caller-provided slots by default (parity with the CuTe decode kernels; the per-call `torch._assert_async` chains cost ~40 us of GPU time around a ~5 us kernel) and perform the same asynchronous fail-closed check only when `FLASHINFER_CAKE_GDN_VALIDATE_SLOTS=1`.


## Prefill performance refresh

The generated prefill kernels are validated on B200 and B300. See
[RESULTS.md](RESULTS.md) for the measured comparisons, correctness coverage,
reference revisions, and sanitizer status; [validation-header.json](validation-header.json)
records the qualified manifest and timing protocol.

Use the existing GDN prefill API with `backend="cake_gdn"` to select this backend
explicitly. Automatic prefill dispatch continues to select CuTe. The existing
19 generated decode variants and their public dispatch are preserved.

## Qwen3.5 TP=2 speculative-verify rows

Qwen3.5-35B-A3B served with `--tp 2 --speculative-num-draft-tokens 7` verifies
T=7 tokens per step per rank (H=8 / HV=16, BF16 state pool, seven-step BF16
checkpoint cache, `disable_state_update=True`). The package carries four extra
decode variants for that workload: `gdn_decode_pretranspose_t4_bf16state_tile16`
specialized to `T_STEPS=7` and `T_STEPS=8` (full-warp tile-v16 route, B<=4) and
`gdn_decode_pretranspose_mtp_t4_bf16state_wide128` with `T_STEPS=7`,
`TILE_V_WIDE=32` for H=8/HV=16 (B>=5) and H=16/HV=32 (TP=1). The selector
admits `(B, 7, 8, 16, strided, True, True, 7)` for B=1..8, `(1, 8, 8, 16, …, 8)`
and `(1, 7, 16, 32, …, 7)`; the launch tile size now follows the resolved route
(`.tile16_fullwarp` -> 16). The generating change and its B200 evidence live in
the Cake repository (CAKE-458). FI_README_RESULTS_PLACEHOLDER
