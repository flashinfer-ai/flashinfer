/*
 * TVM-FFI binding for the MoE monokernel (megamoe).
 *
 * Exports a single function:
 *   - moe_monokernel_topk  (BS8 / E256 / Qwen3.5-35B block-FP8 WGMMA+TMA path)
 *
 * The implementation (and the whole `src/moe.cu` unity build) lives in
 * `moe_wrapper.cu`, which is `#include`d here so the macro-generated launcher
 * is in this translation unit.  This mirrors the `csrc/bgmv_moe/` pattern
 * where the binding TU `#include`s the ops `.cu`.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_wrapper.cu"

// Forward declaration of the macro-generated launcher (global scope, defined
// by MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION at the bottom of moe_wrapper.cu).
void moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_impl(
    TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
    TensorView expert_scales_up, TensorView expert_weights_down, TensorView expert_scales_down,
    TensorView activations_out, TensorView scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize);

// Scratchpad size (bytes) for the BS8 variant.  Exported so Python sizes the
// global scratchpad from the authoritative `sizeof(MoEGemmSpec<Dims>)` rather
// than re-deriving the struct layout (which would silently desync if any
// field is added — see the TEMP_FP8_OFFSET regression anchor in
// moe_wrapper.cu).  The kernel's barrier counters live at the tail of this
// struct, so the buffer MUST be at least this large.
int64_t moe_monokernel_scratchpad_size() {
  return static_cast<int64_t>(sizeof(moe_monokernel::MoEGemmSpec<
                                     moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    moe_monokernel_topk, moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_monokernel_scratchpad_size, moe_monokernel_scratchpad_size);
