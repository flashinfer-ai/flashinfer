/*
 * TVM-FFI binding for the MonoMoe kernel.
 *
 * Exports:
 *   - monomoe_topk             shape-dispatched top-K MoE entry point
 *   - monomoe_scratchpad_size  scratchpad size (bytes) for the active shape
 *
 * The implementation (and the whole `src/moe.cuh` unity build) lives in
 * `monomoe_wrapper.cuh`, which is `#include`d here so the launcher template is
 * in this translation unit.  This mirrors the `csrc/bgmv_moe/` pattern where
 * the binding TU `#include`s the ops sources.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "monomoe_wrapper.cuh"

// Declaration of the launcher function template (defined and explicitly
// instantiated for each Dims variant at the bottom of monomoe_wrapper.cuh).  The
// explicit instantiation there materializes the symbol; here we only need the
// declaration so a variant can be named in the dispatcher below.
template <class Dims>
void monomoe_topk_launcher(TensorView activations_in, TensorView router_logits,
                           TensorView expert_weights_up, TensorView expert_scales_up,
                           TensorView expert_weights_down, TensorView expert_scales_down,
                           TensorView activations_out, TensorView scratchpad, int64_t top_k,
                           int64_t scoring_func, bool renormalize);

namespace {

// Does the (E, N, K) weight-shape signature and the token cap match this Dims
// variant?  The kernel is hard-specialized to a fixed shape, so the runtime
// tensors must match it exactly before we launch — otherwise the kernel would
// read/write out of bounds.  Keyed on the up/down weight shapes (which pin
// E, N, K) plus the BS token cap from `activations_in`.
template <class Dims>
bool shape_matches(const TensorView& activations_in, const TensorView& expert_weights_up,
                   const TensorView& expert_weights_down) {
  return activations_in.ndim() == 2 && activations_in.size(0) <= Dims::BS &&
         activations_in.size(1) == Dims::K &&  //
         expert_weights_up.ndim() == 3 && expert_weights_up.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_up.size(1) == 2 * Dims::N && expert_weights_up.size(2) == Dims::K &&  //
         expert_weights_down.ndim() == 3 && expert_weights_down.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_down.size(1) == Dims::K && expert_weights_down.size(2) == Dims::N;
}

}  // namespace

// Single top-K MoE entry point: dispatch on the runtime tensor shape to the
// matching hard-specialized Dims variant.  Today there is exactly one variant
// (BS8 / E256 / Qwen3.5-35B block-FP8 WGMMA+TMA); a new shape gets a new
// `if` branch plus its explicit instantiation in monomoe_wrapper.cuh.
void monomoe_topk(TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
                  TensorView expert_scales_up, TensorView expert_weights_down,
                  TensorView expert_scales_down, TensorView activations_out, TensorView scratchpad,
                  int64_t top_k, int64_t scoring_func, bool renormalize) {
  using Qwen35 = monomoe::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA;
  if (shape_matches<Qwen35>(activations_in, expert_weights_up, expert_weights_down)) {
    monomoe_topk_launcher<Qwen35>(activations_in, router_logits, expert_weights_up,
                                  expert_scales_up, expert_weights_down, expert_scales_down,
                                  activations_out, scratchpad, top_k, scoring_func, renormalize);
    return;
  }
  TVM_FFI_ICHECK(false) << "monomoe_topk: no kernel for this shape. Supported: E="
                        << Qwen35::NUM_EXPERTS << ", N=" << Qwen35::N << ", K=" << Qwen35::K
                        << ", BS<=" << Qwen35::BS << " (got activations_in=["
                        << activations_in.size(0) << ", "
                        << (activations_in.ndim() == 2 ? activations_in.size(1) : -1)
                        << "], expert_weights_up.dim0="
                        << (expert_weights_up.ndim() == 3 ? expert_weights_up.size(0) : -1) << ").";
}

// Scratchpad size (bytes).  Exported so Python sizes the global scratchpad from
// the authoritative `sizeof(MoEGemmSpec<Dims>)` rather than re-deriving the
// struct layout (which would silently desync if any field is added — see the
// TEMP_FP8_OFFSET regression anchor in monomoe_wrapper.cuh).  The kernel's
// barrier counters live at the tail of this struct, so the buffer MUST be at
// least this large.  With a single Dims variant this is its size; once more
// variants exist, size to the max so any shape fits the same buffer.
int64_t monomoe_scratchpad_size() {
  return static_cast<int64_t>(
      sizeof(monomoe::MoEGemmSpec<monomoe::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_topk, monomoe_topk);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_scratchpad_size, monomoe_scratchpad_size);
