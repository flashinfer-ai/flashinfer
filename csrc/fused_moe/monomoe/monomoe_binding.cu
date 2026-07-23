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

#include <type_traits>

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
                           int64_t scoring_func, bool renormalize,
                           ffi::Optional<TensorView> expert_bias, double routed_scaling_factor);

namespace {

// Does the (E, N, K) weight-shape signature match this Dims variant?  The
// kernel is hard-specialized per shape, so the runtime tensors must match
// exactly before launch (otherwise the kernel reads/writes out of bounds).
template <class Dims>
bool shape_matches(const TensorView& activations_in, const TensorView& expert_weights_up,
                   const TensorView& expert_weights_down) {
  return activations_in.ndim() == 2 && activations_in.size(1) == Dims::K &&
         expert_weights_up.ndim() == 3 && expert_weights_up.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_up.size(1) == 2 * Dims::N && expert_weights_up.size(2) == Dims::K &&
         expert_weights_down.ndim() == 3 && expert_weights_down.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_down.size(1) == Dims::K && expert_weights_down.size(2) == Dims::N;
}

}  // namespace

// TEMP_FP8_OFFSET regression anchor (design doc §4): the down-activation TMA
// descriptor addresses `spec->temp_fp8` as `scratchpad_ptr + TEMP_FP8_OFFSET`,
// so that constant must equal `offsetof(MoEGemmSpec<Dims>, temp_fp8)`; new
// fields go at the tail of MoEGemmSpec<Dims>.
static_assert(
    offsetof(::monomoe::MoEGemmSpec<::monomoe::Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA>,
             temp_fp8) ==
        ::monomoe::MoEGemmSpec<
            ::monomoe::Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA>::TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8).");

// Top-K MoE entry point: validate the runtime tensor shape and token count,
// then launch the single hard-specialized Dims variant.  BS8 serves M <= 8.
void monomoe_topk(TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
                  TensorView expert_scales_up, TensorView expert_weights_down,
                  TensorView expert_scales_down, TensorView activations_out, TensorView scratchpad,
                  int64_t top_k, int64_t scoring_func, bool renormalize,
                  ffi::Optional<TensorView> expert_bias, double routed_scaling_factor) {
  using Dims = ::monomoe::Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA;
  const int64_t m = activations_in.ndim() == 2 ? activations_in.size(0) : -1;

  if (!shape_matches<Dims>(activations_in, expert_weights_up, expert_weights_down)) {
    TVM_FFI_ICHECK(false) << "monomoe_topk: this module is built for E=" << Dims::NUM_EXPERTS
                          << ", N=" << Dims::N << ", K=" << Dims::K << " but got activations_in=["
                          << m << ", " << (activations_in.ndim() == 2 ? activations_in.size(1) : -1)
                          << "], expert_weights_up.dim0="
                          << (expert_weights_up.ndim() == 3 ? expert_weights_up.size(0) : -1)
                          << ".";
    return;
  }

  if (m < 1 || m > static_cast<int64_t>(Dims::BS)) {
    TVM_FFI_ICHECK(false) << "monomoe_topk: token count M=" << m
                          << " out of range for E=" << Dims::NUM_EXPERTS << ", N=" << Dims::N
                          << ", K=" << Dims::K << " (supported 1 <= M <= " << Dims::BS << ").";
    return;
  }

  monomoe_topk_launcher<Dims>(activations_in, router_logits, expert_weights_up, expert_scales_up,
                              expert_weights_down, expert_scales_down, activations_out, scratchpad,
                              top_k, scoring_func, renormalize, expert_bias, routed_scaling_factor);
}

// Scratchpad size (bytes) for the single Dims variant, sourced from
// sizeof(MoEGemmSpec<Dims>) so the buffer can never desync from the kernel's
// struct layout.
int64_t monomoe_scratchpad_size() {
  return static_cast<int64_t>(
      sizeof(::monomoe::MoEGemmSpec<::monomoe::Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA>));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_topk, monomoe_topk);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_scratchpad_size, monomoe_scratchpad_size);
