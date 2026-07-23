#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "src/moe.cuh"
#include "tvm_ffi_utils.h"

// FlashInfer is framework-agnostic through TVM-FFI, so this binding takes
// `TensorView` operands (not `torch::Tensor`) and reports errors through
// `TVM_FFI_ICHECK`.  The vLLM tree's `cuda_utils.h` (which supplied
// `CUDA_CHECK`) does not exist here; define a self-contained equivalent that
// raises a TVM-FFI error on a non-success CUDA status.
#define CUDA_CHECK(call)                                                                     \
  do {                                                                                       \
    cudaError_t _e = (call);                                                                 \
    TVM_FFI_ICHECK(_e == cudaSuccess)                                                        \
        << "CUDA error " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__; \
  } while (0)

// TEMP_FP8_OFFSET regression anchor: the down-activation TMA descriptor
// addresses `spec->temp_fp8` as `scratchpad_ptr + TEMP_FP8_OFFSET`, so that
// constant must equal `offsetof(MoEGemmSpec<Dims>, temp_fp8)`.  The check is
// emitted once per instantiated config (a generated per-config static_assert
// in monomoe_binding.cu, since the layout is Dims-dependent), so it lives
// there rather than being hard-coded to one shape here
// (docs/design_docs/monomoe_kernel.md §4).

/**
 * @brief Host launcher for `moe_kernel_topk<Dims>`, with runtime top_k /
 * scoring_func / renormalize / expert_bias / routed_scaling_factor.
 *
 * Design doc: docs/design_docs/monomoe_kernel.md (section numbers below).
 * Checks the co-residency invariant (§3), zero-inits the scratchpad on first
 * use (§4), builds the TMA descriptors (§5), and launches the §1 pipeline.
 * Each shape variant is materialized by an explicit instantiation below and
 * exported from monomoe_binding.cu.
 */
template <class Dims>
void monomoe_topk_launcher(TensorView activations_in, TensorView router_logits,
                           TensorView expert_weights_up, TensorView expert_scales_up,
                           TensorView expert_weights_down, TensorView expert_scales_down,
                           TensorView activations_out, TensorView scratchpad, int64_t top_k,
                           int64_t scoring_func, bool renormalize,
                           ffi::Optional<TensorView> expert_bias, double routed_scaling_factor) {
  CHECK_INPUT(activations_in);
  CHECK_INPUT(router_logits);
  CHECK_INPUT(expert_weights_up);
  CHECK_INPUT(expert_scales_up);
  CHECK_INPUT(expert_weights_down);
  CHECK_INPUT(expert_scales_down);
  CHECK_INPUT(activations_out);
  CHECK_INPUT(scratchpad);
  CHECK_INPUT_TYPE(activations_in, dl_bfloat16);
  CHECK_INPUT_TYPE(router_logits, dl_bfloat16);
  CHECK_INPUT_TYPE(expert_weights_up, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(expert_scales_up, dl_float32);
  CHECK_INPUT_TYPE(expert_weights_down, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(expert_scales_down, dl_float32);
  CHECK_INPUT_TYPE(activations_out, dl_bfloat16);
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= 8) << "top_k must be between 1 and 8.";
  TVM_FFI_ICHECK(scoring_func == 0 || scoring_func == 1)
      << "scoring_func must be 0 (sigmoid) or 1 (softmax).";

  using namespace monomoe;
  ffi::CUDADeviceGuard device_guard(activations_in.device().device_id);
  const auto* activations_in_ptr = static_cast<const A_element*>(activations_in.data_ptr());
  const auto* router_logits_ptr = static_cast<const __nv_bfloat16*>(router_logits.data_ptr());
  const auto* expert_weights_up_ptr = static_cast<const W_element*>(expert_weights_up.data_ptr());
  const auto* expert_scales_up_ptr = static_cast<const S_element*>(expert_scales_up.data_ptr());
  const auto* expert_weights_down_ptr =
      static_cast<const W_element*>(expert_weights_down.data_ptr());
  const auto* expert_scales_down_ptr = static_cast<const S_element*>(expert_scales_down.data_ptr());
  auto* activations_out_ptr = static_cast<R_element*>(activations_out.data_ptr());
  char* scratchpad_ptr = reinterpret_cast<char*>(scratchpad.data_ptr());

  // Optional per-expert selection bias (float32 [NUM_EXPERTS]); nullptr =>
  // raw-logit ranking.
  const float* expert_bias_ptr = nullptr;
  if (expert_bias.has_value()) {
    CHECK_INPUT(expert_bias.value());
    CHECK_INPUT_TYPE(expert_bias.value(), dl_float32);
    TVM_FFI_ICHECK(expert_bias.value().ndim() == 1 &&
                   expert_bias.value().size(0) == Dims::NUM_EXPERTS)
        << "expert_bias must be 1-D with NUM_EXPERTS (=" << Dims::NUM_EXPERTS
        << ") elements; the routing kernel reads all NUM_EXPERTS entries.";
    expert_bias_ptr = static_cast<const float*>(expert_bias.value().data_ptr());
  }

  const uint32_t num_tokens = activations_in.size(0);
  const size_t shmem_size = get_moe_shmem_size<Dims>();
  const size_t scratchpad_size =
      static_cast<size_t>(scratchpad.numel()) * get_element_size(scratchpad);
  const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);
  const float routed_scaling_factor_f = static_cast<float>(routed_scaling_factor);

  // TMA descriptors (docs/design_docs/monomoe_kernel.md §5).  Non-TMA variants leave these
  // zero-initialized; the kernel params are always present but only the
  // TMA path consumes them.  Per-tensor caller contracts (pre-interleave
  // up-weights, raw down-weights, swizzle modes) live in the factory doc
  // comments in moe_tma.h.
  CUtensorMap up_weights_desc{};
  CUtensorMap activations_desc{};
  CUtensorMap down_weights_desc{};
  CUtensorMap down_activations_desc{};
  if constexpr (use_tma<Dims>::value) {
    up_weights_desc = create_up_weight_tma_desc(
        reinterpret_cast<const void*>(expert_weights_up_ptr), Dims::NUM_EXPERTS, Dims::N, Dims::K);
    // Row extent = the REAL token count, not Dims::BS: the TMA engine
    // bounds-checks against globalDim and zero-fills out-of-bounds box rows,
    // so for M < BS the routing-window load reads exactly M rows and
    // hardware-zeros the phantom rows.  Under CUDA graphs the descriptor is
    // captured per graph (one graph per batch size), so the baked-in M
    // always matches replays.
    activations_desc = create_activations_tma_desc(
        reinterpret_cast<const void*>(activations_in_ptr), num_tokens, Dims::HIDDEN_STATES);
    // row_box pinned to 128: DOWN_COL_TILE can exceed the 256-row TMA boxDim
    // cap (e.g. 384), so the kernel issues one 128-row TMA per M-atom.
    down_weights_desc =
        create_down_weight_tma_desc(reinterpret_cast<const void*>(expert_weights_down_ptr),
                                    Dims::NUM_EXPERTS, Dims::HIDDEN_STATES, Dims::N,
                                    /*row_box=*/128u);
    // Down-activation descriptor reads `spec->temp_fp8` inside the
    // scratchpad: base + compile-time offset (docs/design_docs/monomoe_kernel.md §4).
    const void* temp_fp8_ptr =
        reinterpret_cast<const char*>(scratchpad_ptr) + MoEGemmSpec<Dims>::TEMP_FP8_OFFSET;
    down_activations_desc =
        create_down_activation_tma_desc(temp_fp8_ptr, MoEGemmSpec<Dims>::TEMP_ROWS_TMA, Dims::N);
  }

  const cudaStream_t stream = get_stream(activations_in.device());
  CUDA_CHECK(cudaFuncSetAttribute(moe_kernel_topk<Dims>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
  // Co-residency invariant (design doc §3): the sentinel/flag handoffs spin
  // on values other blocks publish, so every block must be scheduled for the
  // kernel's full lifetime — GRID_SIZE <= SM count.
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                         activations_in.device().device_id);
  TVM_FFI_ICHECK(Dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count))
      << "monomoe requires GRID_SIZE (=" << Dims::KernelConfig::GRID_SIZE
      << ") <= SM count (=" << sm_count << ") for the co-residency invariant.";
  // First-use scratchpad zero-init (design doc §4): establishes the 0.0f
  // scale sentinel, launch_flip = 0, and down_ready = 0.  All are
  // self-maintaining afterwards (parity double-buffer refills).  Keyed on
  // buffer identity so each distinct scratchpad is initialized exactly once.
  {
    static const void* _zeroed_ptr = nullptr;
    static size_t _zeroed_size = 0;
    static int _zeroed_dev = -1;
    const int _cur_dev = activations_in.device().device_id;
    if (_zeroed_ptr != static_cast<const void*>(scratchpad_ptr) ||
        _zeroed_size != scratchpad_size || _zeroed_dev != _cur_dev) {
      CUDA_CHECK(cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));
      _zeroed_ptr = scratchpad_ptr;
      _zeroed_size = scratchpad_size;
      _zeroed_dev = _cur_dev;
    }
  }

  // Standard launch: cross-block ordering comes from the in-kernel
  // sentinel/flag handoffs (design doc §2), which is what lets the kernel be
  // captured into a CUDA Graph.
  moe_kernel_topk<Dims><<<dim3(Dims::KernelConfig::GRID_SIZE, 1, 1),
                          dim3(Dims::KernelConfig::BLOCK_SIZE, 1, 1), shmem_size, stream>>>(
      activations_in_ptr, num_tokens, router_logits_ptr, expert_weights_up_ptr,
      expert_scales_up_ptr, expert_weights_down_ptr, expert_scales_down_ptr, activations_out_ptr,
      scratchpad_ptr, scratchpad_size, shmem_size, top_k_u32, sf, renormalize, expert_bias_ptr,
      routed_scaling_factor_f, up_weights_desc, activations_desc, down_weights_desc,
      down_activations_desc);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiation of the launcher for the single hard-specialized
// shape (E=256, N=512, K=2048, block-wise FP8, BS8).  This is byte-identical
// to the pre-tuning shipped kernel (formerly config 0).
template void monomoe_topk_launcher<::monomoe::Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA>(
    TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
    TensorView expert_scales_up, TensorView expert_weights_down, TensorView expert_scales_down,
    TensorView activations_out, TensorView scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize, ffi::Optional<TensorView> expert_bias, double routed_scaling_factor);
