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
// constant must equal `offsetof(MoEGemmSpec<Dims>, temp_fp8)`.  New fields
// MUST be appended after temp_fp8 or this fires (docs/design_docs/monomoe_kernel.md §4).
static_assert(offsetof(monomoe::MoEGemmSpec<monomoe::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>,
                       temp_fp8) ==
                  monomoe::MoEGemmSpec<
                      monomoe::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>::TEMP_FP8_OFFSET,
              "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
              "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA. Do not insert fields "
              "before temp_fp8; grid_barrier / partial_barrier belong at the tail of "
              "MoEGemmSpec<Dims>.");

/**
 * @brief Host launcher for `moe_kernel_topk`, parameterized on the compile-time
 * @p Dims shape variant with runtime top_k / scoring_func / renormalize.
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
                           int64_t scoring_func, bool renormalize) {
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

  const uint32_t num_tokens = activations_in.size(0);
  const size_t shmem_size = get_moe_shmem_size<Dims>();
  const size_t scratchpad_size =
      static_cast<size_t>(scratchpad.numel()) * get_element_size(scratchpad);
  const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);

  // TMA descriptors (docs/design_docs/monomoe_kernel.md §5).  Non-TMA variants leave these
  // zero-initialized; the kernel params are always present but only the
  // TMA path consumes them.  Per-tensor caller contracts (pre-interleave
  // up-weights, raw down-weights, swizzle modes) live in the factory doc
  // comments in moe_tma.h.
  CUtensorMap up_weights_desc{};
  CUtensorMap activations_desc{};
  CUtensorMap down_weights_desc{};
  CUtensorMap down_activations_desc{};
  if constexpr (use_tma_v<Dims>) {
    up_weights_desc = create_up_weight_tma_desc(
        reinterpret_cast<const void*>(expert_weights_up_ptr), Dims::NUM_EXPERTS, Dims::N, Dims::K);
    activations_desc = create_activations_tma_desc(
        reinterpret_cast<const void*>(activations_in_ptr), Dims::BS, Dims::HIDDEN_STATES);
    down_weights_desc = create_down_weight_tma_desc(
        reinterpret_cast<const void*>(expert_weights_down_ptr), Dims::NUM_EXPERTS,
        Dims::HIDDEN_STATES, Dims::N, /*row_box=*/MoECoreDims<Dims>::DOWN_COL_TILE);
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
  // Co-residency invariant (docs/design_docs/monomoe_kernel.md §3): the barrier spin only avoids
  // deadlock if GRID_SIZE <= SM count.  Single cheap query off the hot path.
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                         activations_in.device().device_id);
  TVM_FFI_ICHECK(Dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count))
      << "monomoe requires GRID_SIZE (=" << Dims::KernelConfig::GRID_SIZE
      << ") <= SM count (=" << sm_count << ") for software grid barrier co-residency invariant.";
  // First-use scratchpad zero-init (docs/design_docs/monomoe_kernel.md §4): the barrier counters
  // must start at 0; the ping-pong reset is self-maintaining thereafter.
  // Key the guard on buffer identity (ptr, size, device) — a process-wide
  // one-shot flag would leave a second distinct scratchpad with
  // uninitialized counters and deadlock the spin.
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

  moe_kernel_topk<Dims><<<dim3(Dims::KernelConfig::GRID_SIZE, 1, 1),
                          dim3(Dims::KernelConfig::BLOCK_SIZE, 1, 1), shmem_size, stream>>>(
      activations_in_ptr, num_tokens, router_logits_ptr, expert_weights_up_ptr,
      expert_scales_up_ptr, expert_weights_down_ptr, expert_scales_down_ptr, activations_out_ptr,
      scratchpad_ptr, scratchpad_size, shmem_size, top_k_u32, sf, renormalize, up_weights_desc,
      activations_desc, down_weights_desc, down_activations_desc);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiation for the only BS8 variant (TMA + WGMMA +
// SWIZZLE_128B); materializes the launcher symbol that monomoe_binding.cu
// exports through TVM-FFI.
template void monomoe_topk_launcher<monomoe::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>(
    TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
    TensorView expert_scales_up, TensorView expert_weights_down, TensorView expert_scales_down,
    TensorView activations_out, TensorView scratchpad, int64_t top_k, int64_t scoring_func,
    bool renormalize);
