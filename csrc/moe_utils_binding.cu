/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tvm_ffi_utils.h"

using namespace tensorrt_llm::kernels::cute_dsl;

namespace {
// Helper function to compute log2 of a value (returns -1 if not power of 2)
inline int32_t computeLog2(int32_t val) {
  int32_t n = val;
  int32_t out = 0;
  while (n >>= 1) {
    ++out;
  }
  if ((1 << out) != val) {
    out = -1;
  }
  return out;
}
}  // namespace

// ============================ moePermute bindings ============================

void moe_permute_fp16(int64_t input_ptr, int64_t permuted_output_ptr, int64_t input_sf_ptr,
                      int64_t permuted_sf_ptr, int64_t tile_idx_to_mn_limit_ptr,
                      int64_t permuted_idx_to_expanded_idx_ptr, int64_t num_non_exiting_tiles_ptr,
                      int32_t max_num_permuted_tokens, int32_t hidden_size, int32_t top_k,
                      int32_t tile_size, bool enable_pdl) {
  moePermute<half, uint8_t>(
      reinterpret_cast<half const*>(input_ptr), reinterpret_cast<half*>(permuted_output_ptr),
      reinterpret_cast<uint8_t const*>(input_sf_ptr), reinterpret_cast<uint8_t*>(permuted_sf_ptr),
      reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
      reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
      reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr), max_num_permuted_tokens,
      hidden_size, top_k, tile_size, enable_pdl, get_current_stream());
}

#ifdef ENABLE_BF16
void moe_permute_bf16(int64_t input_ptr, int64_t permuted_output_ptr, int64_t input_sf_ptr,
                      int64_t permuted_sf_ptr, int64_t tile_idx_to_mn_limit_ptr,
                      int64_t permuted_idx_to_expanded_idx_ptr, int64_t num_non_exiting_tiles_ptr,
                      int32_t max_num_permuted_tokens, int32_t hidden_size, int32_t top_k,
                      int32_t tile_size, bool enable_pdl) {
  moePermute<__nv_bfloat16, uint8_t>(
      reinterpret_cast<__nv_bfloat16 const*>(input_ptr),
      reinterpret_cast<__nv_bfloat16*>(permuted_output_ptr),
      reinterpret_cast<uint8_t const*>(input_sf_ptr), reinterpret_cast<uint8_t*>(permuted_sf_ptr),
      reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
      reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
      reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr), max_num_permuted_tokens,
      hidden_size, top_k, tile_size, enable_pdl, get_current_stream());
}
#endif

#ifdef ENABLE_FP8
void moe_permute_fp8(int64_t input_ptr, int64_t permuted_output_ptr, int64_t input_sf_ptr,
                     int64_t permuted_sf_ptr, int64_t tile_idx_to_mn_limit_ptr,
                     int64_t permuted_idx_to_expanded_idx_ptr, int64_t num_non_exiting_tiles_ptr,
                     int32_t max_num_permuted_tokens, int32_t hidden_size, int32_t top_k,
                     int32_t tile_size, bool enable_pdl) {
  moePermute<__nv_fp8_e4m3, uint8_t>(
      reinterpret_cast<__nv_fp8_e4m3 const*>(input_ptr),
      reinterpret_cast<__nv_fp8_e4m3*>(permuted_output_ptr),
      reinterpret_cast<uint8_t const*>(input_sf_ptr), reinterpret_cast<uint8_t*>(permuted_sf_ptr),
      reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
      reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
      reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr), max_num_permuted_tokens,
      hidden_size, top_k, tile_size, enable_pdl, get_current_stream());
}
#endif

#ifdef ENABLE_FP4
void moe_permute_fp4(int64_t input_ptr, int64_t permuted_output_ptr, int64_t input_sf_ptr,
                     int64_t permuted_sf_ptr, int64_t tile_idx_to_mn_limit_ptr,
                     int64_t permuted_idx_to_expanded_idx_ptr, int64_t num_non_exiting_tiles_ptr,
                     int32_t max_num_permuted_tokens, int32_t hidden_size, int32_t top_k,
                     int32_t tile_size, bool enable_pdl) {
  moePermute<__nv_fp4_e2m1, uint8_t>(
      reinterpret_cast<__nv_fp4_e2m1 const*>(input_ptr),
      reinterpret_cast<__nv_fp4_e2m1*>(permuted_output_ptr),
      reinterpret_cast<uint8_t const*>(input_sf_ptr), reinterpret_cast<uint8_t*>(permuted_sf_ptr),
      reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
      reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
      reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr), max_num_permuted_tokens,
      hidden_size, top_k, tile_size, enable_pdl, get_current_stream());
}
#endif

// ============================ moeUnpermute bindings ============================

void moe_unpermute_fp16_float_scale(int64_t permuted_input_ptr, int64_t output_ptr,
                                    int64_t expanded_idx_to_permuted_idx_ptr,
                                    int64_t topk_scales_ptr, int32_t num_tokens,
                                    int32_t hidden_size, int32_t top_k, bool enable_pdl) {
  moeUnpermute<half, float>(reinterpret_cast<half const*>(permuted_input_ptr),
                            reinterpret_cast<half*>(output_ptr),
                            reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
                            reinterpret_cast<float const*>(topk_scales_ptr), num_tokens,
                            hidden_size, top_k, enable_pdl, get_current_stream());
}

void moe_unpermute_fp16_half_scale(int64_t permuted_input_ptr, int64_t output_ptr,
                                   int64_t expanded_idx_to_permuted_idx_ptr,
                                   int64_t topk_scales_ptr, int32_t num_tokens, int32_t hidden_size,
                                   int32_t top_k, bool enable_pdl) {
  moeUnpermute<half, half>(reinterpret_cast<half const*>(permuted_input_ptr),
                           reinterpret_cast<half*>(output_ptr),
                           reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
                           reinterpret_cast<half const*>(topk_scales_ptr), num_tokens, hidden_size,
                           top_k, enable_pdl, get_current_stream());
}

#ifdef ENABLE_BF16
void moe_unpermute_bf16_float_scale(int64_t permuted_input_ptr, int64_t output_ptr,
                                    int64_t expanded_idx_to_permuted_idx_ptr,
                                    int64_t topk_scales_ptr, int32_t num_tokens,
                                    int32_t hidden_size, int32_t top_k, bool enable_pdl) {
  moeUnpermute<__nv_bfloat16, float>(
      reinterpret_cast<__nv_bfloat16 const*>(permuted_input_ptr),
      reinterpret_cast<__nv_bfloat16*>(output_ptr),
      reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
      reinterpret_cast<float const*>(topk_scales_ptr), num_tokens, hidden_size, top_k, enable_pdl,
      get_current_stream());
}

void moe_unpermute_bf16_bf16_scale(int64_t permuted_input_ptr, int64_t output_ptr,
                                   int64_t expanded_idx_to_permuted_idx_ptr,
                                   int64_t topk_scales_ptr, int32_t num_tokens, int32_t hidden_size,
                                   int32_t top_k, bool enable_pdl) {
  moeUnpermute<__nv_bfloat16, __nv_bfloat16>(
      reinterpret_cast<__nv_bfloat16 const*>(permuted_input_ptr),
      reinterpret_cast<__nv_bfloat16*>(output_ptr),
      reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
      reinterpret_cast<__nv_bfloat16 const*>(topk_scales_ptr), num_tokens, hidden_size, top_k,
      enable_pdl, get_current_stream());
}
#endif

// ============================ moeOutputMemset bindings ============================

void moe_output_memset_fp16(int64_t input_ptr, int64_t tile_idx_to_mn_limit_ptr,
                            int64_t expanded_idx_to_permuted_idx_ptr,
                            int64_t permuted_idx_to_expanded_idx_ptr,
                            int64_t num_non_exiting_tiles_ptr, int32_t max_num_permuted_tokens,
                            int32_t hidden_size, int32_t top_k, int32_t tile_size,
                            bool enable_pdl) {
  moeOutputMemset<half>(reinterpret_cast<half*>(input_ptr),
                        reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
                        reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
                        reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
                        reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr),
                        max_num_permuted_tokens, hidden_size, top_k, tile_size, enable_pdl,
                        get_current_stream());
}

#ifdef ENABLE_BF16
void moe_output_memset_bf16(int64_t input_ptr, int64_t tile_idx_to_mn_limit_ptr,
                            int64_t expanded_idx_to_permuted_idx_ptr,
                            int64_t permuted_idx_to_expanded_idx_ptr,
                            int64_t num_non_exiting_tiles_ptr, int32_t max_num_permuted_tokens,
                            int32_t hidden_size, int32_t top_k, int32_t tile_size,
                            bool enable_pdl) {
  moeOutputMemset<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(input_ptr),
                                 reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
                                 reinterpret_cast<int32_t const*>(expanded_idx_to_permuted_idx_ptr),
                                 reinterpret_cast<int32_t const*>(permuted_idx_to_expanded_idx_ptr),
                                 reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr),
                                 max_num_permuted_tokens, hidden_size, top_k, tile_size, enable_pdl,
                                 get_current_stream());
}
#endif

// ============================ moeActivation bindings ============================

void moe_activation_fp16(int64_t input_ptr, int64_t output_ptr, int64_t tile_idx_to_mn_limit_ptr,
                         int64_t num_non_exiting_tiles_ptr, int32_t activation_type,
                         int32_t max_num_permuted_tokens, int32_t interm_size, int32_t tile_size,
                         bool enable_pdl) {
  moeActivation<half>(reinterpret_cast<half const*>(input_ptr), reinterpret_cast<half*>(output_ptr),
                      reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
                      reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr),
                      static_cast<MoeActivationType>(activation_type), max_num_permuted_tokens,
                      interm_size, tile_size, enable_pdl, get_current_stream());
}

#ifdef ENABLE_BF16
void moe_activation_bf16(int64_t input_ptr, int64_t output_ptr, int64_t tile_idx_to_mn_limit_ptr,
                         int64_t num_non_exiting_tiles_ptr, int32_t activation_type,
                         int32_t max_num_permuted_tokens, int32_t interm_size, int32_t tile_size,
                         bool enable_pdl) {
  moeActivation<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 const*>(input_ptr),
                               reinterpret_cast<__nv_bfloat16*>(output_ptr),
                               reinterpret_cast<int32_t const*>(tile_idx_to_mn_limit_ptr),
                               reinterpret_cast<int32_t const*>(num_non_exiting_tiles_ptr),
                               static_cast<MoeActivationType>(activation_type),
                               max_num_permuted_tokens, interm_size, tile_size, enable_pdl,
                               get_current_stream());
}
#endif

// ============================ TVM FFI Registration ============================

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_permute_fp16, moe_permute_fp16);
#ifdef ENABLE_BF16
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_permute_bf16, moe_permute_bf16);
#endif
#ifdef ENABLE_FP8
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_permute_fp8, moe_permute_fp8);
#endif
#ifdef ENABLE_FP4
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_permute_fp4, moe_permute_fp4);
#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_unpermute_fp16_float_scale,
                              moe_unpermute_fp16_float_scale);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_unpermute_fp16_half_scale,
                              moe_unpermute_fp16_half_scale);
#ifdef ENABLE_BF16
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_unpermute_bf16_float_scale,
                              moe_unpermute_bf16_float_scale);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_unpermute_bf16_bf16_scale,
                              moe_unpermute_bf16_bf16_scale);
#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_output_memset_fp16, moe_output_memset_fp16);
#ifdef ENABLE_BF16
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_output_memset_bf16, moe_output_memset_bf16);
#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_activation_fp16, moe_activation_fp16);
#ifdef ENABLE_BF16
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_activation_bf16, moe_activation_bf16);
#endif

// ============================ moeSort bindings ============================
// moe_sort - Sort tokens by expert assignment and generate mapping tensors
// This uses DeepSeekV3 routing method with pre-computed expert selections
//
// Returns via output pointers:
// - tile_idx_to_expert_idx: [max_num_tiles], mapping from tile to local expert index
// - tile_idx_to_mn_limit: [max_num_tiles], M/N limit for each tile
// - expanded_idx_to_permuted_idx: [num_tokens, top_k], mapping from expanded to permuted index
// - permuted_idx_to_expanded_idx: [max_num_permuted_tokens], mapping from permuted to expanded
// - total_num_padded_tokens: [1], total number of padded tokens
// - num_non_exiting_tiles: [1], number of non-exiting tiles

void moe_sort(
    // Inputs
    int64_t token_selected_experts_ptr,  // [num_tokens, top_k], int32
    int64_t token_final_scales_ptr,      // [num_tokens, top_k], float32 or bf16
    int32_t num_tokens, int32_t num_experts, int32_t top_k, int32_t local_expert_offset,
    int32_t num_local_experts, int32_t tile_tokens_dim, bool use_pdl,
    // Outputs (pre-allocated buffers)
    int64_t tile_idx_to_expert_idx_ptr, int64_t tile_idx_to_mn_limit_ptr,
    int64_t expanded_idx_to_permuted_idx_ptr, int64_t permuted_idx_to_expanded_idx_ptr,
    int64_t total_num_padded_tokens_ptr, int64_t num_non_exiting_tiles_ptr,
    // Optional: expert counts buffer for large token counts (>1024)
    // Should be size 2 * num_experts, int32
    int64_t expert_counts_ptr,
    // Optional: explicit CUDA stream pointer for CUDA graph compatibility
    // If 0, uses TVM FFI's current stream
    int64_t cuda_stream_ptr) {
  // Set up the routing data structure
  moe::dev::routing::routingDeepSeek::Data routingData;

  // Configure dtypes
  routingData.mDtypeExpW = batchedGemm::trtllm::gen::Dtype::Bfloat16;
  routingData.mDtypeBias = batchedGemm::trtllm::gen::Dtype::Bfloat16;
  routingData.mDtypeScore = batchedGemm::trtllm::gen::Dtype::Fp32;
  routingData.mUsePdl = use_pdl;

  // Input tensors (pre-computed expert selections)
  routingData.mPtrTopKIds = reinterpret_cast<int32_t*>(token_selected_experts_ptr);
  routingData.mPtrTopKWeights = reinterpret_cast<void*>(token_final_scales_ptr);
  routingData.mPtrScores = nullptr;       // Not using routing logits
  routingData.mPtrRoutingBias = nullptr;  // Not using bias

  // Output tensors
  routingData.mPtrCtaIdxXyToBatchIdx = reinterpret_cast<int32_t*>(tile_idx_to_expert_idx_ptr);
  routingData.mPtrCtaIdxXyToMnLimit = reinterpret_cast<int32_t*>(tile_idx_to_mn_limit_ptr);
  routingData.mPtrExpandedIdxToPermutedIdx =
      reinterpret_cast<int32_t*>(expanded_idx_to_permuted_idx_ptr);
  routingData.mPtrPermutedIdxToTokenIdx =
      reinterpret_cast<int32_t*>(permuted_idx_to_expanded_idx_ptr);
  routingData.mPtrPermutedIdxSize = reinterpret_cast<int32_t*>(total_num_padded_tokens_ptr);
  routingData.mPtrNumNonExitingCtas = reinterpret_cast<int32_t*>(num_non_exiting_tiles_ptr);

  // Not using packed format since we have explicit TopK IDs
  routingData.mPtrTopKPacked = nullptr;

  // Expert counts buffer: required when num_tokens > 1024
  // The kernel will set this to nullptr internally for small token counts
  routingData.mPtrExpertCounts = reinterpret_cast<int32_t*>(expert_counts_ptr);

  // Metadata
  routingData.mNumTokens = num_tokens;
  routingData.mNumExperts = num_experts;
  routingData.mTopK = top_k;
  routingData.mPaddingLog2 = computeLog2(tile_tokens_dim);
  routingData.mTileTokensDim = tile_tokens_dim;
  routingData.mLocalExpertsStartIdx = local_expert_offset;
  routingData.mLocalExpertsStrideLog2 = 0;
  routingData.mNumLocalExperts = num_local_experts;

  // DeepSeekV3 specific parameters
  // For moe_sort, we use n_group=1, topk_group=1 since experts are already selected
  routingData.mNumExpertGroups = 1;
  routingData.mNumLimitedGroups = 1;
  routingData.mRouteScale = 1.0f;
  routingData.mUseRoutingSoftmax = false;

  // Run the routing kernel
  // Use explicit stream if provided (for CUDA graph compatibility), otherwise fall back to TVM FFI
  // stream
  cudaStream_t stream =
      cuda_stream_ptr != 0 ? reinterpret_cast<cudaStream_t>(cuda_stream_ptr) : get_current_stream();
  moe::dev::routing::routingDeepSeek::run(routingData, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_moe_sort, moe_sort);
