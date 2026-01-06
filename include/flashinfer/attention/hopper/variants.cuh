/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// NOTE(Zihao): we should merge this with include/flashinfer/attention/variants.cuh in the future
#ifndef FLASHINFER_ATTENTION_HOPPER_VARIANTS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_VARIANTS_CUH_
#include <cuda_runtime.h>

#include "../../math.cuh"
#include "../../utils.cuh"
#include "attention_updater.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

// SFINAE to detect maybe_scale_v and scale_v_scalar members
DEFINE_HAS_MEMBER(maybe_scale_v)
DEFINE_HAS_MEMBER(scale_v_scalar)
DEFINE_HAS_MEMBER(maybe_scale_q)
DEFINE_HAS_MEMBER(scale_q_scalar)
DEFINE_HAS_MEMBER(maybe_scale_k)
DEFINE_HAS_MEMBER(scale_k_scalar)
DEFINE_HAS_MEMBER(scale_pv)

// Helper to get scale value from tensor pointer or scalar fallback
template <typename T>
__device__ __forceinline__ float get_scale(const T* tensor_ptr, float scalar_val, uint32_t idx) {
  return tensor_ptr != nullptr ? static_cast<float>(tensor_ptr[idx]) : scalar_val;
}

// Helper to get v_scale from additional_params (returns 1.0 if fields don't exist)
template <typename AdditionalParams>
__device__ __forceinline__ float get_v_scale(const AdditionalParams& params, uint32_t kv_head_idx) {
  if constexpr (has_maybe_scale_v_v<AdditionalParams> && has_scale_v_scalar_v<AdditionalParams>) {
    return get_scale(params.maybe_scale_v, params.scale_v_scalar, kv_head_idx);
  } else {
    return 1.0f;
  }
}

// Helper to get q_scale from additional_params (returns 1.0 if fields don't exist)
template <typename AdditionalParams>
__device__ __forceinline__ float get_q_scale(const AdditionalParams& params, uint32_t qo_head_idx) {
  if constexpr (has_maybe_scale_q_v<AdditionalParams> && has_scale_q_scalar_v<AdditionalParams>) {
    return get_scale(params.maybe_scale_q, params.scale_q_scalar, qo_head_idx);
  } else {
    return 1.0f;
  }
}

// Helper to get k_scale from additional_params (returns 1.0 if fields don't exist)
template <typename AdditionalParams>
__device__ __forceinline__ float get_k_scale(const AdditionalParams& params, uint32_t kv_head_idx) {
  if constexpr (has_maybe_scale_k_v<AdditionalParams> && has_scale_k_scalar_v<AdditionalParams>) {
    return get_scale(params.maybe_scale_k, params.scale_k_scalar, kv_head_idx);
  } else {
    return 1.0f;
  }
}

// Helper to get scale_pv from attention variant (returns 1.0 if field doesn't exist)
template <typename AttentionVariant>
__device__ __forceinline__ float get_variant_scale_pv(const AttentionVariant& variant) {
  if constexpr (has_scale_pv_v<AttentionVariant>) {
    return variant.scale_pv;
  } else {
    return 1.0f;
  }
}

struct StandardAttention {
  float sm_scale_log2;
  float scale_pv;  // v_scale for non-FP8

  template <typename MainloopParams, typename BlockCoord>
  __device__ StandardAttention(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    scale_pv = get_v_scale(params.additional_params, kv_head_idx);
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_log2);
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return logits; })
};

struct LogitsSoftCap {
  float pre_tanh_scale;
  float post_tanh_scale;
  float scale_pv;  // v_scale for non-FP8

  template <typename MainloopParams, typename BlockCoord>
  __device__ LogitsSoftCap(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    pre_tanh_scale =
        params.additional_params.sm_scale * math::ptx_rcp(params.additional_params.logits_soft_cap);
    post_tanh_scale = math::log2e * params.additional_params.logits_soft_cap;
    scale_pv = get_v_scale(params.additional_params, kv_head_idx);
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(post_tanh_scale);
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return math::tanh(logits * pre_tanh_scale); })
};

struct StandardFP8Attention {
  float p_scale, scale_pv, sm_scale_with_qk_log2;

  template <typename MainloopParams, typename BlockCoord>
  __device__ StandardFP8Attention(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;
    // 448 for e4m3; 57344 for e5m2
    p_scale = std::numeric_limits<typename MainloopParams::DTypeKV>::max();
    float v_scale = get_v_scale(params.additional_params, kv_head_idx);
    scale_pv = v_scale / p_scale;
    float q_scale = get_q_scale(params.additional_params, qo_head_idx);
    float k_scale = get_k_scale(params.additional_params, kv_head_idx);
    sm_scale_with_qk_log2 = q_scale * k_scale * params.additional_params.sm_scale * math::log2e;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE=*/true>(sm_scale_with_qk_log2);
  }

  template <typename Tensor0>
  __device__ __forceinline__ void PQuantize(Tensor0& tSrS) {
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      tSrS(i) *= p_scale;
    }
  }

  template <typename MainloopParams, typename Tensor0>
  __device__ __forceinline__ void ODequantize(const MainloopParams& params, Tensor0& tOrO,
                                              uint32_t qo_head_idx, uint32_t kv_head_idx) {
    // we fuse the PV dequantization into online_softmax.finalize
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return logits; })
};

template <bool use_logits_soft_cap>
using DefaultAttention = std::conditional_t<use_logits_soft_cap, LogitsSoftCap, StandardAttention>;
using DefaultFP8Attention = StandardFP8Attention;

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_VARIANTS_CUH_
