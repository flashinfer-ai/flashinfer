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
#include "attention_updater.cuh"

namespace flashinfer {

struct StandardAttention {
  template <int NUM_ROWS_PER_THREAD>
  using Updater = OnlineSoftmaxWithScale<NUM_ROWS_PER_THREAD>;

  template <typename MainloopParams, typename BlockCoord>
  __device__ StandardAttention(const MainloopParams& params, const BlockCoord& block_coord) {}

  template <typename MainloopParams, typename T>
  __device__ __forceinline__ T LogitsTransform(const MainloopParams& params, T logits,
                                               uint32_t batch_idx, uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }
};

struct LogitsSoftCap {
  float pre_tanh_scale;
  float post_tanh_scale;

  template <int NUM_ROWS_PER_THREAD>
  using Updater = OnlineSoftmaxWithoutScale<NUM_ROWS_PER_THREAD>;

  template <typename MainloopParams, typename BlockCoord>
  __device__ LogitsSoftCap(const MainloopParams& params, const BlockCoord& block_coord) {
    pre_tanh_scale =
        params.additional_params.sm_scale * math::ptx_rcp(params.additional_params.logits_soft_cap);
    post_tanh_scale = math::log2e * params.additional_params.logits_soft_cap;
  }

  template <typename MainloopParams, typename T>
  __device__ __forceinline__ T LogitsTransform(const MainloopParams& params, T logits,
                                               uint32_t batch_idx, uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return math::tanh(logits * pre_tanh_scale) * post_tanh_scale;
  }
};

template <bool use_logits_soft_cap>
using DefaultAttention = std::conditional_t<use_logits_soft_cap, LogitsSoftCap, StandardAttention>;

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_VARIANTS_CUH_
