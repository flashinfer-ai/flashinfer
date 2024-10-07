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
#ifndef FLASHINFER_ATTENTION_VARIANTS_CUH_
#define FLASHINFER_ATTENTION_VARIANTS_CUH_
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "../math.cuh"

namespace flashinfer {

// Query Transform function that multiplies the query matrix by sm_scale
template <typename ParamsT_>
struct StandardAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;

  // Create closure
  __device__ __host__ StandardAttention(const ParamsT& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {}

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

template <typename ParamsT_>
struct CustomMaskAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;

  uint8_t* custom_mask_ptr;
  uint32_t qo_len, kv_len;

  // Create closure
  __device__ __host__ CustomMaskAttention(const ParamsT& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.get_batch_local_mask_ptr(batch_idx);
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    const uint32_t offset = qo_idx * kv_len + kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  }
};

template <typename ParamsT_>
struct SlidingWindowAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  __device__ __host__ __forceinline__ SlidingWindowAttention(const ParamsT& params,
                                                             uint32_t batch_idx,
                                                             uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  }
};

template <typename ParamsT>
struct LogitsSoftCap {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;

  __device__ __host__ LogitsSoftCap(const ParamsT& params, uint32_t batch_idx, uint8_t* smem_ptr) {}

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return params.logits_soft_cap * math::log2e * float(math::tanh(logits));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

template <typename ParamsT>
struct ALIBIAttention {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;

  __device__ __host__ ALIBIAttention(const ParamsT& params, uint32_t batch_idx, uint8_t* smem_ptr) {
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits + params.alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

constexpr uint32_t CUSTOM_MASK = 1U;
constexpr uint32_t SLIDING_WINDOW = 2U;
constexpr uint32_t LOGITS_SOFT_CAP = 4U;
constexpr uint32_t ALIBI = 8U;

constexpr uint32_t get_variant_code(bool use_custom_mask, bool use_sliding_window,
                                    bool use_logits_soft_cap, bool use_alibi) {
  return (use_custom_mask ? CUSTOM_MASK : 0U) | (use_sliding_window ? SLIDING_WINDOW : 0U) |
         (use_logits_soft_cap ? LOGITS_SOFT_CAP : 0U) | (use_alibi ? ALIBI : 0U);
}

template <typename ParamsT_, uint32_t VARIANT_CODE>
struct ComposedAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static constexpr bool use_custom_mask = (VARIANT_CODE & CUSTOM_MASK) != 0;
  static constexpr bool use_sliding_window = (VARIANT_CODE & SLIDING_WINDOW) != 0;
  static constexpr bool use_logits_soft_cap = (VARIANT_CODE & LOGITS_SOFT_CAP) != 0;
  static constexpr bool use_alibi = (VARIANT_CODE & ALIBI) != 0;

  uint32_t qo_len, kv_len;
  uint8_t* custom_mask_ptr;
  uint32_t window_left;

  // Create closure
  __device__ __host__ ComposedAttention(const ParamsT& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_custom_mask) {
      custom_mask_ptr = params.get_batch_local_mask_ptr(batch_idx);
    }
    if constexpr (use_sliding_window) {
      window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    }
  }

  template <typename T>
  __device__ __forceinline__ T QueryTransform(const ParamsT& params, T q) {
    if constexpr (use_logits_soft_cap) {
      return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
    } else {
      return float(q) * params.sm_scale * math::log2e;
    }
  }

  template <typename T>
  __device__ __forceinline__ T LogitsTransform(const ParamsT& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = logits + params.alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      logits = params.logits_soft_cap * math::log2e * float(math::tanh(logits));
    }
    return logits;
  }

  __device__ __forceinline__ bool LogitsMask(const ParamsT& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    bool mask = true;
    if constexpr (use_custom_mask) {
      const uint32_t offset = qo_idx * kv_len + kv_idx;
      mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
    }
    if constexpr (use_sliding_window) {
      mask &= (kv_idx + qo_len + window_left >= kv_len + qo_idx);
    }
    return mask;
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_VARIANTS_CUH_