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
#include "../utils.cuh"

namespace flashinfer {

// Query Transform function that multiplies the query matrix by sm_scale
struct StandardAttention {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

DEFINE_HAS_MEMBER(maybe_mask_indptr)

struct CustomMaskAttention {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ CustomMaskAttention(const Params& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    if constexpr (has_maybe_mask_indptr_v<Params>) {
      custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
    } else {
      custom_mask_ptr = params.maybe_custom_mask;
    }
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    const uint32_t offset = qo_idx * kv_len + kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  }
};

struct SlidingWindowAttention {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  // Create closure
  template <typename Params>
  __device__ __host__ __forceinline__ SlidingWindowAttention(const Params& params,
                                                             uint32_t batch_idx,
                                                             uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  }
};

struct LogitsSoftCap {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  template <typename Params>
  __device__ __host__ LogitsSoftCap(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return params.logits_soft_cap * math::log2e * float(math::tanh(logits));
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

struct ALIBIAttention {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;

  template <typename Params>
  __device__ __host__ ALIBIAttention(const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    return logits + params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                             uint32_t qo_idx, uint32_t kv_idx, uint32_t qo_head_idx,
                                             uint32_t kv_head_idx) {
    return true;
  }
};

template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
struct DefaultAttention {
  static constexpr bool use_softmax = true;

  uint32_t qo_len, kv_len;
  uint8_t* custom_mask_ptr;
  uint32_t window_left;

  // Create closure
  template <typename Params>
  __device__ __host__ DefaultAttention(const Params& params, uint32_t batch_idx,
                                       uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    if constexpr (use_custom_mask) {
      if constexpr (has_maybe_mask_indptr_v<Params>) {
        custom_mask_ptr = params.maybe_custom_mask + params.maybe_mask_indptr[batch_idx];
      } else {
        custom_mask_ptr = params.maybe_custom_mask;
      }
    }
    if constexpr (use_sliding_window) {
      window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    }
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T QueryTransform(const Params& params, T q) {
    if constexpr (use_logits_soft_cap) {
      return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
    } else {
      return float(q) * params.sm_scale * math::log2e;
    }
  }

  template <typename Params, typename T>
  __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = logits + params.maybe_alibi_slopes[qo_head_idx] * float(int(kv_idx) - int(qo_idx));
    }
    if constexpr (use_logits_soft_cap) {
      logits = params.logits_soft_cap * math::log2e * float(math::tanh(logits));
    }
    return logits;
  }

  template <typename Params>
  __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
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
