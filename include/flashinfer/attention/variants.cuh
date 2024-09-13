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
  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    return float(q) * params.sm_scale * math::log2e;
  }

  template <typename T>
  static __device__ __forceinline__ T DecodeLogitsTransform(const ParamsT& params, T logits,
                                                            int32_t qo_idx, int32_t kv_idx,
                                                            int32_t qo_head_idx,
                                                            int32_t kv_head_idx) {
    return logits;
  }

  template <typename T>
  static __device__ __forceinline__ T PrefillLogitsTransform(const ParamsT& params, T logits,
                                                             int32_t qo_idx, int32_t kv_idx,
                                                             int32_t qo_head_idx,
                                                             int32_t kv_head_idx) {
    return logits;
  }

  template <typename T>
  static __device__ __forceinline__ T BatchDecodeLogitsTransform(const ParamsT& params, T logits,
                                                                 int32_t batch_idx, int32_t qo_idx,
                                                                 int32_t kv_idx,
                                                                 int32_t qo_head_idx,
                                                                 int32_t kv_head_idx) {
    return logits;
  }

  template <typename T>
  static __device__ __forceinline__ T BatchPrefillLogitsTransform(const ParamsT& params, T logits,
                                                                  int32_t batch_idx, int32_t qo_idx,
                                                                  int32_t kv_idx,
                                                                  int32_t qo_head_idx,
                                                                  int32_t kv_head_idx) {
    return logits;
  }
};

template <typename ParamsT_>
struct CustomMaskAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    return StandardAttention<ParamsT>::QueryTransform(params, q);
  }

  template <typename T>
  static __device__ __forceinline__ T PrefillLogitsTransform(const ParamsT& params, T logits,
                                                             int32_t qo_idx, int32_t kv_idx,
                                                             int32_t qo_head_idx,
                                                             int32_t kv_head_idx) {
    uint8_t* custom_mask = params.custom_mask;
    const uint32_t kv_len = params.kv_len;
    return ((custom_mask[(qo_idx * kv_len + kv_idx) / 8] >> ((qo_idx * kv_len + kv_idx) % 8)) & 1)
               ? logits
               : -T(math::inf);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchPrefillLogitsTransform(const ParamsT& params, T logits,
                                                                  int32_t batch_idx, int32_t qo_idx,
                                                                  int32_t kv_idx,
                                                                  int32_t qo_head_idx,
                                                                  int32_t kv_head_idx) {
    uint8_t* custom_mask = params.custom_mask + params.mask_indptr[batch_idx];
    const uint32_t kv_len = params.kv_indptr[batch_idx + 1] - params.kv_indptr[batch_idx];
    return ((custom_mask[(qo_idx * kv_len + kv_idx) / 8] >> ((qo_idx * kv_len + kv_idx) % 8)) & 1)
               ? logits
               : -T(math::inf);
  }
};

template <typename ParamsT_>
struct SlidingWindowAttention {
  using ParamsT = ParamsT_;
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    return StandardAttention<ParamsT>::QueryTransform(params, q);
  }

  template <typename T>
  static __device__ __forceinline__ T DecodeLogitsTransform(const ParamsT& params, T logits,
                                                            int32_t qo_idx, int32_t kv_idx,
                                                            int32_t qo_head_idx,
                                                            int32_t kv_head_idx) {
    const int32_t qo_len = 1;
    const int32_t kv_len = params.info.kv_len;
    return (kv_idx + qo_len + params.window_left < kv_len + qo_idx) ? logits : -T(math::inf);
  }

  template <typename T>
  static __device__ __forceinline__ T PrefillLogitsTransform(const ParamsT& params, T logits,
                                                             int32_t qo_idx, int32_t kv_idx,
                                                             int32_t qo_head_idx,
                                                             int32_t kv_head_idx) {
    const int32_t qo_len = params.qo_len;
    const int32_t kv_len = params.kv_len;
    return (kv_idx + qo_len + params.window_left < kv_len + qo_idx) ? logits : -T(math::inf);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchDecodeLogitsTransform(const ParamsT& params, T logits,
                                                                 int32_t batch_idx, int32_t qo_idx,
                                                                 int32_t kv_idx,
                                                                 int32_t qo_head_idx,
                                                                 int32_t kv_head_idx) {
    const int32_t qo_len = 1;
    const int32_t kv_len = params.kv_indptr[batch_idx + 1] - params.kv_indptr[batch_idx];
    return (kv_idx + qo_len + params.window_left < kv_len + qo_idx) ? logits : -T(math::inf);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchPrefillLogitsTransform(const ParamsT& params, T logits,
                                                                  int32_t batch_idx, int32_t qo_idx,
                                                                  int32_t kv_idx,
                                                                  int32_t qo_head_idx,
                                                                  int32_t kv_head_idx) {
    const int32_t qo_len = params.qo_indptr[batch_idx + 1] - params.qo_indptr[batch_idx];
    const int32_t kv_len = params.kv_indptr[batch_idx + 1] - params.kv_indptr[batch_idx];
    return (kv_idx + qo_len + params.window_left < kv_len + qo_idx) ? logits : -T(math::inf);
  }
};

template <typename ParamsT>
struct LogitsSoftCap {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    return float(q) * params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
  }

  template <typename T>
  static __device__ __forceinline__ T DecodeLogitsTransform(const ParamsT& params, T logits,
                                                            int32_t qo_idx, int32_t kv_idx,
                                                            int32_t qo_head_idx,
                                                            int32_t kv_head_idx) {
    return params.logits_soft_cap * math::log2e * math::tanh(logits);
  }

  template <typename T>
  static __device__ __forceinline__ T PrefillLogitsTransform(const ParamsT& params, T logits,
                                                             int32_t qo_idx, int32_t kv_idx,
                                                             int32_t qo_head_idx,
                                                             int32_t kv_head_idx) {
    return params.logits_soft_cap * math::log2e * math::tanh(logits);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchDecodeLogitsTransform(const ParamsT& params, T logits,
                                                                 int32_t batch_idx, int32_t qo_idx,
                                                                 int32_t kv_idx,
                                                                 int32_t qo_head_idx,
                                                                 int32_t kv_head_idx) {
    return DecodeLogitsTransform(params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchPrefillLogitsTransform(const ParamsT& params, T logits,
                                                                  int32_t batch_idx, int32_t qo_idx,
                                                                  int32_t kv_idx,
                                                                  int32_t qo_head_idx,
                                                                  int32_t kv_head_idx) {
    return PrefillLogitsTransform(params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
  }
};

template <typename ParamsT>
struct ALIBIAttention {
  using DTypeQ = typename ParamsT::DTypeQ;
  using DTypeKV = typename ParamsT::DTypeKV;
  using DTypeO = typename ParamsT::DTypeO;
  using IdType = typename ParamsT::IdType;
  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    return StandardAttention<ParamsT>::QueryTransform(params, q);
  }

  template <typename T>
  static __device__ __forceinline__ T DecodeLogitsTransform(const ParamsT& params, T logits,
                                                            int32_t qo_idx, int32_t kv_idx,
                                                            int32_t qo_head_idx,
                                                            int32_t kv_head_idx) {
    return logits + params.alibi_slopes[qo_head_idx] * float(kv_idx - qo_idx);
  }

  template <typename T>
  static __device__ __forceinline__ T PrefillLogitsTransform(const ParamsT& params, T logits,
                                                             int32_t qo_idx, int32_t kv_idx,
                                                             int32_t qo_head_idx,
                                                             int32_t kv_head_idx) {
    return logits + params.alibi_slopes[qo_head_idx] * float(kv_idx - qo_idx);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchDecodeLogitsTransform(const ParamsT& params, T logits,
                                                                 int32_t batch_idx, int32_t qo_idx,
                                                                 int32_t kv_idx,
                                                                 int32_t qo_head_idx,
                                                                 int32_t kv_head_idx) {
    return DecodeLogitsTransform(params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
  }

  template <typename T>
  static __device__ __forceinline__ T BatchPrefillLogitsTransform(const ParamsT& params, T logits,
                                                                  int32_t batch_idx, int32_t qo_idx,
                                                                  int32_t kv_idx,
                                                                  int32_t qo_head_idx,
                                                                  int32_t kv_head_idx) {
    return PrefillLogitsTransform(params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
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

  static __device__ __forceinline__ DTypeQ QueryTransform(const ParamsT& params, DTypeQ q) {
    if constexpr (use_logits_soft_cap) {
      return LogitsSoftCap<ParamsT>::QueryTransform(params, q);
    } else {
      return StandardAttention<ParamsT>::QueryTransform(params, q);
    }
  }

  static __device__ __forceinline__ float DecodeLogitsTransform(const ParamsT& params, float logits,
                                                                int32_t qo_idx, int32_t kv_idx,
                                                                int32_t qo_head_idx,
                                                                int32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = ALIBIAttention<ParamsT>::DecodeLogitsTransform(params, logits, qo_idx, kv_idx,
                                                              qo_head_idx, kv_head_idx);
    }
    if constexpr (use_sliding_window) {
      logits = SlidingWindowAttention<ParamsT>::DecodeLogitsTransform(
          params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    } else {
      logits = StandardAttention<ParamsT>::DecodeLogitsTransform(params, logits, qo_idx, kv_idx,
                                                                 qo_head_idx, kv_head_idx);
    }
    return logits;
  }

  static __device__ __forceinline__ float PrefillLogitsTransform(const ParamsT& params,
                                                                 float logits, int32_t qo_idx,
                                                                 int32_t kv_idx,
                                                                 int32_t qo_head_idx,
                                                                 int32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = ALIBIAttention<ParamsT>::PrefillLogitsTransform(params, logits, qo_idx, kv_idx,
                                                               qo_head_idx, kv_head_idx);
    }
    if constexpr (use_custom_mask) {
      logits = CustomMaskAttention<ParamsT>::PrefillLogitsTransform(params, logits, qo_idx, kv_idx,
                                                                    qo_head_idx, kv_head_idx);
    } else if constexpr (use_sliding_window) {
      logits = SlidingWindowAttention<ParamsT>::PrefillLogitsTransform(
          params, logits, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    } else {
      logits = StandardAttention<ParamsT>::PrefillLogitsTransform(params, logits, qo_idx, kv_idx,
                                                                  qo_head_idx, kv_head_idx);
    }
    return logits;
  }

  static __device__ __forceinline__ float BatchDecodeLogitsTransform(
      const ParamsT& params, float logits, int32_t batch_idx, int32_t qo_idx, int32_t kv_idx,
      int32_t qo_head_idx, int32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = ALIBIAttention<ParamsT>::BatchDecodeLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    }
    if constexpr (use_sliding_window) {
      logits = SlidingWindowAttention<ParamsT>::BatchDecodeLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    } else {
      logits = StandardAttention<ParamsT>::BatchDecodeLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    }
    return logits;
  }

  static __device__ __forceinline__ float BatchPrefillLogitsTransform(
      const ParamsT& params, float logits, int32_t batch_idx, int32_t qo_idx, int32_t kv_idx,
      int32_t qo_head_idx, int32_t kv_head_idx) {
    if constexpr (use_alibi) {
      logits = ALIBIAttention<ParamsT>::BatchPrefillLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    }
    if constexpr (use_custom_mask) {
      logits = CustomMaskAttention<ParamsT>::BatchPrefillLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    } else if constexpr (use_sliding_window) {
      logits = SlidingWindowAttention<ParamsT>::BatchPrefillLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    } else {
      logits = StandardAttention<ParamsT>::BatchPrefillLogitsTransform(
          params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx);
    }
    return logits;
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_VARIANTS_CUH_