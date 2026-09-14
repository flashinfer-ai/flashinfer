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
#ifndef FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H
#define FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace flashinfer {

#define REGISTER_QUERY_TRANSFORM(params, q, ...)                                            \
  template <typename MainloopParams, typename T>                                            \
  __device__ __forceinline__ T QueryTransform(const MainloopParams& params, void* q_smem) { \
    __VA_ARGS__                                                                             \
  }

#define REGISTER_KEY_TRANSFORM(params, k, ...)                                            \
  template <typename MainloopParams, typename T>                                          \
  __device__ __forceinline__ T KeyTransform(const MainloopParams& params, void* k_smem) { \
    __VA_ARGS__                                                                           \
  }

#define REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, \
                                  kv_head_idx, ...)                                       \
  template <typename MainloopParams, typename T>                                          \
  __device__ __forceinline__ T LogitsTransform(                                           \
      const MainloopParams& params, T logits, uint32_t batch_idx, uint32_t qo_idx,        \
      uint32_t kv_idx, uint32_t qo_head_idx, uint32_t kv_head_idx) {                      \
    __VA_ARGS__                                                                           \
  }

#define REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, ...) \
  template <typename MainloopParams>                                                           \
  __device__ __forceinline__ bool LogitsMask(const MainloopParams& params, uint32_t batch_idx, \
                                             uint32_t qo_idx, uint32_t kv_idx,                 \
                                             uint32_t qo_head_idx, uint32_t kv_head_idx) {     \
    __VA_ARGS__                                                                                \
  }

// Transforms one output element after normalization (1 / row_sum and, for FP8, the PV
// dequantization scale are already applied), right before it is converted to DTypeO and
// written. Unlike the FA2 hook it does not own normalization.
//   output       fp32 value of query row qo_idx, head qo_head_idx, column d_idx
//   batch_idx    request index within the batch (0 for single-request prefill)
//   qo_idx       request-local query position, as in LogitsTransform
//   d_idx        column of the value head dimension
//   lse          the row's log2-domain log-sum-exp as stored in the lse output; meaningful
//                for softmax updaters only
// Rows that attend to no keys also pass through (output 0, lse -inf). Variants without the
// hook compile exactly as before: the epilogues detect it with has_output_transform_v.
#define REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, d_idx, lse, ...) \
  template <typename MainloopParams, typename T>                                                   \
  __device__ __forceinline__ T OutputTransform(const MainloopParams& params, T output,             \
                                               uint32_t batch_idx, uint32_t qo_idx,                \
                                               uint32_t qo_head_idx, uint32_t d_idx, float lse) {  \
    __VA_ARGS__                                                                                    \
  }

// True when AttentionVariant registered an output transform callable with MainloopParams.
template <typename AttentionVariant, typename MainloopParams, typename = void>
struct has_output_transform : std::false_type {};

template <typename AttentionVariant, typename MainloopParams>
struct has_output_transform<AttentionVariant, MainloopParams,
                            std::void_t<decltype(std::declval<AttentionVariant&>().OutputTransform(
                                std::declval<const MainloopParams&>(), 0.f, uint32_t(0),
                                uint32_t(0), uint32_t(0), uint32_t(0), 0.f))>> : std::true_type {};

template <typename AttentionVariant, typename MainloopParams>
inline constexpr bool has_output_transform_v =
    has_output_transform<AttentionVariant, MainloopParams>::value;

struct AttentionVariantBase {
  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                            { return logits; })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx,
                       { return true; })
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_VARIANT_HELPER_H
