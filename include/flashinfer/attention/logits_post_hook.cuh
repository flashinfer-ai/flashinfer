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
#ifndef FLASHINFER_ATTENTION_LOGITS_POST_HOOK_CUH_
#define FLASHINFER_ATTENTION_LOGITS_POST_HOOK_CUH_

#include "../math.cuh"

namespace flashinfer {

enum class LogitsPostHook {
  kNone = 0U,
  kSoftCap = 1U,
};

/*!
 * \brief Grok's logits cap function
 * \ref
 * https://github.com/xai-org/grok-1/blob/7050ed204b8206bb8645c7b7bbef7252f79561b0/model.py#L864-L865
 */
__forceinline__ __device__ float logits_soft_cap_impl(float x, const float soft_cap) {
  return (soft_cap * math::log2e) * math::tanh(x);
}

__forceinline__ __device__ half2 logits_soft_cap_impl(half2 x, const float soft_cap) {
  return __hmul2(__float2half2_rn(soft_cap * math::log2e), math::tanh(x));
}

template <LogitsPostHook mode, typename T>
__forceinline__ __device__ T apply_logits_post_hook(T x, const float soft_cap);

template <>
__forceinline__ __device__ float apply_logits_post_hook<LogitsPostHook::kNone, float>(
    float x, const float soft_cap) {
  return x;
}

template <>
__forceinline__ __device__ float apply_logits_post_hook<LogitsPostHook::kSoftCap, float>(
    float x, const float soft_cap) {
  return logits_soft_cap_impl(x, soft_cap);
}

template <>
__forceinline__ __device__ half2
apply_logits_post_hook<LogitsPostHook::kNone, half2>(half2 x, const float soft_cap) {
  return x;
}

template <>
__forceinline__ __device__ half2
apply_logits_post_hook<LogitsPostHook::kSoftCap, half2>(half2 x, const float soft_cap) {
  return logits_soft_cap_impl(x, soft_cap);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_LOGITS_POST_HOOK_CUH_
