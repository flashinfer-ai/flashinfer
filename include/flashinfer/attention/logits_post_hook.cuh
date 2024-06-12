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
  kCap30 = 1U,
};

/*!
 * \brief Grok's logits cap function
 * \ref
 * https://github.com/xai-org/grok-1/blob/7050ed204b8206bb8645c7b7bbef7252f79561b0/model.py#L864-L865
 */
__forceinline__ __device__ float logits_cap_30(float x) { return 30.f * math::tanh(x / 30.f); }

template <LogitsPostHook mode>
__forceinline__ __device__ float apply_logits_post_hook(float x);

template <>
__forceinline__ __device__ float apply_logits_post_hook<LogitsPostHook::kNone>(float x) {
  return x;
}

template <>
__forceinline__ __device__ float apply_logits_post_hook<LogitsPostHook::kCap30>(float x) {
  return logits_cap_30(x);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_LOGITS_POST_HOOK_CUH_