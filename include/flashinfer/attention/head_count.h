/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_HEAD_COUNT_H_
#define FLASHINFER_ATTENTION_HEAD_COUNT_H_

#include <cstdint>

#include "../exception.h"

namespace flashinfer {

inline void CheckAttentionHeadCounts(int64_t num_qo_heads, int64_t num_kv_heads) {
  FLASHINFER_CHECK(num_qo_heads > 0, "num_qo_heads must be positive, got", num_qo_heads);
  FLASHINFER_CHECK(num_kv_heads > 0, "num_kv_heads must be positive, got", num_kv_heads);
  FLASHINFER_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads", num_qo_heads,
                   "must be divisible by num_kv_heads", num_kv_heads);
}

inline int64_t GetGQAGroupSize(int64_t num_qo_heads, int64_t num_kv_heads) {
  CheckAttentionHeadCounts(num_qo_heads, num_kv_heads);
  return num_qo_heads / num_kv_heads;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HEAD_COUNT_H_
