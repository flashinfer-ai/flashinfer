/*
 * Copyright (c) 2023 by FlashInfer team.
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
#pragma once
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/rope.cuh>

#define INST_BatchPrefillPagedWrapper(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION,    \
                                      LAYOUT, ROTARY_MODE)                                         \
  namespace flashinfer {                                                                           \
  template cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched<                              \
      PageStorage::kIndices, LAYOUT, GROUP_SIZE, HEAD_DIM, ROTARY_MODE, ALLOW_FP16_QK_REDUCTION,   \
      CAUSAL, T, T, int32_t>(BatchPrefillHandler * handler, T* q, int32_t* qo_indptr,              \
                             int32_t* q_rope_position,                                             \
                             paged_kv_t<PageStorage::kIndices, LAYOUT, T, int32_t> paged_kv, T* o, \
                             float* lse, float rope_scale, float rope_theta, cudaStream_t stream); \
  }

#define INST_BatchPrefillRaggedWrapper(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION,   \
                                       LAYOUT, ROTARY_MODE)                                        \
  namespace flashinfer {                                                                           \
  template cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched<                             \
      GROUP_SIZE, HEAD_DIM, LAYOUT, ROTARY_MODE, ALLOW_FP16_QK_REDUCTION, CAUSAL, T, T, int32_t>(  \
      BatchPrefillHandler * handler, T* q, int32_t* qo_indptr, T* k, T* v, int32_t* kv_indptr,     \
      int32_t* q_rope_position, int32_t* k_rope_pos_offset, T* o, float* lse, uint32_t batch_size, \
      uint32_t num_kv_heads, float rope_scale, float rope_theta, cudaStream_t stream);             \
  }

#define INST_SinglePrefill(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION, LAYOUT,   \
                           ROTARY_MODE)                                                        \
  namespace flashinfer {                                                                       \
  template cudaError_t SinglePrefillWithKVCacheDispatched<                                     \
      GROUP_SIZE, HEAD_DIM, LAYOUT, ROTARY_MODE, ALLOW_FP16_QK_REDUCTION, CAUSAL, T, T>(       \
      T * q, T* k, T* v, T* o, float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, \
      uint32_t kv_len, float rope_scale, float rope_theta, cudaStream_t stream);               \
  }

namespace flashinfer {

class BatchPrefillHandler;

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT, RotaryMode ROTARY_MODE,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, IdType* q_rope_position, IdType* k_rope_pos_offset, DTypeOut* o, float* lse,
    const uint32_t batch_size, const uint32_t num_kv_heads, const float rope_scale,
    const float rope_theta, cudaStream_t stream);

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          RotaryMode ROTARY_MODE, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, IdType* q_rope_position,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    float rope_scale, float rope_theta, cudaStream_t stream);

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT, RotaryMode ROTARY_MODE,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                               float* tmp, float* lse, uint32_t num_kv_heads,
                                               uint32_t qo_len, uint32_t kv_len, float rope_scale,
                                               float rope_theta, cudaStream_t stream);

}  // namespace flashinfer
