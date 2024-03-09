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
#include <flashinfer/attention_decl.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#define INST_BatchPrefillPagedWrapper(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION, \
                                      LAYOUT, pos_encoding_mode)                                \
  namespace flashinfer {                                                                        \
  template cudaError_t BatchPrefillWithPagedKVCacheWrapperDispatched<                           \
      PageStorage::kIndices, LAYOUT, GROUP_SIZE, HEAD_DIM, pos_encoding_mode,                   \
      ALLOW_FP16_QK_REDUCTION, CAUSAL, T, T, int32_t>(                                          \
      BatchPrefillHandler * handler, T* q, int32_t* qo_indptr, int32_t* q_offset,               \
      paged_kv_t<PageStorage::kIndices, LAYOUT, T, int32_t> paged_kv, T* o, float* lse,         \
      float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream);                 \
  }

#define INST_BatchPrefillRaggedWrapper(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION, \
                                       LAYOUT, pos_encoding_mode)                                \
  namespace flashinfer {                                                                         \
  template cudaError_t BatchPrefillWithRaggedKVCacheWrapperDispatched<                           \
      GROUP_SIZE, HEAD_DIM, LAYOUT, pos_encoding_mode, ALLOW_FP16_QK_REDUCTION, CAUSAL, T, T,    \
      int32_t>(BatchPrefillHandler * handler, T* q, int32_t* qo_indptr, T* k, T* v,              \
               int32_t* kv_indptr, int32_t* q_offset, int32_t* k_rope_pos_offset, T* o,          \
               float* lse, uint32_t batch_size, uint32_t num_kv_heads, float sm_scale,           \
               float rope_scale, float rope_theta, cudaStream_t stream);                         \
  }

#define INST_SinglePrefill(T, GROUP_SIZE, HEAD_DIM, CAUSAL, ALLOW_FP16_QK_REDUCTION, LAYOUT,     \
                           pos_encoding_mode)                                                    \
  namespace flashinfer {                                                                         \
  template cudaError_t SinglePrefillWithKVCacheDispatched<                                       \
      GROUP_SIZE, HEAD_DIM, LAYOUT, pos_encoding_mode, ALLOW_FP16_QK_REDUCTION, CAUSAL, T, T>(   \
      T * q, T* k, T* v, T* o, float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len,   \
      uint32_t kv_len, float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream); \
  }
