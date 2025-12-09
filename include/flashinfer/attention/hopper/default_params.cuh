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
#ifndef FLASHINFER_ATTENTION_HOPPER_PARAMS_CUH
#define FLASHINFER_ATTENTION_HOPPER_PARAMS_CUH

#include <cuda.h>

#include <vector>

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct SinglePrefillParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  struct AdditionalParams {
    float logits_soft_cap;
    float sm_scale;
    float* scale_q;
    float* scale_k;
    float* scale_v;
  } additional_params;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;

  int qo_len;
  int kv_len;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int window_left;

  bool causal;
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillRaggedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  IdType* qo_tile_indices;
  IdType* qo_indptr;
  IdType* kv_indptr;
  IdType* qo_lens;
  IdType* kv_lens;
  IdType* head_indices;
  IdType* work_indptr;
  IdType* batch_indices;

  struct AdditionalParams {
    float logits_soft_cap;
    float sm_scale;
    uint32_t* maybe_prefix_len_ptr;
    uint16_t* maybe_token_pos_in_items_ptr;
    uint32_t token_pos_in_items_len;
    uint16_t* maybe_max_item_len_ptr;
  } additional_params;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int64_t nnz_qo;
  int64_t nnz_kv;

  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int window_left;

  bool causal;
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  IdType* qo_tile_indices;
  IdType* qo_indptr;
  IdType* kv_indptr;
  IdType* kv_indices;
  IdType* qo_lens;
  IdType* kv_lens;
  IdType* head_indices;
  IdType* work_indptr;
  IdType* batch_indices;

  struct AdditionalParams {
    float logits_soft_cap;
    float sm_scale;
    uint32_t* maybe_prefix_len_ptr;
    uint16_t* maybe_token_pos_in_items_ptr;
    uint32_t token_pos_in_items_len;
    uint16_t* maybe_max_item_len_ptr;
  } additional_params;

  int64_t q_stride_n;
  int64_t k_stride_n;
  int64_t v_stride_n;
  int64_t o_stride_n;
  int64_t q_stride_h;
  int64_t k_stride_h;
  int64_t v_stride_h;
  int64_t o_stride_h;
  int64_t nnz_qo;

  // NOTE: For sparse paged KV cache, we need the stride between pages
  // This is paged_k_cache.stride(0), not the layout stride
  int64_t k_page_stride;  // Stride between pages for K
  int64_t v_page_stride;  // Stride between pages for V

  int num_qo_heads;
  int num_kv_heads;
  int group_size;
  int page_size;
  int window_left;

  bool causal;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_PARAMS_CUH
