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
#ifndef FLASHINFER_PREFILL_PARAMS_CUH_
#define FLASHINFER_PREFILL_PARAMS_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "../page.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SinglePrefillParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  uint8_t* custom_mask;
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;

  bool partition_kv;

  __host__ SinglePrefillParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask, DTypeO* o,
                               float* lse, float* alibi_slopes, uint32_t num_qo_heads,
                               uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
                               uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                               uint32_t kv_stride_h, uint32_t head_dim, int32_t window_left,
                               float logits_soft_cap, float sm_scale, float rope_scale,
                               float rope_theta)
      : q(q),
        k(k),
        v(v),
        custom_mask(custom_mask),
        o(o),
        lse(lse),
        alibi_slopes(alibi_slopes),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        qo_len(qo_len),
        kv_len(kv_len),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        head_dim(head_dim),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        log2_rope_rcp_scale(-std::log2f(rope_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_theta)),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }

  __host__ __device__ __forceinline__ uint8_t* get_batch_local_mask_ptr(uint32_t batch_idx) const {
    return this->custom_mask;
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillRaggedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  uint8_t* custom_mask;
  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* mask_indptr;
  IdType* q_offset;           // q_offset is only used for fused-rope attention
  IdType* k_rope_pos_offset;  // k_rope_pos_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ BatchPrefillRaggedParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask,
                                    IdType* q_indptr, IdType* kv_indptr, IdType* mask_indptr,
                                    IdType* q_offset, IdType* k_rope_pos_offset, DTypeO* o,
                                    float* lse, float* alibi_slopes, uint32_t num_qo_heads,
                                    uint32_t num_kv_heads, uint32_t q_stride_n, uint32_t q_stride_h,
                                    uint32_t kv_stride_n, uint32_t kv_stride_h, int32_t window_left,
                                    float logits_soft_cap, float sm_scale, float rope_scale,
                                    float rope_theta)
      : q(q),
        k(k),
        v(v),
        custom_mask(custom_mask),
        q_indptr(q_indptr),
        kv_indptr(kv_indptr),
        mask_indptr(mask_indptr),
        q_offset(q_offset),
        k_rope_pos_offset(k_rope_pos_offset),
        o(o),
        lse(lse),
        alibi_slopes(alibi_slopes),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        log2_rope_rcp_scale(-std::log2f(rope_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_theta)),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint8_t* get_batch_local_mask_ptr(uint32_t batch_idx) const {
    return this->custom_mask + mask_indptr[batch_idx];
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  uint8_t* custom_mask;
  IdType* q_indptr;
  IdType* mask_indptr;
  IdType* q_offset;  // q_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ BatchPrefillPagedParams(DTypeQ* q, paged_kv_t<DTypeKV, IdType> paged_kv,
                                   uint8_t* custom_mask, IdType* q_indptr, IdType* mask_indptr,
                                   IdType* q_offset, DTypeO* o, float* lse, float* alibi_slopes,
                                   uint32_t num_qo_heads, IdType q_stride_n, IdType q_stride_h,
                                   int32_t window_left, float logits_soft_cap, float sm_scale,
                                   float rope_scale, float rope_theta)
      : q(q),
        paged_kv(paged_kv),
        custom_mask(custom_mask),
        q_indptr(q_indptr),
        mask_indptr(mask_indptr),
        q_offset(q_offset),
        o(o),
        lse(lse),
        alibi_slopes(alibi_slopes),
        num_qo_heads(num_qo_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        log2_rope_rcp_scale(-std::log2f(rope_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_theta)),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        block_valid_mask(nullptr),
        kv_chunk_size_ptr(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }

  __host__ __device__ __forceinline__ uint8_t* get_batch_local_mask_ptr(uint32_t batch_idx) const {
    return this->custom_mask + mask_indptr[batch_idx];
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
