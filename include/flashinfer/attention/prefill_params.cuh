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

#include "../fastdiv.cuh"
#include "../layout.cuh"
#include "../page.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct PrefillParamsBase {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  DTypeQ* q;
  uint8_t* custom_mask;
  DTypeO* o;
  float* lse;
  float sm_scale;
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SinglePrefillParams : public PrefillParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = int32_t;
  DTypeKV* k;
  DTypeKV* v;
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
  uint32_t window_left;
  float logits_soft_cap;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;
  uint_fastdiv group_size;

  bool partition_kv;

  __host__ SinglePrefillParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask, DTypeO* o,
                               float* lse, float* alibi_slopes, uint32_t num_qo_heads,
                               uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
                               uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                               uint32_t kv_stride_h, uint32_t head_dim, uint32_t window_left,
                               float logits_soft_cap, float sm_scale, float rope_scale,
                               float rope_theta)
      : PrefillParamsBase<DTypeQ, DTypeKV, DTypeO>{q, custom_mask, o, lse, sm_scale},
        k(k),
        v(v),
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
        log2_rope_rcp_scale(-std::log2f(rope_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_theta)),
        group_size(num_qo_heads / num_kv_heads),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ size_t get_q_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, q_stride_n, q_stride_h);
  }

  __host__ __device__ __forceinline__ size_t get_o_elem_offset(uint32_t qo_idx,
                                                               uint32_t qo_head_idx,
                                                               uint32_t feat_idx) const {
    return get_elem_offset_impl(qo_idx, qo_head_idx, feat_idx, num_qo_heads * head_dim, head_dim);
  }

  __host__ __device__ __forceinline__ size_t get_kv_elem_offset(uint32_t kv_idx,
                                                                uint32_t kv_head_idx,
                                                                uint32_t feat_idx) const {
    return get_elem_offset_impl(kv_idx, kv_head_idx, feat_idx, kv_stride_n, kv_stride_h);
  }

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType_>
struct BatchPrefillRaggedParams : public PrefillParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = IdType_;

  IdType* request_indices;
  IdType* q_tile_indices;
  IdType* kv_tile_indices;
  IdType* q_indptr;
  DTypeKV* k;
  DTypeKV* v;
  IdType* kv_indptr;
  uint8_t* custom_mask;
  IdType* qk_indptr;
  IdType* q_offset;
  IdType* k_rope_pos_offset;
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  bool* block_valid_mask;
  uint32_t* kv_chunk_size_ptr;
  bool partition_kv;
  uint_fastdiv group_size_fastdiv;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;

  __host__ BatchPrefillRaggedParams(DTypeQ* q, IdType* q_indptr, DTypeKV* k, DTypeKV* v,
                                    IdType* kv_indptr, uint8_t* custom_mask, IdType* qk_indptr,
                                    DTypeO* o, float* lse, float* alibi_slopes, uint32_t group_size,
                                    uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                                    uint32_t kv_stride_h, int32_t window_left,
                                    float logits_soft_cap, float sm_scale, float rope_scale,
                                    float rope_theta)
      : PrefillParamsBase<DTypeQ, DTypeKV, DTypeO>{q, custom_mask, o, lse, sm_scale},
        q_indptr(q_indptr),
        k(k),
        v(v),
        kv_indptr(kv_indptr),
        custom_mask(custom_mask),
        qk_indptr(qk_indptr),
        o(o),
        lse(lse),
        alibi_slopes(alibi_slopes),
        group_size_fastdiv(group_size),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        kv_stride_n(kv_stride_n),
        kv_stride_h(kv_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        log2_rope_rcp_scale(-std::log2f(rope_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_theta)) {}
    
  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType_>
struct BatchPrefillPagedParams : public PrefillParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = IdType_;

  IdType* request_indices;
  IdType* q_tile_indices;
  IdType* kv_tile_indices;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  IdType* q_indptr;
  uint8_t* custom_mask;
  IdType* qk_indptr;
  IdType* q_offset;
  IdType* o_indptr;
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  bool* block_valid_mask;
  uint32_t* kv_chunk_size_ptr;
  bool partition_kv;
  uint_fastdiv group_size_fastdiv;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float log2_rope_rcp_scale;
  float log2_rope_rcp_theta;

  __host__ BatchPrefillPagedParams(DTypeQ* q, IdType* q_indptr, paged_kv_t<DTypeKV, IdType> paged_kv,
    uint8_t* custom_mask, IdType* qk_indptr, DTypeO* o, float* lse, uint32_t num_qo_heads, int32_t window_left, float logits_soft_cap,
    float sm_scale, float rope_scale, float rope_theta):
    PrefillParamsBase<DTypeQ, DTypeKV, DTypeO>{q, custom_mask, o, lse, sm_scale},
    q_indptr(q_indptr),
    paged_kv(paged_kv),
    custom_mask(custom_mask),
    qk_indptr(qk_indptr),
    o(o),
    lse(lse),
    group_size_fastdiv(num_qo_heads / paged_kv.num_heads),
    window_left(window_left),
    logits_soft_cap(logits_soft_cap),
    log2_rope_rcp_scale(-std::log2f(rope_scale)),
    log2_rope_rcp_theta(-std::log2f(rope_theta)) {}
  
  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_