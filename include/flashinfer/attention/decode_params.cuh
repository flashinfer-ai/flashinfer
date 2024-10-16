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
#ifndef FLASHINFER_DECODE_PARAMS_CUH_
#define FLASHINFER_DECODE_PARAMS_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "../layout.cuh"
#include "../page.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SingleDecodeParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
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
  float rope_rcp_scale;
  float rope_rcp_theta;
  uint32_t kv_chunk_size;

  __device__ __host__ SingleDecodeParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                                         float* alibi_slopes, uint32_t seq_len,
                                         uint32_t num_qo_heads, uint32_t num_kv_heads,
                                         QKVLayout kv_layout, uint32_t head_dim,
                                         int32_t window_left, float logits_soft_cap, float sm_scale,
                                         float rope_scale, float rope_theta)
      : q(q),
        k(k),
        v(v),
        o(o),
        lse(nullptr),
        alibi_slopes(alibi_slopes),
        kv_len(seq_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(num_qo_heads * head_dim),
        q_stride_h(head_dim),
        kv_stride_n((kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim),
        kv_stride_h((kv_layout == QKVLayout::kNHD) ? head_dim : seq_len * head_dim),
        head_dim(head_dim),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        kv_chunk_size(0) {}

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

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchDecodeParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  IdType* q_offset;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  DTypeO* o;
  float* lse;
  float* alibi_slopes;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __device__ __host__ BatchDecodeParams(DTypeQ* q, IdType* q_offset,
                                        paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse,
                                        float* alibi_slopes, uint32_t num_qo_heads,
                                        int32_t window_left, float logits_soft_cap, float sm_scale,
                                        float rope_scale, float rope_theta)
      : q(q),
        q_offset(q_offset),
        paged_kv(paged_kv),
        o(o),
        lse(lse),
        alibi_slopes(alibi_slopes),
        padded_batch_size(0),
        num_qo_heads(num_qo_heads),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        request_indices(nullptr),
        kv_tile_indices(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
