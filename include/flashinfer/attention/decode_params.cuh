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
struct DecodeParamsBase {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  DTypeQ* q;
  DTypeO* o;
  float* lse;
  float sm_scale;
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SingleDecodeParams : public DecodeParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = int32_t;
  DTypeKV* k;
  DTypeKV* v;
  float* alibi_slopes;
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
  float rope_rcp_scale;
  float rope_rcp_theta;
  uint32_t kv_chunk_size;

  __device__ __host__ SingleDecodeParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                                         float* alibi_slopes, uint32_t seq_len,
                                         uint32_t num_qo_heads, uint32_t num_kv_heads,
                                         QKVLayout kv_layout, uint32_t head_dim,
                                         uint32_t window_left, float logits_soft_cap,
                                         float sm_scale, float rope_scale, float rope_theta)
      : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO>{q, o, /*lse=*/nullptr, sm_scale},
        k(k),
        v(v),
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

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType_>
struct BatchDecodeParams : public DecodeParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = IdType_;
  IdType* q_offset;
  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  bool* block_valid_mask;
  float* alibi_slopes;
  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  uint32_t window_left;
  float logits_soft_cap;
  float rope_rcp_scale;
  float rope_rcp_theta;

  bool partition_kv;

  __device__ __host__ BatchDecodeParams(DTypeQ* q, IdType* q_offset,
                                        paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse,
                                        float* alibi_slopes, uint32_t num_qo_heads,
                                        uint32_t window_left, float logits_soft_cap, float sm_scale,
                                        float rope_scale, float rope_theta)
      : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO>{q, o, lse, sm_scale},
        q_offset(q_offset),
        paged_kv(paged_kv),
        block_valid_mask(nullptr),
        alibi_slopes(alibi_slopes),
        padded_batch_size(0),
        num_qo_heads(num_qo_heads),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
