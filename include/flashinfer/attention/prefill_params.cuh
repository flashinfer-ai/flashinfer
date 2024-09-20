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

#include <cstdint>
#include <cmath>

#include "../layout.cuh"
#include "../page.cuh"
#include "../fastdiv.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct PrefillParamsBase {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  DTypeQ* q;
  uint8_t *custom_mask;
  DTypeO *o;
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

  __host__ SinglePrefillParams(
      DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask, DTypeO* o, float* lse,
      float* alibi_slopes, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len,
      uint32_t kv_len, uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
      uint32_t kv_stride_h, uint32_t head_dim, uint32_t window_left, float logits_soft_cap,
      float sm_scale, float rope_rcp_scale, float rope_rcp_theta)
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
        log2_rope_rcp_scale(-std::log2f(rope_rcp_scale)),
        log2_rope_rcp_theta(-std::log2f(rope_rcp_theta)),
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
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_