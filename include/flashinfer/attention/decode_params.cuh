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
  uint32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SingleDecodeParams : DecodeParamsBase<DTypeQ_, DTypeKV_, DTypeO_> {
  DTypeKV* k;
  DTypeKV* v;
  tensor_info_t info;
  uint32_t kv_chunk_size;

  __device__ __host__ SingleDecodeParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
                                         tensor_info_t info, uint32_t window_left,
                                         float logits_soft_cap, float sm_scale,
                                         float rope_rcp_scale, float rope_rcp_theta,
                                         uint32_t kv_chunk_size)
      : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO>{q,
                                                  o,
                                                  lse,
                                                  window_left,
                                                  logits_soft_cap,
                                                  sm_scale,
                                                  rope_rcp_scale,
                                                  rope_rcp_theta},
        k(k),
        v(v),
        info(info),
        kv_chunk_size(kv_chunk_size) {}
};

template <PageStorage page_storage, typename DTypeQ, typename DTypeKV, typename DTypeO,
          typename IdType>
struct BatchDecodeParams : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO> {
  IdType* q_offset;
  paged_kv_t<page_storage, DTypeKV, IdType> paged_kv;
  kv_partition_info_t<IdType> kv_partition_info;
  bool* bool_valid_mask;

  __device__ __host__ BatchDecodeParams(DTypeQ* q, IdType* q_offset,
                                        paged_kv_t<page_storage, DTypeKV, IdType> paged_kv,
                                        kv_partition_info_t<IdType> kv_partition_info,
                                        DTypeO* o, float* lse, bool* bool_valid_mask,
                                        uint32_t window_left, float logits_soft_cap, float sm_scale,
                                        float rope_rcp_scale, float rope_rcp_theta)
      : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO>{q,
                                                  o,
                                                  lse,
                                                  window_left,
                                                  logits_soft_cap,
                                                  sm_scale,
                                                  rope_rcp_scale,
                                                  rope_rcp_theta},
        q_offset(q_offset),
        paged_kv(paged_kv),
        kv_partition_info(kv_partition_info),
        bool_valid_mask(bool_valid_mask) {}
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
