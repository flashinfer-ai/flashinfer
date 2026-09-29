/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * This code is based on code from FlashAttention3, https://github.com/Dao-AILab/flash-attention
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
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

#include <cstdint>

namespace flash {

template <bool Varlen = true>
struct BlockInfo {
  template <typename Params>
  __device__ BlockInfo(const Params& params, const int bidb)
      : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb]),
        sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative
                    ? -1
                    : params.cu_seqlens_k[bidb]),
        actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr
                            ? params.seqlen_q
                            : params.cu_seqlens_q[bidb + 1] - sum_s_q)

        ,
        seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr
                           ? params.seqlen_k
                           : (params.is_seqlens_k_cumulative
                                  ? params.cu_seqlens_k[bidb + 1] - sum_s_k
                                  : params.cu_seqlens_k[bidb])),
        actual_seqlen_k(params.seqused_k
                            ? params.seqused_k[bidb]
                            : seqlen_k_cache +
                                  (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride,
                                              const int bidb) const {
    return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride,
                                              const int bidb) const {
    return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
  }

  const int sum_s_q;
  const int sum_s_k;
  const int actual_seqlen_q;

  const int seqlen_k_cache;
  const int actual_seqlen_k;
};

}  // namespace flash
