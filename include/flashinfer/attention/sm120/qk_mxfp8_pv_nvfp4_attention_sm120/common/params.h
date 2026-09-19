/*
 * Copyright (c) 2025 by SageAttention team.
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

#include <cstdint>

namespace qk_mxfp8_pv_nvfp4_attention {

struct Qkv_params {
  using index_t = int64_t;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  void* __restrict__ sfq_ptr;
  void* __restrict__ sfk_ptr;
  void* __restrict__ sfv_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  int h, h_k;
  int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
  void* __restrict__ o_ptr;

  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  void* __restrict__ softmax_lse_ptr;

  int device_id, b, seqlen_q, seqlen_k, d, unpadded_seqlen_q, unpadded_seqlen_k;
  float scale_softmax_log2;
  bool is_bf16, is_causal;
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
