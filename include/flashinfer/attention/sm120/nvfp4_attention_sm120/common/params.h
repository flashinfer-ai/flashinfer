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

#include <cuda.h>

#include <cstdint>

#include "cutlass/fast_math.h"

struct Qkv_params {
  using index_t = int64_t;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;
  void* __restrict__ delta_s_ptr;

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
  index_t ds_batch_stride;
  index_t ds_row_stride;
  index_t ds_head_stride;

  index_t sfq_batch_stride;
  index_t sfk_batch_stride;
  index_t sfv_batch_stride;
  index_t sfq_row_stride;
  index_t sfk_row_stride;
  index_t sfv_row_stride;
  index_t sfq_head_stride;
  index_t sfk_head_stride;
  index_t sfv_head_stride;

  int h, h_k;

  int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
  void* __restrict__ o_ptr;
  void* __restrict__ oaccum_ptr;
  void* __restrict__ s_ptr;

  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  void* __restrict__ p_ptr;

  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded,
      rotary_dim, unpadded_seqlen_k;
  cutlass::FastDivmod head_divmod, m_block_divmod;
  int total_blocks;
  int seqlen_s;

  float scale_softmax;
  float scale_softmax_log2;
  uint32_t scale_softmax_log2_half2;

  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;

  int* __restrict__ seqused_k;

  int* __restrict__ blockmask;

  void* __restrict__ knew_ptr;
  void* __restrict__ vnew_ptr;

  index_t knew_batch_stride;
  index_t vnew_batch_stride;
  index_t knew_row_stride;
  index_t vnew_row_stride;
  index_t knew_head_stride;
  index_t vnew_head_stride;

  void* __restrict__ rotary_cos_ptr;
  void* __restrict__ rotary_sin_ptr;

  int* __restrict__ cache_batch_idx;

  int* __restrict__ block_table;
  index_t block_table_batch_stride;
  int page_block_size;

  float p_dropout;

  uint8_t p_dropout_in_uint8_t;

  float rp_dropout;
  float scale_softmax_rp_dropout;

  int window_size_left, window_size_right;

  uint64_t philox_args[2];

  uint64_t* rng_state;

  bool is_bf16;
  bool is_e4m3;
  bool is_causal;
  bool per_block_mean;

  bool is_seqlens_k_cumulative;

  bool is_rotary_interleaved;

  int num_splits;

  void* __restrict__ alibi_slopes_ptr;
  index_t alibi_slopes_batch_stride;

  int* __restrict__ tile_count_semaphore;
};
