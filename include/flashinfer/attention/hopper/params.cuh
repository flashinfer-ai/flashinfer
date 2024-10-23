/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>

#include <vector>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod


template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SinglePrefillParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  // The QKV matrices.
  DTypeQ* q_ptr;
  DTypeKV* k_ptr;
  DTypeKV* v_ptr;
  DTypeO* o_ptr;
  float* lse_ptr;

  // The stride between rows of the Q, K and V matrices.
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
  int head_dim;
  int num_qo_heads;
  int num_kv_heads;
  int group_size;

  float sm_scale_log2;
  bool causal;
};
