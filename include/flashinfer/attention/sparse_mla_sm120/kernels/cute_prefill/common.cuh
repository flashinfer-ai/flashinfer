// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include "copy_atom.cuh"
#include "../../arch/common.cuh"
#include "../../arch/barrier.cuh"
#include "../../arch/cp_async.cuh"
#include "../../compute/mega.cuh"

namespace flashinfer::sparse_mla_sm120::cute_prefill {
using namespace cute;
using E4 = cute::float_e4m3_t;
using Mma = decltype(make_tiled_mma(SM89_16x8x32_F32E4M3E4M3F32_TN{}));

// Standalone experimental ABI: inline UE8M0-g32 cache, contiguous Q and indices.
struct Params {
  const bf16* q;
  const uint8_t* kv;
  const uint8_t* extra_kv;
  const int* indices;
  const int* extra_indices;
  const int* lengths;
  const int* extra_lengths;
  const int* positions;
  const float* cos_sin;
  const float* sink;
  uint8_t* output;
  uint8_t* scales;
  float* lse;
  int tokens, heads, topk, extra_topk;
  int kv_stride, extra_stride;
  float sm_scale;
};
__device__ float row_max4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}
__device__ float row_sum4(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1);
  return x + __shfl_xor_sync(0xffffffff, x, 2);
}
__device__ float ue8(uint8_t x) { return __uint_as_float(uint32_t(x) << 23); }

}  // namespace flashinfer::sparse_mla_sm120::cute_prefill
