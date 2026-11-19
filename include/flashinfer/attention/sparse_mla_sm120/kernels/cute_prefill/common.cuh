// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cute/tensor.hpp>
#include <flashinfer/fastdiv.cuh>

#include "../../arch/common.cuh"
#include "../../compute/mega.cuh"
#include "copy_atom.cuh"

namespace flashinfer::sparse_mla_sm120::cute_prefill {
using namespace cute;
using E4 = cute::float_e4m3_t;
using Mma = decltype(make_tiled_mma(SM89_16x8x32_F32E4M3E4M3F32_TN{}));

// Inline UE8M0-g32 cache parameters; contiguous Q and indices.
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

struct PagedParams : Params {
  bf16* bf16_output;
  size_t lse_stride;
  float lse_scale;
  size_t page_stride, extra_page_stride;
  int page_size, extra_page_size;
  flashinfer::uint_fastdiv page_div, extra_page_div;
};

// Footer pages store all payload rows followed by all 16-byte scale rows.
// Inline rows keep their original flat addressing specialization.
template <bool Inline, bool Scale>
__device__ __forceinline__ const uint8_t* cache_row(const PagedParams& p, bool extra, int id) {
  const uint8_t* cache = extra ? p.extra_kv : p.kv;
  if constexpr (Inline) {
    return cache + size_t(max(id, 0)) * (extra ? p.extra_stride : p.kv_stride) + (Scale ? 512 : 0);
  } else {
    uint32_t page, row;
    (extra ? p.extra_page_div : p.page_div).divmod(uint32_t(max(id, 0)), page, row);
    const size_t page_stride = extra ? p.extra_page_stride : p.page_stride;
    const int page_size = extra ? p.extra_page_size : p.page_size;
    return cache + size_t(page) * page_stride +
           (Scale ? size_t(page_size) * 512 + row * 16 : row * 512);
  }
}

static __device__ __forceinline__ float row_max4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffff, x, 2));
}
static __device__ __forceinline__ float row_sum4(float x) {
  x += __shfl_xor_sync(0xffffffff, x, 1);
  return x + __shfl_xor_sync(0xffffffff, x, 2);
}
static __device__ __forceinline__ float ue8(uint8_t x) {
  return __uint_as_float(uint32_t(x) << 23);
}

}  // namespace flashinfer::sparse_mla_sm120::cute_prefill
