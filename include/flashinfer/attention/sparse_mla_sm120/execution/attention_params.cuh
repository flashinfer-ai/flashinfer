// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <flashinfer/fastdiv.cuh>

using bf16 = __nv_bfloat16;

struct PrefillColdParams {
  float sm_scale;
  int num_tokens;
  size_t kv_stride_bytes;  // Inline row stride or footer page stride.
  size_t extra_page_stride_bytes;
  size_t out_lse_stride_elems;
  int topk;
  int extra_topk;
  const float* attn_sink;
  const int* topk_length;
  const int* extra_topk_length;
  const uint8_t* extra_kv = nullptr;
  const int32_t* extra_indices = nullptr;
  int extra_page_block_size = 0;
  int page_block_size = 64;
  flashinfer::uint_fastdiv main_div;
  flashinfer::uint_fastdiv extra_div;
  float lse_scale = 1.f;
};

struct Dsv4PageDivisors {
  flashinfer::uint_fastdiv main;
  flashinfer::uint_fastdiv extra;
};

// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
namespace flashinfer::sparse_mla_sm120::nvfp4 {

struct Dsv4Nvfp4AttentionParams {
  const bf16* q;
  const uint8_t* cache;
  const int32_t* indices;
  bf16* mid_out;
  float* mid_lse;
  bf16* output;
  float* out_lse;
  const int* topk_length;
  const float* attn_sink;
  const uint8_t* extra_cache;
  const int32_t* extra_indices;
  const int* extra_topk_length;
  int extra_topk;
  int extra_page_size;
  size_t extra_page_stride_bytes;
  int num_tokens;
  float sm_scale;
  size_t page_stride_bytes;
  float lse_scale = 1.f;
};

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

namespace flashinfer::sparse_mla_sm120::execution {

struct AttentionParams {
  int num_heads;
  int topk;
  const bf16* q;
  const uint8_t* kv;
  const int32_t* indices;
  bf16* mid_out;
  float* mid_lse;
  const int* topk_length;
  bf16* output;
  float* out_lse;
  const float* attn_sink;
  const uint8_t* extra_kv;
  const int32_t* extra_indices;
  const int* extra_topk_length;
  int extra_topk;
  int extra_page_size;
  size_t extra_page_stride_bytes;
  int num_tokens;
  int allocated_splits;  // FP8/full-BF16 scratch keeps this stride after CPB selection.
  int chunks_per_block;
  float sm_scale;
  size_t page_stride_bytes;
  size_t indices_stride_elems;
  size_t extra_indices_stride_elems;
  size_t out_lse_stride_elems;
  int page_size;
  bool extra_fp4;
  float lse_scale = 1.f;
};

}  // namespace flashinfer::sparse_mla_sm120::execution
