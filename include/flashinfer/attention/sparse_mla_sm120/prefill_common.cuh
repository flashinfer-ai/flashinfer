// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "arch/barrier.cuh"
#include "arch/common.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "arch/stmatrix_sm120.cuh"
#include "common/d2_load_b.cuh"
#include "common/fp8_quant.cuh"
#include "common/kv_cache_io.cuh"
#include "common/online_softmax.cuh"
#include "common/q_rope.cuh"
#include "common/scale_mma.cuh"
#include "common/smem_layout.cuh"
#include "common/xv_rope_mma.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

// Cold (launch-invariant) parameters shared by every prefill kernel family
// (SG/MG/dual in prefill_mg_kernel.cuh, swapAB in prefill_swapab_kernel.cuh);
// passed by grid constant.
struct PrefillColdParams {
  float sm_scale;
  int num_tokens;
  size_t stride_kv_block;
  // Dual-cache only (sparse_mla_prefill_mg_dual_kernel); ignored elsewhere.
  size_t stride_kv_block_extra;
  int topk_extra;          // dual-cache only. Runtime topk_extra so callers can
                           // pass any cdiv(max_model_len, compress_ratio) value
                           // without per-bound template instantiations.
  const float* attn_sink;  // [NUM_HEADS] float32, natural log domain. nullptr = disabled.
  const int* topk_length;  // [num_tokens] int32, nullptr = uniform TOPK.
  const int*
      topk_length_extra;  // [num_tokens] int32, dual-cache only. nullptr = uniform topk_extra.
};
