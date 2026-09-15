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

#include "../../arch/common.cuh"
#include "../../common/kv_cache_io.cuh"
#include "../../execution/attention_params.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "tile_config.cuh"

struct PrefillSection {
  const uint8_t* kv;
  const int32_t* indices;
  size_t block_stride;
  int tile;
  int length;
  int page_block_size;
  bool extra;

  __device__ __forceinline__ int index(int entry, int bi) const {
    const int pos = tile * bi + entry;
    return pos < length ? __ldg(indices + pos) : -1;
  }
};

__device__ __forceinline__ int prefill_extra_length(const PrefillColdParams& cold, int token) {
  if (cold.extra_kv == nullptr) return 0;
  int len = cold.extra_topk_length ? __ldg(cold.extra_topk_length + token) : cold.extra_topk;
  return len < 0 ? 0 : (len > cold.extra_topk ? cold.extra_topk : len);
}

template <int BI, int PBS>
__device__ __forceinline__ PrefillSection prefill_section(const PrefillColdParams& cold,
                                                          const uint8_t* kv, const int32_t* indices,
                                                          int token, int main_len, int extra_len,
                                                          int tile) {
  const int main_tiles = (main_len + BI - 1) / BI;
  if (tile >= main_tiles && cold.extra_kv != nullptr) {
    return {cold.extra_kv,
            cold.extra_indices + (size_t)token * cold.extra_topk,
            cold.extra_page_stride_bytes,
            tile - main_tiles,
            extra_len,
            cold.extra_page_block_size,
            true};
  }
  return {kv,
          indices + (size_t)token * cold.topk,
          cold.page_stride_bytes,
          tile,
          main_len,
          cold.page_block_size,
          false};
}

struct Fp8PrefillSync {
  static constexpr int MATH = 2;
  static constexpr int QK = MATH;
  static constexpr int CTA_INIT = 3;
  static constexpr int NORMALIZER = 4;
  static constexpr int OUTPUT = 10;
  template <int QkThreads, int XvThreads>
  using WeightsReady =
      flashinfer::sparse_mla_sm120::pipeline::StoreHandoff<6, 7, QkThreads, XvThreads>;
  template <int QkThreads, int XvThreads>
  using WeightsFree =
      flashinfer::sparse_mla_sm120::pipeline::SlotRelease<8, 9, QkThreads, XvThreads>;
  template <int IoThreads, int MathThreads>
  using KvFree = flashinfer::sparse_mla_sm120::pipeline::SlotRelease<1, 5, IoThreads, MathThreads>;
};

template <ModelType MT, int PAGE_BLOCK_SIZE>
__device__ __forceinline__ const uint8_t* prefill_kv_entry_base(
    const uint8_t* __restrict__ kv_global, int idx, size_t stride_kv_block, PageGeom pg = {}) {
  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  // Addressing mode follows the scale layout, not V_HAS_ROPE: an inline-scale
  // model (DSV3_2 / GLM_NSA / GLM53_NOPE) is a flat token array, a footer-scale
  // model (DSV4 / DOTS3_SWA) is paged with the footer after the block's data.
  // This matches io_bulk_gather_tile. Keying it on V_HAS_ROPE happened to agree
  // for the three DeepSeek-family models and disagrees for DOTS3_SWA, which is
  // footer-scaled with no rope in V.
  // Masked lanes redirect to the zero row, not slot 0: rope segments read
  // through this base feed MMAs whose result is only partially masked
  // downstream, so a poisoned slot-0 payload would leak NaN.
  const bool valid = idx >= 0;
  idx = valid ? idx : 0;
  const uint8_t* base;
  if constexpr (PAGE_BLOCK_SIZE == 0) {
    static_assert(MT == ModelType::DSV4);
    int bi, li;
    page_divmod(idx, pg, bi, li);
    base = kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
  } else if constexpr (!KV::SCALE_IN_KV_SMEM) {
    const int bi = idx / PAGE_BLOCK_SIZE;
    const int li = idx % PAGE_BLOCK_SIZE;
    base = kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
  } else {
    // Flat inline array: the row advance is the runtime stride (payload 528
    // for GLM53_NOPE; a legacy 656B vLLM pool advances by 656).
    base = kv_global + (size_t)idx * inline_row_advance(stride_kv_block, PAGE_BLOCK_SIZE);
  }
  return valid ? base : sparse_mla_zero_row;
}
