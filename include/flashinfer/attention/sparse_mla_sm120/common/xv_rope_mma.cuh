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

#include "../arch/common.cuh"
#include "../arch/ldmatrix_sm120.cuh"
#include "../arch/mma_sm120.cuh"
#include "../model/kv_cache_traits.cuh"
#include "kv_cache_io.cuh"

// XV rope via BF16 MMA m16n8k16 (DSV4 only).
//
// Computes: acc_rope[h][d] += Σ_e softmax_weight[h][e] * v_rope[e][d]
// Per tile: 64 entries, 16 heads, 64 rope dims.
//
// Each warp handles 1 n-tile of 8 rope dims (n_start = mwarp * 8).
// 4 k-steps of 16 entries each cover all 64 entries. No cross-warp reduction.
//
// A operand: softmax weights as bf16, stored to smem (overlay on dead v_trans),
//            loaded via ldmatrix.x4.
// B operand: V rope bf16, loaded from global memory (L2 cached), scalar packing.
// C output:  c0=C[gid][tid*2], c1=C[gid][tid*2+1],
//            c2=C[gid+8][tid*2], c3=C[gid+8][tid*2+1]

template <ModelType MT, int PAGE_BLOCK_SIZE>
__device__ __forceinline__ void xv_rope_mma(float acc_rope[4], float w0, float w1, float w2,
                                            float w3, const int32_t* __restrict__ tile_indices,
                                            const uint8_t* __restrict__ KV_cache, int mwarp,
                                            int lane, size_t stride_kv_block, bf16* weight_smem) {
  if constexpr (!KVCacheTraits<MT>::V_HAS_ROPE) return;

  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int qk_nb = mwarp * ENTRIES_PER_WARP;

  // Step 1: all warps store softmax weights as bf16 to smem
  // Layout: weight_smem[head * BI + entry], stride = BI bf16 elements
  weight_smem[gid * BI + qk_nb + tid * 2] = __float2bfloat16(w0);
  weight_smem[gid * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(w1);
  weight_smem[(gid + 8) * BI + qk_nb + tid * 2] = __float2bfloat16(w2);
  weight_smem[(gid + 8) * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(w3);
  bar_sync_t<2, MATH_THREADS>();

  // Step 2: BF16 MMA loop — 4 k-steps, each warp handles n_start..n_start+7
  int n_start = mwarp * 8;
  int dim_n = n_start + gid;

  for (int ks = 0; ks < BI / 16; ks++) {
    int k_base = ks * 16;

    // A operand via ldmatrix.x4 from weight_smem
    uint32_t a0, a1, a2, a3;
    ldmatrix_load_A_bf16(a0, a1, a2, a3, weight_smem + k_base, BI, lane);

    // B operand: 4 scalar loads from global (L2 cached). Invalid entries
    // (idx < 0) return v=0 directly rather than reading slot 0 — if slot 0
    // holds non-finite KV (unwritten BF16 NaN/inf from prior workload),
    // 0 * NaN = NaN would propagate through the MMA. The QK side masks
    // invalid entries to -1e30 so weights round to 0, but that's not safe
    // against non-finite V.
    auto load_rope_v = [&](int entry_offset) -> uint16_t {
      int idx = tile_indices[entry_offset];
      if (idx < 0) return 0;
      const uint8_t* base;
      if constexpr (KV::SCALE_IN_KV_SMEM) {
        base = KV_cache + (size_t)idx * IO::IO_STRIDE;
      } else {
        constexpr int pbs = PAGE_BLOCK_SIZE;
        int bi = idx / pbs;
        int li = idx % pbs;
        base = KV_cache + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
      }
      const bf16* rp = reinterpret_cast<const bf16*>(base + KV::KV_ROPE_GMEM_OFFSET);
      return *reinterpret_cast<const uint16_t*>(&rp[dim_n]);
    };

    uint16_t v0 = load_rope_v(k_base + tid * 2);
    uint16_t v1 = load_rope_v(k_base + tid * 2 + 1);
    uint16_t v8 = load_rope_v(k_base + tid * 2 + 8);
    uint16_t v9 = load_rope_v(k_base + tid * 2 + 9);

    uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
    uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);

    // BF16 MMA m16n8k16
    MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc_rope[0], acc_rope[1],
                                        acc_rope[2], acc_rope[3]);
    acc_rope[0] = r.d0;
    acc_rope[1] = r.d1;
    acc_rope[2] = r.d2;
    acc_rope[3] = r.d3;
  }
  // No trailing barrier is needed here. The next sm.w_fp8 reuse is guarded by
  // a later math-wide reduction barrier before W quantization starts.
}

template <ModelType MT, int PAGE_BLOCK_SIZE, int N_HG>
__device__ __forceinline__ void xv_rope_mma_mg(float acc_rope[N_HG][4], const float w_grp[N_HG][4],
                                               const int32_t* __restrict__ tile_indices,
                                               const uint8_t* __restrict__ KV_cache, int mwarp,
                                               int lane, size_t stride_kv_block,
                                               bf16* weight_smem) {
  if constexpr (!KVCacheTraits<MT>::V_HAS_ROPE) return;

  using KV = KVCacheTraits<MT>;
  using IO = KVIOTraits<MT>;
  static_assert(N_HG > 0, "N_HG must be positive");

  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int qk_nb = mwarp * ENTRIES_PER_WARP;
  constexpr int GROUP_STRIDE = HPB * BI;

#pragma unroll
  for (int g = 0; g < N_HG; g++) {
    bf16* group_weight = weight_smem + g * GROUP_STRIDE;
    group_weight[gid * BI + qk_nb + tid * 2] = __float2bfloat16(w_grp[g][0]);
    group_weight[gid * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(w_grp[g][1]);
    group_weight[(gid + 8) * BI + qk_nb + tid * 2] = __float2bfloat16(w_grp[g][2]);
    group_weight[(gid + 8) * BI + qk_nb + tid * 2 + 1] = __float2bfloat16(w_grp[g][3]);
  }
  bar_sync_t<2, MATH_THREADS>();

  int n_start = mwarp * 8;
  int dim_n = n_start + gid;

  for (int ks = 0; ks < BI / 16; ks++) {
    int k_base = ks * 16;

    auto load_rope_v = [&](int entry_offset) -> uint16_t {
      int idx = tile_indices[entry_offset];
      if (idx < 0) return 0;
      const uint8_t* base;
      if constexpr (KV::SCALE_IN_KV_SMEM) {
        base = KV_cache + (size_t)idx * IO::IO_STRIDE;
      } else {
        constexpr int pbs = PAGE_BLOCK_SIZE;
        int bi = idx / pbs;
        int li = idx % pbs;
        base = KV_cache + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;
      }
      const bf16* rp = reinterpret_cast<const bf16*>(base + KV::KV_ROPE_GMEM_OFFSET);
      return *reinterpret_cast<const uint16_t*>(&rp[dim_n]);
    };

    uint16_t v0 = load_rope_v(k_base + tid * 2);
    uint16_t v1 = load_rope_v(k_base + tid * 2 + 1);
    uint16_t v8 = load_rope_v(k_base + tid * 2 + 8);
    uint16_t v9 = load_rope_v(k_base + tid * 2 + 9);

    uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
    uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);

#pragma unroll
    for (int g = 0; g < N_HG; g++) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, weight_smem + g * GROUP_STRIDE + k_base, BI, lane);
      MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc_rope[g][0], acc_rope[g][1],
                                          acc_rope[g][2], acc_rope[g][3]);
      acc_rope[g][0] = r.d0;
      acc_rope[g][1] = r.d1;
      acc_rope[g][2] = r.d2;
      acc_rope[g][3] = r.d3;
    }
  }
  // No trailing barrier is needed here. The next sm.w_fp8 reuse is guarded by
  // a later math-wide reduction barrier before W quantization starts.
}
