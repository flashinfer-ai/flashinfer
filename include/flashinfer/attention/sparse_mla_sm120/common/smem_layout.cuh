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

#include "../model/kv_cache_traits.cuh"

// Smem layout: constexpr offset computation for each buffer.
// Parameterized by ModelType and ComputeMode.
//
// Buffers (decode / prefill SG):
//   q_nope_fp8, q_nope_sc, q_rope, kv_buf×2, [kv_scale_buf×2 for DSV4],
//   reduce_buf, sum_reduce_buf (or union), m_smem, l_smem,
//   w_head_sc_all, w_fp8 (FP8 mode), v_trans, mbar_kv
//
// All offsets are in bytes.

template <ModelType MT, ComputeMode CM>
struct SmemLayout {
  using KV = KVCacheTraits<MT>;
  using CT = ComputeTraits<MT, CM>;

  // Q buffers
  static constexpr bool BF16_Q = (CM == ComputeMode::BF16);
  static constexpr size_t SMEM_Q_NOPE =
      BF16_Q ? HPB * KV::Q_NOPE_BF16_STRIDE * sizeof(bf16) : HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = BF16_Q ? 0 : HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_Q_ROPE = HPB * D_ROPE * sizeof(bf16);

  // KV double buffer
  static constexpr size_t SMEM_KV_BUF = BI * KV::KV_SMEM_STRIDE;

  // KV scale buffer: needed when bulk copy doesn't include scales.
  // DSV3_2: copies 528B (nope+scale), scales in kv_smem → no extra buffer.
  // DSV4: copies 448B (nope only), scales at offset 576 → need separate buffer.
  static constexpr bool NEED_SCALE_BUF =
      (KV::KV_SMEM_COPY_BYTES < KV::KV_SCALE_GMEM_OFFSET + KV::SCALE_BYTES_PER_TOKEN);
  static constexpr size_t SMEM_KV_SCALE_BUF = NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

  // Cross-warp reduction; reduce_buf and sum_reduce_buf share memory.
  static constexpr size_t SMEM_REDUCE = N_MATH_WARPS * HPB * sizeof(float);

  // Per-head online softmax state
  static constexpr size_t SMEM_M = HPB * sizeof(float);
  static constexpr size_t SMEM_L = HPB * sizeof(float);

  // XV phase — w_fp8 for all V chunks (batch W quant, single barrier).
  // XV is always FP8; CM only flips the QK side.
  static constexpr size_t SMEM_W_SC_ALL = CT::N_V_CHUNKS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_FP8_ONE = HPB * (BI + 16);
  static constexpr size_t SMEM_W_FP8 = SMEM_W_FP8_ONE * CT::N_V_CHUNKS;

  // Mbarrier (double-buffered)
  static constexpr size_t SMEM_MBAR_KV = 2 * sizeof(uint64_t);

  // Offsets.
  static constexpr size_t OFF_Q_NOPE = 0;
  static constexpr size_t OFF_Q_SC = OFF_Q_NOPE + SMEM_Q_NOPE;
  static constexpr size_t OFF_Q_ROPE = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV0 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_KV1 = OFF_KV0 + SMEM_KV_BUF;
  static constexpr size_t OFF_KV_SC0 = OFF_KV1 + SMEM_KV_BUF;
  static constexpr size_t OFF_KV_SC1 = OFF_KV_SC0 + SMEM_KV_SCALE_BUF;
  static constexpr size_t OFF_REDUCE = OFF_KV_SC1 + SMEM_KV_SCALE_BUF;
  static constexpr size_t OFF_SUM_RED = OFF_REDUCE;  // shares memory with reduce_buf
  static constexpr size_t OFF_M = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_L = OFF_M + SMEM_M;
  static constexpr size_t OFF_W_SC_ALL = OFF_L + SMEM_L;
  static constexpr size_t OFF_W_FP8 = OFF_W_SC_ALL + SMEM_W_SC_ALL;
  static constexpr size_t OFF_MBAR_KV = (OFF_W_FP8 + SMEM_W_FP8 + 7) / 8 * 8;
  static constexpr size_t TOTAL = OFF_MBAR_KV + SMEM_MBAR_KV;

  static_assert(TOTAL <= 101376, "SG smem exceeds 99KB per-block limit");
};

// MG (multi-group) layout: 2 head groups, shared reduce/sum_reduce buffer.
template <ModelType MT, ComputeMode CM>
struct SmemLayoutMG {
  using KV = KVCacheTraits<MT>;
  using CT = ComputeTraits<MT, CM>;
  static constexpr int N_HG = 2;

  static constexpr bool BF16_Q = (CM == ComputeMode::BF16);
  static constexpr size_t SMEM_Q_NOPE =
      BF16_Q ? HPB * KV::Q_NOPE_BF16_STRIDE * sizeof(bf16) : HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = BF16_Q ? 0 : HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_KV_BUF = BI * KV::KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SCALE_BUF =
      SmemLayout<MT, CM>::NEED_SCALE_BUF ? BI * KV::SCALE_BYTES_PER_TOKEN : 0;

  // reduce_buf and sum_reduce_buf share the same memory.
  static constexpr size_t SMEM_REDUCE_MG = N_HG * N_MATH_WARPS * HPB * sizeof(float);

  static constexpr size_t SMEM_M = N_HG * HPB * sizeof(float);
  static constexpr size_t SMEM_L = N_HG * HPB * sizeof(float);
  static constexpr size_t SMEM_W_SC_ALL = N_HG * CT::N_V_CHUNKS * HPB * sizeof(float);
  // Two parities let adjacent V chunks use separate FP8 weight buffers.
  static constexpr int W_FP8_PARITIES = 2;
  static constexpr size_t SMEM_W_FP8_MG = W_FP8_PARITIES * N_HG * HPB * (BI + 16);
  // q_rope is only needed before the main loop; reuse the W_FP8 region.
  static_assert(N_HG * HPB * D_ROPE * sizeof(bf16) <= SMEM_W_FP8_MG);
  static constexpr size_t SMEM_SCRATCH = 0;
  static constexpr size_t SMEM_MBAR_KV = 2 * sizeof(uint64_t);

  static constexpr size_t OFF_Q_NOPE0 = 0;
  static constexpr size_t OFF_Q_NOPE1 = OFF_Q_NOPE0 + SMEM_Q_NOPE;
  static constexpr size_t OFF_Q_SC0 = OFF_Q_NOPE1 + SMEM_Q_NOPE;
  static constexpr size_t OFF_Q_SC1 = OFF_Q_SC0 + SMEM_Q_SC;
  static constexpr size_t OFF_KV0 = OFF_Q_SC1 + SMEM_Q_SC;
  static constexpr size_t OFF_KV1 = OFF_KV0 + SMEM_KV_BUF;
  static constexpr size_t OFF_KV_SC0 = OFF_KV1 + SMEM_KV_BUF;
  static constexpr size_t OFF_KV_SC1 = OFF_KV_SC0 + SMEM_KV_SCALE_BUF;
  // Single buffer used as both reduce and sum_reduce.
  static constexpr size_t OFF_REDUCE = OFF_KV_SC1 + SMEM_KV_SCALE_BUF;
  static constexpr size_t OFF_M = OFF_REDUCE + SMEM_REDUCE_MG;
  static constexpr size_t OFF_L = OFF_M + SMEM_M;
  static constexpr size_t OFF_W_SC_ALL = OFF_L + SMEM_L;
  static constexpr size_t OFF_W_FP8 = OFF_W_SC_ALL + SMEM_W_SC_ALL;
  static constexpr size_t OFF_SCRATCH = OFF_W_FP8;
  static constexpr size_t OFF_MBAR_KV = (OFF_W_FP8 + SMEM_W_FP8_MG + 7) / 8 * 8;
  static constexpr size_t TOTAL = OFF_MBAR_KV + SMEM_MBAR_KV;

  static_assert(TOTAL <= 101376, "MG smem exceeds 99KB per-block limit");
};

// MG convenience accessor
template <ModelType MT, ComputeMode CM>
struct SmemPtrsMG {
  using LMG = SmemLayoutMG<MT, CM>;
  using CT = ComputeTraits<MT, CM>;

  static constexpr int N_HG = LMG::N_HG;
  static constexpr int REDUCE_GRP_STRIDE = N_MATH_WARPS * HPB;
  static constexpr int ML_GRP_STRIDE = HPB;
  static constexpr int WSC_GRP_STRIDE = CT::N_V_CHUNKS * HPB;
  static constexpr int WFP8_GRP_SIZE = HPB * (BI + 16);
  // Stride between W_FP8 ping-pong parities.
  static constexpr int WFP8_PARITY_STRIDE = LMG::N_HG * WFP8_GRP_SIZE;

  char* base;

  __device__ static SmemPtrsMG init(char* base) { return SmemPtrsMG{base}; }

  // q_nope_fp8 / q_nope_bf16 alias the same OFF_Q_NOPE region; one is used
  // per ComputeMode. q_nope_sc is empty under CM=BF16.
  __device__ __forceinline__ uint8_t* q_nope_fp8(int g) const {
    return reinterpret_cast<uint8_t*>(base + LMG::OFF_Q_NOPE0 + g * LMG::SMEM_Q_NOPE);
  }
  __device__ __forceinline__ bf16* q_nope_bf16(int g) const {
    return reinterpret_cast<bf16*>(base + LMG::OFF_Q_NOPE0 + g * LMG::SMEM_Q_NOPE);
  }
  __device__ __forceinline__ float* q_nope_sc(int g) const {
    return reinterpret_cast<float*>(base + LMG::OFF_Q_SC0 + g * LMG::SMEM_Q_SC);
  }
  __device__ __forceinline__ bf16* q_rope() const {
    return reinterpret_cast<bf16*>(base + LMG::OFF_SCRATCH);
  }
  __device__ __forceinline__ uint8_t* kv_buf(int i) const {
    return reinterpret_cast<uint8_t*>(base + LMG::OFF_KV0 + i * LMG::SMEM_KV_BUF);
  }
  __device__ __forceinline__ uint8_t* kv_scale_buf(int i) const {
    if constexpr (SmemLayout<MT, CM>::NEED_SCALE_BUF) {
      return reinterpret_cast<uint8_t*>(base + LMG::OFF_KV_SC0 + i * LMG::SMEM_KV_SCALE_BUF);
    } else {
      return nullptr;
    }
  }
  __device__ __forceinline__ float* reduce_buf() const {
    return reinterpret_cast<float*>(base + LMG::OFF_REDUCE);
  }
  __device__ __forceinline__ float* m_smem() const {
    return reinterpret_cast<float*>(base + LMG::OFF_M);
  }
  __device__ __forceinline__ float* l_smem() const {
    return reinterpret_cast<float*>(base + LMG::OFF_L);
  }
  __device__ __forceinline__ float* w_head_sc_all() const {
    return reinterpret_cast<float*>(base + LMG::OFF_W_SC_ALL);
  }
  __device__ __forceinline__ uint8_t* w_fp8() const {
    return reinterpret_cast<uint8_t*>(base + LMG::OFF_W_FP8);
  }
  __device__ __forceinline__ uint64_t* mbar_kv(int i) const {
    return reinterpret_cast<uint64_t*>(base + LMG::OFF_MBAR_KV) + i;
  }
};

// SG convenience accessor (initialized from smem base pointer)
template <ModelType MT, ComputeMode CM>
struct SmemPtrs {
  using L = SmemLayout<MT, CM>;

  // q_nope_fp8 / q_nope_bf16 alias the same OFF_Q_NOPE region; one is used
  // per ComputeMode. q_nope_sc is empty under CM=BF16.
  uint8_t* q_nope_fp8;
  bf16* q_nope_bf16;
  float* q_nope_sc;
  bf16* q_rope;
  uint8_t* kv_bufs[2];
  uint8_t* kv_scale_bufs[2];  // nullptr for DSV3_2 (inline scales)
  float* reduce_buf;
  float* sum_reduce_buf;
  float* m_smem;
  float* l_smem;
  float* w_head_sc_all;
  uint8_t* w_fp8;  // base, index by vc * SMEM_W_FP8_ONE
  uint64_t* mbar_kv;

  __device__ static SmemPtrs init(char* base) {
    SmemPtrs s;
    s.q_nope_fp8 = (uint8_t*)(base + L::OFF_Q_NOPE);
    s.q_nope_bf16 = (bf16*)(base + L::OFF_Q_NOPE);
    s.q_nope_sc = (float*)(base + L::OFF_Q_SC);
    s.q_rope = (bf16*)(base + L::OFF_Q_ROPE);
    s.kv_bufs[0] = (uint8_t*)(base + L::OFF_KV0);
    s.kv_bufs[1] = (uint8_t*)(base + L::OFF_KV1);
    if constexpr (L::NEED_SCALE_BUF) {
      s.kv_scale_bufs[0] = (uint8_t*)(base + L::OFF_KV_SC0);
      s.kv_scale_bufs[1] = (uint8_t*)(base + L::OFF_KV_SC1);
    } else {
      s.kv_scale_bufs[0] = nullptr;
      s.kv_scale_bufs[1] = nullptr;
    }
    s.reduce_buf = (float*)(base + L::OFF_REDUCE);
    s.sum_reduce_buf = (float*)(base + L::OFF_SUM_RED);
    s.m_smem = (float*)(base + L::OFF_M);
    s.l_smem = (float*)(base + L::OFF_L);
    s.w_head_sc_all = (float*)(base + L::OFF_W_SC_ALL);
    s.w_fp8 = (uint8_t*)(base + L::OFF_W_FP8);
    s.mbar_kv = (uint64_t*)(base + L::OFF_MBAR_KV);
    return s;
  }
};
