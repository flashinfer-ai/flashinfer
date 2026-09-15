// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../compute/tile_traits.cuh"
#include "../../model/kv_cache_traits.cuh"

namespace flashinfer::sparse_mla_sm120 {

constexpr int DSV32_MATH_BARRIER = 3;
constexpr int DSV32_N_WARPS = 8;  // math warps
constexpr int DSV32_IO_WARPS = 1;
constexpr int DSV32_N_TOTAL_WARPS = DSV32_N_WARPS + DSV32_IO_WARPS;  // 9
constexpr int DSV32_BLOCK_THREADS = DSV32_N_TOTAL_WARPS * 32;        // 288
constexpr int DSV32_MATH_THREADS = DSV32_N_WARPS * 32;               // 256
constexpr int DSV32_IO_THREADS = DSV32_IO_WARPS * 32;                // 32
constexpr int DSV32_CAND_WINDOW = 64;
constexpr int DSV32_BI = DSV32_CAND_WINDOW;
constexpr int DSV32_KV_BUF_COUNT = 2;
constexpr int DSV32_ENTRIES_PER_WARP = DSV32_BI / DSV32_N_WARPS;  // 8
constexpr int DSV32_QK_N_TILES = DSV32_ENTRIES_PER_WARP / 8;      // 1

template <ModelType MT>
struct Dsv32DecodeSmem {
  using KV = KVCacheTraits<MT>;

  static constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;
  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP8 = HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_KV_FP8_BUF = DSV32_BI * KV::KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_ROPE_BUF = DSV32_BI * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * DSV32_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_HEAD_SC = N_V_CHUNKS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_FP8_BUF = HPB * (DSV32_BI + 16);

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP8 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP8 + SMEM_Q_FP8;
  static constexpr size_t OFF_KV_FP8 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_FP8 + DSV32_KV_BUF_COUNT * SMEM_KV_FP8_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + DSV32_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_W_HEAD_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP8 = OFF_W_HEAD_SC + SMEM_W_HEAD_SC;
  static constexpr size_t LAUNCH_BYTES = OFF_MBAR_FULL_UNALIGNED + 16 + 2 * SMEM_MBAR_PAIR +
                                         SMEM_REDUCE + SMEM_W_HEAD_SC + 2 * SMEM_W_FP8_BUF;

  char* base;

  __device__ static Dsv32DecodeSmem init(char* base) { return Dsv32DecodeSmem{base}; }
  __device__ __forceinline__ bf16* q_rope() const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE);
  }
  __device__ __forceinline__ uint8_t* q_fp8() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP8);
  }
  __device__ __forceinline__ float* q_sc() const {
    return reinterpret_cast<float*>(base + OFF_Q_SC);
  }
  __device__ __forceinline__ uint8_t* kv_fp8(int i) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP8 + i * SMEM_KV_FP8_BUF);
  }
  __device__ __forceinline__ bf16* kv_rope(int i) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE + i * SMEM_KV_ROPE_BUF);
  }
  __device__ __forceinline__ uint64_t* mbar_full(int i) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + i;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int i) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + i;
  }
  __device__ __forceinline__ float* reduce() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ float* warp_max() const { return reduce(); }
  __device__ __forceinline__ float* warp_sum() const { return reduce() + DSV32_N_WARPS * HPB; }
  __device__ __forceinline__ float* w_head_sc() const {
    return reinterpret_cast<float*>(base + OFF_W_HEAD_SC);
  }
  __device__ __forceinline__ uint8_t* w_fp8(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP8 + parity * SMEM_W_FP8_BUF);
  }
};

}  // namespace flashinfer::sparse_mla_sm120
