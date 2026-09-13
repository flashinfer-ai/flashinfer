// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../compute/tile_traits.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "../dsv41_fp8/resources.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Decode tile configuration, per model type. Only these four values are chosen;
// everything else in DecodeTileCfg is derived.
//
// The DeepSeek-family models run a 64-candidate tile across 8 math warps.
// DOTS3_SWA halves both. Its 1040-byte KV smem stride does not fit BI=64: the
// resulting 173872 B dynamic request is rejected outright by the driver on
// sm120 (cudaFuncSetAttribute -> invalid argument), against a 101376 B
// per-block opt-in cap. At BI=32 it needs 97072 B and fits, with ~3.2 KB spare.
// Math warps must halve with BI, or ENTRIES_PER_WARP drops below 8 and
// QK_N_TILES floors to 0 — see the asserts in DecodeTileCfg.
template <ModelType MT>
struct DecodeTilePrimary {
  static constexpr int N_WARPS = 8;  // math warps
  static constexpr int IO_WARPS = 1;
  static constexpr int CAND_WINDOW = 64;
  static constexpr int KV_BUF_COUNT = 2;
  // Sliding-window bound on a token's candidate count, 0 for models whose
  // candidate list is a genuine top-k with no positional bound.
  static constexpr int WINDOW = 0;
};

template <>
struct DecodeTilePrimary<ModelType::DOTS3_SWA> {
  static constexpr int N_WARPS = 4;
  static constexpr int IO_WARPS = 1;
  static constexpr int CAND_WINDOW = 32;
  static constexpr int KV_BUF_COUNT = 2;  // keep the copy/compute overlap
  // the family's sliding-window size. The candidate list is the window, so no
  // token can have more than this many valid entries however wide TOPK is.
  static constexpr int WINDOW = KVCacheTraits<ModelType::DOTS3_SWA>::WINDOW;
};

template <ModelType MT, bool MIXED_PIPELINE = false>
struct DecodeTileCfg {
  using P = DecodeTilePrimary<MT>;
  using KV = KVCacheTraits<MT>;
  static constexpr int MATH_BARRIER = 3;
  using MixedPlan = kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources;
  static constexpr int N_WARPS =
      MIXED_PIPELINE ? MixedPlan::MATH_THREADS / MixedPlan::WARP_THREADS : P::N_WARPS;
  static constexpr int IO_WARPS = P::IO_WARPS;
  static constexpr int CAND_WINDOW = MIXED_PIPELINE ? MixedPlan::BI : P::CAND_WINDOW;
  static constexpr int KV_BUF_COUNT = P::KV_BUF_COUNT;

  // XV W-fold arity. The FP8 weight buffer carries the V dequant scale folded
  // in per (candidate, scale group), so one buffer serves exactly one
  // QUANT_TILE-wide group and a chunk feeds at most QUANT_TILE/8 warps. When
  // the math warps outnumber that (DSV4_1: 8 warps, 32-wide groups), each
  // loop step folds W once per group in an XV_FOLD-wide chunk group
  // (pair-fold) instead of narrowing the warp count.
  static constexpr int XV_FOLD = (N_WARPS * 8 + KV::QUANT_TILE - 1) / KV::QUANT_TILE;
  static constexpr int XV_WARPS = N_WARPS / XV_FOLD;

  static constexpr int N_TOTAL_WARPS = N_WARPS + IO_WARPS;       // DSV4 9,   DOTS3_SWA 5
  static constexpr int BASE_BLOCK_THREADS = N_TOTAL_WARPS * 32;  // DSV4 288, DOTS3_SWA 160
  static constexpr int MATH_THREADS = N_WARPS * 32;              // DSV4 256, DOTS3_SWA 128
  static constexpr int IO_THREADS = IO_WARPS * 32;               // 32
  static constexpr int BI = CAND_WINDOW;                         // DSV4 64,  DOTS3_SWA 32
  static constexpr int ENTRIES_PER_WARP = BI / N_WARPS;          // 8 for both
  static constexpr int QK_N_TILES = ENTRIES_PER_WARP / 8;        // 1 for both

  static constexpr int WINDOW = P::WINDOW;
  static constexpr bool HAS_WINDOW = WINDOW > 0;

  static_assert(BI % N_WARPS == 0, "each math warp must own a whole number of entries");
  static_assert(ENTRIES_PER_WARP >= 8,
                "ENTRIES_PER_WARP < 8 floors QK_N_TILES to 0 and silently drops the QK MMA; "
                "halve N_WARPS along with CAND_WINDOW");
  static_assert(QK_N_TILES >= 1, "QK tiling degenerate");
  static_assert(XV_FOLD >= 1 && XV_FOLD <= 2,
                "beyond pair-fold the XV stage needs a wider W buffer design");
  static_assert(N_WARPS % XV_FOLD == 0, "the XV folds must split the math warps evenly");
  static_assert(KV::D_NOPE % (KV::QUANT_TILE * XV_FOLD) == 0,
                "the XV chunk group must tile the nope dims");

  template <typename Schedule>
  static constexpr int block_threads() {
    if constexpr (Schedule::RAW_PIPELINE) {
      static_assert(MATH_THREADS == Schedule::MATH_THREADS);
      static_assert(BI == Schedule::BI);
      return Schedule::BLOCK_THREADS;
    } else {
      return BASE_BLOCK_THREADS + Schedule::EXTRA_THREADS;
    }
  }
};

template <ModelType MT, bool MIXED_PIPELINE = false>
struct Fp8DecodeSharedLayout {
  using KV = KVCacheTraits<MT>;
  using Cfg = DecodeTileCfg<MT, MIXED_PIPELINE>;
  // This layout assumes footer scales (a separate kv_sc buffer) and a KV smem
  // region holding nope only. DSV4, DOTS3_SWA, and DSV4_1 satisfy that; the
  // inline-scale models (DSV3_2 / GLM_NSA) bulk-copy their scales inside the
  // KV region and use kernels/dsv32_fp8/resources.cuh instead.
  static_assert(MT == ModelType::DSV4 || MT == ModelType::DOTS3_SWA || MT == ModelType::DSV4_1);
  static_assert(!KV::SCALE_IN_KV_SMEM, "this smem layout keeps scales in a separate buffer");

  static constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;
  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP8 = HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_KV_FP8_BUF = Cfg::BI * KV::KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = Cfg::BI * KV::SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF = Cfg::BI * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * Cfg::N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_HEAD_SC = N_V_CHUNKS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_FP8_BUF = HPB * (Cfg::BI + 16);
  // W buffers: one per (double-buffer parity, XV fold) — see DecodeTileCfg.
  static constexpr int W_FP8_SLOTS = 2 * Cfg::XV_FOLD;

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP8 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP8 + SMEM_Q_FP8;
  static constexpr size_t OFF_KV_FP8 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_SC = OFF_KV_FP8 + Cfg::KV_BUF_COUNT * SMEM_KV_FP8_BUF;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_SC + Cfg::KV_BUF_COUNT * SMEM_KV_SC_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + Cfg::KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_W_HEAD_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP8 = OFF_W_HEAD_SC + SMEM_W_HEAD_SC;
  static constexpr size_t BASE_BYTES = OFF_W_FP8 + W_FP8_SLOTS * SMEM_W_FP8_BUF;
  static constexpr size_t TOTAL_BYTES =
      BASE_BYTES +
      (MIXED_PIPELINE ? kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources::EXTRA_SMEM : 0);

  char* base;

  __device__ static Fp8DecodeSharedLayout init(char* base) { return Fp8DecodeSharedLayout{base}; }
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
  __device__ __forceinline__ uint8_t* kv_sc(int i) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC + i * SMEM_KV_SC_BUF);
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
  __device__ __forceinline__ float* warp_sum() const { return reduce() + Cfg::N_WARPS * HPB; }
  __device__ __forceinline__ float* w_head_sc() const {
    return reinterpret_cast<float*>(base + OFF_W_HEAD_SC);
  }
  // slot = parity * Cfg::XV_FOLD + fold
  __device__ __forceinline__ uint8_t* w_fp8(int slot) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP8 + slot * SMEM_W_FP8_BUF);
  }
};

}  // namespace flashinfer::sparse_mla_sm120
