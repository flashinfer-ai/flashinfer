// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../compute/tile_traits.cuh"
static constexpr int BI = 64;
static constexpr int N_MATH_WARPS = 8;
static constexpr int N_IO_WARPS = 4;
static constexpr int N_TOTAL_WARPS = N_MATH_WARPS + N_IO_WARPS;
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;
static constexpr int MATH_THREADS = N_MATH_WARPS * 32;
static constexpr int IO_THREADS = N_IO_WARPS * 32;
static constexpr int ENTRIES_PER_WARP = BI / N_MATH_WARPS;

template <ModelType MT>
struct PrefillTilePrimary {
  static constexpr int CAND_WINDOW = BI;
  static constexpr int MATH_WARPS = N_MATH_WARPS;
  static constexpr int IO_WARPS = N_IO_WARPS;
  static constexpr bool REG_REALLOC = true;
  static constexpr bool L2_EVICT_FIRST = true;
  static constexpr int WINDOW = 0;
};

template <>
struct PrefillTilePrimary<ModelType::DSV4_1> {
  static constexpr int CAND_WINDOW = 32;
  static constexpr int MATH_WARPS = 8;
  static constexpr int IO_WARPS = 4;
  static constexpr bool REG_REALLOC = true;
  static constexpr bool L2_EVICT_FIRST = true;
  static constexpr int WINDOW = 0;
};

template <>
struct PrefillTilePrimary<ModelType::DOTS3_SWA> {
  static constexpr int CAND_WINDOW = 32;
  static constexpr int MATH_WARPS = 8;
  static constexpr int IO_WARPS = 4;
  static constexpr bool REG_REALLOC = true;
  static constexpr bool L2_EVICT_FIRST = false;
  static constexpr int WINDOW = KVCacheTraits<ModelType::DOTS3_SWA>::WINDOW;
};

template <ModelType MT>
struct PrefillTileCfg {
  using P = PrefillTilePrimary<MT>;
  static constexpr int MATH_WARPS = P::MATH_WARPS;
  static constexpr int IO_WARPS = P::IO_WARPS;
  static constexpr bool REG_REALLOC = P::REG_REALLOC;
  static constexpr bool L2_EVICT_FIRST = P::L2_EVICT_FIRST;

  static constexpr int N_TOTAL_WARPS = MATH_WARPS + IO_WARPS;
  static constexpr int BASE_BLOCK_THREADS = N_TOTAL_WARPS * 32;
  static constexpr int MATH_THREADS = MATH_WARPS * 32;
  static constexpr int IO_THREADS = IO_WARPS * 32;
  static constexpr int ENTRIES_PER_WARP = 8;
  static constexpr int BI = P::CAND_WINDOW;
  static constexpr int QK_WARPS = BI / ENTRIES_PER_WARP;
  static constexpr int QK_THREADS = QK_WARPS * 32;
  static constexpr bool SPLIT_QK_XV = MATH_WARPS != QK_WARPS;
  static constexpr int MATH_MAXNREG = 232;
  static constexpr int WINDOW = P::WINDOW;
  static constexpr bool HAS_WINDOW = WINDOW > 0;

  static_assert(BI >= 32, "FP8 XV consumes k=32 entries per step; BI < 32 reads past the tile");
  static_assert(BI % 32 == 0, "BI must be a whole number of FP8 XV k-steps");
  static_assert(MATH_WARPS >= QK_WARPS,
                "the XV split may widen the QK warp set but never narrow it");
  static_assert(MATH_WARPS % 4 == 0 && IO_WARPS % 4 == 0,
                "setmaxnreg is warpgroup-aligned; both halves must be whole warpgroups");
  static_assert(QK_WARPS % 4 == 0,
                "the QK warps must be a whole warpgroup prefix of the math warps");
  static_assert(!REG_REALLOC || MATH_MAXNREG * MATH_THREADS + 32 * IO_THREADS <= 65536,
                "redistributed register budget exceeds the per-SM file");
};
