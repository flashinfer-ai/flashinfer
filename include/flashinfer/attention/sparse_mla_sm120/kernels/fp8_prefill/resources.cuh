// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <type_traits>

#include "tile_config.cuh"
#include "smem_layout.cuh"

template <ModelType MT, QkComputeMode QkMode, typename Schedule>
struct Fp8PrefillResources {
  using Tile = PrefillTileCfg<MT>;
  using Layout = SmemLayout<MT, QkMode, Tile::BI, Tile::MATH_WARPS>;
  static constexpr int BLOCK_THREADS = [] {
    if constexpr (std::is_void_v<Schedule>) {
      return Tile::BASE_BLOCK_THREADS;
    } else if constexpr (Schedule::RAW_PIPELINE) {
      static_assert(Tile::MATH_THREADS == Schedule::MATH_THREADS);
      static_assert(Tile::BI == Schedule::BI);
      return Schedule::BLOCK_THREADS;
    } else {
      return Tile::BASE_BLOCK_THREADS + Schedule::EXTRA_THREADS;
    }
  }();
  static constexpr size_t SHARED_BYTES = [] {
    if constexpr (std::is_void_v<Schedule>) {
      return Layout::TOTAL;
    } else if constexpr (Schedule::RAW_PIPELINE) {
      return Schedule::template Layout<Layout::TOTAL>::TOTAL_BYTES;
    } else {
      return Layout::TOTAL + Schedule::EXTRA_SMEM;
    }
  }();
};
