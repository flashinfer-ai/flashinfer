// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../model/kv_cache_traits.cuh"

static constexpr int HPB = 16;

template <ScaleFormat SF>
struct WeightFp8PassTraits {
  static constexpr int PASSES = (SF == ScaleFormat::ARBITRARY_FP32) ? 2 : 1;
};

// V_CHUNK follows the cache scale group. BF16 QK traits retain layout constants;
// they do not select the independent full-BF16 attention implementation.
template <ModelType MT, QkComputeMode QkMode, int TILE_BI, int TILE_MATH_WARPS>
struct ComputeTraits;

template <ModelType MT, int TILE_BI, int TILE_MATH_WARPS>
struct ComputeTraits<MT, QkComputeMode::FP8, TILE_BI, TILE_MATH_WARPS> {
  using KV = KVCacheTraits<MT>;
  static constexpr int V_CHUNK = KV::QUANT_TILE;
  static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;
  static constexpr int V_TRANS_STRIDE = TILE_BI + 16;
  static constexpr int W_FP8_STRIDE = TILE_BI + 16;
  static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / TILE_MATH_WARPS;
  static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;
  static constexpr int XV_KSTEPS = TILE_BI / 32;
  static_assert(NT_PER_WARP_XV >= 1, "V chunk too narrow for this math-warp count");
};

template <ModelType MT, int TILE_BI, int TILE_MATH_WARPS>
struct ComputeTraits<MT, QkComputeMode::BF16, TILE_BI, TILE_MATH_WARPS> {
  using KV = KVCacheTraits<MT>;
  static constexpr int V_CHUNK = KV::QUANT_TILE;
  static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;
  static constexpr int V_TRANS_STRIDE = TILE_BI + 8;
  static constexpr int W_FP8_STRIDE = 0;
  static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / TILE_MATH_WARPS;
  static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;
  static constexpr int XV_KSTEPS = TILE_BI / 16;
  static_assert(NT_PER_WARP_XV >= 1, "V chunk too narrow for this math-warp count");
};
