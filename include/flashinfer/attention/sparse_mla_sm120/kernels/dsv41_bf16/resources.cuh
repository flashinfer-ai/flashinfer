// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../model/kv_cache_traits.cuh"

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16 {

struct Dsv41Bf16Resources {
  using Geometry = Dsv41Geometry;
  static constexpr int HEADS_PER_CTA = 16;
  static constexpr int CANDIDATES = 64;
  static constexpr int WARPS = 8;
  static constexpr int WARP_THREADS = 32;
  static constexpr int BLOCK_THREADS = WARPS * WARP_THREADS;
  static constexpr int VECTOR_ELEMS = sizeof(uint4) / sizeof(bf16);
  static constexpr int QK_STRIDE_ELEMS = Geometry::D_QK + VECTOR_ELEMS;
  static constexpr int P_STRIDE_ELEMS = CANDIDATES + VECTOR_ELEMS;
  static constexpr int QK_VECTORS = Geometry::D_QK / VECTOR_ELEMS;
  static constexpr int MMA_K = 16;
  static constexpr int MMA_N = 8;
  static constexpr int PV_TILES_PER_WARP = Geometry::D_V / (WARPS * MMA_N);
};

struct Dsv41Bf16Smem {
  using R = Dsv41Bf16Resources;
  bf16 q[R::HEADS_PER_CTA][R::QK_STRIDE_ELEMS];
  bf16 kv[R::CANDIDATES][R::QK_STRIDE_ELEMS];
  bf16 p[R::HEADS_PER_CTA][R::P_STRIDE_ELEMS];
  float reduce[2][R::WARPS][R::HEADS_PER_CTA];
  int valid[R::CANDIDATES];
};
static_assert(sizeof(Dsv41Bf16Smem) == 86784);

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16
