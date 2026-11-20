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

// Cache addressing for the scalar BF16 gathers. DSV4_1 splits an index into a
// footer page and a slot at the runtime page size; DSV4_1_INLINE indices are
// flat rows of inline records at the row pitch `stride`.
template <ModelType MT>
struct Dsv41Bf16Cache {
  static_assert(is_dsv41(MT));
  static constexpr bool INLINE = KVCacheTraits<MT>::SCALE_IN_KV_SMEM;
  using Fp8 = Dsv41Fp8LayoutT<INLINE>;
  using Fp4 = Dsv41Fp4LayoutT<INLINE>;

  // Returns the page (footer) or record (inline) holding `index`, and its
  // slot there for the layouts' offset helpers.
  __device__ __forceinline__ static const uint8_t* locate(const uint8_t* cache, int index, int pbs,
                                                          size_t stride, bool power_of_two,
                                                          size_t& slot) {
    if constexpr (INLINE) {
      slot = 0;
      return cache + (size_t)index * stride;
    } else {
      const int page = power_of_two ? (index >> (__ffs(pbs) - 1)) : (index / pbs);
      slot = size_t(index - page * pbs);
      return cache + (size_t)page * stride;
    }
  }
  __device__ __forceinline__ static size_t data_offset(bool fp4, size_t slot) {
    return Fp4::template selected_data_offset<Fp8>(fp4, slot);
  }
  __device__ __forceinline__ static size_t scale_offset(bool fp4, int pbs, size_t slot) {
    return Fp4::template selected_scale_offset<Fp8>(fp4, size_t(pbs), slot);
  }
};

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16
