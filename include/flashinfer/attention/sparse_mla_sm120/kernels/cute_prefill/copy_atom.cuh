// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/atom/copy_atom.hpp>

namespace cute {
// Raw m16n16 transpose, without SM100_U8x16_LDSM_T's STSM-oriented byte repack.
// Maps V[candidate,feature] directly into m16n8k32 A=V^T register ownership.
struct SM120_U8x16_LDSM_T_RAW {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];
  CUTE_DEVICE static void copy(uint128_t const& src, uint32_t& d0, uint32_t& d1,
                               uint32_t& d2, uint32_t& d3) {
    uint32_t address = cast_smem_ptr_to_uint(&src);
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(address));
  }
};
template <> struct Copy_Traits<SM120_U8x16_LDSM_T_RAW> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32, _128>, Stride<_128, _1>>;
  using DstLayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _4, _2, _2>>,
                           Stride<Stride<_512, _8>, Stride<_1, _128, _64, _2048>>>;
  using RefLayout = DstLayout;
};
}  // namespace cute
