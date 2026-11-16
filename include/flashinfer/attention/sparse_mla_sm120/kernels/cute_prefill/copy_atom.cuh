// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/tensor.hpp>

namespace cute {
// Raw m16n16 transpose, without SM100_U8x16_LDSM_T's STSM-oriented byte repack.
// Maps V[candidate,feature] directly into m16n8k32 A=V^T register ownership.
//
// One instruction reads two row-major 16x16 byte matrices X[b][r][c].
// Source address supplied by lane t: &X[t / 16][t % 16][0].
// Let row = 4*(t % 4), col = t / 4. Bytes within each 32-bit register:
//
//              byte 0          byte 1          byte 2          byte 3
//   d0       X[0][row+0][col] X[0][row+1][col] X[0][row+2][col] X[0][row+3][col]
//   d1       X[0][row+0][col+8] ...             ...             X[0][row+3][col+8]
//   d2       X[1][row+0][col] X[1][row+1][col] X[1][row+2][col] X[1][row+3][col]
//   d3       X[1][row+0][col+8] ...             ...             X[1][row+3][col+8]
//
// For example, lane 0 receives rows 0:4 at columns 0 and 8 of both matrices.
//
// Raw transpose ownership             CUTLASS STSM-oriented repack
//   d0 = [a0 a1 a2 a3]  ----+---->     dst0 = [a0 a1 b0 b1]
//   d1 = [b0 b1 b2 b3]  ----+---->     dst1 = [a2 a3 b2 b3]
//   d2/d3 are repacked the same way into dst2/dst3.
// PV's MMA A fragment needs the raw column bytes, so no repack is applied.
// DstLayout maps (lane, register-bit) to the source bit offset in X;
// partition_S composes it with V's swizzled shared-memory layout.
struct SM120_U8x16_LDSM_T_RAW {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];
  CUTE_DEVICE static void copy(uint128_t const& src, uint32_t& d0, uint32_t& d1, uint32_t& d2,
                               uint32_t& d3) {
    uint32_t address = cast_smem_ptr_to_uint(&src);
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(address));
  }
};
template <>
struct Copy_Traits<SM120_U8x16_LDSM_T_RAW> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32, _128>, Stride<_128, _1>>;
  using DstLayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _4, _2, _2>>,
                           Stride<Stride<_512, _8>, Stride<_1, _128, _64, _2048>>>;
  using RefLayout = DstLayout;
};
}  // namespace cute
