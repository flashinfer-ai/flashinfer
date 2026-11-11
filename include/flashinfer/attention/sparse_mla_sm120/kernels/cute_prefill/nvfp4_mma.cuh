// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>

#include "../dsv4_nvfp4/resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {
using Nvfp4Mma =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cute::float_e2m1_t, cute::float_e2m1_t, float,
                                                  cute::float_ue4m3_t, 16>;
using RopeMma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
struct CuteMmaResult {
  float d0, d1, d2, d3;
};

// Operand addresses are views of the shared tile. LDSM preserves the CuTe MMA
// register ownership: A=(head,feature), B=(candidate,feature), C=(head,candidate).
__device__ __forceinline__ void cute_ldsm4(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
                                           const void* ptr) {
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(ptr), a0, a1, a2, a3);
}
__device__ __forceinline__ void cute_load_a(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
                                            const uint8_t* ptr, int stride, int lane) {
  auto tile = cute::make_tensor(cute::make_smem_ptr(ptr),
                                cute::make_layout(cute::make_shape(cute::_16{}, cute::_32{}),
                                                  cute::make_stride(stride, cute::_1{})));
  cute_ldsm4(a0, a1, a2, a3, &tile((lane & 7) + ((lane >> 3) & 1) * 8, (lane >> 4) * 16));
}
__device__ __forceinline__ void cute_load_b(uint32_t& b0, uint32_t& b1, const uint8_t* ptr,
                                            int stride, int lane) {
  auto tile = cute::make_tensor(cute::make_smem_ptr(ptr),
                                cute::make_layout(cute::make_shape(cute::_8{}, cute::_32{}),
                                                  cute::make_stride(stride, cute::_1{})));
  cute::SM75_U32x2_LDSM_N::copy(
      *reinterpret_cast<const cute::uint128_t*>(&tile(lane & 7, ((lane >> 3) & 1) * 16)), b0, b1);
}
__device__ __forceinline__ void cute_load_rope_a(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                                 uint32_t& a3, const bf16* ptr, int stride,
                                                 int lane) {
  auto tile = cute::make_tensor(cute::make_smem_ptr(ptr),
                                cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}),
                                                  cute::make_stride(stride, cute::_1{})));
  cute_ldsm4(a0, a1, a2, a3, &tile((lane & 7) + ((lane >> 3) & 1) * 8, (lane >> 4) * 8));
}
__device__ __forceinline__ CuteMmaResult cute_mma_nvfp4(uint32_t a0, uint32_t a1, uint32_t a2,
                                                        uint32_t a3, uint32_t b0, uint32_t b1,
                                                        float c0, float c1, float c2, float c3,
                                                        uint32_t sa, uint32_t sb) {
  CuteMmaResult r;
  Nvfp4Mma::fma(r.d0, r.d1, r.d2, r.d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sa, sb);
  return r;
}
__device__ __forceinline__ CuteMmaResult cute_mma_rope(uint32_t a0, uint32_t a1, uint32_t a2,
                                                       uint32_t a3, uint32_t b0, uint32_t b1,
                                                       float c0, float c1, float c2, float c3) {
  CuteMmaResult r;
  RopeMma::fma(r.d0, r.d1, r.d2, r.d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
  return r;
}
template <int VALID_GROUPS, int Q_STRIDE, int Q_SCALE_STRIDE, int K_STRIDE, int K_SCALE_STRIDE,
          int KSTEPS, int GROUPS>
__device__ __forceinline__ void cute_qk_nvfp4(float (&qk)[GROUPS][1][4], const uint8_t* q,
                                              const uint8_t* q_sc, const uint8_t* k,
                                              const uint8_t* k_sc, int cand, int lane) {
  const int gid = lane >> 2;
#pragma unroll
  for (int kt = 0; kt < KSTEPS; ++kt) {
    const uint32_t scale_b =
        *reinterpret_cast<const uint32_t*>(k_sc + (size_t)(cand + gid) * K_SCALE_STRIDE + kt * 4);
    uint32_t b0, b1;
    cute_load_b(b0, b1, k + (size_t)cand * K_STRIDE + kt * 32, K_STRIDE, lane);
#pragma unroll
    for (int group = 0; group < VALID_GROUPS; ++group) {
      const int scale_row = gid + (lane & 1) * 8;
      const uint32_t scale_a = *reinterpret_cast<const uint32_t*>(
          q_sc + (group * HPB + scale_row) * Q_SCALE_STRIDE + kt * 4);
      uint32_t a0, a1, a2, a3;
      cute_load_a(a0, a1, a2, a3, q + group * HPB * Q_STRIDE + kt * 32, Q_STRIDE, lane);
      CuteMmaResult r = cute_mma_nvfp4(a0, a1, a2, a3, b0, b1, qk[group][0][0], qk[group][0][1],
                                       qk[group][0][2], qk[group][0][3], scale_a, scale_b);
      qk[group][0][0] = r.d0;
      qk[group][0][1] = r.d1;
      qk[group][0][2] = r.d2;
      qk[group][0][3] = r.d3;
    }
  }
}

template <int VALID_GROUPS, int P_STRIDE, int WARPS, int GROUPS, int SLOTS>
__device__ __forceinline__ void cute_pv_nvfp4(float (&acc)[GROUPS][SLOTS][2][4], const uint8_t* p,
                                              const uint8_t* p_sc, const uint8_t* vt,
                                              const uint8_t* vt_sc, int warp, int lane) {
  uint32_t scale_a[GROUPS], a0[GROUPS], a1[GROUPS], a2[GROUPS], a3[GROUPS];
  const int gid = lane >> 2;
#pragma unroll
  for (int g = 0; g < VALID_GROUPS; ++g) {
    const int row = gid + (lane & 1) * 8;
    scale_a[g] = *reinterpret_cast<const uint32_t*>(p_sc + (g * HPB + row) * NVFP4_VT_SCALE_GROUPS);
    cute_load_a(a0[g], a1[g], a2[g], a3[g], p + g * HPB * P_STRIDE, P_STRIDE, lane);
  }
#pragma unroll
  for (int slot = 0; slot < SLOTS; ++slot) {
    const int scale_group = slot * WARPS + warp;
    if (scale_group >= DSV4NVFP4Cache::D_NOPE / DSV4NVFP4Cache::Scales::GROUP) continue;
    const int dim = scale_group * DSV4NVFP4Cache::Scales::GROUP;
    uint32_t b00, b01, b10, b11;
    cute_load_b(b00, b01, vt + dim * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane);
    cute_load_b(b10, b11, vt + (dim + 8) * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane);
    const uint32_t scale_b0 =
        *reinterpret_cast<const uint32_t*>(vt_sc + (dim + gid) * NVFP4_VT_SCALE_GROUPS);
    const uint32_t scale_b1 =
        *reinterpret_cast<const uint32_t*>(vt_sc + (dim + 8 + gid) * NVFP4_VT_SCALE_GROUPS);
#pragma unroll
    for (int g = 0; g < VALID_GROUPS; ++g) {
      CuteMmaResult r0 = cute_mma_nvfp4(a0[g], a1[g], a2[g], a3[g], b00, b01, 0.f, 0.f, 0.f, 0.f,
                                        scale_a[g], scale_b0);
      CuteMmaResult r1 = cute_mma_nvfp4(a0[g], a1[g], a2[g], a3[g], b10, b11, 0.f, 0.f, 0.f, 0.f,
                                        scale_a[g], scale_b1);
      acc[g][slot][0][0] += r0.d0;
      acc[g][slot][0][1] += r0.d1;
      acc[g][slot][0][2] += r0.d2;
      acc[g][slot][0][3] += r0.d3;
      acc[g][slot][1][0] += r1.d0;
      acc[g][slot][1][1] += r1.d1;
      acc[g][slot][1][2] += r1.d2;
      acc[g][slot][1][3] += r1.d3;
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
