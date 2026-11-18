// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>

#include "../dsv4_nvfp4/resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {
using Nvfp4Mma =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cute::float_e2m1_t, cute::float_e2m1_t, float,
                                                  cute::float_ue4m3_t, 16>;
using RopeMma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
// A and B fragments follow the CuTe MMA register layout. Copy atoms consume
// each lane's 16-byte shared-memory address; they perform the warp-wide LDSM.
template <class T, class Fragment>
__device__ __forceinline__ void cute_ldsm4(T const* ptr, Fragment& rA) {
  using namespace cute;
  auto sA = make_tensor(make_smem_ptr(ptr), make_layout(Int<16 / sizeof(T)>{}));
  Copy_Atom<SM75_U32x4_LDSM_N, uint32_t>{}.call(recast<uint32_t>(sA), rA);
}
template <class Fragment>
__device__ __forceinline__ void cute_load_a(const uint8_t* ptr, int stride, int lane,
                                            Fragment& rA) {
  using namespace cute;
  auto sA = make_tensor(make_smem_ptr(ptr),
                        make_layout(make_shape(_16{}, _32{}), make_stride(stride, _1{})));
  cute_ldsm4(&sA((lane & 7) + ((lane >> 3) & 1) * 8, (lane >> 4) * 16), rA);
}
template <class Fragment>
__device__ __forceinline__ void cute_load_b(const uint8_t* ptr, int stride, int lane,
                                            Fragment& rB) {
  using namespace cute;
  auto sB = make_tensor(make_smem_ptr(ptr),
                        make_layout(make_shape(_8{}, _32{}), make_stride(stride, _1{})));
  auto sVector =
      make_tensor(sB.data() + sB.layout()(lane & 7, ((lane >> 3) & 1) * 16), make_layout(_16{}));
  Copy_Atom<SM75_U32x2_LDSM_N, uint32_t>{}.call(recast<uint32_t>(sVector), rB);
}
template <class Fragment>
__device__ __forceinline__ void cute_load_rope_a(const bf16* ptr, int stride, int lane,
                                                 Fragment& rA) {
  using namespace cute;
  auto sA = make_tensor(make_smem_ptr(ptr),
                        make_layout(make_shape(_16{}, _16{}), make_stride(stride, _1{})));
  cute_ldsm4(&sA((lane & 7) + ((lane >> 3) & 1) * 8, (lane >> 4) * 8), rA);
}

// Four scale bytes cover K=64, with one scale per 16 contracted features.
// Zip each operand with its broadcast scale view for the block-scaled atom.
template <class A, class B, class C>
__device__ __forceinline__ void cute_mma_nvfp4(A const& rA, B const& rB, C& rC, uint32_t scale_a,
                                               uint32_t scale_b) {
  using namespace cute;
  auto rSA = make_tensor(&scale_a, make_layout(_1{}, _1{}));
  auto rSB = make_tensor(&scale_b, make_layout(_1{}, _1{}));
  auto sf_layout = Layout<Shape<Shape<_16, _4>>, Stride<Stride<_0, _1>>>{};
  auto rSFA = make_tensor(recast<uint8_t>(rSA).data(), sf_layout);
  auto rSFB = make_tensor(recast<uint8_t>(rSB).data(), sf_layout);
  MMA_Atom<Nvfp4Mma>{}.call(make_zip_tensor(rA, rSFA), make_zip_tensor(rB, rSFB), rC);
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
    auto rB = cute::make_tensor<uint32_t>(cute::make_layout(cute::_2{}));
    cute_load_b(k + (size_t)cand * K_STRIDE + kt * 32, K_STRIDE, lane, rB);
#pragma unroll
    for (int group = 0; group < VALID_GROUPS; ++group) {
      const int scale_row = gid + (lane & 1) * 8;
      const uint32_t scale_a = *reinterpret_cast<const uint32_t*>(
          q_sc + (group * HPB + scale_row) * Q_SCALE_STRIDE + kt * 4);
      auto rA = cute::make_tensor<uint32_t>(cute::make_layout(cute::_4{}));
      cute_load_a(q + group * HPB * Q_STRIDE + kt * 32, Q_STRIDE, lane, rA);
      auto rC = cute::make_tensor(&qk[group][0][0], cute::make_layout(cute::_4{}));
      cute_mma_nvfp4(rA, rB, rC, scale_a, scale_b);
    }
  }
}

template <int VALID_GROUPS, int P_STRIDE, int WARPS, int GROUPS, int SLOTS>
__device__ __forceinline__ void cute_pv_nvfp4(float (&acc)[GROUPS][SLOTS][2][4], const uint8_t* p,
                                              const uint8_t* p_sc, const uint8_t* vt,
                                              const uint8_t* vt_sc, int warp, int lane) {
  using namespace cute;
  auto rA = make_tensor<uint32_t>(make_layout(make_shape(_4{}, Int<GROUPS>{})));
  uint32_t scale_a[GROUPS];
  const int gid = lane >> 2;
#pragma unroll
  for (int g = 0; g < VALID_GROUPS; ++g) {
    const int row = gid + (lane & 1) * 8;
    scale_a[g] = *reinterpret_cast<const uint32_t*>(p_sc + (g * HPB + row) * NVFP4_VT_SCALE_GROUPS);
    auto rAg = rA(_, g);
    cute_load_a(p + g * HPB * P_STRIDE, P_STRIDE, lane, rAg);
  }
#pragma unroll
  for (int slot = 0; slot < SLOTS; ++slot) {
    const int scale_group = slot * WARPS + warp;
    if (scale_group >= DSV4NVFP4Cache::D_NOPE / DSV4NVFP4Cache::Scales::GROUP) continue;
    const int dim = scale_group * DSV4NVFP4Cache::Scales::GROUP;
    auto rB0 = make_tensor<uint32_t>(make_layout(_2{}));
    auto rB1 = make_tensor<uint32_t>(make_layout(_2{}));
    cute_load_b(vt + dim * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane, rB0);
    cute_load_b(vt + (dim + 8) * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane, rB1);
    const uint32_t scale_b0 =
        *reinterpret_cast<const uint32_t*>(vt_sc + (dim + gid) * NVFP4_VT_SCALE_GROUPS);
    const uint32_t scale_b1 =
        *reinterpret_cast<const uint32_t*>(vt_sc + (dim + 8 + gid) * NVFP4_VT_SCALE_GROUPS);
#pragma unroll
    for (int g = 0; g < VALID_GROUPS; ++g) {
      auto rD0 = make_tensor<float>(make_layout(_4{}));
      auto rD1 = make_tensor<float>(make_layout(_4{}));
      clear(rD0);
      clear(rD1);
      cute_mma_nvfp4(rA(_, g), rB0, rD0, scale_a[g], scale_b0);
      cute_mma_nvfp4(rA(_, g), rB1, rD1, scale_a[g], scale_b1);
      // Preserve the existing arithmetic: round each MMA result before adding
      // it to the running online-softmax accumulator.
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc[g][slot][0][v] += rD0(v);
        acc[g][slot][1][v] += rD1(v);
      }
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
