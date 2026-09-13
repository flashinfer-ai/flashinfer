// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../arch/matrix_memory.cuh"
#include "../arch/mma_sm120.cuh"
#include "../arch/mma_sm120_nvfp4.cuh"
#include "../common/d2_load_b.cuh"
#include "../model/scale_convert.cuh"

#include "nvfp4_vt_layout.cuh"
#include "scale_mma.cuh"
#include "tile_traits.cuh"

namespace flashinfer::sparse_mla_sm120 {

template <typename KV>
__device__ __forceinline__ void qk_fp8_nope_16x8(
    float qk[4], const uint8_t* q, const float* q_sc, const uint8_t* k,
    const uint8_t* k_sc, int lane) {
  constexpr int SCALE_STRIDE = KV::SCALE_IN_KV_SMEM ? KV::KV_SMEM_STRIDE : KV::SCALE_BYTES_PER_TOKEN;
  const int gid = lane >> 2, tid = lane & 3;
#pragma unroll
  for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
    uint8_t sfa = fp32_exponent_byte(q_sc[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
    float acc0, acc1, acc2, acc3;
    init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
    uint8_t sfb = qk_k_scale_selector<KV>(k_sc + gid * SCALE_STRIDE, blk);
#pragma unroll
    for (int ks = 0; ks < KV::QUANT_TILE / 32; ks++) {
      const int ko = blk * KV::QUANT_TILE + ks * 32;
      uint32_t a0, a1, a2, a3, b0, b1;
      ldmatrix_load_a_packed_16x32_bytes(a0, a1, a2, a3, q + ko, KV::Q_NOPE_STRIDE, lane);
      ldmatrix_load_b_packed_8x32_bytes(b0, b1, k + ko, KV::KV_SMEM_STRIDE, lane);
      MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
          a0, a1, a2, a3, b0, b1, acc0, acc1, acc2, acc3, sfa, sfb);
      acc0 = r.d0;
      acc1 = r.d1;
      acc2 = r.d2;
      acc3 = r.d3;
    }
    commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, k_sc + tid * 2 * SCALE_STRIDE,
                      k_sc + (tid * 2 + 1) * SCALE_STRIDE, blk);
  }
}

template <typename KV>
__device__ __forceinline__ void qk_fp8_scale_group_16x8(
    float& acc0, float& acc1, float& acc2, float& acc3, const uint8_t* q,
    const uint8_t* k, int group, uint8_t sfa, uint8_t sfb, int lane) {
#pragma unroll
  for (int ks = 0; ks < KV::QUANT_TILE / 32; ks++) {
    const int ko = group * KV::QUANT_TILE + ks * 32;
    uint32_t a0, a1, a2, a3, b0, b1;
    ldmatrix_load_a_packed_16x32_bytes(a0, a1, a2, a3, q + ko, KV::Q_NOPE_STRIDE, lane);
    ldmatrix_load_b_packed_8x32_bytes(b0, b1, k + ko, KV::KV_SMEM_STRIDE, lane);
    MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
        a0, a1, a2, a3, b0, b1, acc0, acc1, acc2, acc3, sfa, sfb);
    acc0 = r.d0;
    acc1 = r.d1;
    acc2 = r.d2;
    acc3 = r.d3;
  }
}

template <int KV_STRIDE, int W_STRIDE, int KSTEPS, bool ROW_XOR = false>
__device__ __forceinline__ void pv_fp8_d2_16x8(float (&acc)[4], const uint8_t* w,
                                             const uint8_t* v, int dim, int lane) {
#pragma unroll
  for (int ks = 0; ks < KSTEPS; ks++) {
    uint32_t a0, a1, a2, a3, b0, b1;
    ldmatrix_load_a_packed_16x32_bytes_layout<ROW_XOR>(a0, a1, a2, a3, w + ks * 32, W_STRIDE, lane);
    d2_load_b_fp8<KV_STRIDE>(b0, b1, v, ks * 32, dim, lane);
    MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, acc[0], acc[1], acc[2], acc[3]);
    acc[0] = r.d0;
    acc[1] = r.d1;
    acc[2] = r.d2;
    acc[3] = r.d3;
  }
}

template <int KV_STRIDE, int W_STRIDE, int KSTEPS, bool ROW_XOR>
__device__ __forceinline__ void pv_fp8_d2_16x8_pair(
    float (&acc0)[4], float (&acc1)[4], const uint8_t* w0, const uint8_t* w1,
    const uint8_t* v, int dim, int lane) {
#pragma unroll
  for (int ks = 0; ks < KSTEPS; ks++) {
    uint32_t b0, b1;
    d2_load_b_fp8<KV_STRIDE>(b0, b1, v, ks * 32, dim, lane);
    uint32_t a00, a01, a02, a03, a10, a11, a12, a13;
    ldmatrix_load_a_packed_16x32_bytes_layout<ROW_XOR>(a00, a01, a02, a03, w0 + ks * 32, W_STRIDE, lane);
    ldmatrix_load_a_packed_16x32_bytes_layout<ROW_XOR>(a10, a11, a12, a13, w1 + ks * 32, W_STRIDE, lane);
    MmaFp8Result r0 = mma_fp8_m16n8k32(a00, a01, a02, a03, b0, b1, acc0[0], acc0[1], acc0[2], acc0[3]);
    acc0[0] = r0.d0;
    acc0[1] = r0.d1;
    acc0[2] = r0.d2;
    acc0[3] = r0.d3;
    MmaFp8Result r1 = mma_fp8_m16n8k32(a10, a11, a12, a13, b0, b1, acc1[0], acc1[1], acc1[2], acc1[3]);
    acc1[0] = r1.d0;
    acc1[1] = r1.d1;
    acc1[2] = r1.d2;
    acc1[3] = r1.d3;
  }
}

template <typename KV, int GROUPS>
__device__ __forceinline__ void qk_bf16_from_fp8_nope_16x8(
    float (&qk)[GROUPS][4], const bf16* q, int q_group_stride,
    const uint8_t* k_gid, const uint8_t* k_scale_gid, int lane) {
  static_assert(GROUPS == 1 || GROUPS == 2);
  const int tid = lane & 3;
#pragma unroll
  for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
    float scale_f = kv_scale_fp32<KV>(k_scale_gid, blk);
#pragma unroll
    for (int ks = 0; ks < KV::QUANT_TILE / 16; ks++) {
      const int ko = blk * KV::QUANT_TILE + ks * 16;
      uint16_t p0 = *reinterpret_cast<const uint16_t*>(k_gid + ko + 2 * tid);
      uint16_t p1 = *reinterpret_cast<const uint16_t*>(k_gid + ko + 2 * tid + 8);
      uint32_t f16x2_0, f16x2_1;
      asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_0) : "h"(p0));
      asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2_1) : "h"(p1));
      __half2 h2_0 = *reinterpret_cast<__half2*>(&f16x2_0);
      __half2 h2_1 = *reinterpret_cast<__half2*>(&f16x2_1);
      float fk0 = __low2float(h2_0) * scale_f, fk1 = __high2float(h2_0) * scale_f;
      float fk2 = __low2float(h2_1) * scale_f, fk3 = __high2float(h2_1) * scale_f;
      uint32_t b0, b1;
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b0) : "f"(fk1), "f"(fk0));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b1) : "f"(fk3), "f"(fk2));
#pragma unroll
      for (int g = 0; g < GROUPS; g++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, q + g * q_group_stride + ko, KV::Q_NOPE_BF16_STRIDE, lane);
        MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[g][0], qk[g][1], qk[g][2], qk[g][3]);
        qk[g][0] = r.d0;
        qk[g][1] = r.d1;
        qk[g][2] = r.d2;
        qk[g][3] = r.d3;
      }
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <int VALID_GROUPS, int Q_STRIDE, int Q_SCALE_STRIDE, int K_STRIDE,
          int K_SCALE_STRIDE, int KSTEPS, int GROUPS>
__device__ __forceinline__ void qk_nvfp4_nope_16x8(
    float (&qk)[GROUPS][1][4], const uint8_t* q, const uint8_t* q_sc,
    const uint8_t* k, const uint8_t* k_sc, int cand, int lane) {
  const int gid = lane >> 2;
#pragma unroll
  for (int kt = 0; kt < KSTEPS; ++kt) {
    const uint32_t scale_b = *reinterpret_cast<const uint32_t*>(
        k_sc + (size_t)(cand + gid) * K_SCALE_STRIDE + kt * 4);
    uint32_t b0, b1;
    ldmatrix_load_b_packed_8x32_bytes(b0, b1, k + (size_t)cand * K_STRIDE + kt * 32,
                                    K_STRIDE, lane);
#pragma unroll
    for (int group = 0; group < VALID_GROUPS; ++group) {
      const int scale_row = gid + (lane & 1) * 8;
      const uint32_t scale_a = *reinterpret_cast<const uint32_t*>(
          q_sc + (group * HPB + scale_row) * Q_SCALE_STRIDE + kt * 4);
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_a_packed_16x32_bytes(a0, a1, a2, a3,
          q + group * HPB * Q_STRIDE + kt * 32, Q_STRIDE, lane);
      MmaNvfp4Result r = mma_nvfp4_block_scaled_m16n8k64(
          a0, a1, a2, a3, b0, b1, qk[group][0][0], qk[group][0][1],
          qk[group][0][2], qk[group][0][3], scale_a, scale_b);
      qk[group][0][0] = r.d0;
      qk[group][0][1] = r.d1;
      qk[group][0][2] = r.d2;
      qk[group][0][3] = r.d3;
    }
  }
}

template <int VALID_GROUPS, int P_STRIDE, int WARPS, int GROUPS, int SLOTS>
__device__ __forceinline__ void pv_nvfp4_vt_16x16(
    float (&acc)[GROUPS][SLOTS][2][4], const uint8_t* p, const uint8_t* p_sc,
    const uint8_t* vt, const uint8_t* vt_sc, int warp, int lane) {
  uint32_t scale_a[GROUPS], a0[GROUPS], a1[GROUPS], a2[GROUPS], a3[GROUPS];
  const int gid = lane >> 2;
#pragma unroll
  for (int g = 0; g < VALID_GROUPS; ++g) {
    const int row = gid + (lane & 1) * 8;
    scale_a[g] = *reinterpret_cast<const uint32_t*>(p_sc + (g * HPB + row) * NVFP4_VT_SCALE_GROUPS);
    ldmatrix_load_a_packed_16x32_bytes(a0[g], a1[g], a2[g], a3[g], p + g * HPB * P_STRIDE, P_STRIDE, lane);
  }
#pragma unroll
  for (int slot = 0; slot < SLOTS; ++slot) {
    const int scale_group = slot * WARPS + warp;
    if (scale_group >= DSV4NVFP4Cache::D_NOPE / DSV4NVFP4Cache::Scales::GROUP) continue;
    const int dim = scale_group * DSV4NVFP4Cache::Scales::GROUP;
    uint32_t b00, b01, b10, b11;
    ldmatrix_load_b_packed_8x32_bytes(b00, b01, vt + dim * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane);
    ldmatrix_load_b_packed_8x32_bytes(b10, b11, vt + (dim + 8) * NVFP4_VT_PACKED_K_BYTES, NVFP4_VT_PACKED_K_BYTES, lane);
    const uint32_t scale_b0 = *reinterpret_cast<const uint32_t*>(vt_sc + (dim + gid) * NVFP4_VT_SCALE_GROUPS);
    const uint32_t scale_b1 = *reinterpret_cast<const uint32_t*>(vt_sc + (dim + 8 + gid) * NVFP4_VT_SCALE_GROUPS);
#pragma unroll
    for (int g = 0; g < VALID_GROUPS; ++g) {
      MmaNvfp4Result r0 = mma_nvfp4_block_scaled_m16n8k64(
          a0[g], a1[g], a2[g], a3[g], b00, b01, 0.f, 0.f, 0.f, 0.f, scale_a[g], scale_b0);
      MmaNvfp4Result r1 = mma_nvfp4_block_scaled_m16n8k64(
          a0[g], a1[g], a2[g], a3[g], b10, b11, 0.f, 0.f, 0.f, 0.f, scale_a[g], scale_b1);
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
