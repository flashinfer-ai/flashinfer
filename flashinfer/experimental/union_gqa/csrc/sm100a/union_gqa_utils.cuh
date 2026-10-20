/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file union_gqa_utils.cuh
 * @brief Device-side utilities for the SM100 union-GQA prefill kernel.
 *
 * Private helpers used by the union-GQA 2WG kernel. MMA wrappers issue
 * `cute::gemm` on the NOELECT
 * atoms, so the surrounding kernel hoists one elect_one_sync at the
 * MMA-warp entry.
 */
#pragma once

#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>
#include <cute/arch/simd_sm100.hpp>
#include <cute/tensor.hpp>

#include "union_gqa_mma_noelect.cuh"

namespace flashinfer::union_gqa::kernel_utils {

using namespace cute;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

// ====================================================================
// TMA copy helpers
// ====================================================================
template <typename TMA, typename TensorG, typename TensorS>
__device__ __forceinline__ void tma_copy_g2s(
    TMA const& tma_copy, TensorG const& src, TensorS const& dst, transac_bar_t& bar,
    uint32_t tx_bytes,
    cute::TMA::CacheHintSm90 cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL) {
  auto thr_tma = tma_copy.get_slice(cute::_0{});
  bar.arrive_and_expect_tx(tx_bytes);
  cute::copy(tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar),
                           static_cast<uint16_t>(0), cache_hint),
             thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

template <typename TMA, typename TensorS, typename TensorG>
__device__ __forceinline__ void tma_copy_s2g(TMA const& tma_copy, TensorS const& src,
                                             TensorG const& dst) {
  auto thr_tma = tma_copy.get_slice(cute::_0{});
  cute::copy(tma_copy, thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

__device__ __forceinline__ void tma_store_commit_group() {
#if !defined(__VSCODE_IDE__)
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void tma_store_wait_group_read() {
  static_assert(N >= 0 && N <= 7);
#if !defined(__VSCODE_IDE__)
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
#endif
}

// ====================================================================
// tcgen05 mbarrier-arrive helpers
// ====================================================================
__device__ __forceinline__ void umma_commit_arrive(transac_bar_t& bar) {
#if !defined(__VSCODE_IDE__)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
  if (cute::elect_one_sync()) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
            bar_intptr)
        : "memory");
  }
#endif
}

// Same as umma_commit_arrive but assumes the caller already wrapped the
// surrounding region in a single elect_one_sync(). Avoids nesting
// elect.sync (PTX-UB because cute::elect_one_sync uses membermask 0xFFFFFFFF
// unconditionally).
__device__ __forceinline__ void umma_commit_arrive_noelect(transac_bar_t& bar) {
#if !defined(__VSCODE_IDE__)
  uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                   bar_intptr)
               : "memory");
#endif
}

// ====================================================================
// Tensor memory load/store: 32 data-path lanes, 32-bit pattern, repeated N
// ====================================================================
template <int kNumElements>
__device__ __forceinline__ void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
  uint32_t* data = reinterpret_cast<uint32_t*>(data_);
  static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                    kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                    kNumElements == 128,
                "Invalid kNumElements");
#ifndef __VSCODE_IDE__
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (kNumElements == 1) {
      cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 2) {
      cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 4) {
      cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 8) {
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 16) {
      cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 32) {
      cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 64) {
      cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 128) {
      cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
    }
  }(cute::make_index_sequence<kNumElements>{});
#endif
}

template <int kNumElements>
__device__ __forceinline__ void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
  uint32_t const* data = reinterpret_cast<uint32_t const*>(data_);
  static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                    kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                    kNumElements == 128,
                "Invalid kNumElements");
#ifndef __VSCODE_IDE__
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (kNumElements == 1) {
      cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 2) {
      cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 4) {
      cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 8) {
      cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 16) {
      cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 32) {
      cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 64) {
      cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
    } else if constexpr (kNumElements == 128) {
      cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
    }
  }(cute::make_index_sequence<kNumElements>{});
#endif
}

__device__ __forceinline__ void tmem_load_32dp32b32x(uint32_t tmem_col, float* s) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  tmem_ld_32dp32bNx<32>(tmem_col, s);
}

// ====================================================================
// MMA wrappers via cute::gemm on NOELECT atoms (no inline PTX in MMA path).
// ====================================================================

// SMEM × SMEM → TMEM (QK).
template <typename TiledMMA, typename TensorA, typename TensorB, typename TensorC>
__device__ __forceinline__ void umma_ss(TiledMMA& tiled_mma, TensorA const& sA, TensorB const& sB,
                                        TensorC& tC, bool clear_accum) {
  tiled_mma.accumulate_ = clear_accum ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
  cute::ThrMMA thr_mma = tiled_mma.get_slice(cute::_0{});
  auto sA_frag = thr_mma.partition_fragment_A(sA);
  auto sB_frag = thr_mma.partition_fragment_B(sB);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(sA_frag); ++k) {
    cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC);
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
  }
}

// TMEM × SMEM → TMEM (PV), full K range.
template <typename TiledMMA, typename TensorA, typename TensorB, typename TensorC>
__device__ __forceinline__ void umma_ts(TiledMMA& tiled_mma, TensorA const& tA, TensorB const& sB,
                                        TensorC& tC, bool clear_accum) {
  tiled_mma.accumulate_ = clear_accum ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
  cute::ThrMMA thr_mma = tiled_mma.get_slice(cute::_0{});
  auto sB_frag = thr_mma.partition_fragment_B(sB);

  CUTE_UNROLL
  for (int k = 0; k < cute::size<2>(tA); ++k) {
    cute::gemm(tiled_mma, tA(_, _, k), sB_frag(_, _, k), tC);
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
  }
}

// TMEM × SMEM → TMEM (PV) with a split point. Issues K-atoms [0, kSplit),
// then waits on `lastsplit_bar` (the softmax bar that flags "the rest of P
// is now in tmem"), then issues K-atoms [kSplit, K).
template <int kSplit, typename TiledMMA, typename TensorA, typename TensorB, typename TensorC>
__device__ __forceinline__ void umma_ts_split(TiledMMA& tiled_mma, TensorA const& tA,
                                              TensorB const& sB, TensorC& tC,
                                              transac_bar_t& lastsplit_bar, int phase,
                                              bool clear_accum) {
  tiled_mma.accumulate_ = clear_accum ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
  cute::ThrMMA thr_mma = tiled_mma.get_slice(cute::_0{});
  auto sB_frag = thr_mma.partition_fragment_B(sB);

  constexpr int kTotal = decltype(cute::size<2>(tA))::value;
  static_assert(kSplit > 0 && kSplit < kTotal, "umma_ts_split: kSplit must be in (0, K)");

  CUTE_UNROLL
  for (int k = 0; k < kSplit; ++k) {
    cute::gemm(tiled_mma, tA(_, _, k), sB_frag(_, _, k), tC);
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
  }

  lastsplit_bar.wait(phase);
  asm volatile("tcgen05.fence::after_thread_sync;");

  CUTE_UNROLL
  for (int k = kSplit; k < kTotal; ++k) {
    cute::gemm(tiled_mma, tA(_, _, k), sB_frag(_, _, k), tC);
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
  }
}

// ====================================================================
// 3-degree minimax 2^x emulation (packed fp32x2 variant).
// Sollya-derived coefficients used selectively in the middle of the exp loop
// where MUFU.EX2 throttles.
// ====================================================================
__device__ __forceinline__ float fmax_ftz(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float2 add_rd_ftz_f32x2(float2 a, float2 b) {
  float2 r;
  asm("add.rm.ftz.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<const uint64_t&>(a)), "l"(reinterpret_cast<const uint64_t&>(b)));
  return r;
}

__device__ __forceinline__ float2 sub_rn_ftz_f32x2(float2 a, float2 b) {
  float2 r;
  asm("sub.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<const uint64_t&>(a)), "l"(reinterpret_cast<const uint64_t&>(b)));
  return r;
}

__device__ __forceinline__ float2 fma_rn_ftz_f32x2(float2 a, float2 b, float2 c) {
  float2 r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(reinterpret_cast<uint64_t&>(r))
      : "l"(reinterpret_cast<const uint64_t&>(a)), "l"(reinterpret_cast<const uint64_t&>(b)),
        "l"(reinterpret_cast<const uint64_t&>(c)));
  return r;
}

__device__ __forceinline__ float combine_int_frac_ex2(float x_rounded, float frac_ex2) {
  float r;
  asm("{\n\t"
      ".reg .s32 xri, fei, xre, oi;\n\t"
      "mov.b32 xri, %1;\n\t"
      "mov.b32 fei, %2;\n\t"
      "shl.b32 xre, xri, 23;\n\t"
      "add.s32 oi,  xre, fei;\n\t"
      "mov.b32 %0,  oi;\n\t"
      "}"
      : "=f"(r)
      : "f"(x_rounded), "f"(frac_ex2));
  return r;
}

__device__ __forceinline__ float2 ex2_emulation_3deg_pair(float2 x) {
  constexpr float kMagic = 12582912.0f;  // 2^23 + 2^22
  const float2 M = make_float2(kMagic, kMagic);

  float2 x_clamped = make_float2(fmax_ftz(x.x, -127.0f), fmax_ftz(x.y, -127.0f));
  float2 x_rounded = add_rd_ftz_f32x2(x_clamped, M);
  float2 x_rounded_back = sub_rn_ftz_f32x2(x_rounded, M);
  float2 x_frac = sub_rn_ftz_f32x2(x_clamped, x_rounded_back);

  float2 poly = make_float2(0.077119089663028717f, 0.077119089663028717f);
  poly = fma_rn_ftz_f32x2(poly, x_frac, make_float2(0.227564394474029541f, 0.227564394474029541f));
  poly = fma_rn_ftz_f32x2(poly, x_frac, make_float2(0.695146143436431885f, 0.695146143436431885f));
  poly = fma_rn_ftz_f32x2(poly, x_frac, make_float2(1.0f, 1.0f));

  return make_float2(combine_int_frac_ex2(x_rounded.x, poly.x),
                     combine_int_frac_ex2(x_rounded.y, poly.y));
}

__device__ __forceinline__ float2 mul_packed_f32x2(float2 a, float2 b) {
  // ptxas on sm_100a does not lower `mul.rn.ftz.f32x2` to a packed FMUL
  // SASS; forcing `fma.rn.ftz.f32x2 d, a, b, 0` emits the packed HFMA2.F32.
  float2 out;
  asm volatile(
      "{\n\t"
      ".reg .b64 lhs, rhs, zero, dst;\n\t"
      "mov.b64 lhs, {%2, %3};\n\t"
      "mov.b64 rhs, {%4, %5};\n\t"
      "mov.b64 zero, 0;\n\t"
      "fma.rn.ftz.f32x2 dst, lhs, rhs, zero;\n\t"
      "mov.b64 {%0, %1}, dst;\n\t"
      "}\n"
      : "=f"(out.x), "=f"(out.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return out;
}

// ====================================================================
// Row-wise reductions over the per-lane fp32 row buffer.
// ====================================================================
__device__ __forceinline__ float fmax3(float a, float b, float c) {
  float r;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

template <int N>
__device__ __forceinline__ float row_max(const float (&s)[N], float init) {
  static_assert(N % 8 == 0, "row_max expects a multiple of 8 elements");
  float local_max_0 = fmax3(init, s[0], s[1]);
  float local_max[4] = {local_max_0, fmaxf(s[2], s[3]), fmaxf(s[4], s[5]), fmaxf(s[6], s[7])};

#pragma unroll
  for (int i = 8; i < N; i += 8) {
    local_max[0] = fmax3(local_max[0], s[i], s[i + 1]);
    local_max[1] = fmax3(local_max[1], s[i + 2], s[i + 3]);
    local_max[2] = fmax3(local_max[2], s[i + 4], s[i + 5]);
    local_max[3] = fmax3(local_max[3], s[i + 6], s[i + 7]);
  }
  local_max[0] = fmaxf(local_max[0], local_max[1]);
  return fmax3(local_max[0], local_max[2], local_max[3]);
}

template <int Offset, int N, int Total>
__device__ __forceinline__ float row_max_range(const float (&s)[Total], float init) {
  static_assert(N % 8 == 0, "row_max_range expects a multiple of 8 elements");
  static_assert(Offset >= 0 && Offset + N <= Total, "row_max_range out of bounds");
  float local_max_0 = fmax3(init, s[Offset + 0], s[Offset + 1]);
  float local_max[4] = {local_max_0, fmaxf(s[Offset + 2], s[Offset + 3]),
                        fmaxf(s[Offset + 4], s[Offset + 5]), fmaxf(s[Offset + 6], s[Offset + 7])};

#pragma unroll
  for (int i = 8; i < N; i += 8) {
    local_max[0] = fmax3(local_max[0], s[Offset + i], s[Offset + i + 1]);
    local_max[1] = fmax3(local_max[1], s[Offset + i + 2], s[Offset + i + 3]);
    local_max[2] = fmax3(local_max[2], s[Offset + i + 4], s[Offset + i + 5]);
    local_max[3] = fmax3(local_max[3], s[Offset + i + 6], s[Offset + i + 7]);
  }
  local_max[0] = fmaxf(local_max[0], local_max[1]);
  return fmax3(local_max[0], local_max[2], local_max[3]);
}

template <int N>
__device__ __forceinline__ float row_sum(const float (&s)[N], float init) {
  static_assert(N % 8 == 0, "row_sum expects a multiple of 8 elements");
  float2 local_sum_0;
  cute::add(local_sum_0, make_float2(init, 0.0f), make_float2(s[0], s[1]));

  float2 local_sum[4] = {local_sum_0, make_float2(s[2], s[3]), make_float2(s[4], s[5]),
                         make_float2(s[6], s[7])};

#pragma unroll
  for (int i = 8; i < N; i += 8) {
    cute::add(local_sum[0], local_sum[0], make_float2(s[i], s[i + 1]));
    cute::add(local_sum[1], local_sum[1], make_float2(s[i + 2], s[i + 3]));
    cute::add(local_sum[2], local_sum[2], make_float2(s[i + 4], s[i + 5]));
    cute::add(local_sum[3], local_sum[3], make_float2(s[i + 6], s[i + 7]));
  }
  cute::add(local_sum[0], local_sum[0], local_sum[1]);
  cute::add(local_sum[2], local_sum[2], local_sum[3]);
  cute::add(local_sum[0], local_sum[0], local_sum[2]);
  return local_sum[0].x + local_sum[0].y;
}

}  // namespace flashinfer::union_gqa::kernel_utils
