/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_W8A16_SM100_CONVERT_HPP_
#define FLASHINFER_GEMM_W8A16_SM100_CONVERT_HPP_

// e4m3 -> 16-bit conversion for the Blackwell W8A16 transform warps.
//
// The mixed-input transform converts in packs of 4 (gcd(fragment, 32 / 8)); the
// NumericArrayConverter specializations below are more specialized than CUTLASS's
// <T, float_e4m3_t, N, Round> and are declared before any instantiation in the module, so
// they take over for every matching instantiation in it.  The Hopper module is a separate
// JIT build with its own converter.
//
// Numerical contract: exact for every finite e4m3 code, denormals included.  The two
// e4m3fn NaN encodings (0x7F, 0xFF) are not handled: the ALU path turns them into +-480,
// the native cvt into NaN.  Callers must hand the kernel a finite weight.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace flashinfer::gemm::w8a16_sm100_detail {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Version B10: the native packed converter.  Blackwell has a one-instruction e4m3x2 ->
// bf16x2 convert (PTX ISA 9.2, `cvt.rn.bf16x2.e4m3x2`, SASS F2FP.BF16.E4M3.UNPACK_B);
// one F2FP per pair replaces the four-instruction ALU chain below.  PerfSim on version B9
// (gb100, 2SM 256x192x128, M=192 N=256 K=4096) showed why that chain matters: each
// transform warp needs ~200 instructions and ~750 cycles per 128-deep k-block, the MMA
// takes 768, and the TMEM-store -> mbarrier -> 2-CTA barrier -> MMA-issue handoff adds a
// fixed ~226 cycles, so the convert chain runs ~50 cycles longer than the MMA per
// k-block and the tensor pipe idles 6%.  Halving the chain puts the convert back behind
// the MMA.  Exact for every finite e4m3 code (all are representable in bf16, denormals
// included); byte 0 lands in lo[15:0].
//
// The ALU path is kept for toolchains older than CUDA 13.2 (no PTX 9.2) and for
// -DW8A16_FORCE_ALU_CONVERT (A/B testing).
#if !defined(W8A16_FORCE_ALU_CONVERT) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define W8A16_NATIVE_E4M3_BF16_CVT 1
#else
#define W8A16_NATIVE_E4M3_BF16_CVT 0
#endif

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept.
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
#if W8A16_NATIVE_E4M3_BF16_CVT
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
#else
  // Version B1 ALU path.  prmt selector nibbles 0x9180 / 0xB3A2: byte k into the low byte
  // of lane k and its sign replicated into the high byte (selector bit 3).  (v << 4)
  // puts the 7 magnitude bits at [10:4] and the replicated sign at [15], mask 0x87F0
  // keeps exactly those.  The exponent field then holds e4m3's raw exponent, so
  // multiply by 2^(127 - 7) = 2^120 (bf16 0x7B80).  Denormals: field 0 -> bf16 denormal
  // (m / 8) * 2^-126, times 2^120 = (m / 8) * 2^-6, the e4m3 denormal value.
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 4) & 0x87F087F0u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 4) & 0x87F087F0u;
  const uint32_t k_bits = 0x7B807B80u;  // bf16x2 {2^120, 2^120}
  const __nv_bfloat162 k = *reinterpret_cast<const __nv_bfloat162*>(&k_bits);
  __nv_bfloat162 l = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo_s), k);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
#endif
}

// Same for f16.  `cvt.rn.f16x2.e4m3x2` has existed since sm_89 (PTX ISA 7.8), so the
// native form needs no toolchain guard; the ALU path ((v << 7) puts the magnitude at
// [13:7] and the replicated sign at [15], mask 0xBF80, bias fix 2^(15 - 7) = 2^8,
// f16 0x5C00) stays behind W8A16_FORCE_ALU_CONVERT.
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
#if !defined(W8A16_FORCE_ALU_CONVERT)
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
#else
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 7) & 0xBF80BF80u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 7) & 0xBF80BF80u;
  const uint32_t k_bits = 0x5C005C00u;  // f16x2 {256, 256}
  const __half2 k = *reinterpret_cast<const __half2*>(&k_bits);
  __half2 l = __hmul2(*reinterpret_cast<const __half2*>(&lo_s), k);
  __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
#endif
}

}  // namespace flashinfer::gemm::w8a16_sm100_detail

namespace cutlass {

template <FloatRoundStyle Round>
struct NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e4m3_t, 4, Round> {
  using result_element = cutlass::bfloat16_t;
  using source_element = cutlass::float_e4m3_t;
  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type out;
    uint32_t* o = reinterpret_cast<uint32_t*>(&out);
    flashinfer::gemm::w8a16_sm100_detail::e4m3x4_to_bf16x4(
        *reinterpret_cast<uint32_t const*>(&source), o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

template <FloatRoundStyle Round>
struct NumericArrayConverter<cutlass::half_t, cutlass::float_e4m3_t, 4, Round> {
  using result_element = cutlass::half_t;
  using source_element = cutlass::float_e4m3_t;
  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  CUTLASS_DEVICE
  static result_type convert(source_type const& source) {
    result_type out;
    uint32_t* o = reinterpret_cast<uint32_t*>(&out);
    flashinfer::gemm::w8a16_sm100_detail::e4m3x4_to_f16x4(
        *reinterpret_cast<uint32_t const*>(&source), o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

}  // namespace cutlass

#endif  // FLASHINFER_GEMM_W8A16_SM100_CONVERT_HPP_
