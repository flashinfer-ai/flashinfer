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
#ifndef FLASHINFER_GEMM_W8A16_CONVERT_HPP_
#define FLASHINFER_GEMM_W8A16_CONVERT_HPP_

// e4m3 -> 16-bit conversion of the W8A16 weight fragments, for both the Hopper mainloop
// (register-sourced mixed input: each consumer warpgroup converts in front of wgmma) and
// the Blackwell transform warpgroups (convert between the TMA load and tcgen05.mma).
//
// Both CUTLASS mainloops convert with NumericArrayConverter<T, float_e4m3_t, 4> (packs of
// gcd(fragment, 32 / 8) = 4).  The specializations at the bottom are more specialized than
// CUTLASS's <T, float_e4m3_t, N, Round> and are visible before any instantiation in a
// translation unit that includes this header, so they take over for every matching
// instantiation in it.  Each architecture is a separate JIT module.
//
// Numerical contract: exact for every finite e4m3 code, denormals included.  The two
// e4m3fn NaN encodings (0x7F, 0xFF) are not handled: the ALU bf16 path turns them into a
// finite value (+-480), the native cvt paths into NaN.  Callers must hand the kernel a
// finite weight; the Python op documents this.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

// `cvt.rn.bf16x2.e4m3x2` (PTX ISA 9.2, sm_100+; SASS F2FP.BF16.E4M3.UNPACK_B) converts a
// pair in one instruction.  It needs CUDA 13.2 or newer and a Blackwell target; the
// Hopper module and older toolchains take the ALU path below.
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
#define W8A16_NATIVE_E4M3_BF16_CVT 1
#else
#define W8A16_NATIVE_E4M3_BF16_CVT 0
#endif

namespace flashinfer::gemm::w8a16_convert {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept, without
// a packed e4m3 -> bf16 instruction.  Four ALU instructions per pair instead of CUTLASS's
// e4m3x2 -> f16x2 -> 2 x f32 -> bf16x2 chain: prmt selector nibbles 0x9180 / 0xB3A2 put
// byte k in the low byte of lane k with its sign replicated into the high byte (selector
// bit 3); (v << 4) puts the 7 magnitude bits at [10:4] and the replicated sign at [15],
// mask 0x87F0 keeps exactly those.  The bf16 exponent field then holds e4m3's raw
// exponent, so one bf16x2 multiply by 2^(127 - 7) = 2^120 (bf16 0x7B80) fixes the bias.
// Denormals: field 0 -> bf16 denormal (m / 8) * 2^-126, times 2^120 = (m / 8) * 2^-6, the
// e4m3 denormal value.  Exact for every finite e4m3 code.
__device__ __forceinline__ void e4m3x4_to_bf16x4_alu(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 4) & 0x87F087F0u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 4) & 0x87F087F0u;
  const uint32_t k_bits = 0x7B807B80u;  // bf16x2 {2^120, 2^120}
  const __nv_bfloat162 k = *reinterpret_cast<const __nv_bfloat162*>(&k_bits);
  __nv_bfloat162 l = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo_s), k);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
}

// Four e4m3 bytes -> two bf16x2 registers.  On Blackwell the transform warps run this in
// the shadow of the MMA: per 128-deep k-block a transform warp needs about as many cycles
// as the MMA itself plus a fixed TMEM-store -> mbarrier -> 2-CTA barrier -> MMA-issue
// handoff, and the four-instruction ALU chain is slightly longer than the MMA per k-block,
// which idles the tensor pipe; the one-instruction convert puts the transform back behind
// the MMA.  Byte 0 lands in lo[15:0].
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
#if W8A16_NATIVE_E4M3_BF16_CVT && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
#else
  e4m3x4_to_bf16x4_alu(q, lo, hi);
#endif
}

// Four e4m3 bytes -> two f16x2 registers: `cvt.rn.f16x2.e4m3x2` exists since sm_89 (PTX
// ISA 7.8), so both architectures use it unguarded.
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
}

}  // namespace flashinfer::gemm::w8a16_convert

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
    flashinfer::gemm::w8a16_convert::e4m3x4_to_bf16x4(*reinterpret_cast<uint32_t const*>(&source),
                                                      o[0], o[1]);
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
    flashinfer::gemm::w8a16_convert::e4m3x4_to_f16x4(*reinterpret_cast<uint32_t const*>(&source),
                                                     o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

}  // namespace cutlass

#endif  // FLASHINFER_GEMM_W8A16_CONVERT_HPP_
