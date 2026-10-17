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
#ifndef FLASHINFER_GEMM_W8A16_SM90_CONVERT_HPP_
#define FLASHINFER_GEMM_W8A16_SM90_CONVERT_HPP_

// e4m3 -> 16-bit conversion for the Hopper W8A16 mainloop.
//
// CUTLASS's register-sourced mixed-input mainloop converts the weight fragment with
// NumericArrayConverter<T, float_e4m3_t, 4> (packs of gcd(fragment, 32 / 8) = 4).  The
// specializations below replace CUTLASS's own for exactly that signature; they must be
// visible before any instantiation in the translation unit, and they apply to every
// matching instantiation in it.  The Hopper and Blackwell modules are separate JIT
// builds, so their converters never meet.
//
// Numerical contract: exact for every finite e4m3 code, denormals included.  The two
// e4m3fn NaN encodings (0x7F, 0xFF) are not handled: the bf16 path turns them into a
// finite value (+-480), the f16 path into NaN.  Callers must hand the kernel a finite
// weight; the Python op documents this.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

namespace flashinfer::gemm::w8a16_sm90_detail {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Four e4m3 bytes (little-endian in `q`) -> two bf16x2 registers, value order kept.
// Hopper has no packed e4m3 -> bf16 convert (that is sm_100's cvt.rn.bf16x2.e4m3x2);
// CUTLASS's own converter goes e4m3x2 -> f16x2 -> 2 x f32 -> bf16x2, four dependent
// cvt per pair.  This is the ALU path of the hand kernel / Blackwell version B1: a
// sign-replicating prmt puts byte k in the low byte of lane k with its sign copied
// into the high byte, (v << 4) & 0x87F0 leaves the sign at [15] and the 7 magnitude
// bits at [10:4], so the bf16 exponent field holds e4m3's raw exponent, and one
// bf16x2 multiply by 2^(127 - 7) fixes the bias.  Exact for every finite e4m3 code,
// denormals included (they land as bf16 denormals and the multiply rescales them).
__device__ __forceinline__ void e4m3x4_to_bf16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  const uint32_t lo_s = (prmt_b32(q, 0u, 0x9180u) << 4) & 0x87F087F0u;
  const uint32_t hi_s = (prmt_b32(q, 0u, 0xB3A2u) << 4) & 0x87F087F0u;
  const uint32_t k_bits = 0x7B807B80u;  // bf16x2 {2^120, 2^120}
  const __nv_bfloat162 k = *reinterpret_cast<const __nv_bfloat162*>(&k_bits);
  __nv_bfloat162 l = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo_s), k);
  __nv_bfloat162 h = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi_s), k);
  lo = *reinterpret_cast<uint32_t*>(&l);
  hi = *reinterpret_cast<uint32_t*>(&h);
}

// f16 has the native packed convert since sm_89 (PTX ISA 7.8).
__device__ __forceinline__ void e4m3x4_to_f16x4(uint32_t q, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, l;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, h;\n\t}"
      : "=r"(lo), "=r"(hi)
      : "r"(q));
}

}  // namespace flashinfer::gemm::w8a16_sm90_detail

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
    flashinfer::gemm::w8a16_sm90_detail::e4m3x4_to_bf16x4(
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
    flashinfer::gemm::w8a16_sm90_detail::e4m3x4_to_f16x4(
        *reinterpret_cast<uint32_t const*>(&source), o[0], o[1]);
    return out;
  }
  CUTLASS_DEVICE
  result_type operator()(source_type const& s) const { return convert(s); }
};

}  // namespace cutlass

#endif  // FLASHINFER_GEMM_W8A16_SM90_CONVERT_HPP_
