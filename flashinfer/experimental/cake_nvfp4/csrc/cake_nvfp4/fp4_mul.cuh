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
#pragma once
// NVFP4 dequantization for the decode kernels: E2M1 values times their E4M3 block scales, rounded
// to nearest even and saturated, as E4M3 values.
//
// The ptxas of the CUDA 13.4 toolkit assembles this as one instruction per four values. Older
// ones reject it (13.0 and 13.2: "Illegal modifier '.satfinite' for instruction 'mul'", "Unknown
// modifier '.e2m1p4x4'"), so for them the same product goes through f16: both factors convert
// exactly, the product has at most 6 significant bits and is exact too, and the one rounding left
// is the conversion to E4M3. The result is bit-identical for every input; the cost is six
// instructions per four values instead of one, plus the conversion of the scales, which Fp4Scale
// lets a caller do once per scale word.
#include <cstdint>

#ifndef CAKE_NVFP4_FUSED_FP4_MUL
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
#define CAKE_NVFP4_FUSED_FP4_MUL 1
#else
#define CAKE_NVFP4_FUSED_FP4_MUL 0
#endif
#endif

namespace flashinfer {
namespace cake_nvfp4 {
namespace fp4 {

#if CAKE_NVFP4_FUSED_FP4_MUL

// Four E4M3 scales, one per byte.
using Fp4Scale = uint32_t;

__device__ __forceinline__ Fp4Scale fp4Scale(uint32_t sf) { return sf; }

// The same scale in all four bytes.
__device__ __forceinline__ Fp4Scale fp4ScaleUniform(uint32_t sf) { return sf; }

// FP4 (packed, 4 values in 16 bits) x E4M3 scale (x4) -> 4 x E4M3.
__device__ __forceinline__ uint32_t mulFp4x4Packed(uint16_t fp4, Fp4Scale sf) {
  uint32_t r;
  asm("mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, %1, %2;" : "=r"(r) : "h"(fp4), "r"(sf));
  return r;
}

// FP4 (packed, 8 values in 32 bits) x E4M3 scale (x4, applied to each half) -> 8 x E4M3.
__device__ __forceinline__ void mulFp4x8Packed(uint32_t fp4, Fp4Scale sf, uint32_t& lo,
                                               uint32_t& hi) {
  lo = mulFp4x4Packed(static_cast<uint16_t>(fp4 & 0xFFFFu), sf);
  hi = mulFp4x4Packed(static_cast<uint16_t>(fp4 >> 16), sf);
}

// FP4 (one value in the low nibble of each byte) x 4 E4M3 scales -> 4 x E4M3.
__device__ __forceinline__ uint32_t mulFp4x4Unpacked(uint32_t fp4, Fp4Scale sf) {
  uint32_t r;
  asm("mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 %0, %1, %2;" : "=r"(r) : "r"(fp4), "r"(sf));
  return r;
}

#else  // toolkits older than CUDA 13.4

// Four E4M3 scales as two f16x2 words (scales 0-1 and 2-3).
struct Fp4Scale {
  uint32_t lo, hi;
};

__device__ __forceinline__ Fp4Scale fp4Scale(uint32_t sf) {
  Fp4Scale s;
  asm("{\n\t.reg .b16 s0, s1;\n\tmov.b32 {s0, s1}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, s0;\n\tcvt.rn.f16x2.e4m3x2 %1, s1;\n}"
      : "=r"(s.lo), "=r"(s.hi)
      : "r"(sf));
  return s;
}

// The same scale in all four bytes: one conversion serves both words.
__device__ __forceinline__ Fp4Scale fp4ScaleUniform(uint32_t sf) {
  Fp4Scale s;
  asm("{\n\t.reg .b16 s0, s1;\n\tmov.b32 {s0, s1}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, s0;\n}"
      : "=r"(s.lo)
      : "r"(sf));
  s.hi = s.lo;
  return s;
}

// FP4 (packed, 8 values in 32 bits) x E4M3 scale (x4, applied to each half) -> 8 x E4M3.
__device__ __forceinline__ void mulFp4x8Packed(uint32_t fp4, Fp4Scale sf, uint32_t& lo,
                                               uint32_t& hi) {
  asm("{\n\t.reg .b8 a0, a1, a2, a3;\n\t.reg .b32 h0, h1, h2, h3;\n\t.reg .b16 e0, e1, e2, e3;\n\t"
      "mov.b32 {a0, a1, a2, a3}, %2;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, a0;\n\tcvt.rn.f16x2.e2m1x2 h1, a1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h2, a2;\n\tcvt.rn.f16x2.e2m1x2 h3, a3;\n\t"
      "mul.rn.f16x2 h0, h0, %3;\n\tmul.rn.f16x2 h1, h1, %4;\n\t"
      "mul.rn.f16x2 h2, h2, %3;\n\tmul.rn.f16x2 h3, h3, %4;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\tcvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\tcvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
      "mov.b32 %0, {e0, e1};\n\tmov.b32 %1, {e2, e3};\n}"
      : "=r"(lo), "=r"(hi)
      : "r"(fp4), "r"(sf.lo), "r"(sf.hi));
}

// FP4 (packed, 4 values in 16 bits) x E4M3 scale (x4) -> 4 x E4M3.
__device__ __forceinline__ uint32_t mulFp4x4Packed(uint16_t fp4, Fp4Scale sf) {
  uint32_t lo, hi;
  mulFp4x8Packed(fp4, sf, lo, hi);
  return lo;
}

// FP4 (one value in the low nibble of each byte) x 4 E4M3 scales -> 4 x E4M3.
__device__ __forceinline__ uint32_t mulFp4x4Unpacked(uint32_t fp4, Fp4Scale sf) {
  // Values 0-1 pack into byte 0 and values 2-3 into byte 2; bytes 1 and 3 are not read.
  uint32_t const nib = fp4 & 0x0F0F0F0Fu;
  uint32_t const pairs = nib | (nib >> 4);
  uint32_t r;
  asm("{\n\t.reg .b8 a0, a1, x0, x1;\n\t.reg .b32 h0, h1;\n\t.reg .b16 e0, e1;\n\t"
      "mov.b32 {a0, x0, a1, x1}, %1;\n\t"
      "cvt.rn.f16x2.e2m1x2 h0, a0;\n\tcvt.rn.f16x2.e2m1x2 h1, a1;\n\t"
      "mul.rn.f16x2 h0, h0, %2;\n\tmul.rn.f16x2 h1, h1, %3;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\tcvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
      "mov.b32 %0, {e0, e1};\n}"
      : "=r"(r)
      : "r"(pairs), "r"(sf.lo), "r"(sf.hi));
  return r;
}

#endif

}  // namespace fp4
}  // namespace cake_nvfp4
}  // namespace flashinfer
