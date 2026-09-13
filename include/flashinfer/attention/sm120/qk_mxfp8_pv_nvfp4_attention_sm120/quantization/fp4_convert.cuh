/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace qk_mxfp8_pv_nvfp4_attention {

/**
 *
 *
 *
 */

/**
 *
 *
 */
CUTLASS_DEVICE void packed_float_to_ue4m3(float const& f0, float const& f1, float const& f2,
                                          float const& f3, uint32_t& out) {
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(out)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
}

/**
 *
 *
 */
CUTLASS_DEVICE void packed_float_to_e2m1(float const& f0, float const& f1, float const& f2,
                                         float const& f3, float const& f4, float const& f5,
                                         float const& f6, float const& f7, uint32_t& out) {
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(out)
      : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "f"(f4), "f"(f5), "f"(f6), "f"(f7));
}

/**
 * FP8 E4M3 Converter
 *
 */
template <int N>
struct FP8E4M3Converter {
  static_assert(N % 4 == 0, "N must be multiple of 4");

  /**
   *
   */
  __device__ __forceinline__ static void convert(float const inputs[N], uint32_t outputs[N / 4]) {
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      packed_float_to_ue4m3(inputs[i * 4 + 0], inputs[i * 4 + 1], inputs[i * 4 + 2],
                            inputs[i * 4 + 3], outputs[i]);
    }
  }
};

/**
 * FP4 E2M1 Converter
 *
 */
template <int N>
struct FP4E2M1Converter {
  static_assert(N % 8 == 0, "N must be multiple of 8");

  /**
   *
   */
  __device__ __forceinline__ static void convert(float const inputs[N], uint32_t outputs[N / 8]) {
#pragma unroll
    for (int i = 0; i < N / 8; ++i) {
      packed_float_to_e2m1(inputs[i * 8 + 0], inputs[i * 8 + 1], inputs[i * 8 + 2],
                           inputs[i * 8 + 3], inputs[i * 8 + 4], inputs[i * 8 + 5],
                           inputs[i * 8 + 6], inputs[i * 8 + 7], outputs[i]);
    }
  }
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
