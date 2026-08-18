/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Softmax helper functions — CuTe tensor style (matching blk128 softmax.py / utils.py).
// All functions accept cute::Tensor instead of raw float arrays to avoid
// taking addresses of local variables (which forces them into local memory).
#pragma once

#include <cuda_bf16.h>
#include <math_constants.h>

#include "cute/tensor.hpp"

namespace flash {

// 3-operand fmax: max(a, b, c) in one instruction (SM100 FMNMX3)
static __device__ __forceinline__ float fmax3(float a, float b, float c) {
  float r;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

// Packed f32x2 FMA: (a,b) = (a,b) * (c,c) + (d,d)
static __device__ __forceinline__ void ffma2(float& a, float& b, float c, float d) {
  asm("{\n\t"
      ".reg .b64 la, lc, ld;\n\t"
      "mov.b64 la, {%0, %1};\n\t"
      "mov.b64 lc, {%2, %2};\n\t"
      "mov.b64 ld, {%3, %3};\n\t"
      "fma.rn.f32x2 la, la, lc, ld;\n\t"
      "mov.b64 {%0, %1}, la;\n\t"
      "}\n"
      : "+f"(a), "+f"(b)
      : "f"(c), "f"(d));
}

// Packed f32x2 MUL: (a,b) = (a,b) * (c,c)
static __device__ __forceinline__ void fmul2(float& a, float& b, float c) {
  asm("{\n\t"
      ".reg .b64 la, lc;\n\t"
      "mov.b64 la, {%0, %1};\n\t"
      "mov.b64 lc, {%2, %2};\n\t"
      "mul.rn.f32x2 la, la, lc;\n\t"
      "mov.b64 {%0, %1}, la;\n\t"
      "}\n"
      : "+f"(a), "+f"(b)
      : "f"(c));
}

// Packed f32x2 ADD: (a0,a1) += (b0,b1)
static __device__ __forceinline__ void fadd2(float& a0, float& a1, float b0, float b1) {
  asm("{\n\t"
      ".reg .b64 la, lb;\n\t"
      "mov.b64 la, {%0, %1};\n\t"
      "mov.b64 lb, {%2, %3};\n\t"
      "add.rn.f32x2 la, la, lb;\n\t"
      "mov.b64 {%0, %1}, la;\n\t"
      "}\n"
      : "+f"(a0), "+f"(a1)
      : "f"(b0), "f"(b1));
}

// Emulated exp2 for 2 values (FMA pipe polynomial + ALU pipe integer combine).
static __device__ __forceinline__ void exp2_emu2(float& x, float& y) {
  unsigned int int0, int1, frac0, frac1;
  asm volatile(
      "{\n\t"
      ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
      ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l9, l10;\n\t"
      "max.ftz.f32 f1, %4, 0fC2FE0000;\n\t"
      "max.ftz.f32 f2, %5, 0fC2FE0000;\n\t"
      "mov.b64 l1, {f1, f2};\n\t"
      "mov.f32 f3, 0f4B400000;\n\t"
      "mov.b64 l2, {f3, f3};\n\t"
      "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
      ".reg .b64 l8;\n\t"
      "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
      "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
      "mov.f32 f7, 0.077119089663028717;\n\t"
      "mov.b64 l6, {f7, f7};\n\t"
      "mov.f32 f6, 0.227564394474029541;\n\t"
      "mov.b64 l5, {f6, f6};\n\t"
      "mov.f32 f5, 0.695146143436431885;\n\t"
      "mov.b64 l4, {f5, f5};\n\t"
      "mov.f32 f4, 1.0;\n\t"
      "mov.b64 l3, {f4, f4};\n\t"
      "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
      "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
      "mov.b64 {%0, %1}, l7;\n\t"
      "mov.b64 {%2, %3}, l10;\n\t"
      "}\n"
      : "=r"(int0), "=r"(int1), "=r"(frac0), "=r"(frac1)
      : "f"(x), "f"(y));
  asm volatile(
      "{\n\t"
      ".reg .s32 e0, e1, o0, o1;\n\t"
      "shl.b32 e0, %2, 23;\n\t"
      "shl.b32 e1, %3, 23;\n\t"
      "add.s32 o0, e0, %4;\n\t"
      "add.s32 o1, e1, %5;\n\t"
      "mov.b32 %0, o0;\n\t"
      "mov.b32 %1, o1;\n\t"
      "}\n"
      : "=f"(x), "=f"(y)
      : "r"(int0), "r"(int1), "r"(frac0), "r"(frac1));
}

// ============================================================================
// fmax_reduce: CuTe tensor version (matches blk128 utils.fmax_reduce)
// ============================================================================

template <typename Tensor>
static __device__ __forceinline__ float fmax_reduce(Tensor const& data) {
  constexpr int N = decltype(cute::size(data))::value;
  static_assert(N >= 8 && N % 8 == 0, "fmax_reduce requires N>=8, N%8==0");
  float lm0 = fmaxf(data(0), data(1));
  float lm1 = fmaxf(data(2), data(3));
  float lm2 = fmaxf(data(4), data(5));
  float lm3 = fmaxf(data(6), data(7));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 8; i < N; i += 8) {
    lm0 = fmax3(lm0, data(i + 0), data(i + 1));
    lm1 = fmax3(lm1, data(i + 2), data(i + 3));
    lm2 = fmax3(lm2, data(i + 4), data(i + 5));
    lm3 = fmax3(lm3, data(i + 6), data(i + 7));
  }
  lm0 = fmaxf(lm0, lm1);
  return fmax3(lm0, lm2, lm3);
}

template <typename Tensor>
static __device__ __forceinline__ float fmax_reduce(Tensor const& data, float init_val) {
  constexpr int N = decltype(cute::size(data))::value;
  static_assert(N >= 8 && N % 8 == 0, "fmax_reduce requires N>=8, N%8==0");
  float lm0 = fmax3(init_val, data(0), data(1));
  float lm1 = fmaxf(data(2), data(3));
  float lm2 = fmaxf(data(4), data(5));
  float lm3 = fmaxf(data(6), data(7));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 8; i < N; i += 8) {
    lm0 = fmax3(lm0, data(i + 0), data(i + 1));
    lm1 = fmax3(lm1, data(i + 2), data(i + 3));
    lm2 = fmax3(lm2, data(i + 4), data(i + 5));
    lm3 = fmax3(lm3, data(i + 6), data(i + 7));
  }
  lm0 = fmaxf(lm0, lm1);
  return fmax3(lm0, lm2, lm3);
}

// ============================================================================
// fadd_reduce: CuTe tensor version (matches blk128 utils.fadd_reduce)
// ============================================================================

template <typename Tensor>
static __device__ __forceinline__ float fadd_reduce(Tensor const& data) {
  constexpr int N = decltype(cute::size(data))::value;
  static_assert(N >= 8 && N % 8 == 0, "fadd_reduce requires N>=8, N%8==0");
  float s0a = data(0), s0b = data(1);
  float s1a = data(2), s1b = data(3);
  float s2a = data(4), s2b = data(5);
  float s3a = data(6), s3b = data(7);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 8; i < N; i += 8) {
    fadd2(s0a, s0b, data(i + 0), data(i + 1));
    fadd2(s1a, s1b, data(i + 2), data(i + 3));
    fadd2(s2a, s2b, data(i + 4), data(i + 5));
    fadd2(s3a, s3b, data(i + 6), data(i + 7));
  }
  fadd2(s0a, s0b, s1a, s1b);
  fadd2(s2a, s2b, s3a, s3b);
  fadd2(s0a, s0b, s2a, s2b);
  return s0a + s0b;
}

template <typename Tensor>
static __device__ __forceinline__ float fadd_reduce(Tensor const& data, float init_val) {
  constexpr int N = decltype(cute::size(data))::value;
  static_assert(N >= 8 && N % 8 == 0, "fadd_reduce requires N>=8, N%8==0");
  float s0a = init_val, s0b = 0.0f;
  fadd2(s0a, s0b, data(0), data(1));
  float s1a = data(2), s1b = data(3);
  float s2a = data(4), s2b = data(5);
  float s3a = data(6), s3b = data(7);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 8; i < N; i += 8) {
    fadd2(s0a, s0b, data(i + 0), data(i + 1));
    fadd2(s1a, s1b, data(i + 2), data(i + 3));
    fadd2(s2a, s2b, data(i + 4), data(i + 5));
    fadd2(s3a, s3b, data(i + 6), data(i + 7));
  }
  fadd2(s0a, s0b, s1a, s1b);
  fadd2(s2a, s2b, s3a, s3b);
  fadd2(s0a, s0b, s2a, s2b);
  return s0a + s0b;
}

// ============================================================================
// update_row_max: CuTe tensor version (matches blk128 SoftmaxSm100.update_row_max)
// ============================================================================

static constexpr float kRescaleThreshold = 8.0f;

template <bool IsFirst, typename Tensor>
static __device__ __forceinline__ void update_row_max(Tensor const& s_data, float sm_scale,
                                                      float& row_max, float& acc_scale,
                                                      float& row_max_safe) {
  if constexpr (IsFirst) {
    float local_max = fmax_reduce(s_data);
    row_max_safe = (local_max > -CUDART_INF_F) ? local_max : 0.0f;
    acc_scale = 0.0f;
    row_max = local_max;
  } else {
    float row_max_old = row_max;
    float m_new = fmax_reduce(s_data, row_max_old);
    float m_new_safe = (m_new > -CUDART_INF_F) ? m_new : 0.0f;
    float acc_scale_log2 = (row_max_old - m_new_safe) * sm_scale;
    acc_scale = exp2f(acc_scale_log2);
    if (acc_scale_log2 >= -kRescaleThreshold) {
      m_new = row_max_old;
      m_new_safe = row_max_old;
      acc_scale = 1.0f;
    }
    row_max_safe = m_new_safe;
    row_max = m_new;
  }
}

// ============================================================================
// update_row_sum: CuTe tensor version (matches blk128 SoftmaxSm100.update_row_sum)
// ============================================================================

template <bool IsFirst, typename Tensor>
static __device__ __forceinline__ void update_row_sum(Tensor const& s_data, float acc_scale,
                                                      float& row_sum) {
  if constexpr (IsFirst) {
    row_sum = fadd_reduce(s_data);
  } else {
    row_sum = fadd_reduce(s_data, row_sum * acc_scale);
  }
}

}  // namespace flash
