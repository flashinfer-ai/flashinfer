/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * Device helpers of the generated generic-shape Cake BGMV MoE bodies. Each helper holds
 * one copy of an architecture-guarded FMA sequence that the grouped-shrink kernels expand
 * at every use: sm_100 and later take the packed or bf16 instruction, earlier targets
 * the scalar FP32 FMAs. The helpers are always inlined; the kernels' device code does not
 * change.
 */
#pragma once

// a0 += w0 * s; a1 += w1 * s: one fma.rn.f32x2 on sm_100+, two scalar FMAs below.
__device__ __forceinline__ void bgmv_moe_fma2(float& a0, float& a1, float w0, float w1, float s) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  unsigned long long s2;
  asm volatile("mov.b64 %0, {%1, %1};" : "=l"(s2) : "f"(s));
  asm volatile(
      "{\n\t"
      ".reg .b64 _src2, _acc2, _out2;\n\t"
      "mov.b64 _src2, {%2, %3};\n\t"
      "mov.b64 _acc2, {%0, %1};\n\t"
      "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t"
      "mov.b64 {%0, %1}, _out2;\n\t"
      "}"
      : "+f"(a0), "+f"(a1)
      : "f"(w0), "f"(w1), "l"(s2));
#else
  a0 = __fmaf_rn(w0, s, a0);
  a1 = __fmaf_rn(w1, s, a1);
#endif
}

// acc += sum over k of lo(x[k]) * lo(w[k]) + hi(x[k]) * hi(w[k]) for k = 0..3, where lo and
// hi are the bf16 halves of a 32-bit word: one fma.rn.f32.bf16 chain on sm_100+, eight
// widened scalar FMAs below.
__device__ __forceinline__ void bgmv_moe_dot8_bf16(float& acc, unsigned int x0, unsigned int x1,
                                                   unsigned int x2, unsigned int x3,
                                                   unsigned int w0, unsigned int w1,
                                                   unsigned int w2, unsigned int w3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm("{\n\t.reg .b16 _a0l, _a0h, _b0l, _b0h, _a1l, _a1h, _b1l, _b1h, _a2l, _a2h, _b2l, "
      "_b2h, _a3l, _a3h, _b3l, _b3h;\n\tmov.b32 {_a0l, _a0h}, %1;\n\tmov.b32 {_b0l, "
      "_b0h}, %5;\n\tmov.b32 {_a1l, _a1h}, %2;\n\tmov.b32 {_b1l, _b1h}, %6;\n\tmov.b32 "
      "{_a2l, _a2h}, %3;\n\tmov.b32 {_b2l, _b2h}, %7;\n\tmov.b32 {_a3l, _a3h}, "
      "%4;\n\tmov.b32 {_b3l, _b3h}, %8;\n\tfma.rn.f32.bf16 %0, _a0l, _b0l, "
      "%0;\n\tfma.rn.f32.bf16 %0, _a0h, _b0h, %0;\n\tfma.rn.f32.bf16 %0, _a1l, _b1l, "
      "%0;\n\tfma.rn.f32.bf16 %0, _a1h, _b1h, %0;\n\tfma.rn.f32.bf16 %0, _a2l, _b2l, "
      "%0;\n\tfma.rn.f32.bf16 %0, _a2h, _b2h, %0;\n\tfma.rn.f32.bf16 %0, _a3l, _b3l, "
      "%0;\n\tfma.rn.f32.bf16 %0, _a3h, _b3h, %0;\n\t}"
      : "+f"(acc)
      : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(w0), "r"(w1), "r"(w2), "r"(w3));
#else
  acc = __fmaf_rn(__uint_as_float(x0 << 16), __uint_as_float(w0 << 16), acc);
  acc = __fmaf_rn(__uint_as_float(x0 & 0xFFFF0000u), __uint_as_float(w0 & 0xFFFF0000u), acc);
  acc = __fmaf_rn(__uint_as_float(x1 << 16), __uint_as_float(w1 << 16), acc);
  acc = __fmaf_rn(__uint_as_float(x1 & 0xFFFF0000u), __uint_as_float(w1 & 0xFFFF0000u), acc);
  acc = __fmaf_rn(__uint_as_float(x2 << 16), __uint_as_float(w2 << 16), acc);
  acc = __fmaf_rn(__uint_as_float(x2 & 0xFFFF0000u), __uint_as_float(w2 & 0xFFFF0000u), acc);
  acc = __fmaf_rn(__uint_as_float(x3 << 16), __uint_as_float(w3 << 16), acc);
  acc = __fmaf_rn(__uint_as_float(x3 & 0xFFFF0000u), __uint_as_float(w3 & 0xFFFF0000u), acc);
#endif
}
