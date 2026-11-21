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
// Numerics of the DeepSeek-V4 shared expert as SGLang runs it, operation for operation:
//   x -> per-(token, 128-group) e4m3 quantization -> FP8 GEMM with per-K128 promotion (DeepGEMM
//   1d2d) -> BF16 gate_up
//   -> SiLU-and-mul with clamp (fast-math, BF16) -> per-group quantization -> FP8 GEMM -> BF16
//   shared output.
// The fast-math operations are written out explicitly (MUFU.RCP / MUFU.EX2 with FTZ multiplies), so
// the result does not depend on the module's -use_fast_math setting.
#pragma once
#include <cuda_bf16.h>

#include <cstdint>

namespace flashinfer {
namespace tp_moe {
namespace sm90 {
namespace shnum {

__device__ __forceinline__ float rcp_approx_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float ex2_approx_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
  float r;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
  float r;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float max_ftz(float a, float b) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float min_ftz(float a, float b) {
  float r;
  asm("min.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float absmax_ftz(float m, float v) {
  float r;
  asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(m), "f"(fabsf(v)));
  return r;
}

// group quant (v2, group 128, fp8 e4m3, float scales): amax = max(1e-10, max |x|); multiplier = 448
// * rcp(amax); stored scale (the GEMM's dequant factor) = amax * fp32(1/448); q = RN satfinite
// e4m3(clamp(x * multiplier, -448, 448))
__device__ __forceinline__ float gq_amax_init() { return 1.00000001335143196e-10f; }
__device__ __forceinline__ float gq_mult(float amax) {
  return mul_ftz(rcp_approx_ftz(amax), 448.0f);
}
__device__ __forceinline__ float gq_scale(float amax) {
  return mul_ftz(amax, 0.0022321429569274187088f);
}
__device__ __forceinline__ uint16_t gq_pack2(float a, float b, float mult) {
  float qa = min_ftz(max_ftz(mul_ftz(a, mult), -448.0f), 448.0f);
  float qb = min_ftz(max_ftz(mul_ftz(b, mult), -448.0f), 448.0f);
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(qb), "f"(qa));
  return r;
}

// silu_and_mul_clamp (limit 10): gate = min(gate, 10), up = clamp(up, -10, 10) in BF16; silu(g) = g
// * rcp(1 + ex2(g * -log2 e)); out = BF16_rn(u * silu(g)) -- FTZ throughout
__device__ __forceinline__ __nv_bfloat162 silu_clamp2(__nv_bfloat162 g, __nv_bfloat162 u) {
  const __nv_bfloat162 lim = __floats2bfloat162_rn(10.0f, 10.0f),
                       nlim = __floats2bfloat162_rn(-10.0f, -10.0f);
  g = __hmin2(g, lim);
  u = __hmin2(__hmax2(u, nlim), lim);
  const float g0 = __low2float(g), g1 = __high2float(g), u0 = __low2float(u), u1 = __high2float(u);
  const float s0 = mul_ftz(
      g0, rcp_approx_ftz(add_ftz(ex2_approx_ftz(mul_ftz(g0, -1.4426950216293334961f)), 1.0f)));
  const float s1 = mul_ftz(
      g1, rcp_approx_ftz(add_ftz(ex2_approx_ftz(mul_ftz(g1, -1.4426950216293334961f)), 1.0f)));
  return __floats2bfloat162_rn(mul_ftz(u0, s0), mul_ftz(u1, s1));
}

}  // namespace shnum
}  // namespace sm90
}  // namespace tp_moe
}  // namespace flashinfer
