/*
 * Copyright (c) 2025 by SageAttention team.
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

#include <cstdint>

#include "common.cuh"

// SM120 register-register NVFP4 block-scaled MMA used by sparse MLA.
//
// A is a row-major 16x64 E2M1 tile (four 32-bit registers per lane), B is a
// column-major 64x8 E2M1 tile (two registers per lane), and each uint32 scale
// operand packs the four E4M3 scale-vector entries for the K64 tile. The scale
// selectors match the first N8 instruction in FlashInfer's upstream dense
// SM120 NVFP4 composite atom. The instruction form is adapted from
// attention/sm120/nvfp4_attention_sm120/common/cute_extension.h; this compact
// wrapper avoids pulling the dense kernel's CuTe/CUTLASS type stack into the
// sparse MLA hot path.

struct MmaNvfp4Result {
  float d0, d1, d2, d3;
};

__device__ __forceinline__ MmaNvfp4Result mma_nvfp4_block_scaled_m16n8k64(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, float c0,
    float c1, float c2, float c3, uint32_t scale_a, uint32_t scale_b) {
  MmaNvfp4Result r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
      "f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(scale_a), "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)), "r"(scale_b),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
#else
  (void)a0;
  (void)a1;
  (void)a2;
  (void)a3;
  (void)b0;
  (void)b1;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
  (void)scale_a;
  (void)scale_b;
#endif
  return r;
}
