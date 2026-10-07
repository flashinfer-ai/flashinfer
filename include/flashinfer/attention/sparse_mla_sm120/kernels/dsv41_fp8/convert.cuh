// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <flashinfer/math.cuh>

#include "../../compute/nvfp4_quantization.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8 {

template <int MathGroups, int MathRegs>
__device__ inline void Dsv41MixedCacheRoleResources<MathGroups, MathRegs>::gather_registers() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(GATHER_REGS));
}
template <int MathGroups, int MathRegs>
__device__ inline void Dsv41MixedCacheRoleResources<MathGroups, MathRegs>::convert_registers() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(CONVERT_REGS));
}
template <int MathGroups, int MathRegs>
__device__ inline void Dsv41MixedCacheRoleResources<MathGroups, MathRegs>::math_registers() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(MATH_REGS));
}

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8

// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
namespace flashinfer::sparse_mla_sm120 {

// Exact ceil(log2(x)) for x > 0 via the fp32 exponent/mantissa split.
__device__ __forceinline__ int ceil_log2_positive(float x) {
  const uint32_t bits = __float_as_uint(x);
  return int(bits >> 23) - 127 + ((bits & 0x7fffff) ? 1 : 0);
}

// Convert one 32-wide group of a DSV41_FP4 row (16 packed bytes + two E4M3
// scale bytes, already in registers) into 32 canonical FP8 bytes + one
// UE8M0 scale byte.
__device__ __forceinline__ void requantize_e2m1_e4m3_g16_to_e4m3_ue8m0_g32(uint4 packed,
                                                                           uint32_t sc2,
                                                                           uint8_t* smem_data,
                                                                           uint8_t* smem_scale) {
  using nvfp4::e2m1x2_code_to_float2;
  using nvfp4::e4m3_byte_to_float;
  const float sc[2] = {e4m3_byte_to_float(static_cast<uint8_t>(sc2 & 0xFF)),
                       e4m3_byte_to_float(static_cast<uint8_t>(sc2 >> 8))};
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
  float v[32];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float2 d = e2m1x2_code_to_float2((words[w] >> (b * 8)) & 0xFF);
      v[w * 8 + b * 2] = d.x * sc[w / 2];
      v[w * 8 + b * 2 + 1] = d.y * sc[w / 2];
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  // UE8M0 scale = 2^ceil(log2(max(amax/448, 1e-4))); the 1e-4 clamp mirrors
  // FlashMLA's FP8_AMAX_MARGIN alignment. __fdiv_rn: -use_fast_math would
  // otherwise let an approximate division flip the ceil at pow2 boundaries.
  const int e = ceil_log2_positive(fmaxf(__fdiv_rn(amax, 448.f), 1e-4f));
  *smem_scale = static_cast<uint8_t>(e + 127);
  const float inv = exp2f(float(-e));  // exact, scale is a power of two
  uint32_t out[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[j] = math::fp32_vec_to_e4m3(v[4 * j] * inv, v[4 * j + 1] * inv, v[4 * j + 2] * inv,
                                    v[4 * j + 3] * inv);
  }
  *reinterpret_cast<uint4*>(smem_data) = make_uint4(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<uint4*>(smem_data + 16) = make_uint4(out[4], out[5], out[6], out[7]);
}

}  // namespace flashinfer::sparse_mla_sm120

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8 {

template <typename Resources, typename Valid>
__device__ void convert_raw(const uint8_t* raw, uint8_t* values, uint8_t* scales, int tid,
                            Valid valid) {
  using Raw = typename Resources::Raw;
  using KV = typename Resources::KV;
  constexpr int Groups = Resources::CONVERSION_GROUPS_PER_ROW;
  for (int task = tid; task < Resources::BI * Groups; task += Resources::GROUP_THREADS) {
    const int row = task / Groups, group = task % Groups;
    auto* dst = values + row * KV::KV_SMEM_STRIDE + group * KV::QUANT_TILE;
    auto* sc = scales + row * KV::SCALE_BYTES_PER_TOKEN + group;
    if (!valid(row)) {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(dst + sizeof(uint4)) = make_uint4(0, 0, 0, 0);
      *sc = 0;
    } else {
      requantize_e2m1_e4m3_g16_to_e4m3_ue8m0_g32(
          *reinterpret_cast<const uint4*>(raw + row * Raw::DATA_BYTES + group * sizeof(uint4)),
          *reinterpret_cast<const uint16_t*>(raw + Resources::RAW_SCALE_OFFSET +
                                             row * Raw::SCALE_BYTES + group * sizeof(uint16_t)),
          dst, sc);
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8
