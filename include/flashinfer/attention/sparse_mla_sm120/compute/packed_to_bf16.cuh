// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../arch/common.cuh"

namespace flashinfer::sparse_mla_sm120 {

__device__ __forceinline__ __nv_bfloat162 decode_e2m1_or_e4m3_pair_bf16(uint32_t raw, bool is_e2m1) {
  uint32_t half_pair;
  if (is_e2m1) {
    asm("{ .reg .b8 x; mov.b32 {x,_,_,_}, %1; cvt.rn.f16x2.e2m1x2 %0,x; }"
        : "=r"(half_pair)
        : "r"(raw));
  } else {
    asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(half_pair) : "h"(uint16_t(raw)));
  }
  return __float22bfloat162_rn(__half22float2(*reinterpret_cast<__half2*>(&half_pair)));
}

__device__ __forceinline__ __nv_bfloat162 decode_e4m3_or_ue8m0_scale_bf16(uint8_t code, bool is_e4m3) {
  if (is_e4m3) return decode_e2m1_or_e4m3_pair_bf16(uint32_t(code) * 0x101u, false);
  uint32_t result;
  asm("cvt.rn.bf16x2.ue8m0x2 %0,%1;" : "=r"(result) : "h"(uint16_t(uint32_t(code) * 0x101u)));
  return *reinterpret_cast<__nv_bfloat162*>(&result);
}

}  // namespace flashinfer::sparse_mla_sm120
