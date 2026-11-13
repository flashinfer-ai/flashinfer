// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "../arch/common.cuh"

// RoPE convention: int32 positions index FP32 rows [cos[32], sin[32]].
// Rotate adjacent pairs in the last 64 features: forward for Q, inverse for O.
// This matches FlashMLA 395cb3af, fused_norm_rope_attn_rope_cast_fwd/core_attn/
// kernel.h:27-31 and kernel.cuh:553-562, 730-750 (convention, not bitwise parity).
// Output is contiguous [T,H,512], with one UE8M0 scale per 32 features.
// Positions are device inputs read on each replay; callers must keep them
// within the cos_sin table's row bounds.
struct SparseMlaMegaParams {
  const int32_t* positions = nullptr;
  const float* cos_sin = nullptr;
  uint8_t* output = nullptr;
  uint8_t* scales = nullptr;
  __device__ __forceinline__ const float* row(int token) const {
    return positions ? cos_sin + size_t(positions[token]) * 64 : nullptr;
  }
};

template <bool CheckNull = true>
__device__ __forceinline__ float mega_rotate(float x, float other, int d, const float* cs,
                                             bool inverse = false) {
  if constexpr (CheckNull) {
    if (!cs) return x;
  }
  if (d < 448) return x;
  const int r = (d - 448) / 2;
  const float sign = ((d & 1) != 0) != inverse ? 1.f : -1.f;
  // Match the unfused RoPE producer: round other*sin, then fuse x*cos + product.
  // Keep this ordering explicit: choosing the other product for FMA can change
  // the subsequent BF16/FP8 rounding. See tl.fma in sparse_mla_prefill_utils.py.
  return __fmaf_rn(x, __ldg(cs + r), __fmul_rn(sign * other, __ldg(cs + 32 + r)));
}

template <bool CheckNull = true>
__device__ __forceinline__ uint4 mega_rotate8(uint4 packed, int d, const float* cs) {
  if constexpr (CheckNull) {
    if (!cs) return packed;
  }
  if (d >= 448) {
    bf16* v = reinterpret_cast<bf16*>(&packed);
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float x = __bfloat162float(v[j]), y = __bfloat162float(v[j + 1]);
      v[j] = __float2bfloat16_rn(mega_rotate<CheckNull>(x, y, d + j, cs));
      v[j + 1] = __float2bfloat16_rn(mega_rotate<CheckNull>(y, x, d + j + 1, cs));
    }
  }
  return packed;
}

// Four consecutive lanes own one aligned 32-value scale group. Called only
// with whole groups active, including partial head tiles (512/8 = 64 lanes).
__device__ __forceinline__ void mega_store8(uint4 packed, size_t offset, int token,
                                            const SparseMlaMegaParams& mega) {
  const int d = offset % 512;
  const float* cs = mega.cos_sin + size_t(mega.positions[token]) * 64;
  const bf16* v = reinterpret_cast<const bf16*>(&packed);
  float f[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    f[j] = mega_rotate<false>(__bfloat162float(v[j]), __bfloat162float(v[j ^ 1]), d + j, cs, true);
  }
  float amax = fmaxf(fmaxf(fmaxf(fabsf(f[0]), fabsf(f[1])), fmaxf(fabsf(f[2]), fabsf(f[3]))),
                     fmaxf(fmaxf(fabsf(f[4]), fabsf(f[5])), fmaxf(fabsf(f[6]), fabsf(f[7]))));
  amax = fmaxf(amax, 1.e-4f);
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
  const uint32_t bits = (__float_as_uint(amax / 448.f) + 0x007fffff) & 0x7f800000;
  // The clamped, finite scale is a normal power of two; exponent reflection
  // gives its reciprocal without division.
  const float inv = bits < 0x7f800000u ? __uint_as_float(0x7f000000u - bits) : 0.f;
  uint2 out;
  out.x = cvt_e4m3x4(f[0] * inv, f[1] * inv, f[2] * inv, f[3] * inv);
  out.y = cvt_e4m3x4(f[4] * inv, f[5] * inv, f[6] * inv, f[7] * inv);
  *reinterpret_cast<uint2*>(mega.output + offset) = out;
  if ((d & 31) == 0) mega.scales[offset / 32] = bits >> 23;
}
