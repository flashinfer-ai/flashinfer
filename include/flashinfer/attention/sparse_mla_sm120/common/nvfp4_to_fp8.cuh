// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_fp4.h>

#include "../arch/common.cuh"
#include "../model/kv_cache_traits.cuh"

// Expand a gathered GLM NVFP4 row in its existing FP8 shared-memory slot.
// One warp owns a row and each lane owns one 16-element NVFP4 block. All
// source bytes enter registers before any lane overwrites the packed row;
// no extra tile or global-memory intermediate is needed. The caller waits
// for the async gather before entry and synchronizes the math group after
// return, before another warp consumes this row.
//
// One power-of-two scale covers the complete 512-element row. Moving the
// binary point preserves the E4M3 rounding grid for normal values, while a
// common scale lets PV reuse one quantized probability tile across all V
// chunks. Four identical FP32 scales preserve the baseline QK block-128 ABI.
template <ModelType MT, int TILE_BI, int MATH_THREADS,
          int SMEM_STRIDE = KVCacheTraits<MT>::KV_SMEM_STRIDE, bool MOVE_ROPE = false,
          bool CONTIGUOUS_WARP_ROWS = false>
__device__ __forceinline__ void expand_nvfp4_kv_tile(uint8_t* dst, int math_tid,
                                                     const float* global_scale) {
  static_assert(KVCacheTraits<MT>::D_NOPE == 512);
  static_assert(MATH_THREADS % 32 == 0);
  constexpr int MATH_WARPS = MATH_THREADS / 32;
  const int warp = math_tid / 32;
  const int lane = math_tid & 31;
  const float layer_scale = global_scale ? __ldg(global_scale) : 1.f;

  // Decode QK assigns consecutive groups of eight candidates to each warp.
  // That owner can consume its expanded rows after a warp fence; other users
  // retain the original interleaved row assignment and caller synchronization.
  static_assert(!CONTIGUOUS_WARP_ROWS || TILE_BI % MATH_WARPS == 0);
  constexpr int ROWS_PER_WARP = TILE_BI / MATH_WARPS;
  const int row_start = CONTIGUOUS_WARP_ROWS ? warp * ROWS_PER_WARP : warp;
  const int row_end = CONTIGUOUS_WARP_ROWS ? row_start + ROWS_PER_WARP : TILE_BI;
  constexpr int ROW_STEP = CONTIGUOUS_WARP_ROWS ? 1 : MATH_WARPS;
#pragma unroll
  for (int row = row_start; row < row_end; row += ROW_STEP) {
    uint8_t* row_ptr = dst + row * SMEM_STRIDE;
    const uint2 packed = *reinterpret_cast<const uint2*>(row_ptr + lane * 8);
    const uint32_t scale_byte = row_ptr[256 + lane];
    const float block_scale = dequantize_e4m3_byte(static_cast<uint8_t>(scale_byte));
    uint32_t rope = 0;
    if constexpr (MOVE_ROPE) {
      static_assert(SMEM_STRIDE >= 656);
      rope = *reinterpret_cast<const uint32_t*>(row_ptr + 288 + lane * 4);
    }
    // The destination FP8 bytes overlap both the packed values and scales.
    __syncwarp();

    // Nonnegative finite E4M3 scale bytes are monotonically ordered. REDUX
    // replaces five shuffle/FMAX pairs with one warp-wide integer maximum.
    const float max_scale =
        dequantize_e4m3_byte(static_cast<uint8_t>(__reduce_max_sync(0xffffffff, scale_byte)));
    const float lower_bound = max_scale * (6.f / 448.f);
    uint32_t scale_bits = __float_as_uint(lower_bound);
    if (scale_bits & 0x007fffff) scale_bits = (scale_bits + 0x00800000) & 0x7f800000;
    const float compute_scale = __uint_as_float(scale_bits);
    // compute_scale is either zero or a normal FP32 power of two in [2^-15, 8].
    // Reflecting its biased exponent computes the exact reciprocal and avoids
    // a MUFU.RCP on each row's unpack/convert dependency chain.
    const float inv_compute_scale = __uint_as_float(0x7f000000u - scale_bits);
    const float multiplier = compute_scale > 0.f ? block_scale * inv_compute_scale : 0.f;
    // E2M1 * E4M3 / power-of-two is exact in half throughout the storage
    // scale range. Keep it packed until the final E4M3 rounding.
    const __half2 multiplier_h = __float2half2_rn(multiplier);
    uint4 result;
    uint32_t* result_words = reinterpret_cast<uint32_t*>(&result);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const uint32_t word = group < 2 ? packed.x : packed.y;
      const uint16_t values = static_cast<uint16_t>(word >> ((group & 1) * 16));
      const __half2 lo =
          static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(static_cast<uint8_t>(values), __NV_E2M1));
      const __half2 hi = static_cast<__half2>(
          __nv_cvt_fp4x2_to_halfraw2(static_cast<uint8_t>(values >> 8), __NV_E2M1));
      const __half2 normalized_lo = __hmul2(lo, multiplier_h);
      const __half2 normalized_hi = __hmul2(hi, multiplier_h);
      const uint16_t fp8_lo = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(normalized_lo),
                                                         __NV_SATFINITE, __NV_E4M3);
      const uint16_t fp8_hi = __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(normalized_hi),
                                                         __NV_SATFINITE, __NV_E4M3);
      result_words[group] = static_cast<uint32_t>(fp8_lo) | (static_cast<uint32_t>(fp8_hi) << 16);
    }
    *reinterpret_cast<uint4*>(row_ptr + lane * 16) = result;
    if ((lane & 7) == 0) {
      reinterpret_cast<float*>(row_ptr + 512)[lane / 8] = compute_scale * layer_scale;
    }
    if constexpr (MOVE_ROPE) {
      *reinterpret_cast<uint32_t*>(row_ptr + 528 + lane * 4) = rope;
    }
  }
}
