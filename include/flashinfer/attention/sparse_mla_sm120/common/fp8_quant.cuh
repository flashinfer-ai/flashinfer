// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../arch/barrier.cuh"
#include "../arch/common.cuh"
#include "../model/kv_cache_traits.cuh"

// On-the-fly Q quantization: BF16 → FP8 E4M3 with per-tile scaling.
//
// Single vectorized gmem pass (uint4 = 8 BF16); each tile's absmax is reduced
// across the LANES_PER_TILE lanes covering it with warp shuffles (no smem
// atomics, no amax scratch), the scale is computed redundantly per lane, and
// the values are quantized straight out of the registers they were loaded
// into. One trailing bar:2 makes the rope/scale/FP8 smem writes visible to the
// math group before return.
//
// Steps:
//   1. Copy Q rope to smem (BF16, unquantized, uint4-vectorized)
//   2. Per uint4: load once, per-thread absmax over its 8 elements, shuffle-
//      reduce to the tile absmax, scale = absmax / FP8_MAX (power-of-2
//      friendly), quantize the register-held values, write FP8 to smem
//
// Template on ModelType to get correct Q_NOPE_STRIDE and NUM_SCALES.

// BF16 Q load: cooperative gmem→smem copy. Counterpart to quantize_q_to_smem
// for the ComputeMode::BF16 QK path.
template <ModelType MT, int _MATH_THREADS>
__device__ __forceinline__ void load_q_bf16_to_smem(bf16* q_nope_bf16, bf16* q_rope,
                                                    const bf16* q_base, int valid_hpb = HPB) {
  using KV = KVCacheTraits<MT>;
  constexpr int D_NOPE = KV::D_NOPE;
  constexpr int D_ROPE = KV::D_ROPE;
  constexpr int DIM = KV::D_QK;
  constexpr int BF16_STRIDE = KV::Q_NOPE_BF16_STRIDE;

  for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += _MATH_THREADS) {
    int h = idx / D_NOPE, d = idx % D_NOPE;
    q_nope_bf16[h * BF16_STRIDE + d] =
        (h < valid_hpb) ? q_base[h * DIM + d] : __float2bfloat16(0.f);
  }
  if constexpr (D_ROPE > 0) {
    for (int i = threadIdx.x; i < HPB * D_ROPE; i += _MATH_THREADS) {
      int h = i / D_ROPE, d = i % D_ROPE;
      q_rope[h * D_ROPE + d] =
          (h < valid_hpb) ? q_base[h * DIM + D_NOPE + d] : __float2bfloat16(0.f);
    }
  }
  bar_sync_t<2, _MATH_THREADS>();
}

// swapAB Q quantization: the four B-fragment lanes of a head partition its
// D_NOPE dims exactly, so Q is quantized into registers and never reaches smem.
template <ModelType MT>
struct QSwapABRegs {
  using KV = KVCacheTraits<MT>;
  uint32_t nope[KV::D_NOPE / 32][2];
  float scale[KV::NUM_SCALES];  // of head (lane >> 2)
};

template <ModelType MT>
__device__ __forceinline__ QSwapABRegs<MT> quantize_q_to_regs_swapab(const bf16* q_base, int lane) {
  using KV = KVCacheTraits<MT>;
  constexpr int NOPE_KSTEPS = KV::D_NOPE / 32;
  constexpr int STEPS_PER_GRP = KV::QUANT_TILE / 32;
  const int tid = lane & 3;
  QSwapABRegs<MT> q;

#pragma unroll
  for (int g = 0; g < KV::NUM_SCALES; g++) q.scale[g] = 0.f;

#pragma unroll
  for (int ik = 0; ik < NOPE_KSTEPS; ik++) {
    const bf16* p = q_base + ik * 32 + tid * 4;
    uint2 lo = *reinterpret_cast<const uint2*>(p);
    uint2 hi = *reinterpret_cast<const uint2*>(p + 16);
    const bf16* el = reinterpret_cast<const bf16*>(&lo);
    const bf16* eh = reinterpret_cast<const bf16*>(&hi);
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) a = fmaxf(a, fmaxf(fabsf(to_float(el[j])), fabsf(to_float(eh[j]))));
    q.scale[ik / STEPS_PER_GRP] = fmaxf(q.scale[ik / STEPS_PER_GRP], a);
  }

#pragma unroll
  for (int g = 0; g < KV::NUM_SCALES; g++) {
    float amax = q.scale[g];
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float s = fmaxf(amax, 1e-4f) / FP8_MAX;
    if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
      q.scale[g] = s;  // the software fold takes FP32, so keep the exact amax
    } else {
      // Round up to a power of 2 so the UE8M0 block-scaled MMA is exact.
      uint32_t bits = __float_as_uint(s);
      if (bits & 0x007FFFFF) bits = (bits + 0x00800000) & 0x7F800000;
      q.scale[g] = __uint_as_float(bits);
    }
  }

#pragma unroll
  for (int ik = 0; ik < NOPE_KSTEPS; ik++) {
    const float si = 1.f / q.scale[ik / STEPS_PER_GRP];
    const bf16* p = q_base + ik * 32 + tid * 4;
    uint2 lo = *reinterpret_cast<const uint2*>(p);
    uint2 hi = *reinterpret_cast<const uint2*>(p + 16);
    const bf16* el = reinterpret_cast<const bf16*>(&lo);
    const bf16* eh = reinterpret_cast<const bf16*>(&hi);
    q.nope[ik][0] = cvt_e4m3x4(to_float(el[0]) * si, to_float(el[1]) * si, to_float(el[2]) * si,
                               to_float(el[3]) * si);
    q.nope[ik][1] = cvt_e4m3x4(to_float(eh[0]) * si, to_float(eh[1]) * si, to_float(eh[2]) * si,
                               to_float(eh[3]) * si);
  }
  return q;
}

// Called by the math threads only; the trailing bar:2 syncs the math group.
// q_base must be 16B-aligned row-wise: every row offset is a multiple of
// D_QK * 2 bytes, and D_QK * 2 % 16 == 0 for every supported model type.
template <ModelType MT, int _MATH_THREADS>
__device__ __forceinline__ void quantize_q_to_smem(uint8_t* q_nope_fp8, float* q_nope_sc,
                                                   bf16* q_rope, const bf16* q_base,
                                                   int valid_hpb = HPB) {
  using KV = KVCacheTraits<MT>;
  constexpr int D_NOPE = KV::D_NOPE;
  constexpr int D_ROPE = KV::D_ROPE;
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;
  constexpr int QUANT_TILE = KV::QUANT_TILE;
  constexpr int NUM_SCALES = KV::NUM_SCALES;
  constexpr int DIM = KV::D_QK;

  const int tid = threadIdx.x;

  // Step 1: copy Q rope to smem, one uint4 (8 BF16) per thread per iteration
  // (only valid heads from gmem; zero-fill rest).
  if constexpr (D_ROPE > 0) {
    constexpr int ROPE_VECS_PER_HEAD = D_ROPE / 8;
    for (int v = tid; v < HPB * ROPE_VECS_PER_HEAD; v += _MATH_THREADS) {
      const int h = v / ROPE_VECS_PER_HEAD, r = v % ROPE_VECS_PER_HEAD;
      uint4 val = make_uint4(0, 0, 0, 0);
      if (h < valid_hpb) val = *reinterpret_cast<const uint4*>(q_base + h * DIM + D_NOPE + r * 8);
      *reinterpret_cast<uint4*>(q_rope + h * D_ROPE + r * 8) = val;
    }
  }

  // Steps 2-4, fused into a single gmem pass over Q nope.
  //
  // Lane assignment: iteration k maps warp w's 32 lanes to the 32 consecutive
  // uint4 vectors [k * _MATH_THREADS + 32w, +32). A scale tile spans
  // LANES_PER_TILE consecutive vectors, so each aligned group of
  // LANES_PER_TILE lanes covers exactly one tile and the tile absmax is a
  // warp-local shuffle reduce — no smem amax scratch, no atomicMax.
  constexpr int VECS_PER_HEAD = D_NOPE / 8;  // uint4 vectors per head row
  constexpr int TOTAL_VECS = HPB * VECS_PER_HEAD;
  constexpr int LANES_PER_TILE = QUANT_TILE / 8;  // vectors (and lanes) per tile
  constexpr int MAX_VECS = (TOTAL_VECS + _MATH_THREADS - 1) / _MATH_THREADS;
  static_assert(32 % LANES_PER_TILE == 0, "a tile must not straddle a warp");
  static_assert(_MATH_THREADS % 32 == 0, "warp-contiguous lane assignment");
  // A partially-filled tail iteration must cut at a whole-warp boundary so
  // every shuffle group is uniformly in/out of range.
  static_assert(TOTAL_VECS % _MATH_THREADS % 32 == 0, "tail iteration must not split a warp");

  const int lane = tid & 31;
#pragma unroll
  for (int k = 0; k < MAX_VECS; k++) {
    const int v = k * _MATH_THREADS + tid;
    const bool in_range = v < TOTAL_VECS;
    const int h = in_range ? v / VECS_PER_HEAD : 0;
    const int d0 = (v % VECS_PER_HEAD) * 8;
    const bool load = in_range && h < valid_hpb;

    uint4 pk = make_uint4(0, 0, 0, 0);
    if (load) pk = *reinterpret_cast<const uint4*>(q_base + h * DIM + d0);
    const bf16* e = reinterpret_cast<const bf16*>(&pk);

    // Per-thread absmax over this thread's 8 elements, then across the tile's
    // lanes. Out-of-range and invalid-head lanes contribute 0, matching the
    // old zero-initialized amax; fmaxf on exact |values| is order-independent,
    // so the result is bitwise identical to the atomicMax reduction.
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) a = fmaxf(a, fabsf(__bfloat162float(e[j])));
#pragma unroll
    for (int m = 1; m < LANES_PER_TILE; m <<= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));

    // Scale, rounded up to power-of-2 for exact UE8M0 block-scaled MMA. Every
    // lane of the tile computes the identical value; the tile's first lane
    // stores it (invalid heads get the amax=0 scale, as before).
    const float raw = fmaxf(a, 1e-4f) / FP8_MAX;
    uint32_t bits = __float_as_uint(raw);
    if (bits & 0x007FFFFF) bits = (bits + 0x00800000) & 0x7F800000;
    const float s = __uint_as_float(bits);
    if (in_range && lane % LANES_PER_TILE == 0) q_nope_sc[h * NUM_SCALES + d0 / QUANT_TILE] = s;

    // Quantize the register-held values; cvt.rn.satfinite subsumes the old
    // explicit [FP8_MIN, FP8_MAX] clamp.
    if (in_range) {
      uint2 out = make_uint2(0, 0);
      if (load) {
        const float si = 1.f / s;
        out.x = cvt_e4m3x4(__bfloat162float(e[0]) * si, __bfloat162float(e[1]) * si,
                           __bfloat162float(e[2]) * si, __bfloat162float(e[3]) * si);
        out.y = cvt_e4m3x4(__bfloat162float(e[4]) * si, __bfloat162float(e[5]) * si,
                           __bfloat162float(e[6]) * si, __bfloat162float(e[7]) * si);
      }
      *reinterpret_cast<uint2*>(q_nope_fp8 + h * Q_NOPE_STRIDE + d0) = out;
    }
  }

  // Make the rope / scale / FP8 writes visible to the whole math group (the
  // prefill callers preload q_rope right after this returns).
  bar_sync_t<2, _MATH_THREADS>();
}
