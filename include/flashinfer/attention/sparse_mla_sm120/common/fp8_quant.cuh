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
// Steps:
//   1. Copy Q rope to smem (BF16, unquantized)
//   2. Compute per-tile absmax via atomicMax
//   3. Compute scale = absmax / FP8_MAX (power-of-2 friendly)
//   4. Quantize Q nope to FP8 and write to smem
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

template <ModelType MT, int _MATH_THREADS>
__device__ __forceinline__ void quantize_q_to_smem(uint8_t* q_nope_fp8, float* q_nope_sc,
                                                   bf16* q_rope, const bf16* q_base,
                                                   float* reduce_buf, int valid_hpb = HPB) {
  using KV = KVCacheTraits<MT>;
  constexpr int D_NOPE = KV::D_NOPE;
  constexpr int D_ROPE = KV::D_ROPE;
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;
  constexpr int QUANT_TILE = KV::QUANT_TILE;
  constexpr int NUM_SCALES = KV::NUM_SCALES;
  constexpr int DIM = KV::D_QK;

  float* amax = reduce_buf;

  // Step 1: copy Q rope to smem (only valid heads from gmem; zero-fill rest)
  if constexpr (D_ROPE > 0) {
    for (int i = threadIdx.x; i < HPB * D_ROPE; i += _MATH_THREADS) {
      int h = i / D_ROPE, d = i % D_ROPE;
      q_rope[h * D_ROPE + d] =
          (h < valid_hpb) ? q_base[h * DIM + D_NOPE + d] : __float2bfloat16(0.f);
    }
  }
  // Step 2: init amax
  for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += _MATH_THREADS) amax[i] = 0.f;
  bar_sync_t<2, _MATH_THREADS>();

  // Compute absmax per tile (only valid heads)
  for (int idx = threadIdx.x; idx < valid_hpb * D_NOPE; idx += _MATH_THREADS) {
    int h = idx / D_NOPE, blk = (idx % D_NOPE) / QUANT_TILE;
    atomicMax(reinterpret_cast<int*>(&amax[h * NUM_SCALES + blk]),
              __float_as_int(fabsf(__bfloat162float(q_base[h * DIM + idx % D_NOPE]))));
  }
  bar_sync_t<2, _MATH_THREADS>();

  // Step 3: compute scale, rounded up to power-of-2 for exact UE8M0 block-scaled MMA
  for (int i = threadIdx.x; i < HPB * NUM_SCALES; i += _MATH_THREADS) {
    float raw = fmaxf(amax[i], 1e-4f) / FP8_MAX;
    uint32_t bits = __float_as_uint(raw);
    if (bits & 0x007FFFFF) bits = (bits + 0x00800000) & 0x7F800000;
    q_nope_sc[i] = __uint_as_float(bits);
  }
  bar_sync_t<2, _MATH_THREADS>();

  // Step 4: quantize (valid heads from gmem; zero-fill rest)
  for (int idx = threadIdx.x; idx < HPB * D_NOPE; idx += _MATH_THREADS) {
    int h = idx / D_NOPE, d = idx % D_NOPE, blk = d / QUANT_TILE;
    if (h < valid_hpb) {
      float si = 1.f / q_nope_sc[h * NUM_SCALES + blk];
      float v = fmaxf(FP8_MIN, fminf(FP8_MAX, __bfloat162float(q_base[h * DIM + d]) * si));
      __nv_fp8_e4m3 fp8v(v);
      q_nope_fp8[h * Q_NOPE_STRIDE + d] = fp8v.__x;
    } else {
      q_nope_fp8[h * Q_NOPE_STRIDE + d] = 0;
    }
  }
}
