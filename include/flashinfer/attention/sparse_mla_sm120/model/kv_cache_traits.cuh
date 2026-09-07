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

#include "../arch/common.cuh"
#include "model_type.h"

// KVCacheTraits<ModelType>: compile-time constants for KV cache layout.
//
// These determine smem strides, MMA loop counts, IO gather sizes,
// and all dimension-dependent kernel parameters.
//
// The DeepSeek-family model types share: D_ROPE=64, D_V=512, HPB=16, BI=64.
// GLM53_NOPE diverges on D_ROPE (0); DOTS3_SWA diverges on D_V (1024) and runs
// at BI=32 — see its traits below.

template <ModelType MT>
struct KVCacheTraits;

template <>
struct KVCacheTraits<ModelType::DSV3_2> {
  // Dimensions
  static constexpr int D_NOPE = 512;
  static constexpr int D_ROPE = 64;
  static constexpr int D_QK = D_NOPE + D_ROPE;  // 576
  static constexpr int D_V = 512;

  // FP8 quantization
  static constexpr int QUANT_TILE = 128;
  static constexpr int NUM_SCALES = D_NOPE / QUANT_TILE;  // 4
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::POW2_FP32;

  // KV cache layout (FlashMLA ABI): INLINE, 656 bytes per token
  //   [0:512)   FP8 E4M3 nope (4 tiles × 128)
  //   [512:528) 4 × FP32 scale
  //   [528:656) BF16 rope (64 elements × 2B)
  static constexpr bool SCALE_INLINE = true;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_SCALES * sizeof(float);  // 16
  static constexpr int KV_GMEM_STRIDE =
      D_NOPE + SCALE_BYTES_PER_TOKEN + D_ROPE * sizeof(bf16);                 // 656
  static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE;                         // 512
  static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE + SCALE_BYTES_PER_TOKEN;  // 528

  // Smem layout: bulk copy includes nope + scales (528B)
  // stride=528: 528/4=132, 132%32=4 → 4-way bank conflict (acceptable)
  static constexpr int KV_SMEM_STRIDE = D_NOPE + SCALE_BYTES_PER_TOKEN;  // 528
  static constexpr int KV_SMEM_COPY_BYTES = KV_SMEM_STRIDE;              // copy 528B per entry
  // DSV3_2: scales are within the bulk-copied region → accessible from kv_smem
  static constexpr bool SCALE_IN_KV_SMEM = true;

  // Q nope stride (padded for ldmatrix alignment + bank conflict avoidance)
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;  // 528
  // Unused for DSV3_2 prefill; declared so SmemLayout<DSV3_2, BF16> compiles.
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 520

  // V = pure nope (no rope component)
  static constexpr bool V_HAS_ROPE = false;

  // FP32→UE8M0 scale conversion for block-scaled MMA
  // FlashMLA stores power-of-2 FP32 scales → bit-shift gives exact UE8M0
  __device__ static __forceinline__ uint8_t scale_to_ue8m0(float scale) {
    return static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFF);
  }
};

template <>
struct KVCacheTraits<ModelType::GLM_NSA> : KVCacheTraits<ModelType::DSV3_2> {
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::ARBITRARY_FP32;
};

template <>
struct KVCacheTraits<ModelType::GLM53_NOPE> {
  // GLM-5.3-Flash is a native NoPE model. The absorbed query and latent KV
  // dimensions are both 512; no positional-key lane exists.
  static constexpr int D_NOPE = 512;
  static constexpr int D_ROPE = 0;
  static constexpr int D_QK = D_NOPE;
  static constexpr int D_V = 512;

  static constexpr int QUANT_TILE = 128;
  static constexpr int NUM_SCALES = D_NOPE / QUANT_TILE;
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::ARBITRARY_FP32;

  // vLLM's fp8_ds_mla cache ABI remains 656 bytes/token. The first 528
  // bytes contain the 512 FP8 latent values plus four inline FP32 scales;
  // the trailing 128 bytes are reserved padding and must never be treated as
  // RoPE data by this specialization.
  static constexpr bool SCALE_INLINE = true;
  static constexpr int SCALE_BYTES_PER_TOKEN = NUM_SCALES * sizeof(float);
  static constexpr int KV_GMEM_STRIDE = 656;
  static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE;
  static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE + SCALE_BYTES_PER_TOKEN;
  static constexpr int KV_SMEM_STRIDE = D_NOPE + SCALE_BYTES_PER_TOKEN;
  static constexpr int KV_SMEM_COPY_BYTES = KV_SMEM_STRIDE;
  static constexpr bool SCALE_IN_KV_SMEM = true;

  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;
  static constexpr bool V_HAS_ROPE = false;

  __device__ static __forceinline__ uint8_t scale_to_ue8m0(float scale) {
    return static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFF);
  }
};

template <>
struct KVCacheTraits<ModelType::DOTS3_SWA> {
  // Sliding-window MLA: 1024-wide latent + 64-wide rope. First model with
  // D_V != 512 — see the assert block below.
  static constexpr int D_NOPE = 1024;
  static constexpr int D_ROPE = 64;
  static constexpr int D_QK = D_NOPE + D_ROPE;  // 1088
  static constexpr int D_V = D_NOPE;            // 1024, rope excluded (V_HAS_ROPE=false)

  // FP8 quantization. QUANT_TILE=128 (not DSV4's 64) keeps NUM_SCALES a power
  // of two, so the footer is 8B with no pad — unlike DSV4's 7+1.
  static constexpr int QUANT_TILE = 128;
  static constexpr int NUM_SCALES = D_NOPE / QUANT_TILE;  // 8
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::UE8M0_BYTE;

  // KV cache layout (FlashMLA ABI): FOOTER, 1160 logical bytes per token.
  // Physical layout per block (page_block_size tokens):
  //   [0 : block_size*1152)                 nope+rope data (1152B each)
  //     per token: [0:1024) FP8 nope, [1024:1152) BF16 rope
  //   [block_size*1152 : block_size*1160)   scale footer (8B each: 8×UE8M0)
  //
  // IO stride = 1152 (data only), 1152 % 16 = 0 ✓ for cp.async.bulk.
  static constexpr bool SCALE_INLINE = false;
  static constexpr int SCALE_BYTES_PER_TOKEN = 8;
  static constexpr int KV_GMEM_STRIDE =
      D_NOPE + D_ROPE * sizeof(bf16) + SCALE_BYTES_PER_TOKEN;                  // 1160
  static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE;                           // 1024
  static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE + D_ROPE * sizeof(bf16);  // 1152

  // Smem layout (nope only + padding, no rope, no inline scales).
  // stride=1040: 1040/4=260, 260%32=4 → same 4-way conflict class as DSV3_2's
  // 528, not DSV4's conflict-free 464. UNMEASURED — the M4b-equivalent
  // benchmark has not been run for this stride.
  // Must be 16B aligned for cp.async.bulk: 1040%16=0 ✓
  static constexpr int KV_SMEM_STRIDE = D_NOPE + 16;  // 1040
  static constexpr int KV_SMEM_COPY_BYTES = D_NOPE;   // copy 1024B nope per entry
  static constexpr bool SCALE_IN_KV_SMEM = false;

  // Q nope stride
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;      // 1040
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 1032 bf16 (2064 B)

  // V = pure nope. In this family's reference implementation the scores bmm
  // uses the full 1088-wide head dim, but the output bmm reads only the
  // 1024-wide latent (rope excluded).
  static constexpr bool V_HAS_ROPE = false;

  // UE8M0 scales are native — no conversion needed
  __device__ static __forceinline__ uint8_t scale_to_ue8m0(uint8_t scale) { return scale; }
};

template <>
struct KVCacheTraits<ModelType::DSV4> {
  // Dimensions
  static constexpr int D_NOPE = 448;
  static constexpr int D_ROPE = 64;
  static constexpr int D_QK = D_NOPE + D_ROPE;  // 512
  static constexpr int D_V = 512;               // = D_NOPE + D_ROPE

  // FP8 quantization
  static constexpr int QUANT_TILE = 64;
  static constexpr int NUM_SCALES = 7;  // D_NOPE / QUANT_TILE = 448/64
  static constexpr ScaleFormat SCALE_FORMAT = ScaleFormat::UE8M0_BYTE;

  // KV cache layout (FlashMLA ABI): FOOTER, 584 logical bytes per token
  // Physical layout per block (page_block_size tokens):
  //   [0 : block_size*576)                nope+rope data (576B each)
  //     per token: [0:448) FP8 nope, [448:576) BF16 rope
  //   [block_size*576 : block_size*584)   scale footer (8B each: 7×UE8M0 + 1 pad)
  //
  // stride_kv_row = 584 = logical bytes_per_token (PyTorch API stride, NOT IO stride)
  // IO stride = 576 (data only, 16B aligned for cp.async.bulk)
  static constexpr bool SCALE_INLINE = false;  // scales in footer, not inline
  static constexpr int SCALE_BYTES_PER_TOKEN = 8;
  static constexpr int KV_GMEM_STRIDE =
      D_NOPE + D_ROPE * sizeof(bf16) + SCALE_BYTES_PER_TOKEN;                  // 584
  static constexpr int KV_ROPE_GMEM_OFFSET = D_NOPE;                           // 448
  static constexpr int KV_SCALE_GMEM_OFFSET = D_NOPE + D_ROPE * sizeof(bf16);  // 576

  // Smem layout (nope only + padding, no rope, no inline scales)
  // stride=464: 464/4=116, 116%32=20 → clean (M4b benchmark verified: 12.9 ns/MMA)
  // Must be 16B aligned for cp.async.bulk: 464%16=0 ✓
  static constexpr int KV_SMEM_STRIDE = D_NOPE + 16;  // 464
  static constexpr int KV_SMEM_COPY_BYTES = D_NOPE;   // copy 448B nope per entry
  // DSV4: scales NOT in the bulk-copied region → loaded separately to kv_scale_bufs
  static constexpr bool SCALE_IN_KV_SMEM = false;

  // Q nope stride
  static constexpr int Q_NOPE_STRIDE = D_NOPE + 16;      // 464
  static constexpr int Q_NOPE_BF16_STRIDE = D_NOPE + 8;  // 456 bf16 (912 B)

  // V = nope[0:448] + rope[0:64]
  // XV nope: V_CHUNK=128, pad 448→512, MMA same as DSV3_2
  // XV rope: CUDA core scalar FMA from global (zero smem, M5b verified)
  static constexpr bool V_HAS_ROPE = true;

  // UE8M0 scales are native — no conversion needed
  __device__ static __forceinline__ uint8_t scale_to_ue8m0(uint8_t scale) { return scale; }
};

// ============================================================================
// Shared constants across all model types
// ============================================================================

static constexpr int HPB = 16;
static constexpr int BI = 64;
// D_V is shared across the DeepSeek-family models only; the asserts below pin
// the shared values to KVCacheTraits<...> so a new model with diverging values
// has to opt out explicitly (GLM53_NOPE already opts out of D_ROPE).
//
// DOTS3_SWA is the D_V opt-out: D_V = 1024. Anything reading the bare `D_V`
// below is therefore DeepSeek-family-only and must not be reached from a
// DOTS3_SWA instantiation — use KVCacheTraits<MT>::D_V.
static constexpr int D_ROPE = 64;
static constexpr int D_V = 512;
static_assert(KVCacheTraits<ModelType::DSV3_2>::D_ROPE == D_ROPE);
static_assert(KVCacheTraits<ModelType::DSV3_2>::D_V == D_V);
static_assert(KVCacheTraits<ModelType::DSV4>::D_ROPE == D_ROPE);
static_assert(KVCacheTraits<ModelType::DSV4>::D_V == D_V);
static_assert(KVCacheTraits<ModelType::GLM_NSA>::D_ROPE == D_ROPE);
static_assert(KVCacheTraits<ModelType::GLM_NSA>::D_V == D_V);
static_assert(KVCacheTraits<ModelType::GLM53_NOPE>::D_ROPE == 0);
static_assert(KVCacheTraits<ModelType::GLM53_NOPE>::D_V == D_V);
static_assert(KVCacheTraits<ModelType::DOTS3_SWA>::D_ROPE == D_ROPE);
static_assert(KVCacheTraits<ModelType::DOTS3_SWA>::D_V != D_V,
              "DOTS3_SWA is the D_V opt-out; if it ever equals 512, fold it back "
              "into the shared assert above");

// Warp configuration
static constexpr int N_MATH_WARPS = 8;
static constexpr int N_IO_WARPS = 4;
static constexpr int N_TOTAL_WARPS = N_MATH_WARPS + N_IO_WARPS;  // 12
static constexpr int BLOCK_THREADS = N_TOTAL_WARPS * 32;         // 384
static constexpr int MATH_THREADS = N_MATH_WARPS * 32;           // 256
static constexpr int IO_THREADS = N_IO_WARPS * 32;               // 128

static constexpr int ENTRIES_PER_WARP = BI / N_MATH_WARPS;  // 8
static constexpr int N_ROPE_CHUNKS = D_ROPE / 16;           // 4

// Output staging (reuses KV buffer after main loop). Each kernel derives
// its own staging geometry from OUT_VEC.
static constexpr int OUT_VEC = 8;

// ============================================================================
// ComputeMode + ModelType dependent parameters
// ============================================================================
//
// V_CHUNK = QUANT_TILE for each model (1:1 scale mapping, no max-of-tiles):
//   DSV3_2:    V_CHUNK=128 (QUANT_TILE=128, 4 chunks for D_NOPE=512)
//   DSV4: V_CHUNK=64  (QUANT_TILE=64,  7 chunks for D_NOPE=448)
//
// FP8 mode:
//   QK nope: FP8 MMA m16n8k32
//   QK rope: BF16 MMA m16n8k16
//   XV nope: FP8 MMA m16n8k32 (W quantized to FP8, V stays FP8)
//   XV rope (DSV4): BF16 MMA m16n8k16 (B from global, L2 cached)
//   Byte transpose required for V (FP8)
//
// BF16 mode:
//   IO dequants FP8 KV → BF16 in smem
//   QK/XV: BF16 MMA m16n8k16
//   V in smem is BF16 → ldmatrix.x2.trans (no byte transpose)

// ComputeTraits is parameterized on the tile config (candidates per iteration
// and math-warp count). TILE_BI / TILE_MATH_WARPS default to the namespace-scope
// BI / N_MATH_WARPS, so every `ComputeTraits<MT, CM>` spelling keeps the 64/8
// values it had before. A model running a different tile — DOTS3_SWA decodes at
// BI=32 with 4 math warps — must pass them explicitly, or V_TRANS_STRIDE /
// NT_PER_WARP_XV / XV_KSTEPS come out silently wrong.
template <ModelType MT, ComputeMode CM, int TILE_BI = BI, int TILE_MATH_WARPS = N_MATH_WARPS>
struct ComputeTraits;

template <ModelType MT, int TILE_BI, int TILE_MATH_WARPS>
struct ComputeTraits<MT, ComputeMode::FP8, TILE_BI, TILE_MATH_WARPS> {
  using KV = KVCacheTraits<MT>;
  static constexpr int V_CHUNK = KV::QUANT_TILE;                        // DSV3_2=128, DSV4=64
  static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;               // DSV3_2=4, DSV4=7
  static constexpr int V_TRANS_STRIDE = TILE_BI + 16;                   // 80 at BI=64
  static constexpr int W_FP8_STRIDE = TILE_BI + 16;                     // 80 at BI=64
  static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / TILE_MATH_WARPS;  // DSV3_2=2, DSV4=1
  static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;         // DSV3_2=8, DSV4=7
  static constexpr int XV_KSTEPS = TILE_BI / 32;                        // 2 (FP8 k=32)
  static_assert(NT_PER_WARP_XV >= 1, "V chunk too narrow for this math-warp count");
};

template <ModelType MT, int TILE_BI, int TILE_MATH_WARPS>
struct ComputeTraits<MT, ComputeMode::BF16, TILE_BI, TILE_MATH_WARPS> {
  using KV = KVCacheTraits<MT>;
  static constexpr int V_CHUNK = KV::QUANT_TILE;
  static constexpr int N_V_CHUNKS = KV::D_NOPE / V_CHUNK;
  static constexpr int V_TRANS_STRIDE = TILE_BI + 8;  // 72 bf16 elements at BI=64
  static constexpr int W_FP8_STRIDE = 0;
  static constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / TILE_MATH_WARPS;
  static constexpr int ACC_TILES = N_V_CHUNKS * NT_PER_WARP_XV;
  static constexpr int XV_KSTEPS = TILE_BI / 16;  // 4 (BF16 k=16)
  static_assert(NT_PER_WARP_XV >= 1, "V chunk too narrow for this math-warp count");
};
