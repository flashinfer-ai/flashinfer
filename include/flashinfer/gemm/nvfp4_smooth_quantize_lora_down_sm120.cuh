/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"

namespace flashinfer::gemm {

namespace smooth_quantize_lora_down_sm120_detail {

namespace quant = smooth_quantize_detail;

__device__ __forceinline__ std::uint32_t load_bf16_pair(quant::Type const* ptr) {
  return *reinterpret_cast<std::uint32_t const*>(ptr);
}

__device__ __forceinline__ std::uint32_t pack_bf16_pair(quant::Type const* low,
                                                        quant::Type const* high) {
  std::uint32_t const low_bits = *reinterpret_cast<std::uint16_t const*>(low);
  std::uint32_t const high_bits = *reinterpret_cast<std::uint16_t const*>(high);
  return low_bits | (high_bits << 16);
}

__device__ __forceinline__ void mma_m16n8k16_row_col(float& d0, float& d1, float& d2, float& d3,
                                                     std::uint32_t a0, std::uint32_t a1,
                                                     std::uint32_t a2, std::uint32_t a3,
                                                     std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}  // namespace smooth_quantize_lora_down_sm120_detail

namespace smooth_quantize_lora_down_small_m_sm120_detail {

namespace base = smooth_quantize_lora_down_sm120_detail;
namespace quant = smooth_quantize_detail;

constexpr int kRank = 32;
constexpr int kMmaN = 8;
constexpr int kL2PrefetchBytes = 128;

template <int K, int BlockThreads, int DownWarps, int DownTileCols, int RowsPerBlock>
__global__
__launch_bounds__(BlockThreads) void nvfp4_smooth_quantize_lora_down_m537_mixed_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, bool signal_quant_ready) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  constexpr int kM = 537;
  constexpr int kPaddedM = 640;
  constexpr int kBlockWarps = BlockThreads / 32;
  constexpr int kQuantWarps = kBlockWarps - DownWarps;
  constexpr int kQuantThreads = kQuantWarps * 32;
  constexpr int kActiveQuantThreads = kQuantThreads == 0 ? BlockThreads : kQuantThreads;
  constexpr int kDownNTiles = kRank / DownTileCols;
  constexpr int kDownElements = 16 * DownTileCols;
  constexpr int kGridBlocks = ((kM + 15) / 16) * kDownNTiles;
  constexpr int kSfCols = K / quant::SF_VEC_SIZE;
  constexpr int kKSteps = K / 16;
  constexpr int kL2PrefetchLines =
      K * kRank * static_cast<int>(sizeof(quant::Type)) / kL2PrefetchBytes;
  static_assert(BlockThreads % 32 == 0 && BlockThreads <= 1024);
  static_assert(DownWarps > 0 && DownWarps <= kBlockWarps);
  static_assert(DownTileCols % kMmaN == 0 && kRank % DownTileCols == 0);
  static_assert(BlockThreads >= kDownElements);
  static_assert(K % 16 == 0 && K % (4 * quant::SF_VEC_SIZE) == 0);
  static_assert(kGridBlocks * RowsPerBlock >= kM);
  static_assert(kGridBlocks * RowsPerBlock <= kPaddedM);
  static_assert(RowsPerBlock == kQuantWarps, "one quant warp must own each row in the mixed CTA");

  int const block = static_cast<int>(blockIdx.x);
  int const warp = static_cast<int>(threadIdx.x) / 32;
  int const lane = static_cast<int>(threadIdx.x) % 32;
  __shared__ __align__(8) float accumulator_tiles[DownWarps][kDownElements];

  constexpr int kPrefetchLinesPerBlock = (kL2PrefetchLines + kGridBlocks - 1) / kGridBlocks;
  int const prefetch_begin = block * kPrefetchLinesPerBlock;
  int const prefetch_limit = prefetch_begin + kPrefetchLinesPerBlock;
  int const prefetch_end = prefetch_limit < kL2PrefetchLines ? prefetch_limit : kL2PrefetchLines;
  for (int line = prefetch_begin + static_cast<int>(threadIdx.x); line < prefetch_end;
       line += BlockThreads) {
    std::uint8_t const* prefetch_address =
        reinterpret_cast<std::uint8_t const*>(l2t_smoothed) + line * kL2PrefetchBytes;
    asm volatile("prefetch.global.L2 [%0];" : : "l"(prefetch_address));
  }

  if (warp < DownWarps) {
    int const tile_m = block / kDownNTiles;
    int const tile_n = block % kDownNTiles;
    int const group = lane / 4;
    int const pair_col = (lane % 4) * 2;
    constexpr int kNFragments = DownTileCols / kMmaN;
    float d[kNFragments][4];
#pragma unroll
    for (int fragment = 0; fragment < kNFragments; ++fragment) {
      d[fragment][0] = 0.0f;
      d[fragment][1] = 0.0f;
      d[fragment][2] = 0.0f;
      d[fragment][3] = 0.0f;
    }

    int const step_begin = warp * kKSteps / DownWarps;
    int const step_end = (warp + 1) * kKSteps / DownWarps;
    int const a_row = tile_m * 16 + group;
    int const b_col = tile_n * DownTileCols + group;
#pragma unroll 1
    for (int step = step_begin; step < step_end; ++step) {
      int const k_base = step * 16;
      std::uint32_t a0 = 0u;
      std::uint32_t a1 = 0u;
      std::uint32_t a2 = 0u;
      std::uint32_t a3 = 0u;
      if (a_row < kM) {
        quant::Type const* a_base = x + a_row * K + k_base + pair_col;
        a0 = base::load_bf16_pair(a_base);
        a2 = base::load_bf16_pair(a_base + 8);
      }
      if (a_row + 8 < kM) {
        quant::Type const* a_upper = x + (a_row + 8) * K + k_base + pair_col;
        a1 = base::load_bf16_pair(a_upper);
        a3 = base::load_bf16_pair(a_upper + 8);
      }
#pragma unroll
      for (int fragment = 0; fragment < kNFragments; ++fragment) {
        std::uint32_t b0;
        std::uint32_t b1;
        if constexpr (K == 5120 || K == 5376 || K == 7168) {
          // The admitted M537 L2T is prepacked once as
          // [K/16, rank/16, DownTileCols/8, 32 lanes, 4 bf16].  One aligned
          // 64-bit load then supplies both B operands, replacing four
          // scattered U16 loads and their dependent pack-address IMAD chain.
          auto const* packed_l2t = reinterpret_cast<uint2 const*>(l2t_smoothed);
          int const packed_index =
              (((step * kDownNTiles + tile_n) * kNFragments + fragment) * 32) + lane;
          uint2 const packed_b = packed_l2t[packed_index];
          b0 = packed_b.x;
          b1 = packed_b.y;
        } else {
          quant::Type const* b_base = l2t_smoothed + (k_base + pair_col) * kRank + b_col;
          quant::Type const* b_fragment = b_base + fragment * kMmaN;
          b0 = base::pack_bf16_pair(b_fragment, b_fragment + kRank);
          b1 = base::pack_bf16_pair(b_fragment + 8 * kRank, b_fragment + 9 * kRank);
        }
        base::mma_m16n8k16_row_col(d[fragment][0], d[fragment][1], d[fragment][2], d[fragment][3],
                                   a0, a1, a2, a3, b0, b1);
      }
    }

#pragma unroll
    for (int fragment = 0; fragment < kNFragments; ++fragment) {
      int const output0 = (lane / 4) * DownTileCols + fragment * kMmaN + (lane % 4) * 2;
      int const output1 = output0 + 8 * DownTileCols;
      *reinterpret_cast<float2*>(&accumulator_tiles[warp][output0]) =
          make_float2(d[fragment][0], d[fragment][1]);
      *reinterpret_cast<float2*>(&accumulator_tiles[warp][output1]) =
          make_float2(d[fragment][2], d[fragment][3]);
    }
  }
  if (warp >= DownWarps || DownWarps == kBlockWarps) {
    int const quant_thread = kQuantThreads == 0 ? static_cast<int>(threadIdx.x)
                                                : static_cast<int>(threadIdx.x) - DownWarps * 32;
    int const row_base = block * RowsPerBlock;
    float const scale = global_scale[0];
    int const quant_warp = quant_thread / 32;
    int const quant_lane = quant_thread % 32;
    int const row = row_base + quant_warp;
    for (int sf_col = quant_lane; sf_col < kSfCols; sf_col += 32) {
      std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
      if (row < kM) {
        std::int64_t const vec_offset = static_cast<std::int64_t>(row) * kSfCols + sf_col;
        quant::Bf16x8 x_lo;
        quant::Bf16x8 x_hi;
        quant::Bf16x8 pqs_lo;
        quant::Bf16x8 pqs_hi;
        quant::loadBf16x8(x + vec_offset * quant::FAST_ELTS_PER_THREAD, x_lo);
        quant::loadBf16x8(x + vec_offset * quant::FAST_ELTS_PER_THREAD + 8, x_hi);
        quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD, pqs_lo);
        quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD + 8, pqs_hi);
        xq[vec_offset] =
            quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, sf + sf_offset);
      } else {
        sf[sf_offset] = 0u;
      }
    }

    constexpr int kCoveredRows = kGridBlocks * RowsPerBlock;
    constexpr int kPaddingRows = kPaddedM - kCoveredRows;
    constexpr std::int64_t kPaddingItems = static_cast<std::int64_t>(kPaddingRows) * kSfCols;
    for (std::int64_t item = static_cast<std::int64_t>(block) * kActiveQuantThreads + quant_thread;
         item < kPaddingItems;
         item += static_cast<std::int64_t>(kGridBlocks) * kActiveQuantThreads) {
      int const padding_row = static_cast<int>(item / kSfCols);
      int const sf_col = static_cast<int>(item - static_cast<std::int64_t>(padding_row) * kSfCols);
      int const row = kCoveredRows + padding_row;
      std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
      sf[sf_offset] = 0u;
    }

    // Exact row-44 combined path only: the residual GEMM depends on XQ/SF but
    // does not consume LoRA-down until its rank-32 tail.  Let the eight quant
    // warps in every CTA jointly signal that XQ/SF are complete while the 24
    // down warps continue their independent K reduction.  The launch attribute
    // is enabled only by the admitted PDL wrapper below.
    if (signal_quant_ready) {
      asm volatile("bar.sync 1, %0;" ::"r"(kQuantThreads) : "memory");
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }

  __syncthreads();
  int const element = static_cast<int>(threadIdx.x);
  if (element < kDownElements) {
    float accum = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < DownWarps; ++source_warp) {
      accum += accumulator_tiles[source_warp][element];
    }
    int const tile_m = block / kDownNTiles;
    int const tile_n = block % kDownNTiles;
    int const row = tile_m * 16 + element / DownTileCols;
    int const col = tile_n * DownTileCols + element % DownTileCols;
    if (row < kM) {
      down[row * kRank + col] = __float2bfloat16_rn(accum);
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

template <int K, int BlockThreads, int DownWarps, int DownTileCols, int RowsPerBlock>
inline cudaError_t launch_m537_mixed_kernel(void const* x, void const* pre_quant_scale,
                                            float const* global_scale, void* xq, void* sf,
                                            void const* l2t_smoothed, void* down,
                                            cudaStream_t stream, bool signal_quant_ready = false) {
  constexpr int kGridBlocks = ((537 + 15) / 16) * (kRank / DownTileCols);
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kGridBlocks);
  config.blockDim = dim3(BlockThreads);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attributes[1];
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = signal_quant_ready ? 1 : 0;
  config.attrs = attributes;
  config.numAttrs = 1;
  auto* kernel =
      &nvfp4_smooth_quantize_lora_down_m537_mixed_sm120_kernel<K, BlockThreads, DownWarps,
                                                               DownTileCols, RowsPerBlock>;
  cudaLaunchKernelEx(&config, kernel, static_cast<quant::Type const*>(x),
                     static_cast<quant::Type const*>(pre_quant_scale), global_scale,
                     static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
                     static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down),
                     signal_quant_ready);
  return cudaGetLastError();
}

template <int BlockThreads, int DownTileCols, int RowsPerQuantBlock, bool PackedL2T = false,
          bool SignalLaunch = false>
__global__
__launch_bounds__(BlockThreads) void nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, int M, int K) {
  static_assert(!SignalLaunch || PackedL2T, "launch signaling requires packed L2T");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  if constexpr (SignalLaunch) {
    // The PDL consumer can set up early, but waits for this grid before reading outputs.
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  int const kPaddedM = (M + 127) / 128 * 128;
  int const kQuantBlocks = (M + RowsPerQuantBlock - 1) / RowsPerQuantBlock;
  constexpr int kDownWarps = BlockThreads / 32;
  int const kSfCols = K / quant::SF_VEC_SIZE;
  int const kDownKPerWarp = K / kDownWarps;
  int const kL2PrefetchLines = K * kRank * static_cast<int>(sizeof(quant::Type)) / kL2PrefetchBytes;
  constexpr int kNFragments = DownTileCols / kMmaN;
  constexpr int kDownNTiles = kRank / DownTileCols;
  constexpr int kDownElements = 16 * DownTileCols;

  int const block = static_cast<int>(blockIdx.x);
  if (block < kQuantBlocks) {
    for (int line = block * BlockThreads + static_cast<int>(threadIdx.x); line < kL2PrefetchLines;
         line += kQuantBlocks * BlockThreads) {
      std::uint8_t const* prefetch_address =
          reinterpret_cast<std::uint8_t const*>(l2t_smoothed) + line * kL2PrefetchBytes;
      asm volatile("prefetch.global.L2 [%0];" : : "l"(prefetch_address));
    }

    float const scale = global_scale[0];
    int const row_base = block * RowsPerQuantBlock;
    int const kWorkItems = RowsPerQuantBlock * kSfCols;
    for (int item = static_cast<int>(threadIdx.x); item < kWorkItems; item += BlockThreads) {
      int const row_offset = item / kSfCols;
      int const sf_col = item - row_offset * kSfCols;
      int const row = row_base + row_offset;
      std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
      if (row < M) {
        std::int64_t const vec_offset = static_cast<std::int64_t>(row) * kSfCols + sf_col;
        quant::Bf16x8 x_lo;
        quant::Bf16x8 x_hi;
        quant::Bf16x8 pqs_lo;
        quant::Bf16x8 pqs_hi;
        quant::loadBf16x8(x + vec_offset * quant::FAST_ELTS_PER_THREAD, x_lo);
        quant::loadBf16x8(x + vec_offset * quant::FAST_ELTS_PER_THREAD + 8, x_hi);
        quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD, pqs_lo);
        quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD + 8, pqs_hi);
        xq[vec_offset] =
            quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, sf + sf_offset);
      } else {
        sf[sf_offset] = 0u;
      }
    }
    int const kCoveredRows = kQuantBlocks * RowsPerQuantBlock;
    int const kRemainingPaddingRows = kPaddedM - kCoveredRows;
    std::int64_t const kRemainingPaddingItems =
        static_cast<std::int64_t>(kRemainingPaddingRows) * kSfCols;
    for (std::int64_t item = static_cast<std::int64_t>(block) * BlockThreads + threadIdx.x;
         item < kRemainingPaddingItems;
         item += static_cast<std::int64_t>(kQuantBlocks) * BlockThreads) {
      int const padding_row = static_cast<int>(item / kSfCols);
      int const sf_col = static_cast<int>(item - static_cast<std::int64_t>(padding_row) * kSfCols);
      int const row = kCoveredRows + padding_row;
      std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
      sf[sf_offset] = 0u;
    }
    return;
  }

  int const down_tile = block - kQuantBlocks;
  int const tile_m = down_tile / kDownNTiles;
  int const tile_n = down_tile % kDownNTiles;
  __shared__ __align__(8) float accumulator_tiles[kDownWarps][kDownElements];

  int const warp = static_cast<int>(threadIdx.x) / 32;
  int const lane = static_cast<int>(threadIdx.x) % 32;
  int const group = lane / 4;
  int const pair_col = (lane % 4) * 2;
  float d[kNFragments][4];
#pragma unroll
  for (int fragment = 0; fragment < kNFragments; ++fragment) {
    d[fragment][0] = 0.0f;
    d[fragment][1] = 0.0f;
    d[fragment][2] = 0.0f;
    d[fragment][3] = 0.0f;
  }

  int const k_begin = warp * kDownKPerWarp;
  int const k_end = k_begin + kDownKPerWarp;
  int const a_row = tile_m * 16 + group;
  int const b_col = tile_n * DownTileCols + group;
#pragma unroll 1
  for (int k_base = k_begin; k_base < k_end; k_base += 16) {
    std::uint32_t a0 = 0u;
    std::uint32_t a1 = 0u;
    std::uint32_t a2 = 0u;
    std::uint32_t a3 = 0u;
    if (a_row < M) {
      quant::Type const* a_base = x + a_row * K + k_base + pair_col;
      a0 = base::load_bf16_pair(a_base);
      a2 = base::load_bf16_pair(a_base + 8);
    }
    if (a_row + 8 < M) {
      quant::Type const* a_upper = x + (a_row + 8) * K + k_base + pair_col;
      a1 = base::load_bf16_pair(a_upper);
      a3 = base::load_bf16_pair(a_upper + 8);
    }

    quant::Type const* b_base = l2t_smoothed + (k_base + pair_col) * kRank + b_col;
#pragma unroll
    for (int fragment = 0; fragment < kNFragments; ++fragment) {
      std::uint32_t b0;
      std::uint32_t b1;
      if constexpr (PackedL2T) {
        int const packed_index =
            (((k_base / 16 * kDownNTiles + tile_n) * kNFragments + fragment) * 32) + lane;
        uint2 const packed_b = reinterpret_cast<uint2 const*>(l2t_smoothed)[packed_index];
        b0 = packed_b.x;
        b1 = packed_b.y;
      } else {
        quant::Type const* b_fragment = b_base + fragment * kMmaN;
        b0 = base::pack_bf16_pair(b_fragment, b_fragment + kRank);
        b1 = base::pack_bf16_pair(b_fragment + 8 * kRank, b_fragment + 9 * kRank);
      }
      base::mma_m16n8k16_row_col(d[fragment][0], d[fragment][1], d[fragment][2], d[fragment][3], a0,
                                 a1, a2, a3, b0, b1);
    }
  }

#pragma unroll
  for (int fragment = 0; fragment < kNFragments; ++fragment) {
    int const output0 = group * DownTileCols + fragment * kMmaN + pair_col;
    int const output1 = output0 + 8 * DownTileCols;
    *reinterpret_cast<float2*>(&accumulator_tiles[warp][output0]) =
        make_float2(d[fragment][0], d[fragment][1]);
    *reinterpret_cast<float2*>(&accumulator_tiles[warp][output1]) =
        make_float2(d[fragment][2], d[fragment][3]);
  }
  __syncthreads();

  int const element = static_cast<int>(threadIdx.x);
  if (element < kDownElements) {
    float accum = 0.0f;
#pragma unroll
    for (int source_warp = 0; source_warp < kDownWarps; ++source_warp) {
      accum += accumulator_tiles[source_warp][element];
    }
    int const row = tile_m * 16 + element / DownTileCols;
    int const col = tile_n * DownTileCols + element % DownTileCols;
    if (row < M) {
      down[row * kRank + col] = __float2bfloat16_rn(accum);
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

// Runtime-M/K small-M launch. Only the tiling stays in the type; the shape and
// everything derived from it are arguments, so one instantiation per tiling
// serves every shape whose K divides the warp split.
template <int BlockThreads, int DownTileCols, int RowsPerQuantBlock, bool PackedL2T = false,
          bool SignalLaunch = false>
inline cudaError_t launch_small_m_dyn_kernel(void const* x, void const* pre_quant_scale,
                                             float const* global_scale, void* xq, void* sf,
                                             void const* l2t_smoothed, void* down, int m, int k,
                                             cudaStream_t stream) {
  constexpr int kDownWarps = BlockThreads / 32;
  constexpr int kNFragments = DownTileCols / kMmaN;
  constexpr int kDownNTiles = kRank / DownTileCols;
  if constexpr (BlockThreads % 32 || kDownWarps <= 0 || DownTileCols % kMmaN || kNFragments <= 0 ||
                kDownNTiles * DownTileCols != kRank) {
    return cudaErrorInvalidValue;
  } else {
    // What SmallMLaunchGeometry asserted at compile time is a precondition here.
    if (m <= 0 || RowsPerQuantBlock <= 0) return cudaErrorInvalidValue;
    if (k % kDownWarps || (k / kDownWarps) % 16) return cudaErrorInvalidValue;
    if (k % (4 * quant::SF_VEC_SIZE)) return cudaErrorInvalidValue;
    int const padded_m = (m + 127) / 128 * 128;
    int const quant_blocks = (m + RowsPerQuantBlock - 1) / RowsPerQuantBlock;
    if (quant_blocks * RowsPerQuantBlock > padded_m) return cudaErrorInvalidValue;
    int const grid_blocks = quant_blocks + ((m + 15) / 16) * kDownNTiles;
    // Keep the primary's ordinary stream dependency, including the preceding GEMM.
    nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel<
        BlockThreads, DownTileCols, RowsPerQuantBlock, PackedL2T, SignalLaunch>
        <<<grid_blocks, BlockThreads, 0, stream>>>(
            static_cast<quant::Type const*>(x), static_cast<quant::Type const*>(pre_quant_scale),
            global_scale, static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
            static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down), m, k);
    return cudaGetLastError();
  }
}

}  // namespace smooth_quantize_lora_down_small_m_sm120_detail

namespace smooth_quantize_lora_down_large_m_sm120_detail {

namespace base = smooth_quantize_lora_down_sm120_detail;
namespace quant = smooth_quantize_detail;

constexpr int kRank = 32;
constexpr int kDownWarps = kRank / 8;

__device__ __forceinline__ std::uint64_t make_l2_evict_first_policy() {
  std::uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  return policy;
}

template <bool UsePackedStore>
__device__ __forceinline__ void store_down_pair(quant::Type* destination, float low, float high) {
  if constexpr (UsePackedStore) {
    __nv_bfloat162_raw const packed =
        static_cast<__nv_bfloat162_raw>(__floats2bfloat162_rn(low, high));
    *reinterpret_cast<std::uint32_t*>(destination) =
        static_cast<std::uint32_t>(packed.x) | (static_cast<std::uint32_t>(packed.y) << 16);
  } else {
    destination[0] = __float2bfloat16_rn(low);
    destination[1] = __float2bfloat16_rn(high);
  }
}

__device__ __forceinline__ void commit_x_tile_async() {
  asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void wait_x_tile_async() {
  asm volatile("cp.async.wait_group 0;\n" : :);
}

// How the runtime-K row offset is formed. Recompute derives it from the loop
// counter every iteration, which leaves the compiler free to overlap
// iterations; Accumulate carries it, which trades that freedom for one add in
// place of a multiply. Which wins is a property of the geometry, not of the
// shape, and it is measured rather than assumed.
enum class DynAddress { kRecompute, kAccumulate };

template <int BlockThreads, bool LongK, int KernelTileM, int KernelTileK = 128,
          DynAddress Policy = DynAddress::kRecompute>
__device__ __forceinline__ void copy_x_tile_async_dyn(quant::Type const* __restrict__ x,
                                                      quant::Type* __restrict__ x_tile,
                                                      int row_base, int tile_k, int thread,
                                                      std::uint64_t l2_policy, int M, int K) {
  constexpr int kKernelTileM = KernelTileM;
  constexpr bool kUseStreamingX = LongK;
  constexpr int kKernelQuantThreads = BlockThreads - kDownWarps * 32;
  constexpr int kLoadThreads = LongK ? kKernelQuantThreads : BlockThreads;
  constexpr int kTileK = KernelTileK;
  constexpr int kTileStride = kTileK + 8;
  constexpr int kLoadItems = kKernelTileM * kTileK / 8;
  int const load_thread = LongK ? thread - kDownWarps * 32 : thread;
  constexpr int kColsPerRow = kTileK / 8;
  constexpr bool kWholeRowStep = kLoadThreads % kColsPerRow == 0;
  std::int64_t const k64 = K;
  // Accumulate needs the thread stride to cover whole rows, otherwise the
  // column moves too and there is nothing to carry.
  if constexpr (Policy == DynAddress::kAccumulate && kWholeRowStep) {
    constexpr int kRowsPerStep = kLoadThreads / kColsPerRow;
    int const load_col = (load_thread % kColsPerRow) * 8;
    int load_row = load_thread / kColsPerRow;
    int global_row = row_base + load_row;
    std::int64_t row_offset = static_cast<std::int64_t>(global_row) * k64;
    std::int64_t const row_step = static_cast<std::int64_t>(kRowsPerStep) * k64;
    for (int item = load_thread; item >= 0 && item < kLoadItems; item += kLoadThreads) {
      quant::Type const* source = x + (global_row < M ? row_offset : 0) + tile_k + load_col;
      quant::Type* destination = x_tile + load_row * kTileStride + load_col;
      std::uint32_t const shared_address =
          static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
      int const source_bytes = global_row < M ? 16 : 0;
      if constexpr (kUseStreamingX) {
        asm volatile(
            "cp.async.cg.shared.global.L2::cache_hint "
            "[%0], [%1], 16, %2, %3;\n"
            :
            : "r"(shared_address), "l"(source), "r"(source_bytes), "l"(l2_policy));
      } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(shared_address), "l"(source), "r"(source_bytes));
      }
      load_row += kRowsPerStep;
      global_row += kRowsPerStep;
      row_offset += row_step;
    }
    return;
  }
  for (int item = load_thread; item >= 0 && item < kLoadItems; item += kLoadThreads) {
    int const load_row = item / kColsPerRow;
    int const load_col = (item % kColsPerRow) * 8;
    int const global_row = row_base + load_row;
    quant::Type const* source =
        x + static_cast<std::int64_t>(global_row < M ? global_row : 0) * k64 + tile_k + load_col;
    quant::Type* destination = x_tile + load_row * kTileStride + load_col;
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
    int const source_bytes = global_row < M ? 16 : 0;
    if constexpr (kUseStreamingX) {
      asm volatile(
          "cp.async.cg.shared.global.L2::cache_hint "
          "[%0], [%1], 16, %2, %3;\n"
          :
          : "r"(shared_address), "l"(source), "r"(source_bytes), "l"(l2_policy));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :
                   : "r"(shared_address), "l"(source), "r"(source_bytes));
    }
  }
}

template <bool LongK, int KernelTileK = 128>
__device__ __forceinline__ void copy_pre_quant_scale_tile_async_dyn(
    quant::Type const* __restrict__ pre_quant_scale, quant::Type* __restrict__ scale_tile,
    int tile_k, int thread) {
  constexpr int kTileK = KernelTileK;
  constexpr int kLoadItems = kTileK / 8;
  int const load_thread = LongK ? thread - kDownWarps * 32 : thread;
  if (load_thread >= 0 && load_thread < kLoadItems) {
    quant::Type const* source = pre_quant_scale + tile_k + load_thread * 8;
    quant::Type* destination = scale_tile + load_thread * 8;
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(shared_address), "l"(source));
  }
}

template <int BlockThreads, bool LongK, int KernelTileM, int KernelTileK = 128,
          DynAddress Policy = DynAddress::kRecompute>
__global__
__launch_bounds__(BlockThreads) void nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, int M, int K) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  constexpr int kKernelTileM = KernelTileM;
  constexpr bool kUseStreamingX = LongK;
  constexpr int kKernelQuantThreads = BlockThreads - kDownWarps * 32;
  constexpr int kTileK = KernelTileK;
  constexpr int kTileStride = kTileK + 8;
  constexpr int kSfColsPerTile = kTileK / quant::SF_VEC_SIZE;
  int const kPaddedM = (M + 127) / 128 * 128;
  static_assert((kKernelTileM * kSfColsPerTile) % kKernelQuantThreads == 0);
  int const kSfCols = K / quant::SF_VEC_SIZE;
  __shared__ __align__(16) quant::Type x_tiles[2][kKernelTileM * kTileStride];
  __shared__ __align__(16) quant::Type scale_tiles[2][kTileK];
  __shared__ __align__(4) std::uint8_t sf_tiles[2][LongK ? kKernelTileM * kSfColsPerTile : 1];

  int const thread = static_cast<int>(threadIdx.x);
  int const warp = thread / 32;
  int const lane = thread % 32;
  int const group = lane / 4;
  int const pair_col = (lane % 4) * 2;
  float const scale = global_scale[0];
  std::uint64_t l2_evict_first_policy = 0;
  if constexpr (kUseStreamingX) {
    l2_evict_first_policy = make_l2_evict_first_policy();
  }
  if constexpr (LongK) {
    constexpr int kL2PrefetchBytes = 128;
    int const kL2Bytes = K * kRank * static_cast<int>(sizeof(quant::Type));
    int const kL2PrefetchLines = kL2Bytes / kL2PrefetchBytes;
    int const prefetch_line = static_cast<int>(blockIdx.x) * BlockThreads + thread;
    if (prefetch_line < kL2PrefetchLines) {
      std::uint8_t const* prefetch_address =
          reinterpret_cast<std::uint8_t const*>(l2t_smoothed) + prefetch_line * kL2PrefetchBytes;
      asm volatile("prefetch.global.L2 [%0];" : : "l"(prefetch_address));
    }
  }

  int const row_tile = static_cast<int>(blockIdx.x);
  int const row_base = row_tile * kKernelTileM;
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  float d4 = 0.0f;
  float d5 = 0.0f;
  float d6 = 0.0f;
  float d7 = 0.0f;
  float d8 = 0.0f;
  float d9 = 0.0f;
  float d10 = 0.0f;
  float d11 = 0.0f;
  float d12 = 0.0f;
  float d13 = 0.0f;
  float d14 = 0.0f;
  float d15 = 0.0f;
  float d16 = 0.0f;
  float d17 = 0.0f;
  float d18 = 0.0f;
  float d19 = 0.0f;

  copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy>(
      x, x_tiles[0], row_base, 0, thread, l2_evict_first_policy, M, K);
  copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK>(pre_quant_scale, scale_tiles[0], 0,
                                                          thread);
  commit_x_tile_async();
  wait_x_tile_async();
  __syncthreads();

  int tile_buffer = 0;
#pragma unroll 1
  for (int tile_k = 0; tile_k < K; tile_k += kTileK) {
    int const next_tile_k = tile_k + kTileK;
    int const next_buffer = tile_buffer ^ 1;
    if (next_tile_k < K) {
      copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy>(
          x, x_tiles[next_buffer], row_base, next_tile_k, thread, l2_evict_first_policy, M, K);
      copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK>(
          pre_quant_scale, scale_tiles[next_buffer], next_tile_k, thread);
      commit_x_tile_async();
    }
    quant::Type const* x_tile = x_tiles[tile_buffer];
    quant::Type const* scale_tile = scale_tiles[tile_buffer];
    std::uint8_t* sf_tile = sf_tiles[tile_buffer];

    if (warp < kDownWarps) {
#pragma unroll
      for (int local_k = 0; local_k < kTileK; local_k += 16) {
        quant::Type const* a_base = x_tile + group * kTileStride + local_k + pair_col;
        std::uint32_t const a0 = base::load_bf16_pair(a_base);
        std::uint32_t const a1 = base::load_bf16_pair(a_base + 8 * kTileStride);
        std::uint32_t const a2 = base::load_bf16_pair(a_base + 8);
        std::uint32_t const a3 = base::load_bf16_pair(a_base + 8 * kTileStride + 8);

        int const b_col = warp * 8 + group;
        int const b_row = tile_k + local_k + pair_col;
        quant::Type const* b_base = l2t_smoothed + b_row * kRank + b_col;
        std::uint32_t const b0 = base::pack_bf16_pair(b_base, b_base + kRank);
        std::uint32_t const b1 = base::pack_bf16_pair(b_base + 8 * kRank, b_base + 9 * kRank);
        base::mma_m16n8k16_row_col(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        if constexpr (kKernelTileM >= 32) {
          quant::Type const* a_upper = a_base + 16 * kTileStride;
          std::uint32_t const a4 = base::load_bf16_pair(a_upper);
          std::uint32_t const a5 = base::load_bf16_pair(a_upper + 8 * kTileStride);
          std::uint32_t const a6 = base::load_bf16_pair(a_upper + 8);
          std::uint32_t const a7 = base::load_bf16_pair(a_upper + 8 * kTileStride + 8);
          base::mma_m16n8k16_row_col(d4, d5, d6, d7, a4, a5, a6, a7, b0, b1);
        }
        if constexpr (kKernelTileM >= 48) {
          quant::Type const* a_lower = a_base + 32 * kTileStride;
          std::uint32_t const a8 = base::load_bf16_pair(a_lower);
          std::uint32_t const a9 = base::load_bf16_pair(a_lower + 8 * kTileStride);
          std::uint32_t const a10 = base::load_bf16_pair(a_lower + 8);
          std::uint32_t const a11 = base::load_bf16_pair(a_lower + 8 * kTileStride + 8);
          base::mma_m16n8k16_row_col(d8, d9, d10, d11, a8, a9, a10, a11, b0, b1);
        }
        if constexpr (kKernelTileM >= 64) {
          quant::Type const* a_bottom = a_base + 48 * kTileStride;
          std::uint32_t const a12 = base::load_bf16_pair(a_bottom);
          std::uint32_t const a13 = base::load_bf16_pair(a_bottom + 8 * kTileStride);
          std::uint32_t const a14 = base::load_bf16_pair(a_bottom + 8);
          std::uint32_t const a15 = base::load_bf16_pair(a_bottom + 8 * kTileStride + 8);
          base::mma_m16n8k16_row_col(d12, d13, d14, d15, a12, a13, a14, a15, b0, b1);
        }
        if constexpr (kKernelTileM >= 80) {
          quant::Type const* a_tail = a_base + 64 * kTileStride;
          std::uint32_t const a16 = base::load_bf16_pair(a_tail);
          std::uint32_t const a17 = base::load_bf16_pair(a_tail + 8 * kTileStride);
          std::uint32_t const a18 = base::load_bf16_pair(a_tail + 8);
          std::uint32_t const a19 = base::load_bf16_pair(a_tail + 8 * kTileStride + 8);
          base::mma_m16n8k16_row_col(d16, d17, d18, d19, a16, a17, a18, a19, b0, b1);
        }
      }
    } else {
      constexpr int kQuantItems = kKernelTileM * kSfColsPerTile;
      for (int quant_item = thread - kDownWarps * 32; quant_item < kQuantItems;
           quant_item += kKernelQuantThreads) {
        int const row_offset = quant_item / kSfColsPerTile;
        int const sf_in_tile = quant_item - row_offset * kSfColsPerTile;
        int const row = row_base + row_offset;
        int const sf_col = tile_k / quant::SF_VEC_SIZE + sf_in_tile;
        if (row < M) {
          quant::Bf16x8 x_lo;
          quant::Bf16x8 x_hi;
          quant::Bf16x8 pqs_lo;
          quant::Bf16x8 pqs_hi;
          quant::Type const* tile_ptr =
              x_tile + row_offset * kTileStride + sf_in_tile * quant::FAST_ELTS_PER_THREAD;
          quant::loadBf16x8(tile_ptr, x_lo);
          quant::loadBf16x8(tile_ptr + 8, x_hi);
          quant::Type const* pqs_ptr = scale_tile + sf_in_tile * quant::FAST_ELTS_PER_THREAD;
          quant::loadBf16x8(pqs_ptr, pqs_lo);
          quant::loadBf16x8(pqs_ptr + 8, pqs_hi);
          std::int64_t const vec_offset = static_cast<std::int64_t>(row) * kSfCols + sf_col;
          if constexpr (LongK) {
            xq[vec_offset] =
                quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, sf_tile + quant_item);
          } else {
            std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
            std::uint64_t const quantized =
                quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, sf + sf_offset);
            xq[vec_offset] = quantized;
          }
        } else if (row < kPaddedM) {
          if constexpr (LongK) {
            sf_tile[quant_item] = 0u;
          } else {
            std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
            sf[sf_offset] = 0u;
          }
        }
      }
    }
    if (next_tile_k < K) {
      wait_x_tile_async();
    }
    if constexpr (LongK) {
      __syncthreads();
      constexpr int kSfGroupsPerTile = kSfColsPerTile / 4;
      constexpr int kSfStoresPerTile = kKernelTileM * kSfGroupsPerTile;
      for (int sf_store = thread - kDownWarps * 32; sf_store >= 0 && sf_store < kSfStoresPerTile;
           sf_store += kKernelQuantThreads) {
        int const row_offset = sf_store / kSfGroupsPerTile;
        int const sf_group = sf_store - row_offset * kSfGroupsPerTile;
        int const row = row_base + row_offset;
        int const sf_col = tile_k / quant::SF_VEC_SIZE + sf_group * 4;
        if (row < kPaddedM) {
          std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
          *reinterpret_cast<std::uint32_t*>(sf + sf_offset) =
              *reinterpret_cast<std::uint32_t const*>(sf_tile + row_offset * kSfColsPerTile +
                                                      sf_group * 4);
        }
      }
      if (next_tile_k < K) {
        tile_buffer = next_buffer;
      }
    } else if (next_tile_k < K) {
      __syncthreads();
      tile_buffer = next_buffer;
    }
  }

  if (warp < kDownWarps) {
    constexpr bool kUsePackedDownStore = false;
    int const row0 = row_base + group;
    int const row1 = row0 + 8;
    int const col = warp * 8 + pair_col;
    if (row0 < M) {
      store_down_pair<kUsePackedDownStore>(down + row0 * kRank + col, d0, d1);
    }
    if (row1 < M) {
      store_down_pair<kUsePackedDownStore>(down + row1 * kRank + col, d2, d3);
    }
    if constexpr (kKernelTileM >= 32) {
      int const row2 = row0 + 16;
      int const row3 = row0 + 24;
      if (row2 < M) {
        store_down_pair<kUsePackedDownStore>(down + row2 * kRank + col, d4, d5);
      }
      if (row3 < M) {
        store_down_pair<kUsePackedDownStore>(down + row3 * kRank + col, d6, d7);
      }
    }
    if constexpr (kKernelTileM >= 48) {
      int const row4 = row0 + 32;
      int const row5 = row0 + 40;
      if (row4 < M) {
        store_down_pair<kUsePackedDownStore>(down + row4 * kRank + col, d8, d9);
      }
      if (row5 < M) {
        store_down_pair<kUsePackedDownStore>(down + row5 * kRank + col, d10, d11);
      }
    }
    if constexpr (kKernelTileM >= 64) {
      int const row6 = row0 + 48;
      int const row7 = row0 + 56;
      if (row6 < M) {
        store_down_pair<kUsePackedDownStore>(down + row6 * kRank + col, d12, d13);
      }
      if (row7 < M) {
        store_down_pair<kUsePackedDownStore>(down + row7 * kRank + col, d14, d15);
      }
    }
    if constexpr (kKernelTileM >= 80) {
      int const row8 = row0 + 64;
      int const row9 = row0 + 72;
      if (row8 < M) {
        store_down_pair<kUsePackedDownStore>(down + row8 * kRank + col, d16, d17);
      }
      if (row9 < M) {
        store_down_pair<kUsePackedDownStore>(down + row9 * kRank + col, d18, d19);
      }
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

// Runtime-M/K launch. The geometry stays a template parameter -- shared memory
// and every unroll constant come from it -- while M and K become arguments, so
// one instantiation serves every shape whose K divides the tile.
template <int BlockThreads, bool LongK, int TileM, int TileK, DynAddress Policy>
inline cudaError_t launch_large_m_dyn_kernel(void const* x, void const* pre_quant_scale,
                                             float const* global_scale, void* xq, void* sf,
                                             void const* l2t_smoothed, void* down, int m, int k,
                                             cudaStream_t stream) {
  // KernelLaunchGeometry passes TileM straight through; the kernel's default
  // template argument (K >= 8192 ? 80 : kTileM) is never what the launcher uses.
  constexpr int kKernelTileM = TileM;
  constexpr int kQuantThreads = BlockThreads - kDownWarps * 32;
  constexpr int kSfColsPerTile = TileK / quant::SF_VEC_SIZE;
  // Guard the instantiation, not just the call: a (geometry, LongK) pair whose
  // quantize work does not divide across its threads has no kernel to build,
  // and the templated path never produced one because such a pair never got
  // matched to a shape.
  constexpr int kSharedBytes =
      2 * kKernelTileM * (TileK + 8) * static_cast<int>(sizeof(quant::Type)) +
      2 * TileK * static_cast<int>(sizeof(quant::Type)) +
      2 * (LongK ? kKernelTileM * kSfColsPerTile : 1);
  if constexpr ((kKernelTileM * kSfColsPerTile) % kQuantThreads != 0 || kSharedBytes > 48 * 1024) {
    return cudaErrorInvalidValue;
  } else {
    // What the templated kernel asserted at compile time is a precondition here.
    if (k % TileK != 0) return cudaErrorInvalidValue;
    if (LongK != (k >= 8192)) return cudaErrorInvalidValue;
    int const padded_m = (m + 127) / 128 * 128;
    int const grid_blocks = (padded_m + kKernelTileM - 1) / kKernelTileM;
    if constexpr (LongK) {
      int const l2_lines = k * kRank * static_cast<int>(sizeof(quant::Type)) / 128;
      if (grid_blocks * BlockThreads < l2_lines) return cudaErrorInvalidValue;
    }
    nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel<BlockThreads, LongK, kKernelTileM,
                                                             TileK, Policy>
        <<<grid_blocks, BlockThreads, 0, stream>>>(
            static_cast<quant::Type const*>(x), static_cast<quant::Type const*>(pre_quant_scale),
            global_scale, static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
            static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down), m, k);
    return cudaGetLastError();
  }
}

}  // namespace smooth_quantize_lora_down_large_m_sm120_detail

// Runtime-M/K entry. Geometry is still a template parameter -- shared memory and
// every unroll constant come from it -- so this switch is over the ladder, not
// over shapes. One instantiation per (geometry, LongK) serves every shape.
// The instantiated candidate ladders, written once. `dyn`/`family` expand them
// to dispatch; the default entry below expands the same lists to enumerate. They
// mirror SM120_PRODUCER_GEOMETRY_LADDER / SM120_SMALL_M_TILING_LADDER in
// flashinfer/gemm/svdquant_sm120_routes.py, which is what lets Python offer a
// candidate this file can actually launch.
#define FI_SVDQ_LARGE_M_GEOMETRIES(X) \
  X(192, 16, 512)                     \
  X(192, 32, 128)                     \
  X(192, 48, 128)                     \
  X(224, 48, 128)                     \
  X(256, 32, 256)                     \
  X(256, 80, 128)                     \
  X(288, 80, 128)                     \
  X(384, 32, 256)

#define FI_SVDQ_SMALL_M_TILINGS(X) \
  X(256, 16, 4)                    \
  X(768, 8, 8)                     \
  X(768, 16, 16)                   \
  X(768, 32, 16)                   \
  X(896, 16, 16)                   \
  X(1024, 8, 4)                    \
  X(1024, 8, 16)                   \
  X(1024, 16, 16)

inline cudaError_t nvfp4_smooth_quantize_lora_down_dyn_sm120(
    void const* x, void const* pre_quant_scale, float const* global_scale, void* xq, void* sf,
    void const* l2t_smoothed, void* down, int m, int k, int block_threads, int tile_m, int tile_k,
    cudaStream_t stream, int address_policy = 0) {
  namespace large_m = smooth_quantize_lora_down_large_m_sm120_detail;
  using large_m::DynAddress;
  bool const long_k = k >= 8192;
  bool const accumulate = address_policy == 1;
#define FI_SVDQ_DYN_CASE(BT, TM, TK)                                                               \
  if (block_threads == (BT) && tile_m == (TM) && tile_k == (TK)) {                                 \
    if (long_k) {                                                                                  \
      return accumulate                                                                            \
                 ? large_m::launch_large_m_dyn_kernel<BT, true, TM, TK, DynAddress::kAccumulate>(  \
                       x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream) \
                 : large_m::launch_large_m_dyn_kernel<BT, true, TM, TK, DynAddress::kRecompute>(   \
                       x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k,         \
                       stream);                                                                    \
    }                                                                                              \
    return accumulate                                                                              \
               ? large_m::launch_large_m_dyn_kernel<BT, false, TM, TK, DynAddress::kAccumulate>(   \
                     x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream)   \
               : large_m::launch_large_m_dyn_kernel<BT, false, TM, TK, DynAddress::kRecompute>(    \
                     x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);  \
  }
  FI_SVDQ_LARGE_M_GEOMETRIES(FI_SVDQ_DYN_CASE)
#undef FI_SVDQ_DYN_CASE
  return cudaErrorInvalidValue;
}

// Producer launch by family and tiling, with no shape lookup at all. The caller
// enumerates what a shape may run from the same constraints the launchers check,
// so this switch selects compiled families and tilings, and a shape
// nobody wrote down reaches the same kernels as one that was.
//   family 0 large-M (geometry a, b, c = block threads, tile M, tile K)
//   family 1 row-major small-M (a, b, c = block threads, down tile cols, rows/quant block)
//   family 2 fixed-M537 with packed L2T (a, b, c = block threads, down warps, rows/quant block)
//   family 4 packed small-M (same tiling fields as family 1)
// Family 3 uses the separate cuBLASLt prefix entry.
inline cudaError_t nvfp4_smooth_quantize_lora_down_family_sm120(
    void const* x, void const* pre_quant_scale, float const* global_scale, void* xq, void* sf,
    void const* l2t_smoothed, void* down, int m, int k, int family, int a, int b, int c,
    int address_policy, cudaStream_t stream, bool signal_launch = false) {
  namespace large_m = smooth_quantize_lora_down_large_m_sm120_detail;
  namespace small_m = smooth_quantize_lora_down_small_m_sm120_detail;
  using large_m::DynAddress;
  if (family == 2) {
    // One M, parameterised on K: the role split bakes M in, so this family
    // enumerates for the M it was written for and nothing else.
    if (m != 537) return cudaErrorInvalidValue;
    if (k == 5120) {
      return small_m::launch_m537_mixed_kernel<5120, 1024, 24, 16, 8>(
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, stream);
    }
    if (k == 5376) {
      return small_m::launch_m537_mixed_kernel<5376, 1024, 24, 16, 8>(
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, stream);
    }
    if (k == 7168) {
      return small_m::launch_m537_mixed_kernel<7168, 1024, 24, 16, 8>(
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, stream);
    }
    return cudaErrorInvalidValue;
  }
  if (family == 4) {
#define FI_SVDQ_PACKED_SMALL_CASE(BT, DTC, RPQB)                                       \
  if (a == (BT) && b == (DTC) && c == (RPQB)) {                                        \
    if (signal_launch) {                                                               \
      return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB, true, true>(            \
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream); \
    }                                                                                  \
    return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB, true>(                    \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);   \
  }
    FI_SVDQ_SMALL_M_TILINGS(FI_SVDQ_PACKED_SMALL_CASE)
#undef FI_SVDQ_PACKED_SMALL_CASE
    return cudaErrorInvalidValue;
  }
  if (family == 1) {
#define FI_SVDQ_SMALL_CASE(BT, DTC, RPQB)                                            \
  if (a == (BT) && b == (DTC) && c == (RPQB)) {                                      \
    return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB>(                        \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream); \
  }
    FI_SVDQ_SMALL_M_TILINGS(FI_SVDQ_SMALL_CASE)
#undef FI_SVDQ_SMALL_CASE
    return cudaErrorInvalidValue;
  }
  return nvfp4_smooth_quantize_lora_down_dyn_sm120(x, pre_quant_scale, global_scale, xq, sf,
                                                   l2t_smoothed, down, m, k, a, b, c, stream,
                                                   address_policy);
}

inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120(
    void const* x, void const* pre_quant_scale, float const* global_scale, void* xq, void* sf,
    void const* l2t_smoothed, void* down, int m, int k, cudaStream_t stream,
    bool signal_m537_quant_ready = false, int geometry_variant = 0) {
  // This was 399 lines of `if (m == A && k == B)`, each branch launching the
  // producer that shape had been pinned to -- the last place a shape chose its
  // own kernel. Choosing is the autotuner's job now: the fused route passes an
  // explicit (family, tiling, address policy) that came from
  // sm120_producer_variants(), and reaches the launchers through
  // nvfp4_smooth_quantize_lora_down_family_sm120.
  //
  // What remains here is the entry for a caller with no opinion, and it forms
  // one by asking rather than by naming shapes: every launcher validates its own
  // preconditions and returns cudaErrorInvalidValue before it launches anything,
  // so the first candidate that does not refuse is one this shape can run. The
  // ladders come from the same lists the dispatch expands, so a candidate
  // offered here is always a candidate that exists.
  //
  // Row-major families only. M537 and packed small-M read L2T prepacked;
  // making either a default would hand it whatever layout a caller happened to
  // pass, which is silent when it is wrong. A caller that wants it names
  // family 2 or 4.
  (void)signal_m537_quant_ready;
  (void)geometry_variant;
#define FI_SVDQ_TRY(FAMILY, A, B, C)                                                            \
  {                                                                                             \
    cudaError_t const status = nvfp4_smooth_quantize_lora_down_family_sm120(                    \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, FAMILY, A, B, C, 0, \
        stream);                                                                                \
    if (status != cudaErrorInvalidValue) return status;                                         \
  }
#define FI_SVDQ_TRY_LARGE(BT, TM, TK) FI_SVDQ_TRY(0, BT, TM, TK)
#define FI_SVDQ_TRY_SMALL(BT, DTC, RPQB) FI_SVDQ_TRY(1, BT, DTC, RPQB)
  FI_SVDQ_LARGE_M_GEOMETRIES(FI_SVDQ_TRY_LARGE)
  FI_SVDQ_SMALL_M_TILINGS(FI_SVDQ_TRY_SMALL)
#undef FI_SVDQ_TRY_SMALL
#undef FI_SVDQ_TRY_LARGE
#undef FI_SVDQ_TRY
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm
