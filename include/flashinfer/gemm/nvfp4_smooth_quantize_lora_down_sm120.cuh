/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "flashinfer/gemm/nvfp4_smooth_quantize_lora_down_ws_sm120.cuh"
#include "flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"
#include "flashinfer/gemm/svdquant_quantize_barrier_sm120.cuh"

namespace flashinfer::gemm {

namespace smooth_quantize_lora_down_sm120_detail {

namespace quant = smooth_quantize_detail;

__device__ __forceinline__ std::uint32_t load_bf16_pair(quant::Type const* ptr) {
  return *reinterpret_cast<std::uint32_t const*>(ptr);
}

/**
 * @brief Four 8x8 bf16 matrices from shared memory; lane l supplies the 16-byte row
 * address of row l % 8 of matrix l / 8, and r[j] receives matrix j in the mma
 * fragment layout (row = lane / 4, columns 2 * (lane % 4) and +1).
 */
__device__ __forceinline__ void ldmatrix_x4(std::uint32_t& r0, std::uint32_t& r1, std::uint32_t& r2,
                                            std::uint32_t& r3, quant::Type const* row) {
  std::uint32_t const address = static_cast<std::uint32_t>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
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

/**
 * @brief Quantize CTAs of the row-major small-M kernel. Ranks 32/64 give each CTA
 * RowsPerQuantBlock rows; rank 128 gives each thread one (row, scale column)
 * item of the 128-row-padded SF tile, so the count follows M x K rather than M.
 */
template <int BlockThreads, int RowsPerQuantBlock, int Rank>
__host__ __device__ __forceinline__ int small_m_quant_blocks(int m, int sf_cols) {
  if constexpr (Rank > 64) {
    std::int64_t const items = static_cast<std::int64_t>((m + 127) / 128 * 128) * sf_cols;
    return static_cast<int>((items + BlockThreads - 1) / BlockThreads);
  } else {
    (void)sf_cols;
    return (m + RowsPerQuantBlock - 1) / RowsPerQuantBlock;
  }
}

/**
 * @brief Fuse smoothing, NVFP4 quantization, and LoRA-down with separate CTA groups.
 * Rank 128 reads K-major L2T; lower ranks use row-major or explicitly packed L2T.
 * Optional PDL signaling publishes quantize stores before the down CTAs finish.
 */
template <int BlockThreads, int DownTileCols, int RowsPerQuantBlock, bool PackedL2T = false,
          bool SignalLaunch = false, int Rank = 32, bool GateLaunchOnQuant = false, int KSplit = 1>
__global__
__launch_bounds__(BlockThreads) void nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, int M, int K,
    SvdquantQuantizeBarrierSm120 quantize_barrier = {}) {
  constexpr int kRank = Rank;
  // Rank only sets how many DownTileCols-wide column tiles the down CTAs cover;
  // a CTA's fragments, shared accumulator and K split do not depend on it.
  static_assert(Rank == 32 || Rank == 64 || Rank == 128);
  static_assert(!PackedL2T || Rank == 32);
  static_assert(!SignalLaunch || PackedL2T || Rank == 128,
                "launch signaling is built for packed L2T and rank 128 only");
  static_assert(!GateLaunchOnQuant || (SignalLaunch && Rank == 128),
                "quantize-gated launch signaling is built for rank 128 only");
  static_assert(KSplit == 1 || Rank == 128, "split-K down is built for rank 128 only");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  if constexpr (SignalLaunch && !GateLaunchOnQuant) {
    // The PDL consumer can set up early, but waits for this grid before reading outputs.
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  int const kPaddedM = (M + 127) / 128 * 128;
  // Split-K launches KSplit-CTA clusters; the quantize CTAs are rounded up so no
  // cluster straddles the quantize/down boundary (surplus quantize CTAs find no item).
  int const kQuantBlocks =
      (small_m_quant_blocks<BlockThreads, RowsPerQuantBlock, Rank>(M, K / quant::SF_VEC_SIZE) +
       KSplit - 1) /
      KSplit * KSplit;
  if constexpr (GateLaunchOnQuant) {
    // The consumer GEMM runs its FP4 mainloop without waiting for this grid and
    // waits only before loading LoRA-down, so it may launch only once XQ/SF are
    // written. Down CTAs signal now; quantize CTAs signal by exiting after
    // fencing their stores, so the GEMM launches when the quantize half is done
    // while LoRA-down keeps running under its mainloop.
    if (static_cast<int>(blockIdx.x) >= kQuantBlocks && threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  constexpr int kDownWarps = BlockThreads / 32;
  int const kSfCols = K / quant::SF_VEC_SIZE;
  int const kDownKPerWarp = K / kDownWarps;
  std::int64_t const kL2PrefetchLines =
      static_cast<std::int64_t>(K) * kRank * sizeof(quant::Type) / kL2PrefetchBytes;
  constexpr int kNFragments = DownTileCols / kMmaN;
  constexpr int kDownNTiles = kRank / DownTileCols;
  constexpr int kDownElements = 16 * DownTileCols;

  int const block = static_cast<int>(blockIdx.x);
  if (block < kQuantBlocks) {
    for (std::int64_t line = static_cast<std::int64_t>(block) * BlockThreads + threadIdx.x;
         line < kL2PrefetchLines; line += static_cast<std::int64_t>(kQuantBlocks) * BlockThreads) {
      std::uint8_t const* prefetch_address =
          reinterpret_cast<std::uint8_t const*>(l2t_smoothed) + line * kL2PrefetchBytes;
      asm volatile("prefetch.global.L2 [%0];" : : "l"(prefetch_address));
    }

    float const scale = global_scale[0];
    if constexpr (Rank > 64) {
      // One (row, scale column) item per thread over the whole padded SF tile,
      // padding rows included (zero scales). Splitting by item exposes more
      // independent CTAs than splitting only by row for small M and large K.
      std::int64_t const items = static_cast<std::int64_t>(kPaddedM) * kSfCols;
      std::int64_t const item = static_cast<std::int64_t>(block) * BlockThreads + threadIdx.x;
      if (item < items) {
        int const row = static_cast<int>(item / kSfCols);
        int const sf_col = static_cast<int>(item - static_cast<std::int64_t>(row) * kSfCols);
        std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, kSfCols);
        if (row < M) {
          quant::Bf16x8 x_lo;
          quant::Bf16x8 x_hi;
          quant::Bf16x8 pqs_lo;
          quant::Bf16x8 pqs_hi;
          quant::loadBf16x8(x + item * quant::FAST_ELTS_PER_THREAD, x_lo);
          quant::loadBf16x8(x + item * quant::FAST_ELTS_PER_THREAD + 8, x_hi);
          quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD, pqs_lo);
          quant::loadBf16x8(pre_quant_scale + sf_col * quant::FAST_ELTS_PER_THREAD + 8, pqs_hi);
          xq[item] = quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, sf + sf_offset);
        } else {
          sf[sf_offset] = 0u;
        }
      }
      if constexpr (GateLaunchOnQuant) {
        // Exiting is this CTA's launch signal; see the top of the kernel.
        __threadfence();
        quantize_barrier.arrive();
      }
      return;
    }
    int const row_base = block * RowsPerQuantBlock;
    int const kWorkItems = RowsPerQuantBlock * kSfCols;
    for (std::int64_t item = threadIdx.x; item < kWorkItems; item += BlockThreads) {
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

  int const down_tile = (block - kQuantBlocks) / KSplit;
  int const split = (block - kQuantBlocks) % KSplit;
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

  int k_begin = warp * kDownKPerWarp;
  int k_end = k_begin + kDownKPerWarp;
  if constexpr (KSplit > 1) {
    // The cluster's CTAs take consecutive whole 32-k chunks of K, and each CTA's
    // warps split its share the same way, so K need only be a multiple of 32.
    int const chunks = K / 32;
    int const cta_begin = split * chunks / KSplit;
    int const cta_chunks = (split + 1) * chunks / KSplit - cta_begin;
    k_begin = (cta_begin + warp * cta_chunks / kDownWarps) * 32;
    k_end = (cta_begin + (warp + 1) * cta_chunks / kDownWarps) * 32;
  }
  int const a_row = tile_m * 16 + group;
  int const b_col = tile_n * DownTileCols + group;
  if constexpr (Rank > 64) {
    // Rank 128 reads K-major L2T (l2[rank][K]). The MMA sums over k, so A and
    // B only have to agree on which physical k sits in each k slot: lane t of a
    // quad takes k 8t..8t+7 of a 32-k chunk with one 16-byte load per operand
    // row, feeding two k16 steps (slots {2t,2t+1} and {2t+8,2t+9} get k 8t..8t+3
    // in the first step, 8t+4..8t+7 in the second). That is a quarter of the
    // load instructions of per-step 32-bit loads and whole 32-byte sectors per
    // quad, and every chunk of a kDownPrefetch group is loaded before its MMAs.
    // The launcher requires K / warps to be a multiple of 32.
    constexpr int kDownPrefetch = 2;
    bool const has_low = a_row < M;
    bool const has_high = a_row + 8 < M;
    int const quad_k = (lane % 4) * 8;
    quant::Type const* const a_low = x + static_cast<std::int64_t>(a_row) * K + quad_k;
    quant::Type const* const a_high = a_low + static_cast<std::int64_t>(8) * K;
    quant::Type const* const b_lane = l2t_smoothed + static_cast<std::int64_t>(b_col) * K + quad_k;
#pragma unroll 1
    for (int group_base = k_begin; group_base < k_end; group_base += kDownPrefetch * 32) {
      uint4 a_lo[kDownPrefetch];
      uint4 a_hi[kDownPrefetch];
      uint4 b[kDownPrefetch][kNFragments];
#pragma unroll
      for (int c = 0; c < kDownPrefetch; ++c) {
        int const k_base = group_base + c * 32;
        bool const live = k_base < k_end;
        uint4 const zero = make_uint4(0u, 0u, 0u, 0u);
        a_lo[c] = live && has_low ? *reinterpret_cast<uint4 const*>(a_low + k_base) : zero;
        a_hi[c] = live && has_high ? *reinterpret_cast<uint4 const*>(a_high + k_base) : zero;
#pragma unroll
        for (int fragment = 0; fragment < kNFragments; ++fragment) {
          b[c][fragment] =
              live ? *reinterpret_cast<uint4 const*>(
                         b_lane + static_cast<std::int64_t>(fragment * kMmaN) * K + k_base)
                   : zero;
        }
      }
#pragma unroll
      for (int c = 0; c < kDownPrefetch; ++c) {
#pragma unroll
        for (int fragment = 0; fragment < kNFragments; ++fragment) {
          base::mma_m16n8k16_row_col(d[fragment][0], d[fragment][1], d[fragment][2], d[fragment][3],
                                     a_lo[c].x, a_hi[c].x, a_lo[c].y, a_hi[c].y, b[c][fragment].x,
                                     b[c][fragment].y);
          base::mma_m16n8k16_row_col(d[fragment][0], d[fragment][1], d[fragment][2], d[fragment][3],
                                     a_lo[c].z, a_hi[c].z, a_lo[c].w, a_hi[c].w, b[c][fragment].z,
                                     b[c][fragment].w);
        }
      }
    }
  } else {
#pragma unroll 1
    for (int k_base = k_begin; k_base < k_end; k_base += 16) {
      std::uint32_t a0 = 0u;
      std::uint32_t a1 = 0u;
      std::uint32_t a2 = 0u;
      std::uint32_t a3 = 0u;
      if (a_row < M) {
        quant::Type const* a_base = x + static_cast<std::int64_t>(a_row) * K + k_base + pair_col;
        a0 = base::load_bf16_pair(a_base);
        a2 = base::load_bf16_pair(a_base + 8);
      }
      if (a_row + 8 < M) {
        quant::Type const* a_upper =
            x + (static_cast<std::int64_t>(a_row) + 8) * K + k_base + pair_col;
        a1 = base::load_bf16_pair(a_upper);
        a3 = base::load_bf16_pair(a_upper + 8);
      }

      quant::Type const* b_base =
          l2t_smoothed + (static_cast<std::int64_t>(k_base) + pair_col) * kRank + b_col;
#pragma unroll
      for (int fragment = 0; fragment < kNFragments; ++fragment) {
        std::uint32_t b0;
        std::uint32_t b1;
        if constexpr (PackedL2T) {
          std::int64_t const packed_index =
              (((static_cast<std::int64_t>(k_base) / 16 * kDownNTiles + tile_n) * kNFragments +
                fragment) *
               32) +
              lane;
          uint2 const packed_b = reinterpret_cast<uint2 const*>(l2t_smoothed)[packed_index];
          b0 = packed_b.x;
          b1 = packed_b.y;
        } else {
          quant::Type const* b_fragment = b_base + fragment * kMmaN;
          b0 = base::pack_bf16_pair(b_fragment, b_fragment + kRank);
          b1 = base::pack_bf16_pair(b_fragment + 8 * kRank, b_fragment + 9 * kRank);
        }
        base::mma_m16n8k16_row_col(d[fragment][0], d[fragment][1], d[fragment][2], d[fragment][3],
                                   a0, a1, a2, a3, b0, b1);
      }
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
  float accum = 0.0f;
  if (element < kDownElements) {
#pragma unroll
    for (int source_warp = 0; source_warp < kDownWarps; ++source_warp) {
      accum += accumulator_tiles[source_warp][element];
    }
  }
  if constexpr (KSplit > 1) {
    // Fixed-order cluster reduction through distributed shared memory: the
    // leader adds the CTAs' partials in split order, so the result does not
    // depend on which CTA finishes first.
    __shared__ float cta_partial[kDownElements];
    if (element < kDownElements) {
      cta_partial[element] = accum;
    }
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (cluster.block_rank() == 0 && element < kDownElements) {
      accum = 0.0f;
#pragma unroll
      for (int source = 0; source < KSplit; ++source) {
        accum += cluster.map_shared_rank(cta_partial, source)[element];
      }
    }
    cluster.sync();
    if (split != 0) {
      return;
    }
  }
  if (element < kDownElements) {
    int const row = tile_m * 16 + element / DownTileCols;
    int const col = tile_n * DownTileCols + element % DownTileCols;
    if (row < M) {
      down[static_cast<std::int64_t>(row) * kRank + col] = __float2bfloat16_rn(accum);
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

/**
 * @brief Runtime-M/K small-M launch. Only the tiling stays in the type; the shape and
 * everything derived from it are arguments, so one instantiation per tiling
 * serves every shape whose K divides the warp split.
 */
template <int BlockThreads, int DownTileCols, int RowsPerQuantBlock, bool PackedL2T = false,
          bool SignalLaunch = false, int Rank = 32, bool GateLaunchOnQuant = false, int KSplit = 1>
inline cudaError_t launch_small_m_dyn_kernel(void const* x, void const* pre_quant_scale,
                                             float const* global_scale, void* xq, void* sf,
                                             void const* l2t_smoothed, void* down, int m, int k,
                                             cudaStream_t stream,
                                             SvdquantQuantizeBarrierSm120 quantize_barrier = {}) {
  constexpr int kRank = Rank;
  constexpr int kDownWarps = BlockThreads / 32;
  constexpr int kNFragments = DownTileCols / kMmaN;
  constexpr int kDownNTiles = kRank / DownTileCols;
  if constexpr (BlockThreads % 32 || kDownWarps <= 0 || DownTileCols % kMmaN || kNFragments <= 0 ||
                kDownNTiles * DownTileCols != kRank) {
    return cudaErrorInvalidValue;
  } else {
    // What SmallMLaunchGeometry asserted at compile time is a precondition here.
    if (m <= 0 || m > std::numeric_limits<int>::max() - 127 || k <= 0 || RowsPerQuantBlock <= 0) {
      return cudaErrorInvalidValue;
    }
    if constexpr (KSplit > 1) {
      // Every warp of every split owns at least one 32-k chunk.
      if (k % 32 || k / 32 < KSplit * kDownWarps) return cudaErrorInvalidValue;
    } else {
      if (k % kDownWarps || (k / kDownWarps) % 16) return cudaErrorInvalidValue;
      // The rank-128 down loop consumes K in 32-element chunks.
      if (kRank > 64 && (k / kDownWarps) % 32) return cudaErrorInvalidValue;
    }
    if (k % (4 * quant::SF_VEC_SIZE)) return cudaErrorInvalidValue;
    int const padded_m = (m + 127) / 128 * 128;
    int const quant_blocks =
        small_m_quant_blocks<BlockThreads, RowsPerQuantBlock, Rank>(m, k / quant::SF_VEC_SIZE);
    if (kRank <= 64 && quant_blocks * RowsPerQuantBlock > padded_m) return cudaErrorInvalidValue;
    int const grid_blocks =
        (quant_blocks + KSplit - 1) / KSplit * KSplit + ((m + 15) / 16) * kDownNTiles * KSplit;
    auto* kernel = &nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel<
        BlockThreads, DownTileCols, RowsPerQuantBlock, PackedL2T, SignalLaunch, Rank,
        GateLaunchOnQuant, KSplit>;
    if constexpr (KSplit > 1) {
      // Keep the primary's ordinary stream dependency, including the preceding GEMM.
      cudaLaunchConfig_t config = {};
      config.gridDim = dim3(grid_blocks);
      config.blockDim = dim3(BlockThreads);
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      cudaLaunchAttribute attributes[1];
      attributes[0].id = cudaLaunchAttributeClusterDimension;
      attributes[0].val.clusterDim.x = KSplit;
      attributes[0].val.clusterDim.y = 1;
      attributes[0].val.clusterDim.z = 1;
      config.attrs = attributes;
      config.numAttrs = 1;
      cudaLaunchKernelEx(&config, kernel, static_cast<quant::Type const*>(x),
                         static_cast<quant::Type const*>(pre_quant_scale), global_scale,
                         static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
                         static_cast<quant::Type const*>(l2t_smoothed),
                         static_cast<quant::Type*>(down), m, k, quantize_barrier);
    } else {
      // Keep the primary's ordinary stream dependency, including the preceding GEMM.
      kernel<<<grid_blocks, BlockThreads, 0, stream>>>(
          static_cast<quant::Type const*>(x), static_cast<quant::Type const*>(pre_quant_scale),
          global_scale, static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
          static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down), m, k,
          quantize_barrier);
    }
    return cudaGetLastError();
  }
}

}  // namespace smooth_quantize_lora_down_small_m_sm120_detail

namespace smooth_quantize_lora_down_large_m_sm120_detail {

namespace base = smooth_quantize_lora_down_sm120_detail;
namespace quant = smooth_quantize_detail;

__device__ __forceinline__ std::uint64_t make_l2_evict_first_policy() {
  std::uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  return policy;
}

/**
 * @brief LoRA-down warps of the large-M producer. Up to rank 64 every warp owns one
 * n8 column fragment (Rank / 8 warps). Wider ranks keep eight down warps and
 * give each one Rank / 64 adjacent fragments, so the CTA keeps room for its
 * quantize warps and one A fragment load feeds several MMAs.
 */
constexpr int large_m_down_warps(int rank) { return rank <= 64 ? rank / 8 : 8; }

/**
 * @brief Wide ranks read L2T from a K-major copy, l2[rank][K] stored in the [K, rank]
 * tensor. Row-major L2T makes each B-fragment pair two bf16 reads 2*Rank bytes
 * apart, so a warp's load touches 32 lines for 64 useful bytes.
 * K-major turns each pair into one aligned 32-bit read and a warp
 * load into eight 16-byte segments. Rank 32/64 keep the row-major layout.
 */
constexpr bool large_m_kmajor_l2(int rank) { return rank > 64; }

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

/**
 * @brief Stage an X tile with cp.async and zero-fill rows outside M.
 * The caller commits and waits for the copy group before reading shared memory.
 */
template <int BlockThreads, bool LongK, int KernelTileM, int KernelTileK = 128,
          DynAddress Policy = DynAddress::kRecompute, int Rank = 32>
__device__ __forceinline__ void copy_x_tile_async_dyn(quant::Type const* __restrict__ x,
                                                      quant::Type* __restrict__ x_tile,
                                                      int row_base, int tile_k, int thread,
                                                      std::uint64_t l2_policy, int M, int K) {
  constexpr int kDownWarps = large_m_down_warps(Rank);
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

/**
 * @brief Stage one K tile of smoothing scales; the caller commits and waits for copies.
 */
template <bool LongK, int KernelTileK = 128, int Rank = 32>
__device__ __forceinline__ void copy_pre_quant_scale_tile_async_dyn(
    quant::Type const* __restrict__ pre_quant_scale, quant::Type* __restrict__ scale_tile,
    int tile_k, int thread) {
  constexpr int kDownWarps = large_m_down_warps(Rank);
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

/**
 * @brief Fuse quantization and LoRA-down using double-buffered X tiles.
 * Rank 128 loads B fragments from K-major L2T; ranks 32/64 use row-major L2T.
 */
template <int BlockThreads, bool LongK, int KernelTileM, int KernelTileK = 128,
          DynAddress Policy = DynAddress::kRecompute, int Rank = 32>
__global__
__launch_bounds__(BlockThreads) void nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, int M, int K) {
  constexpr int kRank = Rank;
  constexpr int kDownWarps = large_m_down_warps(Rank);
  static_assert(Rank == 32 || Rank == 64 || Rank == 128);
  // n8 column fragments per down warp; 1 keeps the rank-32/64 code path verbatim.
  constexpr int kFrags = Rank / (8 * kDownWarps);
  static_assert(kFrags * 8 * kDownWarps == Rank);
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
    std::int64_t const kL2Bytes = static_cast<std::int64_t>(K) * kRank * sizeof(quant::Type);
    std::int64_t const kL2PrefetchLines = kL2Bytes / kL2PrefetchBytes;
    std::int64_t const prefetch_line =
        static_cast<std::int64_t>(blockIdx.x) * BlockThreads + thread;
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
  // Wide-rank accumulators: [fragment][m16 block][mma.sync C register].
  constexpr int kMBlocks = kKernelTileM / 16;
  static_assert(kFrags == 1 || kKernelTileM % 16 == 0);
  float wide_acc[kFrags == 1 ? 1 : kFrags][kFrags == 1 ? 1 : kMBlocks][4] = {};

  copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy, Rank>(
      x, x_tiles[0], row_base, 0, thread, l2_evict_first_policy, M, K);
  copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK, Rank>(pre_quant_scale, scale_tiles[0], 0,
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
      copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy, Rank>(
          x, x_tiles[next_buffer], row_base, next_tile_k, thread, l2_evict_first_policy, M, K);
      copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK, Rank>(
          pre_quant_scale, scale_tiles[next_buffer], next_tile_k, thread);
      commit_x_tile_async();
    }
    quant::Type const* x_tile = x_tiles[tile_buffer];
    quant::Type const* scale_tile = scale_tiles[tile_buffer];
    std::uint8_t* sf_tile = sf_tiles[tile_buffer];

    if (warp < kDownWarps) {
      if constexpr (kFrags == 1) {
#pragma unroll
        for (int local_k = 0; local_k < kTileK; local_k += 16) {
          quant::Type const* a_base = x_tile + group * kTileStride + local_k + pair_col;
          std::uint32_t const a0 = base::load_bf16_pair(a_base);
          std::uint32_t const a1 = base::load_bf16_pair(a_base + 8 * kTileStride);
          std::uint32_t const a2 = base::load_bf16_pair(a_base + 8);
          std::uint32_t const a3 = base::load_bf16_pair(a_base + 8 * kTileStride + 8);

          int const b_col = warp * 8 + group;
          int const b_row = tile_k + local_k + pair_col;
          quant::Type const* b_base =
              l2t_smoothed + static_cast<std::int64_t>(b_row) * kRank + b_col;
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
        // Wide rank: one A fragment per m16 block feeds kFrags adjacent n8 MMAs.
        // Every B fragment of the tile is issued before the first MMA so the
        // global-load latency overlaps instead of serialising each k step.
        constexpr int kKSteps = kTileK / 16;
        std::uint32_t b0[kKSteps][kFrags];
        std::uint32_t b1[kKSteps][kFrags];
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
          int const b_row = tile_k + ks * 16 + pair_col;
#pragma unroll
          for (int f = 0; f < kFrags; ++f) {
            int const b_col = (warp * kFrags + f) * 8 + group;
            if constexpr (large_m_kmajor_l2(Rank)) {
              quant::Type const* b_base =
                  l2t_smoothed + static_cast<std::int64_t>(b_col) * K + b_row;
              b0[ks][f] = base::load_bf16_pair(b_base);
              b1[ks][f] = base::load_bf16_pair(b_base + 8);
            } else {
              quant::Type const* b_base =
                  l2t_smoothed + static_cast<std::int64_t>(b_row) * kRank + b_col;
              b0[ks][f] = base::pack_bf16_pair(b_base, b_base + kRank);
              b1[ks][f] = base::pack_bf16_pair(b_base + 8 * kRank, b_base + 9 * kRank);
            }
          }
        }
#pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
          int const local_k = ks * 16;
#pragma unroll
          for (int mb = 0; mb < kMBlocks; ++mb) {
            quant::Type const* a_base =
                x_tile + (mb * 16 + group) * kTileStride + local_k + pair_col;
            std::uint32_t const a0 = base::load_bf16_pair(a_base);
            std::uint32_t const a1 = base::load_bf16_pair(a_base + 8 * kTileStride);
            std::uint32_t const a2 = base::load_bf16_pair(a_base + 8);
            std::uint32_t const a3 = base::load_bf16_pair(a_base + 8 * kTileStride + 8);
#pragma unroll
            for (int f = 0; f < kFrags; ++f) {
              base::mma_m16n8k16_row_col(wide_acc[f][mb][0], wide_acc[f][mb][1], wide_acc[f][mb][2],
                                         wide_acc[f][mb][3], a0, a1, a2, a3, b0[ks][f], b1[ks][f]);
            }
          }
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

  if (warp < kDownWarps && kFrags > 1) {
    // Wide rank: each thread owns a bf16 pair per (fragment, m16 block, row half).
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      int const col = (warp * kFrags + f) * 8 + pair_col;
#pragma unroll
      for (int mb = 0; mb < kMBlocks; ++mb) {
        int const row_lo = row_base + mb * 16 + group;
        int const row_hi = row_lo + 8;
        if (row_lo < M) {
          store_down_pair<false>(down + static_cast<std::int64_t>(row_lo) * kRank + col,
                                 wide_acc[f][mb][0], wide_acc[f][mb][1]);
        }
        if (row_hi < M) {
          store_down_pair<false>(down + static_cast<std::int64_t>(row_hi) * kRank + col,
                                 wide_acc[f][mb][2], wide_acc[f][mb][3]);
        }
      }
    }
  } else if (warp < kDownWarps) {
    constexpr bool kUsePackedDownStore = false;
    int const row0 = row_base + group;
    int const row1 = row0 + 8;
    int const col = warp * 8 + pair_col;
    if (row0 < M) {
      store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row0) * kRank + col, d0,
                                           d1);
    }
    if (row1 < M) {
      store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row1) * kRank + col, d2,
                                           d3);
    }
    if constexpr (kKernelTileM >= 32) {
      int const row2 = row0 + 16;
      int const row3 = row0 + 24;
      if (row2 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row2) * kRank + col,
                                             d4, d5);
      }
      if (row3 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row3) * kRank + col,
                                             d6, d7);
      }
    }
    if constexpr (kKernelTileM >= 48) {
      int const row4 = row0 + 32;
      int const row5 = row0 + 40;
      if (row4 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row4) * kRank + col,
                                             d8, d9);
      }
      if (row5 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row5) * kRank + col,
                                             d10, d11);
      }
    }
    if constexpr (kKernelTileM >= 64) {
      int const row6 = row0 + 48;
      int const row7 = row0 + 56;
      if (row6 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row6) * kRank + col,
                                             d12, d13);
      }
      if (row7 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row7) * kRank + col,
                                             d14, d15);
      }
    }
    if constexpr (kKernelTileM >= 80) {
      int const row8 = row0 + 64;
      int const row9 = row0 + 72;
      if (row8 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row8) * kRank + col,
                                             d16, d17);
      }
      if (row9 < M) {
        store_down_pair<kUsePackedDownStore>(down + static_cast<std::int64_t>(row9) * kRank + col,
                                             d18, d19);
      }
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

/**
 * @brief Wide ranks with a K64 tile stage B (the K-major L2T) in shared memory.
 *
 * The K128 wide-rank path above reads its B fragments with per-fragment global
 * loads at the start of every k tile, so those loads -- their latency and the
 * L2 traffic of every CTA re-reading the whole L2T -- sit on the down warps'
 * critical path once per tile. Here each
 * down warp copies its own B slice of the next tile with cp.async while it runs
 * the current one, and reads the current one from shared memory. A K128 tile's
 * 32 KB B tile does not fit next to two x buffers at two CTAs per SM, which is
 * why this is the K64 path; the K128 kernel is left untouched.
 */
constexpr bool large_m_stages_b(int rank, int tile_k) { return rank > 64 && tile_k == 64; }
// One down warp's slice: its 16 columns (two n8 fragments), K-major, 64 k per
// row padded by eight bf16 so the eight fragment groups hit distinct bank quads.
constexpr int kStagedBCols = 16;
constexpr int kStagedBStride = 64 + 8;
constexpr int kStagedBSliceBytes = kStagedBCols * kStagedBStride * 2;
constexpr int kStagedBSlices = 8;  // Rank / kStagedBCols column groups at rank 128
/**
 * @brief A row tile above 80 rows is split between two sets of eight down warps that
 * share each column group's B slice: the CTA reads L2T once for twice the rows,
 * halving the L2T traffic per row (one CTA per SM, dynamic shared memory).
 */
constexpr int staged_b_row_split(int tile_m) { return tile_m > 80 ? 2 : 1; }
/**
 * @brief Allocate one down warp per B column group and row partition.
 */
constexpr int staged_b_down_warps(int tile_m) {
  return kStagedBSlices * staged_b_row_split(tile_m);
}
/**
 * @brief Size the two X/scale buffers, aligned SF scratch, and shared B slices in bytes.
 */
constexpr int staged_b_shared_bytes(int tile_m, int tile_k, bool long_k) {
  return 2 * tile_m * (tile_k + 8) * 2 + 2 * tile_k * 2 +
         (2 * (long_k ? tile_m * (tile_k / 16) : 1) + 15) / 16 * 16 +
         kStagedBSlices * kStagedBSliceBytes;
}

/**
 * @brief Stage part of one column group's K-major B slice with cp.async.
 * The caller commits and waits for the group before reading the shared slice.
 */
template <int Parts>
__device__ __forceinline__ void copy_b_slice_async(quant::Type const* __restrict__ l2t_kmajor,
                                                   quant::Type* __restrict__ slice, int col_group,
                                                   int part, int lane, int tile_k, int K) {
  constexpr int kChunksPerCol = 64 / 8;
  constexpr int kChunks = kStagedBCols * kChunksPerCol / Parts;
#pragma unroll
  for (int i = lane; i < kChunks; i += 32) {
    int const chunk = part * kChunks + i;
    int const col = chunk / kChunksPerCol;
    int const k8 = (chunk % kChunksPerCol) * 8;
    quant::Type const* source =
        l2t_kmajor + static_cast<std::int64_t>(col_group * kStagedBCols + col) * K + tile_k + k8;
    std::uint32_t const shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(slice + col * kStagedBStride + k8));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(shared_address), "l"(source));
  }
}

/**
 * @brief Prefetch x into L2 for the staged-B producer, two K64 tiles per row at a time.
 *
 * A K64 tile reads 128 contiguous bytes of each x row, which keeps DRAM below
 * the bandwidth that longer row segments reach. Every kStagedBXPrefetchGroup
 * tiles the quantize threads bulk-prefetch the next kStagedBXPrefetchGroup
 * tiles of each row (256 bytes), kStagedBXPrefetchLead tiles ahead of the
 * cp.async that reads them. Prefetching 128 bytes per row did not help; larger
 * groups and a longer lead helped less or not at all.
 */
constexpr int kStagedBXPrefetchGroup = 2;
constexpr int kStagedBXPrefetchLead = 2;

/**
 * @brief Bulk-prefetch x[row][k_begin, min(k_end, K)) into L2 for the tile's rows.
 */
template <int TileM>
__device__ __forceinline__ void prefetch_x_rows_l2(quant::Type const* __restrict__ x, int row_base,
                                                   int k_begin, int k_end, int first_row,
                                                   int row_step, int M, int K) {
  int const bytes = (min(k_end, K) - k_begin) * static_cast<int>(sizeof(quant::Type));
  if (bytes <= 0) {
    return;
  }
  for (int row = first_row; row >= 0 && row < TileM && row_base + row < M; row += row_step) {
    quant::Type const* address = x + static_cast<std::int64_t>(row_base + row) * K + k_begin;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" : : "l"(address), "r"(bytes));
  }
}

/**
 * @brief Fuse quantization and LoRA-down with shared B slices and double-buffered X.
 * Down warps share K-major L2T slices across row partitions; larger row tiles
 * use one CTA per SM, while smaller tiles allow two.
 */
template <int BlockThreads, bool LongK, int KernelTileM, int KernelTileK, DynAddress Policy,
          int Rank>
__global__
__launch_bounds__(BlockThreads, 3 - staged_b_row_split(KernelTileM)) void nvfp4_smooth_quantize_lora_down_large_m_staged_b_sm120_kernel(
    quant::Type const* __restrict__ x, quant::Type const* __restrict__ pre_quant_scale,
    float const* __restrict__ global_scale, std::uint64_t* __restrict__ xq,
    std::uint8_t* __restrict__ sf, quant::Type const* __restrict__ l2t_smoothed,
    quant::Type* __restrict__ down, int M, int K) {
  constexpr int kRank = Rank;
  constexpr int kRowSplit = staged_b_row_split(KernelTileM);
  constexpr int kDownWarps = staged_b_down_warps(KernelTileM);
  constexpr int kFrags = kStagedBCols / 8;
  static_assert(large_m_stages_b(Rank, KernelTileK) && large_m_kmajor_l2(Rank));
  static_assert(Rank == kStagedBSlices * kStagedBCols);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  constexpr int kKernelTileM = KernelTileM;
  constexpr bool kUseStreamingX = LongK;
  constexpr int kKernelQuantThreads = BlockThreads - kDownWarps * 32;
  constexpr int kTileK = KernelTileK;
  constexpr int kTileStride = kTileK + 8;
  constexpr int kSfColsPerTile = kTileK / quant::SF_VEC_SIZE;
  constexpr int kKSteps = kTileK / 16;
  constexpr int kWarpRows = kKernelTileM / kRowSplit;
  constexpr int kMBlocks = kWarpRows / 16;
  static_assert(kWarpRows % 16 == 0);
  int const kPaddedM = (M + 127) / 128 * 128;
  // The quantize loop strides by kKernelQuantThreads and bounds-checks every
  // item, so a tile whose items do not divide across the quantize threads only
  // leaves some threads one item short.
  int const kSfCols = K / quant::SF_VEC_SIZE;
  // Dynamic shared memory, carved in the order of staged_b_shared_bytes():
  // x_tiles[2][TM * stride], scale_tiles[2][TK], sf_tiles[2][...] (16-byte
  // padded), b_tiles[kStagedBSlices][kStagedBCols * kStagedBStride].
  extern __shared__ __align__(16) std::uint8_t staged_smem[];
  using XTile = quant::Type[kKernelTileM * kTileStride];
  using ScaleTile = quant::Type[kTileK];
  using SfTile = std::uint8_t[LongK ? kKernelTileM * kSfColsPerTile : 1];
  using BSlice = quant::Type[kStagedBCols * kStagedBStride];
  constexpr int kXBytes = 2 * kKernelTileM * kTileStride * 2;
  constexpr int kScaleBytes = 2 * kTileK * 2;
  constexpr int kSfBytes = (2 * (LongK ? kKernelTileM * kSfColsPerTile : 1) + 15) / 16 * 16;
  XTile* const x_tiles = reinterpret_cast<XTile*>(staged_smem);
  ScaleTile* const scale_tiles = reinterpret_cast<ScaleTile*>(staged_smem + kXBytes);
  SfTile* const sf_tiles = reinterpret_cast<SfTile*>(staged_smem + kXBytes + kScaleBytes);
  BSlice* const b_tiles = reinterpret_cast<BSlice*>(staged_smem + kXBytes + kScaleBytes + kSfBytes);
  static_assert(kXBytes + kScaleBytes + kSfBytes + kStagedBSlices * kStagedBSliceBytes ==
                staged_b_shared_bytes(KernelTileM, KernelTileK, LongK));

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
    std::int64_t const kL2Bytes = static_cast<std::int64_t>(K) * kRank * sizeof(quant::Type);
    std::int64_t const kL2PrefetchLines = kL2Bytes / kL2PrefetchBytes;
    std::int64_t const prefetch_line =
        static_cast<std::int64_t>(blockIdx.x) * BlockThreads + thread;
    if (prefetch_line < kL2PrefetchLines) {
      std::uint8_t const* prefetch_address =
          reinterpret_cast<std::uint8_t const*>(l2t_smoothed) + prefetch_line * kL2PrefetchBytes;
      asm volatile("prefetch.global.L2 [%0];" : : "l"(prefetch_address));
    }
  }

  int const row_base = static_cast<int>(blockIdx.x) * kKernelTileM;
  // [fragment][m16 block][mma.sync C register].
  float acc[kFrags][kMBlocks][4] = {};

  copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy, Rank>(
      x, x_tiles[0], row_base, 0, thread, l2_evict_first_policy, M, K);
  copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK, Rank>(pre_quant_scale, scale_tiles[0], 0,
                                                                thread);
  int const col_group = warp % kStagedBSlices;
  int const row_part = warp / kStagedBSlices;
  if (warp < kDownWarps) {
    copy_b_slice_async<kRowSplit>(l2t_smoothed, b_tiles[col_group], col_group, row_part, lane, 0,
                                  K);
  }
  commit_x_tile_async();
  wait_x_tile_async();
  __syncthreads();

  int tile_buffer = 0;
#pragma unroll 1
  for (int tile_k = 0; tile_k < K; tile_k += kTileK) {
    int const next_tile_k = tile_k + kTileK;
    int const next_buffer = tile_buffer ^ 1;
    if (next_tile_k < K) {
      copy_x_tile_async_dyn<BlockThreads, LongK, KernelTileM, KernelTileK, Policy, Rank>(
          x, x_tiles[next_buffer], row_base, next_tile_k, thread, l2_evict_first_policy, M, K);
      copy_pre_quant_scale_tile_async_dyn<LongK, KernelTileK, Rank>(
          pre_quant_scale, scale_tiles[next_buffer], next_tile_k, thread);
      commit_x_tile_async();
    }
    if (tile_k % (kStagedBXPrefetchGroup * kTileK) == 0) {
      int const prefetch_k = tile_k + kStagedBXPrefetchLead * kTileK;
      prefetch_x_rows_l2<kKernelTileM>(x, row_base, prefetch_k,
                                       prefetch_k + kStagedBXPrefetchGroup * kTileK,
                                       thread - kDownWarps * 32, kKernelQuantThreads, M, K);
    }
    quant::Type const* x_tile = x_tiles[tile_buffer];
    quant::Type const* scale_tile = scale_tiles[tile_buffer];
    std::uint8_t* sf_tile = sf_tiles[tile_buffer];

    if (warp < kDownWarps) {
      quant::Type* const slice = b_tiles[col_group];
      std::uint32_t b0[kKSteps][kFrags];
      std::uint32_t b1[kKSteps][kFrags];
      // ldmatrix replaces four 32-bit shared loads per fragment pair (B) and per
      // m16 block (A), reducing shared-memory instruction issue pressure.
      // Lane l addresses row l % 8 of matrix l / 8; the matrices are ordered as
      // the mma operands below, so the fragments are bit-identical.
      int const ld_row = lane % 8;
      int const ld_matrix = lane / 8;
      static_assert(kFrags == 2, "one ldmatrix.x4 covers both n8 fragments of a slice");
#pragma unroll
      for (int ks = 0; ks < kKSteps; ++ks) {
        // Matrix j: fragment j / 2, k half j % 2.
        base::ldmatrix_x4(b0[ks][0], b1[ks][0], b0[ks][1], b1[ks][1],
                          slice + ((ld_matrix / 2) * 8 + ld_row) * kStagedBStride + ks * 16 +
                              (ld_matrix % 2) * 8);
      }
      // Once every warp reading this slice holds its fragments, the next tile's
      // copy can land in the same buffer; the end-of-tile cp.async wait and
      // barrier cover it together with the x tile.
      if constexpr (kRowSplit == 1) {
        __syncwarp();
      } else {
        asm volatile("bar.sync %0, %1;\n" : : "r"(1 + col_group), "n"(32 * kRowSplit));
      }
      if (next_tile_k < K) {
        copy_b_slice_async<kRowSplit>(l2t_smoothed, slice, col_group, row_part, lane, next_tile_k,
                                      K);
        commit_x_tile_async();
      }
#pragma unroll
      for (int ks = 0; ks < kKSteps; ++ks) {
        int const local_k = ks * 16;
#pragma unroll
        for (int mb = 0; mb < kMBlocks; ++mb) {
          // Matrix j: rows 8 * (j % 2), k half j / 2 (a0, a1, a2, a3).
          std::uint32_t a0, a1, a2, a3;
          base::ldmatrix_x4(
              a0, a1, a2, a3,
              x_tile +
                  (row_part * kWarpRows + mb * 16 + (ld_matrix % 2) * 8 + ld_row) * kTileStride +
                  local_k + (ld_matrix / 2) * 8);
#pragma unroll
          for (int f = 0; f < kFrags; ++f) {
            base::mma_m16n8k16_row_col(acc[f][mb][0], acc[f][mb][1], acc[f][mb][2], acc[f][mb][3],
                                       a0, a1, a2, a3, b0[ks][f], b1[ks][f]);
          }
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
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      int const col = (col_group * kFrags + f) * 8 + pair_col;
#pragma unroll
      for (int mb = 0; mb < kMBlocks; ++mb) {
        int const row_lo = row_base + row_part * kWarpRows + mb * 16 + group;
        int const row_hi = row_lo + 8;
        if (row_lo < M) {
          store_down_pair<false>(down + static_cast<std::int64_t>(row_lo) * kRank + col,
                                 acc[f][mb][0], acc[f][mb][1]);
        }
        if (row_hi < M) {
          store_down_pair<false>(down + static_cast<std::int64_t>(row_hi) * kRank + col,
                                 acc[f][mb][2], acc[f][mb][3]);
        }
      }
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __trap();
  }
#endif
}

/**
 * @brief Runtime-M/K launch. The geometry stays a template parameter -- shared memory
 * and every unroll constant come from it -- while M and K become arguments, so
 * one instantiation serves every shape whose K divides the tile.
 */
template <int BlockThreads, bool LongK, int TileM, int TileK, DynAddress Policy, int Rank = 32>
inline cudaError_t launch_large_m_dyn_kernel(void const* x, void const* pre_quant_scale,
                                             float const* global_scale, void* xq, void* sf,
                                             void const* l2t_smoothed, void* down, int m, int k,
                                             cudaStream_t stream) {
  constexpr int kRank = Rank;
  constexpr int kDownWarps = large_m_down_warps(Rank);
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
  // The staged-B kernel takes its shared memory dynamically (up to the 99 KB a
  // CTA may opt into) and tolerates a quantize-item remainder.
  constexpr bool kStagedB = large_m_stages_b(Rank, TileK);
  constexpr int kStagedSharedBytes = staged_b_shared_bytes(TileM, TileK, LongK);
  constexpr bool kInvalid =
      kStagedB ? (BlockThreads <= staged_b_down_warps(TileM) * 32 ||
                  kStagedSharedBytes > 99 * 1024 || TileM % (16 * staged_b_row_split(TileM)) != 0)
               : ((kKernelTileM * kSfColsPerTile) % kQuantThreads != 0 || kSharedBytes > 48 * 1024);
  if constexpr (kInvalid) {
    return cudaErrorInvalidValue;
  } else {
    // What the templated kernel asserted at compile time is a precondition here.
    if (m <= 0 || m > std::numeric_limits<int>::max() - 127 || k <= 0) {
      return cudaErrorInvalidValue;
    }
    if (k % TileK != 0) return cudaErrorInvalidValue;
    if (LongK != (k >= 8192)) return cudaErrorInvalidValue;
    int const padded_m = (m + 127) / 128 * 128;
    int const grid_blocks = (padded_m + kKernelTileM - 1) / kKernelTileM;
    if constexpr (LongK) {
      std::int64_t const l2_lines =
          static_cast<std::int64_t>(k) * kRank * sizeof(quant::Type) / 128;
      if (static_cast<std::int64_t>(grid_blocks) * BlockThreads < l2_lines) {
        return cudaErrorInvalidValue;
      }
    }
    if constexpr (kStagedB) {
      auto* const kernel = nvfp4_smooth_quantize_lora_down_large_m_staged_b_sm120_kernel<
          BlockThreads, LongK, kKernelTileM, TileK, Policy, Rank>;
      if constexpr (kStagedSharedBytes > 48 * 1024) {
        cudaError_t const attribute_status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kStagedSharedBytes);
        if (attribute_status != cudaSuccess) return attribute_status;
      }
      kernel<<<grid_blocks, BlockThreads, kStagedSharedBytes, stream>>>(
          static_cast<quant::Type const*>(x), static_cast<quant::Type const*>(pre_quant_scale),
          global_scale, static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
          static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down), m, k);
      return cudaGetLastError();
    } else {
      nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel<BlockThreads, LongK, kKernelTileM,
                                                               TileK, Policy, Rank>
          <<<grid_blocks, BlockThreads, 0, stream>>>(
              static_cast<quant::Type const*>(x), static_cast<quant::Type const*>(pre_quant_scale),
              global_scale, static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
              static_cast<quant::Type const*>(l2t_smoothed), static_cast<quant::Type*>(down), m, k);
      return cudaGetLastError();
    }
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

// Rank-128 split-K small-M tilings: (block threads, down tile columns, K splits).
// Mirrors SM120_SMALL_M_SPLIT_TILING_LADDER in svdquant_sm120_routes.py.
#define FI_SVDQ_SMALL_M_SPLIT_TILINGS(X) \
  X(512, 16, 4)                          \
  X(1024, 16, 2)                         \
  X(1024, 16, 4)

// Rank 64 reserves eight warps for LoRA-down; the remaining warps quantize.
// Rank 128 reuses the same eight down warps, each owning two n8 fragments.
#define FI_SVDQ_RANK64_LARGE_M_GEOMETRIES(X) \
  X(320, 16, 128)                            \
  X(320, 32, 128)                            \
  X(352, 48, 128)                            \
  X(384, 32, 256)                            \
  X(384, 64, 128)                            \
  X(384, 80, 128)                            \
  X(512, 32, 256)

// Rank 128 adds K64 geometries whose kernel stages B in shared memory
// (large_m_stages_b). Mirrors SM120_RANK128_STAGED_B_GEOMETRY_LADDER.
#define FI_SVDQ_RANK128_STAGED_B_GEOMETRIES(X) \
  X(320, 64, 64)                               \
  X(352, 48, 64)                               \
  X(384, 64, 64)                               \
  X(384, 80, 64)                               \
  X(416, 80, 64)                               \
  X(768, 160, 64)                              \
  X(832, 160, 64)

/**
 * @brief Dispatch a compiled large-M geometry, returning cudaErrorInvalidValue if absent.
 * Each launcher validates runtime dimensions before enqueueing work on stream.
 */
template <int Rank = 32>
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
                 ? large_m::launch_large_m_dyn_kernel<BT, true, TM, TK, DynAddress::kAccumulate,   \
                                                      Rank>(x, pre_quant_scale, global_scale, xq,  \
                                                            sf, l2t_smoothed, down, m, k, stream)  \
                 : large_m::launch_large_m_dyn_kernel<BT, true, TM, TK, DynAddress::kRecompute,    \
                                                      Rank>(x, pre_quant_scale, global_scale, xq,  \
                                                            sf, l2t_smoothed, down, m, k, stream); \
    }                                                                                              \
    return accumulate                                                                              \
               ? large_m::launch_large_m_dyn_kernel<BT, false, TM, TK, DynAddress::kAccumulate,    \
                                                    Rank>(x, pre_quant_scale, global_scale, xq,    \
                                                          sf, l2t_smoothed, down, m, k, stream)    \
               : large_m::launch_large_m_dyn_kernel<BT, false, TM, TK, DynAddress::kRecompute,     \
                                                    Rank>(x, pre_quant_scale, global_scale, xq,    \
                                                          sf, l2t_smoothed, down, m, k, stream);   \
  }
  if constexpr (Rank >= 64) {
    FI_SVDQ_RANK64_LARGE_M_GEOMETRIES(FI_SVDQ_DYN_CASE)
  } else {
    FI_SVDQ_LARGE_M_GEOMETRIES(FI_SVDQ_DYN_CASE)
  }
  if constexpr (Rank == 128) {
    FI_SVDQ_RANK128_STAGED_B_GEOMETRIES(FI_SVDQ_DYN_CASE)
  }
#undef FI_SVDQ_DYN_CASE
  return cudaErrorInvalidValue;
}

/**
 * @brief Producer launch by family and tiling, with no shape lookup at all. The caller
 * enumerates what a shape may run from the same constraints the launchers check,
 * so this switch selects compiled families and tilings, and a shape
 * nobody wrote down reaches the same kernels as one that was.
 *   family 0 large-M (geometry a, b, c = block threads, tile M, tile K)
 *   family 1 row-major small-M (a, b, c = block threads, down tile cols, rows/quant block)
 *   family 2 fixed-M537 with packed L2T (a, b, c = block threads, down warps, rows/quant block)
 *   family 4 packed small-M (same tiling fields as family 1)
 *   family 5 warp-specialized rank-128 large-M (a, b, c = tile M, tile K,
 *            down layout * 1000 + stages * 100 + quantize warps;
 *            address_policy = grid mode)
 *   family 6 split-K rank-128 small-M (a, b, c = block threads, down tile cols, K splits)
 * Family 3 uses the separate cuBLASLt prefix entry.
 */
template <int Rank = 32>
inline cudaError_t nvfp4_smooth_quantize_lora_down_family_sm120(
    void const* x, void const* pre_quant_scale, float const* global_scale, void* xq, void* sf,
    void const* l2t_smoothed, void* down, int m, int k, int family, int a, int b, int c,
    int address_policy, cudaStream_t stream, bool signal_launch = false,
    SvdquantQuantizeBarrierSm120 quantize_barrier = {}) {
  namespace large_m = smooth_quantize_lora_down_large_m_sm120_detail;
  namespace small_m = smooth_quantize_lora_down_small_m_sm120_detail;
  using large_m::DynAddress;
  if constexpr (Rank == 64) {
    if (family != 0 && family != 1) return cudaErrorInvalidValue;
  }
  // Rank 128 has the large-M (eight wide down warps), the warp-specialized
  // large-M and the row-major small-M families.
  if constexpr (Rank == 128) {
    if (family == 5) {
      return nvfp4_smooth_quantize_lora_down_ws_rank128_sm120<Rank>(
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, a, b, c,
          address_policy, stream);
    }
    if (family == 6) {
      // The small-M kernel with LoRA-down split over K across a KSplit-CTA cluster;
      // same launch signaling and address policies as family 1.
#define FI_SVDQ_SPLIT_CASE(BT, DTC, KS)                                                    \
  if (a == (BT) && b == (DTC) && c == (KS)) {                                              \
    if (signal_launch && address_policy == 1) {                                            \
      return small_m::launch_small_m_dyn_kernel<BT, DTC, 4, false, true, Rank, true, KS>(  \
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream,      \
          quantize_barrier);                                                               \
    }                                                                                      \
    if (signal_launch) {                                                                   \
      return small_m::launch_small_m_dyn_kernel<BT, DTC, 4, false, true, Rank, false, KS>( \
          x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);     \
    }                                                                                      \
    return small_m::launch_small_m_dyn_kernel<BT, DTC, 4, false, false, Rank, false, KS>(  \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);       \
  }
      FI_SVDQ_SMALL_M_SPLIT_TILINGS(FI_SVDQ_SPLIT_CASE)
#undef FI_SVDQ_SPLIT_CASE
      return cudaErrorInvalidValue;
    }
    if (family != 0 && family != 1) return cudaErrorInvalidValue;
  }
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
  // The small-M kernel tiles the rank into DownTileCols-wide column tiles, so
  // every compiled rank reuses the same tilings.
  if (family == 1) {
    // At rank 128 the linear route lets the PDL-launched GEMM start its prologue
    // while this prefix runs; the GEMM waits on the grid before reading outputs.
    // Address policy 1 instead holds the GEMM back until the quantize half is
    // done and lets its mainloop run under LoRA-down (see GateLaunchOnQuant).
#define FI_SVDQ_SMALL_CASE(BT, DTC, RPQB)                                                  \
  if (a == (BT) && b == (DTC) && c == (RPQB)) {                                            \
    if constexpr (Rank == 128) {                                                           \
      if (signal_launch && address_policy == 1) {                                          \
        return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB, false, true, Rank, true>( \
            x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream,    \
            quantize_barrier);                                                             \
      }                                                                                    \
      if (signal_launch) {                                                                 \
        return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB, false, true, Rank>(       \
            x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);   \
      }                                                                                    \
    }                                                                                      \
    return small_m::launch_small_m_dyn_kernel<BT, DTC, RPQB, false, false, Rank>(          \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, stream);       \
  }
    FI_SVDQ_SMALL_M_TILINGS(FI_SVDQ_SMALL_CASE)
#undef FI_SVDQ_SMALL_CASE
    return cudaErrorInvalidValue;
  }
  return nvfp4_smooth_quantize_lora_down_dyn_sm120<Rank>(x, pre_quant_scale, global_scale, xq, sf,
                                                         l2t_smoothed, down, m, k, a, b, c, stream,
                                                         address_policy);
}

/**
 * @brief Try compiled default geometries until one accepts the runtime dimensions.
 * Excludes packed-L2T families; callers must provide the rank's default layout.
 * Returns cudaErrorInvalidValue when no candidate accepts the shape.
 */
template <int Rank = 32>
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
    cudaError_t const status = nvfp4_smooth_quantize_lora_down_family_sm120<Rank>(              \
        x, pre_quant_scale, global_scale, xq, sf, l2t_smoothed, down, m, k, FAMILY, A, B, C, 0, \
        stream);                                                                                \
    if (status != cudaErrorInvalidValue) return status;                                         \
  }
#define FI_SVDQ_TRY_LARGE(BT, TM, TK) FI_SVDQ_TRY(0, BT, TM, TK)
#define FI_SVDQ_TRY_SMALL(BT, DTC, RPQB) FI_SVDQ_TRY(1, BT, DTC, RPQB)
  if constexpr (Rank >= 64) {
    FI_SVDQ_RANK64_LARGE_M_GEOMETRIES(FI_SVDQ_TRY_LARGE)
  } else {
    FI_SVDQ_LARGE_M_GEOMETRIES(FI_SVDQ_TRY_LARGE)
  }
  if constexpr (Rank <= 64) {
    FI_SVDQ_SMALL_M_TILINGS(FI_SVDQ_TRY_SMALL)
  }
#undef FI_SVDQ_TRY_SMALL
#undef FI_SVDQ_TRY_LARGE
#undef FI_SVDQ_TRY
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm
