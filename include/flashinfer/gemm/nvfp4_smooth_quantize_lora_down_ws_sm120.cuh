/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Warp-specialized rank-128 large-M fused prefix for SM120: smooth-quantize x
// to NVFP4 (xq, swizzled 128x4 SF) and compute down = x @ L2T (rank 128, bf16).
//
// Why a separate kernel. The large-M producers in
// nvfp4_smooth_quantize_lora_down_sm120.cuh run every K tile as
// "cp.async next tile -> compute this tile -> wait -> __syncthreads", with every
// thread both a loader and a consumer. At 16384x3072x3072 rank 128 the staged-B
// kernel took 168us; with B copies and the down MMAs removed (quantize only) it
// still took 158us against an x-read floor of ~100us, so the per-tile barrier
// structure itself leaves DRAM idle, and it couples the quantize and down warps.
//
// Here one producer warp streams x tiles (and the matching K-major L2T tile and
// pre-quant-scale slice) into an N-stage shared-memory ring with 2-D TMA
// (128-byte swizzle, 16-row boxes) + mbarrier transaction counts. Quantize
// warps and down warps consume each stage independently and release it by
// arriving on the stage's empty barrier; nothing in the main loop is a CTA-wide
// barrier. The grid is persistent: at most one CTA per SM, with the padded rows
// split evenly across CTAs in 16-row units so there is no partial wave, and
// each CTA walks its rows in chunks of TileM.
//
// Measured on RTX PRO 5000 (SM120, 110 SMs), 16384x3072x3072 cold L2:
//   - per-row 1-D cp.async.bulk copies instead of TMA boxes: 459us (copy issue
//     bound); TMA boxes: 150us.
//   - quantize-only ablation (no B, no down warps) at the same K64 tiling:
//     145us; with K256 stages 134us; standalone nvfp4_quantize_smooth 128us.
//     The full kernel is ~5us over its own quantize-only ablation, and doubling
//     the down warps recovers <2us, so it is the x/quantize stream at K64, not
//     the LoRA-down MMAs, that bounds it. Wider K stages do not fit: a K64
//     rank-128 B tile is 16 KB per stage, and row tiles under ~144 rows re-read
//     L2T often enough to lose more than they gain (32x128: 158us).
//   - reading B fragments straight from global/L2 (no B stage) to free shared
//     memory for wider x stages: 152-275us, slower in every configuration.
//
// Outputs are bit-identical to nvfp4_quantize_smooth for xq/sf (the same
// quantizeSmoothed16 per 16-element block, SF zero-filled for rows in
// [M, padded M)), and down is the same bf16 mma.sync accumulation in K order.

#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"

namespace flashinfer::gemm {

namespace smooth_quantize_lora_down_ws_sm120_detail {

namespace quant = smooth_quantize_detail;

constexpr int kRank = 128;

__device__ __forceinline__ std::uint32_t smem_u32(void const* ptr) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbar_init(std::uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_wait(std::uint64_t* bar, std::uint32_t parity) {
  std::uint32_t const address = smem_u32(bar);
  std::uint32_t done = 0;
  do {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(address), "r"(parity)
        : "memory");
  } while (!done);
}

__device__ __forceinline__ void mbar_arrive(std::uint64_t* bar) {
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
      "}\n"
      :
      : "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(std::uint64_t* bar, std::uint32_t bytes) {
  asm volatile(
      "{\n"
      ".reg .b64 state;\n"
      "mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
      "}\n"
      :
      : "r"(smem_u32(bar)), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ std::uint64_t make_l2_policy_evict_first() {
  std::uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  return policy;
}

// L2T and the smoothing scale are re-read by every CTA, but evict_normal already keeps
// them resident against the evict_first X stream. evict_last would outlive the kernel:
// those lines survive later cold-L2 flushes and pin each layer's L2T in L2.
__device__ __forceinline__ std::uint64_t make_l2_policy_evict_normal() {
  std::uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(policy));
  return policy;
}

// One contiguous global -> shared bulk copy, completing `bytes` on `bar`.
__device__ __forceinline__ void bulk_copy_g2s(void* dst, void const* src, std::uint32_t bytes,
                                              std::uint64_t* bar, std::uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;\n"
      :
      : "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}

// One 2-D TMA box (inner coordinate c0 in elements, outer c1 in rows).
__device__ __forceinline__ void tma_load_2d(void* dst, CUtensorMap const* map, int c0, int c1,
                                            std::uint64_t* bar, std::uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
      :
      : "r"(smem_u32(dst)), "l"(reinterpret_cast<std::uint64_t>(map)), "r"(c0), "r"(c1),
        "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}

// 128-byte swizzle (CU_TENSOR_MAP_SWIZZLE_128B) of a tile whose rows are
// exactly 128 bytes and whose base is 1024-byte aligned: 16-byte chunk c of
// row r lives at chunk c ^ (r & 7).
__device__ __forceinline__ int swz128(int row, int chunk) {
  return row * 128 + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t& r0, std::uint32_t& r1, std::uint32_t& r2,
                                            std::uint32_t& r3, void const* ptr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_u32(ptr)));
}

__device__ __forceinline__ void mma_bf16_16816(float* d, std::uint32_t a0, std::uint32_t a1,
                                               std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
                                               std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Compile-time configuration. Warp roles: warp 0 produces, the next
// QuantWarps quantize, the last DownWarpsM * DownWarpsN compute LoRA-down
// (each owning TileM / DownWarpsM rows and Rank / DownWarpsN columns).
template <int TileM_, int TileK_, int Stages_, int QuantWarps_, int DownWarpsM_, int DownWarpsN_>
struct WsConfig {
  static constexpr int kTileM = TileM_;
  static constexpr int kTileK = TileK_;
  static constexpr int kStages = Stages_;
  static constexpr int kQuantWarps = QuantWarps_;
  static constexpr int kDownWarpsM = DownWarpsM_;
  static constexpr int kDownWarpsN = DownWarpsN_;
  static constexpr int kDownWarps = kDownWarpsM * kDownWarpsN;
  static constexpr int kWarps = 1 + kQuantWarps + kDownWarps;
  static constexpr int kThreads = kWarps * 32;
  // TMA tiles with the 128-byte swizzle: a stage holds kSub sub-tiles of K64
  // (one 128-byte line per row each), bank conflicts are removed by the
  // swizzle instead of padding.
  static constexpr int kBoxRows = 16;
  static constexpr int kSub = kTileK / 64;
  static constexpr int kXSubBytes = kTileM * 64 * 2;
  static constexpr int kBSubBytes = kRank * 64 * 2;
  static constexpr int kXBytes = kTileM * kTileK * 2;
  static constexpr int kBBytes = kRank * kTileK * 2;
  static constexpr int kPqsBytes = kTileK * 2;
  static constexpr int kStageBytes = (kXBytes + kBBytes + kPqsBytes + 1023) / 1024 * 1024;
  static constexpr int kBarrierBytes = 2 * kStages * 8;
  // Slack to round the dynamic shared memory base up to 1024 bytes.
  static constexpr int kSmemBytes = 1024 + kStages * kStageBytes + kBarrierBytes;
  static constexpr int kSfPerTile = kTileK / quant::SF_VEC_SIZE;
  // A quantize item is one row's group of up to four SF blocks (one 32-bit,
  // or for K32 tiles 16-bit, SF store and a contiguous run of xq).
  static constexpr int kSfPerItem = 4;
  static constexpr int kItemsPerRow = kSfPerTile / kSfPerItem;
  static constexpr int kWarpRows = kTileM / DownWarpsM_;
  static constexpr int kMBlocks = kWarpRows / 16;
  static constexpr int kWarpCols = kRank / DownWarpsN_;
  static constexpr int kFrags = kWarpCols / 8;
  static_assert(kTileK % 64 == 0, "rows are 128-byte swizzle lines per K64 sub-tile");
  static_assert(kSfPerTile % kSfPerItem == 0);
  static_assert(kTileM % (16 * DownWarpsM_) == 0 && kRank % (16 * DownWarpsN_) == 0);
  static_assert(kTileM % 16 == 0);
  static_assert(kXBytes % 1024 == 0 && kBBytes % 1024 == 0);
};

template <class Cfg>
__global__ void __launch_bounds__(Cfg::kThreads, 1)
    nvfp4_smooth_quantize_lora_down_ws_sm120_kernel(__grid_constant__ CUtensorMap const x_map,
                                                    __grid_constant__ CUtensorMap const b_map,
                                                    quant::Type const* __restrict__ pre_quant_scale,
                                                    float const* __restrict__ global_scale,
                                                    std::uint64_t* __restrict__ xq,
                                                    std::uint8_t* __restrict__ sf,
                                                    quant::Type* __restrict__ down, int M, int K) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  constexpr int kTileM = Cfg::kTileM;
  constexpr int kTileK = Cfg::kTileK;
  constexpr int kStages = Cfg::kStages;
  extern __shared__ __align__(16) std::uint8_t ws_smem_raw[];
  std::uint8_t* const stage_base = reinterpret_cast<std::uint8_t*>(
      (reinterpret_cast<std::uintptr_t>(ws_smem_raw) + 1023) / 1024 * 1024);
  std::uint64_t* const full_bar =
      reinterpret_cast<std::uint64_t*>(stage_base + kStages * Cfg::kStageBytes);
  std::uint64_t* const empty_bar = full_bar + kStages;

  int const thread = static_cast<int>(threadIdx.x);
  int const warp = thread / 32;
  int const lane = thread % 32;

  if (thread == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], Cfg::kQuantWarps + Cfg::kDownWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  // Balanced persistent split of the padded rows in 16-row units.
  int const padded_m = (M + 127) / 128 * 128;
  int const units = padded_m / 16;
  int const ctas = static_cast<int>(gridDim.x);
  int const cta = static_cast<int>(blockIdx.x);
  int const row_begin = static_cast<int>(static_cast<std::int64_t>(units) * cta / ctas) * 16;
  int const row_end = static_cast<int>(static_cast<std::int64_t>(units) * (cta + 1) / ctas) * 16;
  int const k_tiles = K / kTileK;

  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    std::uint64_t const x_policy = make_l2_policy_evict_first();
    std::uint64_t const keep_policy = make_l2_policy_evict_normal();
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];\n" : : "l"(reinterpret_cast<std::uint64_t>(&x_map)));
      asm volatile("prefetch.tensormap [%0];\n" : : "l"(reinterpret_cast<std::uint64_t>(&b_map)));
    }
    int it = 0;
    for (int chunk = row_begin; chunk < row_end; chunk += kTileM) {
      // Boxes past M are zero-filled by TMA and still count their full bytes.
      int const boxes = (min(kTileM, row_end - chunk) + Cfg::kBoxRows - 1) / Cfg::kBoxRows;
      std::uint32_t const tx_bytes = static_cast<std::uint32_t>(boxes * Cfg::kBoxRows * kTileK * 2 +
                                                                Cfg::kBBytes + kTileK * 2);
      for (int kt = 0; kt < k_tiles; ++kt, ++it) {
        int const s = it % kStages;
        std::uint32_t const phase = static_cast<std::uint32_t>(it / kStages) & 1u;
        mbar_wait(&empty_bar[s], phase ^ 1u);
        std::uint8_t* const stage = stage_base + s * Cfg::kStageBytes;
        int const k0 = kt * kTileK;
        if (lane == 0) mbar_arrive_expect_tx(&full_bar[s], tx_bytes);
        __syncwarp();
        // Sub-tile-major order so each row's kSub boxes are issued back to back.
        for (int box = lane; box < boxes * Cfg::kSub; box += 32) {
          int const rb = box / Cfg::kSub;
          int const sub = box - rb * Cfg::kSub;
          tma_load_2d(stage + sub * Cfg::kXSubBytes + rb * Cfg::kBoxRows * 128, &x_map,
                      k0 + sub * 64, chunk + rb * Cfg::kBoxRows, &full_bar[s], x_policy);
        }
        // L2T is K-major [Rank][K]: one box of 128 rows x 64 k per sub-tile.
        if (lane >= 32 - Cfg::kSub) {
          int const sub = lane - (32 - Cfg::kSub);
          tma_load_2d(stage + Cfg::kXBytes + sub * Cfg::kBSubBytes, &b_map, k0 + sub * 64, 0,
                      &full_bar[s], keep_policy);
        }
        if (lane == 32 - Cfg::kSub - 1) {
          bulk_copy_g2s(stage + Cfg::kXBytes + Cfg::kBBytes, pre_quant_scale + k0, kTileK * 2,
                        &full_bar[s], keep_policy);
        }
      }
    }
  } else if (warp <= Cfg::kQuantWarps) {
    // ---------------------------------------------------------------- quantize
    constexpr int kQuantThreads = Cfg::kQuantWarps * 32;
    constexpr int kSfPerItem = Cfg::kSfPerItem;
    int const qthread = thread - 32;
    float const scale = global_scale[0];
    int const sf_cols = K / quant::SF_VEC_SIZE;
    int it = 0;
    for (int chunk = row_begin; chunk < row_end; chunk += kTileM) {
      int const chunk_rows = min(kTileM, row_end - chunk);
      int const items = chunk_rows * Cfg::kItemsPerRow;
      for (int kt = 0; kt < k_tiles; ++kt, ++it) {
        int const s = it % kStages;
        std::uint32_t const phase = static_cast<std::uint32_t>(it / kStages) & 1u;
        mbar_wait(&full_bar[s], phase);
        std::uint8_t const* const stage = stage_base + s * Cfg::kStageBytes;
        quant::Type const* const pqs_tile =
            reinterpret_cast<quant::Type const*>(stage + Cfg::kXBytes + Cfg::kBBytes);
        for (int item = qthread; item < items; item += kQuantThreads) {
          // Row-fastest: consecutive threads read consecutive rows of one
          // sub-tile (distinct swizzle phases, conflict-free 16-byte reads).
          int const group = item / chunk_rows;
          int const r = item - group * chunk_rows;
          std::uint8_t const* const sub_tile = stage + group * Cfg::kXSubBytes;
          int const row = chunk + r;
          int const sf_col = kt * Cfg::kSfPerTile + group * kSfPerItem;
          std::int64_t const sf_offset = quant::get_sf_out_offset_128x4(row, sf_col, sf_cols);
          if (row < M) {
            std::uint8_t sf_bytes[kSfPerItem];
            std::uint64_t packed[kSfPerItem];
#pragma unroll
            for (int b = 0; b < kSfPerItem; ++b) {
              int const col = (group * kSfPerItem + b) * quant::SF_VEC_SIZE;
              quant::Bf16x8 x_lo, x_hi, pqs_lo, pqs_hi;
              int const chunk16 = (col % 64) / 8;
              quant::loadBf16x8(reinterpret_cast<quant::Type const*>(sub_tile + swz128(r, chunk16)),
                                x_lo);
              quant::loadBf16x8(
                  reinterpret_cast<quant::Type const*>(sub_tile + swz128(r, chunk16 + 1)), x_hi);
              quant::loadBf16x8(pqs_tile + col, pqs_lo);
              quant::loadBf16x8(pqs_tile + col + 8, pqs_hi);
              packed[b] =
                  quant::quantizeSmoothed16(x_lo, x_hi, pqs_lo, pqs_hi, scale, &sf_bytes[b]);
            }
            std::uint64_t* const xq_row = xq + static_cast<std::int64_t>(row) * sf_cols + sf_col;
            {
              reinterpret_cast<uint4*>(xq_row)[0] =
                  make_uint4(static_cast<std::uint32_t>(packed[0]),
                             static_cast<std::uint32_t>(packed[0] >> 32),
                             static_cast<std::uint32_t>(packed[1]),
                             static_cast<std::uint32_t>(packed[1] >> 32));
              reinterpret_cast<uint4*>(xq_row)[1] =
                  make_uint4(static_cast<std::uint32_t>(packed[2]),
                             static_cast<std::uint32_t>(packed[2] >> 32),
                             static_cast<std::uint32_t>(packed[3]),
                             static_cast<std::uint32_t>(packed[3] >> 32));
              *reinterpret_cast<std::uint32_t*>(sf + sf_offset) =
                  static_cast<std::uint32_t>(sf_bytes[0]) |
                  (static_cast<std::uint32_t>(sf_bytes[1]) << 8) |
                  (static_cast<std::uint32_t>(sf_bytes[2]) << 16) |
                  (static_cast<std::uint32_t>(sf_bytes[3]) << 24);
            }
          } else {
            // Rows in [M, padded M) only carry zero scale factors.
            *reinterpret_cast<std::uint32_t*>(sf + sf_offset) = 0u;
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
      }
    }
  } else {
    // ---------------------------------------------------------------- down
    constexpr int kMBlocks = Cfg::kMBlocks;
    constexpr int kFrags = Cfg::kFrags;
    int const down_warp = warp - 1 - Cfg::kQuantWarps;
    int const warp_m = down_warp / Cfg::kDownWarpsN;
    int const warp_n = down_warp % Cfg::kDownWarpsN;
    int const group = lane / 4;
    int const pair_col = (lane % 4) * 2;
    // ldmatrix row addresses. A (x, [m][k]): matrices (r0-7,k0-7), (r8-15,k0-7),
    // (r0-7,k8-15), (r8-15,k8-15) = mma a0..a3. B (K-major L2T, [n][k]):
    // (n0-7,k0-7), (n0-7,k8-15), (n8-15,k0-7), (n8-15,k8-15) = b0,b1 of two frags.
    int const a_row = (lane % 8) + ((lane / 8) % 2) * 8;
    int const a_col = (lane / 16) * 8;
    int const b_row = (lane % 8) + (lane / 16) * 8;
    int const b_col = ((lane / 8) % 2) * 8;
    int it = 0;
    for (int chunk = row_begin; chunk < row_end; chunk += kTileM) {
      int const chunk_rows = min(kTileM, row_end - chunk);
      int const warp_row0 = warp_m * Cfg::kWarpRows;
      // m16 blocks of this warp that hold rows of the chunk (warp-uniform).
      int const live_mblocks = max(0, min(kMBlocks, (chunk_rows - warp_row0 + 15) / 16));
      float acc[kFrags][kMBlocks][4];
#pragma unroll
      for (int f = 0; f < kFrags; ++f)
#pragma unroll
        for (int mb = 0; mb < kMBlocks; ++mb)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[f][mb][i] = 0.0f;
      for (int kt = 0; kt < k_tiles; ++kt, ++it) {
        int const s = it % kStages;
        std::uint32_t const phase = static_cast<std::uint32_t>(it / kStages) & 1u;
        mbar_wait(&full_bar[s], phase);
        std::uint8_t const* const stage = stage_base + s * Cfg::kStageBytes;
#pragma unroll
        for (int kk = 0; kk < kTileK / 16; ++kk) {
          int const ks = kk % 4;
          std::uint8_t const* const x_sub = stage + (kk / 4) * Cfg::kXSubBytes;
          std::uint8_t const* const b_tile = stage + Cfg::kXBytes + (kk / 4) * Cfg::kBSubBytes;
          std::uint32_t b[kFrags][2];
#pragma unroll
          for (int f = 0; f < kFrags; f += 2) {
            ldmatrix_x4(
                b[f][0], b[f][1], b[f + 1][0], b[f + 1][1],
                b_tile + swz128(warp_n * Cfg::kWarpCols + f * 8 + b_row, ks * 2 + b_col / 8));
          }
#pragma unroll
          for (int mb = 0; mb < kMBlocks; ++mb) {
            if (mb < live_mblocks) {
              std::uint32_t a0, a1, a2, a3;
              ldmatrix_x4(a0, a1, a2, a3,
                          x_sub + swz128(warp_row0 + mb * 16 + a_row, ks * 2 + a_col / 8));
#pragma unroll
              for (int f = 0; f < kFrags; ++f) {
                mma_bf16_16816(acc[f][mb], a0, a1, a2, a3, b[f][0], b[f][1]);
              }
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
      }
#pragma unroll
      for (int mb = 0; mb < kMBlocks; ++mb) {
        int const row_lo = chunk + warp_row0 + mb * 16 + group;
        int const row_hi = row_lo + 8;
        bool const lo_ok = mb < live_mblocks && row_lo < M && row_lo < chunk + chunk_rows;
        bool const hi_ok = mb < live_mblocks && row_hi < M && row_hi < chunk + chunk_rows;
#pragma unroll
        for (int f = 0; f < kFrags; ++f) {
          int const col = warp_n * Cfg::kWarpCols + f * 8 + pair_col;
          if (lo_ok) {
            __nv_bfloat162 const v = __floats2bfloat162_rn(acc[f][mb][0], acc[f][mb][1]);
            *reinterpret_cast<__nv_bfloat162*>(down + static_cast<std::int64_t>(row_lo) * kRank +
                                               col) = v;
          }
          if (hi_ok) {
            __nv_bfloat162 const v = __floats2bfloat162_rn(acc[f][mb][2], acc[f][mb][3]);
            *reinterpret_cast<__nv_bfloat162*>(down + static_cast<std::int64_t>(row_hi) * kRank +
                                               col) = v;
          }
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

inline PFN_cuTensorMapEncodeTiled_v12000 ws_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault,
                                     &status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &status);
#endif
    return status == cudaDriverEntryPointSuccess
               ? reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr)
               : nullptr;
  }();
  return fn;
}

// Row-major bf16 [rows, cols] with a 64 x box_rows box and the 128-byte swizzle.
inline bool ws_make_map(CUtensorMap* map, void const* base, int rows, int cols, int box_rows) {
  auto const encode = ws_encode_fn();
  if (encode == nullptr) return false;
  if (reinterpret_cast<std::uintptr_t>(base) % 16 != 0 || (cols * 2) % 16 != 0) return false;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(cols) * 2};
  cuuint32_t box[2] = {64, static_cast<cuuint32_t>(box_rows)};
  cuuint32_t elem_strides[2] = {1, 1};
  CUresult const result =
      encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box,
             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS;
}

inline int ws_device_sm_count() {
  static int cached[64] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) return 0;
  if (cached[device] == 0) {
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
      return 0;
    }
    cached[device] = count;
  }
  return cached[device];
}

// grid_mode 0: one CTA per SM (capped by the 16-row units); 1: as many CTAs as
// TileM chunks, capped by the SM count.
template <int TileM, int TileK, int Stages, int QuantWarps, int DownWarpsM, int DownWarpsN>
inline cudaError_t launch_ws_kernel(void const* x, void const* pre_quant_scale,
                                    float const* global_scale, void* xq, void* sf,
                                    void const* l2t_kmajor, void* down, int m, int k, int grid_mode,
                                    cudaStream_t stream) {
  using Cfg = WsConfig<TileM, TileK, Stages, QuantWarps, DownWarpsM, DownWarpsN>;
  if constexpr (Cfg::kSmemBytes > 99 * 1024) {
    return cudaErrorInvalidValue;
  } else {
    // K a multiple of 64 keeps every 4-wide SF group inside one K tile pair.
    if (m <= 0 || k <= 0 || k % TileK != 0 || k % 64 != 0 || m > (1 << 30)) {
      return cudaErrorInvalidValue;
    }
    if (grid_mode != 0 && grid_mode != 1) return cudaErrorInvalidValue;
    int const sms = ws_device_sm_count();
    if (sms <= 0) return cudaErrorInvalidValue;
    int const padded_m = (m + 127) / 128 * 128;
    int const units = padded_m / 16;
    int grid = grid_mode == 0 ? sms : (padded_m + TileM - 1) / TileM;
    grid = grid < sms ? grid : sms;
    grid = grid < units ? grid : units;
    CUtensorMap x_map;
    CUtensorMap b_map;
    if (!ws_make_map(&x_map, x, m, k, Cfg::kBoxRows)) return cudaErrorInvalidValue;
    static_assert(Cfg::kSub < 30, "producer lanes");
    if (!ws_make_map(&b_map, l2t_kmajor, kRank, k, kRank)) return cudaErrorInvalidValue;
    auto* const kernel = nvfp4_smooth_quantize_lora_down_ws_sm120_kernel<Cfg>;
    static cudaError_t const attribute_status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSmemBytes);
    if (attribute_status != cudaSuccess) return attribute_status;
    kernel<<<grid, Cfg::kThreads, Cfg::kSmemBytes, stream>>>(
        x_map, b_map, static_cast<quant::Type const*>(pre_quant_scale), global_scale,
        static_cast<std::uint64_t*>(xq), static_cast<std::uint8_t*>(sf),
        static_cast<quant::Type*>(down), m, k);
    return cudaGetLastError();
  }
}

}  // namespace smooth_quantize_lora_down_ws_sm120_detail

// Warp-specialized rank-128 tilings: (TileM, TileK, Stages, QuantWarps, down
// layout: 0 = 2x4 down warps (rows x columns), 1 = 2x8, 2 = 1x8). Tile M spans
// the rows one persistent CTA owns at the target M (~37 at M=4096, ~149 at
// M=16384), so a CTA walks one chunk. Mirrors SM120_RANK128_WS_TILING_LADDER in
// flashinfer/gemm/svdquant_sm120_routes.py.
#define FI_SVDQ_RANK128_WS_TILINGS(X) \
  X(160, 64, 2, 5, 1)                 \
  X(160, 64, 2, 5, 0)                 \
  X(160, 64, 2, 10, 1)                \
  X(144, 64, 2, 9, 2)                 \
  X(128, 64, 2, 4, 0)                 \
  X(96, 64, 3, 6, 0)                  \
  X(96, 64, 3, 3, 1)                  \
  X(80, 64, 3, 5, 2)                  \
  X(64, 64, 3, 4, 0)                  \
  X(64, 64, 3, 4, 1)                  \
  X(48, 64, 4, 3, 2)                  \
  X(32, 64, 4, 2, 0)                  \
  X(32, 64, 4, 4, 0)

// A template so that only the rank-128 module, whose dispatch names it inside
// `if constexpr (Rank == 128)`, instantiates these kernels.
template <int Rank>
inline cudaError_t nvfp4_smooth_quantize_lora_down_ws_rank128_sm120(
    void const* x, void const* pre_quant_scale, float const* global_scale, void* xq, void* sf,
    void const* l2t_kmajor, void* down, int m, int k, int tile_m, int tile_k, int packed_tiling,
    int grid_mode, cudaStream_t stream) {
  static_assert(Rank == smooth_quantize_lora_down_ws_sm120_detail::kRank);
  namespace ws = smooth_quantize_lora_down_ws_sm120_detail;
  // The third tiling field packs down layout * 1000 + stages * 100 + quant
  // warps; down layout 0 = 2x4 down warps (rows x columns), 1 = 2x8, 2 = 1x8.
  int const layout = packed_tiling / 1000;
  int const stages = packed_tiling / 100 % 10;
  int const quant_warps = packed_tiling % 100;
#define FI_SVDQ_WS_CASE(TM, TK, ST, QW, DL)                                                   \
  if (tile_m == (TM) && tile_k == (TK) && stages == (ST) && quant_warps == (QW) &&            \
      layout == (DL)) {                                                                       \
    return ws::launch_ws_kernel < TM, TK, ST, QW, (DL) == 2 ? 1 : 2,                          \
           (DL) == 0 ? 4                                                                      \
                     : 8 > (x, pre_quant_scale, global_scale, xq, sf, l2t_kmajor, down, m, k, \
                            grid_mode, stream);                                               \
  }
  FI_SVDQ_RANK128_WS_TILINGS(FI_SVDQ_WS_CASE)
#undef FI_SVDQ_WS_CASE
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::gemm
