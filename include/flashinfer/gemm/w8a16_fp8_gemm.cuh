/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_

// W8A16 GEMM: FP8 e4m3 weight with one per-tensor fp32 scale, 16-bit
// activation, output in the activation dtype, optional per-N bias.
//
//   out[m, n] = sum_k act[m, k] * (float(w[n, k]) * scale)  (+ bias[n])
//
// The weight is dequantized on the fly; the 16-bit copy TRT-LLM's cast-copy
// path materialises never exists.  The weight is [N, K] row-major (the
// torch Linear layout, "NT" GEMM), act is [M, K] row-major.
//
// Rung 16 of the optimization ladder: opt-in prepared weight layout with
// ldmatrix fragment loads.  Through rung 15 every consumer thread built its
// wgmma A fragment with four 2-byte shared-memory loads per k16 (16 LDS.U16
// per slab, each with its own address arithmetic) because the fragment
// wants (k, k+1) and (k+8, k+9) of one row while memory holds k..k+15
// contiguous.  With `prepare_w8a16_fp8_weight()` the weight's K axis is
// permuted once, inside each 16-wide block, to [2c, 2c+1, 2c+8, 2c+9] for
// c = 0..3 (same [N, K] shape, same per-tensor scale); then a single
// ldmatrix.x4 on the fp8 tile delivers, per thread, exactly the four
// bytes of two A registers for two k16 blocks at once: 2 LDSM per slab
// instead of 16 LDS, no per-fragment address math, and the 64 B swizzle
// keeps the 8-row phases conflict-free.  The plain checkpoint layout stays
// the default (`kPrepared = false`, rung-15 path) and stays in the
// benchmark; the prepared path is selected by the caller.  Requires
// sm_90a, K % 64 == 0, 16 B aligned bases, N % 8 == 0.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileN = 128;  // n rows per block (wgmma M)
constexpr int kW8A16TileM = 128;  // m cols per block (wgmma N)
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 6;
constexpr int kW8A16GroupM = 8;                          // m-tiles per rasterization group
constexpr int kW8A16ConsumerWGs = 2;                     // 64 n rows each
constexpr int kW8A16Warpgroups = 1 + kW8A16ConsumerWGs;  // + 1 producer
constexpr int kW8A16Threads = 128 * kW8A16Warpgroups;
constexpr int kW8A16ConsumerThreads = 128 * kW8A16ConsumerWGs;
constexpr int kW8A16ConsumerWarps = kW8A16ConsumerThreads / 32;
constexpr int kW8A16ARowBytes = kW8A16TileK * 2;  // act row = 64 bf16 = 128 B (SW128 atom)
constexpr int kW8A16WRowBytes = kW8A16TileK;      // weight row = 64 fp8 = 64 B (SW64 atom)
constexpr int kW8A16AStageBytes = kW8A16TileM * kW8A16ARowBytes;  // 16 KB
constexpr int kW8A16WStageBytes = kW8A16TileN * kW8A16WRowBytes;  // 8 KB
constexpr int kW8A16EpiPad = kW8A16TileN + 8;  // 16-bit elements per staged output row
constexpr int kW8A16EpiBytes = kW8A16TileM * kW8A16EpiPad * 2;  // 34 KB, reuses the stages
constexpr int kW8A16BarrierBytes = 2 * kW8A16Stages * 8;        // full[] + empty[]
constexpr int kW8A16SmemBytes =
    kW8A16Stages * (kW8A16AStageBytes + kW8A16WStageBytes) + kW8A16BarrierBytes + 1024;

// Two e4m3 values (packed in the low 16 bits) -> two T values packed in a
// 32-bit register, laid out as an mma A/B operand expects.
template <typename T>
__device__ __forceinline__ uint32_t fp8x2_to_pair(uint32_t packed16);
template <>
__device__ __forceinline__ uint32_t fp8x2_to_pair<__half>(uint32_t packed16) {
  const __half2_raw h =
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed16), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}
template <>
__device__ __forceinline__ uint32_t fp8x2_to_pair<__nv_bfloat16>(uint32_t packed16) {
  const __half2 h =
      __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed16), __NV_E4M3));
  const __nv_bfloat162 b = __float22bfloat162_rn(__half22float2(h));
  return *reinterpret_cast<const uint32_t*>(&b);
}

// ---- wgmma plumbing (sm_90a) -------------------------------------------

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// ---- mbarrier / TMA plumbing ---------------------------------------------

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .b64 st;\n"
      "mbarrier.arrive.shared::cta.b64 st, [%0];\n"
      "}\n" ::"r"(a)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(a),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
// 2-D TMA tile load: box at (c0 = innermost coordinate, c1) -> smem, bytes
// counted on the mbarrier.
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* map, uint64_t* bar,
                                            int c0, int c1) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %4}], [%2];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(b), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
}
// Arrive on the mbarrier at the same smem offset in CTA `rank` of the cluster.
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .b32 ra;\n"
      "mapa.shared::cluster.u32 ra, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [ra];\n"
      "}\n" ::"r"(a),
      "r"(rank)
      : "memory");
}
// 2-D TMA tile load multicast to every CTA in `cta_mask`: the box lands at
// the same smem offset in each and credits each CTA's barrier at `bar`.
__device__ __forceinline__ void tma_load_2d_multicast(void* smem_dst, const CUtensorMap* map,
                                                      uint64_t* bar, int c0, int c1,
                                                      uint16_t cta_mask) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  const uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::"
      "cluster [%0], [%1, {%3, %4}], [%2], %5;\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(b), "r"(c0), "r"(c1), "h"(cta_mask)
      : "memory");
}

// ldmatrix x4 (four 8x8 b16 matrices; lane l supplies the row address of
// row l % 8 of matrix l / 8).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem_row_ptr) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_row_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}

// Named barrier among the consumer threads only (id 1; 0 is __syncthreads).
__device__ __forceinline__ void consumer_bar_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kW8A16ConsumerThreads) : "memory");
}

// K-major, 128 B swizzle matrix descriptor: 8-row core-matrix groups are
// 1024 B apart (SBO), rows are 128 B, the 16 B chunks of a row are XOR
// swizzled by (row % 8) inside a 1024 B aligned atom.
__device__ __forceinline__ uint64_t make_smem_desc_sw128(const void* smem_ptr) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((addr & 0x3FFFF) >> 4);  // start address
  desc |= static_cast<uint64_t>(1) << 16;                // LBO (unused for SW128 K-major)
  desc |= static_cast<uint64_t>(1024 >> 4) << 32;        // SBO
  desc |= static_cast<uint64_t>(1) << 62;                // layout: 128 B swizzle
  return desc;
}

// D[64 x 128] (+)= A[64 x 16] (registers) . B[16 x 128] (smem descriptor).
template <typename T>
__device__ __forceinline__ void wgmma_m64n128k16_rs(float (&d)[64], const uint32_t (&a)[4],
                                                    uint64_t desc_b, int scale_d);
template <>
__device__ __forceinline__ void wgmma_m64n128k16_rs<__nv_bfloat16>(float (&d)[64],
                                                                   const uint32_t (&a)[4],
                                                                   uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
      "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
      "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
        "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
        "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
        "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
}
template <>
__device__ __forceinline__ void wgmma_m64n128k16_rs<__half>(float (&d)[64], const uint32_t (&a)[4],
                                                            uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
      "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
      "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
        "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
        "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
        "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),
        "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
        "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),
        "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
}

template <typename T, int kClusterSize, bool kPrepared>
__global__ void __launch_bounds__(kW8A16Threads, 1)
    W8A16Fp8GemmTmaKernel(const __grid_constant__ CUtensorMap act_map,
                          const __grid_constant__ CUtensorMap w_map,
                          const float* __restrict__ scale, const T* __restrict__ bias,
                          T* __restrict__ out, int M, int N, int K) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) &
                                             ~static_cast<uintptr_t>(1023));
  uint8_t* a_base = smem;                                     // act stages (SW128)
  uint8_t* w_base = smem + kW8A16Stages * kW8A16AStageBytes;  // weight stages (SW64)
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(w_base + kW8A16Stages * kW8A16WStageBytes);
  uint64_t* empty_bar = full_bar + kW8A16Stages;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wg = warp >> 2;  // 0 = producer, 1..2 = consumers

  // Grouped rasterization (rung 10) over cluster pairs: the two CTAs of a
  // cluster (consecutive blockIdx.x) share tile_m and take n-tiles 2p, 2p+1.
  const uint32_t rank = kClusterSize > 1 ? cluster_ctarank() : 0u;
  const int pair = blockIdx.x / kClusterSize;
  const int tiles_m = (M + kW8A16TileM - 1) / kW8A16TileM;
  const int tiles_n = (N + kW8A16TileN - 1) / kW8A16TileN;
  const int pairs_n = (tiles_n + kClusterSize - 1) / kClusterSize;
  const int group_size = kW8A16GroupM * pairs_n;
  const int group = pair / group_size;
  const int first_m = group * kW8A16GroupM;
  const int this_group_m = min(kW8A16GroupM, tiles_m - first_m);
  const int in_group = pair - group * group_size;
  const int tile_m = first_m + in_group % this_group_m;
  const int tile_n = (in_group / this_group_m) * kClusterSize + static_cast<int>(rank);
  const int n0 = tile_n * kW8A16TileN;  // may be >= N for an odd tile count: computes zeros
  const int m0 = tile_m * kW8A16TileM;
  const int num_slabs = K / kW8A16TileK;

  if (threadIdx.x == 0) {
    prefetch_tensormap(&act_map);
    prefetch_tensormap(&w_map);
#pragma unroll
    for (int s = 0; s < kW8A16Stages; ++s) {
      mbar_init(&full_bar[s], 1);  // one arrive.expect_tx per slab (tx from both copies)
      // One arrive per consumer warp of every CTA in the cluster.
      mbar_init(&empty_bar[s], kClusterSize * kW8A16ConsumerWarps);
    }
    fence_barrier_init();
  }
  if constexpr (kClusterSize > 1) {
    cluster_sync_all();  // peers' barriers are initialised before any multicast / remote arrive
  } else {
    __syncthreads();
  }

  if (wg == 0) {
    // ------------------------------ producer ------------------------------
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {
      constexpr uint32_t kSlabBytes = kW8A16AStageBytes + kW8A16WStageBytes;
      for (int kt = 0; kt < num_slabs; ++kt) {
        const int stage = kt % kW8A16Stages;
        const uint32_t round = kt / kW8A16Stages;
        mbar_wait(&empty_bar[stage], (round & 1) ^ 1);  // fresh barrier: passes at once
        mbar_arrive_expect_tx(&full_bar[stage], kSlabBytes);
        const int k0 = kt * kW8A16TileK;
        if (rank == 0) {
          tma_load_2d_multicast(a_base + stage * kW8A16AStageBytes, &act_map, &full_bar[stage], k0,
                                m0, static_cast<uint16_t>((1u << kClusterSize) - 1));
        }
        tma_load_2d(w_base + stage * kW8A16WStageBytes, &w_map, &full_bar[stage], k0, n0);
      }
    }
  } else {
    // ------------------------------ consumers -----------------------------
    setmaxnreg_inc<232>();
    const int cwg = wg - 1;   // consumer warpgroup: n rows cwg*64 .. +63
    const int wq = warp & 3;  // warp within it: rows wq*16 .. +15 of those
    float d[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;

    // Plain layout: rows r = cwg*64 + wq*16 + lane/4 (+8), k = 2*(lane%4)
    // (+8) within each k16; the 16 B chunk holding k16 of row r sits at
    // chunk index (k16/16) ^ ((r>>1)&3) under the 64 B swizzle.
    const int a_r = cwg * 64 + wq * 16 + lane / 4;
    const int a_k = (lane % 4) * 2;
    const uint32_t swz0 = (a_r >> 1) & 3;
    const uint32_t swz1 = ((a_r + 8) >> 1) & 3;
    // Prepared layout: ldmatrix row for this lane = warp row (lane % 8) +
    // 8 * ((lane / 8) % 2), chunk = 2 * pair + lane / 16.
    const int ld_r = cwg * 64 + wq * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
    const uint32_t ld_swz = (ld_r >> 1) & 3;
    const uint32_t ld_chunk_hi = lane / 16;

    for (int kt = 0; kt < num_slabs; ++kt) {
      const int stage = kt % kW8A16Stages;
      const uint32_t round = kt / kW8A16Stages;
      mbar_wait(&full_bar[stage], round & 1);
      const uint8_t* a_stage = a_base + stage * kW8A16AStageBytes;
      const uint8_t* w_stage = w_base + stage * kW8A16WStageBytes;
      if constexpr (kPrepared) {
#pragma unroll
        for (int pair = 0; pair < kW8A16TileK / 32; ++pair) {
          const uint32_t chunk = 2 * pair + ld_chunk_hi;
          uint32_t r[4];
          ldmatrix_x4(r, w_stage + ld_r * kW8A16WRowBytes + ((chunk ^ ld_swz) << 4));
#pragma unroll
          for (int h = 0; h < 2; ++h) {  // k16 block 2*pair + h
            uint32_t a_frag[4];
            a_frag[0] = fp8x2_to_pair<T>(r[2 * h] & 0xffffu);
            a_frag[1] = fp8x2_to_pair<T>(r[2 * h + 1] & 0xffffu);
            a_frag[2] = fp8x2_to_pair<T>(r[2 * h] >> 16);
            a_frag[3] = fp8x2_to_pair<T>(r[2 * h + 1] >> 16);
            const uint64_t desc_b = make_smem_desc_sw128(a_stage + (2 * pair + h) * 32);
            wgmma_fence();
            wgmma_m64n128k16_rs<T>(d, a_frag, desc_b, 1);
          }
        }
      } else {
#pragma unroll
        for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
          const uint32_t chunk = k16 >> 4;
          const uint8_t* r0 = w_stage + a_r * kW8A16WRowBytes + ((chunk ^ swz0) << 4) + a_k;
          const uint8_t* r1 = w_stage + (a_r + 8) * kW8A16WRowBytes + ((chunk ^ swz1) << 4) + a_k;
          uint32_t a_frag[4];
          a_frag[0] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0));
          a_frag[1] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1));
          a_frag[2] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0 + 8));
          a_frag[3] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1 + 8));
          const uint64_t desc_b = make_smem_desc_sw128(a_stage + k16 * 2);
          wgmma_fence();
          wgmma_m64n128k16_rs<T>(d, a_frag, desc_b, 1);
        }
      }
      wgmma_commit();
      wgmma_wait<1>();
      if (kt > 0 && lane == 0) {
        uint64_t* eb = &empty_bar[(kt - 1) % kW8A16Stages];
        if constexpr (kClusterSize > 1) {
#pragma unroll
          for (uint32_t c = 0; c < kClusterSize; ++c) mbar_arrive_remote(eb, c);
        } else {
          mbar_arrive(eb);
        }
      }
    }
    wgmma_wait<0>();
    if (num_slabs > 0 && lane == 0) {
      uint64_t* eb = &empty_bar[(num_slabs - 1) % kW8A16Stages];
      if constexpr (kClusterSize > 1) {
#pragma unroll
        for (uint32_t c = 0; c < kClusterSize; ++c) mbar_arrive_remote(eb, c);
      } else {
        mbar_arrive(eb);
      }
    }

    // Epilogue.  Fragment layout per warp (m64nNk16 D): d[j*4 + 0..1] at
    // (n = wq*16 + lane/4, m = 8j + 2*(lane%4) + {0,1}), d[j*4 + 2..3] at
    // n + 8.  Stage the scaled tile as out[m][n] in smem, then store rows.
    static_assert(kW8A16EpiBytes <= kW8A16Stages * (kW8A16AStageBytes + kW8A16WStageBytes),
                  "epilogue tile must fit in the pipeline stages");
    consumer_bar_sync();  // both warpgroups are done reading the stages
    T* tile = reinterpret_cast<T*>(smem);
    const float sc = *scale;
    const int n_lo = cwg * 64 + wq * 16 + lane / 4;  // tile-local n
    const int m_lo = (lane % 4) * 2;                 // tile-local m
    float bias_lo = 0.f, bias_hi = 0.f;
    if (bias != nullptr) {
      if (n0 + n_lo < N) bias_lo = static_cast<float>(bias[n0 + n_lo]);
      if (n0 + n_lo + 8 < N) bias_hi = static_cast<float>(bias[n0 + n_lo + 8]);
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int m = m_lo + j * 8 + q;
        tile[m * kW8A16EpiPad + n_lo] = static_cast<T>(d[j * 4 + q] * sc + bias_lo);
        tile[m * kW8A16EpiPad + n_lo + 8] = static_cast<T>(d[j * 4 + 2 + q] * sc + bias_hi);
      }
    }
    consumer_bar_sync();  // the whole 128 x 128 tile is staged
    // 128 rows x 16 chunks of 16 B; thread t owns chunk t % 16 of rows t / 16 + 16 i.
    const int ct = threadIdx.x - 128;  // consumer-local thread id
    const int chunk = ct % 16;
    const int row0 = ct / 16;
    const int n_chunk = n0 + chunk * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = row0 + 16 * i;
      const int m = m0 + r;
      if (m >= M) continue;
      const T* src = tile + r * kW8A16EpiPad + chunk * 8;
      T* dst = out + static_cast<int64_t>(m) * N + n_chunk;
      if (n_chunk + 8 <= N) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      } else {
        for (int e = 0; n_chunk + e < N; ++e) dst[e] = src[e];
      }
    }
  }
  // No CTA may exit while its peer can still multicast into its smem or
  // arrive on its barriers.
  if constexpr (kClusterSize > 1) cluster_sync_all();
#else
  __trap();  // wgmma / TMA need sm_90a
#endif
}

// Host side: 2-D tiled tensor map over a row-major [rows x cols] matrix with
// element size `elem_bytes`, box [box_cols (inner) x box_rows].
inline cudaError_t MakeTmaMap2D(CUtensorMap* map, const void* base, uint64_t rows, uint64_t cols,
                                uint32_t elem_bytes, CUtensorMapDataType dtype, uint32_t box_rows,
                                uint32_t box_cols, CUtensorMapSwizzle swizzle) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  if (encode == nullptr) {
    cudaDriverEntryPointQueryResult q;
    void* fn = nullptr;
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    if (e != cudaSuccess || fn == nullptr) return cudaErrorUnknown;
    encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  const uint64_t dims[2] = {cols, rows};
  const uint64_t strides[1] = {cols * elem_bytes};  // bytes, for dim 1
  const uint32_t box[2] = {box_cols, box_rows};
  const uint32_t estr[2] = {1, 1};
  CUresult r = encode(map, dtype, 2, const_cast<void*>(base), dims, strides, box, estr,
                      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <typename T, int kClusterSize = 1>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, bool weight_prepared, cudaStream_t stream) {
  if (K % kW8A16TileK != 0) return cudaErrorInvalidValue;  // whole slabs only
  if ((reinterpret_cast<uintptr_t>(act) & 15) || (reinterpret_cast<uintptr_t>(w) & 15)) {
    return cudaErrorInvalidValue;  // TMA needs 16 B aligned bases
  }
  if (N % 8 != 0 || (reinterpret_cast<uintptr_t>(out) & 15)) {
    return cudaErrorInvalidValue;  // 16 B output stores need N % 8 == 0 and an aligned out
  }
  CUtensorMap act_map, w_map;
  cudaError_t e = MakeTmaMap2D(&act_map, act, M, K, sizeof(T),
                               std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                                         : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               kW8A16TileM, kW8A16TileK, CU_TENSOR_MAP_SWIZZLE_128B);
  if (e != cudaSuccess) return e;
  e = MakeTmaMap2D(&w_map, w, N, K, 1, CU_TENSOR_MAP_DATA_TYPE_UINT8, kW8A16TileN, kW8A16TileK,
                   CU_TENSOR_MAP_SWIZZLE_64B);
  if (e != cudaSuccess) return e;

  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  const unsigned tiles_n = ceil_div(N, kW8A16TileN);
  const unsigned pairs_n = ceil_div(tiles_n, static_cast<unsigned>(kClusterSize));
  auto* kernel = weight_prepared ? W8A16Fp8GemmTmaKernel<T, kClusterSize, true>
                                 : W8A16Fp8GemmTmaKernel<T, kClusterSize, false>;
  static bool attr_set[2] = {false, false};  // per instantiation and layout
  if (!attr_set[weight_prepared]) {
    e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8A16SmemBytes);
    if (e != cudaSuccess) return e;
    attr_set[weight_prepared] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(tiles_m * pairs_n * kClusterSize);  // 1-D; the kernel rasterizes it
  cfg.blockDim = dim3(kW8A16Threads);
  cfg.dynamicSmemBytes = kW8A16SmemBytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kClusterSize;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = kClusterSize > 1 ? 1 : 0;
  e = cudaLaunchKernelEx(&cfg, kernel, act_map, w_map, scale, bias, out, M, N, K);
  if (e != cudaSuccess) return e;
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
