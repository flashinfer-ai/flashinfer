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
// Rung 23 of the optimization ladder: 128(n) x 256(m) tile with
// wgmma.m64n256k16, revisited on the mature mainloop.  Rung 22's skewed
// warpgroups showed that one warpgroup alone cannot keep the tensor pipe
// busy with m64n128 wgmmas: its per-slab instruction stream (barrier wait,
// 2 ldmatrix, 16 conversions, 8 wgmma issues with fences) is too thin to
// cover the pipe when the other warpgroup is in its epilogue.  The same
// 32768 outputs per block are kept but each warpgroup owns ONE m64 row block
// of n and the full 256 m columns: every wgmma is twice as long
// (64 x 256 x 16), one A fragment per k16 instead of two, half the
// per-FLOP conversion work, and a single warpgroup's wgmma covers more of
// the other's epilogue.  The act slab doubles (32 KB) and the ring drops to
// 4 stages (160 KB + 32 KB epilogue buffers); L2 bytes per FLOP rise 22%
// versus rung 19 (the L2 sat at 56%).  Requires sm_90a, K % 64 == 0, 16 B
// aligned bases, N % 8 == 0.

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

constexpr int kW8A16TileN = 128;  // n rows per block (one wgmma M per consumer WG)
constexpr int kW8A16TileM = 256;  // m cols per block (wgmma N)
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 4;
constexpr int kW8A16GroupM = 8;       // m-tiles per rasterization group
constexpr int kW8A16ConsumerWGs = 2;  // 128 n rows each (2 row blocks of 64)
constexpr int kW8A16RowBlocks = kW8A16TileN / (64 * kW8A16ConsumerWGs);  // 1
constexpr int kW8A16Warpgroups = 1 + kW8A16ConsumerWGs;                  // + 1 producer
constexpr int kW8A16Threads = 128 * kW8A16Warpgroups;
constexpr int kW8A16ConsumerThreads = 128 * kW8A16ConsumerWGs;
constexpr int kW8A16ConsumerWarps = kW8A16ConsumerThreads / 32;
constexpr int kW8A16ARowBytes = kW8A16TileK * 2;  // act row = 64 bf16 = 128 B (SW128 atom)
constexpr int kW8A16WRowBytes = kW8A16TileK;      // weight row = 64 fp8 = 64 B (SW64 atom)
constexpr int kW8A16AStageBytes = kW8A16TileM * kW8A16ARowBytes;  // 32 KB
constexpr int kW8A16WStageBytes = kW8A16TileN * kW8A16WRowBytes;  // 8 KB
constexpr int kW8A16EpiPad = kW8A16TileN + 8;                     // elements per staged output row
constexpr int kW8A16MaxSplits = 8;
constexpr int kW8A16MinSlabsPerSplit = 8;  // amortize the partial-tile round trip
constexpr int kW8A16EpiBytes =
    kW8A16TileM * kW8A16EpiPad * 4;  // fp32 partial tile (split-K), reuses the stages
constexpr int kW8A16EpiPasses = 2;   // 16-bit epilogue: half the tile per pass
constexpr int kW8A16EpiRows = kW8A16TileM / kW8A16EpiPasses;
constexpr int kW8A16EpiWgCols = kW8A16TileN / kW8A16ConsumerWGs;  // 64 n per warpgroup
// Dense [64 m][128 n] per warpgroup; the 16 B chunk index is XOR-swizzled by
// (m >> 1) & 7 so the four even rows a warp writes at once hit different banks.
constexpr int kW8A16EpiWgBytes = kW8A16EpiRows * kW8A16EpiWgCols * 2;  // 16 KB per warpgroup
constexpr int kW8A16EpiOutBytes = kW8A16EpiWgBytes * kW8A16ConsumerWGs;
constexpr int kW8A16SkewNs = 1500;  // warpgroup 2 start delay (about one epilogue)
constexpr int kW8A16BarrierBytes = 2 * kW8A16Stages * 8;  // full[] + empty[]
constexpr int kW8A16StageBytes = kW8A16Stages * (kW8A16AStageBytes + kW8A16WStageBytes);
constexpr int kW8A16SmemBytesPersistent =
    kW8A16StageBytes + kW8A16EpiOutBytes + kW8A16BarrierBytes + 1024;
constexpr int kW8A16SmemBytesSplit = kW8A16StageBytes + kW8A16BarrierBytes + 1024;
static_assert(kW8A16SmemBytesPersistent <= 232448, "exceeds the 227 KB opt-in limit");

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
// Named barrier among the 128 threads of one consumer warpgroup (ids 2, 3).
__device__ __forceinline__ void warpgroup_bar_sync(int cwg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(2 + cwg) : "memory");
}
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
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

// D[64 x 256] (+)= A[64 x 16] (registers) . B[16 x 256] (smem descriptor).
template <typename T>
__device__ __forceinline__ void wgmma_m64n256k16_rs(float (&d)[128], const uint32_t (&a)[4],
                                                    uint64_t desc_b, int scale_d);
template <>
__device__ __forceinline__ void wgmma_m64n256k16_rs<__nv_bfloat16>(float (&d)[128],
                                                                   const uint32_t (&a)[4],
                                                                   uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %133, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
      "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
      "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, "
      "%74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, "
      "%92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, "
      "%108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
      "%123, %124, %125, %126, %127}, "
      "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
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
        "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
        "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]),
        "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
        "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]),
        "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]),
        "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]),
        "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
        "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]),
        "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
}
template <>
__device__ __forceinline__ void wgmma_m64n256k16_rs<__half>(float (&d)[128], const uint32_t (&a)[4],
                                                            uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %133, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, "
      "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
      "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, "
      "%74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, "
      "%92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, "
      "%108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
      "%123, %124, %125, %126, %127}, "
      "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
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
        "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]),
        "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]),
        "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
        "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]),
        "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]),
        "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]),
        "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
        "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]),
        "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
}

template <typename T, int kClusterSize, bool kPrepared, bool kSplitK>
__global__ void __launch_bounds__(kW8A16Threads, 1)
    W8A16Fp8GemmTmaKernel(const __grid_constant__ CUtensorMap act_map,
                          const __grid_constant__ CUtensorMap w_map,
                          const float* __restrict__ scale, const T* __restrict__ bias,
                          T* __restrict__ out, float* __restrict__ ws, int M, int N, int K) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) &
                                             ~static_cast<uintptr_t>(1023));
  uint8_t* a_base = smem;                                     // act stages (SW128)
  uint8_t* w_base = smem + kW8A16Stages * kW8A16AStageBytes;  // weight stages (SW64)
  uint8_t* epi_base = smem + kW8A16StageBytes;  // 16-bit epilogue buffer (absent for split-K)
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(epi_base + (kSplitK ? 0 : kW8A16EpiOutBytes));
  uint64_t* empty_bar = full_bar + kW8A16Stages;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wg = warp >> 2;  // 0 = producer, 1..2 = consumers

  // Grouped rasterization (rung 10) over cluster pairs: the two CTAs of a
  // cluster (consecutive blockIdx.x) share tile_m and take n-tiles 2p, 2p+1.
  // Persistent: a block walks units pair, pair + stride, ... (one unit when
  // the grid covers every unit, as in split-K mode).
  const uint32_t rank = kClusterSize > 1 ? cluster_ctarank() : 0u;
  const int pair = blockIdx.x / kClusterSize;
  const int unit_stride = gridDim.x / kClusterSize;
  const int tiles_m = (M + kW8A16TileM - 1) / kW8A16TileM;
  const int tiles_n = (N + kW8A16TileN - 1) / kW8A16TileN;
  const int pairs_n = (tiles_n + kClusterSize - 1) / kClusterSize;
  const int num_units = tiles_m * pairs_n;
  const int group_size = kW8A16GroupM * pairs_n;
  auto map_unit = [&](int unit, int& m0, int& n0) {
    const int group = unit / group_size;
    const int first_m = group * kW8A16GroupM;
    const int this_group_m = min(kW8A16GroupM, tiles_m - first_m);
    const int in_group = unit - group * group_size;
    const int tile_m = first_m + in_group % this_group_m;
    const int tile_n = (in_group / this_group_m) * kClusterSize + static_cast<int>(rank);
    n0 = tile_n * kW8A16TileN;  // may be >= N for an odd tile count: computes zeros
    m0 = tile_m * kW8A16TileM;
  };
  // Split-K: blockIdx.y selects a contiguous range of slabs.
  const int total_slabs = K / kW8A16TileK;
  const int per_split =
      (total_slabs + static_cast<int>(gridDim.y) - 1) / static_cast<int>(gridDim.y);
  const int slab_begin = kSplitK ? min(static_cast<int>(blockIdx.y) * per_split, total_slabs) : 0;
  const int num_slabs = kSplitK ? min(per_split, total_slabs - slab_begin) : total_slabs;

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
      uint32_t g = 0;  // running slab index across tiles: stage = g % S
      for (int unit = pair; unit < num_units; unit += unit_stride) {
        int m0, n0;
        map_unit(unit, m0, n0);
        for (int kt = 0; kt < num_slabs; ++kt, ++g) {
          const int stage = g % kW8A16Stages;
          const uint32_t round = g / kW8A16Stages;
          mbar_wait(&empty_bar[stage], (round & 1) ^ 1);  // fresh barrier: passes at once
          mbar_arrive_expect_tx(&full_bar[stage], kSlabBytes);
          const int k0 = (slab_begin + kt) * kW8A16TileK;
          if constexpr (kClusterSize > 1) {
            if (rank == 0) {
              tma_load_2d_multicast(a_base + stage * kW8A16AStageBytes, &act_map, &full_bar[stage],
                                    k0, m0, static_cast<uint16_t>((1u << kClusterSize) - 1));
            }
          } else {
            tma_load_2d(a_base + stage * kW8A16AStageBytes, &act_map, &full_bar[stage], k0, m0);
          }
          tma_load_2d(w_base + stage * kW8A16WStageBytes, &w_map, &full_bar[stage], k0, n0);
        }
      }
    }
  } else {
    // ------------------------------ consumers -----------------------------
    setmaxnreg_inc<232>();
    const int cwg = wg - 1;  // consumer warpgroup: n rows cwg*128 .. +127
    if (cwg == 1 && !kSplitK) {
      // Start one epilogue late so the two warpgroups alternate between
      // mainloop and epilogue instead of idling the tensor pipe together.
      const uint64_t t0 = globaltimer_ns();
      while (globaltimer_ns() - t0 < kW8A16SkewNs) {
      }
    }
    const int wq = warp & 3;  // warp within it: rows wq*16 .. +15 of each row block
    float d[kW8A16RowBlocks][128];

    // Row block b covers tile rows cwg*128 + b*64 .. +63.
    // Plain layout: rows r = base + wq*16 + lane/4 (+8), k = 2*(lane%4) (+8)
    // within each k16; the 16 B chunk holding k16 of row r sits at chunk
    // index (k16/16) ^ ((r>>1)&3) under the 64 B swizzle.
    const int a_r = cwg * (64 * kW8A16RowBlocks) + wq * 16 + lane / 4;
    const int a_k = (lane % 4) * 2;
    const uint32_t swz0 = (a_r >> 1) & 3;  // same for a_r + 64 (64 is 0 mod 8)
    const uint32_t swz1 = ((a_r + 8) >> 1) & 3;
    // Prepared layout: ldmatrix row for this lane = base + (lane % 8) +
    // 8 * ((lane / 8) % 2), chunk = 2 * pair + lane / 16.
    const int ld_r = cwg * (64 * kW8A16RowBlocks) + wq * 16 + (lane % 8) + 8 * ((lane / 8) % 2);
    const uint32_t ld_swz = (ld_r >> 1) & 3;
    const uint32_t ld_chunk_hi = lane / 16;

    // Load + convert the A fragments of one slab for both row blocks
    // (2 blocks x 4 k16 x 4 regs).
    auto load_frags = [&](uint32_t g, uint32_t(&frag)[kW8A16RowBlocks][4][4]) {
      const uint8_t* w_stage = w_base + (g % kW8A16Stages) * kW8A16WStageBytes;
#pragma unroll
      for (int b = 0; b < kW8A16RowBlocks; ++b) {
        if constexpr (kPrepared) {
#pragma unroll
          for (int pair = 0; pair < kW8A16TileK / 32; ++pair) {
            const uint32_t chunk = 2 * pair + ld_chunk_hi;
            uint32_t r[4];
            ldmatrix_x4(r, w_stage + (ld_r + b * 64) * kW8A16WRowBytes + ((chunk ^ ld_swz) << 4));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              frag[b][2 * pair + h][0] = fp8x2_to_pair<T>(r[2 * h] & 0xffffu);
              frag[b][2 * pair + h][1] = fp8x2_to_pair<T>(r[2 * h + 1] & 0xffffu);
              frag[b][2 * pair + h][2] = fp8x2_to_pair<T>(r[2 * h] >> 16);
              frag[b][2 * pair + h][3] = fp8x2_to_pair<T>(r[2 * h + 1] >> 16);
            }
          }
        } else {
#pragma unroll
          for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
            const uint32_t chunk = k16 >> 4;
            const uint8_t* r0 =
                w_stage + (a_r + b * 64) * kW8A16WRowBytes + ((chunk ^ swz0) << 4) + a_k;
            const uint8_t* r1 =
                w_stage + (a_r + b * 64 + 8) * kW8A16WRowBytes + ((chunk ^ swz1) << 4) + a_k;
            frag[b][k16 / 16][0] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0));
            frag[b][k16 / 16][1] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1));
            frag[b][k16 / 16][2] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0 + 8));
            frag[b][k16 / 16][3] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1 + 8));
          }
        }
      }
    };

    // One slab: issue `cur` for both row blocks, retire the previous group,
    // refill `nxt` with slab kt+1.  Buffers are named so they stay in
    // registers; the caller alternates them.
    auto step = [&](int kt, uint32_t g, uint32_t(&cur)[kW8A16RowBlocks][4][4],
                    uint32_t(&nxt)[kW8A16RowBlocks][4][4]) {
      const uint8_t* a_stage = a_base + (g % kW8A16Stages) * kW8A16AStageBytes;
      wgmma_fence();
#pragma unroll
      for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
        const uint64_t desc_b = make_smem_desc_sw128(a_stage + k16 * 2);
#pragma unroll
        for (int b = 0; b < kW8A16RowBlocks; ++b) {
          wgmma_m64n256k16_rs<T>(d[b], cur[b][k16 / 16], desc_b, 1);
        }
      }
      wgmma_commit();
      wgmma_wait<1>();
      if (kt > 0 && lane == 0) {
        uint64_t* eb = &empty_bar[(g - 1) % kW8A16Stages];
        if constexpr (kClusterSize > 1) {
#pragma unroll
          for (uint32_t c = 0; c < kClusterSize; ++c) mbar_arrive_remote(eb, c);
        } else {
          mbar_arrive(eb);
        }
      }
      if (kt + 1 < num_slabs) {
        const uint32_t ng = g + 1;
        mbar_wait(&full_bar[ng % kW8A16Stages], (ng / kW8A16Stages) & 1);
        load_frags(ng, nxt);  // overlaps slab kt's wgmmas
      }
    };

    uint32_t frag0[kW8A16RowBlocks][4][4], frag1[kW8A16RowBlocks][4][4];
    uint32_t g = 0;  // running slab index across tiles (mirrors the producer's)
    for (int unit = pair; unit < num_units; unit += unit_stride) {
      int m0, n0;
      map_unit(unit, m0, n0);
#pragma unroll
      for (int b = 0; b < kW8A16RowBlocks; ++b)
#pragma unroll
        for (int i = 0; i < 128; ++i) d[b][i] = 0.f;
      if (num_slabs > 0) {
        mbar_wait(&full_bar[g % kW8A16Stages], (g / kW8A16Stages) & 1);
        load_frags(g, frag0);
      }
      for (int kt = 0; kt < num_slabs; kt += 2) {
        step(kt, g + kt, frag0, frag1);
        if (kt + 1 < num_slabs) step(kt + 1, g + kt + 1, frag1, frag0);
      }
      g += num_slabs;
      wgmma_wait<0>();
      if (num_slabs > 0 && lane == 0) {
        uint64_t* eb = &empty_bar[(g - 1) % kW8A16Stages];
        if constexpr (kClusterSize > 1) {
#pragma unroll
          for (uint32_t c = 0; c < kClusterSize; ++c) mbar_arrive_remote(eb, c);
        } else {
          mbar_arrive(eb);
        }
      }

      // Epilogue.  Fragment layout per warp (m64nNk16 D): d[b][j*4 + 0..1] at
      // (n = b*64 + wq*16 + lane/4, m = 8j + 2*(lane%4) + {0,1}), d[b][j*4 + 2..3]
      // at n + 8.  Stage the tile as out[m][n] in smem, then store rows with
      // 16 B writes: scaled 16-bit values to `out`, or raw fp32 partials to the
      // split-K workspace (scale / bias / cast happen in the reduce kernel).
      static_assert(kW8A16EpiBytes <= kW8A16StageBytes,
                    "fp32 split-K tile must fit in the pipeline stages");
      using TOut = std::conditional_t<kSplitK, float, T>;
      // Split-K (one unit per block) stages its fp32 tile in the idle stages;
      // the persistent 16-bit path uses the dedicated buffer, half a tile per
      // pass, so the producer can keep filling stages for the next tile.
      constexpr int kPasses = kSplitK ? 1 : kW8A16EpiPasses;
      constexpr int kRowsPerTilePass = kW8A16TileM / kPasses;
      // Split-K keeps the shared fp32 tile in the idle stages (both warpgroups
      // sync); the persistent path gives each warpgroup its own [64 m][136 n]
      // buffer for its 128 n columns and never synchronizes across warpgroups.
      constexpr int kWgCols = kSplitK ? kW8A16TileN : kW8A16EpiWgCols;
      TOut* tile = kSplitK ? reinterpret_cast<TOut*>(smem)
                           : reinterpret_cast<TOut*>(epi_base + cwg * kW8A16EpiWgBytes);
      const float sc = kSplitK ? 1.f : *scale;
      const int m_lo = (lane % 4) * 2;  // tile-local m
      // Element index inside the staging buffer: padded rows for the fp32
      // split-K tile, dense swizzled rows for the per-warpgroup 16-bit tile.
      auto epi_idx = [](int m, int n) -> int {
        if constexpr (kSplitK) return m * kW8A16EpiPad + n;
        return m * kW8A16EpiWgCols + ((((n >> 3) ^ ((m >> 1) & 7))) << 3) + (n & 7);
      };
      constexpr int kElemsPerChunk = 16 / sizeof(TOut);  // 8 or 4
      constexpr int kChunks = kWgCols / kElemsPerChunk;  // 16 or 64
      constexpr int kThreadsHere = kSplitK ? kW8A16ConsumerThreads : 128;
      constexpr int kRowsPerPass = kThreadsHere / kChunks;  // 8 or 4
      const int ct = kSplitK ? threadIdx.x - 128 : threadIdx.x - 128 - cwg * 128;
      const int chunk = ct % kChunks;
      const int row0 = ct / kChunks;
      const int n_chunk = n0 + (kSplitK ? 0 : cwg * kW8A16EpiWgCols) + chunk * kElemsPerChunk;
      TOut* dst_base = kSplitK
                           ? reinterpret_cast<TOut*>(ws) + static_cast<int64_t>(blockIdx.y) * M * N
                           : reinterpret_cast<TOut*>(out);
      auto sync_here = [&]() {
        if constexpr (kSplitK) {
          consumer_bar_sync();
        } else {
          warpgroup_bar_sync(cwg);
        }
      };
      if constexpr (kSplitK) consumer_bar_sync();  // both warpgroups are done reading the stages
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        sync_here();  // previous pass / tile copied out before the buffer is reused
#pragma unroll
        for (int b = 0; b < kW8A16RowBlocks; ++b) {
          const int n_lo =
              cwg * (64 * kW8A16RowBlocks) + b * 64 + wq * 16 + lane / 4;   // tile-local n
          const int n_buf = kSplitK ? n_lo : n_lo - cwg * kW8A16EpiWgCols;  // buffer-local n
          float bias_lo = 0.f, bias_hi = 0.f;
          if (!kSplitK && bias != nullptr) {
            if (n0 + n_lo < N) bias_lo = static_cast<float>(bias[n0 + n_lo]);
            if (n0 + n_lo + 8 < N) bias_hi = static_cast<float>(bias[n0 + n_lo + 8]);
          }
#pragma unroll
          for (int jj = 0; jj < (kW8A16TileM / 8) / kPasses; ++jj) {
            const int j = pass * ((kW8A16TileM / 8) / kPasses) + jj;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
              const int m = m_lo + jj * 8 + q;  // row within this pass's buffer
              tile[epi_idx(m, n_buf)] = static_cast<TOut>(d[b][j * 4 + q] * sc + bias_lo);
              tile[epi_idx(m, n_buf + 8)] = static_cast<TOut>(d[b][j * 4 + 2 + q] * sc + bias_hi);
            }
          }
        }
        sync_here();  // this pass's rows are staged
#pragma unroll
        for (int i = 0; i < kRowsPerTilePass / kRowsPerPass; ++i) {
          const int r = row0 + kRowsPerPass * i;
          const int m = m0 + pass * kRowsPerTilePass + r;
          if (m >= M) continue;
          const TOut* src = tile + epi_idx(r, chunk * kElemsPerChunk);
          TOut* dst = dst_base + static_cast<int64_t>(m) * N + n_chunk;
          if (n_chunk + kElemsPerChunk <= N) {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
          } else {
            for (int e = 0; n_chunk + e < N; ++e) dst[e] = src[e];
          }
        }
      }
    }  // unit loop
  }
  // No CTA may exit while its peer can still multicast into its smem or
  // arrive on its barriers.
  if constexpr (kClusterSize > 1) cluster_sync_all();
#else
  __trap();  // wgmma / TMA need sm_90a
#endif
}

// Split-K reduce: out[i] = T(sum_s ws[s][i] * scale + bias[i % N]), 8 elements
// per thread (two 16 B loads per split, one 16 B store).  Requires N % 8 == 0.
template <typename T>
__global__ void W8A16SplitKReduceKernel(const float* __restrict__ ws, int splits,
                                        const float* __restrict__ scale, const T* __restrict__ bias,
                                        T* __restrict__ out, int64_t mn, int N) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= mn) return;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const float4* src = reinterpret_cast<const float4*>(ws + static_cast<int64_t>(sp) * mn + i);
    const float4 a = src[0], b = src[1];
    acc[0] += a.x;
    acc[1] += a.y;
    acc[2] += a.z;
    acc[3] += a.w;
    acc[4] += b.x;
    acc[5] += b.y;
    acc[6] += b.z;
    acc[7] += b.w;
  }
  const float sc = *scale;
  const int n = static_cast<int>(i % N);
  alignas(16) T v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float bb = bias != nullptr ? static_cast<float>(bias[n + e]) : 0.f;
    v[e] = static_cast<T>(acc[e] * sc + bb);
  }
  *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(v);
}

// Split-K plan: number of K splits for a problem (1 = no split).  Split only
// when the tile grid is under a third of a wave of SMs, aim for one wave, and
// keep at least 8 slabs per split so the fp32 partial round trip stays small
// next to the mainloop.
inline int W8A16Fp8GemmSplits(int M, int N, int K, int num_sms) {
  const int tiles =
      static_cast<int>(ceil_div(M, kW8A16TileM)) * static_cast<int>(ceil_div(N, kW8A16TileN));
  const int total_slabs = K / kW8A16TileK;
  // Measured on H100: half a wave of 256 x 128 tiles with K = 5120 already
  // beats the split (partial round trip + reduce); split below a third.
  if (tiles * 3 > num_sms || total_slabs < 2 * kW8A16MinSlabsPerSplit) return 1;
  int splits = static_cast<int>(ceil_div(num_sms, tiles));
  splits = min(splits, kW8A16MaxSplits);
  splits = min(splits, total_slabs / kW8A16MinSlabsPerSplit);
  return max(splits, 1);
}

// Workspace bytes a call needs (0 when it will not split).
inline int64_t W8A16Fp8GemmWorkspaceBytes(int M, int N, int K, int num_sms) {
  const int splits = W8A16Fp8GemmSplits(M, N, K, num_sms);
  return splits > 1 ? static_cast<int64_t>(splits) * M * N * static_cast<int64_t>(sizeof(float))
                    : 0;
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
                         T* out, int M, int N, int K, bool weight_prepared, float* workspace,
                         int64_t workspace_bytes, int num_sms, cudaStream_t stream) {
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
  int splits = W8A16Fp8GemmSplits(M, N, K, num_sms);
  if (splits > 1 &&
      (workspace == nullptr || workspace_bytes < static_cast<int64_t>(splits) * M * N * 4)) {
    splits = 1;  // no room for partials: run the single-pass kernel
  }
  const bool split = splits > 1;
  using KernelFn = void (*)(const CUtensorMap, const CUtensorMap, const float*, const T*, T*,
                            float*, int, int, int);
  static const KernelFn kernels[2][2] = {{W8A16Fp8GemmTmaKernel<T, kClusterSize, false, false>,
                                          W8A16Fp8GemmTmaKernel<T, kClusterSize, false, true>},
                                         {W8A16Fp8GemmTmaKernel<T, kClusterSize, true, false>,
                                          W8A16Fp8GemmTmaKernel<T, kClusterSize, true, true>}};
  KernelFn kernel = kernels[weight_prepared][split];
  static bool attr_set[2][2] = {{false, false}, {false, false}};
  const int smem_bytes = split ? kW8A16SmemBytesSplit : kW8A16SmemBytesPersistent;
  if (!attr_set[weight_prepared][split]) {
    e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (e != cudaSuccess) return e;
    attr_set[weight_prepared][split] = true;
  }
  cudaLaunchConfig_t cfg = {};
  // Persistent: one block (cluster) per SM walks the units; split-K keeps one
  // unit per block so its fp32 tile can live in the idle stages.
  const unsigned units = tiles_m * pairs_n;
  const unsigned clusters =
      split ? units : min(units, static_cast<unsigned>(num_sms / kClusterSize));
  cfg.gridDim = dim3(clusters * kClusterSize, splits);  // x rasterized; y = K split
  cfg.blockDim = dim3(kW8A16Threads);
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kClusterSize;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = kClusterSize > 1 ? 1 : 0;
  e = cudaLaunchKernelEx(&cfg, kernel, act_map, w_map, scale, bias, out, workspace, M, N, K);
  if (e != cudaSuccess) return e;
  if (split) {
    const int64_t mn = static_cast<int64_t>(M) * N;
    const int threads = 256;
    const int64_t blocks = (mn / 8 + threads - 1) / threads;
    W8A16SplitKReduceKernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        workspace, splits, scale, bias, out, mn, N);
  }
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
