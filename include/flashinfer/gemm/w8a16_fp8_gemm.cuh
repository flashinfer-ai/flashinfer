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
// Rung 9 of the optimization ladder: warp specialization with mbarriers.
// Rung 8 had every warp both copying and computing, and had to drain its
// wgmma groups (wait_group 0) before each slab so the stage could be
// refilled; ncu: tensor pipe ~47%, issue 37%, 25% occupancy.  Here the
// block is three warpgroups: warpgroup 0 is the producer (it only issues
// the cp.async copies and signals a per-stage "full" mbarrier through
// cp.async.mbarrier.arrive), warpgroups 1-2 are the consumers (they wait on
// "full", build the A fragments, issue the wgmmas, and after wait_group 1
// release the previous stage through an "empty" mbarrier).  The producer
// never waits on compute and the consumers keep one slab of wgmma in
// flight.  setmaxnreg moves registers from the producer (40) to the
// consumers (232), and the ring grows to 6 stages (150 KB, 1 block / SM).
// A fence.proxy.async makes the cp.async-written tiles visible to the
// wgmma async proxy.  Requires sm_90a and K % 64 == 0.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "../cp_async.cuh"
#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileN = 128;  // n rows per block (wgmma M)
constexpr int kW8A16TileM = 128;  // m cols per block (wgmma N)
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 6;
constexpr int kW8A16ConsumerWGs = 2;                     // 64 n rows each
constexpr int kW8A16Warpgroups = 1 + kW8A16ConsumerWGs;  // + 1 producer
constexpr int kW8A16Threads = 128 * kW8A16Warpgroups;
constexpr int kW8A16ConsumerThreads = 128 * kW8A16ConsumerWGs;
constexpr int kW8A16WPad = kW8A16TileK + 8;       // 72 fp8 rows (8 B aligned, 18 words)
constexpr int kW8A16ARowBytes = kW8A16TileK * 2;  // act row = 64 bf16 = 128 B (swizzle atom)
constexpr int kW8A16AStageBytes = kW8A16TileM * kW8A16ARowBytes;  // 16 KB
constexpr int kW8A16WStageBytes = kW8A16TileN * kW8A16WPad;       // 9 KB
constexpr int kW8A16BarrierBytes = 2 * kW8A16Stages * 8;          // full[] + empty[]
constexpr int kW8A16SmemBytes =
    kW8A16Stages * (kW8A16AStageBytes + kW8A16WStageBytes) + kW8A16BarrierBytes + 1024;

// 8-byte predicated cp.async with zero fill (the shared helper only has 16 B).
__device__ __forceinline__ void cp_async_pred_load_64b(void* smem_ptr, const void* gmem_ptr,
                                                       bool predicate) {
  const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  const int src_in_bytes = predicate ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr),
               "n"(8), "r"(src_in_bytes));
}

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

// ---- mbarrier plumbing ----------------------------------------------------

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
// Arrive once every prior cp.async of this thread has completed (no pending
// count increment: the init count must include these arrivals).
__device__ __forceinline__ void mbar_cp_async_arrive(uint64_t* bar) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(a) : "memory");
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
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
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

template <typename T>
__global__ void __launch_bounds__(kW8A16Threads, 1)
    W8A16Fp8GemmWgmmaWsKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                              const float* __restrict__ scale, const T* __restrict__ bias,
                              T* __restrict__ out, int M, int N, int K) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using namespace cp_async;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) &
                                             ~static_cast<uintptr_t>(1023));
  uint8_t* a_base = smem;                                     // act stages (swizzled)
  uint8_t* w_base = smem + kW8A16Stages * kW8A16AStageBytes;  // weight stages (padded rows)
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(w_base + kW8A16Stages * kW8A16WStageBytes);
  uint64_t* empty_bar = full_bar + kW8A16Stages;
  using WTile = __nv_fp8_e4m3[kW8A16TileN][kW8A16WPad];
  WTile* w_s = reinterpret_cast<WTile*>(w_base);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wg = warp >> 2;  // 0 = producer, 1..2 = consumers
  const int n0 = blockIdx.x * kW8A16TileN;
  const int m0 = blockIdx.y * kW8A16TileM;
  const int num_slabs = K / kW8A16TileK;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kW8A16Stages; ++s) {
      mbar_init(&full_bar[s], 128);                     // one cp.async arrive per producer thread
      mbar_init(&empty_bar[s], kW8A16ConsumerThreads);  // every consumer thread releases
    }
  }
  __syncthreads();

  if (wg == 0) {
    // ------------------------------ producer ------------------------------
    setmaxnreg_dec<40>();
    constexpr int kCopies = 8;         // 1024 chunks of each operand / 128 threads
    constexpr int kRowStep = 128 / 8;  // 16
    const int cr = threadIdx.x / 8;
    const int cc = threadIdx.x % 8;
    bool a_ok[kCopies], w_ok[kCopies];
    const T* a_src[kCopies];
    const __nv_fp8_e4m3* w_src[kCopies];
    uint32_t a_dst_off[kCopies];
#pragma unroll
    for (int i = 0; i < kCopies; ++i) {
      const int r = cr + kRowStep * i;
      a_ok[i] = (m0 + r) < M;
      w_ok[i] = (n0 + r) < N;
      a_src[i] = act + static_cast<int64_t>(m0 + (a_ok[i] ? r : 0)) * K + cc * 8;
      w_src[i] = w + static_cast<int64_t>(n0 + (w_ok[i] ? r : 0)) * K + cc * 8;
      a_dst_off[i] = r * kW8A16ARowBytes + ((cc ^ (r & 7)) << 4);
    }
    for (int kt = 0; kt < num_slabs; ++kt) {
      const int stage = kt % kW8A16Stages;
      const uint32_t round = kt / kW8A16Stages;
      // A fresh barrier passes a wait on parity 1, so round 0 never blocks.
      mbar_wait(&empty_bar[stage], (round & 1) ^ 1);
      const int k0 = kt * kW8A16TileK;
      uint8_t* a_stage = a_base + stage * kW8A16AStageBytes;
#pragma unroll
      for (int i = 0; i < kCopies; ++i) {
        pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(
            reinterpret_cast<T*>(a_stage + a_dst_off[i]), a_src[i] + k0, a_ok[i]);
        cp_async_pred_load_64b(&w_s[stage][cr + kRowStep * i][cc * 8], w_src[i] + k0, w_ok[i]);
      }
      mbar_cp_async_arrive(&full_bar[stage]);
    }
    // Drain before exit so no copy is outstanding when the block retires.
    wait_group<0>();
  } else {
    // ------------------------------ consumers -----------------------------
    setmaxnreg_inc<232>();
    const int cwg = wg - 1;   // consumer warpgroup: n rows cwg*64 .. +63
    const int wq = warp & 3;  // warp within it: rows wq*16 .. +15 of those
    float d[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) d[i] = 0.f;

    const int a_r = cwg * 64 + wq * 16 + lane / 4;
    const int a_k = (lane % 4) * 2;

    for (int kt = 0; kt < num_slabs; ++kt) {
      const int stage = kt % kW8A16Stages;
      const uint32_t round = kt / kW8A16Stages;
      mbar_wait(&full_bar[stage], round & 1);
      fence_proxy_async_smem();  // cp.async writes -> visible to the wgmma async proxy
      const uint8_t* a_stage = a_base + stage * kW8A16AStageBytes;
#pragma unroll
      for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
        const __nv_fp8_e4m3* r0 = &w_s[stage][a_r][k16 + a_k];
        const __nv_fp8_e4m3* r1 = &w_s[stage][a_r + 8][k16 + a_k];
        uint32_t a_frag[4];
        a_frag[0] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0));
        a_frag[1] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1));
        a_frag[2] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r0 + 8));
        a_frag[3] = fp8x2_to_pair<T>(*reinterpret_cast<const uint16_t*>(r1 + 8));
        const uint64_t desc_b = make_smem_desc_sw128(a_stage + k16 * 2);
        wgmma_fence();
        wgmma_m64n128k16_rs<T>(d, a_frag, desc_b, 1);
      }
      wgmma_commit();
      // Keep this slab's group in flight; the previous slab's group is now
      // complete, so its stage can go back to the producer.
      wgmma_wait<1>();
      if (kt > 0) mbar_arrive(&empty_bar[(kt - 1) % kW8A16Stages]);
    }
    wgmma_wait<0>();
    if (num_slabs > 0) mbar_arrive(&empty_bar[(num_slabs - 1) % kW8A16Stages]);

    // Epilogue (rows are n, cols are m; see rung 8).
    const float sc = *scale;
    const int n_lo = n0 + cwg * 64 + wq * 16 + lane / 4;
    const int m_base = m0 + (lane % 4) * 2;
    float bias_lo = 0.f, bias_hi = 0.f;
    if (bias != nullptr) {
      if (n_lo < N) bias_lo = static_cast<float>(bias[n_lo]);
      if (n_lo + 8 < N) bias_hi = static_cast<float>(bias[n_lo + 8]);
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int m = m_base + j * 8;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int mm = m + q;
        if (mm >= M) continue;
        T* row = out + static_cast<int64_t>(mm) * N;
        if (n_lo < N) row[n_lo] = static_cast<T>(d[j * 4 + q] * sc + bias_lo);
        if (n_lo + 8 < N) row[n_lo + 8] = static_cast<T>(d[j * 4 + 2 + q] * sc + bias_hi);
      }
    }
  }
#else
  __trap();  // wgmma needs sm_90a
#endif
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  if (K % kW8A16TileK != 0) return cudaErrorInvalidValue;  // cp.async slabs must be whole
  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  if (tiles_m > 65535u) return cudaErrorInvalidValue;  // gridDim.y limit
  auto* kernel = W8A16Fp8GemmWgmmaWsKernel<T>;
  static bool attr_set = false;  // per instantiation; the attribute is per function
  if (!attr_set) {
    cudaError_t e =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kW8A16SmemBytes);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  const dim3 grid(ceil_div(N, kW8A16TileN), tiles_m);
  kernel<<<grid, kW8A16Threads, kW8A16SmemBytes, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
