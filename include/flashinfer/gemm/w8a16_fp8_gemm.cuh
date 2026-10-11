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
// Rung 5 of the optimization ladder: cp.async multi-stage pipeline.  Rung 4
// still copied each K slab synchronously (vector loads into registers,
// store to smem, syncthreads, compute, syncthreads), so every slab's global
// latency sat on the critical path and ncu showed the fp32 pipe at ~82%
// issue with 37% occupancy.  Here the slab copy is a 16-byte cp.async into
// a 3-stage shared-memory ring: two slabs are in flight while the third is
// consumed, with cp.async.wait_group + one __syncthreads per slab.  BK grows
// to 64: act moves as two 16 B cp.async per thread per slab, the weight as
// two 8 B ones (8 B keeps the 72-byte, conflict-free weight rows; a 16 B
// copy needs 16 B-aligned rows and those conflict 2-way).  Tile, register
// block, packed conversion and SIMT FMA are unchanged.  Requires K % 64 == 0.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "../cp_async.cuh"
#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileM = 64;
constexpr int kW8A16TileN = 64;
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 3;
constexpr int kW8A16ThreadM = 4;  // outputs per thread along m
constexpr int kW8A16ThreadN = 4;  // outputs per thread along n
constexpr int kW8A16Threads = (kW8A16TileM / kW8A16ThreadM) * (kW8A16TileN / kW8A16ThreadN);
constexpr int kW8A16APad = kW8A16TileK + 8;  // 72 bf16 = 144 B rows (16 B aligned, 36 words)
constexpr int kW8A16WPad = kW8A16TileK + 8;  // 72 fp8  =  72 B rows (8 B aligned, 18 words)

// 8-byte predicated cp.async with zero fill (the shared helper only has 16 B).
__device__ __forceinline__ void cp_async_pred_load_64b(void* smem_ptr, const void* gmem_ptr,
                                                       bool predicate) {
  const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  const int src_in_bytes = predicate ? 8 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr), "l"(gmem_ptr),
               "n"(8), "r"(src_in_bytes));
}

// 4 e4m3 bytes -> 4 floats, two packed conversions.
__device__ __forceinline__ void fp8x4_to_float4(uint32_t packed, float (&f)[4]) {
  const __half2 lo = __half2(
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3));
  const __half2 hi = __half2(
      __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3));
  const float2 a = __half22float2(lo);
  const float2 b = __half22float2(hi);
  f[0] = a.x;
  f[1] = a.y;
  f[2] = b.x;
  f[3] = b.y;
}

template <typename T>
__device__ __forceinline__ float2 pair_to_float2(uint32_t packed);
template <>
__device__ __forceinline__ float2 pair_to_float2<__half>(uint32_t packed) {
  __half2 h = *reinterpret_cast<__half2*>(&packed);
  return __half22float2(h);
}
template <>
__device__ __forceinline__ float2 pair_to_float2<__nv_bfloat16>(uint32_t packed) {
  __nv_bfloat162 h = *reinterpret_cast<__nv_bfloat162*>(&packed);
  return __bfloat1622float2(h);
}

template <typename T>
__global__ void __launch_bounds__(kW8A16Threads)
    W8A16Fp8GemmCpAsyncKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                              const float* __restrict__ scale, const T* __restrict__ bias,
                              T* __restrict__ out, int M, int N, int K) {
  using namespace cp_async;
  constexpr int kThreadsN = kW8A16TileN / kW8A16ThreadN;  // 16
  constexpr int kThreadsM = kW8A16TileM / kW8A16ThreadM;  // 16
  __shared__ __align__(16) T a_s[kW8A16Stages][kW8A16TileM][kW8A16APad];
  __shared__ __align__(16) __nv_fp8_e4m3 w_s[kW8A16Stages][kW8A16TileN][kW8A16WPad];

  const int tx = threadIdx.x % kThreadsN;
  const int ty = threadIdx.x / kThreadsN;
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;

  // Copy assignment per slab: act = 64 rows x 128 B = 512 x 16 B, weight =
  // 64 rows x 64 B = 512 x 8 B; two copies per thread per operand (rows r
  // and r + 32).  Rows past M / N copy zeros (cp.async zfill).
  const int a_r = threadIdx.x / 8;        // 0..31
  const int a_c = (threadIdx.x % 8) * 8;  // first of 8 bf16
  const int w_r = threadIdx.x / 8;        // 0..31
  const int w_c = (threadIdx.x % 8) * 8;  // first of 8 fp8
  const bool a_ok0 = (m0 + a_r) < M;
  const bool a_ok1 = (m0 + a_r + 32) < M;
  const bool w_ok0 = (n0 + w_r) < N;
  const bool w_ok1 = (n0 + w_r + 32) < N;
  const T* a_src0 = act + static_cast<int64_t>(m0 + (a_ok0 ? a_r : 0)) * K + a_c;
  const T* a_src1 = act + static_cast<int64_t>(m0 + (a_ok1 ? a_r + 32 : 0)) * K + a_c;
  const __nv_fp8_e4m3* w_src0 = w + static_cast<int64_t>(n0 + (w_ok0 ? w_r : 0)) * K + w_c;
  const __nv_fp8_e4m3* w_src1 = w + static_cast<int64_t>(n0 + (w_ok1 ? w_r + 32 : 0)) * K + w_c;

  auto issue_slab = [&](int kt, int stage) {
    const int k0 = kt * kW8A16TileK;
    pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(&a_s[stage][a_r][a_c],
                                                                            a_src0 + k0, a_ok0);
    pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(
        &a_s[stage][a_r + 32][a_c], a_src1 + k0, a_ok1);
    cp_async_pred_load_64b(&w_s[stage][w_r][w_c], w_src0 + k0, w_ok0);
    cp_async_pred_load_64b(&w_s[stage][w_r + 32][w_c], w_src1 + k0, w_ok1);
  };

  float acc[kW8A16ThreadM][kW8A16ThreadN];
#pragma unroll
  for (int i = 0; i < kW8A16ThreadM; ++i)
#pragma unroll
    for (int j = 0; j < kW8A16ThreadN; ++j) acc[i][j] = 0.f;

  const int num_slabs = K / kW8A16TileK;
  // Prologue: slabs 0 .. S-2 in flight.
#pragma unroll
  for (int s = 0; s < kW8A16Stages - 1; ++s) {
    if (s < num_slabs) issue_slab(s, s);
    commit_group();
  }

  for (int kt = 0; kt < num_slabs; ++kt) {
    // Slab kt has landed once at most S-2 newer groups are still pending.
    wait_group<kW8A16Stages - 2>();
    __syncthreads();  // also: everyone is done reading stage (kt-1) % S
    const int next = kt + kW8A16Stages - 1;
    if (next < num_slabs) issue_slab(next, next % kW8A16Stages);
    commit_group();  // always commit so the group count stays uniform

    const int stage = kt % kW8A16Stages;
#pragma unroll
    for (int kk = 0; kk < kW8A16TileK; kk += 4) {
      float a_frag[kW8A16ThreadM][4];
      float w_frag[kW8A16ThreadN][4];
#pragma unroll
      for (int i = 0; i < kW8A16ThreadM; ++i) {
        const uint2 p = *reinterpret_cast<const uint2*>(&a_s[stage][ty + i * kThreadsM][kk]);
        const float2 f0 = pair_to_float2<T>(p.x);
        const float2 f1 = pair_to_float2<T>(p.y);
        a_frag[i][0] = f0.x;
        a_frag[i][1] = f0.y;
        a_frag[i][2] = f1.x;
        a_frag[i][3] = f1.y;
      }
#pragma unroll
      for (int j = 0; j < kW8A16ThreadN; ++j) {
        const uint32_t p = *reinterpret_cast<const uint32_t*>(&w_s[stage][tx + j * kThreadsN][kk]);
        fp8x4_to_float4(p, w_frag[j]);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int i = 0; i < kW8A16ThreadM; ++i)
#pragma unroll
          for (int j = 0; j < kW8A16ThreadN; ++j)
            acc[i][j] = fmaf(a_frag[i][q], w_frag[j][q], acc[i][j]);
    }
  }
  wait_group<0>();

  const float s = *scale;
#pragma unroll
  for (int j = 0; j < kW8A16ThreadN; ++j) {
    const int n = n0 + tx + j * kThreadsN;
    if (n >= N) continue;
    const float b = bias != nullptr ? static_cast<float>(bias[n]) : 0.f;
#pragma unroll
    for (int i = 0; i < kW8A16ThreadM; ++i) {
      const int m = m0 + ty + i * kThreadsM;
      if (m < M) out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc[i][j] * s + b);
    }
  }
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  if (K % kW8A16TileK != 0) return cudaErrorInvalidValue;  // cp.async slabs must be whole
  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  if (tiles_m > 65535u) return cudaErrorInvalidValue;  // gridDim.y limit
  const dim3 grid(ceil_div(N, kW8A16TileN), tiles_m);
  W8A16Fp8GemmCpAsyncKernel<T>
      <<<grid, kW8A16Threads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
