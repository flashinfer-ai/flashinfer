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
// Rung 4 of the optimization ladder: vectorized loads and packed fp8
// conversion.  Rung 3 (64x64 tile, 4x4 register block) was still issue
// bound (ncu: 86% issue slots busy, IPC 3.4, 72% L1/TEX pipe) because every
// operand element cost one 1-byte / 2-byte LDG, one 1-byte / 2-byte STS, one
// LDS and one scalar conversion.  Here the K slab is moved with 16-byte
// (act) and 8-byte (weight) global loads straight into shared memory, the
// inner loop reads 4 consecutive k per row as one 32-bit word each (2 words
// for act), and the 4 fp8 are converted with two packed e4m3x2 -> half2
// instructions instead of four scalar ones.  Smem rows are padded to keep
// the 16 B / 8 B vector alignment (40 bf16 = 20 words, 40 fp8 = 10 words,
// both conflict-free for rows 16 apart).  Still SIMT fp32 FMA, synchronous
// loads, one output tile per block.  Requires K % 32 == 0.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileM = 64;
constexpr int kW8A16TileN = 64;
constexpr int kW8A16TileK = 32;
constexpr int kW8A16ThreadM = 4;  // outputs per thread along m
constexpr int kW8A16ThreadN = 4;  // outputs per thread along n
constexpr int kW8A16Threads = (kW8A16TileM / kW8A16ThreadM) * (kW8A16TileN / kW8A16ThreadN);
constexpr int kW8A16APad = kW8A16TileK + 8;  // 40 bf16 = 80 B rows (16 B aligned)
constexpr int kW8A16WPad = kW8A16TileK + 8;  // 40 fp8  = 40 B rows (8 B aligned)

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
    W8A16Fp8GemmVecLoadKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                              const float* __restrict__ scale, const T* __restrict__ bias,
                              T* __restrict__ out, int M, int N, int K) {
  constexpr int kThreadsN = kW8A16TileN / kW8A16ThreadN;  // 16
  constexpr int kThreadsM = kW8A16TileM / kW8A16ThreadM;  // 16
  __shared__ __align__(16) T a_s[kW8A16TileM][kW8A16APad];
  __shared__ __align__(16) __nv_fp8_e4m3 w_s[kW8A16TileN][kW8A16WPad];

  const int tx = threadIdx.x % kThreadsN;
  const int ty = threadIdx.x / kThreadsN;
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;

  // Staging assignment: act slab = 64 rows x 64 B = 256 x 16 B, one uint4 per
  // thread; weight slab = 64 rows x 32 B = 256 x 8 B, one uint2 per thread.
  const int a_r = threadIdx.x / 4;        // row
  const int a_c = (threadIdx.x % 4) * 8;  // first of 8 bf16
  const int w_r = threadIdx.x / 4;
  const int w_c = (threadIdx.x % 4) * 8;  // first of 8 fp8
  const bool a_ok = (m0 + a_r) < M;
  const bool w_ok = (n0 + w_r) < N;
  const T* a_src = act + static_cast<int64_t>(m0 + (a_ok ? a_r : 0)) * K + a_c;
  const __nv_fp8_e4m3* w_src = w + static_cast<int64_t>(n0 + (w_ok ? w_r : 0)) * K + w_c;

  float acc[kW8A16ThreadM][kW8A16ThreadN];
#pragma unroll
  for (int i = 0; i < kW8A16ThreadM; ++i)
#pragma unroll
    for (int j = 0; j < kW8A16ThreadN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += kW8A16TileK) {
    uint4 a_v = a_ok ? *reinterpret_cast<const uint4*>(a_src + k0) : make_uint4(0, 0, 0, 0);
    uint2 w_v = w_ok ? *reinterpret_cast<const uint2*>(w_src + k0) : make_uint2(0, 0);
    *reinterpret_cast<uint4*>(&a_s[a_r][a_c]) = a_v;
    *reinterpret_cast<uint2*>(&w_s[w_r][w_c]) = w_v;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kW8A16TileK; kk += 4) {
      float a_frag[kW8A16ThreadM][4];
      float w_frag[kW8A16ThreadN][4];
#pragma unroll
      for (int i = 0; i < kW8A16ThreadM; ++i) {
        const uint2 p = *reinterpret_cast<const uint2*>(&a_s[ty + i * kThreadsM][kk]);
        const float2 f0 = pair_to_float2<T>(p.x);
        const float2 f1 = pair_to_float2<T>(p.y);
        a_frag[i][0] = f0.x;
        a_frag[i][1] = f0.y;
        a_frag[i][2] = f1.x;
        a_frag[i][3] = f1.y;
      }
#pragma unroll
      for (int j = 0; j < kW8A16ThreadN; ++j) {
        const uint32_t p = *reinterpret_cast<const uint32_t*>(&w_s[tx + j * kThreadsN][kk]);
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
    __syncthreads();
  }

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
  if (K % kW8A16TileK != 0) return cudaErrorInvalidValue;  // vector loads need whole slabs
  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  if (tiles_m > 65535u) return cudaErrorInvalidValue;  // gridDim.y limit
  const dim3 grid(ceil_div(N, kW8A16TileN), tiles_m);
  W8A16Fp8GemmVecLoadKernel<T>
      <<<grid, kW8A16Threads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
