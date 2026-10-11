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
// Rung 6 of the optimization ladder: mma.sync tensor cores.  Rungs 0-5 were
// SIMT fp32 FMA and ended at ~30 TFLOP/s with the FMA pipe at 88% issue.
// Here the 64x64x64 block tile is computed by 4 warps (32x32 each) with
// mma.sync.m16n8k16 (f16 or bf16 inputs, f32 accumulate): the A fragments
// come from the 16-bit act tile with ldmatrix.x4; the B fragments are built
// in registers from the fp8 weight tile, each thread reading the two fp8
// pairs its B slots need (k, k+1 and k+8, k+9 of column n) and converting
// them with one e4m3x2 -> half2 instruction each (plus half2 -> float2 ->
// bf162 when the activation is bf16, since e4m3 -> bf16 has no packed
// conversion).  The weight is still never widened in memory.  Same 3-stage
// cp.async ring as rung 5, 128 threads per block.  Requires K % 64 == 0.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "../cp_async.cuh"
#include "../mma.cuh"
#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileM = 64;
constexpr int kW8A16TileN = 64;
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 3;
constexpr int kW8A16Warps = 4;  // 2 (m) x 2 (n), 32x32 outputs each
constexpr int kW8A16Threads = 32 * kW8A16Warps;
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

// Two e4m3 values (packed in the low 16 bits) -> two T values packed in a
// 32-bit register, laid out as an mma B operand expects.
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

// mma.sync.m16n8k16 row.col, 16-bit inputs, fp32 accumulate (in place).
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

template <typename T>
__device__ __forceinline__ uint32_t pack_pair(float x, float y) {
  if constexpr (std::is_same_v<T, __half>) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
  } else {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<const uint32_t*>(&h);
  }
}

template <typename T>
__global__ void __launch_bounds__(kW8A16Threads)
    W8A16Fp8GemmMmaSyncKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                              const float* __restrict__ scale, const T* __restrict__ bias,
                              T* __restrict__ out, int M, int N, int K) {
  using namespace cp_async;
  __shared__ __align__(16) T a_s[kW8A16Stages][kW8A16TileM][kW8A16APad];
  __shared__ __align__(16) __nv_fp8_e4m3 w_s[kW8A16Stages][kW8A16TileN][kW8A16WPad];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wm = warp >> 1;  // warp row: rows wm*32 .. +31 of the tile
  const int wn = warp & 1;   // warp col: cols wn*32 .. +31 of the tile
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;

  // Copy assignment per slab: act = 64 rows x 128 B = 512 x 16 B, weight =
  // 64 rows x 64 B = 512 x 8 B; four copies per thread per operand (rows
  // r, r+16, r+32, r+48).  Rows past M / N copy zeros (cp.async zfill).
  const int a_r = threadIdx.x / 8;        // 0..15
  const int a_c = (threadIdx.x % 8) * 8;  // first of 8 bf16
  const int w_r = threadIdx.x / 8;        // 0..15
  const int w_c = (threadIdx.x % 8) * 8;  // first of 8 fp8
  bool a_ok[4], w_ok[4];
  const T* a_src[4];
  const __nv_fp8_e4m3* w_src[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a_ok[i] = (m0 + a_r + 16 * i) < M;
    w_ok[i] = (n0 + w_r + 16 * i) < N;
    a_src[i] = act + static_cast<int64_t>(m0 + (a_ok[i] ? a_r + 16 * i : 0)) * K + a_c;
    w_src[i] = w + static_cast<int64_t>(n0 + (w_ok[i] ? w_r + 16 * i : 0)) * K + w_c;
  }

  auto issue_slab = [&](int kt, int stage) {
    const int k0 = kt * kW8A16TileK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(
          &a_s[stage][a_r + 16 * i][a_c], a_src[i] + k0, a_ok[i]);
      cp_async_pred_load_64b(&w_s[stage][w_r + 16 * i][w_c], w_src[i] + k0, w_ok[i]);
    }
  };

  float acc[2][4][4];  // [m16 tile][n8 tile][c fragment]
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  const int num_slabs = K / kW8A16TileK;
#pragma unroll
  for (int s = 0; s < kW8A16Stages - 1; ++s) {
    if (s < num_slabs) issue_slab(s, s);
    commit_group();
  }

  // Per-lane fragment coordinates (m16n8k16 layouts).
  const int a_row = lane % 16;        // ldmatrix row within the m16 tile
  const int a_col = (lane / 16) * 8;  // ldmatrix column half (0 or 8) within k16
  const int b_n = lane / 4;           // column within the n8 tile
  const int b_k = (lane % 4) * 2;     // first k of the pair within k16

  for (int kt = 0; kt < num_slabs; ++kt) {
    wait_group<kW8A16Stages - 2>();
    __syncthreads();
    const int next = kt + kW8A16Stages - 1;
    if (next < num_slabs) issue_slab(next, next % kW8A16Stages);
    commit_group();

    const int stage = kt % kW8A16Stages;
#pragma unroll
    for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
      uint32_t a_frag[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        mma::ldmatrix_m8n8x4(a_frag[i], &a_s[stage][wm * 32 + i * 16 + a_row][k16 + a_col]);
      }
      uint32_t b_frag[4][2];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const __nv_fp8_e4m3* row = &w_s[stage][wn * 32 + j * 8 + b_n][k16 + b_k];
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(row);
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + 8);
        b_frag[j][0] = fp8x2_to_pair<T>(lo);
        b_frag[j][1] = fp8x2_to_pair<T>(hi);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_m16n8k16<T>(acc[i][j], a_frag[i], b_frag[j]);
    }
  }
  wait_group<0>();

  // Epilogue: c[0],c[1] at (row g, cols 2t, 2t+1); c[2],c[3] at row g+8.
  const float s = *scale;
  const int g = lane / 4;
  const int t2 = (lane % 4) * 2;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int n = n0 + wn * 32 + j * 8 + t2;
    if (n >= N) continue;  // N % 2 == 0 is not assumed below for n+1
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr) {
      b0 = static_cast<float>(bias[n]);
      if (n + 1 < N) b1 = static_cast<float>(bias[n + 1]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + wm * 32 + i * 16 + g + h * 8;
        if (m >= M) continue;
        const float v0 = acc[i][j][2 * h] * s + b0;
        const float v1 = acc[i][j][2 * h + 1] * s + b1;
        T* dst = out + static_cast<int64_t>(m) * N + n;
        if (n + 1 < N) {
          *reinterpret_cast<uint32_t*>(dst) = pack_pair<T>(v0, v1);
        } else {
          dst[0] = static_cast<T>(v0);
        }
      }
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
  W8A16Fp8GemmMmaSyncKernel<T>
      <<<grid, kW8A16Threads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
