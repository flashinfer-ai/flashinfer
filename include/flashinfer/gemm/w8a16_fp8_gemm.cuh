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
// Rung 7 of the optimization ladder: 128x128 block tile.  Rung 6 (mma.sync,
// 64x64 tile) ran at ~200 TFLOP/s with ncu showing L2 at 83% and DRAM at
// 49% of peak while the tensor pipe idled: a 64x64 tile re-reads each A and
// B element from L2 once per 64 outputs.  Here the block tile grows to
// 128x128 (8 warps as 2 x 4, 64x32 outputs each, i.e. 4 m16 x 4 n8 mma
// tiles per warp per k16), halving the L2 traffic per FLOP.  The 3-stage
// ring grows to 83 KB and moves to dynamic shared memory (opt-in above
// 48 KB; 2 blocks per SM).  Fragment construction, cp.async pipeline and
// the epilogue are unchanged from rung 6.  Requires K % 64 == 0.

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

constexpr int kW8A16TileM = 128;
constexpr int kW8A16TileN = 128;
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Stages = 3;
constexpr int kW8A16WarpsM = 2;                              // warps along m, 64 rows each
constexpr int kW8A16WarpsN = 4;                              // warps along n, 32 cols each
constexpr int kW8A16WarpTileM = kW8A16TileM / kW8A16WarpsM;  // 64 = 4 m16 tiles
constexpr int kW8A16WarpTileN = kW8A16TileN / kW8A16WarpsN;  // 32 = 4 n8 tiles
constexpr int kW8A16MTiles = kW8A16WarpTileM / 16;
constexpr int kW8A16NTiles = kW8A16WarpTileN / 8;
constexpr int kW8A16Warps = kW8A16WarpsM * kW8A16WarpsN;
constexpr int kW8A16Threads = 32 * kW8A16Warps;
constexpr int kW8A16APad = kW8A16TileK + 8;  // 72 bf16 = 144 B rows (16 B aligned, 36 words)
constexpr int kW8A16WPad = kW8A16TileK + 8;  // 72 fp8  =  72 B rows (8 B aligned, 18 words)
constexpr int kW8A16AStageBytes = kW8A16TileM * kW8A16APad * 2;
constexpr int kW8A16WStageBytes = kW8A16TileN * kW8A16WPad;
constexpr int kW8A16SmemBytes = kW8A16Stages * (kW8A16AStageBytes + kW8A16WStageBytes);

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
    W8A16Fp8GemmMmaSync128Kernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                                 const float* __restrict__ scale, const T* __restrict__ bias,
                                 T* __restrict__ out, int M, int N, int K) {
  using namespace cp_async;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  using ATile = T[kW8A16TileM][kW8A16APad];
  using WTile = __nv_fp8_e4m3[kW8A16TileN][kW8A16WPad];
  ATile* a_s = reinterpret_cast<ATile*>(smem_raw);
  WTile* w_s = reinterpret_cast<WTile*>(smem_raw + kW8A16Stages * kW8A16AStageBytes);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / kW8A16WarpsN;  // rows wm*64 .. +63 of the tile
  const int wn = warp % kW8A16WarpsN;  // cols wn*32 .. +31 of the tile
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;

  // Copy assignment per slab: act = 128 rows x 128 B = 1024 x 16 B, weight =
  // 128 rows x 64 B = 1024 x 8 B; four copies per thread per operand (rows
  // r, r+32, r+64, r+96).  Rows past M / N copy zeros (cp.async zfill).
  constexpr int kCopies = 4;
  constexpr int kRowStep = kW8A16Threads / 8;  // 32
  const int a_r = threadIdx.x / 8;             // 0..31
  const int a_c = (threadIdx.x % 8) * 8;       // first of 8 bf16
  const int w_r = threadIdx.x / 8;             // 0..31
  const int w_c = (threadIdx.x % 8) * 8;       // first of 8 fp8
  bool a_ok[kCopies], w_ok[kCopies];
  const T* a_src[kCopies];
  const __nv_fp8_e4m3* w_src[kCopies];
#pragma unroll
  for (int i = 0; i < kCopies; ++i) {
    a_ok[i] = (m0 + a_r + kRowStep * i) < M;
    w_ok[i] = (n0 + w_r + kRowStep * i) < N;
    a_src[i] = act + static_cast<int64_t>(m0 + (a_ok[i] ? a_r + kRowStep * i : 0)) * K + a_c;
    w_src[i] = w + static_cast<int64_t>(n0 + (w_ok[i] ? w_r + kRowStep * i : 0)) * K + w_c;
  }

  auto issue_slab = [&](int kt, int stage) {
    const int k0 = kt * kW8A16TileK;
#pragma unroll
    for (int i = 0; i < kCopies; ++i) {
      pred_load<128, PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(
          &a_s[stage][a_r + kRowStep * i][a_c], a_src[i] + k0, a_ok[i]);
      cp_async_pred_load_64b(&w_s[stage][w_r + kRowStep * i][w_c], w_src[i] + k0, w_ok[i]);
    }
  };

  float acc[kW8A16MTiles][kW8A16NTiles][4];
#pragma unroll
  for (int i = 0; i < kW8A16MTiles; ++i)
#pragma unroll
    for (int j = 0; j < kW8A16NTiles; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  const int num_slabs = K / kW8A16TileK;
#pragma unroll
  for (int s = 0; s < kW8A16Stages - 1; ++s) {
    if (s < num_slabs) issue_slab(s, s);
    commit_group();
  }

  const int a_row = lane % 16;
  const int a_col = (lane / 16) * 8;
  const int b_n = lane / 4;
  const int b_k = (lane % 4) * 2;

  for (int kt = 0; kt < num_slabs; ++kt) {
    wait_group<kW8A16Stages - 2>();
    __syncthreads();
    const int next = kt + kW8A16Stages - 1;
    if (next < num_slabs) issue_slab(next, next % kW8A16Stages);
    commit_group();

    const int stage = kt % kW8A16Stages;
#pragma unroll
    for (int k16 = 0; k16 < kW8A16TileK; k16 += 16) {
      uint32_t a_frag[kW8A16MTiles][4];
#pragma unroll
      for (int i = 0; i < kW8A16MTiles; ++i) {
        mma::ldmatrix_m8n8x4(a_frag[i],
                             &a_s[stage][wm * kW8A16WarpTileM + i * 16 + a_row][k16 + a_col]);
      }
      uint32_t b_frag[kW8A16NTiles][2];
#pragma unroll
      for (int j = 0; j < kW8A16NTiles; ++j) {
        const __nv_fp8_e4m3* row = &w_s[stage][wn * kW8A16WarpTileN + j * 8 + b_n][k16 + b_k];
        const uint32_t lo = *reinterpret_cast<const uint16_t*>(row);
        const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + 8);
        b_frag[j][0] = fp8x2_to_pair<T>(lo);
        b_frag[j][1] = fp8x2_to_pair<T>(hi);
      }
#pragma unroll
      for (int i = 0; i < kW8A16MTiles; ++i)
#pragma unroll
        for (int j = 0; j < kW8A16NTiles; ++j) mma_m16n8k16<T>(acc[i][j], a_frag[i], b_frag[j]);
    }
  }
  wait_group<0>();

  const float s = *scale;
  const int g = lane / 4;
  const int t2 = (lane % 4) * 2;
#pragma unroll
  for (int j = 0; j < kW8A16NTiles; ++j) {
    const int n = n0 + wn * kW8A16WarpTileN + j * 8 + t2;
    if (n >= N) continue;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr) {
      b0 = static_cast<float>(bias[n]);
      if (n + 1 < N) b1 = static_cast<float>(bias[n + 1]);
    }
#pragma unroll
    for (int i = 0; i < kW8A16MTiles; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + wm * kW8A16WarpTileM + i * 16 + g + h * 8;
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
  auto* kernel = W8A16Fp8GemmMmaSync128Kernel<T>;
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
