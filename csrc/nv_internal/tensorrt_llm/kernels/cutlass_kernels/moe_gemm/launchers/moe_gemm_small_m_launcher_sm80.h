/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../../include/moe_gemm_kernels.h"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/mma.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::cutlass_kernels_oss {

// The small-M grouped GEMM of the SM80 FP4 MoE on the operands of its CUTLASS kernels: weights
// prepared as [E][N/4][K/64][4 columns][32 B] (word w = h*4 + j of a column's 32 B holds the
// nibbles k = 32h + 2j + {0,1,8,9,16,17,24,25}), block scales [E][K/16][N] BF16, BF16 activation
// rows (restored, or the token rows read in place). A CTA computes up to 8 rows of one expert x 128
// columns over K in stages of 128 k with 256 threads: 2 N-warps of 64 columns x 4 K-warps of 32 k.
// The weights take mma.m16n8k16's A slot (16 rows = 16 columns) and the activations its B slot, so
// one ldmatrix.x4 of weights feeds two HMMA; the K-warps' partials are summed through shared memory
// after the loop. Output rows are scaled by their row scale, else by expert_scales[e] when given,
// else by their expert's alpha. Grid (N/128, m-blocks): an m-block is (expert, first row, up to 8
// rows), numbered expert by expert from the experts' row ranges; each CTA finds its own from them,
// so no launch or workspace is added.
namespace sm80_fp4_small_m {

constexpr int kThreads = 256, kTileN = 128, kTileK = 128, kMaxRows = 8, kStages = 4;
constexpr int kActPitch = 272;                     // 256 B row + 16 B against ldmatrix conflicts
constexpr int kWBytes = kTileN * kTileK / 2;       // 32 column groups x 2 K/64 lines x 128 B
constexpr int kSBytes = kTileK / 16 * kTileN * 2;  // 8 k16 rows x 128 columns
constexpr int kABytes = kMaxRows * kActPitch;      // 8 rows
constexpr int kStageBytes = kWBytes + kSBytes + kABytes;  // 12416
constexpr int kScanBytes = 64;                            // the m-block scan's shared words
constexpr int kSmemBytes = kStages * kStageBytes + kScanBytes;

// flashinfer::e2m1x8_to_bf16x8 with its BF16 magnitude tables passed in: high and low bytes of
// +{0.0, 0.5, 1.0, 1.5} (hi0, lo0) and of +{2.0, 3.0, 4.0, 6.0} (hi1, lo1). `prmt` reads both table
// words from registers, so ptxas rebuilds immediate tables before each conversion in a loop; the
// main loop builds them once, OR'ed with a zero the compiler cannot prove (derived from a kernel
// argument), to keep the four words in registers instead.
struct E2m1Bf16Lut {
  uint32_t hi0, hi1, lo0, lo1;
};

__device__ __forceinline__ E2m1Bf16Lut e2m1_bf16_lut(uint32_t zero) {
  return {0x3F3F3F00u | zero, 0x40404040u | zero, 0xC0800000u | zero, 0xC0804000u | zero};
}

__device__ __forceinline__ void e2m1x8_to_bf16x8(uint32_t packed, uint32_t* d,
                                                 E2m1Bf16Lut const& t) {
  uint32_t const mag = packed & 0x77777777u;
  uint32_t const mag_a = mag, mag_b = mag >> 16;
  uint32_t hi_a = __byte_perm(t.hi0, t.hi1, mag_a);
  uint32_t hi_b = __byte_perm(t.hi0, t.hi1, mag_b);
  uint32_t const lo_a = __byte_perm(t.lo0, t.lo1, mag_a);
  uint32_t const lo_b = __byte_perm(t.lo0, t.lo1, mag_b);
  hi_a |= flashinfer::e2m1_sign_bytes(packed, 0xD9C8u);
  hi_b |= flashinfer::e2m1_sign_bytes(packed, 0xFBEAu);
  d[0] = __byte_perm(lo_a, hi_a, 0x5140u);
  d[1] = __byte_perm(lo_a, hi_a, 0x7362u);
  d[2] = __byte_perm(lo_b, hi_b, 0x5140u);
  d[3] = __byte_perm(lo_b, hi_b, 0x7362u);
}

__device__ __forceinline__ void mma_bf16(float* d, uint32_t const* a, uint32_t const* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// a x {s, s} for two BF16 pairs, exact: an E2M1 value times a block scale (E4M3, or a power of two
// for MX) fits BF16's 8-bit significand. The pair is built inside the asm so ptxas folds it into
// HFMA2's operand swizzle instead of spending an instruction per scale on the broadcast.
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint16_t s) {
  uint32_t r;
  asm("{\n.reg .b32 s2;\nmov.b32 s2, {%2, %2};\nfma.rn.bf16x2 %0, %1, s2, %3;\n}\n"
      : "=r"(r)
      : "r"(a), "h"(s), "r"(0x80008000u));
  return r;
}

// Byte offset of half h of a column's 32 B in a stage's weight buffer: column group G (4 columns),
// K/64 line tl, column n4 of the group. The 16-B units of a 128-B bank row are XOR-swizzled with
// G & 1 so the eight rows of one ldmatrix matrix (8 consecutive columns = two groups) take eight
// different units.
__device__ __forceinline__ uint32_t w_offset(int G, int tl, int n4, int h) {
  return G * 256 + tl * 128 + (((n4 * 2 + h) ^ (G & 1)) << 4);
}

// The CTA's m-block from the experts' row ranges: the blocks of up to 8 rows of expert 0, then of
// expert 1, ... Each thread owns consecutive experts; the block counts are scanned over the threads
// and the owner of the block writes it to s_scan[8..10]. False past the last block.
__device__ __forceinline__ bool find_block(int64_t const* total_tokens_including_expert,
                                           int num_experts, int block, int& expert, int& row0,
                                           int& rows, int* s_scan) {
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const chunk = (num_experts + kThreads - 1) / kThreads;
  int const e0 = tid * chunk, e1 = min(e0 + chunk, num_experts);
  int count = 0;
  int64_t const start = e0 > 0 && e0 < num_experts ? total_tokens_including_expert[e0 - 1] : 0;
  int64_t prev = start;
  for (int e = e0; e < e1; ++e) {
    int64_t const end = total_tokens_including_expert[e];
    count += static_cast<int>((end - prev + kMaxRows - 1) / kMaxRows);
    prev = end;
  }
  int inclusive = count;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int const v = __shfl_up_sync(0xffffffffu, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  if (lane == 31) s_scan[warp] = inclusive;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) {
    total += s_scan[w];
    if (w < warp) before += s_scan[w];
  }
  if (block >= total) return false;
  int const first = before + inclusive - count;
  if (block >= first && block < first + count) {
    int b = block - first;
    prev = start;
    for (int e = e0; e < e1; ++e) {
      int64_t const end = total_tokens_including_expert[e];
      int const blocks = static_cast<int>((end - prev + kMaxRows - 1) / kMaxRows);
      if (b < blocks) {
        s_scan[8] = e;
        s_scan[9] = static_cast<int>(prev) + b * kMaxRows;
        s_scan[10] = min(kMaxRows, static_cast<int>(end - prev) - b * kMaxRows);
        break;
      }
      b -= blocks;
      prev = end;
    }
  }
  __syncthreads();
  expert = s_scan[8];
  row0 = s_scan[9];
  rows = s_scan[10];
  return true;
}

template <int kNumStages>
__global__ void __launch_bounds__(kThreads, 1)
    kernel(uint8_t const* __restrict__ w, uint16_t const* __restrict__ s,
           float const* __restrict__ row_scales, float const* const* __restrict__ alpha,
           float const* __restrict__ expert_scales, __nv_bfloat16 const* __restrict__ act,
           int const* __restrict__ act_rows, int act_tokens,
           int64_t const* __restrict__ total_tokens_including_expert,
           __nv_bfloat16* __restrict__ out, int N, int K, int num_experts) {
  extern __shared__ __align__(128) uint8_t smem[];
  int* s_scan = reinterpret_cast<int*>(smem + kNumStages * kStageBytes);
  int e, row0, rows;
  if (!find_block(total_tokens_including_expert, num_experts, blockIdx.y, e, row0, rows, s_scan)) {
    return;
  }
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const warp_n = warp & 1, warp_k = warp >> 1, tl = warp_k >> 1, h = warp_k & 1;
  int const g = lane >> 2, c = lane & 3;
  int const ntile = blockIdx.x;
  int const Tn = K / 64, nstage = K / kTileK;

  // Stage loads, 16 B per cp.async: weights 512 chunks (two per thread), scales 128 (threads <
  // 128), activations 128 (threads >= 128; rows past `rows` zero-filled). Per stage the sources
  // advance by two K/64 lines (256 B), eight scale rows and 256 B of a row.
  uint8_t const* wsrc[2];
  uint32_t wdst[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int const q = tid + kThreads * i, G = q >> 4, u = q & 15, ltl = u >> 3, n4 = (u >> 1) & 3,
              lh = u & 1;
    wsrc[i] = w + (static_cast<size_t>(e) * (N / 4) + ntile * 32 + G) * Tn * 128 + ltl * 128 +
              n4 * 32 + lh * 16;
    wdst[i] = w_offset(G, ltl, n4, lh);
  }
  uint8_t const* osrc;
  uint32_t odst;
  bool opred;
  size_t ostep;
  if (tid < 128) {
    int const k16 = tid >> 4, ch = tid & 15;
    osrc = reinterpret_cast<uint8_t const*>(s + (static_cast<size_t>(e) * (K / 16) + k16) * N +
                                            ntile * kTileN + ch * 8);
    odst = kWBytes + k16 * 256 + ch * 16;
    opred = true;
    ostep = static_cast<size_t>(8) * N * 2;
  } else {
    int const r = (tid - 128) >> 4, ch = tid & 15, row = row0 + (r < rows ? r : 0);
    osrc = reinterpret_cast<uint8_t const*>(
        act + static_cast<size_t>(act_rows ? act_rows[row] % act_tokens : row) * K + ch * 8);
    odst = kWBytes + kSBytes + r * kActPitch + ch * 16;
    opred = r < rows;
    ostep = 256;
  }
  using namespace flashinfer::cp_async;
  auto load_stage = [&](int st, int buf) {
    uint8_t* const stage = smem + buf * kStageBytes;
    load_128b<PrefetchMode::kNoPrefetch>(stage + wdst[0], wsrc[0] + static_cast<size_t>(st) * 256);
    load_128b<PrefetchMode::kNoPrefetch>(stage + wdst[1], wsrc[1] + static_cast<size_t>(st) * 256);
    pred_load_128b<PrefetchMode::kNoPrefetch, SharedMemFillMode::kFillZero>(
        stage + odst, osrc + static_cast<size_t>(st) * ostep, opred);
  };

  // Compute-side offsets within a stage buffer. Weights: two ldmatrix.x4; matrix m of the i2-th
  // covers the eight columns warp_n*64 + i2*32 + m*8 + (0..7), half h of K/64 line tl, so register
  // i = i2*4 + m holds column (8i + g)'s word c: the nibbles k = 2c + {0,1,8,9} and + 16 of this
  // warp's 32-k slice. Activations: the four matrices are the slice's 8-k segments, rows the
  // tokens. Scales: rows k16 = 4tl + 2h + {0,1}, column 8i + g.
  uint32_t woff[2];
#pragma unroll
  for (int i2 = 0; i2 < 2; ++i2) {
    int const n = warp_n * 64 + i2 * 32 + (lane >> 3) * 8 + (lane & 7);
    woff[i2] = w_offset(n >> 2, tl, n & 3, h);
  }
  uint32_t const aoff =
      kWBytes + kSBytes + (lane & 7) * kActPitch + (tl * 64 + h * 32 + (lane >> 3) * 8) * 2;
  uint32_t const soff = kWBytes + (tl * 4 + h * 2) * 256 + (warp_n * 64 + g) * 2;

  // The E2M1 tables stay in registers over the main loop (N is never negative).
  E2m1Bf16Lut const lut = e2m1_bf16_lut(N < 0);
  float acc[4][4] = {};
#pragma unroll
  for (int st = 0; st < kNumStages - 1; ++st) {
    if (st < nstage) load_stage(st, st);
    commit_group();
  }
  for (int kt = 0; kt < nstage; ++kt) {
    wait_group<kNumStages - 2>();
    __syncthreads();
    int const nx = kt + kNumStages - 1;
    if (nx < nstage) load_stage(nx, nx % kNumStages);
    commit_group();

    int const buf = kt % kNumStages;
    uint8_t const* sb = smem + buf * kStageBytes;
    uint32_t b[4], wr[8];
    uint16_t sc[2][8];
    flashinfer::mma::ldmatrix_m8n8x4(b, sb + aoff);
    flashinfer::mma::ldmatrix_m8n8x4(wr, sb + woff[0]);
    flashinfer::mma::ldmatrix_m8n8x4(wr + 4, sb + woff[1]);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        sc[ks][i] = *reinterpret_cast<uint16_t const*>(sb + soff + ks * 256 + i * 16);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {  // columns 16j + g (register 2j) and 16j + 8 + g (2j + 1)
      uint32_t d0[4], d1[4], a[4];
      e2m1x8_to_bf16x8(wr[2 * j], d0, lut);
      e2m1x8_to_bf16x8(wr[2 * j + 1], d1, lut);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {  // k16 steps: d[2ks] = k 2c + {0,1}, d[2ks + 1] = 2c + {8,9}
        a[0] = mul_bf16x2(d0[2 * ks], sc[ks][2 * j]);
        a[1] = mul_bf16x2(d1[2 * ks], sc[ks][2 * j + 1]);
        a[2] = mul_bf16x2(d0[2 * ks + 1], sc[ks][2 * j]);
        a[3] = mul_bf16x2(d1[2 * ks + 1], sc[ks][2 * j + 1]);
        mma_bf16(acc[j], a, b + 2 * ks);
      }
    }
  }

  // K-warps 1..3 of each N-warp store their partials, K-warp 0 adds them in warp order and writes
  // the scaled sums. acc[j][0..3] = (column 16j + g, row 2c), (same column, 2c + 1),
  // (column 16j + 8 + g, 2c), (same, 2c + 1).
  wait_group<0>();
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
  if (warp_k) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      red[(((warp_k - 1) * 2 + warp_n) * 16 + i) * 32 + lane] = acc[i / 4][i % 4];
    }
  }
  __syncthreads();
  if (warp_k == 0) {
    float scale[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      int const token = 2 * c + t;
      scale[t] = token >= rows   ? 0.f
                 : row_scales    ? row_scales[row0 + token]
                 : expert_scales ? expert_scales[e]
                                 : alpha[e][0];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float v = acc[i / 4][i % 4];
#pragma unroll
      for (int p = 0; p < 3; ++p) v += red[((p * 2 + warp_n) * 16 + i) * 32 + lane];
      int const token = 2 * c + (i & 1);
      int const n = ntile * kTileN + warp_n * 64 + (i / 4) * 16 + g + ((i & 2) ? 8 : 0);
      if (token < rows) {
        out[static_cast<size_t>(row0 + token) * N + n] = __float2bfloat16_rn(v * scale[i & 1]);
      }
    }
  }
}

// An upper bound of the m-blocks from the host's values: every block has a row, every expert has at
// most one partial block.
inline int64_t max_blocks(int64_t num_rows, int64_t num_experts) {
  return std::min(num_rows, (num_rows + kMaxRows - 1) / kMaxRows + std::min(num_experts, num_rows));
}

}  // namespace sm80_fp4_small_m

template <typename WeightType, typename ScaleBiasType>
using Sm80FP4SmallMInput =
    cutlass_kernels::GroupedGemmInput<__nv_bfloat16, WeightType, ScaleBiasType, __nv_bfloat16>;

// Whether the small-M kernel takes a GEMM of these sizes, also asked by the MoE runner before its
// W4A16 GEMM1 reads the token rows in place of restored rows.
inline bool sm80FP4SmallMSupportsShape(int64_t n, int64_t k, int64_t num_rows,
                                       int64_t num_experts) {
  using namespace sm80_fp4_small_m;
  return n % kTileN == 0 && k % kTileK == 0 && num_rows > 0 && num_experts > 0 &&
         max_blocks(num_rows, num_experts) <= 65535;
}

template <typename WeightType, typename ScaleBiasType>
bool sm80FP4SmallMSupports(Sm80FP4SmallMInput<WeightType, ScaleBiasType> const& inputs) {
  if (inputs.occupancy) {
    return true;
  }
  return sm80FP4SmallMSupportsShape(inputs.n, inputs.k, inputs.num_rows, inputs.num_experts) &&
         inputs.biases == nullptr && inputs.zeros == nullptr &&
         (inputs.row_scales != nullptr || inputs.alpha_scales != nullptr ||
          inputs.expert_scales != nullptr);
}

template <typename WeightType, typename ScaleBiasType>
void sm80FP4SmallMGemm(Sm80FP4SmallMInput<WeightType, ScaleBiasType> const& inputs) {
  using namespace sm80_fp4_small_m;
  static_assert(std::is_same_v<ScaleBiasType, __nv_bfloat16>, "BF16 block scales");
  auto const kernel_fn = kernel<kStages>;
  cudaError_t result =
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  TLLM_CHECK_WITH_INFO(result == cudaSuccess,
                       "Fail to set the max smem size to %d for the small-M FP4 MoE kernel",
                       kSmemBytes);
  if (inputs.occupancy) {
    tensorrt_llm::common::check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        inputs.occupancy, kernel_fn, kThreads, kSmemBytes));
    return;
  }
  dim3 const grid(static_cast<unsigned>(inputs.n / kTileN),
                  static_cast<unsigned>(max_blocks(inputs.num_rows, inputs.num_experts)));
  kernel_fn<<<grid, kThreads, kSmemBytes, inputs.stream>>>(
      reinterpret_cast<uint8_t const*>(inputs.B), reinterpret_cast<uint16_t const*>(inputs.scales),
      inputs.row_scales, inputs.alpha_scales, inputs.expert_scales, inputs.A, inputs.a_rows,
      inputs.a_tokens, inputs.total_tokens_including_expert, inputs.C, static_cast<int>(inputs.n),
      static_cast<int>(inputs.k), inputs.num_experts);
  result = cudaGetLastError();
  TLLM_CHECK_WITH_INFO(result == cudaSuccess,
                       "Fail to execute the small-M FP4 MoE kernel, cuda error %d", (int)(result));
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
