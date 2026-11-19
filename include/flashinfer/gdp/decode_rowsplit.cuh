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
#ifndef FLASHINFER_GDP_DECODE_ROWSPLIT_CUH_
#define FLASHINFER_GDP_DECODE_ROWSPLIT_CUH_

// Row-split decode / MTP-verification step of the Gated DeltaProduct (K = 128,
// V = 64, n Householder updates per token, FP32 state pool [pool_size, H, V, K])
// for small batch x heads.
//
// Per batch row b and head h, for every token t (gates fused, q / k L2-normalized):
//
//   alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))      softplus threshold 20
//   S = alpha_t * S
//   for j < n:  kq = l2norm(k_tj);  u = S kq;  S += sigmoid(b_tj) (v_tj - u) (x) kq
//   out_t = S (scale * l2norm(q_t))
//
// The V = 64 state rows of one (batch row, head) are independent of each other, so
// one CTA owns kRowsPerCta of them and a head's rows spread over 64 / kRowsPerCta
// CTAs: a call with few batch rows x heads fills more SMs than one CTA per
// (batch row, head).  Each CTA recomputes the per-token gates and the normalized
// q / k of its (batch row, head), which is cheap next to the state traffic, and
// issues every global load of its window (its state rows, all q / k / v / a / b of
// the T tokens) before the first use.
//
// Per state row the arithmetic is the lane-owns-rows body of the Cake GDP decode
// programs on SM100 / SM103, op for op: kLanes lanes per row, lane u owning the
// columns u * 4 + 4 * kLanes * c + {0, 1, 2, 3}; one f32x2 FMA chain over the
// lane's columns, its two halves added, then xor-shuffle sums over 1, 2 (, 4);
// the same approximate intrinsics for the gates and the normalization.  With the
// lane width of the Cake program a call selects, the outputs and every stored
// state are bitwise equal to that program's.  Build with fast math, as the Cake
// programs are built.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace gdp {
namespace rowsplit {

constexpr int kHeadK = 128;
constexpr int kHeadV = 64;
constexpr int kStaticSmemLimit = 48 * 1024;
// Bound on the 8-byte q / k pieces one lane keeps in registers between the
// prologue loads and their first use.
constexpr int kMaxQkPiecesPerLane = 40;

// Model tensors are BF16 (passed as their 16-bit patterns); strides are in elements.
struct Params {
  const uint16_t* q;             // [B, T, H, K]
  const uint16_t* k;             // [B, T, n, H, K]
  const uint16_t* v;             // [B, T, n, H, V]
  const uint16_t* a;             // [B, T, H]
  const uint16_t* b;             // [B, T, n, H]
  const float* A_log;            // [H]
  const float* dt_bias;          // [H]
  float* state;                  // [pool_size, H, V, K], dense rows
  const int32_t* read_slots;     // [B]
  const int32_t* write_slots;    // [B]
  const int32_t* scatter_slots;  // [B, T], read only when use_scatter != 0
  uint16_t* out;                 // [B, T, H, V]
  int64_t pool_size;
  int heads;
  float scale;
  int use_scatter;
  int disable_state_update;
  int64_t q_stride[3];        // b, t, h
  int64_t k_stride[4];        // b, t, j, h
  int64_t v_stride[4];        // b, t, j, h
  int64_t a_stride[3];        // b, t, h
  int64_t b_stride[4];        // b, t, j, h
  int64_t state_stride[3];    // slot, h, row
  int64_t scatter_stride[2];  // b, t
  int64_t out_stride[3];      // b, t, h
};

template <int kNumTokens, int kNumHouseholder, int kRowsPerCta, int kRowsPerLane, int kLanes>
struct KernelTraits {
  // k / v / b vectors of the window (n per token) and the q vectors next to them.
  static constexpr int kVecs = kNumTokens * kNumHouseholder;
  static constexpr int kQkVecs = kVecs + kNumTokens;
  static constexpr int kThreads = kRowsPerCta / kRowsPerLane * kLanes;
  static constexpr int kWarps = kThreads / 32;
  // float4 column chunks of a row per lane, as float2 pairs.
  static constexpr int kChunks = kHeadK / (4 * kLanes);
  static constexpr int kPairs = 2 * kChunks;
  static constexpr int kQkPerWarp = kWarps > 0 ? (kQkVecs + kWarps - 1) / kWarps : 0;
  static constexpr int kVPerThread = (kVecs * kRowsPerCta + kThreads - 1) / kThreads;
  static constexpr int kBPerThread = (kVecs + kThreads - 1) / kThreads;
  static constexpr int kSmemBytes =
      4 * (kQkVecs * kHeadK + kVecs * kRowsPerCta + kVecs + 2 * kNumTokens);
  static constexpr bool kSupported =
      (kNumTokens >= 1) && (kNumTokens <= 16) && (kNumHouseholder >= 1) && (kNumHouseholder <= 4) &&
      (kLanes == 4 || kLanes == 8) && (kRowsPerCta > 0) && (kHeadV % kRowsPerCta == 0) &&
      (kRowsPerLane > 0) && (kRowsPerCta % kRowsPerLane == 0) && (kRowsPerLane <= kLanes) &&
      (kThreads % 32 == 0) && (kThreads <= 1024) && (kThreads >= kNumTokens) &&
      (kQkPerWarp <= kMaxQkPiecesPerLane) && (kSmemBytes <= kStaticSmemLimit);
};

namespace detail {

__device__ __forceinline__ float bf16_lo(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf16_bits_to_float(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float lg2_approx(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The packed f32x2 operations of SM100+; the scalar forms elsewhere (other
// architectures are not bitwise-matched to the Cake programs).
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float2 r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(*reinterpret_cast<uint64_t*>(&r))
      : "l"(*reinterpret_cast<const uint64_t*>(&a)), "l"(*reinterpret_cast<const uint64_t*>(&b)),
        "l"(*reinterpret_cast<const uint64_t*>(&c)));
  return r;
#else
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  float2 r;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(*reinterpret_cast<uint64_t*>(&r))
      : "l"(*reinterpret_cast<const uint64_t*>(&a)), "l"(*reinterpret_cast<const uint64_t*>(&b)));
  return r;
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

}  // namespace detail

// Grid (B * H, 64 / kRowsPerCta): blockIdx.x = b * H + h, blockIdx.y = row group.
template <int kNumTokens, int kNumHouseholder, int kRowsPerCta, int kRowsPerLane, int kLanes>
__global__ void __launch_bounds__(
    KernelTraits<kNumTokens, kNumHouseholder, kRowsPerCta, kRowsPerLane, kLanes>::kThreads)
    GdpDecodeRowSplitKernel(const Params p) {
  using Traits = KernelTraits<kNumTokens, kNumHouseholder, kRowsPerCta, kRowsPerLane, kLanes>;
  constexpr int T = kNumTokens;
  constexpr int NH = kNumHouseholder;
  constexpr int NV = Traits::kVecs;
  constexpr int RPL = kRowsPerLane;
  constexpr int L = kLanes;
  constexpr int kThreads = Traits::kThreads;
  constexpr int kWarps = Traits::kWarps;
  constexpr int kChunks = Traits::kChunks;
  constexpr int kPairs = Traits::kPairs;

  __shared__ __align__(16) float sK[NV][kHeadK];
  __shared__ __align__(16) float sQ[T][kHeadK];
  __shared__ float sV[NV][kRowsPerCta];
  __shared__ float sBeta[NV];
  __shared__ float sAlpha[T];
  __shared__ int sSlot[T];

  const int tid = threadIdx.x;
  const int n = blockIdx.x / p.heads;
  const int h = blockIdx.x - n * p.heads;
  const int row_base = blockIdx.y * kRowsPerCta;
  // A row whose read slot is negative or outside the pool is skipped entirely: no
  // state read, no output, no writeback and no scatter.
  const int read_slot = p.read_slots[n];
  if (read_slot < 0 || read_slot >= p.pool_size) return;

  // Lane u of row group g owns the rows row_base + g * RPL + i (i < RPL).
  const int g = tid / L;
  const int u = tid - g * L;
  const int64_t head_off = static_cast<int64_t>(h) * p.state_stride[1] +
                           static_cast<int64_t>(row_base + g * RPL) * p.state_stride[2] + u * 4;

  float2 S[RPL][kPairs];
  {
    const float* src = p.state + static_cast<int64_t>(read_slot) * p.state_stride[0] + head_off;
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        const float4 x = *reinterpret_cast<const float4*>(src + i * p.state_stride[2] + c * 4 * L);
        S[i][2 * c] = make_float2(x.x, x.y);
        S[i][2 * c + 1] = make_float2(x.z, x.w);
      }
    }
  }

  // Every global load of the window is issued before any of them is used: warp w
  // reads the k / q vectors w, w + kWarps, ... (four BF16 per lane).
  const int warp = tid >> 5;
  const int lane = tid & 31;
  uint2 qk_raw[Traits::kQkPerWarp];
#pragma unroll
  for (int i = 0; i < Traits::kQkPerWarp; ++i) {
    const int vec = warp + i * kWarps;
    if (vec < NV) {
      const int t = vec / NH, j = vec - (vec / NH) * NH;
      qk_raw[i] = *reinterpret_cast<const uint2*>(p.k + n * p.k_stride[0] + t * p.k_stride[1] +
                                                  j * p.k_stride[2] + h * p.k_stride[3] + lane * 4);
    } else if (vec < Traits::kQkVecs) {
      qk_raw[i] = *reinterpret_cast<const uint2*>(
          p.q + n * p.q_stride[0] + (vec - NV) * p.q_stride[1] + h * p.q_stride[2] + lane * 4);
    }
  }
  uint16_t v_raw[Traits::kVPerThread];
#pragma unroll
  for (int i = 0; i < Traits::kVPerThread; ++i) {
    const int idx = tid + i * kThreads;
    if (idx < NV * kRowsPerCta) {
      const int vec = idx / kRowsPerCta, rr = idx - (idx / kRowsPerCta) * kRowsPerCta;
      const int t = vec / NH, j = vec - (vec / NH) * NH;
      v_raw[i] = p.v[n * p.v_stride[0] + t * p.v_stride[1] + j * p.v_stride[2] + h * p.v_stride[3] +
                     row_base + rr];
    }
  }
  uint16_t b_raw[Traits::kBPerThread];
#pragma unroll
  for (int i = 0; i < Traits::kBPerThread; ++i) {
    const int vec = tid + i * kThreads;
    if (vec < NV) {
      const int t = vec / NH, j = vec - (vec / NH) * NH;
      b_raw[i] = p.b[n * p.b_stride[0] + t * p.b_stride[1] + j * p.b_stride[2] + h * p.b_stride[3]];
    }
  }
  uint16_t a_raw = 0;
  int slot_raw = -1;
  float dt_bias = 0.0f, a_log = 0.0f;
  if (tid < T) {
    a_raw = p.a[n * p.a_stride[0] + tid * p.a_stride[1] + h * p.a_stride[2]];
    dt_bias = p.dt_bias[h];
    a_log = p.A_log[h];
    if (p.use_scatter)
      slot_raw = p.scatter_slots[n * p.scatter_stride[0] + tid * p.scatter_stride[1]];
  }

  // q / k: x * rsqrt(sum(x^2) + 1e-6) (q also times scale), one warp per vector.
#pragma unroll
  for (int i = 0; i < Traits::kQkPerWarp; ++i) {
    const int vec = warp + i * kWarps;
    if (vec < Traits::kQkVecs) {
      const float x0 = detail::bf16_lo(qk_raw[i].x), x1 = detail::bf16_hi(qk_raw[i].x);
      const float x2 = detail::bf16_lo(qk_raw[i].y), x3 = detail::bf16_hi(qk_raw[i].y);
      const float2 sq = detail::ffma2(
          make_float2(x2, x3), make_float2(x2, x3),
          detail::ffma2(make_float2(x0, x1), make_float2(x0, x1), make_float2(0.0f, 0.0f)));
      float ss = sq.x + sq.y;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
      float norm = rsqrtf(ss + 1e-6f);
      float* dst;
      if (vec < NV) {
        dst = &sK[vec][lane * 4];
      } else {
        norm = norm * p.scale;
        dst = &sQ[vec - NV][lane * 4];
      }
      *reinterpret_cast<float4*>(dst) = make_float4(x0 * norm, x1 * norm, x2 * norm, x3 * norm);
    }
  }
#pragma unroll
  for (int i = 0; i < Traits::kVPerThread; ++i) {
    const int idx = tid + i * kThreads;
    if (idx < NV * kRowsPerCta) {
      const int vec = idx / kRowsPerCta, rr = idx - (idx / kRowsPerCta) * kRowsPerCta;
      sV[vec][rr] = detail::bf16_bits_to_float(v_raw[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < Traits::kBPerThread; ++i) {
    const int vec = tid + i * kThreads;
    if (vec < NV) {
      sBeta[vec] = detail::rcp_approx(1.0f + __expf(-detail::bf16_bits_to_float(b_raw[i])));
    }
  }
  if (tid < T) {
    const float x = detail::bf16_bits_to_float(a_raw) + dt_bias;
    float softplus = x;
    if (x <= 20.0f) softplus = detail::lg2_approx(1.0f + __expf(x)) * 0.6931472f;
    sAlpha[tid] = __expf((-__expf(a_log)) * softplus);
    sSlot[tid] = slot_raw;
  }
  __syncthreads();

#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float2 alpha = make_float2(sAlpha[t], sAlpha[t]);
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
#pragma unroll
      for (int c = 0; c < kPairs; ++c) S[i][c] = detail::fmul2(S[i][c], alpha);
    }
#pragma unroll
    for (int j = 0; j < NH; ++j) {
      const int vec = t * NH + j;
      float2 kk[kPairs];
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        const float4 x = *reinterpret_cast<const float4*>(&sK[vec][u * 4 + 4 * L * c]);
        kk[2 * c] = make_float2(x.x, x.y);
        kk[2 * c + 1] = make_float2(x.z, x.w);
      }
      const float beta = sBeta[vec];
      float dot[RPL];
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
        float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int c = 0; c < kPairs; ++c) acc = detail::ffma2(S[i][c], kk[c], acc);
        dot[i] = acc.x + acc.y;
      }
#pragma unroll
      for (int off = 1; off < L; off <<= 1) {
#pragma unroll
        for (int i = 0; i < RPL; ++i) dot[i] += __shfl_xor_sync(0xffffffffu, dot[i], off);
      }
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
        const float coeff = (sV[vec][g * RPL + i] - dot[i]) * beta;
        const float2 coeff2 = make_float2(coeff, coeff);
#pragma unroll
        for (int c = 0; c < kPairs; ++c) S[i][c] = detail::ffma2(kk[c], coeff2, S[i][c]);
      }
    }
    const int slot = sSlot[t];
    if (slot >= 0 && slot < p.pool_size) {
      float* dst = p.state + static_cast<int64_t>(slot) * p.state_stride[0] + head_off;
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          *reinterpret_cast<float4*>(dst + i * p.state_stride[2] + c * 4 * L) =
              make_float4(S[i][2 * c].x, S[i][2 * c].y, S[i][2 * c + 1].x, S[i][2 * c + 1].y);
        }
      }
    }
    float2 qq[kPairs];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      const float4 x = *reinterpret_cast<const float4*>(&sQ[t][u * 4 + 4 * L * c]);
      qq[2 * c] = make_float2(x.x, x.y);
      qq[2 * c + 1] = make_float2(x.z, x.w);
    }
    float o[RPL];
#pragma unroll
    for (int i = 0; i < RPL; ++i) {
      float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int c = 0; c < kPairs; ++c) acc = detail::ffma2(S[i][c], qq[c], acc);
      o[i] = acc.x + acc.y;
    }
#pragma unroll
    for (int off = 1; off < L; off <<= 1) {
#pragma unroll
      for (int i = 0; i < RPL; ++i) o[i] += __shfl_xor_sync(0xffffffffu, o[i], off);
    }
    // Lane u < RPL of each row group stores row g * RPL + u.
    if (u < RPL) {
      float value = o[0];
#pragma unroll
      for (int i = 1; i < RPL; ++i) value = (u == i) ? o[i] : value;
      p.out[n * p.out_stride[0] + t * p.out_stride[1] + h * p.out_stride[2] + row_base + g * RPL +
            u] = detail::float_to_bf16_bits(value);
    }
  }

  if (!p.use_scatter && !p.disable_state_update) {
    const int write_slot = p.write_slots[n];
    if (write_slot >= 0 && write_slot < p.pool_size) {
      float* dst = p.state + static_cast<int64_t>(write_slot) * p.state_stride[0] + head_off;
#pragma unroll
      for (int i = 0; i < RPL; ++i) {
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          *reinterpret_cast<float4*>(dst + i * p.state_stride[2] + c * 4 * L) =
              make_float4(S[i][2 * c].x, S[i][2 * c].y, S[i][2 * c + 1].x, S[i][2 * c + 1].y);
        }
      }
    }
  }
}

template <int kNumTokens, int kNumHouseholder, int kRowsPerCta, int kRowsPerLane, int kLanes>
cudaError_t LaunchGdpDecodeRowSplit(const Params& params, int64_t batch, cudaStream_t stream) {
  using Traits = KernelTraits<kNumTokens, kNumHouseholder, kRowsPerCta, kRowsPerLane, kLanes>;
  static_assert(Traits::kSupported, "unsupported row-split configuration");
  const int64_t ctas = batch * params.heads;
  if (batch <= 0 || params.heads <= 0 || ctas > 0x7fffffffLL) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(ctas), kHeadV / kRowsPerCta);
  GdpDecodeRowSplitKernel<kNumTokens, kNumHouseholder, kRowsPerCta, kRowsPerLane, kLanes>
      <<<grid, Traits::kThreads, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace rowsplit
}  // namespace gdp
}  // namespace flashinfer

#endif  // FLASHINFER_GDP_DECODE_ROWSPLIT_CUH_
