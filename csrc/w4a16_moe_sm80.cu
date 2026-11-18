/*
 * Copyright (c) 2025 by FlashInfer team.
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
// Weight-only NVFP4 MoE for SM80 to SM90 with BF16 activations. NVFP4 weights are converted once
// into records (flashinfer.fused_moe.cute_dsl_w4a16_sm80_weights): per expert, 128 output
// columns and 128 k, 8192 B of E2M1 codes in MMA fragment order, then 1024 B of block scales (the
// E4M3 magnitude as a 5-bit exponent and 3-bit mantissa; the sign moves into the codes); a K tail
// of at most 64 k takes a 5120 B record. Products carry e2m1 x scale x 2^-7; the per-expert global
// scale x 2^7 scales the fp32 accumulator. A forward is prep (routing -> blocks of R rows), GEMM1
// (+ activation), GEMM2 and finalize. prep places entry f = token * top_k + slot at the rank of f
// among its expert's entries, so the placement and every fp32 sum order repeat bit for bit.
// R 8, 16, 32: 8 warps of 16 columns; R 64: 4 warps of 32 columns, swizzled activation rows,
// multi-CTA prep. R 96, kW64s (GEMM1), kG2W64x (GEMM2) and kG2W64v (kG2W64x, one 16 B scale load
// per k128) cover two records per CTA, 8 warps of 32 or 4 of 64 columns; kX96 and kQ12 run R 96 at
// 2 stages with the R 64 two-record epilogue, a hoisted activation address and shift-only dequant,
// kQ12 in 12 row parts, not 6. One CTA per output tile. sm_80 and sm_90 builds (cp.async, ldmatrix,
// bf16 mma.sync); sm_86/87/89 run sm_80 code. All configs but R 96 at 3 stages (133632 B, only
// where a block has that much) fit sm_86/89's 99 KB per-block shared memory (at most 89088 B).
// NVFP4 activations (W4A4) run the same records: prep also restores the E2M1 x E4M3 input to
// BF16, and GEMM1 with FQ quantizes its activation to NVFP4 (16-column blocks, E4M3 scales under
// the FC2 global scale, as the CUTLASS SM80 path's doActivation) and writes code x scale in BF16.
// Both restores are exact; the global scales go to the GEMM alphas.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "cute/arch/mma_sm80.hpp"
#include "cutlass/fast_math.h"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/quantization_utils.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::w4a16_moe_sm80 {

constexpr int kTileN = 128, kTileK = 128;
constexpr int kWBytes = kTileN * kTileK / 2;   // 8192
constexpr int kSBytes = kTileN * kTileK / 16;  // 1024
constexpr int kRec = kWBytes + kSBytes;        // 9216
constexpr int kHalfW = kWBytes / 2;            // 4096: k 0..63 of a record, all warps
constexpr int kTailRec = kHalfW + kSBytes;     // 5120: a K tail of at most 64
constexpr int kActPitch = 272;                 // 256 B row + 16 B against ldmatrix conflicts
// Smallest per-block shared memory limit from sm_80 to sm_90 (sm_86, sm_89).
constexpr int kSmemLimit = 99 * 1024;
// The R < 64 prep zeroes kFinCnt counters (column chunks, H <= 2048) of the GEMM2's top-k sum.
constexpr int kFinCnt = 16;

template <int R, int NC = 1>
struct Cfg {
  static constexpr int kABytes = R * kActPitch;
  static constexpr int kStageBytes = NC * kRec + kABytes;
};

__device__ __forceinline__ void cp16z(uint32_t s, const void* g, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(g),
               "r"(valid ? 16 : 0));
}
// Volatile shared loads: ptxas does not merge them into wider loads.
__device__ __forceinline__ void lds32v(uint32_t* d, uint32_t a) {
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(d[0]) : "r"(a));
}
// gt / d for d = 1 + __expf(-gt), bit-identical to the IEEE division (checked
// over all 2^32 fp32 gt on sm_80): the division's own fast path (MUFU.RCP + 4
// FFMA) with the cases its FCHK sends to the slow path written out as selects,
// no branch.
__device__ __forceinline__ float div_fast(float n, float d) {
  float r = flashinfer::math::ptx_rcp(d);
  const float t = fmaf(-d, r, 1.f);
  r = fmaf(r, t, r);
  const float q0 = fmaf(n, r, 0.f);
  const float e = fmaf(-d, q0, n);
  return fmaf(r, e, q0);
}
__device__ __forceinline__ float silu_div(float n, float d) {
  const float inf = __int_as_float(0x7f800000);
  const float s = d >= 0x1p126f ? 0x1p-24f : 1.f;
  float q = div_fast(n, d * s) * s;
  q = d == inf ? n * 0.f : q;
  q = (n == 0.f || n == inf) ? n : q;
  return q;
}
__device__ __forceinline__ void lds64v(uint32_t* d, uint32_t a) {
  asm volatile("ld.volatile.shared.v2.u32 {%0, %1}, [%2];" : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
__device__ __forceinline__ void lds128(uint32_t* d, uint32_t a) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* d, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* d, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t const* a, uint32_t const* b) {
  cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0],
                                            b[1], d[0], d[1], d[2], d[3]);
}
// a x {lo(s2), lo(s2)} and a x {hi(s2), hi(s2)}, exact; the broadcast is built
// inside the asm so ptxas can fold it into HFMA2's operand swizzle.
__device__ __forceinline__ uint32_t mul_lo(uint32_t a, uint32_t s2) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\n.reg .b32 b;\nmov.b32 {l, h}, %2;\nmov.b32 b, {l, "
      "l};\n"
      "fma.rn.bf16x2 %0, %1, b, %3;\n}\n"
      : "=r"(r)
      : "r"(a), "r"(s2), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ uint32_t mul_hi(uint32_t a, uint32_t s2) {
  uint32_t r;
  asm("{\n.reg .b16 l, h;\n.reg .b32 b;\nmov.b32 {l, h}, %2;\nmov.b32 b, {h, "
      "h};\n"
      "fma.rn.bf16x2 %0, %1, b, %3;\n}\n"
      : "=r"(r)
      : "r"(a), "r"(s2), "r"(0x80008000u));
  return r;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
constexpr bool kSm90 = true;
#else
constexpr bool kSm90 = false;
#endif

// 3+1 layout -> four bf16x2 registers of e2m1 x 2^-126.
template <bool SHF = false>
__device__ __forceinline__ void cvt31(uint32_t w, uint32_t* d) {
  d[0] = (w << 6) & 0x81C081C0u;
  d[1] = (w << 3) & 0x81C081C0u;
  d[2] = w & 0x81C081C0u;
  // m = w & 0x2C002C00u has bits only in {10, 11, 13, 26, 27, 29} and
  // 0x22000000 = 2^29 + 2^25, so __umulhi(m, 0x22000000) = (m >> 3) +
  // (m >> 7), two bit-disjoint shifts. SHF computes the same bits with shifts
  // instead of the IMAD.HI; sm_90 keeps the multiply.
  const uint32_t m = w & 0x2C002C00u;
  const uint32_t x = SHF && !kSm90 ? (m >> 3) | (m >> 7) : __umulhi(m, 0x22000000u);
  d[3] = (x & 0x01C001C0u) | ((w & 0x40004000u) << 1);
}
// Two U5 scale pairs of one word: even j (bytes 0, 2) and odd j (bytes 1, 3),
// as s x 2^119.
__device__ __forceinline__ uint32_t u5_even(uint32_t q) {
  return ((q << 4) & 0x0FF00FF0u) | 0x70007000u;
}
__device__ __forceinline__ uint32_t u5_odd(uint32_t q) {
  return ((q >> 4) & 0x0FF00FF0u) | 0x70007000u;
}
// The MMA A fragment of one 3+1 word w times the scale pair s2.
template <bool SHF = false>
__device__ __forceinline__ void dq4(uint32_t w, uint32_t s2, uint32_t* a) {
  uint32_t d[4];
  cvt31<SHF>(w, d);
  a[0] = mul_lo(d[0], s2);
  a[1] = mul_hi(d[1], s2);
  a[2] = mul_lo(d[2], s2);
  a[3] = mul_hi(d[3], s2);
}

struct GemmParams {
  const uint8_t* w;             // records [E][chunks][S]: 9216 B each, the last 5120 B for kr <= 64
  const float* alpha;           // [E] per-expert global dequant scale
  const __nv_bfloat16* act;     // GEMM1: hidden [tokens][K]; GEMM2: blocked intermediate [nb][R][K]
  const int* nblocks;           // device scalar
  const int* block_expert;      // [nb]
  const int* block_rows;        // [nb]
  const int* rowmap;            // [nb][R]: token * top_k + slot, or -1
  __nv_bfloat16* out;           // GEMM1: [nb][R][Nout] (blocked); GEMM2: [tokens*top_k][Nout]
  float* ws;                    // gemm_pt: GEMM1 token scales [tokens], GEMM2 row scales [nb][R]
  int* cnt;                     // FQ GEMM2: [chunks] arrival counts, zeroed by prep
  int top_k, N, K;              // N: record columns, a multiple of 128 (gated: 2 x I padded to 64)
  int Nout;                     // output columns (gated: I)
  int units_per_cta;            // unused, as ws
  cutlass::FastDivmod topk_dm;  // f / top_k (G1 only); appended to preserve layout
};

// FQ GEMM1: GemmParams and the FC2 global scale num / den (den null or zero: num), each [E]
// (stride 1) or one value. FQ GEMM2 (R 8, decode): sum_out [tokens][Nout] = out's top-k sum with
// fin_w (float or BF16) and fin_ids, as finalize. Its own type, so the other kernels keep their
// parameters.
struct GemmParamsFq : GemmParams {
  const float* fq_num;
  const float* fq_den;
  int fq_num_stride, fq_den_stride;
  const void* fin_w;
  const int* fin_ids;
  __nv_bfloat16* sum_out;
  int tokens;
  bool fin_bf16;
};
template <bool FQ>
using GemmParamsT = std::conditional_t<FQ, GemmParamsFq, GemmParams>;

// out[t][8 c, 8 c + 8) = sum_j w[t][j] y[t top_k + j][...], fp32 in slot order over ids >= 0.
// U > 1 loads the ids, weights and rows of U slots at once (y and w hold every slot, so the unused
// ones stay in range), then sums the valid ones. WT: the routing weights' type, float or BF16.
template <typename WT, int U>
__device__ __forceinline__ void fin_sum(const __nv_bfloat16* __restrict__ y,
                                        const WT* __restrict__ w, const int* __restrict__ ids,
                                        __nv_bfloat16* __restrict__ out, int t, int c, int top_k,
                                        int H) {
  const int h8 = H / 8;
  float a[8] = {};
  if constexpr (U == 1) {
    for (int j = 0; j < top_k; ++j) {
      if (ids[t * top_k + j] < 0) continue;
      const float wj = static_cast<float>(w[t * top_k + j]);
      const uint4 v =
          __ldg(reinterpret_cast<const uint4*>(y + (static_cast<long>(t) * top_k + j) * H + c * 8));
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(b2[i]);
        a[2 * i] += wj * f.x;
        a[2 * i + 1] += wj * f.y;
      }
    }
  } else {
    for (int j0 = 0; j0 < top_k; j0 += U) {
      const long f = static_cast<long>(t) * top_k + j0;
      const uint4* r = reinterpret_cast<const uint4*>(y + f * H + c * 8);
      int e[U];
      float wj[U];
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        e[u] = j0 + u < top_k ? __ldg(ids + f + u) : -1;
        if (j0 + u < top_k) {
          wj[u] = static_cast<float>(w[f + u]);
          v[u] = __ldg(r + static_cast<long>(u) * h8);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (e[u] < 0) continue;
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v[u]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(b2[i]);
          a[2 * i] += wj[u] * f.x;
          a[2 * i + 1] += wj[u] * f.y;
        }
      }
    }
  }
  uint4 o;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i) o2[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
  *reinterpret_cast<uint4*>(out + static_cast<long>(t) * H + c * 8) = o;
}

// FQ GEMM2: once every CTA of column chunk c (CW columns) has stored its tile, the last of them to
// count itself in p.cnt[c] writes the chunk's columns of sum_out. Thread 0's acq_rel add, after the
// barrier, releases this CTA's stores and acquires the other CTAs' (all adds to p.cnt[c] form one
// release sequence); the barrier after it orders the other threads' loads. nb 0: no tile, the
// first CTAs write the (zero) sums. fin_sum's sums, a stage of whole tokens at a time: the CTA
// stages their rows of y (through L2), weights and ids in the ring (RING bytes, free: no copy is
// pending after the last k step, and the barriers follow every warp's reads; the host bounds top_k
// by one token's stage), then a thread per token and 8 columns adds the slots in order.
template <int NT, int CW, int RING>
__device__ __forceinline__ void fin_chunk(GemmParamsFq const& p, int c, int nb) {
  constexpr int G = CW / 8;  // 8-column groups
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int last;
  __syncthreads();
  if (threadIdx.x == 0) {
    int old = -1;
    if (nb > 0)
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                   : "=r"(old)
                   : "l"(p.cnt + c)
                   : "memory");
    last = old == nb - 1;
  }
  __syncthreads();
  if (!last) return;
  const int top_k = p.top_k, H = p.Nout, ng = min(CW, H - c * CW) / 8, g = threadIdx.x % G;
  const int ts = RING / (top_k * (G * 16 + 8));  // tokens per stage
  static_assert(RING / (G * 16 + 8) <= NT, "a stage's rows: one weight and id a thread");
  const uint32_t sb = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  for (int t0 = 0; t0 < p.tokens; t0 += ts) {
    const int nr = min(ts, p.tokens - t0) * top_k;  // rows: entries t0 top_k + [0, nr)
    const long f0 = static_cast<long>(t0) * top_k;
    uint4* ys = reinterpret_cast<uint4*>(smem);         // [nr][G]
    float* wl = reinterpret_cast<float*>(ys + nr * G);  // [nr]
    int* il = reinterpret_cast<int*>(wl + nr);          // [nr]
    if (t0 > 0) __syncthreads();                        // the last stage's reads
    for (int i0 = threadIdx.x; i0 < nr * G; i0 += 4 * NT) {
      // every load ahead of the stores; the first round also stages row i0's weight and id
      const bool row = i0 < nr;
      int e = 0;
      float wv = 0.f;
      if (row) {
        e = __ldg(p.fin_ids + f0 + i0);
        wv = p.fin_bf16 ? static_cast<float>(static_cast<const __nv_bfloat16*>(p.fin_w)[f0 + i0])
                        : static_cast<const float*>(p.fin_w)[f0 + i0];
      }
      uint4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = i0 + u * NT;
        const bool in = i < nr * G && i % G < ng;
        v[u] = __ldcg(reinterpret_cast<const uint4*>(
            in ? p.out + (f0 + i / G) * H + (c * G + i % G) * 8 : p.out));
      }
      if (row) {
        il[i0] = e;
        wl[i0] = wv;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = i0 + u * NT;
        if (i < nr * G && i % G < ng) ys[i] = v[u];
      }
    }
    __syncthreads();
    if (g >= ng) continue;
    const uint32_t sw = sb + nr * G * 16, si = sw + nr * 4;
    for (int t = threadIdx.x / G; t * top_k < nr; t += NT / G) {
      float a[8] = {};
      for (int j0 = 0; j0 < top_k; j0 += 4) {
        uint32_t e[4], wj[4], v[4][4];
        // four slots' loads ahead of their tests (volatile: not sunk)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = t * top_k + min(j0 + u, top_k - 1);
          lds32v(e + u, si + r * 4);
          lds32v(wj + u, sw + r * 4);
          lds128(v[u], sb + (r * G + g) * 16);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          if (j0 + u >= top_k || static_cast<int>(e[u]) < 0) continue;
          const float w = __uint_as_float(wj[u]);
          const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(v[u]);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(b2[i]);
            a[2 * i] += w * f.x;
            a[2 * i + 1] += w * f.y;
          }
        }
      }
      uint4 o;
      __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) o2[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
      *reinterpret_cast<uint4*>(p.sum_out + static_cast<long>(t0 + t) * H + (c * G + g) * 8) = o;
    }
  }
}

// 8 E2M1 codes (cvt_warp_fp16_to_fp4 order) times the E4M3 scale sf, as 8 BF16 (4 words): the
// CUTLASS path's restoreNvfp4x8. Every product is exact in BF16. SW (below sm_90): the scale's
// value is the quantizer's e4m3_to_fp32_sw, the same float without the cast's half emulation.
// The R > 8 GEMM epilogues keep the cast: with SW, ptxas reallocates their main loops (fewer
// CTAs, a spill).
template <bool SW = true>
__device__ __forceinline__ void restore_nvfp4x8(uint32_t codes, uint8_t sf, uint32_t* out) {
  __nv_fp8_e4m3 scale;
  scale.__x = sf;
  const __nv_bfloat162 s2 = __float2bfloat162_rn(
      SW && !kSm90 ? tensorrt_llm::kernels::e4m3_to_fp32_sw(sf) : static_cast<float>(scale));
  flashinfer::e2m1x8_to_16bx8<__nv_bfloat16>(codes, out);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162& v = reinterpret_cast<__nv_bfloat162&>(out[i]);
    v = __hmul2(v, s2);
  }
}

// bf16 pair (lo in the low 16 bits) and an 8 x 8 b16 transpose across the warp
// (movmatrix, sm_75+): lane l reads row l / 4, columns 2 (l % 4) and + 1 of the
// piece and gets the same positions of the transposed piece.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// Per-token NVFP4 (PT, gemm_pt): the reference's FP32 products as its -use_fast_math modules run
// them (FTZ), and its relu2, CUTLASS's ReLu (max.NaN: NaN propagates) then the square.
__device__ __forceinline__ float mul_ftz(float a, float b) {
  return tensorrt_llm::kernels::nvfp4_quantize_mul(a, b);
}
__device__ __forceinline__ float relu2_nan(float y) {
  float x;
  asm("max.NaN.f32 %0, %1, 0f00000000;" : "=f"(x) : "f"(y));
  return mul_ftz(x, x);
}
__device__ __forceinline__ float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint32_t movm_trans(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}

// ACT: 0 none (GEMM2), 1 silu(gate) * up (GEMM1, gated), 2 relu(x)^2 (GEMM1,
// not gated) KT: K % 128 != 0 (a tail k-step per chunk). G1E, Q12F: the R 96
// 2-stage paths of kX96 and kQ12. VS: the scale load of kG2W64v. FQ (GEMM1): the
// activation quantized to NVFP4 and written as BF16 code x scale; (R 8 GEMM2) fin_chunk. PT
// (gemm_pt, per-token NVFP4): GEMM1 (relu2) scales a row by alpha s_t of its token (p.ws) and
// GEMM2 by its row scale p.ws[b R + r], both on the FP32 sum as the reference's epilogue.
template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1, bool G1E = false,
          bool Q12F = false, bool VS = false, bool FQ = false, bool PT = false>
__device__ __forceinline__ void gemm_body(GemmParamsT<FQ> const& p) {
  static_assert(!FQ || ACT != 0 || R == 8, "FQ: a GEMM1 epilogue or the R 8 GEMM2 sum");
  using C = Cfg<R, NC>;
  constexpr int NT = NW * 32;
  constexpr int SC = 8 * NC / WC, WT = NW / WC, TW = R / WT, G = TW / 8;
  static_assert(!PT || (ACT != 1 && NC == 1 && SC < 4 && !G1E && !Q12F && !VS && (!FQ || !ACT)),
                "PT: the kBase relu2 GEMM1 or GEMM2");
  constexpr int RP = NT / 16;  // activation rows per pass: 16 threads x 16 B per 256 B row
  constexpr int WP = kWBytes / (NT * 16);  // weight cp16 per thread and record
  static_assert((WC == 4 || WC == 8) && NW % WC == 0 && (R % RP == 0 || RP % R == 0) && TW % 8 == 0,
                "warp tile");
  static_assert(kWBytes % (NT * 16) == 0 && kHalfW % (NT * 16) == 0 && kSBytes / 16 <= NT, "issue");
  // At R 64 with 4 warps the activation rows sit at a 256 B pitch, the 16 B
  // chunk q of row r at chunk q ^ (r & 7) (a permutation inside the chunk's 128
  // B half), in place of the 272 B padded pitch. The cp.async writes and the
  // ldmatrix reads take the same mapping; the stage keeps its R x 272 B.
  constexpr bool SWZ = R >= 64;
  constexpr int AP = SWZ ? 256 : kActPitch;
  static_assert(!SWZ || (RP % 8 == 0 && TW % 8 == 0),
                "swizzle: r & 7 is fixed per thread and per lane");
  constexpr uint32_t kStage = C::kStageBytes;
  constexpr uint32_t kRing = STAGES * kStage;
  extern __shared__ __align__(128) uint8_t smem[];
  const int nb = *p.nblocks;
  const int K = p.K;
  const int record_chunks = p.N / kTileN;
  const int chunks = (record_chunks + NC - 1) / NC;
  const int S = KT ? (K + kTileK - 1) / kTileK : K / kTileK;
  const int kr = K - (S - 1) * kTileK;
  const bool half = KT && kr <= kTileK / 2;
  const long cbytes = static_cast<long>(S - 1) * kRec + (half ? kTailRec : kRec);
  if (blockIdx.x >= nb * chunks) {
    if constexpr (FQ && ACT == 0)
      if (nb == 0 && blockIdx.x < chunks) fin_chunk<NT, NC * kTileN, kRing>(p, blockIdx.x, 0);
    return;
  }
  const int upb = record_chunks * S;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, tq = lane & 3;
  const int wc = warp % WC, wt = warp / WC;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  const int ib = blockIdx.x / chunks;
  int iw = (blockIdx.x % chunks) * NC * S;
  int iks = 0;
  uint32_t islot = 0;
  const uint8_t* gws[NC];
  bool valid_record[NC];
  constexpr int RJ = R > RP ? R / RP : 1;
  // Linear GEMM2 (ACT 0) at R 64, 2 stages: the RJ activation-row source
  // pointers differ only by a constant RP * K, so keep one and rebuild the
  // rest from it to free registers. Other ACT/R/STAGES keep gact[RJ].
  constexpr bool kLinSingle = ACT == 0 && R == 64 && STAGES == 2;
  constexpr int GJ = kLinSingle ? 1 : RJ;
  const __nv_bfloat16* gact[GJ];
  bool avalid[RJ];
  // SC 4: the rows as indices (-1: absent), the source formed at each issue
  constexpr bool kRowIndex = SC == 4 && ACT != 0;
  int arow[kRowIndex ? RJ : 1];
#pragma unroll
  for (int j = 0; j < RJ; ++j) avalid[j] = true;
  const int ar = tid >> 4, aq = tid & 15;
  const bool has_act = R >= RP || tid < R * 16;
  const bool has_s = tid < kSBytes / 16;
  const uint32_t sw_i = sbase + tid * 16;
  // rows ar + RP j share ar & 7 (RP % 8 == 0), so the swizzled chunk is one
  // constant per thread
  const uint32_t sa_i =
      sbase + NC * kRec + (has_act ? ar : 0) * AP + (SWZ ? (aq ^ (ar & 7)) : aq) * 16;
  // PT GEMM1: the s_t of this thread's rows (aq 0) and alpha, stored past the ring as row scales
  [[maybe_unused]] float pt_s[PT && ACT ? RJ : 1];
  [[maybe_unused]] float pt_a = 0.f;
  auto load_block = [&]() {
    const long e = p.block_expert[ib];
    if constexpr (PT && ACT) pt_a = p.alpha[e];
    const int brows = ACT == 0 ? p.block_rows[ib] : R;
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) {
      const int c = iw / S + rec;
      valid_record[rec] = NC == 1 || c < record_chunks;
      if constexpr (KT) {
        gws[rec] = p.w + (e * record_chunks + c) * cbytes + tid * 16;
      } else {
        gws[rec] = p.w + (e * upb + iw + rec * S) * kRec + tid * 16;
      }
    }
#pragma unroll
    for (int j = 0; j < RJ; ++j) {
      const int r = ar + RP * j;
      long row = 0;
      if (has_act) {
        if constexpr (ACT != 0) {
          const int f = p.rowmap[ib * R + r];
          avalid[j] = f >= 0;
          row = avalid[j] ? p.topk_dm.div(f) : 0;
        } else {
          row = static_cast<long>(ib) * R + r;
          avalid[j] = r < brows;  // a padding row of the intermediate:
                                  // zero-filled, not read
        }
      }
      if constexpr (kLinSingle) {
        if (j == 0) gact[0] = p.act + row * K + aq * 8 + iks * kTileK;
      } else {
        gact[j] = p.act + row * K + aq * 8 + iks * kTileK;
      }
      if constexpr (kRowIndex) arow[j] = avalid[j] ? static_cast<int>(row) : -1;
      if constexpr (PT && ACT) pt_s[j] = has_act && aq == 0 && avalid[j] ? __ldg(p.ws + row) : 0.f;
    }
  };
  load_block();
  auto issue = [&]() {
    bool tl = false, hf = false;
    if constexpr (KT) {
      tl = iks == S - 1;
      hf = tl && half;
    }
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) {
      const uint32_t dst = sw_i + islot + rec * kRec;
      const uint8_t* src = valid_record[rec] ? gws[rec] : p.w;
#pragma unroll
      for (int q = 0; q < WP; ++q) {
        const int o = q * NT * 16;
        if (!hf || o < kHalfW) cp16z(dst + o, src + o, valid_record[rec]);
      }
      if (has_s) cp16z(dst + kWBytes, src + (hf ? kHalfW : kWBytes), valid_record[rec]);
    }
#pragma unroll
    for (int j = 0; j < RJ; ++j) {
      if (!has_act) continue;
      const uint32_t sa = sa_i + j * RP * AP + islot;
      if constexpr (kRowIndex) {
        const bool av = arow[j] >= 0;
        const __nv_bfloat16* ga =
            p.act + static_cast<long>(av ? arow[j] : 0) * K + aq * 8 + iks * kTileK;
        if constexpr (KT) {
          const bool in = !tl || aq * 8 < kr;
          if (!hf || aq < 8) cp16z(sa, in ? ga : p.act, av && in);
        } else {
          cp16z(sa, ga, av);
        }
      } else if constexpr (kLinSingle) {
        const __nv_bfloat16* gj = gact[0] + static_cast<long>(j) * RP * K;
        if constexpr (KT) {
          const bool in = !tl || aq * 8 < kr;
          if (!hf || aq < 8) cp16z(sa, in ? gj : p.act, avalid[j] && in);
        } else {
          cp16z(sa, gj, avalid[j]);
        }
      } else if constexpr (KT) {
        const bool in = !tl || aq * 8 < kr;
        if (!hf || aq < 8) cp16z(sa, in ? gact[j] : p.act, avalid[j] && in);
      } else {
        cp16z(sa, gact[j], avalid[j]);
      }
    }
    islot = islot + kStage == kRing ? 0u : islot + kStage;
#pragma unroll
    for (int rec = 0; rec < NC; ++rec) gws[rec] += hf ? kTailRec : kRec;
    if constexpr (kLinSingle) {
      gact[0] += kTileK;
    } else {
#pragma unroll
      for (int j = 0; j < RJ; ++j) gact[j] += kTileK;
    }
    ++iks;
  };

  // Writes one tile from per-lane fp32 values v[SC][G][4] (summed over k);
  // gemm's mapping with record warp slot ws = wc SC + s in place of the warp
  // and rows offset by wt TW.
  auto write_tile = [&](int b, int c, float(*v)[G][4]) {
    const float a = p.alpha[p.block_expert[b]] * 128.f;  // products carry 2^-7
    const int Nout = p.Nout;
    // R 8 GEMM1 with SiLU or relu2 on SM8x: rows at or past the block's row
    // count are padding that GEMM2 zero-fills on load, so they are not stored.
    constexpr bool kSkipPad = !kSm90 && R == 8 && ACT != 0;
    [[maybe_unused]] const int prows = kSkipPad ? p.block_rows[b] : R;
    if constexpr (ACT == 0 && (SC == 4 || (G1E && R == 96 && STAGES == 2 && !KT))) {
      // v[s][gi][2 hh + h] is column ws 16 + 8 hh + g8, row gi 8 + 2 tq + h:
      // per (s, gi, hh) an 8 x 8 piece, columns x rows, one bf16 pair (rows 2
      // tq, 2 tq + 1) per lane. Transposed, lane (g8, tq) holds row gi 8 + g8,
      // columns 8 hh + 2 tq and + 1: one 4 B store (Nout even, out 4 B
      // aligned).
      if ((Nout & 1) == 0 && (reinterpret_cast<uintptr_t>(p.out) & 3) == 0) {
        const int rows = p.block_rows[b];
        int drow[G];
#pragma unroll
        for (int gi = 0; gi < G; ++gi) drow[gi] = p.rowmap[b * R + wt * TW + gi * 8 + g8];
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
          if (wt * TW + gi * 8 + g8 >= rows) drow[gi] = -1;
        const int n0 = c * NC * kTileN + wc * SC * 16 + 2 * tq;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
          uint32_t t[SC][2];
#pragma unroll
          for (int s = 0; s < SC; ++s)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
              t[s][hh] = movm_trans(pack_bf16x2(v[s][gi][2 * hh] * a, v[s][gi][2 * hh + 1] * a));
          if (drow[gi] >= 0) {
            __nv_bfloat16* o = p.out + static_cast<long>(drow[gi]) * Nout + n0;
#pragma unroll
            for (int s = 0; s < SC; ++s)
#pragma unroll
              for (int hh = 0; hh < 2; ++hh)
                if (n0 + s * 16 + hh * 8 < Nout)
                  *reinterpret_cast<uint32_t*>(o + s * 16 + hh * 8) = t[s][hh];
          }
        }
        return;
      }
    }
    // GEMM2 scatters rows: every destination row is loaded first (independent
    // loads within the block), so no load waits behind the stores before it; -1
    // marks a padding row.
    int dst[G][2];
    [[maybe_unused]] float2 rs2[G];  // PT GEMM2: rows 2 tq and 2 tq + 1's scales
    if constexpr (ACT == 0) {
      const int rows = p.block_rows[b];
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h) dst[gi][h] = p.rowmap[b * R + wt * TW + gi * 8 + 2 * tq + h];
      if constexpr (PT)
#pragma unroll
        for (int gi = 0; gi < G; ++gi)
          rs2[gi] = *reinterpret_cast<const float2*>(p.ws + b * R + wt * TW + gi * 8 + 2 * tq);
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          if (wt * TW + gi * 8 + 2 * tq + h >= rows) dst[gi][h] = -1;
    }
#pragma unroll
    for (int s = 0; s < SC; ++s) {
      const int ws = wc * SC + s;
#pragma unroll
      for (int gi = 0; gi < G; ++gi) {
        const int r0 = wt * TW + gi * 8 + 2 * tq;
        if constexpr (ACT == 1) {
          const int i = c * NC * 64 + ws * 8 + g8;
          if (i < Nout) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              if (kSkipPad && r0 + h >= prows) continue;  // padding row
              const float gt = v[s][gi][h] * a, up = v[s][gi][2 + h] * a;
              float y;
              if constexpr (SC == 4 || (G1E && R == 96 && STAGES == 2 && !KT))
                y = silu_div(gt, 1.f + __expf(-gt)) * up;
              else
                y = gt / (1.f + __expf(-gt)) * up;
              p.out[(static_cast<long>(b) * R + r0 + h) * Nout + i] = __float2bfloat16_rn(y);
            }
          }
        } else if constexpr (ACT == 2) {
          const int n = c * NC * kTileN + ws * 16 + g8;
          [[maybe_unused]] float2 rs;  // PT: rows r0 and r0 + 1's alpha s_t
          if constexpr (PT) rs = reinterpret_cast<const float2*>(smem + kRing)[r0 / 2];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int nn = n + (q >> 1) * 8;
            if (kSkipPad && r0 + (q & 1) >= prows) continue;  // padding row
            if (nn < Nout) {
              if constexpr (PT) {
                // the reference's BF16 GEMM1 output (the sum times the row scale), then relu2
                const float y = bf16_round(mul_ftz(v[s][gi][q] * 128.f, q & 1 ? rs.y : rs.x));
                p.out[(static_cast<long>(b) * R + r0 + (q & 1)) * Nout + nn] =
                    __float2bfloat16_rn(relu2_nan(y));
              } else {
                const float x = fmaxf(v[s][gi][q] * a, 0.f);
                p.out[(static_cast<long>(b) * R + r0 + (q & 1)) * Nout + nn] =
                    __float2bfloat16_rn(x * x);
              }
            }
          }
        } else {
          const int n = c * NC * kTileN + ws * 16 + g8;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (dst[gi][h] >= 0) {
              const long f = dst[gi][h];
              if constexpr (PT) {  // the sum times the row's scale (alpha folded in), as reference
                const float r = h ? rs2[gi].y : rs2[gi].x;
                if (n < Nout)
                  p.out[f * Nout + n] = __float2bfloat16_rn(mul_ftz(v[s][gi][h] * 128.f, r));
                if (n + 8 < Nout)
                  p.out[f * Nout + n + 8] =
                      __float2bfloat16_rn(mul_ftz(v[s][gi][2 + h] * 128.f, r));
              } else {
                if (n < Nout) p.out[f * Nout + n] = __float2bfloat16_rn(v[s][gi][h] * a);
                if (n + 8 < Nout)
                  p.out[f * Nout + n + 8] = __float2bfloat16_rn(v[s][gi][2 + h] * a);
              }
            }
          }
        }
      }
    }
  };

  // FQ: the tile's activation (write_tile's, of BF16-rounded GEMM1 outputs as the CUTLASS path's
  // activation reads them; zero past Nout) staged as BF16 in the ring, free once every warp is past
  // its last step; then two threads per 16-column block quantize it with cvt_warp_fp16_to_fp4 (the
  // CUTLASS path's NVFP4 recipe and fast-math default) and store code x scale. Rows past the
  // block's count are not stored (GEMM2 zero-fills them on load).
  [[maybe_unused]] auto write_tile_fq = [&](int b, int c, float(*v)[G][4], uint32_t slot) {
    if constexpr (FQ && ACT != 0) {
      auto bf16_rn = [](float x) { return __bfloat162float(__float2bfloat16_rn(x)); };
      constexpr int FC = ACT == 1 ? 64 * NC : kTileN * NC;  // the tile's output columns
      constexpr int YP = FC * 2 + 16;                       // staging row pitch (bytes)
      constexpr int UPR = FC / 8, U = R * UPR;              // 16 B units: per row, per tile
      static_assert(R * YP <= static_cast<int>(kRing) && U % 32 == 0 && UPR % 2 == 0, "FQ tile");
      const int e = p.block_expert[b];
      const float a = p.alpha[e] * 128.f;  // products carry 2^-7
      const int Nout = p.Nout, col0 = c * FC;
      // Loaded ahead of the barriers, which they would otherwise follow.
      const float num = p.fq_num[e * p.fq_num_stride];
      const float den = p.fq_den ? p.fq_den[e * p.fq_den_stride] : 1.f;
      const float gs = den == 0.f ? num : num / den;
      const int rows = p.block_rows[b];
      constexpr bool kReg = ACT == 1 && R == 8 && SC == 1 && G == 1 && !KT;
      if constexpr (kReg) {
        // lane (g8, tq) of warp wc: column wc 8 + g8, rows 2 tq + h; a 16-column block is warps
        // (2k, 2k+1). After an 8 x 8 transpose a lane holds row g8, columns 2 tq, 2 tq + 1: elts[0]
        // of cvt_warp_fp16_to_fp4, whose x / y |max| (over elts, then lane tq ^ 1) gets the pair
        // maxima of lanes tq ^ 2 and of the partner warp in elts[1..3] (64-thread barrier, free
        // slot). KT keeps the staged path: its k-loop allocates worse with this one.
        const int i = wc * 8 + g8;
        __nv_bfloat162 yb;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float gt = bf16_rn(v[0][0][h] * a), up = bf16_rn(v[0][0][2 + h] * a);
          const float y = gt / (1.f + __expf(-gt)) * up;
          (h ? yb.y : yb.x) = __float2bfloat16_rn(col0 + i < Nout ? y : 0.f);
        }
        tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 8> vec;
        uint32_t m = movm_trans(*reinterpret_cast<uint32_t*>(&yb));
        vec.elts[0] = *reinterpret_cast<__nv_bfloat162*>(&m);
        auto mx = [](uint32_t x, uint32_t y) {
          const __nv_bfloat162 r = __hmax2(*reinterpret_cast<__nv_bfloat162*>(&x),
                                           *reinterpret_cast<__nv_bfloat162*>(&y));
          return *reinterpret_cast<const uint32_t*>(&r);
        };
        m &= 0x7FFF7FFFu;  // __habs2
        m = mx(__shfl_xor_sync(0xffffffffu, m, 2), m);
        uint32_t* xs = reinterpret_cast<uint32_t*>(smem + slot);
        if (tq < 2) xs[i * 2 + tq] = m;
        asm volatile("bar.sync %0, 64;" ::"r"(1 + (wc >> 1)) : "memory");
        m = mx(xs[((wc ^ 1) * 8 + g8) * 2 + (tq & 1)], m);
        vec.elts[1] = vec.elts[2] = vec.elts[3] = *reinterpret_cast<__nv_bfloat162*>(&m);
        uint8_t sf = 0;
        const uint32_t cc =
            tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 8, false>(
                vec, gs, nullptr, 0.f, &sf);
        uint32_t o[4];
        restore_nvfp4x8<true>(cc & 0xFFu, sf, o);  // o[0]: byte 0, elts[0]
        const int n0 = col0 + wc * 8;
        if (g8 < rows && n0 < Nout)
          *reinterpret_cast<uint32_t*>(p.out + (static_cast<long>(b) * R + g8) * Nout + n0 +
                                       2 * tq) = o[0];
        return;
      }
      // R 8: the ring slot after the last k step (cslot) was last read STAGES steps earlier, behind
      // the loop's last barrier, and no copy targets it: the tile is staged there without a
      // barrier.
      constexpr bool kFree = R == 8;
      static_assert(!kFree || R * YP <= static_cast<int>(kStage), "FQ tile in one stage");
      const uint32_t sl = kFree ? slot : 0u;
      flashinfer::cp_async::wait_group<0>();
      if constexpr (!kFree) __syncthreads();
      __nv_bfloat16* ys = reinterpret_cast<__nv_bfloat16*>(smem + sl);
#pragma unroll
      for (int s = 0; s < SC; ++s) {
        const int ws = wc * SC + s;
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
          const int r0 = wt * TW + gi * 8 + 2 * tq;
          if constexpr (ACT == 1) {
            const int i = ws * 8 + g8;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const float gt = bf16_rn(v[s][gi][h] * a), up = bf16_rn(v[s][gi][2 + h] * a);
              float y;
              if constexpr (SC == 4 || (G1E && R == 96 && STAGES == 2 && !KT))
                y = silu_div(gt, 1.f + __expf(-gt)) * up;
              else
                y = gt / (1.f + __expf(-gt)) * up;
              ys[(r0 + h) * (YP / 2) + i] = __float2bfloat16_rn(col0 + i < Nout ? y : 0.f);
            }
          } else {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int n = ws * 16 + g8 + (q >> 1) * 8;
              const float x = fmaxf(bf16_rn(v[s][gi][q] * a), 0.f);
              ys[(r0 + (q & 1)) * (YP / 2) + n] =
                  __float2bfloat16_rn(col0 + n < Nout ? x * x : 0.f);
            }
          }
        }
      }
      __syncthreads();
      // U and NT are multiples of 32: every lane of a warp reaches the shuffles of a block pair
      for (int u = tid; u < U; u += NT) {
        const int r = u / UPR, q8 = u % UPR;
        tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 8> vec;
        lds128(reinterpret_cast<uint32_t*>(&vec), sbase + sl + r * YP + q8 * 16);
        uint8_t sf = 0;
        const uint32_t codes =
            tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 8, false>(
                vec, gs, nullptr, 0.f, &sf);
        uint4 o;
        restore_nvfp4x8<R == 8>(codes, sf, reinterpret_cast<uint32_t*>(&o));
        const int n = col0 + q8 * 8;
        if (r < rows && n < Nout)
          *reinterpret_cast<uint4*>(p.out + (static_cast<long>(b) * R + r) * Nout + n) = o;
      }
    }
  };

  const uint32_t sw_c = sbase + (wc * SC / 8) * kRec + (wc * SC % 8) * 512 + lane * 16;
  const uint32_t ss_c = sbase + (wc * SC / 8) * kRec + kWBytes + (wc * SC % 8) * 128 + g8 * 16;
  const uint32_t sa_c = sbase + NC * kRec + (wt * TW + (lane & 7)) * kActPitch + (lane >> 3) * 16;
  // Swizzled: a lane's ldmatrix rows wt TW + gi 8 + (lane & 7) share r & 7 =
  // lane & 7, and step j2 reads logical chunk (lane >> 3) + 4 j2: j2 even sits
  // at sa_s0 + (j2 >> 1) 128 and j2 odd (chunk bit 2 set, which the XOR flips)
  // at sa_s1 + (j2 >> 1) 128
  const uint32_t sa_s0 =
      sbase + NC * kRec + (wt * TW + (lane & 7)) * AP + (((lane >> 3)) ^ (lane & 7)) * 16;
  const uint32_t sa_s1 =
      sbase + NC * kRec + (wt * TW + (lane & 7)) * AP + (((lane >> 3) + 4) ^ (lane & 7)) * 16;
  // R 96 at 2 stages with G1E: the row part of the half-fragment address
  const uint32_t sa_base_hf = sbase + NC * kRec + (wt * TW + (lane & 7)) * AP;
  float acc[SC][G][4];
#pragma unroll
  for (int s = 0; s < SC; ++s)
#pragma unroll
    for (int gi = 0; gi < G; ++gi)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[s][gi][q] = 0.f;

  // Rows past the block's count are padding: zero activations whose results are
  // dropped. A warp runs the ldmatrix and MMA of its 8-row groups only up to
  // its last real row, in quarters of its G groups, so the group count is a
  // constant in each copy of the step and acc stays in registers.
  constexpr int Q = (Q12F && R == 96 && STAGES == 2 && !KT && (ACT == 0 || ACT == 1)) ? 12
                    : R == 96                                                         ? 6
                              : (SC > 1 && G % 4 == 0 ? 4 : 1);
  auto step = [&](auto ngc, uint32_t sb, bool hf) {
    constexpr int NG = decltype(ngc)::value;
    if constexpr (SC == 4) {
      // Four 16-column slots per warp (128 fp32 accumulators): a slot's weights
      // (8 B) and scales (4 B) of one k32 are read one k32 ahead, in place of
      // the whole k128 up front. Same operands, same MMA order.
      static_assert(G <= 8 && R == 64 && STAGES == 2,
                    "SC 4 step: the k16 activation fragments of R 64, 2 stages");
      // VS (GEMM2 without a K tail, below sm_90): a slot's four scale words
      // of the k128 are read once with one 16 B load, in place of one 4 B
      // load per k32.
      constexpr bool kVS = !kSm90 && ACT == 0 && !KT && VS;
      uint32_t wn[SC][2], sn[SC];
      uint32_t sc[SC][4];
      if constexpr (kVS) {
#pragma unroll
        for (int s = 0; s < SC; ++s) lds128(sc[s], ss_c + sb + s * 128);
      }
      auto load_ws = [&](int j2) {
#pragma unroll
        for (int s = 0; s < SC; ++s) {
          lds64v(wn[s], sw_c + sb + s * 512 + (j2 >> 1) * kHalfW + (j2 & 1) * 8);
          if constexpr (!kVS) lds32v(sn + s, ss_c + sb + s * 128 + j2 * 4);
        }
      };
      load_ws(0);
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (KT && j2 == 2 && hf) break;  // a 5120 B tail record: k 0..63 only
        uint32_t wq[SC][2], sq[SC];
#pragma unroll
        for (int s = 0; s < SC; ++s) {
          wq[s][0] = wn[s][0];
          wq[s][1] = wn[s][1];
          if constexpr (!kVS) sq[s] = sn[s];
        }
        if (j2 < 3 && !(KT && j2 == 1 && hf)) load_ws(j2 + 1);
        constexpr bool kV3 = kSm90 && ACT != 0;  // sm_90 G1: v3 code (x4 spills there)
        // Without kV3, one ldmatrix x4 per 8-row group and k32: chunks 4 j2 .. 4 j2 + 3
        // of the row (both k16 halves; bf[gi] + 2 h is half h), in place of one x2 per
        // half. Same fragments and MMA order, half the ldmatrix instructions.
        uint32_t bf[NG][kV3 ? 2 : 4];
        if constexpr (!kV3) {
#pragma unroll
          for (int gi = 0; gi < NG; ++gi)
            ldsm_x4(bf[gi], ((j2 & 1) ? sa_s1 : sa_s0) + sb + gi * 8 * AP + (j2 >> 1) * 128);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (kV3) {
#pragma unroll
            for (int gi = 0; gi < NG; ++gi) {
              const uint32_t address = sbase + NC * kRec + (wt * TW + gi * 8 + (lane & 7)) * AP +
                                       (((((lane >> 3) & 1) + 2 * h + 4 * j2) ^ (lane & 7)) * 16);
              ldsm_x2(bf[gi], address + sb);
            }
          }
#pragma unroll
          for (int s = 0; s < SC; ++s) {
            const uint32_t sw = kVS ? sc[s][j2] : sq[s];
            uint32_t a[4];
            dq4(wq[s][h], h ? u5_odd(sw) : u5_even(sw), a);
#pragma unroll
            for (int gi = 0; gi < NG; ++gi) mma_bf16(acc[s][gi], a, bf[gi] + (kV3 ? 0 : 2 * h));
          }
        }
      }
    } else {
      uint32_t wq[SC][8], sq[SC][4];
#pragma unroll
      for (int s = 0; s < SC; ++s) {
        lds128(wq[s], sw_c + sb + s * 512);
        lds128(wq[s] + 4, sw_c + sb + s * 512 + kHalfW);
        lds128(sq[s], ss_c + sb + s * 128);
      }
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (KT && j2 == 2 && hf) break;  // a 5120 B tail record: k 0..63 only
        constexpr bool half_fragment = G > 8 || (R == 64 && STAGES == 2);
        uint32_t bf[NG][half_fragment ? 2 : 4];
        auto load_activation = [&](int h) {
#pragma unroll
          for (int gi = 0; gi < NG; ++gi) {
            if constexpr (half_fragment) {
              const uint32_t address =
                  (R == 96 && STAGES == 2 && G1E
                       ? sa_base_hf + gi * 8 * AP
                       : sbase + NC * kRec + (wt * TW + gi * 8 + (lane & 7)) * AP) +
                  (((((lane >> 3) & 1) + 2 * h + 4 * j2) ^ (lane & 7)) * 16);
              ldsm_x2(bf[gi], address + sb);
            } else if constexpr (SWZ) {
              ldsm_x4(bf[gi], ((j2 & 1) ? sa_s1 : sa_s0) + sb + gi * 8 * AP + (j2 >> 1) * 128);
            } else {
              ldsm_x4(bf[gi], sa_c + sb + gi * 8 * kActPitch + j2 * 64);
            }
          }
        };
        if constexpr (!half_fragment) load_activation(0);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (half_fragment) load_activation(h);
#pragma unroll
          for (int s = 0; s < SC; ++s) {
            uint32_t a[4];
            dq4<R == 96 && STAGES == 2 && G1E>(wq[s][j2 * 2 + h],
                                               h ? u5_odd(sq[s][j2]) : u5_even(sq[s][j2]), a);
#pragma unroll
            for (int gi = 0; gi < NG; ++gi)
              mma_bf16(acc[s][gi], a, bf[gi] + (half_fragment ? 0 : 2 * h));
          }
        }
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < S) issue();
    flashinfer::cp_async::commit_group();
  }
  if constexpr (PT && ACT) {
    // GEMM1's row scales, the reference's FTZ alpha s_t (padding rows 0), read by write_tile past
    // the k loop's barriers
    if (has_act && aq == 0)
#pragma unroll
      for (int j = 0; j < RJ; ++j)
        reinterpret_cast<float*>(smem + kRing)[ar + RP * j] = mul_ftz(pt_a, pt_s[j]);
  }
  uint32_t cslot = 0;
  const int tile = blockIdx.x;
  // quarters of this warp's TW rows that hold a real row of the tile's block
  const int nq =
      SC == 1 ? 1 : min(Q, (max(0, p.block_rows[tile / chunks] - wt * TW) * Q + TW - 1) / TW);
  for (int u = 0; u < S; ++u) {
    flashinfer::cp_async::wait_group<STAGES - 2>();
    __syncthreads();
    if (u + STAGES - 1 < S) issue();
    flashinfer::cp_async::commit_group();
    const uint32_t sb = cslot;
    cslot = cslot + kStage == kRing ? 0u : cslot + kStage;
    bool hf = false;
    if constexpr (KT) hf = half && u == S - 1;
    // clang-format off
#define W4A16_STEP(n) case n: if constexpr (n < Q) { \
        step(std::integral_constant<int, n * G / Q>{}, sb, hf); break; } [[fallthrough]];
    switch (nq) {
      case 0: break;
      W4A16_STEP(1) W4A16_STEP(2) W4A16_STEP(3) W4A16_STEP(4) W4A16_STEP(5) W4A16_STEP(6)
      W4A16_STEP(7) W4A16_STEP(8) W4A16_STEP(9) W4A16_STEP(10) W4A16_STEP(11)
      default: step(std::integral_constant<int, G>{}, sb, hf);
    }
#undef W4A16_STEP
    // clang-format on
  }
  if constexpr (FQ && ACT != 0) {
    write_tile_fq(tile / chunks, tile % chunks, acc, cslot);
  } else {
    write_tile(tile / chunks, tile % chunks, acc);
    if constexpr (FQ) fin_chunk<NT, NC * kTileN, kRing>(p, tile % chunks, nb);
  }
}

template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1, bool G1E = false,
          bool Q12F = false, bool VS = false, bool FQ = false>
__global__ void __launch_bounds__(NW * 32) gemm(GemmParamsT<FQ> p) {
  gemm_body<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS, FQ>(p);
}

// Per-token NVFP4 (PT): gemm's kBase tiles with the row scales; GEMM1 takes R x 4 B of shared
// memory past the ring.
template <int R, int WC, int NW, int ACT, int STAGES, bool KT, bool FQ = false>
__global__ void __launch_bounds__(NW * 32) gemm_pt(GemmParamsT<FQ> p) {
  gemm_body<R, WC, NW, ACT, STAGES, KT, 1, false, false, false, FQ, true>(p);
}

// Per-token NVFP4 GEMM2 input (PT): the real rows of GEMM1's blocked output y [nb x R][kb x 16]
// quantized as nvfp4QuantAndPerTokenScaleKernel (a thread per 16-value block, a row on whole
// warps: the row's maximum |y|, s_r = amax x gs, its blocks under 1 / s_r) and restored in place;
// rs[b R + r] = alpha2[e] x s_r (FTZ, as nvfp4RowScalesKernel) with alpha2 = out2[e] x out1[e]
// rounded as torch's product (out1 [E] or [1]: o1s 1 or 0). CTA (b, z) holds rows z rpc to
// z rpc + rpc - 1 of block b.
__global__ void __launch_bounds__(1024)
    row_quant(__nv_bfloat16* y, const int* __restrict__ meta, int max_blocks, int R, int kb,
              const float* __restrict__ out2, const float* __restrict__ out1, int o1s, float gs,
              float* __restrict__ rs) {
  namespace tk = tensorrt_llm::kernels;
  __shared__ float wmax[32];
  const int b = blockIdx.x, kbp = (kb + 31) & ~31, rpc = blockDim.x / kbp;
  if (b >= meta[0]) return;  // the CTA's block is past the count, or its rows are padding
  const int rows = meta[1 + max_blocks + b], r = blockIdx.y * rpc + threadIdx.x / kbp;
  if (static_cast<int>(blockIdx.y) * rpc >= rows) return;
  const int j = threadIdx.x % kbp, warp = threadIdx.x >> 5;
  const bool live = r < rows && j < kb;
  const int e = meta[1 + b];
  const float a2 = __fmul_rn(__ldg(out2 + e), __ldg(out1 + e * o1s));
  const long i = (static_cast<long>(b) * R + r) * kb + j;
  tk::PackedVec<__nv_bfloat16, 16> v;
  uint4* src = reinterpret_cast<uint4*>(y) + 2 * i;
  reinterpret_cast<uint4*>(&v)[0] = live ? src[0] : make_uint4(0, 0, 0, 0);
  reinterpret_cast<uint4*>(&v)[1] = live ? src[1] : make_uint4(0, 0, 0, 0);
  __nv_bfloat162 m2 = __float2bfloat162_rn(0.f);
#pragma unroll
  for (int k = 0; k < 8; ++k) m2 = __hmax2(m2, __habs2(v.elts[k]));
  float m = fmaxf(0.f, __bfloat162float(__hmax(m2.x, m2.y)));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) wmax[warp] = m;
  __syncthreads();
  if (!live) return;
  const int w0 = threadIdx.x / kbp * kbp / 32;
  for (int w = w0; w < w0 + kbp / 32; ++w) m = fmaxf(m, wmax[w]);
  const float sr = tk::nvfp4_quantize_mul(m, gs);
  const float enc = sr != 0.f ? tk::reciprocal_approximate_ftz(sr) : 0.f;
  uint8_t sf;
  const uint64_t c =
      tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false>(v, enc, nullptr, 0.f, &sf);
  uint4 o[2];
  restore_nvfp4x8(static_cast<uint32_t>(c), sf, reinterpret_cast<uint32_t*>(o));
  restore_nvfp4x8(static_cast<uint32_t>(c >> 32), sf, reinterpret_cast<uint32_t*>(o + 1));
  src[0] = o[0];
  src[1] = o[1];
  if (j == 0) rs[static_cast<long>(b) * R + r] = tk::nvfp4_quantize_mul(a2, sr);
}

// NVFP4 input restored to BF16 for GEMM1 (W4A4): CTAs from cta0 on, each thread one 16-value
// block of x [tokens][K / 2] (E2M1 pairs) with its E4M3 scale sf: [tokens][sf_pitch], or with
// swizzled the 128x4 swizzled layout (sf_pitch = K / 64 tiles of 128 rows x 4 scales).
struct DqArgs {
  const uint8_t* x;                // E2M1 pairs, or (gs) the BF16 input
  const uint8_t* sf;               // its E4M3 scales (not with gs)
  const float* gs;                 // BF16 input: the global scale it is quantized under here
  __nv_bfloat16* out;              // [tokens][K]
  int blocks, kb, sf_pitch, cta0;  // tokens x K / 16, K / 16
  bool swizzled;
  const int* lut;  // prep<R, DQ, true>: ids are slots, lut[slot] their experts
  int* cnt;        // prep (R < 64): the kFinCnt GEMM2 sum counters to zero, or null
  float* ts;       // DQ 3 (per-token BF16 input): [tokens] the token scales s_t, written
};

// One token, BF16 input of at most kPairBlocks blocks (3 warps a scheduler): the recipe's
// two-thread form (PAIR), 8 values a thread and one shuffle for the block maximum, a shorter chain
// a thread. More blocks would make the second thread's scale issue cost more than the chain saves.
constexpr int kPairBlocks = 192;

// BF: the input is BF16, quantized first under d.gs.
template <bool BF, bool PAIR = false>
__device__ __forceinline__ void dequant_input(DqArgs const& d) {
  const int stride = (gridDim.x - d.cta0) * blockDim.x;
  [[maybe_unused]] const float gs = BF ? __ldg(d.gs) : 0.f;
  if constexpr (PAIR) {
    // whole warps (full-mask shuffle); lanes past the end store nothing
    if (d.blocks <= kPairBlocks) {
      const int n = 2 * d.blocks;
      for (int u0 = (blockIdx.x - d.cta0) * blockDim.x + (threadIdx.x & ~31); u0 < n;
           u0 += stride) {
        const int u = u0 + (threadIdx.x & 31);
        tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 8> v;
        reinterpret_cast<uint4*>(&v)[0] =
            u < n ? __ldg(reinterpret_cast<const uint4*>(d.x) + u) : make_uint4(0, 0, 0, 0);
        uint8_t s;
        const uint32_t c = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 8, false>(
            v, gs, nullptr, 0.f, &s);
        uint4 o;
        restore_nvfp4x8(c, s, reinterpret_cast<uint32_t*>(&o));
        if (u < n) reinterpret_cast<uint4*>(d.out)[u] = o;
      }
      return;
    }
  }
  for (int i = (blockIdx.x - d.cta0) * blockDim.x + threadIdx.x; i < d.blocks; i += stride) {
    const int t = i / d.kb, j = i % d.kb;
    uint2 c;
    uint8_t s;
    if constexpr (BF) {  // the input quantizer's recipe (16 values a thread: no shuffle)
      tensorrt_llm::kernels::PackedVec<__nv_bfloat16, 16> v;
      const uint4* src = reinterpret_cast<const uint4*>(d.x) + 2 * static_cast<long>(i);
      reinterpret_cast<uint4*>(&v)[0] = __ldg(src);
      reinterpret_cast<uint4*>(&v)[1] = __ldg(src + 1);
      const uint64_t codes =
          tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false>(v, gs, nullptr,
                                                                                    0.f, &s);
      c = make_uint2(static_cast<uint32_t>(codes), static_cast<uint32_t>(codes >> 32));
    } else {
      c = __ldg(reinterpret_cast<const uint2*>(d.x) + i);
      s = d.sf[d.swizzled ? (static_cast<long>(t / 128) * d.sf_pitch + j / 4) * 512 + t % 32 * 16 +
                                t % 128 / 32 * 4 + j % 4
                          : static_cast<long>(t) * d.sf_pitch + j];
    }
    uint4 o[2];
    restore_nvfp4x8(c.x, s, reinterpret_cast<uint32_t*>(o));
    restore_nvfp4x8(c.y, s, reinterpret_cast<uint32_t*>(o + 1));
    uint4* dst = reinterpret_cast<uint4*>(d.out) + 2 * static_cast<long>(i);
    dst[0] = o[0];
    dst[1] = o[1];
  }
}

// Per-token BF16 input (DQ 3), as nvfp4QuantAndPerTokenScaleKernel: a thread per 16-value block,
// a token on whole warps (kb <= blockDim), blockDim / kbp tokens a pass. The token's maximum |x|
// (NaN ignored: any order), s_t = amax x gs (gs the base inverse scale), its blocks quantized under
// 1 / s_t (0 for s_t 0) and restored; ts[t] = s_t. pt_max: 128 B of dynamic shared memory.
__device__ __forceinline__ void dequant_pt(DqArgs const& d) {
  namespace tk = tensorrt_llm::kernels;
  extern __shared__ float pt_max[];  // [warps] the warps' maxima
  const int kbp = (d.kb + 31) & ~31, tpp = blockDim.x / kbp, tokens = d.blocks / d.kb;
  const int lt = threadIdx.x / kbp, j = threadIdx.x % kbp, warp = threadIdx.x >> 5;
  const float gs = __ldg(d.gs);
  // the same t0 across the CTA: every thread reaches the barriers
  for (int t0 = (blockIdx.x - d.cta0) * tpp; t0 < tokens; t0 += (gridDim.x - d.cta0) * tpp) {
    const int t = t0 + lt;
    const bool live = lt < tpp && t < tokens && j < d.kb;
    const long i = static_cast<long>(t) * d.kb + j;
    tk::PackedVec<__nv_bfloat16, 16> v;
    const uint4* src = reinterpret_cast<const uint4*>(d.x) + 2 * i;
    reinterpret_cast<uint4*>(&v)[0] = live ? __ldg(src) : make_uint4(0, 0, 0, 0);
    reinterpret_cast<uint4*>(&v)[1] = live ? __ldg(src + 1) : make_uint4(0, 0, 0, 0);
    __nv_bfloat162 m2 = __float2bfloat162_rn(0.f);
#pragma unroll
    for (int k = 0; k < 8; ++k) m2 = __hmax2(m2, __habs2(v.elts[k]));
    float m = fmaxf(0.f, __bfloat162float(__hmax(m2.x, m2.y)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) pt_max[warp] = m;
    __syncthreads();
    if (live) {
      for (int w = lt * kbp / 32; w < (lt + 1) * kbp / 32; ++w) m = fmaxf(m, pt_max[w]);
      const float s = tk::nvfp4_quantize_mul(m, gs);
      const float enc = s != 0.f ? tk::reciprocal_approximate_ftz(s) : 0.f;
      uint8_t sf;
      const uint64_t c =
          tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 16, false>(v, enc, nullptr, 0.f, &sf);
      uint4 o[2];
      restore_nvfp4x8(static_cast<uint32_t>(c), sf, reinterpret_cast<uint32_t*>(o));
      restore_nvfp4x8(static_cast<uint32_t>(c >> 32), sf, reinterpret_cast<uint32_t*>(o + 1));
      uint4* dst = reinterpret_cast<uint4*>(d.out) + 2 * i;
      dst[0] = o[0];
      dst[1] = o[1];
      if (j == 0) d.ts[t] = s;
    }
    __syncthreads();  // the next pass rewrites pt_max
  }
}

// One-CTA prep: warp w < W counts entries [w Q, (w + 1) Q) per expert in its own table row.
// DQ: CTAs past the first restore the NVFP4 input instead (2: quantize the BF16 input first; 3:
// per token, dequant_pt).
// ONE (one token, M <= 32): a one-row block per entry in entry order, its experts being distinct;
// a BF16 input may take the PAIR restore.
template <int R, int DQ = 0, bool LUT = false, bool ONE = false>
__global__ void __launch_bounds__(1024, 2)
    prep(const int* __restrict__ ids, int M, int E, int W, int Q, int* nblocks, int* block_expert,
         int* block_rows, int* rowmap, DqArgs dq) {
  if constexpr (DQ == 3) {
    if (blockIdx.x >= dq.cta0) {
      dequant_pt(dq);
      return;
    }
  } else if constexpr (DQ) {
    if (blockIdx.x >= dq.cta0) {
      dequant_input<DQ == 2, ONE && DQ == 2>(dq);
      return;
    }
  }
  if (dq.cnt && threadIdx.x < kFinCnt) dq.cnt[threadIdx.x] = 0;
  if constexpr (ONE) {
    const int t = threadIdx.x;
    for (int i = t; i < M * R; i += blockDim.x)
      if (i % R) rowmap[i] = -1;
    if (t < 32) {
      const int e = t < M ? (LUT ? dq.lut[ids[t]] : ids[t]) : -1;
      const unsigned valid = __ballot_sync(0xffffffffu, e >= 0);  // a negative id has no slot
      const int b = __popc(valid & ((1u << t) - 1u));
      if (e >= 0) {
        block_expert[b] = e;
        block_rows[b] = 1;
        rowmap[b * R] = t;
      }
      if (t == 0) *nblocks = __popc(valid);
    }
    return;
  }
  extern __shared__ int sh[];
  int* tab = sh;          // [W][E]: warp w's entries of expert e, then its next rank
  int* cnt = sh + W * E;  // [E]
  int* off = cnt + E;     // [E] first block of each expert
  __shared__ int wsum[32];
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < W * E; i += nt) tab[i] = 0;
  __syncthreads();
  const int f0 = warp * Q, f1 = min(f0 + Q, M);
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? (LUT ? dq.lut[ids[f]] : ids[f]) : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && lane == __ffs(peers) - 1) tab[warp * E + e] += __popc(peers);
      __syncwarp();  // the next 32 may give this expert another leader lane
    }
  }
  __syncthreads();
  // This thread's experts [e0, e1): counts, warp prefixes, blocks; contiguous
  // runs keep expert order.
  const int per = (E + 1023) / 1024;  // the host launches !ONE prep with 1024 threads
  const int e0 = min(tid * per, E), e1 = min(e0 + per, E);
  int v = 0;
  for (int e = e0; e < e1; ++e) {
    int c = 0;
    for (int w = 0; w < W; ++w) {
      const int t = tab[w * E + e];
      tab[w * E + e] = c;
      c += t;
    }
    cnt[e] = c;
    v += (c + R - 1) / R;
  }
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (tid < 32) {
    int s = tid < static_cast<int>(nt >> 5) ? wsum[tid] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, o);
      if (tid >= o) s += y;
    }
    wsum[tid] = s;
  }
  __syncthreads();
  const int excl = x - v + (warp > 0 ? wsum[warp - 1] : 0);
  int b = excl;
  for (int e = e0; e < e1; ++e) {
    const int c = cnt[e], n = (c + R - 1) / R;
    off[e] = b;
    for (int bb = 0; bb < n; ++bb) {
      const int rows = min(R, c - bb * R);
      block_expert[b + bb] = e;
      block_rows[b + bb] = rows;
      for (int r = rows; r < R; ++r) rowmap[(b + bb) * R + r] = -1;
    }
    b += n;
  }
  if (tid == nt - 1) *nblocks = excl + v;
  __syncthreads();
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? (LUT ? dq.lut[ids[f]] : ids[f]) : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0) {
        const int q = tab[warp * E + e] + __popc(peers & ((1u << lane) - 1u));
        rowmap[(off[e] + q / R) * R + q % R] = f;
      }
      __syncwarp();
      if (e >= 0 && lane == __ffs(peers) - 1) tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
}

// Multi-CTA prep, the same blocks and rowmap as prep (entries of an expert in f
// order): CTA c of nC owns entries [c CH, (c + 1) CH). prepx_count: per-chunk
// expert counts pc[c][E]. prepx_scan (one CTA): pc becomes each chunk's first
// rank within its expert, blocks as in prep, off[e] = the expert's first block.
// prepx_place: prep's two passes over the chunk (warp w owns [f0 + w Q, f0 + (w
// + 1) Q)), ranks offset by the chunk's base.
template <int DQ = 0>
__global__ void __launch_bounds__(1024, 2) prepx_count(const int* __restrict__ ids, int M, int E,
                                                       int CH, int* __restrict__ pc, DqArgs dq) {
  if constexpr (DQ == 3) {
    if (blockIdx.x >= dq.cta0) {
      dequant_pt(dq);
      return;
    }
  } else if constexpr (DQ) {
    if (blockIdx.x >= dq.cta0) {
      dequant_input<DQ == 2>(dq);
      return;
    }
  }
  extern __shared__ int hist[];
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int i = tid; i < E; i += nt) hist[i] = 0;
  __syncthreads();
  const int f0 = blockIdx.x * CH, f1 = min(f0 + CH, M);
  for (int f = f0 + tid; f < f1; f += nt) {
    const int e = ids[f];
    if (e >= 0) atomicAdd(hist + e, 1);  // as prep: a negative id has no slot
  }
  __syncthreads();
  for (int i = tid; i < E; i += nt) pc[blockIdx.x * E + i] = hist[i];
}

template <int R>
__global__ void __launch_bounds__(1024)
    prepx_scan(int E, int nC, int* __restrict__ pc, int* __restrict__ off, int* nblocks,
               int* block_expert, int* block_rows, int* rowmap) {
  extern __shared__ int cnt[];  // [E]
  __shared__ int wsum[32];
  // the host's one launch uses 1024 threads
  const int tid = threadIdx.x, nt = 1024, warp = tid >> 5, lane = tid & 31;
  const int per = (E + nt - 1) / nt;
  const int e0 = min(tid * per, E), e1 = min(e0 + per, E);
  int v = 0;
  for (int e = e0; e < e1; ++e) {
    int c = 0;
    for (int k0 = 0; k0 < nC; k0 += 8) {  // 8 chunk counts loaded before their ranks are stored
      int t[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) t[j] = k0 + j < nC ? pc[(k0 + j) * E + e] : 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (k0 + j < nC) {
          pc[(k0 + j) * E + e] = c;
          c += t[j];
        }
    }
    cnt[e] = c;
    v += (c + R - 1) / R;
  }
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[warp] = x;
  __syncthreads();
  if (tid < 32) {
    int s = tid < static_cast<int>(nt >> 5) ? wsum[tid] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, o);
      if (tid >= o) s += y;
    }
    wsum[tid] = s;
  }
  __syncthreads();
  const int excl = x - v + (warp > 0 ? wsum[warp - 1] : 0);
  int b = excl;
  for (int e = e0; e < e1; ++e) {
    const int c = cnt[e], n = (c + R - 1) / R;
    off[e] = b;
    for (int bb = 0; bb < n; ++bb) {
      const int rows = min(R, c - bb * R);
      block_expert[b + bb] = e;
      block_rows[b + bb] = rows;
      for (int r = rows; r < R; ++r) rowmap[(b + bb) * R + r] = -1;
    }
    b += n;
  }
  if (tid == nt - 1) *nblocks = excl + v;
}

template <int R>
__global__ void __launch_bounds__(1024)
    prepx_place(const int* __restrict__ ids, int M, int E, int CH, int W, int Q,
                const int* __restrict__ pc, const int* __restrict__ off, int* __restrict__ rowmap) {
  extern __shared__ int sh[];
  int* tab = sh;           // [W][E]: warp w's entries of expert e, then its next rank
  int* base = sh + W * E;  // [E]: the expert's first row (off R) plus this
                           // chunk's first rank
  const int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < W * E; i += nt) tab[i] = 0;
  __syncthreads();
  const int c0 = blockIdx.x * CH, c1 = min(c0 + CH, M);
  const int f0 = c0 + warp * Q, f1 = min(f0 + Q, c1);
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && lane == __ffs(peers) - 1) tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
  __syncthreads();
  for (int e = tid; e < E; e += nt) {
    int c = 0;
    for (int w = 0; w < W; ++w) {
      const int t = tab[w * E + e];
      tab[w * E + e] = c;
      c += t;
    }
    base[e] = off[e] * R + pc[blockIdx.x * E + e];
  }
  __syncthreads();
  if (warp < W) {
    for (int c = f0; c < f1; c += 32) {
      const int f = c + lane;
      const int e = f < f1 ? ids[f] : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0) rowmap[base[e] + tab[warp * E + e] + __popc(peers & ((1u << lane) - 1u))] = f;
      __syncwarp();
      if (e >= 0 && lane == __ffs(peers) - 1) tab[warp * E + e] += __popc(peers);
      __syncwarp();
    }
  }
}

// The top-k sum (fin_sum), a thread per 8 columns of a token.
template <typename WT, int U>
__global__ void finalize(const __nv_bfloat16* __restrict__ y, const WT* __restrict__ w,
                         const int* __restrict__ ids, __nv_bfloat16* __restrict__ out, int tokens,
                         int top_k, int H) {
  const int h8 = H / 8;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tokens * h8) return;
  fin_sum<WT, U>(y, w, ids, out, idx / h8, idx % h8, top_k, H);
}

template <cudaDeviceAttr A>
int dev_attr() {  // attribute A of the current device, cached per device
  static int cache[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (cache[dev] == 0) cudaDeviceGetAttribute(&cache[dev], A, dev);
  return cache[dev];
}
int max_smem_optin() { return dev_attr<cudaDevAttrMaxSharedMemoryPerBlockOptin>(); }
int max_smem_per_sm() { return dev_attr<cudaDevAttrMaxSharedMemoryPerMultiprocessor>(); }

template <auto K, int SMEM>
int ctas_per_sm() {  // resident 256-thread CTAs per SM of kernel K at SMEM bytes, cached per device
  static int cache[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (cache[dev] == 0) {
    cudaFuncSetAttribute(K, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cache[dev], K, 256, SMEM);
  }
  return cache[dev];
}

template <int R, int WC, int NW, int ACT, int STAGES, bool KT, int NC = 1, bool G1E = false,
          bool Q12F = false, bool VS = false, bool FQ = false>
void launch_gemm(GemmParamsT<FQ> const& p, int grid, cudaStream_t st) {
  static_assert(NC > 1 || STAGES * Cfg<R, NC>::kStageBytes <= kSmemLimit);
  const int smem = STAGES * Cfg<R, NC>::kStageBytes;
  TVM_FFI_ICHECK(smem <= max_smem_optin()) << "W4A16 MoE GEMM: tile exceeds device shared memory";
  cudaFuncSetAttribute(gemm<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS, FQ>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm<R, WC, NW, ACT, STAGES, KT, NC, G1E, Q12F, VS, FQ><<<grid, NW * 32, smem, st>>>(p);
}

template <int R, int WC, int NW, int ACT, int STAGES, bool KT, bool FQ = false>
void launch_gemm_pt(GemmParamsT<FQ> const& p, int grid, cudaStream_t st) {
  const int smem = STAGES * Cfg<R, 1>::kStageBytes + (ACT ? R * 4 : 0);  // GEMM1's row scales
  TVM_FFI_ICHECK(smem <= max_smem_optin()) << "W4A16 MoE GEMM: tile exceeds device shared memory";
  cudaFuncSetAttribute(gemm_pt<R, WC, NW, ACT, STAGES, KT, FQ>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm_pt<R, WC, NW, ACT, STAGES, KT, FQ><<<grid, NW * 32, smem, st>>>(p);
}

// PT: launch_tile's kBase tiles (FQ: the R 8 GEMM2 with the top-k sum).
template <int ACT, bool KT, bool FQ = false>
void launch_tile_pt(int R, GemmParamsT<FQ> const& p, int grid, cudaStream_t st) {
  if constexpr (FQ)
    launch_gemm_pt<8, 8, 8, 0, 4, KT, true>(p, grid, st);
  else if (R == 8) {  // 3 stages where they hold more CTAs per SM than 4 (SM80: 4 against 3)
    constexpr int s = Cfg<8, 1>::kStageBytes, rs = ACT ? 32 : 0;
    if (ctas_per_sm<gemm_pt<8, 8, 8, ACT, 3, KT, false>, 3 * s + rs>() >
        ctas_per_sm<gemm_pt<8, 8, 8, ACT, 4, KT, false>, 4 * s + rs>())
      launch_gemm_pt<8, 8, 8, ACT, 3, KT>(p, grid, st);
    else
      launch_gemm_pt<8, 8, 8, ACT, 4, KT>(p, grid, st);
  } else if (R == 16)
    launch_gemm_pt<16, 8, 8, ACT, 2, KT>(p, grid, st);
  else if (R == 32) {
    if (max_smem_per_sm() <= 100 * 1024)
      launch_gemm_pt<32, 8, 8, ACT, 2, KT>(p, grid, st);
    else
      launch_gemm_pt<32, 8, 8, ACT, 4, KT>(p, grid, st);
  } else
    launch_gemm_pt<64, 4, 4, ACT, 3, KT>(p, grid, st);
}

// R 32 at 2 stages where an SM has at most 100 KiB; R 64 at 3 stages fits two CTAs per SM.
template <int ACT, bool KT, bool G1S2 = false, bool G1E = false, bool Q12F = false, bool FQ = false>
void launch_tile(int R, GemmParamsT<FQ> const& p, int grid, cudaStream_t st) {
  if (R == 8) {  // as launch_tile_pt
    constexpr int s = Cfg<8, 1>::kStageBytes;
    if (ctas_per_sm<gemm<8, 8, 8, ACT, 3, KT, 1, false, false, false, FQ>, 3 * s>() >
        ctas_per_sm<gemm<8, 8, 8, ACT, 4, KT, 1, false, false, false, FQ>, 4 * s>())
      launch_gemm<8, 8, 8, ACT, 3, KT, 1, false, false, false, FQ>(p, grid, st);
    else
      launch_gemm<8, 8, 8, ACT, 4, KT, 1, false, false, false, FQ>(p, grid, st);
  } else if (R == 16)
    launch_gemm<16, 8, 8, ACT, 2, KT, 1, false, false, false, FQ>(p, grid, st);
  else if (R == 32) {
    if (max_smem_per_sm() <= 100 * 1024)
      launch_gemm<32, 8, 8, ACT, 2, KT, 1, false, false, false, FQ>(p, grid, st);
    else
      launch_gemm<32, 8, 8, ACT, 4, KT, 1, false, false, false, FQ>(p, grid, st);
  } else if (R == 96) {
    if (max_smem_optin() >= 3 * Cfg<96, 2>::kStageBytes)
      launch_gemm<96, 8, 8, ACT, 3, KT, 2, G1E, Q12F, false, FQ>(p, grid, st);
    else
      launch_gemm<96, 8, 8, ACT, 2, KT, 2, G1E, Q12F, false, FQ>(p, grid, st);
  } else
    launch_gemm<64, 4, 4, ACT, G1S2 && ACT != 0 ? 2 : 3, KT, 1, false, false, false, FQ>(p, grid,
                                                                                         st);
}

template <int R>
int prep_static_smem() {  // prep's own __shared__ bytes (wsum)
  static const int s = [] {
    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, prep<R>);
    return static_cast<int>(a.sharedSizeBytes);
  }();
  return s;
}

constexpr int kPrepChunk = 2048;  // multi-CTA prep: routed entries per CTA

// GEMM configs (see the file comment); kStage2 (GEMM1), kG2Lin (GEMM2): R 64 at 2 stages.
enum Variant : int64_t {
  kBase = 0,
  kStage2 = 1,
  kG2Lin = 2,
  kW64s = 3,
  kG2W64x = 4,
  kG2W64v = 5,
  kX96 = 6,
  kQ12 = 7
};

}  // namespace flashinfer::w4a16_moe_sm80

using namespace flashinfer::w4a16_moe_sm80;

// topk_ids [tokens, top_k] int32 -> meta: [0] block count, [1, 1 + max_blocks)
// each block's expert, [1 + max_blocks, 1 + 2 max_blocks) its row count; rowmap
// [max_blocks x row_tile]: each block row's entry (token * top_k + slot) or -1.
// Row tiles 64 and 96 take the multi-CTA prep, whose per-chunk expert counts and
// bases live in workspace (at least (ceil(M / 2048) + 1) x num_experts int32).
// NVFP4 input (W4A4): x [tokens][K / 2] uint8 E2M1 pairs and x_sf E4M3, [tokens][>= K / 16] in
// row-major order or (sf_swizzled) 128x4 swizzled, are restored into x_out [tokens][K] BF16 by
// extra CTAs of the first prep kernel. A BF16 x [tokens][K] is quantized there first, under the
// FP32 global scale x_sf [1]. With slot_experts (with x, row_tile < 64) topk_ids are slots,
// slot_experts[slot] their experts. With x_ts (FP32 [tokens], a BF16 x) the BF16 input is
// quantized per token (x_sf the base inverse scale) and x_ts receives the token scales.
void w4a16_moe_sm80_prep(TensorView topk_ids, TensorView meta, TensorView rowmap,
                         TensorView workspace, int64_t num_experts, int64_t row_tile,
                         int64_t max_blocks, tvm::ffi::Optional<TensorView> x,
                         tvm::ffi::Optional<TensorView> x_sf, tvm::ffi::Optional<TensorView> x_out,
                         bool sf_swizzled, tvm::ffi::Optional<TensorView> slot_experts,
                         tvm::ffi::Optional<TensorView> x_ts) {
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(meta, dl_int32);
  CHECK_INPUT_AND_TYPE(rowmap, dl_int32);
  CHECK_INPUT_AND_TYPE(workspace, dl_int32);
  TVM_FFI_ICHECK(row_tile == 8 || row_tile == 16 || row_tile == 32 || row_tile == 64 ||
                 row_tile == 96)
      << "row_tile must be 8, 16, 32, 64 or 96";
  const int R = static_cast<int>(row_tile);
  const int M = static_cast<int>(topk_ids.numel());
  const int e = static_cast<int>(num_experts);
  const int nb = static_cast<int>(max_blocks);
  TVM_FFI_ICHECK(e >= 1) << "num_experts must be positive";
  const int q = std::min(e, M);  // nonempty experts: a block each, a further one per R more rows
  TVM_FFI_ICHECK(nb >= q + (M - q) / R) << "max_blocks is below the routing bound";
  TVM_FFI_ICHECK(meta.numel() >= 1 + 2 * max_blocks && rowmap.numel() >= max_blocks * row_tile)
      << "meta or rowmap is too small";
  ffi::CUDADeviceGuard device_guard(topk_ids.device().device_id);
  const cudaStream_t st = get_stream(topk_ids.device());
  const int max_smem = max_smem_optin();
  const int* ids = static_cast<const int*>(topk_ids.data_ptr());
  int* m = static_cast<int*>(meta.data_ptr());
  int* rm = static_cast<int*>(rowmap.data_ptr());
  const bool dq = x.has_value();
  TVM_FFI_ICHECK(dq == x_sf.has_value() && dq == x_out.has_value())
      << "x, x_sf and x_out go together";
  DqArgs d{};
  int dq_ctas = 0;
  if (dq) {
    TensorView xv = x.value(), sv = x_sf.value(), ov = x_out.value();
    const bool bf16 = xv.dtype() == dl_bfloat16;
    if (!bf16) {
      CHECK_INPUT_AND_TYPE(xv, dl_uint8);
    }
    CHECK_INPUT_AND_TYPE(ov, dl_bfloat16);
    CHECK_INPUT(sv);
    CHECK_DIM(2, xv);
    if (!sf_swizzled && !bf16) CHECK_DIM(2, sv);
    CHECK_DIM(2, ov);
    const int64_t tokens = ov.size(0), K = ov.size(1);
    const int64_t tiles = (K / 16 + 3) / 4;  // swizzled: 128-row x 4-scale tiles per row band
    if (bf16) {
      CHECK_INPUT(xv);
      TVM_FFI_ICHECK(sv.dtype() == dl_float32 && sv.numel() == 1 && !sf_swizzled)
          << "BF16 input: x_sf must be the FP32 global scale";
      TVM_FFI_ICHECK(K % 16 == 0 && xv.size(0) == tokens && xv.size(1) == K)
          << "BF16 input: x and x_out [tokens, K]";
      d.gs = static_cast<const float*>(sv.data_ptr());
    } else {
      TVM_FFI_ICHECK(sv.dtype().bits == 8) << "x_sf must hold one byte per scale";
      TVM_FFI_ICHECK(K % 16 == 0 && xv.size(0) == tokens && xv.size(1) == K / 2 &&
                     (sf_swizzled ? sv.numel() >= (tokens + 127) / 128 * 128 * tiles * 4
                                  : sv.size(0) == tokens && sv.size(1) >= K / 16))
          << "NVFP4 input: x [tokens, K / 2], x_sf [tokens, >= K / 16] (or 128x4 swizzled), "
             "x_out [tokens, K]";
      d.sf = static_cast<const uint8_t*>(sv.data_ptr());
      d.sf_pitch = static_cast<int>(sf_swizzled ? tiles : sv.size(1));
    }
    TVM_FFI_ICHECK(tokens * (K / 16) <= INT32_MAX) << "NVFP4 input is too large";
    d.x = static_cast<const uint8_t*>(xv.data_ptr());
    d.out = static_cast<__nv_bfloat16*>(ov.data_ptr());
    d.blocks = static_cast<int>(tokens * (K / 16));
    d.kb = static_cast<int>(K / 16);
    d.swizzled = sf_swizzled;
    TVM_FFI_ICHECK((reinterpret_cast<uintptr_t>(d.x) & (bf16 ? 15 : 7)) == 0 &&
                   (reinterpret_cast<uintptr_t>(d.out) & 15) == 0)
        << "NVFP4 input: x must be 8 B (BF16: 16 B) and x_out 16 B aligned";
    dq_ctas = std::max(1, (d.blocks + 4095) / 4096);  // four blocks a thread
    if (x_ts.has_value()) {
      TensorView tv = x_ts.value();
      CHECK_INPUT_AND_TYPE(tv, dl_float32);
      TVM_FFI_ICHECK(bf16 && tv.numel() == tokens && d.kb <= 1024)
          << "x_ts: FP32 [tokens] with a BF16 x of K <= 16384";
      d.ts = static_cast<float*>(tv.data_ptr());
      // whole warps a token, 1024 threads a CTA, four passes a CTA
      const int tpp = 1024 / ((d.kb + 31) / 32 * 32);
      dq_ctas = std::max(1, static_cast<int>((tokens + 4 * tpp - 1) / (4 * tpp)));
    }
  }
  TVM_FFI_ICHECK(dq || !x_ts.has_value()) << "x_ts goes with x";
  if (slot_experts.has_value()) {
    CHECK_INPUT_AND_TYPE(slot_experts.value(), dl_int32);
    TVM_FFI_ICHECK(dq && R < 64) << "slot_experts go with x, row_tile < 64";
    d.lut = static_cast<const int*>(slot_experts.value().data_ptr());
  }

  if (R < 64) {
    if (workspace.numel() >= kFinCnt) d.cnt = static_cast<int*>(workspace.data_ptr());
    const int stat = R == 8    ? prep_static_smem<8>()
                     : R == 16 ? prep_static_smem<16>()
                               : prep_static_smem<32>();
    const int tab_ints = (max_smem - stat) / static_cast<int>(sizeof(int));  // [W][E] + 2 x [E]
    TVM_FFI_ICHECK(3 * e <= tab_ints) << "num_experts is too large for the shared table";
    const int W = std::max(1, std::min({32, tab_ints / e - 2, (M + 31) / 32}));
    const int Q = ((M + W - 1) / W + 31) / 32 * 32;
    const int smem = (W + 2) * e * static_cast<int>(sizeof(int));
    // One token (topk_ids [1, top_k]): its entries need no expert table, and the CTAs a thread
    // per restore unit (two a block for PAIR), at least the routing warp.
    const bool one = topk_ids.ndim() == 2 && topk_ids.size(0) == 1 && M <= 32 && nb >= M &&
                     d.blocks <= INT32_MAX / 2;
    const int units =
        one ? std::min(1024, (d.gs && !d.ts && d.blocks <= kPairBlocks ? 2 : 1) * d.blocks) : 0;
    const int nt = one ? std::max(32, (units + 31) / 32 * 32) : 1024;
    // DQ 3: at least dequant_pt's 128 B
    const int dyn = d.ts ? std::max(one ? 0 : smem, 128) : one ? 0 : smem;
    auto run = [&](auto kern) {
      // the 48 KB default bounds dynamic plus static shared memory
      if (!one && dyn + stat > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
      d.cta0 = 1;
      kern<<<1 + dq_ctas, nt, dyn, st>>>(ids, M, e, W, Q, m, m + 1, m + 1 + nb, rm, d);
    };
    auto pick = [&](auto tile) {
      constexpr int RT = decltype(tile)::value;
      auto go = [&](auto o) {
        constexpr bool O = decltype(o)::value;
        !dq     ? run(prep<RT, 0, false, O>)
        : d.ts  ? (d.lut ? run(prep<RT, 3, true, O>) : run(prep<RT, 3, false, O>))
        : d.lut ? (d.gs ? run(prep<RT, 2, true, O>) : run(prep<RT, 1, true, O>))
        : d.gs  ? run(prep<RT, 2, false, O>)
                : run(prep<RT, 1, false, O>);
      };
      one ? go(std::true_type{}) : go(std::false_type{});
    };
    if (R == 8)
      pick(std::integral_constant<int, 8>{});
    else if (R == 16)
      pick(std::integral_constant<int, 16>{});
    else
      pick(std::integral_constant<int, 32>{});
    return;
  }

  const int ch = kPrepChunk;
  const int nC = std::max(1, (M + ch - 1) / ch);
  TVM_FFI_ICHECK(workspace.numel() >= static_cast<int64_t>(nC + 1) * e) << "workspace is too small";
  const int ints = max_smem / static_cast<int>(sizeof(int)) - 64;  // [W][E] + [E], wsum aside
  TVM_FFI_ICHECK(2 * e <= ints) << "num_experts is too large for the shared table";
  const int W = std::max(1, std::min({32, ints / e - 1, (ch + 31) / 32}));
  const int Q = ((ch + W - 1) / W + 31) / 32 * 32;
  int* pc = static_cast<int*>(workspace.data_ptr());
  int* off = pc + static_cast<long>(nC) * e;
  const int sm_e = e * static_cast<int>(sizeof(int)), sm_p = (W + 1) * sm_e;
  auto big = [&](auto kern, int smem) {
    if (smem > 48 * 1024 - 256)
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  };
  big(prepx_count<0>, sm_e);
  const int sm_q = d.ts ? std::max(sm_e, 128) : sm_e;  // DQ 3: dequant_pt's 128 B
  if (dq)
    d.ts ? big(prepx_count<3>, sm_q) : d.gs ? big(prepx_count<2>, sm_e) : big(prepx_count<1>, sm_e);
  d.cta0 = nC;
  auto route = [&](auto tile) {
    constexpr int RT = decltype(tile)::value;
    big(prepx_scan<RT>, sm_e);
    big(prepx_place<RT>, sm_p);
    if (!dq)
      prepx_count<<<nC, 1024, sm_e, st>>>(ids, M, e, ch, pc, d);
    else if (d.ts)
      prepx_count<3><<<nC + dq_ctas, 1024, sm_q, st>>>(ids, M, e, ch, pc, d);
    else if (d.gs)
      prepx_count<2><<<nC + dq_ctas, 1024, sm_e, st>>>(ids, M, e, ch, pc, d);
    else
      prepx_count<1><<<nC + dq_ctas, 1024, sm_e, st>>>(ids, M, e, ch, pc, d);
    prepx_scan<RT><<<1, 1024, sm_e, st>>>(e, nC, pc, off, m, m + 1, m + 1 + nb, rm);
    prepx_place<RT><<<nC, 1024, sm_p, st>>>(ids, M, e, ch, W, Q, pc, off, rm);
  };
  if (R == 96)
    route(std::integral_constant<int, 96>{});
  else
    route(std::integral_constant<int, 64>{});
}

// One GEMM over the routed blocks. GEMM1 (activation 1: silu(gate) * up, 2:
// relu(x)^2) reads act [tokens][K] through rowmap and writes the blocked
// intermediate [max_blocks x row_tile][Nout]; GEMM2 (activation 0) reads that
// and writes [tokens * top_k][Nout]. records: [E][N / 128 chunks][k steps]; N
// is the record column count, a multiple of 128 (gated: gate and up each padded
// to 64). alpha: [E] per-expert global scales. variant: a Variant. fc2_num (GEMM1 only): the
// activation is quantized to NVFP4 under the global scale fc2_num / fc2_den (den absent or zero:
// fc2_num), each float32 [E] or [1], and written as BF16 code x scale. topk_weights (GEMM2):
// sum_out = out's top-k sum with topk_ids (finalize); counters: the prep's workspace. row_scales
// (per-token NVFP4, variant kBase, row_tile <= 64, relu2 or GEMM2): GEMM1 the token scales s_t
// [tokens] (a row scale alpha s_t), GEMM2 row_quant's [max_blocks x row_tile] (alpha unused).
void w4a16_moe_sm80_finalize(TensorView y, TensorView topk_weights, TensorView topk_ids,
                             TensorView out, int64_t top_k);
void w4a16_moe_sm80_gemm(
    TensorView records, TensorView alpha, TensorView act, TensorView meta, TensorView rowmap,
    TensorView out, int64_t row_tile, int64_t activation, int64_t top_k, int64_t N, int64_t K,
    int64_t Nout, int64_t max_blocks, int64_t variant, tvm::ffi::Optional<TensorView> fc2_num,
    tvm::ffi::Optional<TensorView> fc2_den, tvm::ffi::Optional<TensorView> topk_weights,
    tvm::ffi::Optional<TensorView> topk_ids, tvm::ffi::Optional<TensorView> sum_out,
    tvm::ffi::Optional<TensorView> counters, tvm::ffi::Optional<TensorView> row_scales) {
  CHECK_INPUT_AND_TYPE(records, dl_uint8);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(act, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  TVM_FFI_ICHECK(N % kTileN == 0 && K % 16 == 0 && Nout % 8 == 0 && Nout <= N)
      << "W4A16 MoE GEMM: shape";
  TVM_FFI_ICHECK(row_tile == 8 || row_tile == 16 || row_tile == 32 || row_tile == 64 ||
                 row_tile == 96)
      << "row_tile must be 8, 16, 32, 64 or 96";
  TVM_FFI_ICHECK(activation >= 0 && activation <= 2) << "activation must be 0, 1 or 2";
  const bool g1 = activation != 0;
  TVM_FFI_ICHECK(
      variant == kBase || (variant == kStage2 && row_tile == 64) ||
      ((variant == kG2Lin || variant == kG2W64x || variant == kG2W64v) && row_tile == 64 && !g1) ||
      (variant == kW64s && row_tile == 64 && g1) ||
      ((variant == kX96 || variant == kQ12) && row_tile == 96))
      << "variant " << variant << " does not run row_tile " << row_tile << ", activation "
      << activation;
  ffi::CUDADeviceGuard device_guard(act.device().device_id);
  const cudaStream_t st = get_stream(act.device());
  const int* m = static_cast<const int*>(meta.data_ptr());
  GemmParamsFq q{};
  GemmParams& p = q;
  p.w = static_cast<const uint8_t*>(records.data_ptr());
  p.alpha = static_cast<const float*>(alpha.data_ptr());
  p.act = static_cast<const __nv_bfloat16*>(act.data_ptr());
  p.nblocks = m;
  p.block_expert = m + 1;
  p.block_rows = m + 1 + max_blocks;
  p.rowmap = static_cast<const int*>(rowmap.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  if (g1) p.topk_dm = cutlass::FastDivmod(static_cast<int>(top_k));  // G1 only; G2 keeps identity
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);
  p.Nout = static_cast<int>(Nout);
  const bool fq = fc2_num.has_value();
  TVM_FFI_ICHECK(fq || !fc2_den.has_value()) << "fc2_den needs fc2_num";
  if (fq) {
    TVM_FFI_ICHECK(g1) << "NVFP4 activation output is a GEMM1 epilogue";
    TVM_FFI_ICHECK((reinterpret_cast<uintptr_t>(p.out) & 15) == 0) << "out must be 16 B aligned";
    const int64_t E = alpha.numel();
    auto scale = [&](TensorView t, const float*& ptr, int& stride) {
      CHECK_INPUT_AND_TYPE(t, dl_float32);
      TVM_FFI_ICHECK(t.numel() == 1 || t.numel() == E) << "FC2 global scale must be [1] or [E]";
      ptr = static_cast<const float*>(t.data_ptr());
      stride = t.numel() == 1 ? 0 : 1;
    };
    scale(fc2_num.value(), q.fq_num, q.fq_num_stride);
    if (fc2_den.has_value()) scale(fc2_den.value(), q.fq_den, q.fq_den_stride);
  }
  const bool pt = row_scales.has_value();
  if (pt) {
    TensorView rs = row_scales.value();
    CHECK_INPUT_AND_TYPE(rs, dl_float32);
    TVM_FFI_ICHECK(variant == kBase && row_tile <= 64 && activation != 1 && !fq)
        << "row_scales: variant kBase, row_tile <= 64, relu2 GEMM1 or GEMM2";
    TVM_FFI_ICHECK(g1 ? rs.numel() >= act.numel() / K : rs.numel() >= max_blocks * row_tile)
        << "row_scales: GEMM1 [tokens], GEMM2 [max_blocks x row_tile]";
    p.ws = static_cast<float*>(rs.data_ptr());
    TVM_FFI_ICHECK((reinterpret_cast<uintptr_t>(p.ws) & 7) == 0) << "row_scales: 8 B aligned";
  }
  const int columns = static_cast<int>(N / kTileN);
  const bool nc2 = row_tile == 96 || variant == kW64s || variant == kG2W64x || variant == kG2W64v;
  const int grid = static_cast<int>(max_blocks) * (nc2 ? (columns + 1) / 2 : columns);
  const int R = static_cast<int>(row_tile);
  const bool kt = K % kTileK != 0;
  // Decode (finalize's n <= 256) at R 8: the last CTA of each column chunk sums it (fin_chunk).
  const bool sum = topk_weights.has_value();
  bool fuse = false;
  if (sum) {
    TVM_FFI_ICHECK(!g1 && topk_ids.has_value() && sum_out.has_value() && counters.has_value())
        << "the top-k sum takes GEMM2, topk_ids, sum_out and counters";
    TensorView tw = topk_weights.value(), so = sum_out.value(), ct = counters.value();
    CHECK_INPUT(tw);
    TVM_FFI_ICHECK(tw.dtype() == dl_bfloat16 || tw.dtype() == dl_float32)
        << "topk_weights must be float32 or bfloat16";
    CHECK_INPUT_AND_TYPE(topk_ids.value(), dl_int32);
    CHECK_INPUT_AND_TYPE(so, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(ct, dl_int32);
    CHECK_DIM(2, so);
    TVM_FFI_ICHECK(so.size(1) == Nout && Nout % 8 == 0)
        << "sum_out must be [tokens, Nout], 8 | Nout";
    constexpr int kFinTopK = 4 * Cfg<8, 1>::kStageBytes / (kTileN * 2 + 8);  // fin_chunk's stage
    fuse = R == 8 && variant == kBase && so.size(0) * (Nout / 8) <= 256 && columns <= kFinCnt &&
           ct.numel() >= kFinCnt && top_k <= kFinTopK;
    if (fuse) {
      q.fin_w = tw.data_ptr();
      q.fin_ids = static_cast<const int*>(topk_ids.value().data_ptr());
      q.sum_out = static_cast<__nv_bfloat16*>(so.data_ptr());
      q.tokens = static_cast<int>(so.size(0));
      q.fin_bf16 = tw.dtype() == dl_bfloat16;
      p.cnt = static_cast<int*>(ct.data_ptr());
      p.top_k = static_cast<int>(top_k);
    }
  }
  auto run = [&](auto act_c, auto kt_c, auto fq_c) {
    constexpr int A = decltype(act_c)::value;
    constexpr bool T = decltype(kt_c)::value;
    constexpr bool F = decltype(fq_c)::value;
    GemmParamsT<F> const& pq = q;
    if (pt) {
      if constexpr (A != 1 && (!F || A == 0)) launch_tile_pt<A, T, F>(R, pq, grid, st);
    } else if constexpr (F && A == 0) {
      launch_gemm<8, 8, 8, 0, 4, T, 1, false, false, false, true>(pq, grid, st);  // fuse
    } else if (variant == kBase)
      launch_tile<A, T, false, false, false, F>(R, pq, grid, st);
    else if (variant == kStage2)
      launch_tile<A, T, true, false, false, F>(R, pq, grid, st);
    else if (variant == kX96)
      launch_tile<A, T, false, true, false, F>(R, pq, grid, st);
    else if (variant == kQ12)
      launch_tile<A, T, false, true, true, F>(R, pq, grid, st);
    else if constexpr (A == 0) {
      if (variant == kG2Lin)
        launch_gemm<64, 4, 4, 0, 2, T>(p, grid, st);
      else if (variant == kG2W64v)
        launch_gemm<64, 4, 4, 0, 2, T, 2, false, false, true>(p, grid, st);
      else
        launch_gemm<64, 4, 4, 0, 2, T, 2>(p, grid, st);
    } else {
      launch_gemm<64, 4, 4, A, 2, T, 2, false, false, false, F>(pq, grid, st);
    }
  };
  auto run_kt = [&](auto act_c) {
    if (fq || fuse)
      kt ? run(act_c, std::true_type{}, std::true_type{})
         : run(act_c, std::false_type{}, std::true_type{});
    else
      kt ? run(act_c, std::true_type{}, std::false_type{})
         : run(act_c, std::false_type{}, std::false_type{});
  };
  if (activation == 0)
    run_kt(std::integral_constant<int, 0>{});
  else if (activation == 1)
    run_kt(std::integral_constant<int, 1>{});
  else
    run_kt(std::integral_constant<int, 2>{});
  if (sum && !fuse)
    w4a16_moe_sm80_finalize(out, topk_weights.value(), topk_ids.value(), sum_out.value(), top_k);
}

// Per-token NVFP4 GEMM2 input: the real rows of GEMM1's blocked output y [max_blocks x row_tile][I]
// (I % 16 == 0) quantized per row under gs (the base inverse scale) and restored in place;
// row_scales [max_blocks x row_tile]: out2[e] out1[e] times the row's scale (see row_quant).
void w4a16_moe_sm80_row_quant(TensorView y, TensorView meta, TensorView out2, TensorView out1,
                              TensorView row_scales, int64_t row_tile, int64_t max_blocks,
                              double gs) {
  CHECK_INPUT_AND_TYPE(y, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(meta, dl_int32);
  CHECK_INPUT_AND_TYPE(out2, dl_float32);
  CHECK_INPUT_AND_TYPE(out1, dl_float32);
  CHECK_INPUT_AND_TYPE(row_scales, dl_float32);
  CHECK_DIM(2, y);
  const int64_t I = y.size(1), rows = max_blocks * row_tile;
  TVM_FFI_ICHECK(row_tile >= 8 && row_tile <= 64 && I % 16 == 0 && I / 16 <= 1024 &&
                 y.size(0) >= rows && meta.numel() >= 1 + 2 * max_blocks &&
                 row_scales.numel() >= rows && (out1.numel() == 1 || out2.numel() == out1.numel()))
      << "row_quant: y [>= max_blocks x row_tile, I] with 16 | I <= 16384, meta, row_scales "
         "[max_blocks x row_tile], out2 [E] and out1 [E] or [1]";
  TVM_FFI_ICHECK((reinterpret_cast<uintptr_t>(y.data_ptr()) & 15) == 0) << "y must be 16 B aligned";
  if (max_blocks == 0) return;
  ffi::CUDADeviceGuard device_guard(y.device().device_id);
  const cudaStream_t st = get_stream(y.device());
  const int kb = static_cast<int>(I / 16), kbp = (kb + 31) / 32 * 32;
  const int rpc = std::max(1, 256 / kbp), R = static_cast<int>(row_tile);
  const dim3 grid(static_cast<unsigned>(max_blocks), static_cast<unsigned>((R + rpc - 1) / rpc));
  row_quant<<<grid, rpc * kbp, 0, st>>>(
      static_cast<__nv_bfloat16*>(y.data_ptr()), static_cast<const int*>(meta.data_ptr()),
      static_cast<int>(max_blocks), R, kb, static_cast<const float*>(out2.data_ptr()),
      static_cast<const float*>(out1.data_ptr()), out1.numel() == 1 ? 0 : 1, static_cast<float>(gs),
      static_cast<float*>(row_scales.data_ptr()));
}

// out = the top-k weighted sum of y: see the finalize kernel. topk_weights: float32 or bfloat16.
void w4a16_moe_sm80_finalize(TensorView y, TensorView topk_weights, TensorView topk_ids,
                             TensorView out, int64_t top_k) {
  CHECK_INPUT_AND_TYPE(y, dl_bfloat16);
  CHECK_INPUT(topk_weights);
  const bool bf16_w = topk_weights.dtype() == dl_bfloat16;
  TVM_FFI_ICHECK(bf16_w || topk_weights.dtype() == dl_float32)
      << "topk_weights must be float32 or bfloat16";
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_DIM(2, out);
  const int tokens = static_cast<int>(out.size(0)), H = static_cast<int>(out.size(1));
  TVM_FFI_ICHECK(H % 8 == 0) << "hidden size must be a multiple of 8";
  ffi::CUDADeviceGuard device_guard(out.device().device_id);
  const cudaStream_t st = get_stream(out.device());
  const int n = tokens * (H / 8);
  if (n == 0) return;
  auto run = [&](auto w) {
    using WT = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    // Decode: all slots' loads at once (a batch of 6, 11 or 22 >= top_k), CTAs of 32 over the SMs.
    const bool dec = n <= 256;
    auto kern = !dec          ? finalize<WT, 1>
                : top_k <= 6  ? finalize<WT, 6>
                : top_k <= 11 ? finalize<WT, 11>
                              : finalize<WT, 22>;
    const int nt = dec ? 32 : 256;
    kern<<<(n - 1) / nt + 1, nt, 0, st>>>(static_cast<const __nv_bfloat16*>(y.data_ptr()), w,
                                          static_cast<const int*>(topk_ids.data_ptr()),
                                          static_cast<__nv_bfloat16*>(out.data_ptr()), tokens,
                                          static_cast<int>(top_k), H);
  };
  if (bf16_w)
    run(static_cast<const __nv_bfloat16*>(topk_weights.data_ptr()));
  else
    run(static_cast<const float*>(topk_weights.data_ptr()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prep, w4a16_moe_sm80_prep);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, w4a16_moe_sm80_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(finalize, w4a16_moe_sm80_finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(row_quant, w4a16_moe_sm80_row_quant);
