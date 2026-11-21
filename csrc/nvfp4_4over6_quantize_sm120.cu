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
// NVFP4 4over6 activation quantization for SM12x (BF16 input), static and dynamic per-tensor global
// scale.
//
// Output is bit-identical to fp4_quantize(..., nvfp4_4over6=NVFP44Over6Config(e4m3_max=256,
// err_mode="MSE")): the per-block arithmetic follows cvt_warp_fp16_to_fp4<...,
// NVFP44Over6Config<256, MSE, false>> step for step (fast-math scale reciprocals, E4M3 candidates
// for /6 and /4, fp32 scaling, cvt.rn.satfinite.e2m1x2.f32, squared errors of (code * scale) *
// global decode scale summed strictly in element order, /4 only on a strictly smaller error,
// all-zero blocks written as zero). One thread owns a whole 16-element scale block, so the block
// max and the candidate errors need no lane shuffles; the in-order 16-term sum equals the
// reference's lane-pair sum. Small inputs use the reference's lane-pair layout instead (fewer
// instructions per thread).
//
//   nvfp4_4over6_quantize_sm120          static: global_scale [1] fp32 given
//   nvfp4_4over6_dynamic_quantize_sm120  dynamic: deq = max(amax(x), 1e-12) * rcp_q, global_scale =
//   1 / deq,
//                                        alpha = deq * weight_global_scale, then as static, in one
//                                        launch
// Outputs: packed e2m1 [M, K/2] uint8 and UE4M3 scales in the 128x4 swizzled layout (padding
// zeroed), the layout fp4_quantize produces. Tuning arguments (unroll, ctas_per_sm, v8,
// small_threads, hold, order) are 0 / -1 for the built-in heuristic; nonzero values force a kernel
// variant (tests and benchmarks).
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;

// PDL: every kernel waits for its predecessor grid before touching global memory (a no-op without
// the launch attribute). Static kernels let dependents launch once their blocks are quantized (as
// FlashInfer / TRT-LLM do); the dynamic kernels do not by default: CUTLASS epilogues read
// *alpha_ptr at kernel entry, before their own wait.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

struct alignas(32) Blk {  // one 16-element scale block of BF16
  uint32_t w[8];
};

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <bool V8>
__device__ __forceinline__ Blk load_blk(const Blk* p) {
  Blk b;
  if constexpr (V8) {
    asm("ld.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
        : "=r"(b.w[0]), "=r"(b.w[1]), "=r"(b.w[2]), "=r"(b.w[3]), "=r"(b.w[4]), "=r"(b.w[5]),
          "=r"(b.w[6]), "=r"(b.w[7])
        : "l"(p));
  } else {
    const uint4* q = reinterpret_cast<const uint4*>(p);
    const uint4 lo = __ldg(q);
    const uint4 hi = __ldg(q + 1);
    b.w[0] = lo.x;
    b.w[1] = lo.y;
    b.w[2] = lo.z;
    b.w[3] = lo.w;
    b.w[4] = hi.x;
    b.w[5] = hi.y;
    b.w[6] = hi.z;
    b.w[7] = hi.w;
  }
  return b;
}

__device__ __forceinline__ __nv_bfloat162 absmax2(const Blk& b) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(b.w);
  __nv_bfloat162 m = __habs2(h[0]);
#pragma unroll
  for (int j = 1; j < 8; ++j) m = __hmax2(m, __habs2(h[j]));
  return m;
}

// 8 floats -> 8 e2m1 nibbles, element 2i in the low nibble of byte i (the TRT-LLM / fp4_quantize
// operand order).
__device__ __forceinline__ uint32_t to_e2m1x8(const float2 (&v)[4]) {
  uint32_t out;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(out)
      : "f"(v[0].x), "f"(v[0].y), "f"(v[1].x), "f"(v[1].y), "f"(v[2].x), "f"(v[2].y), "f"(v[3].x),
        "f"(v[3].y));
  return out;
}

// 8 e2m1 codes -> 8 exact float values (the inverse conversion is exact).
__device__ __forceinline__ void from_e2m1x8(uint32_t packed, float2 (&v)[4]) {
  uint32_t h[4];
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}\n"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(packed));
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = __half22float2(*reinterpret_cast<const __half2*>(&h[j]));
}

// acc + the 8 squared errors in element order; dq = (code * scale) * gds and every step rounded as
// FlashInfer does.
__device__ __forceinline__ float acc_sq_err8(float acc, const float2 (&f)[4], uint32_t packed,
                                             float scale, float gds) {
  float2 q[4];
  from_e2m1x8(packed, q);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float dx = __fsub_rn(__fmul_rn(__fmul_rn(q[j].x, scale), gds), f[j].x);
    const float dy = __fsub_rn(__fmul_rn(__fmul_rn(q[j].y, scale), gds), f[j].y);
    acc = __fadd_rn(acc, __fmul_rn(dx, dx));
    acc = __fadd_rn(acc, __fmul_rn(dy, dy));
  }
  return acc;
}

struct Recipe {
  float gs;   // global encode scale
  float gds;  // rcp.approx(gs), the global decode scale of the error model
  float r6;   // rcp.approx(6)
};

__device__ __forceinline__ Recipe make_recipe(float gs) {
  return Recipe{gs, rcp_approx_ftz(gs), rcp_approx_ftz(6.0f)};
}

// The 4over6 block recipe of cvt_warp_fp16_to_fp4<..., NVFP44Over6Config<256, MSE, false>> (and of
// the CuTe-DSL nvfp4_quantize kernel): fast-math scales, strict MSE; /4 wins only if its error is
// strictly smaller; all-zero blocks are 0.
__device__ __forceinline__ void quant16(const Blk& b, const Recipe& r, uint2& packed,
                                        uint8_t& sf_byte) {
  const __nv_bfloat162 m2 = absmax2(b);
  const float block_max = __bfloat162float(__hmax(m2.x, m2.y));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(b.w);
  float2 lo[4], hi[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    lo[j] = __bfloat1622float2(h[j]);
    hi[j] = __bfloat1622float2(h[j + 4]);
  }
  const float sf6_hp = r.gs * (block_max * r.r6);
  const float sf4_hp = sf6_hp * 1.5f;
  const __nv_fp8_e4m3 s4 = __nv_fp8_e4m3(sf4_hp);
  const __nv_fp8_e4m3 s6 = __nv_fp8_e4m3(sf6_hp);
  const float scale4 = static_cast<float>(s4);
  const float scale6 = static_cast<float>(s6);
  const float os4 = rcp_approx_ftz(scale4 * r.gds);
  const float os6 = rcp_approx_ftz(scale6 * r.gds);
  float2 t[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) t[j] = make_float2(lo[j].x * os4, lo[j].y * os4);
  const uint32_t p4lo = to_e2m1x8(t);
#pragma unroll
  for (int j = 0; j < 4; ++j) t[j] = make_float2(hi[j].x * os4, hi[j].y * os4);
  const uint32_t p4hi = to_e2m1x8(t);
#pragma unroll
  for (int j = 0; j < 4; ++j) t[j] = make_float2(lo[j].x * os6, lo[j].y * os6);
  const uint32_t p6lo = to_e2m1x8(t);
#pragma unroll
  for (int j = 0; j < 4; ++j) t[j] = make_float2(hi[j].x * os6, hi[j].y * os6);
  const uint32_t p6hi = to_e2m1x8(t);
  const float e4 = acc_sq_err8(acc_sq_err8(0.f, lo, p4lo, scale4, r.gds), hi, p4hi, scale4, r.gds);
  const float e6 = acc_sq_err8(acc_sq_err8(0.f, lo, p6lo, scale6, r.gds), hi, p6hi, scale6, r.gds);
  const bool use4 = e4 < e6;
  packed = use4 ? make_uint2(p4lo, p4hi) : make_uint2(p6lo, p6hi);
  sf_byte = use4 ? s4.__x : s6.__x;
  if (block_max == 0.f) {
    packed = make_uint2(0u, 0u);
    sf_byte = 0;
  }
}

// n / d for n < 2^31 (CUTLASS FastDivmod).
struct FastDiv {
  uint32_t d, mul, shr;
};

FastDiv make_fastdiv(uint32_t d) {
  FastDiv f{d, 0u, 0u};
  if (d != 1) {
    int l = 0;
    while ((uint64_t{1} << l) < d) ++l;  // ceil(log2 d)
    const int p = 31 + l;
    f.mul = static_cast<uint32_t>(((uint64_t{1} << p) + d - 1) / d);
    f.shr = static_cast<uint32_t>(p - 32);
  }
  return f;
}

__device__ __forceinline__ uint32_t fast_div(uint32_t n, const FastDiv& f) {
  return f.d == 1 ? n : (__umulhi(n, f.mul) >> f.shr);
}

// [mTile, kTile, 32 (outer m), 4 (inner m), 4 (inner k)] swizzle of the UE4M3 scale matrix.
__device__ __forceinline__ uint32_t sf_offset(uint32_t row, uint32_t kb, uint32_t num_k_tiles) {
  return (row / 128) * num_k_tiles * 512 + (kb / 4) * 512 + (row % 32) * 16 +
         ((row % 128) / 32) * 4 + (kb % 4);
}

__device__ __forceinline__ void quant_store(const Blk& v, uint32_t bi, const Recipe& r,
                                            const FastDiv& kb_div, uint32_t num_k_tiles,
                                            uint2* __restrict__ out, uint8_t* __restrict__ sf_out) {
  uint2 p;
  uint8_t s;
  quant16(v, r, p, s);
  out[bi] = p;
  const uint32_t row = fast_div(bi, kb_div);
  sf_out[sf_offset(row, bi - row * kb_div.d, num_k_tiles)] = s;
}

// Lane-pair layout for small inputs (half the instructions per thread): chunk c (8 values) is the
// low (even c) or high (odd c) half of block c / 2, and the two halves sit in adjacent lanes of
// `mask`, as in the reference kernel (8 values per thread); the pair's error sum stays strictly in
// element order.
__device__ __forceinline__ void squared_errors8(const float2 (&f)[4], uint32_t packed, float scale,
                                                float gds, float (&sq)[8]) {
  float2 q[4];
  from_e2m1x8(packed, q);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float dx = __fsub_rn(__fmul_rn(__fmul_rn(q[j].x, scale), gds), f[j].x);
    const float dy = __fsub_rn(__fmul_rn(__fmul_rn(q[j].y, scale), gds), f[j].y);
    sq[2 * j] = __fmul_rn(dx, dx);
    sq[2 * j + 1] = __fmul_rn(dy, dy);
  }
}

__device__ __forceinline__ float pair_ordered_sum16(unsigned mask, bool odd, const float (&sq)[8]) {
  float partial = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) partial = __fadd_rn(partial, sq[j]);
  float total = __shfl_xor_sync(mask, partial, 1);
#pragma unroll
  for (int j = 0; j < 8; ++j) total = __fadd_rn(total, sq[j]);
  const float other = __shfl_xor_sync(mask, total, 1);
  return odd ? total : other;
}

__device__ __forceinline__ void quant8_pair_store(const uint4& u, uint32_t c, const Recipe& r,
                                                  unsigned mask, const FastDiv& kb_div,
                                                  uint32_t num_k_tiles, uint32_t* __restrict__ out,
                                                  uint8_t* __restrict__ sf_out) {
  const bool odd = (c & 1) != 0;
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  __nv_bfloat162 lmax = __habs2(h[0]);
#pragma unroll
  for (int j = 1; j < 4; ++j) lmax = __hmax2(lmax, __habs2(h[j]));
  lmax = __hmax2(__shfl_xor_sync(mask, lmax, 1), lmax);
  const float block_max = __bfloat162float(__hmax(lmax.x, lmax.y));
  float2 f[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) f[j] = __bfloat1622float2(h[j]);
  const float sf6_hp = r.gs * (block_max * r.r6);
  const float sf4_hp = sf6_hp * 1.5f;
  const __nv_fp8_e4m3 s4 = __nv_fp8_e4m3(sf4_hp);
  const __nv_fp8_e4m3 s6 = __nv_fp8_e4m3(sf6_hp);
  const float scale4 = static_cast<float>(s4);
  const float scale6 = static_cast<float>(s6);
  const float os4 = rcp_approx_ftz(scale4 * r.gds);
  const float os6 = rcp_approx_ftz(scale6 * r.gds);
  float2 a[4], b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    a[j] = make_float2(f[j].x * os4, f[j].y * os4);
    b[j] = make_float2(f[j].x * os6, f[j].y * os6);
  }
  const uint32_t p4 = to_e2m1x8(a);
  const uint32_t p6 = to_e2m1x8(b);
  float sq[8];
  squared_errors8(f, p4, scale4, r.gds, sq);
  const float err4 = pair_ordered_sum16(mask, odd, sq);
  squared_errors8(f, p6, scale6, r.gds, sq);
  const float err6 = pair_ordered_sum16(mask, odd, sq);
  const bool use4 = err4 < err6;
  uint32_t packed = use4 ? p4 : p6;
  uint8_t sf_byte = use4 ? s4.__x : s6.__x;
  if (block_max == 0.f) {
    packed = 0u;
    sf_byte = 0;
  }
  out[c] = packed;
  if (!odd) {
    const uint32_t bi = c >> 1;
    const uint32_t row = fast_div(bi, kb_div);
    sf_out[sf_offset(row, bi - row * kb_div.d, num_k_tiles)] = sf_byte;
  }
}

// Scale-padding bytes of the swizzled layout, zeroed by first/stride; only bytes no valid (row < m,
// kb < num_kb) scale occupies are written, so zeroing and scale stores need no ordering.
__device__ __forceinline__ void zero_sf_padding_disjoint(uint8_t* __restrict__ sf_out, int64_t m,
                                                         int64_t num_kb, int64_t m_padded,
                                                         int64_t first, int64_t stride) {
  const int64_t tile_bytes = (num_kb + 3) / 4 * 512;
  if (num_kb % 4 == 0) {
    if (m % 128 == 0) return;
    const int m_rel = static_cast<int>(m % 128);
    uint4* tile = reinterpret_cast<uint4*>(sf_out + m / 128 * tile_bytes);
    for (int64_t v = first; v < tile_bytes / 16; v += stride) {
      const int j = static_cast<int>(v & 31);
      if (j >= m_rel) {
        tile[v] = make_uint4(0u, 0u, 0u, 0u);
      } else {
        uint32_t* w = reinterpret_cast<uint32_t*>(&tile[v]);
        for (int t = 1; t < 4; ++t) {
          if (t * 32 + j >= m_rel) w[t] = 0u;
        }
      }
    }
    return;
  }
  const int64_t end = m_padded / 128 * tile_bytes;
  uint32_t* words = reinterpret_cast<uint32_t*>(sf_out);
  for (int64_t w = first; w < end / 4; w += stride) {
    const int64_t off = w * 4, r = off % 512;
    const int64_t row = off / tile_bytes * 128 + (r % 16) / 4 * 32 + r / 16;
    const int64_t kb0 = off % tile_bytes / 512 * 4;
    if (row >= m) {
      words[w] = 0u;
    } else if (kb0 + 3 >= num_kb) {
      for (int b = 0; b < 4; ++b) {
        if (kb0 + b >= num_kb) sf_out[off + b] = 0;
      }
    }
  }
}

// Block-wide max of non-negative values, valid in every thread. Two calls need a block barrier
// between them.
template <int THREADS>
__device__ __forceinline__ float block_max(float local) {
  __shared__ float warp_max[THREADS / 32];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local;
  __syncthreads();
  float b = (threadIdx.x & 31) < THREADS / 32 ? warp_max[threadIdx.x & 31] : 0.f;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, off));
  return b;
}

__device__ __forceinline__ float finalize_scale(float amax, float rcp_q,
                                                const float* __restrict__ wgs,
                                                float* __restrict__ inv_out,
                                                float* __restrict__ alpha_out, bool publish) {
  const float deq = fmaxf(amax, 1.0e-12f) * rcp_q;
  const float sf_scale = __fdiv_rn(1.0f, deq);
  if (publish) {
    *inv_out = sf_scale;
    *alpha_out = deq * (*wgs);
  }
  return sf_scale;
}

struct Args {
  const Blk* x;
  uint32_t nblk;
  FastDiv kb_div;
  uint32_t num_kb;
  uint32_t m;
  uint32_t m_padded;
  uint2* out;
  uint8_t* sf;
};

// ---------------------------------------------------------------------------------------------------------------
// static: grid-stride over blocks, U blocks per thread per iteration with all loads issued first
// The first loads are issued before the global scale is read; P prefetches the next iteration's
// blocks before quantizing the current ones. Threads past the blocks (extra CTAs sized for the
// padding) only zero padding, last. REV walks the blocks last-to-first: an input just written
// front-to-back by its producer has its tail in L2.
template <int U, bool P, bool V8, bool REV = false>
__global__ void __launch_bounds__(kThreads)
    quant_4o6_static_kernel(Args a, const float* __restrict__ gs_ptr, bool trigger) {
  pdl_wait();
  const uint32_t t = blockIdx.x * kThreads + threadIdx.x;
  const uint32_t stride = gridDim.x * kThreads;
  const uint32_t nkt = (a.num_kb + 3) / 4;
  auto at = [&](uint32_t i) { return REV ? a.nblk - 1 - i : i; };
  Blk cur[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const uint32_t bi = t + u * stride;
    if (bi < a.nblk) cur[u] = load_blk<V8>(a.x + at(bi));
  }
  const Recipe r = make_recipe(__ldg(gs_ptr));
  for (uint32_t b0 = t; b0 < a.nblk; b0 += stride * U) {
    if constexpr (P) {
      Blk nxt[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t bi = b0 + (U + u) * stride;
        if (bi < a.nblk) nxt[u] = load_blk<V8>(a.x + at(bi));
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t bi = b0 + u * stride;
        if (bi < a.nblk) quant_store(cur[u], at(bi), r, a.kb_div, nkt, a.out, a.sf);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) cur[u] = nxt[u];
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t bi = b0 + u * stride;
        if (bi < a.nblk) quant_store(cur[u], at(bi), r, a.kb_div, nkt, a.out, a.sf);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint32_t bi = b0 + (U + u) * stride;
        if (bi < a.nblk) cur[u] = load_blk<V8>(a.x + at(bi));
      }
    }
  }
  if (trigger) pdl_trigger();
  zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, t, stride);
}

// ---------------------------------------------------------------------------------------------------------------
// dynamic, small inputs (nblk <= THREADS * LOADS): no inter-block dependency. Every block loads the
// whole input (block j of thread t is t + j * THREADS), reduces |x| max itself, and block c
// quantizes its own blocks c * THREADS + t straight from registers (stage[c]).
template <int THREADS, int LOADS, bool V8>
__global__ void __launch_bounds__(THREADS)
    quant_4o6_dyn_small_kernel(Args a, float rcp_q, const float* __restrict__ wgs,
                               float* __restrict__ inv_out, float* __restrict__ alpha_out,
                               bool trigger) {
  pdl_wait();
  const uint32_t first = blockIdx.x * THREADS + threadIdx.x;
  if (blockIdx.x * THREADS >=
      a.nblk) {  // padding-only block (uniform per block: no barrier below is reached)
    zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, first, gridDim.x * THREADS);
    return;
  }
  Blk stage[LOADS];
#pragma unroll
  for (int j = 0; j < LOADS; ++j) {
    const uint32_t bi = threadIdx.x + j * THREADS;
    stage[j] = bi < a.nblk ? load_blk<V8>(a.x + bi) : Blk{};
  }
  __nv_bfloat162 m2 = absmax2(stage[0]);
#pragma unroll
  for (int j = 1; j < LOADS; ++j) m2 = __hmax2(m2, absmax2(stage[j]));
  const float amax = block_max<THREADS>(__bfloat162float(__hmax(m2.x, m2.y)));
  const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, first == 0);
  const Recipe r = make_recipe(sf_scale);
  Blk v = stage[0];
#pragma unroll
  for (int j = 1; j < LOADS; ++j) {
    if (j == static_cast<int>(blockIdx.x)) v = stage[j];
  }
  if (first < a.nblk) quant_store(v, first, r, a.kb_div, (a.num_kb + 3) / 4, a.out, a.sf);
  if (trigger) pdl_trigger();
  zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, first, gridDim.x * THREADS);
}

// ---------------------------------------------------------------------------------------------------------------
// dynamic, cooperative grid: each thread loads its first R blocks into registers and keeps them
// across the one grid barrier (blocks beyond grid * R per thread are read again after it).
// Per-block maxima go to their own scratch slots, rewritten by every launch (CUDA-graph safe). REV
// maps logical block i to nblk - 1 - i: the first sweep then starts at the tail (in L2 after a
// front-to-back producer) and the re-read after the barrier at the head (read last by the sweep).
template <int R, bool V8, bool REV = false>
__global__ void __launch_bounds__(kThreads)
    quant_4o6_dyn_coop_kernel(Args a, float rcp_q, const float* __restrict__ wgs,
                              float* __restrict__ inv_out, float* __restrict__ alpha_out,
                              float* __restrict__ block_maxima, bool trigger) {
  pdl_wait();
  cg::grid_group grid = cg::this_grid();
  const uint32_t t = blockIdx.x * kThreads + threadIdx.x;
  const uint32_t stride = gridDim.x * kThreads;
  auto at = [&](uint32_t i) { return REV ? a.nblk - 1 - i : i; };
  Blk held[R];
#pragma unroll
  for (int j = 0; j < R; ++j) {
    const uint32_t bi = t + j * stride;
    held[j] = bi < a.nblk ? load_blk<V8>(a.x + at(bi)) : Blk{};
  }
  __nv_bfloat162 m2 = absmax2(held[0]);
#pragma unroll
  for (int j = 1; j < R; ++j) m2 = __hmax2(m2, absmax2(held[j]));
  constexpr int kSweep = 4;
  for (uint32_t b0 = t + R * stride; b0 < a.nblk; b0 += kSweep * stride) {
    Blk v[kSweep];
#pragma unroll
    for (int u = 0; u < kSweep; ++u) {
      const uint32_t bi = b0 + u * stride;
      v[u] = bi < a.nblk ? load_blk<V8>(a.x + at(bi)) : Blk{};
    }
#pragma unroll
    for (int u = 0; u < kSweep; ++u) m2 = __hmax2(m2, absmax2(v[u]));
  }
  const float b = block_max<kThreads>(__bfloat162float(__hmax(m2.x, m2.y)));
  if (threadIdx.x == 0) block_maxima[blockIdx.x] = b;
  grid.sync();
  float partial = 0.f;
  for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += kThreads)
    partial = fmaxf(partial, __ldcg(&block_maxima[i]));
  const float amax = block_max<kThreads>(partial);
  const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, t == 0);
  const Recipe r = make_recipe(sf_scale);
  const uint32_t nkt = (a.num_kb + 3) / 4;
#pragma unroll
  for (int j = 0; j < R; ++j) {
    const uint32_t bi = t + j * stride;
    if (bi < a.nblk) quant_store(held[j], at(bi), r, a.kb_div, nkt, a.out, a.sf);
  }
  // Blocks beyond the held ones were swept forward before the barrier; walk them backwards so the
  // most recently read (the likeliest still in L2) come first.
  const int64_t over0 = int64_t{t} + int64_t{R} * stride;
  if (over0 < a.nblk) {
    constexpr int kTail = 2;
    for (int64_t i = (a.nblk - 1 - over0) / stride; i >= 0; i -= kTail) {
      Blk v[kTail];
#pragma unroll
      for (int u = 0; u < kTail; ++u) {
        if (i - u >= 0)
          v[u] = load_blk<V8>(a.x + at(static_cast<uint32_t>(over0 + (i - u) * stride)));
      }
#pragma unroll
      for (int u = 0; u < kTail; ++u) {
        if (i - u >= 0)
          quant_store(v[u], at(static_cast<uint32_t>(over0 + (i - u) * stride)), r, a.kb_div, nkt,
                      a.out, a.sf);
      }
    }
  }
  if (trigger) pdl_trigger();
  zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, t, stride);
}

// ---------------------------------------------------------------------------------------------------------------
// static, lane-pair layout (small inputs): one 16-byte chunk per thread per iteration. The stride
// is a multiple of 32 and the chunk count even, so both lanes of a pair are always active together.
__global__ void __launch_bounds__(kThreads)
    quant_4o6_static_pair_kernel(Args a, const float* __restrict__ gs_ptr, bool trigger) {
  pdl_wait();
  const uint32_t t = blockIdx.x * kThreads + threadIdx.x;
  const uint32_t stride = gridDim.x * kThreads;
  const uint32_t chunks = a.nblk * 2;
  const uint4* xv = reinterpret_cast<const uint4*>(a.x);
  uint32_t* out = reinterpret_cast<uint32_t*>(a.out);
  const uint32_t nkt = (a.num_kb + 3) / 4;
  uint4 cur = t < chunks ? __ldg(xv + t) : make_uint4(0u, 0u, 0u, 0u);
  const Recipe r = make_recipe(__ldg(gs_ptr));
  for (uint32_t c = t; c < chunks; c += stride) {
    const uint4 v = cur;
    if (c + stride < chunks) cur = __ldg(xv + c + stride);
    quant8_pair_store(v, c, r, __activemask(), a.kb_div, nkt, out, a.sf);
  }
  if (trigger) pdl_trigger();
  zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, t, stride);
}

// dynamic, small inputs, lane-pair layout: as quant_4o6_dyn_small_kernel with 16-byte chunks (chunk
// j of thread t is t + j * THREADS); block c quantizes chunks c * THREADS + t from stage[c].
template <int THREADS, int LOADS>
__global__ void __launch_bounds__(THREADS)
    quant_4o6_dyn_small_pair_kernel(Args a, float rcp_q, const float* __restrict__ wgs,
                                    float* __restrict__ inv_out, float* __restrict__ alpha_out,
                                    bool trigger) {
  pdl_wait();
  const uint32_t first = blockIdx.x * THREADS + threadIdx.x;
  const uint32_t chunks = a.nblk * 2;
  if (blockIdx.x * THREADS >= chunks) {
    zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, first, gridDim.x * THREADS);
    return;
  }
  const uint4* xv = reinterpret_cast<const uint4*>(a.x);
  uint4 stage[LOADS];
#pragma unroll
  for (int j = 0; j < LOADS; ++j) {
    const uint32_t c = threadIdx.x + j * THREADS;
    stage[j] = c < chunks ? __ldg(xv + c) : make_uint4(0u, 0u, 0u, 0u);
  }
  __nv_bfloat162 m2 = __float2bfloat162_rn(0.f);
#pragma unroll
  for (int j = 0; j < LOADS; ++j) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&stage[j]);
#pragma unroll
    for (int q = 0; q < 4; ++q) m2 = __hmax2(m2, __habs2(h[q]));
  }
  const float amax = block_max<THREADS>(__bfloat162float(__hmax(m2.x, m2.y)));
  const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, first == 0);
  const Recipe r = make_recipe(sf_scale);
  uint4 v = stage[0];
#pragma unroll
  for (int j = 1; j < LOADS; ++j) {
    if (j == static_cast<int>(blockIdx.x)) v = stage[j];
  }
  if (first < chunks) {
    quant8_pair_store(v, first, r, __activemask(), a.kb_div, (a.num_kb + 3) / 4,
                      reinterpret_cast<uint32_t*>(a.out), a.sf);
  }
  if (trigger) pdl_trigger();
  zero_sf_padding_disjoint(a.sf, a.m, a.num_kb, a.m_padded, first, gridDim.x * THREADS);
}

// ---------------------------------------------------------------------------------------------------------------
// host

#define NVFP4_4O6_CUDA_CHECK(expr)                                                        \
  do {                                                                                    \
    const cudaError_t err_ = (expr);                                                      \
    TVM_FFI_ICHECK(err_ == cudaSuccess) << #expr " failed: " << cudaGetErrorString(err_); \
  } while (0)

int sm_count(int dev) {
  static std::mutex mu;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;
  int n = 0;
  NVFP4_4O6_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  cache[dev] = n;
  return n;
}

int blocks_per_sm(const void* kernel, int threads, int dev) {
  static std::mutex mu;
  static std::unordered_map<int, std::unordered_map<const void*, int>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto& per_dev = cache[dev];
  auto it = per_dev.find(kernel);
  if (it != per_dev.end()) return it->second;
  int n = 0;
  NVFP4_4O6_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, 0));
  per_dev[kernel] = n;
  return n;
}

struct Prepared {
  Args a;
  cudaStream_t stream;
  int dev;
  bool pdl;      // launch with programmatic stream serialization (PDL)
  bool trigger;  // kernels call griddepcontrol.launch_dependents after their blocks
  bool rev;      // walk blocks last to first (static unroll 11, cooperative dynamic path)
};

template <typename... KArgs, typename... Act>
void launch_ex(void (*kernel)(KArgs...), int grid, int threads, const Prepared& p, bool cooperative,
               Act&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = p.stream;
  cudaLaunchAttribute attrs[2];
  unsigned n = 0;
  if (p.pdl) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cooperative) {
    attrs[n].id = cudaLaunchAttributeCooperative;
    attrs[n].val.cooperative = 1;
    ++n;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = n;
  NVFP4_4O6_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Act>(args)...));
}

// Inputs from this many blocks (96 MiB of BF16) up are walked last block first: after a producer
// that wrote them front to back, their tail is still in L2.
constexpr uint32_t kReverseMinBlocks = 3u << 20;

// pdl: -1 auto (static: PDL launch + trigger; dynamic: PDL launch, no trigger), 0 off, 1 launch +
// trigger, 2 launch
void set_pdl(Prepared& p, int64_t pdl, int64_t auto_mode) {
  if (pdl < 0) pdl = auto_mode;
  TVM_FFI_ICHECK(pdl >= 0 && pdl <= 2)
      << "pdl must be -1 (auto), 0 (off), 1 (launch + trigger) or 2 (launch)";
  p.pdl = pdl != 0;
  p.trigger = pdl == 1;
}

// order: -1 auto (reversed from kReverseMinBlocks), 0 front to back, 1 back to front
void set_order(Prepared& p, int64_t order) {
  TVM_FFI_ICHECK(order >= -1 && order <= 1) << "order must be -1 (auto), 0 or 1";
  p.rev = order < 0 ? p.a.nblk >= kReverseMinBlocks : order == 1;
}

Prepared prepare(TensorView input, TensorView out_fp4, TensorView out_sf) {
  CHECK_INPUT(input);
  CHECK_INPUT(out_fp4);
  CHECK_INPUT(out_sf);
  check_dtype(input, dl_bfloat16, "input");
  check_dtype(out_fp4, dl_uint8, "out_fp4");
  check_dtype(out_sf, dl_uint8, "out_sf");
  TVM_FFI_ICHECK_EQ(input.ndim(), 2) << "input must be 2-D [M, K]";
  const int64_t m = input.size(0), k = input.size(1);
  TVM_FFI_ICHECK(k % 16 == 0) << "K must be a multiple of 16, got " << k;
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 32 == 0)
      << "input must be 32-byte aligned";
  const int64_t m_padded = (m + 127) / 128 * 128;
  const int64_t sf_cols = (k / 16 + 3) / 4 * 4;
  TVM_FFI_ICHECK(m * (k / 16) < (int64_t{1} << 31) && m_padded * sf_cols < (int64_t{1} << 32))
      << "input too large";
  TVM_FFI_ICHECK_EQ(out_fp4.numel(), m * k / 2) << "out_fp4 must hold M * K / 2 bytes";
  TVM_FFI_ICHECK_GE(out_sf.numel(), m_padded * sf_cols)
      << "out_sf must hold roundup(M, 128) * roundup(K / 16, 4) bytes";
  check_same_device(out_fp4, input, "out_fp4", "input");
  check_same_device(out_sf, input, "out_sf", "input");
  Prepared p;
  p.a.x = reinterpret_cast<const Blk*>(input.data_ptr());
  p.a.nblk = static_cast<uint32_t>(m * (k / 16));
  p.a.kb_div = make_fastdiv(static_cast<uint32_t>(k / 16));
  p.a.num_kb = static_cast<uint32_t>(k / 16);
  p.a.m = static_cast<uint32_t>(m);
  p.a.m_padded = static_cast<uint32_t>(m_padded);
  p.a.out = reinterpret_cast<uint2*>(out_fp4.data_ptr());
  p.a.sf = static_cast<uint8_t*>(out_sf.data_ptr());
  p.dev = input.device().device_id;
  p.stream = get_stream(input.device());
  p.pdl = p.trigger = p.rev = false;
  return p;
}

// 16-byte (or 4-byte, when K/16 % 4 != 0) units of scale padding zero_sf_padding_disjoint walks.
int64_t padding_units(const Args& a) {
  const int64_t tile_bytes = (int64_t{a.num_kb} + 3) / 4 * 512;
  if (a.num_kb % 4 == 0) return a.m % 128 == 0 ? 0 : tile_bytes / 16;
  return int64_t{a.m_padded} / 128 * tile_bytes / 4;
}

// CTAs that give every thread at most kPadPerThread padding units.
constexpr int64_t kPadPerThread = 2;
int64_t padding_ctas(const Args& a, int threads) {
  return (padding_units(a) + threads * kPadPerThread - 1) / (threads * kPadPerThread);
}

template <int U, bool P, bool V8, bool REV = false>
void launch_static(const Prepared& p, const float* gs, int64_t ctas_per_sm) {
  auto kernel = quant_4o6_static_kernel<U, P, V8, REV>;
  const int64_t want =
      std::max((int64_t{p.a.nblk} + int64_t{kThreads} * U - 1) / (int64_t{kThreads} * U),
               padding_ctas(p.a, kThreads));
  const int64_t per_sm =
      ctas_per_sm > 0 ? ctas_per_sm
                      : blocks_per_sm(reinterpret_cast<const void*>(kernel), kThreads, p.dev);
  const int grid =
      static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(want, per_sm * sm_count(p.dev))));
  launch_ex(kernel, grid, kThreads, p, false, p.a, gs, p.trigger);
}

void launch_static_pair(const Prepared& p, const float* gs, int64_t ctas_per_sm) {
  auto kernel = quant_4o6_static_pair_kernel;
  const int64_t want =
      std::max((int64_t{p.a.nblk} * 2 + kThreads - 1) / kThreads, padding_ctas(p.a, kThreads));
  const int64_t per_sm =
      ctas_per_sm > 0 ? ctas_per_sm
                      : blocks_per_sm(reinterpret_cast<const void*>(kernel), kThreads, p.dev);
  const int grid =
      static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(want, per_sm * sm_count(p.dev))));
  launch_ex(kernel, grid, kThreads, p, false, p.a, gs, p.trigger);
}

// unroll: 1, 2 (no prefetch), 11, 12 (prefetch the next iteration), 21 (lane-pair layout), 31 (11,
// last block first)
template <bool V8>
void dispatch_static(const Prepared& p, const float* gs, int64_t unroll, int64_t ctas_per_sm) {
  switch (unroll) {
    case 21:
      launch_static_pair(p, gs, ctas_per_sm);
      break;
    case 31:
      launch_static<1, true, V8, true>(p, gs, ctas_per_sm);
      break;
    case 1:
      launch_static<1, false, V8>(p, gs, ctas_per_sm);
      break;
    case 2:
      launch_static<2, false, V8>(p, gs, ctas_per_sm);
      break;
    case 11:
      if (p.rev) {
        launch_static<1, true, V8, true>(p, gs, ctas_per_sm);
      } else {
        launch_static<1, true, V8>(p, gs, ctas_per_sm);
      }
      break;
    case 12:
      launch_static<2, true, V8>(p, gs, ctas_per_sm);
      break;
    default:
      TVM_FFI_ICHECK(false) << "unroll must be 1, 2, 11, 12, 21 or 31";
  }
}

template <int THREADS, bool V8>
void launch_small(const Prepared& p, float rcp_q, const float* wgs, float* inv, float* alpha) {
  const int quant_ctas = static_cast<int>((p.a.nblk + THREADS - 1) / THREADS);
  TVM_FFI_ICHECK(quant_ctas >= 1 && quant_ctas <= 8)
      << "small path covers at most 8 * THREADS blocks";
  const int loads = quant_ctas <= 1 ? 1 : quant_ctas <= 2 ? 2 : quant_ctas <= 4 ? 4 : 8;
  const int grid = static_cast<int>(std::max<int64_t>(quant_ctas, padding_ctas(p.a, THREADS)));
  switch (loads) {
    case 1:
      launch_ex(quant_4o6_dyn_small_kernel<THREADS, 1, V8>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    case 2:
      launch_ex(quant_4o6_dyn_small_kernel<THREADS, 2, V8>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    case 4:
      launch_ex(quant_4o6_dyn_small_kernel<THREADS, 4, V8>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    default:
      launch_ex(quant_4o6_dyn_small_kernel<THREADS, 8, V8>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
  }
}

template <int THREADS>
void launch_small_pair(const Prepared& p, float rcp_q, const float* wgs, float* inv, float* alpha) {
  const int quant_ctas = static_cast<int>((int64_t{p.a.nblk} * 2 + THREADS - 1) / THREADS);
  TVM_FFI_ICHECK(quant_ctas >= 1 && quant_ctas <= 8)
      << "small lane-pair path covers at most 4 * THREADS blocks";
  const int loads = quant_ctas <= 1 ? 1 : quant_ctas <= 2 ? 2 : quant_ctas <= 4 ? 4 : 8;
  const int grid = static_cast<int>(std::max<int64_t>(quant_ctas, padding_ctas(p.a, THREADS)));
  switch (loads) {
    case 1:
      launch_ex(quant_4o6_dyn_small_pair_kernel<THREADS, 1>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    case 2:
      launch_ex(quant_4o6_dyn_small_pair_kernel<THREADS, 2>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    case 4:
      launch_ex(quant_4o6_dyn_small_pair_kernel<THREADS, 4>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
    default:
      launch_ex(quant_4o6_dyn_small_pair_kernel<THREADS, 8>, grid, THREADS, p, false, p.a, rcp_q,
                wgs, inv, alpha, p.trigger);
      break;
  }
}

template <int R, bool V8, bool REV>
void launch_coop(const Prepared& p, float rcp_q, const float* wgs, float* inv, float* alpha,
                 float* scratch, int64_t scratch_numel, int64_t ctas_per_sm) {
  auto kernel = quant_4o6_dyn_coop_kernel<R, V8, REV>;
  const int occ = blocks_per_sm(reinterpret_cast<const void*>(kernel), kThreads, p.dev);
  TVM_FFI_ICHECK(occ >= 1) << "cooperative kernel does not fit on an SM";
  const int64_t per_sm = ctas_per_sm > 0 ? std::min<int64_t>(ctas_per_sm, occ) : occ;
  const int64_t want = (int64_t{p.a.nblk} + int64_t{kThreads} * R - 1) / (int64_t{kThreads} * R);
  const int64_t sms = sm_count(p.dev);
  const int64_t cap = std::min<int64_t>(per_sm * sms, scratch_numel);
  int64_t g = std::min<int64_t>(want, cap);
  // An uncapped grid between one and a few waves of SMs is rounded to whole waves; blocks past the
  // held ones are re-read after the barrier.
  if (g == want && want > sms)
    g = std::min<int64_t>(cap, sms * std::max<int64_t>(1, (want + sms / 2) / sms));
  const int grid = static_cast<int>(std::max<int64_t>(1, g));
  launch_ex(kernel, grid, kThreads, p, true, p.a, rcp_q, wgs, inv, alpha, scratch, p.trigger);
}

template <bool V8>
void dispatch_dyn(const Prepared& p, float rcp_q, const float* wgs, float* inv, float* alpha,
                  float* scratch, int64_t scratch_numel, int64_t small_threads, int64_t hold,
                  int64_t ctas_per_sm) {
  if (small_threads == 0) {  // heuristic, tuned on RTX PRO 5000 (SM120, 110 SMs)
    const uint32_t n = p.a.nblk;
    small_threads = n <= 8 * 128 ? 128 : n <= 8 * 256 ? 256 : -1;
    if (small_threads < 0 && hold == 0) {
      if (n <= 24576) {
        hold = 1;
      } else if (n <= 49152) {
        hold = 2;
        if (ctas_per_sm == 0) ctas_per_sm = 2;
      } else if (n <= (8u << 20)) {
        hold = 4;
        if (ctas_per_sm == 0) ctas_per_sm = 2;
      } else {
        hold = 2;
        if (ctas_per_sm == 0) ctas_per_sm = 1;
      }
    }
  }
  if (small_threads > 2000) {  // lane-pair small path: 2000 + threads
    switch (small_threads - 2000) {
      case 128:
        launch_small_pair<128>(p, rcp_q, wgs, inv, alpha);
        return;
      case 256:
        launch_small_pair<256>(p, rcp_q, wgs, inv, alpha);
        return;
      case 512:
        launch_small_pair<512>(p, rcp_q, wgs, inv, alpha);
        return;
      case 1024:
        launch_small_pair<1024>(p, rcp_q, wgs, inv, alpha);
        return;
      default:
        TVM_FFI_ICHECK(false) << "small lane-pair threads must be 128, 256, 512 or 1024";
    }
  }
  if (small_threads > 0) {
    switch (small_threads) {
      case 128:
        launch_small<128, V8>(p, rcp_q, wgs, inv, alpha);
        return;
      case 256:
        launch_small<256, V8>(p, rcp_q, wgs, inv, alpha);
        return;
      case 512:
        launch_small<512, V8>(p, rcp_q, wgs, inv, alpha);
        return;
      case 1024:
        launch_small<1024, V8>(p, rcp_q, wgs, inv, alpha);
        return;
      default:
        TVM_FFI_ICHECK(false) << "small_threads must be 128, 256, 512 or 1024";
    }
  }
  switch ((hold == 0 ? 2 : hold) * 2 + (p.rev ? 1 : 0)) {
    case 2:
      launch_coop<1, V8, false>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    case 3:
      launch_coop<1, V8, true>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    case 4:
      launch_coop<2, V8, false>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    case 5:
      launch_coop<2, V8, true>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    case 8:
      launch_coop<4, V8, false>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    case 9:
      launch_coop<4, V8, true>(p, rcp_q, wgs, inv, alpha, scratch, scratch_numel, ctas_per_sm);
      break;
    default:
      TVM_FFI_ICHECK(false) << "hold must be 1, 2 or 4";
  }
}

void check_scalar_f32(TensorView t, TensorView input, const char* name) {
  CHECK_CUDA(t);
  check_dtype(t, dl_float32, name);
  TVM_FFI_ICHECK_GE(t.numel(), 1) << name << " must hold one float";
  check_same_device(t, input, name, "input");
}

}  // namespace

void nvfp4_4over6_quantize_sm120(TensorView input, TensorView global_scale, TensorView out_fp4,
                                 TensorView out_sf, int64_t unroll, int64_t ctas_per_sm, bool v8,
                                 int64_t pdl, int64_t order) {
  check_scalar_f32(global_scale, input, "global_scale");
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  Prepared p = prepare(input, out_fp4, out_sf);
  if (p.a.m == 0) return;
  set_pdl(p, pdl, 1);
  set_order(p, order);
  if (unroll == 0) {  // heuristic, tuned on RTX PRO 5000 (SM120, 110 SMs)
    const uint32_t n = p.a.nblk;
    if (n >= 1024 && n <= 12288) {
      unroll = 21;
      if (ctas_per_sm == 0) ctas_per_sm = 2;
    } else {
      unroll = 11;
      v8 = true;
      if (ctas_per_sm == 0 && n >= 65536 && n <= 1572864) ctas_per_sm = 8;
    }
  }
  const float* gs = static_cast<const float*>(global_scale.data_ptr());
  if (v8) {
    dispatch_static<true>(p, gs, unroll, ctas_per_sm);
  } else {
    dispatch_static<false>(p, gs, unroll, ctas_per_sm);
  }
}

void nvfp4_4over6_dynamic_quantize_sm120(TensorView input, TensorView global_scale_out,
                                         TensorView alpha_out, TensorView weight_global_scale,
                                         double rcp_q, TensorView scratch, TensorView out_fp4,
                                         TensorView out_sf, int64_t small_threads, int64_t hold,
                                         int64_t ctas_per_sm, bool v8, int64_t pdl, int64_t order) {
  check_scalar_f32(global_scale_out, input, "global_scale_out");
  check_scalar_f32(alpha_out, input, "alpha_out");
  check_scalar_f32(weight_global_scale, input, "weight_global_scale");
  check_scalar_f32(scratch, input, "scratch");
  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  Prepared p = prepare(input, out_fp4, out_sf);
  TVM_FFI_ICHECK(p.a.nblk > 0) << "dynamic quantization needs a non-empty input";
  set_pdl(p, pdl, 2);
  set_order(p, order);
  if (small_threads == 0) v8 = true;  // heuristic
  float* inv = static_cast<float*>(global_scale_out.data_ptr());
  float* alpha = static_cast<float*>(alpha_out.data_ptr());
  const float* wgs = static_cast<const float*>(weight_global_scale.data_ptr());
  float* scp = static_cast<float*>(scratch.data_ptr());
  const float rq = static_cast<float>(rcp_q);
  if (v8) {
    dispatch_dyn<true>(p, rq, wgs, inv, alpha, scp, scratch.numel(), small_threads, hold,
                       ctas_per_sm);
  } else {
    dispatch_dyn<false>(p, rq, wgs, inv, alpha, scp, scratch.numel(), small_threads, hold,
                        ctas_per_sm);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_4over6_quantize_sm120, nvfp4_4over6_quantize_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_4over6_dynamic_quantize_sm120,
                              nvfp4_4over6_dynamic_quantize_sm120);
