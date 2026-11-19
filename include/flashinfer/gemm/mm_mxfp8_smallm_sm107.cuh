/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FLASHINFER_GEMM_MM_MXFP8_SMALLM_SM107_CUH_
#define FLASHINFER_GEMM_MM_MXFP8_SMALLM_SM107_CUH_

// MXFP8 (E4M3 data + UE8M0 block-32 scales, swizzled 128x4 SF layout) NT GEMM
//   result[M,N] = dequant(aq[M,K]) * dequant(bq[N,K])^T   (fp32 accum, bf16 out)
// Decode-size M (1..32).  sm_107 has no native warp-level fp8 MMA (ptxas expands
// mma...e4m3 into 12 F2FP unpacks + 2 HMMA), so operands are converted to bf16
// once and pre-multiplied by the power-of-two block scale; mma.m16n8k16.bf16
// then accumulates in place.  The *B* matrix drives the mma 16-row axis so the
// streamed operand is the wide one and the tiny A matrix is the 8-wide one.
//
// Each lane streams 16 contiguous k bytes per load.  A lane's 16 bytes sit
// inside one 32-element scale block, so one scale decode covers four mma
// fragments instead of one, and one load replaces two.  The mma k index is a
// permutation of the real k index; A and B use the same permutation, so the
// dot product is unchanged.
//
// Framework-agnostic device code (raw-pointer kernel + helpers); the
// TVM-FFI binding and the per-shape route table live in
// csrc/mm_mxfp8_smallm_sm107.cu.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace flashinfer {
namespace mxfp8_smallm_sm107 {

#define DEVI __device__ __forceinline__

// Global default for the fused scaled up-conversion; individual routes override
// it where the classic cvt + bf16x2 multiply measured faster.
#ifndef SCVD
#define SCVD true
#endif
#define PEER2A_SCV_LO false
#define PEER2A_SCV_HI false

// Cluster primitives written directly: the cooperative_groups wrappers add a
// generic-proxy memory fence (ERRBAR in SASS) around every cluster operation,
// and NCU attributes ~45% of this kernel's stall cycles to those barriers.
// A DSMEM producer only needs the release carried by barrier.cluster.arrive.
DEVI unsigned cta_rank() {
  unsigned r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEVI void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
// store one float4 into `rank`'s copy of a shared-memory object
DEVI void st_rank_f4(const void* p, unsigned rank, const float4& v) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile(
      "{ .reg .u32 m;\n mapa.shared::cluster.u32 m, %0, %1;\n"
      " st.shared::cluster.v4.f32 [m], {%2,%3,%4,%5};\n}" ::"r"(a),
      "r"(rank), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
      : "memory");
}
DEVI unsigned smaddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
// Blackwell/Rubin packed single-precision add: one instruction per two floats.
// The split-K epilogue is a chain of float4 accumulations, so halving its add
// count removes a measurable slice of a kernel whose epilogue is nearly as long
// as its mainloop.
DEVI float4 add4(float4 a, float4 b) {
  float4 r;
  asm("{ .reg .b64 pa, pb, pc, pd;\n"
      " mov.b64 pa, {%4, %5};\n mov.b64 pb, {%8, %9};\n"
      " add.rn.f32x2 pc, pa, pb;\n mov.b64 {%0, %1}, pc;\n"
      " mov.b64 pa, {%6, %7};\n mov.b64 pb, {%10, %11};\n"
      " add.rn.f32x2 pd, pa, pb;\n mov.b64 {%2, %3}, pd;\n}"
      : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
      : "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w), "f"(b.x), "f"(b.y), "f"(b.z), "f"(b.w));
  return r;
}

// Balanced split-K fold: all partials are already live in registers, so use
// ceil(log2(WK)) dependent add levels instead of a WK-1 serial chain.
template <int N>
DEVI float4 fold4(float4* v) {
  if constexpr (N == 1) {
    return v[0];
  } else {
    constexpr int H = (N + 1) / 2;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) v[i] = add4(v[i], v[i + H]);
    return fold4<H>(v);
  }
}

// WK16 small-M routes already sit at the register/occupancy cliff.  The
// balanced tree makes all 16 partials simultaneously schedulable and ptxas
// lengthens their live ranges enough to cost ~37% on those routes, so retain
// the compact serial chain for this one depth.
template <>
DEVI float4 fold4<16>(float4* v) {
#pragma unroll
  for (int i = 1; i < 16; ++i) v[0] = add4(v[0], v[i]);
  return v[0];
}

DEVI void mbar_init1(const void* mb) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smaddr(mb)) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
DEVI void mbar_expect(const void* mb, unsigned bytes) {
  asm volatile(
      "{ .reg .b64 st_;\n mbarrier.arrive.expect_tx.shared::cta.b64 st_, [%0], %1;\n}" ::"r"(
          smaddr(mb)),
      "r"(bytes)
      : "memory");
}
template <bool SUSPEND = false>
DEVI void mbar_wait(const void* mb) {
  if constexpr (SUSPEND) {
    const unsigned suspend_ns = 10000000u;
    asm volatile(
        "{ .reg .pred p_;\n"
        "W%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p_, [%0], 0, %1;\n"
        " @!p_ bra W%=;\n}" ::"r"(smaddr(mb)),
        "r"(suspend_ns)
        : "memory");
  } else {
    asm volatile(
        "{ .reg .pred p_;\n"
        "W%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p_, [%0], 0;\n"
        " @!p_ bra W%=;\n}" ::"r"(smaddr(mb))
        : "memory");
  }
}
DEVI void st_async_f4(const void* p, const void* mb, unsigned rank, const float4& v) {
  asm volatile(
      "{ .reg .u32 ra_, rm_;\n mapa.shared::cluster.u32 ra_, %0, %2;\n"
      " mapa.shared::cluster.u32 rm_, %1, %2;\n"
      " st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
      "[ra_], {%3,%4,%5,%6}, [rm_];\n}" ::"r"(smaddr(p)),
      "r"(smaddr(mb)), "r"(rank), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
      : "memory");
}

DEVI uint4 ldg128(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
// weight stream: every byte is read once, so keeping it out of L1 leaves the
// line budget to the activation rows and the scale words, which every k group
// re-reads (measured L1 hit rate on this kernel is ~4%).
DEVI uint4 ldg128b(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
// same stream, but asking L2 to pull 256 B per miss.  Each CTA walks a long
// contiguous k run of every weight row, so the next group's bytes are already
// in L2 by the time the warp asks for them.
DEVI uint4 ldg128p(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
// Q32 maps a lane's two 16-byte chunks onto one aligned 32-byte scale block.
// Fetch both halves in one instruction to halve the load wavefront count.
DEVI void ldg256(const uint8_t* p, uint4& a, uint4& b) {
  asm volatile("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w), "=r"(b.x), "=r"(b.y), "=r"(b.z),
                 "=r"(b.w)
               : "l"(p));
}
DEVI void ldg256b(const uint8_t* p, uint4& a, uint4& b) {
  asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w), "=r"(b.x), "=r"(b.y), "=r"(b.z),
                 "=r"(b.w)
               : "l"(p));
}
DEVI void ldg256p(const uint8_t* p, uint4& a, uint4& b) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w), "=r"(b.x), "=r"(b.y), "=r"(b.z),
                 "=r"(b.w)
               : "l"(p));
}
DEVI uint2 ldg64(const uint8_t* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
DEVI unsigned ldg32(const uint8_t* p) {
  unsigned r;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
DEVI unsigned ldg8(const uint8_t* p) {
  unsigned r;
  asm volatile("ld.global.nc.u8 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}

// byte selected by `sel` of a ue8m0 scale word, replicated into both halves of
// a .b16 so a single scale operand covers both e4m3 lanes of a scaled cvt.
DEVI unsigned sfsel(unsigned w, unsigned sel) {
  unsigned t;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(t) : "r"(w), "r"(sel));
  return t;
}
DEVI unsigned sfbf(unsigned w, int c) { return sfsel(w, 0x4400u | (unsigned)(c * 0x11)); }

// 4 e4m3 bytes -> 2 bf16x2 registers, each already multiplied by the ue8m0
// block scale `sc`.  Rubin folds the power-of-two block-scale multiply into the
// up-conversion, so the dequant costs one instruction per 2 bytes instead of a
// cvt plus a separate bf16x2 multiply, and no unscaled temporaries stay live.
// Two forms of the same dequant.  SCV uses Rubin's fused scaled up-conversion
// (one instruction per 2 bytes, all work on the conversion pipe); the other
// keeps the classic cvt plus a bf16x2 multiply, which splits the work across
// the conversion and the fp16x2 math pipes.  Which one wins depends on the
// route's pipe balance, so it is a per-route template choice.
template <bool SCV>
DEVI void cvt2(unsigned v, unsigned sc, unsigned* o) {
  if constexpr (SCV) {
    asm("{ .reg .b16 lo,hi,s0,s1;\n mov.b32 {lo,hi}, %2;\n mov.b32 {s0,s1}, %3;\n"
        " cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, lo, s0;\n"
        " cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %1, hi, s0;\n}"
        : "=r"(o[0]), "=r"(o[1])
        : "r"(v), "r"(sc));
  } else {
    unsigned c0, c1, m;
    asm("{ .reg .b16 lo,hi;\n mov.b32 {lo,hi}, %2;\n"
        " cvt.rn.bf16x2.e4m3x2 %0, lo;\n cvt.rn.bf16x2.e4m3x2 %1, hi;\n}"
        : "=r"(c0), "=r"(c1)
        : "r"(v));
    // sc holds the ue8m0 byte b replicated in bytes 0 and 1; the bf16 bits of
    // 2^(b-127) are b<<7, and byte 0 means 2^-127 (the bf16 subnormal 0x0040).
    m = (sc & 0xffu) * 0x00800080u;
    m = m > 0x00400040u ? m : 0x00400040u;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(o[0]) : "r"(c0), "r"(m));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(o[1]) : "r"(c1), "r"(m));
  }
}
// 8 e4m3 bytes -> 4 bf16x2 registers (ragged-tail path)
template <bool SCV>
DEVI void cvt4(uint2 v, unsigned sc, unsigned* o) {
  cvt2<SCV>(v.x, sc, o);
  cvt2<SCV>(v.y, sc, o + 2);
}
DEVI unsigned pick(const uint4& v, int t) {
  return t == 0 ? v.x : (t == 1 ? v.y : (t == 2 ? v.z : v.w));
}

// UE8M0 bytes below three can lose low E4M3 bits when rounded to bf16.
// Test the scales already loaded for the mainloop; ordinary blocks retain the
// existing conversion/MMA path, as do tiny blocks whose values convert exactly.
DEVI bool small_scale_word(unsigned w) { return ((w - 0x03030303u) & ~w & 0x80808080u) != 0; }
DEVI bool loses_bits(unsigned v, unsigned scale) {
  if (scale >= 3) return false;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned x = (v >> (i * 8)) & 127u;
    int e = x >> 3;
    int lost = 4 - int(scale) - (e ? e : 1);
    if (lost > 0 && (x & ((1u << lost) - 1u))) return true;
  }
  return false;
}
template <int BYTES>
DEVI bool loses_chunk(uint4 v, unsigned scale) {
  return loses_bits(v.x, scale) || loses_bits(v.y, scale) ||
         (BYTES == 16 && (loses_bits(v.z, scale) || loses_bits(v.w, scale)));
}
DEVI float dequant_f32(unsigned v, unsigned scale) {
  unsigned x = v & 127u;
  unsigned e = x >> 3;
  float q;
  if (x == 127u)
    q = __uint_as_float(0x7fc00000u);
  else if (e)
    q = __uint_as_float(((e + 120u) << 23) | ((x & 7u) << 20));
  else
    q = float(x & 7u) * 0x1p-9f;
  if (v & 128u) q = -q;
  // UE8M0 byte 255 encodes NaN (not 2^128): the tensor-core path propagates
  // it as NaN, so the precise path must too instead of scaling by +inf.
  float sc = (scale == 255u) ? __uint_as_float(0x7fc00000u)
                             : __uint_as_float(scale ? scale << 23 : 0x00400000u);
  float r;
  // Explicit non-FTZ instructions preserve the reference's FP32 subnormals,
  // independently of the translation unit's --use_fast_math option.
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(q), "f"(sc));
  return r;
}
DEVI float fma_f32(float a, float b, float c) {
  float r;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
DEVI float add_f32(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// UE8M0 bytes at the top of the range can expand a scaled E4M3 operand past
// BF16's (and even FP32's) finite range although the dot product itself is
// finite: 448 * 2^(247-127) already exceeds BF16-max, and byte 254 with
// value 4 is 2^129, outside FP32 too.  Any byte >= 240 (a cheap conservative
// word test: bits 7..4 all set) routes the k group to the rebalanced precise
// path below, which never materializes a scaled operand.
DEVI bool huge_scale_word(unsigned w) {
  return ((w & (w << 1) & (w << 2) & (w << 3)) & 0x80808080u) != 0;
}
// One rebalanced product term: raw E4M3 magnitudes (exact in FP32), with the
// combined block-scale exponent applied AFTER the product via scalbnf.  An
// operand whose own dequant overflows (scale 254, value 4) still contributes
// its exact finite product, and a zero partner gives 0, not NaN; the result
// is inf only when the true term exceeds FP32 range, which the FP32
// tensor-core accumulate could not represent either.  UE8M0 byte 255 is NaN.
DEVI float term_rebalanced(unsigned vb, unsigned va, unsigned sb, unsigned sa) {
  if (sb == 255u || sa == 255u) return __uint_as_float(0x7fc00000u);
  float bv = dequant_f32(vb, 127u);
  float av = dequant_f32(va, 127u);
  float p;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(p) : "f"(bv), "f"(av));
  return scalbnf(p, int(sb) + int(sa) - 254);
}
// Rare lossy conversion: each helper computes one output dot product from
// the same register fragments.  Keeping one accumulator live reduces the
// helper's register requirement on the usual, tensor-core-only path.
template <int BYTES>
__device__ __noinline__ float precise_dot(uint4 b, uint4 a, unsigned sb, unsigned sa, float acc,
                                          int acol) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
#pragma unroll 1
  for (int kp = 0; kp < 4; ++kp) {
    const int bl = g * 4 + kp, al = (2 * q + acol) * 4 + kp;
    unsigned bs = __shfl_sync(0xffffffffu, sb, bl);
    unsigned as = __shfl_sync(0xffffffffu, sa, al);
#pragma unroll 1
    for (int w = 0; w < BYTES / 4; ++w) {
      unsigned bw = __shfl_sync(0xffffffffu, pick(b, w), bl);
      unsigned aw = __shfl_sync(0xffffffffu, pick(a, w), al);
#pragma unroll 1
      for (int c = 0; c < 4; ++c) {
        float bv = dequant_f32((bw >> (8 * c)) & 255u, bs);
        float av = dequant_f32((aw >> (8 * c)) & 255u, as);
        acc = fma_f32(bv, av, acc);
      }
    }
  }
  return acc;
}
template <int BYTES>
DEVI float4 precise_chunk(uint4 b0, uint4 b1, uint4 a, unsigned sb0, unsigned sb1, unsigned sa,
                          float4 acc) {
  acc.x = precise_dot<BYTES>(b0, a, sb0, sa, acc.x, 0);
  acc.y = precise_dot<BYTES>(b0, a, sb0, sa, acc.y, 1);
  acc.z = precise_dot<BYTES>(b1, a, sb1, sa, acc.z, 0);
  acc.w = precise_dot<BYTES>(b1, a, sb1, sa, acc.w, 1);
  return acc;
}

// Rebalanced copy of precise_dot for k groups holding a huge (>= 240) scale
// byte: same fragments, same shuffles, same accumulation order, but every
// term goes through term_rebalanced so no scaled operand is ever formed.
// The small-scale precise_dot above is kept unchanged: it bitwise-matches
// the FP32-dequant reference (subnormal operand rounding included), which
// the rebalanced form intentionally does not reproduce.
template <int BYTES>
__device__ __noinline__ float precise_dot_rb(uint4 b, uint4 a, unsigned sb, unsigned sa, float acc,
                                             int acol) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, q = lane & 3;
#pragma unroll 1
  for (int kp = 0; kp < 4; ++kp) {
    const int bl = g * 4 + kp, al = (2 * q + acol) * 4 + kp;
    unsigned bs = __shfl_sync(0xffffffffu, sb, bl);
    unsigned as = __shfl_sync(0xffffffffu, sa, al);
#pragma unroll 1
    for (int w = 0; w < BYTES / 4; ++w) {
      unsigned bw = __shfl_sync(0xffffffffu, pick(b, w), bl);
      unsigned aw = __shfl_sync(0xffffffffu, pick(a, w), al);
#pragma unroll 1
      for (int c = 0; c < 4; ++c) {
        acc = add_f32(acc, term_rebalanced((bw >> (8 * c)) & 255u, (aw >> (8 * c)) & 255u, bs, as));
      }
    }
  }
  return acc;
}
template <int BYTES>
DEVI float4 precise_chunk_rb(uint4 b0, uint4 b1, uint4 a, unsigned sb0, unsigned sb1, unsigned sa,
                             float4 acc) {
  acc.x = precise_dot_rb<BYTES>(b0, a, sb0, sa, acc.x, 0);
  acc.y = precise_dot_rb<BYTES>(b0, a, sb0, sa, acc.y, 1);
  acc.z = precise_dot_rb<BYTES>(b1, a, sb1, sa, acc.z, 0);
  acc.w = precise_dot_rb<BYTES>(b1, a, sb1, sa, acc.w, 1);
  return acc;
}

#define MMA16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1)        \
  asm volatile(                                              \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"   \
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)               \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// 4 consecutive bf16 outputs, vector store when the tail allows it.  FULL is
// set on the routes whose n tile divides N exactly, which drops the tail
// compare and its branch from the store path; the pack itself is one
// cvt.rn.bf16x2.f32 per output pair instead of two converts plus a shift-or.
template <bool FULL = false, typename Index = int>
DEVI void store4(__nv_bfloat16* op, const float4& v, Index n, Index N) {
  // Exact measured routes have N divisible by four and therefore preserve the
  // allocator's 8-byte alignment on every row.  Generic rows may have an odd
  // bf16 stride, so only issue the 64-bit vector store when its address is
  // actually aligned; the scalar tail remains valid for every in-scope N.
  if (FULL || ((N - n >= 4) && ((((uintptr_t)op) & 7u) == 0))) {
    unsigned x, y;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(x) : "f"(v.x), "f"(v.y));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(y) : "f"(v.z), "f"(v.w));
    asm volatile("st.global.v2.u32 [%0], {%1,%2};" ::"l"(op), "r"(x), "r"(y) : "memory");
  } else {
    op[0] = __float2bfloat16(v.x);
    if (N - n > 1) op[1] = __float2bfloat16(v.y);
    if (N - n > 2) op[2] = __float2bfloat16(v.z);
    if (N - n > 3) op[3] = __float2bfloat16(v.w);
  }
}

// linear (row, kgroup-of-4) -> offset in the 1D swizzled 128x4 scale buffer
template <typename Offset = int>
DEVI Offset sfoff(Offset r, Offset k4d4) {
  return Offset(r >> 7) * (Offset(k4d4) * 512) + (r & 31) * 16 + ((r & 127) >> 5) * 4;
}

// ---------------------------------------------------------------------------
// NSUB : 16-row n sub-tiles handled by one warp      (per-warp n tile = 16*NSUB)
// WN   : warps splitting the n range of the CTA      (CTA n tile = 16*NSUB*WN)
// WK   : warps splitting the k range of the CTA      (reduced through smem)
// MSUB : 8-row m sub-tiles (one CTA always covers all of M)
// ---------------------------------------------------------------------------
template <int MSUB, int NSUB, int WN, int WK, bool PIPE = false, int BPS = 1, bool CLUSTER = false,
          int CLSC = 0, bool SCV = true, bool HALF = false, int GPW = 0, bool Q32 = false,
          bool AFFB = false, bool AFFA = false, bool FS = false, bool DACC = false, int CM = 0,
          int CN = 0, int CK = 0, int CGPC = 0, bool L2P = false, bool MBH = false,
          bool L1B = false, bool WIDE = false>
__global__ void __launch_bounds__(32 * WN * WK, BPS) __cluster_dims__(1, (CLSC > 0 ? CLSC : 1), 1)
    kmain(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ bq,
          const uint8_t* __restrict__ asf, const uint8_t* __restrict__ bsf,
          __nv_bfloat16* __restrict__ out, int64_t M_, int64_t N_, int64_t K_, int64_t nkb_,
          int64_t k4d4_, int64_t gpc_, int CLS, int64_t m0) {
  // For every measured family the shape is known at dispatch time.  Folding
  // M/N/K (and the groups-per-rank) into the instantiation removes the whole
  // constant-bank load + integer chain that otherwise sits between the kernel
  // entry and the first global load, which is pure latency on a kernel whose
  // warps each issue one burst of loads and then wait.
  using Index = std::conditional_t<WIDE, int64_t, int>;
  const Index M = CM ? CM : Index(M_);
  const Index N = CN ? CN : N_;
  const Index K = CK ? CK : K_;
  const Index nkb = CK ? (CK / 32) : nkb_;
  const Index k4d4 = CK ? (((CK / 32) + 3) / 4) : k4d4_;
  const Index gpc = CGPC ? CGPC : gpc_;
  const Index mbase = CM ? 0 : (m0 + (int)blockIdx.z * (8 * MSUB));
  constexpr int NT = 16 * NSUB;
  constexpr int NTOT = NT * WN;
  constexpr int MP = 8 * MSUB;
  constexpr int NTH = 32 * WN * WK;
  constexpr int NTP = NTOT + 4;  // pad so warp-fold stores hit 32 distinct banks
  const int cls = CLUSTER ? (CLSC ? CLSC : CLS) : 1;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  // Establish every cluster CTA's DSMEM lifetime without putting a startup
  // barrier on the critical path.  The relaxed arrival is issued before the
  // mainloop; its matching wait is deferred until the first remote write.
  __shared__ __align__(16) unsigned long long cmb[2];
  if constexpr (CLUSTER) {
    if constexpr (FS) {
      if (tid == 0) mbar_init1(cmb);
      __syncthreads();
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }
  const int wn = (WN == 1) ? 0 : (warp % WN);
  const int wk = (WN == 1) ? warp : (warp / WN);
  const int q = lane & 3, g = lane >> 2;
  const Index nbase = Index(blockIdx.x) * NTOT;
  const Index n0 = nbase + wn * NT;
  // a lane's 16 k bytes live in scale block (2*h + q/2) of the 128-k group
  const unsigned sel0 = 0x4400u + (unsigned)((q >> 1) * 0x11);
  const unsigned sel1 = sel0 + 0x22u;

  // The generic route can address tensors larger than 2 GiB.  Keep the
  // measured, compile-time-bounded routes on their existing 32-bit arithmetic.
  using Offset = std::conditional_t<WIDE, int64_t, int>;
  Offset boff[NSUB][2], sboff[NSUB][2];
  Offset aoff[MSUB], saoff[MSUB];
  if constexpr (AFFB) {
    const Offset bb = Offset(n0 + g) * K + q * (HALF ? 16 : (Q32 ? 32 : 16));
    const Offset sb = sfoff<Offset>(n0 + g, k4d4);
#pragma unroll
    for (int s = 0; s < NSUB; ++s)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        boff[s][i] = bb + Offset(s * 16 + 8 * i) * K;
        sboff[s][i] = sb + (s * 16 + 8 * i) * 16;
      }
  } else {
#pragma unroll
    for (int s = 0; s < NSUB; ++s)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        Index r = n0 + s * 16 + g + 8 * i;
        boff[s][i] = Offset(r < N ? r : N - 1) * K + q * (HALF ? 16 : (Q32 ? 32 : 16));
        sboff[s][i] = sfoff<Offset>(r < N ? r : N - 1, k4d4);
      }
  }
  if constexpr (AFFA) {
    const Offset ab = Offset(mbase + g) * K + q * (HALF ? 16 : (Q32 ? 32 : 16));
    const Offset sa = sfoff<Offset>(mbase + g, k4d4);
#pragma unroll
    for (int j = 0; j < MSUB; ++j) {
      aoff[j] = ab + Offset(j * 8) * K;
      saoff[j] = sa + j * 8 * 16;
    }
  } else {
#pragma unroll
    for (int j = 0; j < MSUB; ++j) {
      Index r = mbase + j * 8 + g;  // one CTA covers 32 m rows; M > 32 uses more launches
      aoff[j] = Offset(r < M ? r : M - 1) * K + q * (HALF ? 16 : (Q32 ? 32 : 16));
      saoff[j] = sfoff<Offset>(r < M ? r : M - 1, k4d4);
    }
  }

  float acc[NSUB][MSUB][4];
  // Second accumulator set for the two-deep pipelined mainloops.  Consecutive
  // k steps of one group write the same accumulator, so the tensor pipe sees a
  // RAW chain only NSUB*MSUB mma deep; giving the paired group its own set
  // doubles the independent chains and removes the fixed-latency wait that NCU
  // charges to those mma.  The two sets are summed once, before the smem fold.
  float acc2[DACC ? NSUB : 1][DACC ? MSUB : 1][4];
#pragma unroll
  for (int s = 0; s < NSUB; ++s)
#pragma unroll
    for (int j = 0; j < MSUB; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[s][j][c] = 0.f;
  if constexpr (DACC) {
#pragma unroll
    for (int s = 0; s < NSUB; ++s)
#pragma unroll
      for (int j = 0; j < MSUB; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc2[s][j][c] = 0.f;
  }

  const Index gtot = (nkb + 3) >> 2;
  const Index gfull = nkb >> 2;  // number of complete 4-block groups
  Index gbeg = (cls == 1) ? 0 : blockIdx.y * gpc;
  Index gend = (cls == 1) ? gtot : gbeg + gpc;
  if (gend > gtot) gend = gtot;
  const Index gLast = gend < gfull ? gend : gfull;

#define LOADG(G, WB, WA, BV, AV)                                                    \
  {                                                                                 \
    const Offset kb_ = Offset(G) << 7, so_ = Offset(G) * 512;                       \
    if constexpr (Q32 && CK == 1024) {                                              \
      _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                            \
        if constexpr (L1B) {                                                        \
          ldg256(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                  \
          ldg256(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                  \
        } else if constexpr (L2P) {                                                 \
          ldg256p(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                 \
          ldg256p(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                 \
        } else {                                                                    \
          ldg256b(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                 \
          ldg256b(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                 \
        }                                                                           \
      }                                                                             \
      _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                              \
          ldg256(aq + aoff[j] + kb_, AV[j][0], AV[j][1]);                           \
    }                                                                               \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s)                                \
        _Pragma("unroll") for (int i = 0; i < 2; ++i) if constexpr (Q32) WB[s][i] = \
            ldg8(bsf + sboff[s][i] + so_ + q);                                      \
    else WB[s][i] = ldg32(bsf + sboff[s][i] + so_);                                 \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j) {                              \
      if constexpr (Q32)                                                            \
        WA[j] = ldg8(asf + saoff[j] + so_ + q);                                     \
      else                                                                          \
        WA[j] = ldg32(asf + saoff[j] + so_);                                        \
    }                                                                               \
    if constexpr (Q32 && CK != 1024) {                                              \
      _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                            \
        if constexpr (L1B) {                                                        \
          ldg256(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                  \
          ldg256(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                  \
        } else if constexpr (L2P) {                                                 \
          ldg256p(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                 \
          ldg256p(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                 \
        } else {                                                                    \
          ldg256b(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                 \
          ldg256b(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                 \
        }                                                                           \
      }                                                                             \
      _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                              \
          ldg256(aq + aoff[j] + kb_, AV[j][0], AV[j][1]);                           \
    } else if constexpr (!Q32) {                                                    \
      _Pragma("unroll") for (int h = 0; h < 2; ++h)                                 \
          _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                        \
        BV[s][0][h] = L1B ? ldg128(bq + boff[s][0] + kb_ + h * 64)                  \
                          : (L2P ? ldg128p(bq + boff[s][0] + kb_ + h * 64)          \
                                 : ldg128b(bq + boff[s][0] + kb_ + h * 64));        \
        BV[s][1][h] = L1B ? ldg128(bq + boff[s][1] + kb_ + h * 64)                  \
                          : (L2P ? ldg128p(bq + boff[s][1] + kb_ + h * 64)          \
                                 : ldg128b(bq + boff[s][1] + kb_ + h * 64));        \
      }                                                                             \
      _Pragma("unroll") for (int h = 0; h < 2; ++h)                                 \
          _Pragma("unroll") for (int j = 0; j < MSUB; ++j) AV[j][h] =               \
              ldg128(aq + aoff[j] + kb_ + h * 64);                                  \
    }                                                                               \
  }
#define LOADGBQ(G, WB, BV)                                                                         \
  {                                                                                                \
    const Offset kb_ = Offset(G) << 7, so_ = Offset(G) * 512;                                      \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s) _Pragma("unroll") for (int i = 0; i < 2; ++i) \
        WB[s][i] = ldg8(bsf + sboff[s][i] + so_ + q);                                              \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                             \
      if constexpr (L2P) {                                                                         \
        ldg256p(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                                  \
        ldg256p(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                                  \
      } else {                                                                                     \
        ldg256b(bq + boff[s][0] + kb_, BV[s][0][0], BV[s][0][1]);                                  \
        ldg256b(bq + boff[s][1] + kb_, BV[s][1][0], BV[s][1][1]);                                  \
      }                                                                                            \
    }                                                                                              \
  }
#define LOADGAQ(G, WA, AV)                                                                   \
  {                                                                                          \
    const Offset kb_ = Offset(G) << 7, so_ = Offset(G) * 512;                                \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j) WA[j] = ldg8(asf + saoff[j] + so_ + q); \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                                         \
        ldg256(aq + aoff[j] + kb_, AV[j][0], AV[j][1]);                                      \
  }
#define MATHG_NORMAL(WB, WA, BV, AV, AC)                                                      \
  {                                                                                           \
    _Pragma("unroll") for (int h = 0; h < 2; ++h) {                                           \
      const unsigned sl_ = h ? sel1 : sel0;                                                   \
      unsigned sb_[NSUB][2], sa_[MSUB];                                                       \
      _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                      \
        if constexpr (Q32) {                                                                  \
          sb_[s][0] = WB[s][0] * 0x101u;                                                      \
          sb_[s][1] = WB[s][1] * 0x101u;                                                      \
        } else {                                                                              \
          sb_[s][0] = sfsel(WB[s][0], sl_);                                                   \
          sb_[s][1] = sfsel(WB[s][1], sl_);                                                   \
        }                                                                                     \
      }                                                                                       \
      _Pragma("unroll") for (int j = 0; j < MSUB; ++j) {                                      \
        if constexpr (Q32)                                                                    \
          sa_[j] = WA[j] * 0x101u;                                                            \
        else                                                                                  \
          sa_[j] = sfsel(WA[j], sl_);                                                         \
      }                                                                                       \
      _Pragma("unroll") for (int t = 0; t < 4; ++t) {                                         \
        if constexpr (MSUB >= 2 * NSUB - 1) {                                                 \
          unsigned Bc[NSUB][2][2];                                                            \
          _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                  \
            cvt2<SCV>(pick(BV[s][0][h], t), sb_[s][0], Bc[s][0]);                             \
            cvt2<SCV>(pick(BV[s][1][h], t), sb_[s][1], Bc[s][1]);                             \
          }                                                                                   \
          _Pragma("unroll") for (int j = 0; j < MSUB; ++j) {                                  \
            unsigned Ac[2];                                                                   \
            cvt2<SCV>(pick(AV[j][h], t), sa_[j], Ac);                                         \
            _Pragma("unroll") for (int s = 0; s < NSUB; ++s)                                  \
                MMA16(AC[s][j][0], AC[s][j][1], AC[s][j][2], AC[s][j][3], Bc[s][0][0],        \
                      Bc[s][1][0], Bc[s][0][1], Bc[s][1][1], Ac[0], Ac[1]);                   \
          }                                                                                   \
        } else {                                                                              \
          unsigned Ac[MSUB][2];                                                               \
          _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                                    \
              cvt2<SCV>(pick(AV[j][h], t), sa_[j], Ac[j]);                                    \
          _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                  \
            unsigned Bc[2][2];                                                                \
            cvt2<SCV>(pick(BV[s][0][h], t), sb_[s][0], Bc[0]);                                \
            cvt2<SCV>(pick(BV[s][1][h], t), sb_[s][1], Bc[1]);                                \
            _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                                  \
                MMA16(AC[s][j][0], AC[s][j][1], AC[s][j][2], AC[s][j][3], Bc[0][0], Bc[1][0], \
                      Bc[0][1], Bc[1][1], Ac[j][0], Ac[j][1]);                                \
          }                                                                                   \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }

#define MATHG(WB, WA, BV, AV, AC)                                                                \
  {                                                                                              \
    bool tiny_ = false, huge_ = false;                                                           \
    _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                          \
        _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) {                                       \
      tiny_ |= Q32 ? WB[s_][i_] < 3u : small_scale_word(WB[s_][i_]);                             \
      huge_ |= Q32 ? WB[s_][i_] >= 240u : huge_scale_word(WB[s_][i_]);                           \
    }                                                                                            \
    _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) {                                        \
      tiny_ |= Q32 ? WA[j_] < 3u : small_scale_word(WA[j_]);                                     \
      huge_ |= Q32 ? WA[j_] >= 240u : huge_scale_word(WA[j_]);                                   \
    }                                                                                            \
    bool loss_ = false;                                                                          \
    if (__any_sync(0xffffffffu, tiny_)) {                                                        \
      _Pragma("unroll") for (int h_ = 0; h_ < 2; ++h_) {                                         \
        unsigned bs_[NSUB][2], as_[MSUB];                                                        \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                      \
            _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) bs_[s_][i_] =                       \
                Q32 ? WB[s_][i_] : (sfsel(WB[s_][i_], h_ ? sel1 : sel0) & 255u);                 \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) as_[j_] =                            \
            Q32 ? WA[j_] : (sfsel(WA[j_], h_ ? sel1 : sel0) & 255u);                             \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_) {                                    \
          loss_ |= loses_chunk<16>(BV[s_][0][h_], bs_[s_][0]);                                   \
          loss_ |= loses_chunk<16>(BV[s_][1][h_], bs_[s_][1]);                                   \
        }                                                                                        \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) loss_ |=                             \
            loses_chunk<16>(AV[j_][h_], as_[j_]);                                                \
      }                                                                                          \
    }                                                                                            \
    if (__any_sync(0xffffffffu, huge_)) {                                                        \
      _Pragma("unroll") for (int h_ = 0; h_ < 2; ++h_) {                                         \
        unsigned bs_[NSUB][2], as_[MSUB];                                                        \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                      \
            _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) bs_[s_][i_] =                       \
                Q32 ? WB[s_][i_] : (sfsel(WB[s_][i_], h_ ? sel1 : sel0) & 255u);                 \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) as_[j_] =                            \
            Q32 ? WA[j_] : (sfsel(WA[j_], h_ ? sel1 : sel0) & 255u);                             \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                      \
            _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) {                                \
          float4 tmp_ = make_float4(AC[s_][j_][0], AC[s_][j_][1], AC[s_][j_][2], AC[s_][j_][3]); \
          tmp_ = precise_chunk_rb<16>(BV[s_][0][h_], BV[s_][1][h_], AV[j_][h_], bs_[s_][0],      \
                                      bs_[s_][1], as_[j_], tmp_);                                \
          AC[s_][j_][0] = tmp_.x;                                                                \
          AC[s_][j_][1] = tmp_.y;                                                                \
          AC[s_][j_][2] = tmp_.z;                                                                \
          AC[s_][j_][3] = tmp_.w;                                                                \
        }                                                                                        \
      }                                                                                          \
    } else if (__any_sync(0xffffffffu, loss_)) {                                                 \
      _Pragma("unroll") for (int h_ = 0; h_ < 2; ++h_) {                                         \
        unsigned bs_[NSUB][2], as_[MSUB];                                                        \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                      \
            _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) bs_[s_][i_] =                       \
                Q32 ? WB[s_][i_] : (sfsel(WB[s_][i_], h_ ? sel1 : sel0) & 255u);                 \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) as_[j_] =                            \
            Q32 ? WA[j_] : (sfsel(WA[j_], h_ ? sel1 : sel0) & 255u);                             \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                      \
            _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) {                                \
          float4 tmp_ = make_float4(AC[s_][j_][0], AC[s_][j_][1], AC[s_][j_][2], AC[s_][j_][3]); \
          tmp_ = precise_chunk<16>(BV[s_][0][h_], BV[s_][1][h_], AV[j_][h_], bs_[s_][0],         \
                                   bs_[s_][1], as_[j_], tmp_);                                   \
          AC[s_][j_][0] = tmp_.x;                                                                \
          AC[s_][j_][1] = tmp_.y;                                                                \
          AC[s_][j_][2] = tmp_.z;                                                                \
          AC[s_][j_][3] = tmp_.w;                                                                \
        }                                                                                        \
      }                                                                                          \
    } else {                                                                                     \
      MATHG_NORMAL(WB, WA, BV, AV, AC)                                                           \
    }                                                                                            \
  }

  if constexpr (HALF) {
    // Split k into 64-element units instead of 128.  A lane still fetches 16
    // contiguous bytes, so the L1 wavefronts per streamed byte are unchanged,
    // but the number of warp k partitions doubles.  That is exactly what the
    // shallow-K families lack: K/128 leaves them 3-5 warps per CTA, i.e. under
    // one warp per scheduler, and NCU shows them latency-starved rather than
    // bandwidth- or math-bound.  Requires K to be a multiple of 64.
    const Index utot = nkb >> 1;
    Index ubeg = (cls == 1) ? 0 : blockIdx.y * gpc;
    Index uend = (cls == 1) ? utot : ubeg + gpc;
    if (uend > utot) uend = utot;
#define HLOAD(U, WB, WA, BV, AV)                                                                   \
  {                                                                                                \
    const Offset so_ = Offset((U) >> 1) * 512, koff_ = Offset(U) * 64;                             \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s) _Pragma("unroll") for (int i = 0; i < 2; ++i) \
        WB[s][i] = ldg32(bsf + sboff[s][i] + so_);                                                 \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j) WA[j] = ldg32(asf + saoff[j] + so_);          \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                             \
      BV[s][0] = ldg128b(bq + boff[s][0] + koff_);                                                 \
      BV[s][1] = ldg128b(bq + boff[s][1] + koff_);                                                 \
    }                                                                                              \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j) AV[j] = ldg128(aq + aoff[j] + koff_);         \
  }
#define HMATH_NORMAL(U, WB, WA, BV, AV)                                                            \
  {                                                                                                \
    const unsigned sl_ = ((U) & 1) ? sel1 : sel0;                                                  \
    unsigned sb_[NSUB][2], sa_[MSUB];                                                              \
    _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                             \
      sb_[s][0] = sfsel(WB[s][0], sl_);                                                            \
      sb_[s][1] = sfsel(WB[s][1], sl_);                                                            \
    }                                                                                              \
    _Pragma("unroll") for (int j = 0; j < MSUB; ++j) sa_[j] = sfsel(WA[j], sl_);                   \
    _Pragma("unroll") for (int t = 0; t < 4; ++t) {                                                \
      if constexpr (MSUB >= 2 * NSUB - 1) {                                                        \
        unsigned Bc[NSUB][2][2];                                                                   \
        _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                         \
          cvt2<SCV>(pick(BV[s][0], t), sb_[s][0], Bc[s][0]);                                       \
          cvt2<SCV>(pick(BV[s][1], t), sb_[s][1], Bc[s][1]);                                       \
        }                                                                                          \
        _Pragma("unroll") for (int j = 0; j < MSUB; ++j) {                                         \
          unsigned Ac[2];                                                                          \
          cvt2<SCV>(pick(AV[j], t), sa_[j], Ac);                                                   \
          _Pragma("unroll") for (int s = 0; s < NSUB; ++s)                                         \
              MMA16(acc[s][j][0], acc[s][j][1], acc[s][j][2], acc[s][j][3], Bc[s][0][0],           \
                    Bc[s][1][0], Bc[s][0][1], Bc[s][1][1], Ac[0], Ac[1]);                          \
        }                                                                                          \
      } else {                                                                                     \
        unsigned Ac[MSUB][2];                                                                      \
        _Pragma("unroll") for (int j = 0; j < MSUB; ++j) cvt2<SCV>(pick(AV[j], t), sa_[j], Ac[j]); \
        _Pragma("unroll") for (int s = 0; s < NSUB; ++s) {                                         \
          unsigned Bc[2][2];                                                                       \
          cvt2<SCV>(pick(BV[s][0], t), sb_[s][0], Bc[0]);                                          \
          cvt2<SCV>(pick(BV[s][1], t), sb_[s][1], Bc[1]);                                          \
          _Pragma("unroll") for (int j = 0; j < MSUB; ++j)                                         \
              MMA16(acc[s][j][0], acc[s][j][1], acc[s][j][2], acc[s][j][3], Bc[0][0], Bc[1][0],    \
                    Bc[0][1], Bc[1][1], Ac[j][0], Ac[j][1]);                                       \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }

#define HMATH(U, WB, WA, BV, AV)                                                                  \
  {                                                                                               \
    bool tiny_ = false, huge_ = false;                                                            \
    _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                           \
        _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) {                                        \
      const unsigned sb__ = sfsel(WB[s_][i_], ((U) & 1) ? sel1 : sel0) & 255u;                    \
      tiny_ |= sb__ < 3u;                                                                         \
      huge_ |= sb__ >= 240u;                                                                      \
    }                                                                                             \
    _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) {                                         \
      const unsigned sa__ = sfsel(WA[j_], ((U) & 1) ? sel1 : sel0) & 255u;                        \
      tiny_ |= sa__ < 3u;                                                                         \
      huge_ |= sa__ >= 240u;                                                                      \
    }                                                                                             \
    bool loss_ = false;                                                                           \
    if (__any_sync(0xffffffffu, tiny_)) {                                                         \
      _Pragma("unroll") for (int h_ = 0; h_ < 1; ++h_) {                                          \
        unsigned bs_[NSUB][2], as_[MSUB];                                                         \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                       \
            _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) bs_[s_][i_] =                        \
                (sfsel(WB[s_][i_], ((U) & 1) ? sel1 : sel0) & 255u);                              \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) as_[j_] =                             \
            (sfsel(WA[j_], ((U) & 1) ? sel1 : sel0) & 255u);                                      \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_) {                                     \
          loss_ |= loses_chunk<16>(BV[s_][0], bs_[s_][0]);                                        \
          loss_ |= loses_chunk<16>(BV[s_][1], bs_[s_][1]);                                        \
        }                                                                                         \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) loss_ |=                              \
            loses_chunk<16>(AV[j_], as_[j_]);                                                     \
      }                                                                                           \
    }                                                                                             \
    if (__any_sync(0xffffffffu, huge_ | loss_)) {                                                 \
      const bool rb_ = __any_sync(0xffffffffu, huge_);                                            \
      _Pragma("unroll") for (int h_ = 0; h_ < 1; ++h_) {                                          \
        unsigned bs_[NSUB][2], as_[MSUB];                                                         \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                       \
            _Pragma("unroll") for (int i_ = 0; i_ < 2; ++i_) bs_[s_][i_] =                        \
                (sfsel(WB[s_][i_], ((U) & 1) ? sel1 : sel0) & 255u);                              \
        _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) as_[j_] =                             \
            (sfsel(WA[j_], ((U) & 1) ? sel1 : sel0) & 255u);                                      \
        _Pragma("unroll") for (int s_ = 0; s_ < NSUB; ++s_)                                       \
            _Pragma("unroll") for (int j_ = 0; j_ < MSUB; ++j_) {                                 \
          float4 tmp_ =                                                                           \
              make_float4(acc[s_][j_][0], acc[s_][j_][1], acc[s_][j_][2], acc[s_][j_][3]);        \
          tmp_ = rb_ ? precise_chunk_rb<16>(BV[s_][0], BV[s_][1], AV[j_], bs_[s_][0], bs_[s_][1], \
                                            as_[j_], tmp_)                                        \
                     : precise_chunk<16>(BV[s_][0], BV[s_][1], AV[j_], bs_[s_][0], bs_[s_][1],    \
                                         as_[j_], tmp_);                                          \
          acc[s_][j_][0] = tmp_.x;                                                                \
          acc[s_][j_][1] = tmp_.y;                                                                \
          acc[s_][j_][2] = tmp_.z;                                                                \
          acc[s_][j_][3] = tmp_.w;                                                                \
        }                                                                                         \
      }                                                                                           \
    } else {                                                                                      \
      HMATH_NORMAL(U, WB, WA, BV, AV)                                                             \
    }                                                                                             \
  }

    if constexpr (PIPE) {
      // Two 64-element units in flight: the second unit's global loads issue
      // before the first unit's conversions consume it, so a warp that owns
      // more than one unit eats one memory latency instead of one per unit.
      Index u = ubeg + wk;
      for (; u + WK < uend; u += 2 * WK) {
        unsigned wb1[NSUB][2], wa1[MSUB], wb2[NSUB][2], wa2[MSUB];
        uint4 Bv1[NSUB][2], Av1[MSUB], Bv2[NSUB][2], Av2[MSUB];
        HLOAD(u, wb1, wa1, Bv1, Av1)
        HLOAD(u + WK, wb2, wa2, Bv2, Av2)
        HMATH(u, wb1, wa1, Bv1, Av1)
        HMATH(u + WK, wb2, wa2, Bv2, Av2)
      }
      if (u < uend) {
        unsigned wb1[NSUB][2], wa1[MSUB];
        uint4 Bv1[NSUB][2], Av1[MSUB];
        HLOAD(u, wb1, wa1, Bv1, Av1)
        HMATH(u, wb1, wa1, Bv1, Av1)
      }
    } else {
      for (Index u = ubeg + wk; u < uend; u += WK) {
        unsigned wb1[NSUB][2], wa1[MSUB];
        uint4 Bv1[NSUB][2], Av1[MSUB];
        HLOAD(u, wb1, wa1, Bv1, Av1)
        HMATH(u, wb1, wa1, Bv1, Av1)
      }
    }
#undef HLOAD
#undef HMATH_NORMAL
#undef HMATH
  } else if constexpr (GPW == 1) {
    // Exact measured routes with one 128-element group per K warp.  Exposing
    // this at compile time removes runtime K bounds, loop control and tail
    // handling from the latency-critical single-wave kernel.
    unsigned wb1[NSUB][2], wa1[MSUB];
    uint4 Bv1[NSUB][2][2], Av1[MSUB][2];
    LOADG(gbeg + wk, wb1, wa1, Bv1, Av1)
    MATHG(wb1, wa1, Bv1, Av1, acc)
  } else if constexpr (GPW == 2 && PIPE) {
    unsigned wb1[NSUB][2], wa1[MSUB], wb2[NSUB][2], wa2[MSUB];
    uint4 Bv1[NSUB][2][2], Av1[MSUB][2], Bv2[NSUB][2][2], Av2[MSUB][2];
    LOADG(gbeg + wk, wb1, wa1, Bv1, Av1)
    if constexpr (Q32 && CM == 24 && CN == 5120 && CK == 2048) {
      LOADGBQ(gbeg + wk + WK, wb2, Bv2)
      MATHG(wb1, wa1, Bv1, Av1, acc)
      LOADGAQ(gbeg + wk + WK, wa2, Av2)
    } else {
      LOADG(gbeg + wk + WK, wb2, wa2, Bv2, Av2)
      MATHG(wb1, wa1, Bv1, Av1, acc)
    }
    if constexpr (DACC) {
      MATHG(wb2, wa2, Bv2, Av2, acc2)
    } else {
      MATHG(wb2, wa2, Bv2, Av2, acc)
    }
  } else if constexpr (GPW == 2) {
    {
      unsigned wb1[NSUB][2], wa1[MSUB];
      uint4 Bv1[NSUB][2][2], Av1[MSUB][2];
      LOADG(gbeg + wk, wb1, wa1, Bv1, Av1)
      MATHG(wb1, wa1, Bv1, Av1, acc)
    }
    {
      unsigned wb1[NSUB][2], wa1[MSUB];
      uint4 Bv1[NSUB][2][2], Av1[MSUB][2];
      LOADG(gbeg + wk + WK, wb1, wa1, Bv1, Av1)
      if constexpr (DACC) {
        MATHG(wb1, wa1, Bv1, Av1, acc2)
      } else {
        MATHG(wb1, wa1, Bv1, Av1, acc)
      }
    }
  } else if constexpr (PIPE) {
    // Two k groups deep: the second group's global loads are issued before the
    // first group's conversions consume the first, so the warp only eats one
    // memory latency per pair instead of one per group.
    Index grp = gbeg + wk;
    for (; grp + WK < gLast; grp += 2 * WK) {
      unsigned wb1[NSUB][2], wa1[MSUB], wb2[NSUB][2], wa2[MSUB];
      uint4 Bv1[NSUB][2][2], Av1[MSUB][2], Bv2[NSUB][2][2], Av2[MSUB][2];
      LOADG(grp, wb1, wa1, Bv1, Av1)
      LOADG(grp + WK, wb2, wa2, Bv2, Av2)
      MATHG(wb1, wa1, Bv1, Av1, acc)
      if constexpr (DACC) {
        MATHG(wb2, wa2, Bv2, Av2, acc2)
      } else {
        MATHG(wb2, wa2, Bv2, Av2, acc)
      }
    }
    if (grp < gLast) {
      unsigned wb1[NSUB][2], wa1[MSUB];
      uint4 Bv1[NSUB][2][2], Av1[MSUB][2];
      LOADG(grp, wb1, wa1, Bv1, Av1)
      MATHG(wb1, wa1, Bv1, Av1, acc)
    }
  } else {
    for (Index grp = gbeg + wk; grp < gLast; grp += WK) {
      unsigned wb1[NSUB][2], wa1[MSUB];
      uint4 Bv1[NSUB][2][2], Av1[MSUB][2];
      LOADG(grp, wb1, wa1, Bv1, Av1)
      MATHG(wb1, wa1, Bv1, Av1, acc)
    }
  }
#undef LOADG
#undef LOADGBQ
#undef LOADGAQ
#undef MATHG_NORMAL
#undef MATHG

  // ragged tail: nkb not a multiple of 4 (only the last group is partial).
  // One 32-element block at a time, so the lane stride drops back to 8 bytes;
  // A and B still share one k permutation inside each mma.
  if constexpr (!HALF && GPW == 0)
    if (gfull < gend && gfull >= gbeg && ((gfull - gbeg) % WK) == wk) {
      const int rem = nkb & 3;
      const Offset soff = Offset(gfull) * 512;
      unsigned wb[NSUB][2], wa[MSUB];
#pragma unroll
      for (int s = 0; s < NSUB; ++s)
#pragma unroll
        for (int i = 0; i < 2; ++i) wb[s][i] = ldg32(bsf + sboff[s][i] + soff);
#pragma unroll
      for (int j = 0; j < MSUB; ++j) wa[j] = ldg32(asf + saoff[j] + soff);
      for (int c = 0; c < rem; ++c) {
        const Index kbase = ((gfull << 2) + c) * 32 - Index(q) * (Q32 ? 24 : 8);
        uint2 Bv[NSUB][2], Av[MSUB];
        bool loss = false, huge = false;
#pragma unroll
        for (int s = 0; s < NSUB; ++s)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            Bv[s][i] = ldg64(bq + boff[s][i] + kbase);
            loss |= loses_chunk<8>(make_uint4(Bv[s][i].x, Bv[s][i].y, 0, 0),
                                   (wb[s][i] >> (8 * c)) & 255u);
            huge |= ((wb[s][i] >> (8 * c)) & 255u) >= 240u;
          }
#pragma unroll
        for (int j = 0; j < MSUB; ++j) {
          Av[j] = ldg64(aq + aoff[j] + kbase);
          loss |= loses_chunk<8>(make_uint4(Av[j].x, Av[j].y, 0, 0), (wa[j] >> (8 * c)) & 255u);
          huge |= ((wa[j] >> (8 * c)) & 255u) >= 240u;
        }
        if (__any_sync(0xffffffffu, loss | huge)) {
          const bool rb = __any_sync(0xffffffffu, huge);
#pragma unroll
          for (int s = 0; s < NSUB; ++s)
#pragma unroll
            for (int j = 0; j < MSUB; ++j) {
              float4 tmp = make_float4(acc[s][j][0], acc[s][j][1], acc[s][j][2], acc[s][j][3]);
              const uint4 b0 = make_uint4(Bv[s][0].x, Bv[s][0].y, 0, 0);
              const uint4 b1 = make_uint4(Bv[s][1].x, Bv[s][1].y, 0, 0);
              const uint4 a0 = make_uint4(Av[j].x, Av[j].y, 0, 0);
              const unsigned sb0 = (wb[s][0] >> (8 * c)) & 255u;
              const unsigned sb1 = (wb[s][1] >> (8 * c)) & 255u;
              const unsigned sa0 = (wa[j] >> (8 * c)) & 255u;
              tmp = rb ? precise_chunk_rb<8>(b0, b1, a0, sb0, sb1, sa0, tmp)
                       : precise_chunk<8>(b0, b1, a0, sb0, sb1, sa0, tmp);
              acc[s][j][0] = tmp.x;
              acc[s][j][1] = tmp.y;
              acc[s][j][2] = tmp.z;
              acc[s][j][3] = tmp.w;
            }
        } else {
          unsigned Bc[NSUB][2][4], Ac[MSUB][4];
#pragma unroll
          for (int s = 0; s < NSUB; ++s) {
            cvt4<SCV>(Bv[s][0], sfbf(wb[s][0], c), Bc[s][0]);
            cvt4<SCV>(Bv[s][1], sfbf(wb[s][1], c), Bc[s][1]);
          }
#pragma unroll
          for (int j = 0; j < MSUB; ++j) cvt4<SCV>(Av[j], sfbf(wa[j], c), Ac[j]);
#pragma unroll
          for (int j = 0; j < MSUB; ++j)
#pragma unroll
            for (int s = 0; s < NSUB; ++s) {
              MMA16(acc[s][j][0], acc[s][j][1], acc[s][j][2], acc[s][j][3], Bc[s][0][0],
                    Bc[s][1][0], Bc[s][0][1], Bc[s][1][1], Ac[j][0], Ac[j][1]);
              MMA16(acc[s][j][0], acc[s][j][1], acc[s][j][2], acc[s][j][3], Bc[s][0][2],
                    Bc[s][1][2], Bc[s][0][3], Bc[s][1][3], Ac[j][2], Ac[j][3]);
            }
        }
      }
    }

  if constexpr (DACC) {
#pragma unroll
    for (int s = 0; s < NSUB; ++s)
#pragma unroll
      for (int j = 0; j < MSUB; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[s][j][c] += acc2[s][j][c];
  }

  // ---- fold the WK k-partitions through smem, then reduce across the cluster ----
  // Every fold issues all of its partial loads up front: a serial chain of
  // dependent shared/dsmem loads costs far more than the arithmetic it feeds.
  __shared__ float sred[WK][MP][NTP];
  __shared__ float4 rbuf[CLUSTER ? (8 * MSUB * (16 * NSUB * WN / 4) + 8) : 1];
#pragma unroll
  for (int s = 0; s < NSUB; ++s)
#pragma unroll
    for (int j = 0; j < MSUB; ++j) {
      const int nn = wn * NT + s * 16 + g;
      sred[wk][8 * j + 2 * q + 0][nn] = acc[s][j][0];
      sred[wk][8 * j + 2 * q + 1][nn] = acc[s][j][1];
      sred[wk][8 * j + 2 * q + 0][nn + 8] = acc[s][j][2];
      sred[wk][8 * j + 2 * q + 1][nn + 8] = acc[s][j][3];
    }
  __syncthreads();

  constexpr int NV = NTOT / 4;  // 16-byte output vectors per m row
  // Exact non-cluster routes only need to fold rows that can reach the output.
  // MMA/shared writes keep their existing rectangular shape, but M1/M2/M4 and
  // M12 no longer load and reduce padded rows that the final store discards.
  constexpr int VROWS = (!CLUSTER && CM > 0 && CM < MP) ? CM : MP;
  constexpr int NE = VROWS * NV;
  const Index mrem = M - mbase;
  const int mlim = AFFA ? MP : (mrem < MP ? mrem : MP);
  if constexpr (!CLUSTER) {
    for (int raw = tid; raw < NE; raw += NTH) {
      int idx = raw;
      // The 8-row permutation only preserves a complete 32-vector group.
      // Exact-row pruning can leave a partial group (M2/M4/M12); applying the
      // permutation there maps some valid vectors into the pruned rows.
      if constexpr (NV == 4 && VROWS == MP) {
        const int l = raw & 31;
        const int q4 = l >> 2;
        idx = (raw & ~31) + ((((q4 >> 1) + ((q4 & 1) << 2)) << 2) | (l & 3));
      }
      const int m = idx / NV;
      const int n = (idx - m * NV) * 4;
      float4 t[WK];
#pragma unroll
      for (int w = 0; w < WK; ++w) {
        const float* p = &sred[w][m][n];
        t[w] = make_float4(p[0], p[1], p[2], p[3]);
      }
      t[0] = fold4<WK>(t);
      if (m < mlim && n < N - nbase) {
        store4<AFFB>(out + (size_t)(mbase + m) * N + nbase + n, t[0], nbase + n, N);
      }
    }
    return;
  }

  if constexpr (CLUSTER) {
    // Push every local partial into the DSMEM of the rank that owns its output
    // element.  One cluster barrier then makes the rank-local reduction buffer
    // complete, avoiding CLS latency-exposed peer reads and a retirement barrier.
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    const int rank = (int)cta_rank();
    constexpr int EC2 = NE / 2;
    constexpr int EC4 = NE / 4;
    constexpr int EC6 = (NE + 5) / 6;  // rounded up: NE need not divide by 6
    constexpr int EC8 = NE / 8;
    const int ec =
        (CLSC == 4)
            ? EC4
            : ((CLSC == 6) ? EC6
                           : ((CLSC == 8) ? EC8 : ((CLS == 2) ? EC2 : ((CLS == 4) ? EC4 : EC8))));
    if constexpr (FS) {
      const int q0 = rank * ec;
      const int q1 = (q0 + ec < NE) ? q0 + ec : NE;
      if (tid == 0) mbar_expect(cmb, (unsigned)(cls * (q1 - q0) * 16));
    }
    for (int raw = tid; raw < NE; raw += NTH) {
      int idx = raw;
      if constexpr (NV == 4) {
        const int l = raw & 31;
        const int q4 = l >> 2;
        idx = (raw & ~31) + ((((q4 >> 1) + ((q4 & 1) << 2)) << 2) | (l & 3));
      }
      const int m = idx / NV;
      const int n = (idx - m * NV) * 4;
      float4 t[WK];
#pragma unroll
      for (int w = 0; w < WK; ++w) {
        const float* p = &sred[w][m][n];
        t[w] = make_float4(p[0], p[1], p[2], p[3]);
      }
      t[0] = fold4<WK>(t);
      int own, local;
      if constexpr (CLSC == 4) {
        own = idx / EC4;
        local = idx - own * EC4;
      } else if constexpr (CLSC == 6) {
        own = idx / EC6;
        local = idx - own * EC6;
      } else if constexpr (CLSC == 8) {
        own = idx / EC8;
        local = idx - own * EC8;
      } else if (CLS == 2) {
        own = idx / EC2;
        local = idx - own * EC2;
      } else if (CLS == 4) {
        own = idx / EC4;
        local = idx - own * EC4;
      } else {
        own = idx / EC8;
        local = idx - own * EC8;
      }
      if constexpr (FS)
        st_async_f4(&rbuf[rank * ec + local], cmb, (unsigned)own, t[0]);
      else
        st_rank_f4(&rbuf[rank * ec + local], (unsigned)own, t[0]);
    }
    if constexpr (FS)
      mbar_wait<MBH>(cmb);
    else
      cluster_sync();
    const int e0 = rank * ec;
    const int e1 = (e0 + ec < NE) ? e0 + ec : NE;
    for (int idx = e0 + tid; idx < e1; idx += NTH) {
      const int local = idx - e0;
      float4 sum = rbuf[local];
      if constexpr (CLSC == 4) {
        float4 p1 = rbuf[ec + local];
        const float4 p2 = rbuf[2 * ec + local];
        const float4 p3 = rbuf[3 * ec + local];
        sum = add4(sum, p2);
        p1 = add4(p1, p3);
        sum = add4(sum, p1);
      } else if constexpr (CLSC == 6) {
        float4 p[5];
#pragma unroll
        for (int r = 0; r < 5; ++r) p[r] = rbuf[(r + 1) * ec + local];
        p[0] = add4(p[0], p[3]);
        p[1] = add4(p[1], p[4]);
        sum = add4(sum, p[2]);
        p[0] = add4(p[0], p[1]);
        sum = add4(sum, p[0]);
      } else if constexpr (CLSC == 8) {
        float4 p[7];
#pragma unroll
        for (int r = 0; r < 7; ++r) p[r] = rbuf[(r + 1) * ec + local];
#pragma unroll
        for (int r = 0; r < 3; ++r) p[r] = add4(p[r], p[r + 4]);
        sum = add4(sum, p[3]);
        p[0] = add4(p[0], p[2]);
        sum = add4(sum, p[1]);
        sum = add4(sum, p[0]);
      } else if (CLS == 2) {
        sum = add4(sum, rbuf[ec + local]);
      } else if (CLS == 4) {
        float4 p1 = rbuf[ec + local];
        const float4 p2 = rbuf[2 * ec + local];
        const float4 p3 = rbuf[3 * ec + local];
        sum = add4(sum, p2);
        p1 = add4(p1, p3);
        sum = add4(sum, p1);
      } else {
        float4 p[7];
#pragma unroll
        for (int r = 0; r < 7; ++r) p[r] = rbuf[(r + 1) * ec + local];
#pragma unroll
        for (int r = 0; r < 3; ++r) p[r] = add4(p[r], p[r + 4]);
        sum = add4(sum, p[3]);
        p[0] = add4(p[0], p[2]);
        sum = add4(sum, p[1]);
        sum = add4(sum, p[0]);
      }
      const int m = idx / NV;
      const int n = (idx - m * NV) * 4;
      if (m < mlim && n < N - nbase) {
        store4<AFFB>(out + (size_t)(mbase + m) * N + nbase + n, sum, nbase + n, N);
      }
    }
  }
}

}  // namespace mxfp8_smallm_sm107
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_MM_MXFP8_SMALLM_SM107_CUH_
