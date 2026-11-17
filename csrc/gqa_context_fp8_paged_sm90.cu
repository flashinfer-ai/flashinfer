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

// Specialized SM90 batch-prefill (chunked-prefill context) attention for an
// fp8 (e4m3) query over an fp8 (e4m3) paged KV cache: GQA 8 query heads over
// 1 KV head, head_dim 128, page size 16, NHD layout, causal, bf16 output.
// Per-sequence float32 Q/K/V descales ((batch, num_kv_heads) ABI; the K and Q
// descales fold into the softmax scale, the V descale folds into the final
// normalisation).
//
// Algorithm: a Hopper warpgroup kernel. QK^T runs as fp8 wgmma
// (m64n64k32 e4m3 x e4m3, fp32 accumulate) over K-major staged Q and K tiles
// (register-resident Q, cp.async two-stage K/V pipeline through dynamic
// shared memory, BN=96 key tiles); softmax statistics are fp32 online-softmax
// with a deferred-renormalisation cap; P is quantised once to e4m3 and the
// P.V product runs as fp8 wgmma (m64n128k32) against a keys-contiguous V^T
// scratch that a small pass (`vt_build`) builds from the paged V cache (every
// covered physical page exactly once per call, also emitting the exact
// long-query tile list); the fp32 O accumulator is renormalised and rounded
// once to bf16. Short-query (decode-like, q_len <= 16) requests run split-KV
// with an fp32 partial-state merge (`combine_kernel`) — packed q_len <= 8
// rows in a 128-thread kernel, the rest at full width; long-query requests
// run one CTA per 16-token query tile, optionally split over the KV axis for
// wave balance (plan-time `psplit`, compiled as a static specialisation, with
// fp16 partial states merged by `combine_prefill`). A single-request prefill
// step skips the decode-side kernels entirely.
//
// OUTPUT PRECISION (disclosed in the Python dispatcher's docstring): the P.V
// product accumulates the e4m3-quantised P against the e4m3 V — an in-class
// quantisation for an fp8-Q/fp8-KV contract (the FA3-style fp8 kernels this
// contract is otherwise served by quantise P the same way), NOT a downgrade of
// a higher-precision input contract. The error is deterministic (the kernel is
// atomics-free and bit-stable) and scales with input magnitude.
//
// OPERAND RANGE (f16 accumulation): the QK^T scores and the unnormalised P.V
// partial sums accumulate in packed f16 (the measured kernel's
// register-budget choice; wgmma_qk16_r / wgmma_pv8_h below). Raw per-row dot
// products (sum over the 128 dims of q_k * k_k, before the descales fold in)
// and per-split unnormalised P.V sums must therefore stay inside f16 range
// (|x| <= 65504); operands quantised to exploit the full e4m3 range (values
// up to 448) can push either past it and saturate to +-inf. The measured
// serving contract (calibrated per-sequence descales over O(1)-magnitude
// quantised operands) sits well inside the bound; widening both accumulators
// to fp32 is a different (slower) kernel, not a patch to this one.
//
// Host contract of the device code (asserted by the Python dispatcher at plan
// time, never read back from the device):
//  * q / out are contiguous [total_q, 8, 128] (1024-element rows);
//  * k_cache / v_cache are contiguous [num_pages, 16, 1, 128] e4m3 (2048-byte
//    pages);
//  * block_table is a contiguous int32 [batch, max_pages_per_seq] table of
//    physical page ids, one leading run per request (zero-padded tails are
//    never dereferenced past a request's page count);
//  * seq_lens (kv length) >= per-request q_len (chunked-prefill alignment:
//    query row `pos` of request r sees kv_len_r - q_len_r + pos + 1 keys);
//  * every referenced physical page id is < vt_pages = vt.numel() / 2048, and
//    v_cache holds at least vt_pages pages (vt_build transforms pages
//    [0, vt_pages) exactly once per call);
//  * the plan-time workspace relations checked in `run` below hold
//    (po/pml split-KV partial states, po2/pml2 prefill partial states when
//    psplit > 1, int2 task list sized for every 16-row query tile).
//
// `init` must run once per device before the first `run` (dynamic smem
// opt-in), outside CUDA-graph capture. `run` performs no allocation, no
// synchronisation and no attribute queries: it launches on the caller's
// current stream and is CUDA-graph capturable.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#define HQ 8
#define DIMH 128
#define PAGE 16
#define BQ 16
#define BN 96
#define NWARP 8
#define NTHREAD (NWARP * 32)
#define KPITCH 144 /* bytes per key row in Ksh */
#define KSTAGE 2
#define VSTAGE 2
#define VPITCH 16384 /* bytes per V^T stage (16 atoms x 8dim x 128B) */
#define NSPLIT_MAX 64
#define MIN_TILES_PER_SPLIT 6

__device__ __forceinline__ uint32_t sm_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

template <bool WIDE>
__device__ __forceinline__ const uint8_t* page_addr(const uint8_t* base, int page, uint32_t tail) {
  if (WIDE) return base + (size_t)(uint32_t)page * 2048 + tail;
  return base + (uint32_t)page * 2048u + tail;
}

__device__ __forceinline__ int split_count(int tiles) {
  int ns = tiles / MIN_TILES_PER_SPLIT;
  if (ns < 1) ns = 1;
  if (ns > NSPLIT_MAX) ns = NSPLIT_MAX;
  return ns;
}

#define MMA_F8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)       \
  asm volatile(                                              \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)               \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

#define MMA_F16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)      \
  asm volatile(                                              \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "   \
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)               \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

#define LDMT4(r0, r1, r2, r3, a, off)                                                          \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4+" #off "];\n" \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)                                        \
               : "r"(a))

#define LDS32(d, a, off) asm volatile("ld.shared.b32 %0, [%1+" #off "];\n" : "=r"(d) : "r"(a))

__device__ __forceinline__ uint64_t make_wgmma_desc(const void* p, uint32_t dim_k_bytes,
                                                    uint32_t dim_mn_bytes) {
  const uint64_t addr = (uint64_t)sm_u32(p);
  return ((addr & 0x3ffffu) >> 4) | ((((uint64_t)dim_k_bytes & 0x3ffffu) >> 4) << 16) |
         ((((uint64_t)dim_mn_bytes & 0x3ffffu) >> 4) << 32);
}

__device__ __forceinline__ uint64_t make_wgmma_desc_sw128(const void* p, uint32_t dim_k_bytes,
                                                          uint32_t dim_mn_bytes) {
  return make_wgmma_desc(p, dim_k_bytes, dim_mn_bytes) | (1ull << 62);
}

__device__ __forceinline__ void wgmma_pv(float (&d)[16][4], const uint32_t (&a)[4],
                                         uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "{%64,%65,%66,%67},%68,1,1,1,1;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]),
        "+f"(d[15][2]), "+f"(d[15][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b)
      : "memory");
}

__device__ __forceinline__ void wgmma_qk(float (&d)[8][4], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "%32,%33,1,1,1;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(desc_a), "l"(desc_b)
      : "memory");
}

/* QK with f16 accumulator: 16 packed registers instead of 32 f32 registers.
   The result is already paired for the f16x2 softmax and e4m3x2 quantizer. */
__device__ __forceinline__ void wgmma_qk16(uint32_t (&d)[24], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n96k32.f16.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23},%24,%25,1,1,1;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
        "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
        "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
        "+r"(d[21]), "+r"(d[22]), "+r"(d[23])
      : "l"(desc_a), "l"(desc_b)
      : "memory");
}

__device__ __forceinline__ uint32_t cvt_f8x2_f16x2(unsigned short v) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(v));
  return r;
}

__device__ __forceinline__ void cvt16(uint4 in, uint32_t out[8]) {
  const uint32_t* w = (const uint32_t*)&in;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i + 0] = cvt_f8x2_f16x2((unsigned short)(w[i] & 0xffffu));
    out[2 * i + 1] = cvt_f8x2_f16x2((unsigned short)(w[i] >> 16));
  }
}

__device__ __forceinline__ uint32_t pack_h2(float v) {
  __half2 h = __float2half2_rn(v);
  return *(const uint32_t*)&h;
}

__device__ __forceinline__ unsigned short cvt2_e4m3(float hi, float lo) {
  unsigned short r;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

/* fp8 PV: A = quantised P in registers, B = V^T (K-major, 128B swizzle) */
__device__ __forceinline__ void wgmma_qk16_r(uint32_t (&d)[24], const uint32_t* a,
                                             uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n96k32.f16.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23},{%24,%25,%26,%27},%28,1,1,1;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
        "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
        "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
        "+r"(d[21]), "+r"(d[22]), "+r"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b)
      : "memory");
}

__device__ __forceinline__ void wgmma_pv8(float (&d)[16][4], const uint32_t* a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "{%64,%65,%66,%67},%68,1,1,1;\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]),
        "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]),
        "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]),
        "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]),
        "+f"(d[11][3]), "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]),
        "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[15][0]), "+f"(d[15][1]),
        "+f"(d[15][2]), "+f"(d[15][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b)
      : "memory");
}

/* Packed-f16 PV accumulation halves the accumulator register footprint and
   shortens the dependent WGMMA drain. */
__device__ __forceinline__ void wgmma_pv8_h(uint32_t (&d)[32], const uint32_t* a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
      "{%32,%33,%34,%35},%36,1,1,1;\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]),
        "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]),
        "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]),
        "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]),
        "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b)
      : "memory");
}

/* within-page key that supplies PV slot j (bit permutation k = [s0,s2,s3,s1]) */
#define VPERM(j) \
  (((j) & 1) | ((((j) >> 2) & 1) << 1) | ((((j) >> 3) & 1) << 2) | ((((j) >> 1) & 1) << 3))

/* Build the transposed V scratch: one CTA per paged KV block, emitting a
   [128 dim][16 slot] fp8 tile so the PV wgmma can read V as a K-major
   (keys-contiguous) B operand without any in-kernel transpose. */
extern "C" __global__ __launch_bounds__(128, 8) void vt_build(const uint8_t* __restrict__ vcp,
                                                              uint8_t* __restrict__ gvt,
                                                              const int* __restrict__ cup,
                                                              int2* __restrict__ tasks, int B,
                                                              int npt) {
  /* Every physical cache page is transformed exactly once.  The former
     (NB,B) launch revisited the same page-table address space and created
     many empty CTAs for ragged batches. */
  const int pg = blockIdx.x;
  __shared__ __align__(16) uint8_t buf[2048];
  const int tid = threadIdx.x;
  if (pg == 0)
    for (int i = tid; i < npt; i += blockDim.x) tasks[i] = make_int2(-1, 0);
  *(uint4*)(buf + tid * 16) = *(const uint4*)(vcp + (size_t)pg * 2048 + tid * 16);
  __syncthreads();
  if (pg == 0 && tid == 0) {
    int ntask = 0;
    for (int r = 0; r < B; ++r) {
      const int qlen = cup[r + 1] - cup[r];
      const int nt = qlen > BQ ? (qlen + BQ - 1) / BQ : 0;
      for (int bx = 0; bx < nt; ++bx) tasks[ntask++] = make_int2(r, bx);
    }
  }
  uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (int j = 0; j < 16; ++j) w[j >> 2] |= (uint32_t)buf[VPERM(j) * 128 + tid] << ((j & 3) * 8);
  *(uint4*)(gvt + (size_t)pg * 2048 + tid * 16) = make_uint4(w[0], w[1], w[2], w[3]);
}

/* Build an exact compact list of long-query tiles on the current stream.
   This small planning launch removes the rectangular B x max_q grid without
   adding a request scan or CTA barrier to every attention tile. */
extern "C" __global__ __launch_bounds__(128,
                                        1) void build_prefill_tasks(const int* __restrict__ cup,
                                                                    int2* __restrict__ tasks, int B,
                                                                    int npt) {
  for (int i = threadIdx.x; i < npt; i += blockDim.x) tasks[i] = make_int2(-1, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    int out = 0;
    for (int r = 0; r < B; ++r) {
      const int qlen = cup[r + 1] - cup[r];
      const int nt = qlen > BQ ? (qlen + BQ - 1) / BQ : 0;
      for (int bx = 0; bx < nt; ++bx) tasks[out++] = make_int2(r, bx);
    }
  }
}

template <int MODE, bool WIDE, int PS = 0> /* 0: direct, 1: split prefill, 2/3: decode */
__global__ __launch_bounds__((MODE == 3 ? 128 : NTHREAD), 2) void attn_kernel(
    const uint8_t* __restrict__ qp, const uint8_t* __restrict__ kcp,
    const uint8_t* __restrict__ vcp, const int* __restrict__ btp, const int* __restrict__ slp,
    const int* __restrict__ cup, const float* __restrict__ qds, const float* __restrict__ kds,
    const float* __restrict__ vds, __nv_bfloat16* __restrict__ outp, float* __restrict__ po,
    float2* __restrict__ pml, int NB, int B, __half* __restrict__ po2, float2* __restrict__ pml2,
    int TQ, int PSPLIT, const int2* __restrict__ tasks) {
  constexpr bool PACK = (MODE == 3);
  constexpr bool SPLIT = (MODE >= 2);
  constexpr bool PART = (MODE == 1);
  int r, bx;
  if (SPLIT) {
    r = blockIdx.y;
    bx = blockIdx.x;
  } else {
    const int2 task = tasks[blockIdx.x];
    r = task.x;
    bx = task.y;
    if (r < 0) return;
  }
  const int q0 = cup[r], q1 = cup[r + 1];
  const int qlen = q1 - q0;
  int p0;
  if (SPLIT) {
    if (qlen <= 0 || qlen > BQ) return;
    if (PACK && qlen > 8) return;
    if (!PACK && qlen <= 8) return;
    p0 = 0;
  } else {
    if (qlen <= BQ) return;
    p0 = bx * BQ;
    if (p0 >= qlen) return;
  }

  const int kvlen = slp[r];
  const int npages = (kvlen + PAGE - 1) / PAGE;
  const int base = kvlen - qlen; /* row pos sees keys <= base+pos */
  const int lastrow = min(p0 + BQ - 1, qlen - 1);
  const int maxkey = base + lastrow;  /* inclusive */
  const int nkv = (maxkey + BN) / BN; /* ceil((maxkey+1)/BN) */
  const int minbound = base + p0;

  int tb = 0, te = nkv;
  const int zs = PART ? (int)blockIdx.z : 0;
  if (SPLIT) {
    const int ns = split_count(nkv);
    if (bx >= ns) return;
    tb = (nkv * bx) / ns;
    te = (nkv * (bx + 1)) / ns;
  } else if (PART) {
    const int ps = PS ? PS : PSPLIT;
    const int per = (nkv + ps - 1) / ps;
    tb = zs * per;
    te = min(nkv, tb + per);
    if (tb >= nkv) te = tb; /* empty chunk: emit a null partial */
  }

  extern __shared__ __align__(16) char smem[];
  uint8_t* Ksh = (uint8_t*)smem;
  uint8_t* Vsh = (uint8_t*)(smem + KSTAGE * BN * DIMH);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hA = PACK ? 2 * warp : warp;
  const int hB = PACK ? hA + 1 : warp;

  /* Clamp the folded softmax scale from below at the smallest positive f16
     value (2^-24): the packed-f16 softmax multiplies scores by this scale,
     and a pathologically small q/k descale product would otherwise round the
     packed scale to +0 and turn the masked -inf scores into NaN (-inf * 0).
     exp2 of a score scaled by <= 2^-24 is ~1 either way (f16 scores are
     bounded by +-65504, so the exponent magnitude stays <= 2^-8), making the
     clamp the correct uniform-softmax limit of scale -> 0. */
  const float scale =
      fmaxf(0.08838834764831845f * 1.4426950408889634f * qds[r] * kds[r], 5.9604644775390625e-8f);
  const float vdsc = vds[r];

  /* ---- load loop-invariant Q into the wgmma A register fragment ---- */
  const int rowA = lane >> 2, rowB = PACK ? rowA : rowA + 8;
  const int pA = min(p0 + rowA, qlen - 1);
  const int pB = min(p0 + rowB, qlen - 1);
  const uint8_t* qgA = qp + (size_t)(q0 + pA) * 1024 + hA * 128 + (lane & 3) * 4;
  const uint8_t* qgB = qp + (size_t)(q0 + pB) * 1024 + hB * 128 + (lane & 3) * 4;
  uint32_t qa[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = ((i >> 1) << 2) + ((i & 1) << 1);
    qa[idx + 0] = *(const uint32_t*)(qgA + i * 16);
    qa[idx + 1] = *(const uint32_t*)(qgB + i * 16);
  }
  /* ---- loader index setup ---- */
  const int kkey0 = tid >> 3;      /* 0..31 */
  const int koff = (tid & 7) * 16; /* byte offset in row */
  const int page_sel = PACK ? 0 : tid >> 7;
  const uint32_t page_byte = (uint32_t)(tid & 127) << 4;
  const int* btrow = btp + (size_t)r * NB;
  /* V^T staging: 16B chunk = one (dim, page) pair of the transposed tile */
  const int vdim = tid & 127;
  const int vpg = PACK ? 0 : tid >> 7;
  const int vx = vdim & 7;
  const int vsh0 = (vdim >> 3) * 1024 + vx * 128;
  const uint32_t vbyte = (uint32_t)vdim << 4;

  uint32_t acc[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) acc[i] = 0u;
  float mA = -1e30f, mB = -1e30f, lA = 0.f, lB = 0.f;
  const float capd = 8.0f / scale; /* allow p up to 2^8 before refreshing */
  /* These packed constants only change when the lagging row maximum moves. */
  const uint32_t rsc = pack_h2(scale);
  uint32_t thrA = 0xfc00fc00u, thrB = 0xfc00fc00u;
  uint32_t rnA = pack_h2(1e4f * scale), rnB = rnA;

  /* ---- prologue: stage the first K/V tile (prefetch distance 1) ---- */
  if (te > tb) {
#pragma unroll
    for (int i = 0; i < (PACK ? BN / 16 : BN / 32); ++i) {
      const int key = kkey0 + i * (PACK ? 16 : 32);
      const uint32_t ko = (uint32_t)(key * 128 + (((tid & 7) * 16) ^ ((key & 7) * 16)));
      int pg = btrow[min(tb * (BN / PAGE) + (PACK ? i : i * 2 + page_sel), npages - 1)];
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sm_u32(Ksh) + ko),
                   "l"(page_addr<WIDE>(kcp, pg, page_byte)));
      const int vc = PACK ? i : vpg + 2 * i;
      const int pgv = btrow[min(tb * (BN / PAGE) + vc, npages - 1)];
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                       sm_u32(Vsh) + (uint32_t)(vsh0 + ((vc ^ vx) << 4))),
                   "l"(page_addr<WIDE>(vcp, pgv, vbyte)));
    }
    asm volatile("cp.async.commit_group;\n");
  }
  int kbi = 0;

  for (int t = tb; t < te; ++t) {
    const int kv0 = t * BN;
    uint8_t* Kb = Ksh + kbi * (BN * DIMH);
    uint8_t* Vb = Vsh + kbi * VPITCH;
    int kbw = kbi + 1;
    if (kbw >= KSTAGE) kbw -= KSTAGE;
    kbi = kbi + 1;
    if (kbi >= KSTAGE) kbi = 0;

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    /* ---- QK: Hopper FP8 WGMMA, packed f16 accumulator ---- */
    uint32_t sv[24];
#pragma unroll
    for (int j = 0; j < 24; ++j) sv[j] = 0u;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t desc_b0 = make_wgmma_desc_sw128(Kb, 1, 8 * 128);
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      wgmma_qk16_r(sv, qa + kb * 4, desc_b0 + 2ull * kb);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");

    /* Issue the next-tile staging inside the QK wgmma shadow.  The block
       table lookup is a dependent global load feeding the cp.async address,
       and running it while the four chained QK wgmma are still in flight
       takes that ~40-instruction burst off the critical path instead of
       paying it before the GEMM starts. */
    {
      const int tk = min(t + 1, te - 1);
#pragma unroll
      for (int i = 0; i < (PACK ? BN / 16 : BN / 32); ++i) {
        const int key = kkey0 + i * (PACK ? 16 : 32);
        const uint32_t ko = (uint32_t)(key * 128 + (((tid & 7) * 16) ^ ((key & 7) * 16)));
        const int pgk = btrow[min(tk * (BN / PAGE) + (PACK ? i : i * 2 + page_sel), npages - 1)];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                         sm_u32(Ksh + kbw * (BN * DIMH)) + ko),
                     "l"(page_addr<WIDE>(kcp, pgk, page_byte)));
        const int vc = PACK ? i : vpg + 2 * i;
        const int pgv = btrow[min(tk * (BN / PAGE) + vc, npages - 1)];
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                         sm_u32(Vsh + kbw * VPITCH) + (uint32_t)(vsh0 + ((vc ^ vx) << 4))),
                     "l"(page_addr<WIDE>(vcp, pgv, vbyte)));
      }
      asm volatile("cp.async.commit_group;\n");
    }

    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    /* ---- mask + packed f16x2 softmax ---- */
    const int boundA = base + p0 + rowA;
    const int boundB = base + p0 + rowB;
    const bool domask = (kv0 + BN - 1) > minbound;
    uint32_t mxA = thrA, mxB = thrB;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int k0 = kv0 + j * 8 + (lane & 3) * 2;
      if (domask) {
        if (k0 > boundA) sv[2 * j] = (sv[2 * j] & 0xffff0000u) | 0x0000fc00u;
        if (k0 + 1 > boundA) sv[2 * j] = (sv[2 * j] & 0x0000ffffu) | 0xfc000000u;
        if (k0 > boundB) sv[2 * j + 1] = (sv[2 * j + 1] & 0xffff0000u) | 0x0000fc00u;
        if (k0 + 1 > boundB) sv[2 * j + 1] = (sv[2 * j + 1] & 0x0000ffffu) | 0xfc000000u;
      }
      asm("max.f16x2 %0, %0, %1;\n" : "+r"(mxA) : "r"(sv[2 * j]));
      asm("max.f16x2 %0, %0, %1;\n" : "+r"(mxB) : "r"(sv[2 * j + 1]));
    }
    /* Lagging reference max: mA/mB are only refreshed when this tile would
       push a probability past 2^PCAP, which keeps p inside f16 range with a
       wide margin.  The common path compares packed values against the
       persistent threshold and recovers the exact maximum only on refresh. */
    if (__any_sync(0xffffffffu, ((mxA ^ thrA) | (mxB ^ thrB)) != 0u)) {
      uint32_t tmA = 0xfc00fc00u, tmB = 0xfc00fc00u;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        asm("max.f16x2 %0, %0, %1;\n" : "+r"(tmA) : "r"(sv[2 * j]));
        asm("max.f16x2 %0, %0, %1;\n" : "+r"(tmB) : "r"(sv[2 * j + 1]));
      }
      const float2 fmA = __half22float2(*(__half2*)&tmA);
      const float2 fmB = __half22float2(*(__half2*)&tmB);
      float rmA = fmaxf(fmA.x, fmA.y), rmB = fmaxf(fmB.x, fmB.y);
      rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 1));
      rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 2));
      rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 1));
      rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 2));
      const float nmA = fmaxf(fmaxf(mA, rmA), -1e30f);
      const float nmB = fmaxf(fmaxf(mB, rmB), -1e30f);
      const float cA = exp2f((mA - nmA) * scale);
      const float cB = exp2f((mB - nmB) * scale);
      mA = nmA;
      mB = nmB;
      lA *= cA;
      lB *= cB;
      const uint32_t pcA = pack_h2(cA), pcB = pack_h2(cB);
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        asm("mul.rn.f16x2 %0, %0, %1;\n" : "+r"(acc[2 * ii]) : "r"(pcA));
        asm("mul.rn.f16x2 %0, %0, %1;\n" : "+r"(acc[2 * ii + 1]) : "r"(pcB));
      }
      thrA = pack_h2(fminf(mA + capd, 60000.f));
      thrB = pack_h2(fminf(mB + capd, 60000.f));
      /* The renormaliser -m*scale must stay f16-finite, but clamping the raw
         max at -1e4 subtracted the clamp instead of the recorded maximum:
         rows whose scores sit uniformly below -1e4 - ~25/scale (e.g. a
         constant -16384) had every probability flushed to zero and returned
         0 instead of the softmax-weighted value. Clamp in scaled (log2)
         units instead: +-30000 fits f16 with margin and is far beyond exp2's
         dynamic range, so the clamp can only engage where the probabilities
         are 0/inf in any arithmetic. */
      rnA = pack_h2(-fminf(fmaxf(mA * scale, -30000.f), 30000.f));
      rnB = pack_h2(-fminf(fmaxf(mB * scale, -30000.f), 30000.f));
    }
    uint32_t sumA = 0u, sumB = 0u;
    /* Pack P straight into the fp8 wgmma A fragment.  Accumulator group j
       owns keys 8j+2u, 8j+2u+1 of lane u; the matching A slot is the bit
       permutation s=[k0,k3,k1,k2,k4,k5], which V^T is pre-built to match. */
    /* The packed P fragments overwrite score registers whose pairs have
       already been consumed.  This keeps the eight WGMMA-A registers from
       extending the 128-register live set. */
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      uint32_t tA, tB;
      asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(tA) : "r"(sv[2 * j]), "r"(rsc), "r"(rnA));
      asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(tB) : "r"(sv[2 * j + 1]), "r"(rsc), "r"(rnB));
      asm("ex2.approx.f16x2 %0, %0;\n" : "+r"(tA));
      asm("ex2.approx.f16x2 %0, %0;\n" : "+r"(tB));
      asm("add.f16x2 %0, %0, %1;\n" : "+r"(sumA) : "r"(tA));
      asm("add.f16x2 %0, %0, %1;\n" : "+r"(sumB) : "r"(tB));
      unsigned short qa, qb;
      asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(qa) : "r"(tA));
      asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(qb) : "r"(tB));
      const int c = j >> 2, g = ((j >> 1) & 1) * 2;
      if ((j & 1) == 0) {
        sv[c * 4 + g + 0] = (uint32_t)qa;
        sv[c * 4 + g + 1] = (uint32_t)qb;
      } else {
        sv[c * 4 + g + 0] |= (uint32_t)qa << 16;
        sv[c * 4 + g + 1] |= (uint32_t)qb << 16;
      }
      /* Start each complete K32 PV slice while later score pairs are still
         being transformed.  The final slice is issued after the loop. */
      if ((j & 3) == 3 && j + 4 < BN / 8) {
        const int slice = j >> 2;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        const uint64_t desc = make_wgmma_desc_sw128(Vb + slice * 32, 1, 8 * 128);
        wgmma_pv8_h(acc, sv + slice * 4, desc);
      }
    }
    const float2 fsA = __half22float2(*(__half2*)&sumA);
    const float2 fsB = __half22float2(*(__half2*)&sumB);
    lA += fsA.x + fsA.y;
    lB += fsB.x + fsB.y;

    /* ---- final PV slice; all slices retire in the same group. ---- */
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint64_t desc1 = make_wgmma_desc_sw128(Vb + (BN / 32 - 1) * 32, 1, 8 * 128);
    wgmma_pv8_h(acc, sv + (BN / 32 - 1) * 4, desc1);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    /* Inline PTX operands are compiler-synchronous even though WGMMA is not.
       Keep all register-A fragments live through the covering wait. */
    asm volatile("" ::"r"(sv[0]), "r"(sv[1]), "r"(sv[2]), "r"(sv[3]), "r"(sv[4]), "r"(sv[5]),
                 "r"(sv[6]), "r"(sv[7]), "r"(sv[8]), "r"(sv[9]), "r"(sv[10]), "r"(sv[11]));
  }

  /* Denominator lanes share the same lagging reference max and rescale, so
     their partial sums can stay independent through the key loop. */
  lA += __shfl_xor_sync(0xffffffffu, lA, 1);
  lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1);
  lB += __shfl_xor_sync(0xffffffffu, lB, 2);

  /* ---- epilogue ---- */
  if (SPLIT) {
    /* partial results: unnormalised acc + (scaled max, denom) per row/head */
    float* oA = po + ((((size_t)bx * B + r) * BQ + rowA) * HQ + hA) * DIMH + (lane & 3) * 2;
    float* oB = po + ((((size_t)bx * B + r) * BQ + rowB) * HQ + hB) * DIMH + (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float2 aA = __half22float2(*(__half2*)&acc[2 * nt]);
      const float2 aB = __half22float2(*(__half2*)&acc[2 * nt + 1]);
      if (rowA < qlen) *(float2*)(oA + nt * 8) = aA;
      if (rowB < qlen) *(float2*)(oB + nt * 8) = aB;
    }
    if ((lane & 3) == 0) {
      size_t mbase = (((size_t)bx * B + r) * BQ) * HQ;
      if (rowA < qlen) pml[mbase + (size_t)rowA * HQ + hA] = make_float2(mA * scale, lA);
      if (rowB < qlen) pml[mbase + (size_t)rowB * HQ + hB] = make_float2(mB * scale, lB);
    }
    return;
  }
  if (PART) {
    const int gA0 = p0 + rowA, gB0 = p0 + rowB;
    const size_t rowbase = (size_t)zs * TQ + q0;
    if (gA0 < qlen) {
      __half* oA = po2 + ((rowbase + gA0) * HQ + hA) * DIMH + (lane & 3) * 2;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        *(__half2*)(oA + nt * 8) = *(__half2*)&acc[2 * nt];
      }
      if ((lane & 3) == 0) pml2[(rowbase + gA0) * HQ + hA] = make_float2(mA * scale, lA);
    }
    if (gB0 < qlen) {
      __half* oB = po2 + ((rowbase + gB0) * HQ + hB) * DIMH + (lane & 3) * 2;
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        *(__half2*)(oB + nt * 8) = *(__half2*)&acc[2 * nt + 1];
      }
      if ((lane & 3) == 0) pml2[(rowbase + gB0) * HQ + hB] = make_float2(mB * scale, lB);
    }
    return;
  }
  float iA = (lA > 0.f ? 1.f / lA : 0.f) * vdsc;
  float iB = (lB > 0.f ? 1.f / lB : 0.f) * vdsc;
  const int gA = p0 + rowA, gB = p0 + rowB;
  __nv_bfloat16* oA = outp + (size_t)(q0 + gA) * 1024 + hA * 128 + (lane & 3) * 2;
  __nv_bfloat16* oB = outp + (size_t)(q0 + gB) * 1024 + hB * 128 + (lane & 3) * 2;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    if (gA < qlen) {
      const float2 a = __half22float2(*(__half2*)&acc[2 * nt]);
      __nv_bfloat162 v = __floats2bfloat162_rn(a.x * iA, a.y * iA);
      *(__nv_bfloat162*)(oA + nt * 8) = v;
    }
    if (gB < qlen) {
      const float2 a = __half22float2(*(__half2*)&acc[2 * nt + 1]);
      __nv_bfloat162 v = __floats2bfloat162_rn(a.x * iB, a.y * iB);
      *(__nv_bfloat162*)(oB + nt * 8) = v;
    }
  }
}

/* combine split-KV partials for the long-query (prefill) rows */
template <int PS>
__global__ __launch_bounds__(256, 2) void combine_prefill(const __half* __restrict__ po2,
                                                          const float2* __restrict__ pml2,
                                                          const int* __restrict__ cup,
                                                          const float* __restrict__ vds,
                                                          __nv_bfloat16* __restrict__ outp, int TQ,
                                                          int B, const int2* __restrict__ tasks) {
  const int tid = threadIdx.x;
  const int h = tid >> 5;
  const int lane = tid & 31;
  const int2 task = tasks[blockIdx.x];
  if (task.x < 0) return;
  const int rr = task.x;
  const int local_row = task.y * BQ + (int)blockIdx.y;
  if (local_row >= cup[rr + 1] - cup[rr]) return;
  const int row = cup[rr] + local_row;
  const int d = lane * 4;
  const size_t moff = (size_t)row * HQ + h;
  const size_t mstride = (size_t)TQ * HQ;
  float2 ml = lane < PS ? pml2[moff + (size_t)lane * mstride] : make_float2(-INFINITY, 0.f);
  float M = ml.x;
  for (int delta = 4; delta; delta >>= 1) M = fmaxf(M, __shfl_down_sync(0xffffffffu, M, delta));
  M = __shfl_sync(0xffffffffu, M, 0);
  float wl = lane < PS && ml.x > -1e29f ? exp2f(ml.x - M) : 0.f;
  float den = wl * ml.y;
  for (int delta = 4; delta; delta >>= 1) den += __shfl_down_sync(0xffffffffu, den, delta);
  den = __shfl_sync(0xffffffffu, den, 0);
  float num0 = 0.f, num1 = 0.f, num2 = 0.f, num3 = 0.f;
#pragma unroll
  for (int z = 0; z < PS; ++z) {
    float w = __shfl_sync(0xffffffffu, wl, z);
    const __half2* pv = (const __half2*)(po2 + (moff + (size_t)z * mstride) * DIMH + d);
    const float2 v01 = __half22float2(pv[0]);
    const float2 v23 = __half22float2(pv[1]);
    num0 = fmaf(w, v01.x, num0);
    num1 = fmaf(w, v01.y, num1);
    num2 = fmaf(w, v23.x, num2);
    num3 = fmaf(w, v23.y, num3);
  }
  float inv = lane == 0 ? (den > 0.f ? 1.f / den : 0.f) * vds[rr] : 0.f;
  inv = __shfl_sync(0xffffffffu, inv, 0);
  __nv_bfloat16* dst = outp + (size_t)row * 1024 + h * 128 + d;
  *(__nv_bfloat162*)dst = __floats2bfloat162_rn(num0 * inv, num1 * inv);
  *(__nv_bfloat162*)(dst + 2) = __floats2bfloat162_rn(num2 * inv, num3 * inv);
}

/* combine split-KV partials for the short-query (decode) requests */
extern "C" __global__ __launch_bounds__(256, 2) void combine_kernel(
    const float* __restrict__ po, const float2* __restrict__ pml, const int* __restrict__ slp,
    const int* __restrict__ cup, const float* __restrict__ vds, __nv_bfloat16* __restrict__ outp,
    int B) {
  const int r = blockIdx.y;
  const int q0 = cup[r], q1 = cup[r + 1];
  const int qlen = q1 - q0;
  if (qlen <= 0 || qlen > BQ) return;
  const int p = blockIdx.x;
  if (p >= qlen) return;
  const int h = threadIdx.x >> 5;
  const int kvlen = slp[r];
  const int nkv = (kvlen + BN - 1) / BN;
  const int ns = split_count(nkv);
  const int lane = threadIdx.x & 31;
  const int d = lane * 4;

  const size_t moff = (((size_t)r * BQ + p) * HQ + h);
  const size_t ooff = ((((size_t)r * BQ + p) * HQ + h) * DIMH + d);
  const size_t mstride = (size_t)B * BQ * HQ;
  const size_t ostride = (size_t)B * BQ * HQ * DIMH;

  const int z0 = lane;
  const int z1 = lane + 32;
  float2 ml0 = z0 < ns ? pml[moff + (size_t)z0 * mstride] : make_float2(-INFINITY, 0.f);
  float2 ml1 = z1 < ns ? pml[moff + (size_t)z1 * mstride] : make_float2(-INFINITY, 0.f);
  float M = fmaxf(ml0.x, ml1.x);
  for (int delta = 16; delta; delta >>= 1) M = fmaxf(M, __shfl_down_sync(0xffffffffu, M, delta));
  M = __shfl_sync(0xffffffffu, M, 0);
  float w0 = z0 < ns ? exp2f(ml0.x - M) : 0.f;
  float w1 = z1 < ns ? exp2f(ml1.x - M) : 0.f;
  float den = fmaf(w0, ml0.y, w1 * ml1.y);
  for (int delta = 16; delta; delta >>= 1) den += __shfl_down_sync(0xffffffffu, den, delta);
  den = __shfl_sync(0xffffffffu, den, 0);
  float num0 = 0.f, num1 = 0.f, num2 = 0.f, num3 = 0.f;
  for (int z = 0; z < min(ns, 32); ++z) {
    float w = __shfl_sync(0xffffffffu, w0, z);
    float4 v = *(const float4*)(po + ooff + (size_t)z * ostride);
    num0 = fmaf(w, v.x, num0);
    num1 = fmaf(w, v.y, num1);
    num2 = fmaf(w, v.z, num2);
    num3 = fmaf(w, v.w, num3);
  }
  for (int z = 32; z < ns; ++z) {
    float w = __shfl_sync(0xffffffffu, w1, z - 32);
    float4 v = *(const float4*)(po + ooff + (size_t)z * ostride);
    num0 = fmaf(w, v.x, num0);
    num1 = fmaf(w, v.y, num1);
    num2 = fmaf(w, v.z, num2);
    num3 = fmaf(w, v.w, num3);
  }
  float inv = lane == 0 ? (den > 0.f ? 1.f / den : 0.f) * vds[r] : 0.f;
  inv = __shfl_sync(0xffffffffu, inv, 0);
  __nv_bfloat16* dst = outp + (size_t)(q0 + p) * 1024 + h * 128 + d;
  *(__nv_bfloat162*)dst = __floats2bfloat162_rn(num0 * inv, num1 * inv);
  *(__nv_bfloat162*)(dst + 2) = __floats2bfloat162_rn(num2 * inv, num3 * inv);
}

// ---------------------------------------------------------------------------
// Host side. The launch heuristic below is carried verbatim from the measured
// kernel (the measured agent-generated kernel's same-SKU re-time); only the
// workspace ownership moved out of the launcher (plan-time allocation by the
// Python dispatcher) so that `run` is CUDA-graph capturable.
// ---------------------------------------------------------------------------

namespace {
constexpr size_t kSmemBytes = KSTAGE * BN * DIMH + VSTAGE * VPITCH;
}  // namespace

// Plan-time KV-split factor for the long-query (prefill) tiles. Verbatim from
// the measured launcher: wave-balance the prefill tiles, aim for >= ~8 full
// waves of CTAs; partial-state footprint capped at 768 MiB (the measured
// launcher's fp32-denominated cap, kept bit-identical so the split decisions
// match the measured kernel exactly); pure-decode steps never split.
int64_t gqa_context_fp8_paged_sm90_plan_psplit(int64_t total_q, int64_t max_seqlen_q,
                                               int64_t batch_size) {
  const int TQ = static_cast<int>(total_q);
  const int B = static_cast<int>(batch_size);
  /* wave-balance the prefill tiles: aim for >= ~8 full waves of CTAs */
  const int NSM = 264; /* two resident BN32 CTAs on each of 132 H200 SMs */
  int est_tokens = TQ;
  if (TQ - (int)max_seqlen_q <= B * BQ) est_tokens = (int)max_seqlen_q;
  int est = (est_tokens + BQ - 1) / BQ;
  if (est < 1) est = 1;
  int psplit = 1;
  float bv = (float)((est + NSM - 1) / NSM);
  for (int p = 2; p <= 8; ++p) {
    float v = (float)(((long)est * p + NSM - 1) / NSM) / (float)p;
    if (v < bv * 0.98f) {
      bv = v;
      psplit = p;
    }
  }
  if ((size_t)psplit * TQ * HQ * DIMH * sizeof(float) > (size_t)768 * 1024 * 1024) psplit = 1;
  if ((int)max_seqlen_q <= BQ) psplit = 1; /* nothing but decode requests */
  return psplit;
}

// Number of int2 task slots (16-row query tiles) the task buffer must hold.
int64_t gqa_context_fp8_paged_sm90_plan_num_tasks(int64_t total_q, int64_t batch_size) {
  int npt = static_cast<int>((total_q + (BQ - 1) * batch_size) / BQ);
  if (npt < 1) npt = 1;
  return npt;
}

// One-time per-device opt-in to the attention kernels' dynamic shared memory
// (56 KiB > the 48 KiB default), covering every instantiation `run` can
// launch: the four modes in both page-addressing widths plus the
// statically-split prefill specialisations. Outside CUDA-graph capture,
// before the first `run` on the device.
void gqa_context_fp8_paged_sm90_init(int64_t device_id) {
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  using AttnFn = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, const int*, const int*,
                          const int*, const float*, const float*, const float*, __nv_bfloat16*,
                          float*, float2*, int, int, __half*, float2*, int, int, const int2*);
  const AttnFn fns[] = {
      attn_kernel<0, false>,    attn_kernel<1, false>,    attn_kernel<2, false>,
      attn_kernel<3, false>,    attn_kernel<0, true>,     attn_kernel<1, true>,
      attn_kernel<2, true>,     attn_kernel<3, true>,     attn_kernel<1, false, 2>,
      attn_kernel<1, true, 2>,  attn_kernel<1, false, 3>, attn_kernel<1, true, 3>,
      attn_kernel<1, false, 4>, attn_kernel<1, true, 4>,  attn_kernel<1, false, 5>,
      attn_kernel<1, true, 5>,  attn_kernel<1, false, 6>, attn_kernel<1, true, 6>,
      attn_kernel<1, false, 7>, attn_kernel<1, true, 7>,  attn_kernel<1, false, 8>,
      attn_kernel<1, true, 8>,
  };
  for (AttnFn fn : fns) {
    cudaError_t status = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(kSmemBytes));
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "gqa_context_fp8_paged_sm90: cudaFuncSetAttribute failed: "
        << cudaGetErrorString(status);
  }
}

// Launch the measured kernel sequence on the caller's current stream:
// vt_build (V^T scratch over the covered pages + exact long-query tile list),
// attn_kernel<3>/<2> (split-KV decode-like rows, packed and regular — skipped
// on a pure single-request prefill step), then either attn_kernel<0> (direct
// prefill tiles) or the statically-split attn_kernel<1, ·, psplit> +
// combine_prefill<psplit> (KV-split prefill tiles, fp16 partial states), and
// combine_kernel (split-KV merge, skipped with the decode kernels). No
// allocation, no sync, no attribute queries: CUDA-graph capturable. All
// workspace buffers are plan-allocated by the Python dispatcher; psplit comes
// from `plan_psplit` at plan time.
void gqa_context_fp8_paged_sm90_run(TensorView q, TensorView k_cache, TensorView v_cache,
                                    TensorView block_table, TensorView seq_lens,
                                    TensorView cu_seqlens_q, TensorView q_descale,
                                    TensorView k_descale, TensorView v_descale,
                                    int64_t max_seqlen_q, TensorView out, TensorView vt,
                                    TensorView po, TensorView pml, TensorView po2, TensorView pml2,
                                    TensorView tasks, int64_t psplit_) {
  const int B = static_cast<int>(seq_lens.size(0));
  const int NB = static_cast<int>(block_table.size(1));
  const int TQ = static_cast<int>(q.size(0));
  const int psplit = static_cast<int>(psplit_);

  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_DIM(3, q);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK_EQ(q.size(1), HQ);
  TVM_FFI_ICHECK_EQ(q.size(2), DIMH);
  TVM_FFI_ICHECK_EQ(out.size(0), TQ);
  TVM_FFI_ICHECK_EQ(out.size(1), HQ);
  TVM_FFI_ICHECK_EQ(out.size(2), DIMH);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(v_cache);
  CHECK_INPUT_TYPE(k_cache, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(v_cache, dl_float8_e4m3fn);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(4, v_cache);
  for (const TensorView* cache : {&k_cache, &v_cache}) {
    TVM_FFI_ICHECK_EQ(cache->size(1), PAGE) << "page size must be 16";
    TVM_FFI_ICHECK_EQ(cache->size(2), 1) << "exactly one KV head";
    TVM_FFI_ICHECK_EQ(cache->size(3), DIMH) << "KV head_dim must be 128";
    TVM_FFI_ICHECK_EQ(cache->stride(3), 1);
    TVM_FFI_ICHECK_EQ(cache->stride(2), DIMH);
    TVM_FFI_ICHECK_EQ(cache->stride(1), DIMH);
    TVM_FFI_ICHECK_EQ(cache->stride(0), PAGE * DIMH) << "KV pages must be 2048 contiguous bytes";
  }
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_q, dl_int32);
  CHECK_INPUT_AND_TYPE(q_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(k_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(v_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(vt, dl_uint8);
  CHECK_INPUT_AND_TYPE(po, dl_float32);
  CHECK_INPUT_AND_TYPE(pml, dl_float32);
  CHECK_INPUT_AND_TYPE(po2, dl_float32);
  CHECK_INPUT_AND_TYPE(pml2, dl_float32);
  CHECK_INPUT_AND_TYPE(tasks, dl_int32);
  CHECK_DIM(2, block_table);
  TVM_FFI_ICHECK_EQ(block_table.size(0), B);
  TVM_FFI_ICHECK_GE(cu_seqlens_q.size(0), B + 1);
  TVM_FFI_ICHECK_GE(q_descale.numel(), B);
  TVM_FFI_ICHECK_GE(k_descale.numel(), B);
  TVM_FFI_ICHECK_GE(v_descale.numel(), B);

  const int64_t vt_pages = vt.numel() / 2048;
  TVM_FFI_ICHECK_GT(vt_pages, 0);
  TVM_FFI_ICHECK_GE(v_cache.size(0), vt_pages)
      << "V cache must hold every page covered by the plan-sized V^T scratch";
  int npt = (TQ + (BQ - 1) * B) / BQ;
  if (npt < 1) npt = 1;
  TVM_FFI_ICHECK_GE(tasks.numel(), 2 * (int64_t)npt) << "task buffer too small";
  TVM_FFI_ICHECK_GE(po.numel(), (int64_t)NSPLIT_MAX * B * BQ * HQ * DIMH);
  TVM_FFI_ICHECK_GE(pml.numel(), (int64_t)NSPLIT_MAX * B * BQ * HQ * 2);
  TVM_FFI_ICHECK_GE(psplit, 1);
  TVM_FFI_ICHECK_LE(psplit, 8);
  if (psplit > 1) {
    // The split-prefill partial states are fp16 (po2) / 2xfp32 (pml2) on the
    // device side; the dispatcher plan-allocates both as fp32 with the same
    // element counts (ABI kept stable across candidate generations), so the
    // fp16 payload is over-provisioned 2x by construction.
    TVM_FFI_ICHECK_GE(po2.numel(), (int64_t)psplit * TQ * HQ * DIMH);
    TVM_FFI_ICHECK_GE(pml2.numel(), (int64_t)psplit * TQ * HQ * 2);
  }
  CHECK_DEVICE(q, k_cache);
  CHECK_DEVICE(q, v_cache);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, block_table);
  CHECK_DEVICE(q, seq_lens);
  CHECK_DEVICE(q, cu_seqlens_q);
  CHECK_DEVICE(q, q_descale);
  CHECK_DEVICE(q, k_descale);
  CHECK_DEVICE(q, v_descale);
  CHECK_DEVICE(q, vt);
  CHECK_DEVICE(q, po);
  CHECK_DEVICE(q, pml);
  CHECK_DEVICE(q, po2);
  CHECK_DEVICE(q, pml2);
  CHECK_DEVICE(q, tasks);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t s = get_stream(q.device());

  const uint8_t* qd = static_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* kd = static_cast<const uint8_t*>(k_cache.data_ptr());
  const uint8_t* vd = static_cast<const uint8_t*>(v_cache.data_ptr());
  const int* bt = static_cast<const int*>(block_table.data_ptr());
  const int* sl = static_cast<const int*>(seq_lens.data_ptr());
  const int* cu = static_cast<const int*>(cu_seqlens_q.data_ptr());
  const float* qs = static_cast<const float*>(q_descale.data_ptr());
  const float* ks = static_cast<const float*>(k_descale.data_ptr());
  const float* vs = static_cast<const float*>(v_descale.data_ptr());
  __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());
  uint8_t* g_vt = static_cast<uint8_t*>(vt.data_ptr());
  float* g_po = static_cast<float*>(po.data_ptr());
  float2* g_pml = reinterpret_cast<float2*>(pml.data_ptr());
  __half* g_po2 = reinterpret_cast<__half*>(po2.data_ptr());
  float2* g_pml2 = reinterpret_cast<float2*>(pml2.data_ptr());
  int2* g_tasks = reinterpret_cast<int2*>(tasks.data_ptr());
  const size_t smem = kSmemBytes;
  /* page-addressing width follows the physical cache size, as in the
     measured launcher */
  const bool wide_pages = (uint64_t)(uint32_t)v_cache.size(0) * 2048ull > 0xffffffffull;

#define LAUNCH_ATTN(MODE, GRID, BLOCK, SHM)                                                       \
  do {                                                                                            \
    if (wide_pages)                                                                               \
      attn_kernel<MODE, true><<<GRID, BLOCK, SHM, s>>>(qd, kd, g_vt, bt, sl, cu, qs, ks, vs, op,  \
                                                       g_po, g_pml, NB, B, g_po2, g_pml2, TQ,     \
                                                       psplit, g_tasks);                          \
    else                                                                                          \
      attn_kernel<MODE, false><<<GRID, BLOCK, SHM, s>>>(qd, kd, g_vt, bt, sl, cu, qs, ks, vs, op, \
                                                        g_po, g_pml, NB, B, g_po2, g_pml2, TQ,    \
                                                        psplit, g_tasks);                         \
  } while (0)
#define LAUNCH_PART(PVAL)                                                                        \
  do {                                                                                           \
    if (wide_pages)                                                                              \
      attn_kernel<1, true, PVAL><<<dim3(npt, 1, PVAL), NTHREAD, smem, s>>>(                      \
          qd, kd, g_vt, bt, sl, cu, qs, ks, vs, op, g_po, g_pml, NB, B, g_po2, g_pml2, TQ, PVAL, \
          g_tasks);                                                                              \
    else                                                                                         \
      attn_kernel<1, false, PVAL><<<dim3(npt, 1, PVAL), NTHREAD, smem, s>>>(                     \
          qd, kd, g_vt, bt, sl, cu, qs, ks, vs, op, g_po, g_pml, NB, B, g_po2, g_pml2, TQ, PVAL, \
          g_tasks);                                                                              \
    combine_prefill<PVAL>                                                                        \
        <<<dim3(npt, BQ), 256, 0, s>>>(g_po2, g_pml2, cu, vs, op, TQ, B, g_tasks);               \
  } while (0)

  /* One pass over the covered physical pages turns V into the
     keys-contiguous form the fp8 PV wgmma needs (each page is transformed
     exactly once) and writes the exact long-query tile list. */
  vt_build<<<(int)vt_pages, 128, 0, s>>>(vd, g_vt, cu, g_tasks, B, npt);
  const bool all_prefill = B == 1 && TQ > BQ;
  if (!all_prefill) {
    LAUNCH_ATTN(3, dim3(NSPLIT_MAX, B), 128, smem);
    LAUNCH_ATTN(2, dim3(NSPLIT_MAX, B), NTHREAD, smem);
  }
  if (psplit > 1) {
    switch (psplit) {
      case 2:
        LAUNCH_PART(2);
        break;
      case 3:
        LAUNCH_PART(3);
        break;
      case 4:
        LAUNCH_PART(4);
        break;
      case 5:
        LAUNCH_PART(5);
        break;
      case 6:
        LAUNCH_PART(6);
        break;
      case 7:
        LAUNCH_PART(7);
        break;
      default:
        LAUNCH_PART(8);
        break;
    }
  } else {
    LAUNCH_ATTN(0, dim3(npt), NTHREAD, smem);
  }
  if (!all_prefill) combine_kernel<<<dim3(BQ, B), 256, 0, s>>>(g_po, g_pml, sl, cu, vs, op, B);
#undef LAUNCH_ATTN
#undef LAUNCH_PART
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "gqa_context_fp8_paged_sm90 launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gqa_context_fp8_paged_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, gqa_context_fp8_paged_sm90_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_psplit, gqa_context_fp8_paged_sm90_plan_psplit);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_num_tasks, gqa_context_fp8_paged_sm90_plan_num_tasks);
