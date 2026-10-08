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

// Specialized SM90 batch-prefill attention for speculative-decoding target
// verification over an fp8 (e4m3) paged KV cache with a custom bit mask:
// every request contributes exactly 4 query tokens x 4 query heads (16 packed
// rows), all heads read KV head 0 (GQA 4:1), head_dim 256, page size 1.
//
// Algorithm: split-KV. Each CTA (128 threads, 4 warps, 3 CTAs per SM) owns a
// contiguous range of 64-token KV tiles of one request, streams fp8 K/V tiles
// through a 2-stage cp.async pipeline (slot ids prefetched one tile ahead so
// the page-table indirection is off the critical path), computes QK^T and P.V
// with mma.sync m16n8k16 (bf16 x bf16, fp32 accumulate; e4m3 -> bf16
// dequantisation is exact) with fp32 online-softmax statistics, and writes a
// partial (m, l, O) state with an L2 evict_last hint. A second kernel merges
// the partial states of a request in fp32 and rounds once to bf16. The merge
// is a plain stream-ordered launch: the attention grid has fully retired before
// any merge CTA starts.
//
// Numerics match FlashInfer's FA2 custom-mask path: bf16 Q x exact K, fp32
// logits / max / sum / rescale, P rounded once to bf16 before the P.V MMA, fp32
// O accumulator, single final bf16 rounding (see tests/attention/
// test_eagle_verify_fp8kv_sm90.py and the precision notes in the Python
// dispatcher, flashinfer/prefill.py).
//
// Host contract the device code relies on (asserted by the Python dispatcher
// at plan time, never read back from the device):
//  * qo_indptr == [0, 4, 8, ...]: the kernel derives the query row base of
//    request r as 4*r (the FA2 scheduler has already validated that every
//    request has exactly uniform_q_len == 4 query rows; the Python guard
//    additionally checks the host copy of qo_indptr arithmetically);
//  * the packed custom mask is followed by at least 8 readable bytes inside
//    packed_custom_mask: a lane fetches its 64-bit mask window as two aligned
//    32-bit words, so the last tile of the last request reads up to 7 bytes
//    past the end of the packed bits (those bits are masked off, never used);
//  * the workspace holds all batch_size x num_splits partial states and is
//    never assumed zeroed: inactive splits publish l = 0 only and the merge
//    drops their (uninitialised) O slot through a select.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

#ifndef KV_TILE
#define KV_TILE 64
#endif
#ifndef STAGES
#define STAGES 2
#endif
#ifndef CTAPSM
#define CTAPSM 3
#endif
#define NWARP 4
#define PSTRIDE 64
#define NNT (KV_TILE / NWARP / 8)
#define CPW (16 / NWARP)
#define NCP (CPW / 2)
#define NTHR (NWARP * 32)
#define HD 256
#define NROW 16
#define MNEG (-1e30f)

// ---------------- primitives ----------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}
// exact e4m3 -> bf16 (e4m3 values are exactly representable in bf16)
__device__ __forceinline__ uint32_t cvt_fp8x2_bf16x2(uint32_t u) {
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(u & 0xffffu), __NV_E4M3);
  __half2 h = *reinterpret_cast<__half2*>(&hr);
  float2 f = __half22float2(h);
  __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<uint32_t*>(&b);
}

#define LDM_X4(r0, r1, r2, r3, addr)                                           \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)                        \
               : "r"(addr))
#define LDM_X4T(r0, r1, r2, r3, addr)                                                \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)                              \
               : "r"(addr))
#define MMA_BF16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)      \
  asm volatile(                                               \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "  \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ uint64_t pol_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void st_el_f4(float* p, const float4& v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "f"(v.x),
               "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol)
               : "memory");
}
__device__ __forceinline__ float4 ld_el_f4(const float* p, uint64_t pol) {
  float4 v;
  asm volatile("ld.global.nc.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz));
}
// K at [dst], V at the constant immediate offset KV_TILE*HD -> one address register, no extra IADD
#if KV_TILE == 64
#define VOFF_STR "16384"
#elif KV_TILE == 32
#define VOFF_STR "8192"
#elif KV_TILE == 128
#define VOFF_STR "32768"
#endif
__device__ __forceinline__ void cp_async16_kv(uint32_t dst, const void* ks, const void* vs) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(ks));
  asm volatile("cp.async.cg.shared.global [%0+" VOFF_STR "], [%1], 16;" ::"r"(dst), "l"(vs));
}

// ---------------- smem layout ----------------
// byte offset of 16B chunk `c` of token `t` inside a 64x256 fp8 tile (xor swizzle)
__device__ __forceinline__ int kv_off(int t, int c) { return (t << 8) + (((c ^ (t & 7))) << 4); }
// element offset (bf16) of logical-k index l in Q row `row` (256 elems/row, xor swizzle over 8-elem
// chunks)
__device__ __forceinline__ int q_off(int row, int l) {
  return (row << 8) + ((((l >> 3) ^ (row & 7))) << 3) + (l & 7);
}
// element offset (bf16) of token tok in P row `row` (64 elems/row)
__device__ __forceinline__ int p_off(int row, int tok) {
  return (row * PSTRIDE) + ((((tok >> 3) ^ (row & 7))) << 3) + (tok & 7);
}

struct Stage {
  __align__(16) uint8_t K[KV_TILE * HD];
  __align__(16) uint8_t V[KV_TILE * HD];
};
struct SMem {
  Stage st[STAGES];
  __align__(16) __nv_bfloat16 Q[NROW * HD];
  __align__(16) __nv_bfloat16 P[NROW * PSTRIDE];
  float4 red[NROW];
  float4 red2[NROW];
};

// inverse of the logical-k permutation: given dim -> logical index
__device__ __forceinline__ int linv(int dim) {
  int g = dim >> 4, d = dim & 15;
  int r = d & 1, u = (d - r) >> 1;
  return (g << 4) + ((u & 1) << 3) + ((u >> 1) << 1) + r;
}

__global__ void __launch_bounds__(NTHR, CTAPSM)
    eagle_verify_kernel(const __nv_bfloat16* __restrict__ qg, const uint8_t* __restrict__ kcache,
                        const uint8_t* __restrict__ vcache, const int* __restrict__ qo_indptr,
                        const int* __restrict__ kv_indptr, const int* __restrict__ kv_indices,
                        const uint8_t* __restrict__ pmask, const int* __restrict__ mask_indptr,
                        float* __restrict__ po, float* __restrict__ pml, int R, int SPLITS) {
  extern __shared__ uint8_t smem_raw[];
  SMem& sm = *reinterpret_cast<SMem*>(smem_raw);

  const int blk = blockIdx.x;
  const int r = blk / SPLITS;
  const int sp = blk - r * SPLITS;
  const int tid = threadIdx.x;
  const int w = tid >> 5, lane = tid & 31;
  const int p4 = lane & 3, q8 = lane >> 2;  // q8 = row group (0..7), p4 = col group

  // Hoist the four per-request metadata loads together so their latency overlaps the
  // split-boundary setup and the first page-index prefetch.
  const int kvb = kv_indptr[r];
  const int kvend = kv_indptr[r + 1];
  // The deployment contract fixes four draft rows per request and qo_indptr is
  // always [0,4,8,...], so this removes one redundant global metadata load.
  // (The host dispatcher asserts qo_indptr == arange(0, 4*(R+1), 4) at plan time.)
  const int qo_begin = 4 * r;
  const int mask_bit0 = mask_indptr[r] << 3;
  const int kv_len = kvend - kvb;
  const int ntiles_all = (kv_len + KV_TILE - 1) / KV_TILE;
  const int active = min(SPLITS, (ntiles_all + 1) >> 1);
  const long long pbase = (long long)(r * SPLITS + sp);

  // Inactive split: publish l = 0 only. The merge skips the 4 KB po read for any split
  // with l <= 0, so the 16 KB zero-fill (and its read back) is pure waste -- on short-KV
  // rows the partial-state traffic was larger than the KV traffic itself.
  if (sp >= active || ntiles_all == 0) {
    if (tid < NROW) {
      pml[pbase * NROW * 2 + tid * 2] = MNEG;
      pml[pbase * NROW * 2 + tid * 2 + 1] = 0.f;
    }
    return;
  }
  // All products are below 2^31 for the supported pool, avoiding two 64-bit divides.
  const int t0 = (ntiles_all * sp) / active;
  const int t1 = (ntiles_all * (sp + 1)) / active;
  const int j0 = t0 * KV_TILE;
  const int j1 = min(t1 * KV_TILE, kv_len);
  const int ntiles = (j1 - j0 + KV_TILE - 1) / KV_TILE;

  // ---- async stage loader ----
  // Affine loader: with idx = i*NTHR + tid, tok = i*8 + (tid>>4) and ch = tid&15, so
  //   kv_off(tok,ch) = i*2048 + [thread-only term]        -> smem dst = 1 reg + immediate
  //   global src      = (kcache + ch*16) + slot*HD        -> 1 reg + slot stride
  //   slot pointer    = (kv_indices + base + tid>>4) + i*8 -> 1 reg + immediate
  // This removes the per-iteration shift/xor/mad chain from the inner loader.
  const int t16 = tid >> 4, ch16 = tid & 15;
  const uint32_t off0 = (uint32_t)((t16 << 8) + (((ch16 ^ (t16 & 7))) << 4));
  const uint8_t* kb16 = kcache + ch16 * 16;
  const uint8_t* vb16 = vcache + ch16 * 16;

  // Slot-id prefetch.  The gather is a two-level indirection: kv_indices[j] must land in a
  // register before the cp.async that reads k_cache[slot] can even be issued, so with the
  // index load inside the loader the whole stage sits behind one extra global round trip
  // (~600 cy) that the 2-deep cp.async pipeline has no slack to hide -- the loop is exactly
  // bandwidth-paced, one stage of compute per stage of bytes.  Here the ids for the stage
  // issued at iteration t+1 are fetched at iteration t, so only the cp.asyncs themselves are
  // on the critical path and the index latency overlaps a full tile of compute.
#define NLD ((KV_TILE * 16) / NTHR)
  int idxb[NLD];
  bool fullb;
  auto fetch_idx = [&](int tile) {
    int jb = j0 + tile * KV_TILE;
    int nvalid = min(KV_TILE, j1 - jb);
    fullb = (nvalid >= KV_TILE);
    if (tile >= ntiles) {
      fullb = false;
      nvalid = 0;
    }
    const int* kip = kv_indices + (kvb + jb + t16);
#pragma unroll
    for (int i = 0; i < NLD; ++i) {
      int tok = i * 8 + t16;
      idxb[i] = (tok < nvalid) ? __ldg(kip + i * 8) : -1;
    }
  };
  auto issue = [&](int tile, int stage) {
    if (tile >= ntiles) {
      return;
    }
    const uint32_t kd = smem_u32(sm.st[stage].K) + off0;
    if (fullb) {  // full tile: no bounds predicate at all
#pragma unroll
      for (int i = 0; i < NLD; ++i) {
        long long slot = idxb[i];
        cp_async16_kv(kd + (uint32_t)(i * 8 * HD), kb16 + slot * HD, vb16 + slot * HD);
      }
    } else {
#pragma unroll
      for (int i = 0; i < NLD; ++i) {
        int sz = (idxb[i] >= 0) ? 16 : 0;
        long long slot = (idxb[i] >= 0) ? idxb[i] : 0;
        uint32_t d = kd + (uint32_t)(i * 8 * HD);
        cp_async16(d, kb16 + slot * HD, sz);
        cp_async16(d + KV_TILE * HD, vb16 + slot * HD, sz);
      }
    }
  };

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    fetch_idx(i);
    issue(i, i);
    asm volatile("cp.async.commit_group;");
  }
  fetch_idx(STAGES - 1);

  // ---- load Q into smem (logical-k permuted, swizzled) ----
  {
    const __nv_bfloat16* qp = qg + (long long)qo_begin * 4 * HD;
#pragma unroll
    for (int i = 0; i < (NROW * HD) / NTHR; ++i) {
      int idx = i * NTHR + tid;
      int m = idx >> 8, dim = idx & 255;
      sm.Q[q_off(m, linv(dim))] = qp[(long long)m * HD + dim];
    }
  }

  const uint32_t Qbase = smem_u32(sm.Q);
  const uint32_t Pbase = smem_u32(sm.P);

  // Q A-fragments are loop-invariant: hoist all 16 k-blocks into registers once,
  // removing 512 B/token of redundant smem traffic (every warp re-read all of Q per tile).
#if NWARP == 4
#define QREG 1
#else
#define QREG 0
#endif
  __syncthreads();
#if QREG
  uint32_t qf[16][4];
  {
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int kb = 0; kb < 16; ++kb)
      LDM_X4(qf[kb][0], qf[kb][1], qf[kb][2], qf[kb][3],
             Qbase + 2 * q_off(lrow, kb * 16 + 8 * (lane >> 4)));
  }
#endif

  // ---- running state ----
  float acc[CPW][2][4];
#pragma unroll
  for (int a = 0; a < CPW; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  float mrun0 = MNEG, mrun1 = MNEG, lrun0 = 0.f, lrun1 = 0.f;

  const int row0 = q8, row1 = q8 + 8;
  const int dt0 = row0 >> 2, dt1 = row1 >> 2;
  const int mb0 = dt0 * kv_len, mb1 = dt1 * kv_len;

  // Mask tile fetch, hoisted. A lane's 4 mask bits per row sit at fixed offsets
  // {0,1,8,9} from one bit base that advances by exactly KV_TILE bits per tile, so the
  // bit shift is loop-invariant and the word pointer just steps by KV_TILE/8 bytes.
  // Aligned to pmask (mask_indptr[r] is an arbitrary BYTE offset, so mrow may be odd).
  const int ab0 = mask_bit0 + mb0 + j0 + w * (8 * NNT) + 2 * p4;
  const int ab1 = mask_bit0 + mb1 + j0 + w * (8 * NNT) + 2 * p4;
  const uint32_t* mw0 = (const uint32_t*)(pmask + ((ab0 >> 3) & ~3));
  const uint32_t* mw1 = (const uint32_t*)(pmask + ((ab1 >> 3) & ~3));
  const int msh0 = ab0 & 31, msh1 = ab1 & 31;

  const float SCALE = 0.0625f * 1.4426950408889634f;

  for (int tile = 0; tile < ntiles; ++tile) {
    // Barrier budget is 3 per tile, not 4.  Issuing the next stage *after* the top barrier
    // (instead of before the wait) means that barrier alone separates the previous tile's
    // P.V reads from this tile's cp.async writes, so the end-of-loop __syncthreads goes away
    // with no change to the prefetch distance: the stage is still issued exactly one full
    // iteration before it is consumed.
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
    __syncthreads();
    issue(tile + STAGES - 1, (tile + STAGES - 1) % STAGES);
    asm volatile("cp.async.commit_group;");
    fetch_idx(tile + STAGES);

    const int stage = tile % STAGES;
    const uint32_t Kb = smem_u32(sm.st[stage].K);
    const uint32_t Vb = Kb + (KV_TILE * HD);
    const int jb = j0 + tile * KV_TILE;
    const int tokW = w * (8 * NNT);

    // Mask words are fetched here, before the QK^T mmas, not between the row-max barrier and
    // the softmax where they used to sit: they depend only on `tile`, but __syncthreads kept
    // the compiler from hoisting them, so a global round trip was exposed every tile.
    uint32_t rng = 0xFFFFFFFFu;
    if (jb + KV_TILE > j1) {
      int lim = j1 - jb - tokW - 2 * p4;
      rng = (lim >= 32) ? 0xFFFFFFFFu : (lim > 0 ? ((1u << lim) - 1u) : 0u);
    }
    const uint32_t* t0 = mw0 + tile * (KV_TILE / 32);
    const uint32_t* t1 = mw1 + tile * (KV_TILE / 32);
    const uint32_t g0 = (uint32_t)((((uint64_t)__ldg(t0 + 1) << 32) | __ldg(t0)) >> msh0) & rng;
    const uint32_t g1 = (uint32_t)((((uint64_t)__ldg(t1 + 1) << 32) | __ldg(t1)) >> msh1) & rng;

    // ---------- QK^T ----------
    float s[NNT][4];
#pragma unroll
    for (int n = 0; n < NNT; ++n) {
      s[n][0] = 0.f;
      s[n][1] = 0.f;
      s[n][2] = 0.f;
      s[n][3] = 0.f;
    }

#pragma unroll
    for (int g = 0; g < 4; ++g) {
      uint32_t kk[NNT][4];
#pragma unroll
      for (int n = 0; n < NNT; ++n)
        LDM_X4(kk[n][0], kk[n][1], kk[n][2], kk[n][3],
               Kb + kv_off(tokW + 8 * n + (lane & 7), 4 * g + (lane >> 3)));
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#if QREG
        const uint32_t a0 = qf[4 * g + c][0], a1 = qf[4 * g + c][1];
        const uint32_t a2 = qf[4 * g + c][2], a3 = qf[4 * g + c][3];
#else
        uint32_t a0, a1, a2, a3;
        LDM_X4(a0, a1, a2, a3,
               Qbase + 2 * q_off((lane & 7) + ((lane >> 3) & 1) * 8,
                                 (4 * g + c) * 16 + 8 * (lane >> 4)));
#endif
#pragma unroll
        for (int n = 0; n < NNT; ++n) {
          uint32_t b0 = cvt_fp8x2_bf16x2(kk[n][c]);
          uint32_t b1 = cvt_fp8x2_bf16x2(kk[n][c] >> 16);
          MMA_BF16(s[n][0], s[n][1], s[n][2], s[n][3], a0, a1, a2, a3, b0, b1);
        }
      }
    }

    // ---------- scale + mask ----------
#pragma unroll
    for (int n = 0; n < NNT; ++n) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        bool v0 = (g0 >> (8 * n + c)) & 1;
        bool v1 = (g1 >> (8 * n + c)) & 1;
        s[n][c] = v0 ? s[n][c] * SCALE : MNEG;
        s[n][c + 2] = v1 ? s[n][c + 2] * SCALE : MNEG;
      }
    }

    // ---------- row max ----------
    float mx0 = MNEG, mx1 = MNEG;
#pragma unroll
    for (int n = 0; n < NNT; ++n) {
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    if (p4 == 0) {
      ((float*)&sm.red[row0])[w] = mx0;
      ((float*)&sm.red[row1])[w] = mx1;
    }
    __syncthreads();
    float4 ra = sm.red[row0], rb = sm.red[row1];
    float tm0 = fmaxf(fmaxf(fmaxf(mrun0, ra.x), fmaxf(ra.y, ra.z)), ra.w);
    float tm1 = fmaxf(fmaxf(fmaxf(mrun1, rb.x), fmaxf(rb.y, rb.z)), rb.w);

    float corr0 = exp2f(mrun0 - tm0);
    float corr1 = exp2f(mrun1 - tm1);
    mrun0 = tm0;
    mrun1 = tm1;

    // ---------- P = exp2(s - m), store bf16, row sums from the bf16 values ----------
    float sum0 = 0.f, sum1 = 0.f;
    const float z0 = (tm0 > -1e29f) ? tm0 : 0.f;  // no key seen yet -> every e underflows
    const float z1 = (tm1 > -1e29f) ? tm1 : 0.f;
#pragma unroll
    for (int n = 0; n < NNT; ++n) {
      float e0 = exp2f(s[n][0] - z0);
      float e1 = exp2f(s[n][1] - z0);
      float e2 = exp2f(s[n][2] - z1);
      float e3 = exp2f(s[n][3] - z1);
      __nv_bfloat162 pr0 = __floats2bfloat162_rn(e0, e1);
      __nv_bfloat162 pr1 = __floats2bfloat162_rn(e2, e3);
      sum0 += __bfloat162float(pr0.x) + __bfloat162float(pr0.y);
      sum1 += __bfloat162float(pr1.x) + __bfloat162float(pr1.y);
      int tk = tokW + 8 * n + 2 * p4;
      *reinterpret_cast<uint32_t*>(&sm.P[p_off(row0, tk)]) = *reinterpret_cast<uint32_t*>(&pr0);
      *reinterpret_cast<uint32_t*>(&sm.P[p_off(row1, tk)]) = *reinterpret_cast<uint32_t*>(&pr1);
    }
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    // Each warp keeps its own fp32 row-sum subtotal over its own 16 KV columns, rescaled by
    // the same global corr; the 4 subtotals are folded once after the loop.  The barrier here
    // is still needed (it publishes P), but the smem round trip for l is gone.
    lrun0 = lrun0 * corr0 + sum0;
    lrun1 = lrun1 * corr1 + sum1;
    __syncthreads();

    // rescale O (skipped whenever no lane in the warp saw its running max move)
    if (__any_sync(0xffffffff, (corr0 != 1.f) | (corr1 != 1.f))) {
#pragma unroll
      for (int a = 0; a < CPW; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          acc[a][b][0] *= corr0;
          acc[a][b][1] *= corr0;
          acc[a][b][2] *= corr1;
          acc[a][b][3] *= corr1;
        }
    }

    // ---------- P * V ----------
#pragma unroll
    for (int ks = 0; ks < KV_TILE / 16; ++ks) {
      int prow = (lane & 7) + ((lane >> 3) & 1) * 8;
      int ptok = ks * 16 + 8 * (lane >> 4);
      uint32_t a0, a1, a2, a3;
      LDM_X4(a0, a1, a2, a3, Pbase + 2 * p_off(prow, ptok));
#pragma unroll
      for (int cp = 0; cp < NCP; ++cp) {
        int vtok = ks * 16 + 8 * ((lane >> 3) & 1) + (lane & 7);
        int vch = CPW * w + 2 * cp + (lane >> 4);
        uint32_t v0, v1, v2, v3;
        LDM_X4T(v0, v1, v2, v3, Vb + kv_off(vtok, vch));
        uint32_t vr[4] = {v0, v1, v2, v3};
#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
          uint32_t lo = vr[2 * ch], hi = vr[2 * ch + 1];
          uint32_t be0 = cvt_fp8x2_bf16x2(prmt_b32(lo, 0, 0x0020));
          uint32_t be1 = cvt_fp8x2_bf16x2(prmt_b32(hi, 0, 0x0020));
          uint32_t bo0 = cvt_fp8x2_bf16x2(prmt_b32(lo, 0, 0x0031));
          uint32_t bo1 = cvt_fp8x2_bf16x2(prmt_b32(hi, 0, 0x0031));
          int cc = 2 * cp + ch;
          MMA_BF16(acc[cc][0][0], acc[cc][0][1], acc[cc][0][2], acc[cc][0][3], a0, a1, a2, a3, be0,
                   be1);
          MMA_BF16(acc[cc][1][0], acc[cc][1][1], acc[cc][1][2], acc[cc][1][3], a0, a1, a2, a3, bo0,
                   bo1);
        }
      }
    }
  }

  // ---------- fold the 4 per-warp row-sum subtotals ----------
  if (p4 == 0) {
    ((float*)&sm.red2[row0])[w] = lrun0;
    ((float*)&sm.red2[row1])[w] = lrun1;
  }
  __syncthreads();

  // ---------- write partials ----------
  // The KV stream (90-347 MB of use-once bytes) sweeps the 60 MB L2 clean several times per
  // launch, so the 6.5 MB fp32 partial-state block is cold in DRAM by the time the merge
  // reads it.  Tagging only these stores (and the merge's matching loads) evict-last keeps
  // the block resident without reserving a carve-out or touching the stream's attributes.
  const uint64_t opol = pol_evict_last();
  float* op = po + pbase * (NROW * HD);
#pragma unroll
  for (int cc = 0; cc < CPW; ++cc) {
    int dbase = 16 * (CPW * w + cc) + 4 * p4;
    float4 r0 = make_float4(acc[cc][0][0], acc[cc][1][0], acc[cc][0][1], acc[cc][1][1]);
    float4 r1 = make_float4(acc[cc][0][2], acc[cc][1][2], acc[cc][0][3], acc[cc][1][3]);
    st_el_f4(op + row0 * HD + dbase, r0, opol);
    st_el_f4(op + row1 * HD + dbase, r1, opol);
  }
  if (w == 0 && p4 == 0) {
    float4 la = sm.red2[row0], lb = sm.red2[row1];
    pml[pbase * NROW * 2 + row0 * 2] = mrun0;
    pml[pbase * NROW * 2 + row0 * 2 + 1] = (la.x + la.y) + (la.z + la.w);
    pml[pbase * NROW * 2 + row1 * 2] = mrun1;
    pml[pbase * NROW * 2 + row1 * 2 + 1] = (lb.x + lb.y) + (lb.z + lb.w);
  }
}

// ---------------- merge ----------------
// The merge is pure latency: the online-softmax chain is serial in `s`, and the old shape
// (4 B per thread, 128 blocks) left <1 MB of loads in flight -> 850 GB/s.  Here each thread
// owns a float4 of O and issues UF independent (predicated) splits per iteration, so the
// block has 512 x 4 x 16 B outstanding.  512 threads = 32 split-subgroups x 16 float4 lanes
// (= 64 dims); grid = R x 16 rows x 4 dim-quarters.  All state fp32; one bf16 round at the end.
#define MSG 32
#define MLANE 16
__device__ __forceinline__ void mcomb(float& m, float& l, float4& a, float mm, float ll,
                                      const float4& vv) {
  // One exponential per combine instead of two: of the two online-softmax rescales exactly
  // one is 2^0, so only the smaller-max side needs exp2f.  The surviving argument is
  // -|m - mm|, which is also the numerically safe one (never overflows).
  float d = m - mm;
  float e = exp2f(-fabsf(d));
  float mn = fmaxf(m, mm);
  float c1 = (d >= 0.f) ? 1.f : e;
  float c2 = (ll > 0.f) ? ((d >= 0.f) ? e : 1.f) : 0.f;
  a.x = fmaf(c2, vv.x, a.x * c1);
  a.y = fmaf(c2, vv.y, a.y * c1);
  a.z = fmaf(c2, vv.z, a.z * c1);
  a.w = fmaf(c2, vv.w, a.w * c1);
  l = fmaf(ll, c2, l * c1);
  m = mn;
}
template <int UF>
__global__ void __launch_bounds__(MSG* MLANE)
    merge_kernel(const float* __restrict__ po, const float* __restrict__ pml,
                 __nv_bfloat16* __restrict__ o, int R, int SPLITS) {
  __shared__ float4 sa[MSG][MLANE];
  __shared__ float smx[MSG], slx[MSG];
  __shared__ float4 sb[4][MLANE];
  __shared__ float sbm[4], sbl[4];

  const int tx = threadIdx.x;
  const int dl = tx & (MLANE - 1), sg = tx >> 4;
  const unsigned kSubgroupMask = 0x0000ffffu << (16 * (sg & 1));  // the lanes of this subgroup
  int b = blockIdx.x;
  const int dq = b & 3;
  b >>= 2;
  const int row = b & 15, r = b >> 4;
  const int d = dq * (MLANE * 4) + dl * 4;
  const long long base = (long long)r * SPLITS;
  const float* pmr = pml + row * 2;
  const float4* por = (const float4*)(po + row * HD + d);

  const uint64_t opol = pol_evict_last();
  float m = MNEG, l = 0.f;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int s0 = sg; s0 < SPLITS; s0 += UF * MSG) {
    float mm[UF], ll[UF];
    float4 vv[UF];
#pragma unroll
    for (int u = 0; u < UF; ++u) {
      int s = s0 + u * MSG;
      bool ok = (s < SPLITS);
      long long ix = base + (ok ? s : 0);
      // Issued unconditionally: predicating this 16 B fetch on the loaded l put a second
      // global round trip in front of every one of them.  Splits with l == 0 never wrote po,
      // so the value is dropped by the select instead of never being fetched.
      float4 t = ld_el_f4((const float*)(por + ix * (NROW * HD / 4)), opol);
      // (m,l) does not depend on the dim lane, so the old code had all 16 lanes of a
      // subgroup issue the same two 4 B loads: 2 x 16-way redundant memory instructions per
      // split against one useful 16 B float4.  One lane fetches the pair as a single 8 B
      // load and broadcasts it; the LSU issue count per split drops from 3 to 2.
      float2 ml = make_float2(MNEG, 0.f);
      if (dl == 0 && ok) ml = *reinterpret_cast<const float2*>(pmr + ix * (NROW * 2));
      // Broadcast within THIS 16-lane split-subgroup only. The two subgroups that share a
      // warp (sg = 2w and 2w+1) run this loop a different number of times whenever
      // SPLITS - sg crosses a multiple of UF*MSG between them (SPLITS mod 64 odd and < 32,
      // e.g. 7 splits at 64 requests, 3 splits at 132-198 requests): a full-warp
      // __shfl_sync here then waits for lanes that already left the loop and sit at the
      // __syncthreads below -- a deadlock. A 16-lane mask makes each half-warp
      // self-contained, which is also what width = MLANE already assumed.
      mm[u] = __shfl_sync(kSubgroupMask, ml.x, 0, MLANE);
      ll[u] = __shfl_sync(kSubgroupMask, ml.y, 0, MLANE);
      vv[u] = (ll[u] > 0.f) ? t : z4;
    }
#pragma unroll
    for (int u = 0; u < UF; ++u) mcomb(m, l, a, mm[u], ll[u], vv[u]);
  }

  // 32 -> 4 -> 1 across split-subgroups (the combine is associative and commutative)
  sa[sg][dl] = a;
  if (dl == 0) {
    smx[sg] = m;
    slx[sg] = l;
  }
  __syncthreads();
  if (sg < 4) {
    float m2 = MNEG, l2 = 0.f;
    float4 a2 = z4;
#pragma unroll
    for (int g = 0; g < MSG / 4; ++g) {
      int gi = sg * (MSG / 4) + g;
      mcomb(m2, l2, a2, smx[gi], slx[gi], sa[gi][dl]);
    }
    sb[sg][dl] = a2;
    if (dl == 0) {
      sbm[sg] = m2;
      sbl[sg] = l2;
    }
  }
  __syncthreads();
  if (sg == 0) {
    float mo = MNEG, lo = 0.f;
    float4 ao = z4;
#pragma unroll
    for (int g = 0; g < 4; ++g) mcomb(mo, lo, ao, sbm[g], sbl[g], sb[g][dl]);
    float inv = (lo > 0.f) ? (1.f / lo) : 0.f;
    __nv_bfloat16* op = o + ((long long)r * NROW + row) * HD + d;
    __nv_bfloat162 o0 = __floats2bfloat162_rn(ao.x * inv, ao.y * inv);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(ao.z * inv, ao.w * inv);
    *reinterpret_cast<uint32_t*>(op) = *reinterpret_cast<uint32_t*>(&o0);
    *reinterpret_cast<uint32_t*>(op + 2) = *reinterpret_cast<uint32_t*>(&o1);
  }
}

// ---------------- launcher ----------------
// A plain stream-ordered launch of the merge: the attention grid has fully
// retired before any merge CTA starts. Programmatic dependent launch is
// deliberately not used: a merge grid made resident while the attention grid
// still has unscheduled CTAs (batch x splits above 3 x SMs) competes with it
// for SM slots and deadlocks CUDA-graph capture.
template <int UF>
cudaError_t launch_merge(int grid, int thr, cudaStream_t stream, const float* po, const float* pml,
                         __nv_bfloat16* o, int R, int SPLITS) {
  merge_kernel<UF><<<grid, thr, 0, stream>>>(po, pml, o, R, SPLITS);
  return cudaGetLastError();
}

constexpr int64_t kNumRows = NROW;
constexpr int64_t kHeadDim = HD;
constexpr int64_t kNumQoHeads = 4;
constexpr int64_t kQoLen = 4;
// Bytes the kernel may read past the end of the packed custom mask (two aligned
// 32-bit mask words per lane per tile); the caller guarantees this slack.
constexpr int64_t kMaskTailSlackBytes = 8;

int64_t workspace_floats(int64_t batch_size, int64_t num_splits) {
  return batch_size * num_splits * kNumRows * (kHeadDim + 2);
}

}  // namespace

// Number of fp32 words the caller must provide in `workspace` for a launch
// over `batch_size` requests with `num_splits` KV splits per request.
int64_t eagle_verify_fp8kv_sm90_workspace_floats(int64_t batch_size, int64_t num_splits) {
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(num_splits, 0);
  return workspace_floats(batch_size, num_splits);
}

// One-time per-device opt-in to the kernel's dynamic shared memory. Must be
// called outside CUDA-graph capture, before the first `run` on that device.
void eagle_verify_fp8kv_sm90_init(int64_t device_id) {
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  cudaError_t status =
      cudaFuncSetAttribute(eagle_verify_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(sizeof(SMem)));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "eagle_verify_fp8kv_sm90: cudaFuncSetAttribute failed: " << cudaGetErrorString(status);
}

// q [batch_size*4, 4, 256] bf16 contiguous; k_cache / v_cache 4-D NHD paged
// caches [num_slots, 1, 1, 256] e4m3 (256 contiguous bytes per slot);
// qo_indptr / kv_indptr / mask_indptr [>= batch_size+1] int32 (mask_indptr in
// BYTES; qo_indptr must equal [0, 4, 8, ...], asserted by the caller on its host
// copy); kv_indices int32; packed_custom_mask uint8 (little-endian bits, one
// byte-aligned segment per request, followed by >= 8 readable bytes); workspace
// fp32 (>= workspace_floats, all slots kept, never assumed zeroed); o
// [batch_size*4, 4, 256] bf16 contiguous.
// Launches on the caller's current stream; no host synchronisation, no
// allocation (CUDA-graph capturable: two stream-ordered kernel nodes).
void eagle_verify_fp8kv_sm90_run(TensorView q, TensorView k_cache, TensorView v_cache,
                                 TensorView qo_indptr, TensorView kv_indptr, TensorView kv_indices,
                                 TensorView packed_custom_mask, TensorView mask_indptr,
                                 TensorView workspace, TensorView o, int64_t batch_size,
                                 int64_t num_splits) {
  TVM_FFI_ICHECK_GT(batch_size, 0);
  TVM_FFI_ICHECK_GT(num_splits, 0);
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(o, dl_bfloat16);
  CHECK_DIM(3, q);
  CHECK_DIM(3, o);
  TVM_FFI_ICHECK_EQ(q.size(0), batch_size * kQoLen) << "q must hold 4 query tokens per request";
  TVM_FFI_ICHECK_EQ(q.size(1), kNumQoHeads) << "q must have 4 query heads";
  TVM_FFI_ICHECK_EQ(q.size(2), kHeadDim) << "q head_dim must be 256";
  TVM_FFI_ICHECK_EQ(o.size(0), q.size(0));
  TVM_FFI_ICHECK_EQ(o.size(1), q.size(1));
  TVM_FFI_ICHECK_EQ(o.size(2), q.size(2));

  CHECK_CUDA(k_cache);
  CHECK_CUDA(v_cache);
  CHECK_INPUT_TYPE(k_cache, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(v_cache, dl_float8_e4m3fn);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(4, v_cache);
  for (const TensorView* cache : {&k_cache, &v_cache}) {
    TVM_FFI_ICHECK_EQ(cache->size(1), 1) << "page size must be 1";
    TVM_FFI_ICHECK_EQ(cache->size(2), 1) << "exactly one KV head";
    TVM_FFI_ICHECK_EQ(cache->size(3), kHeadDim) << "KV head_dim must be 256";
    TVM_FFI_ICHECK_EQ(cache->stride(3), 1) << "KV head_dim must be contiguous";
    TVM_FFI_ICHECK_EQ(cache->stride(0), kHeadDim) << "KV slots must be 256 contiguous bytes";
  }
  CHECK_DEVICE(q, k_cache);
  CHECK_DEVICE(q, v_cache);
  CHECK_DEVICE(q, o);

  CHECK_INPUT_AND_TYPE(qo_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(kv_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(kv_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(mask_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(packed_custom_mask, dl_uint8);
  CHECK_INPUT_AND_TYPE(workspace, dl_float32);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, kv_indptr);
  CHECK_DIM(1, kv_indices);
  CHECK_DIM(1, mask_indptr);
  CHECK_DIM(1, packed_custom_mask);
  TVM_FFI_ICHECK_GE(qo_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_GE(kv_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_GE(mask_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_GE(packed_custom_mask.numel(), kMaskTailSlackBytes)
      << "packed_custom_mask must carry >= 8 bytes of tail slack";
  TVM_FFI_ICHECK_GE(workspace.numel(), workspace_floats(batch_size, num_splits))
      << "workspace too small for batch_size x num_splits partial states";
  CHECK_DEVICE(q, qo_indptr);
  CHECK_DEVICE(q, kv_indptr);
  CHECK_DEVICE(q, kv_indices);
  CHECK_DEVICE(q, mask_indptr);
  CHECK_DEVICE(q, packed_custom_mask);
  CHECK_DEVICE(q, workspace);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  float* po = static_cast<float*>(workspace.data_ptr());
  float* pml = po + batch_size * num_splits * kNumRows * kHeadDim;
  const int R = static_cast<int>(batch_size);
  const int SPLITS = static_cast<int>(num_splits);

  eagle_verify_kernel<<<R * SPLITS, NTHR, sizeof(SMem), stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const uint8_t*>(k_cache.data_ptr()),
      static_cast<const uint8_t*>(v_cache.data_ptr()),
      static_cast<const int*>(qo_indptr.data_ptr()), static_cast<const int*>(kv_indptr.data_ptr()),
      static_cast<const int*>(kv_indices.data_ptr()),
      static_cast<const uint8_t*>(packed_custom_mask.data_ptr()),
      static_cast<const int*>(mask_indptr.data_ptr()), po, pml, R, SPLITS);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "eagle_verify_fp8kv_sm90 attention launch failed: " << cudaGetErrorString(status);

  // One UF that covers every split in a single pass: a second pass costs a full dependent
  // DRAM round trip for 2-3 useful loads, and a too-large UF only burns registers.
  const int need = (SPLITS + MSG - 1) / MSG;
  const int grid = R * NROW * 4, thr = MSG * MLANE;
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(o.data_ptr());
  if (need <= 2)
    status = launch_merge<2>(grid, thr, stream, po, pml, out, R, SPLITS);
  else if (need <= 4)
    status = launch_merge<4>(grid, thr, stream, po, pml, out, R, SPLITS);
  else if (need <= 8)
    status = launch_merge<8>(grid, thr, stream, po, pml, out, R, SPLITS);
  else
    status = launch_merge<16>(grid, thr, stream, po, pml, out, R, SPLITS);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "eagle_verify_fp8kv_sm90 merge launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, eagle_verify_fp8kv_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, eagle_verify_fp8kv_sm90_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_floats, eagle_verify_fp8kv_sm90_workspace_floats);
