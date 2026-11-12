// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// QSA stage-1 indexer for SM107 (Rubin).
//
//  Row t (request r, pos = t - cu[r]) holds, ascending, the token ids of the
//  selected compressed blocks (x4 expansion) then the causal tail, -1 padding
//  and the valid count in the last column.
//
//  vis = (pos+1)/4 and the selection keeps min(vis,512) blocks, so whenever
//  vis <= 512 every visible block is kept: the ascending expansion is
//  0..4*vis-1 and the tail completes it to 0..pos.  Such a row is exactly the
//  identity sequence 0..pos and cannot depend on the fp8 scores.  Scoring +
//  exact top-512 therefore only runs for rows with pos+1 >= 2052.
#include <cuda_runtime.h>
#include <cstdint>
#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

#define CR 4
#define BLOCK_TOPK 512
#define PWIDTH 2052
#define CPAGE 64
#define IDX_D 128
#define SLOW_CNT 2052            // pos+1 >= 2052  <=>  vis > 512

// warp-cooperative 32-ary search: largest r in [0,B) with cu[r] <= t
__device__ __forceinline__ int find_req32(const int* __restrict__ cu, int B, int t, int lane) {
  int lo = 0, len = B;
  while (len > 1) {
    const int span = (len + 31) >> 5;
    const int idx = lo + lane * span;
    const bool inr = idx < lo + len;
    const int v = cu[inr ? idx : lo];
    const unsigned m = __ballot_sync(0xffffffffu, inr && (v <= t));
    const int sel = 31 - __clz(m);
    const int nlo = lo + sel * span;
    len = min(span, lo + len - nlo);
    lo = nlo;
  }
  return lo;
}

// same search, but also hands back cu[r] (the last level already loaded it)
__device__ __forceinline__ int find_req32_base(const int* __restrict__ cu, int B, int t,
                                               int lane, int& base) {
  int lo = 0, len = B, bv = cu[0];
  while (len > 1) {
    const int span = (len + 31) >> 5;
    const int idx = lo + lane * span;
    const bool inr = idx < lo + len;
    const int v = cu[inr ? idx : lo];
    const unsigned m = __ballot_sync(0xffffffffu, inr && (v <= t));
    const int sel = 31 - __clz(m);
    bv = __shfl_sync(0xffffffffu, v, sel);
    const int nlo = lo + sel * span;
    len = min(span, lo + len - nlo);
    lo = nlo;
  }
  base = bv;
  return lo;
}

__device__ __forceinline__ void st_cs(int4* p, const int4& v) {
  asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};\n" ::"l"(p),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// -------------------------------------------------------- fast (identity) rows
// Rows are PWIDTH*4 = 8208 B, i.e. 16B- but not 128B-aligned, so a per-row warp
// store straddles sectors.  Instead every CTA owns a flat, 128B-aligned range of
// int4s of the whole output and derives (row, col) from the flat index.
#define FK_THR 256
#define RW 513                      // int4 per row

template <int PT, bool SINGLE_REQ>
__global__ __launch_bounds__(FK_THR) void qsa_fast(
    const int* __restrict__ cu, int B, int T, int ntot, int4* __restrict__ out4) {
  const int tid = threadIdx.x;
  const int g0 = blockIdx.x * (FK_THR * PT);
  if (g0 >= ntot) return;

  const int row0 = g0 / RW;
  __shared__ int s_cnt[8];
  int cbase = row0 + 1;            // cnt == row + cbase - row0 when uniform
  bool uni = true;
  if (!SINGLE_REQ) {
    const int warp = tid >> 5, lane = tid & 31;   // one warp per covered row
    const int rr = row0 + warp;
    int bse = 0;
    find_req32_base(cu, B, rr, lane, bse);
    if (lane == 0) s_cnt[warp] = rr - bse + 1;
    __syncthreads();
    cbase = s_cnt[0];
    uni = (s_cnt[7] == cbase + 7);                // cnt-w is monotone: ends equal => all equal
  }

  int row = row0;
  int col4 = g0 + tid - row0 * RW;
  if (col4 >= RW) { row += col4 / RW; col4 -= (col4 / RW) * RW; }
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int g = g0 + tid + k * FK_THR;
    if (PT > 1 && g >= ntot) break;
    const int cnt = SINGLE_REQ ? (row + 1) : (uni ? (row - row0 + cbase) : s_cnt[row - row0]);
    if (cnt < SLOW_CNT) {                     // else handled by qsa_slow
      const int c = col4 * 4;
      int4 v;
      v.x = (c     < cnt) ? c     : -1;
      v.y = (c + 1 < cnt) ? c + 1 : -1;
      v.z = (c + 2 < cnt) ? c + 2 : -1;
      v.w = (col4 == RW - 1) ? cnt : ((c + 3 < cnt) ? c + 3 : -1);
      st_cs(out4 + g, v);
    }
    col4 += FK_THR;
    if (col4 >= RW) { col4 -= RW; ++row; }
  }
}

// Request-local prefix writer for long, low-B inputs.  Only the first 2051
// rows of each request are score-independent identity rows, so avoid probing
// the much larger slow-row suffix with idle flat CTAs.
template <int PT>
__global__ __launch_bounds__(FK_THR) void qsa_fast_prefix(
    const int* __restrict__ cu, int B, int4* __restrict__ out4) {
  const int req = static_cast<int>(blockIdx.y);
  if (req >= B) return;
  const int begin = cu[req];
  const int rows = min(cu[req + 1] - begin, SLOW_CNT - 1);
  const int items = rows * RW;
  const int g0 = static_cast<int>(blockIdx.x) * FK_THR * PT;
  if (g0 >= items) return;

  int row = g0 / RW;
  int col4 = g0 + static_cast<int>(threadIdx.x) - row * RW;
  if (col4 >= RW) { col4 -= RW; ++row; }
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int g = g0 + static_cast<int>(threadIdx.x) + k * FK_THR;
    if (g >= items) break;
    const int count = row + 1;
    const int c = col4 << 2;
    int4 v;
    v.x = (c < count) ? c : -1;
    v.y = (c + 1 < count) ? c + 1 : -1;
    v.z = (c + 2 < count) ? c + 2 : -1;
    v.w = (col4 == RW - 1) ? count : ((c + 3 < count) ? c + 3 : -1);
    st_cs(out4 + static_cast<size_t>(begin) * RW + g, v);
    col4 += FK_THR;
    if (col4 >= RW) { col4 -= RW; ++row; }
  }
}

// ------------------------------------------------------------------ fp8 MMA
#define MMA(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)                               \
  asm volatile(                                                                   \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                      \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                   \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                    \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

#define SROW 36            // smem words per 128B operand row (144B, conflict-free)
#define NTILE 128          // ck blocks per pipeline stage
#define GMAX 16            // tokens owned by one CTA
#define NTHR 512
#define NWARP (NTHR / 32)
#define HBK 288          // 256 radix buckets + 32 per-lane dump slots

// --------------------------------------------------------------- slow rows
template <int NSTAGE>
__global__ __launch_bounds__(NTHR, 1) void qsa_slow(
    const uint8_t* __restrict__ qi, const uint8_t* __restrict__ ckc,
    const int* __restrict__ cu, int B, int T, const int* __restrict__ cpb,
    int scap, int* __restrict__ out) {
  extern __shared__ uint32_t smem[];
  uint32_t* q_s  = smem;                                   // 4*GMAX rows
  int*      hist = (int*)(q_s + 4 * GMAX * SROW);          // NWARP * HBK
  int*      meta = hist + NWARP * HBK;                     // 2*GMAX
  float*    sc   = (float*)(meta + 2 * GMAX);              // scap floats
  uint32_t* ck_s = (uint32_t*)(sc + scap);                 // NSTAGE * NTILE rows

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int t0 = (gridDim.x - 1 - blockIdx.x) * GMAX;
  if (t0 >= T) return;

  {   // one warp per token: 16 independent searches issue in parallel
    const int j = warp;
    const int t = t0 + j;
    int r = -1, pos = -1;
    if (j < GMAX && t < T) {
      if (B == 1) { r = 0; pos = t; }
      else { r = find_req32(cu, B, t, lane); pos = t - cu[r]; }
    }
    if (j < GMAX && lane == 0) { meta[j] = r; meta[GMAX + j] = pos; }
  }
  __syncthreads();

  uint32_t todo = 0;
#pragma unroll
  for (int j = 0; j < GMAX; ++j)
    if (meta[GMAX + j] + 1 >= SLOW_CNT) todo |= (1u << j);
  if (todo == 0) return;

  {   // q -> smem, rows laid out as head*GMAX + token (row == MMA M index)
    const int row = tid >> 3, jj = tid & 7;     // 64 rows x 8 chunks
    const int tok = row & (GMAX - 1), h = row >> 4;
    uint4 val = make_uint4(0, 0, 0, 0);
    if (t0 + tok < T)
      val = *reinterpret_cast<const uint4*>(qi + ((size_t)(t0 + tok) * 4 + h) * IDX_D + jj * 16);
    *reinterpret_cast<uint4*>(q_s + row * SROW + jj * 4) = val;
  }
  __syncthreads();

  // A-fragments for the 4 index heads stay resident in registers
  uint32_t af[4][8];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const uint32_t* p0 = q_s + (h * GMAX + (lane >> 2)) * SROW + (lane & 3);
    const uint32_t* p1 = p0 + 8 * SROW;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      af[h][2 * kk]     = p0[8 * kk];            // a0 : row,   k=32kk
      af[h][2 * kk + 1] = p1[8 * kk];            // a1 : row+8, k=32kk
    }
  }
  uint32_t af2[4][8];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    const uint32_t* p0 = q_s + (h * GMAX + (lane >> 2)) * SROW + (lane & 3) + 4;
    const uint32_t* p1 = p0 + 8 * SROW;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      af2[h][2 * kk]     = p0[8 * kk];           // a2 : row,   k=32kk+16
      af2[h][2 * kk + 1] = p1[8 * kk];           // a3 : row+8, k=32kk+16
    }
  }

  const int stbytes = NTILE * SROW * 4;
  const uint32_t ck_base_addr = (uint32_t)__cvta_generic_to_shared(ck_s);

  while (todo) {
    const int j0 = __ffs(todo) - 1;
    const int r0 = meta[j0];
    uint32_t sub = 0;
#pragma unroll
    for (int j = 0; j < GMAX; ++j)
      if (((todo >> j) & 1) && meta[j] == r0) sub |= (1u << j);
    const int jhi = 31 - __clz(sub);
    const int V = (meta[GMAX + jhi] + 1) >> 2;          // max vis in this subset
    const size_t ckbase = (size_t)cpb[r0] * CPAGE;
    const int Vs = (V + 3) & ~3;                        // float4-aligned score stride
    const int gb = min(GMAX, max(1, scap / max(Vs, 1)));
    if ((size_t)gb * (size_t)Vs > (size_t)scap) break;  // V > scap: unsupported extent
    const int ntiles = (V + NTILE - 1) / NTILE;

    for (int bst = 0; bst < GMAX; bst += gb) {
      const uint32_t bmask = ((1u << gb) - 1u) << bst;
      if ((sub & bmask) == 0) continue;

      int issued = 0;
      for (int s = 0; s < NSTAGE - 1; ++s) {
        const int n0 = s * NTILE;
        const uint32_t dst = ck_base_addr + s * stbytes;
#pragma unroll
        for (int it = 0; it < 2; ++it) {
          const int bb = warp * 8 + (lane >> 2);
          const int jj = (lane & 3) + it * 4;
          const int v = n0 + bb;
          if (v < V) cp16(dst + (bb * SROW + jj * 4) * 4, ckc + (ckbase + v) * IDX_D + jj * 16);
        }
        cp_commit();
        ++issued;
      }

      int buf = 0;
      for (int i = 0; i < ntiles; ++i) {
        if (issued < ntiles) {
          const int s = issued;
          const int n0 = s * NTILE;
          const uint32_t dst = ck_base_addr + (s % NSTAGE) * stbytes;
          // Each warp owns and consumes the same eight rows of every stage.
          if (s >= NSTAGE) __syncwarp();
#pragma unroll
          for (int it = 0; it < 2; ++it) {
            const int bb = warp * 8 + (lane >> 2);
            const int jj = (lane & 3) + it * 4;
            const int v = n0 + bb;
            if (v < V) cp16(dst + (bb * SROW + jj * 4) * 4, ckc + (ckbase + v) * IDX_D + jj * 16);
          }
          cp_commit();
          ++issued;
        } else {
          cp_commit();
        }
        cp_wait<NSTAGE - 1>();
        __syncwarp();

        const uint32_t* tile = ck_s + buf * (NTILE * SROW);
        const int n0 = i * NTILE;
        const int colb = warp * 8 + (lane >> 2);
        const uint32_t* bp = tile + colb * SROW + (lane & 3);
        uint32_t bf[4], bf2[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) { bf[kk] = bp[8 * kk]; bf2[kk] = bp[8 * kk + 4]; }

        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)
            MMA(c0, c1, c2, c3, af[h][2 * kk], af[h][2 * kk + 1],
                af2[h][2 * kk], af2[h][2 * kk + 1], bf[kk], bf2[kk]);
          a0 += qsa_relu_exact(c0); a1 += qsa_relu_exact(c1);
          a2 += qsa_relu_exact(c2); a3 += qsa_relu_exact(c3);
        }
        {
          const int tA = lane >> 2, tB = tA + 8;
          const int cA = n0 + warp * 8 + 2 * (lane & 3);
          if (((bmask >> tA) & 1) && cA < V) {
            float* d = sc + (size_t)(tA - bst) * Vs;
            d[cA] = a0;
            if (cA + 1 < V) d[cA + 1] = a1;
          }
          if (((bmask >> tB) & 1) && cA < V) {
            float* d = sc + (size_t)(tB - bst) * Vs;
            d[cA] = a2;
            if (cA + 1 < V) d[cA + 1] = a3;
          }
        }
        buf = (buf + 1 == NSTAGE) ? 0 : buf + 1;
      }
      cp_wait<0>();
      if (tid < gb) {
        float* d = sc + (size_t)tid * Vs + V;
#pragma unroll
        for (int p = 0; p < 3; ++p)
          if (p < Vs - V) d[p] = 0.f;
      }
      __syncthreads();

      // -------- exact top-512 per token: 32-bit radix select over smem scores
      for (int jj = 0; jj < gb; jj += NWARP) {
        const int j = bst + jj + warp;
        if ((jj + warp) < gb && ((sub >> j) & 1)) {
          const int t = t0 + j;
          const int pos = meta[GMAX + j];
          const int vis = (pos + 1) >> 2;
          const float* ks = sc + (size_t)(j - bst) * Vs;
          int* h = hist + warp * HBK;

          uint32_t prefix = 0u, pmask = 0u;
          int k = BLOCK_TOPK;
          bool exact = false;
          for (int shift = 24; shift >= 0; shift -= 8) {
            // Prevent faster lanes from clearing the next radix histogram
            // while the owner lane is still reading the current one.
            __syncwarp();
            for (int i = lane; i < HBK; i += 32) h[i] = 0;
            __syncwarp();
            // branchless: non-matching keys land in a private dump slot
            for (int v = lane; v < vis; v += 32) {
              const uint32_t key = qsa_score_key_exact(ks[v]);
              const int b = ((key & pmask) == prefix) ? (int)((key >> shift) & 255)
                                                      : (256 + lane);
              atomicAdd(&h[b], 1);
            }
            __syncwarp();
            int s = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) s += h[lane * 8 + i];
            int x = s;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
              const int y = __shfl_down_sync(0xffffffffu, x, off);
              if (lane + off < 32) x += y;
            }
            const unsigned m = __ballot_sync(0xffffffffu, x >= k);
            const int L = 31 - __clz(m);
            int d = 0, above = 0, cntd = 0;
            if (lane == L) {
              int acc = x - s;
              for (int i = 7; i >= 0; --i) {
                const int c = h[L * 8 + i];
                if (acc + c >= k) { d = L * 8 + i; above = acc; cntd = c; break; }
                acc += c;
              }
            }
            d = __shfl_sync(0xffffffffu, d, L);
            above = __shfl_sync(0xffffffffu, above, L);
            cntd = __shfl_sync(0xffffffffu, cntd, L);
            prefix |= ((uint32_t)d) << shift;
            pmask |= 0xFFu << shift;
            k -= above;
            if (cntd == k) { exact = true; break; }   // whole bucket is taken
          }
          const uint32_t tau = prefix;
          const uint32_t tmask = pmask;
          const int keq = exact ? (1 << 30) : k;

          int rank = 0, neq = 0;
          int* orow = out + (size_t)t * PWIDTH;
          if (exact) {
            // whole critical bucket is taken: selection is a plain masked compare,
            // so four keys per lane need only one warp scan to rank them
            const uint4* e4 = reinterpret_cast<const uint4*>(ks);
            const int n4 = (vis + 3) >> 2;
            for (int q0 = 0; q0 < n4 && rank < BLOCK_TOPK; q0 += 32) {
              const int q = q0 + lane;
              uint4 kv = make_uint4(0u, 0u, 0u, 0u);
              if (q < n4) kv = e4[q];
              const int v0 = q << 2;
              const bool b0 = (v0     < vis) && ((kv.x & tmask) >= tau);
              const bool b1 = (v0 + 1 < vis) && ((kv.y & tmask) >= tau);
              const bool b2 = (v0 + 2 < vis) && ((kv.z & tmask) >= tau);
              const bool b3 = (v0 + 3 < vis) && ((kv.w & tmask) >= tau);
              const int c = (int)b0 + (int)b1 + (int)b2 + (int)b3;
              int inc = c;
#pragma unroll
              for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, inc, off);
                if (lane >= off) inc += y;
              }
              const int tot = __shfl_sync(0xffffffffu, inc, 31);
              int r = rank + inc - c;
              if (b0) { int4 z; z.x = 4*v0;     z.y = z.x+1; z.z = z.x+2; z.w = z.x+3;
                        *reinterpret_cast<int4*>(orow + 4*r) = z; ++r; }
              if (b1) { int4 z; z.x = 4*(v0+1); z.y = z.x+1; z.z = z.x+2; z.w = z.x+3;
                        *reinterpret_cast<int4*>(orow + 4*r) = z; ++r; }
              if (b2) { int4 z; z.x = 4*(v0+2); z.y = z.x+1; z.z = z.x+2; z.w = z.x+3;
                        *reinterpret_cast<int4*>(orow + 4*r) = z; ++r; }
              if (b3) { int4 z; z.x = 4*(v0+3); z.y = z.x+1; z.z = z.x+2; z.w = z.x+3;
                        *reinterpret_cast<int4*>(orow + 4*r) = z; }
              rank += tot;
            }
          }
          for (int vb0 = 0; !exact && vb0 < vis && rank < BLOCK_TOPK; vb0 += 32) {
            const int v = vb0 + lane;
            const uint32_t key = (v < vis) ? (qsa_score_key_exact(ks[v]) & tmask) : 0u;
            const bool gt = (v < vis) && (key > tau);
            const bool eq = (v < vis) && (key == tau);
            const unsigned be = __ballot_sync(0xffffffffu, eq);
            const int eqb = __popc(be & ((1u << lane) - 1u));
            const bool take = gt || (eq && (neq + eqb < keq));
            const unsigned bt = __ballot_sync(0xffffffffu, take);
            const int myrank = rank + __popc(bt & ((1u << lane) - 1u));
            if (take) {
              int4 q4; q4.x = 4 * v; q4.y = 4 * v + 1; q4.z = 4 * v + 2; q4.w = 4 * v + 3;
              *reinterpret_cast<int4*>(orow + 4 * myrank) = q4;
            }
            rank += __popc(bt);
            neq += __popc(be);
          }
          if (lane == 0) {
            const int rem = (pos + 1) & 3;
            const int ts = vis * CR;
            int4 tl;
            tl.x = (0 < rem) ? ts     : -1;
            tl.y = (1 < rem) ? ts + 1 : -1;
            tl.z = (2 < rem) ? ts + 2 : -1;
            tl.w = CR * BLOCK_TOPK + rem;
            *reinterpret_cast<int4*>(orow + CR * BLOCK_TOPK) = tl;
          }
        }
        __syncthreads();
      }
    }
    todo &= ~sub;
  }
}

// ------------------------------------------------------------------- launcher
void Run(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cu_seqlens,
         ffi::TensorView cpage_base, ffi::TensorView packed) {
  const int T = (int)qi.shape()[0];
  const int B = (int)cpage_base.shape()[0];
  if (T <= 0) return;
  cudaStream_t stream =
      (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id);

  const int* cu = (const int*)cu_seqlens.data_ptr();
  int* out = (int*)packed.data_ptr();

  if (T > 10000 && B <= 4) {
    constexpr int pt = 8;
    constexpr int rows = SLOW_CNT - 1;
    constexpr int grid = (rows * RW + FK_THR * pt - 1) / (FK_THR * pt);
    qsa_fast_prefix<pt><<<dim3(grid, B), FK_THR, 0, stream>>>(
        cu, B, reinterpret_cast<int4*>(out));
  } else {
    const int ntot = T * RW;
    int pt = ntot / (FK_THR * 424);
    pt = (pt >= 8) ? 8 : ((pt >= 4) ? 4 : ((pt >= 2) ? 2 : 1));
    const int grid = (ntot + FK_THR * pt - 1) / (FK_THR * pt);
#define LF(P)                                                                        \
    do { if (B == 1) qsa_fast<P, true><<<grid, FK_THR, 0, stream>>>(cu, B, T, ntot, (int4*)out);  \
         else        qsa_fast<P, false><<<grid, FK_THR, 0, stream>>>(cu, B, T, ntot, (int4*)out); } while (0)
    switch (pt) { case 8: LF(8); break; case 4: LF(4); break; case 2: LF(2); break; default: LF(1); }
#undef LF
  }

  // T is the conservative request-length upper bound even when the valid
  // partition contains empty requests.  T-B+1 is only safe when every
  // request is nonempty.
  const int maxlen = T;
  if (maxlen >= SLOW_CNT) {
    // SM107 has a 227 KiB regular opt-in limit after the 1 KiB reservation.
    // CUDA 13.4's per-launch mode bypasses the legacy 48 KiB function limit,
    // avoiding device queries and mutable function attributes on this path.
    constexpr int scap = 32768;
    constexpr int fixed = (4 * GMAX * SROW + NWARP * HBK + 2 * GMAX) * 4;
    constexpr int stbytes = NTILE * SROW * 4;
    constexpr size_t smem =
        (size_t)fixed + (size_t)scap * 4 + (size_t)3 * stbytes;
    static_assert(smem == 214144 && smem <= 227 * 1024,
                  "SM107 non-portable shared-memory budget exceeded");
    const int grid = (T + GMAX - 1) / GMAX;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeSharedMemoryMode;
    attr.val.sharedMemoryMode = cudaSharedMemoryModeAllowNonPortable;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid);
    config.blockDim = dim3(NTHR);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(
        &config, qsa_slow<3>,
        (const uint8_t*)qi.data_ptr(), (const uint8_t*)ck_cache.data_ptr(), cu, B, T,
        (const int*)cpage_base.data_ptr(), scap, out);
  }
}

#ifndef DEVICE_ONLY
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif




