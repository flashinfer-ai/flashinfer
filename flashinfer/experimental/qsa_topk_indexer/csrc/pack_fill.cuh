// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Trivial-row (identity) output writer for the QSA stage-1 indexer.
//
// pos <= 2050  =>  vis = (pos+1)/4 <= 512  =>  every visible block is kept and
// the packed row is exactly 0..pos, -1 padding, count in the last column.  The
// whole row therefore depends only on pos, so these rows are a pure streaming
// write of T * 8208 B and the only question is how to issue it.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace packfill {

constexpr int QF_RW = 513;    // int4 per packed row (PWIDTH = 2052 int32)
constexpr int QF_NT = 256;    // threads, flat writer
constexpr int QF_RQNT = 128;  // threads, per-request writer

// warp-cooperative 32-ary search: largest r in [0,B) with cu[r] <= t
__device__ __forceinline__ int qf_find(const int* __restrict__ cu, int B, int t, int lane) {
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

__device__ __forceinline__ int4 qf_rowval(int col4, int cnt) {
  const int c = col4 * 4;
  int4 v;
  v.x = (c < cnt) ? c : -1;
  v.y = (c + 1 < cnt) ? c + 1 : -1;
  v.z = (c + 2 < cnt) ? c + 2 : -1;
  v.w = (col4 == QF_RW - 1) ? cnt : ((c + 3 < cnt) ? c + 3 : -1);
  return v;
}

__device__ __forceinline__ void qf_store2(int4* p, const int4& a, const int4& b) {
  asm volatile("st.global.cs.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"l"(p), "r"(a.x),
               "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w)
               : "memory");
}

// Flat writer: every CTA owns a contiguous, 128B-aligned int4 range of the whole
// output, so warps never straddle a sector.  B == 1 skips the cu probe entirely;
// otherwise each warp probes once for its own first row and then walks request
// boundaries forward, which never fires inside a request.
template <int PT, bool B1, bool SKIP>
__global__ __launch_bounds__(QF_NT) void qf_flat(const int* __restrict__ cu, int B, int T, int ntot,
                                                 int4* __restrict__ out4) {
  const int tid = threadIdx.x;
  const int g0 = blockIdx.x * (QF_NT * PT);
  if (g0 >= ntot) return;
  int q = 0, lo = 0, hi = T;
  if (!B1) {
    const int lane = tid & 31;
    int wr = (g0 + (tid & ~31)) / QF_RW;
    if (wr >= T) wr = T - 1;
    q = qf_find(cu, B, wr, lane);
    lo = cu[q];
    hi = cu[q + 1];
  }
  int row = (g0 + tid) / QF_RW;
  int col4 = g0 + tid - row * QF_RW;
  if (!B1)
    while (row >= hi && q + 1 < B) {
      lo = hi;
      hi = cu[++q + 1];
    }
  int cnt = row - lo + 1;
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int g = g0 + tid + k * QF_NT;
    // The final CTA is generally partial even for PT==1.  Always apply the
    // flattened-output bound before the store; output is poison-filled and
    // has exactly T*QF_RW int4 elements.
    if (g >= ntot) break;
    if (!SKIP || cnt < 2052) out4[g] = qf_rowval(col4, cnt);
    col4 += QF_NT;
    if (col4 >= QF_RW) {
      col4 -= QF_RW;
      ++row;
      while (!B1 && row >= hi && q + 1 < B) {
        lo = hi;
        hi = cu[++q + 1];
      }
      cnt = row - lo + 1;
    }
  }
}

// One CTA per request: short requests make the flat writer's cu probe dominate.
template <bool SKIP, int NT>
__global__ __launch_bounds__(NT) void qf_req(const int* __restrict__ cu, int* __restrict__ out) {
  const int r = blockIdx.x;
  const int beg = cu[r], end = cu[r + 1];
  // A v8.b32 store requires 32-byte alignment.  Each row is 8208 bytes, so
  // the paired-row path is aligned exactly when the first token is even.
  if (end == beg + 2 && (beg & 1) == 0) {
    int4* o0 = (int4*)(out + (size_t)beg * (QF_RW * 4));
    int4* o1 = o0 + QF_RW;
    if constexpr (NT == 128) {
      const int tid = threadIdx.x;
      const int4 neg = make_int4(-1, -1, -1, -1);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pair = tid + k * 128;
        int4 a = neg, b = neg;
        if (pair == 0) a.x = 0;
        if (pair == 256) {
          a.w = 1;
          b.x = 0;
          b.y = 1;
        }
        qf_store2(o0 + 2 * pair, a, b);
      }
      if (tid == 0) {
        const int4 meta1 = make_int4(-1, -1, -1, 2);
        qf_store2(o0 + 1024, neg, meta1);
      }
      return;
    }
    for (int v = threadIdx.x; v < QF_RW; v += NT) {
      int4 a = make_int4(-1, -1, -1, -1);
      int4 b = make_int4(-1, -1, -1, -1);
      if (v == 0) {
        a.x = 0;
        b.x = 0;
        b.y = 1;
      }
      if (v == QF_RW - 1) {
        a.w = 1;
        b.w = 2;
      }
      o0[v] = a;
      o1[v] = b;
    }
    return;
  }
  for (int t = beg; t < end; ++t) {
    const int cnt = t - beg + 1;
    if (SKIP && cnt >= 2052) continue;
    int4* o = (int4*)(out + (size_t)t * (QF_RW * 4));
    for (int v = threadIdx.x; v < QF_RW; v += NT) o[v] = qf_rowval(v, cnt);
  }
}

// SKIP == false requires T <= 2051, so every row is trivial.  With SKIP the
// writer leaves pos >= 2051 rows to the scoring kernel.
template <bool SKIP>
inline void LaunchTrivial(const int* cu, int* out, int T, int B, cudaStream_t stream) {
  if (T == 2048 && B == 1024) {
    qf_req<SKIP, 128><<<B, 128, 0, stream>>>(cu, out);
    return;
  }
  if (B > 1 && T <= 4 * B) {
    qf_req<SKIP, QF_RQNT><<<B, QF_RQNT, 0, stream>>>(cu, out);
    return;
  }
  const int ntot = T * QF_RW;
  int pt = ntot / (QF_NT * 424);
#define QF_LF(P)                                                                       \
  do {                                                                                 \
    const int grid = (ntot + QF_NT * (P) - 1) / (QF_NT * (P));                         \
    if (B == 1)                                                                        \
      qf_flat<P, true, SKIP><<<grid, QF_NT, 0, stream>>>(cu, B, T, ntot, (int4*)out);  \
    else                                                                               \
      qf_flat<P, false, SKIP><<<grid, QF_NT, 0, stream>>>(cu, B, T, ntot, (int4*)out); \
  } while (0)
  if (pt >= 8) {
    QF_LF(8);
  } else if (pt >= 4) {
    QF_LF(4);
  } else if (pt >= 2) {
    QF_LF(2);
  } else {
    QF_LF(1);
  }
#undef QF_LF
}

}  // namespace packfill
