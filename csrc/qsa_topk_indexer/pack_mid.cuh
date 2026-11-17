// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// Mid-size regime (2051 < T <= 8208) in a single launch.
//
// A second kernel launch costs ~2.4us here even when all of its CTAs exit
// immediately, so trivial rows and scored rows must share one kernel.  Each
// CTA owns MR consecutive rows: trivial rows are a flat vectorised fill over
// the CTA's contiguous int4 range, and any row with pos >= 2051 is scored and
// selected exactly.  Scores live in that row's own destination words (vis <=
// PWIDTH here), keeping shared memory small so the fill keeps its occupancy.
#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace packmid {

constexpr int MR = 2;     // rows per CTA
constexpr int MT = 192;   // threads per CTA
constexpr int MW4 = 513;  // int4 per packed row
constexpr int MW = 2052;  // int32 per packed row
constexpr int MFLAT = MR * MW4;
constexpr int MPT = (MFLAT + MT - 1) / MT;
constexpr int MTOPK = 512;
constexpr int MBMW = (MW + 31) / 32;  // bitmap words covering any in-regime vis

__device__ __forceinline__ void qf_mid_store_cs(int4* p, const int4& v) {
  asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ float f8(uint32_t byte) {
  __nv_fp8_e4m3 v;
  v.__x = (__nv_fp8_storage_t)byte;
  return (float)v;
}

__global__ __maxnreg__(32) void qf_mid(const uint8_t* __restrict__ qi,
                                       const uint8_t* __restrict__ ckc, const int* __restrict__ cu,
                                       const int* __restrict__ cpb, int B, int T,
                                       int* __restrict__ out) {
  __shared__ int s_cnt[MR];
  __shared__ int s_req[MR];
  __shared__ float s_q[512];
  __shared__ int s_hist[256];
  __shared__ uint32_t s_bm[MBMW];
  __shared__ int s_pc[MBMW + 1];
  __shared__ int s_sel[2];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;

  // The shipped T=8181/B=4 case is all identity rows.  Verify that fact from
  // cu_seqlens on device, then use 1024-int4 CTA ranges: unlike two 513-int4
  // rows, every range and every warp pass stays 128B aligned.
  if (B == 4 && T == 8181) {
    const int c1 = cu[1], c2 = cu[2], c3 = cu[3];
    if (c1 <= 2051 && c2 - c1 <= 2051 && c3 - c2 <= 2051 && T - c3 <= 2051) {
      constexpr int kChunk = 1024;
      const int ntot = T * MW4;
      const int g0 = (int)blockIdx.x * kChunk;
      int4* out4 = reinterpret_cast<int4*>(out);
      // Keep the only integer divide block-uniform; lanes need one carry.
      const int row0 = g0 / MW4;
      int col4 = g0 - row0 * MW4 + tid;
      int row = row0;
      if (col4 >= MW4) {
        col4 -= MW4;
        ++row;
      }
      int req = (row >= c1) + (row >= c2) + (row >= c3);
      int base = req == 0 ? 0 : (req == 1 ? c1 : (req == 2 ? c2 : c3));
#pragma unroll
      for (int k = 0; k < (kChunk + MT - 1) / MT; ++k) {
        const int x = tid + k * MT;
        const int g = g0 + x;
        if (x < kChunk && g < ntot) {
          const int4 v = packfill::qf_rowval(col4, row - base + 1);
          qf_mid_store_cs(out4 + g, v);
        }
        col4 += MT;
        if (col4 >= MW4) {
          col4 -= MW4;
          ++row;
          if (row == c1 || row == c2 || row == c3) base = row;
        }
      }
      return;
    }
  }
  const int r0 = blockIdx.x * MR;

  // ---- trivial rows: one contiguous int4 range for the whole CTA ----------
  // Every warp probes cu_seqlens for the CTA's first row itself and then walks
  // request boundaries forward, so no barrier stands between the launch and
  // the first store.
  int anyslow = 0;
  {
    int q = packfill::qf_find(cu, B, r0 < T ? r0 : T - 1, lane);
    int lo = cu[q], hi = cu[q + 1];
    int4* o4 = (int4*)out + (size_t)r0 * MW4;
    int row = r0, col4 = tid, cnt = r0 - lo + 1;
#pragma unroll
    for (int k = 0; k < MPT; ++k) {
      const int f = tid + k * MT;
      if (f < MFLAT && row < T) {
        if (cnt < 2052)
          o4[f] = packfill::qf_rowval(col4, cnt);
        else
          anyslow = 1;
      }
      col4 += MT;
      if (col4 >= MW4) {
        col4 -= MW4;
        ++row;
        if (row >= hi && q + 1 < B) {
          lo = hi;
          hi = cu[++q + 1];
        }
        cnt = row - lo + 1;
      }
    }
  }
  if (!__syncthreads_or(anyslow)) return;

  // ---- scored rows: re-derive per-row metadata only when some row is slow --
  if (warp < MR) {
    const int row = r0 + warp;
    int c = 0, q = 0;
    if (row < T) {
      q = packfill::qf_find(cu, B, row, lane);
      c = row - cu[q] + 1;
    }
    if (lane == 0) {
      s_cnt[warp] = c;
      s_req[warp] = q;
    }
  }
  __syncthreads();

  int slow = 0;
#pragma unroll
  for (int rr = 0; rr < MR; ++rr)
    if (s_cnt[rr] >= 2052) slow |= (1 << rr);

  while (slow) {
    const int rr = __ffs(slow) - 1;
    slow &= slow - 1;
    const int t = r0 + rr;
    const int pos = s_cnt[rr] - 1;
    const int vis = (pos + 1) >> 2;  // 513 .. MW in this regime
    const size_t kbase = (size_t)cpb[s_req[rr]] * 64;
    int* orow = out + (size_t)t * MW;
    float* gsc = (float*)orow;  // scores reuse this row, overwritten on emit
    __syncthreads();

    for (int i = tid; i < 512; i += MT) s_q[i] = f8(qi[(size_t)t * 512 + i]);
    for (int i = tid; i < MBMW; i += MT) s_bm[i] = 0u;
    __syncthreads();

    for (int b = tid; b < vis; b += MT) {
      const uint8_t* kp = ckc + (kbase + b) * 128;
      float s = 0.f;
#pragma unroll 1
      for (int h = 0; h < 4; ++h) {
        float d = 0.f;
#pragma unroll 1
        for (int j = 0; j < 8; ++j) {
          const uint4 kv = *(const uint4*)(kp + j * 16);
          const uint32_t w[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
          for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int bb = 0; bb < 4; ++bb)
              d = fmaf(s_q[h * 128 + j * 16 + c * 4 + bb], f8((w[c] >> (8 * bb)) & 0xFFu), d);
        }
        s += qsa_relu_exact(d);
      }
      gsc[b] = s;
    }
    __syncthreads();

    // exact 32-bit byte radix: scores are non-negative, so their float bit
    // patterns order them.  prefix ends as the exact key of rank MTOPK.
    uint32_t prefix = 0u, pmask = 0u;
    int need = MTOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < 256; i += MT) s_hist[i] = 0;
      __syncthreads();
      for (int v = tid; v < vis; v += MT) {
        const uint32_t key = qsa_score_key_exact(gsc[v]);
        if ((key & pmask) == prefix) atomicAdd(&s_hist[(key >> shift) & 255], 1);
      }
      __syncthreads();
      if (tid == 0) {
        int cum = 0, b = 255;
        for (; b > 0; --b) {
          if (cum + s_hist[b] >= need) break;
          cum += s_hist[b];
        }
        s_sel[0] = b;
        s_sel[1] = need - cum;
      }
      __syncthreads();
      prefix |= ((uint32_t)s_sel[0]) << shift;
      pmask |= 0xFFu << shift;
      need = s_sel[1];
      __syncthreads();
    }

    // Selection bitmap first: emitting straight from the scores would clobber
    // score words that later lanes still have to read.
    if (warp == 0) {
      int neq = 0, rank = 0;
      for (int v0 = 0; v0 < vis && rank < MTOPK; v0 += 32) {
        const int v = v0 + lane;
        const uint32_t key = (v < vis) ? qsa_score_key_exact(gsc[v]) : 0u;
        const bool gt = (v < vis) && (key > prefix);
        const bool eq = (v < vis) && (key == prefix);
        const unsigned be = __ballot_sync(0xffffffffu, eq);
        const int eqb = __popc(be & ((1u << lane) - 1u));
        const bool take = gt || (eq && (neq + eqb < need));
        const unsigned bt = __ballot_sync(0xffffffffu, take);
        if (lane == 0) s_bm[v0 >> 5] = bt;
        neq += __popc(be);
        rank += __popc(bt);
      }
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0;
      for (int w = 0; w < MBMW; ++w) {
        s_pc[w] = acc;
        acc += __popc(s_bm[w]);
      }
    }
    __syncthreads();

    for (int v = tid; v < vis; v += MT) {
      const uint32_t wrd = s_bm[v >> 5];
      if ((wrd >> (v & 31)) & 1u) {
        const int myrank = s_pc[v >> 5] + __popc(wrd & ((1u << (v & 31)) - 1u));
        int4 q4;
        q4.x = 4 * v;
        q4.y = 4 * v + 1;
        q4.z = 4 * v + 2;
        q4.w = 4 * v + 3;
        *(int4*)(orow + 4 * myrank) = q4;
      }
    }
    if (tid == 0) {
      const int rem = (pos + 1) & 3;
      const int ts = vis * 4;
      int4 tl;
      tl.x = (0 < rem) ? ts : -1;
      tl.y = (1 < rem) ? ts + 1 : -1;
      tl.z = (2 < rem) ? ts + 2 : -1;
      tl.w = 4 * MTOPK + rem;
      *(int4*)(orow + 4 * MTOPK) = tl;
    }
    __syncthreads();
  }
}

// Requires T <= 8208 so vis <= PWIDTH and the destination row can hold scores.
inline void Launch(const uint8_t* qi, const uint8_t* ckc, const int* cu, const int* cpb, int B,
                   int T, int* out, cudaStream_t stream) {
  const unsigned grid =
      (B == 4 && T == 8181) ? (unsigned)((T * MW4 + 1023) / 1024) : (unsigned)((T + MR - 1) / MR);
  qf_mid<<<grid, MT, 0, stream>>>(qi, ckc, cu, cpb, B, T, out);
}

}  // namespace packmid
