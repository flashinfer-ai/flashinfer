// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// QSA stage-1 token top-k indexer, SM107 (Rubin), single fused kernel.
//
// Exact algebraic facts the kernel exploits (no approximation anywhere):
//   * vis(t) = (pos+1)/CR visible compressed blocks, nb = min(vis, 512).
//   * When vis <= 512 every visible block is selected regardless of its score,
//     so the packed row is just ascending ids 0..pos with count pos+1.
//   * Otherwise exactly 512 blocks are selected; expansion is monotone, so the
//     row is "selected b -> 4b..4b+3" in ascending b, then the causal tail.
//   * e4m3 products are exact in fp32 and every partial sum is a multiple of
//     2^-18, so fp32 accumulation in any order (incl. tensor cores) reproduces
//     the reference bit-exactly.
//
// One CTA owns RPC consecutive rows.  Trivial rows are filled with int4 vector
// stores; the rest are scored in groups of STILE tokens with
// mma.sync.m16n8k32.e4m3 (A rows = token*4+head, so the 4 index heads reduce
// with two shfl_xor), selected with a 3-round radix histogram over the fp32
// score bits, and emitted through a ballot/popcount compaction.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace ffi = tvm::ffi;

#define TPB 256
#define NWARP (TPB / 32)
#define RI4 513  // PWIDTH/4 = 2052/4
#define TOPK 512
#define TRIVP 2050  // pos <= TRIVP  =>  trivial row
#define MAXRPC 64
#define STILE 8                 // tokens scored together by one CTA
#define VCAP 2048               // blocks per token in the smem score buffer
#define SCORE_STRIDE (RI4 * 4)  // destination row reinterpreted as float scratch
#define SCAP SCORE_STRIDE       // chunk capacity of the scalar fallback
#define NBT 128                 // blocks staged per K tile
#define META_I 144              // ints reserved for s_pos/s_req/s_tmp/s_res
#define HPW 544                 // padded per-warp histogram (32 groups of 16, stride 17)
#define KROW 160                // padded smem row stride for a 128-byte fp8 vector
#define FULLM 0xffffffffu

__device__ __forceinline__ float f8tof(unsigned char b) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
  return __half2float(__ushort_as_half(hr.x));
}

__device__ __forceinline__ void mma_e4m3(float* d, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ int blockExclScan(int v, int* tmp, int* tot) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(FULLM, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) tmp[wid] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < NWARP; ++i) {
      int t = tmp[i];
      tmp[i] = acc;
      acc += t;
    }
    tmp[NWARP] = acc;
  }
  __syncthreads();
  int r = tmp[wid] + x - v;
  *tot = tmp[NWARP];
  __syncthreads();
  return r;
}

// Largest bin b with suffixAbove(b) < K <= suffixAbove(b)+hist[b].
// res[0]<-b, res[1]<-suffixAbove(b), res[2]<-hist[b].  NB % TPB == 0.
__device__ __forceinline__ void findCross(const int* hist, int NB, int K, int* tmp, int* res) {
  const int g = NB / TPB;
  const int base = threadIdx.x * g;
  int local = 0;
#pragma unroll 1
  for (int i = 0; i < g; ++i) local += hist[base + i];
  int tot;
  int pre = blockExclScan(local, tmp, &tot);
  int cum = tot - pre - local;
#pragma unroll 1
  for (int i = g - 1; i >= 0; --i) {
    int h = hist[base + i];
    if (cum < K && cum + h >= K) {
      res[0] = base + i;
      res[1] = cum;
      res[2] = h;
    }
    cum += h;
  }
  __syncthreads();
}

__device__ __forceinline__ void scoreChunkScalar(const unsigned char* __restrict__ Kb,
                                                 const float* __restrict__ sq,
                                                 float* __restrict__ sc, int cs, int cl) {
  for (int li = threadIdx.x; li < cl; li += TPB) {
    const unsigned char* kp = Kb + (size_t)(cs + li) * 128;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 1
    for (int c = 0; c < 128; c += 16) {
      uint4 w = *(const uint4*)(kp + c);
      const unsigned char* bb = (const unsigned char*)&w;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        float kv = f8tof(bb[t]);
        int d = c + t;
        a0 = fmaf(sq[d], kv, a0);
        a1 = fmaf(sq[128 + d], kv, a1);
        a2 = fmaf(sq[256 + d], kv, a2);
        a3 = fmaf(sq[384 + d], kv, a3);
      }
    }
    sc[li] = qsa_relu_exact(a0) + qsa_relu_exact(a1) + qsa_relu_exact(a2) + qsa_relu_exact(a3);
  }
}

// Write the packed row for a token whose selected block ids are in sel[0..511].
__device__ __forceinline__ void writeRow(int4* __restrict__ out, int tok, const int* sel, int pos,
                                         int V) {
  int4* orow = out + (size_t)tok * RI4;
  for (int x = threadIdx.x; x < TOPK; x += TPB) {
    int c = sel[x] << 2;
    int4 v;
    v.x = c;
    v.y = c + 1;
    v.z = c + 2;
    v.w = c + 3;
    orow[x] = v;
  }
  if (threadIdx.x == 0) {
    int rem = (pos + 1) & 3;
    int t0 = V << 2;
    int4 v;
    v.x = (rem > 0) ? t0 : -1;
    v.y = (rem > 1) ? t0 + 1 : -1;
    v.z = (rem > 2) ? t0 + 2 : -1;
    v.w = 2048 + rem;
    orow[TOPK] = v;
  }
}

__global__ void __launch_bounds__(TPB, 3)
    qsa_kernel(const unsigned char* __restrict__ qi, const unsigned char* __restrict__ ckc,
               const int* __restrict__ cu, const int* __restrict__ cpb, int4* __restrict__ out,
               int T, int B, int RPC, int nCTA) {
  // Layout is ordered so everything a trivial-only launch touches lives in the
  // first SMEM_SMALL bytes; such launches then request only that much dynamic
  // smem and get full 4-CTA/SM occupancy for the pure-store work.
  extern __shared__ __align__(16) char smem[];
  int* s_pos = (int*)smem;            // MAXRPC
  int* s_req = s_pos + MAXRPC;        // MAXRPC
  int* s_tmp = s_req + MAXRPC;        // NWARP+2
  int* s_res = s_tmp + (NWARP + 2);   // 4
  int* s_hist = (int*)smem + META_I;  // NWARP*HPW (also cu cache)
  unsigned char* s_K = reinterpret_cast<unsigned char*>(smem + META_I * 4 + NWARP * HPW * 4);
  int* s_sel = (int*)s_K;  // aliased: K tiles are dead by then
  unsigned char* s_Q = s_K + (size_t)NBT * KROW;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gq = lane >> 2;
  const int tg = lane & 3;

  const int cta = nCTA - 1 - (int)blockIdx.x;  // heaviest rows first
  const int row0 = cta * RPC;
  if (row0 >= T) return;
  int nr = T - row0;
  if (nr > RPC) nr = RPC;

  const bool cuSmem = (B + 1) <= 2048;
  if (cuSmem) {
    for (int i = tid; i <= B; i += TPB) s_hist[i] = cu[i];
    __syncthreads();
  }
  if (tid < nr) {
    int row = row0 + tid;
    int lo = 0, hi = B - 1;
    if (cuSmem) {
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (s_hist[mid] <= row)
          lo = mid;
        else
          hi = mid - 1;
      }
      s_pos[tid] = row - s_hist[lo];
    } else {
      while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (cu[mid] <= row)
          lo = mid;
        else
          hi = mid - 1;
      }
      s_pos[tid] = row - cu[lo];
    }
    s_req[tid] = lo;
  }
  __syncthreads();

  // ---------------- phase 1: trivial rows (pos <= 2050) ----------------
  {
    const int total = nr * RI4;
    int4* obase = out + (size_t)row0 * RI4;
    for (int i = tid; i < total; i += TPB) {
      int lr = i / RI4;
      int j = i - lr * RI4;
      int pos = s_pos[lr];
      if (pos > TRIVP) continue;
      int c = j << 2;
      int4 v;
      if (j < 512) {
        v.x = (c <= pos) ? c : -1;
        v.y = (c + 1 <= pos) ? c + 1 : -1;
        v.z = (c + 2 <= pos) ? c + 2 : -1;
        v.w = (c + 3 <= pos) ? c + 3 : -1;
      } else {
        v.x = (2048 <= pos) ? 2048 : -1;
        v.y = (2049 <= pos) ? 2049 : -1;
        v.z = (2050 <= pos) ? 2050 : -1;
        v.w = pos + 1;
      }
      obase[i] = v;
    }
  }

  if (T <= TRIVP + 1) return;

  // ---------------- phase 2: scored rows (pos >= 2051) -----------------
  // Groups are maximal runs of <= STILE consecutive non-trivial rows from the
  // SAME request: a group that straddled a request boundary would need two K
  // bases, and dropping it to the scalar fallback made that one CTA the whole
  // kernel's critical path.
  int lrc = 0;
  while (lrc < nr) {
    if (s_pos[lrc] <= TRIVP) {
      ++lrc;
      continue;
    }
    const int req0 = s_req[lrc];
    int cnt = 1, maxpos = s_pos[lrc];
    while (cnt < STILE && lrc + cnt < nr) {
      int p = s_pos[lrc + cnt];
      if (p <= TRIVP || s_req[lrc + cnt] != req0) break;
      if (p > maxpos) maxpos = p;
      ++cnt;
    }
    const int g0 = lrc;
    const int ng = cnt;
    lrc += cnt;
    const int Vtile = (maxpos + 1) >> 2;
    const bool fast = (Vtile <= VCAP);

    if (fast) {
      float* g_sc = reinterpret_cast<float*>(out + (size_t)(row0 + g0) * RI4);
      const int qrows = ng * 4;
      for (int idx = tid; idx < STILE * 4 * 8; idx += TPB) {
        int row = idx >> 3, ch = idx & 7;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (row < qrows)
          val =
              *((const uint4*)(qi + (size_t)(row0 + g0 + (row >> 2)) * 512 + (row & 3) * 128) + ch);
        *(uint4*)(s_Q + row * KROW + ch * 16) = val;
      }
      const unsigned char* Kb = ckc + ((size_t)cpb[req0] * 64) * 128;
      __syncthreads();

      for (int nb0 = 0; nb0 < Vtile; nb0 += NBT) {
        const int nbn = min(NBT, Vtile - nb0);
        for (int idx = tid; idx < NBT * 8; idx += TPB) {
          int row = idx >> 3, ch = idx & 7;
          uint4 val = make_uint4(0u, 0u, 0u, 0u);
          if (row < nbn) val = *((const uint4*)(Kb + (size_t)(nb0 + row) * 128) + ch);
          *(uint4*)(s_K + row * KROW + ch * 16) = val;
        }
        __syncthreads();

        float acc[2][2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          unsigned a[2][4];
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            uint2 lo = *(const uint2*)(s_Q + (m * 16 + gq) * KROW + ks * 32 + tg * 8);
            uint2 hi = *(const uint2*)(s_Q + (m * 16 + gq + 8) * KROW + ks * 32 + tg * 8);
            a[m][0] = lo.x;
            a[m][2] = lo.y;
            a[m][1] = hi.x;
            a[m][3] = hi.y;
          }
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            int blk = (warp * 2 + n) * 8 + gq;
            uint2 bb = *(const uint2*)(s_K + blk * KROW + ks * 32 + tg * 8);
            unsigned b[2] = {bb.x, bb.y};
#pragma unroll
            for (int m = 0; m < 2; ++m) mma_e4m3(acc[m][n], a[m], b);
          }
        }

        const bool wr = ((lane & 12) == 0);
        const int half = lane >> 4;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) v[e] = qsa_relu_exact(acc[m][n][e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              v[e] += __shfl_xor_sync(FULLM, v[e], 4);
              v[e] += __shfl_xor_sync(FULLM, v[e], 8);
            }
            if (wr) {
              int col = nb0 + (warp * 2 + n) * 8 + tg * 2;
              const int rowA = m * 4 + half;
              const int rowB = rowA + 2;
              if (rowA < ng) *(float2*)&g_sc[rowA * SCORE_STRIDE + col] = make_float2(v[0], v[1]);
              if (rowB < ng) *(float2*)&g_sc[rowB * SCORE_STRIDE + col] = make_float2(v[2], v[3]);
            }
          }
        }
        __syncthreads();
      }

      // ---- warp-parallel selection: warp w owns token w, no block barriers ----
      if (warp < ng) {
        const int pos = s_pos[g0 + warp];
        if (pos > TRIVP) {
          const int tok = row0 + g0 + warp;
          const int V = (pos + 1) >> 2;
          const float* sc = g_sc + warp * SCORE_STRIDE;
          int* H = s_hist + warp * HPW;
          int* WS = s_sel + warp * TOPK;
          const unsigned lmlt = (1u << lane) - 1u;

          unsigned hi = 0u;
          int hiShift = 32, need = TOPK;
          unsigned tau = 0u;
          int rNeed = -1;
#pragma unroll
          for (int rd = 0; rd < 4; ++rd) {
            const int sh = (rd == 0) ? 23 : (rd == 1) ? 14 : (rd == 2) ? 5 : 0;
            const int nb = (rd == 3) ? 32 : 512;
            const unsigned msk = (rd == 3) ? 0x1fu : 0x1ffu;
            const int hsz = (nb >> 4) * 17;
            for (int i = lane; i < hsz; i += 32) H[i] = 0;
            __syncwarp();
            for (int v = lane; v < V; v += 32) {
              unsigned u = qsa_score_key_exact(sc[v]);
              if (rd == 0 || (u >> hiShift) == hi) {
                int b = (int)((u >> sh) & msk);
                atomicAdd(&H[(b >> 4) * 17 + (b & 15)], 1);
              }
            }
            __syncwarp();
            const int gpl = nb >> 5;
            const int bse = lane * gpl;
            int local = 0;
#pragma unroll 1
            for (int i = 0; i < gpl; ++i) {
              int b = bse + i;
              local += H[(b >> 4) * 17 + (b & 15)];
            }
            int x = local;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
              int y = __shfl_down_sync(FULLM, x, o);
              if (lane + o < 32) x += y;
            }
            int cum = x - local;
            int fb = -1, fc = 0, fh = 0;
#pragma unroll 1
            for (int i = gpl - 1; i >= 0; --i) {
              int b = bse + i;
              int h = H[(b >> 4) * 17 + (b & 15)];
              if (cum < need && cum + h >= need) {
                fb = b;
                fc = cum;
                fh = h;
              }
              cum += h;
            }
            unsigned mm = __ballot_sync(FULLM, fb >= 0);
            int srcl = mm ? (__ffs((int)mm) - 1) : 0;
            int bb = __shfl_sync(FULLM, fb, srcl);
            int cc = __shfl_sync(FULLM, fc, srcl);
            int hh = __shfl_sync(FULLM, fh, srcl);
            need -= cc;
            hi = (rd == 0) ? (unsigned)bb : ((hi << ((rd == 3) ? 5 : 9)) | (unsigned)bb);
            hiShift = sh;
            if (hh == need) {
              tau = hi << sh;
              break;
            }
            if (rd == 3) {
              tau = hi;
              rNeed = need;
              break;
            }
          }

          int vcut = 0x7fffffff;
          if (rNeed >= 0) {
            int tb = 0, vc = 0x7fffffff;
            for (int v0 = 0; v0 < V; v0 += 32) {
              int v = v0 + lane;
              int f = (v < V && qsa_score_key_exact(sc[v]) == tau) ? 1 : 0;
              unsigned mk = __ballot_sync(FULLM, f);
              int rk = tb + __popc(mk & lmlt);
              if (f && rk == rNeed - 1) vc = v;
              tb += __popc(mk);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vc = min(vc, __shfl_xor_sync(FULLM, vc, o));
            vcut = vc;
          }

          int bse2 = 0;
          for (int v0 = 0; v0 < V; v0 += 32) {
            int v = v0 + lane;
            int f = 0;
            if (v < V) {
              unsigned u = qsa_score_key_exact(sc[v]);
              f = (u > tau || (u == tau && v <= vcut)) ? 1 : 0;
            }
            unsigned mk = __ballot_sync(FULLM, f);
            int rk = bse2 + __popc(mk & lmlt);
            if (f && rk < TOPK) WS[rk] = v;
            bse2 += __popc(mk);
          }
          __syncwarp();

          int4* orow = out + (size_t)tok * RI4;
          for (int x = lane; x < TOPK; x += 32) {
            int c = WS[x] << 2;
            int4 vv;
            vv.x = c;
            vv.y = c + 1;
            vv.z = c + 2;
            vv.w = c + 3;
            orow[x] = vv;
          }
          if (lane == 0) {
            int rem = (pos + 1) & 3;
            int t0 = V << 2;
            int4 vv;
            vv.x = (rem > 0) ? t0 : -1;
            vv.y = (rem > 1) ? t0 + 1 : -1;
            vv.z = (rem > 2) ? t0 + 2 : -1;
            vv.w = 2048 + rem;
            orow[TOPK] = vv;
          }
        }
      }
      __syncthreads();
    } else {
      // ---------- generic fallback: chunked scalar scoring + selection -------
      for (int i = 0; i < ng; ++i) {
        const int pos = s_pos[g0 + i];
        if (pos <= TRIVP) continue;
        const int tok = row0 + g0 + i;
        const int V = (pos + 1) >> 2;
        const unsigned char* Kb = ckc + ((size_t)cpb[s_req[g0 + i]] * 64) * 128;
        float* g_sc = reinterpret_cast<float*>(out + (size_t)tok * RI4);
        float* sq = (float*)s_Q;
        __syncthreads();
        {
          const unsigned char* qp = qi + (size_t)tok * 512;
          for (int x = tid; x < 512; x += TPB) sq[x] = f8tof(qp[x]);
        }
        __syncthreads();
        const int nch = (V + SCAP - 1) / SCAP;

        for (int x = tid; x < 1024; x += TPB) s_hist[x] = 0;
        __syncthreads();
        for (int ch = 0; ch < nch; ++ch) {
          const int cs = ch * SCAP, cl = min(SCAP, V - cs);
          scoreChunkScalar(Kb, sq, g_sc, cs, cl);
          __syncthreads();
          for (int x = tid; x < cl; x += TPB)
            atomicAdd(&s_hist[qsa_score_key_exact(g_sc[x]) >> 21], 1);
          __syncthreads();
        }
        findCross(s_hist, 1024, TOPK, s_tmp, s_res);
        const int b1 = s_res[0];
        const int r1 = TOPK - s_res[1];

        for (int x = tid; x < 2048; x += TPB) s_hist[x] = 0;
        __syncthreads();
        for (int ch = 0; ch < nch; ++ch) {
          const int cs = ch * SCAP, cl = min(SCAP, V - cs);
          if (nch > 1) {
            scoreChunkScalar(Kb, sq, g_sc, cs, cl);
            __syncthreads();
          }
          for (int x = tid; x < cl; x += TPB) {
            unsigned u = qsa_score_key_exact(g_sc[x]);
            if ((int)(u >> 21) == b1) atomicAdd(&s_hist[(u >> 10) & 0x7ffu], 1);
          }
          __syncthreads();
        }
        findCross(s_hist, 2048, r1, s_tmp, s_res);
        const int b2 = s_res[0];
        const int r2 = r1 - s_res[1];

        const unsigned pf = ((unsigned)b1 << 11) | (unsigned)b2;
        for (int x = tid; x < 1024; x += TPB) s_hist[x] = 0;
        __syncthreads();
        for (int ch = 0; ch < nch; ++ch) {
          const int cs = ch * SCAP, cl = min(SCAP, V - cs);
          if (nch > 1) {
            scoreChunkScalar(Kb, sq, g_sc, cs, cl);
            __syncthreads();
          }
          for (int x = tid; x < cl; x += TPB) {
            unsigned u = qsa_score_key_exact(g_sc[x]);
            if ((u >> 10) == pf) atomicAdd(&s_hist[u & 0x3ffu], 1);
          }
          __syncthreads();
        }
        findCross(s_hist, 1024, r2, s_tmp, s_res);
        const int b3 = s_res[0];
        const int r3 = r2 - s_res[1];
        const int cntEq = s_res[2];
        const unsigned tau = ((unsigned)b1 << 21) | ((unsigned)b2 << 10) | (unsigned)b3;

        int vcut = 0x7fffffff;
        if (r3 < cntEq) {
          if (tid == 0) s_res[3] = 0x7fffffff;
          __syncthreads();
          int tieBase = 0;
          for (int ch = 0; ch < nch; ++ch) {
            const int cs = ch * SCAP, cl = min(SCAP, V - cs);
            if (nch > 1) {
              scoreChunkScalar(Kb, sq, g_sc, cs, cl);
              __syncthreads();
            }
            for (int v0 = 0; v0 < cl; v0 += TPB) {
              int li = v0 + tid;
              int f = (li < cl && qsa_score_key_exact(g_sc[li]) == tau) ? 1 : 0;
              int tot;
              int pre = blockExclScan(f, s_tmp, &tot);
              if (f && tieBase + pre == r3 - 1) s_res[3] = cs + li;
              tieBase += tot;
            }
          }
          __syncthreads();
          vcut = s_res[3];
        }

        int selBase = 0;
        for (int ch = 0; ch < nch; ++ch) {
          const int cs = ch * SCAP, cl = min(SCAP, V - cs);
          if (nch > 1) {
            scoreChunkScalar(Kb, sq, g_sc, cs, cl);
            __syncthreads();
          }
          for (int v0 = 0; v0 < cl; v0 += TPB) {
            int li = v0 + tid;
            int v = cs + li;
            int f = 0;
            if (li < cl) {
              unsigned u = qsa_score_key_exact(g_sc[li]);
              f = (u > tau || (u == tau && v <= vcut)) ? 1 : 0;
            }
            int tot;
            int pre = blockExclScan(f, s_tmp, &tot);
            int slot = selBase + pre;
            if (f && slot < TOPK) s_sel[slot] = v;
            selBase += tot;
          }
        }
        __syncthreads();
        writeRow(out, tok, s_sel, pos, V);
        __syncthreads();
      }
    }
  }
}

static size_t smem_full() {
  return (size_t)(META_I + NWARP * HPW) * 4 + (size_t)NBT * KROW + (size_t)STILE * 4 * KROW;
}
static size_t smem_small() { return (size_t)(META_I + 2048) * 4; }

// For a single request with at most 512 visible blocks, every block is kept.
// Each CTA writes one packed row without metadata loads or synchronization.
__global__ __launch_bounds__(TPB, 4) void qsa_trivial_b1(int4* __restrict__ out, int T) {
  const int tok = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  if (tok >= T) return;
  int4* row = out + (size_t)tok * RI4;
  const int count = tok + 1;
  const int c0 = tid * 4;
  const int c1 = c0 + 1024;
  row[tid] = make_int4(c0 < count ? c0 : -1, c0 + 1 < count ? c0 + 1 : -1,
                       c0 + 2 < count ? c0 + 2 : -1, c0 + 3 < count ? c0 + 3 : -1);
  row[tid + 256] = make_int4(c1 < count ? c1 : -1, c1 + 1 < count ? c1 + 1 : -1,
                             c1 + 2 < count ? c1 + 2 : -1, c1 + 3 < count ? c1 + 3 : -1);
  if (tid == 0) {
    row[512] = make_int4(2048 < count ? 2048 : -1, 2049 < count ? 2049 : -1,
                         2050 < count ? 2050 : -1, count);
  }
}

// Tiny batches do not expose enough row CTAs to the 212-SM device.  Four
// independent 128-thread CTAs cover disjoint int4 column ranges of each row.
__global__ __launch_bounds__(128, 4) void qsa_trivial_b1_split(int4* __restrict__ out, int T) {
  const int tok = (int)blockIdx.x;
  const int part = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  if (tok >= T) return;
  constexpr int kParts = 4;
  constexpr int kPartItems = (RI4 + kParts - 1) / kParts;
  const int begin = part * kPartItems;
  const int end = min(begin + kPartItems, RI4);
  int4* row = out + (size_t)tok * RI4;
  const int count = tok + 1;
  int j = begin + tid;
  if (j < end) {
    const int c = j * 4;
    int4 v = make_int4(c < count ? c : -1, c + 1 < count ? c + 1 : -1, c + 2 < count ? c + 2 : -1,
                       c + 3 < count ? c + 3 : -1);
    if (j == RI4 - 1) v.w = count;
    row[j] = v;
  }
  j += 128;
  if (tid == 0 && j < end) {
    const int c = j * 4;
    int4 v = make_int4(c < count ? c : -1, c + 1 < count ? c + 1 : -1, c + 2 < count ? c + 2 : -1,
                       c + 3 < count ? c + 3 : -1);
    if (j == RI4 - 1) v.w = count;
    row[j] = v;
  }
}

// At very small T, eight 64-thread column CTAs expose almost one block per SM.
__global__ __launch_bounds__(64, 8) void qsa_trivial_b1_split8(int4* __restrict__ out, int T) {
  const int tok = (int)blockIdx.x;
  const int part = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  if (tok >= T) return;
  constexpr int kParts = 8;
  constexpr int kPartItems = (RI4 + kParts - 1) / kParts;
  const int begin = part * kPartItems;
  const int end = min(begin + kPartItems, RI4);
  int4* row = out + (size_t)tok * RI4;
  const int count = tok + 1;
  int j = begin + tid;
  if (j < end) {
    const int c = j * 4;
    int4 v = make_int4(c < count ? c : -1, c + 1 < count ? c + 1 : -1, c + 2 < count ? c + 2 : -1,
                       c + 3 < count ? c + 3 : -1);
    if (j == RI4 - 1) v.w = count;
    row[j] = v;
  }
  j += 64;
  if (tid == 0 && j < end) {
    const int c = j * 4;
    int4 v = make_int4(c < count ? c : -1, c + 1 < count ? c + 1 : -1, c + 2 < count ? c + 2 : -1,
                       c + 3 < count ? c + 3 : -1);
    if (j == RI4 - 1) v.w = count;
    row[j] = v;
  }
}

constexpr int kGrouped4Rows = 4;

// High-B short requests benefit from enough independent row groups to fill
// the machine while staging cu_seqlens only once for four output rows.
__global__ __launch_bounds__(TPB, 4) void qsa_trivial_group4(const int* __restrict__ cu,
                                                             int4* __restrict__ out, int T, int B) {
  const int tid = (int)threadIdx.x;
  const int row0 = (int)blockIdx.x * kGrouped4Rows;
  const int nr = min(kGrouped4Rows, T - row0);
  __shared__ int scu[1025];
  __shared__ int pos[kGrouped4Rows];
  for (int i = tid; i <= B; i += blockDim.x) scu[i] = cu[i];
  __syncthreads();
  if (tid < nr) {
    const int tok = row0 + tid;
    int lo = 0, hi = B;
    while (lo + 1 < hi) {
      const int mid = (lo + hi) >> 1;
      if (scu[mid] <= tok)
        lo = mid;
      else
        hi = mid;
    }
    pos[tid] = tok - scu[lo];
  }
  __syncthreads();

#pragma unroll
  for (int r = 0; r < kGrouped4Rows; ++r) {
    if (r < nr) {
      int4* row = out + (size_t)(row0 + r) * RI4;
      const int count = pos[r] + 1;
      const int c0 = tid * 4;
      const int c1 = c0 + 1024;
      row[tid] = make_int4(c0 < count ? c0 : -1, c0 + 1 < count ? c0 + 1 : -1,
                           c0 + 2 < count ? c0 + 2 : -1, c0 + 3 < count ? c0 + 3 : -1);
      row[tid + 256] = make_int4(c1 < count ? c1 : -1, c1 + 1 < count ? c1 + 1 : -1,
                                 c1 + 2 < count ? c1 + 2 : -1, c1 + 3 < count ? c1 + 3 : -1);
      if (tid == 0) {
        row[512] = make_int4(2048 < count ? 2048 : -1, 2049 < count ? 2049 : -1,
                             2050 < count ? 2050 : -1, count);
      }
    }
  }
}

// Reuse the proven per-row exact selector as a low-overhead path for a few
// short requests.  DEVICE_ONLY omits its host entry point and TVM export.
#ifndef DEVICE_ONLY
#define MID_LOCAL_DEVICE_ONLY 1
#define DEVICE_ONLY 1
#endif
#include "v18_device.cuh"
#ifdef MID_LOCAL_DEVICE_ONLY
#undef DEVICE_ONLY
#undef MID_LOCAL_DEVICE_ONLY
#endif

void Run(ffi::TensorView qi, ffi::TensorView ck_cache, ffi::TensorView cu_seqlens,
         ffi::TensorView cpage_base, ffi::TensorView packed) {
  const int T = (int)qi.shape()[0];
  const int B = (int)cpage_base.shape()[0];
  if (T <= 0) return;

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id));
  if (B == 1 && T <= TRIVP + 1) {
    if (T <= 64)
      qsa_trivial_b1_split8<<<dim3((unsigned)T, 8u), 64, 0, stream>>>((int4*)packed.data_ptr(), T);
    else if (T <= 256)
      qsa_trivial_b1_split<<<dim3((unsigned)T, 4u), 128, 0, stream>>>((int4*)packed.data_ptr(), T);
    else
      qsa_trivial_b1<<<T, TPB, 0, stream>>>((int4*)packed.data_ptr(), T);
    return;
  }
  if (B > 4 && B <= 1024 && T <= TRIVP + 1) {
    qsa_trivial_group4<<<(T + kGrouped4Rows - 1) / kGrouped4Rows, TPB, 0, stream>>>(
        (const int*)cu_seqlens.data_ptr(), (int4*)packed.data_ptr(), T, B);
    return;
  }
  if (B <= 4 && (T + B - 1) / B <= TRIVP + 1) {
    qsa_indexer_kernel<<<T, kThreads, 0, stream>>>(
        (const __nv_fp8_e4m3*)qi.data_ptr(), (const __nv_fp8_e4m3*)ck_cache.data_ptr(),
        (const int*)cu_seqlens.data_ptr(), (const int*)cpage_base.data_ptr(),
        (int*)packed.data_ptr(), T, B, false);
    return;
  }

  static int s_init = 0;
  if (!s_init) {
    cudaFuncSetAttribute(qsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_full());
    cudaFuncSetAttribute(qsa_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    s_init = 1;
  }
  const size_t sm = (T <= TRIVP + 1) ? smem_small() : smem_full();

  // Rows per CTA: enough to cover the output, and enough that re-probing the
  // B+1 cu_seqlens entries per CTA stays small next to that CTA's stores.
  int RPC = (T + 1023) / 1024;
  int rb = (B + 256) / 256;
  if (rb > RPC) RPC = rb;
  if (RPC < 1) RPC = 1;
  if (RPC > STILE) RPC = (RPC + STILE - 1) / STILE * STILE;
  if (RPC > MAXRPC) RPC = MAXRPC;
  const int nCTA = (T + RPC - 1) / RPC;

  qsa_kernel<<<nCTA, TPB, sm, stream>>>(
      (const unsigned char*)qi.data_ptr(), (const unsigned char*)ck_cache.data_ptr(),
      (const int*)cu_seqlens.data_ptr(), (const int*)cpage_base.data_ptr(),
      (int4*)packed.data_ptr(), T, B, RPC, nCTA);
}

#ifndef DEVICE_ONLY
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif
