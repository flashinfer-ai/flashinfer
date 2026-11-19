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
#ifndef FLASHINFER_MLA_BATCH_INVARIANT_SPARSE_CUH_
#define FLASHINFER_MLA_BATCH_INVARIANT_SPARSE_CUH_

// Kernels of the sparse MLA classes; the term is flashinfer/mla/batch_invariant/spec.py.

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace mla {
namespace bi_sparse {

#include "batch_invariant_sparse_plan.inc"

constexpr int kTile = 128;
constexpr int kLatent = 512;
constexpr int kRowPad = 8;
constexpr int kWsRow = kLatent + 4;  // O[512], m, l, present count, pad
constexpr int kFixupThreads = 128;
constexpr int kCvtBatch = 4;  // loads a thread issues before it uses the first
constexpr int kFixupBatch = 18;
constexpr int kSmemHead = 6144;  // index list, mask words, the warps' exchange slots

struct Params {
  const void* cache;
  const void* q;
  const int32_t* idx;
  const int32_t* lens;
  float* ws;
  uint16_t* out;
  float c, nc, s2;
  int rows, heads, width, tiles;
  int head_block, tiles_per_unit;
  int64_t cache_rows;
};

__device__ __forceinline__ float Ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t Bf16Bits(float x) {
  uint16_t r;
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void Mma16(float (&d)[4], const uint32_t (&a)[4],
                                      const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void LdMatrix4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void CpAsync16(void* dst, const void* src) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ void CpAsyncWait() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void LdMatrix4T(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// Two e4m3 bytes to the two bf16 values they denote: fp8 -> f16 -> f32 -> bf16, every step exact.
__device__ __forceinline__ uint32_t E4m3x2ToBf16(uint32_t two) {
  const uint16_t in = static_cast<uint16_t>(two);
  uint32_t h;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(in));
  const uint16_t h0 = static_cast<uint16_t>(h), h1 = static_cast<uint16_t>(h >> 16);
  float f0, f1;
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(h0));
  asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(h1));
  uint32_t b;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b) : "f"(f1), "f"(f0));
  return b;
}

__device__ __forceinline__ uint4 E4m3x8ToBf16(uint64_t w) {
  const uint32_t lo = static_cast<uint32_t>(w), hi = static_cast<uint32_t>(w >> 32);
  return make_uint4(E4m3x2ToBf16(lo), E4m3x2ToBf16(lo >> 16), E4m3x2ToBf16(hi),
                    E4m3x2ToBf16(hi >> 16));
}

// A thread's own steps of the tile reductions: the whole tree, or its warp's share of it.
__device__ __forceinline__ float RowMaxOwn(const float (&x)[32]) { return sparse_mla_rowmax(x); }
__device__ __forceinline__ float RowMaxOwn(const float (&x)[8]) {
  return sparse_mla_rowmax_w4_part(x);
}
__device__ __forceinline__ float RowSumOwn(const float (&x)[32]) { return sparse_mla_rowsum(x); }
__device__ __forceinline__ float RowSumOwn(const float (&x)[8]) {
  return sparse_mla_rowsum_w4_part(x);
}

template <int D>
constexpr int SmemRowBytes() {
  return (D + kRowPad) * 2;
}

template <int D>
constexpr size_t SmemBytes(int head_block) {
  return kSmemHead + static_cast<size_t>(kTile + head_block) * SmemRowBytes<D>();
}

// A raw e4m3 row is copied into the upper half of its own slot and expanded in place.
template <int D>
constexpr int RawOffset() {
  return SmemRowBytes<D>() - D;
}

// A warp expands rows warp, warp + nwarps, ...: a store only reaches raw bytes of lower units,
// loaded before it.
template <int D>
__device__ __forceinline__ void ExpandRows(unsigned char* slots, int nrows, int warp, int nwarps,
                                           int lane) {
  constexpr int SK = SmemRowBytes<D>(), U = D / 16;
  const int units = nrows / nwarps * U;
  for (int b0 = 0; b0 < units; b0 += 32 * kCvtBatch) {
    uint4 w[kCvtBatch];
    for (int k = 0; k < kCvtBatch; ++k) {
      const int f = b0 + 32 * k + lane;
      if (f < units)
        w[k] = *reinterpret_cast<const uint4*>(slots + (warp + f / U * nwarps) * SK +
                                               RawOffset<D>() + 16 * (f % U));
    }
    __syncwarp();
    for (int k = 0; k < kCvtBatch; ++k) {
      const int f = b0 + 32 * k + lane;
      if (f >= units) continue;
      unsigned char* dst = slots + (warp + f / U * nwarps) * SK + 32 * (f % U);
      const uint64_t lo = w[k].x | (static_cast<uint64_t>(w[k].y) << 32);
      const uint64_t hi = w[k].z | (static_cast<uint64_t>(w[k].w) << 32);
      *reinterpret_cast<uint4*>(dst) = E4m3x8ToBf16(lo);
      *reinterpret_cast<uint4*>(dst + 16) = E4m3x8ToBf16(hi);
    }
  }
}

// One CTA: tiles [t0, t1) of one row for one block of heads (tiles_per_unit 0: all, and it folds).
template <int D, bool FP8, int PVP, int SPLIT>
__global__ void UnitKernel(const Params p) {
  constexpr int SK = SmemRowBytes<D>();
  constexpr int KSTEPS = D / 16;
  constexpr int NB = PVP / 8;
  constexpr int NBK = 16 / SPLIT;
  extern __shared__ __align__(16) unsigned char smem[];
  int32_t* s_idx = reinterpret_cast<int32_t*>(smem);
  uint32_t* s_mask = reinterpret_cast<uint32_t*>(smem + 512);
  float* s_x = reinterpret_cast<float*>(smem + 1024);
  uint32_t* s_p = reinterpret_cast<uint32_t*>(smem + 2048);
  unsigned char* s_keys = smem + kSmemHead;
  unsigned char* s_q = s_keys + kTile * SK;

  const int tid = threadIdx.x, nthr = blockDim.x, lane = tid & 31, warp = tid >> 5;
  const int HB = p.head_block, H = p.heads;
  const int nhb = (H + HB - 1) / HB;
  const bool fused = p.tiles_per_unit == 0;
  const int tpu = fused ? p.tiles : p.tiles_per_unit;
  const int upr = (p.tiles + tpu - 1) / tpu;
  const int u = blockIdx.x;
  const int r = u / (upr * nhb), h0 = ((u / upr) % nhb) * HB;
  const int t0 = (u % upr) * tpu, t1 = min(p.tiles, t0 + tpu);
  const int n_row = p.lens ? max(0, min(p.width, p.lens[r])) : p.width;
  const bool mma = warp < (SPLIT == 4 ? 4 : HB / 16);
  const int gw = SPLIT == 4 ? 0 : warp, kblk0 = SPLIT == 4 ? 4 * warp : 0;
  const int g = lane >> 2, t4 = lane & 3;
  const int hrow[2] = {h0 + 16 * gw + g, h0 + 16 * gw + g + 8};
  const bool hok[2] = {mma && hrow[0] < min(H, h0 + HB), mma && hrow[1] < min(H, h0 + HB)};
  const unsigned char* qa =
      s_q + (16 * gw + (lane & 7) + 8 * ((lane >> 3) & 1)) * SK + 16 * (lane >> 4);
  const unsigned char* kb =
      s_keys + (8 * kblk0 + 8 * (lane >> 4) + (lane & 7)) * SK + 16 * ((lane >> 3) & 1);
  const unsigned char* vbase =
      s_keys + ((lane & 7) + 8 * ((lane >> 3) & 1)) * SK + 16 * (lane >> 4);

  // a thread copies one 16-byte column of every ns-th row: no index arithmetic inside the loops
  constexpr int CPR = (FP8 ? D : 2 * D) / 16;
  constexpr int ROFF = FP8 ? RawOffset<D>() : 0;
  const int ns = nthr / CPR, pc = tid % CPR;
  const unsigned char* qsrc =
      static_cast<const unsigned char*>(p.q) + (static_cast<size_t>(r) * H + h0) * (CPR * 16);
  if (tid < ns * CPR) {
    for (int hh = tid / CPR; hh < HB; hh += ns) {
      void* dst = s_q + hh * SK + ROFF + pc * 16;
      if (h0 + hh < H) {
        CpAsync16(dst, qsrc + hh * (CPR * 16) + pc * 16);
      } else {
        *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
      }
    }
  }

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  bool any = false;

  for (int t = t0; t < t1; ++t) {
    if (tid < kTile) {
      const int pos = t * kTile + tid;
      int j = -1;
      if (pos < n_row) j = p.idx[static_cast<size_t>(r) * p.width + pos];
      const bool ok = j >= 0 && j < p.cache_rows;
      s_idx[tid] = ok ? j : -1;
      const uint32_t bal = __ballot_sync(0xffffffffu, ok);
      if (lane == 0) s_mask[warp] = bal;
    }
    __syncthreads();
    const unsigned char* cache = static_cast<const unsigned char*>(p.cache);
    if (tid < ns * CPR) {
      for (int key = tid / CPR; key < kTile; key += ns) {
        const int j = s_idx[key];
        void* dst = s_keys + key * SK + ROFF + pc * 16;
        if (j >= 0) {
          CpAsync16(dst, cache + static_cast<size_t>(j) * (CPR * 16) + pc * 16);
        } else {
          *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
      }
    }
    CpAsyncWait();
    __syncthreads();
    if (FP8) {
      ExpandRows<D>(s_keys, kTile, warp, nthr >> 5, lane);
      if (t == t0) ExpandRows<D>(s_q, HB, warp, nthr >> 5, lane);
      __syncthreads();
    }

    uint32_t mask[4];
    int cnt = 0;
    float acc[NBK][4];
    float m_t[2], l_t[2];
    uint32_t pf[8][4];
    if (mma) {
      for (int i = 0; i < 4; ++i) {
        mask[i] = s_mask[i];
        cnt += __popc(mask[i]);
      }
      for (int b = 0; b < NBK; ++b)
        for (int i = 0; i < 4; ++i) acc[b][i] = 0.f;
      for (int ks = 0; ks < KSTEPS; ++ks) {
        uint32_t a[4], bf[NBK / 2][4];
        LdMatrix4(a, qa + 32 * ks);
        for (int j = 0; j < NBK / 2; ++j) LdMatrix4(bf[j], kb + 16 * j * SK + 32 * ks);
        for (int j = 0; j < NBK / 2; ++j) {
          const uint32_t b0[2] = {bf[j][0], bf[j][1]}, b1[2] = {bf[j][2], bf[j][3]};
          Mma16(acc[2 * j], a, b0);
          Mma16(acc[2 * j + 1], a, b1);
        }
      }
      for (int rs = 0; rs < 2; ++rs) {
        float x[2 * NBK];
        for (int b = 0; b < NBK; ++b)
          for (int e = 0; e < 2; ++e) {
            const int k = 8 * (kblk0 + b) + 2 * t4 + e;
            x[2 * b + e] = ((mask[k >> 5] >> (k & 31)) & 1u) ? acc[b][2 * rs + e] : -INFINITY;
          }
        const float own = RowMaxOwn(x);
        if (SPLIT == 4) {
          s_x[(warp * 32 + lane) * 2 + rs] = own;
        } else {
          m_t[rs] = own;
        }
      }
    }
    if (SPLIT == 4) __syncthreads();
    if (mma) {
      for (int rs = 0; rs < 2; ++rs) {
        if (SPLIT == 4) {
          const float sib[4] = {s_x[lane * 2 + rs], s_x[(32 + lane) * 2 + rs],
                                s_x[(64 + lane) * 2 + rs], s_x[(96 + lane) * 2 + rs]};
          m_t[rs] = sparse_mla_rowmax_w4_finish(sib);
        }
      }
    }
    if (SPLIT == 4) __syncthreads();
    if (mma) {
      for (int rs = 0; rs < 2; ++rs) {
        const float nms = __fmul_rn(m_t[rs], p.nc);
        float x[2 * NBK];
        for (int b = 0; b < NBK; ++b)
          for (int e = 0; e < 2; ++e) {
            const int k = 8 * (kblk0 + b) + 2 * t4 + e;
            x[2 * b + e] = ((mask[k >> 5] >> (k & 31)) & 1u)
                               ? Ex2(__fmaf_rn(acc[b][2 * rs + e], p.c, nms))
                               : 0.f;
          }
        for (int b = 0; b < NBK; ++b) {
          const uint32_t bits = Bf16Bits(x[2 * b]) | (Bf16Bits(x[2 * b + 1]) << 16);
          const int gb = kblk0 + b;
          if (SPLIT == 4) {
            s_p[((gb >> 1) * 32 + lane) * 4 + rs + 2 * (gb & 1)] = bits;
          } else {
            pf[gb >> 1][rs + 2 * (gb & 1)] = bits;
          }
        }
        const float own = RowSumOwn(x);
        if (SPLIT == 4) {
          s_x[(warp * 32 + lane) * 2 + rs] = own;
        } else {
          l_t[rs] = own;
        }
      }
    }
    if (SPLIT == 4) __syncthreads();
    if (mma) {
      if (SPLIT == 4) {
        for (int rs = 0; rs < 2; ++rs) {
          const float sib[4] = {s_x[lane * 2 + rs], s_x[(32 + lane) * 2 + rs],
                                s_x[(64 + lane) * 2 + rs], s_x[(96 + lane) * 2 + rs]};
          l_t[rs] = sparse_mla_rowsum_w4_finish(sib);
        }
        for (int ks = 0; ks < 8; ++ks) {
          const uint4 v = *reinterpret_cast<const uint4*>(s_p + (ks * 32 + lane) * 4);
          pf[ks][0] = v.x;
          pf[ks][1] = v.y;
          pf[ks][2] = v.z;
          pf[ks][3] = v.w;
        }
      }
      float a1[2], a2[2];
      if (fused && cnt > 0) {
        for (int rs = 0; rs < 2; ++rs) {
          const float mn = fmaxf(m_run[rs], m_t[rs]);
          const float nms = __fmul_rn(mn, p.nc);
          a1[rs] = Ex2(__fadd_rn(__fmul_rn(m_run[rs], p.c), nms));
          a2[rs] = Ex2(__fadd_rn(__fmul_rn(m_t[rs], p.c), nms));
          l_run[rs] = __fmaf_rn(l_t[rs], a2[rs], __fmul_rn(l_run[rs], a1[rs]));
          m_run[rs] = mn;
        }
      }
      float* wrow[2];
      for (int rs = 0; rs < 2; ++rs)
        wrow[rs] = p.ws + (static_cast<size_t>(r) * H + hrow[rs]) * (fused ? 1 : p.tiles) * kWsRow +
                   (fused ? 0 : static_cast<size_t>(t) * kWsRow);
      if (cnt > 0) {
        for (int pass = 0; pass < kLatent / (PVP * SPLIT); ++pass) {
          const int dbase = (SPLIT == 4 ? 128 * warp : 0) + pass * PVP;
          float o[NB][4];
          for (int nb = 0; nb < NB; ++nb)
            for (int i = 0; i < 4; ++i) o[nb][i] = 0.f;
          for (int ks = 0; ks < 8; ++ks) {
            const unsigned char* vb = vbase + 16 * ks * SK + 2 * dbase;
            uint32_t bf[NB / 2][4];
            for (int j = 0; j < NB / 2; ++j) LdMatrix4T(bf[j], vb + 32 * j);
            for (int j = 0; j < NB / 2; ++j) {
              const uint32_t b0[2] = {bf[j][0], bf[j][1]}, b1[2] = {bf[j][2], bf[j][3]};
              Mma16(o[2 * j], pf[ks], b0);
              Mma16(o[2 * j + 1], pf[ks], b1);
            }
          }
          for (int nb = 0; nb < NB; ++nb) {
            const int d0 = dbase + 8 * nb + 2 * t4;
            for (int rs = 0; rs < 2; ++rs) {
              if (!hok[rs]) continue;
              float2 v = make_float2(o[nb][2 * rs], o[nb][2 * rs + 1]);
              float2* dst = reinterpret_cast<float2*>(wrow[rs] + d0);
              if (fused) {
                const float2 prev = any ? *dst : make_float2(0.f, 0.f);
                v.x = __fmaf_rn(v.x, a2[rs], __fmul_rn(prev.x, a1[rs]));
                v.y = __fmaf_rn(v.y, a2[rs], __fmul_rn(prev.y, a1[rs]));
              }
              *dst = v;
            }
          }
        }
      }
      if (!fused && t4 == 0 && (SPLIT == 1 || warp == 0)) {
        for (int rs = 0; rs < 2; ++rs) {
          if (!hok[rs]) continue;
          wrow[rs][kLatent] = m_t[rs];
          wrow[rs][kLatent + 1] = l_t[rs];
          wrow[rs][kLatent + 2] = __int_as_float(cnt);
        }
      }
      any = any || cnt > 0;
    }
    __syncthreads();
  }

  if (fused && mma) {
    const int nb0 = SPLIT == 4 ? 16 * warp : 0, nb1 = SPLIT == 4 ? nb0 + 16 : kLatent / 8;
    for (int rs = 0; rs < 2; ++rs) {
      if (!hok[rs]) continue;
      const size_t rh = static_cast<size_t>(r) * H + hrow[rs];
      const float inv = __fdiv_rn(1.f, l_run[rs]);
      for (int nb = nb0; nb < nb1; ++nb) {
        const int d0 = 8 * nb + 2 * t4;
        uint32_t bits = 0;
        if (any) {
          const float2 o = *reinterpret_cast<const float2*>(p.ws + rh * kWsRow + d0);
          bits = Bf16Bits(__fadd_rn(__fmul_rn(__fmul_rn(o.x, inv), p.s2), 0.f)) |
                 (Bf16Bits(__fadd_rn(__fmul_rn(__fmul_rn(o.y, inv), p.s2), 0.f)) << 16);
        }
        *reinterpret_cast<uint32_t*>(p.out + rh * kLatent + d0) = bits;
      }
    }
  }
}

// One CTA per (row, head): folds the row's partial rows in tile order and writes the output.
__global__ void FixupKernel(const Params p) {
  const size_t rh = blockIdx.x;
  const int d = 4 * threadIdx.x;
  const float* base = p.ws + rh * p.tiles * kWsRow;
  float m = -INFINITY, l = 0.f;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  bool any = false;
  for (int tb = 0; tb < p.tiles; tb += kFixupBatch) {
    float4 bo[kFixupBatch], sc[kFixupBatch];
    for (int k = 0; k < kFixupBatch; ++k) {
      if (tb + k < p.tiles) {
        const float* row = base + static_cast<size_t>(tb + k) * kWsRow;
        sc[k] = *reinterpret_cast<const float4*>(row + kLatent);
        bo[k] = *reinterpret_cast<const float4*>(row + d);
      }
    }
    for (int k = 0; k < kFixupBatch; ++k) {
      if (tb + k >= p.tiles || __float_as_int(sc[k].z) <= 0) continue;
      const float bm = sc[k].x, bl = sc[k].y;
      const float mn = fmaxf(m, bm);
      const float nms = __fmul_rn(mn, p.nc);
      const float a1 = Ex2(__fadd_rn(__fmul_rn(m, p.c), nms));
      const float a2 = Ex2(__fadd_rn(__fmul_rn(bm, p.c), nms));
      l = __fmaf_rn(bl, a2, __fmul_rn(l, a1));
      o[0] = __fmaf_rn(bo[k].x, a2, __fmul_rn(o[0], a1));
      o[1] = __fmaf_rn(bo[k].y, a2, __fmul_rn(o[1], a1));
      o[2] = __fmaf_rn(bo[k].z, a2, __fmul_rn(o[2], a1));
      o[3] = __fmaf_rn(bo[k].w, a2, __fmul_rn(o[3], a1));
      m = mn;
      any = true;
    }
  }
  uint64_t bits = 0;
  if (any) {
    const float inv = __fdiv_rn(1.f, l);
    for (int i = 0; i < 4; ++i)
      bits |= static_cast<uint64_t>(Bf16Bits(__fadd_rn(__fmul_rn(__fmul_rn(o[i], inv), p.s2), 0.f)))
              << (16 * i);
  }
  *reinterpret_cast<uint64_t*>(p.out + rh * kLatent + d) = bits;
}

__global__ void Ex2Kernel(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = Ex2(x[i]);
}

template <int D, bool FP8, int PVP, int SPLIT>
cudaError_t LaunchUnits(const Params& p, int gather_warps, cudaStream_t stream) {
  auto kernel = UnitKernel<D, FP8, PVP, SPLIT>;
  const size_t smem = SmemBytes<D>(p.head_block);
  cudaError_t st = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem));
  if (st != cudaSuccess) return st;
  const int tpu = p.tiles_per_unit == 0 ? p.tiles : p.tiles_per_unit;
  const int64_t units = static_cast<int64_t>(p.rows) *
                        ((p.heads + p.head_block - 1) / p.head_block) * ((p.tiles + tpu - 1) / tpu);
  if (units <= 0 || units > 0x7fffffffll) return cudaErrorInvalidValue;
  kernel<<<static_cast<unsigned>(units), 32 * gather_warps, smem, stream>>>(p);
  return cudaGetLastError();
}

template <int D, bool FP8>
cudaError_t LaunchLayout(const Params& p, int gather_warps, int pv_pass, int mma_split,
                         cudaStream_t stream) {
  if (pv_pass == 128 && mma_split == 1) return LaunchUnits<D, FP8, 128, 1>(p, gather_warps, stream);
  if (pv_pass == 64 && mma_split == 1) return LaunchUnits<D, FP8, 64, 1>(p, gather_warps, stream);
  if (pv_pass == 128 && mma_split == 4) return LaunchUnits<D, FP8, 128, 4>(p, gather_warps, stream);
  if (pv_pass == 64 && mma_split == 4) return LaunchUnits<D, FP8, 64, 4>(p, gather_warps, stream);
  return cudaErrorInvalidValue;
}

// Partial rows: one per (row, head, tile), or per (row, head) when the unit kernel folds.
inline int64_t WorkspaceFloats(int rows, int heads, int tiles, int tiles_per_unit) {
  return static_cast<int64_t>(rows) * heads * (tiles_per_unit == 0 ? 1 : tiles) * kWsRow;
}

inline cudaError_t Launch(const Params& p, int rope, bool fp8, int gather_warps, int pv_pass,
                          int mma_split, cudaStream_t stream) {
  if (p.rows <= 0 || p.tiles <= 0) return cudaSuccess;
  cudaError_t st = cudaErrorInvalidValue;
  if (rope == 64 && !fp8)
    st = LaunchLayout<576, false>(p, gather_warps, pv_pass, mma_split, stream);
  if (rope == 64 && fp8) st = LaunchLayout<576, true>(p, gather_warps, pv_pass, mma_split, stream);
  if (rope == 0 && !fp8) st = LaunchLayout<512, false>(p, gather_warps, pv_pass, mma_split, stream);
  if (rope == 0 && fp8) st = LaunchLayout<512, true>(p, gather_warps, pv_pass, mma_split, stream);
  if (st != cudaSuccess || p.tiles_per_unit == 0) return st;
  FixupKernel<<<static_cast<unsigned>(p.rows) * p.heads, kFixupThreads, 0, stream>>>(p);
  return cudaGetLastError();
}

inline cudaError_t LaunchEx2(const float* x, float* y, int n, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  Ex2Kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
  return cudaGetLastError();
}

}  // namespace bi_sparse
}  // namespace mla
}  // namespace flashinfer

#endif  // FLASHINFER_MLA_BATCH_INVARIANT_SPARSE_CUH_
