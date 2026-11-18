// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// High-throughput scored-row path for large-T QSA indexing (SM107).
//
// Two structural changes over the baseline W7 route:
//   * the MMA M dimension packs TWO index heads per 16-row operand, so a CTA
//     owns only 8 tokens while every m16n8k32 is fully utilised.  The shared
//     score slab shrinks to 64 KiB and two CTAs become resident per SM, so
//     the MMA phase of one CTA overlaps the select/store phase of the other.
//   * selection uses monotone float-bit keys with a fused first histogram.
//     An absolute-product MMA bounds each dot: sums at most 64 are exact on
//     the E4M3 product lattice (2^-18). Larger sums are recomputed with
//     sequential round-to-nearest fp32 before the original radix selection.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>

#include <cstdint>

namespace w7n {

#define W7_QROW 36    // padded query rows stay conflict free
#define W7_KROW 32    // compact key rows use an XOR chunk swizzle
#define W7_GM 8       // tokens owned by one CTA
#define W7_NTILE 128  // key rows per pipeline stage
#define W7_NTHR 256
#define W7_NWARP 8
#define W7_VCAP 2056
#define W7_SCAP (W7_GM * W7_VCAP)
#define W7_REGW (W7_NST * W7_NTILE * W7_KROW)  // scratch region words
#define W7_NST 2
#define W7_KTOP 512
#define W7_PW 2052
#define W7_RW 513
#define W7_SLOW 2052  // pos+1 >= 2052  <=>  vis > 512
#define W7_FKTHR 256
#define W7_HROW 260  // 256 bins + 4 word skew between token rows
#define W7_SHISTW (W7_GM * W7_HROW)

// The per-warp 256 bucket histogram keeps its natural layout: adjacent bins --
// the ones a warp's keys collide on -- then land in different banks, and the
// eight bins a lane owns come back in two 128 bit loads.

__device__ __forceinline__ void w7cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void w7commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void w7wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ uint4 w7ld4(uint32_t p) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(p));
  return v;
}
__device__ __forceinline__ void w7st(int4* p, const int4& v) {
  asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void w7st2(int4* p, const int4& a, const int4& b) {
  asm volatile("st.global.cs.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"l"(p), "r"(a.x),
               "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w)
               : "memory");
}
__device__ __forceinline__ int4 w7rowval(int col4, int cnt) {
  const int c = col4 << 2;
  int4 v;
  v.x = (c < cnt) ? c : -1;
  v.y = (c + 1 < cnt) ? c + 1 : -1;
  v.z = (c + 2 < cnt) ? c + 2 : -1;
  v.w = (col4 == W7_RW - 1) ? cnt : ((c + 3 < cnt) ? c + 3 : -1);
  return v;
}

__device__ __forceinline__ float w7relu(float x) {
  float y;
  asm("max.NaN.f32 %0, %1, 0f00000000;" : "=f"(y) : "f"(x));
  return y;
}

// mma.sync with e4m3 operands is emulated on this target (ptxas expands it into
// HMMA plus a full F2FP conversion of *both* operands on every use).  Convert
// once into f16 registers and drive the native f16 HMMA shape directly.
#define W7MMA(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)           \
  asm volatile(                                                 \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "      \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                  \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// Scores are nonnegative, so their float bits are monotone. Add one to keep
// finite zero above NaN/padding key zero; the radix below uses an offset bin
// around the traffic's useful score range and falls back at either edge.
__device__ __forceinline__ uint32_t w7key(float s) {
  const uint32_t raw = __float_as_uint(s);
  uint32_t k = raw + 1u;
  if (isnan(s)) k = 0u;
  return k;
}

__device__ __forceinline__ void w7key2(float s0, float s1, uint32_t& k0, uint32_t& k1) {
  const uint32_t raw0 = __float_as_uint(s0);
  const uint32_t raw1 = __float_as_uint(s1);
  k0 = raw0 + 1u;
  k1 = raw1 + 1u;
  asm volatile(
      "{ .reg .pred n0, n1;\n"
      "  setp.nan.f32 n0, %2, %2;\n"
      "  setp.nan.f32 n1, %3, %3;\n"
      "  @n0 mov.u32 %0, 0;\n"
      "  @n1 mov.u32 %1, 0;\n"
      "}\n"
      : "+r"(k0), "+r"(k1)
      : "f"(s0), "f"(s1));
}

__device__ __forceinline__ void w7add4x2(float a0, float a1, float b0, float b1, float c0, float c1,
                                         float d0, float d1, float& z0, float& z1) {
  asm volatile(
      "{ .reg .b64 x, y;\n"
      "  mov.b64 x, {%2, %3};\n"
      "  mov.b64 y, {%4, %5};\n"
      "  add.rn.f32x2 x, x, y;\n"
      "  mov.b64 y, {%6, %7};\n"
      "  add.rn.f32x2 x, x, y;\n"
      "  mov.b64 y, {%8, %9};\n"
      "  add.rn.f32x2 x, x, y;\n"
      "  mov.b64 {%0, %1}, x;\n"
      "}\n"
      : "=f"(z0), "=f"(z1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1), "f"(d0), "f"(d1));
}

__device__ __forceinline__ uint32_t w7bfe(uint32_t x, int sh) {
  uint32_t r;
  asm("bfe.u32 %0, %1, %2, 8;\n" : "=r"(r) : "r"(x), "r"(sh));
  return r;
}

// The scoring-fused digit uses 2^18-wide monotone float-bit intervals above
// score 2.0. Both saturated edge bins retain an exact full-radix fallback.
__device__ __forceinline__ uint32_t w7firstbin(uint32_t x) {
  const uint32_t q = min(max(x, 0x40000000u), 0x41FE0000u);
  return (q >> 17) & 255u;
}

__device__ __forceinline__ uint32_t w7cvt(uint16_t v) {
  uint32_t d;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(d) : "h"(v));
  return d;
}

template <int WIDTH>
__device__ __forceinline__ uint32_t w7sort_width(uint32_t v, int lane) {
#pragma unroll
  for (int stride = WIDTH >> 1; stride > 0; stride >>= 1) {
    const uint32_t other = __shfl_xor_sync(0xffffffffu, v, stride);
    const bool take_hi = (((lane & WIDTH) == 0) == ((lane & stride) == 0));
    v = take_hi ? max(v, other) : min(v, other);
  }
  return v;
}

__device__ __forceinline__ void w7sort64(uint32_t& a, uint32_t& b, int lane) {
  a = w7sort_width<2>(a, lane);
  b = w7sort_width<2>(b, lane);
  a = w7sort_width<4>(a, lane);
  b = w7sort_width<4>(b, lane);
  a = w7sort_width<8>(a, lane);
  b = w7sort_width<8>(b, lane);
  a = w7sort_width<16>(a, lane);
  b = w7sort_width<16>(b, lane);
  a = w7sort_width<32>(a, lane);
  b = w7sort_width<32>(b, lane);

  const uint32_t br = __shfl_sync(0xffffffffu, b, 31 - lane);
  b = min(a, br);
  a = max(a, br);
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    const uint32_t oa = __shfl_xor_sync(0xffffffffu, a, stride);
    const uint32_t ob = __shfl_xor_sync(0xffffffffu, b, stride);
    const bool high = (lane & stride) == 0;
    a = high ? max(a, oa) : min(a, oa);
    b = high ? max(b, ob) : min(b, ob);
  }
}

__device__ __forceinline__ int w7_find_req(const int* __restrict__ cu, int B, int t, int lane) {
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
__device__ __forceinline__ int w7_find_req_base(const int* __restrict__ cu, int B, int t, int lane,
                                                int& base) {
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

// -------------------------------------------------------- identity rows ----
// 8208 B rows are 16B but not 128B aligned, so CTAs own flat 128B aligned
// int4 ranges of the whole output and derive (row, col) from the flat index.
template <int PT, bool SINGLE_REQ>
__global__ __launch_bounds__(W7_FKTHR) void w7_fill(const int* __restrict__ cu, int B, int T,
                                                    int ntot, int4* __restrict__ out4) {
  const int tid = threadIdx.x;
  const int g0 = blockIdx.x * (W7_FKTHR * PT);
  if (g0 >= ntot) return;

  const int row0 = g0 / W7_RW;
  __shared__ int s_cnt[8];
  int cbase = row0 + 1;
  bool uni = true;
  if (!SINGLE_REQ) {
    const int warp = tid >> 5, lane = tid & 31;
    const int rr = row0 + warp;
    int bse = 0;
    w7_find_req_base(cu, B, rr, lane, bse);
    if (lane == 0) s_cnt[warp] = rr - bse + 1;
    __syncthreads();
    cbase = s_cnt[0];
    uni = (s_cnt[7] == cbase + 7);
  }

  int row = row0;
  int col4 = g0 + tid - row0 * W7_RW;
  if (col4 >= W7_RW) {
    row += col4 / W7_RW;
    col4 -= (col4 / W7_RW) * W7_RW;
  }
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int g = g0 + tid + k * W7_FKTHR;
    if (PT > 1 && g >= ntot) break;
    const int cnt = SINGLE_REQ ? (row + 1) : (uni ? (row - row0 + cbase) : s_cnt[row - row0]);
    if (cnt < W7_SLOW) {
      const int c = col4 * 4;
      int4 v;
      v.x = (c < cnt) ? c : -1;
      v.y = (c + 1 < cnt) ? c + 1 : -1;
      v.z = (c + 2 < cnt) ? c + 2 : -1;
      v.w = (col4 == W7_RW - 1) ? cnt : ((c + 3 < cnt) ? c + 3 : -1);
      w7st(out4 + g, v);
    }
    col4 += W7_FKTHR;
    if (col4 >= W7_RW) {
      col4 -= W7_RW;
      ++row;
    }
  }
}

// Only the first 2051 rows of a request are score independent, so for low B
// address them request-locally instead of probing the long scored suffix.
template <int PT>
__device__ __forceinline__ void w7_fill_prefix_body(const int* __restrict__ cu, int B, int req,
                                                    int chunk, int4* __restrict__ out4) {
  if (req >= B) return;
  const int begin = cu[req];
  const int rows = min(cu[req + 1] - begin, W7_SLOW - 1);
  const int items = rows * W7_RW;
  const int g0 = chunk * W7_FKTHR * PT;
  if (g0 >= items) return;
  constexpr int pair_threads = W7_FKTHR / 2;
  const int tid = (int)threadIdx.x;
  if (tid >= pair_threads) return;

  const int n = min(W7_FKTHR * PT, items - g0);
  const int lead = begin & 1;  // rows are 16 B mod 32 B aligned
  int4* const dst = out4 + (size_t)begin * W7_RW;
  if (tid == 0 && lead) w7st(dst + g0, w7rowval(g0 % W7_RW, g0 / W7_RW + 1));

  const int npair = (n - lead) >> 1;
  const int first = g0 + lead + (tid << 1);
  int row = first / W7_RW;
  int col4 = first - row * W7_RW;
#pragma unroll
  for (int k = 0; k < PT; ++k) {
    const int p = tid + k * pair_threads;
    if (p >= npair) break;
    const int g = first + k * W7_FKTHR;
    const int4 a = w7rowval(col4, row + 1);
    const int4 b = (col4 + 1 < W7_RW) ? w7rowval(col4 + 1, row + 1) : w7rowval(0, row + 2);
    w7st2(dst + g, a, b);
    col4 += W7_FKTHR;
    if (col4 >= W7_RW) {
      col4 -= W7_RW;
      ++row;
    }
  }
  if (tid == 0 && ((n - lead) & 1)) {
    const int g = g0 + n - 1;
    w7st(dst + g, w7rowval(g % W7_RW, g / W7_RW + 1));
  }
}

// ------------------------------------------------------------ scored rows ---
template <bool B2>
__global__ __launch_bounds__(W7_NTHR, 2) void w7_score(const uint8_t* __restrict__ qi,
                                                       const uint8_t* __restrict__ ckc,
                                                       const int* __restrict__ cu, int B, int T,
                                                       const int* __restrict__ cpb,
                                                       int* __restrict__ out, int ntile, int fgx) {
  extern __shared__ uint32_t sm[];
  int blk = (int)blockIdx.x;
  int score_grid = ntile;
  int split = -1, tiles0 = 0, tiles1 = 0;
  if constexpr (B2) {
    split = cu[1];
    const int n0 = max(split - (W7_SLOW - 1), 0);
    const int n1 = max(T - split - (W7_SLOW - 1), 0);
    tiles0 = (n0 + W7_GM - 1) / W7_GM;
    tiles1 = (n1 + W7_GM - 1) / W7_GM;
    score_grid = tiles0 + tiles1;
  }
  if (fgx > 0) {
    if (blk >= score_grid) {
      const int f = blk - score_grid;
      if constexpr (B2) {
        constexpr int chunk_items = W7_FKTHR * 16;
        const int p0 = (min(split, W7_SLOW - 1) * W7_RW + chunk_items - 1) / chunk_items;
        const int p1 = (min(T - split, W7_SLOW - 1) * W7_RW + chunk_items - 1) / chunk_items;
        if (f < p0 + p1) {
          const int req = (f >= p0);
          const int chunk = req ? (f - p0) : f;
          w7_fill_prefix_body<16>(cu, B, req, chunk, reinterpret_cast<int4*>(out));
        }
      } else {
        w7_fill_prefix_body<16>(cu, B, f / fgx, f - (f / fgx) * fgx, reinterpret_cast<int4*>(out));
      }
      return;
    }
  }
  // The q tile, the cp.async key stages and the histogram / emit staging
  // buffers are live in disjoint phases, so they share one scratch region.
  int* meta = (int*)sm;               // 32
  uint32_t* sc = sm + 32;             // SCAP radix keys
  uint32_t* reg = sm + 32 + W7_SCAP;  // W7_REGW
  uint32_t* q_s = reg;                // 32 * SROW
  uint32_t* ck_s = reg;               // NST*NTILE*SROW
  int* hist = (int*)reg;              // NWARP * 256
  // Live across the whole scoring phase, so it cannot share the stage region:
  // the scorer files every key it produces into its token's 256-bin histogram,
  // which removes the radix's first full pass over the score slab.
  uint32_t* shist = reg + W7_REGW;  // GM * HROW

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int t0, tile_req = -1, tile_base = 0, tile_end = T;
  if constexpr (B2) {
    if (blk < tiles1) {
      tile_req = 1;
      tile_base = split;
      tile_end = T;
      t0 = split + (W7_SLOW - 1) + (tiles1 - 1 - blk) * W7_GM;
    } else {
      tile_req = 0;
      tile_base = 0;
      tile_end = split;
      t0 = (W7_SLOW - 1) + (tiles0 - 1 - (blk - tiles1)) * W7_GM;
    }
  } else {
    t0 = (ntile - 1 - blk) * W7_GM;
  }
  if (t0 >= T) return;

  uint32_t todo = 0;
  if constexpr (B2) {
    const int valid = min(W7_GM, tile_end - t0);
    todo = (1u << valid) - 1u;
  } else {
    {  // one warp per token: eight independent request searches
      const int j = warp;
      const int t = t0 + j;
      int r = -1, pos = -1;
      if (t < T) {
        if (B == 1) {
          r = 0;
          pos = t;
        } else {
          r = w7_find_req(cu, B, t, lane);
          pos = t - cu[r];
        }
      }
      if (lane == 0) {
        meta[j] = r;
        meta[W7_GM + j] = pos;
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < W7_GM; ++j)
      if (meta[W7_GM + j] + 1 >= W7_SLOW) todo |= (1u << j);
  }
  if (todo == 0) return;

  {  // q -> smem; row = group*16 + half*8 + token, head = 2*group + half
    const int row = tid >> 3, jj = tid & 7;
    const int tok = row & 7;
    const int head = ((row >> 4) << 1) | ((row >> 3) & 1);
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (t0 + tok < tile_end)
      val = *reinterpret_cast<const uint4*>(qi + ((size_t)(t0 + tok) * 4 + head) * 128 + jj * 16);
    *reinterpret_cast<uint4*>(q_s + row * W7_QROW + jj * 4) = val;
  }
  __syncthreads();

  // A fragments in f16, resident for the whole kernel: 2 head-pairs x 8 k-steps
  uint32_t af[2][8][4];
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const uint32_t* p0 = q_s + (size_t)(g * 16 + (lane >> 2)) * W7_QROW + ((lane & 3) << 3);
    const uint32_t* p1 = p0 + 8 * W7_QROW;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint32_t w0 = p0[kk], w1 = p1[kk];
      af[g][kk][0] = w7cvt((uint16_t)(w0 & 0xffffu));
      af[g][kk][1] = w7cvt((uint16_t)(w1 & 0xffffu));
      af[g][kk][2] = w7cvt((uint16_t)(w0 >> 16));
      af[g][kk][3] = w7cvt((uint16_t)(w1 >> 16));
    }
  }
  __syncthreads();  // q_s is dead; the region becomes the key stages

  const int stw = W7_NTILE * W7_KROW;
  const uint32_t ckaddr = (uint32_t)__cvta_generic_to_shared(ck_s);

  while (todo) {
    const int j0 = __ffs(todo) - 1;
    int r0;
    if constexpr (B2)
      r0 = tile_req;
    else
      r0 = meta[j0];
    uint32_t sub = 0;
    if constexpr (B2) {
      sub = todo;
    } else {
#pragma unroll
      for (int j = 0; j < W7_GM; ++j)
        if (((todo >> j) & 1) && meta[j] == r0) sub |= (1u << j);
    }
    const int jhi = 31 - __clz(sub);
    int pos_hi;
    if constexpr (B2)
      pos_hi = t0 + jhi - tile_base;
    else
      pos_hi = meta[W7_GM + jhi];
    const int V = (pos_hi + 1) >> 2;  // max vis of the subset
    const size_t ckb = (size_t)cpb[r0] * 64;
    const int Vs = (V + 7) & ~7;
    const int Vst = Vs + 8;  // +32B so adjacent token slabs shift 8 banks
    int gb = W7_GM;
    if (Vst > W7_VCAP) gb = W7_SCAP / Vst;
    if (gb < 1) break;  // beyond supported extent
    const int ntiles = (V + W7_NTILE - 1) / W7_NTILE;

    for (int bst = 0; bst < W7_GM; bst += gb) {
      const uint32_t bmask = sub & (((1u << gb) - 1u) << bst);
      if (!bmask) continue;
      if constexpr (B2) {
        if (bst != 0) __syncthreads();
      } else {
        __syncthreads();
      }

#ifdef W7_DBLMMA
      for (int rep = 0; rep < 2; ++rep) {
#endif
        // lane -> (row, chunk) as (lane & 7, lane >> 3): the eight lanes of a
        // half-wavefront then carry eight distinct swizzled chunks, covering all
        // 32 banks exactly once.  The (lane >> 2, lane & 3) map put four rows on
        // the same 16 banks, costing a 2x shared-store conflict on every stage.
        const int bb_issue = warp * 8 + (lane & 7);
        const int sw_issue = bb_issue & 7;
        const int jj_issue = lane >> 3;
        const uint32_t so00 = (uint32_t)((bb_issue * W7_KROW + (jj_issue ^ sw_issue) * 4) * 4);
        const uint32_t so01 =
            (uint32_t)((bb_issue * W7_KROW + ((jj_issue + 4) ^ sw_issue) * 4) * 4);
        const uint32_t so10 =
            (uint32_t)(((bb_issue + 64) * W7_KROW + (jj_issue ^ sw_issue) * 4) * 4);
        const uint32_t so11 =
            (uint32_t)(((bb_issue + 64) * W7_KROW + ((jj_issue + 4) ^ sw_issue) * 4) * 4);
        const uint8_t* cur0 = ckc + (ckb + bb_issue) * 128 + jj_issue * 16;
        const uint8_t* cur2 = cur0 + 64 * 128;
        const int tok_lane = lane >> 2;
        const bool tok_live = ((bmask >> tok_lane) & 1) != 0;
        uint32_t* const score_row = sc + (tok_live ? (tok_lane - bst) * Vst : 0);
        uint32_t* const shist_row = shist + (tok_live ? (tok_lane - bst) * W7_HROW : 0);
        const int colb = warp * 8 + (lane >> 2);
        const int sw = colb & 7;
        const uint32_t rw_base = ckaddr + colb * W7_KROW * 4;
        int vcur = bb_issue;
        uint32_t issue_dst = ckaddr;

#define W7_ISSUE_STAGE()             \
  do {                               \
    const uint32_t dst = issue_dst;  \
    if (vcur < V) {                  \
      w7cp16(dst + so00, cur0);      \
      w7cp16(dst + so01, cur0 + 64); \
    }                                \
    if (vcur + 64 < V) {             \
      w7cp16(dst + so10, cur2);      \
      w7cp16(dst + so11, cur2 + 64); \
    }                                \
    cur0 += W7_NTILE * 128;          \
    cur2 += W7_NTILE * 128;          \
    vcur += W7_NTILE;                \
    issue_dst ^= 1u << 14;           \
    w7commit();                      \
  } while (0)
        static_assert(W7_NTILE * W7_KROW * 4 == (1 << 14), "stage stride changed");

#pragma unroll 1
        for (int s = 0; s < W7_NST - 1; ++s) {
          W7_ISSUE_STAGE();
        }
        // Clear the disjoint fused histogram while the first async key stage is
        // in flight. The barrier completes the clear; cp.async is waited below.
        for (int z = tid; z < W7_SHISTW; z += W7_NTHR) shist[z] = 0u;
        __syncthreads();

        uint32_t rwA = rw_base;
        int cA = warp * 8 + 2 * (lane & 3);
        for (int i = 0; i < ntiles; ++i) {
          // W7_ISSUE_STAGE predicates every copy by vcur<V; once the source
          // cursor reaches the tail it naturally commits an empty drain group.
          W7_ISSUE_STAGE();
          w7wait<W7_NST - 1>();
          __syncwarp();

          const uint32_t rwB = rwA + 64 * W7_KROW * 4;
          float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;  // half 0, heads 0/1
          float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;  // half 1, heads 0/1
          float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;  // half 0, heads 2/3
          float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;  // half 1, heads 2/3
          float za0 = 0.f, za1 = 0.f, za2 = 0.f, za3 = 0.f;
          float zb0 = 0.f, zb1 = 0.f, zb2 = 0.f, zb3 = 0.f;
          float zc0 = 0.f, zc1 = 0.f, zc2 = 0.f, zc3 = 0.f;
          float ze0 = 0.f, ze1 = 0.f, ze2 = 0.f, ze3 = 0.f;
#pragma unroll
          for (int kc = 0; kc < 2; ++kc) {
            const int kch = (((lane & 3) << 1) | kc) ^ sw;
            const uint4 va = w7ld4(rwA + (kch << 4));
            const uint4 vb = w7ld4(rwB + (kch << 4));
            const uint32_t wav[4] = {va.x, va.y, va.z, va.w};
            const uint32_t wbv[4] = {vb.x, vb.y, vb.z, vb.w};
            uint32_t ba0 = w7cvt((uint16_t)(wav[0] & 0xffffu));
            uint32_t ba1 = w7cvt((uint16_t)(wav[0] >> 16));
            uint32_t bb0 = w7cvt((uint16_t)(wbv[0] & 0xffffu));
            uint32_t bb1 = w7cvt((uint16_t)(wbv[0] >> 16));
#pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
              const int kk = kc * 4 + kq;
              // Interleave the next B-fragment conversions with the current
              // HMMA chain so F2FP latency overlaps tensor-pipe execution.
              W7MMA(a0, a1, a2, a3, af[0][kk][0], af[0][kk][1], af[0][kk][2], af[0][kk][3], ba0,
                    ba1);
              W7MMA(za0, za1, za2, za3, af[0][kk][0] & 0x7fff7fffu, af[0][kk][1] & 0x7fff7fffu,
                    af[0][kk][2] & 0x7fff7fffu, af[0][kk][3] & 0x7fff7fffu, ba0 & 0x7fff7fffu,
                    ba1 & 0x7fff7fffu);
              uint32_t nba0 = 0, nba1 = 0, nbb0 = 0, nbb1 = 0;
              if (kq + 1 < 4) {
                const uint32_t wa = wav[kq + 1];
                nba0 = w7cvt((uint16_t)(wa & 0xffffu));
                nba1 = w7cvt((uint16_t)(wa >> 16));
              }
              W7MMA(b0, b1, b2, b3, af[0][kk][0], af[0][kk][1], af[0][kk][2], af[0][kk][3], bb0,
                    bb1);
              W7MMA(zb0, zb1, zb2, zb3, af[0][kk][0] & 0x7fff7fffu, af[0][kk][1] & 0x7fff7fffu,
                    af[0][kk][2] & 0x7fff7fffu, af[0][kk][3] & 0x7fff7fffu, bb0 & 0x7fff7fffu,
                    bb1 & 0x7fff7fffu);
              if (kq + 1 < 4) {
                const uint32_t wb = wbv[kq + 1];
                nbb0 = w7cvt((uint16_t)(wb & 0xffffu));
                nbb1 = w7cvt((uint16_t)(wb >> 16));
              }
              W7MMA(c0, c1, c2, c3, af[1][kk][0], af[1][kk][1], af[1][kk][2], af[1][kk][3], ba0,
                    ba1);
              W7MMA(zc0, zc1, zc2, zc3, af[1][kk][0] & 0x7fff7fffu, af[1][kk][1] & 0x7fff7fffu,
                    af[1][kk][2] & 0x7fff7fffu, af[1][kk][3] & 0x7fff7fffu, ba0 & 0x7fff7fffu,
                    ba1 & 0x7fff7fffu);
              W7MMA(e0, e1, e2, e3, af[1][kk][0], af[1][kk][1], af[1][kk][2], af[1][kk][3], bb0,
                    bb1);
              W7MMA(ze0, ze1, ze2, ze3, af[1][kk][0] & 0x7fff7fffu, af[1][kk][1] & 0x7fff7fffu,
                    af[1][kk][2] & 0x7fff7fffu, af[1][kk][3] & 0x7fff7fffu, bb0 & 0x7fff7fffu,
                    bb1 & 0x7fff7fffu);
              ba0 = nba0;
              ba1 = nba1;
              bb0 = nbb0;
              bb1 = nbb1;
            }
          }
          const uint8_t* rq = qi + (size_t)(t0 + tok_lane) * 512;
          const uint8_t* rk = ckc + (ckb + cA) * 128;
          a0 = qsa_checked_dot(a0, za0, rq + 0, rk + 0, tok_live && cA + 0 < V);
          a1 = qsa_checked_dot(a1, za1, rq + 0, rk + 128, tok_live && cA + 1 < V);
          a2 = qsa_checked_dot(a2, za2, rq + 128, rk + 0, tok_live && cA + 0 < V);
          a3 = qsa_checked_dot(a3, za3, rq + 128, rk + 128, tok_live && cA + 1 < V);
          b0 = qsa_checked_dot(b0, zb0, rq + 0, rk + 8192, tok_live && cA + 64 < V);
          b1 = qsa_checked_dot(b1, zb1, rq + 0, rk + 8320, tok_live && cA + 65 < V);
          b2 = qsa_checked_dot(b2, zb2, rq + 128, rk + 8192, tok_live && cA + 64 < V);
          b3 = qsa_checked_dot(b3, zb3, rq + 128, rk + 8320, tok_live && cA + 65 < V);
          c0 = qsa_checked_dot(c0, zc0, rq + 256, rk + 0, tok_live && cA + 0 < V);
          c1 = qsa_checked_dot(c1, zc1, rq + 256, rk + 128, tok_live && cA + 1 < V);
          c2 = qsa_checked_dot(c2, zc2, rq + 384, rk + 0, tok_live && cA + 0 < V);
          c3 = qsa_checked_dot(c3, zc3, rq + 384, rk + 128, tok_live && cA + 1 < V);
          e0 = qsa_checked_dot(e0, ze0, rq + 256, rk + 8192, tok_live && cA + 64 < V);
          e1 = qsa_checked_dot(e1, ze1, rq + 256, rk + 8320, tok_live && cA + 65 < V);
          e2 = qsa_checked_dot(e2, ze2, rq + 384, rk + 8192, tok_live && cA + 64 < V);
          e3 = qsa_checked_dot(e3, ze3, rq + 384, rk + 8320, tok_live && cA + 65 < V);
          if (i + W7_NST < ntiles) __syncwarp();
          {
            // Every tile except the final partial tile has all 128 columns in
            // range.  Split that overwhelmingly common case so ptxas can remove
            // four per-result range predicates and collapse the max reduction.
            // `full` is warp-uniform (unlike cA, which includes lane columns).
            // Tokens in one eight-row tile differ by at most two visible
            // blocks. Treat the last shared tile as partial for every token so
            // each lane can write its own padding directly.
            const bool full = ((i + 1) * W7_NTILE) <= V - 2;
            if (tok_live && full) {
              float sA0, sA1, sB0, sB1;
              w7add4x2(w7relu(a0), w7relu(a1), w7relu(a2), w7relu(a3), w7relu(c0), w7relu(c1),
                       w7relu(c2), w7relu(c3), sA0, sA1);
              w7add4x2(w7relu(b0), w7relu(b1), w7relu(b2), w7relu(b3), w7relu(e0), w7relu(e1),
                       w7relu(e2), w7relu(e3), sB0, sB1);
              uint32_t kA0, kA1, kB0, kB1;
              w7key2(sA0, sA1, kA0, kA1);
              w7key2(sB0, sB1, kB0, kB1);
              *reinterpret_cast<uint2*>(score_row + cA) = make_uint2(kA0, kA1);
              *reinterpret_cast<uint2*>(score_row + cA + 64) = make_uint2(kB0, kB1);
              atomicAdd(shist_row + w7firstbin(kA0), 1u);
              atomicAdd(shist_row + w7firstbin(kA1), 1u);
              atomicAdd(shist_row + w7firstbin(kB0), 1u);
              atomicAdd(shist_row + w7firstbin(kB1), 1u);
            } else if (tok_live) {
              // row m and row m+8 of the A operand are the same token under the
              // two heads of a pair, so all four heads land in this thread.
              int tok_pos;
              if constexpr (B2)
                tok_pos = t0 + tok_lane - tile_base;
              else
                tok_pos = meta[W7_GM + tok_lane];
              const int tok_vis = (tok_pos + 1) >> 2;
              if (cA < Vs) {
                const uint32_t k0 =
                    (cA < tok_vis) ? w7key(w7relu(a0) + w7relu(a2) + w7relu(c0) + w7relu(c2)) : 0u;
                const uint32_t k1 = (cA + 1 < tok_vis)
                                        ? w7key(w7relu(a1) + w7relu(a3) + w7relu(c1) + w7relu(c3))
                                        : 0u;
                *reinterpret_cast<uint2*>(score_row + cA) = make_uint2(k0, k1);
                atomicAdd(shist_row + w7firstbin(k0), 1u);
                atomicAdd(shist_row + w7firstbin(k1), 1u);
              }
              const int cB = cA + 64;
              if (cB < Vs) {
                const uint32_t k0 =
                    (cB < tok_vis) ? w7key(w7relu(b0) + w7relu(b2) + w7relu(e0) + w7relu(e2)) : 0u;
                const uint32_t k1 = (cB + 1 < tok_vis)
                                        ? w7key(w7relu(b1) + w7relu(b3) + w7relu(e1) + w7relu(e3))
                                        : 0u;
                *reinterpret_cast<uint2*>(score_row + cB) = make_uint2(k0, k1);
                atomicAdd(shist_row + w7firstbin(k0), 1u);
                atomicAdd(shist_row + w7firstbin(k1), 1u);
              }
            }
          }
          cA += W7_NTILE;
          rwA ^= stw * 4;
        }
#undef W7_ISSUE_STAGE
#ifdef W7_DBLMMA
      }
#endif
      // Every score and its fused histogram update must land before selection.
      __syncthreads();

      // ------------------------------------------- exact top-512 per token
      {
        const int jrel = warp;
        const int j = bst + jrel;
        if (jrel < gb && ((bmask >> j) & 1)) {
          const int t = t0 + j;
          int pos;
          if constexpr (B2)
            pos = t - tile_base;
          else
            pos = meta[W7_GM + j];
          const int vis = (pos + 1) >> 2;
          uint32_t* ks = sc + (size_t)jrel * Vst;
          int* h = hist + warp * 256;
          int* compact = hist + W7_NWARP * (256 + 32) + warp * 64;

          const int nvec = Vs >> 2;

          uint32_t prefix = 0u, pmask = 0u;
          int k = W7_KTOP;
          bool exact = false;
#ifndef W7_NORADIX
#ifdef W7_DBLRADIX
          for (int rep = 0; rep < 2; ++rep) {
            prefix = 0u;
            pmask = 0u;
            k = W7_KTOP;
            exact = false;
#endif
            // The scorer already filed every key into its offset float-bit bin.
            bool allow_pre = true;
            // digits are anchored on the live key width, so every pass spends
            // all 256 buckets instead of the 2^(hib%8) a byte grid would leave.
            for (int sh = 17;;) {
              const bool first_digit = (pmask == 0u);
              const bool pre = first_digit && allow_pre;
              int* const hc = pre ? reinterpret_cast<int*>(shist + (size_t)jrel * W7_HROW) : h;
              if (!pre) {
                __syncwarp();
                {
                  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
                  reinterpret_cast<uint4*>(h)[lane] = z;
                  reinterpret_cast<uint4*>(h)[lane + 32] = z;
                }
                __syncwarp();
                if (pmask == 0u) {
#pragma unroll 2
                  for (int i = lane; i < nvec; i += 32) {
                    const uint4 f = *reinterpret_cast<const uint4*>(ks + 4 * i);
                    atomicAdd(&h[w7bfe(f.x, sh)], 1);
                    atomicAdd(&h[w7bfe(f.y, sh)], 1);
                    atomicAdd(&h[w7bfe(f.z, sh)], 1);
                    atomicAdd(&h[w7bfe(f.w, sh)], 1);
                  }
                } else {
#pragma unroll 2
                  for (int i = lane; i < nvec; i += 32) {
                    const uint4 f = *reinterpret_cast<const uint4*>(ks + 4 * i);
                    const uint32_t kv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                      if ((kv[e] & pmask) == prefix) atomicAdd(&h[w7bfe(kv[e], sh)], 1);
                    }
                  }
                }
                __syncwarp();
              }
              int hv[8];
              {
                const uint4 a = reinterpret_cast<const uint4*>(hc)[2 * lane];
                const uint4 b = reinterpret_cast<const uint4*>(hc)[2 * lane + 1];
                hv[0] = (int)a.x;
                hv[1] = (int)a.y;
                hv[2] = (int)a.z;
                hv[3] = (int)a.w;
                hv[4] = (int)b.x;
                hv[5] = (int)b.y;
                hv[6] = (int)b.z;
                hv[7] = (int)b.w;
              }
              int s = 0;
#pragma unroll
              for (int i = 0; i < 8; ++i) s += hv[i];
              int x = s;
#pragma unroll
              for (int off = 1; off < 32; off <<= 1) {
                const int y = __shfl_down_sync(0xffffffffu, x, off);
                if (lane + off < 32) x += y;
              }
              // highest bin whose suffix count still reaches k; resolved entirely
              // in registers so no lane serialises on dependent shared loads.
              int d = -1, above = 0, cntd = 0, acc = x - s;
#pragma unroll
              for (int i = 7; i >= 0; --i) {
                // acc < k makes the threshold bin unique across the warp: a
                // bin whose preceding count already exhausts the budget must
                // not record a candidate (its `above` can exceed the 9-bit
                // field and forge a higher-bin record under redux.max,
                // selecting a wrong threshold).
                if (d < 0 && acc < k && acc + hv[i] >= k) {
                  d = (lane << 3) | i;
                  above = acc;
                  cntd = hv[i];
                }
                acc += hv[i];
              }
              uint32_t rec =
                  (d < 0) ? 0u : ((uint32_t)d << 24) | ((uint32_t)above << 15) | (uint32_t)cntd;
              asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(rec) : "r"(rec));
              d = (int)(rec >> 24);
              above = (int)((rec >> 15) & 0x1ffu);
              cntd = (int)(rec & 0x7fffu);
              if (pre && (d == 0 || (d == 255 && cntd != k))) {
                // Either saturated edge bin needs the ordinary full-width
                // radix when it owns the threshold.
                uint32_t kmax = 0u;
#pragma unroll 2
                for (int i = lane; i < nvec; i += 32) {
                  const uint4 f = *reinterpret_cast<const uint4*>(ks + 4 * i);
                  kmax = max(kmax, max(max(f.x, f.y), max(f.z, f.w)));
                }
                asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(kmax) : "r"(kmax));
                const int hib = (kmax == 0u) ? 0 : (31 - __clz(kmax));
                prefix = 0u;
                pmask = 0u;
                k = W7_KTOP;
                allow_pre = false;
                sh = (hib > 7) ? (hib - 7) : 0;
                continue;
              }
              if (pre) {
                prefix = 0x40000000u + ((uint32_t)d << 17);
                pmask = 0xFFFE0000u;
              } else {
                prefix |= ((uint32_t)d) << sh;
                pmask |= 0xFFu << sh;
              }
              k -= above;
              if (cntd == k) {
                exact = true;
                break;
              }
              if (first_digit && sh > 0 && cntd <= 32 && k == 1) {
                uint32_t local_max = 0u;
                uint32_t local_count = 0u;
#pragma unroll 2
                for (int i = lane; i < nvec; i += 32) {
                  const uint4 f = *reinterpret_cast<const uint4*>(ks + 4 * i);
                  const uint32_t kv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
                  for (int e = 0; e < 4; ++e) {
                    if ((kv[e] & pmask) == prefix) {
                      if (kv[e] > local_max) {
                        local_max = kv[e];
                        local_count = 1u;
                      } else if (kv[e] == local_max) {
                        ++local_count;
                      }
                    }
                  }
                }
                const uint32_t lane_max = local_max;
                asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(local_max) : "r"(local_max));
                local_count = (lane_max == local_max) ? local_count : 0u;
                asm("redux.sync.add.u32 %0, %1, 0xffffffff;"
                    : "=r"(local_count)
                    : "r"(local_count));
                prefix = local_max;
                pmask = ~0u;
                // tau is now a full key, but the maximum may occur more than
                // once while only one threshold key is still needed.  Route
                // that rare case through the greater/equal-budget emitter.
                exact = (local_count == 1u);
                break;
              }
              if (first_digit && sh > 0 && cntd <= 64) {
                // The first selected byte normally contains only a handful of
                // keys.  Compact that bucket exactly and choose its residual
                // threshold in registers instead of clearing and atomically
                // rebuilding another 256-bin histogram.
                const uint32_t lower = (1u << lane) - 1u;
                int nsel = 0;
#pragma unroll 2
                for (int i0 = 0; i0 < nvec; i0 += 32) {
                  const int i = i0 + lane;
                  const bool valid = i < nvec;
                  const uint4 f = valid ? *reinterpret_cast<const uint4*>(ks + 4 * i)
                                        : make_uint4(0u, 0u, 0u, 0u);
                  const uint32_t kv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
                  for (int e = 0; e < 4; ++e) {
                    const bool take = valid && (kv[e] & pmask) == prefix;
                    const uint32_t bm = __ballot_sync(0xffffffffu, take);
                    if (take) compact[nsel + __popc(bm & lower)] = (int)kv[e];
                    nsel += __popc(bm);
                  }
                }
                __syncwarp();
                uint32_t sv = (lane < cntd) ? (uint32_t)compact[lane] : 0u;
                uint32_t sv2 = (lane + 32 < cntd) ? (uint32_t)compact[lane + 32] : 0u;
                uint32_t tau2;
                if (cntd <= 32) {
                  sv = w7sort_width<2>(sv, lane);
                  sv = w7sort_width<4>(sv, lane);
                  sv = w7sort_width<8>(sv, lane);
                  if (cntd > 8) sv = w7sort_width<16>(sv, lane);
                  if (cntd > 16) sv = w7sort_width<32>(sv, lane);
                  tau2 = __shfl_sync(0xffffffffu, sv, k - 1);
                } else {
                  w7sort64(sv, sv2, lane);
                  tau2 = (k <= 32) ? __shfl_sync(0xffffffffu, sv, k - 1)
                                   : __shfl_sync(0xffffffffu, sv2, k - 33);
                }
                const uint32_t gtm = __ballot_sync(0xffffffffu, lane < cntd && sv > tau2);
                const uint32_t eqm = __ballot_sync(0xffffffffu, lane < cntd && sv == tau2);
                const uint32_t gtm2 = __ballot_sync(0xffffffffu, lane + 32 < cntd && sv2 > tau2);
                const uint32_t eqm2 = __ballot_sync(0xffffffffu, lane + 32 < cntd && sv2 == tau2);
                k -= __popc(gtm) + __popc(gtm2);
                prefix = tau2;
                pmask = ~0u;
                exact = (__popc(eqm) + __popc(eqm2) == k);
                break;
              }
              if (sh == 0) break;
              sh = (sh > 8) ? (sh - 8) : 0;
            }
#ifdef W7_DBLRADIX
          }
#endif
#endif

          // Exact prefixes contain precisely the remaining budget.  A full
          // threshold key can instead have excess equals; preserve `exact`
          // so that path admits every greater key plus the lowest-id equals.
          const uint32_t tau = prefix, tmask = pmask;
          const int keq = exact ? (1 << 30) : k;
          int rank = 0, neq = 0;
          int* orow = out + (size_t)t * W7_PW;
          if (exact) {
            const uint32_t lower = (1u << lane) - 1u;
            for (int v0 = 0; v0 < Vs && rank < W7_KTOP; v0 += 256) {
              uint32_t kv[8];
              if (v0 + 256 <= Vs) {
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = ks[v0 + e * 32 + lane];
              } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                  const int v = v0 + e * 32 + lane;
                  kv[e] = (v < Vs) ? ks[v] : 0u;
                }
              }
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                const bool take = kv[e] >= tau;
                const uint32_t m = __ballot_sync(0xffffffffu, take);
                const int r = rank + __popc(m & lower);
                if (take && r < W7_KTOP) {
                  const int c = (v0 + e * 32 + lane) << 2;
                  int4 z;
                  z.x = c;
                  z.y = c + 1;
                  z.z = c + 2;
                  z.w = c + 3;
                  w7st(reinterpret_cast<int4*>(orow + (r << 2)), z);
                }
                rank += __popc(m);
              }
            }
          } else {
            const uint32_t lower = (1u << lane) - 1u;
            for (int v0 = 0; v0 < Vs && rank < W7_KTOP; v0 += 256) {
              uint32_t kv[8];
              if (v0 + 256 <= Vs) {
#pragma unroll
                for (int e = 0; e < 8; ++e) kv[e] = ks[v0 + e * 32 + lane];
              } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                  const int v = v0 + e * 32 + lane;
                  kv[e] = (v < Vs) ? ks[v] : 0u;
                }
              }
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                const uint32_t key = kv[e] & tmask;
                const bool gt = key > tau;
                const bool eq = key == tau;
                const uint32_t mg = __ballot_sync(0xffffffffu, gt);
                const uint32_t me = __ballot_sync(0xffffffffu, eq);
                const int budget = (keq > neq) ? (keq - neq) : 0;
                const int eql = __popc(me & lower);
                const bool take = gt || (eq && eql < budget);
                const int r = rank + __popc(mg & lower) + min(eql, budget);
                if (take && r < W7_KTOP) {
                  const int c = (v0 + e * 32 + lane) << 2;
                  int4 z;
                  z.x = c;
                  z.y = c + 1;
                  z.z = c + 2;
                  z.w = c + 3;
                  w7st(reinterpret_cast<int4*>(orow + (r << 2)), z);
                }
                const int ce = __popc(me);
                rank += __popc(mg) + min(ce, budget);
                neq += ce;
              }
            }
          }
          if (lane == 0) {
            const int rem = (pos + 1) & 3;
            const int ts = vis << 2;
            int4 tl;
            tl.x = (0 < rem) ? ts : -1;
            tl.y = (1 < rem) ? ts + 1 : -1;
            tl.z = (2 < rem) ? ts + 2 : -1;
            tl.w = 4 * W7_KTOP + rem;
            w7st(reinterpret_cast<int4*>(orow + 4 * W7_KTOP), tl);
          }
        }
      }
    }
    todo &= ~sub;
  }
}

// --------------------------------------------------------------- launcher ---
static inline void Run(const uint8_t* qi, const uint8_t* ck, const int* cu, const int* cpb, int B,
                       int T, int* out, cudaStream_t stream) {
  constexpr int fpt = 16;
  constexpr int frows = W7_SLOW - 1;
  constexpr int fgx = (frows * W7_RW + W7_FKTHR * fpt - 1) / (W7_FKTHR * fpt);
  const int ntile = (T + W7_GM - 1) / W7_GM;
  const int score_grid = (B == 2) ? (max(T - (W7_SLOW - 1), 0) + W7_GM - 1) / W7_GM : ntile;
  int fused_gx = 0, grid = score_grid;
  if (B <= 4) {
    const int nf = fgx * B;
    fused_gx = fgx;
    // For B=2 the kernel compacts the two request-local scored suffixes and
    // their exact prefix chunks.  Their combined CTA count is bounded by the
    // whole-T 8-row tile count plus four, independent of the device split.
    grid = (B == 2) ? (ntile + 4) : (score_grid + nf);
  } else {
    const int ntot = T * W7_RW;
    int pt = ntot / (W7_FKTHR * 424);
    pt = (pt >= 8) ? 8 : ((pt >= 4) ? 4 : ((pt >= 2) ? 2 : 1));
    const int fill_grid = (ntot + W7_FKTHR * pt - 1) / (W7_FKTHR * pt);
#define W7LF(P)                                                                          \
  do {                                                                                   \
    if (B == 1)                                                                          \
      w7_fill<P, true><<<fill_grid, W7_FKTHR, 0, stream>>>(cu, B, T, ntot, (int4*)out);  \
    else                                                                                 \
      w7_fill<P, false><<<fill_grid, W7_FKTHR, 0, stream>>>(cu, B, T, ntot, (int4*)out); \
  } while (0)
    switch (pt) {
      case 8:
        W7LF(8);
        break;
      case 4:
        W7LF(4);
        break;
      case 2:
        W7LF(2);
        break;
      default:
        W7LF(1);
        break;
    }
#undef W7LF
  }

  constexpr size_t smem = (size_t)(32 + W7_SCAP + W7_REGW + W7_SHISTW) * 4;
  static_assert(W7_REGW >= W7_NWARP * (256 + 32 + 64),
                "region too small for histograms and compact buckets");
  static_assert(W7_REGW >= 32 * W7_QROW, "region too small for q tile");
  static_assert(2 * smem <= 228 * 1024, "two CTAs per SM budget exceeded");
  // Two CTAs per SM fit in the ordinary non-portable shared-memory partition.
  cudaLaunchAttribute attr[2]{};
  attr[0].id = cudaLaunchAttributeSharedMemoryMode;
  attr[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowNonPortable;
  attr[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
  attr[1].val.sharedMemCarveout = 100;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(W7_NTHR);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = attr;
  config.numAttrs = 2;
#ifdef W7_PROFILE_LAUNCH
  if (B == 2) {
    cudaFuncSetAttribute(w7_score<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    w7_score<true><<<grid, W7_NTHR, smem, stream>>>(qi, ck, cu, B, T, cpb, out, ntile, fused_gx);
  } else {
    cudaFuncSetAttribute(w7_score<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    w7_score<false><<<grid, W7_NTHR, smem, stream>>>(qi, ck, cu, B, T, cpb, out, ntile, fused_gx);
  }
#else
  if (B == 2)
    cudaLaunchKernelEx(&config, w7_score<true>, qi, ck, cu, B, T, cpb, out, ntile, fused_gx);
  else
    cudaLaunchKernelEx(&config, w7_score<false>, qi, ck, cu, B, T, cpb, out, ntile, fused_gx);
#endif
}

}  // namespace w7n

#undef W7_QROW
#undef W7_KROW
#undef W7_GM
#undef W7_NTILE
#undef W7_NTHR
#undef W7_NWARP
#undef W7_VCAP
#undef W7_SCAP
#undef W7_REGW
#undef W7_NST
#undef W7_KTOP
#undef W7_PW
#undef W7_RW
#undef W7_SLOW
#undef W7_FKTHR
#undef W7_HROW
#undef W7_SHISTW
#undef W7MMA
