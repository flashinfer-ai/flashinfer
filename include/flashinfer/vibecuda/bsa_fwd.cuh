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
// VibeCUDA GQA block-sparse attention forward for SM100/SM103 (Blackwell).
//
// HMMA (mma.sync m16n8k16) Q@K^T and P@V with a stable FP32 max/sum online
// softmax over admitted key blocks read from a per-QO-head boolean block
// mask. TMA (cp.async.bulk.tensor) moves Q/K/V into SW128/SW64-swizzled
// shared-memory panels tracked with mbarrier expect_tx / try_wait.parity
// phases; admitted 64-key chunks are split across warps inside a CTA
// (pair/pair32/bm16 in-CTA splits for grid-underfilled shapes, plus a
// cross-CTA G-way split with a PDL-coupled merge kernel for dense rows).
// See csrc/vibecuda_bsa.cu for the framework binding and
// flashinfer/vibecuda_bsa.py for the public Python API.

#ifndef FLASHINFER_VIBECUDA_BSA_FWD_CUH_
#define FLASHINFER_VIBECUDA_BSA_FWD_CUH_


#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flashinfer {
namespace vibecuda {

#define DEVI __device__ __forceinline__

namespace {

constexpr int BM = 64;       // query rows per CTA
constexpr int BN = 64;       // key columns per inner chunk
constexpr int NWARPS = 4;    // one warp per 16 query rows
constexpr int NWWARPS = 8;   // wide kernel warps (128 row-head slots per CTA)
constexpr int MAX_BLOCKS = 1024;  // max admitted key blocks per (head, query block)
// K/V staging slots that fit the 166KB smem budget for a given head dim:
// bytes = Q + 2*nbuf*BN*D*2 + admit table + slack <= 166000.
inline int nbuf_for(int d) {
  const int budget = 166000 - BM * d * 2 - MAX_BLOCKS * (int)sizeof(int) - 2048;
  const int per_chunk = 2 * BN * d * 2;  // K + V, bf16/fp16
  const int n = budget / per_chunk;
  return n < 2 ? 2 : (n > 9 ? 9 : n);
}
// Wide 8-warp CTA: 128-row Q tile + 4KB admit table + 1.25KB per-warp bitsets
// against a 227KB opt-in budget (B300 max is 232448B per block).
// D=128 -> 5 chunks, D=96 -> 7, D=64 -> 9 (stage-array cap).
inline int nbuf_wide(int d) {
  const int budget = 200000 - 128 * d * 2 - MAX_BLOCKS * (int)sizeof(int) - 4096;
  const int per_chunk = 2 * BN * d * 2;
  const int n = budget / per_chunk;
  return n < 2 ? 2 : (n > 9 ? 9 : n);
}
// Pair-split CTA (8 warps, 2 chunk groups): Q + admit table + the group-B
// merge staging buffer (64 rows x (D+8) fp32) share the budget with K/V slots.
inline int nbuf_pair(int d) {
  const int budget = 204000 - BM * d * 2 - MAX_BLOCKS * (int)sizeof(int) -
                     64 * (d + 8) * (int)sizeof(float) - 4096;
  const int per_chunk = 2 * BN * d * 2;
  int n = budget / per_chunk;
  n = n > 9 ? 9 : n;
  // Windows are nbuf slots wide; an even window size keeps every window's
  // chunks on the same even/odd group assignment.
  n &= ~1;
  return n < 2 ? 2 : n;
}
// 32-row pair-split CTA (8 warps, FOUR chunk groups of 2 warps each): Q tile
// is 32 rows and the in-CTA merge staging holds three group partials
// (3 x 32 rows x (D+8) fp32) sharing the budget with the K/V slots. Window
// width must be a multiple of four so every window starts on group 0.
inline int nbuf_pair32(int d) {
  // GP=4 dedicated merge slices are budgeted with the prologue smem. The
  // launch attr is 219000 (device max 232448), so use the same cap here.
  const int budget = 219000 - 32 * d * 2 - MAX_BLOCKS * (int)sizeof(int) -
                     4 * 32 * (d + 8) * (int)sizeof(float) - 4096;
  const int per_chunk = 2 * BN * d * 2;
  int n = budget / per_chunk;
  n = n > 8 ? 8 : n;
  n &= ~3;
  return n < 4 ? 4 : n;
}
// 16-row 8-warp CTA: EIGHT independent 32-key partials (each admitted 64-key
// chunk splits between two warps). Eight fp32 partial slices
// (8 x 16 rows x (D+8)) share the budget with the K/V slots. Window width
// must be even (two warps consume one staged chunk).
inline int nbuf_bm16(int d) {
  const int budget = 219000 - 16 * d * 2 - MAX_BLOCKS * (int)sizeof(int) -
                     8 * 16 * (d + 8) * (int)sizeof(float) - 4096;
  const int per_chunk = 2 * BN * d * 2;
  int n = budget / per_chunk;
  n = n > 8 ? 8 : n;
  n &= ~1;
  return n < 2 ? 2 : n;
}
constexpr float LN2 = 0.69314718056f;
// Finite stand-in for -inf so max updates never produce NaN; exp2f of any
// difference involving it underflows to 0, which is the desired contribution.
constexpr float NEG_INF = -1.0e30f;

// Compile-time phase profiler (worker diagnostic; stripped in production):
// BSA_PHASE=1 records per-CTA cycle spans between named points into a global
// buffer passed via an extra kernel argument.
#ifdef BSA_PHASE
#define PHASE_DECL long long ph_t = clock64(); long long ph_prev = ph_t
#define PHASE_POINT(buf, i, cid)                                        \
  do {                                                                  \
    if (buf && threadIdx.x == 0) {                                      \
      long long n = clock64();                                          \
      buf[cid * 8 + i] = n - ph_prev;                                   \
      ph_prev = n;                                                      \
    }                                                                   \
  } while (0)
#define PHASE_FLUSH(buf, cid) \
  if (buf && threadIdx.x == 0) buf[cid * 8 + 7] = clock64() - ph_t
#else
#define PHASE_DECL
#define PHASE_POINT(buf, i, cid)
#define PHASE_FLUSH(buf, cid)
#endif

DEVI uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Byte address of element (r, c) of a ROWS x D tile stored as 64-column SW128
// panels (plus a 32-column SW64 tail panel when D == 96). TMA fills each panel
// row-major with the hardware swizzle, so element (r, c) of panel p sits at
// row r (128B or 64B per row) with its 16B chunk index XORed by (r % 8) (SW128)
// or (r % 4) (SW64).
template <int D, int ROWS>
DEVI uint32_t tile_elem_addr(uint32_t tile_base, int r, int c) {
  if (D > 64 && c >= 64) {
    const uint32_t p1 = tile_base + ROWS * 64 * (int)sizeof(uint16_t);
    const int cl = c - 64;
    if (D == 96) {  // SW64 panel: 64B rows, chunk idx XOR ((r/2) % 4) (verified)
      return p1 + r * 64 + (((cl * 2) ^ (((r >> 1) & 3) << 4)) & 63);
    }
    return p1 + r * 128 + (((cl * 2) ^ ((r & 7) << 4)) & 127);
  }
  return tile_base + r * 128 + (((c * 2) ^ ((r & 7) << 4)) & 127);
}

DEVI void ldsm_x4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// x2 variant whose two output registers form one MMA B-fragment directly
// (loads tile rows of 8 consecutive n-rows at two adjacent k-halves), so no
// register-pack MOVs are needed between the load and the HMMA operand pair.
DEVI void ldsm_x2(uint32_t addr, uint32_t &r0, uint32_t &r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

DEVI void ldsm_x4_trans(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// ---- mbarrier helpers ----
DEVI void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(bytes)
               : "memory");
}
DEVI void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@P bra DONE_%=;\n\t"
      "bra WAIT_%=;\n"
      "DONE_%=:\n}"
      ::"r"(bar), "r"(phase)
      : "memory");
}
// ---- TMA issue (single thread) ----
DEVI void tma_3d(uint32_t dst, const CUtensorMap *map, int c0, int c1, int c2,
                 uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}

// 4D variant used by the D=128 panel fold: coords {col_off, panel_off, head, row}.
DEVI void tma_4d(uint32_t dst, const CUtensorMap *map, int c0, int c1, int c2,
                 int c3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}

// Issue the Q tile fetch: all D columns for `bmq` rows starting at row0.
template <int D>
DEVI void issue_q(const CUtensorMap &m0, const CUtensorMap &m1, uint32_t sqa,
                  int hq, int row0, uint32_t bar, int bmq) {
  mbar_expect_tx(bar, bmq * D * 2);
  if (D == 128) {
    // One 4D op fetches both 64-col panels (map dims {cols, rows, panels, heads},
    // so rows fill before panels and the smem result equals two 3D panel fills).
    tma_4d(sqa, &m0, 0, row0, 0, hq, bar);
    return;
  }
  tma_3d(sqa, &m0, 0, hq, row0, bar);
  if (D == 96) {
    tma_3d(sqa + bmq * 64 * 2, &m1, 64, hq, row0, bar);
  }
}

// Issue one chunk of K and V (KROWS key rows each) into stage buffers.
template <int D>
DEVI void issue_kv(const CUtensorMap &mk0, const CUtensorMap &mk1,
                   const CUtensorMap &mv0, const CUtensorMap &mv1, uint32_t ska,
                   uint32_t sva, int nbase, int kvh, uint32_t bar, int krows) {
  mbar_expect_tx(bar, 2 * krows * D * 2);
  if (D == 128) {
    // Panel fold: one 4D op per tensor fetches both 64-col panels.
    tma_4d(ska, &mk0, 0, nbase, 0, kvh, bar);
    tma_4d(sva, &mv0, 0, nbase, 0, kvh, bar);
    return;
  }
  tma_3d(ska, &mk0, 0, kvh, nbase, bar);
  tma_3d(sva, &mv0, 0, kvh, nbase, bar);
  if (D == 96) {
    tma_3d(ska + (size_t)krows * 64 * 2, &mk1, 64, kvh, nbase, bar);
    tma_3d(sva + (size_t)krows * 64 * 2, &mv1, 64, kvh, nbase, bar);
  }
}

// Split-barrier K-only / V-only arms for the pair32 kernel: separate barrier
// sets per slot let warps 2-5 arm K while warp 6 arms V, so the two
// first-touch tensormap descriptor reads (tk0, tv0) run in PARALLEL across
// warps instead of chaining serially on the arming thread (~600 cycles off
// the P0 critical path).
template <int D>
DEVI void issue_ks(const CUtensorMap &mk0, const CUtensorMap &mk1, uint32_t ska,
                   int nbase, int kvh, uint32_t bar, int krows) {
  mbar_expect_tx(bar, krows * D * 2);
  if (D == 128) {
    tma_4d(ska, &mk0, 0, nbase, 0, kvh, bar);
    return;
  }
  tma_3d(ska, &mk0, 0, kvh, nbase, bar);
  if (D == 96) {
    tma_3d(ska + (size_t)krows * 64 * 2, &mk1, 64, kvh, nbase, bar);
  }
}
template <int D>
DEVI void issue_vs(const CUtensorMap &mv0, const CUtensorMap &mv1, uint32_t sva,
                   int nbase, int kvh, uint32_t bar, int krows) {
  mbar_expect_tx(bar, krows * D * 2);
  if (D == 128) {
    tma_4d(sva, &mv0, 0, nbase, 0, kvh, bar);
    return;
  }
  tma_3d(sva, &mv0, 0, kvh, nbase, bar);
  if (D == 96) {
    tma_3d(sva + (size_t)krows * 64 * 2, &mv1, 64, kvh, nbase, bar);
  }
}
// Combined arm for the non-critical windowed refill paths: one thread arms
// both tensors onto their respective barrier sets.
template <int D>
DEVI void issue_kv2(const CUtensorMap &mk0, const CUtensorMap &mk1,
                    const CUtensorMap &mv0, const CUtensorMap &mv1,
                    uint32_t ska, uint32_t sva, int nbase, int kvh,
                    uint32_t bark, uint32_t barv, int krows) {
  issue_ks<D>(mk0, mk1, ska, nbase, kvh, bark, krows);
  issue_vs<D>(mv0, mv1, sva, nbase, kvh, barv, krows);
}

// ---- trait for the two packed-accum MMA variants ----
template <typename T> struct FragPack;
template <> struct FragPack<__nv_bfloat16> {
  static constexpr uint32_t ONES = 0x3f803f80u;
  static DEVI __nv_bfloat16 cvt(float a) { return __float2bfloat16_rn(a); }
  // Packed exp2 path: round two fp32 args into bf16x2, then one packed MUFU
  // ex2. Matches pack(a, b) bit order (a -> low half); cvt.rn puts its FIRST
  // operand in the high half (verified against pack on sm_103).
  static DEVI uint32_t cvt2(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(a), "f"(b));
    return d;
  }
  static DEVI uint32_t ex2(uint32_t u) {
    uint32_t d;
    asm volatile("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(d) : "r"(u));
    return d;
  }
  static DEVI void mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
template <> struct FragPack<__half> {
  static constexpr uint32_t ONES = 0x3c003c00u;
  static DEVI __half cvt(float a) { return __float2half_rn(a); }
  static DEVI uint32_t cvt2(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(a), "f"(b));
    return d;
  }
  static DEVI uint32_t ex2(uint32_t u) {
    uint32_t d;
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(d) : "r"(u));
    return d;
  }
  static DEVI void mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};

// ---------------- merge helper: one warp merges one output row ------------
// Combines per-split (O, l, m) partials with max rescaling:
//   m* = max_s m_s, factor_s = exp2(m_s - m*) (0 when l_s == 0),
//   out = sum_s O_s*factor_s / sum_s l_s*factor_s,
//   LSE = m* + ln(sum_s l_s*factor_s).
// Merge is DRAM latency-bound, not bandwidth-bound (long_scoreboard top
// stall at ~10% occupancy, all CTAs resident in one partial wave). D <= 128
// makes NVEC = D/4 <= 32, i.e. exactly one float4 group per lane, so BOTH
// independent memory rounds — lane<G's l/m pair and the per-split float4 —
// are issued up front; the shuffle reductions (lm-dependent only) then
// overlap the float4 DRAM round, collapsing two serialized rounds into one.
// (Measured alternative: two rows per warp with a doubled tmp prefetch was
// 3.64us vs 3.08us one-row on the long case — per-SM in-flight bytes, not
// per-lane load count, is the cap.)
template <typename T, int D, bool WITH_LSE>
DEVI void merge_one_row(const float *__restrict__ ows, T *__restrict__ out,
                        float *__restrict__ lse_g, int M, int HQ, int G,
                        int rows_pad, int row, int hq, int lane) {
  constexpr int NVEC = D / 4;  // float4 groups per row: 16..32
  static_assert(NVEC <= 32, "D <= 128: exactly one float4 group per lane");
  const size_t step = (size_t)rows_pad * HQ * (D + 4);
  const size_t base = ((size_t)row * HQ + hq) * (D + 4);

  // Round issue 1: l/m pairs are adjacent floats -> one 8B load per split.
  float2 lm = {0.f, NEG_INF};
  if (lane < G)
    lm = __ldg(reinterpret_cast<const float2 *>(
        ows + base + (size_t)lane * step + D));
  // Round issue 2 (independent): every split's float4 for this lane.
  float4 tmp[16];
  if (lane < NVEC) {
    const float *pb = ows + base + lane * 4;
#pragma unroll
    for (int s = 0; s < 16; ++s)
      if (s < G)
        tmp[s] =
            __ldg(reinterpret_cast<const float4 *>(pb + (size_t)s * step));
  }

  // Reductions depend on lm only; they run while the float4 stream flies.
  const float l_s = (lane < G) ? lm.x : 0.f;
  const float m_s = (lane < G) ? lm.y : NEG_INF;
  float m_star = m_s;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m_star = fmaxf(m_star, __shfl_xor_sync(0xffffffffu, m_star, off));
  // l_s == 0 (empty split) must contribute exactly 0 regardless of m_s.
  const float fs_lane = (l_s > 0.f) ? exp2f(m_s - m_star) : 0.f;
  float ltot = l_s * fs_lane;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ltot += __shfl_xor_sync(0xffffffffu, ltot, off);
  const float inv = (ltot > 0.f) ? (1.f / ltot) : 0.f;
  if (lane < NVEC) {
    float4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      if (s < G) {
        const float fs = __shfl_sync(0xffffffffu, fs_lane, s);
        acc.x = fmaf(tmp[s].x, fs, acc.x);
        acc.y = fmaf(tmp[s].y, fs, acc.y);
        acc.z = fmaf(tmp[s].z, fs, acc.z);
        acc.w = fmaf(tmp[s].w, fs, acc.w);
      }
    }
    T po[4];
    po[0] = FragPack<T>::cvt(acc.x * inv);
    po[1] = FragPack<T>::cvt(acc.y * inv);
    po[2] = FragPack<T>::cvt(acc.z * inv);
    po[3] = FragPack<T>::cvt(acc.w * inv);
    *reinterpret_cast<uint2 *>(&out[((size_t)row * HQ + hq) * D + lane * 4]) =
        *reinterpret_cast<uint2 *>(&po[0]);
  }
  if (WITH_LSE && lane == 0) {
    lse_g[(size_t)row * HQ + hq] =
        (ltot > 0.f) ? (m_star + log2f(ltot)) * LN2 : -INFINITY;
  }
}

// ------------ split (main) kernel with TMA + mbarrier staging --------------
// 4 warps: each warp owns 16 query rows and the full 64-key chunk. Stable
// online softmax: per-chunk running max with exp2-domain rescaling, so splits
// carry (O, l, m) partials and merge with max rescaling downstream.
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(NWARPS * 32)
bsa_split_kernel(const __grid_constant__ CUtensorMap tq0,
                 const __grid_constant__ CUtensorMap tq1,
                 const __grid_constant__ CUtensorMap tk0,
                 const __grid_constant__ CUtensorMap tk1,
                 const __grid_constant__ CUtensorMap tv0,
                 const __grid_constant__ CUtensorMap tv1,
                 const bool *__restrict__ mask, T *__restrict__ out,
                 float *__restrict__ lse_g, float *__restrict__ ows, int M, int N,
                 int HQ, int HKV, int BS, int rows_pad, int nbuf, bool normalize,
                 float scale_log2e, long long *phbuf) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8 = BN / 8;  // n8 fragments per 64-key chunk
  const int G = gridDim.z;
  PHASE_DECL;
  const int ph_cid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;

  extern __shared__ char dyn_smem[];
  // Align tile base to the 1024B swizzle atom.
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);
  T *sK0 = reinterpret_cast<T *>(sbase + BM * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (BM * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));

  constexpr int MAXSTAGE = 9;
  __shared__ uint64_t bar_q, bar_full[MAXSTAGE];
  __shared__ int sWarpCount[NWARPS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rowgrp = warp;
  const int hq = blockIdx.y;
  const int split = blockIdx.z;
  const int row0 = blockIdx.x * BM;
  const int qblk = row0 / BS;
  const int group = HQ / HKV;
  const int kvh = hq / group;
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;

  // ---- barriers up (parallel per-lane inits); the mask row load is issued
  // first so its global latency overlaps the setup below ----
  const uint8_t *mrow =
      reinterpret_cast<const uint8_t *>(mask) + ((size_t)hq * MBm + qblk) * NB;
  // Early mask-row byte prefetch: per-lane predicated loads covering up to 256
  // blocks. Only the ballot-scan path consumes these registers; NB==4 uses a
  // single uint32 load and NB>256 re-reads gmem in its wide loop, so skip the
  // byte loads there (issue pressure on the prologue critical path).
  bool mrow_pf[8];
  const bool pf_ok = (NB != 4) && (NB <= 256);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = i * 32 + lane;
    mrow_pf[i] = (pf_ok && idx < NB) ? (mrow[idx] != 0) : false;
  }
  uint32_t mbytes_pf = 0;
  if (NB == 4) mbytes_pf = *reinterpret_cast<const uint32_t *>(mrow);
  if (warp == 0) {
    if (lane < nbuf && lane < MAXSTAGE) mbar_init(smem_addr(&bar_full[lane]), 1);
    if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
    __syncwarp();
    if (lane == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), BM);
    }
  }

  PHASE_POINT(phbuf, 0, ph_cid);
  // ---- admitted key blocks for (hq, qblk) ----
  const int CPB = BS / BN;  // 64-key chunks per admitted block
  int nadmit;
  if (NB == 4) {
    // Contiguous 4-byte bool mask row: one broadcast 32-bit load per CTA
    // replaces the cooperative ballot scan (and its two __syncthreads).
    // Elements are whole bytes 0/1 -> first collapse to a 4-bit mask.
    const uint32_t mbytes = mbytes_pf;
    const uint32_t m4 = ((mbytes & 0xFFu) ? 1u : 0u) |
                        ((mbytes & 0xFF00u) ? 2u : 0u) |
                        ((mbytes & 0xFF0000u) ? 4u : 0u) |
                        ((mbytes & 0xFF000000u) ? 8u : 0u);
    nadmit = __popc(m4);
    nadmit = nadmit > MAX_BLOCKS ? MAX_BLOCKS : nadmit;
    if (warp == 0 && lane < nadmit) {
      const uint32_t b0 = m4 & (m4 - 1u);  // rank-0 bit cleared
      const uint32_t b1 = b0 & (b0 - 1u);
      const uint32_t b2 = b1 & (b1 - 1u);
      const uint32_t sel = (lane == 0) ? m4 : (lane == 1) ? b0 : (lane == 2) ? b1 : b2;
      sAdmit[lane] = __ffs(sel) - 1;
    }
  } else if (NB <= 256) {
    // Single-pass ballot scan over prefetched bytes: 4 warps x 8 ballots max,
    // no per-round syncthreads, no multi-block smem prefix chain. Rank->index
    // expansion happens once on <= 32 warp0 lanes; sAdmit keeps the same
    // ascending-rank layout the chunk loop expects.
    uint32_t w[8];
    const int nw = (NB + 31) >> 5;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      w[i] = (i < nw) ? __ballot_sync(0xffffffffu, mrow_pf[i]) : 0u;
    if (warp == 0) {
      int running = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < nw) {
          const uint32_t bal = w[i];
          const int cnt = __popc(bal);
          const int off = __popc(bal & ((1u << lane) - 1));
          const bool adm = (bal >> lane) & 1u;
          if (adm && running + off < MAX_BLOCKS)
            sAdmit[running + off] = i * 32 + lane;
          running += cnt;
        }
      }
      nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
    }
    // every warp arrives at the same count; broadcast via the barrier below
    if (warp != 0) {
      int total = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) total += __popc(w[i]);
      nadmit = total > MAX_BLOCKS ? MAX_BLOCKS : total;
    }
  } else {
    int running = 0;
    for (int base = 0; base < NB; base += NWARPS * 32) {
      int idx = base + tid;
      bool adm = (idx < NB) && (mrow[idx] != 0);
      unsigned bal = __ballot_sync(0xffffffffu, adm);
      int cnt = __popc(bal);
      int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = cnt;
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < NWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (adm && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  }
  const int nchunks = nadmit * CPB;
  __syncthreads();  // sAdmit visible to all (incl. the TMA-issuing thread)
  PHASE_POINT(phbuf, 1, ph_cid);

  const int myRow = rowgrp * 16 + (lane >> 2);
  const int myCol = (lane & 3) * 2;
  const int gr0 = row0 + myRow;
  const int gr8 = row0 + myRow + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;  // running row max (exp2 domain)

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };

  if (split >= nchunks) {
    mbar_wait(smem_addr(&bar_q), 0);  // drain the Q fetch before exit
    if (normalize) {
      T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        if (gr0 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
        if (gr8 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
      }
      if (WITH_LSE) {
        if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
        if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
      }
    } else {
      // empty split: zero the full row so the merge can read every split
      // unconditionally without a validity-mask chain.
      const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
      const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        ows[b0 * (D + 4) + c] = 0.f;
        ows[b0 * (D + 4) + c + 1] = 0.f;
        ows[b8 * (D + 4) + c] = 0.f;
        ows[b8 * (D + 4) + c + 1] = 0.f;
      }
      ows[b0 * (D + 4) + D] = 0.f;
      ows[b0 * (D + 4) + D + 1] = NEG_INF;
      ows[b8 * (D + 4) + D] = 0.f;
      ows[b8 * (D + 4) + D + 1] = NEG_INF;
    }
    return;
  }

  // ---- single issuing thread kicks off the K/V fetches ----
  const int mychunks = (nchunks - 1 - split) / G + 1;
  const bool full_stage = mychunks <= nbuf;
  // Parallel arming: chunk slots have independent mbarriers, so one warp
  // per slot issues its K/V TMA fetches concurrently (was one thread at
  // ~100 cycles per serialized TMA op on the prologue critical path).
  if (full_stage) {
    for (int j = warp; j < mychunks; j += NWARPS) {
      if (lane == 0)
        issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                    smem_addr(sV0 + (size_t)j * BN * D),
                    chunk_nbase(split + j * G), kvh, smem_addr(&bar_full[j]), BN);
    }
  } else {
    if (warp == 0 && lane == 0) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0), smem_addr(sV0),
                  chunk_nbase(split), kvh, smem_addr(&bar_full[0]), BN);
    }
    if (warp == 1 && lane == 0 && split + G < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + BN * D),
                  smem_addr(sV0 + BN * D), chunk_nbase(split + G), kvh,
                  smem_addr(&bar_full[1]), BN);
    }
  }

  PHASE_POINT(phbuf, 2, ph_cid);

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  // ---- hoist the (chunk-invariant) Q fragments into registers once; the
  // loads also overlap the wait for chunk 0's K/V arrival below. Only for
  // D <= 96: at D=128 the +32 registers measurably hurt (register pressure).
  constexpr int DKH = DK;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, BM>(smem_addr(sQ), rowgrp * 16 + (lane & 15),
                                  jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  PHASE_POINT(phbuf, 3, ph_cid);

  if (full_stage) {
    // Fully staged consume loop with a software-pipelined score buffer:
    // QK of chunk j+1 (write-once slot, TMA already armed) interleaves with
    // the softmax/PV of chunk j within the warp. No refills, no syncthreads;
    // each chunk's mbarrier is waited exactly once (parity 0).
    // ---- S = Q @ K^T (fp32) accumulator factory for one 64-key chunk ----
    float S[NS8][4];
    auto qk_stage = [&](int slot, float (&Sacc)[NS8][4]) {
      T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
      for (int jj = 0; jj < NS8; ++jj) {
#pragma unroll
        for (int e = 0; e < 4; ++e) Sacc[jj][e] = 0.f;
      }
#pragma unroll
      for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
        for (int jn = 0; jn < NS8; jn += 2) {
          // Two x2 loads whose outputs ARE the B fragments ({Tn,klo, Tn,khi}),
          // avoiding the x4->pair redistribution MOVs.
          uint32_t b0[2], b1[2];
          ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 7),
                                        jc * 16 + ((lane & 8) >> 3) * 8),
                  b0[0], b0[1]);
          ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + 8 + (lane & 7),
                                        jc * 16 + ((lane & 8) >> 3) * 8),
                  b1[0], b1[1]);
          uint32_t qa[4];
          if (DKH > 0) {
#pragma unroll
            for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
          } else {
            ldsm_x4(tile_elem_addr<D, BM>(smem_addr(sQ),
                                          rowgrp * 16 + (lane & 15),
                                          jc * 16 + (lane >> 4) * 8),
                    qa[0], qa[1], qa[2], qa[3]);
          }
          FragPack<T>::mma(Sacc[jn], qa, b0);
          FragPack<T>::mma(Sacc[jn + 1], qa, b1);
        }
      }
    };

    mbar_wait(smem_addr(&bar_full[0]), 0);
    for (int j = 0; j < mychunks; ++j) {
      if (j > 0) mbar_wait(smem_addr(&bar_full[j]), 0);
      qk_stage(j, S);
      T *sV = sV0 + (size_t)j * BN * D;
      const int nbase = chunk_nbase(split + j * G);

      if (nbase + BN > N) {  // partial final key block
#pragma unroll
        for (int jn = 0; jn < NS8; ++jn) {
          int c0 = nbase + jn * 8 + myCol;
          if (c0 >= N) S[jn][0] = NEG_INF;
          if (c0 + 1 >= N) S[jn][1] = NEG_INF;
          if (c0 >= N) S[jn][2] = NEG_INF;
          if (c0 + 1 >= N) S[jn][3] = NEG_INF;
        }
      }

      // ---- chunk row max (quad-reduced), advance the running max, and
      // rescale previous O/l partials by exp2(m_old - m_new). Both rows are
      // handled under one branch: alpha is exactly 1.0f for a row whose max
      // did not move, so its rescale is a no-op.
      float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
        cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
      }
      cmax0 *= scale_log2e;
      cmax8 *= scale_log2e;
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
        cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
      }
      const float mnew0 = fmaxf(m_r, cmax0);
      const float mnew8 = fmaxf(m_r8, cmax8);
      if (j == 0) {
        // First chunk in this group's chain: O and l are still zero and m is
        // -inf, so the rescale below is provably a no-op (alpha == 0 and it
        // only scales zeros). Skipping it removes 2 MUFU + ~24 FMUL from the
        // one-chunk-per-group path (all fixed-suite pair32 cases).
        m_r = mnew0;
        m_r8 = mnew8;
      } else if (mnew0 > m_r || mnew8 > m_r8) {
        const float alpha0 = exp2f(m_r - mnew0);
        const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
        for (int jj = 0; jj < DN; ++jj) {
          O[jj][0] *= alpha0;
          O[jj][1] *= alpha0;
          O[jj][2] *= alpha8;
          O[jj][3] *= alpha8;
        }
        l_r *= alpha0;
        l_r8 *= alpha8;
        m_r = mnew0;
        m_r8 = mnew8;
      }

      // ---- P = packed ex2(cvt(S*scale - m)); row sums via ones-MMA below ----
      uint32_t pfr[NS8 / 2][4];
#pragma unroll
      for (int jc = 0; jc < NS8 / 2; ++jc) {
        pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][0], scale_log2e, -m_r),
            fmaf(S[2 * jc][1], scale_log2e, -m_r)));
        pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
        pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
            fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
        pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
      }

      // ---- P @ V accumulate, plus row sums via P @ 1 (ones-MMA) ----
      float lacc[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
      for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
        for (int jn = 0; jn < DN; jn += 2) {
          uint32_t vq[4];
          ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV), jk * 16 + (lane & 15),
                                              jn * 8 + (lane >> 4) * 8),
                        vq[0], vq[1], vq[2], vq[3]);
          uint32_t b0[2] = {vq[0], vq[1]};
          uint32_t b1[2] = {vq[2], vq[3]};
          FragPack<T>::mma(O[jn], pfr[jk], b0);
          FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
        }
        FragPack<T>::mma(lacc, pfr[jk], ones2);
      }
      l_r += lacc[0];
      l_r8 += lacc[2];
    }
  } else {
  uint32_t ph0 = 0, ph1 = 0;
  int stage = 0;
  for (int ci = split; ci < nchunks; ci += G, stage ^= 1) {
    T *sK = sK0 + stage * BN * D;
    T *sV = sV0 + stage * BN * D;
    const uint32_t sbar = smem_addr(&bar_full[stage]);
    if (stage) {
      mbar_wait(sbar, ph1);
      ph1 ^= 1;
    } else {
      mbar_wait(sbar, ph0);
      ph0 ^= 1;
    }
    const int nbase = chunk_nbase(ci);

    // ---- S = Q @ K^T (fp32): the full 64-key chunk ----
    float S[NS8][4];
#pragma unroll
    for (int j = 0; j < NS8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[j][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8; jn += 2) {
        uint32_t kb4[4];
        ldsm_x4(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 15),
                                      jc * 16 + (lane >> 4) * 8),
                kb4[0], kb4[1], kb4[2], kb4[3]);
        uint32_t b0[2] = {kb4[0], kb4[2]};
        uint32_t b1[2] = {kb4[1], kb4[3]};
        uint32_t qa[4];
        if (DKH > 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
        } else {
          ldsm_x4(tile_elem_addr<D, BM>(smem_addr(sQ), rowgrp * 16 + (lane & 15),
                                        jc * 16 + (lane >> 4) * 8),
                  qa[0], qa[1], qa[2], qa[3]);
        }
        FragPack<T>::mma(S[jn], qa, b0);
        FragPack<T>::mma(S[jn + 1], qa, b1);
      }
    }

    if (nbase + BN > N) {  // partial final key block (never in the fixed suite)
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        int c0 = nbase + jn * 8 + myCol;
        if (c0 >= N) S[jn][0] = NEG_INF;
        if (c0 + 1 >= N) S[jn][1] = NEG_INF;
        if (c0 >= N) S[jn][2] = NEG_INF;
        if (c0 + 1 >= N) S[jn][3] = NEG_INF;
      }
    }

    // ---- chunk row max (quad-reduced), advance the running max, and
    // rescale previous O/l partials by exp2(m_old - m_new). Both rows are
    // handled under one branch: alpha is exactly 1.0f for a row whose max
    // did not move, so its rescale is a no-op.
    float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
    for (int jn = 0; jn < NS8; ++jn) {
      cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
      cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
    }
    cmax0 *= scale_log2e;
    cmax8 *= scale_log2e;
#pragma unroll
    for (int sh = 1; sh <= 2; sh <<= 1) {
      cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
      cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
    }
    const float mnew0 = fmaxf(m_r, cmax0);
    const float mnew8 = fmaxf(m_r8, cmax8);
    if (mnew0 > m_r || mnew8 > m_r8) {
      const float alpha0 = exp2f(m_r - mnew0);  // NEG_INF first time -> 0
      const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
      for (int j = 0; j < DN; ++j) {
        O[j][0] *= alpha0;
        O[j][1] *= alpha0;
        O[j][2] *= alpha8;
        O[j][3] *= alpha8;
      }
      l_r *= alpha0;
      l_r8 *= alpha8;
      m_r = mnew0;
      m_r8 = mnew8;
    }

    // ---- P = packed ex2(cvt(S*scale - m)); row sums via ones-MMA below ----
    uint32_t pfr[NS8 / 2][4];
#pragma unroll
    for (int jc = 0; jc < NS8 / 2; ++jc) {
      pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][0], scale_log2e, -m_r),
          fmaf(S[2 * jc][1], scale_log2e, -m_r)));
      pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
      pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
          fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
      pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
    }

    // ---- P @ V accumulate, plus row sums via P @ 1 (ones-MMA) ----
    float lacc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
    for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
      for (int jn = 0; jn < DN; jn += 2) {
        uint32_t vq[4];
        ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV), jk * 16 + (lane & 15),
                                            jn * 8 + (lane >> 4) * 8),
                      vq[0], vq[1], vq[2], vq[3]);
        uint32_t b0[2] = {vq[0], vq[1]};
        uint32_t b1[2] = {vq[2], vq[3]};
        FragPack<T>::mma(O[jn], pfr[jk], b0);
        FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
      }
      FragPack<T>::mma(lacc, pfr[jk], ones2);
    }
    l_r += lacc[0];
    l_r8 += lacc[2];

    __syncthreads();  // stage consumed by all warps; safe to refill
    const int ci2 = ci + 2 * G;
    if (tid == 0 && ci2 < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK), smem_addr(sV),
                  chunk_nbase(ci2), kvh, sbar, BN);
    }
  }
  }

  PHASE_POINT(phbuf, 4, ph_cid);

  // ---- epilogue ----
  if (normalize) {
    const float inv0 = (l_r > 0.f) ? (1.f / l_r) : 0.f;
    const float inv8 = (l_r8 > 0.f) ? (1.f / l_r8) : 0.f;
    // Stage the normalized rows in smem (K/V staging is consumed by now), then
    // write out with fully coalesced 16B stores instead of 32 scattered 4B
    // stores per warp ("epi drain uncoalesced" was a top tail cost).
    constexpr int DP = D + 8;  // padded row pitch: 16B-aligned and bank-spread
    T *sOw = reinterpret_cast<T *>(sK0) + (size_t)rowgrp * 16 * DP;
    const int rw0 = (lane >> 2), rw8 = rw0 + 8;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      T pk[4];
      pk[0] = FragPack<T>::cvt(O[jn][0] * inv0);
      pk[1] = FragPack<T>::cvt(O[jn][1] * inv0);
      pk[2] = FragPack<T>::cvt(O[jn][2] * inv8);
      pk[3] = FragPack<T>::cvt(O[jn][3] * inv8);
      *reinterpret_cast<uint32_t *>(sOw + rw0 * DP + c) =
          *reinterpret_cast<uint32_t *>(pk);
      *reinterpret_cast<uint32_t *>(sOw + rw8 * DP + c) =
          *reinterpret_cast<uint32_t *>(pk + 2);
    }
    __syncwarp();
    constexpr int NR16 = D / 8;  // uint4 per row
#pragma unroll
    for (int i = 0; i < 16 * NR16 / 32; ++i) {
      const int u = i * 32 + lane;
      const int rr = u / NR16;
      const int cc = u - rr * NR16;
      const int gr = row0 + rowgrp * 16 + rr;
      if (gr < M) {
        const uint4 x = *reinterpret_cast<const uint4 *>(
            reinterpret_cast<char *>(sOw) + rr * (DP * 2) + cc * 16);
        *reinterpret_cast<uint4 *>(&out[((size_t)gr * HQ + hq) * D + cc * 8]) = x;
      }
    }
    if (WITH_LSE) {
      if (gr0 < M)
        lse_g[(size_t)gr0 * HQ + hq] = (l_r > 0.f) ? (m_r + log2f(l_r)) * LN2 : -INFINITY;
      if (gr8 < M)
        lse_g[(size_t)gr8 * HQ + hq] = (l_r8 > 0.f) ? (m_r8 + log2f(l_r8)) * LN2 : -INFINITY;
    }
  } else {
    const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
    const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      // Adjacent fp32 pairs -> one 8B store each (halves LSU wavefronts; the
      // split epilogue showed lg_throttle as a top-3 stall at long N).
      float2 lo, hi;
      lo.x = O[jn][0];
      lo.y = O[jn][1];
      hi.x = O[jn][2];
      hi.y = O[jn][3];
      *reinterpret_cast<float2 *>(&ows[b0 * (D + 4) + c]) = lo;
      *reinterpret_cast<float2 *>(&ows[b8 * (D + 4) + c]) = hi;
    }
    ows[b0 * (D + 4) + D] = l_r;
    ows[b0 * (D + 4) + D + 1] = m_r;
    ows[b8 * (D + 4) + D] = l_r8;
    ows[b8 * (D + 4) + D + 1] = m_r8;
  }
  PHASE_POINT(phbuf, 5, ph_cid);
  PHASE_FLUSH(phbuf, ph_cid);
  // PDL: all workspace/output writes of this CTA are complete; allow the
  // dependent merge kernel to start. The merge kernel pairs this with
  // griddepcontrol.wait before reading `ows`/`out`.
  asm volatile("griddepcontrol.launch_dependents;");
}

// Two rows per warp, 8 warps per CTA: doubles per-SM load depth where the
// row grid underfills the device (latency-depth-bound streaming merge).
// Uses the proven single-row merge_one_row core twice: the interleaved-load
// two-row variant tripped a warp-convergence miscompile on D=96 lanes>=NVEC
// (intermittent illegal instruction at the r=1 shfl tree under PDL), and the
// sequential version retains the depth benefit.
template <typename T, int D, bool WITH_LSE>
__global__ void bsa_merge2_kernel(const float *__restrict__ ows, T *__restrict__ out,
                                  float *__restrict__ lse_g, int M, int HQ, int G,
                                  int rows_pad) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int hq = blockIdx.y;
  const int row0 = blockIdx.x * 8 + warp;
  if (row0 >= M) return;

  asm volatile("griddepcontrol.wait;");
  merge_one_row<T, D, WITH_LSE>(ows, out, lse_g, M, HQ, G, rows_pad, row0, hq,
                                lane);
}

// ============ GQA head-packed split kernel ============
// One CTA covers RPH query rows x PH query heads that all share one KV head
// (PH = 2 or 4, WPH = NWARPS/PH warps per head, RPH = 16*WPH rows per head,
// 64 Q rows in smem total). Each staged K/V chunk is consumed once per CTA
// instead of once per (head, row-tile) CTA, so TMA traffic per covered
// (row, head) pair drops ~PH-fold when row tiles stay small. The chunk list
// is the UNION of the PH heads' admitted key blocks for the CTA's query
// block; every warp re-checks its own head's mask byte per chunk and skips
// compute for chunks its head did not admit (identical math to the per-head
// kernel). Dispatch is purely on runtime (HQ, HKV, M, N, D, BS, selected).
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(NWARPS * 32)
bsa_pack_kernel(const __grid_constant__ CUtensorMap tq0,
                const __grid_constant__ CUtensorMap tq1,
                const __grid_constant__ CUtensorMap tk0,
                const __grid_constant__ CUtensorMap tk1,
                const __grid_constant__ CUtensorMap tv0,
                const __grid_constant__ CUtensorMap tv1,
                const bool *__restrict__ mask, T *__restrict__ out,
                float *__restrict__ lse_g, float *__restrict__ ows, int M, int N,
                int HQ, int HKV, int BS, int rows_pad, int nbuf, bool normalize,
                float scale_log2e, int PH) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8 = BN / 8;
  const int WPH = NWARPS / PH;     // warps per head
  const int RPH = 16 * WPH;        // query rows per head in this CTA
  const int G = gridDim.z;

  extern __shared__ char dyn_smem[];
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);   // 64 rows: head-major [PH][RPH][D]
  T *sK0 = reinterpret_cast<T *>(sbase + 64 * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (BM * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));

  constexpr int MAXSTAGE = 9;
  __shared__ uint64_t bar_q, bar_full[MAXSTAGE];
  __shared__ int sWarpCount[NWARPS];
  // Per-head admission bitsets (PH heads x up-to-1024 key blocks, 32 blocks
  // per u32 word): the chunk loop re-checks each warp's own head from smem
  // instead of a dependent global byte load on the critical path. Only valid
  // when NB <= 1024; otherwise the chunk loop falls back to the mask row in
  // global memory (still correct, just the old latency profile).
  constexpr int MAXNBW = MAX_BLOCKS / 32;
  uint32_t *sBits = reinterpret_cast<uint32_t *>(sAdmit + MAX_BLOCKS);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hgh = warp / WPH;                  // head index inside the pack
  const int rseg = warp % WPH;                 // 16-row segment inside head
  const int hq0 = blockIdx.y * PH;
  const int hq = hq0 + hgh;
  const int split = blockIdx.z;
  const int row0 = blockIdx.x * RPH;
  const int qblk = row0 / BS;
  const int group = HQ / HKV;
  const int kvh = hq0 / group;
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;

  // ---- barriers up (parallel per-lane inits), Q tile fetches issued ----
  if (warp == 0) {
    if (lane < nbuf && lane < MAXSTAGE) mbar_init(smem_addr(&bar_full[lane]), 1);
    if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
    __syncwarp();
  }
  if (tid == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_expect_tx(smem_addr(&bar_q), 64 * D * 2);
    for (int h = 0; h < PH; ++h) {
      for (int sg = 0; sg < WPH; ++sg) {
        const int dst_rows = (h * WPH + sg) * 16;
        tma_3d(smem_addr(sQ) + (size_t)dst_rows * 64 * 2, &tq0, 0, hq0 + h,
               row0 + sg * 16, smem_addr(&bar_q));
        if (D == 96) {
          tma_3d(smem_addr(sQ) + 64 * 64 * 2 + (size_t)dst_rows * 32 * 2, &tq1,
                 64, hq0 + h, row0 + sg * 16, smem_addr(&bar_q));
        } else if (D == 128) {
          tma_3d(smem_addr(sQ) + 64 * 64 * 2 + (size_t)dst_rows * 64 * 2, &tq0,
                 64, hq0 + h, row0 + sg * 16, smem_addr(&bar_q));
        }
      }
    }
  }

  // ---- union of admitted key blocks across the pack's PH heads ----
  const uint8_t *mbase = reinterpret_cast<const uint8_t *>(mask) +
                         ((size_t)hq0 * MBm + qblk) * NB;
  const int CPB = BS / BN;
  const int NBW = (NB + 31) / 32;  // u32 words per per-head bitset
  const bool useBits = (NB <= MAX_BLOCKS);
  int nadmit;
  if (NB == 4) {
    // One 32-bit broadcast load per head row; OR the collapsed 4-bit sets.
    uint32_t m4 = 0;
    for (int h = 0; h < PH; ++h) {
      const uint32_t mb4 = *reinterpret_cast<const uint32_t *>(mbase + h * MBm * 4);
      const uint32_t m4h = ((mb4 & 0xFFu) ? 1u : 0u) | ((mb4 & 0xFF00u) ? 2u : 0u) |
                           ((mb4 & 0xFF0000u) ? 4u : 0u) | ((mb4 & 0xFF000000u) ? 8u : 0u);
      if (tid == 0) sBits[h] = m4h;
      m4 |= m4h;
    }
    nadmit = __popc(m4);
    nadmit = nadmit > MAX_BLOCKS ? MAX_BLOCKS : nadmit;
    if (warp == 0 && lane < nadmit) {
      const uint32_t b0 = m4 & (m4 - 1u);
      const uint32_t b1 = b0 & (b0 - 1u);
      const uint32_t b2 = b1 & (b1 - 1u);
      const uint32_t sel = (lane == 0) ? m4 : (lane == 1) ? b0 : (lane == 2) ? b1 : b2;
      sAdmit[lane] = __ffs(sel) - 1;
    }
  } else {
    int running = 0;
    for (int base = 0; base < NB; base += NWARPS * 32) {
      int idx = base + tid;
      const int widx = idx >> 5;  // uniform per warp
      unsigned bal = 0;
      for (int h = 0; h < PH; ++h) {
        const bool ah = (idx < NB) && (mbase[h * MBm * NB + idx] != 0);
        const unsigned bh = __ballot_sync(0xffffffffu, ah);
        if (lane == 0 && useBits) sBits[h * NBW + widx] = bh;
        bal |= bh;  // OR of per-head ballots == ballot of the union
      }
      int cnt = __popc(bal);
      int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = cnt;
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < NWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (((bal >> lane) & 1u) && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  }
  const int nchunks = nadmit * CPB;
  __syncthreads();

  // Warp's private row/head constants; this head's own mask row (global, one
  // byte load per chunk re-checks admission for the union-fetched chunks).
  const uint8_t *mrowH = reinterpret_cast<const uint8_t *>(mask) +
                         ((size_t)hq * MBm + qblk) * NB;
  const int myRowCTA = (hgh * WPH + rseg) * 16;  // row base inside sQ tile
  const int grBase = row0 + rseg * 16;           // global row base of warp
  const int myRow = myRowCTA + (lane >> 2);
  const int myCol = (lane & 3) * 2;
  const int gr0 = grBase + (lane >> 2);
  const int gr8 = gr0 + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };
  // Per-head re-admission of a union chunk: smem bitset (NB <= 1024) or the
  // global mask byte otherwise.
  auto chunk_adm = [&](int jb) {  // jb = sAdmit[ci / CPB]
    if (useBits)
      return ((sBits[hgh * NBW + (jb >> 5)] >> (jb & 31)) & 1u) != 0u;
    return mrowH[jb] != 0;
  };
  if (split >= nchunks) {
    mbar_wait(smem_addr(&bar_q), 0);  // drain the Q fetch before exit
    if (normalize) {
      T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        if (gr0 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
        if (gr8 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
      }
      if (WITH_LSE) {
        if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
        if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
      }
    } else {
      const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
      const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        ows[b0 * (D + 4) + c] = 0.f;
        ows[b0 * (D + 4) + c + 1] = 0.f;
        ows[b8 * (D + 4) + c] = 0.f;
        ows[b8 * (D + 4) + c + 1] = 0.f;
      }
      ows[b0 * (D + 4) + D] = 0.f;
      ows[b0 * (D + 4) + D + 1] = NEG_INF;
      ows[b8 * (D + 4) + D] = 0.f;
      ows[b8 * (D + 4) + D + 1] = NEG_INF;
    }
    return;
  }

  // ---- single issuing thread kicks off the K/V fetches (union chunk list) --
  const int mychunks = (nchunks - 1 - split) / G + 1;
  const bool full_stage = mychunks <= nbuf;
  // Parallel arming: chunk slots have independent mbarriers, so one warp
  // per slot issues its K/V TMA fetches concurrently (was one thread at
  // ~100 cycles per serialized TMA op on the prologue critical path).
  if (full_stage) {
    for (int j = warp; j < mychunks; j += NWARPS) {
      if (lane == 0)
        issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                    smem_addr(sV0 + (size_t)j * BN * D),
                    chunk_nbase(split + j * G), kvh, smem_addr(&bar_full[j]), BN);
    }
  } else {
    if (warp == 0 && lane == 0) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0), smem_addr(sV0),
                  chunk_nbase(split), kvh, smem_addr(&bar_full[0]), BN);
    }
    if (warp == 1 && lane == 0 && split + G < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + BN * D),
                  smem_addr(sV0 + BN * D), chunk_nbase(split + G), kvh,
                  smem_addr(&bar_full[1]), BN);
    }
  }

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  constexpr int DKH = (D <= 96) ? DK : 0;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, 64>(smem_addr(sQ), myRowCTA + (lane & 15),
                                  jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  if (full_stage) {
    float S[NS8][4];
    auto qk_stage = [&](int slot, float (&Sacc)[NS8][4]) {
      T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
      for (int jj = 0; jj < NS8; ++jj) {
#pragma unroll
        for (int e = 0; e < 4; ++e) Sacc[jj][e] = 0.f;
      }
#pragma unroll
      for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
        for (int jn = 0; jn < NS8; jn += 2) {
          // Two x2 loads whose outputs ARE the B fragments ({Tn,klo, Tn,khi}),
          // avoiding the x4->pair redistribution MOVs.
          uint32_t b0[2], b1[2];
          ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 7),
                                        jc * 16 + ((lane & 8) >> 3) * 8),
                  b0[0], b0[1]);
          ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + 8 + (lane & 7),
                                        jc * 16 + ((lane & 8) >> 3) * 8),
                  b1[0], b1[1]);
          uint32_t qa[4];
          if (DKH > 0) {
#pragma unroll
            for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
          } else {
            ldsm_x4(tile_elem_addr<D, 64>(smem_addr(sQ),
                                          myRowCTA + (lane & 15),
                                          jc * 16 + (lane >> 4) * 8),
                    qa[0], qa[1], qa[2], qa[3]);
          }
          FragPack<T>::mma(Sacc[jn], qa, b0);
          FragPack<T>::mma(Sacc[jn + 1], qa, b1);
        }
      }
    };

    mbar_wait(smem_addr(&bar_full[0]), 0);
    for (int j = 0; j < mychunks; ++j) {
      if (j > 0) mbar_wait(smem_addr(&bar_full[j]), 0);
      const int ci = split + j * G;
      const int jb = sAdmit[ci / CPB];
      if (chunk_adm(jb)) {
        qk_stage(j, S);
        T *sV = sV0 + (size_t)j * BN * D;
        const int nbase = chunk_nbase(ci);
        if (nbase + BN > N) {
#pragma unroll
          for (int jn = 0; jn < NS8; ++jn) {
            int c0 = nbase + jn * 8 + myCol;
            if (c0 >= N) S[jn][0] = NEG_INF;
            if (c0 + 1 >= N) S[jn][1] = NEG_INF;
            if (c0 >= N) S[jn][2] = NEG_INF;
            if (c0 + 1 >= N) S[jn][3] = NEG_INF;
          }
        }
        float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
        for (int jn = 0; jn < NS8; ++jn) {
          cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
          cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
        }
        cmax0 *= scale_log2e;
        cmax8 *= scale_log2e;
#pragma unroll
        for (int sh = 1; sh <= 2; sh <<= 1) {
          cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
          cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
        }
        const float mnew0 = fmaxf(m_r, cmax0);
        const float mnew8 = fmaxf(m_r8, cmax8);
        if (mnew0 > m_r || mnew8 > m_r8) {
          const float alpha0 = exp2f(m_r - mnew0);
          const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
          for (int jj = 0; jj < DN; ++jj) {
            O[jj][0] *= alpha0;
            O[jj][1] *= alpha0;
            O[jj][2] *= alpha8;
            O[jj][3] *= alpha8;
          }
          l_r *= alpha0;
          l_r8 *= alpha8;
          m_r = mnew0;
          m_r8 = mnew8;
        }
        uint32_t pfr[NS8 / 2][4];
#pragma unroll
        for (int jc = 0; jc < NS8 / 2; ++jc) {
          pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc][0], scale_log2e, -m_r),
              fmaf(S[2 * jc][1], scale_log2e, -m_r)));
          pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc][2], scale_log2e, -m_r8),
              fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
          pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
              fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
          pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
              fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
        }
        float lacc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
        for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
          for (int jn = 0; jn < DN; jn += 2) {
            uint32_t vq[4];
            ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV),
                                                jk * 16 + (lane & 15),
                                                jn * 8 + (lane >> 4) * 8),
                          vq[0], vq[1], vq[2], vq[3]);
            uint32_t b0[2] = {vq[0], vq[1]};
            uint32_t b1[2] = {vq[2], vq[3]};
            FragPack<T>::mma(O[jn], pfr[jk], b0);
            FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
          }
          FragPack<T>::mma(lacc, pfr[jk], ones2);
        }
        l_r += lacc[0];
        l_r8 += lacc[2];
      }
    }
  } else {
  uint32_t ph0 = 0, ph1 = 0;
  int stage = 0;
  for (int ci = split; ci < nchunks; ci += G, stage ^= 1) {
    T *sK = sK0 + stage * BN * D;
    T *sV = sV0 + stage * BN * D;
    const uint32_t sbar = smem_addr(&bar_full[stage]);
    if (stage) {
      mbar_wait(sbar, ph1);
      ph1 ^= 1;
    } else {
      mbar_wait(sbar, ph0);
      ph0 ^= 1;
    }
    const int nbase = chunk_nbase(ci);
    const bool admh = chunk_adm(sAdmit[ci / CPB]);

    if (admh) {
    float S[NS8][4];
#pragma unroll
    for (int j = 0; j < NS8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[j][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8; jn += 2) {
        uint32_t kb4[4];
        ldsm_x4(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 15),
                                      jc * 16 + (lane >> 4) * 8),
                kb4[0], kb4[1], kb4[2], kb4[3]);
        uint32_t b0[2] = {kb4[0], kb4[2]};
        uint32_t b1[2] = {kb4[1], kb4[3]};
        uint32_t qa[4];
        if (DKH > 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
        } else {
          ldsm_x4(tile_elem_addr<D, 64>(smem_addr(sQ), myRowCTA + (lane & 15),
                                        jc * 16 + (lane >> 4) * 8),
                  qa[0], qa[1], qa[2], qa[3]);
        }
        FragPack<T>::mma(S[jn], qa, b0);
        FragPack<T>::mma(S[jn + 1], qa, b1);
      }
    }

    if (nbase + BN > N) {
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        int c0 = nbase + jn * 8 + myCol;
        if (c0 >= N) S[jn][0] = NEG_INF;
        if (c0 + 1 >= N) S[jn][1] = NEG_INF;
        if (c0 >= N) S[jn][2] = NEG_INF;
        if (c0 + 1 >= N) S[jn][3] = NEG_INF;
      }
    }

    float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
    for (int jn = 0; jn < NS8; ++jn) {
      cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
      cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
    }
    cmax0 *= scale_log2e;
    cmax8 *= scale_log2e;
#pragma unroll
    for (int sh = 1; sh <= 2; sh <<= 1) {
      cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
      cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
    }
    const float mnew0 = fmaxf(m_r, cmax0);
    const float mnew8 = fmaxf(m_r8, cmax8);
    if (mnew0 > m_r || mnew8 > m_r8) {
      const float alpha0 = exp2f(m_r - mnew0);
      const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
      for (int j = 0; j < DN; ++j) {
        O[j][0] *= alpha0;
        O[j][1] *= alpha0;
        O[j][2] *= alpha8;
        O[j][3] *= alpha8;
      }
      l_r *= alpha0;
      l_r8 *= alpha8;
      m_r = mnew0;
      m_r8 = mnew8;
    }

    uint32_t pfr[NS8 / 2][4];
#pragma unroll
    for (int jc = 0; jc < NS8 / 2; ++jc) {
      pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][0], scale_log2e, -m_r),
          fmaf(S[2 * jc][1], scale_log2e, -m_r)));
      pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
      pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
          fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
      pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
    }

    float lacc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
    for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
      for (int jn = 0; jn < DN; jn += 2) {
        uint32_t vq[4];
        ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV),
                                            jk * 16 + (lane & 15),
                                            jn * 8 + (lane >> 4) * 8),
                      vq[0], vq[1], vq[2], vq[3]);
        uint32_t b0[2] = {vq[0], vq[1]};
        uint32_t b1[2] = {vq[2], vq[3]};
        FragPack<T>::mma(O[jn], pfr[jk], b0);
        FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
      }
      FragPack<T>::mma(lacc, pfr[jk], ones2);
    }
    l_r += lacc[0];
    l_r8 += lacc[2];
    }  // admh

    __syncthreads();  // stage consumed by all warps; safe to refill
    const int ci2 = ci + 2 * G;
    if (tid == 0 && ci2 < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK), smem_addr(sV),
                  chunk_nbase(ci2), kvh, sbar, BN);
    }
  }
  }

  // ---- epilogue (warp's private head and rows) ----
  if (normalize) {
    const float inv0 = (l_r > 0.f) ? (1.f / l_r) : 0.f;
    const float inv8 = (l_r8 > 0.f) ? (1.f / l_r8) : 0.f;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      T plo[2], phi[2];
      plo[0] = FragPack<T>::cvt(O[jn][0] * inv0);
      plo[1] = FragPack<T>::cvt(O[jn][1] * inv0);
      phi[0] = FragPack<T>::cvt(O[jn][2] * inv8);
      phi[1] = FragPack<T>::cvt(O[jn][3] * inv8);
      if (gr0 < M)
        *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
            *reinterpret_cast<uint32_t *>(plo);
      if (gr8 < M)
        *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
            *reinterpret_cast<uint32_t *>(phi);
    }
    if (WITH_LSE) {
      if (gr0 < M)
        lse_g[(size_t)gr0 * HQ + hq] = (l_r > 0.f) ? (m_r + log2f(l_r)) * LN2 : -INFINITY;
      if (gr8 < M)
        lse_g[(size_t)gr8 * HQ + hq] = (l_r8 > 0.f) ? (m_r8 + log2f(l_r8)) * LN2 : -INFINITY;
    }
  } else {
    const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
    const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      float2 lo, hi;
      lo.x = O[jn][0];
      lo.y = O[jn][1];
      hi.x = O[jn][2];
      hi.y = O[jn][3];
      *reinterpret_cast<float2 *>(&ows[b0 * (D + 4) + c]) = lo;
      *reinterpret_cast<float2 *>(&ows[b8 * (D + 4) + c]) = hi;
    }
    ows[b0 * (D + 4) + D] = l_r;
    ows[b0 * (D + 4) + D + 1] = m_r;
    ows[b8 * (D + 4) + D] = l_r8;
    ows[b8 * (D + 4) + D + 1] = m_r8;
  }
  // PDL: all workspace/output writes of this CTA are complete; allow the
  // dependent merge kernel to start.
  asm volatile("griddepcontrol.launch_dependents;");
}

// ============ 8-warp wide split kernel: 128 row-head slots per CTA =========
// One CTA covers PH query heads x RPH=16*(8/PH) rows (8 warps, 128 row-head
// slots; PH heads must sit inside one GQA group so they share the KV head).
// Warp g owns one 16-row group: head hq0 + g/WPH, global rows
// row0 + (g%WPH)*16 + [0,16). The chunk list is the UNION of all 8 warp mask
// rows (each staged K/V chunk feeds twice the row-head work of the 4-warp
// kernels, so per-slot staging traffic halves); every warp re-admits each
// staged chunk against its OWN (head, query-block) row via a per-warp smem
// bitset (NB <= 1024) or a global byte re-read, and skips non-admitted chunks
// -- identical math to the per-head kernel, with 2 warps per scheduler.
// Rows may straddle query blocks (RPH=128 with BS=64): each warp's query
// block is grBase/BS, so the union is over 8 arbitrary mask rows.
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(NWWARPS * 32)
bsa_wide_kernel(const __grid_constant__ CUtensorMap tq0,
                const __grid_constant__ CUtensorMap tq1,
                const __grid_constant__ CUtensorMap tk0,
                const __grid_constant__ CUtensorMap tk1,
                const __grid_constant__ CUtensorMap tv0,
                const __grid_constant__ CUtensorMap tv1,
                const bool *__restrict__ mask, T *__restrict__ out,
                float *__restrict__ lse_g, float *__restrict__ ows, int M, int N,
                int HQ, int HKV, int BS, int rows_pad, int nbuf, bool normalize,
                float scale_log2e, int PH) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8 = BN / 8;
  const int WPH = NWWARPS / PH;  // warps per head
  const int RPH = 16 * WPH;      // query rows per head in this CTA
  const int G = gridDim.z;

  extern __shared__ char dyn_smem[];
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);  // 128 rows, warp-major [8][16][D]
  T *sK0 = reinterpret_cast<T *>(sbase + 128 * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (128 * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));
  uint32_t *sBits = reinterpret_cast<uint32_t *>(sAdmit + MAX_BLOCKS);
  uint32_t *sUnion = sBits + NWWARPS * (MAX_BLOCKS / 32);

  constexpr int MAXSTAGE = 9;
  __shared__ uint64_t bar_q, bar_full[MAXSTAGE];
  __shared__ int sWarpCount[NWWARPS];
  __shared__ int sNadmit;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hgh = warp / WPH;
  const int rseg = warp % WPH;
  const int hq0 = blockIdx.y * PH;
  const int hq = hq0 + hgh;
  const int split = blockIdx.z;
  const int row0 = blockIdx.x * RPH;
  const int grBase = row0 + rseg * 16;
  const int qb = grBase / BS;
  const int group = HQ / HKV;
  const int kvh = (PH > 1) ? (hq0 / group) : (hq / group);
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;
  const bool useBits = (NB <= MAX_BLOCKS);
  const int NBW = useBits ? (NB + 31) / 32 : 1;
  const int CPB = BS / BN;

  // ---- barriers up (parallel per-lane inits), Q tile fetches issued ----
  if (warp == 0) {
    if (lane < nbuf && lane < MAXSTAGE) mbar_init(smem_addr(&bar_full[lane]), 1);
    if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
    __syncwarp();
  }
  if (tid == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_expect_tx(smem_addr(&bar_q), 128 * D * 2);
    for (int h = 0; h < PH; ++h) {
      tma_3d(smem_addr(sQ) + (size_t)h * RPH * 64 * 2, &tq0, 0, hq0 + h, row0,
             smem_addr(&bar_q));
      if (D == 96) {
        tma_3d(smem_addr(sQ) + 128 * 64 * 2 + (size_t)h * RPH * 32 * 2, &tq1,
               64, hq0 + h, row0, smem_addr(&bar_q));
      } else if (D == 128) {
        tma_3d(smem_addr(sQ) + 128 * 64 * 2 + (size_t)h * RPH * 64 * 2, &tq0,
               64, hq0 + h, row0, smem_addr(&bar_q));
      }
    }
  }

  // ---- per-warp mask row + union of admitted key blocks across the CTA ----
  const uint8_t *mrowW = reinterpret_cast<const uint8_t *>(mask) +
                         ((size_t)hq * MBm + qb) * NB;
  int nadmit;
  if (useBits) {
    if (NB == 4) {
      const uint32_t mb4 = *reinterpret_cast<const uint32_t *>(mrowW);
      const uint32_t m4 = ((mb4 & 0xFFu) ? 1u : 0u) |
                          ((mb4 & 0xFF00u) ? 2u : 0u) |
                          ((mb4 & 0xFF0000u) ? 4u : 0u) |
                          ((mb4 & 0xFF000000u) ? 8u : 0u);
      if (lane == 0) sBits[warp * NBW] = m4;
    } else {
      for (int base = 0; base < NB; base += 32) {
        const int idx = base + lane;
        const bool adm = (idx < NB) && (mrowW[idx] != 0);
        const unsigned bal = __ballot_sync(0xffffffffu, adm);
        if (lane == 0) sBits[warp * NBW + (base >> 5)] = bal;
      }
    }
    __syncthreads();
    if (tid < NBW) {
      uint32_t u = 0;
#pragma unroll
      for (int g = 0; g < NWWARPS; ++g) u |= sBits[g * NBW + tid];
      sUnion[tid] = u;
    }
    __syncthreads();
    int running = 0;
    for (int base = 0; base < NB; base += NWWARPS * 32) {
      const int idx = base + tid;
      const bool adm =
          (idx < NB) && ((sUnion[idx >> 5] >> (idx & 31)) & 1u) != 0u;
      const unsigned bal = __ballot_sync(0xffffffffu, adm);
      const int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = __popc(bal);
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < NWWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (adm && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  } else {
    // NB > 1024: serial warp-0 union scan over global rows; per-warp
    // re-admission below falls back to global byte reads. Same ascending
    // order + MAX_BLOCKS cap as the per-head kernel.
    if (warp == 0) {
      int running = 0;
      for (int base = 0; base < NB; base += 32) {
        const int idx = base + lane;
        uint32_t u = 0;
#pragma unroll
        for (int g = 0; g < NWWARPS; ++g) {
          const int gh = g / WPH, gr = g % WPH;
          const uint8_t *mr = reinterpret_cast<const uint8_t *>(mask) +
                              ((size_t)(hq0 + gh) * MBm +
                               (row0 + gr * 16) / BS) *
                                  NB;
          const bool adm = (idx < NB) && (mr[idx] != 0);
          u |= __ballot_sync(0xffffffffu, adm);
        }
        const int off = __popc(u & ((1u << lane) - 1));
        const int pos = running + off;
        if (((u >> lane) & 1u) && pos < MAX_BLOCKS) sAdmit[pos] = idx;
        running += __popc(u);
      }
      if (lane == 0) sNadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
    }
    __syncthreads();
    nadmit = sNadmit;
  }
  const int nchunks = nadmit * CPB;
  __syncthreads();

  // Per-warp re-admission of a union chunk (bitset or global byte).
  auto chunk_adm = [&](int jb) {
    if (useBits)
      return ((sBits[warp * NBW + (jb >> 5)] >> (jb & 31)) & 1u) != 0u;
    return mrowW[jb] != 0;
  };

  const int myRowCTA = warp * 16;  // row base inside the sQ tile
  const int myCol = (lane & 3) * 2;
  const int gr0 = grBase + (lane >> 2);
  const int gr8 = gr0 + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };

  if (split >= nchunks) {
    mbar_wait(smem_addr(&bar_q), 0);  // drain the Q fetch before exit
    if (normalize) {
      T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        if (gr0 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
        if (gr8 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
      }
      if (WITH_LSE) {
        if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
        if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
      }
    } else {
      const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
      const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        ows[b0 * (D + 4) + c] = 0.f;
        ows[b0 * (D + 4) + c + 1] = 0.f;
        ows[b8 * (D + 4) + c] = 0.f;
        ows[b8 * (D + 4) + c + 1] = 0.f;
      }
      ows[b0 * (D + 4) + D] = 0.f;
      ows[b0 * (D + 4) + D + 1] = NEG_INF;
      ows[b8 * (D + 4) + D] = 0.f;
      ows[b8 * (D + 4) + D + 1] = NEG_INF;
    }
    return;
  }

  // ---- single issuing thread kicks off the K/V fetches (union chunk list) --
  const int mychunks = (nchunks - 1 - split) / G + 1;
  const bool full_stage = mychunks <= nbuf;
  // Parallel arming: chunk slots have independent mbarriers, so one warp
  // per slot issues its K/V TMA fetches concurrently (was one thread at
  // ~100 cycles per serialized TMA op on the prologue critical path).
  if (full_stage) {
    for (int j = warp; j < mychunks; j += NWWARPS) {
      if (lane == 0)
        issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                    smem_addr(sV0 + (size_t)j * BN * D),
                    chunk_nbase(split + j * G), kvh, smem_addr(&bar_full[j]), BN);
    }
  } else {
    if (warp == 0 && lane == 0) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0), smem_addr(sV0),
                  chunk_nbase(split), kvh, smem_addr(&bar_full[0]), BN);
    }
    if (warp == 1 && lane == 0 && split + G < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + BN * D),
                  smem_addr(sV0 + BN * D), chunk_nbase(split + G), kvh,
                  smem_addr(&bar_full[1]), BN);
    }
  }

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  constexpr int DKH = DK;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, 128>(smem_addr(sQ), myRowCTA + (lane & 15),
                                   jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  if (full_stage) {
    float S[NS8][4];
    auto qk_stage = [&](int slot, float (&Sacc)[NS8][4]) {
      T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
      for (int jj = 0; jj < NS8; ++jj) {
#pragma unroll
        for (int e = 0; e < 4; ++e) Sacc[jj][e] = 0.f;
      }
#pragma unroll
      for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
        for (int jn = 0; jn < NS8; jn += 2) {
          uint32_t kb4[4];
          ldsm_x4(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 15),
                                        jc * 16 + (lane >> 4) * 8),
                  kb4[0], kb4[1], kb4[2], kb4[3]);
          uint32_t b0[2] = {kb4[0], kb4[2]};
          uint32_t b1[2] = {kb4[1], kb4[3]};
          FragPack<T>::mma(Sacc[jn], qa_r[jc], b0);
          FragPack<T>::mma(Sacc[jn + 1], qa_r[jc], b1);
        }
      }
    };

    mbar_wait(smem_addr(&bar_full[0]), 0);
    for (int j = 0; j < mychunks; ++j) {
      if (j > 0) mbar_wait(smem_addr(&bar_full[j]), 0);
      const int ci = split + j * G;
      const int jb = sAdmit[ci / CPB];
      if (chunk_adm(jb)) {
        qk_stage(j, S);
        T *sV = sV0 + (size_t)j * BN * D;
        const int nbase = chunk_nbase(ci);
        if (nbase + BN > N) {
#pragma unroll
          for (int jn = 0; jn < NS8; ++jn) {
            int c0 = nbase + jn * 8 + myCol;
            if (c0 >= N) S[jn][0] = NEG_INF;
            if (c0 + 1 >= N) S[jn][1] = NEG_INF;
            if (c0 >= N) S[jn][2] = NEG_INF;
            if (c0 + 1 >= N) S[jn][3] = NEG_INF;
          }
        }
        float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
        for (int jn = 0; jn < NS8; ++jn) {
          cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
          cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
        }
        cmax0 *= scale_log2e;
        cmax8 *= scale_log2e;
#pragma unroll
        for (int sh = 1; sh <= 2; sh <<= 1) {
          cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
          cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
        }
        const float mnew0 = fmaxf(m_r, cmax0);
        const float mnew8 = fmaxf(m_r8, cmax8);
        if (mnew0 > m_r || mnew8 > m_r8) {
          const float alpha0 = exp2f(m_r - mnew0);
          const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
          for (int jj = 0; jj < DN; ++jj) {
            O[jj][0] *= alpha0;
            O[jj][1] *= alpha0;
            O[jj][2] *= alpha8;
            O[jj][3] *= alpha8;
          }
          l_r *= alpha0;
          l_r8 *= alpha8;
          m_r = mnew0;
          m_r8 = mnew8;
        }
        uint32_t pfr[NS8 / 2][4];
#pragma unroll
        for (int jc = 0; jc < NS8 / 2; ++jc) {
          pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc][0], scale_log2e, -m_r),
              fmaf(S[2 * jc][1], scale_log2e, -m_r)));
          pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc][2], scale_log2e, -m_r8),
              fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
          pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
              fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
          pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
              fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
              fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
        }
        float lacc[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
        for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
          for (int jn = 0; jn < DN; jn += 2) {
            uint32_t vq[4];
            ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV),
                                                jk * 16 + (lane & 15),
                                                jn * 8 + (lane >> 4) * 8),
                          vq[0], vq[1], vq[2], vq[3]);
            uint32_t b0[2] = {vq[0], vq[1]};
            uint32_t b1[2] = {vq[2], vq[3]};
            FragPack<T>::mma(O[jn], pfr[jk], b0);
            FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
          }
          FragPack<T>::mma(lacc, pfr[jk], ones2);
        }
        l_r += lacc[0];
        l_r8 += lacc[2];
      }
    }
  } else {
  uint32_t ph0 = 0, ph1 = 0;
  int stage = 0;
  for (int ci = split; ci < nchunks; ci += G, stage ^= 1) {
    T *sK = sK0 + stage * BN * D;
    T *sV = sV0 + stage * BN * D;
    const uint32_t sbar = smem_addr(&bar_full[stage]);
    if (stage) {
      mbar_wait(sbar, ph1);
      ph1 ^= 1;
    } else {
      mbar_wait(sbar, ph0);
      ph0 ^= 1;
    }
    const int nbase = chunk_nbase(ci);
    const bool admh = chunk_adm(sAdmit[ci / CPB]);

    if (admh) {
    float S[NS8][4];
#pragma unroll
    for (int j = 0; j < NS8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[j][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8; jn += 2) {
        uint32_t kb4[4];
        ldsm_x4(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 15),
                                      jc * 16 + (lane >> 4) * 8),
                kb4[0], kb4[1], kb4[2], kb4[3]);
        uint32_t b0[2] = {kb4[0], kb4[2]};
        uint32_t b1[2] = {kb4[1], kb4[3]};
        FragPack<T>::mma(S[jn], qa_r[jc], b0);
        FragPack<T>::mma(S[jn + 1], qa_r[jc], b1);
      }
    }

    if (nbase + BN > N) {
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        int c0 = nbase + jn * 8 + myCol;
        if (c0 >= N) S[jn][0] = NEG_INF;
        if (c0 + 1 >= N) S[jn][1] = NEG_INF;
        if (c0 >= N) S[jn][2] = NEG_INF;
        if (c0 + 1 >= N) S[jn][3] = NEG_INF;
      }
    }

    float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
    for (int jn = 0; jn < NS8; ++jn) {
      cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
      cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
    }
    cmax0 *= scale_log2e;
    cmax8 *= scale_log2e;
#pragma unroll
    for (int sh = 1; sh <= 2; sh <<= 1) {
      cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
      cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
    }
    const float mnew0 = fmaxf(m_r, cmax0);
    const float mnew8 = fmaxf(m_r8, cmax8);
    if (mnew0 > m_r || mnew8 > m_r8) {
      const float alpha0 = exp2f(m_r - mnew0);
      const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
      for (int j = 0; j < DN; ++j) {
        O[j][0] *= alpha0;
        O[j][1] *= alpha0;
        O[j][2] *= alpha8;
        O[j][3] *= alpha8;
      }
      l_r *= alpha0;
      l_r8 *= alpha8;
      m_r = mnew0;
      m_r8 = mnew8;
    }

    uint32_t pfr[NS8 / 2][4];
#pragma unroll
    for (int jc = 0; jc < NS8 / 2; ++jc) {
      pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][0], scale_log2e, -m_r),
          fmaf(S[2 * jc][1], scale_log2e, -m_r)));
      pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
      pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
          fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
      pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
    }

    float lacc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
    for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
      for (int jn = 0; jn < DN; jn += 2) {
        uint32_t vq[4];
        ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV),
                                            jk * 16 + (lane & 15),
                                            jn * 8 + (lane >> 4) * 8),
                      vq[0], vq[1], vq[2], vq[3]);
        uint32_t b0[2] = {vq[0], vq[1]};
        uint32_t b1[2] = {vq[2], vq[3]};
        FragPack<T>::mma(O[jn], pfr[jk], b0);
        FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
      }
      FragPack<T>::mma(lacc, pfr[jk], ones2);
    }
    l_r += lacc[0];
    l_r8 += lacc[2];
    }  // admh

    __syncthreads();  // stage consumed by all warps; safe to refill
    const int ci2 = ci + 2 * G;
    if (tid == 0 && ci2 < nchunks) {
      issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK), smem_addr(sV),
                  chunk_nbase(ci2), kvh, sbar, BN);
    }
  }
  }

  // ---- epilogue (warp's private head and rows) ----
  if (normalize) {
    const float inv0 = (l_r > 0.f) ? (1.f / l_r) : 0.f;
    const float inv8 = (l_r8 > 0.f) ? (1.f / l_r8) : 0.f;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      T plo[2], phi[2];
      plo[0] = FragPack<T>::cvt(O[jn][0] * inv0);
      plo[1] = FragPack<T>::cvt(O[jn][1] * inv0);
      phi[0] = FragPack<T>::cvt(O[jn][2] * inv8);
      phi[1] = FragPack<T>::cvt(O[jn][3] * inv8);
      if (gr0 < M)
        *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
            *reinterpret_cast<uint32_t *>(plo);
      if (gr8 < M)
        *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
            *reinterpret_cast<uint32_t *>(phi);
    }
    if (WITH_LSE) {
      if (gr0 < M)
        lse_g[(size_t)gr0 * HQ + hq] = (l_r > 0.f) ? (m_r + log2f(l_r)) * LN2 : -INFINITY;
      if (gr8 < M)
        lse_g[(size_t)gr8 * HQ + hq] = (l_r8 > 0.f) ? (m_r8 + log2f(l_r8)) * LN2 : -INFINITY;
    }
  } else {
    const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
    const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      float2 lo, hi;
      lo.x = O[jn][0];
      lo.y = O[jn][1];
      hi.x = O[jn][2];
      hi.y = O[jn][3];
      *reinterpret_cast<float2 *>(&ows[b0 * (D + 4) + c]) = lo;
      *reinterpret_cast<float2 *>(&ows[b8 * (D + 4) + c]) = hi;
    }
    ows[b0 * (D + 4) + D] = l_r;
    ows[b0 * (D + 4) + D + 1] = m_r;
    ows[b8 * (D + 4) + D] = l_r8;
    ows[b8 * (D + 4) + D + 1] = m_r8;
  }
  // PDL: all workspace/output writes of this CTA are complete; allow the
  // dependent merge kernel to start.
  asm volatile("griddepcontrol.launch_dependents;");
}

// ============ 8-warp pair-split kernel: in-CTA G=2 chunk split ============
// Same (tile, head) work assignment as bsa_split_kernel with G==1, but the CTA
// has two 4-warp groups walking the INTERLEAVED admitted-chunk list (group g
// takes chunks g, g+2, g+4, ...). Each group keeps an independent (O, l, m)
// partial over its chunks; after the last chunk the groups combine partials in
// shared memory with the same max-rescaling math as merge_one_row, then group
// A runs the fused coalesced epilogue. No gmem workspace, no merge kernel,
// no PDL handshake: the "split-KV merge overhead" that made host-side G=2
// lose to G=1 on short shapes is paid in ~34KB of smem traffic instead.
// Chunks are staged in windows of nbuf (even) slots: window 0 is armed up
// front, later windows are re-armed at __syncthreads boundaries exactly like
// the split kernel's rolling path, so any nchunks works. Host dispatches
// this path only when the whole worst-case chunk count fits two windows
// (NB*(BS/BN) <= 2*nbuf keeps the overlap benefit dominant).
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(2 * NWARPS * 32)
bsa_pair_kernel(const __grid_constant__ CUtensorMap tq0,
                const __grid_constant__ CUtensorMap tq1,
                const __grid_constant__ CUtensorMap tk0,
                const __grid_constant__ CUtensorMap tk1,
                const __grid_constant__ CUtensorMap tv0,
                const __grid_constant__ CUtensorMap tv1,
                const bool *__restrict__ mask, T *__restrict__ out,
                float *__restrict__ lse_g, float *__restrict__ ows, int M, int N,
                int HQ, int HKV, int BS, int rows_pad, int nbuf, bool normalize,
                float scale_log2e, long long *phbuf) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8 = BN / 8;  // n8 fragments per 64-key chunk
  PHASE_DECL;
  const int ph_cid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;

  extern __shared__ char dyn_smem[];
  // Align tile base to the 1024B swizzle atom.
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);
  T *sK0 = reinterpret_cast<T *>(sbase + BM * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (BM * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));
  // Merge staging for group B partials: [64 rows][D+8] f32 (L/M at c=D, D+1).
  float *sMB = reinterpret_cast<float *>(sAdmit + MAX_BLOCKS);

  constexpr int MAXSTAGE = 9;
  __shared__ uint64_t bar_q, bar_full[MAXSTAGE];
  __shared__ int sWarpCount[2 * NWARPS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int grp = warp >> 2;   // chunk-split group: 0 = even chunks, 1 = odd
  const int rowgrp = warp & 3;  // 16-row query segment, same in both groups
  const int hq = blockIdx.y;
  const int row0 = blockIdx.x * BM;
  const int G = gridDim.z;      // external split count (1 <=> normalize mode)
  const int split = blockIdx.z;
  const int qblk = row0 / BS;
  const int group = HQ / HKV;
  const int kvh = hq / group;
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;

  // ---- barriers up (parallel per-lane inits); the mask row load is issued
  // first so its global latency overlaps the setup below ----
  const uint8_t *mrow =
      reinterpret_cast<const uint8_t *>(mask) + ((size_t)hq * MBm + qblk) * NB;
  bool mrow_pf[8];
  const bool pf_ok = (NB != 4) && (NB <= 256);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = i * 32 + lane;
    mrow_pf[i] = (pf_ok && idx < NB) ? (mrow[idx] != 0) : false;
  }
  uint32_t mbytes_pf = 0;
  if (NB == 4) mbytes_pf = *reinterpret_cast<const uint32_t *>(mrow);
  if (warp == 0) {
    if (lane < nbuf && lane < MAXSTAGE) mbar_init(smem_addr(&bar_full[lane]), 1);
    if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
    __syncwarp();
    if (lane == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), BM);
    }
  }

  PHASE_POINT(phbuf, 0, ph_cid);
  // ---- admitted key blocks for (hq, qblk) ----
  const int CPB = BS / BN;  // 64-key chunks per admitted block
  int nadmit;
  if (NB == 4) {
    const uint32_t mbytes = mbytes_pf;
    const uint32_t m4 = ((mbytes & 0xFFu) ? 1u : 0u) |
                        ((mbytes & 0xFF00u) ? 2u : 0u) |
                        ((mbytes & 0xFF0000u) ? 4u : 0u) |
                        ((mbytes & 0xFF000000u) ? 8u : 0u);
    nadmit = __popc(m4);
    nadmit = nadmit > MAX_BLOCKS ? MAX_BLOCKS : nadmit;
    if (warp == 0 && lane < nadmit) {
      const uint32_t b0 = m4 & (m4 - 1u);
      const uint32_t b1 = b0 & (b0 - 1u);
      const uint32_t b2 = b1 & (b1 - 1u);
      const uint32_t sel = (lane == 0) ? m4 : (lane == 1) ? b0 : (lane == 2) ? b1 : b2;
      sAdmit[lane] = __ffs(sel) - 1;
    }
  } else if (NB <= 256) {
    uint32_t w[8];
    const int nw = (NB + 31) >> 5;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      w[i] = (i < nw) ? __ballot_sync(0xffffffffu, mrow_pf[i]) : 0u;
    if (warp == 0) {
      int running = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i < nw) {
          const uint32_t bal = w[i];
          const int cnt = __popc(bal);
          const int off = __popc(bal & ((1u << lane) - 1));
          const bool adm = (bal >> lane) & 1u;
          if (adm && running + off < MAX_BLOCKS)
            sAdmit[running + off] = i * 32 + lane;
          running += cnt;
        }
      }
      nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
    }
    if (warp != 0) {
      int total = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) total += __popc(w[i]);
      nadmit = total > MAX_BLOCKS ? MAX_BLOCKS : total;
    }
  } else {
    int running = 0;
    for (int base = 0; base < NB; base += 2 * NWARPS * 32) {
      int idx = base + tid;
      bool adm = (idx < NB) && (mrow[idx] != 0);
      unsigned bal = __ballot_sync(0xffffffffu, adm);
      int cnt = __popc(bal);
      int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = cnt;
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < 2 * NWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (adm && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  }
  const int nchunks = nadmit * CPB;
  __shared__ int sPrearm;
  {
    // Pre-arm window 0 before the sAdmit __syncthreads (NB == 4 fast path
    // only): warp 0 already holds the admit list in registers, so K/V TMA
    // fetches for the consumed chunk range launch here and their fill
    // latency overlaps the barrier + setup that follows. All other warps
    // skip the normal arm loop via sPrearm after the syncthreads.
    if (tid == 0) sPrearm = 0;
    if (NB == 4 && warp == 0 && split < nchunks) {
      const int nloc_pa = (nchunks - 1 - split) / G + 1;
      const int w1_pa = nloc_pa < nbuf ? nloc_pa : nbuf;
      __syncwarp();  // sAdmit writes from lanes above are visible to lane 0
      if (lane == 0) {
        sPrearm = 1;
        for (int j = 0; j < w1_pa; ++j) {
          const int col = sAdmit[j / CPB] * BS + (j % CPB) * BN;
          issue_kv<D>(tk0, tk1, tv0, tv1,
                      smem_addr(sK0 + (size_t)j * BN * D),
                      smem_addr(sV0 + (size_t)j * BN * D), col, kvh,
                      smem_addr(&bar_full[j]), BN);
        }
      }
    }
  }
  __syncthreads();  // sAdmit visible to all (incl. the TMA-issuing threads)
  PHASE_POINT(phbuf, 1, ph_cid);

  const int myRow = rowgrp * 16 + (lane >> 2);
  const int myCol = (lane & 3) * 2;
  const int gr0 = row0 + myRow;
  const int gr8 = row0 + myRow + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;  // running row max (exp2 domain)

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };

  if (split >= nchunks) {
    // This CTA has no chunks: G==1 => admitted-block list empty (write
    // zeros/-inf LSE directly); G>1 => this split is empty, so it must still
    // publish a zero partial for the merge kernel to read unconditionally
    // (same correctness contract as the split kernel).
    mbar_wait(smem_addr(&bar_q), 0);  // drain the Q fetch before exit
    if (grp == 0) {
      if (normalize) {
        T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
        for (int jn = 0; jn < DN; ++jn) {
          const int c = jn * 8 + myCol;
          if (gr0 < M)
            *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
                *reinterpret_cast<uint32_t *>(z2);
          if (gr8 < M)
            *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
                *reinterpret_cast<uint32_t *>(z2);
        }
        if (WITH_LSE) {
          if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
          if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
        }
      } else {
        const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
        const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
        for (int jn = 0; jn < DN; ++jn) {
          const int c = jn * 8 + myCol;
          ows[b0 * (D + 4) + c] = 0.f;
          ows[b0 * (D + 4) + c + 1] = 0.f;
          ows[b8 * (D + 4) + c] = 0.f;
          ows[b8 * (D + 4) + c + 1] = 0.f;
        }
        ows[b0 * (D + 4) + D] = 0.f;
        ows[b0 * (D + 4) + D + 1] = NEG_INF;
        ows[b8 * (D + 4) + D] = 0.f;
        ows[b8 * (D + 4) + D + 1] = NEG_INF;
      }
    }
    return;
  }

  // ---- arm window 0 now: one warp per slot issues K/V TMA fetches, exactly
  // like the full_stage path of the split kernel. Later windows are armed at
  // the window boundaries inside the consume loop (after __syncthreads, the
  // same refill discipline as the split kernel's rolling path). ----
  // This CTA's local chunk list: local j -> global chunk (split + j*G).
  const int nloc = (nchunks - 1 - split) / G + 1;
  if (!sPrearm) {  // not pre-armed above (long-NB or empty-split paths)
    const int w1 = nloc < nbuf ? nloc : nbuf;
    for (int j = warp; j < w1; j += 2 * NWARPS) {
      if (lane == 0)
        issue_kv<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                    smem_addr(sV0 + (size_t)j * BN * D),
                    chunk_nbase(split + j * G), kvh, smem_addr(&bar_full[j]),
                    BN);
    }
  }

  PHASE_POINT(phbuf, 2, ph_cid);

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  // Hoist the (chunk-invariant) Q fragments (D<=96 only, as in the pack
  // kernel; at D=128 the +32 registers measurably hurt at 8 warps).
  constexpr int DKH = (D <= 96) ? DK : 0;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, BM>(smem_addr(sQ), rowgrp * 16 + (lane & 15),
                                  jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  PHASE_POINT(phbuf, 3, ph_cid);

  float S[NS8][4];
  auto qk_stage = [&](int slot, float (&Sacc)[NS8][4]) {
    T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
    for (int jj = 0; jj < NS8; ++jj) {
#pragma unroll
      for (int e = 0; e < 4; ++e) Sacc[jj][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8; jn += 2) {
        // Two x2 loads whose outputs ARE the B fragments ({Tn,klo, Tn,khi}),
        // avoiding the x4->pair redistribution MOVs.
        uint32_t b0[2], b1[2];
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b0[0], b0[1]);
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b1[0], b1[1]);
        uint32_t qa[4];
        if (DKH > 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
        } else {
          ldsm_x4(tile_elem_addr<D, BM>(smem_addr(sQ),
                                        rowgrp * 16 + (lane & 15),
                                        jc * 16 + (lane >> 4) * 8),
                  qa[0], qa[1], qa[2], qa[3]);
        }
        FragPack<T>::mma(Sacc[jn], qa, b0);
        FragPack<T>::mma(Sacc[jn + 1], qa, b1);
      }
    }
  };

  // ---- windowed consume: chunks [w0, w1) of each window live in slots
  // [0, nbuf); each slot's mbarrier is armed once per window, so slot parity
  // toggles with the window index (wphase = window & 1). nbuf is even, so
  // every window starts on group 0 and the even/odd group interleave is
  // window-invariant. ----
  uint32_t wphase = 0;
  for (int w0 = 0; w0 < nloc; w0 += nbuf, wphase ^= 1) {
    const int w1 = w0 + nbuf < nloc ? w0 + nbuf : nloc;
    if (w0 > 0) {
      __syncthreads();  // previous window fully consumed; safe to refill
      for (int j = w0 + warp; j < w1; j += 2 * NWARPS) {
        if (lane == 0)
          issue_kv<D>(tk0, tk1, tv0, tv1,
                      smem_addr(sK0 + (size_t)(j - w0) * BN * D),
                      smem_addr(sV0 + (size_t)(j - w0) * BN * D),
                      chunk_nbase(split + j * G), kvh,
                      smem_addr(&bar_full[j - w0]), BN);
      }
    }
    for (int j = w0 + grp; j < w1; j += 2) {
      const int slot = j - w0;
      mbar_wait(smem_addr(&bar_full[slot]), wphase);
      qk_stage(slot, S);
      T *sV = sV0 + (size_t)slot * BN * D;
      const int nbase = chunk_nbase(split + j * G);

    if (nbase + BN > N) {  // partial final key block
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        int c0 = nbase + jn * 8 + myCol;
        if (c0 >= N) S[jn][0] = NEG_INF;
        if (c0 + 1 >= N) S[jn][1] = NEG_INF;
        if (c0 >= N) S[jn][2] = NEG_INF;
        if (c0 + 1 >= N) S[jn][3] = NEG_INF;
      }
    }

    // ---- chunk row max (quad-reduced), advance the running max, and
    // rescale previous O/l partials by exp2(m_old - m_new) ----
    float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
    for (int jn = 0; jn < NS8; ++jn) {
      cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
      cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
    }
    cmax0 *= scale_log2e;
    cmax8 *= scale_log2e;
#pragma unroll
    for (int sh = 1; sh <= 2; sh <<= 1) {
      cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
      cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
    }
    const float mnew0 = fmaxf(m_r, cmax0);
    const float mnew8 = fmaxf(m_r8, cmax8);
    if (mnew0 > m_r || mnew8 > m_r8) {
      const float alpha0 = exp2f(m_r - mnew0);  // NEG_INF first time -> 0
      const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
      for (int jj = 0; jj < DN; ++jj) {
        O[jj][0] *= alpha0;
        O[jj][1] *= alpha0;
        O[jj][2] *= alpha8;
        O[jj][3] *= alpha8;
      }
      l_r *= alpha0;
      l_r8 *= alpha8;
      m_r = mnew0;
      m_r8 = mnew8;
    }

    // ---- P = packed ex2(cvt(S*scale - m)); row sums via ones-MMA below ----
    uint32_t pfr[NS8 / 2][4];
#pragma unroll
    for (int jc = 0; jc < NS8 / 2; ++jc) {
      pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][0], scale_log2e, -m_r),
          fmaf(S[2 * jc][1], scale_log2e, -m_r)));
      pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
      pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
          fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
      pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
          fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
          fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
    }

    // ---- P @ V accumulate, plus row sums via P @ 1 (ones-MMA) ----
    float lacc[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
    for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
      for (int jn = 0; jn < DN; jn += 2) {
        uint32_t vq[4];
        ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV), jk * 16 + (lane & 15),
                                            jn * 8 + (lane >> 4) * 8),
                      vq[0], vq[1], vq[2], vq[3]);
        uint32_t b0[2] = {vq[0], vq[1]};
        uint32_t b1[2] = {vq[2], vq[3]};
        FragPack<T>::mma(O[jn], pfr[jk], b0);
        FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
      }
      FragPack<T>::mma(lacc, pfr[jk], ones2);
    }
      l_r += lacc[0];
      l_r8 += lacc[2];
    }
  }

  PHASE_POINT(phbuf, 4, ph_cid);

  // ---- in-CTA merge: group B publishes (O, l, m) to smem; group A folds it
  // into its partials with the merge_one_row max-rescaling combination, then
  // runs the fused epilogue. ----
  if (grp == 1) {
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      *reinterpret_cast<float2 *>(&sMB[myRow * (D + 8) + c]) =
          make_float2(O[jn][0], O[jn][1]);
      *reinterpret_cast<float2 *>(&sMB[(myRow + 8) * (D + 8) + c]) =
          make_float2(O[jn][2], O[jn][3]);
    }
    sMB[myRow * (D + 8) + D] = l_r;
    sMB[myRow * (D + 8) + D + 1] = m_r;
    sMB[(myRow + 8) * (D + 8) + D] = l_r8;
    sMB[(myRow + 8) * (D + 8) + D + 1] = m_r8;
  }
  __syncthreads();

  if (grp == 0) {
    const float mb_l0 = sMB[myRow * (D + 8) + D];
    const float mb_m0 = sMB[myRow * (D + 8) + D + 1];
    const float mb_l8 = sMB[(myRow + 8) * (D + 8) + D];
    const float mb_m8 = sMB[(myRow + 8) * (D + 8) + D + 1];
    const float ms0 = fmaxf(m_r, mb_m0);
    const float ms8 = fmaxf(m_r8, mb_m8);
    // A partial with l == 0 contributes exactly 0 regardless of m.
    const float fa0 = (l_r > 0.f) ? exp2f(m_r - ms0) : 0.f;
    const float fb0 = (mb_l0 > 0.f) ? exp2f(mb_m0 - ms0) : 0.f;
    const float fa8 = (l_r8 > 0.f) ? exp2f(m_r8 - ms8) : 0.f;
    const float fb8 = (mb_l8 > 0.f) ? exp2f(mb_m8 - ms8) : 0.f;
    const float lt0 = l_r * fa0 + mb_l0 * fb0;
    const float lt8 = l_r8 * fa8 + mb_l8 * fb8;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      const float2 b0 = *reinterpret_cast<const float2 *>(&sMB[myRow * (D + 8) + c]);
      const float2 b8 = *reinterpret_cast<const float2 *>(&sMB[(myRow + 8) * (D + 8) + c]);
      O[jn][0] = O[jn][0] * fa0 + b0.x * fb0;
      O[jn][1] = O[jn][1] * fa0 + b0.y * fb0;
      O[jn][2] = O[jn][2] * fa8 + b8.x * fb8;
      O[jn][3] = O[jn][3] * fa8 + b8.y * fb8;
    }
    l_r = lt0;
    l_r8 = lt8;
    m_r = ms0;
    m_r8 = ms8;

    if (normalize) {
      const float inv0 = (l_r > 0.f) ? (1.f / l_r) : 0.f;
      const float inv8 = (l_r8 > 0.f) ? (1.f / l_r8) : 0.f;
      // Stage the normalized rows in smem (chunk staging consumed by both
      // groups now), then write out with fully coalesced 16B stores.
      constexpr int DP = D + 8;  // padded row pitch: 16B-aligned and bank-spread
      T *sOw = reinterpret_cast<T *>(sK0) + (size_t)rowgrp * 16 * DP;
      const int rw0 = (lane >> 2), rw8 = rw0 + 8;
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        T pk[4];
        pk[0] = FragPack<T>::cvt(O[jn][0] * inv0);
        pk[1] = FragPack<T>::cvt(O[jn][1] * inv0);
        pk[2] = FragPack<T>::cvt(O[jn][2] * inv8);
        pk[3] = FragPack<T>::cvt(O[jn][3] * inv8);
        *reinterpret_cast<uint32_t *>(sOw + rw0 * DP + c) =
            *reinterpret_cast<uint32_t *>(pk);
        *reinterpret_cast<uint32_t *>(sOw + rw8 * DP + c) =
            *reinterpret_cast<uint32_t *>(pk + 2);
      }
      __syncwarp();
      constexpr int NR16 = D / 8;  // uint4 per row
#pragma unroll
      for (int i = 0; i < 16 * NR16 / 32; ++i) {
        const int u = i * 32 + lane;
        const int rr = u / NR16;
        const int cc = u - rr * NR16;
        const int gr = row0 + rowgrp * 16 + rr;
        if (gr < M) {
          const uint4 x = *reinterpret_cast<const uint4 *>(
              reinterpret_cast<char *>(sOw) + rr * (DP * 2) + cc * 16);
          *reinterpret_cast<uint4 *>(&out[((size_t)gr * HQ + hq) * D + cc * 8]) = x;
        }
      }
      if (WITH_LSE) {
        if (gr0 < M)
          lse_g[(size_t)gr0 * HQ + hq] = (l_r > 0.f) ? (m_r + log2f(l_r)) * LN2 : -INFINITY;
        if (gr8 < M)
          lse_g[(size_t)gr8 * HQ + hq] = (l_r8 > 0.f) ? (m_r8 + log2f(l_r8)) * LN2 : -INFINITY;
      }
    } else {
      // G>1: publish the merged (over both in-CTA groups) partial for this
      // external split; the merge kernel combines across splits.
      const size_t b0 = ((size_t)split * rows_pad + (size_t)gr0) * HQ + hq;
      const size_t b8 = ((size_t)split * rows_pad + (size_t)gr8) * HQ + hq;
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        float2 lo, hi;
        lo.x = O[jn][0];
        lo.y = O[jn][1];
        hi.x = O[jn][2];
        hi.y = O[jn][3];
        *reinterpret_cast<float2 *>(&ows[b0 * (D + 4) + c]) = lo;
        *reinterpret_cast<float2 *>(&ows[b8 * (D + 4) + c]) = hi;
      }
      ows[b0 * (D + 4) + D] = l_r;
      ows[b0 * (D + 4) + D + 1] = m_r;
      ows[b8 * (D + 4) + D] = l_r8;
      ows[b8 * (D + 4) + D + 1] = m_r8;
    }
  }
  PHASE_POINT(phbuf, 5, ph_cid);
  PHASE_FLUSH(phbuf, ph_cid);
  // PDL: all partial/output writes of this CTA are complete; allow the
  // dependent merge kernel to start (pairs with griddepcontrol.wait there).
  asm volatile("griddepcontrol.launch_dependents;");
}

// ============ pair32: 32-row pair-split kernel, FOUR in-CTA chunk groups ====
// Same 8-warp budget as the pair kernel but the CTA tiles 32 query rows and
// walks the admitted chunks with four 2-warp groups (group g takes chunks
// g, g+4, ...). Rationale (runtime-metric dispatched): when the BM=64 grid
// underfills the device (e.g. 16 or 32 CTAs on 148 SMs), halving the row tile
// doubles the CTA count AND quarters the per-CTA serial chunk chain at no
// extra gmem cost (merge stays in smem). A CTA's chunk chain is
// ceil(nchunks/4) vs ceil(nchunks/2) for the pair kernel; merge folds up to
// three staged group partials instead of one. Normalize path only (no ows,
// no external splits): underfill is a small-problem condition, and small
// problems run at G==1.
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(2 * NWARPS * 32)
bsa_pair32_kernel(const __grid_constant__ CUtensorMap tq0,
                  const __grid_constant__ CUtensorMap tq1,
                  const __grid_constant__ CUtensorMap tk0,
                  const __grid_constant__ CUtensorMap tk1,
                  const __grid_constant__ CUtensorMap tv0,
                  const __grid_constant__ CUtensorMap tv1,
                  const bool *__restrict__ mask, T *__restrict__ out,
                  float *__restrict__ lse_g, int M, int N, int HQ, int HKV,
                  int BS, int nbuf, float scale_log2e, long long *phbuf) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8 = BN / 8;  // n8 fragments per 64-key chunk
  constexpr int ROWS = 32;     // CTA query rows
  constexpr int GP = 4;        // in-CTA chunk groups
  PHASE_DECL;
  const int ph_cid = blockIdx.y * gridDim.x + blockIdx.x;

  extern __shared__ char dyn_smem[];
  // Align tile base to the 1024B swizzle atom.
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);
  T *sK0 = reinterpret_cast<T *>(sbase + ROWS * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (ROWS * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));
  // Merge staging for the (GP) group partials: GP x [32 rows][D+8] f32.
  // Dedicated budget: staging must not wait for a K/V-area barrier.
  float *sMB = reinterpret_cast<float *>(sAdmit + MAX_BLOCKS);

  constexpr int MAXSTAGE = 9;
  // Split barrier sets per K/V slot: warps 2-5 own the K side (bar_k), the
  // otherwise-idle warp 6 owns the V side (bar_v), so prologue arms and their
  // tensormap first-touch reads overlap across warps.
  __shared__ uint64_t bar_q, bar_k[MAXSTAGE], bar_v[MAXSTAGE];
  __shared__ int sWarpCount[2 * NWARPS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int grp = warp >> 1;   // chunk-split group: chunks g, g+GP, ...
  const int rowgrp = warp & 1;  // 16-row query segment, same in all groups
  const int hq = blockIdx.y;
  const int row0 = blockIdx.x * ROWS;
  const int qblk = row0 / BS;
  const int group = HQ / HKV;
  const int kvh = hq / group;
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;

  // ---- parallel prologue (phase-probe: P0+P1 was 3.2k cycles when warp 0
  // ran the whole serial chain). Warp 0 owns the Q barrier + Q TMA issue;
  // warp 1 owns the mask scan + admit list; warps 2-5 own the bar_full stripe
  // inits AND the window-0 K/V pre-arm (barrier j belongs to warp 2 + (j & 3),
  // so init/fence/arm ordering is intra-thread and the arms run concurrently
  // with warp 1's scan). All other warps proceed straight to the barrier. ----
  const uint8_t *mrow =
      reinterpret_cast<const uint8_t *>(mask) +
      (size_t)(((unsigned)hq * (unsigned)MBm + (unsigned)qblk) * (unsigned)NB);
  // Mask word prefetch at kernel entry: the ~650-cycle L2 latency of this
  // single broadcast load otherwise sits entirely on the admit->pre-arm
  // chain. Every lane loads the same word (one sector), so this is one
  // global request whose latency now overlaps the setup below.
  const uint32_t mbytes_pf =
      (NB == 4) ? *reinterpret_cast<const uint32_t *>(mrow) : 0u;
  const int CPB = BS / BN;  // 64-key chunks per admitted block
  // Admit bits for the NB == 4 fast path, computed by every thread (pure
  // ALU on the prefetched word) so the pre-arm warps need no smem handshake.
  const uint32_t m4 = (NB == 4) ? (((mbytes_pf & 0xFFu) ? 1u : 0u) |
                                   ((mbytes_pf & 0xFF00u) ? 2u : 0u) |
                                   ((mbytes_pf & 0xFF0000u) ? 4u : 0u) |
                                   ((mbytes_pf & 0xFF000000u) ? 8u : 0u))
                                : 0u;
  __shared__ int sNadmit;
  int nadmit;  // set below: register path for NB > 256, smem path otherwise
  if (NB <= 256) {
    if (warp == 0) {
      if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
      __syncwarp();
      if (lane == 31) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), ROWS);
      }
    } else if (warp == 1) {
      // Admit scan only. Barrier init and the window-0 pre-arm moved to warps
      // 2-5 (cross-warp distributed below), so the TMA-issue chain no longer
      // sits behind this scan on the P0 critical path.
      int nad;
      if (NB == 4) {
        nad = __popc(m4);  // hoisted mask word: at most 4 bits, no cap needed
        if (lane < nad) {
          const uint32_t b0 = m4 & (m4 - 1u);
          const uint32_t b1 = b0 & (b0 - 1u);
          const uint32_t b2 = b1 & (b1 - 1u);
          const uint32_t sel =
              (lane == 0) ? m4 : (lane == 1) ? b0 : (lane == 2) ? b1 : b2;
          sAdmit[lane] = __ffs(sel) - 1;
        }
      } else {
        uint32_t w[8];
        const int nw = (NB + 31) >> 5;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int idx = i * 32 + lane;
          const bool adm = (i < nw && idx < NB) ? (mrow[idx] != 0) : false;
          w[i] = __ballot_sync(0xffffffffu, adm);
        }
        int running = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (i < nw) {
            const uint32_t bal = w[i];
            const int cnt = __popc(bal);
            const int off = __popc(bal & ((1u << lane) - 1));
            const bool adm = (bal >> lane) & 1u;
            if (adm && running + off < MAX_BLOCKS)
              sAdmit[running + off] = i * 32 + lane;
            running += cnt;
          }
        }
        nad = running > MAX_BLOCKS ? MAX_BLOCKS : running;
      }
      if (lane == 0) sNadmit = nad;
    } else if (warp <= 5) {
      // Cross-warp distributed prologue: K side. Barrier j (K half) is owned
      // by warp 2 + (j & 3), lane (j >> 2); that thread inits, fences, and
      // arms it, so all ordering is intra-thread program order. Chunk
      // coordinates come from the entry-prefetched mask word (m4), NOT from
      // sAdmit, so the K TMA issues do not wait on warp 1's scan.
      const int w = warp - 2;
      const int j = w + 4 * lane;
      if (lane < 3 && j < nbuf && j < MAXSTAGE)
        mbar_init(smem_addr(&bar_k[j]), 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (NB == 4) {
        const int nch = __popc(m4) * CPB;
        const int w1_pa = nch < nbuf ? nch : nbuf;
        if (lane < 3 && j < w1_pa) {
          uint32_t sel = m4;
          int t = j / CPB;
          while (t--) sel &= sel - 1u;
          const int col = (__ffs(sel) - 1) * BS + (j % CPB) * BN;
          issue_ks<D>(tk0, tk1, smem_addr(sK0 + (size_t)j * BN * D), col, kvh,
                      smem_addr(&bar_k[j]), BN);
        }
      }
    } else if (warp == 6) {
      // V side of the distributed prologue, on the otherwise-idle warp: lane
      // j inits bar_v[j] and arms chunk j's V fetch. Running the V arms on
      // their own warp lets the tv0 tensormap first-touch overlap warps 2-5's
      // tk0 first-touch instead of chaining behind it. All ordering here is
      // intra-thread.
      const int j = lane;
      if (lane < nbuf && lane < MAXSTAGE)
        mbar_init(smem_addr(&bar_v[j]), 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (NB == 4) {
        const int nch = __popc(m4) * CPB;
        const int w1_pa = nch < nbuf ? nch : nbuf;
        if (lane < w1_pa && lane < 32) {
          uint32_t sel = m4;
          int t = j / CPB;
          while (t--) sel &= sel - 1u;
          const int col = (__ffs(sel) - 1) * BS + (j % CPB) * BN;
          issue_vs<D>(tv0, tv1, smem_addr(sV0 + (size_t)j * BN * D), col, kvh,
                      smem_addr(&bar_v[j]), BN);
        }
      }
    }
    __syncthreads();  // sAdmit/sNadmit visible to all
    nadmit = sNadmit;
  } else {
    // Wide scan (NB > 256): all warps sweep the row cooperatively; barrier
    // init + Q issue land on warp 0 first so the sweep setup is unchanged.
    if (warp == 0) {
      if (lane < nbuf && lane < MAXSTAGE) {
        mbar_init(smem_addr(&bar_k[lane]), 1);
        mbar_init(smem_addr(&bar_v[lane]), 1);
      }
      if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
      __syncwarp();
      if (lane == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), ROWS);
      }
    }
    int running = 0;
    for (int base = 0; base < NB; base += 2 * NWARPS * 32) {
      int idx = base + tid;
      bool adm = (idx < NB) && (mrow[idx] != 0);
      unsigned bal = __ballot_sync(0xffffffffu, adm);
      int cnt = __popc(bal);
      int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = cnt;
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < 2 * NWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (adm && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  }
  // Local chunk list == the global list (external split count is 1 here).
  const int nchunks = nadmit * CPB;
  PHASE_POINT(phbuf, 0, ph_cid);
  PHASE_POINT(phbuf, 1, ph_cid);

  const int myRow = rowgrp * 16 + (lane >> 2);
  const int myCol = (lane & 3) * 2;
  const int gr0 = row0 + myRow;
  const int gr8 = row0 + myRow + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;  // running row max (exp2 domain)

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };

  if (nchunks == 0) {
    // Empty admitted list: group 0 writes zeros/-inf LSE directly (uniform
    // exit for the whole CTA; no barrier below is reached by anyone).
    mbar_wait(smem_addr(&bar_q), 0);  // drain the Q fetch before exit
    if (grp == 0) {
      T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        if (gr0 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
        if (gr8 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
      }
      if (WITH_LSE) {
        if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
        if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
      }
    }
    return;
  }

  // ---- arm window 0 now unless pre-armed by warps 2-5 (NB != 4 paths only):
  // one warp per slot issues its K/V TMA fetches. Later windows are armed at
  // the window boundaries (same refill discipline). ----
  if (NB != 4) {
    const int w1 = nchunks < nbuf ? nchunks : nbuf;
    for (int j = warp; j < w1; j += 2 * NWARPS) {
      if (lane == 0)
        issue_kv2<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                     smem_addr(sV0 + (size_t)j * BN * D), chunk_nbase(j), kvh,
                     smem_addr(&bar_k[j]), smem_addr(&bar_v[j]), BN);
    }
  }

  PHASE_POINT(phbuf, 2, ph_cid);

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  // Hoist the (chunk-invariant) Q fragments (D<=96 only; at D=128 the +32
  // registers measurably hurt at 8 warps — re-measured on the 219KB build:
  // P4 +550 cycles with the hoist, so it stays off for D=128).
  constexpr int DKH = (D <= 96) ? DK : 0;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, ROWS>(smem_addr(sQ), rowgrp * 16 + (lane & 15),
                                    jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  PHASE_POINT(phbuf, 3, ph_cid);

  float S[NS8][4];
  auto qk_stage = [&](int slot, float (&Sacc)[NS8][4]) {
    T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
    for (int jj = 0; jj < NS8; ++jj) {
#pragma unroll
      for (int e = 0; e < 4; ++e) Sacc[jj][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8; jn += 2) {
        // Two x2 loads whose outputs ARE the B fragments ({Tn,klo, Tn,khi}),
        // avoiding the x4->pair redistribution MOVs.
        uint32_t b0[2], b1[2];
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b0[0], b0[1]);
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK), jn * 8 + 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b1[0], b1[1]);
        uint32_t qa[4];
        if (DKH > 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
        } else {
          ldsm_x4(tile_elem_addr<D, ROWS>(smem_addr(sQ),
                                          rowgrp * 16 + (lane & 15),
                                          jc * 16 + (lane >> 4) * 8),
                  qa[0], qa[1], qa[2], qa[3]);
        }
        FragPack<T>::mma(Sacc[jn], qa, b0);
        FragPack<T>::mma(Sacc[jn + 1], qa, b1);
      }
    }
  };

  // ---- windowed consume: chunks [w0, w1) of each window live in slots
  // [0, nbuf); each slot's mbarrier is armed once per window, so slot parity
  // toggles with the window index (wphase = window & 1). nbuf is a multiple
  // of GP, so every window starts on group 0 and the mod-GP group interleave
  // is window-invariant. ----
  uint32_t wphase = 0;
  for (int w0 = 0; w0 < nchunks; w0 += nbuf, wphase ^= 1) {
    const int w1 = w0 + nbuf < nchunks ? w0 + nbuf : nchunks;
    if (w0 > 0) {
      __syncthreads();  // previous window fully consumed; safe to refill
      for (int j = w0 + warp; j < w1; j += 2 * NWARPS) {
        if (lane == 0)
          issue_kv2<D>(tk0, tk1, tv0, tv1,
                       smem_addr(sK0 + (size_t)(j - w0) * BN * D),
                       smem_addr(sV0 + (size_t)(j - w0) * BN * D),
                       chunk_nbase(j), kvh, smem_addr(&bar_k[j - w0]),
                       smem_addr(&bar_v[j - w0]), BN);
      }
    }
    for (int j = w0 + grp; j < w1; j += GP) {
      const int slot = j - w0;
      mbar_wait(smem_addr(&bar_k[slot]), wphase);
      mbar_wait(smem_addr(&bar_v[slot]), wphase);
      qk_stage(slot, S);
      T *sV = sV0 + (size_t)slot * BN * D;
      const int nbase = chunk_nbase(j);

      if (nbase + BN > N) {  // partial final key block
#pragma unroll
        for (int jn = 0; jn < NS8; ++jn) {
          int c0 = nbase + jn * 8 + myCol;
          if (c0 >= N) S[jn][0] = NEG_INF;
          if (c0 + 1 >= N) S[jn][1] = NEG_INF;
          if (c0 >= N) S[jn][2] = NEG_INF;
          if (c0 + 1 >= N) S[jn][3] = NEG_INF;
        }
      }

      // ---- chunk row max (quad-reduced), advance the running max, rescale --
      float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
      for (int jn = 0; jn < NS8; ++jn) {
        cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
        cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
      }
      cmax0 *= scale_log2e;
      cmax8 *= scale_log2e;
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
        cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
      }
      const float mnew0 = fmaxf(m_r, cmax0);
      const float mnew8 = fmaxf(m_r8, cmax8);
      if (j == grp) {
        // First chunk in this group's chain: O and l are still zero and m is
        // -inf, so the rescale below is provably a no-op (alpha == 0 and it
        // only scales zeros). Skipping it removes 2 MUFU + ~24 FMUL from the
        // one-chunk-per-group path (all fixed-suite pair32 cases).
        m_r = mnew0;
        m_r8 = mnew8;
      } else if (mnew0 > m_r || mnew8 > m_r8) {
        const float alpha0 = exp2f(m_r - mnew0);
        const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
        for (int jj = 0; jj < DN; ++jj) {
          O[jj][0] *= alpha0;
          O[jj][1] *= alpha0;
          O[jj][2] *= alpha8;
          O[jj][3] *= alpha8;
        }
        l_r *= alpha0;
        l_r8 *= alpha8;
        m_r = mnew0;
        m_r8 = mnew8;
      }

      // ---- P = packed ex2(cvt(S*scale - m)); row sums via ones-MMA below --
      uint32_t pfr[NS8 / 2][4];
#pragma unroll
      for (int jc = 0; jc < NS8 / 2; ++jc) {
        pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][0], scale_log2e, -m_r),
            fmaf(S[2 * jc][1], scale_log2e, -m_r)));
        pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
        pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
            fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
        pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
      }

      // ---- P @ V accumulate, plus row sums via P @ 1 (ones-MMA) ----
      float lacc[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
      for (int jk = 0; jk < NS8 / 2; ++jk) {
#pragma unroll
        for (int jn = 0; jn < DN; jn += 2) {
          uint32_t vq[4];
          ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV), jk * 16 + (lane & 15),
                                              jn * 8 + (lane >> 4) * 8),
                        vq[0], vq[1], vq[2], vq[3]);
          uint32_t b0[2] = {vq[0], vq[1]};
          uint32_t b1[2] = {vq[2], vq[3]};
          FragPack<T>::mma(O[jn], pfr[jk], b0);
          FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
        }
        FragPack<T>::mma(lacc, pfr[jk], ones2);
      }
      l_r += lacc[0];
      l_r8 += lacc[2];
    }
  }

  PHASE_POINT(phbuf, 4, ph_cid);

  // ---- in-CTA merge (distributed all-fold): the K/V chunk staging area is
  // dead once every group finishes its chunk chain, so the (GP) partial
  // slices alias it after a full-CTA barrier. EVERY group (all 8 warps) then
  // folds ALL published slices over its own 8-col column band of width
  // DN/GP — the fold walks slices in ascending order from an empty
  // accumulator, which reproduces the old group-0 fold bit-exactly — and
  // stores its band straight to global from registers (no smem restage).
  // Dedicated fp32 slices in their own smem budget (no K/V aliasing), so a
  // group stages its partial the moment its own chunk chain ends — no
  // CTA-wide barrier before staging; a single rendezvous sync afterwards.
  // fp16 staging was measured slower (cvt ALU cost > MIO wavefront savings).
  constexpr int DNB = DN / GP;  // 8-col fragments per group band (D/GP/8)
  constexpr int DPB = D + 8;    // padded slice row pitch (floats)
  float *sMBd = sMB;            // dedicated (GP) x [32 rows][D+8] f32 slices
  const int nact = nchunks < GP ? nchunks : GP;
  if (grp < nact) {
    float *sg = sMBd + (size_t)grp * ROWS * DPB;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      *reinterpret_cast<float2 *>(&sg[myRow * DPB + c]) =
          make_float2(O[jn][0], O[jn][1]);
      *reinterpret_cast<float2 *>(&sg[(myRow + 8) * DPB + c]) =
          make_float2(O[jn][2], O[jn][3]);
    }
    sg[myRow * DPB + D] = l_r;
    sg[myRow * DPB + D + 1] = m_r;
    sg[(myRow + 8) * DPB + D] = l_r8;
    sg[(myRow + 8) * DPB + D + 1] = m_r8;
  }
  __syncthreads();              // all slices published
  PHASE_POINT(phbuf, 6, ph_cid);

  {
    const int rw0 = rowgrp * 16 + (lane >> 2), rw8 = rw0 + 8;
    const int gr0l = row0 + rw0;
    const int gr8l = gr0l + 8;
    float lt0 = 0.f, lt8 = 0.f;
    float ms0 = NEG_INF, ms8 = NEG_INF;
    {
      float Oa[DNB][4];
#pragma unroll
      for (int j = 0; j < DNB; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) Oa[j][e] = 0.f;
      }
#pragma unroll
      for (int g = 0; g < GP; ++g) {
        if (g >= nact) break;
        const float *sg = sMBd + (size_t)g * ROWS * DPB;
        // (l, m) sit adjacent at columns D, D+1: one LDS.64 per row instead
        // of two LDS.32 (8B-aligned: DPB*4 is a multiple of 8 for all D).
        const float2 lm0 =
            *reinterpret_cast<const float2 *>(&sg[rw0 * DPB + D]);
        const float2 lm8 =
            *reinterpret_cast<const float2 *>(&sg[rw8 * DPB + D]);
        const float bl0 = lm0.x;
        const float bm0 = lm0.y;
        const float bl8 = lm8.x;
        const float bm8 = lm8.y;
        const float ns0 = fmaxf(ms0, bm0);
        const float ns8 = fmaxf(ms8, bm8);
        const float fa0 = (lt0 > 0.f) ? exp2f(ms0 - ns0) : 0.f;
        const float fb0 = (bl0 > 0.f) ? exp2f(bm0 - ns0) : 0.f;
        const float fa8 = (lt8 > 0.f) ? exp2f(ms8 - ns8) : 0.f;
        const float fb8 = (bl8 > 0.f) ? exp2f(bm8 - ns8) : 0.f;
#pragma unroll
        for (int jj = 0; jj < DNB; ++jj) {
          const int c = (grp * DNB + jj) * 8 + myCol;
          const float2 b0 =
              *reinterpret_cast<const float2 *>(&sg[rw0 * DPB + c]);
          const float2 b8 =
              *reinterpret_cast<const float2 *>(&sg[rw8 * DPB + c]);
          Oa[jj][0] = Oa[jj][0] * fa0 + b0.x * fb0;
          Oa[jj][1] = Oa[jj][1] * fa0 + b0.y * fb0;
          Oa[jj][2] = Oa[jj][2] * fa8 + b8.x * fb8;
          Oa[jj][3] = Oa[jj][3] * fa8 + b8.y * fb8;
        }
        lt0 = lt0 * fa0 + bl0 * fb0;
        lt8 = lt8 * fa8 + bl8 * fb8;
        ms0 = ns0;
        ms8 = ns8;
      }
      const float inv0 = (lt0 > 0.f) ? (1.f / lt0) : 0.f;
      const float inv8 = (lt8 > 0.f) ? (1.f / lt8) : 0.f;
#pragma unroll
      for (int jj = 0; jj < DNB; ++jj) {
        const int c = (grp * DNB + jj) * 8 + myCol;
        if (gr0l < M) {
          T pk[2] = {FragPack<T>::cvt(Oa[jj][0] * inv0),
                     FragPack<T>::cvt(Oa[jj][1] * inv0)};
          *reinterpret_cast<uint32_t *>(
              &out[((size_t)gr0l * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(pk);
        }
        if (gr8l < M) {
          T pk[2] = {FragPack<T>::cvt(Oa[jj][2] * inv8),
                     FragPack<T>::cvt(Oa[jj][3] * inv8)};
          *reinterpret_cast<uint32_t *>(
              &out[((size_t)gr8l * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(pk);
        }
      }
    }
    if (WITH_LSE && grp == 0) {
      if (gr0l < M)
        lse_g[(size_t)gr0l * HQ + hq] =
            (lt0 > 0.f) ? (ms0 + log2f(lt0)) * LN2 : -INFINITY;
      if (gr8l < M)
        lse_g[(size_t)gr8l * HQ + hq] =
            (lt8 > 0.f) ? (ms8 + log2f(lt8)) * LN2 : -INFINITY;
    }
  }
  PHASE_POINT(phbuf, 5, ph_cid);
  PHASE_FLUSH(phbuf, ph_cid);
}

// ============ bm16: 16-row kernel, EIGHT half-chunk partials ===============
// Deeper underfill specialization of pair32 (runtime-gated, normalize only):
// when even the pair32 grid leaves most SMs idle, halve the row tile again
// (16 rows) AND split every admitted 64-key chunk between TWO warps — warp w
// takes local chunk (w >> 1), 32-key half (w & 1). Per-warp serial chain is
// HALF a 64-key chunk (vs a full chunk in pair32) and the grid doubles once
// more (M=256, HQ=8 -> 128 CTAs on 148 SMs). Each warp publishes one
// (O, l, m) partial slice; all slices are merged once in the epilogue with a
// row-split fold (warp w owns output rows 2w, 2w+1), so every warp folds at
// every D (no idle warps in the merge for D < 128). Stage+fold byte volume is
// identical to pair32's (~8 slices x 16 rows == ~4 slices x 32 rows), so the
// LDS-bound fold cost is unchanged while the per-warp chunk chain halves.
template <typename T, int D, bool WITH_LSE>
__global__ void __launch_bounds__(2 * NWARPS * 32)
bsa_bm16_kernel(const __grid_constant__ CUtensorMap tq0,
                const __grid_constant__ CUtensorMap tq1,
                const __grid_constant__ CUtensorMap tk0,
                const __grid_constant__ CUtensorMap tk1,
                const __grid_constant__ CUtensorMap tv0,
                const __grid_constant__ CUtensorMap tv1,
                const bool *__restrict__ mask, T *__restrict__ out,
                float *__restrict__ lse_g, int M, int N, int HQ, int HKV,
                int BS, int nbuf, float scale_log2e, long long *phbuf) {
  constexpr int DK = D / 16;
  constexpr int DN = D / 8;
  constexpr int NS8H = BN / 16;  // n8 fragments per 32-key half-chunk (4)
  constexpr int ROWS = 16;       // CTA query rows
  constexpr int NPART = 8;       // partial slices (one per warp)
  PHASE_DECL;
  const int ph_cid = blockIdx.y * gridDim.x + blockIdx.x;

  extern __shared__ char dyn_smem[];
  const uint32_t dynu = smem_addr(dyn_smem);
  char *sbase = dyn_smem + (((dynu + 1023u) & ~1023u) - dynu);
  T *sQ = reinterpret_cast<T *>(sbase);
  T *sK0 = reinterpret_cast<T *>(sbase + ROWS * D * 2);
  T *sV0 = sK0 + (size_t)nbuf * BN * D;
  int *sAdmit = reinterpret_cast<int *>(
      sbase + (ROWS * D + (size_t)2 * nbuf * BN * D) * (int)sizeof(T));
  // Dedicated partial staging (never aliases K/V): NPART x [16 rows][D+8] f32.
  float *sMB = reinterpret_cast<float *>(sAdmit + MAX_BLOCKS);

  constexpr int MAXSTAGE = 9;
  __shared__ uint64_t bar_q, bar_k[MAXSTAGE], bar_v[MAXSTAGE];
  __shared__ int sWarpCount[2 * NWARPS];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hq = blockIdx.y;
  const int row0 = blockIdx.x * ROWS;
  const int qblk = row0 / BS;
  const int group = HQ / HKV;
  const int kvh = hq / group;
  const int MBm = (M + BS - 1) / BS;
  const int NB = (N + BS - 1) / BS;

  // ---- distributed prologue (same roles as pair32: warp 0 Q, warp 1 admit
  // scan, warps 2-5 K barrier inits + window-0 K arms, warp 6 the V side) ----
  const uint8_t *mrow =
      reinterpret_cast<const uint8_t *>(mask) +
      (size_t)(((unsigned)hq * (unsigned)MBm + (unsigned)qblk) * (unsigned)NB);
  const uint32_t mbytes_pf =
      (NB == 4) ? *reinterpret_cast<const uint32_t *>(mrow) : 0u;
  const int CPB = BS / BN;
  const uint32_t m4 = (NB == 4) ? (((mbytes_pf & 0xFFu) ? 1u : 0u) |
                                   ((mbytes_pf & 0xFF00u) ? 2u : 0u) |
                                   ((mbytes_pf & 0xFF0000u) ? 4u : 0u) |
                                   ((mbytes_pf & 0xFF000000u) ? 8u : 0u))
                                : 0u;
  __shared__ int sNadmit;
  int nadmit;
  if (NB <= 256) {
    if (warp == 0) {
      if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
      __syncwarp();
      if (lane == 31) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), ROWS);
      }
    } else if (warp == 1) {
      int nad;
      if (NB == 4) {
        nad = __popc(m4);
        if (lane < nad) {
          const uint32_t b0 = m4 & (m4 - 1u);
          const uint32_t b1 = b0 & (b0 - 1u);
          const uint32_t b2 = b1 & (b1 - 1u);
          const uint32_t sel =
              (lane == 0) ? m4 : (lane == 1) ? b0 : (lane == 2) ? b1 : b2;
          sAdmit[lane] = __ffs(sel) - 1;
        }
      } else {
        uint32_t w[8];
        const int nw = (NB + 31) >> 5;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int idx = i * 32 + lane;
          const bool adm = (i < nw && idx < NB) ? (mrow[idx] != 0) : false;
          w[i] = __ballot_sync(0xffffffffu, adm);
        }
        int running = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (i < nw) {
            const uint32_t bal = w[i];
            const int cnt = __popc(bal);
            const int off = __popc(bal & ((1u << lane) - 1));
            const bool adm = (bal >> lane) & 1u;
            if (adm && running + off < MAX_BLOCKS)
              sAdmit[running + off] = i * 32 + lane;
            running += cnt;
          }
        }
        nad = running > MAX_BLOCKS ? MAX_BLOCKS : running;
      }
      if (lane == 0) sNadmit = nad;
    } else if (warp <= 5) {
      const int w = warp - 2;
      const int j = w + 4 * lane;
      if (lane < 3 && j < nbuf && j < MAXSTAGE)
        mbar_init(smem_addr(&bar_k[j]), 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (NB == 4) {
        const int nch = __popc(m4) * CPB;
        const int w1_pa = nch < nbuf ? nch : nbuf;
        if (lane < 3 && j < w1_pa) {
          uint32_t sel = m4;
          int t = j / CPB;
          while (t--) sel &= sel - 1u;
          const int col = (__ffs(sel) - 1) * BS + (j % CPB) * BN;
          issue_ks<D>(tk0, tk1, smem_addr(sK0 + (size_t)j * BN * D), col, kvh,
                      smem_addr(&bar_k[j]), BN);
        }
      }
    } else if (warp == 6) {
      const int j = lane;
      if (lane < nbuf && lane < MAXSTAGE)
        mbar_init(smem_addr(&bar_v[j]), 1);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (NB == 4) {
        const int nch = __popc(m4) * CPB;
        const int w1_pa = nch < nbuf ? nch : nbuf;
        if (lane < w1_pa && lane < 32) {
          uint32_t sel = m4;
          int t = j / CPB;
          while (t--) sel &= sel - 1u;
          const int col = (__ffs(sel) - 1) * BS + (j % CPB) * BN;
          issue_vs<D>(tv0, tv1, smem_addr(sV0 + (size_t)j * BN * D), col, kvh,
                      smem_addr(&bar_v[j]), BN);
        }
      }
    }
    __syncthreads();  // sAdmit/sNadmit visible to all
    nadmit = sNadmit;
  } else {
    if (warp == 0) {
      if (lane < nbuf && lane < MAXSTAGE) {
        mbar_init(smem_addr(&bar_k[lane]), 1);
        mbar_init(smem_addr(&bar_v[lane]), 1);
      }
      if (lane == 31) mbar_init(smem_addr(&bar_q), 1);
      __syncwarp();
      if (lane == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        issue_q<D>(tq0, tq1, smem_addr(sQ), hq, row0, smem_addr(&bar_q), ROWS);
      }
    }
    int running = 0;
    for (int base = 0; base < NB; base += 2 * NWARPS * 32) {
      int idx = base + tid;
      bool adm = (idx < NB) && (mrow[idx] != 0);
      unsigned bal = __ballot_sync(0xffffffffu, adm);
      int cnt = __popc(bal);
      int off = __popc(bal & ((1u << lane) - 1));
      sWarpCount[warp] = cnt;
      __syncthreads();
      int wbase = running;
      for (int w = 0; w < warp; ++w) wbase += sWarpCount[w];
      int total = 0;
      for (int w = 0; w < 2 * NWARPS; ++w) total += sWarpCount[w];
      __syncthreads();
      if (adm && wbase + off < MAX_BLOCKS) sAdmit[wbase + off] = idx;
      running += total;
    }
    nadmit = running > MAX_BLOCKS ? MAX_BLOCKS : running;
  }
  const int nchunks = nadmit * CPB;
  PHASE_POINT(phbuf, 0, ph_cid);
  PHASE_POINT(phbuf, 1, ph_cid);

  const int myRow = lane >> 2;
  const int myCol = (lane & 3) * 2;
  const int gr0 = row0 + myRow;
  const int gr8 = row0 + myRow + 8;
  float l_r = 0.f, l_r8 = 0.f;
  float m_r = NEG_INF, m_r8 = NEG_INF;

  auto chunk_nbase = [&](int ci) { return sAdmit[ci / CPB] * BS + (ci % CPB) * BN; };

  if (nchunks == 0) {
    mbar_wait(smem_addr(&bar_q), 0);
    if (warp == 0) {
      T z2[2] = {FragPack<T>::cvt(0.f), FragPack<T>::cvt(0.f)};
#pragma unroll
      for (int jn = 0; jn < DN; ++jn) {
        const int c = jn * 8 + myCol;
        if (gr0 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr0 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
        if (gr8 < M)
          *reinterpret_cast<uint32_t *>(&out[((size_t)gr8 * HQ + hq) * D + c]) =
              *reinterpret_cast<uint32_t *>(z2);
      }
      if (WITH_LSE) {
        if (gr0 < M) lse_g[(size_t)gr0 * HQ + hq] = -INFINITY;
        if (gr8 < M) lse_g[(size_t)gr8 * HQ + hq] = -INFINITY;
      }
    }
    return;
  }

  if (NB != 4) {
    const int w1 = nchunks < nbuf ? nchunks : nbuf;
    for (int j = warp; j < w1; j += 2 * NWARPS) {
      if (lane == 0)
        issue_kv2<D>(tk0, tk1, tv0, tv1, smem_addr(sK0 + (size_t)j * BN * D),
                     smem_addr(sV0 + (size_t)j * BN * D), chunk_nbase(j), kvh,
                     smem_addr(&bar_k[j]), smem_addr(&bar_v[j]), BN);
    }
  }

  PHASE_POINT(phbuf, 2, ph_cid);

  float O[DN][4];
#pragma unroll
  for (int j = 0; j < DN; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) O[j][e] = 0.f;
  }

  mbar_wait(smem_addr(&bar_q), 0);

  constexpr int DKH = (D <= 96) ? DK : 0;
  uint32_t qa_r[DKH > 0 ? DKH : 1][4];
#pragma unroll
  for (int jc = 0; jc < DKH; ++jc) {
    ldsm_x4(tile_elem_addr<D, ROWS>(smem_addr(sQ), (lane & 15),
                                    jc * 16 + (lane >> 4) * 8),
            qa_r[jc][0], qa_r[jc][1], qa_r[jc][2], qa_r[jc][3]);
  }

  PHASE_POINT(phbuf, 3, ph_cid);

  const int jloc = warp >> 1;  // local 64-key chunk index within a window
  const int hw = warp & 1;     // 32-key half of that chunk
  float S[NS8H][4];
  auto qk_half = [&](int slot) {
    T *sK = sK0 + (size_t)slot * BN * D;
#pragma unroll
    for (int jj = 0; jj < NS8H; ++jj) {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[jj][e] = 0.f;
    }
#pragma unroll
    for (int jc = 0; jc < DK; ++jc) {
#pragma unroll
      for (int jn = 0; jn < NS8H; jn += 2) {
        uint32_t b0[2], b1[2];
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK),
                                      hw * 32 + jn * 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b0[0], b0[1]);
        ldsm_x2(tile_elem_addr<D, BN>(smem_addr(sK),
                                      hw * 32 + jn * 8 + 8 + (lane & 7),
                                      jc * 16 + ((lane & 8) >> 3) * 8),
                b1[0], b1[1]);
        uint32_t qa[4];
        if (DKH > 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) qa[e] = qa_r[jc][e];
        } else {
          ldsm_x4(tile_elem_addr<D, ROWS>(smem_addr(sQ), (lane & 15),
                                          jc * 16 + (lane >> 4) * 8),
                  qa[0], qa[1], qa[2], qa[3]);
        }
        FragPack<T>::mma(S[jn], qa, b0);
        FragPack<T>::mma(S[jn + 1], qa, b1);
      }
    }
  };

  // ---- windowed consume: slot (w >> 1), half (w & 1). Each warp carries a
  // rolling (O, l, m) partial over its half-chunks, one per window. ----
  bool have = false;
  uint32_t wphase = 0;
  for (int w0 = 0; w0 < nchunks; w0 += nbuf, wphase ^= 1) {
    const int w1 = w0 + nbuf < nchunks ? w0 + nbuf : nchunks;
    if (w0 > 0) {
      __syncthreads();  // previous window fully consumed; safe to refill
      for (int j = w0 + warp; j < w1; j += 2 * NWARPS) {
        if (lane == 0)
          issue_kv2<D>(tk0, tk1, tv0, tv1,
                       smem_addr(sK0 + (size_t)(j - w0) * BN * D),
                       smem_addr(sV0 + (size_t)(j - w0) * BN * D),
                       chunk_nbase(j), kvh, smem_addr(&bar_k[j - w0]),
                       smem_addr(&bar_v[j - w0]), BN);
      }
    }
    // A window can hold more chunks than the 4 the eight warps cover in one
    // pass (two warps per chunk, half each), so strides of 4 chunks sweep the
    // rest of the window.
    for (int c = jloc; w0 + c < w1; c += 4) {
      const int slot = c;
      mbar_wait(smem_addr(&bar_k[slot]), wphase);
      mbar_wait(smem_addr(&bar_v[slot]), wphase);
      const int nbase = chunk_nbase(w0 + c) + hw * 32;  // half base
      // Half fully out of range (tail chunk's far half): stage an empty
      // partial and skip the work. Processing it would leave every S at
      // NEG_INF, m at NEG_INF*scale, and fmaf(S, scale, -m) collapses to the
      // multiply's rounding residue (~1e21); one sign exp2()s to +inf and
      // poisons the whole fold with NaN (0 * inf via the OOB-zero V rows).
      if (nbase < N) {
      qk_half(slot);
      T *sV = sV0 + (size_t)slot * BN * D;

      if (nbase + 32 > N) {  // partial tail inside this 32-key half
#pragma unroll
        for (int jn = 0; jn < NS8H; ++jn) {
          int c0 = nbase + jn * 8 + myCol;
          if (c0 >= N) S[jn][0] = NEG_INF;
          if (c0 + 1 >= N) S[jn][1] = NEG_INF;
          if (c0 >= N) S[jn][2] = NEG_INF;
          if (c0 + 1 >= N) S[jn][3] = NEG_INF;
        }
      }

      float cmax0 = NEG_INF, cmax8 = NEG_INF;
#pragma unroll
      for (int jn = 0; jn < NS8H; ++jn) {
        cmax0 = fmaxf(cmax0, fmaxf(S[jn][0], S[jn][1]));
        cmax8 = fmaxf(cmax8, fmaxf(S[jn][2], S[jn][3]));
      }
      cmax0 *= scale_log2e;
      cmax8 *= scale_log2e;
#pragma unroll
      for (int sh = 1; sh <= 2; sh <<= 1) {
        cmax0 = fmaxf(cmax0, __shfl_xor_sync(0xffffffffu, cmax0, sh));
        cmax8 = fmaxf(cmax8, __shfl_xor_sync(0xffffffffu, cmax8, sh));
      }
      const float mnew0 = fmaxf(m_r, cmax0);
      const float mnew8 = fmaxf(m_r8, cmax8);
      if (!have) {
        // First half-chunk of this warp's chain: rescale is a provable no-op.
        m_r = mnew0;
        m_r8 = mnew8;
        have = true;
      } else if (mnew0 > m_r || mnew8 > m_r8) {
        const float alpha0 = exp2f(m_r - mnew0);
        const float alpha8 = exp2f(m_r8 - mnew8);
#pragma unroll
        for (int jj = 0; jj < DN; ++jj) {
          O[jj][0] *= alpha0;
          O[jj][1] *= alpha0;
          O[jj][2] *= alpha8;
          O[jj][3] *= alpha8;
        }
        l_r *= alpha0;
        l_r8 *= alpha8;
        m_r = mnew0;
        m_r8 = mnew8;
      }

      uint32_t pfr[NS8H / 2][4];
#pragma unroll
      for (int jc = 0; jc < NS8H / 2; ++jc) {
        pfr[jc][0] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][0], scale_log2e, -m_r),
            fmaf(S[2 * jc][1], scale_log2e, -m_r)));
        pfr[jc][1] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc][3], scale_log2e, -m_r8)));
        pfr[jc][2] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][0], scale_log2e, -m_r),
            fmaf(S[2 * jc + 1][1], scale_log2e, -m_r)));
        pfr[jc][3] = FragPack<T>::ex2(FragPack<T>::cvt2(
            fmaf(S[2 * jc + 1][2], scale_log2e, -m_r8),
            fmaf(S[2 * jc + 1][3], scale_log2e, -m_r8)));
      }

      float lacc[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t ones2[2] = {FragPack<T>::ONES, FragPack<T>::ONES};
#pragma unroll
      for (int jk = 0; jk < NS8H / 2; ++jk) {
#pragma unroll
        for (int jn = 0; jn < DN; jn += 2) {
          uint32_t vq[4];
          ldsm_x4_trans(tile_elem_addr<D, BN>(smem_addr(sV),
                                              hw * 32 + jk * 16 + (lane & 15),
                                              jn * 8 + (lane >> 4) * 8),
                        vq[0], vq[1], vq[2], vq[3]);
          uint32_t b0[2] = {vq[0], vq[1]};
          uint32_t b1[2] = {vq[2], vq[3]};
          FragPack<T>::mma(O[jn], pfr[jk], b0);
          FragPack<T>::mma(O[jn + 1], pfr[jk], b1);
        }
        FragPack<T>::mma(lacc, pfr[jk], ones2);
      }
      l_r += lacc[0];
      l_r8 += lacc[2];
      }  // nbase < N (half in range)
    }
  }

  PHASE_POINT(phbuf, 4, ph_cid);

  // ---- publish one partial slice per warp, then a single row-split fold:
  // warp w owns output rows 2w and 2w+1 and folds every staged slice. ----
  constexpr int DPB = D + 8;
  const int nstage = (2 * nchunks < NPART) ? 2 * nchunks : NPART;
  if (warp < nstage) {
    float *sg = sMB + (size_t)warp * ROWS * DPB;
#pragma unroll
    for (int jn = 0; jn < DN; ++jn) {
      const int c = jn * 8 + myCol;
      *reinterpret_cast<float2 *>(&sg[myRow * DPB + c]) =
          make_float2(O[jn][0], O[jn][1]);
      *reinterpret_cast<float2 *>(&sg[(myRow + 8) * DPB + c]) =
          make_float2(O[jn][2], O[jn][3]);
    }
    sg[myRow * DPB + D] = l_r;
    sg[myRow * DPB + D + 1] = m_r;
    sg[(myRow + 8) * DPB + D] = l_r8;
    sg[(myRow + 8) * DPB + D + 1] = m_r8;
  }
  __syncthreads();  // all slices published
  PHASE_POINT(phbuf, 6, ph_cid);

#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int r = warp * 2 + rr;
    const int gr = row0 + r;
    float lg[NPART], mg[NPART];
    float mst = NEG_INF;
#pragma unroll
    for (int g = 0; g < NPART; ++g) {
      lg[g] = 0.f;
      mg[g] = NEG_INF;
      if (g < nstage) {
        const float *srow = sMB + (size_t)g * ROWS * DPB + r * DPB;
        lg[g] = srow[D];
        mg[g] = srow[D + 1];
        mst = fmaxf(mst, mg[g]);
      }
    }
    float fg[NPART];
    float lt = 0.f;
#pragma unroll
    for (int g = 0; g < NPART; ++g) {
      fg[g] = (lg[g] > 0.f) ? exp2f(mg[g] - mst) : 0.f;
      lt += lg[g] * fg[g];
    }
    const float inv = (lt > 0.f) ? (1.f / lt) : 0.f;
    if (lane * 4 < D && gr < M) {
      // Issue every slice's LDS up front, then accumulate.
      float4 v4[NPART];
#pragma unroll
      for (int g = 0; g < NPART; ++g) {
        if (g < nstage)
          v4[g] = *reinterpret_cast<const float4 *>(
              sMB + (size_t)g * ROWS * DPB + r * DPB + lane * 4);
      }
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int g = 0; g < NPART; ++g) {
        if (g < nstage) {
          a0 = fmaf(v4[g].x, fg[g], a0);
          a1 = fmaf(v4[g].y, fg[g], a1);
          a2 = fmaf(v4[g].z, fg[g], a2);
          a3 = fmaf(v4[g].w, fg[g], a3);
        }
      }
      T pk[4] = {FragPack<T>::cvt(a0 * inv), FragPack<T>::cvt(a1 * inv),
                 FragPack<T>::cvt(a2 * inv), FragPack<T>::cvt(a3 * inv)};
      uint2 p2;
      p2.x = *reinterpret_cast<uint32_t *>(pk);
      p2.y = *reinterpret_cast<uint32_t *>(pk + 2);
      *reinterpret_cast<uint2 *>(&out[((size_t)gr * HQ + hq) * D + lane * 4]) = p2;
    }
    if (WITH_LSE && lane == 0 && gr < M) {
      lse_g[(size_t)gr * HQ + hq] =
          (lt > 0.f) ? (mst + log2f(lt)) * LN2 : -INFINITY;
    }
  }
  PHASE_POINT(phbuf, 5, ph_cid);
  PHASE_FLUSH(phbuf, ph_cid);
}

CUresult encode_map(CUtensorMap *map, const void *ptr, CUtensorMapDataType dt,
                  uint64_t dcols, uint64_t heads, uint64_t rows,
                  uint32_t box_cols, uint32_t box_rows, CUtensorMapSwizzle sw) {
  const uint64_t dims[3] = {dcols, heads, rows};
  const uint64_t strides[2] = {dcols * 2, dcols * heads * 2};
  const uint32_t box[3] = {box_cols, 1, box_rows};
  const uint32_t estr[3] = {1, 1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      map, dt, 3, const_cast<void *>(ptr), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return res;
}

// D=128 panel fold: dims {64 cols, rows, 2 panels, heads}, box
// {64, box_rows, 2, 1} -> both 64-col panels arrive in ONE TMA op, filled
// rows-before-panels so the smem result is byte-identical to the two 3D
// panel fills (swizzle phase stays r & 7). Layout verified by diag_tma4d.py.
CUresult encode_map4d(CUtensorMap *map, const void *ptr,
                      CUtensorMapDataType dt, uint64_t heads, uint64_t rows,
                      uint32_t box_rows) {
  const uint64_t dims[4] = {64, rows, 2, heads};
  const uint64_t strides[3] = {heads * 256, 128, 256};
  const uint32_t box[4] = {64, box_rows, 2, 1};
  const uint32_t estr[4] = {1, 1, 1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      map, dt, 4, const_cast<void *>(ptr), dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return res;
}

}  // namespace

template <typename T, int D, bool WITH_LSE>
void launch_path(const CUtensorMap &tq0, const CUtensorMap &tq1,
                 const CUtensorMap &tk0, const CUtensorMap &tk1,
                 const CUtensorMap &tv0, const CUtensorMap &tv1,
                 const bool *mask_p, T *out_p, float *lse_p, float *ows_p,
                 int M, int N, int HQ, int HKV, int BS, int rows_pad, int G,
                 bool normalize, float scale_log2e, cudaStream_t stream, long long *php) {
  const int nbuf = nbuf_for(D);
  const int smem = (BM * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) + 2048;
  const int Mtl = (M + BM - 1) / BM;
  auto kern = bsa_split_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 166000);
    attr_done = true;
  }
  kern<<<dim3(Mtl, HQ, (unsigned)G), NWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, ows_p, M, N, HQ, HKV,
      BS, rows_pad, nbuf, normalize, scale_log2e, php);
  if (G > 1) {
    int Mi = M, HQi = HQ, Gi = G, rpi = rows_pad;
    void *kargs[] = {&ows_p, &out_p, &lse_p, &Mi, &HQi, &Gi, &rpi};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((M + 7) / 8, HQ);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute pattr[1];
    pattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pattr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = pattr;
    cfg.numAttrs = 1;
    cudaLaunchKernelExC(&cfg, (const void *)bsa_merge2_kernel<T, D, WITH_LSE>, kargs);
  }
}

template <typename T, int D, bool WITH_LSE>
void launch_pack(const CUtensorMap &tq0, const CUtensorMap &tq1,
                 const CUtensorMap &tk0, const CUtensorMap &tk1,
                 const CUtensorMap &tv0, const CUtensorMap &tv1,
                 const bool *mask_p, T *out_p, float *lse_p, float *ows_p,
                 int M, int N, int HQ, int HKV, int BS, int rows_pad, int G,
                 bool normalize, float scale_log2e, cudaStream_t stream,
                 int PH) {
  const int nbuf = nbuf_for(D);
  const int smem = (BM * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) + 2048;
  const int RPH = 16 * (NWARPS / PH);
  const int gx = (M + RPH - 1) / RPH;
  auto kern = bsa_pack_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 166000);
    attr_done = true;
  }
  kern<<<dim3(gx, HQ / PH, (unsigned)G), NWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, ows_p, M, N, HQ, HKV,
      BS, rows_pad, nbuf, normalize, scale_log2e, PH);
  if (G > 1) {
    int Mi = M, HQi = HQ, Gi = G, rpi = rows_pad;
    void *kargs[] = {&ows_p, &out_p, &lse_p, &Mi, &HQi, &Gi, &rpi};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((M + 7) / 8, HQ);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute pattr[1];
    pattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pattr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = pattr;
    cfg.numAttrs = 1;
    cudaLaunchKernelExC(&cfg, (const void *)bsa_merge2_kernel<T, D, WITH_LSE>, kargs);
  }
}

template <typename T, int D, bool WITH_LSE>
void launch_wide(const CUtensorMap &tq0, const CUtensorMap &tq1,
                 const CUtensorMap &tk0, const CUtensorMap &tk1,
                 const CUtensorMap &tv0, const CUtensorMap &tv1,
                 const bool *mask_p, T *out_p, float *lse_p, float *ows_p,
                 int M, int N, int HQ, int HKV, int BS, int rows_pad, int G,
                 bool normalize, float scale_log2e, cudaStream_t stream,
                 int PH) {
  const int nbuf = nbuf_wide(D);
  const int smem = (128 * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) +
                   (NWWARPS * (MAX_BLOCKS / 32) + 32) * (int)sizeof(uint32_t) +
                   2048;
  const int RPH = 16 * (NWWARPS / PH);
  const int gx = (M + RPH - 1) / RPH;
  auto kern = bsa_wide_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 220000);
    attr_done = true;
  }
  kern<<<dim3(gx, HQ / PH, (unsigned)G), NWWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, ows_p, M, N, HQ, HKV,
      BS, rows_pad, nbuf, normalize, scale_log2e, PH);
  if (G > 1) {
    int Mi = M, HQi = HQ, Gi = G, rpi = rows_pad;
    void *kargs[] = {&ows_p, &out_p, &lse_p, &Mi, &HQi, &Gi, &rpi};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((M + 7) / 8, HQ);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute pattr[1];
    pattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pattr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = pattr;
    cfg.numAttrs = 1;
    cudaLaunchKernelExC(&cfg, (const void *)bsa_merge2_kernel<T, D, WITH_LSE>, kargs);
  }
}

template <typename T, int D, bool WITH_LSE>
void launch_pair(const CUtensorMap &tq0, const CUtensorMap &tq1,
                 const CUtensorMap &tk0, const CUtensorMap &tk1,
                 const CUtensorMap &tv0, const CUtensorMap &tv1,
                 const bool *mask_p, T *out_p, float *lse_p, float *ows_p,
                 int M, int N, int HQ, int HKV, int BS, int rows_pad, int G,
                 float scale_log2e, cudaStream_t stream, long long *php) {
  const int nbuf = nbuf_pair(D);
  const int smem = (BM * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) +
                   64 * (D + 8) * (int)sizeof(float) + 2048;
  const int Mtl = (M + BM - 1) / BM;
  auto kern = bsa_pair_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 204000);
    attr_done = true;
  }
  kern<<<dim3(Mtl, HQ, (unsigned)G), 2 * NWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, ows_p, M, N, HQ, HKV,
      BS, rows_pad, nbuf, G == 1, scale_log2e, php);
}

template <typename T, int D, bool WITH_LSE>
void launch_pair32(const CUtensorMap &tq0, const CUtensorMap &tq1,
                   const CUtensorMap &tk0, const CUtensorMap &tk1,
                   const CUtensorMap &tv0, const CUtensorMap &tv1,
                   const bool *mask_p, T *out_p, float *lse_p, int M, int N,
                   int HQ, int HKV, int BS, float scale_log2e,
                   cudaStream_t stream, long long *php) {
  const int nbuf = nbuf_pair32(D);
  const int smem = (32 * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) +
                   4 * 32 * (D + 8) * (int)sizeof(float) + 2048;
  const int Mtl = (M + 31) / 32;
  auto kern = bsa_pair32_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 219000);
    attr_done = true;
  }
  kern<<<dim3(Mtl, HQ, 1), 2 * NWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, M, N, HQ, HKV, BS,
      nbuf, scale_log2e, php);
}

template <typename T, int D, bool WITH_LSE>
void launch_bm16(const CUtensorMap &tq0, const CUtensorMap &tq1,
                 const CUtensorMap &tk0, const CUtensorMap &tk1,
                 const CUtensorMap &tv0, const CUtensorMap &tv1,
                 const bool *mask_p, T *out_p, float *lse_p, int M, int N,
                 int HQ, int HKV, int BS, float scale_log2e,
                 cudaStream_t stream, long long *php) {
  const int nbuf = nbuf_bm16(D);
  const int smem = (16 * D * 2 + (size_t)2 * nbuf * BN * D * 2) +
                   MAX_BLOCKS * (int)sizeof(int) +
                   8 * 16 * (D + 8) * (int)sizeof(float) + 2048;
  const int Mtl = (M + 15) / 16;
  auto kern = bsa_bm16_kernel<T, D, WITH_LSE>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 219000);
    attr_done = true;
  }
  kern<<<dim3(Mtl, HQ, 1), 2 * NWARPS * 32, smem, stream>>>(
      tq0, tq1, tk0, tk1, tv0, tv1, mask_p, out_p, lse_p, M, N, HQ, HKV, BS,
      nbuf, scale_log2e, php);
}

cudaError_t VibeCUDABSAFwdRaw(void *out_raw, float *lse_p, const void *q_raw,
                              const void *k_raw, const void *v_raw,
                              const bool *mask_p, float *ows_p, int M, int N,
                              int HQ, int HKV, int D, int block_size,
                              bool return_lse, int G, bool is_bf16,
                              float sm_scale, cudaStream_t stream,
                              long long *php = nullptr) {
  if ((reinterpret_cast<uintptr_t>(q_raw) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(k_raw) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(v_raw) & 15) != 0)
    return cudaErrorInvalidValue;  // tensor bases must be 16B aligned for TMA
  if (D != 64 && D != 96 && D != 128) return cudaErrorInvalidValue;
  if (block_size % BN != 0)
    return cudaErrorInvalidValue;  // block size must be a multiple of 64
  const float scale_log2e = sm_scale * 1.4426950408889634f;

  const int Mtl = (M + BM - 1) / BM;
  const int rows_pad = Mtl * BM;
  if (G < 1 || G > 16) return cudaErrorInvalidValue;  // bad split count
  const bool normalize = (G == 1);
  static int nsm = []() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    return p.multiProcessorCount;
  }();

  // GQA head packing: PH query heads that share one KV head ride one CTA and
  // reuse each staged K/V chunk. Purely runtime dispatch on (HQ, HKV, N).
  // Measured on the fixed suite (NCU v36_short_pack vs v36_short_nopack):
  // packing ties on K/V traffic (4 warps x 16 rows invariants) but adds ~14%
  // chunk-overhead latency at the short shapes, so it is gated to the
  // long-N/high-group regime it was designed for; everywhere else the direct
  // low-overhead split kernel runs.
  int PH = 0;
  if (HQ % HKV == 0) {
    const int grp = HQ / HKV;
    const bool long_gqa = (grp >= 2) && ((int64_t)N >= 32768);
    if (long_gqa) {
      if (grp % 4 == 0 && HQ % 4 == 0) PH = 4;
      else if (grp % 2 == 0 && HQ % 2 == 0) PH = 2;
    }
  }

  // 8-warp wide CTA (128 row-head slots so staged K/V is reused 2x per CTA).
  // Purely structural runtime dispatch: GQA group, M, BS. Heads in one pack
  // share a KV head (grp % W == 0); PH=1 pairs two 64-row tiles of one head.
  int WPHsel = 0;
  const int grp = (HQ % HKV == 0) ? HQ / HKV : 0;
  auto pick_wph = [&]() -> int {
    if (grp <= 0) return 0;
    if (grp % 8 == 0 && HQ % 8 == 0) return 8;
    if (grp % 4 == 0 && HQ % 4 == 0) return 4;
    if (grp % 2 == 0 && HQ % 2 == 0) return 2;
    return (M >= 128) ? 1 : 0;
  };
  {
    // Auto policy from NCU evidence (round 7, v41_short_wide vs v41_short_v39):
    // the wide CTA halves the CTA count, so on sub-wave grids it doubles the
    // serial warp-chain work per active SM while other SMs idle (15.3us vs
    // 10.8us kernel time on the short suite case; occupancy 6.2->12.1% but
    // SM count halves). It only serves workloads whose grid still fills the
    // device with ~2+ CTA waves after the 2x tile widening; reserve it for
    // those. Total warp work is workload-invariant, so large-M/large-HQ*MB
    // problems with admitted chunk volume keep full parallelism.
    const int wph_try = (grp >= 2) ? pick_wph() : ((grp == 1 && M >= 128) ? 1 : 0);
    if (wph_try > 0 && PH == 0) {
      const int RPH = 16 * (NWWARPS / wph_try);
      const int64_t ctas =
          (int64_t)((M + RPH - 1) / RPH) * (HQ / wph_try) * (int64_t)G;
      // 2+ CTA waves so the 2x wider CTA does not strand SMs.
      if (ctas >= 2 * nsm) WPHsel = wph_try;
    }
  }
  if (WPHsel > 0 && (HQ % WPHsel != 0 || (grp > 0 && grp % WPHsel != 0))) WPHsel = 0;
  if (WPHsel > 0) PH = 0;  // wide takes precedence over the 4-warp pack

  // In-CTA pair split (8 warps, interleaved chunk groups, smem merge): the
  // direct low-overhead path at G==1 (normalize). A/B on the long case
  // (m=128, n=16384, G=8, torch profiler, round 10) measured pair+merge at
  // 8.37us + 3.87us = 12.24us vs the cross-CTA split path at 6.58us + 3.07us
  // = 9.65us: folding two split streams in-CTA halves the grid (128 -> 64
  // CTAs) and at a 0.86-wave grid the lost SM parallelism outweighs the
  // halved merge traffic, so the cross-CTA split stays the G>1 production
  // path.
  bool use_pair = false;
  if (WPHsel == 0 && PH == 0 && normalize) {
    // Windowed re-staging makes any chunk count safe, so the 8-warp in-CTA
    // pair split is the default normalize path.
    use_pair = true;
  }

  // BM=32 pair variant (8 warps, FOUR in-CTA chunk groups, normalize only):
  // dispatched purely on the runtime fill metric ctas64 = Mtl(64)*HQ. When the
  // BM=64 pair grid underfills the device (ctas64 < nsm, e.g. 16 CTAs on 148
  // SMs), halving the row tile doubles the grid AND quarters the per-CTA
  // serial chunk chain (group g takes chunks g, g+4, ... vs g, g+2, ...), with
  // the in-smem merge folding up to three staged group partials. Grid-full
  // workloads keep the 64-row tile pair kernel (double the K/V reuse per CTA).
  bool use_pair32 = false;
  bool use_bm16 = false;
  if (use_pair) {
    const int64_t ctas64 = (int64_t)Mtl * HQ;
    if (ctas64 < nsm) {
      // Deeper underfill: when even the 32-row grid leaves SMs idle, the 16-row
      // bm16 kernel doubles the grid once more (and halves every warp's serial
      // chunk chain) as long as its own grid still fits one wave.
      const int64_t ctas16 = (int64_t)((M + 15) / 16) * HQ;
      if (ctas16 <= nsm) use_bm16 = true;
      else use_pair32 = true;
    }
  }
  if (use_pair32 || use_bm16) use_pair = false;

  CUresult enc_rc = CUDA_SUCCESS;
  const CUtensorMapDataType dt =
      is_bf16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  CUtensorMap tq0, tq1, tqA0, tqA1, tqW0, tqW1, tk0, tk1, tv0, tv1;
  const int qbox = (PH > 0) ? 16 : BM;  // Q box rows: packed kernels take 16
  if (D == 128) {
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tqA0, q_raw, dt, HQ, M, qbox);
    tqA1 = tqA0;
  } else {
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqA0, q_raw, dt, D, HQ, M, 64, qbox, CU_TENSOR_MAP_SWIZZLE_128B);
    if (D == 96) {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqA1, q_raw, dt, D, HQ, M, 32, qbox,
                 CU_TENSOR_MAP_SWIZZLE_64B);
    } else {
      tqA1 = tqA0;
    }
  }
  tq0 = tqA0;
  tq1 = tqA1;
  // pair32 path: Q map with box_rows=32 to match the 32-row CTA tile.
  CUtensorMap tqB0 = tqA0, tqB1 = tqA1;
  if (use_pair32) {
    if (D == 128) {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tqB0, q_raw, dt, HQ, M, 32);
      tqB1 = tqB0;
    } else {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqB0, q_raw, dt, D, HQ, M, 64, 32,
                 CU_TENSOR_MAP_SWIZZLE_128B);
      if (D == 96) {
        if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqB1, q_raw, dt, D, HQ, M, 32, 32,
                   CU_TENSOR_MAP_SWIZZLE_64B);
      } else {
        tqB1 = tqB0;
      }
    }
  }
  // bm16 path: Q map with box_rows=16 to match the 16-row CTA tile.
  CUtensorMap tqC0 = tqA0, tqC1 = tqA1;
  if (use_bm16) {
    if (D == 128) {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tqC0, q_raw, dt, HQ, M, 16);
      tqC1 = tqC0;
    } else {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqC0, q_raw, dt, D, HQ, M, 64, 16,
                 CU_TENSOR_MAP_SWIZZLE_128B);
      if (D == 96) {
        if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqC1, q_raw, dt, D, HQ, M, 32, 16,
                   CU_TENSOR_MAP_SWIZZLE_64B);
      } else {
        tqC1 = tqC0;
      }
    }
  }
  if (WPHsel > 0) {
    const int wbox = 16 * (NWWARPS / WPHsel);  // RPH rows per Q box
    if (D == 128) {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tqW0, q_raw, dt, HQ, M, wbox);
      tqW1 = tqW0;
    } else {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqW0, q_raw, dt, D, HQ, M, 64, wbox,
                 CU_TENSOR_MAP_SWIZZLE_128B);
      if (D == 96) {
        if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tqW1, q_raw, dt, D, HQ, M, 32, wbox,
                   CU_TENSOR_MAP_SWIZZLE_64B);
      } else {
        tqW1 = tqW0;
      }
    }
  } else {
    tqW0 = tqA0;
    tqW1 = tqA1;
  }
  if (D == 128) {
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tk0, k_raw, dt, HKV, N, BN);
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map4d(&tv0, v_raw, dt, HKV, N, BN);
    tk1 = tk0;
    tv1 = tv0;
  } else {
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tk0, k_raw, dt, D, HKV, N, 64, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tv0, v_raw, dt, D, HKV, N, 64, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    if (D == 96) {
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tk1, k_raw, dt, D, HKV, N, 32, BN, CU_TENSOR_MAP_SWIZZLE_64B);
      if (enc_rc == CUDA_SUCCESS) enc_rc = encode_map(&tv1, v_raw, dt, D, HKV, N, 32, BN, CU_TENSOR_MAP_SWIZZLE_64B);
    } else {
      tk1 = tk0;
      tv1 = tv0;
    }
  }

  if (enc_rc != CUDA_SUCCESS) return cudaErrorInvalidValue;

#define BSA_CALL(DV, LSEB, TYPE)                                                       \
  do {                                                                                 \
    if (use_bm16)                                                                      \
      launch_bm16<TYPE, DV, LSEB>(tqC0, tqC1, tk0, tk1, tv0, tv1, mask_p, op,          \
                                  lse_p, M, N, HQ, HKV, (int)block_size,               \
                                  scale_log2e, stream, php);                           \
    else if (use_pair32)                                                               \
      launch_pair32<TYPE, DV, LSEB>(tqB0, tqB1, tk0, tk1, tv0, tv1, mask_p, op,        \
                                    lse_p, M, N, HQ, HKV, (int)block_size,             \
                                    scale_log2e, stream, php);                          \
    else if (WPHsel > 0)                                                               \
      launch_wide<TYPE, DV, LSEB>(tqW0, tqW1, tk0, tk1, tv0, tv1, mask_p, op,          \
                                  lse_p, ows_p, M, N, HQ, HKV, (int)block_size,        \
                                  rows_pad, (int)G, normalize, scale_log2e, stream,    \
                                  WPHsel);                                             \
    else if (use_pair)                                                                 \
      launch_pair<TYPE, DV, LSEB>(tq0, tq1, tk0, tk1, tv0, tv1, mask_p, op, lse_p,     \
                                  ows_p, M, N, HQ, HKV, (int)block_size, rows_pad,     \
                                  1, scale_log2e, stream, php);                         \
    else if (PH > 0)                                                                   \
      launch_pack<TYPE, DV, LSEB>(tqA0, tqA1, tk0, tk1, tv0, tv1, mask_p, op,          \
                                  lse_p, ows_p, M, N, HQ, HKV, (int)block_size,        \
                                  rows_pad, (int)G, normalize, scale_log2e, stream,    \
                                  PH);                                                 \
    else                                                                               \
      launch_path<TYPE, DV, LSEB>(tq0, tq1, tk0, tk1, tv0, tv1, mask_p, op, lse_p,     \
                                  ows_p, M, N, HQ, HKV, (int)block_size, rows_pad,     \
                                  (int)G, normalize, scale_log2e, stream, php);         \
  } while (0)

#define BSA_DISPATCH(TYPE)                          \
  TYPE *op = reinterpret_cast<TYPE *>(out_raw);  \
  if (return_lse) {                                     \
    if (D == 64) BSA_CALL(64, true, TYPE);              \
    else if (D == 96) BSA_CALL(96, true, TYPE);         \
    else BSA_CALL(128, true, TYPE);                     \
  } else {                                              \
    if (D == 64) BSA_CALL(64, false, TYPE);             \
    else if (D == 96) BSA_CALL(96, false, TYPE);        \
    else BSA_CALL(128, false, TYPE);                    \
  }

  if (is_bf16) {
    BSA_DISPATCH(__nv_bfloat16);
  } else {
    BSA_DISPATCH(__half);
  }
#undef BSA_DISPATCH
#undef BSA_CALL

  return cudaGetLastError();
}

}  // namespace vibecuda
}  // namespace flashinfer

#endif  // FLASHINFER_VIBECUDA_BSA_FWD_CUH_
