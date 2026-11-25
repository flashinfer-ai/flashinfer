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
 *
 * SM90 NVFP4 W4A16 fused MoE, large batch (prefill) kernel: one persistent
 * launch per chunk of up to `chunk_tokens` tokens, grouped by expert.
 *
 * Shares the packed weight format and the register decoder with decode.cuh
 * (same lineage: gauravjain14/muse-okc-submissions, vjeux/okc-2026,
 * alygg77/okc-muse-spark-kernels, all Apache-2.0; see NOTICE). Changes from
 * the version this was ported from: hidden / intermediate sizes are template
 * parameters (they were 4096 / 4096), gate and up take separate global
 * scales, the per-route RMSNorm is optional, routes to experts outside
 * [0, num_experts) are skipped by the finalize kernel, and the multi-GPU paths
 * (peer rows, remote expert pulls, pre-listed routes) were removed.
 *
 * Structure: CTA pairs (cluster of 2, TMA multicast of the token tile),
 * 384 threads per CTA.
 *   - thread 0 producer: claims items in pairs, streams activation tiles (TMA)
 *     and weight blocks into a 4-slot ring;
 *   - warp 1 publisher: TMA-stores SwiGLU tiles (GEMM1) and pushes finished
 *     down rows (GEMM2) to `recv` [8, M, hidden];
 *   - warps 2-3 helpers: gather routed token rows into expert-grouped `xt`;
 *   - warps 4-11 consumers (two warpgroups): route counting, the expert plan,
 *     and the GEMM items (RS-wgmma, weights decoded in registers).
 * A second kernel (finalize) applies the optional per-route RMSNorm and the
 * top-k weighted sum.
 */
#ifndef FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_PREFILL_CUH_
#define FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_PREFILL_CUH_

#include "common.cuh"

namespace flashinfer {
namespace sm90_w4a16_moe {
namespace pf {

#define DEVI FI_W4A16_DEVI
#define AS FI_W4A16_AS

constexpr int NL = kMaxLocalExperts, NE = NL, TOPK = kTopKSlots, WBLK = kWeightBlockBytes;
constexpr int NSLOT = 4, WOFF = 32768, SOFF = 36864, SLOTB = 38912;
constexpr int R2MAX = 96, WOFF_R = 16384, SOFF_R1 = 24576, SOFF_R2 = 32768;
constexpr int EWS = 3328;
constexpr int STG = NSLOT * SLOTB, BAR = STG + 69632;
constexpr int kSmemBytes = BAR + EWS + 1280;
constexpr int C_QUEUE = 96, C_CDONE = 97, C_LDONE = 98, C_EXIT = 99;
constexpr int C_DONE = 128, C_READY = 1152;
constexpr int C_TOT = 3200, C_LB = 3240, C_WORDS = 3400;
// pull claim word: own 128 B line, apart from the producer's C_QUEUE
constexpr int C_GQUEUE = 3328;
constexpr int kCounterWords = C_WORDS + 32;
constexpr int K_G1 = 0, K_G2 = 1, K_CNT = 2, K_CMB = 4, K_PLAN = 5, K_LIST = 7, K_NOP = 8;
constexpr int MAXLAG = 16, MAXSEG = 2 * (NE + 1 + MAXLAG);
// Rows of `xt` / `h` one launch may fill: chunk_tokens * 8 routes plus per-expert padding.
constexpr int kSlotPad = 1152, kSlotSlack = 1088;
// A launch takes at most 8 * chunk_tokens + (kSlotPad - kSlotSlack) routes, so it has at most
// chunk_tokens / 32 + NE tiles of 256 rows. Up to this chunk size the per-tile done
// (C_DONE + tile) and ready (C_READY + tile) words never meet.
constexpr int kMaxChunkTokens = 32 * (C_READY - C_DONE - NE);
static_assert((8 * kMaxChunkTokens + kSlotPad - kSlotSlack + 255 * NE) / 256 <= C_READY - C_DONE &&
                  C_READY + (C_READY - C_DONE) <= C_TOT,
              "tile counters of the largest chunk overlap");
// item widths: 32 .. 256 step 32; 240 = an expert's last 256-wide tile with >= 16 padding rows
constexpr int NW = 9;
DEVI int widx(int n) { return n & 31 ? 8 : (n >> 5) - 1; }

// Problem shape. GEMM1: N = 2 * kInter (gate/up interleaved per 16 rows), K = kHidden;
// GEMM2: N = kHidden, K = kInter.
template <int kHidden_, int kInter_, bool kPostNorm_>
struct Shape {
  static constexpr int kHidden = kHidden_, kInter = kInter_;
  static constexpr bool kPostNorm = kPostNorm_;
  // A CTA pair multicasts one token tile, so a tile's GEMM2 items must come in pairs.
  static_assert(kHidden % 512 == 0 && kInter % 256 == 0, "prefill: hidden % 512, inter % 256");
  static constexpr int kItems1 = 2 * kInter / 128, kItems2 = kHidden / 128;  // 128-row blocks
  static constexpr int kNrb1 = 2 * kInter / 256, kNrb2 = kHidden / 256;      // 256-row blocks
  static constexpr int kNkc1 = kHidden / 256, kNkc2 = kInter / 256;          // 256-wide K chunks
  // two K chunks of 64 per ring slot for <= 64-row tiles
  static constexpr bool kAllowDP = kHidden % 512 == 0 && kInter % 512 == 0;
  static constexpr bool kPow2 = (kItems1 & (kItems1 - 1)) == 0 && (kItems2 & (kItems2 - 1)) == 0;
};

struct alignas(64) Tmap {
  u64 v[16];
};
struct Params {
  Tmap wm[6];  // weight maps: (k blocks, row groups) in {(1, 8), (1, 16), (2, 16)} x {w13, w2}
  const Tmap *th, *tx2, *th2, *txh, *thh;  // activation / intermediate maps (device memory)
  const int* ids;                          // [M, 8] local expert ids
  const u8 *w13, *w2;
  const float *gs13, *gs2, *nw, *gs13_up;
  u16* xt;  // [cap, kHidden] expert-grouped activation rows
  int* ctr;
  int *rev, *ucnt;
  int lag, M, cap, first, ne;
  const u16* x;  // [M, kHidden]
  u16* recv;     // [8, M, kHidden] scaled down rows per (slot, token)
  int cl;        // count and list routes in one pass (one launch covers all tokens)
};
struct Plan {
  int nwork, nlu, u0, u1;
  int seg0[MAXSEG + 1], sege[MAXSEG];
  int ord[NE], rk[NE];
  int cnt[NE], nt[NE], q[NE], r[NE], off[NE], tb[NE], sh[NE], cut[NE];
};
DEVI void tma_load3(u32 dst, const Tmap* m, int c1, int c2, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
      "{%2, %3, %4}], [%5];" ::"r"(dst),
      "l"(m), "r"(0), "r"(c1), "r"(c2), "r"(bar)
      : "memory");
}
DEVI void tma_load4(u32 dst, const Tmap* m, int c1, int c2, int c3, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
      "{%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(m), "r"(0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
}
DEVI void tmap_acquire(const Tmap* m) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" ::"l"(m) : "memory");
}
DEVI void tma_store(const Tmap* m, int c0, int c1, u32 src) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(m),
      "r"(c0), "r"(c1), "r"(src)
      : "memory");
}
DEVI void wait_geq(const int* p, int v) {
  while (ld_acquire(p) < v) __nanosleep(128);
  fence_async_global();
}
DEVI uint4 ld_na(const uint4* a) {
  uint4 v;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(a));
  return v;
}
DEVI void stmatrix_t2(u32 a, u32 r0, u32 r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};" ::"r"(a), "r"(r0),
               "r"(r1)
               : "memory");
}
DEVI float silu(float v) { return __fdividef(v, 1.f + ex2(v * -1.4426950216293334961f)); }
// `rev`: the route list of each expert (up to TOPK * M entries: a token may name an expert in
// several slots), then at list NE the rank of every (token, slot) in its expert's list.
DEVI size_t rev_at(const Params& p, int e, int i) { return (size_t)e * TOPK * p.M + i; }
// gather packs (expert, rank) into one int: a launch lists at most 8 * chunk_tokens + 64 routes
static_assert(8 * kMaxChunkTokens + kSlotPad - kSlotSlack <= (1 << 20), "route rank field");
DEVI int tile_of_row(const Plan& pl, int e, int row) {
  const int q = pl.q[e], r = pl.r[e], v = (row + (row >= pl.cut[e] ? pl.sh[e] : 0)) >> 5;
  return pl.tb[e] + (v < r * (q + 1) ? v / (q + 1) : r + (v - r * (q + 1)) / q);
}
constexpr int B_F = 0, B_E = 8 * NSLOT, B_IF = 16 * NSLOT, B_IE = B_IF + 16, B_PF = B_IE + 16,
              B_PE = B_PF + 8, B_PL = B_PE + 8;
// Per-expert route counts of one 128-token unit (counting pass of a multi-launch batch).
DEVI void count_unit(const Params& p, int u, int q, int lane, int ctid, int* hist) {
  if (ctid < 64) hist[ctid] = 0;
  bar_sync(1, 256);
  const int t = u * 128 + q * 16 + (lane >> 1);
  if (t < p.M) {
    const int4 v = AS(const int4, p.ids)[2 * t + (lane & 1)];
    const int e4[4] = {v.x, v.y, v.z, v.w};
    for (int k = 0; k < 4; ++k)
      if (e4[k] >= 0 && e4[k] < p.ne) atomicAdd(hist + e4[k], 1);
  }
  bar_sync(1, 256);
  if (ctid < 32) {
    const int c = hist[ctid];
    int s = c;
    for (int o = 16; o; o >>= 1) s += shfl_xor(s, o);
    p.ucnt[u * 64 + ctid] = c;
    p.ucnt[u * 64 + NL + ctid] = ctid == 31 ? s : 0;  // word 63: routes of the unit
    if (c) atomicAdd(p.ctr + C_TOT + ctid, c);
    if (ctid == 0 && s) atomicAdd(p.ctr + C_TOT + NE, s);
  }
  bar_sync(1, 256);
  if (ctid == 0) {
    __threadfence();
    red_release(p.ctr + C_CDONE, 1);
  }
}
DEVI void cp_async4(u32 dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
DEVI u32 tid_x() {
  u32 r;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r));
  return r;
}
DEVI u32 lds32(u32 a) {
  u32 v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
DEVI void arrive_peer(u32 bar, u32 cta) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(mapa(bar, cta))
               : "memory");
}
DEVI void st_peer(u32 addr, u32 cta, int v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(mapa(addr, cta)), "r"(v) : "memory");
}
DEVI bool mbar_test_cl(u32 b, u32 ph) {
  u32 ok;
  asm volatile(
      "{ .reg .pred p; mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; "
      "selp.u32 %0, 1, 0, p; }"
      : "=r"(ok)
      : "r"(b), "r"(ph)
      : "memory");
  return ok != 0;
}
DEVI void mbar_wait_cl(u32 b, u32 ph) {
  asm volatile(
      "{\n.reg .pred P;\nW_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], "
      "%1;\n@!P bra W_%=;\n}\n" ::"r"(b),
      "r"(ph)
      : "memory");
}
DEVI void tma_load_mc(u32 dst, const Tmap* m, int c0, int c1, u32 bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::"
      "cluster [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(m), "r"(c0), "r"(c1), "r"(bar), "h"((u16)3)
      : "memory");
}
DEVI void release_slot(u32 bar, u32 crank) {
  mbar_arrive(bar);
  // default .release orders the slot's reads
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(mapa(bar, crank ^ 1))
               : "memory");
}
DEVI u32 word4(const uint4& w, int j) { return j == 0 ? w.x : j == 1 ? w.y : j == 2 ? w.z : w.w; }
// Same decoder as decode.cuh (dq4), on a 16-byte code vector.
DEVI void decode(const uint4& w, int j, u32 R, u32 (&a)[4]) {
  constexpr u32 M = 0x81C081C0u;
  const u32 q = word4(w, j), qp = word4(w, j ^ 1);
  const u32 F = j & 1 ? __byte_perm(qp, q, 0x6341) : __byte_perm(q, qp, 0x6341);
  u32 o1, o2, o3;
  if (!(j & 1)) {
    o1 = ((q << 2) ^ F) & M;
    o2 = (q << 6) & M;
    o3 = rotl(q, 12) & M;
  } else {
    o1 = (q << 4) & M;
    o2 = rotl(q, 10) & M;
    o3 = (rotl(q, 14) ^ F) & M;
  }
  a[0] = bmul2s<0>(q & M, R);
  a[1] = bmul2s<1>(o1, R);
  a[2] = bmul2s<0>(o2, R);
  a[3] = bmul2s<1>(o3, R);
}
template <int KK>
DEVI u32 scale_pair(u32 odd, u32 w0, u32 w1, u32 w2) {
  constexpr u32 M6 = 0x03F003F0u, NB = 0x00F000F0u, C = 0x78007800u;
  u32 v;
  if (KK == 3)
    v = odd ? (((w1 << 4) & NB) | ((w2 << 6) & ~NB)) : (((w0 << 4) & NB) | ((w2 << 8) & ~NB));
  else
    v = (KK == 0 ? w0 : KK == 1 ? w1 : w2) >> (odd ? 6 : 0);
  return (v & M6) | C;
}
DEVI u32 scale_kk(int kk, u32 odd, u32 w0, u32 w1, u32 w2) {
  return kk == 0   ? scale_pair<0>(odd, w0, w1, w2)
         : kk == 1 ? scale_pair<1>(odd, w0, w1, w2)
         : kk == 2 ? scale_pair<2>(odd, w0, w1, w2)
                   : scale_pair<3>(odd, w0, w1, w2);
}
DEVI int halves(const Plan& pl, int e) {
  const int q = pl.q[e], r = pl.r[e];
  return r * (32 * (q + 1) <= R2MAX ? 1 : 2) + (pl.nt[e] - r) * (32 * q <= R2MAX ? 1 : 2);
}
// Expert order, tile split and the work list of the launch (one warp, identical in every CTA).
template <class S>
DEVI void make_plan(const Params& p, Plan& pl, int lane) {
  const int npos = NE + 1 + p.lag;
  const int u0 = pl.u0, nu = (p.M + 127) >> 7, budget = p.cap - kSlotSlack;
  int u1 = nu, c = 0;
  const int* src = p.ctr + (p.cl ? C_LB : C_TOT);
  if (p.cl || __ldcg(src + NE) <= budget) {
    c = __ldcg(src + lane);
  } else {
    // more routes than slots: this launch takes the units that fit
    int run = 0;
    for (int b = u0; b < nu; b += 32) {
      const int inc = wscan(b + lane < nu ? __ldcg(p.ucnt + (b + lane) * 64 + 63) : 0, lane);
      const u32 over = ballot(b + lane < nu && run + inc > budget);
      if (over) {
        u1 = b + __ffs(over) - 1;
        break;
      }
      run += shfl(inc, 31);
    }
    for (int u = u0; u < u1; u += 16) {
      int v[16];
      for (int k = 0; k < 16; ++k) v[k] = u + k < u1 ? __ldcg(p.ucnt + (u + k) * 64 + lane) : 0;
      for (int k = 0; k < 16; ++k)
        if (u + k < u1) c += v[k];
    }
  }
  const int nt = (c + 255) >> 8, u = (c + 31) >> 5;
  // u <= 8 nt: the quotient is 0..8, counted without a division
  auto qdiv = [](int u, int nt) {
    int q = 0;
    for (int k = 1; k <= 8; ++k) q += u >= k * nt;
    return nt ? q : 0;
  };
  auto shr = [&](int c, int nt, int u) {
    const int q = qdiv(u, nt), r = u - q * nt;
    return (r ? q == 7 : q == 8) ? (32 * u - c) & 16 : 0;
  };
  const int sh = shr(c, nt, u);
  const int off = wscan(32 * u - sh, lane), tb = wscan(nt, lane);
  int pos = 0;
#pragma unroll 4
  for (int k = 0; k < NL; ++k) {
    const int ck = shfl(c, k);
    pos += ck > c || (ck == c && k < lane);
  }
  auto put = [&](int e, int c, int nt, int u, int off, int tb, int pos, int sh) {
    const int q = qdiv(u, nt), r = u - q * nt, s = r ? r - 1 : nt - 1;
    pl.cnt[e] = c;
    pl.nt[e] = nt;
    pl.q[e] = q;
    pl.r[e] = r;
    pl.off[e] = off - (32 * u - sh);
    pl.tb[e] = tb - nt;
    pl.ord[pos] = e;
    pl.rk[e] = pos;
    pl.sh[e] = sh;
    pl.cut[e] = 32 * (s * q + min(s, r)) + 256 - sh;
  };
  put(lane, c, nt, u, off, tb, pos, sh);
  if (lane == 0) {
    pl.u1 = u1;
    pl.nlu = p.cl ? 0 : u1 - u0;
  }
  __syncwarp();
  // GEMM1 of expert i runs at position i + 1, its GEMM2 `lag` positions later
  int base = 0;
  for (int i0 = 0; i0 < npos; i0 += 32) {
    const int i = i0 + lane;
    int sz[2] = {0, 0}, ek[2] = {0, 0};
    if (i >= 1 && i - 1 < NE) {
      ek[0] = pl.ord[i - 1];
      sz[0] = (S::kItems1 / 2) * halves(pl, ek[0]);
    }
    if (i >= 1 + p.lag && i - 1 - p.lag < NE) {
      ek[1] = pl.ord[i - 1 - p.lag];
      sz[1] = (S::kItems2 / 2) * halves(pl, ek[1]);
    }
    const int own = sz[0] + sz[1];
    const int tot = wscan(own, lane);
    if (i < npos) {
      int st = base + tot - own;
      for (int k = 0; k < 2; ++k) {
        pl.seg0[2 * i + k] = st;
        pl.sege[2 * i + k] = ek[k] | (k << 8);
        st += sz[k];
      }
    }
    base += shfl(tot, 31);
  }
  if (lane == 0) {
    pl.seg0[2 * npos] = base;
    pl.nwork = base;
  }
  __syncwarp();
}
constexpr int B_MF = 224, B_ME = 232, B_MBOX = 240;
DEVI void prefetch_chunk(const u8* b, int hf) {
  if (hf == 1)
    prefetch_l2(b + 16384, 16384 + 3072);
  else
    prefetch_l2(b, WBLK);
}
template <int KPS, bool RG2>
DEVI void load_item(u32 ring, u32 bars, const Tmap* tx, const Tmap* tw, const u8* wb, int n,
                    int row0, int blk0, int nrb, int hf, int nkb, int& slot, u32& ph, u32 crank,
                    int* queue, u32 mph, int& nxt) {
  constexpr int WB = RG2 ? 8192 : 4096, SB = RG2 ? 3072 : 1536;
  constexpr int WO = RG2 ? WOFF_R : WOFF, SO = RG2 ? (KPS == 2 ? SOFF_R2 : SOFF_R1) : SOFF;
#pragma unroll 4
  for (int kb = 0; kb < nkb; kb += KPS) {
    const int kk = kb & 3, blk = blk0 + (kb >> 2) * nrb;
    if (kk == 0 && kb < nkb - 4) prefetch_chunk(wb + (size_t)(blk + nrb) * WBLK, hf);
    // near the end of the item, claim the pair's next item and hand it to the peer CTA
    if (kb == nkb - 4 && crank == 0) {
      nxt = atomicAdd(queue, 2);
      mbar_wait_cl(bars + B_ME, mph ^ 1);
      st_peer(bars + B_MBOX, 1, nxt);
      arrive_peer(bars + B_MF, 1);
    }
    if (kb == nkb - 2 && crank == 1 && mbar_test_cl(bars + B_MF, mph)) {
      nxt = lds32(bars + B_MBOX) + 1;
      arrive_peer(bars + B_ME, 0);
    }
    mbar_wait(bars + B_E + 8 * slot, ph ^ 1);
    const u32 fb = bars + B_F + 8 * slot, st = ring + slot * SLOTB;
    mbar_expect(fb, KPS * (n * 128 + WB) + (kk ? 0 : SB));
    if (KPS == 2)
      tma_load3(st, tx, row0, kb, fb);
    else
      tma_load_mc(st + crank * n * 64, tx, kb * 64, row0 + crank * (n >> 1), fb);
    tma_load4(st + WO, tw, kk, RG2 ? 0 : hf * 8, blk, fb);
    if (!kk) bulk_g2s(st + SO, wb + (size_t)blk * WBLK + 32768 + (RG2 ? 0 : hf * 1536), SB, fb);
    advance<NSLOT>(slot, ph);
  }
}
template <class S>
DEVI void producer(const Params& p, u8* sm, Plan& pl, u32 bars, int* iq) {
  int slot = 0, seg = 0, stage = 0, nxt = -1;
  u32 ph = 0, qs = 0, qph = 0, mph = 0;
  const u32 crank = cl_rank();
  const int ncu = ((p.M + 127) >> 7) - pl.u0;
  const u32 ring = smem_u32(sm);
  auto qget = [&] {
    mbar_wait(bars + B_IE + 8 * qs, qph ^ 1);
    return iq + 8 * qs;
  };
  auto qput = [&] {
    mbar_arrive(bars + B_IF + 8 * qs);
    advance<2>(qs, qph);
  };
  auto plan = [&] {
    wait_geq(p.ctr + C_CDONE, ncu);
    *qget() = K_PLAN;
    qput();
    mbar_wait(bars + B_PL, 0);
    stage = 1;
  };
  for (;;) {
    int it;
    if (crank == 0) {
      if (nxt >= 0) {
        it = nxt;
        nxt = -1;
      } else {
        it = atomicAdd(p.ctr + C_QUEUE, 2);
        mbar_wait_cl(bars + B_ME, mph ^ 1);
        st_peer(bars + B_MBOX, 1, it);
        arrive_peer(bars + B_MF, 1);
      }
    } else {
      if (nxt >= 0) {
        it = nxt;
        nxt = -1;
      } else {
        mbar_wait_cl(bars + B_MF, mph);
        it = *AS(volatile int, sm + BAR + B_MBOX) + 1;
        arrive_peer(bars + B_ME, 0);
      }
    }
    mph ^= 1;
    int kind = -1, a = 0, b = 0, row0 = 0, n = 0, tile = 0;
    if (it < ncu) {
      kind = K_CNT;
      a = pl.u0 + it;
    } else {
      if (stage < 1) plan();
      int w = it - ncu;
      if (w < pl.nlu) {
        kind = K_LIST;
        a = pl.u0 + w;
      } else {
        if (stage < 2) {
          wait_geq(p.ctr + C_LDONE, pl.nlu);
          stage = 2;
        }
        w -= pl.nlu + ((ncu + pl.nlu) & 1);
        if (w < 0) {
          kind = K_NOP;
        } else if (w < pl.nwork) {
          while (pl.seg0[seg + 1] <= w) ++seg;
          const int e = pl.sege[seg] & 255, li = w - pl.seg0[seg];
          kind = pl.sege[seg] >> 8;
          a = e;
          const int q = pl.q[e], r = pl.r[e], s1 = kind == K_G1 ? S::kItems1 : S::kItems2;
          const int p1 = 32 * (q + 1) <= R2MAX ? s1 / 2 : s1, p0 = 32 * q <= R2MAX ? s1 / 2 : s1;
          const int r1 = r * p1;
          int tl;
          if constexpr (S::kPow2) {
            const int l1 = __ffs(p1) - 1, l0 = __ffs(p0) - 1;
            tl = li < r1 ? li >> l1 : r + ((li - r1) >> l0);
            b = li < r1 ? li & (p1 - 1) : (li - r1) & (p0 - 1);
          } else {
            tl = li < r1 ? li / p1 : r + (li - r1) / p0;
            b = li < r1 ? li - (li / p1) * p1 : (li - r1) - ((li - r1) / p0) * p0;
          }
          const int sv = r ? r - 1 : pl.nt[e] - 1, shv = pl.sh[e];
          n = 32 * (q + (tl < r)) - (tl == sv ? shv : 0);
          row0 = pl.off[e] + 32 * (tl * q + min(tl, r)) - (tl > sv ? shv : 0);
          tile = pl.tb[e] + tl;
        }
      }
    }
    if (kind == K_NOP) continue;
    if (kind < 0 && stage < 3) {
      stage = 3;
      *qget() = K_CMB;
      qput();
    }
    int* d = qget();
    d[0] = kind;
    d[1] = a;
    d[2] = b;
    d[3] = row0;
    d[4] = n;
    d[5] = tile;
    qput();
    if (kind < 0) break;
    if (kind > K_G2) continue;
    const int e = a;
    const bool rg2 = n <= R2MAX;
    const int nrb = kind ? S::kNrb2 : S::kNrb1, nkc = kind ? S::kNkc2 : S::kNkc1;
    const int nkb = (kind ? S::kInter : S::kHidden) / 64;
    const int blk0 = e * nkc * nrb + (rg2 ? b : b >> 1);
    const u8* wb = kind ? p.w2 : p.w13;
    const Tmap* tw = p.wm + kind;
    const int hf = rg2 ? 2 : b & 1;
    prefetch_chunk(wb + (size_t)blk0 * WBLK, hf);
    if (kind == K_G1)
      wait_geq(p.ctr + C_READY + tile, min(n, pl.cnt[e] - (row0 - pl.off[e])));
    else
      wait_geq(p.ctr + C_DONE + tile, S::kItems1);
    const Tmap* txh = (kind ? p.thh : p.txh) + widx(n);
#define LOAD_ITEM(KPS, RG2, TX, TW)                                                         \
  load_item<KPS, RG2>(ring, bars, TX, TW, wb, n, row0, blk0, nrb, hf, nkb, slot, ph, crank, \
                      p.ctr + C_QUEUE, mph, nxt)
    if (S::kAllowDP && rg2 && n <= 64)
      LOAD_ITEM(2, true, (kind ? p.th2 : p.tx2) + widx(n), tw + 4);
    else if (rg2)
      LOAD_ITEM(1, true, txh, tw + 2);
    else
      LOAD_ITEM(1, false, txh, tw);
#undef LOAD_ITEM
  }
}
template <class S>
DEVI void publisher(const Params& p, u8* sm, u32 bars, const int* pq, const Plan& pl, int lane) {
  const u32 stg = smem_u32(sm + STG);
  if (lane == 0)
    for (int i = 0; i < NW; ++i) tmap_acquire(p.th + i);
  u32 ph = 0;
  for (;;) {
    mbar_wait(bars + B_PF, ph);
    ph ^= 1;
    const int kind = pq[0], bi = pq[1], row0 = pq[2], n = pq[3], tile = pq[4], e = pq[5];
    __syncwarp();
    if (kind < 0) break;
    const bool rg2 = n <= R2MAX;
    if (kind == K_G1) {
      const int wq = lane >> 2, wa = lane & 3;
      for (int T = 0; T < (rg2 ? 2 : 1); ++T)
        for (int tok = 0; tok < n; ++tok) {
          const u32 row = stg + T * 16384 + tok * 128, x = tok & 7;
          const u32 v = lds32(row + ((wq ^ x) << 4) + 4 * wa);
          __syncwarp();
          sts32(row + (((4 * (wq & 1) + wa) ^ x) << 4) + 4 * (wq >> 1), v);
          __syncwarp();
        }
      fence_async();
      __syncwarp();
      if (lane == 0) {
        tma_store(p.th + widx(n), bi * (rg2 ? 128 : 64), row0, stg);
        if (rg2) tma_store(p.th + widx(n), bi * 128 + 64, row0, stg + 16384);
        bulk_commit_wait_read();
        mbar_arrive(bars + B_PE);
        bulk_wait();
        fence_async_global();
        red_release(p.ctr + C_DONE + tile, rg2 ? 2 : 1);
      }
      __syncwarp();
    } else {
      const int r0 = row0 - pl.off[e], nr = min(n, pl.cnt[e] - r0), rs = rg2 ? 528 : 272,
                cb = rg2 ? 512 : 256;
      for (int i = lane; i < nr; i += 32) {
        const int tj = __ldcg(p.rev + rev_at(p, e, r0 + i)), t = tj >> 3;
        bulk_s2g(AS(u8, p.recv) + ((size_t)(tj & 7) * p.M + t) * (S::kHidden * 2) + bi * cb,
                 stg + i * rs, cb);
      }
      bulk_commit_wait_read();
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + B_PE);
    }
  }
  bulk_wait();
  fence_async_global();
}
template <class S, int N>
DEVI void item(const Params& p, u8* sm, u32 bars, int* pq, const int* d, int q, int lane, int& slot,
               u32& ph, u32& pph) {
  constexpr bool R2 = N <= R2MAX;
  constexpr int KPS = S::kAllowDP && R2 && N <= 64 ? 2 : 1, NT = R2 ? 2 : 1;
  constexpr bool DP = KPS == 2;
  const int g = lane >> 2;
  const u32 odd = (tid_x() >> 5) & 1, crank = cl_rank();
  float acc[128];
  for (int i = 0; i < N / 2; ++i) {
    acc[i] = 0.f;
    if (R2) acc[64 + i] = 0.f;
  }
  // epilogue-only inputs go to shared memory: no registers held through the main loop
  const u8* ew = sm + BAR + EWS + 160 * q;
  const u32 ews = smem_u32(ew);
  const int nkb = (d[0] == K_G1 ? S::kHidden : S::kInter) / 64;
  {
    const int kind = d[0], e = d[1], bi = d[2];
    if (lane == 0) {
      cp_async4(ews, (kind == K_G1 ? p.gs13 : p.gs2) + e);
      if (kind == K_G1) cp_async4(ews + 4, p.gs13_up + e);
    }
    if (S::kPostNorm && kind != K_G1 && (lane & 3) == 0)
      for (int T = 0; T < NT; ++T) {
        const int o = (R2 ? 2 * bi + T : bi) * 128 + 16 * q + g;
        cp_async4(ews + 16 + (g * NT + T) * 8, p.nw + o);
        cp_async4(ews + 20 + (g * NT + T) * 8, p.nw + o + 8);
      }
  }
  u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, af[DP ? 4 : 2][4];
  int prev = -1;
  constexpr u32 RGS = DP ? 1024 : 512, SO = DP ? SOFF_R2 : R2 ? SOFF_R1 : SOFF;
  const u32 ring = smem_u32(sm), wo = (R2 ? WOFF_R : WOFF) + q * RGS + lane * 16,
            so = SO + (q >> 1) * 384 + lane * 12;
  constexpr int KU = 4 * KPS;
#pragma unroll 1
  for (int kc = 0; kc < nkb / KU; ++kc) {
#pragma unroll
    for (int ku = 0; ku < KU; ++ku) {
      const int kk = ku & 3, kq = KPS == 2 ? (ku & 1) : 0;
      if (kq == 0) mbar_wait(bars + B_F + 8 * slot, ph);
      const u32 st = ring + slot * SLOTB;
      const uint4 wq = lds128(st + wo + kq * 512);
      uint4 wq2 = wq;
      if (R2) wq2 = lds128(st + wo + 8 * RGS + kq * 512);
      if (kk == 0) {
        s0 = lds32(st + so);
        s1 = lds32(st + so + 4);
        s2 = lds32(st + so + 8);
        if (R2) {
          s3 = lds32(st + so + 1536);
          s4 = lds32(st + so + 1540);
          s5 = lds32(st + so + 1544);
        }
      }
      const u32 R = scale_kk(kk, odd, s0, s1, s2), Rb = !R2 ? 0 : scale_kk(kk, odd, s3, s4, s5);
      const u64 desc = sw128_desc(st + kq * N * 128);
      for (int j = 0; j < 4; ++j) {
        const int b0 = DP ? 2 * (j & 1) : R2 ? 0 : j & 1;
        decode(wq, j, R, af[b0]);
        if constexpr (DP) decode(wq2, j, Rb, af[b0 + 1]);
        wg_fence();
        wgmma_rs<N>(acc, af[b0], desc + 2 * j);
        if constexpr (DP) wgmma_rs<N>(acc + 64, af[b0 + 1], desc + 2 * j);
        wg_commit();
        wg_wait<1>();
        if (j == 0 && kq == 0 && prev >= 0) {
          __syncwarp();
          if (lane == 0) release_slot(bars + B_E + 8 * prev, crank);
        }
        if constexpr (R2 && !DP) {
          decode(wq2, j, Rb, af[1]);
          wg_fence();
          wgmma_rs<N>(acc + 64, af[1], desc + 2 * j);
          wg_commit();
          wg_wait<1>();
        }
      }
      if (kq == KPS - 1) {
        prev = slot;
        advance<NSLOT>(slot, ph);
      }
    }
  }
  wg_wait<0>();
  __syncwarp();
  if (lane == 0) release_slot(bars + B_E + 8 * prev, crank);
  const int kind = d[0], bi = d[2], row0 = d[3], tile = d[5];
  u8* stg = sm + STG;
  mbar_wait(bars + B_PE, pph ^ 1);
  pph ^= 1;
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncwarp();
  // global scale(s) of the item: gate (or down) at word 0, up at word 1
  const float cs = AS(const float, ew)[0] * 16384.f;
  const float csu = kind == K_G1 ? AS(const float, ew)[1] * 16384.f : 0.f;
  float nwv[NT][2];
  for (int T = 0; T < NT; ++T)
    for (int u = 0; u < 2; ++u)
      nwv[T][u] = S::kPostNorm ? AS(const float, ew + 16 + (g * NT + T) * 8)[u] : 1.f;
  {
    const u32 tx = tid_x();
    const int q = (tx >> 5) - 4, lane = tx & 31;
#pragma unroll
    for (int T = 0; T < NT; ++T) {
      const int ao = 64 * T;
      if (kind == K_G1) {
        const u32 sbh = smem_u32(stg + T * 16384);
        const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int nb = 0; nb < N / 8; nb += 4) {
          u32 r[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int nj = nb + i;
            if (nj < N / 8)
              r[i] = bf2(silu(acc[ao + 4 * nj] * cs) * (acc[ao + 4 * nj + 2] * csu),
                         silu(acc[ao + 4 * nj + 1] * cs) * (acc[ao + 4 * nj + 3] * csu));
          }
          const int tok = 8 * (nb + mi) + rr;
          const u32 a = sbh + tok * 128 + ((q ^ (tok & 7)) << 4);
          if (nb + 4 <= N / 8)
            stmatrix_t(a, r);
          else
            stmatrix_t2(a, r[0], r[1]);
        }
      } else {
        const float n0 = nwv[T][0], n8 = nwv[T][1];
        constexpr int RS = R2 ? 528 : 272;
        const int m = lane >> 3, rr = lane & 7, c = 2 * (q & 3) + (m & 1);
        const u32 sb = smem_u32(stg) + (8 * (m >> 1) + rr) * RS + (R2 ? T * 256 : 0) +
                       (q >> 2) * 128 + (c << 4);
        for (int j2 = 0; j2 < N / 16; ++j2) {
          u32 r[4];
          for (int u = 0; u < 2; ++u) {
            const int ai = ao + 4 * (2 * j2 + u);
            r[2 * u] = bf2(acc[ai] * cs * n0, acc[ai + 1] * cs * n0);
            r[2 * u + 1] = bf2(acc[ai + 2] * cs * n8, acc[ai + 3] * cs * n8);
          }
          stmatrix_t(sb + 16 * RS * j2, r);
        }
      }
    }
  }
  fence_async();
  bar_sync(1, 256);
  if (q == 0 && lane == 0) {
    pq[0] = kind;
    pq[1] = bi;
    pq[2] = row0;
    pq[3] = N;
    pq[4] = tile;
    pq[5] = d[1];
    mbar_arrive(bars + B_PF);
  }
}
DEVI void transpose4(uint4& v, bool b0, bool b1) {
  u32 s0 = b1 ? v.x : v.z, s1 = b1 ? v.y : v.w;
  s0 = shfl_xor(s0, 2);
  s1 = shfl_xor(s1, 2);
  if (b1) {
    v.x = s0;
    v.y = s1;
  } else {
    v.z = s0;
    v.w = s1;
  }
  s0 = b0 ? v.x : v.y;
  s1 = b0 ? v.z : v.w;
  s0 = shfl_xor(s0, 1);
  s1 = shfl_xor(s1, 1);
  if (b0) {
    v.x = s0;
    v.z = s1;
  } else {
    v.y = s0;
    v.w = s1;
  }
}
// rev of one unit; with p.cl it is also the counting pass
DEVI void list_unit(const Params& p, int u, int ctid, int* hist) {
  if (ctid < NE) hist[ctid] = 0;
  bar_sync(1, 256);
  const int t = 128 * u + ctid;
  int e8[8], lr[8];
  if (ctid < 128 && t < p.M) {
    const int4 a = AS(const int4, p.ids)[2 * t], c = AS(const int4, p.ids)[2 * t + 1];
    e8[0] = a.x;
    e8[1] = a.y;
    e8[2] = a.z;
    e8[3] = a.w;
    e8[4] = c.x;
    e8[5] = c.y;
    e8[6] = c.z;
    e8[7] = c.w;
    for (int j = 0; j < TOPK; ++j)
      if (e8[j] >= 0 && e8[j] < p.ne) lr[j] = atomicAdd(hist + e8[j], 1);
  }
  bar_sync(1, 256);
  if (ctid < NE) hist[NE + ctid] = hist[ctid] ? atomicAdd(p.ctr + C_LB + ctid, hist[ctid]) : 0;
  bar_sync(1, 256);
  if (ctid < 128 && t < p.M)
    for (int j = 0; j < TOPK; ++j)
      if (e8[j] >= 0 && e8[j] < p.ne) {
        const int r = hist[NE + e8[j]] + lr[j];
        p.rev[rev_at(p, e8[j], r)] = t * 8 + j;
        p.rev[rev_at(p, NE, t * 8 + j)] = r;
      }
  bar_sync(1, 256);
  if (ctid == 0) {
    __threadfence();
    red_release(p.ctr + (p.cl ? C_CDONE : C_LDONE), 1);
  }
}
// Every gather row is claimed from C_GQUEUE, so the rows (and the G1 items waiting on
// them) never depend on a CTA that is not resident: the kernel completes with any
// subset of its CTAs resident (SM partitions, MPS limits, other work on the device).
template <class S>
DEVI void helper_gather(const Params& p, const Plan& pl, int lane) {
  if (lane == 0) wait_geq(p.ctr + (p.cl ? C_CDONE : C_LDONE), p.cl ? pl.u1 - pl.u0 : pl.nlu);
  __syncwarp();
  const int c = lane & 7, lo = 8 * (lane >> 3);
  int pos = 0, base = 0;
  constexpr int END = -(1 << 30);
  auto locate = [&](int k) {
    while (pos < NE && base + pl.cnt[pl.ord[pos]] <= k) base += pl.cnt[pl.ord[pos++]];
    return pos >= NE ? END : pl.ord[pos] << 20 | (k - base);
  };
  auto claim_k = [&] {
    int k = 0;
    if (lane == 0) k = atomicAdd(p.ctr + C_GQUEUE, 1);
    return shfl(k, 0);
  };
  // asm volatile keeps the next route's load ahead of this row's data loads
  auto route = [&](int x) {
    int r = 0;
    if (x >= 0)
      asm volatile("ld.global.cg.b32 %0, [%1];"
                   : "=r"(r)
                   : "l"(p.rev + rev_at(p, x >> 20, x & 0xFFFFF)));
    return r;
  };
  constexpr int kCh = S::kHidden / 256;  // 32-lane x 16-byte chunks per row
  int x = locate(claim_k()), tj = route(x);
  while (x != END) {
    const int e = x >> 20, t = tj >> 3;
    const int le = lane < TOPK ? __ldcg(p.ids + 8 * t + lane) : -1;
    int fr = lane < TOPK ? __ldcg(p.rev + rev_at(p, NE, 8 * t + lane)) : 0;
    const int xn = locate(claim_k());
    const bool v = le >= 0 && le < p.ne;
    // a token is copied once, by its first route in plan order, to all its rows
    if (__reduce_min_sync(~0u, v ? pl.rk[le] * TOPK + lane : NE * TOPK) !=
        pl.rk[e] * TOPK + (tj & 7)) {
      x = xn;
      tj = route(x);
      continue;
    }
    const u32 dm = ballot(v);
    if (!v) fr = 0;
    const int drow = v ? pl.off[le] + fr : 0;
    const uint4* src = AS(const uint4, p.x + (size_t)t * S::kHidden) + lo + 2 * (c & 3) + (c >> 2);
    const int tjn = route(xn);
#pragma unroll
    for (int h = 0; h < kCh; h += 8) {
      uint4 w[8];
      for (int k2 = 0; k2 < 8; ++k2)
        if (h + k2 < kCh) w[k2] = ld_na(src + 32 * (h + k2));
      for (int k2 = 0; k2 < 8; ++k2)
        if (h + k2 < kCh) transpose4(w[k2], lane & 1, lane & 2);
      for (u32 m = dm; m; m &= m - 1) {
        uint4* dst = AS(uint4, p.xt + (size_t)shfl(drow, __ffs(m) - 1) * S::kHidden) + lo + c;
        for (int k2 = 0; k2 < 8; ++k2)
          if (h + k2 < kCh) dst[32 * (h + k2)] = w[k2];
      }
    }
    __syncwarp();
    if (v) red_release(p.ctr + C_READY + tile_of_row(pl, le, fr), 1);
    x = xn;
    tj = tjn;
  }
}
template <class S>
DEVI void helper(const Params& p, const Plan& pl, u32 bars, int lane, bool first) {
  if (first) mbar_wait(bars + B_PL, 0);
  helper_gather<S>(p, pl, lane);
}

template <class S>
__global__ void __launch_bounds__(384, 1) moe_prefill_kernel(const __grid_constant__ Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(1024) u8 sm[];
  const u32 bars = smem_u32(sm + BAR);
  int* iq = AS(int, sm + BAR + 128);
  int* pq = iq + 16;
  Plan& pl = *AS(Plan, sm + BAR + 256);
  static_assert(256 + sizeof(Plan) <= EWS, "plan overlaps the epilogue inputs");
  const int tid = threadIdx.x;
  pdl_trigger();  // the next chunk's launch reads nothing before pdl_wait
  pdl_wait();
  const int u0 = p.first ? 0 : __ldcg(p.ctr + C_WORDS);
  if (u0 >= (p.M + 127) >> 7) return;
  if (tid == 0) {
    pl.u0 = u0;
    for (int i = 0; i < NSLOT; ++i) {
      mbar_init(bars + B_F + 8 * i, 1);
      mbar_init(bars + B_E + 8 * i, 16);
    }
    mbar_init(bars + B_MF, 1);
    mbar_init(bars + B_ME, 1);
    for (int i = 0; i < 2; ++i) {
      mbar_init(bars + B_IF + 8 * i, 1);
      mbar_init(bars + B_IE + 8 * i, 8);
    }
    for (int b = B_PF; b <= B_PL; b += 8) mbar_init(bars + b, 1);
    mbar_init_fence();
    for (int i = 0; i < NW; ++i) {
      tmap_acquire(p.txh + i);
      tmap_acquire(p.thh + i);
      if (i < 2) {
        tmap_acquire(p.tx2 + i);
        tmap_acquire(p.th2 + i);
      }
    }
  }
  cl_sync();
  if (tid < 128) {
    nreg_dec<56>();
    if (tid == 0)
      producer<S>(p, sm, pl, bars, iq);
    else if ((tid >> 5) == 1)
      publisher<S>(p, sm, bars, pq, pl, tid & 31);
    else if (tid >= 64)
      helper<S>(p, pl, bars, tid & 31, true);
  } else {
    nreg_inc<224>();
    const int ctid = tid - 128, q = ctid >> 5, lane = ctid & 31;
    int* hist = AS(int, sm + STG);
    int slot = 0;
    u32 ph = 0, qn = 0, pph = 0;
    for (;;) {
      const u32 qs = qn & 1, qph = (qn >> 1) & 1;
      mbar_wait(bars + B_IF + 8 * qs, qph);
      const int* d = iq + 8 * qs;
      const int kind = d[0];
      if (kind == K_CNT) {
        if (p.cl)
          list_unit(p, d[1], ctid, hist);
        else
          count_unit(p, d[1], q, lane, ctid, hist);
      } else if (kind == K_PLAN) {
        if (q == 0) {
          make_plan<S>(p, pl, lane);
          if (lane == 0) mbar_arrive(bars + B_PL);
        }
      } else if (kind == K_CMB)
        helper<S>(p, pl, bars, lane, false);
      else if (kind == K_LIST)
        list_unit(p, d[1], ctid, hist);
      else if (kind >= 0) {
#define ITEM_CASE(W)                                        \
  case W:                                                   \
    item<S, W>(p, sm, bars, pq, d, q, lane, slot, ph, pph); \
    break;
        switch (d[4]) {
          ITEM_CASE(32)
          ITEM_CASE(64)
          ITEM_CASE(96)
          ITEM_CASE(128)
          ITEM_CASE(160)
          ITEM_CASE(192)
          ITEM_CASE(224)
          ITEM_CASE(256)
          ITEM_CASE(240)
        }
#undef ITEM_CASE
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + B_IE + 8 * qs);
      ++qn;
      if (kind < 0) break;
    }
    if (ctid == 0) {
      mbar_wait(bars + B_PE, pph ^ 1);
      pq[0] = -1;
      mbar_arrive(bars + B_PF);
    }
  }
  // last CTA zeroes counters, sets the batch cursor
  asm volatile("barrier.sync 2, 384;" ::: "memory");
  if (tid < 32) {
    int last = 0;
    if (tid == 0) {
      __threadfence();
      last = atomicAdd(p.ctr + C_EXIT, 1) == (int)gridDim.x - 1;
    }
    if (shfl(last, 0)) {
      __threadfence();
      for (int i = tid; i < C_WORDS; i += 32) p.ctr[i] = 0;
      if (tid == 0) p.ctr[C_WORDS] = pl.u1;
    }
  }
  cl_sync();
#endif
}

// out[r] = sum_j w[r, j] * f(recv[j, r]) over the routes with a local expert,
// f = RMSNorm (weight already applied by the GEMM2 epilogue) or identity.
template <int kHidden, bool kPostNorm>
__global__ void __launch_bounds__(256, kHidden <= 4096 ? 2 : 1)
    moe_finalize_kernel(const u16* recv, const int* ids, const float* tw, u16* out, int rows,
                        int num_experts, float eps) {
  static_assert(kHidden % 256 == 0);
  constexpr int kRowVecs = kHidden / 8, kVPL = kHidden / 256;
  extern __shared__ __align__(16) u8 fsm[];
  uint4* ys = AS(uint4, fsm);
  float* cs = AS(float, fsm + TOPK * kHidden * 2);
  const int lane = threadIdx.x & 31, j = threadIdx.x >> 5;
  const uint4* rj = AS(const uint4, recv) + (size_t)j * rows * kRowVecs + lane;
  int r = blockIdx.x;
  uint4 v[kVPL];
  bool live = false;
  auto load = [&](int r) {
    const int e = __ldg(ids + r * TOPK + j);
    live = e >= 0 && e < num_experts;
    for (int i = 0; i < kVPL; ++i)
      v[i] = live ? __ldcs(rj + (size_t)r * kRowVecs + i * 32) : make_uint4(0u, 0u, 0u, 0u);
  };
  load(r);
  for (;;) {
    float ss = 0.f;
    for (int i = 0; i < kVPL; ++i) {
      if constexpr (kPostNorm)
        for (int h = 0; h < 4; ++h) {
          const float lo = bf_lo(word4(v[i], h)), hi = bf_hi(word4(v[i], h));
          ss = fmaf(lo, lo, ss);
          ss = fmaf(hi, hi, ss);
        }
      ys[j * kRowVecs + i * 32 + lane] = v[i];
    }
    if constexpr (kPostNorm)
      for (int o = 16; o; o >>= 1) ss += shfl_xor(ss, o);
    if (lane == 0) {
      const float w = live ? tw[r * TOPK + j] : 0.f;
      cs[j] = kPostNorm ? w * rsqrtf(ss * (1.f / kHidden) + eps) : w;
    }
    const int rn = r + gridDim.x;
    if (rn < rows) load(rn);
    __syncthreads();
    float c[TOPK];
    for (int k = 0; k < TOPK; ++k) c[k] = cs[k];
#pragma unroll
    for (int u = 0; u < (kRowVecs + 255) / 256; ++u) {
      const int ch = 256 * u + threadIdx.x;
      if (kRowVecs % 256 != 0 && ch >= kRowVecs) break;
      float o[8];
      for (int h = 0; h < 8; ++h) o[h] = 0.f;
      for (int k = 0; k < TOPK; ++k) {
        const uint4 w = ys[k * kRowVecs + ch];
        for (int h = 0; h < 4; ++h) {
          o[2 * h] = fmaf(c[k], bf_lo(word4(w, h)), o[2 * h]);
          o[2 * h + 1] = fmaf(c[k], bf_hi(word4(w, h)), o[2 * h + 1]);
        }
      }
      AS(uint4, out)
      [(size_t)r * kRowVecs + ch] =
          make_uint4(bf2(o[0], o[1]), bf2(o[2], o[3]), bf2(o[4], o[5]), bf2(o[6], o[7]));
    }
    if (rn >= rows) break;
    r = rn;
    __syncthreads();
  }
}
template <int kHidden>
constexpr int finalize_smem_bytes() {
  return TOPK * kHidden * 2 + 32;
}

#undef DEVI
#undef AS

}  // namespace pf
}  // namespace sm90_w4a16_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_PREFILL_CUH_
