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
 * SM90 NVFP4 W4A16 fused MoE, small / medium batch kernel (1 .. 1024 tokens
 * per launch): gate/up GEMM -> SwiGLU -> down GEMM -> (optional per-route
 * RMSNorm) -> top-k weighted sum, in one persistent launch.
 *
 * Adapted from gauravjain14/muse-okc-submissions (muse_moe_w4a16, Apache-2.0),
 * which itself builds on vjeux/okc-2026 and alygg77/okc-muse-spark-kernels
 * (Apache-2.0); see NOTICE. Changes from the upstream kernel: hidden and
 * intermediate sizes are template parameters (they were 4096 / 4096), the
 * expert count and the post-norm epsilon are runtime parameters, the gate and
 * up halves take separate global scales, the per-route RMSNorm is optional,
 * and the multi-GPU paths (sequence-parallel scatter, replica plans, remote
 * experts) were removed.
 *
 * Structure (one CTA per SM, 384 threads, persistent):
 *   - warps 0-7  consumers: decode FP4 weights in registers (exact BF16
 *                products of code x block scale), mma.sync / RS-wgmma against
 *                BF16 activation tiles in shared memory, FP32 accumulation;
 *   - warp 8     producer: claims work items from a global queue and streams
 *                weight blocks + activation tiles into an NS-deep ring with
 *                cp.async.bulk;
 *   - warp 9     signals tile completion to other CTAs;
 *   - warps 10-11 asynchronous epilogue (SwiGLU to BF16 for phase 1, scaled
 *                output rows for phase 2); with >= 65 tokens warps 9-11 also
 *                gather the routed activation rows.
 * Work item = (token tile of one expert) x (one block of weight rows) x full K.
 * Phase 1 is the gate/up GEMM, phase 2 the down GEMM (optionally split along
 * K for the last tiles), then every CTA combines finished tokens.
 * Counters in `ctr` are self-resetting (two alternating parity blocks), so the
 * workspace must be zero-initialized once and must not be shared by launches
 * that can run concurrently.
 */
#ifndef FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_DECODE_CUH_
#define FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_DECODE_CUH_

#include "common.cuh"

namespace flashinfer {
namespace sm90_w4a16_moe {
namespace dec {

#define DEVI FI_W4A16_DEVI
#define AS FI_W4A16_AS

struct __align__(16) StageInfo {
  int item, phase, tile, rb, ks, st, last, n, p0, e, pad0, pad1;
};

// Tile order of a lane's expert: count `key` descending, ties by lane (key < 2^(nb + 1)).
DEVI int key_rank(int key, int nb) {
  u32 gtm = 0u, eqm = ~0u;
  for (int b = nb; b >= 0; --b) {
    const bool one = (key >> b) & 1;
    const u32 B = ballot(one);
    if (one)
      eqm &= B;
    else {
      gtm |= eqm & B;
      eqm &= ~B;
    }
  }
  return __popc(gtm) + __popc(eqm & ((1u << (threadIdx.x & 31)) - 1u));
}
// Alternate heavy and light experts, except the 5 lightest (c = this lane's route count).
DEVI int mix_rank(int rank, int c) {
  const int nz = __popc(ballot(c > 0)), m = nz - min(5, nz);
  return rank >= m ? rank : rank < (m + 1) / 2 ? 2 * rank : 2 * (m - 1 - rank) + 1;
}
#define LD_IDS(ids, t, v0, v1)             \
  {                                        \
    v0 = AS(const int4, ids)[(t) * 2];     \
    v1 = AS(const int4, ids)[(t) * 2 + 1]; \
  }
#define EV8(v0, v1) {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w}
DEVI void mbar_arrive_n(u32 bar, u32 n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(n) : "memory");
}
DEVI void mbar_poll(u32 bar, u32 parity) {
  u32 ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, "
        "p; }"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
}
template <int HINT = 1000000>
DEVI void mbar_sleep(u32 bar, u32 parity) {
  u32 ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, "
        "0, p; }"
        : "=r"(ok)
        : "r"(bar), "r"(parity), "n"(HINT)
        : "memory");
}
DEVI int ld_relaxed(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
DEVI bool elect_one() {
  u32 p;
  asm volatile("{ .reg .pred e; elect.sync _|e, 0xffffffff; selp.u32 %0, 1, 0, e; }" : "=r"(p));
  return p;
}
DEVI void consumer_sync() { bar_sync(1, 256); }

// FP4 code bits (sign + 3 magnitude bits) placed into a BF16 pair.
#define M_BF 0x81C081C0u

// The decode constants live in constant memory so that the compiler keeps the
// multiplies as single IMADs instead of expanding them.
__constant__ u32 c_s6o[7] = {16u, 256u, 64u, 8u, 128u, 2048u, 512u};
__constant__ u32 c_s6oc[2] = {0x78007800u, 0x40004000u};
// One stage of one consumer warp: 2 x 4 x 16 B of FP4 codes and the 6-bit scales
// of its two 16-row groups, expanded to BF16 multipliers (R).
DEVI void ld_stage(const u8* wbase, const u8* sclb, uint4 (&qw)[2][4], u32 (&R)[8]) {
  for (int kk = 0; kk < 4; ++kk) {
    qw[0][kk] = *AS(const uint4, wbase + kk * 512);
    qw[1][kk] = *AS(const uint4, wbase + 2048 + kk * 512);
  }
  const u32* sw = AS(const u32, sclb);
  const u32 w0 = sw[0], w1 = sw[1], w2 = sw[2];
  constexpr u32 M6 = 0x03F003F0u, NB = 0x00F000F0u;
  const u32 C = c_s6oc[0];
  R[0] = (w0 & M6) | C;
  R[4] = ((w0 >> 6) & M6) | C;
  R[1] = (w1 & M6) | C;
  R[5] = ((w1 >> 6) & M6) | C;
  R[2] = (w2 & M6) | C;
  R[6] = ((w2 >> 6) & M6) | C;
  const u32 vg = ((w0 * c_s6o[0]) & NB) | ((w2 * c_s6o[1]) & ~NB);
  R[3] = (vg & M6) | C;
  const u32 vh = ((w1 * c_s6o[0]) & NB) | ((w2 * c_s6o[2]) & ~NB);
  R[7] = (vh & M6) | C;
}

__constant__ u32 c_x8[3] = {4u, 64u, 16u};
// 8 FP4 codes of a stored word pair -> 4 BF16x2 A-fragment registers times the block scale.
DEVI void dq4(u32 q, u32 qp, int j, u32 R, u32* a) {
  const u32 F = j == 0 ? __byte_perm(q, qp, 0x6341) : __byte_perm(qp, q, 0x6341);
  u32 o[4];
  o[0] = q & M_BF;
  if (j == 0) {
    o[1] = ((q * c_x8[0]) ^ F) & M_BF;
    o[2] = (q * c_x8[1]) & M_BF;
    o[3] = rotl(q, 12) & M_BF;
  } else {
    o[1] = (q * c_x8[2]) & M_BF;
    o[2] = rotl(q, 10) & M_BF;
    o[3] = (rotl(q, 14) ^ F) & M_BF;
  }
  a[0] = bmul2s<0>(o[0], R);
  a[1] = bmul2s<1>(o[1], R);
  a[2] = bmul2s<0>(o[2], R);
  a[3] = bmul2s<1>(o[3], R);
}
// Weights-loaded arrival (bar id depends on every load; c_z0 = 0). Unused (ew < 0) in
// the single-GPU kernel; kept so that the stage loops stay the validated code.
__constant__ u32 c_z0;
DEVI void ew_arv(const uint4 (&q)[2][4], const u32* R, int ew) {
  u32 z = R[0] ^ R[1] ^ R[2];
  for (int i = 0; i < 8; ++i) z ^= q[i / 4][i % 4].x;
  if (ew >= 0) bar_arrive(ew + (z & c_z0), 288);
}
DEVI void ld_x(uint4 (&b)[2], const u8* p) {
  b[0] = *AS(const uint4, p);
  b[1] = *AS(const uint4, p + 16);
}
DEVI void mma_x(float* c, const u32 (&a)[4], const uint4 (&b)[2], int jj) {
  const u32 B[8] = {b[0].x, b[0].y, b[0].z, b[0].w, b[1].x, b[1].y, b[1].z, b[1].w};
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(B[2 * jj]), "r"(B[2 * jj + 1]));
}
DEVI void dec_kr(const uint4 (&qw)[2][4], const u32 (&R6)[8], int kk, int r, u32 (&A)[4][4]) {
  const uint4 q = qw[r][kk];
  const u32 R = R6[4 * r + kk];
  dq4(q.x, q.y, 0, R, A[0]);
  dq4(q.y, q.x, 1, R, A[1]);
  dq4(q.z, q.w, 0, R, A[2]);
  dq4(q.w, q.z, 1, R, A[3]);
}
DEVI void dec_k(const uint4 (&qw)[2][4], const u32 (&R6)[8], int kk, u32 (&A)[2][4][4]) {
  for (int r = 0; r < 2; ++r) dec_kr(qw, R6, kk, r, A[r]);
}

// 32x32 bit transpose across the warp (expert-major <-> token-major route masks).
DEVI u32 tr32(u32 x) {
  const int lane = threadIdx.x & 31;
  u32 m = 0x0000FFFFu;
  for (int j = 16; j != 0; j >>= 1, m ^= m << j) {
    const u32 y = shfl_xor(x, j);
    x ^= (lane & j) ? (((y >> j) ^ x) & m) : ((((x >> j) ^ y) & m) << j);
  }
  return x;
}
// clang-format off
#define FOR_ACC(B0) \
  _Pragma("unroll") for (int a = 0; a < 2; ++a) \
  _Pragma("unroll") for (int b = (B0); b < NT; ++b) \
  _Pragma("unroll") for (int c = 0; c < 4; ++c)
#define ZERO_ACC(B0) FOR_ACC(B0) { acc[a][b][c] = 0.f; acc2[a][b][c] = 0.f; }
// clang-format on
template <int NT>
DEVI void pin_acc(float (&acc)[2][NT][4]) {
  FOR_ACC(0) asm volatile("" : "+f"(acc[a][b][c])::"memory");
}

DEVI u64 sw128_dlo(u32 lo) { return (u64)lo | ((u64)64 << 32) | ((u64)1 << 62); }
// One stage (256 K) with RS-wgmma: decoded weights are the A operand, the token tile in
// shared memory (128B swizzle) is B.
template <int NT, int NTL, bool RGC>
DEVI void stage_compute_wg(float (&acc)[2][NT][4], const u8* wbase, const u8* sclb, u32 xlo, int T,
                           int ew) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  if constexpr (RGC) {
    u32 A[2][4][4];
    for (int kk = 0; kk < 4; ++kk) {
      for (int r = 0; r < 2; ++r) {
        dec_kr(qw, R6, kk, r, A[r]);
        wg_fence();
        for (int jj = 0; jj < 4; ++jj)
          wgmma_rs<8 * NTL>(acc[r][0], A[r][jj], sw128_dlo(xlo + kk * T * 8 + 2 * jj));
        wg_commit();
        if (kk + r == 1 && r) ew_arv(qw, R6, ew);
        wg_wait<1>();
      }
    }
    wg_wait<0>();
    pin_acc<NT>(acc);
    return;
  }
  const u64 xd = sw128_dlo(xlo);
  u32 A[2][2][4][4];
  for (int kk = 0; kk < 4; ++kk) {
    u32(&Ab)[2][4][4] = A[kk & 1];
    dec_k(qw, R6, kk, Ab);
    wg_fence();
    for (int jj = 0; jj < 4; ++jj)
      for (int r = 0; r < 2; ++r)
        wgmma_rs<8 * NTL>(acc[r][0], Ab[r][jj], xd + (u32)(kk * T * 8) + 2 * jj);
    wg_commit();
    wg_wait<1>();
  }
  wg_wait<0>();
  pin_acc<NT>(acc);
}

// One stage with mma.sync (token tiles of 8 / 16 rows).
template <int NT, int NTL>
DEVI void stage_compute(float (&acc)[2][NT][4], const u8* wbase, const u8* sclb, const u8* xbase,
                        int XP) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  uint4 bx[2][NTL][2];
  for (int nt = 0; nt < NTL; ++nt) ld_x(bx[0][nt], xbase + nt * 8 * XP);
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3)
      for (int nt = 0; nt < NTL; ++nt)
        ld_x(bx[(kk + 1) & 1][nt], xbase + nt * 8 * XP + (kk + 1) * 128);
    u32 A[2][4][4];
    dec_k(qw, R6, kk, A);
    for (int nt = 0; nt < NTL; ++nt)
      for (int jj = 0; jj < 4; ++jj)
        for (int r = 0; r < 2; ++r) mma_x(acc[r][nt], A[r][jj], bx[kk & 1][nt], jj);
  }
}

template <int NTL>
struct DsCarry {
  u32 A[2][4][4];
  uint4 b[NTL][2];
};
template <int SPL, int NT, int NTL>
DEVI void ds_cmma(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const DsCarry<NTL>& c) {
  for (int nt = 0; nt < NTL; ++nt)
    for (int jj = 2; jj < 4; ++jj)
      for (int r = 0; r < 2; ++r)
        mma_x(SPL && (jj & 1) ? acc2[r][nt] : acc[r][nt], c.A[r][jj], c.b[nt], jj);
}
// last chunk's jj 2..3 MMAs are deferred (cr) to the next stage / ds_flush
template <int SPL, int NT, int NTL>
DEVI void stage_ds(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const u8* wbase, const u8* sclb,
                   const u8* xbase, int XP, DsCarry<NTL>& cr, int ew) {
  uint4 qw[2][4];
  u32 R6[8];
  ld_stage(wbase, sclb, qw, R6);
  uint4 bx[4][NTL][2];
  for (int nt = 0; nt < NTL; ++nt) ld_x(bx[0][nt], xbase + nt * 8 * XP);
  ds_cmma<SPL, NT, NTL>(acc, acc2, cr);
  u32 A[4][2][4][4];
  for (int kk = 0; kk < 4; ++kk) {
    if (kk < 3)
      for (int nt = 0; nt < NTL; ++nt) ld_x(bx[kk + 1][nt], xbase + nt * 8 * XP + (kk + 1) * 128);
    dec_k(qw, R6, kk, A[kk]);
    for (int nt = 0; nt < NTL; ++nt)
      for (int jj = 0; jj < (kk == 3 ? 2 : 4); ++jj)
        for (int r = 0; r < 2; ++r)
          mma_x(SPL && (jj & 1) ? acc2[r][nt] : acc[r][nt], A[kk][r][jj], bx[kk][nt], jj);
    if (kk == 0) ew_arv(qw, R6, ew);
  }
  for (int r = 0; r < 2; ++r)
    for (int j = 0; j < 4; ++j)
      for (int e = 0; e < 4; ++e) cr.A[r][j][e] = A[3][r][j][e];
  for (int n = 0; n < NTL; ++n) {
    cr.b[n][0] = bx[3][n][0];
    cr.b[n][1] = bx[3][n][1];
  }
}
__host__ __device__ constexpr int ds_code(int rw, int k, int mw) {
  return rw == 8 && !(mw == 1 && k == 1) ? k : 0;
}
template <int C, int NT, int NTL>
DEVI void ds_flush(float (&acc)[2][NT][4], float (&acc2)[2][NT][4], const DsCarry<NTL>& cr) {
  ds_cmma<C == 2, NT, NTL>(acc, acc2, cr);
  if constexpr (C == 2)
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < NTL; ++b)
        for (int c = 0; c < 4; ++c) {
          acc[a][b][c] += acc2[a][b][c];
          acc2[a][b][c] = 0.f;
        }
}

#define PUT_INFO(f, a, b, c) \
  {                          \
    int4* fi = AS(int4, f);  \
    fi[0] = a;               \
    fi[1] = b;               \
    fi[2] = c;               \
  }
DEVI void decode_item2(int item, int ntiles, int tsplit, int NRB1, int NRB2, int KSX, int& phase,
                       int& tile, int& rb, int& ks, int& kss) {
  const int n1 = ntiles * NRB1;
  if (item < n1) {
    phase = 1;
    tile = item / NRB1;
    rb = item - tile * NRB1;
    ks = 0;
    kss = 1;
    return;
  }
  phase = 2;
  int j = item - n1;
  const int na = tsplit * NRB2;
  if (j < na) {
    kss = 1;
    tile = j / NRB2;
    rb = j - tile * NRB2;
    ks = 0;
  } else {
    kss = KSX;
    j -= na;
    const int t2 = j / (NRB2 * KSX);
    j -= t2 * NRB2 * KSX;
    tile = tsplit + t2;
    rb = j / KSX;
    ks = j - rb * KSX;
  }
}

// Kernel arguments. Pointers to BF16 data are u16, packed weights u32.
struct Params {
  const u16* x;         // [M, kHidden] activations
  const int* ids;       // [M, 8] local expert ids, < 0 or >= num_experts: skipped
  const float* topk_w;  // [M, 8] route weights
  int M;
  const u32* w13;        // packed gate/up weights
  const float* gs13;     // [E] gate global scale
  const u32* w2;         // packed down weights
  const float* gs2;      // [E] down global scale
  const float* nw1;      // [kHidden] post-norm weight (kPostNorm only)
  u16* inter;            // workspace: SwiGLU output tiles
  float* ypart;          // workspace: split-K partial down rows (FP32)
  u16* out;              // [M, kHidden] output
  int* ctr;              // workspace: self-resetting counters (zero-initialized once)
  int pmax;              // 8 * M
  u16* xs;               // workspace: gathered activation tiles
  u16* yh;               // workspace: full-K down rows (BF16)
  float* scr;            // workspace: per-CTA epilogue spill
  const float* gs13_up;  // [E] up global scale
  int num_experts;       // <= 32
  float eps;             // post-norm epsilon (kPostNorm only)
};

DEVI void st_bf4(u16* p, float4 v) { *AS(uint2, p) = make_uint2(bf2(v.x, v.y), bf2(v.z, v.w)); }
DEVI float4 ld_bf4(const u16* p) {
  const uint2 u = __ldcg(AS(const uint2, p));
  return make_float4(bf_lo(u.x), bf_hi(u.x), bf_lo(u.y), bf_hi(u.y));
}
DEVI float rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// SwiGLU of two gate / up accumulator pairs (rows g and g + 8 of a 16-row group):
// silu(gate) * up = gate * up * sigmoid(gate), with gke = -log2(e) * gate scale and
// gs2 = gate scale * up scale.
DEVI u32 swiglu_bf2(const float* rr, float gke, float gs2) {
  const float2 gv = *AS(const float2, rr), uv = *AS(const float2, rr + 8);
  const float g0 = gv.x, g1 = gv.y, u0 = uv.x, u1 = uv.y;
  const float e0 = ex2(g0 * gke), e1 = ex2(g1 * gke), r0 = rcp_ftz(1.f + e0),
              r1 = rcp_ftz(1.f + e1);
  return bf2((g0 * u0) * (gs2 * r0), (g1 * u1) * (gs2 * r1));
}

// First NS stages of a CTA's first item for the 1..32-token kernel, issued while
// the routing table is still being built.
template <int KC, int kHidden, int kInter>
__device__ __noinline__ void wp32_fst(const u32* bm, const u32* w13, const u16* x, int M, u8* info,
                                      u32 st0, u32 b0, int item, int nrb1, int ns, int stage,
                                      int wb, int sb, int srg, int rgs, int xp, int cmax) {
  constexpr int kNkc1 = kHidden / 256, kNrg1 = 2 * kInter / 16;
  const int lane = threadIdx.x & 31;
  const u32 bml = bm[lane];
  const int c = __popc(bml);
  const int tile0 = item / nrb1, rb0 = item - tile0 * nrb1;
  int key = c > 0 ? (c << 5) | (31 - lane) : -1;
  int p0 = 0, mx = -1;
#pragma unroll 1
  for (int k = 0; k <= tile0; ++k) {
    mx = __reduce_max_sync(~0u, key);
    if (mx < 0) break;
    if (k < tile0) {
      p0 += mx >> 5;
      if (lane == 31 - (mx & 31)) key = -1;
    }
  }
  if (mx < 0 || (mx >> 5) > cmax) return;
  const int e0 = 31 - (mx & 31);
  const u32 bme = shfl(bml, e0);
  const int n = __popc(bme);
  const int r = (lane < M && ((bme >> lane) & 1u)) ? __popc(bme & ((1u << lane) - 1u)) : -1;
  const u64 polw = policy_evict_first(), polx = policy_evict_last();
  const u8* wsrc = weight_block(w13, e0, kNkc1, kNrg1, 0, rb0 * rgs);
  const u8* ssrc = weight_scales(w13, e0, kNkc1, kNrg1, 0, rb0 * rgs);
#pragma unroll 1
  for (int st = 0; st < ns; ++st) {
    const u32 full = b0 + st * 8;
    if (lane == 0) {
      PUT_INFO(info + st * 48, make_int4(item, 1, tile0, rb0), make_int4(0, st, 0, n),
               make_int4(p0, e0, 0, 0))
      mbar_expect(full, wb + sb + (u32)(n * 512 * KC));
      for (int j = 0; j < KC; ++j) {
        bulk_g2s(st0 + st * stage + j * rgs * 2048,
                 wsrc + (size_t)(st * KC + j) * (kNrg1 / 16) * kWeightBlockBytes, rgs * 2048, full,
                 polw);
        bulk_g2s(st0 + st * stage + wb + j * rgs * srg,
                 ssrc + (size_t)(st * KC + j) * (kNrg1 / 16) * kWeightBlockBytes, rgs * srg, full,
                 polw);
      }
    }
    __syncwarp();
    if (r >= 0)
      bulk_g2s(st0 + st * stage + wb + sb + r * xp, x + (size_t)lane * kHidden + st * 256 * KC,
               512 * KC, full, polx);
  }
  __syncwarp();
}

// Compile-time shared-memory layout of one decode configuration. Shared with the
// host launcher (dynamic shared memory size) so that both use one formula.
//   RW: 16-row groups per consumer warp (8: 256-row weight blocks, 4: 128 rows with
//       the K range split over two warps), NT: 8-token columns per tile,
//   NS: ring depth, MW: 32-token words of the routing bitmap (M <= 32 * MW).
template <int RW, int NT, int NS, int MW>
struct DecodeLayout {
  static constexpr int KW = 8 / RW;
  static constexpr bool HB = RW == 4;
  static constexpr int RB = 32 * RW, RGS = RB / 16, KST = 256 * KW, T = 8 * NT, XP = KST * 2 + 16,
                       WB = RGS * 2048 * KW;
  static constexpr int WGMIN = KW == 1 ? 3 : 99;
  static constexpr bool HASWG = WGMIN < 99;
  static constexpr int SRG = 192, SB = RGS * SRG * KW;
  static constexpr int XB = HB ? 16 * XP : T * XP;
  static constexpr int STAGE =
      HASWG ? (WB + SB + XB + 1023) / 1024 * 1024 : (WB + SB + XB + 127) / 128 * 128;
  static constexpr int WPA_BUFB = NT == 4 ? (MW == 1 ? 16384 : 15360) : (MW <= 8 ? 13312 : 0);
  static constexpr int MAXP = MW * 32 * kTopKSlots, MAXT = kMaxLocalExperts + MAXP / T;
  static constexpr bool PAIRS = HB || NT < 8;
  static constexpr int ROUTE_BYTES =
      (32 * MW * 8 + MAXT * 16 + MAXP * (PAIRS ? 3 : 1) + 127) / 128 * 128;
  static constexpr int STB = (1024 + ROUTE_BYTES + 1023) / 1024 * 1024;
  static constexpr int kSmemBytes = STB + NS * STAGE + WPA_BUFB;
  static constexpr int kMaxTokens = 32 * MW;
};

// Stages per work item and the down-GEMM K split of one configuration.
//   ST1X: phase-1 stages (K = hidden), ST1H: phase-2 stages over the full K (= intermediate),
//   KSX: K split of the last phase-2 tiles (1: no split), STX = ST1H / KSX.
// Every item needs at least NS stages, so the split is reduced (down to none) when the
// intermediate size is small.
template <int kHidden, int kInter, int RW, int NS>
struct DecodeSplit {
  static constexpr int KW = 8 / RW;
  static constexpr int ST1X = kHidden / 256 / KW, ST1H = kInter / 256 / KW;
  static constexpr int KSX0 = RW == 4 ? 2 : 4;
  static constexpr int KSX = (ST1H % KSX0 == 0 && ST1H / KSX0 >= NS) ? KSX0
                             : (ST1H % 2 == 0 && ST1H / 2 >= NS)     ? 2
                                                                     : 1;
  static constexpr int STX = ST1H / KSX;
};

// Counter workspace layout (int32 words). Two parity blocks alternate between
// launches; each launch zeroes the other one, so they are self-resetting.
constexpr int kMaxTiles = kMaxLocalExperts + 1024 * kTopKSlots / 64;  // largest config: MW 32, T 64
constexpr int kCnt1Off = 8;                         // phase-1 done per (tile, k chunk)
constexpr int kGtcOff = kCnt1Off + kMaxTiles * 32;  // gathered rows per tile
constexpr int kCnt2Off = kGtcOff + kMaxTiles;       // phase-2 done per tile
constexpr int kParWords = kCnt2Off + kMaxTiles;
constexpr int kParOff = 65536, kParWord = kParOff + 16384;
constexpr int kCounterWords = kParWord + 64;
static_assert(kParWords <= kParOff, "counter parity block overflow");

template <int kHidden, int kInter, bool kPostNorm, int RW, int NT, int NS, int MW>
DEVI void moe_body(const Params& P) {
  using L = DecodeLayout<RW, NT, NS, MW>;
  static_assert(kHidden % 256 == 0 && kInter % 256 == 0, "hidden / intermediate % 256");
  static_assert(kHidden / 256 <= 32 && kInter / 256 <= 32, "hidden / intermediate <= 8192");
  constexpr int KW = L::KW;
  constexpr bool HB = L::HB;
  constexpr bool WSC = KW == 1 && !(NT == 4 && MW == 1);
  constexpr bool LEANC = WSC;
  constexpr bool IT2K = NT == 4 && MW == 1;
  constexpr int RB = L::RB, RGS = L::RGS, KST = L::KST, T = L::T, XP = L::XP, WB = L::WB;
  constexpr int WGMIN = L::WGMIN;
  constexpr bool HASWG = L::HASWG;
  constexpr int SRG = L::SRG, SB = L::SB;
  constexpr bool GNEW = !HB && NT >= 8, BMT = GNEW && MW >= 16, CEPI = BMT;
#define BMI(e, w) (BMT ? (w) * 32 + (e) : (e) * MW + (w))
  constexpr int GHC = MW > 8 ? 4 : 1, GUPT = (8 * NT + GHC - 1) / GHC;
  constexpr int STAGE = L::STAGE;
  constexpr int IT2XOFF = WB + SB + 8 * XP;
  constexpr int MOE_WPA_BUFB = L::WPA_BUFB;
  static_assert(!IT2K || (IT2XOFF % 128 == 0 && IT2XOFF + 8 * (RB + 4) * 4 <= STAGE));
  // Model shape. GEMM1: N = 2 * kInter (gate/up rows interleaved per 16), K = kHidden;
  // GEMM2: N = kHidden, K = kInter.
  constexpr int kNkc1 = kHidden / 256, kNkc2 = kInter / 256;
  constexpr int kNrg1 = 2 * kInter / 16, kNrg2 = kHidden / 16;
  constexpr int NRB1 = 2 * kInter / RB, NRB2 = kHidden / RB;
  static_assert(kNkc1 % KW == 0 && kNkc2 % KW == 0, "K chunks per stage");
  // Stages per item: phase 1 (K = hidden), phase 2 full K (K = intermediate), phase 2 split.
  using SPL = DecodeSplit<kHidden, kInter, RW, NS>;
  constexpr int ST1X = SPL::ST1X, ST1H = SPL::ST1H, KSX = SPL::KSX, STX = SPL::STX;
  constexpr int MAXP = L::MAXP, MAXT = L::MAXT;
  static_assert(MAXT <= kMaxTiles, "tile counter overflow");
  constexpr bool MOE_L2_CLDV = (LEANC && MW >= 4) || HB;
  constexpr int NSIG = 4;
  constexpr bool kFullMask = kNkc1 < 32 && kNkc2 < 32;
  extern __shared__ __align__(128) u8 smem[];
  static_assert(NS <= 4);
  u64* bars = AS(u64, smem);
  u64* epi = AS(u64, smem + 64);
  u64* sfree = AS(u64, smem + 96);
  StageInfo* info = AS(StageInfo, smem + 128);
  int2* sinfo = AS(int2, smem + 320);
  int* scal = AS(int, smem + 352);
  int* ftile = AS(int, smem + 416);
  int* off = AS(int, smem + 544);
  const u32 rdy = smem_u32(smem + 672);
  const u32 rtd = smem_u32(smem + 688);
  const u32 plr = smem_u32(smem + 752);
  const u32 eis = smem_u32(smem + 696);
  float* gsm = AS(float, smem + 768);
  u64* aq_full = AS(u64, smem + 704);
  int* aq_slot = AS(int, smem + 736);
  u32* bm = AS(u32, smem + 1024);
  int* wpre = AS(int, bm + 32 * MW);
  int4* tiles = AS(int4, wpre + 32 * MW);
  constexpr bool PAIRS = L::PAIRS;
  u16* pair_tok = AS(u16, tiles + MAXT);
  signed char* lid = AS(signed char, pair_tok + (PAIRS ? MAXP : 0));
  constexpr int STB = L::STB;
  u8* stages = smem + STB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = P.M;
  const int nexp = P.num_experts;
  constexpr bool FT_GSDV = IT2K, MOE_FSTV = IT2K;
  constexpr int PAR_OFF = kParOff, PAR_WORD = kParWord;
  int fi0 = 0, gc0 = 0;
  pdl_wait();
  if (tid == 256) fi0 = atomicAdd(P.ctr + PAR_WORD + 32, 1);
  int par_r = 0;
  if (tid == 0) par_r = __ldcg(P.ctr + PAR_WORD);
  int* CB = P.ctr;
  int4 hv0 = {}, hv1 = {};
  if (warp * 32 + lane < M) LD_IDS(P.ids, warp * 32 + lane, hv0, hv1)
  constexpr int NIT = BMT ? (MW * 32 + 383) / 384 : 1;
  int4 hvx[NIT][2];
  if constexpr (BMT) {
    for (int k = 1; k < NIT; ++k) {
      const int t = warp * 32 + k * 384 + lane;
      hvx[k][0] = hvx[k][1] = {};
      if (t < M) LD_IDS(P.ids, t, hvx[k][0], hvx[k][1])
    }
  }
  // per-expert global scales (gate, down); scaled at use
  const float hgs = tid < 64 ? (tid < 32 ? (tid < nexp ? P.gs13[tid] : 0.f)
                                         : (tid - 32 < nexp ? P.gs2[tid - 32] : 0.f))
                             : 0.f;
  for (int i = tid; i < 32 * MW; i += blockDim.x) bm[i] = 0u;
  if (tid == 352) {
    for (int s = 0; s < NS; ++s) mbar_init(smem_u32(&bars[s]), 1);
    for (int s = 0; s < NSIG; ++s) {
      mbar_init(smem_u32(&epi[s]), 8);
      mbar_init(smem_u32(&sfree[s]), 1);
    }
    for (int s = 0; s < NS; ++s) mbar_init(smem_u32(&aq_full[s]), 8);
    if constexpr (MOE_WPA_BUFB > 0) mbar_init(smem_u32(stages + NS * STAGE + MOE_WPA_BUFB - 16), 1);
    mbar_init(rdy, 1);
    mbar_init(rtd, 1);
    mbar_init(eis, 1);
    mbar_init(plr, 1);
    mbar_init_fence();
    fence_async();
  }
  __syncthreads();
  // Routing bitmap: bm[expert, 32-token word] (transposed per warp), lid[t * 8 + j] = local id.
  for (int k = 0; k < NIT; ++k) {
    const int t0 = warp * 32 + k * 384;
    if (t0 >= MW * 32) break;
    const int t = t0 + lane;
    u32 em = 0;
    if (t < M) {
      const int4 v0 = k == 0 ? hv0 : hvx[k][0], v1 = k == 0 ? hv1 : hvx[k][1];
      const int ev[8] = EV8(v0, v1);
      for (int j = 0; j < 8; ++j) {
        const int e = ev[j];
        const bool loc = (unsigned)e < (unsigned)nexp;
        lid[t * 8 + j] = loc ? (signed char)e : (signed char)-1;
        em |= loc ? 1u << e : 0u;
      }
    }
    bm[BMI(lane, t0 >> 5)] = tr32(em);
  }
  if (tid < 64) {
    float g = hgs * (FT_GSDV ? 1.f : 256.f);
    gsm[tid] = FT_GSDV ? g * 256.f : g;
  }
  if (tid == 0) scal[14] = par_r;
  __syncthreads();
  {
    const int par = scal[14];
    CB = P.ctr + par * PAR_OFF;
    if (NT >= 8 ? tid == 288 : tid == 0) gc0 = atomicAdd(CB + 7, 1);
    int* OB = P.ctr + (1 - par) * PAR_OFF;
    for (int i = blockIdx.x * (int)blockDim.x + tid; i < kParWords;
         i += (int)gridDim.x * (int)blockDim.x)
      OB[i] = 0;
    if (warp == 9 && lane == 0) {
      if (atomicAdd(CB + 6, 1) == (int)gridDim.x - 1) atomicExch(P.ctr + PAR_WORD, 1 - par);
    }
  }
  if constexpr (MOE_FSTV) {
    static_assert(sizeof(StageInfo) == 48 && ST1X > NS, "wp32_fst layout");
    if (warp == 8) {
      bar_arrive(2, 384);
      wp32_fst<KW, kHidden, kInter>(bm, P.w13, P.x, M, AS(u8, info), smem_u32(stages),
                                    smem_u32(bars), shfl(fi0, 0), NRB1, NS, STAGE, WB, SB, SRG, RGS,
                                    XP, HASWG ? 8 * (WGMIN - 1) : 32);
    }
  }
  // Warp 0: per-expert route counts -> tile table (expert, first route, rows, gather unit).
  if (warp == 0) {
    int c = 0;
    u32 bw[MW];
    for (int w = 0; w < MW; ++w) bw[w] = bm[BMI(lane, w)];
    for (int w = 0; w < MW; ++w) {
      wpre[BMI(lane, w)] = c;
      c += __popc(bw[w]);
    }
    const int nt = (c + T - 1) / T;
    int rank = key_rank(c, 10);
    if constexpr (NT >= 8 && MW <= 8) rank = mix_rank(rank, c);
    const int nu = GNEW && c > 0 ? (nt - 1) * GUPT + (c - (nt - 1) * T + GHC - 1) / GHC : 0;
    int4* rk = AS(int4, tiles);
    rk[rank] = make_int4(c, nt, nu, 0);
    __syncwarp();
    const int4 mine = rk[lane];
    int sc = mine.x, st = mine.y, su = mine.z;
    for (int d = 1; d < 32; d <<= 1) {
      const int a = shfl_up(sc, d), b = shfl_up(st, d), u = GNEW ? shfl_up(su, d) : 0;
      if (lane >= d) {
        sc += a;
        st += b;
        su += u;
      }
    }
    const int tu = shfl(su, 31), tot_t = shfl(st, 31), tot_p = shfl(sc, 31);
    __syncwarp();
    rk[lane] = make_int4(sc - mine.x, st - mine.y, su - mine.z, 0);
    __syncwarp();
    const int4 ex = rk[rank];
    const int o = ex.x, to = ex.y, uo = ex.z;
    __syncwarp();
    ftile[lane] = to;
    off[lane] = o;
    for (int k = 0; k < nt; ++k)
      tiles[to + k] = make_int4(lane, o + k * T, min(T, c - k * T), GNEW ? uo + k * GUPT : 0);
    if (GNEW && lane == 0) scal[6] = tu;
    __syncwarp();
    if (lane == 0) {
      // the last nspl tiles split the down GEMM along K
      const int nspl = (KSX > 1 && MW < 32) ? min(tot_t, (int)ceilf((HB        ? 0.75f
                                                                     : NT >= 8 ? 0.6f
                                                                               : 1.0f) *
                                                                    gridDim.x / NRB2))
                                            : 0;
      const int tsplit = tot_t - nspl;
      scal[0] = tot_t;
      scal[1] = NRB1 * tot_t;
      scal[2] = NRB1 * tot_t + NRB2 * tsplit + NRB2 * KSX * nspl;
      scal[3] = tot_p;
      scal[4] = tsplit;
      scal[5] = nspl > 0 ? tiles[tsplit].y : tot_p;
    }
  }
  if constexpr (MOE_FSTV) {
    if (warp != 8) bar_sync(2, 384);
  } else
    __syncthreads();
  if (tid == 0) mbar_arrive(rtd);
  constexpr bool MOE_XDIRV = NT == 4, MOE_XRELV = NT == 4, MOE_RT_WRELV = NT == 4 && MW > 1;
  // Warps 0-7: route position of every (token, slot); with < 65 tokens they also
  // gather the activation rows into the tiles.
  if (warp < 8) {
#pragma unroll 1
    for (int i0 = tid; PAIRS && i0 < M * 8; i0 += 1024) {
      int ev[4], pv[4];
      for (int u = 0; u < 4; ++u) ev[u] = i0 + 256 * u < M * 8 ? lid[i0 + 256 * u] : -1;
      for (int u = 0; u < 4; ++u) {
        const int t = (i0 + 256 * u) >> 3;
        pv[u] = 0;
        if (ev[u] >= 0) {
          const u32 word = bm[ev[u] * MW + (t >> 5)];
          pv[u] = off[ev[u]] + wpre[ev[u] * MW + (t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
        }
      }
      for (int u = 0; u < 4; ++u)
        if (ev[u] >= 0) pair_tok[pv[u]] = (i0 + 256 * u) >> 3;
    }
    consumer_sync();
    if (tid == 0) mbar_arrive(plr);
    if constexpr (!HB && !GNEW) {
      constexpr int kGH = (kHidden / 8 + 255) / 256;  // 16-byte chunks per thread and row
      const int np_all = scal[3], gch = (np_all + (int)gridDim.x - 1) / (int)gridDim.x;
      const int gnc = gch > 0 ? (np_all + gch - 1) / gch : 0;
      int tile = 0;
      if (tid == 0) scal[9] = gc0;
      consumer_sync();
      int gc = scal[9];
      for (;;) {
        consumer_sync();
        if (gc >= gnc) break;
        if (tid == 0) scal[9] = atomicAdd(CB + 7, 1);
        const int pb = gc * gch, pe = min(np_all, pb + gch);
        for (int pg = pb; pg < pe; pg += 4) {
          uint4 val[4][kGH];
          int tl[4], ri[4];
          bool wg[4];
          for (int r = 0; r < 4; ++r) {
            const int p = pg + r;
            tl[r] = 0;
            ri[r] = 0;
            wg[r] = false;
            if (p < pe) {
              while (tile + 1 < scal[0] && tiles[tile + 1].y <= p) ++tile;
              tl[r] = tile;
              ri[r] = p - tiles[tile].y;
              wg[r] = (tiles[tile].z + 7) / 8 >= WGMIN;
              const u16* srow = P.x + (size_t)pair_tok[p] * kHidden;
              for (int h = 0; h < kGH; ++h) {
                const int ch = tid + 256 * h;
                if (kHidden % 2048 != 0 && ch >= kHidden / 8) continue;
                if (wg[r]) {
                  const int c64 = ch >> 3, pc = ch & 7, lc = pc ^ (ri[r] & 7);
                  u32 v[4];
                  for (int e2 = 0; e2 < 4; ++e2) {
                    const int qp = lc * 8 + e2 * 2, jj = qp >> 4, pp = qp & 15;
                    const int k = c64 * 64 + 16 * ((pp & 7) >> 1) + 4 * jj + ((pp >> 3) << 1);
                    v[e2] = __ldg(AS(const u32, srow + k));
                  }
                  val[r][h] = make_uint4(v[0], v[1], v[2], v[3]);
                } else
                  val[r][h] = __ldg(AS(const uint4, srow) + ch);
              }
            }
          }
          for (int r = 0; r < 4; ++r)
            if (pg + r < pe)
              for (int h = 0; h < kGH; ++h) {
                const int ch = tid + 256 * h;
                if (kHidden % 2048 != 0 && ch >= kHidden / 8) continue;
                if (wg[r]) {
                  const int c64 = ch >> 3, pc = ch & 7;
                  u16* dtile = P.xs + (size_t)tl[r] * ST1X * T * XP / 2;
                  *AS(uint4, dtile + ((size_t)c64 * T + ri[r]) * 64 + pc * 8) = val[r][h];
                } else {
                  u8* dst = AS(u8, P.xs) + ((size_t)tl[r] * ST1X * T + ri[r]) * XP;
                  const int st = ch / (KST / 8), w = ch - st * (KST / 8);
                  *AS(uint4, dst + (size_t)st * T * XP + w * 16) = val[r][h];
                }
              }
        }
        consumer_sync();
        if (tid == 0) red_release(CB + 3, 1);
        gc = scal[9];
      }
    }
    if (tid == 0) mbar_arrive(rdy);
  }
  if (MOE_FSTV && warp == 8) {
    if (lane == 0) mbar_sleep(rtd, 0);
    __syncwarp();
  }
  int ntiles = scal[0], total = scal[2], tsplit = scal[4], psplit = scal[5];
  int* head = P.ctr + PAR_WORD + 32;
  int* cnt1 = CB + kCnt1Off;
  int* cnt2 = CB + kCnt2Off;
  if (warp >= 8) {
    nreg_dec<72>();
    if constexpr (GNEW) {
      // Warps 9-11 (>= 65 tokens): gather activation rows, GHC rows per claimed unit.
      if (warp >= 9) {
        const int gt = tid - 288;
        if (lane == 0) mbar_poll(rtd, 0);
        __syncwarp();
        const int NU = scal[6], ntl = scal[0];
        int gtl = 0;
        const int g0 = gt >> 3, pcp = gt & 7;
        if (gt == 0) scal[10] = gc0;
        bar_sync(3, 96);
        int u = scal[10];
        while (u < NU) {
          int nxt = 0;
          if (gt == 0) nxt = atomicAdd(CB + 7, 1);
          while (gtl + 1 < ntl && tiles[gtl + 1].w <= u) ++gtl;
          const int tile = gtl;
          const int4 tv = tiles[tile];
          const int r0 = (u - tv.w) * GHC, nr = min(GHC, tv.z - r0);
          int tok = 0;
          if (lane < nr) {
            const int e = tv.x, pr = tv.y - off[e] + r0 + lane;
            int w0 = 0;
            for (int s = MW / 2; s > 0; s >>= 1)
              if (wpre[BMI(e, w0 + s)] <= pr) w0 += s;
            u32 word = bm[BMI(e, w0)];
            int nb = pr - wpre[BMI(e, w0)], pos = 0;
            for (int s = 16; s > 0; s >>= 1) {
              const int cl = __popc(word & ((1u << s) - 1u));
              if (nb >= cl) {
                nb -= cl;
                word >>= s;
                pos += s;
              }
            }
            tok = w0 * 32 + pos;
          }
          const bool wgt = (tv.z + 7) / 8 >= WGMIN;
          constexpr int kC64 = kHidden / 64, kV16 = kHidden / 8;
          constexpr int kGI = (kC64 + 11) / 12, kVI = (kV16 + 95) / 96;
          for (int r = 0; r < nr; ++r) {
            const int ri = r0 + r, tq = shfl(tok, r);
            const u16* rp = P.x + (size_t)tq * kHidden;
            if (wgt) {
              const u32* sw = AS(const u32, rp) + g0 * 32 + (pcp ^ (ri & 7));
              uint4* dst =
                  AS(uint4, P.xs + (size_t)tile * ST1X * T * XP / 2 + ((size_t)g0 * T + ri) * 64) +
                  pcp;
#pragma unroll
              for (int i0 = 0; i0 < kGI; i0 += 6) {
                uint4 v[6];
                for (int i = 0; i < 6; ++i)
                  if (i0 + i < kGI && g0 + 12 * (i0 + i) < kC64) {
                    const u32* q = sw + (i0 + i) * 12 * 32;
                    v[i] = make_uint4(__ldg(q), __ldg(q + 8), __ldg(q + 16), __ldg(q + 24));
                  }
                for (int i = 0; i < 6; ++i)
                  if (i0 + i < kGI && g0 + 12 * (i0 + i) < kC64)
                    dst[(size_t)(i0 + i) * 96 * T] = v[i];
              }
            } else {
              u8* dst = AS(u8, P.xs) + ((size_t)tile * ST1X * T + ri) * XP +
                        (size_t)(gt >> 5) * T * XP + (gt & 31) * 16;
#pragma unroll
              for (int i0 = 0; i0 < kVI; i0 += 6) {
                uint4 v[6];
                for (int i = 0; i < 6; ++i)
                  if (i0 + i < kVI && gt + 96 * (i0 + i) < kV16)
                    v[i] = __ldg(AS(const uint4, rp) + gt + 96 * (i0 + i));
                for (int i = 0; i < 6; ++i)
                  if (i0 + i < kVI && gt + 96 * (i0 + i) < kV16)
                    *AS(uint4, dst + (size_t)(i0 + i) * 3 * T * XP) = v[i];
              }
            }
          }
          bar_sync(3, 96);
          if (gt == 0) {
            red_release(CB + kGtcOff + tile, 1);
            scal[10] = nxt;
          }
          bar_sync(3, 96);
          u = scal[10];
        }
      }
    }
    // Warps 10-11: asynchronous epilogue of finished stages.
    if (warp >= 10) {
      const int et = tid - 320;
      int q_sig = 0;
      u32 tmc_ph = 0u;
      for (int qe = 0;; ++qe) {
        mbar_sleep(smem_u32(&aq_full[qe % NS]), (qe / NS) & 1);
        const int aqr = aq_slot[qe % NS];
        const int sl = CEPI ? (aqr & 0xFF) - 1 : aqr;
        const int qsg = CEPI ? (aqr >> 8) : q_sig;
        const int j = qsg % NSIG;
        if (sl >= 0) {
          const StageInfo f = info[sl];
          constexpr int SP = RB + 4;
          // phase 1: SwiGLU of the gate / up accumulators -> BF16 intermediate tile
          auto wpa_g1 = [&](const StageInfo& f, const float* src, int tlo, int thi) {
            const int nn = f.n, ntl = (nn + 7) >> 3;
            const float gsc = gsm[f.e] * (IT2K ? 64.f : 1.f);
            const float gsu = __ldg(P.gs13_up + f.e) * (256.f * (IT2K ? 64.f : 1.f));
            const float gke = -1.4426950408889634f * gsc, gs2 = gsc * gsu;
            auto ld2 = [&](int tok, int row) { return swiglu_bf2(src + tok * SP + row, gke, gs2); };
            if (HASWG && ntl >= WGMIN) {
              u16* dt = P.inter + (size_t)f.tile * ST1H * T * XP / 2;
              const int len = min(thi, nn) - tlo, pc = et & 7;
              for (int o = et; o < len * (RB / 16); o += 64) {
                const int rest = o >> 3, c64l = rest >= len ? 1 : 0,
                          tok = tlo + rest - (c64l ? len : 0);
                const int lc = pc ^ (tok & 7);
                u32 h[4];
                for (int e2 = 0; e2 < 4; ++e2) {
                  const int il = 64 * c64l + 16 * e2 + 4 * (lc >> 1) + 2 * (lc & 1);
                  h[e2] = ld2(tok, 16 * (il >> 3) + (il & 7));
                }
                *AS(uint4, dt + ((size_t)((RB / 128) * f.rb + c64l) * T + tok) * 64 + pc * 8) =
                    make_uint4(h[0], h[1], h[2], h[3]);
              }
            } else {
              const int te = min(thi, nn);
              for (int o = et + tlo * (RB / 16); o < te * (RB / 16); o += 64) {
                const int tok = o / (RB / 16), j8 = o - tok * (RB / 16);
                u32 h[4];
                for (int q = 0; q < 4; ++q) h[q] = ld2(tok, 16 * j8 + 2 * q);
                const int ig = (RB / 2) * f.rb + 8 * j8;
                *AS(uint4, AS(u8, P.inter) + (((size_t)f.tile * ST1H + ig / KST) * T + tok) * XP +
                               (ig % KST) * 2) = make_uint4(h[0], h[1], h[2], h[3]);
              }
            }
          };
          // phase 2: down rows times the down global scale -> BF16 (full K) or FP32 partials
          auto wpa_g2 = [&](const StageInfo& f, const float* src, int tlo, int thi) {
            const float gsc = gsm[32 + f.e] * (IT2K ? 64.f : 1.f);
            const bool full_k = f.tile < tsplit;
            float* yb =
                P.ypart + (size_t)f.ks * P.pmax * kHidden + (size_t)f.p0 * kHidden + f.rb * RB;
            const int te = min(thi, f.n);
            if (full_k) {
              for (int o = et + tlo * (RB / 4); o < te * (RB / 4); o += 64) {
                const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
                float4 v = *AS(const float4, src + tok * SP + 4 * r4);
                v.x *= gsc;
                v.y *= gsc;
                v.z *= gsc;
                v.w *= gsc;
                st_bf4(P.yh + (size_t)(f.p0 + tok) * kHidden + f.rb * RB + 4 * r4, v);
              }
            } else
              for (int o = et + tlo * (RB / 4); o < te * (RB / 4); o += 64) {
                const int tok = o / (RB / 4), r4 = o - tok * (RB / 4);
                float4 v = *AS(const float4, src + tok * SP + 4 * r4);
                v.x *= gsc;
                v.y *= gsc;
                v.z *= gsc;
                v.w *= gsc;
                *AS(float4, yb + (size_t)tok * kHidden + 4 * r4) = v;
              }
          };
#define MOE_WPA(...)          \
  {                           \
    if (f.phase == 1)         \
      wpa_g1(f, __VA_ARGS__); \
    else                      \
      wpa_g2(f, __VA_ARGS__); \
  }
          if constexpr (MOE_WPA_BUFB > 0) {
            // copy the stage out first so that the producer can reuse the slot early
            u8* const wbuf = stages + NS * STAGE;
            const float* const stg =
                AS(const float, stages + sl * STAGE + (IT2K && f.n <= 8 ? IT2XOFF : 0));
            int nn = f.n;
            const int nb = min(nn, (MOE_WPA_BUFB - 16) / (SP * 4));
            const u32 bytes = nb * (SP * 4), cb = smem_u32(wbuf + MOE_WPA_BUFB - 16);
            bar_sync(15, 64);
            float* gscr = P.scr + (size_t)blockIdx.x * (64 * 1040 / 4);
            // no same-CTA smem->smem bulk copy
            for (int v = et; v < (int)(bytes / 16); v += 64)
              AS(uint4, wbuf)[v] = AS(const uint4, stg)[v];
            bar_sync(15, 64);
            if (et == 0) {
              if (MW >= 4 && nn > nb) {
                for (int t0 = nb; t0 < nn; t0 += nb) {
                  const u32 cbytes = min(nb, nn - t0) * (SP * 4);
                  bulk_s2g(gscr + (size_t)(t0 - nb) * SP, smem_u32(stg + t0 * SP), cbytes);
                }
                bulk_commit_wait_read();
              }
            }
            if constexpr (MW >= 4) {
              if (nn > nb) bar_sync(15, 64);
            } else if (nn > nb)
              MOE_WPA(stg, nb, 1 << 20)
            __syncwarp();
            bar_arrive(4 + sl, 96);
            const float* cpy = AS(const float, wbuf);
            MOE_WPA(cpy, 0, nb)
            if constexpr (MW >= 4)
              if (nn > nb) {
                if (et == 0) bulk_wait();
                for (int t0 = nb; t0 < nn; t0 += nb) {
                  const u32 cbytes = min(nb, nn - t0) * (SP * 4);
                  bar_sync(15, 64);
                  if (et == 0) {
                    mbar_expect(cb, cbytes);
                    bulk_g2s(smem_u32(wbuf), gscr + (size_t)(t0 - nb) * SP, cbytes, cb);
                  }
                  mbar_sleep(cb, tmc_ph);
                  tmc_ph ^= 1u;
                  const float* ck = cpy - (size_t)t0 * SP;
                  MOE_WPA(ck, t0, t0 + nb)
                }
              }
          } else {
            MOE_WPA(AS(const float, stages + sl * STAGE), 0, 1 << 20)
            __syncwarp();
            bar_arrive(4 + sl, 96);
          }
          if (et == 0) {
            mbar_sleep(smem_u32(&sfree[j]), ((qsg / NSIG) & 1) ^ 1);
            sinfo[j] = make_int2(f.phase, f.phase == 1 ? f.tile * kNkc2 + f.rb * RB / 512 : f.tile);
          }
        } else if (et == 0) {
          mbar_sleep(smem_u32(&sfree[j]), ((qsg / NSIG) & 1) ^ 1);
          sinfo[j] = make_int2(-1, 0);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive_n(smem_u32(&epi[j]), 4);
        ++q_sig;
        if (sl < 0) break;
      }
      return;
    }
    // Warp 8: producer.
    if (warp == 8) {
      const u64 pol_w = policy_evict_first(), pol_x = policy_evict_last();
      const u32 stage0 = smem_u32(stages), bars0 = smem_u32(bars);
      int it = 0;
      bool gathered = false;
      int gtile = -1, pq = 0;
      fi0 = shfl(fi0, 0);
      int item = fi0;
      const int gch8 = (scal[3] + (int)gridDim.x - 1) / (int)gridDim.x;
      const int gnch = gch8 > 0 ? (scal[3] + gch8 - 1) / gch8 : 0;
      int nb_le = -1000;
      for (;;) {
        if (item >= total) {
          const int s = it % NS;
          if (it >= NS) bar_sync(4 + s, it - NS == nb_le ? 96 : 288);
          if (lane == 0) {
            info[s].item = -1;
            mbar_arrive(smem_u32(&bars[s]));
            if (atomicAdd(head + 1, 1) == (int)gridDim.x - 1) {
              atomicExch(head, 0);
              atomicExch(head + 1, 0);
            }
          }
          break;
        }
        int next;
#define MOE_L2_CLAIM()                                                                       \
  asm volatile(                                                                              \
      "{ .reg .pred p; setp.eq.u32 p, %1, 0; @p atom.global.inc.u32 %0, [%2], 0x7fffffff; }" \
      : "=r"(next)                                                                           \
      : "r"(lane), "l"(head)                                                                 \
      : "memory")
        if constexpr (!MOE_L2_CLDV)
          asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, 0; @p atom.global.add.u32 %0, [%2], 1; }"
                       : "=r"(next)
                       : "r"(lane), "l"(head)
                       : "memory");
        int phase, tile, rb, ks, kss;
        decode_item2(item, ntiles, tsplit, NRB1, NRB2, KSX, phase, tile, rb, ks, kss);
        if constexpr (CEPI) {
          if (lane == 0) mbar_sleep(smem_u32(&sfree[pq % NSIG]), ((pq / NSIG) & 1) ^ 1);
          __syncwarp();
          ++pq;
        }
        const int nst = phase == 1 ? ST1X : kss == KSX ? STX : ST1H;
        int ccl = MOE_L2_CLDV && nst >= 8 ? nst - 4 : 0;
        if (MOE_L2_CLDV && ccl <= 0) {
          MOE_L2_CLAIM();
          ccl = 1 << 20;
        }
        const int4 tl = tiles[tile];
        const int e = tl.x, p0 = tl.y, n = tl.z;
        const bool wgl = HASWG && (n + 7) / 8 >= WGMIN;
        const bool epi_item = true;
        const int nrg_tot = phase == 1 ? kNrg1 : kNrg2;
        const int nkc = phase == 1 ? kNkc1 : kNkc2;
        const int kc_item = phase == 1 ? 0 : ks * (kNkc2 / KSX);
        const int xst = phase == 1 ? ST1X : ST1H;  // tile stride (stages) of the activation buffer
        const u16* xrow0 = nullptr;
        u32 rmask = 0;
        const int pkw = lane >> 1;
        const bool pscl = lane & 1;
        const void* wcb = phase == 1 ? (const void*)P.w13 : (const void*)P.w2;
        const u8* csrc = pscl ? weight_scales(wcb, e, nkc, nrg_tot, kc_item + pkw, rb * RGS)
                              : weight_block(wcb, e, nkc, nrg_tot, kc_item + pkw, rb * RGS);
        const size_t cstep = (size_t)KW * (nrg_tot >> 4) * kWeightBlockBytes;
        u32 cdst = pscl ? WB + pkw * RGS * SRG : pkw * RGS * 2048;
        u32 cbytes = pscl ? RGS * SRG : RGS * 2048;
        u32 lrdy = 0;
        const bool xdir = MOE_XDIRV && (HB || item == fi0) && phase == 1 && !wgl;
        if (xdir) {
          if (lane == 0) mbar_sleep(plr, 0);
          __syncwarp();
          if (lane < n) xrow0 = P.x + (size_t)pair_tok[p0 + lane] * kHidden;
        }
        if constexpr (GNEW) {
          bool rd = gtile == tile;
          if (phase == 1 && !rd) {
            int ok = 0;
            if (lane == 0) ok = ld_acquire(CB + kGtcOff + tile) >= (n + GHC - 1) / GHC;
            ok = shfl(ok, 0);
            if (ok) {
              __syncwarp();
              fence_async_global();
              gtile = tile;
              rd = true;
            }
          }
          lrdy = phase == 1 && rd ? ~0u : 0u;
        } else if constexpr (LEANC)
          lrdy = phase == 1 && gathered && !xdir ? ~0u : 0u;
        const int nrx = nrg_tot >> 4;
        const u8* wsrc = AS(const u8, wcb);
        auto wblock = [&](int kc) {
          return wsrc + ((size_t)(e * nkc + kc) * nrx + rb) * (WB + SB);
        };
        int st_first = 0;
        if (MOE_FSTV && item == fi0 && phase == 1) {
          if (xdir) {
            st_first = NS;
            it += NS;
            csrc += NS * cstep;
          }
        }
        for (int st = st_first; st < nst; ++st, ++it) {
          if (MOE_L2_CLDV && st >= ccl) {
            MOE_L2_CLAIM();
            ccl = 1 << 20;
          }
          static_assert(ST1X >= NS && ST1H >= NS && STX >= NS);
          u32 nb_was = it - NS == nb_le;
          if (st == nst - 1 && epi_item) nb_le = it;
          const int s = it % NS;
          u8* sbase = stages + s * STAGE;
          const u32 full = bars0 + s * 8;
          const int kc0 = kc_item + st * KW;
          if constexpr (LEANC) {
            // every remaining stage's inputs are ready: issue them back to back
            const int nrem = nst - st;
            const u32 lrest3 = (kFullMask || nrem < 32 ? ((1u << nrem) - 1u) : ~0u) << kc0;
            if ((lrdy & lrest3) == lrest3) {
              const u8* wq = wblock(kc0);
              const u32 wqs = KW * nrx * (WB + SB);
              const u8* xbq =
                  AS(const u8, phase == 2 ? P.inter : P.xs) + (size_t)tile * xst * T * XP;
              const u8* xq =
                  wgl ? xbq + ((size_t)(kc0 * 4) * T) * 128 : xbq + (size_t)(kc0 / KW) * T * XP;
              const u32 xqs = wgl ? KW * 4 * T * 128 : T * XP;
              const u32 xqb = wgl ? ((n + 7) & ~7) * 128 : n * XP;
              const u32 ltx3 = WB + SB + (wgl ? 4 * xqb : xqb);
              int sl = s, stl = st;
              u32 fl = full;
              auto issue = [&]() {
                const u32 sd = stage0 + sl * STAGE;
                mbar_expect(fl, ltx3);
                bulk_g2s(sd, wq, WB + SB, fl, pol_w);
                if (wgl) {
                  for (int c = 0; c < 4; ++c)
                    bulk_g2s(sd + WB + SB + c * T * 128, xq + c * T * 128, xqb, fl, pol_x);
                } else
                  bulk_g2s(sd + WB + SB, xq, xqb, fl, pol_x);
              };
              for (;;) {
                const int l4e = MOE_L2_CLDV && ccl < nst - 1 ? ccl : nst - 1;
                if (NT >= 8 && stl >= NS && stl < l4e) {
                  for (;;) {
                    bar_sync(4 + sl, 288);
                    if (elect_one()) issue();
                    __syncwarp();
                    wq += wqs;
                    xq += xqs;
                    ++it;
                    ++stl;
                    if (++sl == NS) sl = 0;
                    fl = bars0 + sl * 8;
                    if (stl == l4e) break;
                  }
                  nb_was = 0;
                  nb_le = epi_item ? it : -1000;
                }
                if (MOE_L2_CLDV && stl >= ccl) {
                  MOE_L2_CLAIM();
                  ccl = 1 << 20;
                }
                if (it >= NS) bar_sync(4 + sl, nb_was ? 96 : 288);
                if (elect_one()) {
                  if (stl == 0 || stl == nst - 1)
                    PUT_INFO(&info[sl], make_int4(item, phase, tile, rb),
                             make_int4(ks, stl, stl == nst - 1, n), make_int4(p0, e, 0, 0))
                  issue();
                }
                __syncwarp();
                wq += wqs;
                xq += xqs;
                ++it;
                if (++stl == nst) break;
                if (++sl == NS) sl = 0;
                fl = bars0 + sl * 8;
                nb_was = it - NS == nb_le;
                if (stl == nst - 1 && epi_item) nb_le = it;
              }
              st = nst - 1;
              --it;
              continue;
            }
          }
          if (it >= NS) bar_sync(4 + s, nb_was ? 96 : 288);
          if (lane == 0) {
            if (MOE_RT_WRELV && it == 0) mbar_sleep(eis, 0);
            PUT_INFO(&info[s], make_int4(item, phase, tile, rb),
                     make_int4(ks, st, st == nst - 1, n), make_int4(p0, e, 0, 0))
            mbar_expect(full, WB + SB +
                                  (wgl ? 4 * ((n + 7) & ~7) * 128
                                       : (phase == 2 || !xdir ? n * XP : n * KST * 2)));
          }
          __syncwarp();
          if constexpr (WSC) {
            static_assert(WB == 32768 && (SB == 4096 || SB == 3072) && RGS == 16);
            if (lane == 0) bulk_g2s(stage0 + s * STAGE, wblock(kc0), WB + SB, full, pol_w);
          } else if (lane < KW * 2)
            bulk_g2s(stage0 + s * STAGE + cdst, csrc, cbytes, full, pol_w);
          csrc += cstep;
          if (phase == 2) {
            // the intermediate columns of this K chunk are written by phase-1 items
            u32 need = ((1u << KW) - 1u) << kc0;
            const bool acq = (rmask & need) != need;
            if (acq) {
              rmask = ballot(lane < kNkc2 && ld_acquire(&cnt1[tile * kNkc2 + lane]) >= 512 / RB);
              if constexpr (LEANC) lrdy = rmask;
            }
            if (lane == 0 && (rmask & need) != need)
              for (int kw = 0; kw < KW; ++kw)
                while (ld_acquire(&cnt1[tile * kNkc2 + kc0 + kw]) < 512 / RB) {
                }
            __syncwarp();
            // fence.proxy.async only orders the issuing thread's accesses, and several lanes
            // issue bulk copies: every lane fences, after the barrier that publishes the acquire.
            if (acq) fence_async_global();
          }
          if (xdir) {
            if (xrow0)
              bulk_g2s(smem_u32(sbase + WB + SB + lane * XP), xrow0 + kc0 * 256, KST * 2, full,
                       pol_x);
          } else {
            if constexpr (GNEW) {
              if (phase == 1 && gtile != tile) {
                if (lane == 0)
                  while (ld_acquire(CB + kGtcOff + tile) < (n + GHC - 1) / GHC) {
                  }
                __syncwarp();
                fence_async_global();
                gtile = tile;
                lrdy = ~0u;
              }
            } else if (phase == 1 && !gathered) {
              if (lane == 0)
                while (ld_acquire(CB + 3) < gnch) {
                }
              __syncwarp();
              fence_async_global();
              gathered = true;
              if constexpr (LEANC) lrdy = ~0u;
            }
            const u8* xreg =
                AS(const u8, phase == 2 ? P.inter : P.xs) + (size_t)tile * xst * T * XP;
            if (wgl) {
              if (lane < 4)
                bulk_g2s(smem_u32(sbase + WB + SB + lane * T * 128),
                         xreg + ((size_t)(kc0 * 4 + lane) * T) * 128, ((n + 7) & ~7) * 128, full,
                         pol_x);
            } else if (lane == 0)
              bulk_g2s(smem_u32(sbase + WB + SB), xreg + (size_t)(kc0 / KW) * T * XP, n * XP, full,
                       pol_x);
          }
        }
        if (MOE_L2_CLDV && ccl < (1 << 20)) MOE_L2_CLAIM();
        item = shfl(next, 0);
      }
      return;
    }
    // Warp 9: publishes finished (tile, k chunk) / tiles to the other CTAs.
    if (warp == 9 && lane == 0) {
      if (MOE_XRELV) {
        mbar_sleep(rdy, 0);
        if (MOE_RT_WRELV) mbar_arrive(eis);
      }
      for (int q = 0;; ++q) {
        const int j = q % NSIG;
        mbar_sleep<200>(smem_u32(&epi[j]), (q / NSIG) & 1);
        const int2 si = sinfo[j];
        mbar_arrive(smem_u32(&sfree[j]));
        if (si.x < 0) break;
        red_release(si.x == 1 ? &cnt1[si.y] : &cnt2[si.y], 1);
      }
    }
    return;
  }
  // Warps 0-7: consumers.
  nreg_inc<216>();
  const int kw = warp % KW, rw = warp / KW;
  int q_epi = 0, q_sig_c = 0;
  const int g = lane >> 2, tig = lane & 3;
#define MOE_WOFF ((kw * RGS + rw * 2) * 2048 + lane * 16)
#define MOE_SOFF (WB + (kw * RGS + rw * 2) * SRG + lane * 12)
#define MOE_XOFF (WB + SB + g * XP + kw * 512 + tig * 32)
#define MOE_STG(a, b, c) \
  stg[(kw * T + (b) * 8 + 2 * tig + ((c) & 1)) * SP + 16 * (2 * rw + (a)) + g + 8 * ((c) >> 1)]
  float acc[2][NT][4], acc2[2][NT][4];
  ZERO_ACC(0)
  int it = 0;
  [[maybe_unused]] const u32 xlo0 = sw128_desc(smem_u32(stages + WB + SB));
#define MOE_DSKV(K) ((K) < 100 && ds_code(RW, (K), MW) != 0)
#define MOE_STAGES(K)                                                                              \
  {                                                                                                \
    DsCarry<((K) >= 1 && (K) <= 4) ? (K) : 1> ds_cr{};                                             \
    for (int st = 0; st < nst; ++st) {                                                             \
      if (st > 0) {                                                                                \
        bar_arrive(4 + s, 288);                                                                    \
        advance<NS>(s, ph);                                                                        \
        mbar_poll(smem_u32(&bars[s]), ph);                                                         \
      }                                                                                            \
      const u8* sbase = stages + s * STAGE;                                                        \
      const u8* wbase = sbase + MOE_WOFF;                                                          \
      const u8* sclb = sbase + MOE_SOFF;                                                           \
      if constexpr ((K) >= 100)                                                                    \
        stage_compute_wg<NT, (K) - 100, (MW >= 16)>(acc, wbase, sclb, xlo0 + s * (STAGE >> 4), T,  \
                                                    -1);                                           \
      else if constexpr (MOE_DSKV(K))                                                              \
        stage_ds<ds_code(RW, (K), MW) == 2, NT, (K)>(acc, acc2, wbase, sclb, sbase + MOE_XOFF, XP, \
                                                     ds_cr, -1);                                   \
      else                                                                                         \
        stage_compute<NT, (K)>(acc, wbase, sclb, sbase + MOE_XOFF, XP);                            \
      __syncwarp();                                                                                \
    }                                                                                              \
    if constexpr (MOE_DSKV(K)) ds_flush<ds_code(RW, (K), MW), NT, (K)>(acc, acc2, ds_cr);          \
  }
  for (;;) {
    int s, ntl;
    u32 ph;
    StageInfo f;
    s = it % NS;
    ph = (it / NS) & 1;
    mbar_poll(smem_u32(&bars[s]), ph);
    f = info[s];
    if (f.item < 0) {
      __syncwarp();
      if (warp == 0 && lane == 0) aq_slot[q_epi % NS] = CEPI ? (q_sig_c << 8) : -1;
      if (lane == 0) mbar_arrive(smem_u32(&aq_full[q_epi % NS]));
      break;
    }
    ntl = (f.n + 7) >> 3;
    const int nst = f.phase == 1 ? ST1X : f.tile >= tsplit ? STX : ST1H;
    if (HASWG && ntl >= WGMIN) {
#define MOE_WGC(N) \
  case N:          \
    if constexpr (NT >= N) MOE_STAGES(100 + N) break;
      switch (ntl) {
        MOE_WGC(3)
        MOE_WGC(4)
        MOE_WGC(5)
        MOE_WGC(6)
        MOE_WGC(7)
        MOE_WGC(8)
        MOE_WGC(9)
        MOE_WGC(10)
        MOE_WGC(11)
        MOE_WGC(12)
        MOE_WGC(13)
        MOE_WGC(14)
        MOE_WGC(15)
        default:
          if constexpr (NT >= 16) MOE_STAGES(116) break;
      }
#undef MOE_WGC
    } else {
      switch (ntl) {
        case 1:
          MOE_STAGES(1) { ZERO_ACC(1) }
          break;
        case 2:
          MOE_STAGES(2) { ZERO_ACC(2) }
          break;
      }
    }
    it += nst;
    if constexpr (!IT2K) {
      FOR_ACC(0) acc[a][b][c] *= 64.f;
    }
    constexpr int SP = RB + 4;
    static_assert(T * SP * 4 <= STAGE);
    const bool it2xs = IT2K && ntl == 1;
    float* stg = AS(float, stages + s * STAGE + (it2xs ? IT2XOFF : 0));
    if (!it2xs) consumer_sync();
    if constexpr (HB) {
      // the two K halves of a 128-row block meet in shared memory
      float* red = AS(float, stages + s * STAGE + (RGS + rw * 2) * 2048) + lane;
      if (kw) {
        FOR_ACC(0) if (b < ntl) red[((a * NT + b) * 4 + c) * 32] = acc[a][b][c];
        bar_arrive(8 + rw, 64);
      } else {
        bar_sync(8 + rw, 64);
        FOR_ACC(0) if (b < ntl) acc[a][b][c] += red[((a * NT + b) * 4 + c) * 32];
      }
    }
    if constexpr (IT2K) {
      if (ntl > 0)
        for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
          for (int c = 0; c < 4; ++c) MOE_STG(a, 0, c) = acc[a][0][c];
      if (!it2xs)
        for (int b = 1; b < NT; ++b)
          if (b < ntl)
            for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
              for (int c = 0; c < 4; ++c) MOE_STG(a, b, c) = acc[a][b][c];
    } else
      for (int a = 0; a < ((HB && kw) ? 0 : 2); ++a)
        for (int b = 0; b < NT; ++b)
          if (b < ntl)
            for (int c = 0; c < 4; ++c) MOE_STG(a, b, c) = acc[a][b][c];
    if constexpr (MOE_WPA_BUFB > 0) fence_async();
    __syncwarp();
    if (warp == 0 && lane == 0) aq_slot[q_epi % NS] = CEPI ? (s + 1) | (q_sig_c << 8) : s;
    if (lane == 0) mbar_arrive(smem_u32(&aq_full[q_epi % NS]));
    ++q_epi;
    if constexpr (CEPI) ++q_sig_c;
    FOR_ACC(0) acc[a][b][c] = 0.f;
  }
  // Combine: every CTA claims tokens; out[t] = sum_j w_j * f(y_j), f = RMSNorm(y * nw) or identity.
  float* rbuf = AS(float, stages + ((NS - 1) * STAGE + (T * (RB + 4) * 4 + 127) / 128 * 128));
  static_assert((T * (RB + 4) * 4 + 127) / 128 * 128 + 256 <= STAGE);
  constexpr bool PW =
      MW <= 1;  // <= 32 tokens: each route pair is combined as soon as its two tiles are done
  constexpr int KH = (kHidden + 1023) / 1024;  // float4 columns per thread
  constexpr bool kHFull = kHidden % 1024 == 0;
  float4 w4s[KH];
  if constexpr (kPostNorm)
    for (int k = 0; k < KH; ++k)
      if (kHFull || (tid + 256 * k) * 4 < kHidden)
        w4s[k] = *AS(const float4, P.nw1 + (tid + 256 * k) * 4);
  auto claim = [&](int* ctr_word) -> int {
    if (tid == 0) scal[8] = atomicAdd(ctr_word, 1);
    consumer_sync();
    const int v = scal[8];
    consumer_sync();
    return v;
  };
  int ci = claim(CB + 4);
  for (int epar = 0; ci < M; epar ^= 1) {
    const int t = ci;
    const int e = lane < 8 ? lid[t * 8 + lane] : -1;
    const u32 vm = ballot(e >= 0);
    const int np = __popc(vm);
    int pv = 0, tv = -1;
    float wv = 0.f;
    if (e >= 0) {
      const u32 word = bm[BMI(e, t >> 5)];
      const int pr = wpre[BMI(e, t >> 5)] + __popc(word & ((1u << (t & 31)) - 1u));
      pv = off[e] + pr;
      wv = P.topk_w[t * 8 + lane];
      tv = ftile[e] + pr / T;
    }
    int src = lane < np ? (int)__fns(vm, 0, lane + 1) : 0;
    if constexpr (PW) {  // tiles finish about in index order: pair routes by tile
      int rk = 0;
      for (int i = 0; i < 8; ++i) {
        const int ti = shfl(tv, i);
        rk += ti >= 0 && ti < tv;
      }
      for (int i = 0; i < 8; ++i) {
        const int ri = shfl(rk, i);
        if (((vm >> i) & 1u) && ri == lane) src = i;
      }
    }
    const int pl = shfl(pv, src);
    const float wq = shfl(wv, src);
    int tlq = shfl(tv, src);
    if (lane >= np) tlq = -1;
    auto wait_tile = [&](bool w) {
      if (w && warp == 0 && tlq >= 0) {
        const int want = NRB2 * (tlq >= tsplit ? KSX : 1);
        while (ld_relaxed(&cnt2[tlq]) < want) {
        }
        (void)ld_acquire(&cnt2[tlq]);
      }
      consumer_sync();
    };
    if constexpr (!PW) wait_tile(true);
    int enxt = M;
    if ((!PW || np == 0) && tid == 0) enxt = atomicAdd(CB + 4, 1);
    float accv[4 * KH] = {};
    int rnd = 0;
    for (int q0 = 0; q0 < np; q0 += 2, ++rnd) {
      if constexpr (PW) {
        wait_tile(lane >> 1 == q0 >> 1);
        if (q0 + 2 >= np && tid == 0) enxt = atomicAdd(CB + 4, 1);
      }
      float v[2][4 * KH];
      float ss[2];
      for (int gq = 0; gq < 2; ++gq) {
        ss[gq] = 0.f;
        const int p = shfl(pl, (q0 + gq) & 31);
        const bool live = q0 + gq < np;
        const int kp = p >= psplit ? KSX : 1;
        for (int k = 0; k < KH; ++k) {
          const int n = (tid + 256 * k) * 4;
          const bool col = kHFull || n < kHidden;
          float4 y = {};
          if (live && col && kp == 1) y = ld_bf4(P.yh + (size_t)p * kHidden + n);
          for (int k2 = 0; k2 < KSX; ++k2)
            if (live && col && kp > 1 && k2 < kp) {
              const float4 u =
                  __ldcg(AS(const float4, P.ypart + ((size_t)k2 * P.pmax + p) * kHidden + n));
              y.x += u.x;
              y.y += u.y;
              y.z += u.z;
              y.w += u.w;
            }
          if constexpr (kPostNorm) {
            const float4 w = col ? w4s[k] : make_float4(0.f, 0.f, 0.f, 0.f);
            v[gq][4 * k] = y.x * w.x;
            v[gq][4 * k + 1] = y.y * w.y;
            v[gq][4 * k + 2] = y.z * w.z;
            v[gq][4 * k + 3] = y.w * w.w;
            ss[gq] += v[gq][4 * k] * v[gq][4 * k] + v[gq][4 * k + 1] * v[gq][4 * k + 1] +
                      v[gq][4 * k + 2] * v[gq][4 * k + 2] + v[gq][4 * k + 3] * v[gq][4 * k + 3];
          } else {
            v[gq][4 * k] = y.x;
            v[gq][4 * k + 1] = y.y;
            v[gq][4 * k + 2] = y.z;
            v[gq][4 * k + 3] = y.w;
          }
        }
      }
      if constexpr (kPostNorm) {
        for (int s2 = 16; s2 > 0; s2 >>= 1)
          for (int gq = 0; gq < 2; ++gq) ss[gq] += shfl_xor(ss[gq], s2);
        if (lane == 0)
          for (int gq = 0; gq < 2; ++gq) rbuf[(rnd & 1) * 32 + gq * 8 + warp] = ss[gq];
        consumer_sync();
      }
      for (int gq = 0; gq < 2; ++gq)
        if (q0 + gq < np) {
          const float gw = shfl(wq, (q0 + gq) & 31);
          float r = gw;
          if constexpr (kPostNorm) {
            float tot = 0.f;
            for (int w = 0; w < 8; ++w) tot += rbuf[(rnd & 1) * 32 + gq * 8 + w];
            r = rsqrtf(tot * (1.f / kHidden) + P.eps) * gw;
          }
          for (int k = 0; k < 4 * KH; ++k) accv[k] += r * v[gq][k];
        }
    }
    for (int k = 0; k < KH; ++k) {
      const int n = (tid + 256 * k) * 4;
      if (kHFull || n < kHidden)
        *AS(uint2, P.out + (size_t)t * kHidden + n) =
            make_uint2(bf2(accv[4 * k], accv[4 * k + 1]), bf2(accv[4 * k + 2], accv[4 * k + 3]));
    }
    if (tid == 0) scal[12 + epar] = enxt;
    consumer_sync();
    ci = scal[12 + epar];
  }
}
#undef BMI
#undef MOE_WOFF
#undef MOE_SOFF
#undef MOE_XOFF
#undef MOE_STG
#undef MOE_DSKV
#undef MOE_STAGES
#undef MOE_WPA
#undef MOE_L2_CLAIM

template <int kHidden, int kInter, bool kPostNorm, int RW, int NT, int NS, int MW>
__global__ void __launch_bounds__(384, 1) __cluster_dims__(1, 1, 1) moe_decode_kernel(Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  moe_body<kHidden, kInter, kPostNorm, RW, NT, NS, MW>(p);
#endif
}

#undef LD_IDS
#undef EV8
#undef M_BF
#undef FOR_ACC
#undef ZERO_ACC
#undef PUT_INFO
#undef DEVI
#undef AS

}  // namespace dec
}  // namespace sm90_w4a16_moe
}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_MOE_SM90_NVFP4_W4A16_DECODE_CUH_
