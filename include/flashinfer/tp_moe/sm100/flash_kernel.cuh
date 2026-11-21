// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// SM100 (B200) fused tensor-parallel MoE decode kernel, DeepSeek-V4-Flash routed + shared experts,
// M <= 64 tokens per call, TP4 / TP8.
//
// One persistent launch (one CTA per SM, 148 SMs) runs the whole MoE block of a rank: fused gate
// (router logits -> sqrtsoftplus scores, bias-corrected top-6, renormalized weights; bitwise
// moe_fused_gate) or top-k ids / weights given, MXFP8 quantization of x, routing tables (token
// panels of <= NMAX entries per expert), routed FC1 / FC2 on tcgen05 kind::mxf8f6f4 block-scaled
// MMAs (MXFP4 trtllm-gen weight images x MXFP8 activations, fp32 accumulators in TMEM), the
// shared-expert slice of the rank (FP8 weights, per-128 block scales), T1 (weighted top-k combine +
// shared output) and the tensor-parallel all-reduce (one-shot LL for M <= TPMOE_ONESHOT_MAXM, else
// reduce-scatter + all-gather) over peer buffers that every rank maps (CUDA IPC).  The output, bf16
// [M][4096], is identical on every rank.
//
// Numerics:
//   x  -> MXFP8 (e4m3, one UE8M0 scale per 32: 2^ceil(log2(amax / 448)), as flashinfer
//   mxfp8_quantize)   FC1 fp32 (TMEM, K 4096) act = up * gate * sigmoid(gate), gate <= 10, |up| <=
//   10 (fp32) -> h = MXFP8 per 32, scale h_scale_exp (common.cuh) FC2 fp32 -> bf16 per (token,
//   slot) T1: r = bf16(sum_k bf16(w_k) * o_k) (k order), y = bf16(shared + factor r) all-reduce:
//   rank-ordered fp32 sum -> bf16
//
// Warps: w0 lane 0 producer (weights by TMA + SFA; work descriptors on the ring) | w1 TMEM + MMA |
// w2..w5, w8..w11 epilogue (TMEM lane quadrant warp & 3) | w6 B operand (FC1: x rows by TMA gather
// + SFB; FC2: h image by bulk copy) | w7 FC2-unit publisher (per-256-row-tile counts that release
// the T1).  Prologue and tail use every thread.
//
// Each build serves one M range (TPMOE_MMIN .. TPMOE_MMAX) at one TP size (TPMOE_NDEV); the
// remaining TPMOE_* macros select per-range schedule variants (the host's build table sets them).
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "flashinfer/tp_moe/sm100/common.cuh"

#ifndef TPMOE_FC2S_HI  // static FC2 ticket order for M <= this (the B warp waits for each panel's
                       // h)
#define TPMOE_FC2S_HI 8
#endif
#ifndef TPMOE_TABW
#define TPMOE_TABW 0  // small fused gate, M <= 1: routing tables by one warp (no CTA-wide phases)
#endif

#ifndef TPMOE_PUBRED
#define TPMOE_PUBRED \
  0  // static FC2 order: the h publish is a red.release (no atomic round trip on the epilogue's
     // publishing lane) instead of
     // atom.acq_rel + ready-list append
#endif
#ifndef TPMOE_FC2CAP
#define TPMOE_FC2CAP \
  0  // > 0: a CTA claims at most ceil(FC2 units / G) + TPMOE_FC2CAP - 1 FC2 tickets (idle CTAs no
     // longer take every ticket
     // while the FC1 CTAs stream)
#endif
#ifndef TPMOE_ACTX
#define TPMOE_ACTX \
  0  // routed FC1 epilogue: SwiGLU of 4 tokens per loop round (independent chains interleaved), the
     // per-token warp amax by one
     // redux.max on the bits of |a| (== the fmaxf butterfly for non-negative floats)
#endif
#ifndef TPMOE_GUNR
#define TPMOE_GUNR 0  // gate: activation + routing-weight loops unrolled
#endif
#ifndef TPMOE_ARV
#define TPMOE_ARV \
  0  // all-reduce variants (bits; same arithmetic): 2 T2 + T3 in one poll loop (a thread's T2
     // sources and T3 messages in flight
// together); 8 explicit in-flight source polls (issue all pending loads, then test); 32 epoch from
// a u64 CTA-exit count (red, no fence / returning atomic at exit)
#endif
#ifndef TPMOE_SHGU_LO2
#define TPMOE_SHGU_LO2 0  // (a second M range for SHGU: TPMOE_SHGU_LO2 <= M <= TPMOE_SHGU_HI2)
#endif
#ifndef TPMOE_SHGU_HI2
#define TPMOE_SHGU_HI2 -1
#endif
#ifndef TPMOE_SHGU_MAXM
#define TPMOE_SHGU_MAXM \
  6  // SHGU (a shared FC1 item's gate and up chains on two CTAs) for M <= this
#endif
#ifndef TPMOE_FC2PAIR_OFFM
#define TPMOE_FC2PAIR_OFFM \
  8  // FC2 row tiles stage by stage for M <= this (else both stages of a row tile per MMA round)
#endif
#ifndef TPMOE_ONESHOT_MAXM
#define TPMOE_ONESHOT_MAXM \
  48  // one-shot LL all-reduce for M <= this, else reduce-scatter + all-gather
#endif
#ifndef TPMOE_MMAX
#define TPMOE_MMAX \
  0  // > 0: this build serves M <= TPMOE_MMAX only (the host checks): device reads of M are clamped
     // to it, so the paths of
     // larger M fold away at compile time (less cold code)
#endif
#ifndef TPMOE_GATE
#define TPMOE_GATE 0  // 1: fused-gate entry (router logits in); 0: top-k ids / weights entry
#endif
#ifndef TPMOE_ACTU
#define TPMOE_ACTU \
  0  // 1 (+ TPMOE_MMAX <= 16): FC1 epilogue SwiGLU / h quantization unrolled over the <= TPMOE_MMAX
     // live tokens (no rotations)
#endif
#define TPMOE_ACTU_ON (TPMOE_ACTU && TPMOE_MMAX && TPMOE_MMAX <= 16)
#ifndef TPMOE_SFBI
#define TPMOE_SFBI \
  0  // 1: routed FC1 x scales (SFB) staged per item (two item buffers [16 stages][32 rows][16 B]
     // after the tables) in
// batches of 4 stages with one proxy fence per batch, instead of a generic store + proxy fence per
// stage on the B warp.  The MMA warp copies stage s of item buffer (item & 1).
#endif
#ifndef TPMOE_MMIN
#define TPMOE_MMIN \
  1  // > 1: this build serves M >= TPMOE_MMIN only (the paths of smaller M fold away likewise)
#endif

namespace flashinfer::tp_moe::sm100::flash {

#ifndef TPMOE_NDEV
#define TPMOE_NDEV 4  // TP size: 4 (INTER 512) or 8 (INTER 256)
#endif
constexpr int NDEV = TPMOE_NDEV;
static_assert(NDEV == 4 || NDEV == 8, "TP4 / TP8");
constexpr int DIM = 4096, INTER = 2048 / NDEV, NEXP = 256, TOPK_MAX = 8, MAXM = 512;
constexpr int MAXE =
    MAXM * 6;              // entries (token, expert) upper bound at top-6 (asserted on the host)
constexpr int NMAX = 32;   // token panel width = MMA N bound
constexpr int MAXP = 512;  // panels upper bound (<= NEXP + MAXE / NMAX)
constexpr int NT1 =
    2 * INTER / 128;  // FC1 tiles (128 w13 rows = 64 up + 64 gate, trtllm-gen order) per panel
constexpr int NCH1 = DIM / 128;                      // 32 FC1 k-chunks
constexpr int NRT2 = DIM / 128;                      // 32 FC2 row tiles
constexpr int NCH2 = INTER / 128;                    // FC2 k-chunks
constexpr int KC = 2;                                // k-chunks per pipeline stage
constexpr int NSTG1 = NCH1 / KC, NSTG2 = NCH2 / KC;  // stages per FC1 item / FC2 row tile
constexpr int NS = 4;                                // ring stages (weight stages in flight per SM)
constexpr int THREADS =
    384;  // 12 warps: producer, MMA, 2 x 4 epilogue, B operand, FC2-unit publisher
// FC2 unit = fc2g consecutive 128-row tiles of one panel (one ticket / ring entry); fc2g chosen per
// launch from the panel count
constexpr int NRT = DIM / 256;   // 16 T1 / all-reduce row tiles (256 rows)
constexpr int LRT = NRT / NDEV;  // row tiles per reduce-scatter owner
constexpr int NMSG =
    43;  // LL messages per (row tile, token): 6 bf16 + tag each, the last carries 4
constexpr int MAXG = 160;
constexpr int ACC = 4, ACC_COLS = 64;  // TMEM accumulator buffers (routed: columns 0..31; shared
                                       // FC1: gate 0..31, up 32..63)
constexpr int MAXPS = MAXM / NMAX;     // shared-expert panels (tokens 32 sp .. 32 sp + 31)
constexpr int NT1S =
    INTER /
    128;  // shared FC1 items per panel (128 h columns: gate rows 128 k.., up rows INTER + 128 k..)
constexpr int NSTG1S = NCH1;  // 32 shared FC1 stages per item (one 128-k chunk of both tiles)
// scale factors per ring stage: SFA chunks at columns 0..7 (4 per chunk), SFB at TSF_B .. TSF_B + 3
// (16 columns per stage)
constexpr int TMEM_COLS = 512, TSF_STRIDE = 16, TSF_B = 8, TSF0 = TMEM_COLS - NS * TSF_STRIDE;
static_assert(ACC * ACC_COLS <= TSF0, "TMEM: accumulators + scale factors");
constexpr int WR = 16;  // work ring depth
// shared memory (1024-aligned base): A [NS][KC][16 KB] | B [NS][KC][NMAX][128] | SFA [NS][KC][512]
// | SFB [NS][512] | tables | ring | bars
constexpr int A_BYTES = KC * 16384, B_BYTES = KC * NMAX * 128;
constexpr int OFF_A = 0, OFF_B = OFF_A + NS * A_BYTES, OFF_SFA = OFF_B + NS * B_BYTES,
              OFF_SFB = OFF_SFA + NS * KC * 512;
constexpr int OFF_TAB = OFF_SFB + NS * 512;
constexpr int TAB_ENT = 0;                    // uint16 [MAXE] entry: token | top-k slot mask << 9
constexpr int TAB_PINF = TAB_ENT + MAXE * 2;  // uint32 [MAXP] panel: e | n << 8 | ebase << 14
constexpr int TAB_RW =
    TAB_PINF + MAXP * 4;  // float [8][8] small fused gate (M <= 8): routing weights of (t, k)
constexpr int TAB_SIDS =
    TAB_RW + 256;  // int16 [8][8] small fused gate: expert ids of (t, k) (T1 reads them here)
constexpr int TAB_AM = TAB_SIDS + 128;  // float [4][NMAX] FC1 epilogue: per-warp amax exchange
constexpr int TAB_MISC = TAB_AM + 4 * NMAX * 4;  // int [32]: 0 experts present, 1 entries, 2
                                                 // panels, 3 TMEM base, 4 shared panels, 8 T1 item
constexpr int TAB_LANES = TAB_MISC + 32 * 4;     // uint8 [128] FC2 row of TMEM lane
constexpr int TAB_STG = TAB_LANES + 128;         // bf16 [2][16][128] FC2 epilogue transpose staging
constexpr int TAB_T1S = TAB_STG + 2 * 16 * 128 * 2;  // bf16 [256] T1 warp staging
constexpr int T1Q = 64;               // FC2-unit completion queue (epilogue -> warp-7 publisher)
constexpr int TAB_T1Q = TAB_T1S + 0;  // int [T1Q] unit descriptors (d.x << 24 | d.y), [T1Q] =
                                      // count, [T1Q + 1] = epilogue done, [T1Q + 2] = consumed
constexpr int TAB_RING = TAB_T1Q + (T1Q + 4) * 4;  // int4 [WR] work ring
constexpr int TAB_BAR = TAB_RING + WR * 16;        // uint64 barriers
constexpr int NBAR = 3 * NS + 2 * ACC + 2 * WR;    // FULL / EMPTY / SFR per ring stage, ACCF / ACCE
                                                   // per accumulator, WF / WE per ring slot
constexpr int TAB_BYTES = (TAB_BAR + NBAR * 8 + 15) & ~15;
constexpr int OFF_SFBI =
    (OFF_TAB + TAB_BYTES + 1023) &
    ~1023;  // (TPMOE_SFBI) uint8 [2][NSTG1][512] routed FC1 SFB images per item
constexpr int SFBI_BYTES = TPMOE_SFBI ? 2 * NSTG1 * 512 : 0;
constexpr int SMEM_TOTAL = (TPMOE_SFBI ? OFF_SFBI + SFBI_BYTES : OFF_TAB + TAB_BYTES),
              SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(OFF_B % 1024 == 0 && B_BYTES % 1024 == 0 && OFF_TAB % 16 == 0 && TAB_BAR % 8 == 0,
              "alignment");
// prologue scratch inside the A ring (dead before the first TMA)
constexpr int PRO_BITS =
    2 * A_BYTES;  // uint32 [16 words][NEXP] token bits per expert (above the early stages)
constexpr int PRO_PRE = PRO_BITS + 16 * NEXP * 4;  // uint16 [16][NEXP] exclusive word prefix
constexpr int PRO_OFFE = PRO_PRE + 16 * NEXP * 2;  // int [NEXP] entry offset of expert
constexpr int PRO_WS = PRO_OFFE + NEXP * 4;        // int [32] warp sums
constexpr int PRO_GIDS = PRO_WS + 32 * 4;          // int [8][8] small fused gate: ids
constexpr int PRO_IDS = PRO_GIDS + 256;  // uint16 [MAXM][8] expert of (t, k), 0xFFFF = none
static_assert(PRO_IDS + MAXM * 8 * 2 <= NS * A_BYTES && PRO_IDS % 16 == 0, "prologue scratch");

struct Params {
  CUtensorMap map1, map2;   // w13 [E * 1024 rows][2048 B], w2 [E * 4096 rows][256 B]: 16U4_ALIGN16B
                            // {128, rows, chunks}
  const uint8_t* w13_sf;    // [E][1024][128] block_scale_interleave bytes: 512 B per (tile, chunk)
  const uint8_t* w2_sf;     // [E][DIM][16]
  const uint8_t* lanes;     // [128] FC2 output row of TMEM lane (per 128-row tile)
  const __nv_bfloat16* xb;  // [M][4096] MoE input
  const float* glog;        // fused gate: router logits fp32 [M][experts], else null (ids entry)
  const float* gbias;       // [256] score correction bias
  int* ids;                 // [M][6] (fused gate: written here)
  float* rw;                // [M][6]
  float factor;             // T1 factor on the routed sum
  uint8_t* xq;              // [MAXM][4096] e4m3
  uint8_t* himg;            // [MAXP][NCH2][NMAX][128] h, K-major SW128
  uint8_t* hsf;             // [MAXP][NSTG2][512] h SFB images (lane = token, byte 8 (c & 1) + kb)
  float* part;              // SHGU partials: per CTA [NMAX][128] (epoch | fp32) words
  unsigned* xflag;    // [2][MAXM] quantized 512-element groups of token t (8 = done; epoch parity)
  unsigned* gcnt;     // [2] gated tokens (epoch parity)
  unsigned* pan_cnt;  // [2][MAXP] FC1 items done per panel
  unsigned* rdy;  // [2][MAXP + 32] panels in the order their h became ready (p + 1; 0 = not yet),
                  // [2][MAXP] = slots taken
  unsigned* fc2_next;   // [4] FC2 ticket head [par], T1 item head [2 + par]
  unsigned* rt_cnt;     // [2][NRT] FC2 units stored per 256-row tile
  unsigned* work;       // call counters (epoch)
  __nv_bfloat16* outp;  // [MAXM][TOPK_MAX][4096] per-slot expert outputs
  uint4* rs[NDEV];      // peer reduce-scatter buffers (two parity halves)
  uint4* ag[NDEV];      // peer all-gather buffers (two parity halves)
  __nv_bfloat16* out;   // [M][4096]
  int M, my_rank, oneshot;
  CUtensorMap map1s, map2s;  // gate_up [1024][4096] e4m3: box {128 B, 128 rows, 1 chunk}; down
                             // [4096][512]: box {128, 128, 2}
  CUtensorMap mapxq;         // xq [MAXM][DIM] box {128 B, 1} SW128 (gather4 rows)
  uint8_t* xsf2;  // [MAXM][NSTG1][16] x scales by token in the SFB row layout (byte 8 (c & 1) + kb
                  // of stage c / 2)
  const uint8_t* sfa1s;  // [8 row blocks][32 k blocks][512] SFA images (one UE8M0 byte per 128 x
                         // 128 block, replicated)
  const uint8_t* sfa2s;  // [32][4][512]
  uint8_t* xgs;    // [shared panels][NCH1][NMAX][128] x per (token, 128-group) e4m3 (ceil rule, eps
                   // 1e-10), SW128
  uint8_t* xgsfs;  // [shared panels][NSTG1][512] its SFB images (4 equal bytes per 128-group)
  uint8_t*
      hsimg;  // [shared panels][NCH2][NMAX][128] shared h (silu_and_mul_clamp bf16 -> per-128 FP8)
  uint8_t* hssf;         // [shared panels][NSTG2][512]
  __nv_bfloat16* shout;  // [MAXM][DIM] shared expert output (bf16), added in T1
};
__device__ __forceinline__ int pM(const Params& P) {
  return max(TPMOE_MMAX ? min(P.M, TPMOE_MMAX) : P.M, TPMOE_MMIN);
}  // (TPMOE_MMAX: range known to the compiler)
__device__ __forceinline__ int pOS(const Params& P) {
  return (TPMOE_MMAX && TPMOE_MMAX <= TPMOE_ONESHOT_MAXM)
             ? 1
             : (TPMOE_MMIN > TPMOE_ONESHOT_MAXM ? 0 : P.oneshot);
}

// ------------------------------------------------------------------------------------------------
// device helpers
__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t b, int c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(c) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(b),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
      "W_%=;\n}\n" ::"r"(b),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                      uint32_t bar, uint64_t hint = 0x12F0000000000000ull) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(hint)
      : "memory");
}
constexpr uint64_t SH_HINT = 0x12F0000000000000ull;
__device__ __forceinline__ void tma_gather4(uint32_t dst, const CUtensorMap* map, int col, int r0,
                                            int r1, int r2, int r3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], "
      "[%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
      "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void st_v4_hint(void* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }
__device__ __forceinline__ uint4 ld_v4_last_use(const void* p) {
  return __ldcg(reinterpret_cast<const uint4*>(p));
}
__device__ __forceinline__ void st_relaxed64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_relaxed(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v) {
  while (ld_acquire(p) < v) {
  }
}
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_relaxed_add(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;"
               : "=r"(r)
               : "l"(p), "r"(v)
               : "memory");
  return r;
}
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void st_ll(uint4* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint4 ld_ll(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(pred));
  return pred != 0;
}
__device__ __forceinline__ uint32_t warp_uniform(uint32_t v) {
  return __shfl_sync(0xffffffffu, v, 0);
}
// routed FC1 activation: up * silu(gate) on the clamped values
__device__ __forceinline__ float silu_mul(float g, float u) {
  return u * __fdividef(g, 1.0f + __expf(-g));
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
// MXFP8 scale exponent: 2^ceil(log2(amax / 448)) as an unbiased exponent (amax 0 -> the smallest
// normal scale)
__device__ __forceinline__ int ue8m0_exp(float amax) {
  const uint32_t b = __float_as_uint(fmaxf(amax * (1.0f / 448.0f), 1.17549435e-38f));
  return static_cast<int>((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
}
// (h of the routed FC1: h_scale_exp, common.cuh)
__device__ __forceinline__ float exp2i(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}
__device__ __forceinline__ uint32_t e4m3_of(float v) {
  return static_cast<uint32_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// tcgen05
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo,
                                              uint32_t layout) {
  return static_cast<uint64_t>((addr >> 4) & 0x3FFF) |
         (static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16) |
         (static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) |
         (static_cast<uint64_t>(layout) << 61);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return smem_desc(addr, 0, 1024, 2);
}
__device__ __forceinline__ uint64_t desc_sf(uint32_t addr) { return smem_desc(addr, 0, 128, 0); }
// kind::mxf8f6f4: A e2m1 (5), B e4m3 (0), fp32 D, UE8M0 scales, K-major, M = 128; N >> 3 at bit 17;
// scale byte ids
__device__ __forceinline__ uint32_t
idesc_f4f8(int n, uint32_t sf_id, uint32_t atype = 5u) {  // atype 5 e2m1 (routed), 0 e4m3 (shared)
  return (sf_id << 4) | (atype << 7) | (static_cast<uint32_t>(n >> 3) << 17) | (1u << 23) |
         ((128u >> 4) << 24) | (sf_id << 29);
}
#define TPMOE_CG "cta_group::1"
__device__ __forceinline__ void tc_mma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                                       uint32_t sfa, uint32_t sfb, uint32_t acc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
      " tcgen05.mma." TPMOE_CG
      ".kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb), "r"(acc)
      : "memory");
}
__device__ __forceinline__ void tc_cp_sf(uint32_t taddr, uint64_t desc) {
  asm volatile("tcgen05.cp." TPMOE_CG ".32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(desc)
               : "memory");
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar)
      : "memory");
}
// 16 columns of the warp's TMEM lane quadrant AND the wait in one asm block: tcgen05.ld's
// destination registers are undefined until tcgen05.wait::ld, so the compiler must not see them
// before it (separate asm statements let it copy them early -- read stale values)
__device__ __forceinline__ void tc_ld16_wait(uint32_t taddr, float* v) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n\t"
      "tcgen05.wait::ld.sync.aligned;"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15])
      : "r"(taddr)
      : "memory");
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

// ------------------------------------------------------------------------------------------------
// fused gate (bitwise moe_fused_gate)
__device__ __forceinline__ float tri_act(float x) {
  float r;
  asm("{\n\t"
      ".reg .pred ps, pf, pz, pg;\n\t"
      ".reg .b32 a, e, y, y2, ys, eb, t, mb, fe, f, p, q, lg;\n\t"
      "mul.f32 a, %1, 0f3FB8AA3B;\n\t"
      "ex2.approx.f32 e, a;\n\t"
      "add.f32 y, e, 0f3F800000;\n\t"
      "setp.lt.f32 ps, y, 0f00800000;\n\t"
      "mul.f32 y2, y, 0f4B000000;\n\t"
      "selp.f32 ys, y2, y, ps;\n\t"
      "selp.f32 eb, 0fC1B80000, 0f00000000, ps;\n\t"
      "add.s32 t, ys, -1059760811;\n\t"
      "and.b32 t, t, -8388608;\n\t"
      "sub.s32 mb, ys, t;\n\t"
      "cvt.rn.f32.s32 fe, t;\n\t"
      "fma.rn.ftz.f32 fe, fe, 0f34000000, eb;\n\t"
      "add.f32 f, mb, 0fBF800000;\n\t"
      "fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3E0F2955;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n\t"
      "mul.f32 q, f, p;\n\t"
      "fma.rn.ftz.f32 q, q, f, f;\n\t"
      "fma.rn.ftz.f32 lg, fe, 0f3F317218, q;\n\t"
      "setp.lt.u32 pf, ys, 2139095040;\n\t"
      "@!pf fma.rn.ftz.f32 lg, ys, 0f7F800000, 0f7F800000;\n\t"
      "setp.eq.f32 pz, ys, 0f00000000;\n\t"
      "selp.f32 lg, 0fFF800000, lg, pz;\n\t"
      "setp.gt.f32 pg, %1, 0f41A00000;\n\t"
      "selp.f32 lg, %1, lg, pg;\n\t"
      "sqrt.approx.ftz.f32 %0, lg;\n\t"
      "}"
      : "=f"(r)
      : "f"(x));
  return r;
}
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t gate_key(float v) {
  const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ void gate_cas(uint64_t& a, uint64_t& b) {
  const uint64_t x = a > b ? a : b, y = a > b ? b : a;
  a = x;
  b = y;
}
constexpr int GATE_TOPK = 6;
// one token by one warp: lanes < topk return (expert id, routing weight) of slot lane
__device__ __forceinline__ void gate_token(const Params& P, const float* lg, int lane, int& id,
                                           float& wt) {
  float lv[8], bv[8], sc[8];
  uint64_t h[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    lv[w] = __ldcg(lg + lane + 32 * w);
    bv[w] = __ldg(P.gbias + lane + 32 * w);
    sc[w] = 0.f;
    h[w] = 0ull;
  }
#if TPMOE_GUNR  // (8 independent activation chains interleaved: the rolled loop serializes
                // ~150-cycle dependent chains)
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    const float a = tri_act(lv[w]);
    float v = __fadd_rn(a, bv[w]);
    v = v == v ? v : -1e30f;
    sc[w] = a;
    h[w] = (static_cast<uint64_t>(gate_key(v)) << 32) |
           static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
  }
#else
#pragma unroll 1
  for (int w = 0; w < 8;
       ++w) {  // (input at [0], result appended at [7]: after 8 rounds element w sits at [w])
    const float a = tri_act(lv[0]);
    float v = __fadd_rn(a, bv[0]);
    v = v == v ? v : -1e30f;
    const uint64_t hw = (static_cast<uint64_t>(gate_key(v)) << 32) |
                        static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lane + 32 * w));
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      lv[k] = lv[k + 1];
      bv[k] = bv[k + 1];
      sc[k] = sc[k + 1];
      h[k] = h[k + 1];
    }
    sc[7] = a;
    h[7] = hw;
  }
#endif

  gate_cas(h[0], h[2]);
  gate_cas(h[1], h[3]);
  gate_cas(h[4], h[6]);
  gate_cas(h[5], h[7]);
  gate_cas(h[0], h[4]);
  gate_cas(h[1], h[5]);
  gate_cas(h[2], h[6]);
  gate_cas(h[3], h[7]);
  gate_cas(h[0], h[1]);
  gate_cas(h[2], h[3]);
  gate_cas(h[4], h[5]);
  gate_cas(h[6], h[7]);
  gate_cas(h[2], h[4]);
  gate_cas(h[3], h[5]);
  gate_cas(h[1], h[4]);
  gate_cas(h[3], h[6]);
  gate_cas(h[1], h[2]);
  gate_cas(h[3], h[4]);
  gate_cas(h[5], h[6]);
  int ex[GATE_TOPK];
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) ex[k] = 0;
#pragma unroll 1
  for (int k = 0; k < GATE_TOPK; ++k) {
    const uint32_t hk = static_cast<uint32_t>(h[0] >> 32), hl = static_cast<uint32_t>(h[0]);
    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
    const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
    if (hk == gm && hl == gl) {
#pragma unroll
      for (int i = 0; i < 7; ++i) h[i] = h[i + 1];
      h[7] = 0ull;
    }
#pragma unroll
    for (int i = 0; i < GATE_TOPK - 1; ++i) ex[i] = ex[i + 1];
    ex[GATE_TOPK - 1] = static_cast<int>(0xFFFFFFFFu - gl);
  }
  float R[GATE_TOPK];
#if TPMOE_GUNR  // (6 independent lookups)
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) {
    const int ew = ex[k] >> 5;
    float sv = sc[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
    R[k] = __shfl_sync(0xffffffffu, sv, ex[k] & 31);
  }
#else
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) R[k] = 0.f;
#pragma unroll 1
  for (int k = 0; k < GATE_TOPK; ++k) {  // (ex rotates through once: back in order afterwards)
    const int e = ex[0];
#pragma unroll
    for (int i = 0; i < GATE_TOPK - 1; ++i) {
      ex[i] = ex[i + 1];
      R[i] = R[i + 1];
    }
    ex[GATE_TOPK - 1] = e;
    const int ew = e >> 5;
    float sv = sc[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) sv = ew == w ? sc[w] : sv;
    R[GATE_TOPK - 1] = __shfl_sync(0xffffffffu, sv, e & 31);
  }
#endif
  const float sum =
      __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
  const float norm = sum > 0.0f ? sum : 1.0f;
  float r = R[0];
  id = ex[0];
#pragma unroll
  for (int k = 1; k < GATE_TOPK; ++k) {
    r = lane == k ? R[k] : r;
    id = lane == k ? ex[k] : id;
  }
  wt = div_full(r, norm);
}
// one token gated by warps 0..7 together (expert 32 warp + lane): per-warp top-6 by redux, then
// warp 0 merges the 48 candidates -- the same keys (score bits, then the lower expert index),
// selection order and weight arithmetic as gate_token
__device__ __forceinline__ void gate_token_cta(const Params& P, const float* lg, int warp, int lane,
                                               uint64_t* cand, float* sca, int& id, float& wt) {
  const int e = 32 * warp + lane;
  float lv = __ldcg(lg + e);
  const float bv = __ldg(P.gbias + e);
  const float a = tri_act(lv);
  float v = __fadd_rn(a, bv);
  v = v == v ? v : -1e30f;
  uint64_t h = (static_cast<uint64_t>(gate_key(v)) << 32) |
               static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(e));
  sca[e] = a;
#pragma unroll 1
  for (int k = 0; k < GATE_TOPK; ++k) {
    const uint32_t hk = static_cast<uint32_t>(h >> 32), hl = static_cast<uint32_t>(h);
    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
    const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
    if (lane == 0) cand[warp * GATE_TOPK + k] = (static_cast<uint64_t>(gm) << 32) | gl;
    if (hk == gm && hl == gl) h = 0ull;
  }
  named_bar(6, 256);
  if (warp != 0) return;
  uint64_t c0 = cand[lane], c1 = lane + 32 < 8 * GATE_TOPK ? cand[lane + 32] : 0ull;
  if (c1 > c0) {
    const uint64_t t = c0;
    c0 = c1;
    c1 = t;
  }
  int ex[GATE_TOPK];
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) {
    const uint32_t hk = static_cast<uint32_t>(c0 >> 32), hl = static_cast<uint32_t>(c0);
    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
    const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
    ex[k] = static_cast<int>(0xFFFFFFFFu - gl);
    if (hk == gm && hl == gl) {
      c0 = c1;
      c1 = 0ull;
    }
  }
  float R[GATE_TOPK];
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) R[k] = sca[ex[k]];
  const float sum =
      __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), R[2]), R[3]), __fadd_rn(R[4], R[5]));
  const float norm = sum > 0.0f ? sum : 1.0f;
  float r = R[0];
  id = ex[0];
#pragma unroll
  for (int k = 1; k < GATE_TOPK; ++k) {
    r = lane == k ? R[k] : r;
    id = lane == k ? ex[k] : id;
  }
  wt = div_full(r, norm);
}

// MXFP8 quantization of x in units (token t, 512-element group cg) by one warp, lane l: elements 16
// l .. + 15 (chunk 4 cg + l / 8, 16-B granule l % 8, 32-block = lanes l, l ^ 1).  Writes the
// token-major codes xq and scales xsf2 (the gather4 B operand) and the shared expert's per-128
// image, then counts the unit on xcnt[t].  Units u = gw + i nw, 4 per warp in flight.
__device__ __forceinline__ void quant_x(const Params& P, unsigned* xcnt, int lane, int gw, int nw) {
  const int U = pM(P) * 8;
  const int QN = (8 * TPMOE_MMAX <= nw) ? 1 : 4;  // (units in flight; 1: every warp has <= 1 unit)
  for (int u0 = gw; u0 < U; u0 += QN * nw) {
    uint4 raw[4][2];
#pragma unroll
    for (int j = 0; j < QN; ++j) {
      const int u = u0 + j * nw;
      if (u < U) {
        const uint4* src = reinterpret_cast<const uint4*>(P.xb + static_cast<size_t>(u >> 3) * DIM +
                                                          (u & 7) * 512) +
                           lane * 2;
        raw[j][0] = __ldg(src);
        raw[j][1] = __ldg(src + 1);
      }
    }
#pragma unroll
    for (int j = 0; j < QN; ++j) {
      const int u = u0 + j * nw;
      if (u >= U) break;
      const int t = u >> 3, cg = u & 7, ch = 4 * cg + (lane >> 3), gr = lane & 7;
      const uint32_t w8[8] = {raw[j][0].x, raw[j][0].y, raw[j][0].z, raw[j][0].w,
                              raw[j][1].x, raw[j][1].y, raw[j][1].z, raw[j][1].w};
      float am = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) am = fmaxf(am, fmaxf(fabsf(bf16lo(w8[i])), fabsf(bf16hi(w8[i]))));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
      const int ex = ue8m0_exp(am);
      const float inv = exp2i(-ex);
      uint32_t cw[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        cw[i] = e4m3_of(bf16lo(w8[2 * i]) * inv) | (e4m3_of(bf16hi(w8[2 * i]) * inv) << 8) |
                (e4m3_of(bf16lo(w8[2 * i + 1]) * inv) << 16) |
                (e4m3_of(bf16hi(w8[2 * i + 1]) * inv) << 24);
      const uint4 code = make_uint4(cw[0], cw[1], cw[2], cw[3]);
      const uint32_t eb = static_cast<uint32_t>(ex + 127);
      uint32_t sfw = 0;  // the chunk's 4 scale bytes (blocks of lanes 8 c + 0, 2, 4, 6)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
        sfw |= (__shfl_sync(0xffffffffu, eb, (lane & ~7) + 2 * kb) & 0xFFu) << (8 * kb);
      {  // shared expert input: per (token, 128-group) e4m3, scale 2^ceil(log2(max(amax, 1e-10) /
         // 448)) (per-token-group quant, ue8m0)
        float a128 = am;
        a128 = fmaxf(a128, __shfl_xor_sync(0xffffffffu, a128, 2));
        a128 = fmaxf(a128, __shfl_xor_sync(0xffffffffu, a128, 4));
        const int e128 = ue8m0_exp(fmaxf(a128, 1e-10f));
        const float i128 = exp2i(-e128);
        uint32_t sw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          sw[i] = e4m3_of(bf16lo(w8[2 * i]) * i128) | (e4m3_of(bf16hi(w8[2 * i]) * i128) << 8) |
                  (e4m3_of(bf16lo(w8[2 * i + 1]) * i128) << 16) |
                  (e4m3_of(bf16hi(w8[2 * i + 1]) * i128) << 24);
        const int sp = t >> 5, r = t & 31;
        uint8_t* row = P.xgs + ((static_cast<size_t>(sp) * NCH1 + ch) * NMAX) * 128 +
                       (r >> 3) * 1024 + (r & 7) * 128;
        *reinterpret_cast<uint4*>(row + ((gr ^ (r & 7)) << 4)) =
            make_uint4(sw[0], sw[1], sw[2], sw[3]);
        if (gr == 0)
          *reinterpret_cast<uint32_t*>(
              P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + (ch >> 1)) * 512 + r * 16 +
              8 * (ch & 1)) = static_cast<uint32_t>(e128 + 127) * 0x01010101u;
      }
      *reinterpret_cast<uint4*>(P.xq + static_cast<size_t>(t) * DIM + cg * 512 + lane * 16) = code;
      if (gr == 0)
        *reinterpret_cast<uint32_t*>(P.xsf2 + (static_cast<size_t>(t) * NSTG1 + (ch >> 1)) * 16 +
                                     8 * (ch & 1)) = sfw;
    }
  }
  // one fence for all of this warp's units, then their counts (fence + relaxed RMW = release)
  __syncwarp();
  if (lane == 0 && gw < U) {
    fence_acq_rel();
    for (int u = gw; u < U; u += nw) red_relaxed_add(xcnt + (u >> 3), 1u);
  }
}

// ------------------------------------------------------------------------------------------------
// routing tables
struct Tabs {
  uint16_t* ids;
  uint16_t* ent;
  uint8_t* km;
  uint32_t* pinf;
  float* srw;
  float* am;
  int* misc;
  uint8_t* lanes;
  int4* ring;
  __nv_bfloat16* stg;
  int16_t* sids;
};
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
  uint8_t* t = smem + OFF_TAB;
  return Tabs{reinterpret_cast<uint16_t*>(smem + PRO_IDS),
              reinterpret_cast<uint16_t*>(t + TAB_ENT),
              nullptr,
              reinterpret_cast<uint32_t*>(t + TAB_PINF),
              reinterpret_cast<float*>(t + TAB_RW),
              reinterpret_cast<float*>(t + TAB_AM),
              reinterpret_cast<int*>(t + TAB_MISC),
              t + TAB_LANES,
              reinterpret_cast<int4*>(t + TAB_RING),
              reinterpret_cast<__nv_bfloat16*>(t + TAB_STG),
              reinterpret_cast<int16_t*>(t + TAB_SIDS)};
}
// Every CTA, all THREADS: union = present experts in ascending id; entries of an expert = its
// tokens in ascending order (a token listing the same expert in several slots is ONE entry, km =
// its slots); panels of <= NMAX entries.  ids from smem (gids, small fused gate) or global (P.ids).
// Writes T.ids, T.ent, T.km, T.pinf, misc 0..2.
__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T, const int* gids) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = pM(P), topk = 6, MT = M * topk, W = (M + 31) >> 5;
  uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
  uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
  int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
  int* ws = reinterpret_cast<int*>(smem + PRO_WS);
  for (int i = tid; i < W * NEXP; i += THREADS) bits[i] = 0u;
  for (int i = tid; i < M * 8; i += THREADS) T.ids[i] = 0xFFFF;
  __syncthreads();
  for (int p = tid; p < MT; p += THREADS) {
    const int e = (gids ? gids[p] : __ldcg(P.ids + p));
    const int t = p / topk, k = p - t * topk;
    if (static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
      T.ids[t * 8 + k] = static_cast<uint16_t>(e);
      atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
    }
  }
  __syncthreads();
  // per expert (tid == e, warps 0..7): count, word prefix; block scans of (present, count, panels)
  int c = 0, pres = 0, npan = 0, a = 0, b = 0, d = 0;
  if (tid < NEXP) {
    const int e = tid;
    for (int w = 0; w < W; ++w) {
      pre[w * NEXP + e] = static_cast<uint16_t>(c);
      c += __popc(bits[w * NEXP + e]);
    }
    pres = c > 0;
    npan = (c + NMAX - 1) / NMAX;
    a = pres;
    b = c;
    d = npan;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int ya = __shfl_up_sync(0xffffffffu, a, o), yb = __shfl_up_sync(0xffffffffu, b, o),
                yd = __shfl_up_sync(0xffffffffu, d, o);
      if (lane >= o) {
        a += ya;
        b += yb;
        d += yd;
      }
    }
    if (lane == 31) {
      ws[warp] = a;
      ws[8 + warp] = b;
      ws[16 + warp] = d;
    }
  }
  __syncthreads();
  if (tid < NEXP) {
    int xb = b - c, xd = d - npan;
#pragma unroll
    for (int w = 0; w < 8; ++w)
      if (w < warp) {
        xb += ws[8 + w];
        xd += ws[16 + w];
      }
    offe[tid] = xb;
    if (pres) {
      for (int p = 0; p < npan; ++p)
        T.pinf[xd + p] = static_cast<uint32_t>(tid) |
                         (static_cast<uint32_t>(min(NMAX, c - p * NMAX)) << 8) |
                         (static_cast<uint32_t>(xb + p * NMAX) << 14);
    }
  }
  if (tid == NEXP - 1) {
    int U = 0, E = 0, Pn = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      U += ws[w];
      E += ws[8 + w];
      Pn += ws[16 + w];
    }
    T.misc[0] = U;
    T.misc[1] = E;
    T.misc[2] = Pn;
  }
  __syncthreads();
  // entries: one per (token, expert) at its first slot
  for (int p = tid; p < MT; p += THREADS) {
    const int t = p / topk, k = p - t * topk;
    const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);
    const uint32_t rw[4] = {row.x, row.y, row.z, row.w};
    const int e = static_cast<int>((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
    if (e == 0xFFFF) continue;
    unsigned km = 0;
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2)
      km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == static_cast<unsigned>(e) ? 1u : 0u)
            << k2;
    if (km & ((1u << k) - 1u)) continue;
    const int w = t >> 5;
    const int idx =
        offe[e] + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & ((1u << (t & 31)) - 1u));
    T.ent[idx] = static_cast<uint16_t>(t | (km << 9));
  }
  __syncthreads();
}

// TPMOE_TABW: the same tables for M <= 1 with the small fused gate (ids in smem, one panel per
// expert) by warp 0 alone, without the CTA-wide phases (cold code + 5 __syncthreads on the
// prologue's critical path).  Entries p = lane, lane + 32.
__device__ void build_tables_warp(const Params& P, uint8_t* smem, const Tabs& T, const int* gids,
                                  int lane) {
  const int M = pM(P), topk = 6, MT = M * topk;
  uint32_t* mask = reinterpret_cast<uint32_t*>(smem + PRO_BITS);  // 256-bit present-expert mask
  if (lane < 8) mask[lane] = 0u;
  __syncwarp();
  int ep[2], tp[2], kp[2];
  unsigned km[2];
  bool first[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int pidx = lane + 32 * h;
    ep[h] = pidx < MT ? gids[pidx] : -1;
    tp[h] = pidx < MT ? pidx / topk : 0;
    kp[h] = pidx - tp[h] * topk;
    km[h] = 0u;
    const bool valid = static_cast<unsigned>(ep[h]) < static_cast<unsigned>(NEXP);
    if (valid) {
      for (int k2 = 0; k2 < topk; ++k2) km[h] |= (gids[tp[h] * topk + k2] == ep[h] ? 1u : 0u) << k2;
      atomicOr(&mask[ep[h] >> 5], 1u << (ep[h] & 31));
    }
    first[h] = valid && (km[h] & ((1u << kp[h]) - 1u)) == 0u;
  }
  const unsigned F0 = __ballot_sync(0xffffffffu, first[0]),
                 F1 = __ballot_sync(0xffffffffu, first[1]);
  __syncwarp();
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!first[h]) continue;
    const int e = ep[h], t = tp[h];
    int idx_in = 0, n = 0,
        ebase = 0;  // rank among this expert's tokens, its token count, its first entry
    for (int q = 0; q < MT; ++q) {
      const bool fq = q < 32 ? ((F0 >> q) & 1u) : ((F1 >> (q - 32)) & 1u);
      if (!fq) continue;
      const int eq = gids[q];
      if (eq == e) {
        ++n;
        if (q < t * topk) ++idx_in;
      } else if (eq < e)
        ++ebase;
    }
    int pe = __popc(mask[e >> 5] & ((1u << (e & 31)) - 1u));
    for (int w = 0; w < (e >> 5); ++w) pe += __popc(mask[w]);
    T.ent[ebase + idx_in] = static_cast<uint16_t>(t | (km[h] << 9));
    if (idx_in == 0)
      T.pinf[pe] = static_cast<uint32_t>(e) | (static_cast<uint32_t>(n) << 8) |
                   (static_cast<uint32_t>(ebase) << 14);
  }
  if (lane == 0) {
    int U = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) U += __popc(mask[w]);
    T.misc[0] = U;
    T.misc[1] = __popc(F0) + __popc(F1);
    T.misc[2] = U;
  }
}

// ------------------------------------------------------------------------------------------------
// work Ring entries: x = kind (1 routed FC1 item, 2 routed FC2 unit, 3 shared FC1 item, 4 shared
// FC2 unit, -1 end); FC1: y = item (panel item / NT1, tile item % NT1), z / w = its stage range;
// FC2: y = panel NRT2 + first row tile, z = row tiles.
__device__ __forceinline__ void panel_of(const Tabs& T, int p, int& e, int& n, int& ebase) {
  const uint32_t v = T.pinf[p];
  e = static_cast<int>(v & 255u);
  n = static_cast<int>((v >> 8) & 63u);
  ebase = static_cast<int>(v >> 14);
}

// ------------------------------------------------------------------------------------------------
// tail: T1 + all-reduce
__device__ __forceinline__ size_t rs_idx(int src, int lrt, int t, int m) {
  return (static_cast<size_t>((src * LRT + lrt) * MAXM + t)) * NMSG + m;
}
__device__ __forceinline__ size_t ag_idx(int rtile, int t, int m) {
  return (static_cast<size_t>(rtile * MAXM + t)) * NMSG + m;
}
constexpr int OS_MAXM = MAXM / NDEV;
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) {
  return (static_cast<size_t>((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m;
}
constexpr size_t RS_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;
constexpr int OUTP_LD = DIM + 0;  // outp [MAXM][TOPK_MAX][OUTP_LD]
constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;

// T1 of one (row tile, token) by one warp: lane l owns values rtile * 256 + 8 l .. + 7
struct T1In {
  uint4 v[TOPK_MAX];
  float wk[TOPK_MAX];
  int idk[TOPK_MAX];
  uint4 sh;
};
__device__ __forceinline__ void t1_load(const Params& P, const Tabs& T, bool small_rw, int rtile,
                                        int t, int lane, T1In& in) {
  const int topk = 6;
  const int c0 = rtile * 256 + 8 * lane;
  // every load issued at once: slots past topk / invalid ids are loaded but masked off; small fused
  // gate: ids / weights from smem
  in.sh = ld_v4_last_use(P.shout + static_cast<size_t>(t) * DIM + c0);
#pragma unroll
  for (int k = 0; k < TOPK_MAX; ++k) {
    if (k < topk) {
      in.idk[k] = small_rw ? static_cast<int>(T.sids[t * topk + k]) : __ldcg(P.ids + t * topk + k);
      in.wk[k] = small_rw ? T.srw[t * topk + k] : __ldcg(P.rw + t * topk + k);
      in.v[k] = ld_v4_last_use(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * OUTP_LD + c0);
    } else {
      in.idk[k] = -1;
      in.wk[k] = 0.f;
      in.v[k] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}
__device__ __forceinline__ void t1_emit(const Params& P, int rtile, int t, unsigned epoch,
                                        __nv_bfloat16* stg, int lane, const T1In& in) {
  unsigned on = 0u;
  float wk[TOPK_MAX];
#pragma unroll
  for (int k = 0; k < TOPK_MAX; ++k) {
    if (static_cast<unsigned>(in.idk[k]) < static_cast<unsigned>(NEXP)) on |= 1u << k;
    wk[k] = bf16r(in.wk[k]);  // (bf16 routing weights)
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TOPK_MAX; ++k) {
    if (!((on >> k) & 1u)) continue;
    const uint32_t w4[4] = {in.v[k].x, in.v[k].y, in.v[k].z, in.v[k].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[2 * j] = __fmaf_rn(bf16lo(w4[j]), wk[k], acc[2 * j]);
      acc[2 * j + 1] = __fmaf_rn(bf16hi(w4[j]), wk[k], acc[2 * j + 1]);
    }
  }
  const uint32_t s4[4] = {in.sh.x, in.sh.y, in.sh.z, in.sh.w};
  uint32_t o4[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float r0 = bf16r(acc[2 * j]), r1 = bf16r(acc[2 * j + 1]);
    const float y0 = __fmaf_rn(P.factor, r0, bf16lo(s4[j]));
    const float y1 = __fmaf_rn(P.factor, r1, bf16hi(s4[j]));
    o4[j] = pack_bf16x2(y0, y1);
  }
  *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
  __syncwarp();
  for (int m = lane; m < NMSG; m += 32) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
    const bool full6 = m < NMSG - 1;
    const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
    if (pOS(P)) {
#pragma unroll
      for (int d = 0; d < NDEV; ++d)
        st_ll(P.rs[(P.my_rank + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR +
                  os_idx(P.my_rank, rtile, t, m),
              msg);
    } else {
      st_ll(P.rs[rtile / LRT] + (epoch & 1u) * RS_PAR + rs_idx(P.my_rank, rtile % LRT, t, m), msg);
    }
  }
  __syncwarp();
}

__device__ void tail(const Params& P, const Tabs& T, bool small_rw, int tid, unsigned epoch,
                     unsigned par, uint8_t* smem) {
  const int G = 148, b = blockIdx.x, M = pM(P);
  const unsigned need = static_cast<unsigned>(T.misc[2] + T.misc[4]) *
                        2u;  // FC2 row tiles (routed + shared panels) per 256-row tile
  const int TCH = (M + max(1, G / NRT) - 1) / max(1, G / NRT);
  const int ntc = (M + TCH - 1) / TCH, nitem = NRT * ntc;
  volatile int* s_item = reinterpret_cast<volatile int*>(T.misc + 8);
  unsigned done_rt = 0u, done_it = 0u;
  for (;;) {
    if (tid == 0) {
      const int it = b + static_cast<int>(done_it++) * G;
      if (it < nitem) {
        const int rt = it / ntc;

        if (!((done_rt >> rt) & 1u)) {
          wait_ge(P.rt_cnt + par * NRT + rt, need);
          done_rt |= 1u << rt;
        }
      }
      *s_item = it;
    }
    __syncthreads();
    const int it = *s_item;
    __syncthreads();
    if (it >= nitem) break;
    const int rtile = it / ntc, t0 = (it - rtile * ntc) * TCH, nt = min(TCH, M - t0);
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem) + (tid >> 5) * 256;
    constexpr int NW = THREADS / 32;
    for (int tt = tid >> 5; tt < nt;
         tt += 2 * NW) {  // (a warp's pairs tt, tt + NW: both pairs' loads in one round trip)
      T1In ia, ib;
      t1_load(P, T, small_rw, rtile, t0 + tt, tid & 31, ia);
      const bool two = tt + NW < nt;
      if (two) t1_load(P, T, small_rw, rtile, t0 + tt + NW, tid & 31, ib);
      t1_emit(P, rtile, t0 + tt, epoch, stg, tid & 31, ia);
      if (two) t1_emit(P, rtile, t0 + tt + NW, epoch, stg, tid & 31, ib);
    }
  }
  if (pOS(P)) {
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
    const int nm = NMSG;
    for (int x = tid; x < mine * nm; x += THREADS) {
      const int pi = x / nm, m = x - pi * nm, j = b + pi * G, rtile = j / M, t = j - rtile * M;
      uint4 v[NDEV];
      unsigned pending = (1u << NDEV) - 1u;
      if (TPMOE_ARV & 8) {
        do {
          const unsigned p0 = pending;
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if (p0 & (1u << d)) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if ((p0 & (1u << d)) && v[d].w == epoch) pending &= ~(1u << d);
        } while (pending);
      } else
        do {
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if (pending & (1u << d)) {
              v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
              if (v[d].w == epoch) pending &= ~(1u << d);
            }
        } while (pending);
      float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y),
                    bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
      for (int d = 1; d < NDEV; ++d) {
        a[0] = __fadd_rn(a[0], bf16lo(v[d].x));
        a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
        a[2] = __fadd_rn(a[2], bf16lo(v[d].y));
        a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
        a[4] = __fadd_rn(a[4], bf16lo(v[d].z));
        a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
      }
      const int row0 = 6 * m, nw = (m < NMSG - 1 ? 3 : 2);
      uint32_t* o =
          reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + row0);
      o[0] = pack_bf16x2(a[0], a[1]);
      if (nw > 1) o[1] = pack_bf16x2(a[2], a[3]);
      if (nw > 2) o[2] = pack_bf16x2(a[4], a[5]);
    }
    return;
  }
  if (TPMOE_ARV & 2) {  // (TPMOE_ARV & 2) T2 + T3 in one poll loop: a thread's <= 1 T2 message
                        // (NDEV sources) and <= 4 T3 messages in flight together
    const int n2 = (b < LRT * M ? (LRT * M - b + G - 1) / G : 0) * NMSG,
              n3 = (b < NRT * M ? (NRT * M - b + G - 1) / G : 0) * NMSG;
    if (n2 <= THREADS && n3 <= 4 * THREADS) {
      const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
      const uint4* ag_me = P.ag[P.my_rank] + (epoch & 1u) * AG_PAR;
      unsigned p2 = 0u, p3 = 0u;
      size_t a2 = 0, a3[4];
      int lrt = 0, t2 = 0, m2 = 0;
      if (tid < n2) {
        const int pi = tid / NMSG, j = b + pi * G;
        m2 = tid - pi * NMSG;
        lrt = j / M;
        t2 = j - lrt * M;
        a2 = rs_idx(0, lrt, t2, m2);
        p2 = (1u << NDEV) - 1u;
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int x = tid + k * THREADS;
        a3[k] = 0;
        if (x < n3) {
          const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M,
                    t = j - rtile * M;
          a3[k] = ag_idx(rtile, t, m);
          p3 |= 1u << k;
        }
      }
      constexpr size_t RS_SRC =
          static_cast<size_t>(LRT) * MAXM * NMSG;  // rs_idx(d + 1, ...) - rs_idx(d, ...)
      uint4 v2[NDEV], v3[4];
      while (p2 | p3) {
        const unsigned q2 = p2, q3 = p3;
#pragma unroll
        for (int d = 0; d < NDEV; ++d)
          if (q2 & (1u << d)) v2[d] = ld_ll(rs_me + a2 + d * RS_SRC);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if (q3 & (1u << k)) v3[k] = ld_ll(ag_me + a3[k]);
#pragma unroll
        for (int d = 0; d < NDEV; ++d)
          if ((q2 & (1u << d)) && v2[d].w == epoch) p2 &= ~(1u << d);
        if (q2 && !p2) {  // all sources in: reduce in rank order, all-gather push
          float a[6] = {bf16lo(v2[0].x), bf16hi(v2[0].x), bf16lo(v2[0].y),
                        bf16hi(v2[0].y), bf16lo(v2[0].z), bf16hi(v2[0].z)};
#pragma unroll
          for (int d = 1; d < NDEV; ++d) {
            a[0] = __fadd_rn(a[0], bf16lo(v2[d].x));
            a[1] = __fadd_rn(a[1], bf16hi(v2[d].x));
            a[2] = __fadd_rn(a[2], bf16lo(v2[d].y));
            a[3] = __fadd_rn(a[3], bf16hi(v2[d].y));
            a[4] = __fadd_rn(a[4], bf16lo(v2[d].z));
            a[5] = __fadd_rn(a[5], bf16hi(v2[d].z));
          }
          const uint4 mo = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]),
                                      pack_bf16x2(a[4], a[5]), epoch);
          const size_t ai = ag_idx(P.my_rank * LRT + lrt, t2, m2);
#pragma unroll
          for (int d = 0; d < NDEV; ++d) st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ai, mo);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if ((q3 & (1u << k)) && v3[k].w == epoch) {
            p3 &= ~(1u << k);
            const int x = tid + k * THREADS, pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G,
                      rtile = j / M, t = j - rtile * M;
            uint32_t* o = reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM +
                                                      rtile * 256 + 6 * m);
            o[0] = v3[k].x;
            o[1] = v3[k].y;
            if (m < NMSG - 1) o[2] = v3[k].z;
          }
      }
      return;
    }
  }
  {  // T2: owner reduce in rank order, all-gather push
    const int npair = LRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
    for (int x = tid; x < mine * NMSG; x += THREADS) {
      const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / M, t = j - lrt * M;
      uint4 v[NDEV];
      unsigned pending = (1u << NDEV) - 1u;
      if (TPMOE_ARV & 8) {
        do {
          const unsigned p0 = pending;
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if (p0 & (1u << d)) v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m));
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if ((p0 & (1u << d)) && v[d].w == epoch) pending &= ~(1u << d);
        } while (pending);
      } else
        do {
#pragma unroll
          for (int d = 0; d < NDEV; ++d)
            if (pending & (1u << d)) {
              v[d] = ld_ll(rs_me + rs_idx(d, lrt, t, m));
              if (v[d].w == epoch) pending &= ~(1u << d);
            }
        } while (pending);
      float a[6] = {bf16lo(v[0].x), bf16hi(v[0].x), bf16lo(v[0].y),
                    bf16hi(v[0].y), bf16lo(v[0].z), bf16hi(v[0].z)};
#pragma unroll
      for (int d = 1; d < NDEV; ++d) {
        a[0] = __fadd_rn(a[0], bf16lo(v[d].x));
        a[1] = __fadd_rn(a[1], bf16hi(v[d].x));
        a[2] = __fadd_rn(a[2], bf16lo(v[d].y));
        a[3] = __fadd_rn(a[3], bf16hi(v[d].y));
        a[4] = __fadd_rn(a[4], bf16lo(v[d].z));
        a[5] = __fadd_rn(a[5], bf16hi(v[d].z));
      }
      const uint4 mo = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]),
                                  pack_bf16x2(a[4], a[5]), epoch);
      const int rtile = P.my_rank * LRT + lrt;
#pragma unroll
      for (int d = 0; d < NDEV; ++d)
        st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
    }
  }
  {  // T3: pull the reduced rows
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* ag_me = P.ag[P.my_rank] + (epoch & 1u) * AG_PAR;
    for (int x = tid; x < mine * NMSG; x += THREADS) {
      const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
      uint4 v;
      do {
        v = ld_ll(ag_me + ag_idx(rtile, t, m));
      } while (v.w != epoch);
      uint32_t* o =
          reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
      o[0] = v.x;
      o[1] = v.y;
      if (m < NMSG - 1) o[2] = v.z;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// kernel
__global__ void __launch_bounds__(THREADS, 1) tp_moe_kernel(const __grid_constant__ Params P) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sraw = su32(smem_raw), sb = (sraw + 1023u) & ~1023u;
  uint8_t* smem = smem_raw + (sb - sraw);
  const Tabs T = tabs(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int G = 148, b = blockIdx.x, M = pM(P);
  const bool f2s = (M >= 1 && M <= TPMOE_FC2S_HI);  // FC2 tickets in static panel order
  auto bar = [&](int i) { return sb + OFF_TAB + TAB_BAR + 8 * i; };

  constexpr int B_FULL = 0, B_EMPTY = NS, B_SFR = 2 * NS, B_ACCF = 3 * NS, B_ACCE = 3 * NS + ACC,
                B_WF = 3 * NS + 2 * ACC, B_WE = B_WF + WR;
  auto we_arrive = [&](int slot) { mbar_arrive(bar(B_WE + slot)); };  // ring slot consumed
  auto use4 = [](const int4& d) { asm volatile("" ::"r"(d.x), "r"(d.y), "r"(d.z), "r"(d.w)); };
  // ---- prologue: barriers, TMEM, lane table (before the PDL wait)
  if (tid < NBAR) {  // one barrier per thread: FULL 2 (producer + B), EMPTY / SFR / ACCF / WF 1,
                     // ACCE 8 epilogue warps, WE 10 consumers
    const int i = tid;
    const int cnt = i < B_EMPTY ? 2 : (i < B_ACCE ? 1 : (i < B_WF ? 8 : (i < B_WE ? 1 : 10)));
    mbar_init(bar(i), cnt);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  const bool shgu =
      (M <= TPMOE_SHGU_MAXM ||
       (M >= TPMOE_SHGU_LO2 &&
        M <= TPMOE_SHGU_HI2));  // SHGU: a shared FC1 item's gate and up chains on two CTAs
  const int nshq0 = ((M + NMAX - 1) / NMAX) * NT1S * (shgu ? 2 : 1);  // shared FC1 pieces
  const int shh = (shgu && b < nshq0)
                      ? (b & 1)
                      : -1;  // (SHGU: this CTA's shared piece is the gate (0) or the up (1) chain)
  const bool shearly = nshq0 > 0 && 2 * nshq0 <= G && M > 2;
  const int npre =
      (shearly && b < nshq0)
          ? min(2, NSTG1S)
          : 0;  // ring stages issued before the PDL wait (the shared item's first stages)
  if (npre) {
    __syncthreads();  // (the barrier inits)
    if (tid == 0) {   // shared item b: the producer skips these stages
      const int k = (shgu ? b >> 1 : b) % NT1S;
      for (int j = 0; j < npre; ++j) {
        if (shh >= 0) {  // (SHGU: chunks 2 j, 2 j + 1 of this CTA's gate or up rows + their scales
                         // -- a full 32 KB slot)
          mbar_expect_tx(bar(B_FULL + j), 2 * 16384 + 2 * 512);
          for (int h2 = 0; h2 < 2; ++h2) {
            tma3d(sb + OFF_A + j * A_BYTES + h2 * 16384, &P.map1s, 0, shh * INTER + 128 * k,
                  2 * j + h2, bar(B_FULL + j), SH_HINT);
            bulk_g2s(
                sb + OFF_SFA + j * KC * 512 + h2 * 512,
                P.sfa1s + (static_cast<size_t>(shh * (INTER / 128) + k) * NCH1 + 2 * j + h2) * 512,
                512, bar(B_FULL + j));
          }
          continue;
        }
        mbar_expect_tx(bar(B_FULL + j), (2 * 16384) + 2 * 512);
        tma3d(sb + OFF_A + j * A_BYTES, &P.map1s, 0, 128 * k, j, bar(B_FULL + j), SH_HINT);
        tma3d(sb + OFF_A + j * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, j, bar(B_FULL + j),
              SH_HINT);
        bulk_g2s(sb + OFF_SFA + j * KC * 512, P.sfa1s + (static_cast<size_t>(k) * NCH1 + j) * 512,
                 512, bar(B_FULL + j));
        bulk_g2s(sb + OFF_SFA + j * KC * 512 + 512,
                 P.sfa1s + (static_cast<size_t>(INTER / 128 + k) * NCH1 + j) * 512, 512,
                 bar(B_FULL + j));
      }
    }
  }
  if (tid == 64) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map1) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&P.map2) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&P.mapxq) : "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc." TPMOE_CG
                 ".sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(su32(T.misc + 3)),
                 "r"(TMEM_COLS)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit." TPMOE_CG ".sync.aligned;" ::: "memory");
    tc_fence_before();
  }
  if (tid < 32)
    reinterpret_cast<uint32_t*>(T.lanes)[tid] = reinterpret_cast<const uint32_t*>(P.lanes)[tid];
  griddep_wait();
  // epoch = completed calls + 1, from the u64 CTA-exit count work[2..3] (+ G per call)
  const unsigned epoch = static_cast<unsigned>(
                             __ldcg(reinterpret_cast<const unsigned long long*>(P.work + 2)) /
                             static_cast<unsigned long long>(G)) +
                         1u,
                 par = epoch & 1u;
  if (b == 0) {  // the next launch's parity set (the previous launch, its user, has completed)
    for (int i = tid; i < MAXP; i += THREADS) P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u;
    for (int i = tid; i < MAXP + 32; i += THREADS) P.rdy[(par ^ 1u) * (MAXP + 32) + i] = 0u;
    for (int i = tid; i < MAXM; i += THREADS) P.xflag[(par ^ 1u) * MAXM + i] = 0u;
    if (tid < NRT) P.rt_cnt[(par ^ 1u) * NRT + tid] = 0u;
    if (tid == NRT + 1) {
      P.fc2_next[par ^ 1u] = 0u;
      P.fc2_next[2 + (par ^ 1u)] = 0u;
      P.gcnt[par ^ 1u] = 0u;
    }
  }
  // ---- routing (fused gate) + MXFP8 quantization of x
  const bool small = TPMOE_GATE && M <= 8;
  int* gids = reinterpret_cast<int*>(smem + PRO_GIDS);
  if (TPMOE_GATE) {
    if (small) {  // every CTA gates every token (warp w: tokens w, w + 8, ..); CTA 0 also publishes
                  // them
      for (int t = warp; t < M; t += THREADS / 32) {
        int id;
        float w;
        gate_token(P, P.glog + static_cast<size_t>(t) * NEXP, lane, id, w);
        if (lane < 6) {
          gids[t * 6 + lane] = id;
          T.srw[t * 6 + lane] = w;
          T.sids[t * 6 + lane] = static_cast<int16_t>(id);
          if (b == 0) {
            P.ids[t * 6 + lane] = id;
            P.rw[t * 6 + lane] = w;
          }
        }
      }
    } else {  // token t by CTA t % G (warp (t / G) % 8), counted on gcnt[par]
      unsigned* c = P.gcnt + par;
      {  // (warps 0..7 together per token; warps 8..11 go on to the x quantization)
        uint64_t* cand = reinterpret_cast<uint64_t*>(smem + PRO_IDS);
        float* sca = reinterpret_cast<float*>(smem + PRO_IDS + 8 * GATE_TOPK * 8);
        if (warp < 8)
          for (int t = b; t < M; t += G) {
            int id = 0;
            float w = 0.f;
            gate_token_cta(P, P.glog + static_cast<size_t>(t) * NEXP, warp, lane, cand, sca, id, w);
            if (warp == 0) {
              if (lane < 6) {
                P.ids[t * 6 + lane] = id;
                P.rw[t * 6 + lane] = w;
              }
              __syncwarp();
              if (lane == 0) {
                fence_acq_rel();
                atomicAdd(c, 1u);
              }
            }
            named_bar(6, 256);  // (cand / sca reused by the next token)
          }
      }
    }
  }
  // x -> MXFP8 (+ shared per-128) by token, independent of the routing: right after (and on other
  // warps than) the gate
  quant_x(P, P.xflag + par * MAXM, lane, (THREADS / 32 - 1 - warp) * G + b, (THREADS / 32) * G);
  if (TPMOE_GATE && !small) {
    if (tid == 0) wait_ge(P.gcnt + par, static_cast<unsigned>(M));
    __syncthreads();
  }
  __syncthreads();
  unsigned xv[16];  // (warp 6) every token's x flag, loaded here: in flight over the tables,
                    // checked after the role split
  if (warp == 6) {
#pragma unroll
    for (int i = 0; i < 16; ++i)
      xv[i] = lane + 32 * i < M ? ld_acquire(P.xflag + par * MAXM + lane + 32 * i) : 8u;
  }
  if (TPMOE_TABW && small && M <= 1) {
    if (warp == 0) build_tables_warp(P, smem, T, gids, lane);
    __syncthreads();
  } else
    build_tables(P, smem, T, small ? gids : nullptr);
  const int Ptot = T.misc[2];
  const int Psh =
      (M + NMAX - 1) / NMAX;  // shared panels (FC1 items / FC2 units after the routed ones)
  if (tid == 33) {
    int* q = reinterpret_cast<int*>(smem + OFF_TAB + TAB_T1Q);
    q[T1Q] = 0;
    q[T1Q + 1] = 0;
    q[T1Q + 2] = 0;
    T.misc[26] = 0;
  }
  if (tid == 0) T.misc[4] = Psh;
  const int nshq = Psh * NT1S * (shgu ? 2 : 1);
  const int R1 = Ptot * NT1 * NSTG1, S1 = Psh * NT1S * NSTG1S, totc = R1 + 2 * S1;
  // FC2 unit width: the widest of 8 / 4 / 2 / 1 row tiles that still gives every CTA >= 2 FC2 units
  const int Pall = Ptot + Psh;
  const int fc2g0 = (Pall * (NRT2 / 4) >= 2 * G ? 4 : (Pall * (NRT2 / 2) >= 2 * G ? 2 : 1));
  const int fc2g = min(fc2g0, M >= 64 ? 4 : 8), nrg = NRT2 / fc2g;  // (4-row-tile units at M >= 64)
  // FC2 descriptor (kinds 2 / 4): d.y = panel NRT2 + first row tile, d.z = row tiles
  auto f2p = [&](const int4& d) -> int { return d.y / NRT2; };
  auto f2r0 = [&](const int4& d) -> int { return d.y % NRT2; };
  auto f2n = [&](const int4& d) -> int { return d.z; };
  // tokens of a work descriptor's panel (kinds: 1 routed FC1 piece, 2 routed FC2 unit, 3 shared FC1
  // piece, 4 shared FC2 unit)
  auto panel_n = [&](const int4& d) -> int {
    int e, n, eb;
    if (d.x == 1) {
      panel_of(T, d.y / NT1, e, n, eb);
      return n;
    }
    if (d.x == 2) {
      panel_of(T, f2p(d), e, n, eb);
      return n;
    }
    if (d.x == 3) return min(NMAX, M - NMAX * (d.y / NT1S));
    return min(NMAX, M - NMAX * (f2p(d) - Ptot));
  };
  fence_proxy_async_smem();  // prologue scratch (generic) precedes the ring's async-proxy writes
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = static_cast<uint32_t>(T.misc[3]);
  int4* ring =
      T.ring;  // (written before the slot's WF arrive, read after its wait: the mbarrier orders it)

  if (warp == 0) {
    // ======================================================== producer: work descriptors on the
    // ring + this CTA's weight tile stages (A) and their SFA images
    if (lane == 0) {
      int fills = 0, w = 0;
      auto push = [&](int4 d) {  // -> the ring
        const int slot = w % WR;
        if (w >= WR) mbar_wait(bar(B_WE + slot), ((w / WR) - 1) & 1);
        ring[slot] = d;
        mbar_arrive(bar(B_WF + slot));
        ++w;
      };
      auto stage = [&](const CUtensorMap* map, int row, int ch0, const uint8_t* sfa,
                       uint32_t atx) {  // KC chunks of one 128-row tile
        const int st = fills % NS;
        if (fills >= NS) {
          mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
        }
        mbar_expect_tx(bar(B_FULL + st), atx + (KC * 512));
        tma3d(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, bar(B_FULL + st), 0x12F0000000000000ull);
        bulk_g2s(sb + OFF_SFA + st * KC * 512, sfa, KC * 512, bar(B_FULL + st));
        ++fills;
      };
      auto stage_s1 = [&](int k, int c) {  // shared FC1: chunk c of the gate tile (rows 128 k ..)
                                           // and the up tile (rows INTER + 128 k ..)
        if (fills < npre) {
          ++fills;
          return;
        }  // (issued before the PDL wait)
        const int st = fills % NS;
        if (fills >= NS) {
          mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
        }
        if (shh >= 0) {  // (SHGU: stage c = chunks 2 c, 2 c + 1 of this CTA's gate or up rows +
                         // their scales)
          mbar_expect_tx(bar(B_FULL + st), 2 * 16384 + 2 * 512);
          for (int h2 = 0; h2 < 2; ++h2) {
            tma3d(sb + OFF_A + st * A_BYTES + h2 * 16384, &P.map1s, 0, shh * INTER + 128 * k,
                  2 * c + h2, bar(B_FULL + st), SH_HINT);
            bulk_g2s(
                sb + OFF_SFA + st * KC * 512 + h2 * 512,
                P.sfa1s + (static_cast<size_t>(shh * (INTER / 128) + k) * NCH1 + 2 * c + h2) * 512,
                512, bar(B_FULL + st));
          }
          ++fills;
          return;
        }
        mbar_expect_tx(bar(B_FULL + st), (2 * 16384) + 2 * 512);
        tma3d(sb + OFF_A + st * A_BYTES, &P.map1s, 0, 128 * k, c, bar(B_FULL + st), SH_HINT);
        tma3d(sb + OFF_A + st * A_BYTES + 16384, &P.map1s, 0, INTER + 128 * k, c, bar(B_FULL + st),
              SH_HINT);
        bulk_g2s(sb + OFF_SFA + st * KC * 512, P.sfa1s + (static_cast<size_t>(k) * NCH1 + c) * 512,
                 512, bar(B_FULL + st));
        bulk_g2s(sb + OFF_SFA + st * KC * 512 + 512,
                 P.sfa1s + (static_cast<size_t>(INTER / 128 + k) * NCH1 + c) * 512, 512,
                 bar(B_FULL + st));
        ++fills;
      };
      auto issue = [&](const int4& d) {  // this CTA's stages of a descriptor
        if (d.x == 1) {
          const int item = d.y, tile = item % NT1;
          int e, n, eb;
          panel_of(T, item / NT1, e, n, eb);
          for (int s = d.z; s < d.w; ++s)
            stage(&P.map1, e * (2 * INTER) + tile * 128, KC * s,
                  P.w13_sf + ((static_cast<size_t>(e) * NT1 + tile) * NCH1 + KC * s) * 512,
                  KC * 8192);
        } else if (d.x == 3) {
          const int k = d.y % NT1S;
          for (int s = d.z; s < d.w; ++s) stage_s1(k, s);
        } else {
          const int p = f2p(d), r0 = f2r0(d), cn = f2n(d);
          if (p < Ptot) {
            int e, n, eb;
            panel_of(T, p, e, n, eb);
            for (int j = 0; j < cn; ++j) {
              const int r = (r0 + j);
              for (int s = 0; s < NSTG2; ++s)
                stage(&P.map2, e * DIM + r * 128, KC * s,
                      P.w2_sf + ((static_cast<size_t>(e) * NRT2 + r) * NCH2 + KC * s) * 512,
                      KC * 8192);
            }
          } else {
            for (int j = 0; j < cn; ++j) {
              const int r = (r0 + j);
              for (int s = 0; s < NSTG2; ++s)
                stage(&P.map2s, r * 128, KC * s,
                      P.sfa2s + (static_cast<size_t>(r) * NCH2 + KC * s) * 512, KC * 16384);
            }
          }
        }

      };
      {
        auto put = [&](int4 d) {
          push(d);
          issue(d);
        };
        // FC1 work: CTAs < nshq hold a whole shared item for swv waves of routed items
        constexpr int swv = 3;
        auto put_r = [&](int r) { put(make_int4(1, r, 0, NSTG1)); };
        {  // shared item b first; routed entry r: waves < swv over CTAs nshq .., then every CTA
           // round robin
          if (b < nshq) put(make_int4(3, shgu ? b >> 1 : b, 0, shgu ? NSTG1S / 2 : NSTG1S));
          const int nr = Ptot * NT1, gs = G - nshq, head = swv * gs;
          {  // (waves < swv on CTAs >= nshq, then round robin from head)
            const int nw0 = b >= nshq ? swv : 0;
            for (int k = 0;; ++k) {
              const int r = k < nw0 ? k * gs + (b - nshq) : head + b + (k - nw0) * G;
              if (r >= nr) {
                if (k < nw0) continue;
                break;
              }
              put_r(r);
            }
          }
        }

        // (row tiles R in slot-major order; [0, RA) in units of fc2g, [RA, RB) pairs, [RB, RT)
        // singles)
        const int RT = Pall * NRT2;
        int RA = RT, RB = RT;
        if (fc2g >= 4) {
          RB = max(0, RT - 2 * ((G + 1) / 2));
          RA = RB >= 2 * G ? ((RB - 2 * G) / fc2g) * fc2g : 0;
        } else if (fc2g == 2 && fc2g0 >= 4) {
          RB = max(0, RT - 2 * ((G + 1) / 2));
          RA = RB;
        }  // (pairs, then singles)
        const int nA = RA / fc2g, nB = (RB - RA) / 2;
        auto rmap = [&](int R, int& row) -> int {  // R -> slot, row tile
          row = R % NRT2;
          return R / NRT2;
        };
        auto tk = [&](int u, int& R0) -> int {
          if (u < nA) {
            R0 = u * fc2g;
            return fc2g;
          }
          if (u < nA + nB) {
            R0 = RA + 2 * (u - nA);
            return 2;
          }
          R0 = RB + (u - nA - nB);
          return 1;
        };
        auto tslot = [&](int u) -> int {
          int R0, row;
          tk(u, R0);
          return rmap(R0, row);
        };
        const int units = nA + nB + (RT - RB);
        // FC2 units in the order panels become ready: ticket u -> ready-list slot u / nrg (the
        // panel whose h completed (u / nrg)-th), row group u % nrg.  One ticket ahead, its slot
        // loaded as soon as the ticket arrives (acquire: the panel's h)
        const unsigned* rl = P.rdy + par * (MAXP + 32);
        auto ldf = [&](const unsigned* a) -> unsigned { return ld_acquire(a); };
        int tq = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
        unsigned pf = tq < units ? ldf(rl + tslot(tq)) : 0u;
        const int fcap =
            (units + G - 1) / G + TPMOE_FC2CAP - 1;  // (TPMOE_FC2CAP: tickets this CTA may claim)
        int nclaim = 0;
        for (;;) {
          const int u = tq;
          if (u >= units) break;
          const int rg = u % nrg;
          if (f2s) {  // (static order: slot = panel; the B warp waits for the panel's h)
            const int sl = tslot(u);
            pf = static_cast<unsigned>(sl + 1);
          }
          if (pf == 0u) {
            while ((pf = ldf(rl + tslot(u))) == 0u) {
            }
          }
          const int p = static_cast<int>(pf) - 1;
          int4 d = make_int4(p < Ptot ? 2 : 4, p * nrg + rg, 0, 0);
          {
            int R0, row;
            const int cn = tk(u, R0);
            rmap(R0, row);
            d.y = p * NRT2 + row;
            d.z = cn;
          }
          push(d);
          tq = (TPMOE_FC2CAP && ++nclaim >= fcap)
                   ? units
                   : static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
          pf = (tq < units && !f2s) ? ld_acquire(rl + tslot(tq)) : 0u;
          issue(d);
        }
        push(make_int4(-1, 0, 0, 0));
      }
    }
    __syncwarp();
  } else if (warp == 7) {
    // ======================================================== warp 7: FC2-unit publisher
    if (lane == 0) {  // FC2 units queued by the epilogue -> per-256-row-tile counts (rt_cnt) that
                      // release the T1
      volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
      int c = 0;
      for (;;) {
        const int n = q[T1Q];
        if (c == n) {
          if (q[T1Q + 1] && c == q[T1Q]) break;  // (epilogue done, everything published)
          __nanosleep(64);  // (warp 7 shares an SM sub-partition with epilogue warps 3 / 11)
          continue;
        }
        constexpr bool T1CNT = NRT == 16 && NRT2 == 32 &&
                               T1Q <= 127;  // (<= 2 per tile per unit: bytes hold T1Q units)
        unsigned rn[NRT];
        uint64_t clo = 0ull, chi = 0ull;
#pragma unroll
        for (int i = 0; i < NRT; ++i) rn[i] = 0u;
        for (; c < n; ++c) {
          const int qv = q[c % T1Q];
          const int r0 = (qv & 0xFFFFFF) % NRT2, cn = (qv >> 24);
          if (T1CNT) {  // row tiles r0 .. r0 + cn - 1 -> 2-bit counts per 256-row tile -> byte
                        // fields (tiles 0-7 | 8-15)
            const uint32_t m = ((1u << cn) - 1u) << r0,
                           c2 = (m & 0x55555555u) + ((m >> 1) & 0x55555555u);
            auto spread = [](uint32_t x16) {
              uint64_t x = x16;
              x = (x | (x << 24)) & 0x000000FF000000FFull;
              x = (x | (x << 12)) & 0x000F000F000F000Full;
              return (x | (x << 6)) & 0x0303030303030303ull;
            };
            clo += spread(c2 & 0xFFFFu);
            chi += spread(c2 >> 16);
            continue;
          }
          int ij[8];  // T1 tile (256 rows) of each of the unit's row tiles
#pragma unroll
          for (int j = 0; j < 8; ++j) ij[j] = j < cn ? ((r0 + j) >> 1) : -1;
#pragma unroll
          for (int i = 0; i < NRT; ++i) {
            unsigned a = 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) a += ij[j] == i ? 1u : 0u;
            rn[i] += a;
          }
        }
        q[T1Q + 2] = c;
        fence_acq_rel();  // (the queued units' outp / shout rows: observed through the epilogue's
                          // barrier + queue entry)
#pragma unroll
        for (int i = 0; i < NRT; ++i) {
          const unsigned r =
              T1CNT ? static_cast<unsigned>(((i < 8 ? clo : chi) >> (8 * (i & 7))) & 0xFFull)
                    : rn[i];
          if (r) red_relaxed_add(P.rt_cnt + par * NRT + i, r);
        }
      }
    }
    __syncwarp();
  } else if (warp == 6) {
    // ======================================================== B operand by bulk copies: FC1 x
    // panel images / FC2 h panel images
    int fills = 0, w = 0;
    int rfi = 0;  // (TPMOE_SFBI) routed FC1 items seen: item buffer rfi & 1
    bool x_ready = false;
    constexpr int NPR = 8;
    for (;;) {
      const int slot = w % WR;
      mbar_wait(bar(B_WF + slot), (w / WR) & 1);
      const int4 d = ring[slot];
      use4(d);
      __syncwarp();
      if (lane == 0) we_arrive(slot);
      ++w;
      if (d.x < 0) break;
      if (d.x == 1 || d.x == 3) {
        if (!x_ready) {  // the flags loaded before the tables (acquire): usually every token is in
                         // already
          bool all = true;
#pragma unroll
          for (int i = 0; i < 16; ++i) all &= xv[i] >= 8u;
          if (__all_sync(0xffffffffu, all)) {
            if (lane == 0)
              fence_proxy_async_global();  // generic writes -> TMA gathers (async proxy)
            x_ready = true;
          }
        }
        if (!x_ready) {  // every token quantized: all of a lane's polls in flight together
          const unsigned* xc = P.xflag + par * MAXM;
          for (;;) {
            bool all = true;
            for (int t0 = lane; t0 < M; t0 += 32 * 16) {
              unsigned v[16];
#pragma unroll
              for (int i = 0; i < 16; ++i)
                v[i] = t0 + 32 * i < M ? ld_relaxed(xc + t0 + 32 * i) : 8u;
#pragma unroll
              for (int i = 0; i < 16; ++i) all &= v[i] >= 8u;
            }
            if (__all_sync(0xffffffffu, all)) break;
          }
          fence_acq_rel();
          if (lane == 0) fence_proxy_async_global();  // generic writes -> bulk copies (async proxy)
          x_ready = true;
        }
        if (d.x == 1) {  // routed: the panel's token rows gathered from xq / xsf2 (lane g < npad /
                         // 4: rows 4 g .. 4 g + 3)
          const int p = d.y / NT1;
          int e, n, eb;
          panel_of(T, p, e, n, eb);
          const int npad = (n + NPR - 1) & ~(NPR - 1), nb = npad, rb = 0, ng = nb >> 2;
          // lane r < npad: token of panel row r (padding rows repeat the last); lane g < npad / 4
          // gathers rows 4 g .. 4 g + 3.  The SFB rows go by one 16-B bulk copy per lane (a gather4
          // of 16-B rows is 64 B: TMA destinations need 128-B alignment)
          const int tok = lane < npad ? static_cast<int>(T.ent[eb + min(lane, n - 1)] & 511u) : 0;
          int tr[4];
#pragma unroll
          for (int i = 0; i < 4; ++i)
            tr[i] = __shfl_sync(0xffffffffu, tok, (rb + 4 * lane + i) & 31);
          // SFB rows: generic 16-B loads two stages ahead, stored into the stage's SFB image before
          // the stage's arrive (+ proxy fence)
          const uint4* srow =
              reinterpret_cast<const uint4*>(P.xsf2 + static_cast<size_t>(tok) * NSTG1 * 16);
          uint4 q0 = make_uint4(0u, 0u, 0u, 0u), q1 = q0;
          constexpr int SB = 4;  // (SFBI) stages per SFB batch: one proxy fence per batch, the next
                                 // batch's loads in flight
          uint4 qb[SB];
          const uint32_t ib = static_cast<uint32_t>(OFF_SFBI + (rfi & 1) * (NSTG1 * 512));
          if (TPMOE_SFBI) {
            ++rfi;
#pragma unroll
            for (int i = 0; i < SB; ++i)
              qb[i] = (lane < npad && d.z + i < d.w) ? __ldcg(srow + d.z + i)
                                                     : make_uint4(0u, 0u, 0u, 0u);
          } else if (lane < npad) {
            q0 = __ldcg(srow + d.z);
            if (d.z + 1 < d.w) q1 = __ldcg(srow + d.z + 1);
          }
          for (int s = d.z; s < d.w; ++s) {
            if (TPMOE_SFBI &&
                ((s - d.z) % SB) ==
                    0) {  // batch s .. s + SB - 1 -> the item buffer, one fence; next batch's loads
              if (lane < npad) {
#pragma unroll
                for (int i = 0; i < SB; ++i)
                  if (s + i < d.w)
                    *reinterpret_cast<uint4*>(smem + ib + (s + i) * 512 + lane * 16) = qb[i];
                fence_proxy_async_smem();
              }
#pragma unroll
              for (int i = 0; i < SB; ++i)
                qb[i] = (lane < npad && s + SB + i < d.w) ? __ldcg(srow + s + SB + i)
                                                          : make_uint4(0u, 0u, 0u, 0u);
              __syncwarp();
            }
            const int st = fills % NS;
            if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
            if (!TPMOE_SFBI) {
              const uint4 cur = q0;
              q0 = q1;
              if (lane < npad && s + 2 < d.w) q1 = __ldcg(srow + s + 2);
              if (lane < npad) {
                *reinterpret_cast<uint4*>(smem + OFF_SFB + st * 512 + lane * 16) = cur;
                fence_proxy_async_smem();
              }
            }
            __syncwarp();
            if (lane == 0) mbar_expect_tx(bar(B_FULL + st), KC * nb * 128);
            if (lane < ng) {
#pragma unroll
              for (int c = 0; c < KC; ++c)
                tma_gather4(sb + OFF_B + st * B_BYTES + c * nb * 128 + lane * 512, &P.mapxq,
                            (KC * s + c) * 128, tr[0], tr[1], tr[2], tr[3], bar(B_FULL + st));
            }
            ++fills;
          }
        } else if (lane == 0) {
          {  // shared: chunk s of the shared panel's per-128 x image per stage (SFB image of the
             // chunk pair)
            const int sp = d.y / NT1S, n = min(NMAX, M - NMAX * sp),
                      npad = (n + NPR - 1) & ~(NPR - 1);
            const int nb = npad, rb = 0;
            for (int s = d.z; s < d.w; ++s) {
              const int st = fills % NS;
              if (fills >= NS) mbar_wait(bar(B_EMPTY + st), ((fills / NS) - 1) & 1);
              if (shh >= 0) {  // (SHGU: x chunks 2 s, 2 s + 1 and their SFB image s)
                mbar_expect_tx(bar(B_FULL + st), 2 * nb * 128 + 512);
                for (int h2 = 0; h2 < 2; ++h2)
                  bulk_g2s(
                      sb + OFF_B + st * B_BYTES + h2 * nb * 128,
                      P.xgs + ((static_cast<size_t>(sp) * NCH1 + 2 * s + h2) * NMAX + rb) * 128,
                      nb * 128, bar(B_FULL + st));
                bulk_g2s(sb + OFF_SFB + st * 512,
                         P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + s) * 512, 512,
                         bar(B_FULL + st));
                ++fills;
                continue;
              }
              mbar_expect_tx(bar(B_FULL + st), nb * 128 + 512);
              bulk_g2s(sb + OFF_B + st * B_BYTES,
                       P.xgs + ((static_cast<size_t>(sp) * NCH1 + s) * NMAX + rb) * 128, nb * 128,
                       bar(B_FULL + st));
              bulk_g2s(sb + OFF_SFB + st * 512,
                       P.xgsfs + (static_cast<size_t>(sp) * NSTG1 + (s >> 1)) * 512, 512,
                       bar(B_FULL + st));
              ++fills;
            }
          }
        }
        fills = warp_uniform(static_cast<uint32_t>(fills));
      } else {
        const int p = f2p(d), cn2 = f2n(d);
        int n;
        const uint8_t* himg;
        const uint8_t* hsf;
        if (d.x == 2) {
          int e, eb;
          panel_of(T, p, e, n, eb);
          himg = P.himg + static_cast<size_t>(p) * NCH2 * NMAX * 128;
          hsf = P.hsf + static_cast<size_t>(p) * NSTG2 * 512;
        } else {
          const int sp = p - Ptot;
          n = min(NMAX, M - NMAX * sp);
          himg = P.hsimg + static_cast<size_t>(sp) * NCH2 * NMAX * 128;
          hsf = P.hssf + static_cast<size_t>(sp) * NSTG2 * 512;
        }
        const int npad = (n + NPR - 1) & ~(NPR - 1), nb = npad, rb = 0;
        if (lane == 0) {
          if (f2s) {  // static order: wait until all of the panel's FC1 items are in (acquire)
            wait_ge(P.pan_cnt + par * MAXP + p,
                    d.x == 2 ? static_cast<unsigned>(NT1) : static_cast<unsigned>(NT1S));
          }
          fence_proxy_async_global();  // (generic h writes -> bulk copies; ready order: the
                                       // producer acquired the panel's h flag)
          for (int f = fills; f < fills + cn2 * NSTG2;
               ++f) {  // the panel's h once per row tile (L2)
            const int st = f % NS, s = (f - fills) % NSTG2;
            if (f >= NS) mbar_wait(bar(B_EMPTY + st), ((f / NS) - 1) & 1);
            mbar_expect_tx(bar(B_FULL + st), KC * nb * 128 + 512);
#pragma unroll
            for (int c = 0; c < KC; ++c)
              bulk_g2s(sb + OFF_B + st * B_BYTES + c * nb * 128,
                       himg + (static_cast<size_t>(KC * s + c) * NMAX + rb) * 128, nb * 128,
                       bar(B_FULL + st));
            bulk_g2s(sb + OFF_SFB + st * 512, hsf + static_cast<size_t>(s) * 512, 512,
                     bar(B_FULL + st));
          }
        }
        __syncwarp();
        fills += cn2 * NSTG2;
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    // ======================================================== MMA issuer (converged warp, one
    // elected lane) tcgen05.mma operands must be warp-uniform (uniform registers): TMEM base,
    // descriptor fields via shuffles (else every MMA issue gets an R2UR loop)
    const uint32_t tmem_u = warp_uniform(tmem);
    // stage-0 operands, computed once: stage st adds st * stride
    const uint64_t adesc0 = desc_sw128(sb + OFF_A), bdesc0 = desc_sw128(sb + OFF_B),
                   sfadesc0 = desc_sf(sb + OFF_SFA), sfbdesc0 = desc_sf(sb + OFF_SFB);
    const uint32_t bfull0 = bar(B_FULL), bempty0 = bar(B_EMPTY), tsf0 = tmem_u + TSF0;
    int fills = 0, w = 0,
        kacc = 0;  // accumulators issued so far: buffer kacc % ACC, use kacc / ACC
    int rfi = 0;   // (TPMOE_SFBI) routed FC1 items seen: SFB item buffer rfi & 1
    const uint64_t sfbidesc0 = desc_sf(sb + OFF_SFBI);
    for (;;) {
      const int slot = w % WR;
      mbar_wait(bar(B_WF + slot), (w / WR) & 1);
      int4 d = ring[slot];
      d.x = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.x)));
      d.y = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.y)));
      d.z = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.z)));
      d.w = static_cast<int>(warp_uniform(static_cast<uint32_t>(d.w)));
      use4(d);
      __syncwarp();
      if (lane == 0) we_arrive(slot);
      ++w;
      if (d.x < 0) {
        break;
      }
      const bool sfbi =
          TPMOE_SFBI &&
          d.x == 1;  // (this routed FC1 item's SFB rows: item buffer rfi & 1, stage d.z + j)
      const uint64_t sfbi0 =
          sfbidesc0 + static_cast<uint64_t>((((rfi & 1) * NSTG1) + d.z) * (512 >> 4));
      if (sfbi) ++rfi;
      const int n = panel_n(d);
      const uint32_t npad = warp_uniform(static_cast<uint32_t>((n + 7) & ~7));
      const bool shk = d.x >= 3;  // shared expert: e4m3 A
      const uint32_t id0 = idesc_f4f8(static_cast<int>(npad), 0, shk ? 0u : 5u);
      const bool fc1 = (d.x & 1) != 0;
      const int nacc = fc1 ? 1 : f2n(d), nst = fc1 ? d.w - d.z : NSTG2;
      for (int a = 0; a < nacc; ++a) {  // FC1: one accumulator; FC2: one per row tile of the group
        const int ab = kacc % ACC;
        if (kacc >= ACC) {
          mbar_wait(bar(B_ACCE + ab), ((kacc / ACC) - 1) & 1);
        }
        tc_fence_after();
        const uint32_t dacc = tmem_u + ab * ACC_COLS;
        {  // per-item operands hoisted, stage / parity incremental
          int st = fills % NS;
          uint32_t ph = static_cast<uint32_t>(fills / NS) & 1u;
          const uint64_t bstep = static_cast<uint64_t>((npad * 128) >> 4);
          if (d.x == 3) {
            for (int j = 0; j < nst; ++j) {
              mbar_wait(bfull0 + 8 * st, ph);
              tc_fence_after();
              const uint32_t tsf = tsf0 + st * TSF_STRIDE;
              const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)),
                             bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
              if (elect_one()) {
#pragma unroll
                for (int c = 0; c < KC; ++c)
                  tc_cp_sf(tsf + 4 * c,
                           sfadesc0 + static_cast<uint64_t>((st * KC + c) * (512 >> 4)));
                tc_cp_sf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
                const uint32_t sfb = tsf + TSF_B + 2 * ((d.z + j) & 1);
                if (shh >= 0) {  // SHGU: chunks 2 j, 2 j + 1 of this CTA's chain (A halves, B
                                 // chunks, SFA / SFB entries c), K order
#pragma unroll
                  for (int c = 0; c < 2; ++c)
#pragma unroll
                    for (int kb = 0; kb < 4; ++kb)
                      tc_mma(dacc + 32 * shh, ad0 + c * (16384 >> 4) + 2 * kb,
                             bd0 + c * bstep + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c,
                             tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                } else
#pragma unroll
                  for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = (j | kb) != 0;
                    tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                    tc_mma(dacc + 32, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb,
                           acc);
                  }
                tc_commit(bempty0 + 8 * st);
              }
              __syncwarp();
              if (++st == NS) {
                st = 0;
                ph ^= 1u;
              }
            }
          } else if ((M > TPMOE_FC2PAIR_OFFM && !fc1 && nst == 2)) {
            for (int jp = 0; jp < nst; jp += 2) {
              int st1 = st + 1;
              uint32_t ph1 = ph;
              if (st1 == NS) {
                st1 = 0;
                ph1 ^= 1u;
              }
              mbar_wait(bfull0 + 8 * st, ph);
              mbar_wait(bfull0 + 8 * st1, ph1);
              tc_fence_after();
              if (elect_one()) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                  const int sx = h ? st1 : st;
                  const uint32_t tsf = tsf0 + sx * TSF_STRIDE;
                  const uint64_t ad0 = adesc0 + static_cast<uint64_t>(sx * (A_BYTES >> 4)),
                                 bd0 = bdesc0 + static_cast<uint64_t>(sx * (B_BYTES >> 4));
#pragma unroll
                  for (int c = 0; c < KC; ++c)
                    tc_cp_sf(tsf + 4 * c,
                             sfadesc0 + static_cast<uint64_t>((sx * KC + c) * (512 >> 4)));
                  tc_cp_sf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(sx * (512 >> 4)));
#pragma unroll
                  for (int c = 0; c < KC; ++c) {
                    const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * bstep;
#pragma unroll
                    for (int kb = 0; kb < 4; ++kb)
                      tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29),
                             tsf + 4 * c, tsf + TSF_B + 2 * c, (jp | h | c | kb) != 0);
                  }
                }
                tc_commit(bempty0 + 8 * st);
                tc_commit(bempty0 + 8 * st1);
              }
              __syncwarp();
              st = st1 + 1;
              ph = ph1;
              if (st == NS) {
                st = 0;
                ph ^= 1u;
              }
            }
          } else {
            for (int j = 0; j < nst; ++j) {
              mbar_wait(bfull0 + 8 * st, ph);
              tc_fence_after();
              const uint32_t tsf = tsf0 + st * TSF_STRIDE;
              const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)),
                             bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
              if (elect_one()) {
#pragma unroll
                for (int c = 0; c < KC; ++c)
                  tc_cp_sf(tsf + 4 * c,
                           sfadesc0 + static_cast<uint64_t>((st * KC + c) * (512 >> 4)));
                tc_cp_sf(tsf + TSF_B, sfbi ? sfbi0 + static_cast<uint64_t>(j * (512 >> 4))
                                           : sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
#pragma unroll
                for (int c = 0; c < KC; ++c) {
                  const uint64_t ac = ad0 + c * (16384 >> 4), bc = bd0 + c * bstep;
#pragma unroll
                  for (int kb = 0; kb < 4; ++kb)
                    tc_mma(dacc, ac + 2 * kb, bc + 2 * kb, id0 | (kb << 4) | (kb << 29),
                           tsf + 4 * c, tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                }
                tc_commit(bempty0 + 8 * st);
              }
              __syncwarp();
              if (++st == NS) {
                st = 0;
                ph ^= 1u;
              }
            }
          }
          fills += nst;
        }
        if (elect_one()) tc_commit(bar(B_ACCF + ab));
        __syncwarp();
        ++kacc;
      }
    }
  } else {
    // ======================================================== epilogue: warps 2..5 (group 0: token
    // columns 0..15) and 8..11 (group 1: 16..31); TMEM lanes 32 q .. 32 q + 31, q = warp & 3. Named
    // barriers: 1 + g (the group's 128 threads), 3 (both groups)
    const int q = warp & 3, L = 32 * q + lane, g = warp >= 8 ? 1 : 0, tb = 16 * g,
              gt = tid - (g ? 256 : 64);
    const bool up_lane = (lane & 8) == 0;
    const bool pub = warp == 2 && lane == 0;  // the publishing thread
    const int frow = T.lanes[L];

    __nv_bfloat16* stg = T.stg + g * 16 * 128;  // FC2 transpose staging [16 tokens][128 rows]
    int w = 0, kacc = 0;
    unsigned
        rt_n[NRT];  // (pub) FC2 row tiles stored per 256-row T1 tile, published once at the end
#pragma unroll
    for (int i = 0; i < NRT; ++i) rt_n[i] = 0u;
    int dry = 0;  // dry passes left (warm the epilogue code before the first real item): 2 = routed
                  // FC1, 1 = FC2
    for (;;) {
      int4 d;
      const bool isdry = dry != 0;
      if (isdry) {
        d = dry == 2 ? make_int4(1, 0, 0, NSTG1) : make_int4(2, 0, 1, 0);  // (d.z = 1 row tile)
        --dry;
      } else {
        const int slot = w % WR;
        mbar_wait(bar(B_WF + slot), (w / WR) & 1);
        d = ring[slot];
        use4(d);
        __syncwarp();
        if (lane == 0) we_arrive(slot);
        ++w;
      }
      if (d.x < 0) {
        if (pub) {
          __threadfence_block();
          reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q)[T1Q + 1] = 1;
        }
        break;
      }
      int p, e = 0, n, eb = 0;  // routed: panel; shared: shared panel sp (tokens 32 sp ..)
      if (d.x == 1) {
        p = d.y / NT1;
        panel_of(T, p, e, n, eb);
      } else if (d.x == 2) {
        p = f2p(d);
        panel_of(T, p, e, n, eb);
      } else if (d.x == 3) {
        p = d.y / NT1S;
        n = min(NMAX, M - NMAX * p);
      } else {
        p = f2p(d) - Ptot;
        n = min(NMAX, M - NMAX * p);
      }
      const int npad = (n + 7) & ~7;
      const bool fc1 = (d.x & 1) != 0;
      const int nacc = fc1 ? 1 : f2n(d);
      for (int a = 0; a < nacc; ++a) {
        const int ab = kacc % ACC;
        if (!isdry) mbar_wait(bar(B_ACCF + ab), (kacc / ACC) & 1);
        tc_fence_after();
        float v[16], vu[16];
        const uint32_t ta = tmem + (static_cast<uint32_t>(32 * q) << 16) + ab * ACC_COLS + tb;
        tc_ld16_wait(ta, v);                      // (columns past npad: unused garbage)
        if (d.x == 3) tc_ld16_wait(ta + 32, vu);  // shared FC1: the up tile's accumulator
        tc_fence_before();
        __syncwarp();
        if (!isdry) {
          if (lane == 0) {
            mbar_arrive(bar(B_ACCE + ab));
          }
          ++kacc;
        }
        if (!fc1) {
          // ---- FC2 row tile r: transpose through smem, 16-B rows of bf16 per (token, slot) /
          // shared: per token of shout
          const int r = (f2r0(d) + a);
          const int orow =
              d.x == 4 ? L : frow;  // (the shared down weights are used in checkpoint row order)
#pragma unroll
          for (int i = 0; i < 16; ++i)
            if (tb + i < n) stg[i * 128 + orow] = __float2bfloat16_rn(v[i]);
          named_bar(1 + g, 128);
          for (int x = gt; x < 16 * 16; x += 128) {  // (token i, 16-B chunk c)
            const int i = x >> 4, c = x & 15;
            if (tb + i >= n) break;
            const uint4 val = *reinterpret_cast<const uint4*>(stg + i * 128 + 8 * c);
            if (isdry) continue;
            if (d.x == 4) {
              st_v4_hint(P.shout + static_cast<size_t>(NMAX * p + tb + i) * DIM + r * 128 + 8 * c,
                         val);
              continue;
            }
            const unsigned ev = T.ent[eb + tb + i];
            const int t = static_cast<int>(ev & 511u);
            unsigned km = ev >> 9;
            while (km) {
              const int k = __ffs(km) - 1;
              km &= km - 1;
              st_v4_hint(
                  P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * OUTP_LD + r * 128 + 8 * c,
                  val);
            }
          }
          named_bar(1 + g, 128);  // (stg reused by the next row tile)
          continue;
        }
        // ---- FC1 piece of item d.y, stages [d.z, d.w)
        const int item = d.y;
        if (d.x == 3 && shh >= 0 && !isdry) {  // ---- SHGU: the up CTA hands its tile to the gate
                                               // CTA as (epoch | fp32) words
          unsigned long long* pq0 = reinterpret_cast<unsigned long long*>(P.part);
          const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32;
          if (shh == 1) {
            unsigned long long* pq = pq0 + static_cast<size_t>(b) * NMAX * 128;
#pragma unroll
            for (int i = 0; i < 16; ++i)
              if (tb + i < n) st_relaxed64(pq + (tb + i) * 128 + L, tag | __float_as_uint(vu[i]));
            continue;
          }
          const unsigned long long* pc = pq0 + static_cast<size_t>((b + 1)) * NMAX * 128;
          for (;;) {
            bool ok = true;
#pragma unroll
            for (int i = 0; i < 16; ++i)
              if (tb + i < n) {
                const unsigned long long w = ld_relaxed64(pc + (tb + i) * 128 + L);
                ok &= static_cast<unsigned>(w >> 32) == epoch;
                vu[i] = __uint_as_float(static_cast<unsigned>(w));
              }
            if (__all_sync(0xffffffffu, ok)) break;
            __nanosleep(200);
          }
        }
        if (d.x == 3) {
          // ---- shared FC1 item (128 h columns = lanes): DeepGEMM bf16 gate / up,
          // silu_and_mul_clamp (bf16 clamp, fp32 silu * up, bf16), per-128 FP8 (ceil rule, eps
          // 1e-10, ue8m0) -> the shared h image + SFB (4 equal bytes)
          const int ng = min(TPMOE_ACTU_ON ? min(16, TPMOE_MMAX) : 16, n - tb), kq = item % NT1S;
          float am_mine = 0.f;
          float r[16];  // (unprocessed gate values at the front, finished activations appended at
                        // the back)
#pragma unroll
          for (int i = 0; i < 16; ++i) r[i] = v[i];
#if TPMOE_ACTU_ON
#pragma unroll
          for (int i = 0; i < 16;
               ++i) {  // (r[i] = token i's activation, as the rotation leaves it)
            if (i >= ng) break;
            const float gq = fminf(bf16r(r[i]), 10.0f),
                        uq = fminf(fmaxf(bf16r(vu[i]), -10.0f), 10.0f);
            const float a2 = bf16r((gq / (1.0f + __expf(-gq))) * uq);
            float m = fabsf(a2);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            am_mine = lane == i ? m : am_mine;
            r[i] = a2;
          }
#else
#pragma unroll 1
          for (int i = 0; i < ng; ++i) {
            const float gq = fminf(bf16r(r[0]), 10.0f),
                        uq = fminf(fmaxf(bf16r(vu[0]), -10.0f), 10.0f);
            const float a2 = bf16r((gq / (1.0f + __expf(-gq))) * uq);
            float m = fabsf(a2);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            am_mine = lane == i ? m : am_mine;
#pragma unroll
            for (int k = 0; k < 15; ++k) {
              r[k] = r[k + 1];
              vu[k] = vu[k + 1];
            }
            r[15] = a2;
          }
#pragma unroll 1
          for (int k2 = ng; k2 < 16; ++k2) {  // activations to the front (token i at r[i])
            const float t0 = r[0];
#pragma unroll
            for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
            r[15] = t0;
          }
#endif
          if (lane < 16 && tb + lane < NMAX) T.am[q * NMAX + tb + lane] = am_mine;
          named_bar(1 + g, 128);
          uint8_t* hb = P.hsimg + (static_cast<size_t>(p) * NCH2 + kq) * NMAX * 128;
#if TPMOE_ACTU_ON
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            if (i >= ng) break;
            const float a = r[i];
#else
#pragma unroll 1
          for (int i = 0; i < ng; ++i) {
            const float a = r[0];
#pragma unroll
            for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
#endif
            const int tok = tb + i;
            const float amax = fmaxf(fmaxf(T.am[tok], T.am[NMAX + tok]),
                                     fmaxf(T.am[2 * NMAX + tok], T.am[3 * NMAX + tok]));
            const int ex = ue8m0_exp(fmaxf(amax, 1e-10f));
            const uint32_t byte = e4m3_of(a * exp2i(-ex));
            uint32_t word = 0;  // column L = 32 q + lane: granule 2 q + lane / 16, byte lane % 16;
                                // lanes (lane & 15) < 4 gather word lane & 3
#pragma unroll
            for (int bb = 0; bb < 4; ++bb)
              word |= (__shfl_sync(0xffffffffu, byte, (lane & 16) + 4 * (lane & 3) + bb) & 0xFFu)
                      << (8 * bb);
            if ((lane & 15) < 4)
              *reinterpret_cast<uint32_t*>(hb + (tok >> 3) * 1024 + (tok & 7) * 128 +
                                           (((2 * q + (lane >> 4)) ^ (tok & 7)) << 4) +
                                           4 * (lane & 3)) = word;
            if (q == 0 && lane == 0)
              *reinterpret_cast<uint32_t*>(
                  P.hssf + (static_cast<size_t>(p) * NSTG2 + (kq >> 1)) * 512 + tok * 16 +
                  8 * (kq & 1)) = static_cast<uint32_t>(ex + 127) * 0x01010101u;
          }
          named_bar(3, 256);
          if (pub && TPMOE_PUBRED && f2s) {
            red_release_add(P.pan_cnt + par * MAXP + Ptot + p, 1u);
          } else if (pub) {
            const bool lastit =
                (atom_add_acq_rel(P.pan_cnt + par * MAXP + Ptot + p, 1u)) == NT1S - 1u;
            if (lastit) {  // the panel's last FC1 item: append it
              unsigned* rl = P.rdy + par * (MAXP + 32);
              st_release(rl + atomicAdd(rl + MAXP, 1u), static_cast<unsigned>(Ptot + p + 1));
            }
          }
          continue;
        }
        // SwiGLU: the gate of an up lane's h column sits in lane ^ 8 of the same warp.  Compact
        // loops over the live tokens on a rotating register copy (the code runs i-cache cold on the
        // critical path: its size, not its count, costs)
        const int ng = min(TPMOE_ACTU_ON ? min(16, TPMOE_MMAX) : 16,
                           n - tb);  // this group's live tokens (warp-uniform; <= 0: none)
        float am_mine = 0.f;         // lane i: the warp's amax of token tb + i
        const int tile = item % NT1,
                  q2 = tile >> 1;  // FC2 k-chunk of these 64 h columns: 64 (tile & 1) + 16 q + ..
        uint8_t* himg_p = P.himg + static_cast<size_t>(p) * NCH2 * NMAX * 128 +
                          static_cast<size_t>(q2) * NMAX * 128;
        const int gran =
            4 * (tile & 1) + q;  // 16-B granule of this warp's 16 columns in the 128-B row
#if TPMOE_ACTX
        float r[16];  // unprocessed values at the front, finished activations appended at the back
#pragma unroll
        for (int i = 0; i < 16; ++i) r[i] = v[i];
        const int nr4 = (ng + 3) >> 2;
#pragma unroll 1
        for (int i0 = 0; i0 < 4 * nr4; i0 += 4) {
          float a4[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float x = r[j];
            const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
            float a2 = 0.f;
            if (up_lane) {
              const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(x, -10.0f), 10.0f);
              a2 = silu_mul(gg, uu);
            }
            a4[j] = a2;
          }
          uint32_t m4[4];  // (amax over the warp's lanes: gate lanes hold 0; |a| >= 0 -> max of the
                           // bit patterns == fmaxf)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            m4[j] = __reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(a4[j])));
#pragma unroll
          for (int j = 0; j < 4; ++j) am_mine = lane == i0 + j ? __uint_as_float(m4[j]) : am_mine;
#pragma unroll
          for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
#pragma unroll
          for (int j = 0; j < 4; ++j) r[12 + j] = a4[j];
        }
#pragma unroll 1
        for (int k2 = nr4; k2 < 4; ++k2) {  // rotate the activations to the front (token i at r[i])
          float t4[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) t4[j] = r[j];
#pragma unroll
          for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
#pragma unroll
          for (int j = 0; j < 4; ++j) r[12 + j] = t4[j];
        }
#elif !TPMOE_ACTU_ON
        float r[16];  // unprocessed values at the front, finished activations appended at the back
#pragma unroll
        for (int i = 0; i < 16; ++i) r[i] = v[i];
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
          const float x = r[0];
          const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
          float a2 = 0.f;
          if (up_lane) {
            const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(x, -10.0f), 10.0f);
            a2 = silu_mul(gg, uu);
          }
          float m = fabsf(a2);  // amax over the warp's 16 up lanes (gate lanes hold 0)
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
          am_mine = lane == i ? m : am_mine;
#pragma unroll
          for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
          r[15] = a2;
        }
#pragma unroll 1
        for (int k2 = ng; k2 < 16; ++k2) {  // rotate the activations to the front (token i at r[i])
          const float t0 = r[0];
#pragma unroll
          for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
          r[15] = t0;
        }
#else
      float act[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        act[i] = 0.f;
        if (i < ng) {
          const float gv = __shfl_xor_sync(0xffffffffu, v[i], 8);
          float a2 = 0.f;
          if (up_lane) {
            const float gg = fminf(gv, 10.0f), uu = fminf(fmaxf(v[i], -10.0f), 10.0f);
            a2 = silu_mul(gg, uu);
          }
          act[i] = a2;
          float m = fabsf(a2);  // amax over the warp's 16 up lanes (gate lanes hold 0)
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m,
                                       8));  // (every lane gets it: lane i holds token i's)
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
          am_mine = lane == i ? m : am_mine;
        }
      }
#endif

        if (lane < 16 && tb + lane < NMAX)
          T.am[q * NMAX + tb + lane] = am_mine;  // 32-column scale blocks = warp pairs (q, q ^ 1)
        named_bar(1 + g, 128);
#if TPMOE_ACTX
#pragma unroll 1
        for (int i0 = 0; i0 < ng;
             i0 += 4) {  // 4 tokens per round (tokens tb + i0 + j <= 31: T.am rows hold NMAX)
          uint32_t w4[4];
          int ex4[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int tok = tb + i0 + j;
            const float amax = fmaxf(T.am[q * NMAX + tok], T.am[(q ^ 1) * NMAX + tok]);
            ex4[j] = h_scale_exp(amax);
            const uint32_t byte = up_lane ? e4m3_of(r[j] * exp2i(-ex4[j])) : 0u;
            uint32_t word = 0;
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
              const int jb = 4 * (lane & 3) + bb;
              word |= (__shfl_sync(0xffffffffu, byte, ((jb >> 1) & 7) | ((jb & 1) << 4)) & 0xFFu)
                      << (8 * bb);
            }
            w4[j] = word;
          }
#pragma unroll
          for (int k = 0; k < 12; ++k) r[k] = r[k + 4];
          if (isdry) continue;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int tok = tb + i0 + j;
            if (i0 + j < ng) {
              if (lane < 4)
                *reinterpret_cast<uint32_t*>(himg_p + (tok >> 3) * 1024 + (tok & 7) * 128 +
                                             ((gran ^ (tok & 7)) << 4) + 4 * lane) = w4[j];
              if (lane == 0 && (q & 1) == 0) {
                const int kb = 2 * (tile & 1) + (q >> 1);
                P.hsf[(static_cast<size_t>(p) * NSTG2 + (q2 >> 1)) * 512 + tok * 16 + 8 * (q2 & 1) +
                      kb] = static_cast<uint8_t>(ex4[j] + 127);
              }
            }
          }
        }
#else
#if !TPMOE_ACTU_ON
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
          const float a = r[0];
#pragma unroll
          for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
#else
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          if (i >= ng) break;
          const float a = act[i];
#endif
          const int tok = tb + i;
          const float amax = fmaxf(T.am[q * NMAX + tok], T.am[(q ^ 1) * NMAX + tok]);
          const int ex = h_scale_exp(amax);
          const uint32_t byte = up_lane ? e4m3_of(a * exp2i(-ex)) : 0u;
          uint32_t word = 0;  // lane wd (< 4) gathers bytes 4 wd .. 4 wd + 3 of the granule (byte
                              // jb is in up lane ((jb >> 1) & 7) | (jb & 1) << 4)
#pragma unroll
          for (int bb = 0; bb < 4; ++bb) {
            const int jb = 4 * (lane & 3) + bb;
            word |= (__shfl_sync(0xffffffffu, byte, ((jb >> 1) & 7) | ((jb & 1) << 4)) & 0xFFu)
                    << (8 * bb);
          }
          if (isdry) continue;
          if (lane < 4)
            *reinterpret_cast<uint32_t*>(himg_p + (tok >> 3) * 1024 + (tok & 7) * 128 +
                                         ((gran ^ (tok & 7)) << 4) + 4 * lane) = word;
          if (lane == 0 && (q & 1) == 0) {  // one scale byte per (token, 32-column block)
            const int kb = 2 * (tile & 1) + (q >> 1);
            P.hsf[(static_cast<size_t>(p) * NSTG2 + (q2 >> 1)) * 512 + tok * 16 + 8 * (q2 & 1) +
                  kb] = static_cast<uint8_t>(ex + 127);
          }
        }
#endif

        named_bar(3, 256);
        if (pub && !isdry && TPMOE_PUBRED && f2s) {
          red_release_add(P.pan_cnt + par * MAXP + p, 1u);
        } else if (pub && !isdry) {
          const bool lastit = (atom_add_acq_rel(P.pan_cnt + par * MAXP + p, 1u)) == NT1 - 1u;
          if (lastit) {  // the panel's last FC1 item: append it to the ready list
            unsigned* rl = P.rdy + par * (MAXP + 32);
            st_release(rl + atomicAdd(rl + MAXP, 1u), static_cast<unsigned>(p + 1));
          }
        }
      }
      if (!fc1 && !isdry) {  // FC2 unit done (both groups): queue it for the warp-7 publisher
        named_bar(3, 256);
        if (pub) {
          volatile int* q = reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_T1Q);
          const int qn = q[T1Q];
          while (qn - q[T1Q + 2] >= T1Q) {
          }  // (queue full)
          q[qn % T1Q] = (d.y | (d.z << 24));
          __threadfence_block();
          q[T1Q] = qn + 1;
        }
      }
    }
  }
  // ---- all roles done: TMEM free, then T1 + all-reduce with every thread
  tc_fence_before();
  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc." TPMOE_CG ".sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TMEM_COLS)
                 : "memory");
  }
  tail(P, T, small, tid, epoch, par, smem);
  __syncthreads();
  if (tid == 0) {  // (fire-and-forget: the next call reads the count after its griddepcontrol.wait)
    asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(
                     reinterpret_cast<unsigned long long*>(P.work + 2))
                 : "memory");
  }
}

}  // namespace flashinfer::tp_moe::sm100::flash
