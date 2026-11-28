// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
//
// SM100 (B200) fused tensor-parallel MoE decode kernel, generic shapes (DeepSeek-V4.1-Flash), M <=
// 64, TP4 / TP8.  Shape from TPMOE_DIM (hidden), TPMOE_IFULL (expert intermediate), TPMOE_NEXP
// (experts), TPMOE_NDEV (TP size), TPMOE_ISPLIT and TPMOE_SHI.
//
// Sharding: TPMOE_ISPLIT == TPMOE_NDEV slices every expert's intermediate over the ranks;
// TPMOE_ISPLIT < TPMOE_NDEV is the hybrid: NDEV / ISPLIT expert groups (rank r holds the experts of
// group r / ISPLIT) x ISPLIT intermediate slices.  The shared expert is sliced over all ranks
// (TPMOE_SHI columns per rank, zero-padded to whole 128-column items).
//
// One persistent launch (one CTA per SM, 148 SMs) runs the whole MoE block of a rank: fused gate
// (the DSV4.1 gate: sqrtsoftplus scores, image-token bias, routed scaling in the weights) or top-k
// ids / weights given, MXFP8 quantization of
// x, routing tables, routed FC1 / FC2 on tcgen05 kind::mxf8f6f4 block-scaled MMAs (MXFP4 trtllm-gen
// weight images x MXFP8 activations, fp32 accumulators in TMEM), the shared-expert slice (FP8,
// per-128 block scales), T1 (weighted top-k combine + shared output) and the tensor-parallel
// all-reduce (sentinel-tagged LL messages: one-shot or reduce-scatter + all-gather by token owner)
// over peer buffers that every rank maps (CUDA IPC).  The output, bf16 [M][DIM], is identical on
// every rank.
//
// Numerics:
//   x  -> MXFP8 (e4m3, one UE8M0 scale per 32: 2^ceil(log2(amax / 448)), as flashinfer
//   mxfp8_quantize)   FC1 fp32 (TMEM) act = up * gate * sigmoid(gate), gate <= 10, -10 <= up <= 10
//   (fp32) -> h = MXFP8 per 32, scale h_scale_exp (common.cuh) FC2 fp32 -> bf16 per (token, slot)
//   T1: r = bf16(sum_k bf16(w_k) * o_k) (k order), y = bf16(shared + factor r)     all-reduce:
//   rank-ordered fp32 sum -> bf16
//
// Warps: w0 lane 0 producer | w1 TMEM + MMA | w2..w5, w8..w11 epilogue | w6 B operand | w7 FC2-unit
// publisher.  Prologue and tail use every thread.  Each build serves one M range (TPMOE_MMIN ..
// TPMOE_MMAX); the remaining TPMOE_* macros select per-range schedule variants (the host's build
// table sets them).
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "flashinfer/tp_moe/sm100/common.cuh"

#ifndef TPMOE_EPR
#define TPMOE_EPR \
  0  // FC2 epilogue: register transpose (no staging smem / group barriers per row tile)
#endif
#ifndef TPMOE_FC2G8
#define TPMOE_FC2G8 0  // FC2 units of 8 row tiles when there are >= 2 G of them
#endif

#ifndef TPMOE_ACTX
#define TPMOE_ACTX \
  0  // routed FC1 epilogue: SwiGLU of 4 tokens per loop round (independent chains interleaved), the
     // per-token warp amax by one
     // redux.max on the bits of |a| (== the fmaxf butterfly for non-negative floats)
#endif
#ifndef TPMOE_FC2G4_MINM
#define TPMOE_FC2G4_MINM 64  // FC2 unit width capped at 4 row tiles for M >= this
#endif
#ifndef TPMOE_MMAX
#define TPMOE_MMAX \
  0  // > 0: this build serves M <= TPMOE_MMAX only (the host checks): device reads of M are clamped
     // to it, so the paths of
     // larger M fold away at compile time (less cold code)
#endif
#ifndef TPMOE_NDEV
#define TPMOE_NDEV 0  // TP size
#endif
#ifndef TPMOE_MMIN
#define TPMOE_MMIN \
  1  // > 1: this build serves M >= TPMOE_MMIN only (the paths of smaller M fold away likewise)
#endif

namespace flashinfer::tp_moe::sm100::generic {

#ifndef TPMOE_NDEV
#define TPMOE_NDEV 8
#endif
constexpr int NDEV = TPMOE_NDEV;
#ifndef TPMOE_DIM
#define TPMOE_DIM 4096
#endif
#ifndef TPMOE_IFULL
#define TPMOE_IFULL 2048
#endif
#ifndef TPMOE_NEXP
#define TPMOE_NEXP 256
#endif
#ifndef TPMOE_ISPLIT
#define TPMOE_ISPLIT \
  TPMOE_NDEV  // (hybrid when < NDEV: NDEV / ISPLIT expert groups x ISPLIT intermediate slices)
#endif
constexpr int DIM = TPMOE_DIM, INTER = TPMOE_IFULL / TPMOE_ISPLIT,
              NEXP = TPMOE_NEXP / (NDEV / TPMOE_ISPLIT), TOPK_MAX = 6, TOPK_LD = 8,
              MAXM = 512;  // (NEXP = this rank's experts: its group under the hybrid)
static_assert(NDEV % TPMOE_ISPLIT == 0, "hybrid ISPLIT: expert groups x slices");
static_assert(MAXM <= 512 && MAXM % 32 == 0 && TOPK_MAX <= TOPK_LD && TOPK_LD == 8,
              "entries: token in 9 bits; ids rows of 8 uint16");
constexpr int MAXE = MAXM * TOPK_MAX;  // entries (token, expert) upper bound
constexpr int QU = DIM / 512;          // x quantization units (512 elements) per token
#ifndef TPMOE_NMAX
#define TPMOE_NMAX 32
#endif
constexpr int NMAX = TPMOE_NMAX;                                  // token panel width = MMA N bound
constexpr int MME = (TPMOE_MMAX ? TPMOE_MMAX : MAXM) * TOPK_MAX;  // entries bound of this build
constexpr int MAXP = 1024;  // panels upper bound (<= NEXP + MAXE / NMAX); the same in every build:
                            // the parity-indexed panel counters / ready lists are shared by the
                            // per-M builds of one workspace
static_assert(NEXP + MAXM * TOPK_MAX / NMAX + 1 <= MAXP, "panels");
#ifndef TPMOE_IPAD
#define TPMOE_IPAD \
  (TPMOE_IFULL / TPMOE_ISPLIT)  // per-rank intermediate of the routed weight tensors: INTER valid
                                // rows / K columns, zero-padded to IPAD (never read)
#endif
constexpr int IPAD = TPMOE_IPAD;
constexpr int NT1 =
    (2 * INTER + 127) /
    128;  // FC1 tiles (128 w13 rows = 32-row blocks of interleaved up / gate rows) per panel
constexpr int NT1P = 2 * IPAD / 128;       // FC1 tiles per expert of the weight / scale tensors
constexpr int NCH1 = DIM / 128;            // 32 FC1 k-chunks
constexpr int NRT2 = DIM / 128;            // 32 FC2 row tiles
constexpr int NCH2 = (INTER + 127) / 128;  // h image chunks
constexpr int NCH2P = IPAD / 128;          // FC2 k-chunks per row tile of the w2 scale images
// ragged I (INTER % 128 == 64 or 32): the last FC1 tile is the low half of the last h chunk (its
// rows past 2 INTER are read out of bounds: zeros); FC2 runs the K tail separately (F2T below)
constexpr bool RAGI = INTER % 128 != 0;
static_assert(INTER % 32 == 0 && (INTER % 128 == 0 || INTER % 128 <= 64) && IPAD % 128 == 0 &&
                  IPAD >= INTER && DIM % 512 == 0,
              "FC1 tiles of 64 h columns, FC2 K chunks of 128 (ragged: a half-empty last chunk)");
constexpr int KC = 2;
constexpr int NSTG1 = (NCH1 + KC - 1) / KC,
              NSTG2 = (NCH2 + KC - 1) / KC;  // stages per FC1 item / FC2 row tile
constexpr int KC2L =
    NCH2 -
    KC * (NSTG2 - 1);  // k-chunks of the last FC2 stage (a TMA box past the last chunk reads zeros)
static_assert(NCH1 % KC == 0 && KC2L >= 1 && KC2L <= KC && NCH2 <= NCH2P, "stages");
constexpr int NS = 5;  // ring stages (weight stages in flight per SM)
constexpr int THREADS =
    384;  // 12 warps: producer, MMA, 2 x 4 epilogue, B operand, FC2-unit publisher
// FC2 unit = fc2g consecutive 128-row tiles of one panel (one ticket / ring entry); fc2g chosen per
// launch from the panel count
constexpr int NRT = DIM / 256;  // 16 T1 / all-reduce row tiles (256 rows)
constexpr int NMSG = 43;
constexpr int MAXG = 192;
#ifndef TPMOE_ACC
#define TPMOE_ACC 4  // 32-column routed accumulators in TMEM
#endif
constexpr int ACC = TPMOE_ACC,
              ACC_COLS =
                  32;  // (the shared FC1 gate 0..31; its up accumulator: columns ACC ACC_COLS ..)
#ifdef TPMOE_SHI
constexpr int SHI = TPMOE_SHI;  // this rank's TP-sharded shared-expert slice (zero-padded to whole
                                // 128-column items)
#else
constexpr int SHI = INTER;  // (the shared slice = the routed per-rank intermediate)
#endif
static_assert(SHI % 128 == 0, "shared slice: whole 128-column FC1 items / FC2 K chunks");
// F2T (ragged I): a routed FC2 row tile is K = NCH2F whole 128-column w2 chunks (NSTG2M stages)
// plus its K tail (TAILK = INTER % 128 columns) from the tail image (map2t*): the tails of RPC
// consecutive row tiles side by side in one 128-column chunk.  A unit's tail stages (up to 2
// chunks each) come first and open its accumulators with one K32 MMA per tail block; the main
// stages then accumulate.
constexpr int TAILK = INTER % 128, TB = TAILK / 32, RPC = RAGI ? 4 / TB : 1,
              NTL = RAGI ? NRT2 / RPC : 0, NCH2F = INTER / 128, NSTG2M = NCH2F / KC;
static_assert(!RAGI || (NCH2F % KC == 0 && NRT2 % RPC == 0 && ACC >= 8),
              "F2T: whole main stages and tail chunks; an accumulator per row tile of a unit");
constexpr int NT1S =
    SHI /
    128;  // shared FC1 items per panel (128 h columns: gate rows 128 k.., up rows SHI + 128 k..)
constexpr int NCH2S = SHI / 128, NSTG2S = (NCH2S + KC - 1) / KC,
              KC2LS = NCH2S - KC * (NSTG2S - 1);  // shared FC2: K = SHI
static_assert(SHI == INTER || (NCH2S <= NCH2 && NSTG2S <= NSTG2),
              "shared h / SFB images use the routed panel strides NCH2 / NSTG2");
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
static_assert(9 + TOPK_MAX <= 16, "ENT16");
constexpr int MPB = ((NEXP + MME / NMAX + 32) < MAXP ? (NEXP + MME / NMAX + 32) : MAXP) &
                    ~3;     // panel bound of this build (x4: 16-B aligned tables after it)
constexpr int TAB_ENT = 0;  // uint16 [MME] entry: token | top-k slot mask << 9
constexpr int TAB_PINF =
    TAB_ENT + ((MME * 2 + 15) & ~15);       // uint32 [MPB] panel: e | n << 10 | ebase << 16
constexpr int TAB_AM = TAB_PINF + MPB * 4;  // float [4][NMAX] FC1 epilogue: per-warp amax exchange
static_assert(TAB_AM % 16 == 0 && MPB >= NEXP + MME / NMAX + 1,
              "ENT16: 16-B aligned tables, panel bound");
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
constexpr int SMEM_TOTAL = OFF_TAB + TAB_BYTES, SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(OFF_B % 1024 == 0 && B_BYTES % 1024 == 0 && OFF_TAB % 16 == 0 && TAB_BAR % 8 == 0,
              "alignment");
// prologue scratch inside the A ring (dead before the first TMA)
constexpr int TW = (MAXM + 31) / 32;  // token words
constexpr int PRO_BITS =
    2 * A_BYTES;  // uint32 [TW words][NEXP] token bits per expert (above the early stages)
constexpr int PRO_PRE = PRO_BITS + TW * NEXP * 4;  // uint16 [TW][NEXP] exclusive word prefix
constexpr int PRO_OFFE =
    PRO_PRE + ((TW * NEXP * 2 + 15) & ~15);  // int [NEXP] entry offset of expert
constexpr int PRO_WS = PRO_OFFE + NEXP * 4;  // int [32] warp sums
constexpr int PRO_IDS = PRO_WS + 64 * 4;  // uint16 [MAXM][TOPK_LD] expert of (t, k), 0xFFFF = none
static_assert(PRO_IDS + MAXM * TOPK_LD * 2 <= NS * A_BYTES && PRO_IDS % 16 == 0 &&
                  PRO_OFFE % 4 == 0,
              "prologue scratch");

struct Params {
  CUtensorMap map1, map2;      // w13 {128, rows, chunks, experts}, w2 {128, E * DIM rows, chunks}:
                               // 16U4_ALIGN16B
  CUtensorMap map2t2, map2t1;  // (F2T) the tail image {128, E * 128 rows, NTL chunks}: boxes of 2 /
                               // 1 chunks
  const uint8_t* w13_sf;    // [E][1024][128] block_scale_interleave bytes: 512 B per (tile, chunk)
  const uint8_t* w2_sf;     // [E][row tiles][chunks][512] block_scale_interleave images
  const uint8_t* w2t_sf;    // (F2T) [E][NTL][512] the tail chunks' images
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
  int ebase;  // (hybrid) first global expert id of this rank's group (0 unless ISPLIT < NDEV)
  const float* gbias_alt;  // (GATE41) [TPMOE_NEXP] correction bias of image tokens, or null
  const int* in_ids;       // (GATE41) [M] input token ids of this step or null (no image tokens)
  int img_tok;             // (GATE41) image_token_id
  float gscale;            // routed scaling factor, applied to the routing weights (both entries)
};
__device__ __forceinline__ int lid(const Params& P, int e) {
  return TPMOE_ISPLIT < NDEV ? e - P.ebase : e;
}  // (hybrid: local expert id)
__device__ __forceinline__ int pM(const Params& P) {
  return max(TPMOE_MMAX ? min(P.M, TPMOE_MMAX) : P.M, TPMOE_MMIN);
}  // (TPMOE_MMAX: range known to the compiler)
__device__ __forceinline__ int pOS(const Params& P) {
  return (TPMOE_MMAX && TPMOE_MMAX <= 32) ? 1 : (TPMOE_MMIN > 32 ? 0 : P.oneshot);
}
__device__ __forceinline__ bool pGL(const Params& P) { return P.glog != nullptr; }

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
                                      uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(0x12F0000000000000ull)
      : "memory");
}
__device__ __forceinline__ void tma4d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2,
                                      int c3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar), "l"(0x12F0000000000000ull)
      : "memory");
}
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
// SwiGLU: gate <= 10, -10 <= up <= 10 (fp32), up * gate * sigmoid(gate) (reciprocal form, fast exp)
__device__ __forceinline__ float act_mul(float g, float u) {
  const float gc = fminf(g, 10.0f), uc = fminf(fmaxf(u, -10.0f), 10.0f);
  return uc * __fdividef(gc, 1.0f + __expf(-gc));
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

// FC1 weight stage s (2 chunks) of tile `tile` of expert e
__device__ __forceinline__ void w13_tma(const Params& P, uint32_t dst, int e, int tile, int s,
                                        uint32_t bar) {
  tma4d(dst, &P.map1, 0, tile * 128, KC * s, e, bar);
}
// ------------------------------------------------------------------------------------------------
// fused gate (bitwise moe_fused_gate)
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t gate_key(float v) {
  const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
constexpr int GATE_TOPK = 6;
// one token gated by warps 0 .. GW - 1 together (warp w: experts GPL 32 w .. GPL 32 (w + 1) - 1,
// GPL per lane): per-warp top-6 by redux, then warp 0 merges the 6 GW candidates -- the same keys
// (score bits, then the lower expert index), selection order and weight arithmetic as one warp
// over all experts
constexpr int NEXP_G = TPMOE_NEXP,  // (GATE41) all experts (the hybrid's other groups included)
    GW = 6, GPL = NEXP_G / (32 * GW);
static_assert(NEXP_G == 32 * GW * GPL, "GATE41: whole experts per lane");
__device__ __forceinline__ float act41(float x) {  // sqrtsoftplus: sqrt(x > 20 ? x : log1p(exp(x)))
  float r;
  asm("{\n\t"
      ".reg .pred pa, pb, pab, pz, pg;\n\t"
      ".reg .b32 t, j, k, rr, e, s, y, u, a, b, c, d, f, e2, p, q, lg, t1, lg2, lg3, sp;\n\t"
      "fma.rn.ftz.f32 t, %1, 0f3BBB989D, 0f3F000000;\n\t"
      "cvt.ftz.sat.f32.f32 t, t;\n\t"
      "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n\t"
      "add.f32 k, j, 0fCB40007F;\n\t"
      "neg.f32 k, k;\n\t"
      "fma.rn.ftz.f32 rr, %1, 0f3FB8AA3B, k;\n\t"
      "fma.rn.ftz.f32 rr, %1, 0f32A57060, rr;\n\t"
      "shl.b32 s, j, 23;\n\t"
      "ex2.approx.ftz.f32 e, rr;\n\t"
      "mul.f32 y, e, s;\n\t"
      "add.rz.ftz.f32 u, y, 0f3F800000;\n\t"
      "add.s32 a, u, -1061158912;\n\t"
      "and.b32 a, a, -8388608;\n\t"
      "sub.s32 b, y, a;\n\t"
      "sub.s32 c, 1082130432, a;\n\t"
      "fma.rn.ftz.f32 d, 0f3E800000, c, 0fBF800000;\n\t"
      "add.f32 f, d, b;\n\t"
      "cvt.rn.f32.s32 e2, a;\n\t"
      "mul.f32 e2, e2, 0f34000000;\n\t"
      "fma.rn.ftz.f32 p, 0fBD39BF78, f, 0f3DD80012;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBE0778E0;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3E146475;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBE2A68DD;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3E4CAF9E;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBE800042;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0f3EAAAAE6;\n\t"
      "fma.rn.ftz.f32 p, p, f, 0fBF000000;\n\t"
      "mul.f32 q, f, p;\n\t"
      "fma.rn.ftz.f32 q, q, f, f;\n\t"
      "fma.rn.ftz.f32 lg, e2, 0f3F317218, q;\n\t"
      "setp.lt.u32 pa, y, 2139095040;\n\t"
      "setp.lt.s32 pb, y, -1082130431;\n\t"
      "or.pred pab, pa, pb;\n\t"
      "fma.rn.ftz.f32 t1, y, 0f7F800000, 0f7F800000;\n\t"
      "selp.f32 lg2, lg, t1, pab;\n\t"
      "setp.eq.f32 pz, y, 0f00000000;\n\t"
      "selp.f32 lg3, 0f80000000, lg2, pz;\n\t"
      "selp.f32 lg, lg, lg3, pa;\n\t"
      "setp.gt.f32 pg, %1, 0f41A00000;\n\t"
      "selp.f32 sp, %1, lg, pg;\n\t"
      "sqrt.approx.ftz.f32 %0, sp;\n\t"
      "}"
      : "=f"(r)
      : "f"(x));
  return r;
}
// warp w's part: its experts' scores -> sca[e], its top-6 keys -> cand[6 w ..]
__device__ __forceinline__ void gate41_part(const float* lg, const float* gb, int w, int lane,
                                            uint64_t* cand, float* sca) {
  uint64_t h[GPL];
#pragma unroll
  for (int j = 0; j < GPL; ++j) {
    const int e = 32 * (GPL * w + j) + lane;
    const float a = act41(__ldcg(lg + e));
    float v = __fadd_rn(a, __ldg(gb + e));
    v = v == v ? v
               : __int_as_float(0x7f800000);  // (NaN ranks above every finite score, as torch.topk)
    sca[e] = a;
    h[j] = (static_cast<uint64_t>(gate_key(v)) << 32) |
           static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(e));
  }
#pragma unroll
  for (int k = 0; k < GATE_TOPK; ++k) {  // round k: every lane's largest key, the warp's largest
    uint64_t m = h[0];
#pragma unroll
    for (int j = 1; j < GPL; ++j) m = h[j] > m ? h[j] : m;
    const uint32_t hk = static_cast<uint32_t>(m >> 32), hl = static_cast<uint32_t>(m);
    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
    const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
    if (lane == 0) cand[GATE_TOPK * w + k] = (static_cast<uint64_t>(gm) << 32) | gl;
    if (hk == gm && hl == gl) {  // (keys are unique: the winner lane drops exactly that key)
#pragma unroll
      for (int j = 0; j < GPL; ++j) h[j] = h[j] == m ? 0ull : h[j];
    }
  }
}
// warp 0: the merge -> lanes < topk return (global expert id, routing weight incl. the routed
// scaling) of slot lane
__device__ __forceinline__ void gate41_merge(const Params& P, int lane, const uint64_t* cand,
                                             const float* sca, int& id, float& wt) {
  constexpr int NC = GATE_TOPK * GW;
  static_assert(NC <= 64, "GATE41: two candidates per lane");
  uint64_t c0 = lane < NC ? cand[lane] : 0ull, c1 = lane + 32 < NC ? cand[lane + 32] : 0ull;
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
      __fadd_rn(__fadd_rn(__fadd_rn(R[0], R[1]), __fadd_rn(R[2], R[3])), __fadd_rn(R[4], R[5]));
  const float norm = __fadd_rn(sum, 1e-20f);  // (RENORMALIZE_EPSILON 1e-20)
  float r = R[0];
  id = ex[0];
#pragma unroll
  for (int k = 1; k < GATE_TOPK; ++k) {
    r = lane == k ? R[k] : r;
    id = lane == k ? ex[k] : id;
  }
  wt = __fmul_rn(div_full(r, norm), P.gscale);
}
__device__ __forceinline__ const float* gate_bias41(const Params& P, int t) {
  return (P.gbias_alt && P.in_ids && __ldg(P.in_ids + t) == P.img_tok) ? P.gbias_alt : P.gbias;
}
// MXFP8 quantization of x in units (token t, 512-element group cg) by one warp, lane l: elements 16
// l .. + 15 (chunk 4 cg + l / 8, 16-B granule l % 8, 32-block = lanes l, l ^ 1).  Writes the
// token-major codes xq and scales xsf2 (the gather4 B operand) and the shared expert's per-128
// image, then counts the unit on xcnt[t].  Units u = gw + i nw, 4 per warp in flight.
__device__ __forceinline__ void quant_x(const Params& P, unsigned* xcnt, int lane, int gw, int nw) {
  const int U = pM(P) * QU;
  const int QN = (QU * TPMOE_MMAX <= nw) ? 1 : 4;  // (units in flight; 1: every warp has <= 1 unit)
  for (int u0 = gw; u0 < U; u0 += QN * nw) {
    uint4 raw[4][2];
#pragma unroll
    for (int j = 0; j < QN; ++j) {
      const int u = u0 + j * nw;
      if (u < U) {
        {
          const uint4* src = reinterpret_cast<const uint4*>(
                                 P.xb + static_cast<size_t>(u / QU) * DIM + (u % QU) * 512) +
                             lane * 2;
          raw[j][0] = __ldg(src);
          raw[j][1] = __ldg(src + 1);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < QN; ++j) {
      const int u = u0 + j * nw;
      if (u >= U) break;
      const int t = u / QU, cg = u % QU, ch = 4 * cg + (lane >> 3), gr = lane & 7;
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
        const int sp = t / NMAX,
                  r = t - sp * NMAX;  // (NMAX-token panels, as the shared FC1 loader reads them)
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
    for (int u = gw; u < U; u += nw) red_relaxed_add(xcnt + (u / QU), 1u);
  }
}

// ------------------------------------------------------------------------------------------------
// routing tables
struct Tabs {
  uint16_t* ids;
  uint16_t* ent;
  uint8_t* km;
  uint32_t* pinf;
  float* am;
  int* misc;
  uint8_t* lanes;
  int4* ring;
  __nv_bfloat16* stg;
};
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
  uint8_t* t = smem + OFF_TAB;
  return Tabs{reinterpret_cast<uint16_t*>(smem + PRO_IDS),
              reinterpret_cast<uint16_t*>(t + TAB_ENT),
              nullptr,
              reinterpret_cast<uint32_t*>(t + TAB_PINF),
              reinterpret_cast<float*>(t + TAB_AM),
              reinterpret_cast<int*>(t + TAB_MISC),
              t + TAB_LANES,
              reinterpret_cast<int4*>(t + TAB_RING),
              reinterpret_cast<__nv_bfloat16*>(t + TAB_STG)};
}
// Every CTA, all THREADS: union = present experts in ascending id; entries of an expert = its
// tokens in ascending order (a token listing the same expert in several slots is ONE entry, km =
// its slots); panels of <= NMAX entries.  ids from global (P.ids).
// Writes T.ids, T.ent, T.km, T.pinf, misc 0..2.
__device__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  constexpr int NW = THREADS / 32,
                EPT = (NEXP + THREADS - 1) /
                      THREADS;  // (experts EPT tid .. EPT tid + EPT - 1 per thread: ascending)
  const int M = pM(P), topk = 6, MT = M * topk, W = (M + 31) >> 5;
  uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
  uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
  int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
  int* ws = reinterpret_cast<int*>(smem + PRO_WS);
  constexpr int MAXV =
      (MME / 4 + THREADS) / THREADS;  // int4 groups of the flat [M][6] ids (+ a partial last one)
  static_assert(TOPK_MAX == 6, "ids as int4 groups of the flat [M][6] array");
  const int nv4 = (MT + 3) / 4;
  int4 iv[MAXV];
  {
#pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int v = tid + j * THREADS;
      if (v < MT / 4) {
        iv[j] = __ldcg(reinterpret_cast<const int4*>(P.ids) + v);
        if (TPMOE_ISPLIT < NDEV) {
          iv[j].x -= P.ebase;
          iv[j].y -= P.ebase;
          iv[j].z -= P.ebase;
          iv[j].w -= P.ebase;
        }  // (hybrid)
      } else if (v < nv4) {
        const int* ip = P.ids + 4 * v;
        iv[j] = make_int4(lid(P, __ldcg(ip)), 4 * v + 1 < MT ? lid(P, __ldcg(ip + 1)) : -1,
                          4 * v + 2 < MT ? lid(P, __ldcg(ip + 2)) : -1, -1);
      } else {
        iv[j] = make_int4(-1, -1, -1, -1);
      }
    }
  }
  for (int i = tid; i < W * NEXP; i += THREADS) bits[i] = 0u;
  for (int i = tid; i < M * TOPK_LD; i += THREADS) T.ids[i] = 0xFFFF;
  __syncthreads();
  {
#pragma unroll
    for (int j = 0; j < MAXV; ++j) {
      const int v = tid + j * THREADS;
      if (v >= nv4) break;
      const int ev[4] = {iv[j].x, iv[j].y, iv[j].z, iv[j].w};
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int e = ev[q4], idx = 4 * v + q4, t = idx / TOPK_MAX, k = idx - t * TOPK_MAX;
        if (idx < MT && static_cast<unsigned>(e) < static_cast<unsigned>(NEXP)) {
          T.ids[t * TOPK_LD + k] = static_cast<uint16_t>(e);
          atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
        }
      }
    }
  }
  __syncthreads();
  // per expert: token count + word prefix; scans of (present, count, panels) over the experts in
  // ascending order
  int cnt[EPT];
  int a = 0, bsum = 0, d = 0;
#pragma unroll
  for (int j = 0; j < EPT; ++j) {
    const int e = EPT * tid + j;
    int c = 0;
    if (e < NEXP)
      for (int w = 0; w < W; ++w) {
        pre[w * NEXP + e] = static_cast<uint16_t>(c);
        c += __popc(bits[w * NEXP + e]);
      }
    cnt[j] = c;
    a += c > 0;
    bsum += c;
    d += (c + NMAX - 1) / NMAX;
  }
  const int a0 = a, b0 = bsum, d0 = d;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ya = __shfl_up_sync(0xffffffffu, a, o), yb = __shfl_up_sync(0xffffffffu, bsum, o),
              yd = __shfl_up_sync(0xffffffffu, d, o);
    if (lane >= o) {
      a += ya;
      bsum += yb;
      d += yd;
    }
  }
  if (lane == 31) {
    ws[warp] = a;
    ws[NW + warp] = bsum;
    ws[2 * NW + warp] = d;
  }
  __syncthreads();
  {
    int xb = bsum - b0, xd = d - d0;
    for (int w = 0; w < warp; ++w) {
      xb += ws[NW + w];
      xd += ws[2 * NW + w];
    }
#pragma unroll
    for (int j = 0; j < EPT; ++j) {
      const int e = EPT * tid + j, c = cnt[j];
      if (e >= NEXP) break;
      offe[e] = xb;
      const int npan = (c + NMAX - 1) / NMAX;
      for (int p = 0; p < npan; ++p)
        T.pinf[xd + p] = static_cast<uint32_t>(e) |
                         (static_cast<uint32_t>(min(NMAX, c - p * NMAX)) << 10) |
                         (static_cast<uint32_t>(xb + p * NMAX) << 16);
      xb += c;
      xd += npan;
    }
  }
  if (tid == THREADS - 1) {
    int U = 0, E = 0, Pn = 0;
    for (int w = 0; w < NW; ++w) {
      U += ws[w];
      E += ws[NW + w];
      Pn += ws[2 * NW + w];
    }
    T.misc[0] = U;
    T.misc[1] = E;
    T.misc[2] = Pn;
  }
  __syncthreads();
  // entries: one per (token, expert) at its first slot
  for (int p = tid; p < MT; p += THREADS) {
    const int t = p / topk, k = p - t * topk;
    const uint4 r0 = *reinterpret_cast<const uint4*>(
        T.ids + t * TOPK_LD);  // (TOPK_LD uint16 slots per token row)
    const uint32_t rw[4] = {r0.x, r0.y, r0.z, r0.w};
    const int e = static_cast<int>((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
    if (e == 0xFFFF) continue;
    unsigned km = 0;
#pragma unroll
    for (int k2 = 0; k2 < TOPK_MAX; ++k2)
      km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == static_cast<unsigned>(e) ? 1u : 0u)
            << k2;
    if (km & ((1u << k) - 1u)) continue;
    const int w = t >> 5;
    const int idx =
        offe[e] + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & ((1u << (t & 31)) - 1u));
    T.ent[idx] =
        static_cast<uint16_t>(static_cast<uint32_t>(t) | (km << 9));  // (9 + TOPK_MAX <= 16 bits)
  }
  __syncthreads();
}

// (F2T) a routed FC2 unit of cn row tiles from row tile r0: tail chunks c_lo .. c_lo + nch - 1,
// in nts tail stages of up to 2 chunks; the unit's ring stages
__device__ __forceinline__ void f2t_unit(int r0, int cn, int& c_lo, int& nch, int& nts) {
  c_lo = r0 / RPC;
  nch = (r0 + cn - 1) / RPC - c_lo + 1;
  nts = (nch + 1) >> 1;
}
__device__ __forceinline__ int f2t_nst(int r0, int cn) {
  int c_lo, nch, nts;
  f2t_unit(r0, cn, c_lo, nch, nts);
  return nts + cn * NSTG2M;
}
// ------------------------------------------------------------------------------------------------
// work Ring entries: x = kind (1 routed FC1 item, 2 routed FC2 unit, 3 shared FC1 item, 4 shared
// FC2 unit, -1 end); FC1: y = item (panel item / NT1, tile item % NT1), z / w = its stage range;
// FC2: y = panel NRT2 + first row tile, z = row tiles.
__device__ __forceinline__ void panel_of(const Tabs& T, int p, int& e, int& n, int& ebase) {
  const uint32_t v = T.pinf[p];
  e = static_cast<int>(v & 1023u);
  n = static_cast<int>((v >> 10) & 63u);
  ebase = static_cast<int>(v >> 16);
}

// ------------------------------------------------------------------------------------------------
// tail: T1 + all-reduce
constexpr int OS_MAXM = MAXM;
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) {
  return (static_cast<size_t>((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m;
}
constexpr size_t RS_PAR =
    static_cast<size_t>(NDEV) * NRT * OS_MAXM * NMSG;  // (one-shot: [src][rtile][t][m])
constexpr int OUTP_LD = DIM + 0;                       // outp [MAXM][TOPK_MAX][OUTP_LD]
constexpr size_t AG_PAR = static_cast<size_t>(NRT) * MAXM * NMSG;

// T1 of one (row tile, token) by one warp: lane l owns values rtile * 256 + 8 l .. + 7
// all-reduce message = 8 bf16 (a lane's 16 B of a 256-row tile); its w word never equals the
// sentinel (NaN pairs canonicalized)
constexpr uint32_t LLS_S = 0xFFFFFFFFu;
__device__ __forceinline__ uint4 lls_msg(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  return make_uint4(x, y, z, w == LLS_S ? 0x7FC07FC0u : w);
}
__device__ __forceinline__ void lls_send(
    const Params& P, int rtile, int t, unsigned epoch, int lane,
    const uint32_t* o4) {  // T1 row slice -> all-reduce messages
  const uint4 msg = lls_msg(o4[0], o4[1], o4[2], o4[3]);
  if (pOS(P)) {
#pragma unroll
    for (int d = 0; d < NDEV; ++d)
      st_ll(P.rs[(P.my_rank + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR +
                os_idx(P.my_rank, rtile, t, lane),
            msg);
  } else {
    st_ll(P.rs[t & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(P.my_rank, rtile, t, lane), msg);
  }
}
__device__ __forceinline__ void lls_acc(const uint4& v, float* a, bool first) {
  const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    a[2 * j] = first ? bf16lo(w4[j]) : __fadd_rn(a[2 * j], bf16lo(w4[j]));
    a[2 * j + 1] = first ? bf16hi(w4[j]) : __fadd_rn(a[2 * j + 1], bf16hi(w4[j]));
  }
}
// the all-reduce of the T1 rows: one-shot (every source -> every rank) or RS + AG by token owner t
// % NDEV; sums in rank order
__device__ __noinline__ void lls_allreduce(const Params& P, int b, int G, int M, int tid,
                                           unsigned epoch) {
  if (gridDim.x != static_cast<unsigned>(148)) __trap();  // (one reader per slot)
  const uint4 S4 = make_uint4(LLS_S, LLS_S, LLS_S, LLS_S);
  const int rk = P.my_rank;
  uint4* rs_me = P.rs[rk] + (epoch & 1u) * RS_PAR;
  if (pOS(P)) {
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    for (int x = tid; x < mine * 32; x += THREADS) {
      const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / M, t = j - rtile * M;
      uint4 v[NDEV];
      for (;;) {
#pragma unroll
        for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
        bool ok = true;
#pragma unroll
        for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;
        if (ok) break;
      }
      float a[8];
#pragma unroll
      for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);
      *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) =
          make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]),
                     pack_bf16x2(a[6], a[7]));
#pragma unroll
      for (int d = 0; d < NDEV; ++d)
        *(rs_me + os_idx(d, rtile, t, m)) = S4;  // (slots free for the call after next)
    }
    return;
  }
  const int mown = M > rk ? (M - rk + NDEV - 1) / NDEV : 0;
  const int npo = NRT * mown, mineo = b < npo ? (npo - b + G - 1) / G : 0;
  for (int x = tid; x < mineo * 32; x += THREADS) {  // owner: reduce, write out, push to every
                                                     // other rank (this rank's source slot there)
    const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / mown,
              t = rk + NDEV * (j - rtile * mown);
    uint4 v[NDEV];
    for (;;) {
#pragma unroll
      for (int d = 0; d < NDEV; ++d) v[d] = ld_ll(rs_me + os_idx(d, rtile, t, m));
      bool ok = true;
#pragma unroll
      for (int d = 0; d < NDEV; ++d) ok &= v[d].w != LLS_S;
      if (ok) break;
    }
    float a[8];
#pragma unroll
    for (int d = 0; d < NDEV; ++d) lls_acc(v[d], a, d == 0);
    const uint4 mo = lls_msg(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]),
                             pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));
    *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * m) =
        make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]),
                   pack_bf16x2(a[6], a[7]));
#pragma unroll
    for (int d = 1; d < NDEV; ++d)
      st_ll(P.rs[(rk + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(rk, rtile, t, m), mo);
#pragma unroll
    for (int d = 0; d < NDEV; ++d) *(rs_me + os_idx(d, rtile, t, m)) = S4;
  }
  const int mo = M - mown, npr = NRT * mo, miner = b < npr ? (npr - b + G - 1) / G : 0,
            nrx = miner * 32;
  constexpr int RB = 8;
  for (int x0 = tid; x0 < nrx;
       x0 +=
       RB * THREADS) {  // the other ranks' tokens: their owner's pushed rows, RB polls in flight
    uint4 v[RB];
    uint4* pm[RB];
    int ti[RB], ri[RB];
#pragma unroll
    for (int q = 0; q < RB; ++q) {
      const int x = x0 + q * THREADS;
      pm[q] = nullptr;
      ti[q] = 0;
      ri[q] = 0;
      v[q] = S4;
      if (x >= nrx) continue;
      const int pi = x >> 5, m = x & 31, j = b + pi * G, rtile = j / mo, i = j - rtile * mo,
                rr = i % (NDEV - 1);
      const int t = NDEV * (i / (NDEV - 1)) + (rr < rk ? rr : rr + 1);
      pm[q] = rs_me + os_idx(t & (NDEV - 1), rtile, t, m);
      ti[q] = t;
      ri[q] = rtile * 256 + 8 * m;
      v[q] = ld_ll(pm[q]);
    }
#pragma unroll
    for (int q = 0; q < RB; ++q) {
      if (!pm[q]) continue;
      while (v[q].w == LLS_S) v[q] = ld_ll(pm[q]);
      *reinterpret_cast<uint4*>(P.out + static_cast<size_t>(ti[q]) * DIM + ri[q]) = v[q];
      *pm[q] = S4;
    }
  }
}
#ifndef TPMOE_EPRSGL
#define TPMOE_EPRSGL \
  0  // 2: EPR epilogue on trtllm-gen-shuffled w2 rows (the FC2 row of each TMEM lane, as the host's
     // lanes table)
#endif
#ifndef TPMOE_T1PW
#define TPMOE_T1PW 4  // T1 by independent warps, this many tokens per warp batch
#endif
// T1 result of (rtile, t): r = bf16(acc), y = bf16(shared + factor r) -> all-reduce messages
__device__ __noinline__ void t1_fin(const Params& P, int rtile, int t, unsigned epoch,
                                    __nv_bfloat16* stg, int lane, const float* acc,
                                    uint4 shv = make_uint4(0u, 0u, 0u, 0u), bool shv_ok = false) {
  uint32_t o4[4];
  const __nv_bfloat16* shp_ = P.shout;
  uint4 sh4 = make_uint4(0u, 0u, 0u, 0u);
  if (shp_)
    sh4 =
        shv_ok ? shv : ld_v4_last_use(shp_ + static_cast<size_t>(t) * DIM + rtile * 256 + 8 * lane);
  const uint32_t s4[4] = {sh4.x, sh4.y, sh4.z, sh4.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float r0 = bf16r(acc[2 * j]), r1 = bf16r(acc[2 * j + 1]);
    o4[j] = pack_bf16x2(shp_ ? __fmaf_rn(P.factor, r0, bf16lo(s4[j])) : __fmul_rn(P.factor, r0),
                        shp_ ? __fmaf_rn(P.factor, r1, bf16hi(s4[j])) : __fmul_rn(P.factor, r1));
  }
  if (NDEV > 1) {
    lls_send(P, rtile, t, epoch, lane, o4);
    return;
  }  // (lane l = message l)
  *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
  __syncwarp();
  for (int m = lane; m < NMSG; m += 32) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
    const bool full6 = m < NMSG - 1;
    const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
    if (NDEV == 1) {
      uint32_t* o =
          reinterpret_cast<uint32_t*>(P.out + static_cast<size_t>(t) * DIM + rtile * 256 + 6 * m);
      o[0] = msg.x;
      o[1] = msg.y;
      if (full6) o[2] = msg.z;
    } else if (!pOS(P)) {  // (reduce-scatter: to the token's owner only)
      st_ll(P.rs[t & (NDEV - 1)] + (epoch & 1u) * RS_PAR + os_idx(P.my_rank, rtile, t, m), msg);
    } else {
#pragma unroll 1
      for (int d = 0; d < NDEV; ++d)
        st_ll(P.rs[(P.my_rank + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR +
                  os_idx(P.my_rank, rtile, t, m),
              msg);
    }
  }
  __syncwarp();
}

__device__ void tail(const Params& P, const Tabs& T, int tid, unsigned epoch, unsigned par,
                     uint8_t* smem) {
  const int G = 148, b = blockIdx.x, M = pM(P);
  const int NWT = THREADS / 32, wtail = tid >> 5;
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
    const int NW = NWT;
    {  // T1 by warps, independently: TPMOE_T1PW tokens per warp batch, every slot load of the batch
       // in flight together
      constexpr int PW = TPMOE_T1PW;
      const int topk = 6, lane = tid & 31;
      __nv_bfloat16* stg1 =
          reinterpret_cast<__nv_bfloat16*>(smem) + (tid >> 5) * 256;  // (A ring: dead here)
      for (int tt = tid >> 5; tt < nt; tt += PW * NW) {
        uint4 v[PW][TOPK_MAX];
        float wk[PW][TOPK_MAX];
#pragma unroll
        for (int j = 0; j < PW; ++j) {
          const int ti = tt + j * NW, t = t0 + ti;
#pragma unroll
          for (int k = 0; k < TOPK_MAX; ++k) {
            const bool ok = ti < nt && k < topk;
            const int id = ok ? lid(P, __ldcg(P.ids + t * topk + k)) : -1;
            // (the gate's weights carry the routed scaling factor; the ids entry's given
            // renormalized weights get it here, by the fp32 multiply the gate applies to its own)
            const float w0 = ok ? __ldcg(P.rw + t * topk + k) : 0.f;
            const float wv = pGL(P) ? w0 : __fmul_rn(w0, P.gscale);
            v[j][k] =
                ok ? ld_v4_last_use(P.outp + (static_cast<size_t>(t) * TOPK_MAX + k) * OUTP_LD +
                                    rtile * 256 + 8 * lane)
                   : make_uint4(0u, 0u, 0u, 0u);
            wk[j][k] = static_cast<unsigned>(id) < static_cast<unsigned>(NEXP) ? wv : 0.f;
          }
        }
#pragma unroll
        for (int j = 0; j < PW; ++j) {
          if (tt + j * NW >= nt) break;
          float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int k = 0; k < TOPK_MAX; ++k) {
            if (wk[j][k] == 0.f) continue;
            const uint4 q = v[j][k];
            const float w = wk[j][k];
            acc[0] = __fmaf_rn(bf16lo(q.x), w, acc[0]);
            acc[1] = __fmaf_rn(bf16hi(q.x), w, acc[1]);
            acc[2] = __fmaf_rn(bf16lo(q.y), w, acc[2]);
            acc[3] = __fmaf_rn(bf16hi(q.y), w, acc[3]);
            acc[4] = __fmaf_rn(bf16lo(q.z), w, acc[4]);
            acc[5] = __fmaf_rn(bf16hi(q.z), w, acc[5]);
            acc[6] = __fmaf_rn(bf16lo(q.w), w, acc[6]);
            acc[7] = __fmaf_rn(bf16hi(q.w), w, acc[7]);
          }
          t1_fin(P, rtile, t0 + tt + j * NW, epoch, stg1, lane, acc);
        }
      }
    }
  }
  // all-reduce of the T1 rows
  lls_allreduce(P, b, G, M, tid, epoch);
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
  const bool f2s = (M >= 1 && M <= 8);  // FC2 tickets in static panel order
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
  const bool shgu = (M <= 6);  // SHGU: a shared FC1 item's gate and up chains on two CTAs
  const int nshq0 = ((M + NMAX - 1) / NMAX) * NT1S * (shgu ? 2 : 1);  // shared FC1 pieces
  const int shh = (shgu && b < nshq0)
                      ? (b & 1)
                      : -1;  // (SHGU: this CTA's shared piece is the gate (0) or the up (1) chain)
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
  const unsigned epoch = __ldcg(P.work + (64 + b)) + 1u, par = epoch & 1u;
  if (b == 0) {  // the next launch's parity set (the previous launch, its user, has completed)
    for (int i = tid; i < MAXP; i += THREADS) P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u;
    for (int i = tid; i < MAXP + 32; i += THREADS) P.rdy[(par ^ 1u) * (MAXP + 32) + i] = 0u;
    for (int i = tid; i < MAXM; i += THREADS) P.xflag[(par ^ 1u) * MAXM + i] = 0u;
    if (tid < NRT) P.rt_cnt[(par ^ 1u) * NRT + tid] = 0u;
    if (tid == NRT) {
      P.work[1 + (par ^ 1u)] = 0u;
    }
    if (tid == NRT + 1) {
      P.fc2_next[par ^ 1u] = 0u;
      P.fc2_next[2 + (par ^ 1u)] = 0u;
      P.gcnt[par ^ 1u] = 0u;
    }
  }
  // ---- routing (fused gate) + MXFP8 quantization of x
  if (pGL(P) &&
      warp < GW) {  // token t by CTA t % G, warps 0 .. GW - 1 together; the candidates and
                    // scores in ring stage 0 (dead in the prologue); counted on gcnt[par]
    uint64_t* gcand = reinterpret_cast<uint64_t*>(smem);
    float* gsca = reinterpret_cast<float*>(smem + 8 * GATE_TOPK * GW);
    for (int t = b; t < M; t += G) {
      gate41_part(P.glog + static_cast<size_t>(t) * NEXP_G, gate_bias41(P, t), warp, lane, gcand,
                  gsca);
      named_bar(4, 32 * GW);
      if (warp == 0) {
        int id;
        float w;
        gate41_merge(P, lane, gcand, gsca, id, w);
        if (lane < 6) {
          P.ids[t * 6 + lane] = id;
          P.rw[t * 6 + lane] = w;
        }
        __syncwarp();
        if (lane == 0) {
          fence_acq_rel();
          atomicAdd(P.gcnt + par, 1u);
        }
      }
      if (t + G < M) named_bar(4, 32 * GW);  // (the scratch is reused by the next token)
    }
  }
  // x -> MXFP8 (+ shared per-128) by token, independent of the routing: right after (and on other
  // warps than) the gate
  quant_x(P, P.xflag + par * MAXM, lane, (THREADS / 32 - 1 - warp) * G + b, (THREADS / 32) * G);
  if (pGL(P)) {
    if (tid == 0) wait_ge(P.gcnt + par, static_cast<unsigned>(M));
    __syncthreads();
  }
  __syncthreads();
  unsigned xv[16];  // (warp 6) every token's x flag, loaded here: in flight over the tables,
                    // checked after the role split
  if (warp == 6) {
#pragma unroll
    for (int i = 0; i < 16; ++i)
      xv[i] = lane + 32 * i < M ? (ld_relaxed(P.xflag + par * MAXM + lane + 32 * i))
                                : static_cast<unsigned>(QU);
  }
  build_tables(P, smem, T);
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
  const int fc2g0 = (TPMOE_FC2G8 && Pall * (NRT2 / 8) >= 2 * G)
                        ? 8
                        : (Pall * (NRT2 / 4) >= 2 * G ? 4 : (Pall * (NRT2 / 2) >= 2 * G ? 2 : 1));
  // FC2 units.  Ragged I with 32-column K tails (4 row tiles per tail chunk), and with 64-column
  // tails at M < 16: whole F2T units, 8 row tiles (one 2-chunk tail stage) when every CTA gets >= 4
  // of them, else 4, no finer tail.  Otherwise (64-column tails at M >= 16 too: 0.3 - 1.9 us faster
  // at TP4 M 16 - 64) units of at most 4 row tiles, then pairs / singles.
  const bool f2tw = RAGI && (TAILK == 32 || M < 16);
  const int fc2g = f2tw ? (Pall * (NRT2 / 8) >= 4 * G ? 8 : 4)
                        : min(fc2g0, M >= TPMOE_FC2G4_MINM ? 4 : 8),
            nrg = NRT2 / fc2g;  // (4-row-tile units at M >= TPMOE_FC2G4_MINM)
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
      auto slot_wait = [&](int st) {  // the slot's previous use released
        if (fills >= NS) {
          mbar_wait(bar(B_EMPTY + st), static_cast<uint32_t>((fills / NS) - 1) & 1u);
        }
      };
      auto push = [&](int4 d) {  // -> the ring
        const int slot = w % WR;
        if (w >= WR) mbar_wait(bar(B_WE + slot), ((w / WR) - 1) & 1);
        ring[slot] = d;
        mbar_arrive(bar(B_WF + slot));
        ++w;
      };
      auto stage = [&](const CUtensorMap* map, int row, int ch0, const uint8_t* sfa, uint32_t atx,
                       int nsf, int c3 = -1) {  // KC chunks of one 128-row tile (nsf SFA images)
        if (fills < 0) {
          ++fills;
          return;
        }
        const int st = fills % NS;
        slot_wait(st);
        mbar_expect_tx(bar(B_FULL + st), atx + (nsf * 512));
        if (c3 >= 0)
          tma4d(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, c3, bar(B_FULL + st));
        else
          tma3d(sb + OFF_A + st * A_BYTES, map, 0, row, ch0, bar(B_FULL + st));
        bulk_g2s(sb + OFF_SFA + st * KC * 512, sfa, nsf * 512, bar(B_FULL + st));
        ++fills;
      };
      auto stage_s1 = [&](int k, int c) {  // shared FC1: chunk c of the gate tile (rows 128 k ..)
                                           // and the up tile (rows INTER + 128 k ..)
        if (fills < 0) {
          ++fills;
          return;
        }  // (issued before the PDL wait)
        const int st = fills % NS;
        slot_wait(st);
        if (shh >= 0) {  // (SHGU: stage c = chunks 2 c, 2 c + 1 of this CTA's gate or up rows +
                         // their scales)
          mbar_expect_tx(bar(B_FULL + st), 2 * 16384 + 2 * 512);
          for (int h2 = 0; h2 < 2; ++h2) {
            tma3d(sb + OFF_A + st * A_BYTES + h2 * 16384, &P.map1s, 0, shh * SHI + 128 * k,
                  2 * c + h2, bar(B_FULL + st));
            bulk_g2s(
                sb + OFF_SFA + st * KC * 512 + h2 * 512,
                P.sfa1s + (static_cast<size_t>(shh * (SHI / 128) + k) * NCH1 + 2 * c + h2) * 512,
                512, bar(B_FULL + st));
          }
          ++fills;
          return;
        }
        mbar_expect_tx(bar(B_FULL + st), (2 * 16384) + 2 * 512);
        tma3d(sb + OFF_A + st * A_BYTES, &P.map1s, 0, 128 * k, c, bar(B_FULL + st));
        tma3d(sb + OFF_A + st * A_BYTES + 16384, &P.map1s, 0, SHI + 128 * k, c, bar(B_FULL + st));
        bulk_g2s(sb + OFF_SFA + st * KC * 512, P.sfa1s + (static_cast<size_t>(k) * NCH1 + c) * 512,
                 512, bar(B_FULL + st));
        bulk_g2s(sb + OFF_SFA + st * KC * 512 + 512,
                 P.sfa1s + (static_cast<size_t>(SHI / 128 + k) * NCH1 + c) * 512, 512,
                 bar(B_FULL + st));
        ++fills;
      };
      auto issue = [&](const int4& d) {  // this CTA's stages of a descriptor
        if (d.x == 1) {
          const int item = d.y, tile = item % NT1;
          int e, n, eb;
          panel_of(T, item / NT1, e, n, eb);
          for (int s = d.z; s < d.w; ++s) {
            if (fills < 0) {
              ++fills;
              continue;
            }
            const int st = fills % NS;
            slot_wait(st);
            mbar_expect_tx(bar(B_FULL + st), KC * 8192 + (KC * 512));
            w13_tma(P, sb + OFF_A + st * A_BYTES, e, tile, s, bar(B_FULL + st));
            bulk_g2s(sb + OFF_SFA + st * KC * 512,
                     P.w13_sf + ((static_cast<size_t>(e) * NT1P + tile) * NCH1 + KC * s) * 512,
                     KC * 512, bar(B_FULL + st));
            ++fills;
          }
        } else if (d.x == 3) {
          const int k = d.y % NT1S;
          for (int s = d.z; s < d.w; ++s) stage_s1(k, s);
        } else {
          const int p = f2p(d), r0 = f2r0(d), cn = f2n(d);
          if (p < Ptot) {
            int e, n, eb;
            panel_of(T, p, e, n, eb);
            if (RAGI) {  // (F2T) the tail stages, then each row tile's main stages
              int c_lo, nch, nts;
              f2t_unit(r0, cn, c_lo, nch, nts);
              for (int t = 0; t < nts; ++t) {
                const int nv = min(2, nch - 2 * t), c = c_lo + 2 * t;
                stage(nv == 2 ? &P.map2t2 : &P.map2t1, e * 128, c,
                      P.w2t_sf + (static_cast<size_t>(e) * NTL + c) * 512, nv * 8192, nv);
              }
              for (int j = 0; j < cn; ++j)
                for (int s = 0; s < NSTG2M; ++s)
                  stage(&P.map2, e * DIM + (r0 + j) * 128, KC * s,
                        P.w2_sf + ((static_cast<size_t>(e) * NRT2 + r0 + j) * NCH2P + KC * s) * 512,
                        KC * 8192, KC);
            } else {
              for (int j = 0; j < cn; ++j) {
                const int r = (r0 + j);
                for (int s = 0; s < NSTG2; ++s)
                  stage(&P.map2, e * DIM + r * 128, KC * s,
                        P.w2_sf + ((static_cast<size_t>(e) * NRT2 + r) * NCH2P + KC * s) * 512,
                        KC * 8192, s == NSTG2 - 1 ? KC2L : KC);
              }
            }
          } else {
            for (int j = 0; j < cn; ++j) {
              const int r = (r0 + j);
              for (int s = 0; s < NSTG2S; ++s)
                stage(&P.map2s, r * 128, KC * s,
                      P.sfa2s + (static_cast<size_t>(r) * NCH2S + KC * s) * 512, KC * 16384,
                      s == NSTG2S - 1 ? KC2LS : KC);
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
            auto rk = [&](int k) {
              return k < nw0 ? k * gs + (b - nshq) : head + b + (k - nw0) * G;
            };
            for (int k = 0;; ++k) {
              const int r = rk(k);
              if (r >= nr) {
                if (k < nw0) continue;
                break;
              }
              put_r(r);
            }
          }
        }

        // (row tiles R in slot-major order; [0, RA) in units of fc2g, [RA, RB) pairs, [RB, RT)
        // singles; whole F2T units: none of the finer tail)
        const int RT = Pall * NRT2;
        int RA = RT, RB = RT;
        if (!f2tw && fc2g >= 4) {
          RB = max(0, RT - 2 * ((G + 1) / 2));
          RA = RB >= 2 * G ? ((RB - 2 * G) / fc2g) * fc2g : 0;
        } else if (!f2tw && fc2g == 2 && fc2g0 >= 4) {
          RB = max(0, RT - 2 * ((G + 1) / 2));
          RA = RB;
        }  // (pairs, then singles)
        const int nA = RA / fc2g, nB = (RB - RA) / 2;
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
          int R0;
          tk(u, R0);
          return R0 / NRT2;
        };
        const int units = nA + nB + (RT - RB);
        // FC2 units in the order panels become ready: ticket u -> ready-list slot u / nrg (the
        // panel whose h completed (u / nrg)-th), row group u % nrg.  One ticket ahead, its slot
        // loaded as soon as the ticket arrives (acquire: the panel's h)
        const unsigned* rl = P.rdy + par * (MAXP + 32);
        auto ldf = [&](const unsigned* a) -> unsigned { return ld_acquire(a); };
        int tq = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
        unsigned pf = tq < units ? ldf(rl + tslot(tq)) : 0u;
        for (;;) {
          const int u = tq;
          if (u >= units) break;
          const int rg = (u % nrg);
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
            int R0;
            const int cn = tk(u, R0);
            d.y = p * NRT2 + R0 % NRT2;
            d.z = cn;
          }
          push(d);
          tq = static_cast<int>(atomicAdd(P.fc2_next + par, 1u));
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
        unsigned rn[NRT];
#pragma unroll
        for (int i = 0; i < NRT; ++i) rn[i] = 0u;
        for (; c < n; ++c) {
          const int qv = q[c % T1Q];
          const int r0 = ((qv & 0xFFFFFF) % NRT2), cn = (qv >> 24);
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
          const unsigned r = rn[i];
          if (r) red_relaxed_add(P.rt_cnt + par * NRT + i, r);
        }
      }
    }
    __syncwarp();
  } else if (warp == 6) {
    // ======================================================== B operand by bulk copies: FC1 x
    // panel images / FC2 h panel images
    int fills = 0, w = 0;
    auto bslot = [&](int st, int f) {
      if (f >= NS) mbar_wait(bar(B_EMPTY + st), ((f / NS) - 1) & 1);
    };
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
        if (!x_ready) {  // the flags loaded before the tables (acquire): usually
                         // every token is in already
          bool all = true;
#pragma unroll
          for (int i = 0; i < 16; ++i) all &= xv[i] >= static_cast<unsigned>(QU);
          if (__all_sync(0xffffffffu, all)) {
            fence_acq_rel();  // (relaxed flag loads: acquire by fence)
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
                v[i] = t0 + 32 * i < M ? ld_relaxed(xc + t0 + 32 * i) : static_cast<unsigned>(QU);
#pragma unroll
              for (int i = 0; i < 16; ++i) all &= v[i] >= static_cast<unsigned>(QU);
            }
            if (__all_sync(0xffffffffu, all)) break;
          }
          fence_acq_rel();
          if (lane == 0) fence_proxy_async_global();  // generic writes -> bulk copies (async proxy)
          x_ready = true;
        }
        if (d.x == 1) {  // routed: the panel's token rows gathered from xq / xsf2
                         // (lane g < npad / 4: rows 4 g .. 4 g + 3)
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
          if (lane < npad) {
            q0 = __ldcg(srow + d.z);
            if (d.z + 1 < d.w) q1 = __ldcg(srow + d.z + 1);
          }
          for (int s = d.z; s < d.w; ++s) {
            const int st = fills % NS;
            bslot(st, fills);
            const uint4 cur = q0;
            q0 = q1;
            if (lane < npad && s + 2 < d.w) q1 = __ldcg(srow + s + 2);
            if (lane < npad) {
              *reinterpret_cast<uint4*>(smem + OFF_SFB + st * 512 + lane * 16) = cur;
              fence_proxy_async_smem();
            }
            __syncwarp();
            if (lane == 0) {
              mbar_expect_tx(bar(B_FULL + st), (KC * nb * 128));
            }
            if (lane < ng) {
#pragma unroll
              for (int c = 0; c < KC; ++c)
                tma_gather4(sb + OFF_B + st * B_BYTES + c * nb * 128 + lane * 512, &P.mapxq,
                            (KC * s + c) * 128, tr[0], tr[1], tr[2], tr[3], bar(B_FULL + st));
            }
            ++fills;
          }
        } else if (lane == 0) {
          {  // shared: chunk s of the shared panel's per-128 x image per stage (SFB image of
             // the chunk pair)
            const int sp = d.y / NT1S, n = min(NMAX, M - NMAX * sp),
                      npad = (n + NPR - 1) & ~(NPR - 1);
            const int nb = npad, rb = 0;
            for (int s = d.z; s < d.w; ++s) {
              const int st = fills % NS;
              bslot(st, fills);
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
        const int nf2 = (RAGI && d.x == 2) ? f2t_nst(f2r0(d), cn2)
                                           : cn2 * (d.x == 4 ? NSTG2S : NSTG2);  // ring stages
        if (lane == 0) {
          if (f2s) {  // static order: wait until all of the panel's FC1 items are in (acquire)
            wait_ge(P.pan_cnt + par * MAXP + p,
                    d.x == 2 ? static_cast<unsigned>(NT1) : static_cast<unsigned>(NT1S));
          }
          fence_proxy_async_global();  // (generic h writes -> bulk copies; ready order: the
                                       // producer acquired the panel's h flag)
          const bool f2t = RAGI && d.x == 2;
          const int nts = f2t ? nf2 - cn2 * NSTG2M : 0;
          for (int f = fills; f < fills + nf2; ++f) {  // the panel's h once per row tile (L2)
            const int st = f % NS, q = f - fills;
            if (f >= NS) mbar_wait(bar(B_EMPTY + st), ((f / NS) - 1) & 1);
            int s, kcs;  // (h chunks KC s .. KC s + kcs - 1 and the scale image of stage s)
            if (f2t) {   // (F2T) tail stages: the tail chunk; main stages: chunks KC s ..
              s = q < nts ? NSTG2M : (q - nts) % NSTG2M;
              kcs = q < nts ? 1 : KC;
            } else {
              s = q % (d.x == 4 ? NSTG2S : NSTG2);
              kcs = s == (d.x == 4 ? NSTG2S : NSTG2) - 1 ? (d.x == 4 ? KC2LS : KC2L) : KC;
            }
            mbar_expect_tx(bar(B_FULL + st), kcs * nb * 128 + 512);
            for (int c = 0; c < kcs; ++c)
              bulk_g2s(sb + OFF_B + st * B_BYTES + c * nb * 128,
                       himg + (static_cast<size_t>(KC * s + c) * NMAX + rb) * 128, nb * 128,
                       bar(B_FULL + st));
            bulk_g2s(sb + OFF_SFB + st * 512, hsf + static_cast<size_t>(s) * 512, 512,
                     bar(B_FULL + st));
          }
        }
        __syncwarp();
        fills += nf2;
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
    int lsh = -1;  // kacc of the last shared FC1 item (its buffer release frees the up accumulator)
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
      const int n = panel_n(d);
      const uint32_t npad = warp_uniform(static_cast<uint32_t>((n + 7) & ~7));
      const bool shk = d.x == 3 || d.x == 4;  // shared expert: e4m3 A
      const uint32_t id0 = idesc_f4f8(static_cast<int>(npad), 0, shk ? 0u : 5u);
      const bool fc1 = (d.x & 1) != 0;
      const int nacc = fc1 ? 1 : f2n(d), nst = fc1 ? d.w - d.z : (d.x == 4 ? NSTG2S : NSTG2);
      const int kcl = d.x == 4 ? KC2LS : KC2L;
      if (RAGI && d.x == 2) {  // (F2T) routed FC2 unit: tail stages open the row tiles'
                               // accumulators (one K32 MMA per tail block), main stages accumulate
        const int cn = nacc, r0 = f2r0(d);
        int c_lo, nch, nts;
        f2t_unit(r0, cn, c_lo, nch, nts);
        int st = fills % NS;
        uint32_t ph = static_cast<uint32_t>(fills / NS) & 1u;
        const uint64_t bstep = static_cast<uint64_t>((npad * 128) >> 4);
        for (int t = 0; t < nts; ++t) {
          const int c = c_lo + 2 * t, nv = min(2, nch - 2 * t);
          const int j0 = max(0, c * RPC - r0), j1 = min(cn, (c + nv) * RPC - r0);
          {  // (the accumulators this stage opens: free?  The epilogue releases them in order: the
             // last one's release implies the others')
            const int k = kacc + j1 - 1;
            if (k >= ACC) mbar_wait(bar(B_ACCE + k % ACC), ((k / ACC) - 1) & 1);
          }
          mbar_wait(bfull0 + 8 * st, ph);
          tc_fence_after();
          const uint32_t tsf = tsf0 + st * TSF_STRIDE;
          const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)),
                         bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
          if (elect_one()) {
            for (int cl = 0; cl < nv; ++cl)
              tc_cp_sf(tsf + 4 * cl, sfadesc0 + static_cast<uint64_t>((st * KC + cl) * (512 >> 4)));
            tc_cp_sf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
            for (int j = j0; j < j1; ++j) {
              const int rr = r0 + j, cl = rr / RPC - c, kq0 = (rr % RPC) * TB;
              const uint32_t dj = tmem_u + static_cast<uint32_t>((kacc + j) % ACC) * ACC_COLS;
#pragma unroll
              for (int tb = 0; tb < TB; ++tb)
                tc_mma(dj, ad0 + cl * (16384 >> 4) + 2 * (kq0 + tb), bd0 + 2 * tb,
                       id0 | (static_cast<uint32_t>(kq0 + tb) << 29) | (tb << 4), tsf + 4 * cl,
                       tsf + TSF_B, tb != 0);
            }
            tc_commit(bempty0 + 8 * st);
          }
          __syncwarp();
          if (++st == NS) {
            st = 0;
            ph ^= 1u;
          }
        }
        for (int j = 0; j < cn; ++j) {
          const uint32_t dj = tmem_u + static_cast<uint32_t>((kacc + j) % ACC) * ACC_COLS;
          for (int s = 0; s < NSTG2M; ++s) {
            mbar_wait(bfull0 + 8 * st, ph);
            tc_fence_after();
            const uint32_t tsf = tsf0 + st * TSF_STRIDE;
            const uint64_t ad0 = adesc0 + static_cast<uint64_t>(st * (A_BYTES >> 4)),
                           bd0 = bdesc0 + static_cast<uint64_t>(st * (B_BYTES >> 4));
            if (elect_one()) {
#pragma unroll
              for (int cc = 0; cc < KC; ++cc)
                tc_cp_sf(tsf + 4 * cc,
                         sfadesc0 + static_cast<uint64_t>((st * KC + cc) * (512 >> 4)));
              tc_cp_sf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
#pragma unroll
              for (int cc = 0; cc < KC; ++cc)
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                  tc_mma(dj, ad0 + cc * (16384 >> 4) + 2 * kb, bd0 + cc * bstep + 2 * kb,
                         id0 | (kb << 4) | (kb << 29), tsf + 4 * cc, tsf + TSF_B + 2 * cc, 1u);
              tc_commit(bempty0 + 8 * st);
              if (s == NSTG2M - 1) tc_commit(bar(B_ACCF + (kacc + j) % ACC));
            }
            __syncwarp();
            if (++st == NS) {
              st = 0;
              ph ^= 1u;
            }
          }
        }
        fills += nts + cn * NSTG2M;
        kacc += cn;
        continue;
      }
      for (int a = 0; a < nacc; ++a) {  // FC1: one accumulator; FC2: one per row tile of the group
        const int ab = kacc % ACC;
        if (kacc >= ACC) {
          mbar_wait(bar(B_ACCE + ab), ((kacc / ACC) - 1) & 1);
        }
        tc_fence_after();
        const uint32_t dacc = tmem_u + ab * ACC_COLS;
        const uint32_t dup =
            tmem_u + static_cast<uint32_t>(ACC * ACC_COLS);  // shared FC1 up accumulator
        if (d.x == 3) {  // (the up accumulator is free once the previous shared item's buffer was
                         // released: that release follows its up read)
          static_assert((ACC * ACC_COLS + 32 <= TSF0), "SHUP: TMEM region, MMA paths");
          if (lsh >= 0 && kacc - lsh < ACC)
            mbar_wait(bar(B_ACCE + lsh % ACC), static_cast<uint32_t>(lsh / ACC) & 1u);
          lsh = kacc;
        }
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
                      tc_mma((shh ? dup : dacc), ad0 + c * (16384 >> 4) + 2 * kb,
                             bd0 + c * bstep + 2 * kb, id0 | (kb << 4) | (kb << 29), tsf + 4 * c,
                             tsf + TSF_B + 2 * c, (j | c | kb) != 0);
                } else
#pragma unroll
                  for (int kb = 0; kb < 4; ++kb) {
                    const uint32_t idk = id0 | (kb << 4) | (kb << 29), acc = (j | kb) != 0;
                    tc_mma(dacc, ad0 + 2 * kb, bd0 + 2 * kb, idk, tsf, sfb, acc);
                    tc_mma(dup, ad0 + (16384 >> 4) + 2 * kb, bd0 + 2 * kb, idk, tsf + 4, sfb, acc);
                  }
                tc_commit(bempty0 + 8 * st);
              }
              __syncwarp();
              if (++st == NS) {
                st = 0;
                ph ^= 1u;
              }
            }
          } else if ((M > 8 && !fc1 && nst == 2)) {
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
                    if (jp + h == nst - 1 && c >= kcl) break;  // (ragged last FC2 stage)
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
                tc_cp_sf(tsf + TSF_B, sfbdesc0 + static_cast<uint64_t>(st * (512 >> 4)));
#pragma unroll
                for (int c = 0; c < KC; ++c) {
                  if (!fc1 && j == nst - 1 && c >= kcl) break;  // (ragged last FC2 stage)
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
      const bool epi2 =
          d.x == 2 &&
          n <= 16;  // (n <= 16: row tiles alternate between the epilogue groups, tokens 0..15 each)
      const int tb2 = epi2 ? 0 : tb;
      for (int a = 0; a < nacc; ++a) {
        const int ab = kacc % ACC;
        if (!isdry) mbar_wait(bar(B_ACCF + ab), (kacc / ACC) & 1);
        tc_fence_after();
        float v[16], vu[16];
        const uint32_t ta = tmem + (static_cast<uint32_t>(32 * q) << 16) + ab * ACC_COLS + (tb2);
        if ((epi2 &&
             (a & 1) !=
                 g)) {  // (the other group's row tile: free the buffer's use, nothing to read)
          __syncwarp();
          if (!isdry) {
            if (lane == 0) mbar_arrive(bar(B_ACCE + ab));
            ++kacc;
          }
          continue;
        }
        tc_ld16_wait(ta, v);  // (columns past npad: unused garbage)
        if (d.x == 3)
          tc_ld16_wait(tmem + (static_cast<uint32_t>(32 * q) << 16) + ACC * ACC_COLS + tb2,
                       vu);  // (shared FC1: the up accumulator)
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
          if ((TPMOE_EPR && d.x == 2)) {  // (TPMOE_EPR) register transpose
            const int vl = (TPMOE_EPRSGL == 2 && d.x == 2)
                               ? 4 * (lane & 7) + (lane >> 3)
                               : lane;  // (TPMOE_EPRSGL 2: the row this TMEM lane holds)
            const int pr = vl & 1, e4 = (vl >> 1) & 3, m8 = vl >> 3;
            uint32_t
                wv[8];  // (step 1) wv[i]: rows (lane & ~1, lane | 1) of token i + 8 pr as bf16x2
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const float snd = pr ? v[i] : v[i + 8];
              const float rcv =
                  __shfl_xor_sync(0xffffffffu, snd, (TPMOE_EPRSGL == 2 && d.x == 2) ? 8 : 1);
              wv[i] = pr ? pack_bf16x2(rcv, v[i + 8]) : pack_bf16x2(v[i], rcv);
            }
#pragma unroll
            for (int bk = 0; bk < 2; ++bk) {  // (step 2) M[e4][t] = wv[4 bk + t] (row pair e4 of
                                              // token 4 bk + t) -> lane e4: M[0..3][e4]
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const uint32_t x = (e4 & 1) ? wv[4 * bk + 2 * j] : wv[4 * bk + 2 * j + 1];
                const uint32_t y =
                    __shfl_xor_sync(0xffffffffu, x, (TPMOE_EPRSGL == 2 && d.x == 2) ? 16 : 2);
                wv[4 * bk + 2 * j] = (e4 & 1) ? y : wv[4 * bk + 2 * j];
                wv[4 * bk + 2 * j + 1] = (e4 & 1) ? wv[4 * bk + 2 * j + 1] : y;
              }
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const uint32_t x = (e4 & 2) ? wv[4 * bk + j] : wv[4 * bk + j + 2];
                const uint32_t y =
                    __shfl_xor_sync(0xffffffffu, x, (TPMOE_EPRSGL == 2 && d.x == 2) ? 1 : 4);
                wv[4 * bk + j] = (e4 & 2) ? y : wv[4 * bk + j];
                wv[4 * bk + j + 2] = (e4 & 2) ? wv[4 * bk + j + 2] : y;
              }
            }
            if (!isdry) {
#pragma unroll
              for (int bk = 0; bk < 2; ++bk) {
                const int i = 4 * bk + e4 + 8 * pr;  // (this lane's token of the group's 16)
                if (tb2 + i >= n) continue;
                const unsigned ev = T.ent[eb + tb2 + i];
                const int t = static_cast<int>(ev & 511u);
                const uint4 val =
                    make_uint4(wv[4 * bk], wv[4 * bk + 1], wv[4 * bk + 2], wv[4 * bk + 3]);
                for (unsigned km = ev >> 9; km; km &= km - 1)
                  st_v4_hint(P.outp +
                                 (static_cast<size_t>(t) * TOPK_MAX + (__ffs(km) - 1)) * OUTP_LD +
                                 r * 128 + 32 * q + 8 * m8,
                             val);
              }
            }
            continue;
          }
          __nv_bfloat16* const stgx = stg;
          for (int i = 0; i < 16; ++i)
            if (tb2 + i < n) stgx[i * 128 + orow] = __float2bfloat16_rn(v[i]);
          named_bar(1 + g, 128);
          for (int x = gt; x < 16 * 16; x += 128) {  // (token i, 16-B chunk c)
            const int i = x >> 4, c = x & 15;
            if (tb2 + i >= n) break;
            const uint4 val = *reinterpret_cast<const uint4*>(stgx + i * 128 + 8 * c);
            if (isdry) continue;
            if (d.x == 4) {
              st_v4_hint(P.shout + static_cast<size_t>(NMAX * p + tb + i) * DIM + r * 128 + 8 * c,
                         val);
              continue;
            }
            const unsigned ev = T.ent[eb + tb2 + i];
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
          const int ng = min(16, n - tb), kq = item % NT1S;
          float am_mine = 0.f;
          float r[16];  // (unprocessed gate values at the front, finished activations appended at
                        // the back)
#pragma unroll
          for (int i = 0; i < 16; ++i) r[i] = v[i];
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
          if (lane < 16 && tb + lane < NMAX) T.am[q * NMAX + tb + lane] = am_mine;
          named_bar(1 + g, 128);
          uint8_t* hb = P.hsimg + (static_cast<size_t>(p) * NCH2 + kq) * NMAX * 128;
#pragma unroll 1
          for (int i = 0; i < ng; ++i) {
            const float a = r[0];
#pragma unroll
            for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
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
          if (pub) {
            const bool lastit =
                atom_add_acq_rel(P.pan_cnt + par * MAXP + Ptot + p, 1u) == NT1S - 1u;
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
        const int ng = min(16, n - tb);  // this group's live tokens (warp-uniform; <= 0: none)
        float am_mine = 0.f;             // lane i: the warp's amax of token tb + i
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
              const float gg = gv, uu = x;
              a2 = act_mul(gg, uu);
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
#else
        float r[16];  // unprocessed values at the front, finished activations appended at the back
#pragma unroll
        for (int i = 0; i < 16; ++i) r[i] = v[i];
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
          const float x = r[0];
          const float gv = __shfl_xor_sync(0xffffffffu, x, 8);
          float a2 = 0.f;
          if (up_lane) {
            const float gg = gv, uu = x;
            a2 = act_mul(gg, uu);
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
#pragma unroll 1
        for (int i = 0; i < ng; ++i) {
          const float a = r[0];
#pragma unroll
          for (int k = 0; k < 15; ++k) r[k] = r[k + 1];
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
        if (pub && !isdry) {
          const bool lastit = atom_add_acq_rel(P.pan_cnt + par * MAXP + p, 1u) == NT1 - 1u;
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
  tail(P, T, tid, epoch, par, smem);
  __syncthreads();
  if (tid == 0) {
    P.work[64 + b] = epoch;  // this CTA's epoch
  }
}

}  // namespace flashinfer::tp_moe::sm100::generic
