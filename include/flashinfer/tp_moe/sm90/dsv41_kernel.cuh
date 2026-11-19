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
// Fused tensor-parallel MoE layer for DeepSeek-V4.1-Flash decode on SM90a: hidden 5120, 384 routed
// experts (intermediate 2304 split over TP_MOE_NDEV = 4 | 8 ranks, MXFP4 weights, FP8 activations,
// top-6), the TP-sliced FP8 shared expert (32 x 32 block scales) and the cross-rank all-reduce in
// one persistent kernel.  One CTA per SM, 640 threads: 4 consumer warpgroups + 1 producer
// warpgroup.
//   prologue  fused gate (sqrtsoftplus + bias, top-6, renormalized; image tokens with their own
//   bias) of the router logits, or the
//             given ids / weights; per-token FP8 quantization of x (and its per-128-group
//             quantization for the shared expert); routing tables that every CTA builds
//             identically: the selected experts (ascending id), their token entries (ascending
//             token), panels of <= 32 entries.
//   FC1       (panel, 128-column group) items of 40 k-tiles, stream-K over all CTAs by cost; the
//   CTA owning an item's k-tile 0 adds
//             the other CTAs' fp32 partials in CTA order and writes the BF16 SwiGLU-clamp
//             activation and its per-token amax.  The per-rank intermediate (576 | 288) is not a
//             multiple of 128: its last item reads its own 64-row tail blocks (at TP8 the 32-row
//             tail of narrow panels runs split 4 ways over K).
//   FC2       (panel, 256-row tile) units from a global queue; a panel is quantized per token once
//   its FC1 items are done; the BF16
//             output goes to every (token, top-k slot) of its expert.
//   tail      T1 per (row tile, token): slot-ordered fp32 sum of w_k * y_k, x routed_scaling_factor
//   -> BF16, + shared -> BF16, pushed
//             to the row tile's owner rank (one-shot at small M: to every rank); T2: rank-ordered
//             fp32 sum -> BF16, pushed to all ranks; T3: pull -> output.  Messages carry the launch
//             epoch; the exchange buffers are double-buffered by its parity.
// k-steps: MXFP4 -> e4m3 register dequant of the Humming fused-E8M0 weight layout pipelined with
// e4m3 wgmma (kpipe*.cuh).  The fused shared expert runs as extra panels with SGLang's
// shared-expert numerics (shared_num.cuh) on SS-wgmma k-steps. Results are bitwise deterministic
// (fixed reduction orders) and identical on every rank. Serves up to MAX_TOKENS = 64 tokens per
// call.  Compile-time variant (one build per TP size, see flashinfer/fused_moe/_tp_moe_sm90.py):
// TP_MOE_NDEV.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "chain_ss1.cuh"
#include "kpipe.cuh"
#include "kpipe1.cuh"
#include "shared_num.cuh"

namespace flashinfer {
namespace tp_moe {
namespace sm90 {
namespace dsv41 {

#ifndef TP_MOE_NDEV
#define TP_MOE_NDEV 4  // tensor-parallel ranks: 4 (per-rank intermediate 576) or 8 (288)
#endif
constexpr int NDEV = TP_MOE_NDEV;
static_assert(NDEV == 4 || NDEV == 8, "TP4 or TP8");
constexpr int DIM = 5120, NEXP = 384, TOPK_MAX = 8, MAXM = 512;
constexpr int MAX_TOKENS =
    64;  // tokens per call the op serves (MAXM: the workspace / message capacity)
constexpr int INTER_R = 2304 / NDEV;  // per-rank intermediate (576 | 288)
constexpr int NQG =
    (INTER_R + 127) /
    128;  // FC1 items (128-column groups q) per panel (5 | 3: q = NQG - 1 the tail item)
constexpr int INTER =
    NQG * 128;  // act / h columns per entry (640 | 384: columns INTER_R.. are zero)
constexpr int INTER_M = (INTER_R / 128) * 128;  // main w13 rows per stack (512 | 256)
constexpr bool TAIL =
    INTER != INTER_R;  // the tail item q = NQG - 1 (own weight maps / global scale)
#ifndef TP_MOE_TAIL4
#define TP_MOE_TAIL4 \
  (TP_MOE_NDEV == 8)  // TP8: the N8 tail items as one 64-row block split 4 ways over K (maps 6, 7)
#endif
constexpr bool T4 = TAIL && TP_MOE_TAIL4;
constexpr int MAXE = MAXM * TOPK_MAX;  // entries (token, expert) upper bound
constexpr int MAXP = 512;              // panels upper bound (<= NEXP + MAXE / NMAX)
static_assert(NEXP + MAXE / 32 <= MAXP, "panels bound");
constexpr int NKT1 = DIM / 128;  // FC1 k-tiles per item (40)
constexpr int NKT2 =
    NQG;  // FC2 k-tiles per unit (5 | 3; the last one holds INTER_R % 128 real columns)
constexpr int NSL2 = TAIL ? INTER_R / 32 : INTER / 32;  // w2 k32 slices per expert (18 | 9)
constexpr int CPT = INTER / 16;                         // 16-column act chunks per token (40 | 24)
constexpr int CPE = NKT1 * 8;                           // x gather: 16-B chunks per entry (320)
constexpr int NRT = DIM / 256;                          // FC2 row tiles (20)
static_assert(NRT <= 32, "row-tile bit sets");
constexpr int NSLOT =
    (NRT + NDEV - 1) /
    NDEV;  // reduce-scatter slots per (source, token) at an owner (5 | 3: rtile / NDEV)
// reduce-scatter owner of (row tile, token): (rtile + t) % NDEV -- every owner gets ~NRT / NDEV row
// tiles of every token
__host__ __device__ __forceinline__ int rs_owner(int rtile, int t) {
  return (rtile + t) & (NDEV - 1);
}
// panel info word: e | n << PI_NS | ebase << PI_BS
constexpr int PI_NS = NEXP > 256 ? 9 : 8, PI_BS = PI_NS + 6;
__host__ __device__ __forceinline__ uint32_t pinf_pack(uint32_t e, uint32_t n, uint32_t b) {
  return e | (n << PI_NS) | (b << PI_BS);
}
__device__ __forceinline__ int pinf_e(uint32_t v) { return (int)(v & ((1u << PI_NS) - 1u)); }
__device__ __forceinline__ int pinf_n(uint32_t v) { return (int)((v >> PI_NS) & 63u); }
__device__ __forceinline__ int pinf_b(uint32_t v) { return (int)(v >> PI_BS); }
// FC2 unit word (s_unit): rtile | n << SU_NS | e << SU_ES | pid << SU_PS
constexpr int SU_NS = NRT > 16 ? 5 : 4, SU_ES = SU_NS + 6, SU_PS = SU_ES + PI_NS;
static_assert(SU_PS + 10 <= 31, "unit word");
constexpr int NMSG = 43;  // LL messages per (row tile, token): 6 bf16 + tag each, last carries 4
constexpr int NCONS = 512, NTHREADS = 640, PROD_WARP = 16, GATHER_WARP = 17;
constexpr int NMAX = 32;   // token panel width (wgmma N <= 32)
constexpr int HLB = 1024;  // one 64-row x k32 Humming block
constexpr int STAGE_W = 16384, STAGE_S = 1024,
              STAGE_WS = STAGE_W + STAGE_S;  // [4 k32 slices][4 blocks][1 KB] + [4][4 x 64 B]
#ifndef TP_MOE_T4N16
#define TP_MOE_T4N16 \
  (TP_MOE_NDEV == 8)  // TP8: the T4 split tail also for 9..16-token panels (x tiles of 4 k-chunks x
                      // 16 rows = 8 KB per stage; NST 6)
#endif
constexpr int STAGE_X =
    TP_MOE_T4N16 ? 2 * NMAX * 128
                 : NMAX * 128;  // FC1 x tile: [32 tokens][128 B] (128B swizzle) (T4N16: 8 KB)
constexpr int STAGE = STAGE_WS + STAGE_X;  // 21504 (multiple of 1024)
constexpr int KPS = 1;  // k-tiles per ring stage (one full / empty mbarrier round trip each);
                        // stream-K ranges are KPS-aligned
constexpr int BALIGN =
    T4 ? (KPS > 4 ? KPS : 4) : KPS;  // stream-K bound alignment (T4: a tail ring stage = 4 k-tiles)
constexpr int NST =
    KPS == 2 || TP_MOE_T4N16
        ? 6
        : 7;  // ring slots (one k-tile each); barrier stages = NST / KPS (slot pairs)
constexpr int OFF_STG = NST * STAGE;  // epilogue staging: FC1 up exchange [2][128][NMAX/2] fp32 +
                                      // act tile [NMAX][128] bf16 | FC2 out tile [NMAX][256] bf16
constexpr int T4_SCR = 16384;  // (T4 epilogue scratch: fp32 [4 warpgroups][R <= 8][128] over the
                               // act tile -- written in stage 2, after the scratch is read)
constexpr int T4MAXN =
    TP_MOE_T4N16 ? 16 : 8;  // widest panel (token count) whose tail item runs split 4 ways over K
constexpr int ACT_PITCH =
    136;  // act tile row pitch (bf16): 272 B rows, 16 B apart in banks (no 4-way conflicts)
constexpr int OUT_PITCH = 264;                  // FC2 out tile row pitch (bf16): 528 B rows
constexpr int OUT_TILE = NMAX * OUT_PITCH * 2;  // 16896
constexpr int STG_BYTES =
    2 * OUT_TILE;  // FC2: two out tiles (consumers fill one while the store warps drain the other)
static_assert(16384 + NMAX * ACT_PITCH * 2 <= STG_BYTES,
              "FC1 up exchange + act tile fit the staging area");
constexpr int OFF_TAB = OFF_STG + STG_BYTES;
constexpr int TAB_IDS =
    0;  // uint16 [MAXM][8] expert of (t, k) (row padded to 8 = one 16-B load), 0xFFFF = invalid
constexpr int TAB_ENT = TAB_IDS + MAXE * 2;   // uint16 [MAXE] token of entry
constexpr int TAB_UEXP = TAB_ENT + MAXE * 2;  // uint16 [NEXP] union expert of slot
constexpr int TAB_OFF = TAB_UEXP + NEXP * 2;  // uint16 [NEXP + 8] entry offset of slot (+ total)
constexpr int TAB_PID =
    TAB_OFF + (NEXP + 8) * 2;  // uint16 [NEXP + 8] first panel of slot (+ total)
constexpr int TAB_P2U = TAB_PID + (NEXP + 8) * 2;  // uint16 [MAXP] slot of panel
constexpr int TAB_KM = TAB_P2U + MAXP * 2;         // uint8 [MAXE] top-k slots (bit mask) of entry
constexpr int TAB_E2P = TAB_KM + MAXE;             // uint16 [MAXE] panel of entry
constexpr int TAB_FS = TAB_E2P + MAXE * 2;         // float [NMAX] per-token epilogue factor
constexpr int TAB_AM = TAB_FS + NMAX * 4;          // float [NMAX][16] amax partials (per warp)
constexpr int TAB_MISC =
    TAB_AM + NMAX * 16 * 4;  // int [32]: 0 U, 1 entries, 2 panels, 3 queued quantizations
constexpr int TAB_CCUM =
    TAB_MISC + 32 * 4;  // uint32 [MAXP + 1] exclusive FC1 cost prefix over panels
constexpr int TAB_PROW =
    TAB_CCUM +
    (MAXP + 8) * 4;  // uint32 [MAXP + 1] exclusive prefix of padded panel widths (xg rows)
constexpr int TAB_PINF =
    TAB_PROW + (MAXP + 8) * 4;  // uint32 [MAXP] panel info: e | n << 8 | ebase << 14
constexpr int MAXG = 160;
constexpr int TAB_BND =
    TAB_PINF +
    MAXP * 4;  // int [2][MAXG + 1] FC1 stream-K range starts of every CTA, per panel group
constexpr int TAB_QQ = TAB_BND + (2 * (MAXG + 1) + 6) * 4;  // panel-ready bits / scratch
constexpr int TAB_UNIT =
    TAB_QQ + 96 * 4;  // int2 [8] FC2 unit ring (producer -> consumers), int [NRT] units done per
                      // row tile, int4 [2] out tile info (QQ: panel-ready bits [16] + the store
                      // warps' pending row-tile counts [16 + NRT] / [48 + NRT])
constexpr int TAB_BAR =
    TAB_UNIT +
    (16 + NRT + 8) *
        4;  // uint64 full[NST], empty[NST], epi[4], hdone, gbar, fullo[2], freeo[2], sdone
constexpr int TAB_RSV =
    TAB_BAR +
    (2 * NST + 12) * 8;  // (+ xsb: dense fused-quant token scales in TAB_XS); 384 B reserved
constexpr int TAB_GRW =
    TAB_RSV + 48 * 8;  // float [128] fused gate, small tables: routing weight of (t, k)
constexpr int TAB_XS =
    TAB_GRW + 128 * 4;  // float [NMAX] dense + fused quantization: per-token x scale (warp 19, xsb)
constexpr int TAB_BYTES = TAB_XS + NMAX * 4;
#ifndef TP_MOE_COST
#define TP_MOE_COST \
  100, 104, 109, 132  // relative FC1 k-tile cost by panel width class N8 / N16 / N24 / N32
#endif
#define TP_MOE_A0(a, b, c, d) a
#define TP_MOE_A1(a, b, c, d) b
#define TP_MOE_A2(a, b, c, d) c
#define TP_MOE_A3(a, b, c, d) d
#define TP_MOE_APPLY(m, x) m(x)
constexpr int k_cost0 = TP_MOE_APPLY(TP_MOE_A0, TP_MOE_COST),
              k_cost1 = TP_MOE_APPLY(TP_MOE_A1, TP_MOE_COST),
              k_cost2 = TP_MOE_APPLY(TP_MOE_A2, TP_MOE_COST),
              k_cost3 = TP_MOE_APPLY(TP_MOE_A3, TP_MOE_COST);
#ifndef TP_MOE_EPI_COST
#define TP_MOE_EPI_COST \
  250, 300, 400,        \
      500  // owner epilogue cost of one FC1 item by width class (same units: 100 = one N8 k-tile)
#endif
#ifndef TP_MOE_EPI_SCALE
#define TP_MOE_EPI_SCALE 300  // percent; compact panels only (dense panels: 0)
#endif
constexpr int k_epi0 = TP_MOE_APPLY(TP_MOE_A0, TP_MOE_EPI_COST),
              k_epi1 = TP_MOE_APPLY(TP_MOE_A1, TP_MOE_EPI_COST),
              k_epi2 = TP_MOE_APPLY(TP_MOE_A2, TP_MOE_EPI_COST),
              k_epi3 = TP_MOE_APPLY(TP_MOE_A3, TP_MOE_EPI_COST);
__device__ __forceinline__ uint32_t sel4(int cl, int a, int b, int c, int d) {
  return (uint32_t)(cl == 0 ? a : (cl == 1 ? b : (cl == 2 ? c : d)));
}  // immediates, no constant-bank load
__device__ __forceinline__ uint32_t c_cost(int cl) {
  return sel4(cl, k_cost0, k_cost1, k_cost2, k_cost3);
}
// stream-K cost of one panel of width class cl: NQG items x (owner epilogue + 32 k-tiles); a
// position x (cost units from the panel start) maps to the item-major k-tile whose range it falls
// in (the epilogue cost sits before an item's k-tile 0: its owner pays it)
__device__ __forceinline__ uint32_t c_epi(int cl, bool dense) {
  return dense ? 0u
               : sel4(cl, k_epi0 * TP_MOE_EPI_SCALE / 100, k_epi1 * TP_MOE_EPI_SCALE / 100,
                      k_epi2 * TP_MOE_EPI_SCALE / 100, k_epi3 * TP_MOE_EPI_SCALE / 100);
}
#ifndef TP_MOE_ONESHOT_MAXM_DEF
#define TP_MOE_ONESHOT_MAXM_DEF \
  (TP_MOE_NDEV == 8 ? 8 : 32)  // one-shot all-reduce for M <= this, else RS + AG
#endif
#ifndef TP_MOE_SH_COST
#define TP_MOE_SH_COST \
  170  // percent: a fused shared expert k-block (2 x 16 KB FP8 weights + x) vs a routed k-tile (8
       // KB MXFP4) of its width class
#endif
__device__ __forceinline__ uint32_t c_cost_x(int cl, bool sh) {
  return sh ? c_cost(cl) * TP_MOE_SH_COST / 100u : c_cost(cl);
}
#ifndef TP_MOE_T4_COST
#define TP_MOE_T4_COST \
  25  // stream-K cost of a T4 tail k-unit, % of a k-tile of the panel's width class (4 k-units =
      // one ring stage)
#endif
// df: bit 0 dense mode, bit 1 the wide (N16) T4 tail allowed for this launch (dflag)
__device__ __forceinline__ uint32_t panel_cost(int cl, int df, bool sh = false) {
  const bool dense = df & 1;
  if (T4 && cl <= ((df & 2) ? T4MAXN / 8 - 1 : 0) && !sh)
    return (uint32_t)NQG * c_epi(cl, dense) +
           (uint32_t)NKT1 * c_cost_x(cl, sh) * (100u * (NQG - 1) + (uint32_t)TP_MOE_T4_COST) /
               100u;  // (T4: a tail item k-unit costs TP_MOE_T4_COST % of a k-tile)
  return (uint32_t)NQG * (c_epi(cl, dense) + (uint32_t)NKT1 * c_cost_x(cl, sh));
}
__device__ __forceinline__ int panel_kt(uint32_t x, int cl, int df, bool sh = false) {
  const bool dense = df & 1;
  const uint32_t e = c_epi(cl, dense), cc = c_cost_x(cl, sh), ic = e + (uint32_t)NKT1 * cc;
  const uint32_t q = min((uint32_t)(NQG - 1), x / ic), y = x - q * ic;
  if (T4 && cl <= ((df & 2) ? T4MAXN / 8 - 1 : 0) && !sh && q == NQG - 1)
    return (int)(q * (uint32_t)NKT1 +
                 (y < e ? 0u
                        : min((uint32_t)NKT1 - 1u,
                              (y - e) * 100u / (cc * (uint32_t)TP_MOE_T4_COST)))) &
           ~(BALIGN - 1);
  return (int)(q * (uint32_t)NKT1 + (y < e ? 0u : min((uint32_t)NKT1 - 1u, (y - e) / cc))) &
         ~(BALIGN - 1);  // (KPS-aligned: a ring stage never spans two pieces)
}
constexpr int SMEM_TOTAL = OFF_TAB + TAB_BYTES;
constexpr int SMEM_ALLOC = SMEM_TOTAL + 1024;
static_assert(STAGE % 1024 == 0 && OFF_STG % 1024 == 0, "swizzle atoms");
static_assert(SMEM_ALLOC <= 232448, "smem budget");
static_assert(!T4 || (T4_SCR + 4 * (T4MAXN / 2) * 128 * 4 <= STG_BYTES && KPS == 1 &&
                      4 * T4MAXN * 128 <= STAGE_X),
              "T4 epilogue scratch / one k-tile stages / 4 x tiles per stage");
static_assert(TAB_BAR % 8 == 0 && (TAB_UNIT + (16 + NRT) * 4) % 16 == 0,
              "mbarrier / int4 alignment");
// prologue scratch (inside the ring, dead before the first TMA)
constexpr int PRO_BITS = 0;                        // uint32 [NEXP][16] token bits
constexpr int PRO_PRE = PRO_BITS + NEXP * 16 * 4;  // uint16 [NEXP][16] exclusive word prefix
constexpr int PRO_CNT = PRO_PRE + NEXP * 16 * 2;   // int [NEXP]
constexpr int PRO_OFFE = PRO_CNT + NEXP * 4;       // int [NEXP] entry offset by expert
constexpr int PRO_WS = PRO_OFFE + NEXP * 4;        // int [72 | DSV4.1 96] warp sums
constexpr int PRO_PIDE = PRO_WS + 96 * 4;          // int [NEXP] first panel of expert
// warp-sum scratch layout (ws): per-warp (a, b, d) at [w], [WSS + w], [2 WSS + w]; the cost prefix
// pair at [WS2A + w], [WS2D + w]
constexpr int WSS = NEXP / 32 > 8 ? 16 : 8, WS2A = 48, WS2D = 64, WS_RED = 80;
// dense small-M tables: one expert per thread (BT_DENSE threads, named barrier 2); the other
// consumer threads quantize x (small tables)
constexpr int BT_DENSE = NEXP, NQ_SMALL = 512 - BT_DENSE;
constexpr int GS_N = BT_DENSE;  // fused gate, small tables: gating threads (named barrier 2)
static_assert(PRO_PIDE + NEXP * 4 <= OFF_STG, "prologue scratch fits in the ring");
static_assert((unsigned long long)MAXP * NQG * (600 + NKT1 * 200) * 160 < (1ull << 32),
              "32-bit stream-K cost math (k-tile cost <= 200, epilogue <= 600, <= 160 CTAs)");

struct alignas(64) Params {
  CUtensorMap xmap;  // dense mode: x [M][DIM] e4m3, box {128 B, md rows}, 128B swizzle (OOB rows
                     // zero-filled)
  const CUtensorMap* tmaps;  // [0] w13 5-D, [1] s13 4-D, [2] w2 4-D, [3] s2 4-D, [4] w13 tail 4-D,
                             // [5] s13 tail 3-D, [6, 7] the split tail
  const float* fc1_s2;
  const float* fc1t_s2;   // the tail item's global scales
  const float* fc1t4_s2;  // (T4) the split tail block's global scales
  const float* fc2_s2;
  const uint8_t* xq;        // [M][DIM] e4m3
  const float* xs;          // [M] per-token x scale
  const __nv_bfloat16* xb;  // BF16 x [M][DIM], quantized in the kernel into xq / xs
  const float* glog;        // fused gate: router logits fp32 [M][NEXP] (ids / rw then point at the
                            // workspace), else null
  const float* gbias;       // fused gate: score correction bias [NEXP]
  unsigned* gcnt;           // [2] fused gate (block-wide tables): gated tokens, per epoch parity
  unsigned* xflag;          // [MAXM] fused quantization: token t quantized (tag: the launch epoch)
  const int* ids;           // [M][topk]
  const float* rw;          // [M][topk]
  int topk;
  float factor;
  __nv_bfloat16* act;    // [MAXE][INTER]
  float* amax;           // [MAXE][NQG]
  float* part;           // [grid][NCONS][NMAX/2]
  uint8_t* hq;           // [MAXP][NKT2 k-tiles][NMAX][128] quantized FC2 input (128B-swizzled rows)
  uint8_t* xg;           // [rows][NKT1 k-tiles] x gathered per panel: panel p at row prow[p],
                         // [kt][Npad][128] swizzled
  unsigned* xg_cnt;      // [2][MAXP] CTAs done gathering each panel's x rows (epoch parity)
  unsigned* fc2_next;    // [2] FC2 unit queue head (epoch parity)
  float* hs;             // [MAXP][NMAX] per-token FC2 input scale
  unsigned* h_flags;     // [MAXP] epoch tag: panel quantized
  unsigned* pan_cnt;     // [2][MAXP] finished FC1 items per panel (epoch parity)
  unsigned* part_flags;  // [grid] epoch tags
  unsigned* rt_cnt;      // [2][NRT] FC2 row-tile completion counters (epoch parity)
  unsigned* work;        // [0] epoch, [1 + parity] CTA done counters
  __nv_bfloat16* outp;   // [MAXM][TOPK_MAX][DIM] per-slot expert outputs
  uint4* rs[NDEV];       // peer reduce-scatter buffers [2 parity][NDEV src][NSLOT][MAXM][NMSG]
  uint4* ag[NDEV];       // peer all-gather buffers [2 parity][NRT rtile][MAXM][NMSG]
  __nv_bfloat16* out;    // [M][DIM]
  int M, n_exp, my_rank;
  int dense,
      md;       // dense mode (small M): FC1 B operand = x rows 0..md-1 for every expert, no gather
  int groups;   // FC1 panel groups (1 or 2): host-chosen by M
  int oneshot;  // all-reduce: 1 = one-shot push to all ranks (M <= TP_MOE_ONESHOT_MAXM_DEF), 0 = RS
                // + AG
  // fused shared expert: the TP slice of the FP8 shared expert, used in place
  CUtensorMap w1s, w2s,
      xgs;  // 2D uint8 maps, 128B swizzle: gate_up [2 INTER][DIM] / down [DIM][INTER] box {128 B,
            // 128 rows}; xqg [M][DIM] box {128 B, 32 rows} (OOB rows zero)
  const float* sb1;  // gate_up block scales, fp32 per (32 rows, 32 k) (zero padding: 1.0)
  const float* sb2;  // down block scales, fp32 per (32 rows, 32 k) (zero padding: 1.0)
  uint8_t* xqg;      // [MAXM][DIM] e4m3: x per (token, 128-group) (SGLang per_token_group_quant)
  float*
      xsT;  // [DIM / 128][MAXM] its scales, group-major (one 128-B bulk copy per (k-block, panel))
  float* hs4;            // [MAXP][NKT2][NMAX] shared FC2 input group scales
  __nv_bfloat16* shout;  // [MAXM][DIM] shared expert output (BF16, added in T1)
  // vision gate: token t is an image token iff vids && vids[t] == image_id; it selects its experts
  // with gbias_v
  const int* vids;       // [M] input token ids, or null (every token uses gbias)
  const float* gbias_v;  // [NEXP] score correction bias of image tokens
  int image_id;
};
// panel flags: 1 = dense mode, 2 = the N16 T4 split tail (TP_MOE_T4N16) -- dense panels only
__device__ __forceinline__ int dflag(const Params& P) { return P.dense ? 3 : 0; }

// ------------------------------------------------------------------------------------------------
// device helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\nW_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra "
      "W_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
      "%3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes,
                                         uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void bulk_s2g(
    void* dst, const void* src, unsigned bytes) {  // smem -> global (async proxy), bulk group
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                   reinterpret_cast<uint64_t>(dst)),
               "r"(smem_u32(src)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read0() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait0() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* map, int c0, int c1,
                                            int c2, uint64_t* bar) {
  asm volatile(
      "{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], policy;\n}" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int c0, int c1,
                                            int c2, int c3, uint64_t* bar) {
  asm volatile(
      "{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5}], [%6], policy;\n}" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void tma_load_5d(void* dst, const CUtensorMap* map, int c0, int c1,
                                            int c2, int c3, int c4, uint64_t* bar) {
  asm volatile(
      "{ .reg .b64 policy;\n createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], policy;\n}" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
      "r"(smem_u32(bar))
      : "memory");
}
#ifndef TP_MOE_PROD_REGS
#define TP_MOE_PROD_REGS \
  64  // producer warpgroup registers after setmaxnreg (512 x CONS + 128 x PROD = 61440 = the launch
      // allocation)
#endif
#ifndef TP_MOE_CONS_REGS
#define TP_MOE_CONS_REGS 104
#endif
static_assert(512 * TP_MOE_CONS_REGS + 128 * TP_MOE_PROD_REGS <= 61440, "register pool");
#ifndef TP_MOE_PASSA
#define TP_MOE_PASSA \
  3  // producer pass A: compact panels: at most this many FC1 stages' weights before the x gather
     // (NST = 7; dense: NST -- its pass A also issues x)
#endif
#ifndef TP_MOE_SMALL_MAXMT
#define TP_MOE_SMALL_MAXMT \
  96  // small tables only for M * topk <= this (96 = M <= 16 at top-6: M17..21 go block-wide)
#endif
// programmatic dependent launch: everything that reads the previous kernels' outputs (x, xs,
// shared, routing weights) or this kernel's own global workspace (epoch, counters -- the previous
// launch of this kernel may be the prerequisite grid) follows it
__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// spin with relaxed loads (an ld.acquire per poll invalidates L1 each time: CCTL.IVALL), one
// acquire fence once satisfied
__device__ __forceinline__ void wait_eq(const unsigned* p, unsigned v) {
  while (ld_relaxed_gpu(p) != v) {
  }
  (void)ld_acquire_gpu(p);
}
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v) {
  while (ld_relaxed_gpu(p) < v) {
  }
  (void)ld_acquire_gpu(p);
}
// loads pinned where they are written (asm volatile keeps program order against the k-step asm
// blocks): issued early, waited on only at their use -- the compiler otherwise sinks them next to
// the use and exposes the L2 latency
__device__ __forceinline__ float ldv_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float ldv_nc_f32(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
__device__ __forceinline__ unsigned atom_add_acq_rel_gpu(unsigned* p, unsigned v) {
  unsigned r;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;"
               : "=r"(r)
               : "l"(p), "r"(v)
               : "memory");
  return r;
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
template <int N>
__device__ __forceinline__ void fence_operand(float (&r)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
__device__ __forceinline__ uint64_t
make_desc_sw128(uint32_t a) {  // K-major 128B-swizzled tile: 8-row atoms of 1 KB (SBO 1024)
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) |
         ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wg_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
__device__ __forceinline__ void wg_wait1() {
  asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}

// One k-tile: wait the stage; one asm block dequantizes the 4 k32 slices (slice s + 1 under slice
// s's wgmma), retires the previous chain (D += C, releases its stage) and issues this one
// (kpipe.cuh)
template <int N, bool FIRST>
__device__ __forceinline__ void kstep(uint8_t* smem, uint64_t* full, uint64_t* empty, int& s,
                                      unsigned& ph, int& sprev, float (&C)[N / 2],
                                      float (&D)[N / 2], uint64_t desc, int frag_off, int scl_off,
                                      int lane) {
  const bool sub0 =
      KPS == 1 ||
      (s & (KPS - 1)) ==
          0;  // first k-tile of its ring stage: wait for the stage, release the previous one
  if (sub0) mbar_wait(&full[s / KPS], ph);
  const uint8_t* st = smem + s * STAGE;
  kpipe::KPipe<N, FIRST>::run(C, D, smem_u32(st + frag_off), smem_u32(st + scl_off), desc,
                              smem_u32(&empty[sprev / KPS]), lane == 0 && sub0 ? 1u : 0u);
  sprev = s;
  if (++s == NST) {
    s = 0;
    ph ^= 1;
  }
}
#ifndef TP_MOE_FC2_T1S
#define TP_MOE_FC2_T1S \
  (TP_MOE_NDEV == 8 ? 2 : 0)  // 2 (TP8): the FC2 tail k-tile as ONE k32 slice inside the
                              // slice-pipelined asm (KPipe1; its 3 other slices are zeros)
#endif
static_assert(TP_MOE_FC2_T1S == 0 || (TP_MOE_FC2_T1S == 2 && NKT2 == 3),
              "FC2 tail k-tile: full k-step (0) or KPipe1 (2, TP8)");
// FC2 tail k-tile, slice-pipelined (TP_MOE_FC2_T1S 2): KPipe1 = KPipe<N, false> cut to its first
// k32 slice (the 3 others are zeros)
template <int N>
__device__ __forceinline__ void kstep_p1(uint8_t* smem, uint64_t* full, uint64_t* empty, int& s,
                                         unsigned& ph, int& sprev, float (&C)[N / 2],
                                         float (&D)[N / 2], uint64_t desc, int frag_off,
                                         int scl_off, int lane) {
  static_assert(KPS == 1, "kstep_p1: one k-tile per ring stage");
  mbar_wait(&full[s], ph);
  const uint8_t* st = smem + s * STAGE;
  kpipe::KPipe1<N>::run(C, D, smem_u32(st + frag_off), smem_u32(st + scl_off), desc,
                        smem_u32(&empty[sprev]), lane == 0 ? 1u : 0u);
  sprev = s;
  if (++s == NST) {
    s = 0;
    ph ^= 1;
  }
}
template <int N>
__device__ __forceinline__ void kfinish(uint64_t* empty, int sprev, float (&C)[N / 2],
                                        float (&D)[N / 2], int lane) {
  wg_wait0();
  fence_operand(C);
  if (lane == 0) mbar_arrive(&empty[sprev / KPS]);  // (pieces / units end on a stage's last k-tile)
#pragma unroll
  for (int i = 0; i < N / 2; ++i) D[i] += C[i];
  fence_operand(D);
}

// ------------------------------------------------------------------------------------------------
// routing tables
struct Tabs {
  uint16_t* ids;
  uint16_t* ent;
  uint16_t* uexp;
  uint16_t* off;
  uint16_t* pid;
  uint16_t* p2u;
  float* fs;
  float* am;
  int* misc;
  uint32_t* ccum;
  int* qq;
  uint32_t* prow;
  uint8_t* km;
  uint32_t* pinf;
  int* bnd;
  uint16_t* e2p;
  float* grw;
};
__device__ __forceinline__ Tabs tabs(uint8_t* smem) {
  uint8_t* t = smem + OFF_TAB;
  return Tabs{reinterpret_cast<uint16_t*>(t + TAB_IDS),  reinterpret_cast<uint16_t*>(t + TAB_ENT),
              reinterpret_cast<uint16_t*>(t + TAB_UEXP), reinterpret_cast<uint16_t*>(t + TAB_OFF),
              reinterpret_cast<uint16_t*>(t + TAB_PID),  reinterpret_cast<uint16_t*>(t + TAB_P2U),
              reinterpret_cast<float*>(t + TAB_FS),      reinterpret_cast<float*>(t + TAB_AM),
              reinterpret_cast<int*>(t + TAB_MISC),      reinterpret_cast<uint32_t*>(t + TAB_CCUM),
              reinterpret_cast<int*>(t + TAB_QQ),        reinterpret_cast<uint32_t*>(t + TAB_PROW),
              reinterpret_cast<uint8_t*>(t + TAB_KM),    reinterpret_cast<uint32_t*>(t + TAB_PINF),
              reinterpret_cast<int*>(t + TAB_BND),       reinterpret_cast<uint16_t*>(t + TAB_E2P),
              reinterpret_cast<float*>(t + TAB_GRW)};
}
// Every CTA: union = present experts in ascending id; entries of an expert = its tokens in
// ascending order (a token listing the same expert in several slots is ONE entry; FC2 writes its
// output to every such slot); panels of <= NMAX entries. Small-M tables (M * topk <= 128, top-k <=
// 8, M <= 128): warp 0 alone, warp-synchronous -- bitonic sort of the (expert, token, slot) keys (4
// per lane), then prefix counts by ballot.  Produces exactly the tables of build_tables (one panel
// per expert: n <= M <= NMAX here).  The caller's __syncthreads publishes them.
#ifndef TP_MOE_SH_FIRST
#define TP_MOE_SH_FIRST \
  1  // block-wide tables: the shared expert's panels get the FIRST panel ids (first in the FC1
     // stream-K order and the FC2
// unit order: their FC1 / quantization done early, their larger FC2 units not left for the tail)
#endif
// shared-expert panel test: ids [T.misc[5], T.misc[5] + T.misc[6])
__device__ __forceinline__ bool is_sh(const Tabs& T, int pid) {
  return (unsigned)(pid - T.misc[5]) < (unsigned)T.misc[6];
}

// Small-M tables, bitmap form: warp 0 alone.  Per expert a token bitmask (M <= 32) built with
// shared atomicOr; a warp scan over the NEXP experts (NEXP / 32 per lane, ascending) gives union
// slots, entry offsets and the cost prefix; entries are the set bits in ascending token order.  One
// panel per expert.
__device__ __forceinline__ void build_tables_small(const Params& P, uint8_t* smem, const Tabs& T) {
  const int lane = threadIdx.x & 31;
  const int M = P.M, topk = P.topk, MT = M * topk;
  uint32_t* tm = reinterpret_cast<uint32_t*>(smem + PRO_BITS);  // [NEXP] token mask per expert
  int* eoff =
      reinterpret_cast<int*>(smem + PRO_OFFE);  // [NEXP] entry offset of expert (valid if present)
  int* eslot = reinterpret_cast<int*>(smem + PRO_PIDE);  // [NEXP] union slot (= panel) of expert
#pragma unroll
  for (int i = 0; i < NEXP / 32; ++i) tm[lane + 32 * i] = 0u;
  int ev[4];
  const int* gids = reinterpret_cast<const int*>(smem + PRO_PRE);  // fused gate: this CTA's ids
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int p = lane + 32 * q;
    ev[q] = p < MT ? (P.glog ? gids[p] : __ldcg(P.ids + p)) : -1;
  }
  for (int i = lane; i < M * 8; i += 32)
    if ((i & 7) >= topk) T.ids[i] = (uint16_t)0xFFFF;  // row padding
  for (int i = lane; i < (MT + M + 3) / 4; i += 32)
    reinterpret_cast<unsigned*>(T.km)[i] = 0u;  // (+ shared entries: no slots)
  __syncwarp();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int p = lane + 32 * q;
    if (p < MT) {
      const int e = ev[q], t = p / topk, k = p - t * topk;
      const bool ok = (unsigned)e < (unsigned)P.n_exp;
      T.ids[t * 8 + k] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
      if (ok)
        atomicOr(tm + e, 1u << t);
      else
        ev[q] = -1;
    }
  }
  __syncwarp();
  // experts lane * EPL .. + EPL - 1: counts, presence, panel costs / padded widths
  constexpr int EPL = NEXP / 32;  // (8 | DSV4.1 12)
  uint32_t m[EPL];
  int cnt = 0, pres = 0;
  uint32_t cst = 0, npd = 0;
  const uint32_t dpc =
      P.dense ? panel_cost((P.md + 7) / 8 - 1, dflag(P)) : 0u;  // dense: every panel costs the same
#pragma unroll
  for (int v = 0; v < EPL / 4; ++v) {
    const uint4 mv = reinterpret_cast<const uint4*>(tm)[lane * (EPL / 4) + v];
    m[4 * v] = mv.x;
    m[4 * v + 1] = mv.y;
    m[4 * v + 2] = mv.z;
    m[4 * v + 3] = mv.w;
  }
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    const int n = __popc(m[i]);
    cnt += n;
    pres += n > 0;
    if (n) {
      cst += P.dense ? dpc : panel_cost((n + 7) / 8 - 1, dflag(P));
      npd += (uint32_t)((n + 7) & ~7);
    }
  }
  int ic = cnt, ip = pres;
  uint32_t icst = cst, inp = npd;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y0 = __shfl_up_sync(0xffffffffu, ic, o), y1 = __shfl_up_sync(0xffffffffu, ip, o);
    const uint32_t y2 = __shfl_up_sync(0xffffffffu, icst, o),
                   y3 = __shfl_up_sync(0xffffffffu, inp, o);
    if (lane >= o) {
      ic += y0;
      ip += y1;
      icst += y2;
      inp += y3;
    }
  }
  int E = ic - cnt, u = ip - pres;
  uint32_t cc = icst - cst, nn = inp - npd;
  const int E_tot = __shfl_sync(0xffffffffu, ic, 31), U = __shfl_sync(0xffffffffu, ip, 31);
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    const int n = __popc(m[i]);
    if (!n) continue;
    const int e = lane * EPL + i;
    T.uexp[u] = (uint16_t)e;
    T.off[u] = (uint16_t)E;
    T.pid[u] = (uint16_t)u;
    T.p2u[u] = (uint16_t)u;
    T.pinf[u] = pinf_pack((uint32_t)e, (uint32_t)n, (uint32_t)E);
    T.ccum[u] = cc;
    T.prow[u] = nn;
    eoff[e] = E;
    eslot[e] = u;
    cc += P.dense ? dpc : panel_cost((n + 7) / 8 - 1, dflag(P));
    nn += (uint32_t)((n + 7) & ~7);
    E += n;
    ++u;
  }
  const int PT =
      U + 1;  // + the shared expert's panel (M <= 32: one): tokens 0..M-1 as entries E_tot..
  if (lane == 31) {
    T.off[U] = (uint16_t)E_tot;
    T.pid[U] = (uint16_t)U;
    T.misc[0] = U;
    T.misc[1] = E_tot;
    T.misc[2] = PT;
    T.misc[4] = PT;
    T.misc[5] = U;
    T.misc[6] = PT - U;
    T.ccum[U] = cc;
    T.prow[U] = nn;
    T.pinf[U] = pinf_pack(0u, (uint32_t)M, (uint32_t)E_tot);
    T.ccum[U + 1] = cc + panel_cost(((P.dense ? P.md : M) + 7) / 8 - 1, dflag(P), true);
    T.prow[U + 1] = nn;
  }
  for (int i = lane; i < M; i += 32) {
    T.ent[E_tot + i] = (uint16_t)i;
    T.e2p[E_tot + i] = (uint16_t)U;
  }
  __syncwarp();
  // entries (rank of the token in its expert's mask) and their slot masks, one id per lane-step (a
  // duplicate expert in several slots of a token: one entry -- written twice with the same values
  // -- and several km bits)
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int p = lane + 32 * q;
    if (p < MT && ev[q] >= 0) {
      const int e = ev[q], t = p / topk, k = p - t * topk;
      const int ent = eoff[e] + __popc(tm[e] & ((1u << t) - 1u));
      T.ent[ent] = (uint16_t)t;
      T.e2p[ent] = (uint16_t)eslot[e];
      atomicOr(reinterpret_cast<unsigned*>(T.km + (ent & ~3)), (1u << k) << ((ent & 3) * 8));
    }
  }
  __syncwarp();
  {  // stream-K range starts: this CTA's window
    const unsigned G = gridDim.x, total = T.ccum[PT];
    const int ncl = P.dense ? (P.md + 7) / 8 - 1 : -1;
    for (unsigned b = blockIdx.x + lane; b <= min(G, blockIdx.x + 33u); b += 32) {
      int r;
      if (b == 0)
        r = 0;
      else if (b == G || PT == 0)
        r = PT * NQG * NKT1;
      else {
        const unsigned bcst = (total * b) / G;
        int lo = 0, hi = PT - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (T.ccum[mid] <= bcst)
            lo = mid;
          else
            hi = mid - 1;
        }
        const int cl = ncl >= 0 ? ncl : (pinf_n(T.pinf[lo]) + 7) / 8 - 1;
        r = lo * NQG * NKT1 + panel_kt(bcst - T.ccum[lo], cl, dflag(P), lo >= U);
      }
      T.bnd[b] = r;
      T.bnd[MAXG + 1 + b] = PT * NQG * NKT1;
    }
  }
}

// Dense small-M tables (P.dense: M <= 16, M * topk <= 128): warps 0..7 (named barrier 2), one
// expert / one routing entry per thread, few distinct instructions (the prologue runs from a cold
// instruction cache: unrolled per-lane loops cost more than they save). Same tables as
// build_tables_small.
__device__ __forceinline__ void build_tables_dense(const Params& P, uint8_t* smem, const Tabs& T) {
  constexpr int BT = BT_DENSE;  // (= NEXP: one expert per thread)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = P.M, topk = P.topk, MT = M * topk;
  uint32_t* tm = reinterpret_cast<uint32_t*>(smem + PRO_BITS);  // [NEXP] token mask per expert
  int* eoff = reinterpret_cast<int*>(smem + PRO_OFFE);
  int* eslot = reinterpret_cast<int*>(smem + PRO_PIDE);
  int* ws =
      reinterpret_cast<int*>(smem + PRO_WS);  // [8] presence, [8] entries, [8] padded rows per warp
  int e = -1, t = 0, k = 0;
  if (tid < MT) {
    e = P.glog ? reinterpret_cast<const int*>(smem + PRO_PRE)[tid] : __ldcg(P.ids + tid);
    t = tid / topk;
    k = tid - t * topk;
  }
  tm[tid] = 0u;
  if (tid < M * 8 && (tid & 7) >= topk) T.ids[tid] = (uint16_t)0xFFFF;
  if (tid < (MT + M + 3) / 4)
    reinterpret_cast<unsigned*>(T.km)[tid] = 0u;  // (+ shared entries: no slots)
  const bool ok = tid < MT && (unsigned)e < (unsigned)P.n_exp;
  if (tid < MT) T.ids[t * 8 + k] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
  named_bar_sync(2, BT);
  if (ok) atomicOr(tm + e, 1u << t);
  named_bar_sync(2, BT);
  const uint32_t m = tm[tid];
  const int n = __popc(m), np = (n + 7) & ~7;
  const unsigned bal = __ballot_sync(0xffffffffu, n > 0);
  int c = n, d = np;  // inclusive warp scans: entries, padded rows
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, c, o), z = __shfl_up_sync(0xffffffffu, d, o);
    if (lane >= o) {
      c += y;
      d += z;
    }
  }
  if (lane == 31) {
    ws[warp] = __popc(bal);
    ws[WSS + warp] = c;
    ws[2 * WSS + warp] = d;
  }
  named_bar_sync(2, BT);
  int ub = 0, eb = 0, db = 0, U = 0, Et = 0, Dt = 0;
#pragma unroll 1
  for (int w = 0; w < BT / 32; ++w) {
    const int a = ws[w], b = ws[WSS + w], g = ws[2 * WSS + w];
    if (w < warp) {
      ub += a;
      eb += b;
      db += g;
    }
    U += a;
    Et += b;
    Dt += g;
  }
  const uint32_t dpc =
      panel_cost((P.md + 7) / 8 - 1, dflag(P));  // dense: every panel costs the same
  if (n) {
    const int u = ub + __popc(bal & ((1u << lane) - 1u)), E = eb + c - n;
    T.uexp[u] = (uint16_t)tid;
    T.off[u] = (uint16_t)E;
    T.pid[u] = (uint16_t)u;
    T.p2u[u] = (uint16_t)u;
    T.pinf[u] = pinf_pack((uint32_t)tid, (uint32_t)n, (uint32_t)E);
    T.ccum[u] = (uint32_t)u * dpc;
    T.prow[u] = (uint32_t)(db + d - np);
    eoff[tid] = E;
    eslot[tid] = u;
  }
  // + the shared expert's panel at U, cost-uniform with the dense panels: tokens 0..M-1 as entries
  // Et..
  const int PT = U + 1;
  const int pg = PT;  // one panel group (the shared panel has its own cost)
  if (tid == BT - 1) {
    T.off[U] = (uint16_t)Et;
    T.pid[U] = (uint16_t)U;
    T.misc[0] = U;
    T.misc[1] = Et;
    T.misc[2] = PT;
    T.misc[4] = pg;
    T.misc[5] = U;
    T.misc[6] = PT - U;
    T.ccum[U] = (uint32_t)U * dpc;
    T.prow[U] = (uint32_t)Dt;
    T.pinf[U] = pinf_pack(0u, (uint32_t)M, (uint32_t)Et);
    T.ccum[U + 1] = (uint32_t)U * dpc + panel_cost((P.md + 7) / 8 - 1, dflag(P), true);
    T.prow[U + 1] = (uint32_t)Dt;
  }
  if (tid < M) {
    T.ent[Et + tid] = (uint16_t)tid;
    T.e2p[Et + tid] = (uint16_t)U;
  }
  {  // stream-K range starts of this CTA's window: the uniform routed panels, then the shared panel
     // (its own cost)
    const unsigned G = gridDim.x, b = blockIdx.x + (unsigned)tid;
    if (tid < 34 && b <= G) {
      const uint32_t sc = panel_cost((P.md + 7) / 8 - 1, dflag(P), true),
                     tot = (uint32_t)U * dpc + sc;
      const uint32_t x = (uint32_t)(((unsigned long long)tot * b) / G);
      int r;
      if (b == G)
        r = PT * NQG * NKT1;
      else if (x < (uint32_t)U * dpc) {
        const int pnl = (int)(x / dpc);
        r = pnl * NQG * NKT1 + panel_kt(x - (uint32_t)pnl * dpc, (P.md + 7) / 8 - 1, dflag(P));
      } else
        r = U * NQG * NKT1 + panel_kt(x - (uint32_t)U * dpc, (P.md + 7) / 8 - 1, dflag(P), true);
      T.bnd[b] = r;
      T.bnd[MAXG + 1 + b] = PT * NQG * NKT1;
    }
  }
  named_bar_sync(2, BT);
  if (ok) {  // entries: rank of the token in its expert's mask (a duplicate expert of a token: one
             // entry, several km bits)
    const int ent = eoff[e] + __popc(tm[e] & ((1u << t) - 1u));
    T.ent[ent] = (uint16_t)t;
    T.e2p[ent] = (uint16_t)eslot[e];
    atomicOr(reinterpret_cast<unsigned*>(T.km + (ent & ~3)), (1u << k) << ((ent & 3) * 8));
  }
}

// block-wide tables: the consumer warps only (BT threads, named barrier 1) -- the producer
// warpgroup quantizes x meanwhile (fused)
__device__ __forceinline__ void build_tables(const Params& P, uint8_t* smem, const Tabs& T) {
  constexpr int BT = NCONS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = P.M, topk = P.topk, MT = M * topk, W = (M + 31) >> 5;
  uint32_t* bits = reinterpret_cast<uint32_t*>(smem + PRO_BITS);
  uint16_t* pre = reinterpret_cast<uint16_t*>(smem + PRO_PRE);
  int* offe = reinterpret_cast<int*>(smem + PRO_OFFE);
  int* ws = reinterpret_cast<int*>(smem + PRO_WS);
  int* pide = reinterpret_cast<int*>(smem + PRO_PIDE);
  static_assert(MAX_TOKENS * TOPK_MAX <= BT, "one routing value per thread");
  const int idr = tid < MT ? __ldcg(P.ids + tid) : -1;  // in flight during the zeroing
  for (int i = tid; i < W * NEXP; i += BT)
    bits[i] = 0u;  // [word][expert] (conflict-free scans, no division)
  named_bar_sync(1, BT);
  for (int i = tid; i < M * 8; i += BT)
    if ((i & 7) >= topk) T.ids[i] = (uint16_t)0xFFFF;  // row padding
  auto put = [&](int p, int e) {
    const bool ok = (unsigned)e < (unsigned)P.n_exp;
    const int t = topk == 6 ? p / 6 : p / topk;  // (constant divisor: mul-shift)
    T.ids[t * 8 + (p - t * topk)] = ok ? (uint16_t)e : (uint16_t)0xFFFF;
    if (ok) atomicOr(&bits[(t >> 5) * NEXP + e], 1u << (t & 31));
  };
  if (tid < MT) put(tid, idr);
  named_bar_sync(1, BT);
  int c = 0, xa = 0, xb = 0, xd = 0, pres = 0, npan = 0;
  if (tid < NEXP) {
    for (int w = 0; w < W; ++w) {
      pre[w * NEXP + tid] = (uint16_t)c;
      c += __popc(bits[w * NEXP + tid]);
    }
    pres = c > 0;
    npan = (c + NMAX - 1) / NMAX;
    int a = pres, b = c, d = npan;  // inclusive warp scans of (present, count, panels)
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
      ws[WSS + warp] = b;
      ws[2 * WSS + warp] = d;
    }
    xa = a - pres;
    xb = b - c;
    xd = d - npan;
  }
  named_bar_sync(1, BT);
  if (tid < NEXP) {
#pragma unroll
    for (int w = 0; w < NEXP / 32; ++w)
      if (w < warp) {
        xa += ws[w];
        xb += ws[WSS + w];
        xd += ws[2 * WSS + w];
      }  // independent loads
    xd += TP_MOE_SH_FIRST ? (M + NMAX - 1) / NMAX
                          : 0;  // (shared-first: routed panels after the shared ones)
    offe[tid] = xb;
    pide[tid] = xd;
    if (pres) {
      T.uexp[xa] = (uint16_t)tid;
      T.off[xa] = (uint16_t)xb;
      T.pid[xa] = (uint16_t)xd;
      for (int p = 0; p < npan; ++p) {
        T.p2u[xd + p] = (uint16_t)xa;
        T.pinf[xd + p] =
            pinf_pack((uint32_t)tid, (uint32_t)min(NMAX, c - p * NMAX), (uint32_t)(xb + p * NMAX));
      }
    }
    if (tid == NEXP - 1) {
      const int U = xa + pres;
      T.off[U] = (uint16_t)(xb + c);
      T.pid[U] = (uint16_t)(xd + npan);
      T.misc[0] = U;
      T.misc[1] = xb + c;
      T.misc[2] = xd + npan;
    }
  }
  named_bar_sync(1, BT);
  // + the shared expert's panels: tokens 32 sp .. as entries E_tot + 32 sp .., after the routed
  // panels.  (every thread derives P_r / Ps / P_tot itself -- no barrier; misc[2] / [5] / [6] are
  // written after the next barrier, the shared panels' widths are recomputed by their cost threads
  // instead of read back from pinf)
  const int Ps = (M + NMAX - 1) / NMAX, E_tot = T.misc[1];
  const int P_end = T.misc[2], P_r = TP_MOE_SH_FIRST ? 0 : P_end;  // first shared panel id
  const int P_tot = TP_MOE_SH_FIRST ? P_end : P_end + Ps;
  if (tid < Ps)
    T.pinf[P_r + tid] =
        pinf_pack(0u, (uint32_t)min(NMAX, M - tid * NMAX), (uint32_t)(E_tot + tid * NMAX));
  if (Ps)
    for (int i = tid; i < M; i += BT) {
      T.ent[E_tot + i] = (uint16_t)i;
      T.km[E_tot + i] = 0;
      T.e2p[E_tot + i] = (uint16_t)(P_r + i / NMAX);
    }
  for (int p = tid; p < MT; p += BT) {
    const int t = topk == 6 ? p / 6 : p / topk, k = p - t * topk;
    const uint4 row = *reinterpret_cast<const uint4*>(T.ids + t * 8);  // the token's 8 slots
    const uint32_t rw[4] = {row.x, row.y, row.z, row.w};
    int e = (int)((rw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu);
    if (e == 0xFFFF) continue;
    unsigned km = 0;
#pragma unroll
    for (int k2 = 0; k2 < 8; ++k2)
      km |= (((rw[k2 >> 1] >> ((k2 & 1) * 16)) & 0xFFFFu) == (unsigned)e ? 1u : 0u) << k2;
    if (km & ((1u << k) - 1u))
      continue;  // an earlier slot of this token has the same expert: one entry per (token, expert)
    const int w = t >> 5;
    const int idx =
        offe[e] + pre[w * NEXP + e] + __popc(bits[w * NEXP + e] & ((1u << (t & 31)) - 1u));
    T.ent[idx] = (uint16_t)t;
    T.km[idx] = (uint8_t)km;
    T.e2p[idx] = (uint16_t)(pide[e] + (idx - offe[e]) / NMAX);
  }
  // FC1 cost prefix over panels (stream-K balances cost, not k-tiles): panel cost = 128 k-tiles x
  // c(width class)
  {
    uint32_t c = 0, np = 0;
    if (tid < P_tot) {
      const bool sh = (unsigned)(tid - P_r) < (unsigned)Ps;
      const int n = P.dense ? P.md
                    : sh    ? min(NMAX, M - (tid - P_r) * NMAX)
                            : pinf_n(T.pinf[tid]);  // B operand width of the panel's FC1
      c = panel_cost((n + 7) / 8 - 1, dflag(P), sh);
      np = (uint32_t)((n + 7) & ~7);
    }
    uint32_t a = c, d = np;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, a, o), z = __shfl_up_sync(0xffffffffu, d, o);
      if (lane >= o) {
        a += y;
        d += z;
      }
    }
    if (lane == 31) {
      ws[WS2A + warp] = (int)a;
      ws[WS2D + warp] = (int)d;
    }
    named_bar_sync(1, BT);  // also publishes T.ent / T.km
    uint32_t base = 0, based = 0, total = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w) {
      const uint32_t cw = (uint32_t)ws[WS2A + w];
      total += cw;
      if (w < warp) {
        base += cw;
        based += (uint32_t)ws[WS2D + w];
      }
    }
    if (tid <= P_tot) {
      T.ccum[tid] = base + a - c;
      T.prow[tid] = based + d - np;
    }  // tid == P_tot: c == np == 0 -> totals
    if (tid == 0) {
      T.misc[5] = P_r;
      T.misc[6] = Ps;
      T.misc[2] = P_tot;
    }
    // panel groups: group 0 = panels starting before the cost midpoint, group 1 = the rest (from
    // this thread's own exclusive / inclusive prefix and the total -- no barrier between the prefix
    // and the split)
    if (tid == 0 && (P.groups == 1 || P_tot <= 1))
      T.misc[4] = P_tot;  // else exactly one panel thread writes the split below
    if (P.groups == 2 && P_tot > 1 && tid < P_tot) {
      const unsigned half = total / 2;
      if (base + a - c <= half && half < base + a) T.misc[4] = min(P_tot - 1, max(1, tid + 1));
    }
    named_bar_sync(1, BT);
  }
  // stream-K range starts per group and CTA: panel thread pid writes the CTAs b whose group cost
  // boundary falls in its range (32-bit: total <= MAXP * 128 * 200 < 2^24, total * G < 2^32)
  {
    const unsigned G = gridDim.x;
    const int pg = T.misc[4];
    if (tid < 2) {
      const int p0 = tid ? pg : 0, p1 = tid ? P_tot : pg;
      T.bnd[tid * (MAXG + 1)] = p0 * NQG * NKT1;
      T.bnd[tid * (MAXG + 1) + G] = p1 * NQG * NKT1;
    }
    if (tid < 2 * ((int)G - 1)) {  // thread (g, b), b = 1..G-1: the panel p with c0(p) <= X =
                                   // floor(tot b / G) < c1(p) (binary search; same values)
      const int g = tid >= (int)G - 1 ? 1 : 0;
      const unsigned b = (unsigned)(tid - g * ((int)G - 1) + 1);
      const int pa = g ? pg : 0, pb = g ? P_tot : pg;
      if (pa < pb) {
        const unsigned C0 = T.ccum[pa], tot = T.ccum[pb] - C0, X = (tot * b) / G;
        int lo = pa, hi = pb - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) >> 1;
          if (T.ccum[mid] - C0 <= X)
            lo = mid;
          else
            hi = mid - 1;
        }
        const int n = P.dense ? P.md : pinf_n(T.pinf[lo]), cl = (n + 7) / 8 - 1;
        T.bnd[g * (MAXG + 1) + b] =
            lo * NQG * NKT1 + panel_kt(X - (T.ccum[lo] - C0), cl, dflag(P), is_sh(T, lo));
      }
    }
    if (tid > 0 && tid < (int)G) {  // an empty group: every CTA's range is empty
      if (pg == 0) T.bnd[tid] = 0;
      if (pg == P_tot) T.bnd[MAXG + 1 + tid] = P_tot * NQG * NKT1;
    }
  }
}

// Fused gate of one token by one warp, SGLang's DeepSeek-V4.1 moe_fused_gate (_router_triton_kernel
// with sqrtsoftplus_log1p, top-6, renormalize with epsilon 1e-20) as its Triton build computes it:
// act = sqrt(x > 20 ? x : log1p(exp(x))) with libdevice expf / log1pf in FTZ mode and
// sqrt.approx.ftz (tri41_act: the Triton kernel's PTX for one element); biased = act + bias (NaN
// ranks above every finite value); top-k by repeated max, ties to the lowest expert id, taken ->
// -inf; weights = div.full(act, routed_sum + 1e-20) with tl.sum's pairwise routed_sum ((R0 + R1) +
// (R2 + R3)) + (R4 + R5), unscaled (T1 applies the routed scaling factor).  Expert e = lane + 32 w
// here (the selection does not depend on the layout); keys = order-preserving uints (0 = taken).
__device__ __forceinline__ float tri41_act(float x) {
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
__device__ __forceinline__ float div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ uint32_t
gate_key(float v) {  // order-preserving, > 0 for every finite value (taken = 0)
  const uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);  // -0 == +0
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
constexpr int GATE_TOPK = 6;
// DSV4.1 (NEXP 384): 12 experts per lane; top-k by k rounds of (lane max -> warp max) -- the same
// selection and order as the 256 sorting network (largest key first, ties to the lower expert id)
__device__ __forceinline__ void gate_token_n(const float* gbias, const float* lg, int lane, int& id,
                                             float& wt) {
  constexpr int GW = NEXP / 32;
  float sc[GW];
  uint64_t h[GW];
#pragma unroll
  for (int w = 0; w < GW; ++w) {
    const float l = __ldcg(lg + lane + 32 * w);
    const float a = tri41_act(l);
    float v = __fadd_rn(a, __ldg(gbias + lane + 32 * w));
    v = v == v ? v : __int_as_float(0x7f800000);
    sc[w] = a;
    h[w] = ((uint64_t)gate_key(v) << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)(lane + 32 * w));
  }
  // The k rounds as a rolled loop (cold-i-cache code size: each round once); the selected
  // activation is fetched in its round and the normalizer keeps the pairwise ((R0 + R1) + (R2 +
  // R3)) + (R4 + R5) association
  float acc = 0.f, r2 = 0.f, r4 = 0.f, r = 0.f;
  id = 0;
#pragma unroll 1
  for (int k = 0; k < GATE_TOPK; ++k) {
    uint64_t b = h[0];
#pragma unroll
    for (int w = 1; w < GW; ++w) b = h[w] > b ? h[w] : b;
    const uint32_t hk = (uint32_t)(b >> 32), hl = (uint32_t)b;
    const uint32_t gm = __reduce_max_sync(0xffffffffu, hk);
    const uint32_t gl = __reduce_max_sync(0xffffffffu, hk == gm ? hl : 0u);
    const int ex = (int)(0xFFFFFFFFu - gl);
    if (hk == gm && hl == gl) {
#pragma unroll
      for (int w = 0; w < GW; ++w) h[w] = h[w] == b ? 0ull : h[w];
    }
    const int ew = ex >> 5;
    float sv = sc[0];
#pragma unroll
    for (int w = 1; w < GW; ++w) sv = ew == w ? sc[w] : sv;
    const float R = __shfl_sync(0xffffffffu, sv, ex & 31);
    if (k == 0)
      acc = R;
    else if (k == 1)
      acc = __fadd_rn(acc, R);
    else if (k == 2)
      r2 = R;
    else if (k == 3)
      acc = __fadd_rn(acc, __fadd_rn(r2, R));
    else if (k == 4)
      r4 = R;
    else
      acc = __fadd_rn(acc, __fadd_rn(r4, R));
    r = lane == k ? R : r;
    id = lane == k ? ex : id;
  }
  const float norm = __fadd_rn(acc, 1e-20f);
  wt = div_full(r, norm);
}
__device__ __forceinline__ void gate_token(const Params& P, int t, int lane, int& id, float& wt) {
  const float* gb = (P.vids != nullptr && __ldg(P.vids + t) == P.image_id) ? P.gbias_v : P.gbias;
  gate_token_n(gb, P.glog + (size_t)t * NEXP, lane, id, wt);
}

// Fused input quantization of token t by threads [t0, t0 + 256) (named barrier 7): Humming
// quant_input float8e4m3, bit for bit -- absmax = max(max|x|, 1e-30), scale = absmax * fp32(1/448)
// (Triton's "/ 448"), inv = 1 / scale (div.rn), q = e4m3 RN satfinite(x * inv)
template <int NQ>  // NQ threads (named barrier 7), DIM / 8 / NQ 16-B chunks each
__device__ __forceinline__ void quant_token(const Params& P, int t, int lt, float* red,
                                            unsigned xtag) {
  static_assert((DIM / 8) % NQ == 0, "quantization chunks per thread");
  constexpr int CH = DIM / 8 / NQ;
  const uint4* src = reinterpret_cast<const uint4*>(P.xb + (size_t)t * DIM);
  uint32_t w[CH][4];
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const uint4 v = __ldg(src + lt + c * NQ);
    w[c][0] = v.x;
    w[c][1] = v.y;
    w[c][2] = v.z;
    w[c][3] = v.w;
  }
  float m = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c)
#pragma unroll
    for (int i = 0; i < 4; ++i) m = fmaxf(m, fmaxf(fabsf(bf16lo(w[c][i])), fabsf(bf16hi(w[c][i]))));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((lt & 31) == 0) red[lt >> 5] = m;
  named_bar_sync(7, NQ);
  m = red[0];
#pragma unroll
  for (int i = 1; i < NQ / 32; ++i) m = fmaxf(m, red[i]);
  const float scale = __fmul_rn(fmaxf(m, 1e-30f), 0.0022321429569274187f),
              inv = __fdiv_rn(1.0f, scale);  // absmax / 448 compiles to x * (1/448)
  uint8_t* dst = const_cast<uint8_t*>(P.xq) + (size_t)t * DIM;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    uint32_t q[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t a = w[c][2 * i], b = w[c][2 * i + 1];
      const uint32_t lo = pack_e4m3x2(__fmul_rn(bf16lo(a), inv), __fmul_rn(bf16hi(a), inv));
      const uint32_t hi = pack_e4m3x2(__fmul_rn(bf16lo(b), inv), __fmul_rn(bf16hi(b), inv));
      q[i] = lo | (hi << 16);
    }
    *reinterpret_cast<uint2*>(dst + (lt + c * NQ) * 8) = make_uint2(q[0], q[1]);
  }
  if (lt == 0) const_cast<float*>(P.xs)[t] = scale;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    const int j = lt + c * NQ;
    float f[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = bf16lo(w[c][i]);
      f[2 * i + 1] = bf16hi(w[c][i]);
    }
    float gm = shnum::gq_amax_init();
#pragma unroll
    for (int i = 0; i < 8; ++i) gm = shnum::absmax_ftz(gm, f[i]);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) gm = shnum::max_ftz(gm, __shfl_xor_sync(0xffffffffu, gm, o));
    const float mult = shnum::gq_mult(gm);
    const uint32_t q0 = (uint32_t)shnum::gq_pack2(f[0], f[1], mult) |
                        ((uint32_t)shnum::gq_pack2(f[2], f[3], mult) << 16);
    const uint32_t q1 = (uint32_t)shnum::gq_pack2(f[4], f[5], mult) |
                        ((uint32_t)shnum::gq_pack2(f[6], f[7], mult) << 16);
    *reinterpret_cast<uint2*>(P.xqg + (size_t)t * DIM + j * 8) = make_uint2(q0, q1);
    if ((j & 15) == 0) P.xsT[(j >> 4) * MAXM + t] = shnum::gq_scale(gm);
  }
  asm volatile("fence.proxy.async.global;" ::
                   : "memory");  // x is read by TMA / bulk copies (async proxy)
  named_bar_sync(7, NQ);
  if (lt == 0) {
    fence_acq_rel_gpu();
    st_release_gpu(P.xflag + t, xtag);
  }
}

// wait until tokens [0, M) are quantized: thread lt of nthr polls tokens lt, lt + nthr, .. with all
// its loads in flight together (the caller orders the other threads after it)
__device__ __forceinline__ void wait_tokens(const Params& P, int lt, int nthr, unsigned xtag) {
  for (;;) {
    bool all = true;
    for (int t0 = lt; t0 < P.M; t0 += 16 * nthr) {
      unsigned v[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int t = t0 + i * nthr;
        v[i] = t < P.M ? ld_relaxed_gpu(P.xflag + t) : xtag;
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) all &= v[i] == xtag;
    }
    if (all) break;
  }
  fence_acq_rel_gpu();
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// fused routing, consumer warps (tid < NCONS; named barrier 1).  Small tables: every CTA gates all
// tokens (warp w: w, w + 16, ..) -> ids at PRO_PRE, weights at T.grw (CTA 0 also -> P.ids / P.rw).
// Block-wide: token t by CTA t % G -> P.ids / P.rw + flag, all CTAs wait for all.
__device__ __forceinline__ void route_prologue(const Params& P, uint8_t* smem, const Tabs& T,
                                               bool small, int nthr, int bar) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, M = P.M, topk = P.topk;
  if (small) {
    int* gids = reinterpret_cast<int*>(smem + PRO_PRE);
    for (int t = warp; t < M; t += nthr / 32) {
      int id;
      float w;
      gate_token(P, t, lane, id, w);
      if (lane < topk) {
        gids[t * topk + lane] = id;
        T.grw[t * topk + lane] = w;
        if (blockIdx.x == 0) {
          const_cast<int*>(P.ids)[t * topk + lane] = id;
          const_cast<float*>(P.rw)[t * topk + lane] = w;
        }
      }
    }
  } else {
    // (gated-token counter P.gcnt[par], ONE poller per CTA: thousands of
    // spinning pollers slow the very stores / atomics they wait for)
    unsigned* c = P.gcnt + ((__ldcg(P.work) + 1u) & 1u);
    for (int t = blockIdx.x + warp * gridDim.x; t < M; t += (NCONS / 32) * gridDim.x) {
      int id;
      float w;
      gate_token(P, t, lane, id, w);
      if (lane < topk) {
        const_cast<int*>(P.ids)[t * topk + lane] = id;
        const_cast<float*>(P.rw)[t * topk + lane] = w;
      }
      __syncwarp();
      if (lane == 0) {
        fence_acq_rel_gpu();
        atomicAdd(c, 1u);
      }
    }
    if (tid == 0) {
      while (ld_relaxed_gpu(c) < (unsigned)M) {
      }
      (void)ld_acquire_gpu(c);  // (the counter's RMWs form one release sequence -- no MEMBAR.GPU on
                                // the routing path)
    }
  }
  named_bar_sync(bar, nthr);
}

// ------------------------------------------------------------------------------------------------
// work lists
struct Item {
  int pid, q, u, e, ebase, n;
};
__device__ __forceinline__ Item item_of(const Tabs& T, int pid, int q) {
  Item r;
  const uint32_t v = T.pinf[pid];
  r.pid = pid;
  r.q = q;
  r.u = 0;
  r.e = pinf_e(v);
  r.n = pinf_n(v);
  r.ebase = pinf_b(v);
  return r;
}
constexpr int PART_F = NMAX / 2;  // fp32 partial floats per consumer thread and slot
// fp32 partial slot (stream-K): float4 k of consumer thread tid at [(slot * PART_F / 4 + k) * NCONS
// + tid] (warp accesses coalesced)
__device__ __forceinline__ float4* part_f4(const Params& P, int slot, int k, int tid) {
  return reinterpret_cast<float4*>(P.part) + ((size_t)slot * (PART_F / 4) + k) * NCONS + tid;
}
// k-tile index (global, item-major) where CTA b's FC1 range starts (precomputed in the prologue:
// cost boundary total*b/G)
__device__ __forceinline__ int fc1_lo(const Tabs& T, int g, int b) {
  return T.bnd[g * (MAXG + 1) + b];
}
// x gather: C = entries * CPE 16-B chunks (entry-major: [kt NKT1][chunk 8]) split evenly: CTA b
// gathers [b C / G, (b + 1) C / G)
__device__ __forceinline__ unsigned gather_owner(unsigned j, unsigned C, unsigned G) {
  return ((j + 1) * G + C - 1) / C - 1;
}
__device__ __forceinline__ unsigned gather_contrib(const Item& I, unsigned C,
                                                   unsigned G) {  // CTAs that gather panel I
  const unsigned s0 = (unsigned)I.ebase * (unsigned)CPE,
                 s1 = (unsigned)(I.ebase + I.n) * (unsigned)CPE;
  return gather_owner(s1 - 1, C, G) - gather_owner(s0, C, G) + 1u;
}
// FC2 unit v -> (row group, panel): group 0's panels first (their h is ready earliest), row group
// major inside a group
__device__ __forceinline__ void unit_of(const Tabs& T, int v, int& rtile, int& pid, int nu) {
  const int pg = T.misc[4], P_tot = T.misc[2];
  if (v < pg * nu) {
    rtile = v / pg;
    pid = v - rtile * pg;
  } else {
    const int n1 = P_tot - pg, v2 = v - pg * nu;
    rtile = v2 / n1;
    pid = pg + v2 - rtile * n1;
  }
}

// FC1 weights + offsets of item q's k-tile at slice row `row` (the tail item from its own maps,
// blocks 1 / 3 zero-filled)
__device__ __forceinline__ bool is_t4(const Params& P, const Item& I) {
  return T4 && I.q == NQG - 1 && (P.dense ? P.md <= T4MAXN : I.n <= ((dflag(P) & 2) ? T4MAXN : 8));
}
// T4 ring stage at k-tile kt (multiple of 4): x tiles kt .. kt + 3 (8 rows each) at STAGE_WS + w KB
__device__ __forceinline__ void fc1_xdense(const Params& P, uint8_t* st, int kt, bool t4,
                                           uint64_t* bar) {
  const int r0 = 0;
  if (T4 && t4) {
    for (int w = 0; w < 4; ++w)
      tma_load_2d(st + STAGE_WS + w * P.md * 128, &P.xmap, (kt + w) * 128, r0, bar);
  }  // (x tile w: md rows)
  else
    tma_load_2d(st + STAGE_WS, &P.xmap, kt * 128, r0, bar);
}
__device__ __forceinline__ void fc1_wload(const Params& P, uint8_t* st, int q, int row,
                                          uint64_t* bar, bool t4 = false, int e = 0, int kt = 0) {
  if (T4 && t4) {
    tma_load_4d(st, P.tmaps + 6, 0, 0, 0, e * NKT1 + kt, bar);
    tma_load_3d(st + STAGE_WS - STAGE_S, P.tmaps + 7, 0, 0, e * NKT1 + kt, bar);
  } else if (TAIL && q == NQG - 1) {
    tma_load_4d(st, P.tmaps + 4, 0, 0, 0, row, bar);
    tma_load_3d(st + STAGE_W, P.tmaps + 5, 0, 0, row, bar);
  } else {
    tma_load_5d(st, P.tmaps + 0, 0, 0, q, 0, row, bar);
    tma_load_4d(st + STAGE_W, P.tmaps + 1, 0, q, 0, row, bar);
  }
}
// FC2 weights + offsets of (expert e, row tile rt, k-tile kt) (per-expert 4-D maps, the last
// k-tile's slices 10, 11 zero-filled)
__device__ __forceinline__ void fc2_wload(const Params& P, uint8_t* st, int e, int rt, int kt,
                                          uint64_t* bar) {
  if constexpr (TAIL) {
    tma_load_4d(st, P.tmaps + 2, 0, rt * 4, kt * 4, e, bar);
    tma_load_4d(st + STAGE_W, P.tmaps + 3, 0, rt, kt * 4, e, bar);
  } else {
    const int sl = e * (INTER / 32) + kt * 4;
    tma_load_3d(st, P.tmaps + 2, 0, rt * 4, sl, bar);
    tma_load_3d(st + STAGE_W, P.tmaps + 3, 0, rt, sl, bar);
  }
}
// The producer lane (warp 16, lane 0): per stage one expect_tx for weights + B operand, the w13/w2
// weight + offset TMAs and one bulk copy of the B operand (FC1: the panel's gathered x rows of
// k-tile kt from xg; FC2: the panel's quantized h rows). FC1: the first stages' weights are issued
// before the grid-wide x gather is waited for.  FC2 units are taken from a global queue (work
// stealing); the unit id travels to the consumers through s_unit (written before the unit's first
// stage arrive).
__device__ __forceinline__ void producer(const Params& P, uint8_t* smem, uint64_t* full,
                                         uint64_t* empty, const Tabs& T, int* s_unit,
                                         uint64_t* gbar, bool early_xs, unsigned xtag) {
  const int G = gridDim.x, b = blockIdx.x;
  const int P_tot = T.misc[2];
  int s = 0;
  unsigned ph = 0;
  // the maps were written by a host copy before this launch: no tensormap-proxy acquire fence
  // needed, just the prefetch
  for (int i = 0; i < 4; ++i)
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(P.tmaps + i))
                 : "memory");
  // pass A: weights of the first <= NST FC1 stages (no dependency on the x gather / the previous
  // kernels)
  int nA = 0;
  int kts[NST];
  unsigned kt4m = 0;  // (T4) pass-A stages that are split tail stages
  const int E_sh =
      T.misc[1];  // first shared-expert entry: a panel is the shared expert's iff its ebase >= E_sh
  bool stopA = false;
  for (int g = 0; g < 2 && !stopA; ++g) {
    const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
    for (int gk = lo; gk < hi && nA < (P.dense ? NST : min(NST, TP_MOE_PASSA));) {
      const int it = gk / NKT1, kt0 = gk - it * NKT1, kt1 = min(NKT1, hi - it * NKT1);
      const Item I = item_of(T, it / NQG, it % NQG);
      if (I.ebase >= E_sh) {
        stopA = true;
        break;
      }  // shared expert: issued whole in pass B
      const int np = P.dense ? P.md : (I.n + 7) & ~7;
      const bool t4 = is_t4(P, I);
      for (int kt = kt0; kt < kt1 && nA < (P.dense ? NST : min(NST, TP_MOE_PASSA));
           kt += (t4 ? 4 : 1), ++nA) {
        uint8_t* st = smem + nA * STAGE;
        if ((nA & (KPS - 1)) == 0)
          mbar_arrive_expect_tx(
              &full[nA / KPS],
              (unsigned)(KPS * (STAGE_WS +
                                (t4 ? 4 : 1) * np * 128)));  // (a stage = KPS k-tiles of one item)
        const int row = I.e * (NKT1 * 4) + kt * 4;
        if (P.dense && nA == 1) {  // the TMA unit works in issue order: stage 0's x goes right
                                   // behind its weights
          griddep_wait();  // (PDL: the previous kernel -- shorter than the table build -- has
                           // usually finished)
          if (early_xs)
            asm volatile("fence.proxy.async.global;" ::
                             : "memory");  // (flags acquired by warp 19 before the __syncthreads)
          else
            wait_tokens(P, 0, 1, xtag);
          fc1_xdense(P, smem, kts[0], kt4m & 1, &full[0]);
        }
        fc1_wload(P, st, I.q, row, &full[nA / KPS], t4, I.e, kt);
        kts[nA] = kt;
        if (t4) kt4m |= 1u << nA;
        if (P.dense && nA > 0) {
          fc1_xdense(P, st, kt, t4, &full[nA / KPS]);
        }
      }
      gk = (it + 1) * NKT1;
    }
  }
  griddep_wait();  // pass A read weights only (dense: and x after stage 0); the workspace from here
                   // on
  if (P.dense && nA <= 1) {
    if (early_xs)
      asm volatile("fence.proxy.async.global;" ::: "memory");
    else
      wait_tokens(P, 0, 1, xtag);
  }  // (dense with nA >= 2 waited above; compact panels read xg -- gathered after the wait)
  if (P.dense && nA == 1) {
    fc1_xdense(P, smem, kts[0], kt4m & 1, &full[0]);
  }
  const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
  int n = 0, ready_pid = -1;
  bool rows_ok = false;
  const unsigned C =
      (unsigned)T.misc[1] * (unsigned)CPE;  // (x gather chunks: must match the consumers' split)
  for (int g = 0; g < 2; ++g) {
    const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
    for (int gk = lo; gk < hi;) {
      const int it = gk / NKT1, kt0 = gk - it * NKT1, kt1 = min(NKT1, hi - it * NKT1);
      const Item I = item_of(T, it / NQG, it % NQG);
      if (I.ebase >= E_sh) {  // shared expert item (gate rows 128 q.., up rows 512 + 128 q..): per
                              // k-block two stages (gate, up), each 128 weight rows x K128 + the
                              // panel's group-quantized x tile + its 32 k-block scales
        if (!rows_ok) {
          wait_tokens(P, 0, 1, xtag);
          rows_ok = true;
        }
        const int t0 = I.ebase - E_sh;  // (its tokens 32 sp .. = entries E_sh + 32 sp ..)
        for (int kt = kt0; kt < kt1;
             ++kt) {  // slot 0: gate rows + x tile + scales, slot 1: up rows
          const int s0 = s;
          mbar_wait(&empty[s], ph ^ 1);
          if (++s == NST) {
            s = 0;
            ph ^= 1;
          }
          const int s1 = s;
          mbar_wait(&empty[s], ph ^ 1);
          if (++s == NST) {
            s = 0;
            ph ^= 1;
          }
          uint8_t* st0 = smem + s0 * STAGE;
          mbar_arrive_expect_tx(&full[s0], (unsigned)(STAGE_W + NMAX * 128 + NMAX * 4));
          mbar_arrive_expect_tx(&full[s1], (unsigned)STAGE_W);
          tma_load_2d(st0, &P.w1s, kt * 128, I.q * 128, &full[s0]);
          tma_load_2d(smem + s1 * STAGE, &P.w1s, kt * 128, INTER + I.q * 128, &full[s1]);
          tma_load_2d(st0 + STAGE_WS, &P.xgs, kt * 128, t0, &full[s0]);
          bulk_g2s(st0 + STAGE_W, P.xsT + (size_t)kt * MAXM + t0, (unsigned)(NMAX * 4), &full[s0]);
          ++n;
        }
        gk = (it + 1) * NKT1;
        continue;
      }
      const int np = P.dense ? P.md : (I.n + 7) & ~7;
      const uint8_t* xsrc = P.xg + (size_t)T.prow[I.pid] * DIM;
      if (!P.dense && I.pid != ready_pid) {  // the CTAs that gather this panel's x rows are done
        const unsigned need = gather_contrib(I, C, (unsigned)G);
        while (ld_relaxed_gpu(P.xg_cnt + par * MAXP + I.pid) < need) {
        }
        (void)ld_acquire_gpu(P.xg_cnt + par * MAXP + I.pid);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        ready_pid = I.pid;
      }
      const bool t4 = is_t4(P, I);
      for (int kt = kt0; kt < kt1; kt += (t4 ? 4 : 1), ++n) {
        uint8_t* st = smem + s * STAGE;
        uint64_t* fb = &full[s / KPS];
        if (n >= nA) {  // pass B: full stage
          if ((s & (KPS - 1)) == 0) {
            mbar_wait(&empty[s / KPS], ph ^ 1);
            mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + (t4 ? 4 : 1) * np * 128)));
          }
          const int row = I.e * (NKT1 * 4) + kt * 4;
          fc1_wload(P, st, I.q, row, fb, t4, I.e, kt);
        }
        if (P.dense) {
          if (n >= nA) {
            fc1_xdense(P, st, kt, t4, fb);
          }
        } else if (T4 && t4) {
          for (int w = 0; w < 4; ++w)
            bulk_g2s(st + STAGE_WS + w * np * 128, xsrc + (size_t)(kt + w) * np * 128,
                     (unsigned)(np * 128), fb);
        } else
          bulk_g2s(st + STAGE_WS, xsrc + (size_t)kt * np * 128, (unsigned)(np * 128), fb);
        if (++s == NST) {
          s = 0;
          ph ^= 1;
        }
      }
      gk = (it + 1) * NKT1;
    }
  }
  const int NUr = NRT;  // FC2 units per panel: one per row tile
  const int V2 = P_tot * NUr;
  unsigned* pready =
      reinterpret_cast<unsigned*>(T.qq);  // panel-ready bit set (smem, producer lane only)
  for (int i = 0; i < MAXP / 32; ++i) pready[i] = 0u;
  int vnext = (int)atomicAdd(P.fc2_next + par, 1u);
  for (int j = 0;; ++j) {
    int v = -1;
    if (vnext < V2) {
      v = vnext;
      vnext = (int)atomicAdd(P.fc2_next + par,
                             1u);  // the next unit's grab is in flight during this one
    }
    int2* su = reinterpret_cast<int2*>(s_unit) + (j & 7);
    if (v < 0) {                           // sentinel: consumers read -1 at the next stage
      mbar_wait(&empty[s / KPS], ph ^ 1);  // (s is stage-aligned: every piece / unit is)
      *reinterpret_cast<volatile int*>(&su->x) = -1;
      mbar_arrive(&full[s / KPS]);
      break;
    }
    int rtile, upid;
    unit_of(T, v, rtile, upid, NUr);
    const Item I = item_of(T, upid, 0);
    const int np = (I.n + 7) & ~7;
    if (I.ebase >= E_sh) {  // shared expert unit (256 rows of down): 8 stages (k-block, row half),
                            // each 128 rows x K128 + the panel's
      // quantized activation tile + its 32 scales.  Waits for the panel first (8 stages > the ring:
      // no weights-first pass)
      if (!((pready[I.pid >> 5] >> (I.pid & 31)) & 1u)) {
        while (ld_relaxed_gpu(P.h_flags + I.pid) != epoch) {
        }
        (void)ld_acquire_gpu(P.h_flags + I.pid);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        pready[I.pid >> 5] |= 1u << (I.pid & 31);
      }
      const uint8_t* hsrc = P.hq + (size_t)I.pid * NKT2 * NMAX * 128;
      for (int kb = 0; kb < NKT2;
           ++kb) {  // slot 0: rows 0..127 + B tile + scales, slot 1: rows 128..255
        const int s0 = s;
        mbar_wait(&empty[s], ph ^ 1);
        if (++s == NST) {
          s = 0;
          ph ^= 1;
        }
        if (kb == 0) {
          *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
          *reinterpret_cast<volatile int*>(&su->x) =
              rtile | (I.n << SU_NS) | (I.e << SU_ES) | (I.pid << SU_PS);
        }
        const int s1 = s;
        mbar_wait(&empty[s], ph ^ 1);
        if (++s == NST) {
          s = 0;
          ph ^= 1;
        }
        uint8_t* st0 = smem + s0 * STAGE;
        mbar_arrive_expect_tx(&full[s0], (unsigned)(STAGE_W + np * 128 + NMAX * 4));
        mbar_arrive_expect_tx(&full[s1], (unsigned)STAGE_W);
        tma_load_2d(st0, &P.w2s, kb * 128, rtile * 256, &full[s0]);
        tma_load_2d(smem + s1 * STAGE, &P.w2s, kb * 128, rtile * 256 + 128, &full[s1]);
        bulk_g2s(st0 + STAGE_WS, hsrc + (size_t)kb * NMAX * 128, (unsigned)(np * 128), &full[s0]);
        bulk_g2s(st0 + STAGE_W, P.hs4 + ((size_t)I.pid * NKT2 + kb) * NMAX, (unsigned)(NMAX * 4),
                 &full[s0]);
      }
      continue;
    }
    // panel readiness is cached in a CTA-local bit set: only the first unit of a panel pays an
    // acquire + proxy fence
    bool ready = (pready[I.pid >> 5] >> (I.pid & 31)) & 1u;
    if (!ready && ld_acquire_gpu(P.h_flags + I.pid) == epoch) {
      asm volatile("fence.proxy.async.global;" ::
                       : "memory");  // hq (generic writes, other CTAs) -> async-proxy bulk reads
      pready[I.pid >> 5] |= 1u << (I.pid & 31);
      ready = true;
    }
    const uint8_t* hsrc = P.hq + (size_t)I.pid * NKT2 * NMAX * 128;
    if (ready) {  // one pass: weights + B operand per stage
      for (int kt = 0; kt < NKT2; ++kt) {
        uint64_t* fb = &full[s / KPS];
        if ((kt & (KPS - 1)) == 0) {
          mbar_wait(&empty[s / KPS], ph ^ 1);
          if (kt == 0) {
            *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
            *reinterpret_cast<volatile int*>(&su->x) =
                rtile | (I.n << SU_NS) | (I.e << SU_ES) | (I.pid << SU_PS);
          }
          mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + np * 128)));
        }
        uint8_t* st = smem + s * STAGE;
        fc2_wload(P, st, I.e, rtile, kt, fb);
        bulk_g2s(st + STAGE_WS, hsrc + (size_t)kt * NMAX * 128, (unsigned)(np * 128), fb);
        if (++s == NST) {
          s = 0;
          ph ^= 1;
        }
      }
      continue;
    }
    int s0 = s;
    unsigned ph0 = ph;
    for (int kt = 0; kt < NKT2; ++kt) {  // pass A: the unit's weights (h not published yet)
      uint64_t* fb = &full[s / KPS];
      if ((kt & (KPS - 1)) == 0) {
        mbar_wait(&empty[s / KPS], ph ^ 1);
        if (kt == 0) {
          *reinterpret_cast<volatile int*>(&su->y) = I.ebase;
          *reinterpret_cast<volatile int*>(&su->x) =
              rtile | (I.n << SU_NS) | (I.e << SU_ES) | (I.pid << SU_PS);
        }
        mbar_arrive_expect_tx(fb, (unsigned)(KPS * (STAGE_WS + np * 128)));
      }
      uint8_t* st = smem + s * STAGE;
      fc2_wload(P, st, I.e, rtile, kt, fb);
      if (++s == NST) {
        s = 0;
        ph ^= 1;
      }
    }
    while (ld_relaxed_gpu(P.h_flags + I.pid) != epoch) {
    }
    (void)ld_acquire_gpu(P.h_flags + I.pid);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    pready[I.pid >> 5] |= 1u << (I.pid & 31);
    for (int kt = 0; kt < NKT2; ++kt) {  // pass B: the unit's B operand
      bulk_g2s(smem + s0 * STAGE + STAGE_WS, hsrc + (size_t)kt * NMAX * 128, (unsigned)(np * 128),
               &full[s0 / KPS]);
      if (++s0 == NST) {
        s0 = 0;
        ph0 ^= 1;
      }
    }
  }
}

// Release helper (warp 18): per owner FC1 item of this CTA, in order: wait for the consumers'
// stores (epi[j & 3]), fence + count the item on its panel; when it is the panel's 4th, the warp
// quantizes the panel (per token over 512 columns) and publishes it.
__device__ __forceinline__ void quantize_panel(const Params& P, const Tabs& T, int pid, int t0,
                                               int nt);
__device__ __forceinline__ void quantize_panel_sh(const Params& P, const Tabs& T, int pid, int t0,
                                                  int nt);
__device__ __forceinline__ void release_helper(const Params& P, uint64_t* epi, const Tabs& T,
                                               unsigned epoch, unsigned par) {
  const int b = blockIdx.x, lane = threadIdx.x & 31;
  int j = 0;
  for (int g = 0; g < 2; ++g) {
    const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
    for (int gk = lo; gk < hi;) {
      const int it = gk / NKT1, kt0 = gk - it * NKT1;
      if (kt0 == 0) {
        mbar_wait(&epi[j & 3], (unsigned)(j >> 2) & 1u);
        ++j;
        const int pid = it / NQG;
        unsigned old = 0;
        // one acq_rel atomic: releases this item's act / amax (consumer stores ordered by the epi
        // mbarrier), and the panel's last (NQG-th) item acquires the others (the __shfl_sync orders
        // the other lanes' loads after lane 0's acquire)
        if (lane == 0) old = atom_add_acq_rel_gpu(P.pan_cnt + par * MAXP + pid, 1u);
        old = __shfl_sync(0xffffffffu, old, 0);
        if (old == (unsigned)(NQG - 1)) {
          if (pinf_b(T.pinf[pid]) >= T.misc[1])
            quantize_panel_sh(P, T, pid, lane, 32);
          else
            quantize_panel(P, T, pid, lane, 32);
          asm volatile("fence.proxy.async.global;" ::
                           : "memory");  // hq is read by bulk copies (async proxy)
          __syncwarp();
          if (lane == 0) st_release_gpu(P.h_flags + pid, epoch);
        }
      }
      gk = (it + 1) * NKT1;
    }
  }
}

// FC2 store warps (warps 17, 19): drain each finished out tile to every (token, slot) of its expert
// with 16-B stores.
__device__ __forceinline__ void store_helper(const Params& P, uint8_t* smem, const Tabs& T,
                                             uint64_t* fullo, uint64_t* freeo, uint64_t* sdone,
                                             int half, unsigned par) {
  const int lt = threadIdx.x & 31;
  for (int j = half;; j += 2) {  // tiles half, half + 2, .. (= out buffer half)
    const int bo = j & 1;
    mbar_wait(&fullo[bo], (unsigned)(j >> 1) & 1u);
    const volatile int* oi =
        reinterpret_cast<const volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRT) * 4) + bo * 4;
    const int n = oi[2];
    if (n < 0) break;
    const int rtile = oi[0], ebase = oi[1];
    const __nv_bfloat16* out_t =
        reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);
    const bool shu = ebase >= T.misc[1];  // shared expert unit: its token rows of the shared output
    if (lt <
        n) {  // one lane per token row: a 512-B bulk copy (async proxy) per destination slot row
      {
        const int t = T.ent[ebase + lt];
        const __nv_bfloat16* src = out_t + lt * OUT_PITCH;
        if (shu)
          bulk_s2g(P.shout + (size_t)t * DIM + rtile * 256, src, 512u);
        else {
          unsigned km = T.km[ebase + lt];
          __nv_bfloat16* o = P.outp + (size_t)t * TOPK_MAX * DIM + rtile * 256;
          while (km) {
            const int k = __ffs(km) - 1;
            km &= km - 1;
            bulk_s2g(o + (size_t)k * DIM, src, 512u);
          }
        }
        bulk_commit();
        bulk_wait_read0();  // the tile's smem has been read
      }
    }
    __syncwarp();
    if ((threadIdx.x & 31) == 0) mbar_arrive(&freeo[bo]);  // the tile's smem reads are done
    // this unit's global stores done -> count it on its row tile (T1 of the row tile waits for its
    // count)
    bulk_wait0();
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
      fence_acq_rel_gpu();
      atomicAdd(P.rt_cnt + par * NRT + rtile, 1u);
    }
  }
  __syncwarp();
  if ((threadIdx.x & 31) == 0) mbar_arrive(sdone);
}

// ------------------------------------------------------------------------------------------------
// consumer roles
struct Lane {
  int tid, w, wq, lane, g, tig, Lb, frag_off, scl_off;
};
__device__ __forceinline__ Lane lane_of() {
  Lane L;
  L.tid = threadIdx.x;
  L.lane = L.tid & 31;
  const int warp = L.tid >> 5;
  L.w = warp >> 2;
  L.wq = warp & 3;
  L.g = L.lane >> 2;
  L.tig = L.lane & 3;
  const int sel = ((L.g >> 1) + (L.wq & 1)) & 1;
  L.Lb = (L.wq >> 1) * 32 + sel * 16 +
         L.g;  // logical row (within the 64-row block) of accumulator rows g; + 8 for rows g + 8
  L.frag_off = L.w * HLB + ((L.wq >> 1) * 32 + L.lane) * 16 + sel * 8;
  L.scl_off = STAGE_W + L.w * 64 + 8 * L.g + 4 * (L.wq >> 1) + 2 * sel;
  return L;
}

template <int N>
__device__ __forceinline__ void fc1_piece(const Params& P, uint8_t* smem, uint64_t* full,
                                          uint64_t* empty, const Tabs& T, const Lane& L, int& s,
                                          unsigned& ph, const Item& I, int grp, int it, int kt0,
                                          int kt1, unsigned epoch, uint64_t* epi_bar) {
  constexpr int R = N / 2;
  const int G = gridDim.x, b = blockIdx.x, tid = L.tid;
  float C[R], D[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    C[i] = 0.f;
    D[i] = 0.f;
  }
  const uint64_t desc0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
  // owner: the epilogue's per-column factor -- loads issued now, consumed (stored to smem) after
  // the k-loop (no T.am zeroing here: the previous owner epilogue's tid < n threads may still be
  // reading T.am -- stage 2 writes every
  //  (token, warp) cell of the N tokens, 0 where the warp has no element of that token)
  // (fused quantization: block-wide -- the consumers acquired every token before FC1; dense -- warp
  // 19 stages the scales in
  //  TAB_XS (xsb), read after the k-loop: no consumer warp polls a flag ahead of its first k-step)
  const bool fs_mine = kt0 == 0 && tid < (P.dense ? P.md : I.n);
  const int fs_t =
      fs_mine ? (P.dense ? tid : (int)T.ent[I.ebase + tid]) : P.M;  // dense: column = token
  float fs_x = 0.f, fs_s2 = 0.f;
  if (fs_t < P.M) {
    if (!P.dense) fs_x = __ldcg(P.xs + fs_t);
    fs_s2 =
        __ldg((TAIL && I.q == NQG - 1
                   ? (T4 && (N == 8 || (N <= T4MAXN && (dflag(P) & 2))) ? P.fc1t4_s2 : P.fc1t_s2)
                   : P.fc1_s2) +
              I.e);
  }
  int sprev = 0;
  const bool t4 = T4 && (N == 8 || (N <= T4MAXN && (dflag(P) & 2))) &&
                  I.q == NQG - 1;  // (N8 / N16 <=> panel width <= 8 / 16, dense md 8 / 16)
  const uint64_t dsc = desc0 + (t4 ? (uint64_t)(L.w * ((N * 128) >> 4))
                                   : 0ull);  // T4: warpgroup w reads x tile w of the stage (N rows)
  int nk = t4 ? (kt1 - kt0) >> 2 : kt1 - kt0;
  kstep<N, true>(smem, full, empty, s, ph, sprev, C, D, dsc + (uint64_t)(s * (STAGE >> 4)),
                 L.frag_off, L.scl_off, L.lane);
  --nk;
  for (; nk >= 2; nk -= 2) {
    kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, dsc + (uint64_t)(s * (STAGE >> 4)),
                    L.frag_off, L.scl_off, L.lane);
    kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, dsc + (uint64_t)(s * (STAGE >> 4)),
                    L.frag_off, L.scl_off, L.lane);
  }
  if (nk > 0)
    kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, dsc + (uint64_t)(s * (STAGE >> 4)),
                    L.frag_off, L.scl_off, L.lane);
  kfinish<N>(empty, sprev, C, D, L.lane);
  if (fs_mine) {
    if (P.dense && fs_t < P.M) {
      mbar_wait(reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR) + 2 * NST + 11, 0u);
      fs_x = reinterpret_cast<const float*>(smem + OFF_TAB + TAB_XS)[fs_t];
    }
    T.fs[tid] = fs_x * (fs_s2 * 64.0f);  // published by the epilogue's first barrier
  }

  if (kt0 != 0) {  // not the owner: publish the fp32 partial (thread-private slots) for the CTA
                   // that owns k-tile 0
#pragma unroll
    for (int i = 0; i < R / 4; ++i)
      *part_f4(P, grp * G + b, i, tid) =
          make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
    named_bar_sync(1, NCONS);  // warp 17 fences + flags it (the gbar phase of this group): no
                               // consumer waits on the gpu fence
    if (L.lane == 0)
      mbar_arrive(reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR) + 2 * NST + 5);
    return;
  }
  if (kt1 < NKT1) {  // owner of a split item: add the partials of CTAs b+1.. (their ranges start
                     // inside this item), CTA order
    const int iend = (it + 1) * NKT1;
    int nsrc = 0;
    while (b + 1 + nsrc < G && fc1_lo(T, grp, b + 1 + nsrc) < iend) ++nsrc;
    if (tid < nsrc && (BALIGN == 1 || fc1_lo(T, grp, b + 1 + tid) < fc1_lo(T, grp, b + 2 + tid)))
      wait_eq(P.part_flags + grp * G + b + 1 + tid,
              epoch);  // all sources polled in parallel (an empty range publishes none)
    named_bar_sync(1, NCONS);
#pragma unroll 2
    for (int k = 0; k < nsrc; ++k) {  // added in CTA order
      if (BALIGN > 1 && fc1_lo(T, grp, b + 1 + k) == fc1_lo(T, grp, b + 2 + k))
        continue;  // (empty range: no partial)
      float4 v[R / 4];
#pragma unroll
      for (int i = 0; i < R / 4; ++i) v[i] = __ldcg(part_f4(P, grp * G + b + 1 + k, i, tid));
#pragma unroll
      for (int i = 0; i < R / 4; ++i) {
        D[4 * i] += v[i].x;
        D[4 * i + 1] += v[i].y;
        D[4 * i + 2] += v[i].z;
        D[4 * i + 3] += v[i].w;
      }
    }
  }
  // ---- epilogue: gu = BF16(D * xs * s2 * 64); act = BF16(BF16(silu(min(gate, 10))) * clamp(up,
  // +-10))
  named_bar_sync(1, NCONS);  // T.fs (written at piece start) visible; partial reads done
  // stage 1: every warpgroup writes its BF16 gate (min 10) / up (clamp +-10) values,
  // lane-contiguous: gx[blk][i][128]
  __nv_bfloat16* gx = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG);
  if (T4 && t4) {  // the 4 warpgroups' K-quarter partials of the one tail block (rows 0..31
                   // gate, 32..63 up) -> gate / up of WG 0 / 2 rows 0..31
    float* S = reinterpret_cast<float*>(smem + OFF_STG + T4_SCR);
#pragma unroll
    for (int i = 0; i < R; ++i) S[(L.w * R + i) * 128 + (tid & 127)] = D[i];
    named_bar_sync(1, NCONS);
    const int t = tid & 127;
    const bool live = (L.w == 0 || L.w == 2) && t < 64;
    const int src = t + (L.w == 2 ? 64 : 0);
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1);
      float v = 0.f;
      if (live) {
        const float sum = ((S[(0 * R + i) * 128 + src] + S[(1 * R + i) * 128 + src]) +
                           S[(2 * R + i) * 128 + src]) +
                          S[(3 * R + i) * 128 + src];
        v = bf16r(sum * T.fs[tk]);
        v = L.w == 0 ? fminf(v, 10.f) : fminf(fmaxf(v, -10.f), 10.f);
      }
      gx[(L.w * (NMAX / 2) + i) * 128 + t] = __float2bfloat16_rn(v);
    }
  } else {
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1);
      const float v = bf16r(D[i] * T.fs[tk]);
      gx[(L.w * (NMAX / 2) + i) * 128 + (tid & 127)] =
          __float2bfloat16_rn(L.w < 2 ? fminf(v, 10.f) : fminf(fmaxf(v, -10.f), 10.f));
    }
  }
  named_bar_sync(1, NCONS);
  // stage 2: all 512 threads: (row half rh = w & 1, element half = w >> 1) of the (gate, up) pairs
  // of lane (tid & 127)
  {
    const int rh = L.w & 1, h0 = (L.w >> 1) * (R / 2);
    __nv_bfloat16* act_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + 16384) + rh * 64 +
                           L.Lb;  // [NMAX][ACT_PITCH]
    float am[R / 2];  // by global token slot j = 2 (i >> 2) + (i & 1) (compile-time); slots outside
                      // this half stay 0
#pragma unroll
    for (int j = 0; j < R / 2; ++j) am[j] = 0.f;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      if (i < h0 || i >= h0 + R / 2) continue;  // the other element half belongs to warpgroup w ^ 2
      const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1), r = (i >> 1) & 1;
      const float gate = __bfloat162float(gx[(rh * (NMAX / 2) + i) * 128 + (tid & 127)]);
      const float up = __bfloat162float(gx[((2 + rh) * (NMAX / 2) + i) * 128 + (tid & 127)]);
      const float sg = bf16r(
          __fdividef(gate, 1.0f + __expf(-gate)));  // (BF16-rounded right after: fast-math SiLU)
      const float a = bf16r(sg * up);
      act_t[tk * ACT_PITCH + r * 8] = __float2bfloat16_rn(a);
      if (P.dense || tk < I.n)
        am[2 * (i >> 2) + (i & 1)] = fmaxf(am[2 * (i >> 2) + (i & 1)], fabsf(a));
    }
#pragma unroll
    for (int j = 0; j < R / 2; ++j) {
      float v = am[j];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
      const int tk = 8 * (j >> 1) + 2 * L.tig + (j & 1);
      if (L.g == 0)
        T.am[(L.w * 4 + L.wq) * NMAX + tk] =
            v;  // (token, warp) cells are private: untouched slots write 0
    }
  }
  named_bar_sync(1, NCONS);
  if (tid < I.n) {  // entry tid of the panel: its column is tid (compact) or its token (dense)
    const int col = P.dense ? (int)T.ent[I.ebase + tid] : tid;
    float m = T.am[col];
#pragma unroll
    for (int j = 1; j < 16; ++j) m = fmaxf(m, T.am[j * NMAX + col]);
    P.amax[(size_t)(I.ebase + tid) * NQG + I.q] = m;
  }
  {
    const __nv_bfloat16* act_t = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + 16384);
    for (int c = tid; c < I.n * 16; c += NCONS) {
      const int tk = c >> 4, ch = c & 15, col = P.dense ? (int)T.ent[I.ebase + tk] : tk;
      *reinterpret_cast<uint4*>(P.act + (size_t)(I.ebase + tk) * INTER + I.q * 128 + ch * 8) =
          *reinterpret_cast<const uint4*>(act_t + col * ACT_PITCH + ch * 8);
    }
  }
  __syncwarp();
  if (L.lane == 0)
    mbar_arrive(epi_bar);  // the release helper (warp 18) fences and counts this item
}

// Quantize panel pid (all NQG items finished): per token amax over INTER columns -> hs, e4m3 rows
// [NKT2 k-tiles][NMAX][128] + hs.
__device__ __forceinline__ void quantize_panel(const Params& P, const Tabs& T, int pid, int t0,
                                               int nt) {
  const Item I = item_of(T, pid, 0);
  for (int ch = t0; ch < I.n * CPT; ch += nt) {
    const int tk = ch / CPT, c16 = ch - tk * CPT, kb = c16 >> 3, ci = c16 & 7, en = I.ebase + tk;
    float amax;
    if constexpr (NQG == 4) {
      const float4 a4 = __ldcg(reinterpret_cast<const float4*>(P.amax + (size_t)en * 4));
      amax = fmaxf(fmaxf(a4.x, a4.y), fmaxf(a4.z, a4.w));
    } else if constexpr (NQG == 2) {
      const float2 a2 = __ldcg(reinterpret_cast<const float2*>(P.amax + (size_t)en * 2));
      amax = fmaxf(a2.x, a2.y);
    } else {
      amax = __ldcg(P.amax + (size_t)en * NQG);
#pragma unroll
      for (int q = 1; q < NQG; ++q) amax = fmaxf(amax, __ldcg(P.amax + (size_t)en * NQG + q));
    }
    const float hs = __fmul_rn(fmaxf(amax, 1e-30f), 0.0022321429569274187f);
    float inv;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(hs));
    const uint4* src = reinterpret_cast<const uint4*>(P.act + (size_t)en * INTER + c16 * 16);
    const uint4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    uint32_t q4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      q4[j] = (uint32_t)pack_e4m3x2(bf16lo(w[2 * j]) * inv, bf16hi(w[2 * j]) * inv) |
              ((uint32_t)pack_e4m3x2(bf16lo(w[2 * j + 1]) * inv, bf16hi(w[2 * j + 1]) * inv) << 16);
    *reinterpret_cast<uint4*>(P.hq + ((size_t)(pid * NKT2 + kb) * NMAX + tk) * 128 +
                              ((ci ^ (tk & 7)) << 4)) = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    if (c16 == 0) P.hs[pid * NMAX + tk] = hs;
  }
}

// ------------------------------------------------------------------------------------------------
// fused shared expert
static_assert(KPS == 1, "the fused shared expert assumes one k-tile per ring stage");
// D = fma(sa[t] * sb, C, D) per accumulator column (token t = 8 j + 2 tig + h): deep_gemm's
// per-K128 promotion (scale_a * scale_b, FFMA)
template <int N>
__device__ __forceinline__ void promote_sh(const float* sa, float sb, const float (&C)[N / 2],
                                           float (&D)[N / 2], const Lane& L) {
#pragma unroll
  for (int j = 0; j < N / 8; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float sc = __fmul_rn(sa[8 * j + 2 * L.tig + h], sb);
      D[4 * j + h] = __fmaf_rn(sc, C[4 * j + h], D[4 * j + h]);
      D[4 * j + 2 + h] = __fmaf_rn(sc, C[4 * j + 2 + h], D[4 * j + 2 + h]);
    }
  }
}
// One k-block of the shared expert = two ring slots (a logical stage): slot s0 = rows 0..127
// (warpgroups 0, 1) + the B tile + the B columns' K128 scales, slot s1 = rows 128..255 (warpgroups
// 2, 3).  Every warpgroup runs the same code (no divergent wgmma: the slot is a data select):
// retire the previous chain (D = fma(sa sb, C, D) with its scales, release its two slots), issue
// this one.
__device__ __forceinline__ void adv_slot(int& s, unsigned& ph) {
  if (++s == NST) {
    s = 0;
    ph ^= 1;
  }
}
// DSV4.1 shared expert (SH32): FP8 weight scales per (32 rows, 32 k) -- a rank's 288-row slice
// straddles the checkpoint's 128-row blocks, and DSV4.1 recipes may use 32x32 blocks -- so every
// k32 slice is its own SS wgmma, promoted with sa[t] * sw[row block][k32]: one k-block (two ring
// slots, as kstep_sh) = 4 x (wgmma, wait, promote); both slots released at its end SH32 FC2 tail
// k-block (TAIL builds): real k32 slices = (INTER_R % 128) / 32 -- TP8 288: 1 (k 256..287), TP4
// 576: 2 (k 512..575); the remaining slices of that k-block are zero-filled weights x zero act
constexpr int SH_TAIL_SL = TAIL ? (INTER_R % 128) / 32 : 4;
static_assert(!TAIL || (INTER_R % 32 == 0 && SH_TAIL_SL >= 1 && SH_TAIL_SL <= 3),
              "SH32 FC2 tail k-block: 1..3 real k32 slices");
template <int N>
__device__ __forceinline__ void kstep_sh32(
    uint8_t* smem, uint64_t* full, uint64_t* empty, int& s, unsigned& ph, float (&C)[N / 2],
    float (&D)[N / 2], float4 sb4, const Lane& L, uint64_t descB0,
    bool one =
        false) {  // one: only the first SH_TAIL_SL slices (the FC2 tail k-block: the others are
                  // zero weights x zero act); sb4: this k-block's 4 weight scales (prefetched by
                  // the caller)   // (before the stage wait: off the critical path)
  const int s0 = s;
  mbar_wait(&full[s], ph);
  adv_slot(s, ph);
  const int s1 = s;
  mbar_wait(&full[s], ph);
  adv_slot(s, ph);
  const int sw = (L.w >> 1) ? s1 : s0;
  const uint64_t da =
      make_desc_sw128(smem_u32(smem + sw * STAGE)) + (uint64_t)((L.w & 1) * ((64 * 128) >> 4));
  const uint64_t db = descB0 + (uint64_t)(s0 * (STAGE >> 4));
  const float* sa = reinterpret_cast<const float*>(smem + s0 * STAGE + STAGE_W);
  const float sb[4] = {fabsf(sb4.x), sb4.y, sb4.z, sb4.w};  // (|scale|: scales are positive)
  if (one) {
#pragma unroll
    for (int q = 0; q < SH_TAIL_SL; ++q) {  // (TP8: q = 0 only -- the former single-slice path)
      fence_operand(D);
      fence_operand(C);
      wg_fence();
      chain_ss::ChainSS1<N>::run(C, da + (uint64_t)(2 * q), db + (uint64_t)(2 * q));
      wg_commit();
      wg_wait0();
      fence_operand(C);
      promote_sh<N>(sa, sb[q], C, D, L);
    }
    fence_operand(D);
    if (L.lane == 0) {
      mbar_arrive(&empty[s0]);
      mbar_arrive(&empty[s1]);
    }
    return;
  }
  float C1[N / 2];
  fence_operand(D);
  fence_operand(C);
  wg_fence();
  chain_ss::ChainSS1<N>::run(C, da, db);
  wg_commit();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float(&Cn)[N / 2] = (q & 1) ? C : C1;
    float(&Cc)[N / 2] = (q & 1) ? C1 : C;
    if (q < 3) {
      fence_operand(Cn);
      wg_fence();
      chain_ss::ChainSS1<N>::run(Cn, da + (uint64_t)(2 * (q + 1)), db + (uint64_t)(2 * (q + 1)));
      wg_commit();
      wg_wait1();
    } else {
      wg_wait0();
    }
    fence_operand(Cc);
    promote_sh<N>(sa, sb[q], Cc, D, L);
    fence_operand(D);
  }
  fence_operand(D);
  if (L.lane == 0) {
    mbar_arrive(&empty[s0]);
    mbar_arrive(&empty[s1]);
  }
}

// Shared expert FC1 piece: item q = act columns 128 q .. (gate rows 128 q.. on warpgroups 0, 1; up
// rows 512 + 128 q.. on 2, 3), k-blocks [kt0, kt1).  Stream-K partials / owner merge as fc1_piece;
// the owner's epilogue: gu = BF16(D) (deep_gemm), gate min 10 / up clamp
// +-10 in BF16, act = BF16(up * silu(gate)) with the production fast-math SiLU, per-token amax of
// the item's 128 columns (= its group).
template <int N>
__device__ __forceinline__ void fc1_piece_sh(const Params& P, uint8_t* smem, uint64_t* full,
                                             uint64_t* empty, const Tabs& T, const Lane& L, int& s,
                                             unsigned& ph, const Item& I, int grp, int it, int kt0,
                                             int kt1, unsigned epoch, uint64_t* epi_bar) {
  constexpr int R = N / 2;
  const int G = gridDim.x, b = blockIdx.x, tid = L.tid;
  float C[R], D[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    C[i] = 0.f;
    D[i] = 0.f;
  }
  const uint64_t descB0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
  {  // row block (32) of this thread: gate | up half, item q, warpgroup row half, warp pair
    const float* swr =
        P.sb1 + (size_t)(((L.w >> 1) ? INTER / 32 : 0) + I.q * 4 + (L.w & 1) * 2 + (L.wq >> 1)) *
                    (DIM / 32);
    for (int kt = kt0; kt < kt1; ++kt)
      kstep_sh32<N>(smem, full, empty, s, ph, C, D,
                    __ldg(reinterpret_cast<const float4*>(swr + kt * 4)), L, descB0);
  }
  if (kt0 != 0) {  // not the owner: publish the fp32 partial for the CTA that owns k-block 0
#pragma unroll
    for (int i = 0; i < R / 4; ++i)
      *part_f4(P, grp * G + b, i, tid) =
          make_float4(D[4 * i], D[4 * i + 1], D[4 * i + 2], D[4 * i + 3]);
    named_bar_sync(1, NCONS);  // warp 17 fences + flags it (the gbar phase of this group): no
                               // consumer waits on the gpu fence
    if (L.lane == 0)
      mbar_arrive(reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR) + 2 * NST + 5);
    return;
  }
  if (kt1 < NKT1) {  // owner of a split item: add the partials of CTAs b+1.. in CTA order
    const int iend = (it + 1) * NKT1;
    int nsrc = 0;
    while (b + 1 + nsrc < G && fc1_lo(T, grp, b + 1 + nsrc) < iend) ++nsrc;
    if (tid < nsrc && (BALIGN == 1 || fc1_lo(T, grp, b + 1 + tid) < fc1_lo(T, grp, b + 2 + tid)))
      wait_eq(P.part_flags + grp * G + b + 1 + tid, epoch);  // (an empty range publishes none)
    named_bar_sync(1, NCONS);
#pragma unroll 2
    for (int k = 0; k < nsrc; ++k) {
      if (BALIGN > 1 && fc1_lo(T, grp, b + 1 + k) == fc1_lo(T, grp, b + 2 + k))
        continue;  // (empty range: no partial)
      float4 v[R / 4];
#pragma unroll
      for (int i = 0; i < R / 4; ++i) v[i] = __ldcg(part_f4(P, grp * G + b + 1 + k, i, tid));
#pragma unroll
      for (int i = 0; i < R / 4; ++i) {
        D[4 * i] += v[i].x;
        D[4 * i + 1] += v[i].y;
        D[4 * i + 2] += v[i].z;
        D[4 * i + 3] += v[i].w;
      }
    }
  }
  named_bar_sync(1, NCONS);  // partial reads done (the staging area below is shared with them only
                             // through the ring: none)
  // stage 1: BF16 gate (min 10) / up (clamp +-10), lane-contiguous gx[blk][i][128] (as fc1_piece)
  __nv_bfloat16* gx = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG);
#pragma unroll
  for (int i = 0; i < R; ++i) {
    const __nv_bfloat16 v = __float2bfloat16_rn(D[i]);
    const __nv_bfloat16 lim = __float2bfloat16_rn(10.f), nlim = __float2bfloat16_rn(-10.f);
    gx[(L.w * (NMAX / 2) + i) * 128 + (tid & 127)] =
        L.w < 2 ? __hmin(v, lim) : __hmin(__hmax(v, nlim), lim);
  }
  named_bar_sync(1, NCONS);
  // stage 2: (row half rh = w & 1, element half = w >> 1) of the (gate, up) pairs of lane (tid &
  // 127); natural accumulator rows
  {
    const int rh = L.w & 1, h0 = (L.w >> 1) * (R / 2), Lrow = 16 * L.wq + L.g;
    __nv_bfloat16* act_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + 16384) + rh * 64 +
                           Lrow;  // [NMAX][ACT_PITCH]
    float am[R / 2];
#pragma unroll
    for (int j = 0; j < R / 2; ++j) am[j] = 0.f;
#pragma unroll
    for (int i = 0; i < R; ++i) {
      if (i < h0 || i >= h0 + R / 2) continue;
      const int tk = 8 * (i >> 2) + 2 * L.tig + (i & 1), r = (i >> 1) & 1;
      const float gate = __bfloat162float(gx[(rh * (NMAX / 2) + i) * 128 + (tid & 127)]);
      const float up = __bfloat162float(gx[((2 + rh) * (NMAX / 2) + i) * 128 + (tid & 127)]);
      const float sg = shnum::mul_ftz(
          gate, shnum::rcp_approx_ftz(shnum::add_ftz(
                    shnum::ex2_approx_ftz(shnum::mul_ftz(gate, -1.4426950216293334961f)), 1.0f)));
      const float a = bf16r(shnum::mul_ftz(up, sg));
      act_t[tk * ACT_PITCH + r * 8] = __float2bfloat16_rn(a);
      if (P.dense || tk < I.n)
        am[2 * (i >> 2) + (i & 1)] = fmaxf(am[2 * (i >> 2) + (i & 1)], fabsf(a));
    }
#pragma unroll
    for (int j = 0; j < R / 2; ++j) {
      float v = am[j];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
      const int tk = 8 * (j >> 1) + 2 * L.tig + (j & 1);
      if (L.g == 0) T.am[(L.w * 4 + L.wq) * NMAX + tk] = v;
    }
  }
  named_bar_sync(1, NCONS);
  if (tid < I.n) {  // entry tid = token t0 + tid: this item's group amax
    const int col = P.dense ? (int)T.ent[I.ebase + tid] : tid;
    float m = T.am[col];
#pragma unroll
    for (int j = 1; j < 16; ++j) m = fmaxf(m, T.am[j * NMAX + col]);
    P.amax[(size_t)(I.ebase + tid) * NQG + I.q] = m;
  }
  {
    const __nv_bfloat16* act_t = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_STG + 16384);
    for (int c = tid; c < I.n * 16; c += NCONS) {
      const int tk = c >> 4, ch = c & 15, col = P.dense ? (int)T.ent[I.ebase + tk] : tk;
      *reinterpret_cast<uint4*>(P.act + (size_t)(I.ebase + tk) * INTER + I.q * 128 + ch * 8) =
          *reinterpret_cast<const uint4*>(act_t + col * ACT_PITCH + ch * 8);
    }
  }
  __syncwarp();
  if (L.lane == 0)
    mbar_arrive(epi_bar);  // the release helper counts the item; the panel's last item quantizes it
                           // (quantize_panel_sh)
}

// Quantize shared panel pid: per (token, 128-group = k-block) as SGLang per_token_group_quant (amax
// = max(1e-10, group amax), x * (448 * rcp(amax)), clamp, RN satfinite e4m3; dequant scale amax *
// (1/448)) -> hq rows [NKT2][NMAX][128] + hs4.
__device__ __forceinline__ void quantize_panel_sh(const Params& P, const Tabs& T, int pid, int t0,
                                                  int nt) {
  const Item I = item_of(T, pid, 0);
  for (int ch = t0; ch < I.n * CPT; ch += nt) {
    const int tk = ch / CPT, c16 = ch - tk * CPT, kb = c16 >> 3, ci = c16 & 7, en = I.ebase + tk;
    const float amax =
        shnum::max_ftz(shnum::gq_amax_init(), __ldcg(P.amax + (size_t)en * NQG + kb));
    const float mult = shnum::gq_mult(amax);
    const uint4* src = reinterpret_cast<const uint4*>(P.act + (size_t)en * INTER + c16 * 16);
    const uint4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    uint32_t q4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      q4[j] = (uint32_t)shnum::gq_pack2(bf16lo(w[2 * j]), bf16hi(w[2 * j]), mult) |
              ((uint32_t)shnum::gq_pack2(bf16lo(w[2 * j + 1]), bf16hi(w[2 * j + 1]), mult) << 16);
    *reinterpret_cast<uint4*>(P.hq + ((size_t)(pid * NKT2 + kb) * NMAX + tk) * 128 +
                              ((ci ^ (tk & 7)) << 4)) = make_uint4(q4[0], q4[1], q4[2], q4[3]);
    if (ci == 0) P.hs4[((size_t)pid * NKT2 + kb) * NMAX + tk] = shnum::gq_scale(amax);
  }
}

// Shared expert FC2 unit (256 down rows of row tile rtile): stages (k-block kb, row half hh), half
// hh on warpgroups 2 hh, 2 hh + 1; out = BF16(D) into the out tile (natural rows), the store warps
// write it to the tokens' shared output rows.
template <int N>
__device__ __forceinline__ void fc2_unit_sh(const Params& P, uint8_t* smem, uint64_t* full,
                                            uint64_t* empty, const Tabs& T, const Lane& L, int& s,
                                            unsigned& ph, const Item& I, int rtile, int j,
                                            uint64_t* fullo, uint64_t* freeo) {
  constexpr int R = N / 2;
  float C[R], D[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    C[i] = 0.f;
    D[i] = 0.f;
  }
  const uint64_t descB0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
  {  // row block (32): row tile, warpgroup (64 rows), warp pair
    const float* swr = P.sb2 + (size_t)(rtile * 8 + L.w * 2 + (L.wq >> 1)) * (INTER / 32);
#pragma unroll 1
    for (int kb = 0; kb < NKT2; ++kb)
      kstep_sh32<N>(smem, full, empty, s, ph, C, D,
                    __ldg(reinterpret_cast<const float4*>(swr + kb * 4)), L, descB0,
                    TAIL && kb == NKT2 - 1);
  }
  const int bo = j & 1;
  if (j >= 2) mbar_wait(&freeo[bo], (unsigned)((j >> 1) - 1) & 1u);  // tile drained (unit j - 2)
  __nv_bfloat16* out_t = reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);
  const int Lrow = 16 * L.wq + L.g;
#pragma unroll
  for (int j2 = 0; j2 < R / 2; ++j2) {
    const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
    const int i0 = ((j2 >> 1) << 2) | (j2 & 1);
    __nv_bfloat16* o = out_t + tk * OUT_PITCH + L.w * 64 + Lrow;
    o[0] = __float2bfloat16_rn(D[i0]);
    o[8] = __float2bfloat16_rn(D[i0 + 2]);
  }
  if (L.tid == 0) {
    volatile int* oi =
        reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRT) * 4) + bo * 4;
    oi[0] = rtile;
    oi[1] = I.ebase;
    oi[2] = I.n;
  }
  fence_proxy_async_smem();  // (generic smem writes -> the store warps' bulk copies)
  __syncwarp();
  if (L.lane == 0) mbar_arrive(&fullo[bo]);
}

template <int N>
__device__ __forceinline__ void fc2_unit(const Params& P, uint8_t* smem, uint64_t* full,
                                         uint64_t* empty, const Tabs& T, const Lane& L, int& s,
                                         unsigned& ph, const Item& I, int rtile, int j,
                                         uint64_t* fullo, uint64_t* freeo) {
  constexpr int R = N / 2;
  const float f2r = ldv_nc_f32(P.fc2_s2 + I.e);
  float C[R], D[R];
#pragma unroll
  for (int i = 0; i < R; ++i) {
    C[i] = 0.f;
    D[i] = 0.f;
  }
  const uint64_t desc0 = make_desc_sw128(smem_u32(smem + STAGE_WS));
  int sprev = 0;
  kstep<N, true>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                 L.frag_off, L.scl_off, L.lane);
  float hsr[R / 2];  // the stage landed => the gather lanes saw the panel's h flag: its scales are
                     // readable
#pragma unroll
  for (int j = 0; j < R / 2; ++j)
    hsr[j] = ldv_cg_f32(P.hs + I.pid * NMAX + 8 * (j >> 1) + 2 * L.tig + (j & 1));
  kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                  L.frag_off, L.scl_off, L.lane);
  if constexpr (NKT2 >= 3) {  // K = INTER: 4 k-steps (TP4) | 2 (TP8) | 3 (DSV4.1)
    if constexpr (NKT2 == 3 && TP_MOE_FC2_T1S == 2)
      kstep_p1<N>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                  L.frag_off, L.scl_off, L.lane);
    else
      kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                      L.frag_off, L.scl_off, L.lane);
  }
  if constexpr (NKT2 >= 4)
    kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                    L.frag_off, L.scl_off, L.lane);
  if constexpr (NKT2 >= 5)  // (TP4: k-tile 4 = slices 16, 17 + 2 zero-filled)
    kstep<N, false>(smem, full, empty, s, ph, sprev, C, D, desc0 + (uint64_t)(s * (STAGE >> 4)),
                    L.frag_off, L.scl_off, L.lane);
  static_assert(NKT2 <= 5, "fc2_unit k-steps");
  kfinish<N>(empty, sprev, C, D, L.lane);
  {
    const float f2 = f2r * 64.0f;
#pragma unroll
    for (int j = 0; j < R / 2; ++j) hsr[j] *= f2;  // (hs * (s2 * 64))
  }
  // ---- epilogue: BF16(D * hs * s2 * 64) -> out tile (j & 1); the store warps write it to every
  // (token, slot) of this expert
  const int bo = j & 1;
  if (j >= 2) mbar_wait(&freeo[bo], (unsigned)((j >> 1) - 1) & 1u);  // tile drained (unit j - 2)
  __nv_bfloat16* out_t =
      reinterpret_cast<__nv_bfloat16*>(smem + OFF_STG + bo * OUT_TILE);  // [NMAX][OUT_PITCH]
#pragma unroll
  for (int j2 = 0; j2 < R / 2; ++j2) {
    const int tk = 8 * (j2 >> 1) + 2 * L.tig + (j2 & 1);
    const int i0 = ((j2 >> 1) << 2) |
                   (j2 & 1);  // accumulator index of (token slot j2, row g); + 2 for row g + 8
    __nv_bfloat16* o = out_t + tk * OUT_PITCH + L.w * 64 + L.Lb;
    o[0] = __float2bfloat16_rn(D[i0] * hsr[j2]);
    o[8] = __float2bfloat16_rn(D[i0 + 2] * hsr[j2]);
  }
  if (L.tid == 0) {
    volatile int* oi =
        reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRT) * 4) + bo * 4;
    oi[0] = rtile;
    oi[1] = I.ebase;
    oi[2] = I.n;
  }
  fence_proxy_async_smem();  // (generic smem writes -> the store warps' bulk copies)
  __syncwarp();
  if (L.lane == 0) mbar_arrive(&fullo[bo]);
}

// ---- tail.  Message m of (row tile, token): rows rtile*256 + 6m .. +5 (m = 42: 4 rows), 3 bf16x2
// words + epoch tag. the message's place in its owner's reduce-scatter buffer: [src][t][slot rtile
// / NDEV][m]
__device__ __forceinline__ size_t rs_at(int src, int rtile, int t, int m) {
  return ((size_t)(src * MAXM + t) * NSLOT + (rtile >> (NDEV == 8 ? 3 : 2))) * NMSG + m;
}
__device__ __forceinline__ size_t ag_idx(int rtile, int t, int m) {
  return ((size_t)(rtile * MAXM + t)) * NMSG + m;
}
// one-shot mode (M <= OS_MAXM): every rank pushes all its messages to all NDEV ranks (in the RS
// buffers), each rank reduces locally
constexpr int OS_MAXM =
    MAXM / NDEV;  // 128 | 64: NDEV x NRT x OS_MAXM one-shot rows fit the NDEV x LRT x MAXM RS rows
static_assert(NRT * OS_MAXM <= NSLOT * MAXM, "one-shot messages fit the RS buffer");
__device__ __forceinline__ size_t os_idx(int src, int rtile, int t, int m) {
  return ((size_t)((src * NRT + rtile) * OS_MAXM + t)) * NMSG + m;
}
// rs / ag double-buffered by epoch parity (RS_BYTES / AG_BYTES hold two halves): a rank that pushed
// its call-k messages but has not read its peers' yet (late launch, preemption, time-slicing) is
// not overrun by peers that finished call k and run call k+1 (the equality-tag waits would never
// match again); no rank gets two calls ahead
constexpr size_t RS_REG =
    (size_t)NDEV * NSLOT * MAXM * NMSG;  // uint4: the RS / one-shot message region of one call
constexpr size_t AG_REG = (size_t)NRT * MAXM * NMSG;
constexpr size_t RS_PAR = RS_REG, AG_PAR = AG_REG;  // parity halves

// T1 of one (row tile, token) pair by one warp: lane l owns values rtile * 256 + 8l .. +7 (one 16-B
// load per slot), value by value: slot-ordered fp32 sum of BF16 x weight, x factor -> BF16, +
// shared -> BF16; the 256 BF16 staged in stg (per warp), then the row tile's 43 messages
__device__ __forceinline__ void t1_pair_vec(const Params& P, const Tabs& T, int rtile, int t,
                                            unsigned epoch, __nv_bfloat16* stg, int lane) {
  const int topk = P.topk;
  const bool rws = P.glog && P.M * topk <= TP_MOE_SMALL_MAXMT &&
                   P.M <= NMAX;  // fused gate, small tables: weights in smem
  const uint4 idrow = *reinterpret_cast<const uint4*>(T.ids + t * 8);
  const uint32_t idw[4] = {idrow.x, idrow.y, idrow.z, idrow.w};
  const int c0 = rtile * 256 + 8 * lane;
  uint4 v[TOPK_MAX];
  float wk[TOPK_MAX];
  unsigned on = 0u;
#pragma unroll
  for (int k = 0; k < TOPK_MAX; ++k) {
    if (k < topk && ((idw[k >> 1] >> ((k & 1) * 16)) & 0xFFFFu) != 0xFFFFu) {
      on |= 1u << k;
      wk[k] = rws ? T.grw[t * topk + k] : __ldcg(P.rw + t * topk + k);
      v[k] = __ldcg(reinterpret_cast<const uint4*>(P.outp + ((size_t)t * TOPK_MAX + k) * DIM + c0));
    }
  }
  const uint4 sh = __ldcg(reinterpret_cast<const uint4*>(P.shout + (size_t)t * DIM + c0));
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < TOPK_MAX; ++k) {
    if (!((on >> k) & 1u)) continue;
    const uint32_t w4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[2 * j] = __fadd_rn(acc[2 * j], __fmul_rn(bf16lo(w4[j]), wk[k]));
      acc[2 * j + 1] = __fadd_rn(acc[2 * j + 1], __fmul_rn(bf16hi(w4[j]), wk[k]));
    }
  }
  const uint32_t s4[4] = {sh.x, sh.y, sh.z, sh.w};
  uint32_t o4[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float y0 = bf16r(__fmul_rn(acc[2 * j], P.factor)),
          y1 = bf16r(__fmul_rn(acc[2 * j + 1], P.factor));
    y0 = bf16r(__fadd_rn(y0, bf16lo(s4[j])));
    y1 = bf16r(__fadd_rn(y1, bf16hi(s4[j])));
    o4[j] = pack_bf16x2_rn(y0, y1);
  }
  *reinterpret_cast<uint4*>(stg + 8 * lane) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
  __syncwarp();
  for (int m = lane; m < NMSG; m += 32) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(stg + 6 * m);
    const bool full6 = m < NMSG - 1;
    const uint4 msg = make_uint4(w[0], w[1], full6 ? w[2] : 0u, epoch);
    if (P.oneshot) {
#pragma unroll
      for (int d = 0; d < NDEV; ++d)
        st_ll(P.rs[(P.my_rank + d) & (NDEV - 1)] + (epoch & 1u) * RS_PAR +
                  os_idx(P.my_rank, rtile, t, m),
              msg);  // own copy first
    } else {
      st_ll(P.rs[rs_owner(rtile, t)] + (epoch & 1u) * RS_PAR + rs_at(P.my_rank, rtile, t, m), msg);
    }
  }
  __syncwarp();  // the staging row is rewritten by the warp's next pair
}

__device__ __forceinline__ void tail(const Params& P, const Tabs& T, int tid, unsigned epoch,
                                     unsigned par, uint8_t* smem) {
  const int G = gridDim.x, b = blockIdx.x, M = P.M, topk = P.topk;
  const unsigned need = (unsigned)T.misc[2];
  // tokens per item: the fewest tokens per item with NRT x items-per-row-tile <= G, i.e. at most
  // one item per CTA -- 96 items on 78 CTAs left 18 CTAs a second item (the T1 critical path) at
  // M128 / 256 / 384)
  const int TCH = (M + max(1, G / NRT) - 1) / max(1, G / NRT);
  const int ntc = (M + TCH - 1) / TCH, nitem = NRT * ntc;
  volatile int* s_item = reinterpret_cast<volatile int*>(const_cast<int*>(T.misc) + 16);
  unsigned done_rt = 0u;  // row tiles known complete (thread 0)
  for (;;) {
    if (tid == 0) {
      const int it = (int)atomicAdd(P.fc2_next + 2 + par, 1u);
      if (it < nitem) {
        const int rt = it / ntc;
        if (!((done_rt >> rt) & 1u)) {
          wait_ge(P.rt_cnt + par * NRT + rt, need);
          done_rt |= 1u << rt;
        }
      }
      *s_item = it;
    }
    named_bar_sync(1, NCONS);
    const int it = *s_item;
    named_bar_sync(1, NCONS);  // (s_item reread by thread 0 next round only after everyone read it)
    if (it >= nitem) break;
    const int rtile = it / ntc, t0 = (it - rtile * ntc) * TCH, nt = min(TCH, M - t0);
    // one warp per (row tile, token); staging rows in the idle ring (256 BF16 per warp)
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem) + (tid >> 5) * 256;
    for (int tt = tid >> 5; tt < nt; tt += NCONS / 32)
      t1_pair_vec(P, T, rtile, t0 + tt, epoch, stg, tid & 31);
  }
  if (P.oneshot) {  // every (row tile, token) of this CTA: the NDEV ranks' messages, summed in rank
                    // order (fp32), one BF16 RN
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
    for (int x = tid; x < mine * NMSG; x += NCONS) {
      const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
      uint4 v[NDEV];
      unsigned pending = (1u << NDEV) - 1u;
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
      uint32_t* o = reinterpret_cast<uint32_t*>(P.out + (size_t)t * DIM + rtile * 256 + 6 * m);
      o[0] = pack_bf16x2_rn(a[0], a[1]);
      o[1] = pack_bf16x2_rn(a[2], a[3]);
      if (m < NMSG - 1) o[2] = pack_bf16x2_rn(a[4], a[5]);
    }
    return;
  }
  // T2: owner reduce in rank order (fp32 from rank 0's value, one BF16 RN), all-gather push
  {
    const int npair = NSLOT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* rs_me = P.rs[P.my_rank] + (epoch & 1u) * RS_PAR;
    for (int x = tid; x < mine * NMSG; x += NCONS) {
      const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, lrt = j / M, t = j - lrt * M;
      // the owned row tile of slot lrt for token t (rs_owner inverted); slots past the last row
      // tile are empty
      const int rt_own = ((P.my_rank - t) & (NDEV - 1)) + NDEV * lrt;
      if (rt_own >= NRT) continue;
      uint4 v[NDEV];
      unsigned pending = (1u << NDEV) - 1u;
      do {
#pragma unroll
        for (int d = 0; d < NDEV; ++d)
          if (pending & (1u << d)) {
            v[d] = ld_ll(rs_me + rs_at(d, rt_own, t, m));
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
      const uint4 mo = make_uint4(pack_bf16x2_rn(a[0], a[1]), pack_bf16x2_rn(a[2], a[3]),
                                  pack_bf16x2_rn(a[4], a[5]), epoch);
      const int rtile = rt_own;
#pragma unroll
      for (int d = 0; d < NDEV; ++d)
        st_ll(P.ag[d] + (epoch & 1u) * AG_PAR + ag_idx(rtile, t, m), mo);
    }
  }
  // T3: pull the reduced rows
  {
    const int npair = NRT * M, mine = b < npair ? (npair - b + G - 1) / G : 0;
    const uint4* ag_me = P.ag[P.my_rank] + (epoch & 1u) * AG_PAR;
    for (int x = tid; x < mine * NMSG; x += NCONS) {
      const int pi = x / NMSG, m = x - pi * NMSG, j = b + pi * G, rtile = j / M, t = j - rtile * M;
      uint4 v;
      do {
        v = ld_ll(ag_me + ag_idx(rtile, t, m));
      } while (v.w != epoch);
      uint32_t* o = reinterpret_cast<uint32_t*>(P.out + (size_t)t * DIM + rtile * 256 + 6 * m);
      o[0] = v.x;
      o[1] = v.y;
      if (m < NMSG - 1) o[2] = v.z;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// kernel
__global__ void __launch_bounds__(NTHREADS, 1) tp_moe_kernel(const __grid_constant__ Params P) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
  const Tabs T = tabs(smem);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + OFF_TAB + TAB_BAR);
  uint64_t* empty = full + NST;
  uint64_t* epi = full + 2 * NST;
  uint64_t* gbar = epi + 5;  // (epi + 4: unused slot)
  uint64_t* fullo = gbar + 1;
  uint64_t* freeo = fullo + 2;
  uint64_t* sdone = freeo + 2;
  uint64_t* xsb = sdone + 1;
  int* s_unit = reinterpret_cast<int*>(smem + OFF_TAB + TAB_UNIT);
  const int tid = threadIdx.x, warp = tid >> 5;
  const bool consumer = tid < NCONS;
  const bool small_tab = P.M * P.topk <= TP_MOE_SMALL_MAXMT && P.M <= NMAX;
  // fused quantization threads: small tables -- the consumer warps that neither build the tables
  // nor gate; block-wide tables and the unsplit fused gate -- the producer warpgroup
  const bool gsplit =
      P.glog &&
      small_tab;  // fused gate, small tables: gate on warps 0..7, quantization on warps 8..15
  const bool early_xs =
      P.dense && small_tab && !(P.glog && !gsplit);  // (the producer warpgroup is idle)
  const bool fused_compact = !P.dense;               // compact panels (gathered x)
  const bool early_xq =
      fused_compact;  // compact panels: x quantized (flags acquired) before the tables
  unsigned epoch_e = 0u;
  bool epoch_ok = false;
  griddep_wait();  // (no-op when not PDL-launched: x and the routing come from the previous
                   // kernels)
  // the launch epoch tags this launch's per-token flags: read by every thread before the first
  // __syncthreads (work[0] moves only when the last CTA passes its final barrier, so it cannot
  // change under any thread of this launch)
  const unsigned xtag = __ldcg(P.work) + 1u;
  if (consumer) {
    epoch_e = xtag - 1u;
    epoch_ok = true;
  }
  if (tid < NST / KPS)
    mbar_init(&full[tid], 1u);  // the producer lane's expect_tx arrive (+ tx bytes)
  else if (tid >= 32 && tid < 32 + NST / KPS)
    mbar_init(&empty[tid - 32], (unsigned)(NCONS / 32));
  else if (tid >= 64 && tid < 68)
    mbar_init(&epi[tid - 64], (unsigned)(NCONS / 32));
  else if (tid == 69)
    mbar_init(gbar, (unsigned)(NCONS / 32));
  else if (tid == 70 || tid == 71)
    mbar_init(&fullo[tid - 70], (unsigned)(NCONS / 32));
  else if (tid == 72 || tid == 73)
    mbar_init(&freeo[tid - 72], 1u);
  else if (tid == 74)
    mbar_init(sdone, 2u);
  else if (tid == 75)
    mbar_init(xsb, 1u);
  if (tid == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  if (consumer) {
    if (gsplit && tid < GS_N)
      route_prologue(P, smem, T, true, GS_N, 2);  // (GS_N = BT_DENSE threads: named barrier 2 keeps
                                                  // one count; warps 12..15 quantize meanwhile)
    else if (P.glog && !gsplit)
      route_prologue(P, smem, T, small_tab, NCONS, 1);  // (the producer warpgroup quantizes)
    else if (small_tab && tid >= BT_DENSE) {  // fused input quantization of tokens t = b, b + G, ..
                                              // (small: warps 12..15)
      float* red = reinterpret_cast<float*>(smem + PRO_WS) + WS_RED;
      for (int t = blockIdx.x; t < P.M; t += gridDim.x)
        quant_token<NQ_SMALL>(P, t, tid - BT_DENSE, red, xtag);
    }
  } else if (early_xs &&
             warp == GATHER_WARP + 2) {  // dense: token flags acquired + owners' scales staged
    const int l = tid & 31;              // while the tables are built (xsb arrive after the sync)
    float v = 0.f;
    if (l < P.M) {
      while (ld_relaxed_gpu(P.xflag + l) != xtag) __nanosleep(32);
      (void)ld_acquire_gpu(P.xflag + l);
      v = __ldcg(P.xs + l);
    }
    if (l < NMAX) reinterpret_cast<float*>(smem + OFF_TAB + TAB_XS)[l] = v;
  }
  if (small_tab && P.dense) {
    if (tid < BT_DENSE) build_tables_dense(P, smem, T);
  } else if (small_tab) {
    if (warp == 0) build_tables_small(P, smem, T);
  } else if (consumer)
    build_tables(
        P, smem,
        T);  // consumer warps (named barrier 1); the producer warpgroup quantizes meanwhile
  if ((!small_tab || (P.glog && !gsplit)) && tid >= NCONS &&
      !(early_xs && warp == GATHER_WARP + 2)) {
    // block-wide tables / fused gate: the producer warpgroup's quantization, after the tables (one
    // call site: warps 16..18 arrive at once)
    float* red = reinterpret_cast<float*>(smem + OFF_TAB + TAB_QQ);  // (scratch, 64 words)
    for (int t = blockIdx.x; t < P.M; t += gridDim.x)
      quant_token<NTHREADS - NCONS>(P, t, tid - NCONS, red, xtag);
  }
  if (early_xq && warp == PROD_WARP)
    wait_tokens(P, tid & 31, 32,
                xtag);       // every token's flag: the consumers' gather follows the sync
                             // (one warp polls + fences, the __syncthreads publishes)
  fence_proxy_async_smem();  // prologue scratch (generic writes) precedes the ring's async-proxy
                             // writes
  __syncthreads();
  if (warp >= PROD_WARP) {
    setmaxnreg_dec<TP_MOE_PROD_REGS>();
    if (warp == PROD_WARP) {
      if ((tid & 31) == 0) producer(P, smem, full, empty, T, s_unit, gbar, early_xs, xtag);
      return;
    }  // waits after pass A
    griddep_wait();
    const unsigned epoch = __ldcg(P.work) + 1u, par = epoch & 1u;
    if (blockIdx.x == 0) {  // next launch's parity set (unused by this launch; the previous launch
                            // has completed): warps 17..19
      const int t2 = tid - NCONS - 32;
      if (t2 < NRT) P.rt_cnt[(par ^ 1u) * NRT + t2] = 0u;
      if (t2 == NRT) P.work[1 + (par ^ 1u)] = 0u;
      for (int i = t2; i < MAXP; i += NTHREADS - NCONS - 32) {
        P.pan_cnt[(par ^ 1u) * MAXP + i] = 0u;
        P.xg_cnt[(par ^ 1u) * MAXP + i] = 0u;
      }
      if (t2 == NRT + 2) {
        P.fc2_next[par ^ 1u] = 0u;
        P.fc2_next[2 + (par ^ 1u)] = 0u;
      }
      if (t2 == NRT + 3 && P.gcnt) P.gcnt[par ^ 1u] = 0u;
    }
    if (warp == GATHER_WARP + 2 && P.dense) {  // the owners' token scales: acquired + staged here,
                                               // not polled by consumer warps
      const int l = tid & 31;
      if (!early_xs) {  // (else done in the prologue)
        float v = 0.f;
        if (l < P.M) {
          while (ld_relaxed_gpu(P.xflag + l) != xtag) {
          }
          (void)ld_acquire_gpu(P.xflag + l);
          v = __ldcg(P.xs + l);
        }
        if (l < NMAX) reinterpret_cast<float*>(smem + OFF_TAB + TAB_XS)[l] = v;
      }
      __syncwarp();
      if (l == 0) mbar_arrive(xsb);
    }
    if (warp == GATHER_WARP) {  // FC1 stream-K partials: at most one per group (a range that starts
      if ((tid & 31) == 0) {    // inside an item): wait for the consumers' writes, fence, flag
        unsigned gph = 0u;
        for (int g = 0; g < 2; ++g) {
          const int lo = fc1_lo(T, g, blockIdx.x), hi = fc1_lo(T, g, blockIdx.x + 1);
          if (lo < hi && lo % NKT1 != 0) {
            mbar_wait(gbar, gph);
            gph ^= 1u;
            fence_acq_rel_gpu();
            st_release_gpu(P.part_flags + g * gridDim.x + blockIdx.x, epoch);
          }
        }
      }
      __syncwarp();
    }
    if (warp == GATHER_WARP || warp == GATHER_WARP + 2)
      store_helper(P, smem, T, fullo, freeo, sdone, warp == GATHER_WARP ? 0 : 1, par);
    else if (warp == GATHER_WARP + 1)
      release_helper(P, epi, T, epoch, par);
    return;
  }
  setmaxnreg_inc<TP_MOE_CONS_REGS>();
  griddep_wait();
  const unsigned epoch = (epoch_ok ? epoch_e : __ldcg(P.work)) + 1u, par = epoch & 1u;
  const Lane L = lane_of();
  const int G = gridDim.x, b = blockIdx.x;
  const int P_tot = T.misc[2];
  if (!P.dense) {  // this CTA's share of every panel's x rows into xg (per panel [kt][Npad][128],
                   // 128B-swizzled)
    const unsigned C = (unsigned)T.misc[1] * (unsigned)CPE, j0 = (C * (unsigned)b) / (unsigned)G,
                   j1 = (C * (unsigned)(b + 1)) / (unsigned)G;  // (C G < 2^32: 32-bit)
    if (j1 - j0 <= 2u * NCONS) {
      for (unsigned jb = j0 + tid; jb < j1;
           jb += 2 * NCONS) {  // (small share: the same body 2-wide -- less cold code)
        uint4 v[2];
        int off[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const unsigned j = jb + u * NCONS;
          off[u] = -1;
          if (j < j1) {
            const int e = (int)(j / (unsigned)CPE),
                      kt = (int)((j - (unsigned)e * (unsigned)CPE) >> 3), c = (int)(j & 7);
            const int pid = T.e2p[e];
            const uint32_t pv = T.pinf[pid];
            const int r = e - pinf_b(pv), np = (pinf_n(pv) + 7) & ~7;
            const uint4* xp =
                reinterpret_cast<const uint4*>(P.xq + (size_t)T.ent[e] * DIM + kt * 128 + c * 16);
            v[u] = __ldcg(xp);  // (one load form -- the i-cache is cold here)
            off[u] = (int)(T.prow[pid] * NKT1 * 128u) + (kt * np + r) * 128 + ((c ^ (r & 7)) << 4);
          }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u)
          if (off[u] >= 0) *reinterpret_cast<uint4*>(P.xg + off[u]) = v[u];
      }
    } else {
      for (unsigned jb = j0 + tid; jb < j1;
           jb += 8 * NCONS) {  // 8 independent loads in flight per thread, then the stores
        uint4 v[8];
        int off[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const unsigned j = jb + u * NCONS;
          off[u] = -1;
          if (j < j1) {
            const int e = (int)(j / (unsigned)CPE),
                      kt = (int)((j - (unsigned)e * (unsigned)CPE) >> 3), c = (int)(j & 7);
            const int pid = T.e2p[e];
            const uint32_t pv = T.pinf[pid];
            const int r = e - pinf_b(pv), np = (pinf_n(pv) + 7) & ~7;
            const uint4* xp =
                reinterpret_cast<const uint4*>(P.xq + (size_t)T.ent[e] * DIM + kt * 128 + c * 16);
            v[u] = __ldcg(xp);  // (one load form -- the i-cache is cold here)
            off[u] = (int)(T.prow[pid] * NKT1 * 128u) + (kt * np + r) * 128 + ((c ^ (r & 7)) << 4);
          }
        }
#pragma unroll
        for (int u = 0; u < 8; ++u)
          if (off[u] >= 0) *reinterpret_cast<uint4*>(P.xg + off[u]) = v[u];
      }
    }
    asm volatile("fence.proxy.async.global;" ::
                     : "memory");  // read back by bulk copies (async proxy)
    named_bar_sync(1, NCONS);
    if (tid == 0 && j0 < j1) {
      fence_acq_rel_gpu();
      const int p0 = T.e2p[j0 / (unsigned)CPE], p1 = T.e2p[(j1 - 1) / (unsigned)CPE];
      for (int pid = p0; pid <= p1; ++pid) atomicAdd(P.xg_cnt + par * MAXP + pid, 1u);
    }
  }
  int s = 0;
  unsigned ph = 0;
  {
    int j = 0;
    for (int g = 0; g < 2; ++g) {
      const int lo = fc1_lo(T, g, b), hi = fc1_lo(T, g, b + 1);
      for (int gk = lo; gk < hi;) {
        const int it = gk / NKT1, kt0 = gk - it * NKT1, kt1 = min(NKT1, hi - it * NKT1);
        const Item I = item_of(T, it / NQG, it % NQG);
        uint64_t* eb = &epi[j & 3];
        if (kt0 == 0) ++j;
        if (I.ebase >= T.misc[1]) {  // fused shared expert: one N32 instantiation (code size: the
                                     // routed loops'
          fc1_piece_sh<32>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch,
                           eb);  // i-cache locality)
          gk = (it + 1) * NKT1;
          continue;
        }
        switch (P.dense ? P.md >> 3 : (I.n + 7) >> 3) {
          case 1:
            fc1_piece<8>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb);
            break;
          case 2:
            fc1_piece<16>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb);
            break;
          case 3:
            fc1_piece<24>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb);
            break;
          default:
            fc1_piece<32>(P, smem, full, empty, T, L, s, ph, I, g, it, kt0, kt1, epoch, eb);
            break;
        }
        gk = (it + 1) * NKT1;
      }
    }
  }
  named_bar_sync(
      1, NCONS);  // the last FC1 epilogue's act-tile reads (staging +16384..) finish before the
                  // first two FC2 out tiles (no freeo wait) overwrite the shared staging area
  {
    int j = 0;
    int jo = 0;                  // out tiles handed to the store warps
    const int E_sh = T.misc[1];  // first shared-expert entry (a unit is the shared expert's iff its
                                 // entry base >= E_sh)
    for (;; ++j) {               // units from the producer's queue grabs
      mbar_wait(&full[s / KPS], ph);
      const int2 uv = make_int2(reinterpret_cast<volatile int*>(s_unit)[2 * (j & 7)],
                                reinterpret_cast<volatile int*>(s_unit)[2 * (j & 7) + 1]);
      if (uv.x < 0) break;
      const int rtile = uv.x & ((1 << SU_NS) - 1);
      Item I;
      I.n = (uv.x >> SU_NS) & 63;
      I.e = (uv.x >> SU_ES) & ((1 << PI_NS) - 1);
      I.pid = (uv.x >> SU_PS) & 1023;
      I.ebase = uv.y;
      I.q = 0;
      I.u = 0;
      if (I.ebase >=
          E_sh) {  // fused shared expert: one N32 instantiation (stale B rows -> unused columns)
        fc2_unit_sh<32>(P, smem, full, empty, T, L, s, ph, I, rtile, jo++, fullo, freeo);
        continue;
      }
      switch ((I.n + 7) >> 3) {
        case 1:
          fc2_unit<8>(P, smem, full, empty, T, L, s, ph, I, rtile, jo, fullo, freeo);
          break;
        case 2:
          fc2_unit<16>(P, smem, full, empty, T, L, s, ph, I, rtile, jo, fullo, freeo);
          break;
        case 3:
          fc2_unit<24>(P, smem, full, empty, T, L, s, ph, I, rtile, jo, fullo, freeo);
          break;
        default:
          fc2_unit<32>(P, smem, full, empty, T, L, s, ph, I, rtile, jo, fullo, freeo);
          break;
      }
      ++jo;
    }
    for (int je = jo; je < jo + 2;
         ++je) {              // end marker for the store warps (in the next out tile of
      const int bo = je & 1;  // each buffer), then wait until they drained everything
      if (je >= 2) mbar_wait(&freeo[bo], (unsigned)((je >> 1) - 1) & 1u);
      if (tid == 0)
        reinterpret_cast<volatile int*>(smem + OFF_TAB + TAB_UNIT + (16 + NRT) * 4)[bo * 4 + 2] =
            -1;
      __syncwarp();
      if (L.lane == 0) mbar_arrive(&fullo[bo]);
    }
    mbar_wait(sdone, 0u);
  }
  tail(P, T, tid, epoch, par, smem);
  named_bar_sync(1, NCONS);
  if (tid == 0) {
    fence_acq_rel_gpu();
    if (atomicAdd(P.work + 1 + par, 1u) == (unsigned)G - 1u)
      atomicAdd(P.work, 1u);  // last CTA out: next epoch
  }
}

}  // namespace dsv41
}  // namespace sm90
}  // namespace tp_moe
}  // namespace flashinfer
