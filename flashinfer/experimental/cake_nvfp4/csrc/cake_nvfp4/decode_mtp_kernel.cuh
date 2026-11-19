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
#pragma once
// Paged-KV decode attention for Blackwell (sm_100a): FP8 E4M3 query/output, NVFP4 K/V cache
// (E2M1 values, one E4M3 scale per 16 values), 16 query heads per KV head, head dim 128,
// 128-token pages (HND layout). G query tokens (MTP) of one request share every KV page load:
// the 16 heads x G tokens form the N = 16G columns of the swapped-AB GEMMs. Each token keeps its
// own causal length, max, sum and output.
//
// This is the "mtp" kernel: G = 2, 4 or 8.
//
// Persistent, warp-specialized kernel (one CTA per SM, 896 threads; G = 2 has two softmax
// warpgroups, 640 threads, and the later warps move down by 8):
//   warps 0-15  softmax, four warpgroups: S from TMEM -> reference max / sum, P = exp2(...) as E4M3
//               into smem. Warpgroup g takes the S blocks b = g (mod 4) of each step; warpgroups
//               sharing a token vote / reduce together, the token's owner keeps its O columns
//   warp  16    QK MMA issuer (tcgen05.mma kind::f8f6f4, A operand from TMEM)
//   warp  17    V page loader (TMA)
//   warp  18    computes this CTA's KV segment list, then issues PV MMA
//   warp  19    Q + K page loader (TMA)
//   warps 20-27 two dequant warpgroups: FP4 x E4M3 scale -> E4M3 into TMEM (K: token rows,
//               V: transposed, head-dim rows)
// Swapped-AB GEMMs: S^T[128 tok x 16G] = K[tok x d] Q^T, O^T[128 d x 16G] += V^T P^T.
// Each CTA pair (one CTA per KV head) processes an equal page range of the concatenated virtual
// requests; a unit split into pieces across CTAs is merged by a small combine kernel.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "decode_params.h"
#include "fp4_mul.cuh"

namespace flashinfer {
namespace cake_nvfp4 {
namespace decode_mtp {

namespace {

constexpr int kKStageBytes = 9216;   // 8 KB packed K + 1 KB block scales
constexpr int kVStageBytes = 17408;  // 16 KB unpacked V + 1 KB block scales
constexpr int kKvTxBytes = 9216;     // global bytes per (page, head): 8 KB FP4 + 1 KB scales
constexpr int kKScaleOffset = 8192;  // block scales follow the values in a stage
constexpr int kVScaleOffset = 16384;
constexpr int kMaxSmem = 232448;  // dynamic shared memory of a CTA on sm_100: 227 KiB
constexpr int kRedBufs = 3;       // rotating column-max reduction buffers
constexpr int kNumBars = 72;
constexpr int kCachedSegs = 5;

// Per-G configuration. SP = pages per softmax step; TMEM: S stages | O | K stages | V stages.
template <int G>
struct Cfg {
  static constexpr int N = 16 * G;
  static_assert(G == 2 || G == 4 || G == 8, "fused query tokens");
  // Softmax warpgroups.
  static constexpr int NG = G == 2 ? 2 : 4;
  static constexpr int Threads = 32 * (4 * NG + 4 + 8);
  static constexpr int WarpQk = 4 * NG;  // then V loader, segments + PV, Q + K loader, 8 dequant
  // Registers per thread: the launch share, then softmax / dequant up and the four 1-warp roles
  // down.
  static constexpr int RegsLaunch = 65536 / Threads / 8 * 8;
  static constexpr int RegsSoftmax = G == 8 ? 88 : (G == 4 ? 80 : 112);
  static constexpr int RegsDequant = G == 8 ? 56 : (G == 4 ? 72 : 104);
  static constexpr int RegsMisc = G == 8 ? 32 : (G == 4 ? 40 : 48);
  static_assert(4 * NG * RegsSoftmax + 8 * RegsDequant + 4 * RegsMisc <= RegsLaunch * Threads / 32,
                "register split");
  static constexpr int SP = G == 2 ? 4 : (G == 4 ? 2 : 1);
  // S (TMEM) / P (smem) stages: QK may run Stages - 1 steps ahead of the softmax.
  static constexpr int Stages = 2;
  // TMEM stages of dequantized K and V pages (32 columns each).
  static constexpr int KT = G == 8 ? 3 : 4;
  static constexpr int VT = G == 2 ? 3 : (G == 4 ? 2 : 1);
  // Shared memory stages of NVFP4 K and V pages.
  static constexpr int KS = G == 2 ? 6 : (G == 4 ? 5 : 4);
  static constexpr int VS = G == 2 ? 7 : 6;
  // P (smem) stages may exceed the S (TMEM) stages: QK then waits for S[s] directly and P lags one
  // more step.
  static constexpr int PStages = G == 4 ? 3 : Stages;
  static constexpr int SCols = SP * N;
  static constexpr uint32_t TmemS = 0;
  static constexpr uint32_t TmemO = Stages * SCols;
  static constexpr uint32_t TmemK = TmemO + N;
  static constexpr uint32_t TmemV = TmemK + KT * 32;
  static_assert(TmemV + VT * 32 <= 512, "TMEM columns");
  static constexpr int QStageBytes = 2048 * G;
  static constexpr int PStageBytes = 2048 * SP * G;
  static constexpr int SmemQ = 0;
  static constexpr int SmemK = SmemQ + 2 * QStageBytes;
  static constexpr int SmemV = SmemK + KS * kKStageBytes;
  static constexpr int SmemP = SmemV + VS * kVStageBytes;
  static constexpr int SmemMaxRed = SmemP + PStages * PStageBytes;  // 3 x N x u32 (rotating)
  static constexpr int SmemSumRed = SmemMaxRed + kRedBufs * N * 4;  // 2 x 4NG warps x N f32
  static constexpr int SmemMisc =
      SmemSumRed + 2 * 4 * NG * N * 4;  // tmem base, numSegs, rescale flags
  static constexpr int SmemBars = SmemMisc + 128;
  static constexpr int SmemSegCache = SmemBars + kNumBars * 8;
  static constexpr int SmemBytes = SmemSegCache + kCachedSegs * 16;
  static_assert(SmemBytes <= kMaxSmem, "smem layout");
  static_assert(SmemP % 1024 == 0, "P tiles need 1024-byte alignment");
  // Barrier indices.
  static constexpr int BarQFull = 0;                 // 2: TMA tx
  static constexpr int BarQEmpty = 2;                // 2: MMA commit
  static constexpr int BarKFull = 4;                 // KS: K TMA tx
  static constexpr int BarKEmpty = BarKFull + KS;    // KS: four K dequant warps
  static constexpr int BarTkFull = BarKEmpty + KS;   // KT: 128 K dequant threads
  static constexpr int BarTkEmpty = BarTkFull + KT;  // KT: MMA commit
  static constexpr int BarVFull = BarTkEmpty + KT;   // VS: V TMA tx
  static constexpr int BarVEmpty = BarVFull + VS;    // VS: four V dequant warps
  static constexpr int BarTvFull = BarVEmpty + VS;   // VT: 128 V dequant threads
  static constexpr int BarTvEmpty = BarTvFull + VT;  // VT: MMA commit
  static constexpr int BarSFull = BarTvEmpty + VT;   // Stages: MMA commit
  static constexpr int BarSEmpty =
      BarSFull + Stages;  // Stages: all softmax threads, once S is in registers
  static constexpr int BarOFull = BarSEmpty + Stages;  // 1: MMA commit at segment end
  static constexpr int BarOFree = BarOFull + 1;        // 1: O-owning threads read the final O
  static constexpr int BarODone = BarOFree + 1;  // PStages ring: PV commit after every step's MMAs
  static constexpr int BarPFull =
      BarODone + PStages;  // PStages: all softmax threads, P[s] (and O rescale) done
  static constexpr int BarPCons =
      BarPFull + PStages;  // PStages (!EarlyS): PV commit, the MMAs reading P[s] are complete
  static_assert(BarPCons + PStages <= kNumBars, "barrier storage");
  // tcgen05 instruction descriptor: D F32, A/B E4M3, K-major, M=128, N=16G.
  static constexpr uint32_t Idesc = (1u << 4) | ((uint32_t(N) >> 3) << 17) | ((128u >> 4) << 24);
  // Softmax work split: block b = (page b / G, token b % G) of a step goes to warpgroup b % NG.
  // Shared mode (G divides NG): every block of warpgroup g has token g % G, NG / G warpgroups share
  // a token and warpgroup t owns token t. Otherwise (SP = 1) warpgroup g owns tokens t = g (mod
  // NG).
  static constexpr bool Shared = NG % G == 0;
  static constexpr int Blk = SP * G;
  static constexpr int BlkPerGrp = (Blk + NG - 1) / NG;
  static constexpr int TokPerGrp = Shared ? 1 : (G + NG - 1) / NG;
  static constexpr int ShareGrps = Shared ? NG / G : 1;
  static constexpr int NumOwners = G < NG ? G : NG;
  static_assert(Shared || SP == 1, "softmax split");
  // Headroom (base 2) of P over the reference max: reference updates (and O rescales, which wait
  // for the previous step's PV MMAs) stay rare. With 8 bits P <= 448 * 2^-8 right after an update.
  static constexpr float Headroom = 8.f;
  // Release each S stage as soon as the softmax holds it in registers (P readiness then has its own
  // barrier); otherwise S is released together with P, as in the original pipeline.
  static constexpr bool EarlyS = G == 2;
  // Speculative single-pass softmax: convert against the current reference and only reload S (TMEM)
  // and reduce the exact max when some exponent exceeds log2(448). Needs S in TMEM until P is
  // written.
  static constexpr bool Spec = G == 8;
  static_assert(!(Spec && EarlyS), "speculative softmax reloads S");
  // Dequant in halves (smaller register footprint) where the dequant budget is tight.
  static constexpr bool SplitDequant = RegsDequant < 64;
};

constexpr uint32_t kLane16 = 16u << 16;

// Named barriers (+ barrier group: the warpgroups sharing a token).
constexpr int kNbVote = 1;
constexpr int kNbSoftmax = 5;
constexpr int kNbCorr = 9;
constexpr int kNbTmemDone = 13;
constexpr int kNbPConsumed = 14;

constexpr float kLog2E4m3Max = 8.8073549f;  // log2(448): P is scaled into the E4M3 range
// P = exp2((S - ref) * scale + log2(448) - Cfg::Headroom): scores may exceed the reference max by
// Headroom (base 2) before P would saturate, so the reference (and O) is only updated then.
constexpr float kNegMax = -3.4028235e38f;

struct alignas(16) KvSeg {
  int32_t virt;  // virtual request: request * numGroups + group
  int32_t head;
  int32_t pageStart;
  int32_t numPages;
};

// ------------------------------------------------------------------------------------------------
// PTX helpers.

__device__ __forceinline__ uint32_t smemAddr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count)
               : "memory");
}

// Blocking wait; the suspend-time hint (10 ms, in ns) lets a waiting warp sleep instead of
// spinning.
__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@!p bra WAIT_%=;\n}" ::"r"(smemAddr(bar)),
      "r"(parity), "r"(0x989680u)
      : "memory");
}

__device__ __forceinline__ void mbarArrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smemAddr(bar))
               : "memory");
}

__device__ __forceinline__ void mbarArriveExpectTx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tmaLoad4d(void* dst, CUtensorMap const* map, uint64_t* bar, int c0,
                                          int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(smemAddr(dst)),
      "l"(map), "r"(smemAddr(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}

// One lane of a converged warp; lets ptxas issue warp-level async ops (TMA, UMMA) without
// serializing over the active lanes.
__device__ __forceinline__ bool electOne() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "elect.sync _|P, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, P;\n}"
      : "=r"(pred));
  return pred != 0;
}

__device__ __forceinline__ void namedBarSync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void namedBarArrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// Named barrier that also ORs a predicate over the participating threads.
__device__ __forceinline__ bool namedBarSyncOr(int id, int count, bool pred) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p, q;\n\t"
      "setp.ne.u32 p, %1, 0;\n\t"
      "barrier.cta.red.or.pred q, %2, %3, p;\n\t"
      "selp.u32 %0, 1, 0, q;\n}"
      : "=r"(r)
      : "r"(static_cast<uint32_t>(pred)), "r"(id), "r"(count)
      : "memory");
  return r != 0;
}

template <int N>
__device__ __forceinline__ void setMaxNRegInc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}

template <int N>
__device__ __forceinline__ void setMaxNRegDec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}

__device__ __forceinline__ void tcFenceBefore() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceAfter() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcWaitLd() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcWaitSt() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcCommit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                   smemAddr(bar))
               : "memory");
}

__device__ __forceinline__ void tcMma(uint32_t dTmem, uint32_t aTmem, uint64_t bDesc,
                                      uint32_t idesc, bool accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;\n}" ::"r"(dTmem),
      "r"(aTmem), "l"(bDesc), "r"(idesc), "r"(static_cast<uint32_t>(accumulate))
      : "memory");
}

// 16 lanes x 256 bits, two repetitions (8 regs/thread).
__device__ __forceinline__ void tcLd16x256x2(uint32_t (&r)[8], uint32_t taddr) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
                 "=r"(r[7])
               : "r"(taddr));
}

__device__ __forceinline__ void tcSt16x256x2(uint32_t taddr, uint32_t const (&r)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
      : "memory");
}

__device__ __forceinline__ void tcLd32x32x8(uint32_t (&r)[8], uint32_t taddr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
                 "=r"(r[7])
               : "r"(taddr));
}

__device__ __forceinline__ void tcSt32x32x8(uint32_t taddr, uint32_t const (&r)[8]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
      : "memory");
}

__device__ __forceinline__ void tcSt32x32x32(uint32_t taddr, uint32_t const (&r)[32]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, "
      "%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "
      "%30, "
      "%31, %32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
      "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
      "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]),
      "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]),
      "r"(r[30]), "r"(r[31])
      : "memory");
}

__device__ __forceinline__ void tcSt32x32x16(uint32_t taddr, uint32_t const (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, "
      "%12, %13, %14, %15, %16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
      "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}

__device__ __forceinline__ void tcSt16x128x8(uint32_t taddr, uint32_t const (&r)[16]) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, "
      "%12, %13, %14, %15, %16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
      "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}

// Shared memory matrix descriptor: K-major, 128B swizzle, 8-row atoms 1024 B apart, LBO 2048 B.
__device__ __forceinline__ uint64_t smemDescSw128(void const* p) {
  uint32_t const lo = ((smemAddr(p) & 0x3FFFFu) >> 4) | (uint32_t{2048 >> 4} << 16);
  uint32_t const hi = (1024u >> 4) | (1u << 14) | (2u << 29);
  return (uint64_t{hi} << 32) | lo;
}

// FP4 x E4M3 scale -> E4M3: one instruction from the CUDA 13.4 toolkit on, see fp4_mul.cuh.
using fp4::Fp4Scale;
using fp4::fp4Scale;
using fp4::fp4ScaleUniform;
using fp4::mulFp4x4Unpacked;
using fp4::mulFp4x8Packed;

// Transposed 16x16 FP4 tile load: 4-bit values expanded to bytes.
__device__ __forceinline__ void ldsmFp4Trans16x16(uint32_t& r0, uint32_t& r1, uint32_t saddr) {
  asm volatile("ldmatrix.sync.aligned.shared::cta.m16n16.x1.trans.b8x16.b4x16_p64 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(saddr));
}

__device__ __forceinline__ void stsmTrans8b(uint32_t saddr, uint32_t const (&r)[4]) {
  asm volatile(
      "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};" ::"r"(saddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
      : "memory");
}

__device__ __forceinline__ uint32_t f32x4ToE4m3x4(float a, float b, float c, float d) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return uint32_t{lo} | (uint32_t{hi} << 16);
}

// Packed f32x2 arithmetic (one instruction for two lanes of work).
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rc, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\tmov.b64 rc, {%6, %7};\n\t"
      "fma.rn.f32x2 rd, ra, rb, rc;\n\tmov.b64 {%0, %1}, rd;\n}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}

__device__ __forceinline__ float2 fadd2(float2 a, float2 b) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\tmov.b64 rb, {%4, %5};\n\t"
      "add.rn.f32x2 rd, ra, rb;\n\tmov.b64 {%0, %1}, rd;\n}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}

// Order-preserving float <-> uint32 encoding so that column maxima can use atomicMax.
__device__ __forceinline__ uint32_t encodeMax(float f) {
  uint32_t const u = __float_as_uint(f);
  return u ^ (static_cast<uint32_t>(-static_cast<int32_t>(u >> 31)) | 0x80000000u);
}

__device__ __forceinline__ float decodeMax(uint32_t u) {
  return __uint_as_float(u ^ (((u >> 31) - 1u) | 0x80000000u));
}

// ------------------------------------------------------------------------------------------------
// KV partition.

// Visible length of the last token of a virtual request's group. Bottom-right causal alignment:
// query token q of a request with T tokens and length L attends to cache indices <= L - T + q.
__device__ __forceinline__ int groupSeqLen(DecodeParams const& p, int virt, int G) {
  int const req = virt / p.numGroups;
  int const grpIdx = virt - req * p.numGroups;
  int const tokens = p.cumSeqLensQ ? p.cumSeqLensQ[req + 1] - p.cumSeqLensQ[req] : p.queryTokens;
  if (grpIdx * G >= tokens) return 0;
  return p.seqLens[req] - tokens + (grpIdx + 1) * G;
}

// Ragged calls use G=1, so no fused tile can cross a request boundary.
__device__ __forceinline__ int queryRow(DecodeParams const& p, int virt, int G) {
  int const req = virt / p.numGroups;
  int const group = virt - req * p.numGroups;
  return p.cumSeqLensQ ? p.cumSeqLensQ[req] + group : virt * G;
}

template <int SP>
__device__ __forceinline__ int unitPages(int seqLen) {
  int const pages = (seqLen + (1 << kPageSizeLog2) - 1) >> kPageSizeLog2;
  return ((max(pages, 0) + SP - 1) / SP) * SP;
}

// One warp. The virtual requests are concatenated in SP-page granularity; CTAs 2r and 2r + 1 take
// page range r = [r * share, (r + 1) * share) of KV head 0 and 1 (the grid is even), so the two
// heads of a page (adjacent in memory) are streamed at the same time. Units may span any number of
// ranges: the piece of unit v in range r goes to workspace slot (v + r) * 2 + head, which is unique
// because v + r strictly increases along the concatenation. The first and last range of every
// unit are published for the combine kernel.
template <int G, int SP>
__device__ void computeKvSegments(DecodeParams const& p, KvSeg* segs, KvSeg* cache,
                                  int32_t* numSegsOut) {
  int const lane = threadIdx.x % 32;
  int total = 0;
  for (int v = lane; v < p.numVirt; v += 32) total += unitPages<SP>(groupSeqLen(p, v, G));
#pragma unroll
  for (int off = 16; off > 0; off /= 2) total += __shfl_xor_sync(0xffffffffu, total, off);
  int const numRanges = static_cast<int>(gridDim.x) / 2;
  int const rangeIdx = static_cast<int>(blockIdx.x) / 2;
  int const myHead = static_cast<int>(blockIdx.x) % 2;
  int share = (total + numRanges - 1) / numRanges;
  share = ((share + SP - 1) / SP) * SP;
  int const lo = rangeIdx * share;
  int const hi = min(lo + share, total);
  int* ranges = reinterpret_cast<int*>(p.ws + p.wsRangeOffset);
  int base = 0;
  int count = 0;
  for (int v0 = 0; v0 < p.numVirt && base < hi; v0 += 32) {
    int const v = v0 + lane;
    int const pages = v < p.numVirt ? unitPages<SP>(groupSeqLen(p, v, G)) : 0;
    int incl = pages;
#pragma unroll
    for (int off = 1; off < 32; off *= 2) {
      int const t = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += t;
    }
    int const start = base + incl - pages;
    int const end = start + pages;
    int const s = max(start, lo);
    int const e = min(end, hi);
    bool const emit = (pages > 0) && (s < e);
    uint32_t const mask = __ballot_sync(0xffffffffu, emit);
    if (emit) {
      KvSeg seg;
      seg.virt = v;
      seg.head = myHead;
      seg.pageStart = s - start;
      seg.numPages = e - s;
      int const idx = count + __popc(mask & ((1u << lane) - 1u));
      if (idx < kCachedSegs)
        cache[idx] = seg;
      else
        segs[idx] = seg;
      if (s == start) ranges[(v * kNumHeadsKv + myHead) * 2 + 0] = rangeIdx;
      if (e == end) ranges[(v * kNumHeadsKv + myHead) * 2 + 1] = rangeIdx;
    }
    count += __popc(mask);
    base += __shfl_sync(0xffffffffu, incl, 31);
  }
  if (lane == 0) *numSegsOut = count;
}

// Loads a segment descriptor and broadcasts it from lane 0 so that every derived index is provably
// warp-uniform (keeps UMMA/TMA operands in uniform registers).
__device__ __forceinline__ KvSeg loadSegUniform(KvSeg const* p) {
  KvSeg s = *p;
  s.virt = __shfl_sync(0xffffffffu, s.virt, 0);
  s.head = __shfl_sync(0xffffffffu, s.head, 0);
  s.pageStart = __shfl_sync(0xffffffffu, s.pageStart, 0);
  s.numPages = __shfl_sync(0xffffffffu, s.numPages, 0);
  return s;
}

// Segments [0, kCachedSegs) live in shared memory, later ones in the CTA's global list.
struct KvSegList {
  KvSeg const* cache;
  KvSeg const* global;
};

__device__ __forceinline__ KvSeg loadSegUniform(KvSegList const& l, int si) {
  return loadSegUniform(si < kCachedSegs ? l.cache + si : l.global + si);
}

// ------------------------------------------------------------------------------------------------
// Producers.

template <class C, bool kIsV>
__device__ void issueKvPages(DecodeParams const& p, uint8_t* smem, uint64_t* bars,
                             uint32_t pageBase, int numPages, int32_t const* pageTable,
                             int pageStart, int pageUb, int head) {
  int const lane = threadIdx.x % 32;
  CUtensorMap const* tmData = kIsV ? &p.tmaV : &p.tmaK;
  CUtensorMap const* tmSf = kIsV ? &p.tmaVSf : &p.tmaKSf;
  constexpr int kStagesN = kIsV ? C::VS : C::KS;
  constexpr int kStageBytes = kIsV ? kVStageBytes : kKStageBytes;
  constexpr int kDataOffset = kIsV ? C::SmemV : C::SmemK;
  constexpr int kScaleOffset = kIsV ? kVScaleOffset : kKScaleOffset;
  constexpr int kFullBar = kIsV ? C::BarVFull : C::BarKFull;
  constexpr int kEmptyBar = kIsV ? C::BarVEmpty : C::BarKEmpty;
  uint32_t stage = pageBase % kStagesN;
  uint32_t phase = (pageBase / kStagesN) & 1u;
  int chunk = numPages > 0 ? pageTable[min(pageStart + lane, pageUb)] : 0;
  for (int c = 0; c < numPages; c += 32) {
    int const next = (c + 32 < numPages) ? pageTable[min(pageStart + c + 32 + lane, pageUb)] : 0;
    int const cnt = min(32, numPages - c);
    for (int i = 0; i < cnt; ++i) {
      int const pageIdx = __shfl_sync(0xffffffffu, chunk, i);
      mbarWait(&bars[kEmptyBar + stage], phase ^ 1u);
      if (electOne()) {
        mbarArriveExpectTx(&bars[kFullBar + stage], kKvTxBytes);
        uint8_t* dst = smem + kDataOffset + stage * kStageBytes;
        tmaLoad4d(dst, tmData, &bars[kFullBar + stage], 0, 0, head, pageIdx);
        tmaLoad4d(dst + kScaleOffset, tmSf, &bars[kFullBar + stage], 0, 0, head, pageIdx);
      }
      __syncwarp();
      if (++stage == kStagesN) {
        stage = 0;
        phase ^= 1u;
      }
    }
    chunk = next;
  }
}

// ------------------------------------------------------------------------------------------------
// Dequantization into TMEM.

// K tile (packed bytes, 64B swizzle: 16B chunk c of row r at chunk c ^ ((r >> 1) & 3)). The thread
// owning token row r writes its 128 E4M3 values to its TMEM lane: column c = dims 4c..4c+3.
// waitT() waits for the TMEM stage to be free; it runs only before the first TMEM store, so the
// shared loads and FP4 multiplies overlap the wait for the MMA that last read the stage.
template <bool kSplit, class W>
__device__ __forceinline__ void dequantK(uint8_t const* stage, uint32_t taddr, int warpInWg,
                                         int lane, W const& waitT) {
  int const row = warpInWg * 32 + lane;
  if constexpr (kSplit) {
    // Two halves of 64 dims: 8 packed words + 16 output words live at a time.
    uint8_t const* rowPtrS = stage + row * 64;
    int const swzS = (row >> 1) & 3;
    uint2 const sfS = *reinterpret_cast<uint2 const*>(stage + kKScaleOffset + row * 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t packedH[8];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        uint4 const v = *reinterpret_cast<uint4 const*>(rowPtrS + (((2 * h + c) ^ swzS) * 16));
        packedH[4 * c + 0] = v.x;
        packedH[4 * c + 1] = v.y;
        packedH[4 * c + 2] = v.z;
        packedH[4 * c + 3] = v.w;
      }
      uint32_t outH[16];
#pragma unroll
      for (int gg = 0; gg < 4; ++gg) {
        Fp4Scale const sfb = fp4ScaleUniform(
            __byte_perm(h == 0 ? sfS.x : sfS.y, 0, 0x1111u * static_cast<uint32_t>(gg)));
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          mulFp4x8Packed(packedH[2 * gg + w], sfb, outH[4 * gg + 2 * w + 0],
                         outH[4 * gg + 2 * w + 1]);
        }
      }
      if (h == 0) waitT();
      tcSt32x32x16(taddr + h * 16, outH);
    }
    return;
  }
  uint8_t const* rowPtr = stage + row * 64;
  int const swz = (row >> 1) & 3;
  uint32_t packed[16];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    uint4 const v = *reinterpret_cast<uint4 const*>(rowPtr + ((c ^ swz) * 16));
    packed[4 * c + 0] = v.x;
    packed[4 * c + 1] = v.y;
    packed[4 * c + 2] = v.z;
    packed[4 * c + 3] = v.w;
  }
  uint2 const sf = *reinterpret_cast<uint2 const*>(stage + kKScaleOffset + row * 8);
  uint32_t out[32];
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    Fp4Scale const sfb = fp4ScaleUniform(
        __byte_perm(g < 4 ? sf.x : sf.y, 0, 0x1111u * static_cast<uint32_t>(g % 4)));
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      mulFp4x8Packed(packed[2 * g + w], sfb, out[4 * g + 2 * w + 0], out[4 * g + 2 * w + 1]);
    }
  }
  waitT();
  tcSt32x32x32(taddr, out);
}

// V tile (4-bit values padded to bytes, 128B swizzle) with interleaved scales: the scale of token
// t, group g sits at (t / 4) * 32 + g * 4 + t % 4. Written transposed: TMEM lane = head dim, column
// c = tokens 4c..4c+3.
template <bool kSplit, class W>
__device__ __forceinline__ void dequantV(uint8_t const* stage, uint32_t taddr, int warpInWg,
                                         int lane, W const& waitT) {
  uint8_t const* sfBase = stage + kVScaleOffset + (lane % 4) * 32 + warpInWg * 8;
  if constexpr (kSplit) {
    // Lanes 0-15 then 16-31 of the transposed tile: 16 output words live at a time.
    int const swzS = (lane % 8) * 16;
    uint32_t const rowBaseS = smemAddr(stage) + lane * 128;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t dst[16];
#pragma unroll
      for (int it = 0; it < 8; ++it) {
        Fp4Scale const sfw =
            fp4Scale(*reinterpret_cast<uint32_t const*>(sfBase + it * 128 + h * 4));
        uint32_t s0, s1;
        ldsmFp4Trans16x16(s0, s1, rowBaseS + ((it * 2048 + warpInWg * 32 + h * 16) ^ swzS));
        dst[2 * it + 0] = mulFp4x4Unpacked(s0, sfw);
        dst[2 * it + 1] = mulFp4x4Unpacked(s1, sfw);
      }
      if (h == 0) waitT();
      tcSt16x128x8(taddr + h * kLane16, dst);
    }
    return;
  }
  uint2 sfV[8];
#pragma unroll
  for (int it = 0; it < 8; ++it) sfV[it] = *reinterpret_cast<uint2 const*>(sfBase + it * 128);
  int const swz = (lane % 8) * 16;
  uint32_t const rowBase = smemAddr(stage) + lane * 128;
  uint32_t dst0[16];
  uint32_t dst1[16];
#pragma unroll
  for (int it = 0; it < 8; ++it) {
    uint32_t s0, s1, s2, s3;
    ldsmFp4Trans16x16(s0, s1, rowBase + ((it * 2048 + warpInWg * 32) ^ swz));
    ldsmFp4Trans16x16(s2, s3, rowBase + ((it * 2048 + warpInWg * 32 + 16) ^ swz));
    Fp4Scale const sf0 = fp4Scale(sfV[it].x);
    Fp4Scale const sf1 = fp4Scale(sfV[it].y);
    dst0[2 * it + 0] = mulFp4x4Unpacked(s0, sf0);
    dst0[2 * it + 1] = mulFp4x4Unpacked(s1, sf0);
    dst1[2 * it + 0] = mulFp4x4Unpacked(s2, sf1);
    dst1[2 * it + 1] = mulFp4x4Unpacked(s3, sf1);
  }
  waitT();
  tcSt16x128x8(taddr, dst0);
  tcSt16x128x8(taddr + kLane16, dst1);
}

template <class C, bool kIsV>
__device__ void runDequantPipeline(uint8_t* smem, uint64_t* bars, uint32_t tmem,
                                   KvSegList const& segs, int numSegs, int warpInWg, int lane) {
  constexpr int kSmemStages = kIsV ? C::VS : C::KS;
  constexpr int kTmemStages = kIsV ? C::VT : C::KT;
  constexpr int kSmemStageBytes = kIsV ? kVStageBytes : kKStageBytes;
  constexpr int kSmemBase = kIsV ? C::SmemV : C::SmemK;
  constexpr uint32_t kTmemBase = kIsV ? C::TmemV : C::TmemK;
  constexpr int kSmemFullBar = kIsV ? C::BarVFull : C::BarKFull;
  constexpr int kSmemEmptyBar = kIsV ? C::BarVEmpty : C::BarKEmpty;
  constexpr int kTmemFullBar = kIsV ? C::BarTvFull : C::BarTkFull;
  constexpr int kTmemEmptyBar = kIsV ? C::BarTvEmpty : C::BarTkEmpty;
  uint32_t smemStage = 0;
  uint32_t tmemStage = 0;
  uint32_t smemPhase = 0;
  uint32_t tmemPhase = 0;
  for (int si = 0; si < numSegs; ++si) {
    KvSeg const seg = loadSegUniform(segs, si);
    int const n = seg.numPages;
    for (int page = 0; page < n; ++page) {
      mbarWait(&bars[kSmemFullBar + smemStage], smemPhase);
      uint8_t const* src = smem + kSmemBase + smemStage * kSmemStageBytes;
      uint32_t const taddr = tmem + kTmemBase + tmemStage * 32;
      // The TMEM stage is only needed for the stores: wait for it after the loads and multiplies.
      auto waitT = [&]() {
        mbarWait(&bars[kTmemEmptyBar + tmemStage], tmemPhase ^ 1u);
        tcFenceAfter();
      };
      if constexpr (kIsV) {
        dequantV<C::SplitDequant>(src, taddr, warpInWg, lane, waitT);
      } else {
        dequantK<C::SplitDequant>(src, taddr, warpInWg, lane, waitT);
      }
      tcWaitSt();
      tcFenceBefore();
      mbarArrive(&bars[kTmemFullBar + tmemStage]);
      __syncwarp();
      if (lane == 0) mbarArrive(&bars[kSmemEmptyBar + smemStage]);
      if (++smemStage == kSmemStages) {
        smemStage = 0;
        smemPhase ^= 1u;
      }
      if (++tmemStage == kTmemStages) {
        tmemStage = 0;
        tmemPhase ^= 1u;
      }
    }
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------

template <int G>
__device__ __forceinline__ void nvfp4PagedDecodeBody(DecodeParams const& p) {
  using C = Cfg<G>;
  constexpr int kTmemUserThreads = 32 * (4 * C::NG + 2 + 8);  // softmax, QK, PV, dequant
  extern __shared__ __align__(1024) uint8_t smem[];
  int const warp = __shfl_sync(0xffffffffu, static_cast<int>(threadIdx.x) / 32, 0);
  int const lane = threadIdx.x % 32;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + C::SmemBars);
  uint32_t* tmemSlot = reinterpret_cast<uint32_t*>(smem + C::SmemMisc);
  int32_t* numSegsSlot = reinterpret_cast<int32_t*>(smem + C::SmemMisc + 16);
  // [S stage][warpgroup]: O must be rescaled between the PV MMAs of the previous and this step.
  uint32_t volatile* rescaleFlags = reinterpret_cast<uint32_t volatile*>(smem + C::SmemMisc + 32);
  uint32_t* maxRed = reinterpret_cast<uint32_t*>(smem + C::SmemMaxRed);
  KvSeg* globalSegs = reinterpret_cast<KvSeg*>(p.ws + p.wsSegsOffset) +
                      static_cast<size_t>(blockIdx.x) * p.maxSegsPerCta;
  KvSeg* cachedSegs = reinterpret_cast<KvSeg*>(smem + C::SmemSegCache);

  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < 2; ++i) {
        mbarInit(&bars[C::BarQFull + i], 1);
        mbarInit(&bars[C::BarQEmpty + i], 1);
      }
      for (int i = 0; i < C::Stages; ++i) {
        mbarInit(&bars[C::BarSFull + i], 1);
        mbarInit(&bars[C::BarSEmpty + i], 128 * C::NG);  // all softmax threads
      }
      for (int i = 0; i < C::KS; ++i) {
        mbarInit(&bars[C::BarKFull + i], 1);
        mbarInit(&bars[C::BarKEmpty + i], 4);
      }
      for (int i = 0; i < C::VS; ++i) {
        mbarInit(&bars[C::BarVFull + i], 1);
        mbarInit(&bars[C::BarVEmpty + i], 4);
      }
      for (int i = 0; i < C::KT; ++i) {
        mbarInit(&bars[C::BarTkFull + i], 128);
        mbarInit(&bars[C::BarTkEmpty + i], 1);
      }
      for (int i = 0; i < C::VT; ++i) {
        mbarInit(&bars[C::BarTvFull + i], 128);
        mbarInit(&bars[C::BarTvEmpty + i], 1);
      }
      mbarInit(&bars[C::BarOFull], 1);
      mbarInit(&bars[C::BarOFree], 128 * C::NumOwners);
      for (int i = 0; i < C::PStages; ++i) {
        mbarInit(&bars[C::BarODone + i], 1);
        mbarInit(&bars[C::BarPFull + i], 128 * C::NG);  // all softmax threads
        mbarInit(&bars[C::BarPCons + i], 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncwarp();
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(
                     smemAddr(tmemSlot))
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    tcFenceBefore();
  } else if (warp == 1) {
    // Rotating column-max buffers start at -max; no O rescale requested yet.
    for (int i = lane; i < kRedBufs * C::N; i += 32) maxRed[i] = encodeMax(kNegMax);
    if (lane < 4 * C::PStages) rescaleFlags[lane] = 0;
  } else if (warp == C::WarpQk + 3 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmaQ) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmaK) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmaV) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmaKSf) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&p.tmaVSf) : "memory");
  }
  if (warp == C::WarpQk + 2) {
    computeKvSegments<G, C::SP>(p, globalSegs, cachedSegs, numSegsSlot);
  }
  __syncthreads();
  tcFenceAfter();
  uint32_t const tmem = __shfl_sync(0xffffffffu, *tmemSlot, 0);
  int const numSegs = __shfl_sync(0xffffffffu, *numSegsSlot, 0);
  // Segment si lives in the shared-memory cache for si < kCachedSegs, else in the global list.
  KvSegList const segs{cachedSegs, globalSegs};
  // Empty CTAs have no final segment but must also release their launch dependency.
  if (p.pdlMode != 0 && numSegs == 0 && threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  float const scaleLog2 = (p.scaleSoftmaxLog2Ptr ? *p.scaleSoftmaxLog2Ptr : p.scaleSoftmaxLog2);

  if (warp < C::WarpQk) {
    // ============================================================================== softmax
    // Step = SP pages. Its S tile holds blocks b = (page b / G, token b % G) of 128 tokens x 16
    // heads (TMEM columns 16b..16b+15); warpgroup grp (of NG) takes blocks b = NG j + grp. The
    // ShareGrps warpgroups holding a token form its barrier group (one vote / reduction per step);
    // the owner warpgroup of a token keeps its O columns (rescale, epilogue).
    //
    // Speculative normalization: P uses a per-column reference max `ref` with kHeadroom bits of
    // headroom. Each step one barrier ORs "some score exceeds ref + headroom" over the threads of
    // the token(s); only then is the exact column max reduced, ref raised, the sum rescaled and O
    // rescaled (between the PV MMAs of the previous and this step, flagged to the PV warp).
    setMaxNRegInc<C::RegsSoftmax>();
    int const grp = warp / 4;
    int const tid = threadIdx.x % 128;
    int const w = tid / 32;
    int const rowIdx = w * 32 + lane / 4;  // first TMEM lane (token) of this thread
    int const colGrp = lane % 4;           // heads 2*colGrp, +1, +8, +9
    constexpr int NG = C::NG;
    auto blkTok = [&](int j) { return (NG * j + grp) % G; };
    auto blkPage = [&](int j) { return (NG * j + grp) / G; };
    auto blkValid = [&](int j) { return NG * j + grp < C::Blk; };
    auto ltTok = [&](int lt) { return C::Shared ? grp % G : NG * lt + grp; };
    auto ltValid = [&](int lt) { return C::Shared || NG * lt + grp < G; };
    bool const ownsO = grp < C::NumOwners;
    int const bgroup = C::Shared ? grp % G : grp;
    int const nbVote = kNbVote + bgroup;
    int const nbSoftmax = kNbSoftmax + bgroup;
    int const nbCorr = kNbCorr + bgroup;
    int const nbThreads = 128 * C::ShareGrps;
    uint8_t* smemP = smem + C::SmemP;
    float* sumRed = reinterpret_cast<float*>(smem + C::SmemSumRed);  // [2 buffers][4NG warps][N]
    float const headroomRaw = C::Headroom / scaleLog2;
    float const pBias = kLog2E4m3Max - C::Headroom;
    uint32_t sIt = 0, redIt = 0, segIt = 0;
    int const mtx = lane / 8;
    int const thrRow = lane % 8;
    uint32_t const pOff = ((mtx % 2) * 8 + thrRow) * 128 + (((w * 2 + mtx / 2) ^ thrRow) * 16);
    for (int si = 0; si < numSegs; ++si) {
      KvSeg const seg = loadSegUniform(segs, si);
      int const numSteps = seg.numPages / C::SP;
      if (numSteps == 0) continue;
      int const seqLen0 = groupSeqLen(p, seg.virt, G) - (G - 1);  // visible length of token 0
      int const tokenBase = seg.pageStart << kPageSizeLog2;
      float ref[C::TokPerGrp][4];   // reference max (raw score), -max: no visible score yet
      float nsm[C::TokPerGrp][4];   // pBias - ref * scale (pBias while ref is -max: P = 0)
      float rsum[C::TokPerGrp][4];  // this thread's rows, relative to ref
#pragma unroll
      for (int lt = 0; lt < C::TokPerGrp; ++lt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          ref[lt][c] = kNegMax;
          nsm[lt][c] = pBias;
          rsum[lt][c] = 0.f;
        }
      }
      for (int step = 0; step < numSteps; ++step) {
        uint32_t const sStage = sIt % C::Stages;
        mbarWait(&bars[C::BarSFull + sStage], (sIt / C::Stages) & 1u);
        tcFenceAfter();
        int const stepTok = tokenBase + step * (C::SP << kPageSizeLog2);
        bool const needMask = stepTok + (C::SP << kPageSizeLog2) > seqLen0;
        float lmax[C::TokPerGrp][4];
        float2 stepSum[C::TokPerGrp][2];
        uint32_t pk[C::BlkPerGrp][4];  // P of each block as E4M3, in stmatrix order
#pragma unroll
        for (int lt = 0; lt < C::TokPerGrp; ++lt) {
#pragma unroll
          for (int c = 0; c < 4; ++c) lmax[lt][c] = kNegMax;
          stepSum[lt][0] = make_float2(0.f, 0.f);
          stepSum[lt][1] = make_float2(0.f, 0.f);
        }
        float2 const scale2 = make_float2(scaleLog2, scaleLog2);
        // Block j of this step's S tile (128 tokens x 16 heads, TMEM columns 16b..16b+15), held in
        // registers. Regs (i, i+1) hold heads (0, 1) of the thread's column group when (i / 4) is
        // even, else (2, 3). Tokens beyond the block's token causal length are masked.
        float qk[C::BlkPerGrp][16];
        auto loadBlocks = [&]() {
#pragma unroll
          for (int j = 0; j < C::BlkPerGrp; ++j) {
            if (!blkValid(j)) continue;
            uint32_t const taddr = tmem + C::TmemS + sStage * C::SCols + (NG * j + grp) * 16;
            tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(qk[j][0]), taddr);
            tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(qk[j][8]), taddr + kLane16);
          }
          tcWaitLd();
          if (!needMask) return;
#pragma unroll
          for (int j = 0; j < C::BlkPerGrp; ++j) {
            if (!blkValid(j)) continue;
            int const lim = seqLen0 + blkTok(j);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
              // Rows {0, 8, 16, 24} of the fragment hold regs {0,1,4,5}, {2,3,6,7}, {8,9,12,13},
              // {10,11,14,15}.
              int const tok = stepTok + (blkPage(j) << kPageSizeLog2) + rowIdx + r * 8;
              int const b = (r / 2) * 8 + (r % 2) * 2;
              if (tok >= lim) {
                qk[j][b + 0] = kNegMax;
                qk[j][b + 1] = kNegMax;
                qk[j][b + 4] = kNegMax;
                qk[j][b + 5] = kNegMax;
              }
            }
          }
        };
        // This thread's column maxima of the raw scores.
        auto trackMax = [&]() {
#pragma unroll
          for (int j = 0; j < C::BlkPerGrp; ++j) {
            if (!blkValid(j)) continue;
            int const lt = C::Shared ? 0 : j;
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
              int const c = ((i / 4) % 2) * 2;
              lmax[lt][c] = fmaxf(lmax[lt][c], fmaxf(qk[j][i], qk[j][i + 2]));
              lmax[lt][c + 1] = fmaxf(lmax[lt][c + 1], fmaxf(qk[j][i + 1], qk[j][i + 3]));
            }
          }
        };
        // E4M3 probabilities against the current reference, P = exp2(S * scale - ref * scale +
        // log2(448)
        // - headroom); returns the largest exponent.
        auto convertBlocks = [&]() {
          float xmax = kNegMax;
#pragma unroll
          for (int j = 0; j < C::BlkPerGrp; ++j) {
            if (!blkValid(j)) continue;
            int const lt = C::Shared ? 0 : j;
            float2 const nsm01 = make_float2(nsm[lt][0], nsm[lt][1]);
            float2 const nsm23 = make_float2(nsm[lt][2], nsm[lt][3]);
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
              int const hi = (i / 4) % 2;
              float2 x = ffma2(make_float2(qk[j][i], qk[j][i + 1]), scale2, hi ? nsm23 : nsm01);
              if constexpr (C::Spec) xmax = fmaxf(xmax, fmaxf(x.x, x.y));
              x.x = exp2f(x.x);
              x.y = exp2f(x.y);
              qk[j][i] = x.x;
              qk[j][i + 1] = x.y;
              stepSum[lt][hi] = fadd2(stepSum[lt][hi], x);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k)
              pk[j][k] =
                  f32x4ToE4m3x4(qk[j][4 * k], qk[j][4 * k + 1], qk[j][4 * k + 2], qk[j][4 * k + 3]);
          }
          return xmax;
        };
        loadBlocks();
        if constexpr (C::EarlyS) {
          // S is in registers: the TMEM stage goes straight back to the QK issuer, so the next QK
          // MMAs overlap this step's softmax.
          tcFenceBefore();
          mbarArrive(&bars[C::BarSEmpty + sStage]);
        }
        bool over = false;
        if constexpr (C::Spec) {
          // Speculative single pass: P against the current reference is valid unless some exponent
          // exceeds log2(448), i.e. a score exceeds ref + headroom (or a reference is still unset).
          // Only then (rare once the reference has settled) is S reloaded and the exact column max
          // reduced.
          over = convertBlocks() > kLog2E4m3Max;
#pragma unroll
          for (int lt = 0; lt < C::TokPerGrp; ++lt) {
#pragma unroll
            for (int c = 0; c < 4; ++c) over = over || (ltValid(lt) && ref[lt][c] == kNegMax);
          }
        } else {
          trackMax();
#pragma unroll
          for (int lt = 0; lt < C::TokPerGrp; ++lt) {
#pragma unroll
            for (int c = 0; c < 4; ++c) over = over || (lmax[lt][c] > ref[lt][c] + headroomRaw);
          }
        }
        float scales[C::TokPerGrp][4];
        bool const anyOver = namedBarSyncOr(nbVote, nbThreads, over);
        if (anyOver) {
          if constexpr (C::Spec) {
            loadBlocks();
            trackMax();
          }
          // Exact column max over the threads of the token's barrier group. Lanes with equal
          // lane % 4 hold the same 4 heads for 8 token rows; a 3-level butterfly halves the columns
          // per lane at each level, leaving lane l with the 8-row max of column
          // ((l >> 2) & 1) * 2 + ((l >> 3) & 1).
          uint32_t* red = maxRed + (redIt % kRedBufs) * C::N;
          bool const b1 = (lane >> 2) & 1;
          bool const b2 = (lane >> 3) & 1;
#pragma unroll
          for (int lt = 0; lt < C::TokPerGrp; ++lt) {
            if (!ltValid(lt)) continue;
            float const keep0 = b1 ? lmax[lt][2] : lmax[lt][0];
            float const keep1 = b1 ? lmax[lt][3] : lmax[lt][1];
            float const send0 = b1 ? lmax[lt][0] : lmax[lt][2];
            float const send1 = b1 ? lmax[lt][1] : lmax[lt][3];
            float const m0 = fmaxf(keep0, __shfl_xor_sync(0xffffffffu, send0, 4));
            float const m1 = fmaxf(keep1, __shfl_xor_sync(0xffffffffu, send1, 4));
            float m = fmaxf(b2 ? m1 : m0, __shfl_xor_sync(0xffffffffu, b2 ? m0 : m1, 8));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
            if (lane < 16)
              atomicMax(&red[ltTok(lt) * 16 + colGrp * 4 + (b1 ? 2 : 0) + (b2 ? 1 : 0)],
                        encodeMax(m));
          }
          namedBarSync(nbSoftmax, nbThreads);
#pragma unroll
          for (int lt = 0; lt < C::TokPerGrp; ++lt) {
            if (!ltValid(lt)) {
#pragma unroll
              for (int c = 0; c < 4; ++c) scales[lt][c] = 1.f;
              continue;
            }
            uint4 const v = *reinterpret_cast<uint4 const*>(&red[ltTok(lt) * 16 + colGrp * 4]);
            float const stepMax[4] = {decodeMax(v.x), decodeMax(v.y), decodeMax(v.z),
                                      decodeMax(v.w)};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              float const newRef = fmaxf(ref[lt][c], stepMax[c]);
              scales[lt][c] =
                  (newRef != ref[lt][c]) ? exp2f((ref[lt][c] - newRef) * scaleLog2) : 1.f;
              rsum[lt][c] *= scales[lt][c];
              ref[lt][c] = newRef;
              // A column without any visible score keeps P = 0.
              nsm[lt][c] = newRef == kNegMax ? pBias : fmaf(newRef, -scaleLog2, pBias);
            }
            stepSum[lt][0] = make_float2(0.f, 0.f);
            stepSum[lt][1] = make_float2(0.f, 0.f);
          }
          // Buffer (redIt + 2) % 3 was last read before this barrier and is next written after the
          // next one: the owner resets its columns now.
          uint32_t* nextRed = maxRed + ((redIt + 2) % kRedBufs) * C::N;
          if (ownsO && tid < 16 * C::TokPerGrp && ltValid(tid / 16)) {
            nextRed[ltTok(tid / 16) * 16 + tid % 16] = encodeMax(kNegMax);
          }
          ++redIt;
          if constexpr (C::Spec) convertBlocks();
        }
        if constexpr (!C::Spec) convertBlocks();
        // O accumulated in earlier steps needs the new reference (the first step overwrites O).
        bool const rescaleO = anyOver && ownsO && step > 0;
        if (rescaleO) {
          // Rescale O once the PV MMAs of the previous step are complete (ring of per-step PV
          // commits); the P-full release below orders it before this step's PV MMAs.
          uint32_t const prev = sIt - 1;
          mbarWait(&bars[C::BarODone + prev % C::PStages], (prev / C::PStages) & 1u);
          tcFenceAfter();
#pragma unroll
          for (int lt = 0; lt < C::TokPerGrp; ++lt) {
            if (!ltValid(lt)) continue;
            uint32_t const taddr = tmem + C::TmemO + ltTok(lt) * 16;
            uint32_t o[16];
            tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(o[0]), taddr);
            tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(o[8]), taddr + kLane16);
            tcWaitLd();
#pragma unroll
            for (int i = 0; i < 16; ++i) {
              float const sc = scales[lt][((i / 4) % 2) * 2 + (i % 2)];
              o[i] = __float_as_uint(__uint_as_float(o[i]) * sc);
            }
            tcSt16x256x2(taddr, reinterpret_cast<uint32_t const(&)[8]>(o[0]));
            tcSt16x256x2(taddr + kLane16, reinterpret_cast<uint32_t const(&)[8]>(o[8]));
          }
          tcWaitSt();
        }
        if (C::EarlyS && sIt >= C::PStages) {
          // P[sIt % PStages] was last read by the PV MMAs of step sIt - PStages.
          uint32_t const old = sIt - C::PStages;
          mbarWait(&bars[C::BarODone + old % C::PStages], (old / C::PStages) & 1u);
        }
        {
          uint8_t* pStage = smemP + (sIt % C::PStages) * C::PStageBytes;
#pragma unroll
          for (int j = 0; j < C::BlkPerGrp; ++j) {
            if (blkValid(j)) stsmTrans8b(smemAddr(pStage + (NG * j + grp) * 2048 + pOff), pk[j]);
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        // Tell the PV warp that this part of P (and any O rescale) is complete (without EarlyS this
        // also releases S). P readiness has its own PStages ring: with PStages > Stages, the S ring
        // can complete a phase twice before PV observes it.
        tcFenceBefore();
        mbarArrive(&bars[C::BarPFull + sIt % C::PStages]);
        if constexpr (!C::EarlyS) mbarArrive(&bars[C::BarSEmpty + sStage]);
        ++sIt;
#pragma unroll
        for (int lt = 0; lt < C::TokPerGrp; ++lt) {
          rsum[lt][0] += stepSum[lt][0].x;
          rsum[lt][1] += stepSum[lt][0].y;
          rsum[lt][2] += stepSum[lt][1].x;
          rsum[lt][3] += stepSum[lt][1].y;
        }
      }
      // Row sums: reduce over the lanes sharing heads, then over the warps through smem.
#pragma unroll
      for (int lt = 0; lt < C::TokPerGrp; ++lt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          rsum[lt][c] += __shfl_xor_sync(0xffffffffu, rsum[lt][c], 16);
          rsum[lt][c] += __shfl_xor_sync(0xffffffffu, rsum[lt][c], 8);
          rsum[lt][c] += __shfl_xor_sync(0xffffffffu, rsum[lt][c], 4);
        }
      }
      float* sumBuf = sumRed + (si % 2) * 4 * NG * C::N;
      if (lane < 4) {
#pragma unroll
        for (int lt = 0; lt < C::TokPerGrp; ++lt) {
          if (!ltValid(lt)) continue;
          *reinterpret_cast<float4*>(&sumBuf[(grp * 4 + w) * C::N + ltTok(lt) * 16 + lane * 4]) =
              make_float4(rsum[lt][0], rsum[lt][1], rsum[lt][2], rsum[lt][3]);
        }
      }
      if (!ownsO) {
        namedBarArrive(nbCorr, nbThreads);
        continue;
      }
      namedBarSync(nbCorr, nbThreads);
      // Final O of this segment.
      mbarWait(&bars[C::BarOFull], segIt & 1u);
      tcFenceAfter();
      ++segIt;
      uint32_t o[C::TokPerGrp][16];
#pragma unroll
      for (int lt = 0; lt < C::TokPerGrp; ++lt) {
        if (!ltValid(lt)) continue;
        uint32_t const taddr = tmem + C::TmemO + ltTok(lt) * 16;
        tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(o[lt][0]), taddr);
        tcLd16x256x2(reinterpret_cast<uint32_t(&)[8]>(o[lt][8]), taddr + kLane16);
      }
      tcWaitLd();
      tcFenceBefore();
      mbarArrive(&bars[C::BarOFree]);
      // Thread 0 owns output in every G specialization. Release once per nonempty CTA
      // at its final epilogue; combine still waits for completion before reading any data.
      if (p.pdlMode != 0 && si == numSegs - 1 && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
      // Unnormalized O^T and (ref, sum) of this piece go to the workspace.
      size_t const slot =
          static_cast<size_t>(seg.virt + static_cast<int>(blockIdx.x) / 2) * kNumHeadsKv + seg.head;
      float* wsO = p.ws + slot * (C::N * kHeadDim);
      float* wsStats = p.ws + p.wsStatsOffset + slot * (C::N * 2);
#pragma unroll
      for (int lt = 0; lt < C::TokPerGrp; ++lt) {
        if (!ltValid(lt)) continue;
        int const t = ltTok(lt);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          int const head = (lane % 4) * 2 + (i & 1) + ((i >> 2) & 1) * 8;
          int const dim = w * 32 + lane / 4 + ((i >> 1) & 1) * 8 + ((i >> 3) & 1) * 16;
          wsO[(t * 16 + head) * kHeadDim + dim] = __uint_as_float(o[lt][i]);
        }
        if (tid < 4) {
          float tot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int ww = 0; ww < 4 * C::ShareGrps; ++ww) {
            // Warps of the token's barrier group: warpgroups t + G k (shared mode), else grp.
            int const wrow = (grp + G * (ww / 4)) * 4 + ww % 4;
            float4 const s4 =
                *reinterpret_cast<float4 const*>(&sumBuf[wrow * C::N + t * 16 + colGrp * 4]);
            tot[0] += s4.x;
            tot[1] += s4.y;
            tot[2] += s4.z;
            tot[3] += s4.w;
          }
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            int const head = tid * 2 + (c & 1) + ((c >> 1) & 1) * 8;
            wsStats[(t * 16 + head) * 2 + 0] = ref[lt][c];
            wsStats[(t * 16 + head) * 2 + 1] = tot[c];
          }
        }
      }
    }
    if (grp != 0) {
      tcFenceBefore();
      namedBarArrive(kNbTmemDone, kTmemUserThreads);
    } else {
      namedBarSync(kNbTmemDone, kTmemUserThreads);
      if (w == 0) {
        tcFenceAfter();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem)
                     : "memory");
      }
    }
  } else if (warp == C::WarpQk) {
    // ============================================================================== QK MMA issuer
    setMaxNRegDec<C::RegsMisc>();
    uint32_t qIt = 0, sIt = 0, kStage = 0, kPhase = 0;
    for (int si = 0; si < numSegs; ++si) {
      KvSeg const seg = loadSegUniform(segs, si);
      int const numSteps = seg.numPages / C::SP;
      if (numSteps == 0) continue;
      mbarWait(&bars[C::BarQFull + qIt % 2], (qIt / 2) & 1u);
      tcFenceAfter();
      uint64_t const qDesc = smemDescSw128(smem + C::SmemQ + (qIt % 2) * C::QStageBytes);
      for (int t = 0; t < numSteps; ++t) {
        if constexpr (C::EarlyS) {
          // S[sIt - Stages] has been loaded into the softmax registers (P reuse is guarded by the
          // softmax).
          if (sIt >= C::Stages) {
            uint32_t const prevS = sIt - C::Stages;
            mbarWait(&bars[C::BarSEmpty + prevS % C::Stages], (prevS / C::Stages) & 1u);
          }
        } else {
          // PV publishes that the MMAs reading P[s] are complete before QK makes S[s + PStages]
          // available to softmax, which overwrites the same probability buffer.
          if (C::PStages > C::Stages && sIt >= C::Stages) {
            // S[sIt - Stages] has been read by the softmax (P lags further behind).
            uint32_t const prevS = sIt - C::Stages;
            mbarWait(&bars[C::BarSEmpty + prevS % C::Stages], (prevS / C::Stages) & 1u);
          }
          if (sIt >= C::PStages) {
            uint32_t const prev = sIt - C::PStages;
            mbarWait(&bars[C::BarPCons + prev % C::PStages], (prev / C::PStages) & 1u);
          }
        }
        uint32_t const sCol = tmem + C::TmemS + (sIt % C::Stages) * C::SCols;
#pragma unroll 1
        for (int pp = 0; pp < C::SP; ++pp) {
          mbarWait(&bars[C::BarTkFull + kStage], kPhase);
          tcFenceAfter();
          uint32_t const aTmem = tmem + C::TmemK + kStage * 32;
          if (electOne()) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
              tcMma(sCol + pp * C::N, aTmem + k * 8, qDesc + 2 * k, C::Idesc, k > 0);
            tcCommit(&bars[C::BarTkEmpty + kStage]);
          }
          __syncwarp();
          if (++kStage == C::KT) {
            kStage = 0;
            kPhase ^= 1u;
          }
        }
        if (electOne()) tcCommit(&bars[C::BarSFull + sIt % C::Stages]);
        __syncwarp();
        ++sIt;
      }
      if (electOne()) tcCommit(&bars[C::BarQEmpty + qIt % 2]);
      __syncwarp();
      ++qIt;
    }
    tcFenceBefore();
    namedBarArrive(kNbTmemDone, kTmemUserThreads);
  } else if (warp == C::WarpQk + 1 || warp == C::WarpQk + 3) {
    // ============================================================================== loaders
    setMaxNRegDec<C::RegsMisc>();
    bool const isK = warp == C::WarpQk + 3;
    uint32_t pageBase = 0, qIt = 0;
    for (int si = 0; si < numSegs; ++si) {
      KvSeg const seg = loadSegUniform(segs, si);
      int const numPages = seg.numPages;
      if (numPages == 0) continue;
      int const seqLen = groupSeqLen(p, seg.virt, G);
      int const pageUb = ((seqLen + (1 << kPageSizeLog2) - 1) >> kPageSizeLog2) - 1;
      int32_t const* pageTable =
          p.pageTable + static_cast<size_t>(seg.virt / p.numGroups) * p.maxPagesPerSeq;
      if (isK) {
        uint32_t const stage = qIt % 2;
        mbarWait(&bars[C::BarQEmpty + stage], ((qIt / 2) & 1u) ^ 1u);
        if (electOne()) {
          mbarArriveExpectTx(&bars[C::BarQFull + stage], C::QStageBytes);
          tmaLoad4d(smem + C::SmemQ + stage * C::QStageBytes, &p.tmaQ, &bars[C::BarQFull + stage],
                    0, 0, seg.head, queryRow(p, seg.virt, G));
        }
        __syncwarp();
        ++qIt;
        issueKvPages<C, false>(p, smem, bars, pageBase, numPages, pageTable, seg.pageStart, pageUb,
                               seg.head);
      } else {
        issueKvPages<C, true>(p, smem, bars, pageBase, numPages, pageTable, seg.pageStart, pageUb,
                              seg.head);
      }
      pageBase += static_cast<uint32_t>(numPages);
    }
  } else if (warp == C::WarpQk + 2) {
    // ============================================================================== PV MMA issuer
    setMaxNRegDec<C::RegsMisc>();
    int totalSteps = 0;
    for (int si = 0; si < numSegs; ++si) totalSteps += loadSegUniform(segs, si).numPages / C::SP;
    uint32_t pIt = 0, vStage = 0, vPhase = 0, segDone = 0;
    for (int si = 0; si < numSegs; ++si) {
      KvSeg const seg = loadSegUniform(segs, si);
      int const numSteps = seg.numPages / C::SP;
      if (numSteps == 0) continue;
      for (int t = 0; t < numSteps; ++t) {
        mbarWait(&bars[C::BarPFull + pIt % C::PStages], (pIt / C::PStages) & 1u);
        tcFenceAfter();
        uint32_t const pSlot = pIt % C::PStages;
        uint8_t const* pStage = smem + C::SmemP + pSlot * C::PStageBytes;
        if (t == 0 && segDone > 0) {
          // The previous segment's final O has been read.
          mbarWait(&bars[C::BarOFree], (segDone - 1) & 1u);
          tcFenceAfter();
        }
#pragma unroll 1
        for (int pp = 0; pp < C::SP; ++pp) {
          mbarWait(&bars[C::BarTvFull + vStage], vPhase);
          tcFenceAfter();
          uint32_t const aTmem = tmem + C::TmemV + vStage * 32;
          uint64_t const pDesc = smemDescSw128(pStage + pp * (G * 2048));
          if (electOne()) {
#pragma unroll
            for (int k = 0; k < 4; ++k)
              tcMma(tmem + C::TmemO, aTmem + k * 8, pDesc + 2 * k, C::Idesc,
                    (k > 0) || (t > 0) || (pp > 0));
            tcCommit(&bars[C::BarTvEmpty + vStage]);
          }
          __syncwarp();
          if (++vStage == C::VT) {
            vStage = 0;
            vPhase ^= 1u;
          }
        }
        // Every step's MMAs are tracked by the O-done ring (softmax warpgroups wait on it before an
        // O rescale). The slot's previous phase (PStages steps back, long complete) is consumed
        // before it is reused.
        if (pIt >= C::PStages) {
          uint32_t const old = pIt - C::PStages;
          mbarWait(&bars[C::BarODone + old % C::PStages], (old / C::PStages) & 1u);
        }
        if (electOne()) tcCommit(&bars[C::BarODone + pIt % C::PStages]);
        __syncwarp();
        // P[pSlot] may be overwritten once the MMAs that read it complete (tcgen05.mma is
        // asynchronous, so issuing them is not enough).
        if (!C::EarlyS && static_cast<int>(pIt) + C::PStages < totalSteps) {
          if (electOne()) tcCommit(&bars[C::BarPCons + pIt % C::PStages]);
          __syncwarp();
        }
        if (t == numSteps - 1) {
          if (electOne()) tcCommit(&bars[C::BarOFull]);
          __syncwarp();
        }
        ++pIt;
      }
      ++segDone;
    }
    tcFenceBefore();
    namedBarArrive(kNbTmemDone, kTmemUserThreads);
  } else {
    // ============================================================================== dequant
    if constexpr (C::RegsDequant > C::RegsLaunch)
      setMaxNRegInc<C::RegsDequant>();
    else if constexpr (C::RegsDequant < C::RegsLaunch)
      setMaxNRegDec<C::RegsDequant>();
    int const dq = warp - (C::WarpQk + 4);
    if (dq < 4) {
      runDequantPipeline<C, false>(smem, bars, tmem, segs, numSegs, dq, lane);
    } else {
      runDequantPipeline<C, true>(smem, bars, tmem, segs, numSegs, dq - 4, lane);
    }
    tcFenceBefore();
    namedBarArrive(kNbTmemDone, kTmemUserThreads);
  }
}

// Merges the pieces of one (virtual request, kvHead) unit for one (query token, head) column of its
// group into the FP8 output. 256 threads. kWarpCols: each warp merges one column on its own (units
// with few pieces); otherwise the eight warps share a column, warp w taking pieces w, w + 8, ...
// Lane l owns head dims [4l, 4l + 4). The stats and O rows of kChunk pieces per warp are loaded in
// one round trip and merged with a running max.
constexpr int kCombineThreads = 256;
constexpr int kCombineWarps = kCombineThreads / 32;
template <int kChunk, bool kWarpCols>
__device__ __forceinline__ void combineBody(DecodeParams const& p) {
  constexpr int kPieceStride = kWarpCols ? 1 : kCombineWarps;
  __shared__ float4 red[kWarpCols ? 1 : kCombineWarps][32];
  __shared__ float redM[kCombineWarps];
  __shared__ float redS[kCombineWarps];
  int const unit = blockIdx.x;  // virt * kNumHeadsKv + kvHead
  int const virt = unit / kNumHeadsKv;
  int const kvHead = unit % kNumHeadsKv;
  int const tid = threadIdx.x;
  int const lane = tid % 32;
  int const warp = tid / 32;
  int const N = kWarpCols ? gridDim.y * kCombineWarps : gridDim.y;
  int const G = N / kHeadsPerKv;
  int const col = kWarpCols ? blockIdx.y * kCombineWarps + warp : blockIdx.y;  // t * 16 + head
  int const t = col / kHeadsPerKv;
  int const head = col % kHeadsPerKv;
  int const req = virt / p.numGroups;
  int const grpIdx = virt - req * p.numGroups;
  // This is the first global-memory access, including inactive/ragged group checks.
  if (p.pdlMode != 0) cudaGridDependencySynchronize();
  int const tokens = p.cumSeqLensQ ? p.cumSeqLensQ[req + 1] - p.cumSeqLensQ[req] : p.queryTokens;
  if (grpIdx * G + t >= tokens) return;  // no output row belongs to this inactive group
  // Independent loads: the unit's piece range and the request length.
  int2 const range = reinterpret_cast<int2 const*>(p.ws + p.wsRangeOffset)[unit];
  int const seqLen = p.seqLens[req];
  // out is [rows][32 q heads][128]; row = request * T + group * G + t, q head = kvHead * 16 + head.
  uint8_t* dst = p.out + ((static_cast<size_t>(queryRow(p, virt, G) + t) * kNumHeadsKv + kvHead) *
                              kHeadsPerKv +
                          head) *
                             kHeadDim;
  if (seqLen - tokens + 1 + grpIdx * G + t <= 0) {
    if (kWarpCols || tid < 32)
      reinterpret_cast<uint32_t*>(dst)[lane] = 0u;  // no visible cache entry
    return;
  }
  int const n = range.y - range.x + 1;
  float const scaleLog2 = p.scaleSoftmaxLog2Ptr ? *p.scaleSoftmaxLog2Ptr : p.scaleSoftmaxLog2;
  size_t const slot0 = static_cast<size_t>(virt + range.x) * kNumHeadsKv + kvHead;
  float const* stats = p.ws + p.wsStatsOffset + slot0 * N * 2 + col * 2;
  size_t const statStride = static_cast<size_t>(kNumHeadsKv) * N * 2;
  float const* oBase = p.ws + slot0 * N * kHeadDim + col * kHeadDim + lane * 4;
  size_t const oStride = static_cast<size_t>(kNumHeadsKv) * N * kHeadDim;
  float mW = kNegMax;  // running max of this warp's pieces
  float sW = 0.f;      // their sum, relative to mW
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int base = kWarpCols ? 0 : warp; base < n; base += kPieceStride * kChunk) {
    // Lane k < kChunk holds the (max, sum) of piece base + k * stride; every lane holds 4 dims of
    // each O row.
    int const iSt = base + lane * kPieceStride;
    float2 const st = (lane < kChunk && iSt < n)
                          ? *reinterpret_cast<float2 const*>(stats + iSt * statStride)
                          : make_float2(kNegMax, 0.f);
    float4 x[kChunk];
#pragma unroll
    for (int k = 0; k < kChunk; ++k) {
      int const i = base + k * kPieceStride;
      x[k] = i < n ? *reinterpret_cast<float4 const*>(oBase + i * oStride)
                   : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float cm = st.x;
#pragma unroll
    for (int off = 16; off > 0; off /= 2) cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, off));
    float const newM = fmaxf(mW, cm);
    float const r = exp2f((mW - newM) * scaleLog2);
    mW = newM;
    sW *= r;
    acc.x *= r;
    acc.y *= r;
    acc.z *= r;
    acc.w *= r;
    float const a = exp2f((st.x - mW) * scaleLog2);
    float ps = a * st.y;
#pragma unroll
    for (int off = 16; off > 0; off /= 2) ps += __shfl_xor_sync(0xffffffffu, ps, off);
    sW += ps;
#pragma unroll
    for (int k = 0; k < kChunk; ++k) {
      float const ak = __shfl_sync(0xffffffffu, a, k);
      acc.x = fmaf(ak, x[k].x, acc.x);
      acc.y = fmaf(ak, x[k].y, acc.y);
      acc.z = fmaf(ak, x[k].z, acc.z);
      acc.w = fmaf(ak, x[k].w, acc.w);
    }
  }
  float4 v = acc;
  float tot = sW;
  if constexpr (!kWarpCols) {
    red[warp][lane] = acc;
    if (lane == 0) {
      redM[warp] = mW;
      redS[warp] = sW;
    }
    __syncthreads();
    if (warp != 0) return;
    float m = redM[0];
#pragma unroll
    for (int w = 1; w < kCombineWarps; ++w) m = fmaxf(m, redM[w]);
    tot = 0.f;
    v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < kCombineWarps; ++w) {
      // A warp without pieces has max -max and contributes nothing.
      float const f = exp2f((redM[w] - m) * scaleLog2);
      float4 const y = red[w][lane];
      tot = fmaf(f, redS[w], tot);
      v.x = fmaf(f, y.x, v.x);
      v.y = fmaf(f, y.y, v.y);
      v.z = fmaf(f, y.z, v.z);
      v.w = fmaf(f, y.w, v.w);
    }
  }
  float const inv = (p.outputScalePtr ? *p.outputScalePtr : p.outputScale) / tot;
  uint32_t const lo =
      __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
  uint32_t const hi =
      __nv_cvt_float2_to_fp8x2(make_float2(v.z * inv, v.w * inv), __NV_SATFINITE, __NV_E4M3);
  reinterpret_cast<uint32_t*>(dst)[lane] = lo | (hi << 16);
}

// Combine variants by expected pieces per unit: one CTA per column with 10 / 5 / 3 loads per warp
// (1 to ~4 requests), or one warp per column with up to 12 loads (more requests, fewer pieces
// each).
extern "C" __global__ void __launch_bounds__(kCombineThreads)
    cake_nvfp4_decode_mtp_combine_c10(__grid_constant__ DecodeParams const p) {
  combineBody<10, false>(p);
}
extern "C" __global__ void __launch_bounds__(kCombineThreads)
    cake_nvfp4_decode_mtp_combine_c5(__grid_constant__ DecodeParams const p) {
  combineBody<5, false>(p);
}
extern "C" __global__ void __launch_bounds__(kCombineThreads)
    cake_nvfp4_decode_mtp_combine_c3(__grid_constant__ DecodeParams const p) {
  combineBody<3, false>(p);
}
extern "C" __global__ void __launch_bounds__(kCombineThreads)
    cake_nvfp4_decode_mtp_combine_w12(__grid_constant__ DecodeParams const p) {
  combineBody<12, true>(p);
}

extern "C" __global__ void __launch_bounds__(Cfg<2>::Threads, 1)
    cake_nvfp4_decode_mtp_g2(__grid_constant__ DecodeParams const p) {
  nvfp4PagedDecodeBody<2>(p);
}
extern "C" __global__ void __launch_bounds__(Cfg<4>::Threads, 1)
    cake_nvfp4_decode_mtp_g4(__grid_constant__ DecodeParams const p) {
  nvfp4PagedDecodeBody<4>(p);
}
extern "C" __global__ void __launch_bounds__(Cfg<8>::Threads, 1)
    cake_nvfp4_decode_mtp_g8(__grid_constant__ DecodeParams const p) {
  nvfp4PagedDecodeBody<8>(p);
}

// The host launch code reads Cfg<G>::SmemBytes and Cfg<G>::Threads; pin the tuned values.
static_assert(Cfg<2>::SmemBytes == 221328 && Cfg<2>::Threads == 640, "G2 resources");
static_assert(Cfg<4>::SmemBytes == 225808 && Cfg<4>::Threads == 896, "G4 resources");
static_assert(Cfg<8>::SmemBytes == 225552 && Cfg<8>::Threads == 896, "G8 resources");

}  // namespace decode_mtp
}  // namespace cake_nvfp4
}  // namespace flashinfer
