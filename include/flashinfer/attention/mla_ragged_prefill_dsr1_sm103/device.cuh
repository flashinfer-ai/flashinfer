// Agent-generated ragged MLA prefill attention for the DeepSeek-R1-class
// head geometry (12 q heads, qk head dim 192 = 128 nope + 64 rope, v head
// dim 128): fp8 e4m3 q/k/v streams, bf16 out, softmax scale 192**-0.5.
// Blackwell SM103 tcgen05 (UMMA) implementation with split-KV.
//
// Framework-independent device code: the kernels, device helpers and the
// launch-geometry macros/constants, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/mla_ragged_prefill_dsr1_sm103.cu, which owns the host-side launch
// glue (TensorView adapters, workspace orchestration, split planning) and
// the TVM-FFI entry points. The geometry macros are file-scope by design:
// never include this header in another translation unit.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define HQ 12
#define DQK 192
#define DV 128

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int NTHREADS = 128;
constexpr int NTHREADS2 = 192;
constexpr int STAGES = 2;
constexpr int PREFETCH = 2;

// TMA canonical operand layouts: 128B-swizzled 128-d portions plus the
// 64B-swizzled 64-d rope portion.
constexpr int QNSZ = BM * 128;
constexpr int QRSZ = BM * 64;
constexpr int QSZ = QNSZ + QRSZ;
constexpr int KNSZ = BN * 128;
constexpr int KRSZ = BN * 64;
constexpr int VHALF = 64 * 128;
constexpr int VHSTAGES = 3;
constexpr int VSZ = VHSTAGES * VHALF;
constexpr int VSZ1 = BN * 128;
constexpr int PSZ = BM * BN;
constexpr int KSZ = KNSZ + KRSZ;
constexpr int SMEM_BYTES = QSZ + STAGES * KSZ + VSZ + PSZ + 128;
constexpr int SMEM_BYTES_V1 = QSZ + STAGES * KSZ + VSZ1 + PSZ + 128;
static_assert(SMEM_BYTES == 114816, "unexpected K2/V3 shared-memory size");
static_assert(SMEM_BYTES_V1 == 106624, "unexpected K2/V1 shared-memory size");

constexpr int PKN2SZ = 64 * 128;
constexpr int PKR2SZ = 64 * 64;
constexpr int PV2SZ = BN * 64;
constexpr int KSTAGES2 = 3;
constexpr int PFDISTK2 = 2;
constexpr int PFDISTV2 = 3;
constexpr int VSTAGES2 = 2;
// Non-causal 2-CTA routes carry a third V stage: with the lead-time refill it
// gives the value TMA two iterations of runway, which replaces the per-tile
// cp.async.bulk.prefetch those routes used to issue.  The causal routes keep
// two stages because their second P buffer already uses that shared memory.
constexpr int SMEM2_BASE = QSZ + KSTAGES2 * (PKN2SZ + PKR2SZ) + 3 * PV2SZ + 192;
constexpr int SMEM2_CBASE = QSZ + KSTAGES2 * (PKN2SZ + PKR2SZ) + VSTAGES2 * PV2SZ + 192;
constexpr int SMEM2_NC = SMEM2_BASE + PSZ;
constexpr int SMEM2_C = SMEM2_CBASE + 2 * PSZ;
constexpr int SMEM2_PC = SMEM2_CBASE + PSZ;
constexpr int SMEM2_NC_K4 = SMEM2_CBASE + PSZ + PKN2SZ + PKR2SZ;
constexpr int SMEM2_K4_P2_V1 = QSZ + 4 * (PKN2SZ + PKR2SZ) + PV2SZ + 2 * PSZ + 192;
constexpr int SMEM2_NC_K2V3 = QSZ + 2 * (PKN2SZ + PKR2SZ) + 3 * PV2SZ + PSZ + 192;
constexpr int SMEM2_C_K2V3 = QSZ + 2 * (PKN2SZ + PKR2SZ) + 3 * PV2SZ + 2 * PSZ + 192;
static_assert(SMEM2_NC == 102592, "unexpected 2-CTA non-causal shared-memory size");
static_assert(SMEM2_C == 110784, "unexpected 2-CTA causal shared-memory size");
static_assert(SMEM2_PC == 94400, "unexpected packed-causal shared-memory size");
static_assert(SMEM2_NC_K4 == 106688, "unexpected four-stage non-causal shared-memory size");
static_assert(SMEM2_K4_P2_V1 == 114880, "unexpected K4/P2/V1 shared-memory size");
static_assert(SMEM2_NC_K2V3 == 90304, "unexpected K2/V3 non-causal shared-memory size");
static_assert(SMEM2_C_K2V3 == 106688, "unexpected causal K2/V3 shared-memory size");

constexpr float QSCALE = 0.07216878364870322f * 1.4426950408889634f;
constexpr uint32_t QSCALE_BF2 = 0x3dd53dd5u;
constexpr uint32_t QSCALE_H2 = 0x2eaa2eaau;
constexpr float PSHIFT = 8.0f;
constexpr float MINIT = -1e30f;
constexpr float MMARGIN = 2.0f;

#define DEVFN __device__ __forceinline__

DEVFN uint32_t smaddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVFN float ex2(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// mv is maintained in base-2 score units because QK is multiplied by
// softmax_scale * log2(e) before exp2.  lv also includes the internal
// 2^PSHIFT multiplier used to keep E4M3 weights away from underflow.
// NaN statistics (NaN is data: the dtype checks admit non-finite fp8 inputs)
// must reach the LSE rather than collapse to the empty-row -inf, matching the
// reducer's poisoning of both outputs; -inf is reserved for lv == 0.
DEVFN float lse2(float mv, float lv) {
  return (lv > 0.0f || isnan(lv)) ? mv - PSHIFT + __log2f(lv) : -INFINITY;
}
DEVFN void cpasync16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVFN void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVFN void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
DEVFN uint32_t pack4(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
DEVFN uint32_t pack4e5(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
DEVFN uint32_t cvth2e5(uint16_t x) {
  uint32_t d;
  asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(d) : "h"(x));
  return d;
}
DEVFN uint32_t ex2h2(uint32_t x) {
  uint32_t r;
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x));
  return r;
}
DEVFN uint32_t subh2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
DEVFN uint32_t fmah2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
DEVFN uint32_t pack_h2_to_f8(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("{ .reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 hi, %2;\n"
      "mov.b32 %0, {lo, hi}; }"
      : "=r"(d)
      : "r"(a), "r"(b));
  return d;
}
DEVFN uint32_t pack_h2_to_e5(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("{ .reg .b16 lo, hi;\n"
      "cvt.rn.satfinite.e5m2x2.f16x2 lo, %1;\n"
      "cvt.rn.satfinite.e5m2x2.f16x2 hi, %2;\n"
      "mov.b32 %0, {lo, hi}; }"
      : "=r"(d)
      : "r"(a), "r"(b));
  return d;
}
DEVFN uint32_t cvtbf2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVFN uint32_t ex2bf2(uint32_t x) {
  uint32_t d;
  asm("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(d) : "r"(x));
  return d;
}
DEVFN uint32_t addbf2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVFN uint32_t fmabf2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
DEVFN uint16_t cvtf8bf2(uint32_t x) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(x));
  return d;
}
DEVFN uint32_t cvtf8bf4(uint32_t lo2, uint32_t hi2) {
  uint32_t d;
  asm("{ .reg .b16 lo, hi; "
      "cvt.rn.satfinite.e4m3x2.bf16x2 lo, %1; "
      "cvt.rn.satfinite.e4m3x2.bf16x2 hi, %2; "
      "mov.b32 %0, {lo, hi}; }"
      : "=r"(d)
      : "r"(lo2), "r"(hi2));
  return d;
}
DEVFN uint32_t pack_bf2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVFN float unpack_bf(uint16_t x) { return __uint_as_float((uint32_t)x << 16); }
DEVFN uint32_t addh2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
DEVFN uint32_t pack_h2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVFN uint32_t pack_h2s(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
DEVFN uint32_t cvt_h2_f8x2(uint16_t x) {
  uint32_t d;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(x));
  return d;
}
DEVFN float unpack_h(uint16_t x) {
  float d;
  asm("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(x));
  return d;
}

DEVFN uint64_t mkdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFFull);
  d |= (uint64_t)((lbo >> 4) & 0x3FFFull) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFFull) << 32;
  d |= (uint64_t)1 << 46;
  return d;
}
DEVFN uint64_t mkdesc_sw(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t ltype) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFFu) >> 4);
  d |= (uint64_t)(lbo & 0x3FFFu) << 16;
  d |= (uint64_t)(sbo & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)(ltype & 7u) << 61;
  return d;
}

// ---- mbarrier ----
DEVFN void mbar_init(uint64_t* b, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smaddr(b)), "r"(cnt));
}
DEVFN void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smaddr(b)), "r"(bytes)
               : "memory");
}
DEVFN void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smaddr(b)) : "memory");
}
DEVFN void mbar_wait(uint64_t* b, int phase) {
  uint32_t a = smaddr(b);
  const uint32_t suspend = 0x989680u;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LW_%=:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
      "%1, %2;\n"
      "@!P1 bra.uni LW_%=;\n"
      "}\n" ::"r"(a),
      "r"(phase), "r"(suspend)
      : "memory");
}
// Cluster-scope consumer wait.  The wait must ACQUIRE at cluster scope:
// the peer CTA publishes shared-memory data (its P tile, or its TMA-filled
// K/V stages) before arriving on this barrier, and the subsequent
// tcgen05.fence::after_thread_sync only orders the asynchronous MMA after
// THIS thread's synchronization -- it supplies no visibility of its own, so
// a relaxed wait would let the MMA read the peer's previous-iteration
// bytes.  Paired with the release arrive in mbar_arrive_rank0 (and the
// producer-side release fences on the generic-proxy P stores).
DEVFN void mbar_wait_cluster(uint64_t* b, int phase) {
  uint32_t a = smaddr(b);
  const uint32_t suspend = 0x989680u;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LWC_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
      "P1, [%0], %1, %2;\n"
      "@!P1 bra.uni LWC_%=;\n"
      "}\n" ::"r"(a),
      "r"(phase), "r"(suspend)
      : "memory");
}
// For barriers completed only by tcgen05 operations, the following
// tcgen05.fence::after_thread_sync supplies the required TMEM visibility.
DEVFN void mbar_wait_tc(uint64_t* b, int phase) {
  uint32_t a = smaddr(b);
  const uint32_t suspend = 0x989680u;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LWTC_%=:\n"
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, "
      "[%0], %1, %2;\n"
      "@!P1 bra.uni LWTC_%=;\n"
      "}\n" ::"r"(a),
      "r"(phase), "r"(suspend)
      : "memory");
}
DEVFN void tma3d(uint32_t dst, const void* tmap, int x, int y, int z, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2,%3,%4}], [%5];" ::"r"(dst),
      "l"((uint64_t)tmap), "r"(x), "r"(y), "r"(z), "r"(smaddr(bar))
      : "memory");
}
DEVFN void tma3d_pf(const void* tmap, int x, int y, int z) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1,%2,%3}];" ::"l"((uint64_t)tmap),
      "r"(x), "r"(y), "r"(z)
      : "memory");
}

// ---- tcgen05 ----
DEVFN void tmem_alloc(uint32_t* dst_smem, int ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], "
      "%1;" ::"r"(smaddr(dst_smem)),
      "r"(ncols));
}
DEVFN void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
DEVFN void tmem_dealloc(uint32_t taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
DEVFN void tmem_alloc2(uint32_t* dst_smem, int ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], "
      "%1;" ::"r"(smaddr(dst_smem)),
      "r"(ncols));
}
DEVFN void tmem_relinquish2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
DEVFN void tmem_dealloc2(uint32_t taddr, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
DEVFN void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;"); }
DEVFN void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;"); }
DEVFN void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
DEVFN void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;"); }
DEVFN void tc_commit(uint64_t* b) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
      "cluster.b64 [%0];" ::"r"(smaddr(b)));
}
DEVFN void tc_commit2(uint64_t* b) {
  uint16_t mask = 3;
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;" ::"r"(smaddr(b)),
      "h"(mask)
      : "memory");
}
DEVFN void umma(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t scaleD) {
  uint32_t z0 = 0, z1 = 0, z2 = 0, z3 = 0;
  asm volatile(
      "{\n.reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, "
      "{%5,%6,%7,%8}, p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idesc), "r"(scaleD), "r"(z0), "r"(z1), "r"(z2), "r"(z3));
}
DEVFN void umma2(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t scaleD) {
  uint32_t z0 = 0, z1 = 0, z2 = 0, z3 = 0;
  uint32_t z4 = 0, z5 = 0, z6 = 0, z7 = 0;
  asm volatile(
      "{\n.reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, "
      "{%5,%6,%7,%8,%9,%10,%11,%12}, p;\n}\n" ::"r"(td),
      "l"(da), "l"(db), "r"(idesc), "r"(scaleD), "r"(z0), "r"(z1), "r"(z2), "r"(z3), "r"(z4),
      "r"(z5), "r"(z6), "r"(z7));
}
DEVFN void umma2_ts(uint32_t td, uint32_t ta, uint64_t db, uint32_t idesc, uint32_t scaleD) {
  uint32_t z0 = 0, z1 = 0, z2 = 0, z3 = 0;
  uint32_t z4 = 0, z5 = 0, z6 = 0, z7 = 0;
  asm volatile(
      "{\n.reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, "
      "{%5,%6,%7,%8,%9,%10,%11,%12}, p;\n}\n" ::"r"(td),
      "r"(ta), "l"(db), "r"(idesc), "r"(scaleD), "r"(z0), "r"(z1), "r"(z2), "r"(z3), "r"(z4),
      "r"(z5), "r"(z6), "r"(z7));
}
// Cluster-scope producer arrive on rank 0's barrier.  RELEASE so the
// visibility this thread holds (its own generic P stores, or the TMA data
// it just observed through its CTA-local tbar wait) transfers to the rank-0
// consumer's acquire wait above.
DEVFN void mbar_arrive_rank0(uint64_t* b) {
  uint32_t a = smaddr(b);
  asm volatile(
      "{\n"
      ".reg .b32 ra;\n"
      "mapa.shared::cluster.u32 ra, %0, 0;\n"
      "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];\n"
      "}\n" ::"r"(a)
      : "memory");
}
DEVFN int cluster_rank() {
  int r;
  asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEVFN void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVFN void cluster_sync_dealloc() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVFN void grid_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
DEVFN void grid_wait_prerequisite() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEVFN bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n.reg .pred p;\n.reg .b32 m;\n"
      "elect.sync m|p, 0xffffffff;\n"
      "selp.b32 %0, 1, 0, p;\n}\n"
      : "=r"(pred));
  return pred != 0;
}

DEVFN void tmem_ld_red32(uint32_t* r, float& redv, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
      "25,%26,%27,%28,%29,%30,%31}, %32, [%33];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=f"(redv)
      : "r"(taddr));
}

DEVFN void tmem_ld_red64(uint32_t* r, float& redv, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
      "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%"
      "48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, [%65];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]),
        "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]),
        "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]),
        "=r"(r[63]), "=f"(redv)
      : "r"(taddr));
}

DEVFN void tmem_ld_red128(uint32_t* r, float& redv, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
      "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%"
      "48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%"
      "71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%"
      "94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%"
      "113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, [%129];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]),
        "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]),
        "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]),
        "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]),
        "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]),
        "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]),
        "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]),
        "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]),
        "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]),
        "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]),
        "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]),
        "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]),
        "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127]),
        "=f"(redv)
      : "r"(taddr));
}

DEVFN void tmem_ld32(uint32_t* r, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
      "25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}

DEVFN void tmem_ld128(uint32_t* r, uint32_t taddr) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%"
      "25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%"
      "48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%"
      "71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%"
      "94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%"
      "113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, [%128];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]),
        "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]),
        "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]),
        "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]),
        "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]),
        "=r"(r[63]), "=r"(r[64]), "=r"(r[65]), "=r"(r[66]), "=r"(r[67]), "=r"(r[68]), "=r"(r[69]),
        "=r"(r[70]), "=r"(r[71]), "=r"(r[72]), "=r"(r[73]), "=r"(r[74]), "=r"(r[75]), "=r"(r[76]),
        "=r"(r[77]), "=r"(r[78]), "=r"(r[79]), "=r"(r[80]), "=r"(r[81]), "=r"(r[82]), "=r"(r[83]),
        "=r"(r[84]), "=r"(r[85]), "=r"(r[86]), "=r"(r[87]), "=r"(r[88]), "=r"(r[89]), "=r"(r[90]),
        "=r"(r[91]), "=r"(r[92]), "=r"(r[93]), "=r"(r[94]), "=r"(r[95]), "=r"(r[96]), "=r"(r[97]),
        "=r"(r[98]), "=r"(r[99]), "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]),
        "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]), "=r"(r[108]), "=r"(r[109]),
        "=r"(r[110]), "=r"(r[111]), "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]),
        "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]), "=r"(r[120]), "=r"(r[121]),
        "=r"(r[122]), "=r"(r[123]), "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127])
      : "r"(taddr));
}

DEVFN void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
      "%26,%27,%28,%29,%30,%31,%32};\n" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
      "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
      "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]),
      "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]),
      "r"(r[30]), "r"(r[31]));
}
DEVFN void tmem_st4(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(taddr),
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
}

DEVFN void tmem_st128(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x128.b32 [%0], "
      "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,"
      "%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,"
      "%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,"
      "%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,"
      "%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%"
      "114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127,%128};\n"
      :
      : "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]),
        "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]),
        "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]),
        "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]),
        "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(r[32]), "r"(r[33]), "r"(r[34]), "r"(r[35]),
        "r"(r[36]), "r"(r[37]), "r"(r[38]), "r"(r[39]), "r"(r[40]), "r"(r[41]), "r"(r[42]),
        "r"(r[43]), "r"(r[44]), "r"(r[45]), "r"(r[46]), "r"(r[47]), "r"(r[48]), "r"(r[49]),
        "r"(r[50]), "r"(r[51]), "r"(r[52]), "r"(r[53]), "r"(r[54]), "r"(r[55]), "r"(r[56]),
        "r"(r[57]), "r"(r[58]), "r"(r[59]), "r"(r[60]), "r"(r[61]), "r"(r[62]), "r"(r[63]),
        "r"(r[64]), "r"(r[65]), "r"(r[66]), "r"(r[67]), "r"(r[68]), "r"(r[69]), "r"(r[70]),
        "r"(r[71]), "r"(r[72]), "r"(r[73]), "r"(r[74]), "r"(r[75]), "r"(r[76]), "r"(r[77]),
        "r"(r[78]), "r"(r[79]), "r"(r[80]), "r"(r[81]), "r"(r[82]), "r"(r[83]), "r"(r[84]),
        "r"(r[85]), "r"(r[86]), "r"(r[87]), "r"(r[88]), "r"(r[89]), "r"(r[90]), "r"(r[91]),
        "r"(r[92]), "r"(r[93]), "r"(r[94]), "r"(r[95]), "r"(r[96]), "r"(r[97]), "r"(r[98]),
        "r"(r[99]), "r"(r[100]), "r"(r[101]), "r"(r[102]), "r"(r[103]), "r"(r[104]), "r"(r[105]),
        "r"(r[106]), "r"(r[107]), "r"(r[108]), "r"(r[109]), "r"(r[110]), "r"(r[111]), "r"(r[112]),
        "r"(r[113]), "r"(r[114]), "r"(r[115]), "r"(r[116]), "r"(r[117]), "r"(r[118]), "r"(r[119]),
        "r"(r[120]), "r"(r[121]), "r"(r[122]), "r"(r[123]), "r"(r[124]), "r"(r[125]), "r"(r[126]),
        "r"(r[127]));
}

// Keep the producer's split numbering (the reducer uses the same stride),
// but intersect its increasing tile sequence with the runtime KV prefix.
DEVFN int active_tiles(int count, int first, int step, int kv_end) {
  const int remaining = (kv_end + BN - 1) / BN - first;
  return remaining <= 0 ? 0 : min(count, 1 + (remaining - 1) / step);
}

DEVFN void write_empty_split(__nv_bfloat16* out, float* lse, uint16_t* partial, float2* stats,
                             int row, int stride, int split) {
  const size_t slot = (size_t)row * stride + split;
  uint4* dst = stride == 1 ? reinterpret_cast<uint4*>(out + (size_t)row * DV)
                           : reinterpret_cast<uint4*>(partial + slot * DV);
  for (int i = 0; i < DV / 8; ++i) dst[i] = make_uint4(0, 0, 0, 0);
  if (stride == 1)
    lse[row] = -INFINITY;
  else
    stats[slot] = make_float2(-INFINITY, 0.f);
}

// Called by the PV warp AFTER the TMA wait. Swizzling permutes 16-byte
// chunks within each V row; clearing whole physical rows needs no unswizzle.
// Each writer fences its stores before the elected lane publishes/uses V.
template <int WIDTH, int ROWS>
DEVFN void mask_v_suffix(uint8_t* stage, int valid, int lane) {
  if (valid < ROWS) {
    for (int i = max(valid, 0) * WIDTH + lane * 16; i < ROWS * WIDTH; i += 32 * 16)
      *reinterpret_cast<uint4*>(stage + i) = make_uint4(0, 0, 0, 0);
    asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;" ::
                     : "memory");
    __syncwarp();
  }
}

DEVFN void split_range(int nt, int nsplit, int s, int& S_eff, int& tb, int& te) {
  if (nt >= nsplit) {
    S_eff = nsplit;
    tb = (int)(((long long)s * nt) / nsplit);
    te = (int)(((long long)(s + 1) * nt) / nsplit);
  } else {
    S_eff = nt;
    tb = s;
    te = s + 1;
  }
}

template <bool CAUSAL, bool EXACT_B1 = false, bool FULL_Q = false, bool INTERLEAVED = false,
          bool V3 = true, bool DIRECT_Q128 = false, int FIXED_NS = 0, int FIXED_NT = 0>
__global__ __launch_bounds__(NTHREADS, 2) void mla_kernel(
    const __grid_constant__ CUtensorMap tm_qn, const __grid_constant__ CUtensorMap tm_qr,
    const __grid_constant__ CUtensorMap tm_kn, const __grid_constant__ CUtensorMap tm_kr,
    const __grid_constant__ CUtensorMap tm_v, const uint8_t* __restrict__ Qg,
    const uint8_t* __restrict__ Kg, const uint8_t* __restrict__ Vg, const int* __restrict__ q_lens,
    const int* __restrict__ kv_lens, const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ Out, float* __restrict__ Lse, uint16_t* __restrict__ partial,
    float2* __restrict__ stats, int QT, int NSPLIT, int TQ, int TKV) {
  extern __shared__ __align__(1024) char smem[];
  uint8_t* sQn = (uint8_t*)smem;
  uint8_t* sQr = sQn + QNSZ;
  uint8_t* sKn = sQr + QRSZ;
  uint8_t* sKr = sKn + STAGES * KNSZ;
  uint8_t* sV = sKr + STAGES * KRSZ;
  uint8_t* sP = sV + (V3 ? VSZ : VSZ1);
  // K is double-buffered; V and P are singletons recycled under PV completion.
  uint64_t* bar = (uint64_t*)(sP + (CAUSAL ? 2 : 1) * PSZ);
  uint64_t* b_qk = bar;         // 2
  uint64_t* b_pv = bar + 2;     // 2
  uint64_t* b_drain = bar + 4;  // score tile released by all 4 warps
  uint64_t* b_prdy = bar + 5;   // P tile published by all 4 warps
  uint64_t* tbar_q = bar + 6;
  uint64_t* tbar_k = bar + 7;
  uint64_t* tbar_v = tbar_k + STAGES;
  uint32_t* tslot = (uint32_t*)(tbar_v + (V3 ? VHSTAGES : 1));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  int bid = blockIdx.x;
  int qt, h, s, r = 0;
  if constexpr (DIRECT_Q128) {
    static_assert(!CAUSAL && EXACT_B1 && FULL_Q && INTERLEAVED && !V3);
    static_assert(FIXED_NS > 0 && FIXED_NT > FIXED_NS);
    qt = 0;
    h = blockIdx.x;
    s = blockIdx.y;
  } else if constexpr (CAUSAL) {
    h = bid % HQ;
    bid /= HQ;
    qt = QT - 1 - (bid % QT);
    bid /= QT;
    s = bid % NSPLIT;
    r = bid / NSPLIT;
  } else {
    qt = QT - 1 - (bid % QT);
    bid /= QT;
    h = bid % HQ;
    bid /= HQ;
    s = bid % NSPLIT;
    if constexpr (!EXACT_B1) r = bid / NSPLIT;
  }

  int qbase, qlen;
  if constexpr (EXACT_B1) {
    qbase = 0;
    qlen = TQ;
  } else {
    qbase = cu_q[r];
    qlen = q_lens[r];
  }
  const int m0 = qt * BM;
  if constexpr (!FULL_Q) {
    if (m0 >= qlen) return;
  }
  const int rows = FULL_Q ? BM : min(BM, qlen - m0);
  const int myrow = warp * 32 + lane;
  const int tok = qbase + m0 + myrow;

  int kbase, kvlen;
  if constexpr (CAUSAL) {
    kbase = qbase;
    kvlen = qlen;
  } else if constexpr (EXACT_B1) {
    // The exact B=1 routes keep their full-buffer tile geometry (nt stays
    // TKV/BN, so the compile-time split plans hold), but the ACTIVE key
    // extent is the runtime length: a capacity-sized K/V buffer whose
    // cumulative offsets describe a shorter prefix must not attend the
    // inactive tail.  Keys past kvlen are masked to -inf below, so their
    // e4m3 weights are exactly zero; at full extent the mask never fires.
    kbase = 0;
    kvlen = kv_lens[0];
  } else {
    kbase = cu_kv[r];
    kvlen = kv_lens[r];
  }
  // Two-key softmax uses a residual P fragment in the retired first K stage.
  const bool short_weights = !CAUSAL && !EXACT_B1 && kvlen == 2;
  const int kv_end = CAUSAL ? min(kvlen, m0 + rows) : kvlen;
  int nt;
  if constexpr (FIXED_NT > 0)
    nt = FIXED_NT;
  else
    nt = EXACT_B1 ? (TKV / BN) : ((kv_end + BN - 1) / BN);
  const int split_stride = FIXED_NS > 0 ? FIXED_NS : NSPLIT;

  if constexpr (!EXACT_B1) {
    if (nt == 0) {
      if (NSPLIT == 1 && s == 0 && (FULL_Q || myrow < rows)) {
        uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int c = 0; c < 16; c++) *(uint4*)(Out + (size_t)(tok * HQ + h) * DV + c * 8) = z;
        Lse[tok * HQ + h] = -INFINITY;
      }
      return;
    }
  }

  int S_eff, t_begin, t_end, ntiles;
  if constexpr (FIXED_NS > 0) {
    S_eff = FIXED_NS;
    t_begin = s;
    ntiles = (FIXED_NT - s + FIXED_NS - 1) / FIXED_NS;
    t_end = t_begin + ntiles * FIXED_NS;
  } else if constexpr (INTERLEAVED) {
    S_eff = min(nt, NSPLIT);
    t_begin = s;
    ntiles = (nt - s + S_eff - 1) / S_eff;
    t_end = t_begin + ntiles * S_eff;
  } else {
    split_range(nt, NSPLIT, s, S_eff, t_begin, t_end);
    ntiles = t_end - t_begin;
  }
  if (s >= S_eff) return;
  const int kv_hi = min(t_end * BN, kv_end);
  const int tile_step = INTERLEAVED ? S_eff : 1;
  if constexpr (EXACT_B1) {
    ntiles = active_tiles(ntiles, t_begin, tile_step, kv_end);
    if (ntiles == 0) {
      if (myrow < rows) write_empty_split(Out, Lse, partial, stats, tok * HQ + h, split_stride, s);
      return;
    }
  }

  // ---------------- TMEM + barriers ----------------
  if (warp == 0) tmem_alloc(tslot, 256);
  if (tid == 0) {
    mbar_init(&b_qk[0], 1);
    mbar_init(&b_qk[1], 1);
    mbar_init(&b_pv[0], 1);
    mbar_init(&b_pv[1], 1);
    mbar_init(b_drain, 4);
    mbar_init(b_prdy, 4);
    mbar_init(tbar_q, 1);
#pragma unroll
    for (int st = 0; st < STAGES; st++) mbar_init(&tbar_k[st], 1);
#pragma unroll
    for (int hh = 0; hh < (V3 ? VHSTAGES : 1); hh++) mbar_init(&tbar_v[hh], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // A clamped V tensor-map does not write rows past the global tensor extent.
  // Initialize every V stage when the request has a partial final tile; for
  // the final packed request TMA overwrites the valid prefix and leaves the
  // zero suffix for PV.  Recycled stages are cleared before the tail reload.
  if constexpr (!EXACT_B1) {
    if ((kvlen & (BN - 1)) != 0) {
      if constexpr (V3) {
#pragma unroll
        for (int hh = 0; hh < VHSTAGES; hh++)
          for (int i = tid * 16; i < VHALF; i += NTHREADS * 16)
            *(uint4*)(sV + hh * VHALF + i) = make_uint4(0u, 0u, 0u, 0u);
      } else {
        for (int i = tid * 16; i < VSZ1; i += NTHREADS * 16)
          *(uint4*)(sV + i) = make_uint4(0u, 0u, 0u, 0u);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
  }
  __syncthreads();
  const uint32_t tbase = *tslot;
  if (warp == 0) tmem_relinquish();
  const uint32_t tS = tbase;
  const uint32_t tO = tbase + 128;
  const uint32_t tlane = (uint32_t)(warp * 32) << 16;

  const uint32_t IDESC_QK = (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
  const uint32_t IDESC_PV = (1u << 4) | (1u << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
  const int qbytes = min(BM, TQ) * DQK;
  const int kbytes = min(BN, TKV) * DQK;
  const int vbytes = min(V3 ? 64 : BN, TKV) * DV;

  // Warp 0 lane 0 drives the TMA stream, warp 2 issues QK, warp 1 issues PV;
  // all four warps run the softmax.  No CTA-wide barrier inside the loop.
  if (tid == 0) {
    mbar_expect(tbar_q, (uint32_t)qbytes);
    tma3d(smaddr(sQn), &tm_qn, 0, h, qbase + m0, tbar_q);
    tma3d(smaddr(sQr), &tm_qr, 128, h, qbase + m0, tbar_q);
#pragma unroll
    for (int st = 0; st < STAGES; st++) {
      if (st < ntiles) {
        int kvi = kbase + (t_begin + st * tile_step) * BN;
        mbar_expect(&tbar_k[st], (uint32_t)kbytes);
        tma3d(smaddr(sKn + st * KNSZ), &tm_kn, 0, h, kvi, &tbar_k[st]);
        tma3d(smaddr(sKr + st * KRSZ), &tm_kr, 128, h, kvi, &tbar_k[st]);
      }
    }
    if constexpr (V3) {
#pragma unroll
      for (int hh = 0; hh < VHSTAGES; hh++) {
        if (hh < 2 * ntiles) {
          mbar_expect(&tbar_v[hh], (uint32_t)vbytes);
          const int vt = t_begin + (hh >> 1) * tile_step;
          tma3d(smaddr(sV + hh * VHALF), &tm_v, 0, h, kbase + vt * BN + (hh & 1) * 64, &tbar_v[hh]);
        }
      }
    } else {
      mbar_expect(tbar_v, (uint32_t)vbytes);
      tma3d(smaddr(sV), &tm_v, 0, h, kbase + t_begin * BN, tbar_v);
    }
  }
  if (warp == 2) {
    mbar_wait(tbar_q, 0);
    mbar_wait(&tbar_k[0], 0);
    if (elect_one()) {
      tc_fence_after();
#pragma unroll
      for (int d = 0; d < 4; d++)
        umma(tS, mkdesc_sw(smaddr(sQn + 32 * d), 1, 64, 2),
             mkdesc_sw(smaddr(sKn + 32 * d), 1, 64, 2), IDESC_QK, d ? 1u : 0u);
#pragma unroll
      for (int d = 0; d < 2; d++)
        umma(tS, mkdesc_sw(smaddr(sQr + 32 * d), 1, 32, 4),
             mkdesc_sw(smaddr(sKr + 32 * d), 1, 32, 4), IDESC_QK, 1u);
      tc_commit(&b_qk[0]);
    }
  }

  // lv: exact (pre-quantization) weight sum — feeds the LSE and the split
  // stats.  lq: the dequantized sum of the SAME e4m3 weights the PV
  // numerator uses — the non-causal output normalizer, so the output is a
  // ratio of identical quantized weights (removes the uniform-attention
  // bias).  Causal keeps the exact normalizer: its PV numerator carries
  // the e5m2 residual and already matches it.
  float mv = MINIT, lv = 0.0f, lq = 0.0f;
  uint32_t sr[128];
  int stage = 0, stage_phase = 0;

  for (int it = 0; it < ntiles; it++) {
    const int t = t_begin + it * tile_step;
    const int kb0 = t * BN;
    const int next_stage = (stage == STAGES - 1) ? 0 : stage + 1;
    const int next_phase = (next_stage == 0) ? (stage_phase ^ 1) : stage_phase;

    mbar_wait_tc(&b_qk[it & 1], (it >> 1) & 1);
    tc_fence_after();
    float rmx0, rmx1, rmx2, rmx3;
    uint32_t taddr = tS + tlane;
    tmem_ld_red32(sr, rmx0, taddr);
    tmem_ld_red32(sr + 32, rmx1, taddr + 32);
    tmem_ld_red32(sr + 64, rmx2, taddr + 64);
    tmem_ld_red32(sr + 96, rmx3, taddr + 96);
    tc_wait_ld();
    tc_fence_before();
    __syncwarp();
    if (lane == 0) mbar_arrive(b_drain);

    if (tid == 0 && it + STAGES < ntiles) {
      int kvi = kbase + (t + STAGES * tile_step) * BN;
      mbar_expect(&tbar_k[stage], (uint32_t)kbytes);
      tma3d(smaddr(sKn + stage * KNSZ), &tm_kn, 0, h, kvi, &tbar_k[stage]);
      tma3d(smaddr(sKr + stage * KRSZ), &tm_kr, 128, h, kvi, &tbar_k[stage]);
    }
    if (warp == 2 && it + 1 < ntiles) {
      mbar_wait(b_drain, it & 1);
      mbar_wait(&tbar_k[next_stage], next_phase);
      if (elect_one()) {
        tc_fence_after();
#pragma unroll
        for (int d = 0; d < 4; d++)
          umma(tS, mkdesc_sw(smaddr(sQn + 32 * d), 1, 64, 2),
               mkdesc_sw(smaddr(sKn + next_stage * KNSZ + 32 * d), 1, 64, 2), IDESC_QK,
               d ? 1u : 0u);
#pragma unroll
        for (int d = 0; d < 2; d++)
          umma(tS, mkdesc_sw(smaddr(sQr + 32 * d), 1, 32, 4),
               mkdesc_sw(smaddr(sKr + next_stage * KRSZ + 32 * d), 1, 32, 4), IDESC_QK, 1u);
        tc_commit(&b_qk[(it + 1) & 1]);
      }
    }

    float rmax = fmaxf(fmaxf(rmx0, rmx1), fmaxf(rmx2, rmx3));
    {
      // Runtime-extent masking on every route (EXACT_B1 included): at full
      // extent the predicate is false for every tile and this costs one
      // compare per tile.
      bool masked = (kb0 + BN > kv_hi) || (CAUSAL && (kb0 + BN - 1 > m0 + myrow));
      if (masked) {
        int lim = kv_hi;
        if constexpr (CAUSAL) lim = min(lim, m0 + myrow + 1);
        lim -= kb0;
        rmax = MINIT;
#pragma unroll
        for (int j = 0; j < 128; j++) {
          float v = (j < lim) ? __int_as_float(sr[j]) : -INFINITY;
          sr[j] = __float_as_int(v);
          rmax = fmaxf(rmax, v);
        }
      }
    }
    float mn;
    if constexpr (CAUSAL) {
      float cand = fmaf(rmax, QSCALE, MMARGIN);
      mn = (cand > mv + 1.0f) ? cand : mv;
    } else
      mn = fmaxf(mv, __builtin_ceilf(rmax * QSCALE * 0.25f) * 4.0f);
    float alpha = ex2(mv - mn);
    mv = mn;
    float bshift = PSHIFT - mn;
    // Scale and center before converting to half; raw FP8 dot products can
    // exceed the range and precision of half even for finite inputs.
    uint32_t packed_args[64];
    {
#pragma unroll
      for (int j = 0; j < 64; j++)
        packed_args[j] = pack_h2(fmaf(__int_as_float(sr[2 * j]), QSCALE, bshift),
                                 fmaf(__int_as_float(sr[2 * j + 1]), QSCALE, bshift));
    }

    // PV(it-1) retires the shared P tile, the V buffer and the O accumulator.
    if (it >= 1) mbar_wait(&b_pv[(it - 1) & 1], ((it - 1) >> 1) & 1);
    if constexpr (!EXACT_B1) {
      if constexpr (V3) {
        if (warp == 0 && it >= 1 && (kvlen & (BN - 1)) != 0) {
          bool cleared = false;
#pragma unroll
          for (int d = 0; d < 2; d++) {
            const int hh = 2 * it + 1 + d;
            if (hh < 2 * ntiles) {
              const int vt = t_begin + (hh >> 1) * tile_step;
              const int vrow = vt * BN + (hh & 1) * 64;
              if (vrow + 64 > kvlen) {
                const int b = hh - (hh / VHSTAGES) * VHSTAGES;
                for (int i = lane * 16; i < VHALF; i += 32 * 16)
                  *(uint4*)(sV + b * VHALF + i) = make_uint4(0u, 0u, 0u, 0u);
                cleared = true;
              }
            }
          }
          if (cleared) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
          }
        }
      } else {
        if (warp == 0 && it >= 1 && t * BN + BN > kvlen) {
          for (int i = lane * 16; i < VSZ1; i += 32 * 16)
            *(uint4*)(sV + i) = make_uint4(0u, 0u, 0u, 0u);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
        }
      }
    }
    if (tid == 0 && it >= 1) {
      if constexpr (V3) {
#pragma unroll
        for (int d = 0; d < 2; d++) {
          int hh = 2 * it + 1 + d;
          if (hh < 2 * ntiles) {
            int b = hh - (hh / VHSTAGES) * VHSTAGES;
            mbar_expect(&tbar_v[b], (uint32_t)vbytes);
            const int vt = t_begin + (hh >> 1) * tile_step;
            tma3d(smaddr(sV + b * VHALF), &tm_v, 0, h, kbase + vt * BN + (hh & 1) * 64, &tbar_v[b]);
          }
        }
      } else {
        mbar_expect(tbar_v, (uint32_t)vbytes);
        tma3d(smaddr(sV), &tm_v, 0, h, kbase + t * BN, tbar_v);
      }
    }

    uint32_t short_residual = 0;
    uint32_t ps0 = 0, ps1 = 0, ps2 = 0, ps3 = 0;
    [[maybe_unused]] uint32_t qs0 = 0, qs1 = 0, qs2 = 0, qs3 = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      uint32_t pr[4], pres[CAUSAL ? 4 : 1];
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int j = 4 * c + k;
        uint32_t e01, e23, sum01, sum23;
        if constexpr (CAUSAL) {
          uint32_t p01 = ex2h2(packed_args[2 * j]);
          uint32_t p23 = ex2h2(packed_args[2 * j + 1]);
          sum01 = p01;
          sum23 = p23;
          pr[k] = pack_h2_to_f8(p01, p23);
          e01 = cvt_h2_f8x2((uint16_t)pr[k]);
          e23 = cvt_h2_f8x2((uint16_t)(pr[k] >> 16));
          pres[k] = pack_h2_to_e5(subh2(p01, e01), subh2(p23, e23));
          e01 = addh2(e01, cvth2e5((uint16_t)pres[k]));
          e23 = addh2(e23, cvth2e5((uint16_t)(pres[k] >> 16)));
        } else {
          e01 = ex2h2(packed_args[2 * j]);
          e23 = ex2h2(packed_args[2 * j + 1]);
          sum01 = e01;
          sum23 = e23;
          pr[k] = pack_h2_to_f8(e01, e23);
          uint32_t q01, q23;
          if (j == 0 && short_weights) {
            float p0 = ex2(fmaf(__int_as_float(sr[0]), QSCALE, bshift));
            float p1 = ex2(fmaf(__int_as_float(sr[1]), QSCALE, bshift));
            pr[k] = pack4(p0, p1, 0.f, 0.f);
            e01 = cvt_h2_f8x2((uint16_t)pr[k]);
            short_residual = pack4e5(p0 - unpack_h((uint16_t)e01),
                                     p1 - unpack_h((uint16_t)(e01 >> 16)), 0.f, 0.f);
            e01 = addh2(e01, cvth2e5((uint16_t)short_residual));
            e23 = 0;
            sum01 = e01;
            sum23 = 0;
            // short path ships quantized+e5m2-residual weights: the
            // numerator-matching sum IS sum01.
            q01 = e01;
            q23 = 0;
          } else {
            q01 = cvt_h2_f8x2((uint16_t)pr[k]);
            q23 = cvt_h2_f8x2((uint16_t)(pr[k] >> 16));
          }
          if ((j & 1) == 0) {
            qs0 = addh2(qs0, q01);
            qs1 = addh2(qs1, q23);
          } else {
            qs2 = addh2(qs2, q01);
            qs3 = addh2(qs3, q23);
          }
        }
        if ((j & 1) == 0) {
          ps0 = addh2(ps0, sum01);
          ps1 = addh2(ps1, sum23);
        } else {
          ps2 = addh2(ps2, sum01);
          ps3 = addh2(ps3, sum23);
        }
      }
      if (FULL_Q || myrow < rows) {
        if (short_weights && c < 2)
          *(uint4*)(sKn + c * (BM * 16) + myrow * 16) =
              make_uint4(c == 0 ? short_residual : 0u, 0u, 0u, 0u);
        *(uint4*)(sP + c * (BM * 16) + myrow * 16) = make_uint4(pr[0], pr[1], pr[2], pr[3]);
        if constexpr (CAUSAL)
          *(uint4*)(sP + PSZ + c * (BM * 16) + myrow * 16) =
              make_uint4(pres[0], pres[1], pres[2], pres[3]);
      }
    }
    float l0 = unpack_h((uint16_t)ps0) + unpack_h((uint16_t)(ps0 >> 16));
    float l1 = unpack_h((uint16_t)ps1) + unpack_h((uint16_t)(ps1 >> 16));
    float l2 = unpack_h((uint16_t)ps2) + unpack_h((uint16_t)(ps2 >> 16));
    float l3 = unpack_h((uint16_t)ps3) + unpack_h((uint16_t)(ps3 >> 16));
    lv = lv * alpha + ((l0 + l1) + (l2 + l3));
    if constexpr (!CAUSAL) {
      float q0 = unpack_h((uint16_t)qs0) + unpack_h((uint16_t)(qs0 >> 16));
      float q1 = unpack_h((uint16_t)qs1) + unpack_h((uint16_t)(qs1 >> 16));
      float q2 = unpack_h((uint16_t)qs2) + unpack_h((uint16_t)(qs2 >> 16));
      float q3 = unpack_h((uint16_t)qs3) + unpack_h((uint16_t)(qs3 >> 16));
      lq = lq * alpha + ((q0 + q1) + (q2 + q3));
    }

    if (__any_sync(0xffffffffu, alpha != 1.0f) && it > 0) {
      uint32_t orr[32];
      tc_fence_after();
#pragma unroll
      for (int c = 0; c < 4; c++) {
        tmem_ld32(orr, tO + tlane + c * 32);
        tc_wait_ld();
#pragma unroll
        for (int j = 0; j < 32; j++) orr[j] = __float_as_int(__int_as_float(orr[j]) * alpha);
        tmem_st32(tO + tlane + c * 32, orr);
      }
      tc_wait_st();
      tc_fence_before();
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) mbar_arrive(b_prdy);

    if (warp == 1) {
      if constexpr (V3) {
        const int h0 = 2 * it, h1 = 2 * it + 1;
        const int vb0 = h0 - (h0 / VHSTAGES) * VHSTAGES;
        const int vb1 = h1 - (h1 / VHSTAGES) * VHSTAGES;
        mbar_wait(&tbar_v[vb0], (h0 / VHSTAGES) & 1);
        mbar_wait(&tbar_v[vb1], (h1 / VHSTAGES) & 1);
        if (kb0 + BN > kvlen) {
          mask_v_suffix<DV, 64>(sV + vb0 * VHALF, kvlen - kb0, lane);
          mask_v_suffix<DV, 64>(sV + vb1 * VHALF, kvlen - kb0 - 64, lane);
        }
        mbar_wait(b_prdy, it & 1);
        if (elect_one()) {
          tc_fence_after();
#pragma unroll
          for (int kb = 0; kb < 4; kb++) {
            int vb = (kb < 2) ? vb0 : vb1;
            umma(tO, mkdesc_sw(smaddr(sP + kb * 4096), BM, 8, 0),
                 mkdesc_sw(smaddr(sV + vb * VHALF + (kb & 1) * 4096), 8, 64, 2), IDESC_PV,
                 (it == 0 && kb == 0) ? 0u : 1u);
            if (short_weights && kb == 0)
              umma(tO, mkdesc_sw(smaddr(sKn), BM, 8, 0),
                   mkdesc_sw(smaddr(sV + vb * VHALF), 8, 64, 2), IDESC_PV | (1u << 7), 1u);
            if constexpr (CAUSAL)
              umma(tO, mkdesc_sw(smaddr(sP + PSZ + kb * 4096), BM, 8, 0),
                   mkdesc_sw(smaddr(sV + vb * VHALF + (kb & 1) * 4096), 8, 64, 2),
                   IDESC_PV | (1u << 7), 1u);
          }
          tc_commit(&b_pv[it & 1]);
        }
      } else {
        mbar_wait(b_prdy, it & 1);
        mbar_wait(tbar_v, it & 1);
        mask_v_suffix<DV, BN>(sV, kvlen - kb0, lane);
        if (elect_one()) {
          tc_fence_after();
#pragma unroll
          for (int kb = 0; kb < 4; kb++) {
            umma(tO, mkdesc_sw(smaddr(sP + kb * 4096), BM, 8, 0),
                 mkdesc_sw(smaddr(sV + kb * 4096), 8, 64, 2), IDESC_PV,
                 (it == 0 && kb == 0) ? 0u : 1u);
            if (short_weights && kb == 0)
              umma(tO, mkdesc_sw(smaddr(sKn), BM, 8, 0), mkdesc_sw(smaddr(sV), 8, 64, 2),
                   IDESC_PV | (1u << 7), 1u);
            if constexpr (CAUSAL)
              umma(tO, mkdesc_sw(smaddr(sP + PSZ + kb * 4096), BM, 8, 0),
                   mkdesc_sw(smaddr(sV + kb * 4096), 8, 64, 2), IDESC_PV | (1u << 7), 1u);
          }
          tc_commit(&b_pv[it & 1]);
        }
      }
    }
    stage = next_stage;
    stage_phase = next_phase;
  }

  mbar_wait_tc(&b_pv[(ntiles - 1) & 1], ((ntiles - 1) >> 1) & 1);
  tc_fence_after();

  // ---------------- epilogue ----------------
  // Non-causal normalizes the output by the quantized-weight sum; the LSE
  // and the split stats keep the exact sum (the split combine weights the
  // lq-normalized partials by the exact per-split sums — exact for uniform
  // rows, second-order otherwise).
  const float nrm = CAUSAL ? lv : lq;
  // NaN statistics on a nonempty row must poison the output as well as the
  // LSE (reducer policy); 1/NaN keeps the row NaN.  inv == 0 stays reserved
  // for the empty-KV row.
  float inv = (nrm > 0.0f || isnan(nrm)) ? 1.0f / nrm : 0.0f;
  if (split_stride == 1 && (FULL_Q || myrow < rows)) Lse[tok * HQ + h] = lse2(mv, lv);
  uint32_t orr[32];
  if (split_stride == 1) {
    uint16_t* ep = reinterpret_cast<uint16_t*>(sKn);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      tmem_ld32(orr, tO + tlane + c * 32);
      tc_wait_ld();
      if (FULL_Q || myrow < rows) {
        uint32_t ov[16];
#pragma unroll
        for (int j = 0; j < 16; j++)
          ov[j] = pack_bf2(__int_as_float(orr[2 * j]) * inv, __int_as_float(orr[2 * j + 1]) * inv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int logical = c * 32 + j * 8;
          int physical = logical ^ ((myrow & 7) * 8);
          *(uint4*)(ep + myrow * DV + physical) =
              make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
        }
      }
    }
    __syncwarp();
#pragma unroll
    for (int rb = 0; rb < 32; rb += 2) {
      int row_out = warp * 32 + rb + (lane >> 4);
      if (FULL_Q || row_out < rows) {
        int col = (lane & 15) * 8;
        int physical = col ^ ((row_out & 7) * 8);
        int tok_out = qbase + m0 + row_out;
        uint4 v = *(const uint4*)(ep + row_out * DV + physical);
        *(uint4*)(Out + (size_t)(tok_out * HQ + h) * DV + col) = v;
      }
    }
  } else {
    // Keep normalized partials in FP16: all finite E4M3 values fit.
    uint16_t* ep = reinterpret_cast<uint16_t*>(sKn);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      tmem_ld32(orr, tO + tlane + c * 32);
      tc_wait_ld();
      if (FULL_Q || myrow < rows) {
        uint32_t ov[16];
#pragma unroll
        for (int j = 0; j < 16; j++)
          ov[j] = pack_h2(__int_as_float(orr[2 * j]) * inv, __int_as_float(orr[2 * j + 1]) * inv);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int logical = c * 32 + j * 8;
          int physical = logical ^ ((myrow & 7) * 8);
          *(uint4*)(ep + myrow * DV + physical) =
              make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
        }
      }
    }
    __syncwarp();
#pragma unroll
    for (int rb = 0; rb < 32; rb += 2) {
      int row_out = warp * 32 + rb + (lane >> 4);
      if (FULL_Q || row_out < rows) {
        int col = (lane & 15) * 8;
        int physical = col ^ ((row_out & 7) * 8);
        int tok_out = qbase + m0 + row_out;
        uint16_t* pb = partial + ((size_t)(tok_out * HQ + h) * split_stride + s) * DV;
        *(uint4*)(pb + col) = *(const uint4*)(ep + row_out * DV + physical);
      }
    }
    if (FULL_Q || myrow < rows) {
      const size_t si = (size_t)(tok * HQ + h) * split_stride + s;
      stats[si] = make_float2(mv, lv);
    }
  }
  __syncthreads();
  if constexpr (EXACT_B1 && FULL_Q && INTERLEAVED)
    if (tid == 0) grid_launch_dependents();
  if (warp == 0) tmem_dealloc(tbase, 256);
}

// ---------------- 2-CTA M256 path ----------------
// Two adjacent query tiles form one tcgen05 cta_group::2 operation: A/D are
// split by M (128 rows per CTA) and B by N (64 key columns per CTA), so one
// pass over K/V serves 256 query rows instead of 128.  EXACT keeps the
// branch-free instantiation for the packed B=1 full-tile unsplit shape; the
// general instantiation adds split-KV, ragged batches, partial query/KV
// tiles, odd query-tile counts and the causal form.
template <bool CAUSAL, bool EXACT, bool PADDED = false, bool PACKED_CAUSAL = false,
          bool INTERLEAVED2 = false, bool LATE_P_WAIT = false, bool PF2 = false, int FIXED_NS = 0,
          int FIXED_NT = 0, bool K4 = false, bool P2 = false, bool V1 = false, bool TS_P = false,
          bool PDL = false, bool K2 = false, bool V3 = false, bool PACKED_STATS = false,
          bool V4 = false>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(NTHREADS2, 2) void mla_kernel2(
    const __grid_constant__ CUtensorMap tm_qn, const __grid_constant__ CUtensorMap tm_qr,
    const __grid_constant__ CUtensorMap tm_kn, const __grid_constant__ CUtensorMap tm_kr,
    const __grid_constant__ CUtensorMap tm_v, const int* __restrict__ q_lens,
    const int* __restrict__ kv_lens, const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    __nv_bfloat16* __restrict__ Out, float* __restrict__ Lse, uint16_t* __restrict__ partial,
    float2* __restrict__ stats, int QT, int PT, int NSPLIT, int TQ, int TKV) {
  // Packed causal reuses its two primary-P slots for one primary/residual
  // pair, retaining the existing shared-memory footprint and occupancy.
  constexpr int PSTAGES2 = (P2 && !PACKED_CAUSAL) ? 2 : 1;
  constexpr int PBYTES2 = CAUSAL ? 2 * PSZ : PSZ;
  constexpr int KS = K4 ? 4 : (K2 ? 2 : KSTAGES2);
  constexpr int VS = V4 ? 4 : (V1 ? 1 : ((V3 || (!CAUSAL && !K4)) ? 3 : VSTAGES2));
  extern __shared__ __align__(1024) char smem[];
  uint8_t* sQn = (uint8_t*)smem;
  uint8_t* sQr = sQn + QNSZ;
  uint8_t* sKn = sQr + QRSZ;
  uint8_t* sKr = sKn + KS * PKN2SZ;
  uint8_t* sV = sKr + KS * PKR2SZ;
  uint8_t* sP = sV + VS * PV2SZ;
  uint64_t* bar = (uint64_t*)(sP + (TS_P ? 0 : PSTAGES2 * PBYTES2));
  uint64_t* b_qk = bar;
  uint64_t* b_pv = bar + 2;
  uint64_t* b_kready = bar + 4;
  uint64_t* b_vready = bar + 5;
  uint64_t* b_drain = b_vready + VS;
  uint64_t* b_prdy = b_drain + 1;  // two alternating phases
  uint64_t* tbar_q = b_prdy + 2;
  uint64_t* tbar_k = tbar_q + 1;
  uint64_t* tbar_v = tbar_k + KS;
  uint32_t* tslot = (uint32_t*)(tbar_v + VS);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool consumer = warp < 4;
  const bool producer = warp == 4;
  const bool tma_owner = tid == NTHREADS;
  const int rank = cluster_rank();
  const int split_stride = FIXED_NS > 0 ? FIXED_NS : NSPLIT;
  int pair = blockIdx.x >> 1;
  int qt, h, s = 0, r = 0;
  if constexpr (PACKED_CAUSAL) {
    h = pair % HQ;
    const int qp = pair / HQ;
    qt = 2 * (PT - 1 - qp) + rank;
  } else if constexpr (EXACT) {
    const int qpair = pair % PT;
    h = (pair / PT) % HQ;
    s = (pair / (PT * HQ)) % split_stride;
    qt = QT - 1 - (2 * qpair + rank);
  } else {
    int p = pair, qp;
    if constexpr (CAUSAL) {
      h = p % HQ;
      p /= HQ;
      qp = p % PT;
      p /= PT;
    } else {
      qp = p % PT;
      p /= PT;
      h = p % HQ;
      p /= HQ;
    }
    s = p % NSPLIT;
    r = p / NSPLIT;
    qt = 2 * (PT - 1 - qp) + rank;
  }

  int qbase, qlen;
  if constexpr (EXACT || PADDED || PACKED_CAUSAL) {
    qbase = 0;
    qlen = TQ;
  } else {
    qbase = cu_q[r];
    qlen = q_lens[r];
  }
  const int m0 = qt * BM;
  const int m0p = m0 - rank * BM;  // rank-0 row base of this pair
  if constexpr (!EXACT && !PACKED_CAUSAL) {
    if (m0p >= qlen) return;  // both ranks idle: safe to leave together
  }
  const int rows = (EXACT || PACKED_CAUSAL) ? BM : max(0, min(BM, qlen - m0));
  // The second rank of an odd final query pair has no output rows. It must
  // still feed its K/V half to the pair MMA, but its score and softmax warps
  // can retire once the pair-wide barrier counts exclude them.
  const bool idle_rank = !(EXACT || PACKED_CAUSAL) && (rank == 1) && (m0 >= qlen);
  const bool pair_short = !(EXACT || PACKED_CAUSAL) && (m0p + BM >= qlen);
  const int myrow = warp * 32 + lane;
  const int tok = qbase + m0 + myrow;

  int kbase, kvlen;
  if constexpr (CAUSAL) {
    kbase = qbase;
    kvlen = qlen;
  } else if constexpr (EXACT || PADDED) {
    // Full-buffer tile geometry with the runtime ACTIVE key extent (see
    // mla_kernel): keys past kvlen are masked to -inf, so a capacity-sized
    // buffer with shorter cumulative offsets stays correct; at full extent
    // the mask never fires.
    kbase = 0;
    kvlen = kv_lens[0];
  } else {
    kbase = cu_kv[r];
    kvlen = kv_lens[r];
  }
  // Two-key softmax uses a residual P fragment in the retired first K stage.
  const bool short_weights = !CAUSAL && !EXACT && !PADDED && kvlen == 2;
  // Pair-wide key extent: both CTAs must run the same tile count.
  const int kv_end = CAUSAL ? min(kvlen, m0p + 2 * BM) : kvlen;
  int nt;
  if constexpr (FIXED_NT > 0)
    nt = FIXED_NT;
  else
    nt = (EXACT || PADDED) ? (TKV / BN) : ((kv_end + BN - 1) / BN);
  if constexpr (!EXACT && !PADDED && !PACKED_CAUSAL) {
    if (nt == 0) {
      if (NSPLIT == 1 && myrow < rows) {
        uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int c = 0; c < 16; c++) *(uint4*)(Out + (size_t)(tok * HQ + h) * DV + c * 8) = z;
        Lse[tok * HQ + h] = -INFINITY;
      }
      return;
    }
  }
  int S_eff = 1, t_begin = 0, t_end = nt;
  int ntiles = nt, tile_step = 1;
  if constexpr (FIXED_NS > 0) {
    S_eff = FIXED_NS;
    if constexpr (INTERLEAVED2) {
      t_begin = s;
      ntiles = (FIXED_NT - s + FIXED_NS - 1) / FIXED_NS;
      tile_step = FIXED_NS;
      t_end = t_begin + ntiles * FIXED_NS;
    } else {
      t_begin = (s * FIXED_NT) / FIXED_NS;
      t_end = ((s + 1) * FIXED_NT) / FIXED_NS;
      ntiles = t_end - t_begin;
    }
  } else if constexpr (INTERLEAVED2) {
    // Exact B=1 q256 fast path: split s owns tiles
    // s, s + S_eff, ... instead of one contiguous KV interval.
    S_eff = min(nt, NSPLIT);
    if (s >= S_eff) return;
    t_begin = s;
    ntiles = (nt - s + S_eff - 1) / S_eff;
    tile_step = S_eff;
    t_end = t_begin + ntiles * tile_step;
  } else {
    if constexpr (!PACKED_CAUSAL) {
      if (NSPLIT > 1) {
        split_range(nt, NSPLIT, s, S_eff, t_begin, t_end);
        if (s >= S_eff) return;
      }
    }
    ntiles = t_end - t_begin;
  }
  if constexpr (EXACT || PADDED) {
    // Pair-uniform decision before collective allocation (not based on rows).
    ntiles = active_tiles(ntiles, t_begin, tile_step, kv_end);
    if (ntiles == 0) {
      if (consumer && myrow < rows)
        write_empty_split(Out, Lse, partial, stats, tok * HQ + h, split_stride, s);
      return;
    }
  }
  // Per-rank masking limit (the partner may own more keys than we do).
  const int kv_hi = min(t_end * BN, CAUSAL ? min(kvlen, m0 + rows) : kvlen);

  // The collective allocation requires both CTAs to have entered the cluster.
  cluster_sync();
  if (warp == 0) tmem_alloc2(tslot, 256);
  // Separate the allocation's slot write from the barrier-init writes that
  // share the same shared-memory block; racecheck flagged the pair.
  __syncthreads();
  if (tid == 0) {
    mbar_init(&b_qk[0], 1);
    mbar_init(&b_qk[1], 1);
    mbar_init(&b_pv[0], 1);
    mbar_init(&b_pv[1], 1);
    mbar_init(b_kready, 2);
#pragma unroll
    for (int st = 0; st < VS; ++st) mbar_init(&b_vready[st], 2);
    mbar_init(b_drain, pair_short ? 4 : 8);
#pragma unroll
    for (int st = 0; st < 2; ++st)
      mbar_init(&b_prdy[st], TS_P ? (pair_short ? 4 : 8) : (pair_short ? 128 : 256));
    mbar_init(tbar_q, 1);
#pragma unroll
    for (int st = 0; st < KS; ++st) mbar_init(&tbar_k[st], 1);
#pragma unroll
    for (int st = 0; st < VS; ++st) mbar_init(&tbar_v[st], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (!EXACT && !PADDED && !PACKED_CAUSAL) {
    if ((kvlen & (BN - 1)) != 0) {
      // See mla_kernel: initialize every stage before the first partial TMA.
#pragma unroll
      for (int st = 0; st < VS; ++st)
        for (int i = tid * 16; i < PV2SZ; i += NTHREADS2 * 16)
          *(uint4*)(sV + st * PV2SZ + i) = make_uint4(0u, 0u, 0u, 0u);
      asm volatile(
          "fence.proxy.async::generic.release.sync_restrict::shared::cta."
          "cluster;" ::
              : "memory");
    }
  }
  if constexpr (!TS_P) {
    if (idle_rank) {
      for (int i = tid * 16; i < PSTAGES2 * PBYTES2; i += NTHREADS2 * 16)
        *(uint4*)(sP + i) = make_uint4(0u, 0u, 0u, 0u);
      asm volatile(
          "fence.proxy.async::generic.release.sync_restrict::shared::cta."
          "cluster;" ::
              : "memory");
    }
  }
  // Publish the TMEM allocation result and any V-tail initialization.
  __syncthreads();
  cluster_sync();
  const uint32_t tbase = *tslot;
  if (warp == 0) tmem_relinquish2();
  const uint32_t tS = tbase;
  const uint32_t tO = tbase + 128;
  const uint32_t tP = TS_P ? tS : tbase + 256;
  const uint32_t tlane = (uint32_t)(rank * BM + warp * 32) << 16;

  constexpr uint32_t IDESC_QK2 = (1u << 4) | ((128u >> 3) << 17) | ((256u >> 4) << 24);
  constexpr uint32_t IDESC_PV2 = (1u << 4) | (1u << 16) | ((128u >> 3) << 17) | ((256u >> 4) << 24);
  const int qbytes = min(BM, TQ) * DQK;
  const int kbytes = min(64, TKV) * DQK;
  const int vbytes = min(BN, TKV) * 64;

  if (tma_owner) {
    mbar_expect(tbar_q, qbytes);
    tma3d(smaddr(sQn), &tm_qn, 0, h, qbase + m0, tbar_q);
    tma3d(smaddr(sQr), &tm_qr, 128, h, qbase + m0, tbar_q);
#pragma unroll
    for (int st = 0; st < KS; st++) {
      if (st < ntiles) {
        int kvi = kbase + (t_begin + st * tile_step) * BN + rank * 64;
        mbar_expect(&tbar_k[st], kbytes);
        tma3d(smaddr(sKn + st * PKN2SZ), &tm_kn, 0, h, kvi, &tbar_k[st]);
        tma3d(smaddr(sKr + st * PKR2SZ), &tm_kr, 128, h, kvi, &tbar_k[st]);
      }
    }
#pragma unroll
    for (int st = 0; st < VS; ++st) {
      if (st < ntiles) {
        mbar_expect(&tbar_v[st], vbytes);
        tma3d(smaddr(sV + st * PV2SZ), &tm_v, rank * 64, h, kbase + (t_begin + st * tile_step) * BN,
              &tbar_v[st]);
      }
    }
    if constexpr (PF2) {
#pragma unroll
      for (int pf = 0; pf < PFDISTV2; ++pf) {
        const int kt = KS + pf;
        const int vt = VS + pf;
        if (pf < PFDISTK2 && kt < ntiles) {
          int kvi = kbase + (t_begin + kt * tile_step) * BN + rank * 64;
          tma3d_pf(&tm_kn, 0, h, kvi);
          tma3d_pf(&tm_kr, 128, h, kvi);
        }
        if (vt < ntiles) tma3d_pf(&tm_v, rank * 64, h, kbase + (t_begin + vt * tile_step) * BN);
      }
    }
  }

  if (warp == 2) {
    mbar_wait(tbar_q, 0);
    mbar_wait(&tbar_k[0], 0);
    if (lane == 0) mbar_arrive_rank0(b_kready);
    if (rank == 0) {
      mbar_wait_cluster(b_kready, 0);
      if (elect_one()) {
        tc_fence_after();
#pragma unroll
        for (int d = 0; d < 4; d++)
          umma2(tS, mkdesc_sw(smaddr(sQn + 32 * d), 1, 64, 2),
                mkdesc_sw(smaddr(sKn + 32 * d), 1, 64, 2), IDESC_QK2, d ? 1u : 0u);
#pragma unroll
        for (int d = 0; d < 2; d++)
          umma2(tS, mkdesc_sw(smaddr(sQr + 32 * d), 1, 32, 4),
                mkdesc_sw(smaddr(sKr + 32 * d), 1, 32, 4), IDESC_QK2, 1u);
        tc_commit2(&b_qk[0]);
      }
    }
  }

  if (producer) {
    uint64_t dN0 = 0, dR0 = 0;
    if constexpr ((EXACT && LATE_P_WAIT) || PACKED_CAUSAL) {
      // Exact long paths keep scalar descriptor bases. Q/K share layout bits,
      // so K-stage and 32-byte slice descriptors are encoded-address adds
      // rather than twelve full cvta/mask/shift/OR reconstructions per tile.
      dN0 = mkdesc_sw(smaddr(sQn), 1, 64, 2);
      dR0 = mkdesc_sw(smaddr(sQr), 1, 32, 4);
    }
    int pstage = 0, pstage_phase = 0;
    for (int it = 0; it < ntiles; ++it) {
      const int next_stage = (pstage == KS - 1) ? 0 : pstage + 1;
      const int next_phase = (next_stage == 0) ? (pstage_phase ^ 1) : pstage_phase;
      // QK completion proves the current K stage is no longer consumed.
      mbar_wait(&b_qk[it & 1], (it >> 1) & 1);
      if (tma_owner && it + KS < ntiles) {
        int kvi = kbase + (t_begin + (it + KS) * tile_step) * BN + rank * 64;
        mbar_expect(&tbar_k[pstage], kbytes);
        tma3d(smaddr(sKn + pstage * PKN2SZ), &tm_kn, 0, h, kvi, &tbar_k[pstage]);
        tma3d(smaddr(sKr + pstage * PKR2SZ), &tm_kr, 128, h, kvi, &tbar_k[pstage]);
        const int pfi = it + KS + PFDISTK2;
        if constexpr (PF2)
          if (pfi < ntiles) {
            int pkv = kbase + (t_begin + pfi * tile_step) * BN + rank * 64;
            tma3d_pf(&tm_kn, 0, h, pkv);
            tma3d_pf(&tm_kr, 128, h, pkv);
          }
      }
      // Own the next-QK issue so all four consumer warps can enter softmax
      // immediately after publishing their score-drain arrivals.
      if (it + 1 < ntiles) {
        mbar_wait(&tbar_k[next_stage], next_phase);
        if (lane == 0) mbar_arrive_rank0(b_kready);
        if (rank == 0) {
          if constexpr (TS_P)
            mbar_wait(&b_pv[it & 1], (it >> 1) & 1);
          else
            mbar_wait_cluster(b_drain, it & 1);
          mbar_wait_cluster(b_kready, (it + 1) & 1);
          if (elect_one()) {
            tc_fence_after();
            if constexpr ((EXACT && LATE_P_WAIT) || PACKED_CAUSAL) {
              constexpr uint64_t KN_OFF = (QNSZ + QRSZ) >> 4;
              constexpr uint64_t KR_OFF = (QRSZ + KS * PKN2SZ) >> 4;
              const uint64_t dKn = dN0 + KN_OFF + ((uint64_t)next_stage << 9);
              const uint64_t dKr = dR0 + KR_OFF + ((uint64_t)next_stage << 8);
#pragma unroll
              for (int d = 0; d < 4; d++)
                umma2(tS, dN0 + 2 * d, dKn + 2 * d, IDESC_QK2, d ? 1u : 0u);
#pragma unroll
              for (int d = 0; d < 2; d++) umma2(tS, dR0 + 2 * d, dKr + 2 * d, IDESC_QK2, 1u);
            } else {
#pragma unroll
              for (int d = 0; d < 4; d++)
                umma2(tS, mkdesc_sw(smaddr(sQn + 32 * d), 1, 64, 2),
                      mkdesc_sw(smaddr(sKn + next_stage * PKN2SZ + 32 * d), 1, 64, 2), IDESC_QK2,
                      d ? 1u : 0u);
#pragma unroll
              for (int d = 0; d < 2; d++)
                umma2(tS, mkdesc_sw(smaddr(sQr + 32 * d), 1, 32, 4),
                      mkdesc_sw(smaddr(sKr + next_stage * PKR2SZ + 32 * d), 1, 32, 4), IDESC_QK2,
                      1u);
            }
            tc_commit2(&b_qk[(it + 1) & 1]);
          }
        }
      }
      pstage = next_stage;
      pstage_phase = next_phase;
    }
  } else if (consumer) {
    if (!idle_rank) {
      // lv exact for LSE/stats; lq = dequantized e4m3 weight sum, the
      // non-causal output normalizer (see mla_kernel).
      float mv = MINIT, lv = 0.0f, lq = 0.0f;
      uint32_t sr[128];
      for (int it = 0; it < ntiles; it++) {
        const int kb0 = (t_begin + it * tile_step) * BN;

        mbar_wait_tc(&b_qk[it & 1], (it >> 1) & 1);
        tc_fence_after();
        float rmax;
        uint32_t taddr = tS + tlane;
        if constexpr (PADDED) {
          // Two half reads leave the TMEM port free between them; measured a win
          // only on the odd-tile padded route, where a short pipeline makes the
          // score drain the critical path.
          float rmaxa, rmaxb;
          tmem_ld_red64(sr, rmaxa, taddr);
          tmem_ld_red64(sr + 64, rmaxb, taddr + 64);
          tc_wait_ld();
          rmax = fmaxf(rmaxa, rmaxb);
        } else {
          tmem_ld_red128(sr, rmax, taddr);
          tc_wait_ld();
        }
        tc_fence_before();
        __syncwarp();
        if constexpr (!TS_P)
          if (lane == 0) mbar_arrive_rank0(b_drain);

        {
          // Runtime-extent masking on every route (see mla_kernel).
          bool masked = (kb0 + BN > kv_hi) || (CAUSAL && (kb0 + BN - 1 > m0 + myrow));
          if (masked) {
            int lim = kv_hi;
            if constexpr (CAUSAL) lim = min(lim, m0 + myrow + 1);
            lim -= kb0;
            rmax = MINIT;
#pragma unroll
            for (int j = 0; j < 128; j++) {
              float v = (j < lim) ? __int_as_float(sr[j]) : -INFINITY;
              sr[j] = __float_as_int(v);
              rmax = fmaxf(rmax, v);
            }
          }
        }
        float mn;
        if constexpr (CAUSAL) {
          float cand = fmaf(rmax, QSCALE, MMARGIN);
          mn = (cand > mv + 1.0f) ? cand : mv;
        } else
          mn = fmaxf(mv, __builtin_ceilf(rmax * QSCALE * 0.25f) * 4.0f);
        float alpha = ex2(mv - mn);
        mv = mn;
        // Use one FP32 scale and center for the maximum and every probability.
        float bshift = PSHIFT - mn;

        // Form a whole row of post-FMA exponent arguments before entering the
        // probability loop.  This lets the independent half2 MUFU operations
        // issue as a dense wave and releases the FP32 score fragment earlier.
        uint32_t packed_storage[64];
        uint32_t* packed_args = packed_storage;
#pragma unroll
        for (int j = 0; j < 64; j++)
          packed_args[j] = pack_h2(fmaf(__int_as_float(sr[2 * j]), QSCALE, bshift),
                                   fmaf(__int_as_float(sr[2 * j + 1]), QSCALE, bshift));

        // Recycle P only after the PV operation owning its stage has completed.
        // Two-stage routes can overlap this softmax with the preceding PV.
        // The padded q2101 route computes one 16-byte chunk before polling that
        // reuse barrier, covering part of its long-scoreboard latency with ex2.
        if constexpr (!LATE_P_WAIT) {
          if (it >= PSTAGES2) mbar_wait(&b_pv[(it - PSTAGES2) & 1], ((it - PSTAGES2) >> 1) & 1);
        }
        uint8_t* sPw = sP + (it & (PSTAGES2 - 1)) * PBYTES2;
        uint32_t short_residual = 0;
        uint32_t ps0 = 0, ps1 = 0, ps2 = 0, ps3 = 0;
        [[maybe_unused]] uint32_t qs0 = 0, qs1 = 0, qs2 = 0, qs3 = 0;
        uint32_t held_p[4], held_pres[CAUSAL ? 4 : 1];
#pragma unroll
        for (int c = 0; c < 8; c++) {
          uint32_t pr[4], pres[CAUSAL ? 4 : 1];
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int j = 4 * c + k;
            uint32_t e01, e23, sum01, sum23;
            if constexpr (CAUSAL) {
              uint32_t p01, p23;
              p01 = ex2h2(packed_args[2 * j]);
              p23 = ex2h2(packed_args[2 * j + 1]);
              sum01 = p01;
              sum23 = p23;
              pr[k] = pack_h2_to_f8(p01, p23);
              {
                e01 = cvt_h2_f8x2((uint16_t)pr[k]);
                e23 = cvt_h2_f8x2((uint16_t)(pr[k] >> 16));
                pres[k] = pack_h2_to_e5(subh2(p01, e01), subh2(p23, e23));
                e01 = addh2(e01, cvth2e5((uint16_t)pres[k]));
                e23 = addh2(e23, cvth2e5((uint16_t)(pres[k] >> 16)));
              }
            } else {
              e01 = ex2h2(packed_args[2 * j]);
              e23 = ex2h2(packed_args[2 * j + 1]);
              sum01 = e01;
              sum23 = e23;
              pr[k] = pack_h2_to_f8(e01, e23);
              uint32_t q01, q23;
              if (j == 0 && short_weights) {
                float p0 = ex2(fmaf(__int_as_float(sr[0]), QSCALE, bshift));
                float p1 = ex2(fmaf(__int_as_float(sr[1]), QSCALE, bshift));
                pr[k] = pack4(p0, p1, 0.f, 0.f);
                e01 = cvt_h2_f8x2((uint16_t)pr[k]);
                short_residual = pack4e5(p0 - unpack_h((uint16_t)e01),
                                         p1 - unpack_h((uint16_t)(e01 >> 16)), 0.f, 0.f);
                e01 = addh2(e01, cvth2e5((uint16_t)short_residual));
                e23 = 0;
                sum01 = e01;
                sum23 = 0;
                // short path ships quantized+e5m2-residual weights: the
                // numerator-matching sum IS sum01.
                q01 = e01;
                q23 = 0;
              } else {
                q01 = cvt_h2_f8x2((uint16_t)pr[k]);
                q23 = cvt_h2_f8x2((uint16_t)(pr[k] >> 16));
              }
              if ((j & 1) == 0) {
                qs0 = addh2(qs0, q01);
                qs1 = addh2(qs1, q23);
              } else {
                qs2 = addh2(qs2, q01);
                qs3 = addh2(qs3, q23);
              }
            }
            if ((j & 1) == 0) {
              ps0 = addh2(ps0, sum01);
              ps1 = addh2(ps1, sum23);
            } else {
              ps2 = addh2(ps2, sum01);
              ps3 = addh2(ps3, sum23);
            }
          }
          if (short_weights && c < 2)
            *(uint4*)(sKn + c * (BM * 16) + myrow * 16) =
                make_uint4(c == 0 ? short_residual : 0u, 0u, 0u, 0u);
          if constexpr (LATE_P_WAIT && !EXACT) {
            if (c == 0) {
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                held_p[k] = pr[k];
                if constexpr (CAUSAL) held_pres[k] = pres[k];
              }
              continue;
            }
            if (c == 1) {
              if (it >= PSTAGES2) mbar_wait(&b_pv[(it - PSTAGES2) & 1], ((it - PSTAGES2) >> 1) & 1);
              if constexpr (TS_P)
                tmem_st4(tP + tlane, held_p);
              else
                *(uint4*)(sPw + myrow * 16) =
                    make_uint4(held_p[0], held_p[1], held_p[2], held_p[3]);
              if constexpr (CAUSAL)
                *(uint4*)(sPw + PSZ + myrow * 16) =
                    make_uint4(held_pres[0], held_pres[1], held_pres[2], held_pres[3]);
            }
          } else if constexpr (LATE_P_WAIT) {
            if (c == 0 && it >= PSTAGES2)
              mbar_wait(&b_pv[(it - PSTAGES2) & 1], ((it - PSTAGES2) >> 1) & 1);
          }
          if constexpr (TS_P)
            tmem_st4(tP + tlane + 4 * c, pr);
          else
            *(uint4*)(sPw + c * (BM * 16) + myrow * 16) = make_uint4(pr[0], pr[1], pr[2], pr[3]);
          if constexpr (CAUSAL)
            *(uint4*)(sPw + PSZ + c * (BM * 16) + myrow * 16) =
                make_uint4(pres[0], pres[1], pres[2], pres[3]);
        }
        float l0 = unpack_h((uint16_t)ps0) + unpack_h((uint16_t)(ps0 >> 16));
        float l1 = unpack_h((uint16_t)ps1) + unpack_h((uint16_t)(ps1 >> 16));
        float l2 = unpack_h((uint16_t)ps2) + unpack_h((uint16_t)(ps2 >> 16));
        float l3 = unpack_h((uint16_t)ps3) + unpack_h((uint16_t)(ps3 >> 16));
        lv = lv * alpha + ((l0 + l1) + (l2 + l3));
        if constexpr (!CAUSAL) {
          float q0 = unpack_h((uint16_t)qs0) + unpack_h((uint16_t)(qs0 >> 16));
          float q1 = unpack_h((uint16_t)qs1) + unpack_h((uint16_t)(qs1 >> 16));
          float q2 = unpack_h((uint16_t)qs2) + unpack_h((uint16_t)(qs2 >> 16));
          float q3 = unpack_h((uint16_t)qs3) + unpack_h((uint16_t)(qs3 >> 16));
          lq = lq * alpha + ((q0 + q1) + (q2 + q3));
        }

        if (__any_sync(0xffffffffu, alpha != 1.0f) && it > 0) {
          // A single-stage late-wait route acquired this exact completion phase
          // before its first P store, so repeating the poll only extends the
          // output-rescale dependency chain.
          if constexpr (!(LATE_P_WAIT && PSTAGES2 == 1))
            mbar_wait_tc(&b_pv[(it - 1) & 1], ((it - 1) >> 1) & 1);
          tc_fence_after();
          if constexpr (PACKED_CAUSAL || (!CAUSAL && EXACT && LATE_P_WAIT && !INTERLEAVED2 &&
                                          FIXED_NS == 0 && !PADDED)) {
            // These routes already own a 128-register score fragment; reuse it to
            // rescale the full output row with one TMEM transaction instead of
            // four serialized 32-column load/wait/store groups.
            tmem_ld128(sr, tO + tlane);
            tc_wait_ld();
#pragma unroll
            for (int j = 0; j < 128; j++) sr[j] = __float_as_int(__int_as_float(sr[j]) * alpha);
            tmem_st128(tO + tlane, sr);
          } else {
            uint32_t orr[32];
#pragma unroll
            for (int c = 0; c < 4; c++) {
              tmem_ld32(orr, tO + tlane + c * 32);
              tc_wait_ld();
#pragma unroll
              for (int j = 0; j < 32; j++) orr[j] = __float_as_int(__int_as_float(orr[j]) * alpha);
              tmem_st32(tO + tlane + c * 32, orr);
            }
          }
          tc_wait_st();
          tc_fence_before();
        }

        if constexpr (TS_P) {
          tc_wait_st();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_rank0(&b_prdy[it & 1]);
        } else {
          asm volatile(
              "fence.proxy.async::generic.release.sync_restrict::shared::cta."
              "cluster;" ::
                  : "memory");
          // Each writer publishes its own 16-byte P segment.  The rank-0 barrier
          // therefore completes on the true final store rather than after eight
          // warp rendezvous and lane-0 handoffs across the CTA pair.
          mbar_arrive_rank0(&b_prdy[it & 1]);
        }
      }

      // The running softmax state is final as soon as the last P tile is
      // published.  Start the reciprocal and direct LSE store while the final
      // asynchronous PV MMA is still in flight, then wait only before TMEM O is
      // consumed.
      // Non-causal normalizes the output by the quantized-weight sum; the
      // LSE and the split stats keep the exact sum (see mla_kernel).
      const float nrm = CAUSAL ? lv : lq;
      // NaN statistics on a nonempty row must poison the output as well as
      // the LSE (reducer policy); 1/NaN keeps the row NaN.  inv == 0 stays
      // reserved for the empty-KV row.
      float inv = (nrm > 0.0f || isnan(nrm)) ? 1.0f / nrm : 0.0f;
      uint16_t* ep = reinterpret_cast<uint16_t*>(sQn);
      constexpr bool WIDE_DIRECT_EPI =
          !CAUSAL && EXACT && LATE_P_WAIT && !INTERLEAVED2 && FIXED_NS == 0 && !PADDED;
      const bool split = (PACKED_CAUSAL || WIDE_DIRECT_EPI) ? false : (split_stride > 1);
      if (!split && (EXACT || PACKED_CAUSAL || myrow < rows)) Lse[tok * HQ + h] = lse2(mv, lv);
      mbar_wait_tc(&b_pv[(ntiles - 1) & 1], ((ntiles - 1) >> 1) & 1);
      tc_fence_after();
      uint32_t orr[32];
      if constexpr (WIDE_DIRECT_EPI) {
        tmem_ld128(sr, tO + tlane);
        tc_wait_ld();
#pragma unroll
        for (int c = 0; c < 4; c++) {
          uint32_t ov[16];
#pragma unroll
          for (int j = 0; j < 16; j++)
            ov[j] = pack_bf2(__int_as_float(sr[c * 32 + 2 * j]) * inv,
                             __int_as_float(sr[c * 32 + 2 * j + 1]) * inv);
#pragma unroll
          for (int j = 0; j < 4; j++) {
            int logical = c * 32 + j * 8;
            int physical = logical ^ ((myrow & 7) * 8);
            *(uint4*)(ep + myrow * DV + physical) =
                make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
          }
        }
      } else if (split) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
          tmem_ld32(orr, tO + tlane + c * 32);
          tc_wait_ld();
          if (EXACT || myrow < rows) {
            uint32_t ov[16];
#pragma unroll
            for (int j = 0; j < 16; j++)
              ov[j] =
                  pack_h2(__int_as_float(orr[2 * j]) * inv, __int_as_float(orr[2 * j + 1]) * inv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              int logical = c * 32 + j * 8;
              int physical = logical ^ ((myrow & 7) * 8);
              *(uint4*)(ep + myrow * DV + physical) =
                  make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
            }
          }
        }
      } else if constexpr (PACKED_CAUSAL) {
        // Keep two 32-column TMEM loads in flight before waiting, halving the
        // serialized load-wait groups while reusing the dead score fragment.
#pragma unroll
        for (int cb = 0; cb < 4; cb += 2) {
          tmem_ld32(sr, tO + tlane + cb * 32);
          tmem_ld32(sr + 32, tO + tlane + (cb + 1) * 32);
          tc_wait_ld();
#pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            const int c = cb + cc;
            uint32_t ov[16];
#pragma unroll
            for (int j = 0; j < 16; j++)
              ov[j] = pack_bf2(__int_as_float(sr[cc * 32 + 2 * j]) * inv,
                               __int_as_float(sr[cc * 32 + 2 * j + 1]) * inv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              int logical = c * 32 + j * 8;
              int physical = logical ^ ((myrow & 7) * 8);
              *(uint4*)(ep + myrow * DV + physical) =
                  make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
            }
          }
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; c++) {
          tmem_ld32(orr, tO + tlane + c * 32);
          tc_wait_ld();
          if (EXACT || PACKED_CAUSAL || myrow < rows) {
            {
              uint32_t ov[16];
#pragma unroll
              for (int j = 0; j < 16; j++)
                ov[j] = pack_bf2(__int_as_float(orr[2 * j]) * inv,
                                 __int_as_float(orr[2 * j + 1]) * inv);
#pragma unroll
              for (int j = 0; j < 4; j++) {
                int logical = c * 32 + j * 8;
                int physical = logical ^ ((myrow & 7) * 8);
                *(uint4*)(ep + myrow * DV + physical) =
                    make_uint4(ov[4 * j], ov[4 * j + 1], ov[4 * j + 2], ov[4 * j + 3]);
              }
            }
          }
        }
      }
      __syncwarp();
      if (split) {
#pragma unroll
        for (int rb = 0; rb < 32; rb += 2) {
          int row_out = warp * 32 + rb + (lane >> 4);
          if (EXACT || row_out < rows) {
            int col = (lane & 15) * 8;
            int physical = col ^ ((row_out & 7) * 8);
            int tok_out = qbase + m0 + row_out;
            uint4 val = *(const uint4*)(ep + row_out * DV + physical);
            *(uint4*)(partial + ((size_t)(tok_out * HQ + h) * split_stride + s) * DV + col) = val;
          }
        }
      } else {
#pragma unroll
        for (int rb = 0; rb < 32; rb += 2) {
          int row_out = warp * 32 + rb + (lane >> 4);
          if (EXACT || PACKED_CAUSAL || row_out < rows) {
            int col = (lane & 15) * 8;
            int physical = col ^ ((row_out & 7) * 8);
            int tok_out = qbase + m0 + row_out;
            uint4 val = *(const uint4*)(ep + row_out * DV + physical);
            *(uint4*)(Out + (size_t)(tok_out * HQ + h) * DV + col) = val;
          }
        }
      }
      if constexpr (!PACKED_CAUSAL) {
        if (split && myrow < rows) {
          const size_t si = (size_t)(tok * HQ + h) * split_stride + s;
          stats[si] = make_float2(mv, lv);
        }
      }
    }  // !idle_rank
  } else {
    // A sixth warp retires PV independently while warp 4 keeps the K/QK
    // pipeline moving and warps 0..3 stay entirely on score/softmax work.
    uint64_t dP0 = 0, dV0 = 0;
    if constexpr (EXACT && LATE_P_WAIT) {
      dP0 = mkdesc_sw(smaddr(sP), BM, 8, 0);
      dV0 = mkdesc_sw(smaddr(sV), 8, 32, 4);
    }
    for (int it = 0; it < ntiles; ++it) {
      const int vst = it % VS;
      // Warp 5 owns V recycling as well as PV issue.  This removes the PV
      // completion dependency from warp 4's next-QK pipeline.  The refill is
      // issued after this tile's PV and targets tile it+VS-1, so a V stage is
      // filled VS-1 iterations before it is consumed instead of being awaited
      // in the same iteration it is issued.
      mbar_wait(&tbar_v[vst], (it / VS) & 1);
      // Each rank owns 64 channels of the SAME BN rows, even for an idle Q partner.
      mask_v_suffix<64, BN>(sV + vst * PV2SZ, kvlen - (t_begin + it * tile_step) * BN, lane);
      if (lane == 0) mbar_arrive_rank0(&b_vready[vst]);
      if (rank == 0) {
        mbar_wait_cluster(&b_prdy[it & 1], (it >> 1) & 1);
        mbar_wait_cluster(&b_vready[vst], (it / VS) & 1);
        if (elect_one()) {
          tc_fence_after();
#pragma unroll
          for (int kb = 0; kb < 4; kb++) {
            const uint64_t vd = (EXACT && LATE_P_WAIT)
                                    ? dV0 + ((uint64_t)vst << 9) + ((uint64_t)kb << 7)
                                    : mkdesc_sw(smaddr(sV + vst * PV2SZ + kb * 2048), 8, 32, 4);
            if constexpr (TS_P)
              umma2_ts(tO, tP + 8 * kb, vd, IDESC_PV2, (it == 0 && kb == 0) ? 0u : 1u);
            else if constexpr (EXACT && LATE_P_WAIT)
              umma2(tO,
                    dP0 + (uint64_t)(((it & (PSTAGES2 - 1)) * PBYTES2) >> 4) + ((uint64_t)kb << 8),
                    vd, IDESC_PV2, (it == 0 && kb == 0) ? 0u : 1u);
            else
              umma2(tO,
                    mkdesc_sw(smaddr(sP + (it & (PSTAGES2 - 1)) * PBYTES2 + kb * 4096), BM, 8, 0),
                    vd, IDESC_PV2, (it == 0 && kb == 0) ? 0u : 1u);
            if (short_weights && kb == 0)
              umma2(tO, mkdesc_sw(smaddr(sKn), BM, 8, 0), vd, IDESC_PV2 | (1u << 7), 1u);
            if constexpr (CAUSAL)
              umma2(tO,
                    mkdesc_sw(smaddr(sP + (it & (PSTAGES2 - 1)) * PBYTES2 + PSZ + kb * 4096), BM, 8,
                              0),
                    vd, IDESC_PV2 | (1u << 7), 1u);
          }
          tc_commit2(&b_pv[it & 1]);
        }
      }
      // The stage targeted here was released by PV(it-1), whatever VS is, so
      // the two-slot completion barrier stays unambiguous for any depth.
      const int ld = it + VS - 1;
      if (ld >= VS && ld < ntiles) {
        const int lst = ld % VS;
        mbar_wait(&b_pv[(ld - VS) & 1], ((ld - VS) >> 1) & 1);
        if constexpr (!EXACT && !PADDED && !PACKED_CAUSAL) {
          const int vt = t_begin + ld * tile_step;
          if ((kvlen & (BN - 1)) != 0 && vt * BN + BN > kvlen) {
            for (int i = lane * 16; i < PV2SZ; i += 32 * 16)
              *(uint4*)(sV + lst * PV2SZ + i) = make_uint4(0u, 0u, 0u, 0u);
            asm volatile(
                "fence.proxy.async::generic.release.sync_restrict::shared::cta."
                "cluster;" ::
                    : "memory");
            __syncwarp();
          }
        }
        if (lane == 0) {
          mbar_expect(&tbar_v[lst], vbytes);
          tma3d(smaddr(sV + lst * PV2SZ), &tm_v, rank * 64, h,
                kbase + (t_begin + ld * tile_step) * BN, &tbar_v[lst]);
          const int pfi = ld + PFDISTV2;
          if constexpr (PF2)
            if (pfi < ntiles)
              tma3d_pf(&tm_v, rank * 64, h, kbase + (t_begin + pfi * tile_step) * BN);
        }
      }
    }
  }
  __syncthreads();
  cluster_sync_dealloc();
  // The exact W10 specialization has a separate 12-way combine pass.  Let
  // that dependent grid begin once every attention CTA has published its
  // partial/stat row; TMEM teardown can overlap its prologue.
  if constexpr (PDL || (!CAUSAL && EXACT && INTERLEAVED2 && FIXED_NS == 12 && FIXED_NT == 518))
    if (tid == 0) grid_launch_dependents();
  if (warp == 0) tmem_dealloc2(tbase, 256);
}

// One warp per (token, head) row.  Phase 1 reads that row's NSPLIT (m, l)
// statistics -- contiguous after the layout change -- and turns them into
// per-split weights in shared memory.  Phase 2 streams the normalized fp16
// partials with every load independent, so the whole reduction is one
// memory-latency chain instead of NSPLIT dependent ones.
constexpr int CMB_WARPS = 4;
template <int FIXED_SPLIT, bool PDL = false>
__global__ __launch_bounds__(32 * CMB_WARPS, 1) void combine_kernel(
    const uint16_t* __restrict__ partial, const float2* __restrict__ stats,
    __nv_bfloat16* __restrict__ Out, float* __restrict__ Lse, const int* __restrict__ q_lens,
    const int* __restrict__ kv_lens, const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    int B, int TQ, int NSPLIT, int causal, int mstep) {
  if constexpr (PDL) grid_wait_prerequisite();
  __shared__ float sw[CMB_WARPS][64];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int j = blockIdx.x * CMB_WARPS + warp;
  if (j >= TQ * HQ) return;
  const int tok = j / HQ;
  const int h = j - tok * HQ;

  int lo = 0, hi = B;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (cu_q[mid] <= tok)
      lo = mid;
    else
      hi = mid;
  }
  const int r = lo;
  const int qbase = cu_q[r];
  const int qlen = q_lens[r];
  const int m0 = ((tok - qbase) / mstep) * mstep;
  const int rws = min(mstep, qlen - m0);
  const int kvlen = causal ? qlen : kv_lens[r];
  const int kv_end = causal ? min(kvlen, m0 + rws) : kvlen;
  const int nt = (kv_end + BN - 1) / BN;
  const int split_stride = FIXED_SPLIT ? FIXED_SPLIT : NSPLIT;
  const int S = FIXED_SPLIT ? FIXED_SPLIT : ((nt >= NSPLIT) ? NSPLIT : nt);

  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, L = 0.f;
  float M = -INFINITY;
  if (S > 0) {
    // The partial addresses do not depend on the statistics, so issue that
    // whole batch of loads first and let it fly while the (m, l) reduction
    // runs; otherwise the two global round trips serialize.
    constexpr int PF = FIXED_SPLIT ? FIXED_SPLIT : 16;
    const uint16_t* pb0 = partial + (size_t)j * split_stride * DV + lane * 4;
    uint2 ph0[PF];
#pragma unroll
    for (int t = 0; t < PF; t++)
      if (FIXED_SPLIT || t < S) ph0[t] = *(const uint2*)(pb0 + (size_t)t * DV);
    const float2* st = stats + (size_t)j * split_stride;
    float m0v = -INFINITY, l0v = 0.f, m1v = -INFINITY, l1v = 0.f;
    float g0 = 1.f, g1 = 1.f;
    bool bad = false;
    if (lane < S) {
      float2 ml = st[lane];
      float lq = ml.y;
      bad = (lq != lq) || (ml.x != ml.x);
      if (lq > 0.f) {
        m0v = ml.x;
        l0v = lq;
      }
    }
    if (lane + 32 < S) {
      float2 ml = st[lane + 32];
      float lq = ml.y;
      bad = bad || (lq != lq) || (ml.x != ml.x);
      if (lq > 0.f) {
        m1v = ml.x;
        l1v = lq;
      }
    }
    bad = __any_sync(0xffffffffu, bad);
    M = fmaxf(m0v, m1v);
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    if (bad) {
      // NaN statistics in any split of this row: NaN is data (the public
      // dtype checks admit non-finite fp8 inputs), so poison BOTH outputs
      // -- the fmaxf-based reductions would otherwise drop the NaN split
      // and produce a finite wrong result (or a -inf LSE).
      a0 = a1 = a2 = a3 = __int_as_float(0x7fc00000);
      M = __int_as_float(0x7fc00000);
      L = 1.0f;  // lse2(NaN, 1) propagates NaN to the LSE
    } else if (M > -INFINITY) {
      float w0 = (l0v > 0.f) ? ex2(m0v - M) * l0v : 0.f;
      float w1 = (l1v > 0.f) ? ex2(m1v - M) * l1v : 0.f;
      sw[warp][lane] = w0 * g0;
      sw[warp][lane + 32] = w1 * g1;
      L = w0 + w1;
#pragma unroll
      for (int o = 16; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
      __syncwarp();

#pragma unroll
      for (int t = 0; t < PF; t++) {
        if (FIXED_SPLIT || t < S) {
          float w = sw[warp][t];
          uint32_t lo = ph0[t].x;
          uint32_t hi = ph0[t].y;
          a0 = fmaf(unpack_h((uint16_t)lo), w, a0);
          a1 = fmaf(unpack_h((uint16_t)(lo >> 16)), w, a1);
          a2 = fmaf(unpack_h((uint16_t)hi), w, a2);
          a3 = fmaf(unpack_h((uint16_t)(hi >> 16)), w, a3);
        }
      }
      for (int s0 = PF; s0 < S; s0 += 8) {
        uint2 ph[8];
#pragma unroll
        for (int t = 0; t < 8; t++)
          if (s0 + t < S) ph[t] = *(const uint2*)(pb0 + (size_t)(s0 + t) * DV);
#pragma unroll
        for (int t = 0; t < 8; t++) {
          if (s0 + t < S) {
            float w = sw[warp][s0 + t];
            uint32_t lo = ph[t].x;
            uint32_t hi = ph[t].y;
            a0 = fmaf(unpack_h((uint16_t)lo), w, a0);
            a1 = fmaf(unpack_h((uint16_t)(lo >> 16)), w, a1);
            a2 = fmaf(unpack_h((uint16_t)hi), w, a2);
            a3 = fmaf(unpack_h((uint16_t)(hi >> 16)), w, a3);
          }
        }
      }
    }
    // else: every split published l == 0 (an empty runtime extent).  Keep
    // the empty-KV form -- zero output and lse2(-inf, 0) == -inf -- the
    // NaN case is handled above, from the statistics themselves.
  }
  float iv = (L > 0.0f) ? 1.0f / L : 0.0f;
  if (lane == 0) Lse[j] = lse2(M, L);
  uint2 ov;
  ov.x = pack_bf2(a0 * iv, a1 * iv);
  ov.y = pack_bf2(a2 * iv, a3 * iv);
  *(uint2*)(Out + (size_t)(tok * HQ + h) * DV + lane * 4) = ov;
}

// Exact seven-split combine used by q2101.  Each half warp owns one output
// row: seven lanes cover the split statistics while all sixteen lanes stream
// eight output values each.
template <int S, bool PDL = false, bool PACKED_STATS = false>
__global__ __launch_bounds__(32 * CMB_WARPS,
                             8) void combine_half_kernel(const uint16_t* __restrict__ partial,
                                                         const float2* __restrict__ stats,
                                                         __nv_bfloat16* __restrict__ Out,
                                                         float* __restrict__ Lse, int nj) {
  static_assert(S <= 16);
  __shared__ float sw[2 * CMB_WARPS][16];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half = lane >> 4;
  const int hlane = lane & 15;
  const int local = 2 * warp + half;
  const int j = blockIdx.x * (2 * CMB_WARPS) + local;
  const bool valid = j < nj;

  if constexpr (PDL) grid_wait_prerequisite();

  uint4 ph[S];
  if (valid) {
    const uint16_t* pb = partial + (size_t)j * S * DV + hlane * 8;
#pragma unroll
    for (int s = 0; s < S; s++) ph[s] = *(const uint4*)(pb + (size_t)s * DV);
  }

  float m = -INFINITY, l = 0.0f, g = 1.0f;
  int bad = 0;
  if (valid && hlane < S) {
    if constexpr (PACKED_STATS) {
      uint32_t ml = reinterpret_cast<const uint32_t*>(stats)[(size_t)j * S + hlane];
      l = unpack_bf((uint16_t)(ml >> 16));
      float mraw = unpack_bf((uint16_t)ml);
      bad = (l != l) || (mraw != mraw);
      if (l > 0.0f) m = mraw;
    } else {
      float2 ml = stats[(size_t)j * S + hlane];
      l = ml.y;
      bad = (l != l) || (ml.x != ml.x);
      if (l > 0.0f) m = ml.x;
    }
  }
  float M = m;
#pragma unroll
  for (int o = 8; o; o >>= 1) {
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o, 16));
    bad |= __shfl_xor_sync(0xffffffffu, bad, o, 16);
  }

  float w = (M > -INFINITY && l > 0.0f) ? ex2(m - M) * l : 0.0f;
  sw[local][hlane] = w * g;
  float L = w;
#pragma unroll
  for (int o = 8; o; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o, 16);
  __syncwarp();

  if (valid && bad) {
    // NaN statistics in any split of this row: poison BOTH outputs (see
    // combine_kernel) -- the max/sum reductions would otherwise drop the
    // NaN split and produce a finite wrong result.
    if (hlane == 0) Lse[j] = __int_as_float(0x7fc00000);
    const uint32_t nan2 = 0x7fc07fc0u;
    *(uint4*)(Out + (size_t)j * DV + hlane * 8) = make_uint4(nan2, nan2, nan2, nan2);
  } else if (valid && M > -INFINITY) {
    if (hlane == 0) Lse[j] = lse2(M, L);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
#pragma unroll
    for (int s = 0; s < S; s++) {
      float w = sw[local][s];
      uint32_t q0 = ph[s].x;
      uint32_t q1 = ph[s].y;
      uint32_t q2 = ph[s].z;
      uint32_t q3 = ph[s].w;
      a0 = fmaf(unpack_h((uint16_t)q0), w, a0);
      a1 = fmaf(unpack_h((uint16_t)(q0 >> 16)), w, a1);
      a2 = fmaf(unpack_h((uint16_t)q1), w, a2);
      a3 = fmaf(unpack_h((uint16_t)(q1 >> 16)), w, a3);
      a4 = fmaf(unpack_h((uint16_t)q2), w, a4);
      a5 = fmaf(unpack_h((uint16_t)(q2 >> 16)), w, a5);
      a6 = fmaf(unpack_h((uint16_t)q3), w, a6);
      a7 = fmaf(unpack_h((uint16_t)(q3 >> 16)), w, a7);
    }
    float iv = (L > 0.0f) ? 1.0f / L : 0.0f;
    uint4 ov = make_uint4(pack_bf2(a0 * iv, a1 * iv), pack_bf2(a2 * iv, a3 * iv),
                          pack_bf2(a4 * iv, a5 * iv), pack_bf2(a6 * iv, a7 * iv));
    *(uint4*)(Out + (size_t)j * DV + hlane * 8) = ov;
  } else if (valid) {
    // Every split published l == 0 (an empty runtime extent): keep the
    // empty-KV form, zero output and -inf LSE.  The NaN case is handled
    // above, from the statistics themselves.
    if (hlane == 0) Lse[j] = -INFINITY;
    *(uint4*)(Out + (size_t)j * DV + hlane * 8) = make_uint4(0u, 0u, 0u, 0u);
  }
}

// Small-ragged correctness path.  The fast attention pipeline deliberately
// quantizes its probability tile to E4M3 before PV, so its normalization
// statistic is approximate.  Recompute only LSE from the original FP8 Q/K
// for short segments; all timed serving rows have much larger KV lengths.
__global__ __launch_bounds__(128) void exact_small_lse_kernel(
    const uint8_t* __restrict__ Q, const uint8_t* __restrict__ K, const int* __restrict__ q_lens,
    const int* __restrict__ kv_lens, const int* __restrict__ cu_q, const int* __restrict__ cu_kv,
    int B, int TQ, int causal, float* __restrict__ Lse) {
  const int j = blockIdx.x;
  if (j >= TQ * HQ) return;
  const int tok = j / HQ;
  const int h = j - tok * HQ;
  __shared__ int meta[3];
  __shared__ float red[128];
  if (threadIdx.x == 0) {
    int lo = 0, hi = B;
    while (lo + 1 < hi) {
      int mid = (lo + hi) >> 1;
      if (cu_q[mid] <= tok)
        lo = mid;
      else
        hi = mid;
    }
    const int qbase = cu_q[lo];
    meta[0] = qbase;
    meta[1] = causal ? qbase : cu_kv[lo];
    meta[2] = causal ? (tok - qbase + 1) : kv_lens[lo];
  }
  __syncthreads();
  const int kbase = meta[1];
  const int nk = meta[2];
  const uint8_t* qp = Q + (size_t)(tok * HQ + h) * DQK;
  float local_m = -INFINITY;
  for (int p = threadIdx.x; p < nk; p += blockDim.x) {
    const uint8_t* kp = K + (size_t)((kbase + p) * HQ + h) * DQK;
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < DQK; d += 2) {
      uint32_t q2 = cvt_h2_f8x2(*(const uint16_t*)(qp + d));
      uint32_t k2 = cvt_h2_f8x2(*(const uint16_t*)(kp + d));
      dot = fmaf(unpack_h((uint16_t)q2), unpack_h((uint16_t)k2), dot);
      dot = fmaf(unpack_h((uint16_t)(q2 >> 16)), unpack_h((uint16_t)(k2 >> 16)), dot);
    }
    local_m = fmaxf(local_m, dot * QSCALE);
  }
  red[threadIdx.x] = local_m;
  __syncthreads();
  for (int s = 64; s; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float M = red[0];
  __syncthreads();
  float local_l = 0.0f;
  for (int p = threadIdx.x; p < nk; p += blockDim.x) {
    const uint8_t* kp = K + (size_t)((kbase + p) * HQ + h) * DQK;
    float dot = 0.0f;
#pragma unroll
    for (int d = 0; d < DQK; d += 2) {
      uint32_t q2 = cvt_h2_f8x2(*(const uint16_t*)(qp + d));
      uint32_t k2 = cvt_h2_f8x2(*(const uint16_t*)(kp + d));
      dot = fmaf(unpack_h((uint16_t)q2), unpack_h((uint16_t)k2), dot);
      dot = fmaf(unpack_h((uint16_t)(q2 >> 16)), unpack_h((uint16_t)(k2 >> 16)), dot);
    }
    local_l += ex2(dot * QSCALE - M);
  }
  red[threadIdx.x] = local_l;
  __syncthreads();
  for (int s = 64; s; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) Lse[j] = nk > 0 ? M + __log2f(red[0]) : -INFINITY;
}
