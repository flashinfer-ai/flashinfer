/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include "exact_topk.cuh"
#include "indexer_q4_types.cuh"
#include "ll_ids_protocol.h"
#include "long_topk.cuh"

namespace flashinfer::glm5_tp8::indexer_q4 {

static_assert(kTopk == long_topk::kTopK && kBlock == long_topk::kThreads,
              "long selector matches the indexer query geometry");
static_assert(sizeof(long_topk::Scratch) + smem::kRing <= smem::kDynamicBytes,
              "long selector preserves the indexer shared-memory budget");

__host__ __device__ constexpr bool long_context(int cur_pos) {
  return total_len(cur_pos) > kMaxSeqLen;
}

// ---- named barriers -------------------------------------------------------------------------
__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 4, %0;" ::"n"(kConsumerThreads) : "memory");
}
__device__ __forceinline__ void cta384_sync() {
  asm volatile("bar.sync 4, %0;" ::"n"(kBlock) : "memory");
}
__device__ __forceinline__ void group_sync(const OpCtx& ctx) {
  if (ctx.sync_threads == kBlock)
    cta384_sync();
  else
    consumer_sync();
}

// ---- register split ---------
// Warpgroup-uniform: warps 0..7 (two warpgroups) inc, warps 8..11 (one warpgroup) dec.
__device__ __forceinline__ void setmaxnreg_consumer() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kConsumerMaxReg));
}
__device__ __forceinline__ void setmaxnreg_prefetcher() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kPrefetchMaxReg));
}

// ---- fences ----------------------------------------------------------------------------------
__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}
// generic-proxy writes -> async proxy readers (UMMA / TMA); required, issued writer-side by
// every thread after its cp.async.wait_group and BEFORE the bar.sync that hands the tile to the
// elected MMA thread.
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void spin_backoff() {}

// ---- gpu-scope word primitives (boundary words, S barrier, tickets) -----------------------------
__device__ __forceinline__ uint32_t load_relaxed_gpu(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ uint64_t ld_relaxed_gpu_u64(const uint64_t* address) {
  uint64_t value;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ void st_relaxed_gpu_u64(uint64_t* address, uint64_t value) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}
// Staged refine composites are key32 << 18 | (2^18 - 1 - pos) (50 bits); P5 stores them with a
// non-zero 14-bit tag of the launch epoch in bits [50, 64) so the refine can take them as soon as
// they are written.  The refine clears every slot it consumed, so a slot only ever holds 0 or the
// composite of the launch that wrote it.  Equal tags on all candidates leave the select unchanged.
constexpr int kCompTagShift = 50;
__device__ __forceinline__ uint64_t comp_tag(uint32_t epoch) {
  return static_cast<uint64_t>(1u + epoch % 0x3fffu);
}
__device__ __forceinline__ void red_release_gpu_add(uint32_t* address, uint32_t value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}
__device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* address, uint32_t value) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(address), "r"(value)
               : "memory");
  return old;
}
__device__ __forceinline__ uint32_t atom_relaxed_gpu_add(uint32_t* address, uint32_t value) {
  uint32_t old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(address), "r"(value)
               : "memory");
  return old;
}

// ---- system-scope 128-bit records (ids records and collective lines) ---------------------------
// PTX single-copy atomicity of the scalar .b128 access keeps payload and tag words together (the
// receivers test dwords 1,3 / 2,3 against the tag).
__device__ __forceinline__ void st_relaxed_sys_b128(void* address, uint32_t w0, uint32_t w1,
                                                    uint32_t w2, uint32_t w3) {
  const uint64_t lo = uint64_t(w0) | (uint64_t(w1) << 32);
  const uint64_t hi = uint64_t(w2) | (uint64_t(w3) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.sys.global.b128 [%0], r;\n\t}" ::
          "l"(address),
      "l"(lo), "l"(hi)
      : "memory");
}
__device__ __forceinline__ void ld_relaxed_sys_b128(const void* address, uint64_t& lo,
                                                    uint64_t& hi) {
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.sys.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo), "=l"(hi)
      : "l"(address)
      : "memory");
}

__device__ __forceinline__ bool g2b_refine_fast(const IndexerArgs& args, int cur_pos) {
  return args.mode == kModePersistent && !has_inline_ctx(args) && cur_pos > kLastDenseCurPos &&
         !long_context(cur_pos);
}

// ---- paired fp32 arithmetic of the rank-0 reduce -----------------------------------------
union F32Pair {
  uint64_t bits;
  float value[2];
};
__device__ __forceinline__ uint64_t pack_f32x2(float low, float high) {
  return static_cast<uint64_t>(__float_as_uint(low)) |
         (static_cast<uint64_t>(__float_as_uint(high)) << 32);
}
__device__ __forceinline__ uint64_t
add_f32x2(uint64_t a, uint64_t b) {  // add.rn.f32x2 = two independent correctly rounded fp32 adds
  uint64_t result;
  asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
  return result;
}
__device__ __forceinline__ float bf16_pair_low(uint32_t bits) {
  return __uint_as_float(bits << 16);
}  // lower-address element
__device__ __forceinline__ float bf16_pair_high(uint32_t bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t arrival_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(arrival_count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "MBAR_WAIT_LOOP:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra MBAR_WAIT_DONE;\n\t"
      "bra MBAR_WAIT_LOOP;\n\t"
      "MBAR_WAIT_DONE:\n\t"
      "}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_count(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(tx_bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ void bulk_copy_g2s_hint(uint32_t dst_smem, const void* src_global,
                                                   uint32_t bytes, uint32_t bar,
                                                   uint64_t l2_policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], %4;" ::"r"(dst_smem),
      "l"(src_global), "r"(bytes), "r"(bar), "l"(l2_policy)
      : "memory");
}

__device__ __forceinline__ uint64_t l2_policy_stream() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void tma_load_2d_hint(uint32_t dst_smem, const void* tmap, int32_t c0,
                                                 int32_t c1, uint32_t bar, uint64_t l2_policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst_smem),
      "l"(tmap), "r"(c0), "r"(c1), "r"(bar), "l"(l2_policy)
      : "memory");
}

__device__ __forceinline__ void ldgsts_16(uint32_t dst_smem, const void* src_global) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_smem), "l"(src_global)
               : "memory");
}
__device__ __forceinline__ void ldgsts_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void ldgsts_wait_all() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// ---- shared loads by 32-bit shared-window address ----------------------------------------------
__device__ __forceinline__ void lds_u32x2(uint32_t address, uint32_t& lo, uint32_t& hi) {
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lo), "=r"(hi) : "r"(address));
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t address) {
  uint32_t value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
  return value;
}
__device__ __forceinline__ float lds_f32(uint32_t address) {
  float value;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(value) : "r"(address));
  return value;
}

// ---- TMEM (tcgen05) allocation / release ---------------------
// Whole-warp instructions (warp 0 of the CTA), 128 columns.  The allocated column base lands in
// smem (kTmemBase); dealloc reads it back.
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem_u32, uint32_t ncols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem_u32),
      "r"(ncols)
      : "memory");
}
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ __nv_bfloat16 to_bf16(float value) { return __float2bfloat16_rn(value); }
// PTX cvt.rn.bf16x2.f32 packs its FIRST source into the HIGH half: reversed operands keep `low` at
// bits 15:0.
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  uint32_t bits;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(high), "f"(low));
  return bits;
}
// fp8 e4m3 x8 -> four bf16x2 A-fragment registers
__device__ __forceinline__ void fp8x8_to_bf16_fragment(uint32_t lo, uint32_t hi,
                                                       uint32_t (&fragment)[4]) {
  asm volatile(
      "{\n\t"
      ".reg .b16 h0,h1,h2,h3;\n\t"
      "mov.b32 {h0,h1},%4;\n\t"
      "mov.b32 {h2,h3},%5;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %0,h0;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %1,h1;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %2,h2;\n\t"
      "cvt.rn.bf16x2.e4m3x2 %3,h3;\n\t"
      "}"
      : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
      : "r"(lo), "r"(hi));
}
struct HmmaAcc {
  float value[4];
};
__device__ __forceinline__ void hmma_16816_bf16(HmmaAcc& acc, const uint32_t (&a)[4],
                                                const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(acc.value[0]), "+f"(acc.value[1]), "+f"(acc.value[2]), "+f"(acc.value[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- RoPE + Sylvester H_128 of the rotate family ----
// Data layout of both phases: one 128-dim row is held by 16 consecutive lanes (lane16 = tid & 15),
// 8 consecutive dims [8*lane16, +8) per thread (one 16 B load / store).  Everything is fp32; the
// only bf16 rounding is the caller's final pack.  bf16 pair (one u32 of the 16 B load) -> two fp32:
// pure shifts, no rounding (bf16_pair_low / bf16_pair_high).
__device__ __forceinline__ void p3_unpack_bf16x8(const uint4& raw, float (&v)[8]) {
  v[0] = bf16_pair_low(raw.x);
  v[1] = bf16_pair_high(raw.x);
  v[2] = bf16_pair_low(raw.y);
  v[3] = bf16_pair_high(raw.y);
  v[4] = bf16_pair_low(raw.z);
  v[5] = bf16_pair_high(raw.z);
  v[6] = bf16_pair_low(raw.w);
  v[7] = bf16_pair_high(raw.w);
}
// RoPE on ONE adjacent pair (x[2j], x[2j+1]) with (c, s) = (cos_j, sin_j), the four-instruction
// form (P3 and Group2a):  t = x1*s;  o0 = fma(x0, c, -t);  u = x1*c;  o1 = fma(x0, s, u). __fmul_rn
// / __fmaf_rn pin each rounding, so the compiler cannot contract or reorder them.
__device__ __forceinline__ void p3_rope_pair(float& x0, float& x1, float c, float s) {
  const float t = __fmul_rn(x1, s);
  const float o0 = __fmaf_rn(x0, c, -t);
  const float u = __fmul_rn(x1, c);
  const float o1 = __fmaf_rn(x0, s, u);
  x0 = o0;
  x1 = o1;
}
// The 4 adjacent pairs of one thread's 8 dims with its 8 table words cs = (c0, s0, c1, s1, c2, s2,
// c3, s3) (P3 lanes (tid&15) < 8 = dims 0..63 only; the table row is rope_cos_sin + q*256 B +
// lane16*32 B.
__device__ __forceinline__ void p3_rope_dims8(float (&v)[8], const float (&cs)[8]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) p3_rope_pair(v[2 * j], v[2 * j + 1], cs[2 * j], cs[2 * j + 1]);
}
// In-register butterfly levels of H_128 (dim strides 1, 2, 4 inside the thread's 8 dims): (a, b) ->
// (a + b, a - b) with a the lower dim; IEEE addition is commutative, so the operand order of the
// sums does not matter; the association (which pair at which level) does and is kept.
__device__ __forceinline__ void p3_hadamard8_regs(float (&v)[8]) {
#pragma unroll
  for (int h = 1; h <= 4; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((i & h) == 0) {
        const float a = v[i];
        const float b = v[i + h];
        v[i] = __fadd_rn(a, b);
        v[i + h] = __fadd_rn(a, -b);
      }
    }
  }
}
// Cross-lane levels (dim strides 8, 16, 32, 64 = lane xor 1, 2, 4, 8), per level and per element:
// partner = shuffle-xor; own' = (tid bit) ? -own : own; partner + own': the lower lane of a pair
// gets a + b, the upper gets a - b (a = lower lane's value) -- the same (a + b, a - b) butterfly as
// the register levels.  __shfl_xor_sync(0xffffffff, ., m) with m <= 8 stays inside the 16-lane
// half-warp of the row; all 32 lanes must call it (the rows >= N of a CTA carry zeros and still
// shuffle).
__device__ __forceinline__ void p3_hadamard_lanes16(float (&v)[8], int lane16) {
#pragma unroll
  for (int m = 1; m <= 8; m <<= 1) {
    const bool upper = (lane16 & m) != 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float partner = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = __fadd_rn(partner, upper ? -v[i] : v[i]);
    }
  }
}
// Full Sylvester H_128 (natural order) of one row = 3 register levels + 4 shuffle levels; unscaled
// (the caller multiplies by kHadamardScale = 128^-0.5).  Same butterfly order as a natural-order
// fast Walsh-Hadamard transform with strides 1..64.
__device__ __forceinline__ void p3_hadamard128(float (&v)[8], int lane16) {
  p3_hadamard8_regs(v);
  p3_hadamard_lanes16(v, lane16);
}

// ---- fp16 / fp32 monotone keys of the two-stage top-k  --------
// stage A/B: h16 = fp16(s) rounded toward zero; key = h16 >= 0 ? h16 | 0x8000 : ~h16 (16 bit); bin
// = key >> 8
__device__ __forceinline__ uint32_t score_bin16(float s) {
  const uint32_t h = static_cast<uint32_t>(__half_as_ushort(__float2half_rz(s)));
  const uint32_t key = (h & 0x8000u) ? (~h & 0xffffu) : (h | 0x8000u);
  return key >> 8;
}
// stage C: fp32 key(v) = v >= 0 ? bits(v) | 0x80000000 : ~bits(v)  (monotone), digit = (key >> (16
// - 8r)) & 0xff
__device__ __forceinline__ uint32_t score_key32(float v) {
  const uint32_t b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

}  // namespace flashinfer::glm5_tp8::indexer_q4
// boundary-word arrive/wait idioms, the sense-reversing S barrier, the H-block clear.
// The words are never reset: every launch adds exactly kWordSum (0x100) to every word, so the wait
// target of launch e is word_target(e) = 0x100 * e (monotonic, signed compare).  Arrive = release
// reduction by thread 0 after the group barrier (no round trip); wait = lane-0 relaxed poll, one
// acquire fence, group barrier.

namespace flashinfer::glm5_tp8::indexer_q4 {

__device__ __forceinline__ bool word_reached(const uint32_t* word, uint32_t target) {
  return static_cast<int32_t>(load_relaxed_gpu(word) - target) >= 0;
}

// Boundary arrive: group barrier (all of this CTA's stores of the phase are issued), then thread 0
// adds `weight`. CTA-uniform call by every thread of the group (256 consumers, or 384 in Group2b).
__device__ __forceinline__ void boundary_arrive(const OpCtx& ctx, int word_offset,
                                                uint32_t weight) {
  group_sync(ctx);
  if (ctx.tid == 0) {
    red_release_gpu_add(sync_word(ctx, word_offset), weight);
  }
}

// Poll helper: thread 0 spins until every word reached its target; then one acquire fence, then
// the group barrier broadcasts the acquisition.
__device__ __forceinline__ void boundary_wait_impl(const OpCtx& ctx, const uint32_t* w0,
                                                   uint32_t t0, const uint32_t* w1, uint32_t t1) {
  const bool poller = ctx.tid == 0;
  if (poller) {
    for (;;) {
      const bool r0 = word_reached(w0, t0);
      const bool r1 = (w1 == nullptr) || word_reached(w1, t1);
      if (r0 && r1) break;
      spin_backoff();
    }
    fence_acquire_gpu();
  }
  group_sync(ctx);
}
__device__ __forceinline__ void boundary_wait(const OpCtx& ctx, int word_offset) {
  boundary_wait_impl(ctx, sync_word(ctx, word_offset), word_target(ctx.epoch), nullptr, 0u);
}
// Two words in one poll loop.
__device__ __forceinline__ void boundary_wait2(const OpCtx& ctx, int word_a, int word_b) {
  boundary_wait_impl(ctx, sync_word(ctx, word_a), word_target(ctx.epoch), sync_word(ctx, word_b),
                     word_target(ctx.epoch));
}

// ---- S barrier (B+0x44) ---------------------------------------
// Participants: CTAs c < n_active (all 256 consumer threads call).  FIRST a group barrier: it
// orders the other 255 threads' preceding stores / REDs (CTA0's H clear, every thread's H[q][tid]
// += hist) before thread 0's release atom -- without it S#1/S#2 would publish only thread 0's own
// work.  Then thread 0 of CTA 0 adds 0x80000001 - n_active, thread 0 of every other participant
// adds 1 (sum per generation = 0x80000000: the sign bit flips); the returned old value is broadcast
// to warp 0, which spins until (v ^ old) < 0, then an acquire fence and the group barrier. Requires
// the word to hold 0 or 0x80000000 at generation start (arena zeroed at init; exactly two
// generations per sparse launch, none per dense launch, so it self-resets).  `is_cta0` must be
// exactly one participant (CTA 0 is always active since n_active >= 1 when sparse).
__device__ __forceinline__ void sense_barrier(const OpCtx& ctx, int n_active, bool is_cta0) {
  group_sync(ctx);
  uint32_t* word = sync_word(ctx, kSyncS);
  uint32_t old = 0;
  if (ctx.tid == 0) {
    const uint32_t add = is_cta0 ? (kSenseCta0Base - static_cast<uint32_t>(n_active)) : 1u;
    old = atom_acq_rel_gpu_add(word, add);
  }
  if (ctx.warp == 0) {
    old = __shfl_sync(0xffffffffu, old, 0);
    const bool poller = ctx.lane == 0;
    if (poller) {
      while (static_cast<int32_t>(load_relaxed_gpu(word) ^ old) >= 0) spin_backoff();
      fence_acquire_gpu();
    }
  }
  group_sync(ctx);
}
__device__ __forceinline__ void sense_barrier_presensed(const OpCtx& ctx, int n_active,
                                                        bool is_cta0, uint32_t sense_word) {
  group_sync(ctx);
  if (ctx.tid == 0) {
    uint32_t* word = sync_word(ctx, kSyncS);
    red_release_gpu_add(word, is_cta0 ? (kSenseCta0Base - static_cast<uint32_t>(n_active)) : 1u);
    while (static_cast<int32_t>(load_relaxed_gpu(word) ^ sense_word) >= 0) spin_backoff();
    fence_acquire_gpu();
  }
  group_sync(ctx);
}

// ---- H block clear -------------------------
// plain stores by all 256 consumer threads; same bytes, published by S#1's group_sync + release
// atom.
__device__ __forceinline__ void hist_block_clear(const OpCtx& ctx, uint32_t* hist_blk) {
  for (int i = ctx.tid; i < kSeqLen * kHistBlockDwords; i += kConsumerThreads) hist_blk[i] = 0u;
}

}  // namespace flashinfer::glm5_tp8::indexer_q4
// The 4-slot page ring shared by the prefetch warps (producers, warps 8..11) and the consumer op
// bodies.
//   slots      4 x 0x9000 at smem::kRing; item k -> slot k & 3
//   full[s]    mbarrier count 4 (+ tx bytes): the 4 prefetch warps' lane 0 arrive;
//              thread 256 alone registers the tx bytes and issues ONE cp.async.bulk for a weight
//              page, or expect_tx 0x8000 + TWO cp.async.bulk.tensor.2d for a K tile
//   empty[s]   mbarrier count 256: every consumer thread arrives right after its last shared load
//              of the page (early release)
//   parity     consumers wait full with ((item>>2)&1); producers wait empty with that ^ 1 (the
//              first round passes)
//   waits      producers wait on no device word; the optional kFlagPrefetchWaitC2 makes thread
//              256 poll C2 before the tile that covers cur_pos..+3
// The persistent kernel's only block-wide barrier is the prologue __syncthreads after
// ring_init_barriers.  The prefetch warps (prefetch_main) and the consumer op bodies enumerate
// exactly the same items in the same order (the item counter is monotonic across ops):
//   items 0..5: P1  CTA c < 68: 6 x 0x8080 fp8 pages
//                      q_a_indexer_wk + c*0x30300 + i*0x8080
//                    CTA 68..71: 6 x 0x8000 bf16 pages
//                      indexer_weights_proj + rowblock(c)*0x30000 + i*0x8000
//                    CTA 72..127: none (their item counter stays 0)
//   next 2     : P2  every Group1 CTA: 2 x 0x8080 pages indexer_wq_b + c*0x10100 + i*0x8080
//   next n     : P4  (sparse only) n = k_tiles_of_cta(cur_pos, c) K tiles of 2 x 0x4000 B TMA boxes
//                    rows [i*16384 + c*128, +128), dims [0,64) -> slot+0, [64,128) -> slot+0x4000
// Group2a's slot-0 feed is one extra item for CTA 128 only (g2a_prefetch_feed).  A CTA that
// runs no op must get NO items for it (liveness: its prefetchers would block on empty[] forever).

namespace flashinfer::glm5_tp8::indexer_q4 {

// P1 page `index` of CTA `cta` (Group1 index).  Returns false past the end / for CTAs without P1
// pages.
__device__ __forceinline__ bool p1_page(const CtxRow0& row, int cta, int index, const uint8_t** ptr,
                                        uint32_t* bytes) {
  if (cta < kQaIndexerWkCtas) {
    if (index >= kQaIndexerWkPages) return false;
    *ptr = row.q_a_indexer_wk_pages + static_cast<size_t>(cta) * kQaIndexerWkCtaStride +
           static_cast<size_t>(index) * kPageBytes8080;
    *bytes = kPageBytes8080;
    return true;
  }
  if (cta < kP1AllCtas) {
    if (index >= kIndexerWeightsProjPages) return false;
    *ptr = row.indexer_weights_proj_pages +
           static_cast<size_t>(indexer_weights_proj_rowblock(cta)) *
               kIndexerWeightsProjRowBlockStride +
           static_cast<size_t>(index) * kIndexerWeightsProjPageBytes;
    *bytes = kIndexerWeightsProjPageBytes;
    return true;
  }
  return false;
}
__device__ __forceinline__ int p1_page_count(int cta) {
  return cta < kP1AllCtas ? kQaIndexerWkPages : 0;
}  // 6 for both kinds

// P2 page `index` of CTA `cta`.
__device__ __forceinline__ bool p2_page(const CtxRow1& row, int cta, int index, const uint8_t** ptr,
                                        uint32_t* bytes) {
  if (cta >= kIndexerWqBCtas || index >= kIndexerWqBPages) return false;
  *ptr = row.indexer_wq_b_pages + static_cast<size_t>(cta) * kIndexerWqBCtaStride +
         static_cast<size_t>(index) * kPageBytes8080;
  *bytes = kPageBytes8080;
  return true;
}
__device__ __forceinline__ int p2_page_count(int cta) {
  return cta < kIndexerWqBCtas ? kIndexerWqBPages : 0;
}

// K tile i of CTA c covers kv rows [i*16384 + c*128, +128).
__device__ __forceinline__ int k_tile_base(int cta, int tile) {
  return tile * kKTileStrideRows + cta * kKTileRows;
}
// The tile holds the current 4 rows (cur_pos..cur_pos+3) iff tile_base < L && tile_base + 128 >
// cur_pos:
__device__ __forceinline__ bool k_tile_covers_current(int cur_pos, int tile_base) {
  return tile_base < total_len(cur_pos) && tile_base + kKTileRows > cur_pos;
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

__device__ __forceinline__ void ring_init_barriers(uint8_t* smem) {
  if (threadIdx.x < kRingSlots) {
    mbar_init(smem_u32(smem + smem::kFullBar) + threadIdx.x * 8, kFullArrivals);
    mbar_init(smem_u32(smem + smem::kEmptyBar) + threadIdx.x * 8, kEmptyArrivals);
    mbar_fence_init();
  }
}
__device__ __forceinline__ uint32_t ring_full_bar(const OpCtx& ctx, uint32_t slot) {
  return ctx.smem_s + smem::kFullBar + slot * 8;
}
__device__ __forceinline__ uint32_t ring_empty_bar(const OpCtx& ctx, uint32_t slot) {
  return ctx.smem_s + smem::kEmptyBar + slot * 8;
}
__device__ __forceinline__ uint32_t ring_slot_s(const OpCtx& ctx, uint32_t slot) {
  return ctx.smem_s + smem::kRing + slot * kRingSlotBytes;
}

// ---- consumer side (all 256 consumer threads call both, in item order)
// ------------------------------------ acquire: wait full[item&3] parity (item>>2)&1; returns the
// slot's shared address.
__device__ __forceinline__ uint32_t ring_acquire(const OpCtx& ctx) {
  const uint32_t slot = ctx.item & (kRingSlots - 1);
  mbar_wait_parity(ring_full_bar(ctx, slot), (ctx.item >> 2) & 1u);
  return ring_slot_s(ctx, slot);
}
// release: arrive on empty[item&3], advance the item counter.
__device__ __forceinline__ void ring_release(OpCtx& ctx) {
  const uint32_t slot = ctx.item & (kRingSlots - 1);
  mbar_arrive(ring_empty_bar(ctx, slot));
  ++ctx.item;
}
// Drain n items without reading them (stub op bodies / null mode keep the handshake live).
__device__ __forceinline__ void ring_drain(OpCtx& ctx, int n) {
  for (int i = 0; i < n; ++i) {
    ring_acquire(ctx);
    ring_release(ctx);
  }
}

// ---- producer side (the 128 prefetch threads; ptid = tid - 256)
// ----------------------------------------------
struct PrefetchCtx {
  int cta;          // Group1 CTA index
  int ptid;         // 0..127
  int pwarp;        // 0..3
  int plane;        // 0..31
  uint32_t smem_s;  // shared-window address of the smem base
  uint32_t item;    // monotonic item counter (== consumers' order)
  uint32_t epoch;   // loaded epoch (smem::kEpochWord): target of the optional C2 wait
  const uint8_t* smem;
  const IndexerArgs* args;
};
__device__ __forceinline__ uint32_t pf_full_bar(const PrefetchCtx& p, uint32_t slot) {
  return p.smem_s + smem::kFullBar + slot * 8;
}
__device__ __forceinline__ uint32_t pf_empty_bar(const PrefetchCtx& p, uint32_t slot) {
  return p.smem_s + smem::kEmptyBar + slot * 8;
}
__device__ __forceinline__ uint32_t pf_slot_s(const PrefetchCtx& p, uint32_t slot) {
  return p.smem_s + smem::kRing + slot * kRingSlotBytes;
}

// All 128 prefetch threads wait for the slot to be free (empty parity flipped; first round passes).
__device__ __forceinline__ uint32_t pf_wait_slot(const PrefetchCtx& p) {
  const uint32_t slot = p.item & (kRingSlots - 1);
  mbar_wait_parity(pf_empty_bar(p, slot), ((p.item >> 2) & 1u) ^ 1u);
  return slot;
}
// The 4 lane-0 arrivals that complete full[slot] together with the tx bytes.
__device__ __forceinline__ void pf_arrive_full(const PrefetchCtx& p, uint32_t slot) {
  if (p.plane == 0) mbar_arrive(pf_full_bar(p, slot));
}

// One weight page: thread 256 (ptid 0) registers the bytes (0x8080 / 0x8000) and issues one bulk
// copy; all four warps' lane 0 arrive.  The L2 hint is kL2PolicyEvictNormal (the K-tile TMA
// constant); cache hint only, no correctness effect.
__device__ __forceinline__ void pf_issue_page(PrefetchCtx& p, const uint8_t* src, uint32_t bytes) {
  const uint32_t slot = pf_wait_slot(p);
  const uint32_t full = pf_full_bar(p, slot);
  if (p.ptid == 0) {
    mbar_expect_tx(full, bytes);
    bulk_copy_g2s_hint(pf_slot_s(p, slot), src, bytes, full, l2_policy_stream());
  }
  pf_arrive_full(p, slot);
  ++p.item;
}
// One K tile: thread 256 registers 0x8000 bytes and issues the two 64 x 128 boxes (dims [0,64) and
// [64,128)) of kv rows [tile_base, +128) through the indexer_k_cache tensor map (ctx row 4 +0x40);
// lane 0s arrive.
__device__ __forceinline__ void pf_issue_k_tile(PrefetchCtx& p, const void* k_tmap, int tile_base) {
  const uint32_t slot = pf_wait_slot(p);
  const uint32_t full = pf_full_bar(p, slot);
  if (p.ptid == 0) {
    mbar_expect_tx(full, kKTileBytes);
    const uint32_t dst = pf_slot_s(p, slot);
    const uint64_t policy = l2_policy_stream();
    tma_load_2d_hint(dst, k_tmap, 0, tile_base, full, policy);
    tma_load_2d_hint(dst + kKTileBoxBytes, k_tmap, kIdxDim / 2, tile_base, full, policy);
  }
  pf_arrive_full(p, slot);
  ++p.item;
}

// P1 pages (6 for CTAs 0..71, none for 72..127), then P2 pages (2 for every Group1 CTA).
__device__ __forceinline__ void pf_stream_p1(PrefetchCtx& p, int first_page = 0) {
  const CtxRow0& row = ctx_p1(p.smem);
  const uint8_t* src = nullptr;
  uint32_t bytes = 0;
  for (int i = first_page; p1_page(row, p.cta, i, &src, &bytes); ++i) pf_issue_page(p, src, bytes);
}

// The first kRingSlots P1 pages of a P1 CTA are issued at kernel entry by thread kConsumerThreads,
// which also initializes the ring barriers (the prologue's ring_init_barriers is skipped for these
// CTAs).  The page addresses come straight from the global ctx table, so the weight stream starts
// while the ctx table is still being copied to shared memory.  Each pre-issued full[i] receives its
// four arrivals and the page's tx bytes here; the prefetch warps resume the normal stream at page
// kRingSlots.
constexpr int kP1Preissued = kRingSlots;
static_assert(kP1Preissued <= kQaIndexerWkPages && kP1Preissued <= kIndexerWeightsProjPages,
              "pre-issued pages exist for every P1 CTA");
__device__ __forceinline__ bool p1_preissue_enabled(const IndexerArgs& args, int cta) {
  return cta < kP1AllCtas && !has_inline_ctx(args);
}
__device__ __forceinline__ void prefetch_l2_bytes(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// Entry-time L2 warm of the rank-0 front end: the P1 RMSNorm operands (x, gamma_x), the P2 gamma_q
// and the P3 RoPE table are fetched by CTA 0 ahead of the weight stream (their plain loads
// otherwise queue behind ~26 MB of page traffic), and every P1 CTA warms its two non-preissued P1
// pages and its two P2 pages, so the ring refills after the first pages hit L2.
__device__ __forceinline__ void p1_entry_warm(const CtxRow0& row, int cta) {
  if (cta == 0) {
    prefetch_l2_bytes(row.x, kSeqLen * kHidden * 2);
    prefetch_l2_bytes(row.gamma_x, kHidden * 4);
  }
}

// After the prologue (ctx rows in shared memory, no extra global round trip on the prologue path):
// warm the P2 operands that the ring only reaches after the P1 pages (gamma_q, the P3 RoPE table, a
// P1 CTA's two indexer.wq_b pages) and the K / Q tensor maps.
__device__ __forceinline__ void prefetch_late_warm(const IndexerArgs& args, const uint8_t* smem,
                                                   int cta) {
  if (has_inline_ctx(args) || cta >= kGroup1Ctas) return;
  if (cta == 0) {
    prefetch_l2_bytes(ctx_p2(smem).gamma_q, kQaDim * 4);
    const float* rope = ctx_p3(smem).rope_cos_sin;
    if (rope) rope += size_t(*reinterpret_cast<const int32_t*>(smem + smem::kCurPosWord)) * 64;
    if (rope != nullptr) prefetch_l2_bytes(rope, kSeqLen * 64 * 4);
  }
  const void* k_tmap = ctx_score(smem).k_tmap;
  const void* q_tmap = ctx_score(smem).q_tmap;
  if (k_tmap != nullptr) prefetch_tensormap(k_tmap);
  if (q_tmap != nullptr) prefetch_tensormap(q_tmap);
}

__device__ __forceinline__ void p1_preissue_pages(const IndexerArgs& args, uint8_t* smem, int cta) {
  if (static_cast<int>(threadIdx.x) != kConsumerThreads) return;
  const uint32_t full_s = smem_u32(smem + smem::kFullBar);
  const uint32_t empty_s = smem_u32(smem + smem::kEmptyBar);
#pragma unroll
  for (int i = 0; i < kRingSlots; ++i) {
    mbar_init(full_s + i * 8, kFullArrivals);
    mbar_init(empty_s + i * 8, kEmptyArrivals);
  }
  mbar_fence_init();
  CtxRow0 row;
  row.x = args.ctx_table->p1.x;
  row.gamma_x = args.ctx_table->p1.gamma_x;
  row.q_a_indexer_wk_pages = args.ctx_table->p1.q_a_indexer_wk_pages;
  row.indexer_weights_proj_pages = args.ctx_table->p1.indexer_weights_proj_pages;
  const uint32_t ring_s = smem_u32(smem + smem::kRing);
#pragma unroll
  for (int i = 0; i < kP1Preissued; ++i) {
    const uint8_t* src = nullptr;
    uint32_t bytes = 0;
    p1_page(row, cta, i, &src, &bytes);
    mbar_arrive_expect_tx(full_s + i * 8, bytes);
    mbar_arrive_count(full_s + i * 8, kFullArrivals - 1);
    bulk_copy_g2s_hint(ring_s + i * kRingSlotBytes, src, bytes, full_s + i * 8, l2_policy_stream());
  }
  p1_entry_warm(row, cta);
}
__device__ __forceinline__ void pf_stream_p2(PrefetchCtx& p) {
  const CtxRow1& row = ctx_p2(p.smem);
  const uint8_t* src = nullptr;
  uint32_t bytes = 0;
  for (int i = 0; p2_page(row, p.cta, i, &src, &bytes); ++i) pf_issue_page(p, src, bytes);
}
// K tiles (sparse only; n = 0 -> nothing).
template <bool SparseComputeOnly>
__device__ __forceinline__ void pf_stream_k_tiles(PrefetchCtx& p, int cur_pos,
                                                  bool wait_c2 = false) {
  if constexpr (!SparseComputeOnly) {
    if (!is_sparse(cur_pos)) return;
  }
  const int n = k_tiles_of_cta(cur_pos, p.cta);
  if (n == 0) return;
  const CtxRow4& row = ctx_score(p.smem);
  const void* k_tmap;
  if constexpr (SparseComputeOnly) {
    k_tmap = static_cast<const void*>(&p.args->k_tmap);
  } else {
    k_tmap = has_inline_ctx(*p.args) ? static_cast<const void*>(&p.args->k_tmap) : row.k_tmap;
  }
  for (int i = 0; i < n; ++i) {
    const int base = k_tile_base(p.cta, i);
    if ((wait_c2 || (p.args->flags & kFlagPrefetchWaitC2)) && p.ptid == 0 &&
        k_tile_covers_current(cur_pos, base)) {
      const uint32_t* c2 = reinterpret_cast<const uint32_t*>(
          reinterpret_cast<const uint8_t*>(p.args->sync_words) + kSyncC2);
      while (!word_reached(c2, word_target(p.epoch))) spin_backoff();
      fence_acquire_gpu();
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    pf_issue_k_tile(p, k_tmap, base);
  }
}

__device__ __forceinline__ void g2a_prefetch_feed(PrefetchCtx& p) {
  const CtxRow2& row = ctx_g2a(p.smem);
  const uint32_t slot = pf_wait_slot(p);  // item 0: passes
  const uint32_t slot_s = pf_slot_s(p, slot);
  const int c = p.ptid;
  if (c < 32) {
    ldgsts_16(slot_s + (smem::kG2aGamma - smem::kRing) + c * 16,
              reinterpret_cast<const uint8_t*>(row.ln_gamma) + c * 16);
  } else if (c < 64) {
    ldgsts_16(slot_s + (smem::kG2aBeta - smem::kRing) + (c - 32) * 16,
              reinterpret_cast<const uint8_t*>(row.ln_beta) + (c - 32) * 16);
  } else {
    ldgsts_16(slot_s + (smem::kG2aCosSin - smem::kRing) + (c - 64) * 16,
              reinterpret_cast<const uint8_t*>(
                  row.rope_cos_sin +
                  size_t(*reinterpret_cast<const int32_t*>(p.smem + smem::kCurPosWord)) * 64) +
                  (c - 64) * 16);
  }
  ldgsts_commit();
  ldgsts_wait_all();
  __syncwarp();
  pf_arrive_full(p, slot);
  ++p.item;
}

// Per-CTA prefetch schedule of the persistent kernel (Group1): P1 -> P2 -> [K tiles].
template <bool CompactDense, bool SparseComputeOnly>
__device__ __forceinline__ void prefetch_main(PrefetchCtx& p, int cur_pos) {
  // Fully dense steps do not expose q_a/IQ/logits/ids. Only indexer.wk (P1 CTAs
  // 64..67) feeds the required Group2a cache-row update.
  if constexpr (CompactDense) {
    if (p.cta >= 64 && p.cta < kP1Ctas) pf_stream_p1(p);
    return;
  }
  // kModeNull still streams every page: the ring is part of the protocol floor (consumers drain).
  if constexpr (!SparseComputeOnly) {
    if (p.ptid == 0) prefetch_late_warm(*p.args, p.smem, p.cta);
  }
  if (!SparseComputeOnly && p1_preissue_enabled(*p.args, p.cta)) {
    p.item = kP1Preissued;
    pf_stream_p1(p, kP1Preissued);
  } else {
    pf_stream_p1(p);
  }
  pf_stream_p2(p);
  if constexpr (!SparseComputeOnly) {
    // P2-only CTAs reach their K tiles at kernel entry; request them once C0 is reached so these
    // index-key reads stay out of P1's weight stream from HBM (the front of the full layer's
    // critical path).  P4 starts after C3, well behind C0.
    if (p.cta >= kP1AllCtas && p.ptid == 0 && p.args->mode == kModePersistent &&
        is_sparse(cur_pos)) {
      const uint32_t* c0 = reinterpret_cast<const uint32_t*>(
          reinterpret_cast<const uint8_t*>(p.args->sync_words) + kSyncC0);
      while (!word_reached(c0, word_target(p.epoch))) spin_backoff();
    }
  }
  pf_stream_k_tiles<SparseComputeOnly>(p, cur_pos, !SparseComputeOnly);
}

__device__ __forceinline__ PrefetchCtx make_prefetch_ctx(const IndexerArgs& args, int cta,
                                                         uint8_t* smem, uint32_t epoch) {
  PrefetchCtx p;
  p.cta = cta;
  p.ptid = static_cast<int>(threadIdx.x) - kConsumerThreads;
  p.pwarp = p.ptid >> 5;
  p.plane = p.ptid & 31;
  p.smem_s = smem_u32(smem);
  p.item = 0;
  p.epoch = epoch;
  p.smem = smem;
  p.args = &args;
  return p;
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// ---- P1-local helpers (prefixed op_p1_; nothing here is shared with other ops)
// ---------------------------------------------
constexpr int kP1Chunks = 3;  // 3 x 2048 columns per thread
constexpr int kP1ChunkCols = 2048;
constexpr uint32_t kP1ChunkBytes = 0x1000;  // 2048 bf16
constexpr uint32_t kP1RmsTokStride = 0x20;  // kRmsPartial + tok*0x20 + warp*4
constexpr int kP1RowsPerCta = 32;           // two 16-row MMA tiles (rg 0/1)
constexpr int kP1bRowsPerCta = 16;          // one 16-row tile
constexpr int kP1bTokens = 2;
constexpr uint32_t kP1RedTokStride = 0x80;   // kP1Red: [warp 0x404][tok 8 x 0x80][row 32 x 4]
constexpr uint32_t kP1bRedTokStride = 0x40;  // kP1bRed: [warp 0x84][tok 2 x 0x40][row 16 x 4]
constexpr float kP1RmsEps = 9.9999997473787516356e-06f;  // 0x3727c5ac
constexpr float kP1RmsDenominator = 6144.0f;
static_assert(kP1Chunks * kP1ChunkCols == kHidden && kP1ChunkBytes == kP1ChunkCols * 2,
              "x chunking");
static_assert(kSeqLen * kP1RmsTokStride <= 0x80,
              "RMS partials fit the 0x80 B region (smem::kRmsPartial)");
static_assert(smem::kP1RedWarpStride == 8 * kP1RedTokStride + 4 &&
                  smem::kP1bRedWarpStride == kP1bTokens * kP1bRedTokStride + 4,
              "K-split partial geometry");

// 16 B shared load by shared-window address (P1b A fragment, kP1bOut readback).  asm volatile keeps
// it ordered before the slot release like the other shared loads.
__device__ __forceinline__ void op_p1_lds_u32x4(uint32_t address, uint32_t (&v)[4]) {
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(address));
}

// One bf16x8 word quad of x folded into the running fp32 sum of squares: for word q (elements 2q =
// low half, 2q+1 = high half) p = fma(e_even, e_even, e_odd * e_odd); s = s + p (sequential across
// the 12 words).
__device__ __forceinline__ float op_p1_sumsq_words(float s, const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float e = bf16_pair_low(w[q]);
    const float o = bf16_pair_high(w[q]);
    s = __fadd_rn(s, __fmaf_rn(e, e, __fmul_rn(o, o)));
  }
  return s;
}

// Shuffle-down 16,8,4,2,1 tree with s = s + shfl; lane 0 ends with the warp total.
__device__ __forceinline__ float op_p1_warp_sum_down(float s) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
  return s;
}

// rstd of one token from the 8 warp partials (every thread, after the barrier): ((0 + w0) + .. +
// w7), div.rn(sum, 6144), + 1e-5, rsqrtf (= rsqrt.approx.f32 incl. the denormal guard).
__device__ __forceinline__ float op_p1_rstd_from_partials(const uint8_t* smem, int token) {
  const float4* partial =
      reinterpret_cast<const float4*>(smem + smem::kRmsPartial + token * kP1RmsTokStride);
  const float4 a = partial[0];
  const float4 b = partial[1];
  float sum = __fadd_rn(0.0f, a.x);
  sum = __fadd_rn(sum, a.y);
  sum = __fadd_rn(sum, a.z);
  sum = __fadd_rn(sum, a.w);
  sum = __fadd_rn(sum, b.x);
  sum = __fadd_rn(sum, b.y);
  sum = __fadd_rn(sum, b.z);
  sum = __fadd_rn(sum, b.w);
  const float mean = __fdiv_rn(sum, kP1RmsDenominator);
  return rsqrtf(__fadd_rn(mean, kP1RmsEps));
}

__device__ __forceinline__ uint4 op_p1_xnorm_words(const uint4& x, const float4& g_lo,
                                                   const float4& g_hi, float rstd) {
  const uint32_t w[4] = {x.x, x.y, x.z, x.w};
  const float g[8] = {g_lo.x, g_lo.y, g_lo.z, g_lo.w, g_hi.x, g_hi.y, g_hi.z, g_hi.w};
  uint32_t y[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float ye = __fmul_rn(bf16_pair_low(w[q]), __fmul_rn(g[2 * q], rstd));
    const float yo = __fmul_rn(bf16_pair_high(w[q]), __fmul_rn(g[2 * q + 1], rstd));
    y[q] = pack_bf16_pair(ye, yo);
  }
  return make_uint4(y[0], y[1], y[2], y[3]);
}

// x-RMSNorm of `tokens` rows (P1: 4 rows tok 0..3; P1b: 2 rows tok_base, +1) into smem x_norm with
// row stride `xnorm_tok_stride`. Every consumer thread owns columns {chunk*2048 + 8*tid .. +8:
// chunk 0..2} of every row. Ends with the barrier that publishes x_norm to the page loop. Two
// barriers in total.
template <int Tokens, bool BroadcastRstd = false>
__device__ __forceinline__ void op_p1_rmsnorm_x_to_smem(OpCtx& ctx, const CtxRow0& row,
                                                        int tok_base, uint32_t xnorm_tok_stride) {
  static_assert(Tokens == kSeqLen || Tokens == kP1bTokens, "P1 normalizes 4 rows, P1b 2 rows");
  const int tid = ctx.tid;
  // 1. x rows into registers.
  uint4 xv[Tokens][kP1Chunks];
#pragma unroll
  for (int t = 0; t < Tokens; ++t) {
    const uint4* src =
        reinterpret_cast<const uint4*>(row.x + static_cast<size_t>(tok_base + t) * kHidden) + tid;
#pragma unroll
    for (int c = 0; c < kP1Chunks; ++c) xv[t][c] = src[c * (kP1ChunkCols / 8)];  // +c*0x1000 B
  }
  float sums[Tokens];
#pragma unroll
  for (int t = 0; t < Tokens; ++t) {
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < kP1Chunks; ++c) s = op_p1_sumsq_words(s, xv[t][c]);
    sums[t] = op_p1_warp_sum_down(s);
  }
  if (ctx.lane == 0) {  // kRmsPartial + warp*4 (+ tok*0x20)
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
      *smem_at<float>(ctx.smem, smem::kRmsPartial + t * kP1RmsTokStride + ctx.warp * 4) = sums[t];
    }
  }
  float4 g_lo[kP1Chunks];
  float4 g_hi[kP1Chunks];
#pragma unroll
  for (int c = 0; c < kP1Chunks; ++c) {
    const float4* gp = reinterpret_cast<const float4*>(row.gamma_x + c * kP1ChunkCols + tid * 8);
    g_lo[c] = gp[0];
    g_hi[c] = gp[1];
  }
  consumer_sync();
  // 3. rstd per token, then x_norm rows -> smem.
  float rstd[Tokens];
  if constexpr (BroadcastRstd) {
    // Compact dense launches use only four indexer.wk CTAs.  Compute the four
    // identical token chains once in warp 0 and broadcast one float4.
    if (ctx.warp == 0 && ctx.lane < Tokens) {
      *smem_at<float>(ctx.smem, smem::kRmsPartial + ctx.lane * 4) =
          op_p1_rstd_from_partials(ctx.smem, ctx.lane);
    }
    consumer_sync();
    const float4 packed = *smem_at<const float4>(ctx.smem, smem::kRmsPartial);
    rstd[0] = packed.x;
    rstd[1] = packed.y;
    if constexpr (Tokens == kSeqLen) {
      rstd[2] = packed.z;
      rstd[3] = packed.w;
    }
  } else {
#pragma unroll
    for (int t = 0; t < Tokens; ++t) rstd[t] = op_p1_rstd_from_partials(ctx.smem, t);
  }
#pragma unroll
  for (int t = 0; t < Tokens; ++t) {
#pragma unroll
    for (int c = 0; c < kP1Chunks; ++c) {
      uint4* dst = smem_at<uint4>(ctx.smem,
                                  smem::kAct + t * xnorm_tok_stride + c * kP1ChunkBytes + tid * 16);
      *dst = op_p1_xnorm_words(xv[t][c], g_lo[c], g_hi[c], rstd[t]);
    }
  }
  consumer_sync();
}

// B fragment of one K16 step: lane l < `active_lanes` holds x_norm[tok = l>>2][k0 + 2*(l&3) +
// {0,1}] and [k0 + 8 + ...]; other lanes supply the zero N columns.
__device__ __forceinline__ void op_p1_lds_b_fragment(uint32_t b_address, bool active,
                                                     uint32_t (&b)[2]) {
  b[0] = 0u;
  b[1] = 0u;
  if (active) {
    b[0] = lds_u32(b_address);
    b[1] = lds_u32(b_address + 16u);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// P1: x-RMSNorm + W8A16 GEMV, CTAs 0..67
// ---------------------------------------------------------------------------------------------------------------------------
template <bool StaticMath, bool BroadcastRstd = false>
static __device__ void op_p1_rmsnorm_gemv_q_a_indexer_wk(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  const int pages = p1_page_count(ctx.cta);  // 6 (kQaIndexerWkPages)
  if constexpr (!StaticMath) {
    if (ctx.null_math) {
      ring_drain(ctx, pages);
      return;
    }
  }
  const CtxRow0& row = ctx_p1(ctx.smem);
  const int tid = ctx.tid;
  const int warp = ctx.warp;
  const int lane = ctx.lane;

  op_p1_rmsnorm_x_to_smem<kSeqLen, BroadcastRstd>(ctx, row, 0, smem::kP1XnormTokStride);

  // ---- 6 pages: per (page, warp) two 16-row chains x 8 K16 steps from zero, scale FMA into 8
  // running sums ----
  float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
  const bool b_active = lane < 16;
  // x_norm column k of token (lane>>2): kAct + (lane>>2)*0x3010 + k*2; per (page, warp, step): k =
  // page*1024 + warp*128 + step*16 + 2*(lane&3)
  const uint32_t b_lane =
      ctx.smem_s + smem::kAct + static_cast<uint32_t>(lane >> 2) * smem::kP1XnormTokStride +
      static_cast<uint32_t>(lane & 3) * 4u + static_cast<uint32_t>(warp) * 0x100u;
  for (int page = 0; page < pages; ++page) {
    const uint32_t slot = ring_acquire(ctx);  // wait full[item&3], parity (item>>2)&1
    const uint32_t a_lane =
        slot + static_cast<uint32_t>(warp) * 0x100u + static_cast<uint32_t>(lane) * 8u;
    const uint32_t b_page = b_lane + static_cast<uint32_t>(page) * 0x800u;  // item*0x400 u16
    HmmaAcc d[2] = {{{0.0f, 0.0f, 0.0f, 0.0f}},
                    {{0.0f, 0.0f, 0.0f, 0.0f}}};  // accumulators start from zero
    float scale = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint32_t a0_lo, a0_hi, a1_lo, a1_hi;
      lds_u32x2(a_lane + static_cast<uint32_t>(k) * 0x800u, a0_lo, a0_hi);
      lds_u32x2(a_lane + 0x4000u + static_cast<uint32_t>(k) * 0x800u, a1_lo, a1_hi);
      uint32_t b[2];
      op_p1_lds_b_fragment(b_page + static_cast<uint32_t>(k) * 0x20u, b_active, b);
      if (k == 7) {
        // The slot's last reads are the 16th weight load and the warp's block scale; the empty[]
        // arrive follows them and precedes the tail MMAs / FMAs: early release, never before the
        // scale.
        scale = lds_f32(slot + kPageScaleOffset + static_cast<uint32_t>(warp) * 4u);
        ring_release(ctx);
      }
      uint32_t a[4];
      fp8x8_to_bf16_fragment(a0_lo, a0_hi, a);
      hmma_16816_bf16(d[0], a, b);
      fp8x8_to_bf16_fragment(a1_lo, a1_hi, a);
      hmma_16816_bf16(d[1], a, b);  // rg1 chain
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rg][i] = __fmaf_rn(scale, d[rg].value[i], acc[rg][i]);
    }
  }

  // ---- K-split fold: 8 warp partials [warp][tok][row] -> ordered add by tid < 128 -> bf16 -> smem
  // out -> global ----
  {
    // accumulator i of rg: row = rg*16 + (lane>>2) + 8*(i>>1), token column = 2*(lane&3) + (i&1)
    // (tokens 4..7 = zero columns of lanes with (lane&3) >= 2, stored into the upper half of the
    // warp's 0x400 block
    uint8_t* red = ctx.smem + smem::kP1Red + warp * smem::kP1RedWarpStride;
    if ((lane & 3) < 2) {
#pragma unroll
      for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int tokcol = 2 * (lane & 3) + (i & 1);
          const int r = rg * 16 + (lane >> 2) + 8 * (i >> 1);
          *reinterpret_cast<float*>(red + tokcol * kP1RedTokStride + r * 4) = acc[rg][i];
        }
      }
    }
  }
  consumer_sync();
  if (tid < kSeqLen * kP1RowsPerCta) {
    // element tid = tok*32 + row: partials at kP1Red + w*0x404 + tid*4, 0 + p0 + p1 + ... + p7 in
    // order, bf16 pack
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      sum = __fadd_rn(sum, *smem_at<const float>(
                               ctx.smem, smem::kP1Red + w * smem::kP1RedWarpStride + tid * 4));
    }
    const __nv_bfloat16 value = to_bf16(sum);
    const bool is_ki = ctx.cta >= kQaDim / kP1RowsPerCta;  // CTAs 64..67
    __nv_bfloat16* out = is_ki ? row.ki_out : row.q_a_out;
    const int tok_stride = is_ki ? kIndexerWkRows : kQaDim;  // 0x80 / 0x800 elements
    const int block = is_ki ? ctx.cta - kQaDim / kP1RowsPerCta : ctx.cta;
    out[warp * tok_stride + block * kP1RowsPerCta + lane] = value;
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// P1b: bf16 indexer.weights_proj GEMV, CTAs 68..71: row block
// indexer_weights_proj_rowblock(c) x tokens indexer_weights_proj_token_base(c), +1
// ---------------------------------------------------------------------------------------------------------------------------
template <bool StaticMath>
static __device__ void op_p1b_gemv_indexer_weights_proj(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  const int pages = p1_page_count(ctx.cta);  // 6 (kIndexerWeightsProjPages)
  if constexpr (!StaticMath) {
    if (ctx.null_math) {
      ring_drain(ctx, pages);
      return;
    }
  }
  const CtxRow0& row = ctx_p1(ctx.smem);
  const int tid = ctx.tid;
  const int warp = ctx.warp;
  const int lane = ctx.lane;
  const int cta = ctx.cta;  // Group1 index 68..71
  const int tok_base = indexer_weights_proj_token_base(cta);
  const int rowblock = indexer_weights_proj_rowblock(cta);

  op_p1_rmsnorm_x_to_smem<kP1bTokens>(ctx, row, tok_base, smem::kP1bXnormTokStride);

  // ---- 6 pages x 8 K16 steps, ONE accumulator chain per warp (no scales)  ----
  HmmaAcc d = {{0.0f, 0.0f, 0.0f, 0.0f}};
  const bool b_active = lane < 8;
  const uint32_t b_lane =
      ctx.smem_s + smem::kAct + static_cast<uint32_t>(lane >> 2) * smem::kP1bXnormTokStride +
      static_cast<uint32_t>(lane & 3) * 4u + static_cast<uint32_t>(warp) * 0x100u;
  for (int page = 0; page < pages; ++page) {
    const uint32_t slot = ring_acquire(ctx);  // wait full[item&3]
    const uint32_t a_lane =
        slot + static_cast<uint32_t>(warp) * 0x1000u + static_cast<uint32_t>(lane) * 16u;
    const uint32_t b_page = b_lane + static_cast<uint32_t>(page) * 0x800u;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint32_t a[4];
      op_p1_lds_u32x4(a_lane + static_cast<uint32_t>(k) * 0x200u, a);
      uint32_t b[2];
      op_p1_lds_b_fragment(b_page + static_cast<uint32_t>(k) * 0x20u, b_active, b);
      if (k == 7) ring_release(ctx);
      hmma_16816_bf16(d, a, b);
    }
  }

  // ---- fold: lanes with (lane&3) == 0 hold tokens 0/1 (d0,d2 -> tok 0 rows g, g+8; d1,d3 -> tok
  // 1) -> kP1bRed [warp][tok][row] ----
  if ((lane & 3) == 0) {
    uint8_t* red = ctx.smem + smem::kP1bRed + warp * smem::kP1bRedWarpStride;
    const int g = lane >> 2;
    *reinterpret_cast<float*>(red + 0 * kP1bRedTokStride + g * 4) = d.value[0];
    *reinterpret_cast<float*>(red + 0 * kP1bRedTokStride + (g + 8) * 4) = d.value[2];
    *reinterpret_cast<float*>(red + 1 * kP1bRedTokStride + g * 4) = d.value[1];
    *reinterpret_cast<float*>(red + 1 * kP1bRedTokStride + (g + 8) * 4) = d.value[3];
  }
  consumer_sync();
  if (tid < kP1bTokens * kP1bRowsPerCta) {
    // element tid = tok*16 + row: partials at kP1bRed + w*0x84 + tid*4, ordered sum from 0, bf16
    // pack
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      sum = __fadd_rn(sum, *smem_at<const float>(
                               ctx.smem, smem::kP1bRed + w * smem::kP1bRedWarpStride + tid * 4));
    }
    const int tok = tid / kP1bRowsPerCta;
    const int r = tid - tok * kP1bRowsPerCta;
    row.idx_scores_out[(tok_base + tok) * kIdxHeads + rowblock * kP1bRowsPerCta + r] = to_bf16(sum);
  }
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

constexpr int kP2Rows = kIqDim / kP2Ctas;  // 32 output features (rows) per CTA, from 32 * cta
constexpr int kP2ElemsPerThread = kQaDim / kConsumerThreads;
constexpr uint32_t kP2InvQaDimBits = 0x3a000000u;
constexpr uint32_t kP2RmsEpsBits = 0x3727c5acu;
constexpr uint32_t kP2PageKStride =
    0x800;  // qa_norm bytes per page (1024 K x 2 B): page B B-fragments at +0x24800
constexpr uint32_t kP2RgStride = 0x4000;   // 16-row group stride inside a page
constexpr uint32_t kP2WarpStride = 0x800;  // warp (128-K block) stride inside a page
constexpr uint32_t kP2StepStride = 0x100;  // k16 step stride inside a page
constexpr uint32_t kP2RedTokStride =
    kP2Rows * 4;  // 0x80: red[warp][token][row] f32 (+0x80 for token 2tp+1)
static_assert(kP2Rows == 32 && kP2ElemsPerThread == 8 && kSeqLen * kP2Rows == kConsumerThreads / 2,
              "P2 geometry (128 fold cells)");
static_assert(2 * kP2RedTokStride == 0x100 && 4 * kP2RedTokStride + 4 == smem::kP2RedWarpStride,
              "red layout: (lane&3)*0x100, warp stride 0x204");
static_assert(smem::kP2QnormTokStride == kQaDim * 2 + 0x20 &&
                  kP2PageKStride * kIndexerWqBPages == kQaDim * 2,
              "qa_norm token stride 0x1020, 2 pages x 1024 K");
static_assert(2 * kP2RgStride == kPageMatBytes && kConsumerWarps * kP2WarpStride == kP2RgStride &&
                  8 * kP2StepStride == kP2WarpStride,
              "page: [rg 2][warp 8][k16 8][lane 32][8 B] = 0x8000");
static_assert(kP2Rows * kP2Ctas * 2 == 0x2000, "IQ token stride 0x2000 B");

// Sum of squares of the thread's 8 fp32 elements: p(2i,2i+1) = fma(e[2i], e[2i], e[2i+1] *
// e[2i+1]), summed
// ((((0 + p01) + p23) + p45) + p67).  This association is part of the numerical contract.  Explicit
// intrinsics: no contraction / reassociation by the compiler.
__device__ __forceinline__ float p2_sumsq8(const float (&e)[8]) {
  const float p01 = __fmaf_rn(e[0], e[0], __fmul_rn(e[1], e[1]));
  const float p23 = __fmaf_rn(e[2], e[2], __fmul_rn(e[3], e[3]));
  const float p45 = __fmaf_rn(e[4], e[4], __fmul_rn(e[5], e[5]));
  const float p67 = __fmaf_rn(e[6], e[6], __fmul_rn(e[7], e[7]));
  float s = __fadd_rn(0.0f, p01);
  s = __fadd_rn(s, p23);
  s = __fadd_rn(s, p45);
  s = __fadd_rn(s, p67);
  return s;
}

// 32-lane shuffle-down 16, 8, 4, 2, 1 tree: lane l adds lane l + off; lane 0 ends with the warp
// total.
__device__ __forceinline__ float p2_warp_sum_down(float s) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
  return s;
}

__device__ __forceinline__ void p2_unpack_bf16x8(const uint4& raw, float (&v)[8]) {
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[2 * j] = bf16_pair_low(w[j]);
    v[2 * j + 1] = bf16_pair_high(w[j]);
  }
}

__device__ __forceinline__ void p2_rmsnorm_qa_to_smem(const CtxRow1& row, OpCtx& ctx) {
  const int tid = ctx.tid;
  uint4 q_raw[kSeqLen];
#pragma unroll
  for (int t = 0; t < kSeqLen; ++t) {
    q_raw[t] = *reinterpret_cast<const uint4*>(row.q_a_in + t * kQaDim + tid * kP2ElemsPerThread);
  }
  const float4 g_lo = *reinterpret_cast<const float4*>(row.gamma_q + tid * kP2ElemsPerThread);
  const float4 g_hi = *reinterpret_cast<const float4*>(row.gamma_q + tid * kP2ElemsPerThread + 4);
  const float gamma[8] = {g_lo.x, g_lo.y, g_lo.z, g_lo.w, g_hi.x, g_hi.y, g_hi.z, g_hi.w};

  float q[kSeqLen][8];
  float part[kSeqLen];
#pragma unroll
  for (int t = 0; t < kSeqLen; ++t) {
    p2_unpack_bf16x8(q_raw[t], q[t]);
    part[t] = p2_warp_sum_down(p2_sumsq8(q[t]));
  }
  // lane 0 of every warp: partial[t][warp] at kRmsPartial + t*0x20 + warp*4
  float* partial = smem_at<float>(ctx.smem, smem::kRmsPartial);
  if (ctx.lane == 0) {
#pragma unroll
    for (int t = 0; t < kSeqLen; ++t) partial[t * kConsumerWarps + ctx.warp] = part[t];
  }
  consumer_sync();

  // every thread: the 32 partials, per token ((((0 + w0) + w1) + ...) + w7), fma(sum, 1/2048,
  // 1e-5), rsqrt.approx
  float rstd[kSeqLen];
#pragma unroll
  for (int t = 0; t < kSeqLen; ++t) {
    const float4 w0 = *reinterpret_cast<const float4*>(partial + t * kConsumerWarps);
    const float4 w1 = *reinterpret_cast<const float4*>(partial + t * kConsumerWarps + 4);
    float tot = __fadd_rn(0.0f, w0.x);
    tot = __fadd_rn(tot, w0.y);
    tot = __fadd_rn(tot, w0.z);
    tot = __fadd_rn(tot, w0.w);
    tot = __fadd_rn(tot, w1.x);
    tot = __fadd_rn(tot, w1.y);
    tot = __fadd_rn(tot, w1.z);
    tot = __fadd_rn(tot, w1.w);
    const float mean_eps =
        __fmaf_rn(tot, __uint_as_float(kP2InvQaDimBits), __uint_as_float(kP2RmsEpsBits));
    rstd[t] = rsqrtf(mean_eps);
  }
#pragma unroll
  for (int t = 0; t < kSeqLen; ++t) {
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = __fmul_rn(gamma[2 * j], rstd[t]);
      const float g1 = __fmul_rn(gamma[2 * j + 1], rstd[t]);
      packed[j] = pack_bf16_pair(__fmul_rn(q[t][2 * j], g0), __fmul_rn(q[t][2 * j + 1], g1));
    }
    *smem_at<uint4>(ctx.smem, smem::kAct + t * smem::kP2QnormTokStride + tid * 16) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
  consumer_sync();
}

// All reads of one page for this thread: the 16 weight fragments of the page and the 8 k16 B
// fragments of qa_norm (lanes >= 16: zero).  The caller releases the slot AFTER this returns (and
// after the scale read).
__device__ __forceinline__ void p2_load_page(const OpCtx& ctx, uint32_t slot_s, uint32_t act_s,
                                             int page, uint32_t (&raw)[2][8][2],
                                             uint32_t (&b)[8][2]) {
  const uint32_t a_base = slot_s + ctx.warp * kP2WarpStride + ctx.lane * 8;
  const uint32_t b_base = act_s + (ctx.lane >> 2) * smem::kP2QnormTokStride + ctx.warp * 0x100 +
                          (ctx.lane & 3) * 4 + page * kP2PageKStride;
  const bool real_token = ctx.lane < 16;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
#pragma unroll
    for (int rg = 0; rg < 2; ++rg)
      lds_u32x2(a_base + rg * kP2RgStride + s * kP2StepStride, raw[rg][s][0], raw[rg][s][1]);
    b[s][0] = 0u;
    b[s][1] = 0u;
    if (real_token) {
      b[s][0] = lds_u32(b_base + s * 0x20);
      b[s][1] = lds_u32(b_base + s * 0x20 + 0x10);
    }
  }
}

// Two rg chains x 8 k16 steps from zero, A = weights, B = the qa_norm fragment of the step.
__device__ __forceinline__ void p2_page_hmma(const uint32_t (&raw)[2][8][2],
                                             const uint32_t (&b)[8][2], HmmaAcc (&d)[2]) {
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
    for (int i = 0; i < 4; ++i) d[rg].value[i] = 0.0f;
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      uint32_t a[4];
      fp8x8_to_bf16_fragment(raw[rg][s][0], raw[rg][s][1], a);
      hmma_16816_bf16(d[rg], a, b[s]);
    }
  }
}
}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// Fused P3 hand-off: every P2 CTA publishes its 32 BF16 IQ rows per query as launch-tagged 16-byte
// records {tag << 32 | w0, tag << 32 | w1} (4 rows each, one b128 store); the head's rotating CTA
// (the first of its four P2 CTAs) polls the head's 128 records directly instead of an acq_rel head
// counter followed by a reload of the IQ rows.  The records live in the record scratch of rank 0's
// arena (mega_wire::kScratch); IQ itself is still written.
constexpr size_t kRecIq = mega_wire::kScratch;
static_assert(size_t(kSeqLen) * kIqDim / 4 * 16 <= mega_wire::kScratchBytes,
              "IQ records fit the record scratch");
__device__ __forceinline__ void iq_rec_store(void* address, uint32_t w0, uint32_t w1,
                                             uint32_t tag) {
  const uint64_t lo = (uint64_t(tag) << 32) | w0;
  const uint64_t hi = (uint64_t(tag) << 32) | w1;
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.gpu.global.b128 [%0], r;\n\t}" ::
          "l"(address),
      "l"(lo), "l"(hi)
      : "memory");
}

// Two consecutive IQ records {lo, hi} polled together: both loads are issued before the first tag
// check and only a stale one is reloaded, so the second record costs no extra round trip.  Returns
// {lo.w0, lo.w1, hi.w0, hi.w1}.
__device__ __forceinline__ uint4 iq_rec_poll2(const void* address, uint32_t tag) {
  const uint8_t* a = static_cast<const uint8_t*>(address);
  uint64_t lo0, hi0, lo1, hi1;
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo0), "=l"(hi0)
      : "l"(a)
      : "memory");
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo1), "=l"(hi1)
      : "l"(a + 16)
      : "memory");
  for (;;) {
    const bool ok0 =
        static_cast<uint32_t>(lo0 >> 32) == tag && static_cast<uint32_t>(hi0 >> 32) == tag;
    const bool ok1 =
        static_cast<uint32_t>(lo1 >> 32) == tag && static_cast<uint32_t>(hi1 >> 32) == tag;
    if (ok0 && ok1) break;
    if (!ok0) {
      asm volatile(
          "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
          : "=l"(lo0), "=l"(hi0)
          : "l"(a)
          : "memory");
    }
    if (!ok1) {
      asm volatile(
          "{\n\t.reg .b128 r;\n\tld.relaxed.gpu.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
          : "=l"(lo1), "=l"(hi1)
          : "l"(a + 16)
          : "memory");
    }
  }
  return make_uint4(static_cast<uint32_t>(lo0), static_cast<uint32_t>(hi0),
                    static_cast<uint32_t>(lo1), static_cast<uint32_t>(hi1));
}

__device__ __forceinline__ int64_t p3_row_byte_offset(int q_elems_i32, int row, int lane16) {
  return static_cast<int64_t>(q_elems_i32) * 2 +
         static_cast<int64_t>(static_cast<uint32_t>(row)) * 256 + lane16 * 16;
}

// P3 fused into P2 (persistent kernel, N = 32 heads): a head's 128 IQ rows are produced by four
// consecutive P2 CTAs.  Each of them counts its completed rows on the head's arena counter
// (acq_rel, after the CTA barrier that orders its IQ stores); the fourth arrival (counter value 0
// mod 4, every launch adds exactly four per head) reloads the head's BF16 IQ rows and applies
// exactly P3's RoPE -> Hadamard128 -> x SCALE -> BF16 for the four queries, then arrives C3 with
// 1/32 of the word sum.  This replaces the grid-wide C1 wait and the separate 8-CTA P3 phase; the
// IQ / IQ_RT values are bitwise those of P2 -> P3.
__device__ __forceinline__ bool p3_fused(const IndexerArgs& args, const OpCtx& ctx) {
  return args.mode == kModePersistent && !has_inline_ctx(args) &&
         ctx_p3(ctx.smem).n_heads == static_cast<uint64_t>(kIdxHeads);
}
__device__ __forceinline__ void p2_fused_rotate_head(const OpCtx& ctx) {
  static_assert(kP2Ctas == 4 * kIdxHeads && kIdxDim == 4 * kP2Rows,
                "a head's 128 IQ rows come from 4 consecutive P2 CTAs");
  static_assert(kWordSum % kIdxHeads == 0, "C3 weight per head");
  if ((ctx.cta & 3) != 0) return;
  const CtxRow3& row3 = ctx_p3(ctx.smem);
  const int head = ctx.cta >> 2;
  if (ctx.tid < kSeqLen * 16) {
    const int q = ctx.tid >> 4;
    const int lane = ctx.tid & 15;
    const float* table =
        row3.rope_cos_sin == nullptr ? nullptr : row3.rope_cos_sin + size_t(ctx.cur_pos) * 64;
    const bool do_rope = (table != nullptr) && lane < 8;
    const int q_elems = (kIdxHeads * q) * kIdxDim;
    float v[8];
    float cs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = 0.0f;
      cs[i] = 0.0f;
    }
    // The RoPE table is a read-only launch input: fetch it before polling so its round trip
    // overlaps the wait for the head's IQ records, and poll the two records of this lane together
    // (re-polling only a stale one) instead of one after the other.
    if (do_rope) {
      const float* cs_src = table + q * 64 + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) cs[i] = cs_src[i];
    }
    const uint8_t* rec =
        ctx.arena + kRecIq + size_t((q_elems + head * kIdxDim + lane * 8) / 4) * 16;
    const uint4 w = iq_rec_poll2(rec, ctx.epoch);
    p3_unpack_bf16x8(w, v);
    if (do_rope) p3_rope_dims8(v, cs);
    p3_hadamard128(v, lane);
    uint4 packed;
    packed.x = pack_bf16_pair(__fmul_rn(v[0], kHadamardScale), __fmul_rn(v[1], kHadamardScale));
    packed.y = pack_bf16_pair(__fmul_rn(v[2], kHadamardScale), __fmul_rn(v[3], kHadamardScale));
    packed.z = pack_bf16_pair(__fmul_rn(v[4], kHadamardScale), __fmul_rn(v[5], kHadamardScale));
    packed.w = pack_bf16_pair(__fmul_rn(v[6], kHadamardScale), __fmul_rn(v[7], kHadamardScale));
    uint8_t* dst =
        reinterpret_cast<uint8_t*>(row3.iq_rt_out) + p3_row_byte_offset(q_elems, head, lane);
    *reinterpret_cast<uint4*>(dst) = packed;
    if (row3.second_out_base != nullptr) {
      int32_t idx = *row3.second_out_pos_ptr + q;
      const int32_t m = static_cast<int32_t>(row3.second_out_mod);
      if (m >= 1) idx = idx % m;
      const int32_t ring_elems = (kIdxHeads * kIdxDim) * idx;
      uint8_t* dst2 = reinterpret_cast<uint8_t*>(row3.second_out_base) +
                      p3_row_byte_offset(ring_elems, head, lane);
      *reinterpret_cast<uint4*>(dst2) = packed;
    }
  }
  boundary_arrive(ctx, kSyncC3, kWordSum / kIdxHeads);
}

template <bool StaticMath>
static __device__ void op_p2_rmsnorm_gemv_indexer_wq_b(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  const int pages = p2_page_count(ctx.cta);  // 2 for every Group1 CTA
  if constexpr (!StaticMath) {
    if (ctx.null_math) {
      ring_drain(ctx, pages);
      return;
    }
  }
  const CtxRow1& row = ctx_p2(ctx.smem);

  p2_rmsnorm_qa_to_smem(row, ctx);

  // Page A (item, slot item&3): full wait; scale at the PAGE START (slot + warp*4 + 0x8000); 16
  // weight loads + B fragments; release after the page's last load; the page-B full wait follows
  // before page A's tail MMAs.
  const uint32_t act_s = ctx.smem_s + smem::kAct;
  uint32_t raw_a[2][8][2];
  uint32_t b_a[8][2];
  const uint32_t slot_a = ring_acquire(ctx);
  const float scale_a = lds_f32(slot_a + kPageScaleOffset + ctx.warp * 4);
  p2_load_page(ctx, slot_a, act_s, 0, raw_a, b_a);
  ring_release(ctx);
  const uint32_t slot_b = ring_acquire(ctx);  // wait full[(item+1)&3]
  HmmaAcc d_a[2];
  p2_page_hmma(raw_a, b_a, d_a);

  // Page B (item+1): 16 weight loads + B fragments (+0x800 K offset); scale at the PAGE END (the
  // slot's last read); release.
  uint32_t raw_b[2][8][2];
  uint32_t b_b[8][2];
  p2_load_page(ctx, slot_b, act_s, 1, raw_b, b_b);
  const float scale_b = lds_f32(slot_b + kPageScaleOffset + ctx.warp * 4);
  ring_release(ctx);  // arrive empty[(item+1)&3]
  HmmaAcc d_b[2];
  p2_page_hmma(raw_b, b_b, d_b);

  float acc[2][4];
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[rg][i] = __fmaf_rn(scale_a, d_a[rg].value[i], 0.0f);
  }
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[rg][i] = __fmaf_rn(scale_b, d_b[rg].value[i], acc[rg][i]);
  }

  // Warp partials red[warp][token][row] f32 at kP2Red + warp*0x204 + (lane&3)*0x100 + (lane>>2)*4:
  // D fragment (rows g = lane>>2 / g+8 of group rg, token columns 2(lane&3) / +1); lanes with
  // (lane&3) > 1 hold the zero tokens 4..7 and skip.
  const int g = ctx.lane >> 2;
  const int tp = ctx.lane & 3;
  if (tp < 2) {  // lanes (tid&3) <= 1
    uint8_t* red = ctx.smem + smem::kP2Red + ctx.warp * smem::kP2RedWarpStride +
                   tp * (2 * kP2RedTokStride) + g * 4;
    *reinterpret_cast<float*>(red + 0x00) = acc[0][0];  // token 2tp, row g
    *reinterpret_cast<float*>(red + 0x20) = acc[0][2];  // token 2tp, row g+8
    *reinterpret_cast<float*>(red + 0x80) = acc[0][1];  // token 2tp+1, row g
    *reinterpret_cast<float*>(red + 0xa0) = acc[0][3];  // token 2tp+1, row g+8
    *reinterpret_cast<float*>(red + 0x40) = acc[1][0];  // token 2tp, row g+16
    *reinterpret_cast<float*>(red + 0x60) = acc[1][2];  // token 2tp, row g+24
    *reinterpret_cast<float*>(red + 0xc0) = acc[1][1];  // token 2tp+1, row g+16
    *reinterpret_cast<float*>(red + 0xe0) = acc[1][3];  // token 2tp+1, row g+24
  }
  consumer_sync();

  // tid < 128 = cell (token = tid>>5, row = tid&31): ((((0 + w0) + w1) + ...) + w7) over the 8
  // warps, bf16 pack, 2 B store to kP2Out + tid*2.
  if (ctx.tid < kSeqLen * kP2Rows) {
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      sum = __fadd_rn(sum, *reinterpret_cast<const float*>(
                               ctx.smem + smem::kP2Red + w * smem::kP2RedWarpStride + ctx.tid * 4));
    }
    const int tok = ctx.tid / kP2Rows;
    const int r = ctx.tid - tok * kP2Rows;
    const __nv_bfloat16 value = to_bf16(sum);
    row.iq_out[tok * kIqDim + ctx.cta * kP2Rows + r] = value;
    if constexpr (!StaticMath) {
      if (p3_fused(args, ctx)) {
        const uint32_t bits = static_cast<uint32_t>(__bfloat16_as_ushort(value));
        const uint32_t b1 = __shfl_down_sync(0xffffffffu, bits, 1);
        const uint32_t b2 = __shfl_down_sync(0xffffffffu, bits, 2);
        const uint32_t b3 = __shfl_down_sync(0xffffffffu, bits, 3);
        if ((r & 3) == 0) {
          iq_rec_store(ctx.arena + kRecIq + size_t((tok * kIqDim + ctx.cta * kP2Rows + r) / 4) * 16,
                       bits | (b1 << 16), b2 | (b3 << 16), ctx.epoch);
        }
      }
    }
  }
  if constexpr (!StaticMath) {
    if (p3_fused(args, ctx)) p2_fused_rotate_head(ctx);
  }
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// Byte offset of (q, row, lane) inside IQ / IQ_RT: int32 element offset (N*q) << 7, + row*0x100
// (64-bit, sign-extended),
// + lane*16.

template <bool SparseComputeOnly>
static __device__ void op_p3_rotate_q(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  if constexpr (!SparseComputeOnly) {
    if (ctx.null_math) return;  // protocol only: no memory traffic (no ring items in P3)
  }
  const CtxRow3& row3 = ctx_p3(ctx.smem);
  const int n_heads =
      SparseComputeOnly ? kIdxHeads : static_cast<int>(row3.n_heads);  // the host always passes 32
  if constexpr (!SparseComputeOnly) {
    if (ctx.cta >= p3_ctas(n_heads)) return;
  }
  const int q = p3_query_of_cta(ctx.cta, n_heads);        // quotient of the signed division
  const int hb = p3_head_block_of_cta(ctx.cta, n_heads);  // remainder
  const int lane = ctx.tid & 15;
  const int row = hb * kP3RowsPerBlock + (ctx.tid >> 4);
  const bool row_valid = row < n_heads;  // row >= N: no load, no stores
  const float* table =
      row3.rope_cos_sin == nullptr ? nullptr : row3.rope_cos_sin + size_t(ctx.cur_pos) * 64;
  const bool do_rope =
      (table != nullptr) && lane < 8;  // skipped when the table is null or lane > 7
  const int q_elems = (n_heads * q) * kIdxDim;

  float v[8];   // this thread's 8 dims [8*lane, +8) of row (q, row)
  float cs[8];  // (c0, s0, c1, s1, c2, s2, c3, s3) of the 4 pairs
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = 0.0f;
    cs[i] = 0.0f;
  }
  if (row_valid) {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(row3.iq_in) + p3_row_byte_offset(q_elems, row, lane);
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    p3_unpack_bf16x8(raw, v);
    if (do_rope) {
      const float* cs_src = table + q * 64 + lane * 8;  // q*0x40 + lane*8 f32 = q*256 B + lane*32 B
#pragma unroll
      for (int i = 0; i < 8; ++i) cs[i] = cs_src[i];
    }
  }
  if (do_rope) p3_rope_dims8(v, cs);
  p3_hadamard128(v, lane);
  if (!row_valid) return;  // branch  (after the shuffles: every lane took part)

  uint4 packed;  // scale by 128^-0.5, then the ONLY rounding:
  packed.x = pack_bf16_pair(__fmul_rn(v[0], kHadamardScale), __fmul_rn(v[1], kHadamardScale));
  packed.y = pack_bf16_pair(__fmul_rn(v[2], kHadamardScale), __fmul_rn(v[3], kHadamardScale));
  packed.z = pack_bf16_pair(__fmul_rn(v[4], kHadamardScale), __fmul_rn(v[5], kHadamardScale));
  packed.w = pack_bf16_pair(__fmul_rn(v[6], kHadamardScale), __fmul_rn(v[7], kHadamardScale));
  uint8_t* dst =
      reinterpret_cast<uint8_t*>(row3.iq_rt_out) + p3_row_byte_offset(q_elems, row, lane);
  *reinterpret_cast<uint4*>(dst) = packed;

  // Position-ring copy: row 3 +0x20 = base, +0x28 = pos_ptr; base == 0 -> skip.
  if (row3.second_out_base != nullptr) {
    int32_t idx = *row3.second_out_pos_ptr + q;
    const int32_t m = static_cast<int32_t>(row3.second_out_mod);  // row 3 +0x34
    if (m >= 1) idx = idx % m;                                    // signed truncated remainder
    const int32_t ring_elems = (n_heads * kIdxDim) * idx;
    uint8_t* dst2 = reinterpret_cast<uint8_t*>(row3.second_out_base) +
                    p3_row_byte_offset(ring_elems, row, lane);
    *reinterpret_cast<uint4*>(dst2) = packed;
  }
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// ---- geometry of the compute branch ----------------------
constexpr int kG2aWorkThreads =
    kSeqLen * 16;                 // 64: token tok = tid >> 4 (4 tokens), 16 lanes x 8 dims per row
constexpr int kG2aRopeLanes = 8;  // lanes (tid & 15) < 8 rotate dims 0..63
constexpr float kG2aMeanScale = -0.0078125f;  // -1/128: x = fma(sum, -1/128, x)  (mean subtraction)
constexpr float kG2aVarScale = 0.0078125f;    // 1/128 = 0x3c000000: fma(var, 1/128, eps)
constexpr float kG2aLnEps = 1e-6f;            // 0x358637bd = 9.9999999747524270788e-07
static_assert(smem::kG2aGamma == smem::kRing && smem::kG2aBeta == smem::kG2aGamma + 0x200 &&
                  smem::kG2aCosSin == smem::kG2aGamma + 0x400,
              "G2a parameters are ring slot 0: gamma at +0, beta at +0x200, cos-sin at +0x400");
static_assert(smem::kG2aResult == smem::kG2aKi + 0x1000,
              "G2a result follows the ki rows 0x1000 B later");
static_assert(smem::kG2aKi + kG2aWorkThreads * 16 <= smem::kG2aResult &&
                  smem::kG2aResult + kG2aWorkThreads * 16 <= smem::kUsedEnd,
              "G2a ki / result rows fit their regions");
static_assert(kG2aWorkThreads * 8 == kSeqLen * kIdxDim,
              "64 threads x 8 elements = 4 rows x 128 dims");

// 16-lane butterfly sum: LN sum, centered sum of squares.  Every lane of the 16-lane row ends with
// the same value. All 32 lanes of the warp must call it.
__device__ __forceinline__ float g2a_bfly16_sum(float v) {
#pragma unroll
  for (int m = 1; m <= 8; m <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, m, 16));
  return v;
}

// LayerNorm of one 128-dim row held by 16 lanes x 8 fp32 (in place):
//   lane sum      sequential from zero, x0..x7
//   row sum       g2a_bfly16_sum
//   centre        x_j = fma(sum, -0.0078125, x_j)
//   lane sumsq    q = fma(x_0, x_0, 0); q = fma(x_j, x_j, q) j = 1..7 (sequential)
//   row sumsq     g2a_bfly16_sum
//   rstd          rsqrtf(fma(sumsq, 0x3c000000, 1e-6))                     (non-ftz
//   rsqrt.approx.f32, no Newton step) normalize     xhat_j = x_j * rstd affine        y_j =
//   fma(gamma_j, xhat_j, beta_j)
// Two roundings between centre and affine (mul then fma) -- NOT fma(x, gamma*rstd, beta) like the
// RMSNorms of P1/P2.

__device__ __forceinline__ void g2a_layernorm_row(float (&v)[8], const float (&gamma)[8],
                                                  const float (&beta)[8]) {
  float s = __fadd_rn(0.0f, v[0]);
#pragma unroll
  for (int j = 1; j < 8; ++j) s = __fadd_rn(v[j], s);
  s = g2a_bfly16_sum(s);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = __fmaf_rn(s, kG2aMeanScale, v[j]);
  float q = __fmaf_rn(v[0], v[0], 0.0f);
#pragma unroll
  for (int j = 1; j < 8; ++j) q = __fmaf_rn(v[j], v[j], q);
  q = g2a_bfly16_sum(q);
  const float rstd = rsqrtf(__fmaf_rn(q, kG2aVarScale, kG2aLnEps));
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = __fmul_rn(v[j], rstd);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = __fmaf_rn(gamma[j], v[j], beta[j]);
}

__device__ __forceinline__ uint4 g2a_scale_pack(const float (&v)[8]) {
  uint4 packed;
  packed.x = pack_bf16_pair(__fmul_rn(v[0], kHadamardScale), __fmul_rn(v[1], kHadamardScale));
  packed.y = pack_bf16_pair(__fmul_rn(v[2], kHadamardScale), __fmul_rn(v[3], kHadamardScale));
  packed.z = pack_bf16_pair(__fmul_rn(v[4], kHadamardScale), __fmul_rn(v[5], kHadamardScale));
  packed.w = pack_bf16_pair(__fmul_rn(v[6], kHadamardScale), __fmul_rn(v[7], kHadamardScale));
  return packed;
}

template <bool StaticMath>
static __device__ void op_g2a_layernorm_rope_k(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  if (ctx.cta != 0) return;  // CTAs 129..131: nothing
  if constexpr (!StaticMath) {
    if (ctx.null_math) {
      ring_drain(ctx, 1);
      return;
    }
  }
  const CtxRow2& row = ctx_g2a(ctx.smem);
  const int tid = ctx.tid;
  const bool worker = tid < kG2aWorkThreads;  // warp-uniform: warps 0..1 work, 2..7 idle
  const uint32_t my16 =
      static_cast<uint32_t>(tid) * 16u;  // this thread's 16 B of the [4][128] bf16 block

  if (worker)
    ldgsts_16(ctx.smem_s + smem::kG2aKi + my16, reinterpret_cast<const uint8_t*>(row.ki_in) + my16);
  ldgsts_commit();
  // 2. item 0 = slot 0 (gamma / beta / cos-sin fed by the 4 prefetch warps): wait full[0] parity 0
  // by all 256 threads.
  const uint32_t slot_s = ring_acquire(ctx);
  (void)slot_s;  // == ctx.smem_s + smem::kG2aGamma (static_assert above)
  ldgsts_wait_all();

  if (worker) {  // branch
    const int lane16 = tid & 15;
    const int tok = tid >> 4;                     // (tid*16) & 0x300 = tok*0x100
    const bool do_rope = lane16 < kG2aRopeLanes;  // lanes 8..15 do not rotate
    // 3. loads from smem: ki, gamma, beta and (rotating lanes only) cos-sin; the other lanes keep
    // zeros.
    const uint32_t param16 = static_cast<uint32_t>(lane16) * 32u;  // (tid*32) & 0x1e0 /
    const uint4 raw = *smem_at<const uint4>(ctx.smem, smem::kG2aKi + my16);
    const float4 g0 = *smem_at<const float4>(ctx.smem, smem::kG2aGamma + param16);
    const float4 g1 = *smem_at<const float4>(ctx.smem, smem::kG2aGamma + param16 + 16);
    const float4 b0 = *smem_at<const float4>(ctx.smem, smem::kG2aBeta + param16);
    const float4 b1 = *smem_at<const float4>(ctx.smem, smem::kG2aBeta + param16 + 16);
    const float gamma[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    const float beta[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float cs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f};  // (c0, s0, c1, s1, c2, s2, c3, s3) of the thread's 4 pairs
    if (do_rope) {
      const uint32_t cs_off = smem::kG2aCosSin + static_cast<uint32_t>(tok) * 256u + param16;
      const float4 c0 = *smem_at<const float4>(ctx.smem, cs_off);
      const float4 c1 = *smem_at<const float4>(ctx.smem, cs_off + 16);
      cs[0] = c0.x;
      cs[1] = c0.y;
      cs[2] = c0.z;
      cs[3] = c0.w;
      cs[4] = c1.x;
      cs[5] = c1.y;
      cs[6] = c1.z;
      cs[7] = c1.w;
    }
    // 4. bf16 -> fp32: even element = word << 16, odd = word & 0xffff0000.
    float v[8];
    p3_unpack_bf16x8(raw, v);
    // 5. LayerNorm (fp32, eps 1e-6, gamma / beta).
    g2a_layernorm_row(v, gamma, beta);
    if (do_rope) p3_rope_dims8(v, cs);
    //.
    p3_hadamard128(v, lane16);
    *smem_at<uint4>(ctx.smem, smem::kG2aResult + my16) = g2a_scale_pack(v);
    // 9. write-back (tid < 64): indexer_k_cache row cur_pos (row stride 256 B, no row modulo).
    const uint4 out = *smem_at<const uint4>(ctx.smem, smem::kG2aResult + my16);
    const int32_t row_elems = ctx.cur_pos * kIdxDim;  // int32 cur_pos*0x80 (cur_pos <= 202756 fits)
    uint8_t* dst = reinterpret_cast<uint8_t*>(row.indexer_k_cache) +
                   static_cast<int64_t>(row_elems) * 2 + my16;
    *reinterpret_cast<uint4*>(dst) = out;
  }
  ring_release(ctx);
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// ---- tcgen05 primitives of the scoring op (prefix p4_) ----
// UMMA shared-memory descriptor of a SWIZZLE_128B K-major operand at `window_addr` (1 KB aligned in
// the shared window), K16 step k (0..7): lo = umma_smem_desc_lo(addr) + (k&3)*2 + (k>>2)*0x400, hi
// = kUmmaSmemDescHi (0x40004040: SBO 1024 B, version 1, layout 2 = SWIZZLE_128B).
__device__ __forceinline__ uint64_t p4_umma_desc(uint32_t window_addr, int k) {
  const uint32_t lo = umma_smem_desc_lo(window_addr) + static_cast<uint32_t>(k & 3) * kUmmaKStepLo +
                      static_cast<uint32_t>(k >> 2) * kUmmaBoxStepLo;
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(kUmmaSmemDescHi) << 32);
}
// tcgen05.mma cta_group::1 kind::f16, A and B from shared memory (descriptors), D in TMEM.
// `accumulate` = enable-input-d. Single-thread instruction: called by tid 0 only.
__device__ __forceinline__ void p4_umma_mma_bf16_ss(uint32_t d_tmem, uint64_t a_desc,
                                                    uint64_t b_desc, uint32_t idesc,
                                                    uint32_t accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate));
}
__device__ __forceinline__ void p4_umma_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar)
      : "memory");
}
// tcgen05.ld 32x32b.x64: 64 consecutive fp32 columns of this thread's TMEM lane + wait::ld.
__device__ __forceinline__ void p4_tmem_ld_32x32b_x64(uint32_t taddr, uint32_t (&v)[64]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,"
      "%39,"
      "%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,"
      "%63}, [%64];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]),
        "=r"(v[7]), "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]),
        "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]),
        "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
        "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]), "=r"(v[32]), "=r"(v[33]), "=r"(v[34]),
        "=r"(v[35]), "=r"(v[36]), "=r"(v[37]), "=r"(v[38]), "=r"(v[39]), "=r"(v[40]), "=r"(v[41]),
        "=r"(v[42]), "=r"(v[43]), "=r"(v[44]), "=r"(v[45]), "=r"(v[46]), "=r"(v[47]), "=r"(v[48]),
        "=r"(v[49]), "=r"(v[50]), "=r"(v[51]), "=r"(v[52]), "=r"(v[53]), "=r"(v[54]), "=r"(v[55]),
        "=r"(v[56]), "=r"(v[57]), "=r"(v[58]), "=r"(v[59]), "=r"(v[60]), "=r"(v[61]), "=r"(v[62]),
        "=r"(v[63])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void p4_hist_inc(uint32_t addr_s) {
  asm volatile("red.shared.add.u32 [%0], 1;" ::"r"(addr_s) : "memory");
}
// Byte offset inside a K-tile slot of 16 B chunk `chunk` (0..15 of the 256 B row) of tile row r
// (0..127): the SWIZZLE_128B layout the TMA boxes use (chunk >> 3 selects the 64-dim box, 1 KB atom
// = 8 rows x 128 B, chunk ^ (r & 7) inside the row).
__device__ __forceinline__ uint32_t p4_patch_dst_offset(uint32_t chunk, uint32_t r) {
  return ((chunk & 8u) ? uint32_t(kKTileBoxBytes) : 0u) + ((r >> 3) & 0xfu) * 1024u +
         (r & 7u) * 128u + (((chunk & 7u) ^ (r & 7u)) << 4);
}

// ---- setup pieces ---------------------------
__device__ __forceinline__ void p4_tmem_alloc(OpCtx& ctx) {
  if (ctx.warp == 0) {
    tmem_alloc(ctx.smem_s + smem::kTmemBase, kTmemColumns);
    tmem_relinquish_alloc_permit();  // harmless at 1 CTA/SM
    tcgen05_fence_before_thread_sync();
  }
  consumer_sync();
  tcgen05_fence_after_thread_sync();
}
__device__ __forceinline__ void p4_init_barriers(OpCtx& ctx) {
  if (ctx.tid == 0) {
    mbar_init(ctx.smem_s + smem::kMmaBar, 1u);
    mbar_init(ctx.smem_s + smem::kMmaBar1, 1u);
    mbar_init(ctx.smem_s + smem::kQTmaBar, 1u);
    mbar_fence_init();
  }
  consumer_sync();
}

__device__ __forceinline__ void p4_q_tile_load(OpCtx& ctx, const void* q_tmap) {
  const uint32_t bar = ctx.smem_s + smem::kQTmaBar;
  if (ctx.tid == 0) {
    mbar_arrive_expect_tx(bar, kQTileBytes);  // = expect_tx 0x8000 + the ONE arrival, fused
    for (int h = 0; h < 2; ++h) {
      for (int r = 0; r < 4; ++r) {
        tma_load_2d_hint(
            ctx.smem_s + smem::kQTile + h * smem::kQTileHalfStride + r * smem::kQTileRowBlockStride,
            q_tmap, h * (kIdxDim / 2), r * kQTileBoxRows, bar, kL2PolicyEvictNormal);
      }
    }
  }
  mbar_wait_parity(bar, 0u);
  fence_proxy_async_shared_cta();
  consumer_sync();
}
// 64 fp32 weights of queries 2h, 2h+1: idx_weights + half*128 B (four 32 B loads); bf16 -> fp32 is
// the exact shift. w[i] = weight of query 2h + (i >> 5), head i & 31 = TMEM column half*64 + i
// (identity pairing).
__device__ __forceinline__ void p4_load_weights(const CtxRow4& row, int half, float (&w)[64]) {
  const uint4* src = reinterpret_cast<const uint4*>(row.idx_weights) + half * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint4 v = src[i];
    w[8 * i + 0] = __uint_as_float(v.x << 16);
    w[8 * i + 1] = __uint_as_float(v.x & 0xffff0000u);
    w[8 * i + 2] = __uint_as_float(v.y << 16);
    w[8 * i + 3] = __uint_as_float(v.y & 0xffff0000u);
    w[8 * i + 4] = __uint_as_float(v.z << 16);
    w[8 * i + 5] = __uint_as_float(v.z & 0xffff0000u);
    w[8 * i + 6] = __uint_as_float(v.w << 16);
    w[8 * i + 7] = __uint_as_float(v.w & 0xffff0000u);
  }
}
// Flat clear of the 1032-dword local histogram (4 x 258 u32 at kHist = window 0x30030): stores at
// tid*4 + {0, 0x400, 0x800, 0xc00}, tid < 8 also +0x1000.  The caller's barrier publishes it.
__device__ __forceinline__ void p4_hist_clear(OpCtx& ctx) {
  uint32_t* hist = smem_at<uint32_t>(ctx.smem, smem::kHist);
  hist[ctx.tid] = 0u;
  hist[ctx.tid + 256] = 0u;
  hist[ctx.tid + 512] = 0u;
  hist[ctx.tid + 768] = 0u;
  if (ctx.tid < 8) hist[ctx.tid + 1024] = 0u;
}
__device__ __forceinline__ void p4_tmem_dealloc(OpCtx& ctx) {
  tcgen05_fence_before_thread_sync();
  consumer_sync();
  if (ctx.warp == 0) {
    tcgen05_fence_after_thread_sync();
    const uint32_t base = *smem_at<uint32_t>(ctx.smem, smem::kTmemBase);
    tmem_dealloc(base, kTmemColumns);
  }
}

// ---- one K tile
// ------------------------------------------------------------------------------------------
// cp.async patch of the current rows cur_pos..+3 into the tile that covers them: tid < 128 only;
// the 16 threads with (tid>>4) == (p & 7) copy row p, one 16 B chunk (tid & 15) each; rows outside
// [base, base+128) are skipped by the unsigned r > 0x7f test.
__device__ __forceinline__ void p4_patch_current_rows(const OpCtx& ctx, const CtxRow4& row,
                                                      uint32_t slot, int base) {
  if (ctx.tid < kKTileRows) {  // threads 128..255 go straight to the barrier
    const uint32_t chunk = static_cast<uint32_t>(ctx.tid) & 15u;
    const uint32_t row_sel = (static_cast<uint32_t>(ctx.tid) >> 4) & 7u;  // must equal p & 7
    const uint8_t* src = reinterpret_cast<const uint8_t*>(row.k_patch_src) + chunk * 16u;
#pragma unroll
    for (int j = 0; j < kSeqLen; ++j) {
      const int p = ctx.cur_pos + j;
      const uint32_t r =
          static_cast<uint32_t>(p - base);  // r > 0x7f (unsigned) -> not in this tile
      if (r <= 0x7fu && row_sel == (static_cast<uint32_t>(p) & 7u)) {
        ldgsts_16(slot + p4_patch_dst_offset(chunk, r),
                  src + static_cast<size_t>(p) * (kIdxDim * 2));
      }
    }
    ldgsts_commit();
    ldgsts_wait_all();
  }
}

template <bool SparseComputeOnly, bool WaitInside = false>
static __device__ void op_p4_score_tiles(const IndexerArgs& args, OpCtx& ctx) {
  const int n_tiles = SparseComputeOnly ? k_tiles_of_cta(ctx.cur_pos, ctx.cta)
                                        : (ctx.sparse ? k_tiles_of_cta(ctx.cur_pos, ctx.cta) : 0);
  if constexpr (!SparseComputeOnly) {
    if (!ctx.sparse)
      return;  // dense launches never enter (consumer_main runs P4 on sparse steps only); null
               // mode included (no tiles)
  }
  // CTAs beyond ceil(L/128) have no K rows. Avoid allocating TMEM and
  // redundantly TMA-loading the 32 KiB query tile on those SMs; they still
  // participate in consumer_main's C4 arrival after returning from P4/P5.
  if (n_tiles == 0) return;
  // Null mode keeps the protocol part of P4: the TMEM alloc/dealloc pair and the two count-1
  // mbarrier inits + fence are exercised every sparse launch (TMEM alloc/free every launch); only
  // the tile math is replaced by ring drains. Every sparse CTA runs the setup regardless of
  // n_tiles.
  p4_tmem_alloc(ctx);  // alloc -> barrier
  p4_init_barriers(ctx);
  if constexpr (!SparseComputeOnly) {
    if (ctx.null_math) {
      if constexpr (WaitInside) boundary_wait2(ctx, kSyncC3, kSyncC2);
      ring_drain(ctx, n_tiles);  // protocol only: tiles are consumed, no math, no q-tile TMA (no
                                 // memory traffic elsewhere)
      p4_tmem_dealloc(ctx);
      return;
    }
  }
  {
    const CtxRow4& row = ctx_score(ctx.smem);
    const void* q_tmap;
    if constexpr (SparseComputeOnly) {
      q_tmap = static_cast<const void*>(&args.q_tmap);
    } else {
      q_tmap = has_inline_ctx(args) ? static_cast<const void*>(&args.q_tmap) : row.q_tmap;
    }
    if (q_tmap == nullptr) __trap();  // a null map would fault inside the TMA unit
    const int half = ctx.tid >> 7;
    float w[64];
    const bool need_hist = ctx.cur_pos > kLastDenseCurPos && !long_context(ctx.cur_pos);
    if constexpr (WaitInside) {
      p4_load_weights(row, half, w);
      if (need_hist) p4_hist_clear(ctx);
      boundary_wait2(ctx, kSyncC3, kSyncC2);
      p4_q_tile_load(ctx, q_tmap);
    } else {
      p4_q_tile_load(ctx, q_tmap);
      p4_load_weights(row, half, w);
      if (need_hist) {
        p4_hist_clear(ctx);
        consumer_sync();
      }
    }
    const uint32_t tmem_d = *smem_at<uint32_t>(
        ctx.smem,
        smem::kTmemBase);  // smem 0x30000 (written by the alloc, published by its barrier)
    const uint32_t taddr = tmem_d + static_cast<uint32_t>(half) * 64u;
    const int L = total_len(ctx.cur_pos);  // cur_pos + 4
    const int stride = logits_row_stride(args.flags, ctx.cur_pos);
    const int q0 = 2 * half;
    const int causal0 = ctx.cur_pos + q0;
    const int causal1 = causal0 + 1;
    const uint32_t hist0 = ctx.smem_s + smem::kHist + static_cast<uint32_t>(q0) * kHistBlockBytes;
    const uint32_t hist1 = hist0 + kHistBlockBytes;
    uint32_t* const bin_cache = smem_at<uint32_t>(ctx.smem, smem::kP5BinCache);
    const uint32_t q_desc_base = ctx.smem_s + smem::kQTile;  // window 0x28000 -> lo 0x12800
    const uint32_t item0 = ctx.item;
    auto acquire_and_issue = [&](int t) {
      const uint32_t item_t = item0 + static_cast<uint32_t>(t);
      const uint32_t s_idx = item_t & (kRingSlots - 1);
      mbar_wait_parity(ring_full_bar(ctx, s_idx), (item_t >> 2) & 1u);
      const uint32_t slot = ring_slot_s(ctx, s_idx);
      const int base_t = k_tile_base(ctx.cta, t);
      if (!WaitInside && k_tile_covers_current(ctx.cur_pos, base_t)) {
        p4_patch_current_rows(ctx, row, slot, base_t);
        fence_proxy_async_shared_cta();  // writer side, every thread: its generic-proxy patch is
                                         // ordered before the async-proxy MMA read; canonical
                                         // wait_group -> fence -> bar.sync -> MMA
        consumer_sync();                 // the fenced patch is complete before tid 0 issues the MMA
      }
      // MMA: tid 0 issues the 8 K16 steps (A = K tile slot, B = q tile; first step
      // overwrites D) and the commit to kMmaBar.
      if (ctx.tid == 0) {                   // tid != 0 -> 0xcd50
        tcgen05_fence_after_thread_sync();  // orders every thread's tcgen05.ld of the previous tile
                                            // (fenced before the barrier) before this MMA
        const uint32_t d = tmem_d + static_cast<uint32_t>(t & 1) * 128u;
#pragma unroll
        for (int k = 0; k < kUmmaKStepsPerTile; ++k) {
          p4_umma_mma_bf16_ss(d, p4_umma_desc(slot, k), p4_umma_desc(q_desc_base, k),
                              kUmmaIdescBf16F32M128N128, k == 0 ? 0u : 1u);
        }
        p4_umma_commit(ctx.smem_s + ((t & 1) ? smem::kMmaBar1 : smem::kMmaBar));
      }
    };
    acquire_and_issue(0);
#pragma unroll 1
    for (int it = 0; it < n_tiles; ++it) {
      const int base = k_tile_base(ctx.cta, it);
      if (it + 1 < n_tiles) acquire_and_issue(it + 1);
      mbar_wait_parity(ctx.smem_s + ((it & 1) ? smem::kMmaBar1 : smem::kMmaBar),
                       static_cast<uint32_t>(it >> 1) & 1u);
      tcgen05_fence_after_thread_sync();
      mbar_arrive(ring_empty_bar(ctx, (item0 + static_cast<uint32_t>(it)) & (kRingSlots - 1)));
      uint32_t v[64];
      p4_tmem_ld_32x32b_x64(taddr + static_cast<uint32_t>(it & 1) * 128u, v);
      float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
      for (int h = 0; h < kIdxHeads; ++h) {
        acc0 = __fmaf_rn(w[h], fmaxf(0.0f, __uint_as_float(v[h])), acc0);
        acc1 = __fmaf_rn(w[kIdxHeads + h], fmaxf(0.0f, __uint_as_float(v[kIdxHeads + h])), acc1);
      }
      acc0 = __fmul_rn(
          acc0, kScoreScale);  // exact power-of-two scale, once per query instead of once per head
      acc1 = __fmul_rn(acc1, kScoreScale);
      const int pos = base + (ctx.tid & 0x7f);
      if (pos < L) {  // pos >= L -> nothing written, nothing counted
        float* out0 = row.logits + static_cast<size_t>(q0) * static_cast<size_t>(stride) +
                      static_cast<size_t>(pos);  // (2h)*stride + pos
        *out0 = acc0;
        *(out0 + stride) = acc1;
        if (need_hist) {
          const uint32_t b0 = score_bin16(acc0);
          const uint32_t b1 = score_bin16(acc1);
          const int lane_row = ctx.tid & (kKTileRows - 1);
          if constexpr (SparseComputeOnly) {
            bin_cache[(it * (kSeqLen / 2) + half) * kKTileRows + lane_row] = b0 | (b1 << 8);
          } else {
            if (it < static_cast<int>(smem::kBin8Tiles)) {
              uint8_t* bin8 = smem_at<uint8_t>(ctx.smem, smem::kBin8) +
                              (it * kSeqLen + q0) * kKTileRows + lane_row;
              bin8[0] = static_cast<uint8_t>(b0);
              bin8[kKTileRows] = static_cast<uint8_t>(b1);
            }
            if (it < static_cast<int>(smem::kLogitCacheTiles)) {
              float* lc = smem_at<float>(ctx.smem, smem::kLogitCache) +
                          (it * kSeqLen + q0) * kKTileRows + lane_row;
              lc[0] = acc0;
              lc[kKTileRows] = acc1;
            }
          }
          if (pos <= causal0) p4_hist_inc(hist0 + b0 * 4u);
          if (pos <= causal1) p4_hist_inc(hist1 + b1 * 4u);
        }
      }
      if (long_context(ctx.cur_pos)) {
        uint64_t* maxima = reinterpret_cast<uint64_t*>(row.cand_ws);
        const int gs = long_topk::summary_stride(L);
        long_topk::publish_group(acc0, pos, causal0 + 1, L, maxima + size_t(q0) * gs);
        long_topk::publish_group(acc1, pos, causal1 + 1, L, maxima + size_t(q0 + 1) * gs);
      }
      tcgen05_fence_before_thread_sync();  // this thread's TMEM read is complete (wait::ld) before
                                           // the barrier the next MMA follows
      consumer_sync();
    }
    ctx.item = item0 + static_cast<uint32_t>(n_tiles);
  }
  p4_tmem_dealloc(ctx);  // single dealloc site, null mode included
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// ---- constants of the threshold rule
// --------------------------------------------------------------------------
constexpr uint32_t kP5SureThreshold = 0x801u;
constexpr uint32_t kP5SureLimit = 0x800u;
static_assert(kP5SureLimit == uint32_t(kTopk) && kP5SureThreshold == kP5SureLimit + 1u,
              "threshold rule H[b] > 2048 >= H[b+1]");
static_assert(kHistSureIndex == 256 && kHistCandIndex == 257 && kHistBlockDwords == 258,
              "H / hist block layout [256 bins][n_sure][n_cand]");

// ---- step 3: suffix scan of the local histogram
// ------------------------------------------------------------- Leading barrier, then per (q, d):
// load hist[q][tid], hist[q][tid+d] (tid <= 255-d); barrier; add + store hist[q][tid]; barrier. The
// [256]/[257] words are never read or written here (tid + d <= 255).
__device__ __forceinline__ void p5_suffix_scan(OpCtx& ctx, uint32_t* hist) {
  // Stage-A shared atomics must finish before loading the four bins owned by this thread. Each warp
  // then scans its 32-bin segment entirely in registers. Warp 0 scans the 8 segment totals for all
  // four queries in four independent width-8 groups.
  consumer_sync();
  uint32_t sum0 = hist[0 * kHistBlockDwords + ctx.tid];
  uint32_t sum1 = hist[1 * kHistBlockDwords + ctx.tid];
  uint32_t sum2 = hist[2 * kHistBlockDwords + ctx.tid];
  uint32_t sum3 = hist[3 * kHistBlockDwords + ctx.tid];
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t v0 = __shfl_down_sync(0xffffffffu, sum0, offset);
    const uint32_t v1 = __shfl_down_sync(0xffffffffu, sum1, offset);
    const uint32_t v2 = __shfl_down_sync(0xffffffffu, sum2, offset);
    const uint32_t v3 = __shfl_down_sync(0xffffffffu, sum3, offset);
    if (ctx.lane + offset < 32) {
      sum0 += v0;
      sum1 += v1;
      sum2 += v2;
      sum3 += v3;
    }
  }

  uint32_t* warp_totals = smem_at<uint32_t>(ctx.smem, smem::kG2bHist2);
  if (ctx.lane == 0) {
    warp_totals[0 * 8 + ctx.warp] = sum0;
    warp_totals[1 * 8 + ctx.warp] = sum1;
    warp_totals[2 * 8 + ctx.warp] = sum2;
    warp_totals[3 * 8 + ctx.warp] = sum3;
  }
  consumer_sync();

  if (ctx.warp == 0) {
    const int warp_in_query = ctx.lane & 7;
    const uint32_t own = warp_totals[ctx.lane];
    uint32_t suffix = own;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, suffix, offset, 8);
      if (warp_in_query + offset < 8) suffix += v;
    }
    warp_totals[ctx.lane] = suffix - own;
  }
  consumer_sync();

  hist[0 * kHistBlockDwords + ctx.tid] = sum0 + warp_totals[0 * 8 + ctx.warp];
  hist[1 * kHistBlockDwords + ctx.tid] = sum1 + warp_totals[1 * 8 + ctx.warp];
  hist[2 * kHistBlockDwords + ctx.tid] = sum2 + warp_totals[2 * 8 + ctx.warp];
  hist[3 * kHistBlockDwords + ctx.tid] = sum3 + warp_totals[3 * 8 + ctx.warp];
}

// ---- step 4 (between the S barriers): global H accumulation
// ------------------------------------------------ Every thread: 4 relaxed atomic adds (result
// unused) H[q*258 + tid] += hist[q][tid], q = 0,1,2,3.  Bins 0..255 only (256 threads); [256]/[257]
// untouched.

__device__ __forceinline__ void p5_accumulate_hist_block(OpCtx& ctx, const uint32_t* hist,
                                                         uint32_t* hist_blk) {
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    const uint32_t v = hist[q * kHistBlockDwords + ctx.tid];  // hist[q][tid]
    if (v != 0u) atomicAdd(hist_blk + q * kHistBlockDwords + ctx.tid, v);
  }
}

// ---- step 5: threshold bin per q
// --------------------------------------------------------------------------- tid: H[q][tid] >=
// 0x801 -> load H[q][tid+1]; <= 0x800 -> kThrBin[q] = tid (smem window 0x32070 + 2q). Exactly one
// thread qualifies for a sparse q (the suffix counts are non-increasing and drop from >= 0x801 to
// <= 0x800 once); none for a dense q (H[q][b] <= 2048 everywhere). tid == 255 reads H[q][256] = the
// n_sure ticket word (<= 0x800 at all times: the tickets sum to H[q][thr+1] <= 0x800), harmless.
// Plain (weak) loads: the S#2 acquire + barrier ordered every CTA's atomics before them.
__device__ __forceinline__ void p5_threshold_bins(OpCtx& ctx, const uint32_t* hist_blk,
                                                  uint16_t* thr_bin) {
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    const uint32_t* Hq = hist_blk + q * kHistBlockDwords;
    const uint32_t above = Hq[ctx.tid];
    if (above >= kP5SureThreshold) {
      const uint32_t next = Hq[ctx.tid + 1];
      if (!(next > kP5SureLimit)) thr_bin[q] = static_cast<uint16_t>(ctx.tid);
    }
  }
  consumer_sync();  // barrier
}

// H contains raw bins in the static sparse path.  Every CTA reconstructs the same global suffix
// values after the second S barrier generation, avoiding an additional cross-CTA publication
// barrier.  The four coalesced H loads replace p5_threshold_bins' four suffix loads; only two extra
// local barriers are added.
__device__ __forceinline__ void p5_threshold_bins_raw(OpCtx& ctx, const uint32_t* hist_blk,
                                                      uint16_t* thr_bin) {
  uint32_t sum0 = hist_blk[0 * kHistBlockDwords + ctx.tid];
  uint32_t sum1 = hist_blk[1 * kHistBlockDwords + ctx.tid];
  uint32_t sum2 = hist_blk[2 * kHistBlockDwords + ctx.tid];
  uint32_t sum3 = hist_blk[3 * kHistBlockDwords + ctx.tid];
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t v0 = __shfl_down_sync(0xffffffffu, sum0, offset);
    const uint32_t v1 = __shfl_down_sync(0xffffffffu, sum1, offset);
    const uint32_t v2 = __shfl_down_sync(0xffffffffu, sum2, offset);
    const uint32_t v3 = __shfl_down_sync(0xffffffffu, sum3, offset);
    if (ctx.lane + offset < 32) {
      sum0 += v0;
      sum1 += v1;
      sum2 += v2;
      sum3 += v3;
    }
  }

  uint32_t* warp_totals = smem_at<uint32_t>(ctx.smem, smem::kG2bHist2);
  if (ctx.lane == 0) {
    warp_totals[0 * 8 + ctx.warp] = sum0;
    warp_totals[1 * 8 + ctx.warp] = sum1;
    warp_totals[2 * 8 + ctx.warp] = sum2;
    warp_totals[3 * 8 + ctx.warp] = sum3;
  }
  consumer_sync();

  if (ctx.warp == 0) {
    const int warp_in_query = ctx.lane & 7;
    const uint32_t own = warp_totals[ctx.lane];
    uint32_t suffix = own;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, suffix, offset, 8);
      if (warp_in_query + offset < 8) suffix += v;
    }
    warp_totals[ctx.lane] = suffix - own;
  }
  consumer_sync();

  sum0 += warp_totals[0 * 8 + ctx.warp];
  sum1 += warp_totals[1 * 8 + ctx.warp];
  sum2 += warp_totals[2 * 8 + ctx.warp];
  sum3 += warp_totals[3 * 8 + ctx.warp];
  uint32_t next0 = __shfl_down_sync(0xffffffffu, sum0, 1);
  uint32_t next1 = __shfl_down_sync(0xffffffffu, sum1, 1);
  uint32_t next2 = __shfl_down_sync(0xffffffffu, sum2, 1);
  uint32_t next3 = __shfl_down_sync(0xffffffffu, sum3, 1);
  if (ctx.lane == 31) {
    next0 = warp_totals[0 * 8 + ctx.warp];
    next1 = warp_totals[1 * 8 + ctx.warp];
    next2 = warp_totals[2 * 8 + ctx.warp];
    next3 = warp_totals[3 * 8 + ctx.warp];
  }
  if (sum0 >= kP5SureThreshold && next0 <= kP5SureLimit)
    thr_bin[0] = static_cast<uint16_t>(ctx.tid);
  if (sum1 >= kP5SureThreshold && next1 <= kP5SureLimit)
    thr_bin[1] = static_cast<uint16_t>(ctx.tid);
  if (sum2 >= kP5SureThreshold && next2 <= kP5SureLimit)
    thr_bin[2] = static_cast<uint16_t>(ctx.tid);
  if (sum3 >= kP5SureThreshold && next3 <= kP5SureLimit)
    thr_bin[3] = static_cast<uint16_t>(ctx.tid);
  consumer_sync();
}

// Fused static-path scan: reconstruct this CTA's local suffix histogram and
// the globally merged raw-bin suffixes under the same three CTA barriers.
// Warp 0 scans the 32 local segment totals while warp 1 scans the 32 global
// totals, cutting the two independent scan phases' barrier count in half.
__device__ __forceinline__ void p5_dual_suffix_threshold(OpCtx& ctx, uint32_t* hist,
                                                         const uint32_t* hist_blk,
                                                         uint16_t* thr_bin) {
  uint32_t l0 = hist[0 * kHistBlockDwords + ctx.tid];
  uint32_t l1 = hist[1 * kHistBlockDwords + ctx.tid];
  uint32_t l2 = hist[2 * kHistBlockDwords + ctx.tid];
  uint32_t l3 = hist[3 * kHistBlockDwords + ctx.tid];
  uint32_t g0 = hist_blk[0 * kHistBlockDwords + ctx.tid];
  uint32_t g1 = hist_blk[1 * kHistBlockDwords + ctx.tid];
  uint32_t g2 = hist_blk[2 * kHistBlockDwords + ctx.tid];
  uint32_t g3 = hist_blk[3 * kHistBlockDwords + ctx.tid];
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t lv0 = __shfl_down_sync(0xffffffffu, l0, offset);
    const uint32_t lv1 = __shfl_down_sync(0xffffffffu, l1, offset);
    const uint32_t lv2 = __shfl_down_sync(0xffffffffu, l2, offset);
    const uint32_t lv3 = __shfl_down_sync(0xffffffffu, l3, offset);
    const uint32_t gv0 = __shfl_down_sync(0xffffffffu, g0, offset);
    const uint32_t gv1 = __shfl_down_sync(0xffffffffu, g1, offset);
    const uint32_t gv2 = __shfl_down_sync(0xffffffffu, g2, offset);
    const uint32_t gv3 = __shfl_down_sync(0xffffffffu, g3, offset);
    if (ctx.lane + offset < 32) {
      l0 += lv0;
      l1 += lv1;
      l2 += lv2;
      l3 += lv3;
      g0 += gv0;
      g1 += gv1;
      g2 += gv2;
      g3 += gv3;
    }
  }

  uint32_t* segment = smem_at<uint32_t>(ctx.smem, smem::kG2bHist2);
  if (ctx.lane == 0) {
    segment[0 * 8 + ctx.warp] = l0;
    segment[1 * 8 + ctx.warp] = l1;
    segment[2 * 8 + ctx.warp] = l2;
    segment[3 * 8 + ctx.warp] = l3;
    segment[32 + 0 * 8 + ctx.warp] = g0;
    segment[32 + 1 * 8 + ctx.warp] = g1;
    segment[32 + 2 * 8 + ctx.warp] = g2;
    segment[32 + 3 * 8 + ctx.warp] = g3;
  }
  consumer_sync();

  if (ctx.warp < 2) {
    const int base = ctx.warp * 32;
    const int warp_in_query = ctx.lane & 7;
    const uint32_t own = segment[base + ctx.lane];
    uint32_t suffix = own;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, suffix, offset, 8);
      if (warp_in_query + offset < 8) suffix += v;
    }
    segment[base + ctx.lane] = suffix - own;
  }
  consumer_sync();

  l0 += segment[0 * 8 + ctx.warp];
  l1 += segment[1 * 8 + ctx.warp];
  l2 += segment[2 * 8 + ctx.warp];
  l3 += segment[3 * 8 + ctx.warp];
  g0 += segment[32 + 0 * 8 + ctx.warp];
  g1 += segment[32 + 1 * 8 + ctx.warp];
  g2 += segment[32 + 2 * 8 + ctx.warp];
  g3 += segment[32 + 3 * 8 + ctx.warp];
  hist[0 * kHistBlockDwords + ctx.tid] = l0;
  hist[1 * kHistBlockDwords + ctx.tid] = l1;
  hist[2 * kHistBlockDwords + ctx.tid] = l2;
  hist[3 * kHistBlockDwords + ctx.tid] = l3;

  uint32_t n0 = __shfl_down_sync(0xffffffffu, g0, 1);
  uint32_t n1 = __shfl_down_sync(0xffffffffu, g1, 1);
  uint32_t n2 = __shfl_down_sync(0xffffffffu, g2, 1);
  uint32_t n3 = __shfl_down_sync(0xffffffffu, g3, 1);
  if (ctx.lane == 31) {
    n0 = segment[32 + 0 * 8 + ctx.warp];
    n1 = segment[32 + 1 * 8 + ctx.warp];
    n2 = segment[32 + 2 * 8 + ctx.warp];
    n3 = segment[32 + 3 * 8 + ctx.warp];
  }
  if (g0 >= kP5SureThreshold && n0 <= kP5SureLimit) thr_bin[0] = static_cast<uint16_t>(ctx.tid);
  if (g1 >= kP5SureThreshold && n1 <= kP5SureLimit) thr_bin[1] = static_cast<uint16_t>(ctx.tid);
  if (g2 >= kP5SureThreshold && n2 <= kP5SureLimit) thr_bin[2] = static_cast<uint16_t>(ctx.tid);
  if (g3 >= kP5SureThreshold && n3 <= kP5SureLimit) thr_bin[3] = static_cast<uint16_t>(ctx.tid);
  consumer_sync();
}

// ---- step 6: dense-q causal lengths
// ------------------------------------------------------------------------ cur_pos > 0x7ff ->
// nothing.  Otherwise (cta 0, tid 0) only: H[q][257] = cur_pos + q + 1 for q < n_dense = 2048 -
// cur_pos (cur_pos == 0x7ff -> q = 0 only).  n_dense is 1..3 here (cur_pos >= 0x7fd passed the
// participation test; at cur_pos == 0x7fc nobody reaches this code and Group2b takes len = cp + q +
// 1 from cur_pos itself).  Group2b acquires it through C4.
__device__ __forceinline__ void p5_dense_lengths(OpCtx& ctx, uint32_t* hist_blk, int n_dense) {
  if (ctx.cur_pos > kLastDenseCurPos) return;
  if ((ctx.tid | blockIdx.x) != 0u) return;  // only (cta 0, tid 0)
  for (int q = 0; q < n_dense; ++q) {        // q < n_dense
    hist_blk[q * kHistBlockDwords + kHistCandIndex] = static_cast<uint32_t>(ctx.cur_pos + q + 1);
  }
}

// ---- step 7: tickets
// ------------------------------------------------------------------------------------------ Thread
// tid = q for q in [max(n_dense, 0), 3] (skip if n_dense > tid, signed, or tid > 3): thr =
// kThrBin[q]; this CTA's sure count hist[q][thr+1] and candidate count hist[q][thr] -
// hist[q][thr+1] take tickets sure_base = atomicAdd(H[q][256]) and cand_base =
// atomicAdd(H[q][257]); barrier.  Relaxed gpu-scope atomics with return value.
__device__ __forceinline__ void p5_tickets(OpCtx& ctx, const uint32_t* hist, uint32_t* hist_blk,
                                           const uint16_t* thr_bin, uint32_t* sure_base,
                                           uint32_t* cand_base, int n_dense) {
  if (!(n_dense > ctx.tid) && ctx.tid <= kSeqLen - 1) {  // tid == q in [max(n_dense, 0), 3]
    const int q = ctx.tid;
    const uint32_t thr = thr_bin[q];
    const uint32_t* hq = hist + q * kHistBlockDwords;
    const uint32_t n_above = hq[thr + 1];  // hist[q][thr+1]: this CTA's sure count for q
    const uint32_t n_thr =
        hq[thr] - n_above;  // hist[q][thr] - hist[q][thr+1]: this CTA's candidates
    uint32_t* Hq = hist_blk + q * kHistBlockDwords;
    sure_base[q] = atom_relaxed_gpu_add(Hq + kHistSureIndex, n_above);  //  ->
    cand_base[q] = atom_relaxed_gpu_add(Hq + kHistCandIndex, n_thr);    //  ->
  }
  consumer_sync();  // barrier
}

// ---- step 8: write-out
// --------------------------------------------------------------------------------------- q from
// max(n_dense, 0) to 3.  Per q: n = hist[q][0] (tid >= n skips), last = cur_pos + q, thr, cand_row
// = q*stride + cand_base, sure_row = q*2048 + sure_base.  Loop i = tid; i < n; i += 256: pos = ((i
// & ~0x7f) + c) * 128 + (tid & 0x7f); pos > last -> skip (never taken: pos is monotone in i and n
// counts exactly the causal rows); bin > thr: selects[sure_row + atomicAdd(hist[q][bin+1])] = pos;
// bin == thr: cand_ws[cand_row + atomicAdd(hist[q][257])] = pos.  hist[q][0] is never a cursor
// (bin+1 >= 1), so the bound is stable.
template <bool SparseComputeOnly>
__device__ __forceinline__ void p5_scatter(const IndexerArgs& args, OpCtx& ctx, uint32_t* hist,
                                           const CtxRow4& row, const uint16_t* thr_bin,
                                           const uint32_t* sure_base, const uint32_t* cand_base,
                                           int n_dense) {
  const int logits_stride =
      SparseComputeOnly ? total_len(ctx.cur_pos) : logits_row_stride(args.flags, ctx.cur_pos);
  const int ws_stride =
      SparseComputeOnly ? total_len(ctx.cur_pos) : ws_row_stride(args.flags, ctx.cur_pos);
  const int q0 = n_dense > 0 ? n_dense : 0;  // max(n_dense, 0)
  const int lane_row = ctx.tid & 0x7f;
  for (int q = q0; q < kSeqLen; ++q) {
    uint32_t* hq = hist + q * kHistBlockDwords;
    const int n = static_cast<int>(hq[0]);  // this CTA's causal count for q
    if (ctx.tid >= n) continue;
    const int last = ctx.cur_pos + q;
    const uint32_t thr = thr_bin[q];
    const int cand_row = q * ws_stride + static_cast<int>(cand_base[q]);  // q*L + cand_base
    const int sure_row = q * kTopk + static_cast<int>(sure_base[q]);      // q*2048 + sure_base
    const float* logits_q = row.logits + static_cast<size_t>(q) * logits_stride;
    const uint32_t* const bin_cache = smem_at<uint32_t>(ctx.smem, smem::kP5BinCache);
    for (int i = ctx.tid; i < n; i += kConsumerThreads) {
      const int pos =
          ((i & ~0x7f) + ctx.cta) * kKTileRows + lane_row;  // tile i>>7 of this CTA, row tid&0x7f
      if (pos > last) continue;                             //  (never-taken guard, kept)
      uint32_t b;
      if constexpr (SparseComputeOnly) {
        const int tile = i >> 7;
        const uint32_t packed =
            bin_cache[(tile * (kSeqLen / 2) + (q >> 1)) * kKTileRows + lane_row];
        b = (packed >> ((q & 1) * 8)) & 0xffu;
      } else {
        const int tile = i >> 7;
        if (tile < static_cast<int>(smem::kBin8Tiles)) {
          b = smem_at<uint8_t>(ctx.smem, smem::kBin8)[(tile * kSeqLen + q) * kKTileRows + lane_row];
        } else {
          b = score_bin16(logits_q[pos]);
        }
      }
      if (b > thr) {
        const uint32_t k = atomicAdd(hq + b + 1, 1u);
        row.selects[sure_row + static_cast<int>(k)] = pos;
      } else if (b == thr) {
        const uint32_t j = atomicAdd(hq + kHistCandIndex, 1u);
        row.cand_ws[cand_row + static_cast<int>(j)] = pos;
      }
    }
  }
}

// Ticket-free P5 (persistent kernel, cur_pos > 2047, all four queries sparse).  The raw-bin merge
// atomics return this CTA's offset inside every global bin of H.  After S the global exclusive
// suffix G(b) = sum of H over bins above b gives every row its final slot without a ticket round
// trip: a sure row of bin b goes to selects[q][G(b) + off(b) + i] and a threshold-bin row to
// cand_ws[q][off(thr) + i], i = the row's index among this CTA's rows of that bin (shared-memory
// cursor).  These are the same slot ranges the tickets produced up to a permutation (no ordering
// promise); n_sure = G(thr) and n_cand = H(thr) are written by CTA 0 into the H block's two ticket
// words before its C4 arrival, exactly what Group2b reads.
constexpr uint32_t kP5BaseSure = smem::kRing + 0x1000;  // u32 [4][256] (ring idle after P4)
constexpr uint32_t kP5CandOff = kP5BaseSure + kSeqLen * kHistBins * 4;  // u32 [4]
constexpr uint32_t kP5NCand =
    kP5CandOff + kSeqLen * 4;  // u32 [4] global count of the threshold bin
constexpr uint32_t kP5NSure =
    kP5NCand + kSeqLen * 4;  // u32 [4] global count above the threshold bin (sure IDs)
static_assert(smem::kG2bHist2 + 64 * 4 <= kP5BaseSure && kP5NSure + kSeqLen * 4 <= smem::kRingEnd,
              "ticket-free P5 scratch");

// Distributed exact refine of a large threshold bin ("dsel"; sparse persistent launches with LL
// ID publication). A threshold bin of more than kDistMin candidates is not handed to the single
// Group2b CTA, which would have to stream all of them.  Instead every scoring CTA refines the rows
// it already owns from its stage-A logit / bin caches: an MSB radix select over the unique 50-bit
// composite key32 << kIdBits | (kIdMask - id) (larger composite = larger key; equal keys keep the
// smaller id, the tie order of the reference selection), 8 bits per round, with one global
// histogram merge and one grid barrier per round.  Rows above the selected digit go straight to
// their final selected_ids slots; once the selected digit's population fits kFinalCap it is staged
// for the Group2b CTA, which finishes it with the shared-memory exact select.  Nothing is
// truncated: rounds continue until the digit is taken whole or fits the staging, and a query still
// unresolved after kMaxRounds traps.
//   kDistMin   bins up to this size keep the Group2b-only path (<= the staging capacity)
//   kFinalCap  largest digit population handed to the Group2b select
//   kMaxRounds 50-bit composites need at most 7 rounds
//   kSpecLo/Hi composite bits 31..38 (key32 bits 13..20): the speculative first digit, histogrammed
//   while collecting candidates kListCap   per-CTA, per-query candidate list: every row a CTA owns
//   at kMaxSeqLen (13 tiles of 128 rows) kSyncWord  free word of the sync block used by the
//   per-round grid barrier
namespace dsel {
constexpr uint32_t kDistMin = 8192;
constexpr uint32_t kFinalCap = 8192;
constexpr int kMaxRounds = 8;
constexpr int kSpecLo = 31;
constexpr int kSpecHi = 38;
constexpr int kListCap = 13 * kKTileRows;
static_assert(kListCap >= (kMaxSeqLen + kKTileStrideRows - 1) / kKTileStrideRows * kKTileRows,
              "candidate list holds every row a scoring CTA can own at kMaxSeqLen");
constexpr int kSyncWord = 0x48;
static_assert(kSyncWord % 4 == 0 && kSyncWord < kSyncBlockBytes && kSyncWord != kSyncS &&
                  kSyncWord != kSyncC4 && kSyncWord != kSyncC5,
              "refine barrier word is a free word of the sync block");

// Global scratch behind the Group2b staging, inside the scratch region: per-round digit histograms,
// the merged AND / OR of the composites, CTA 0's result record for the Group2b CTA and the
// per-query done counters.  Cleared in stage B (dsel_clear_scratch).
constexpr size_t kArenaBase =
    (arena::kCompScratch + arena::kCompScratchBytes + size_t(kGroup2bWorkCtas) * 4096 * 8 + 127) &
    ~size_t(127);
constexpr size_t kHistG = kArenaBase;
constexpr size_t kHistGWords = size_t(kMaxRounds) * kSeqLen * kHistBins;
constexpr size_t kAndOr = kHistG + kHistGWords * 4;
constexpr size_t kResult = kAndOr + size_t(kSeqLen) * 16;
constexpr size_t kDone = kResult + size_t(kSeqLen) * 16;
constexpr size_t kEnd = kDone + size_t(kSeqLen) * 4;
static_assert(kAndOr % 16 == 0 && kResult % 16 == 0 &&
                  kEnd <= arena::kScratch + arena::kScratchBytes,
              "refine scratch fits the scratch region");
static_assert(kDistMin <= uint32_t(arena::kCompScratchCap) &&
                  kFinalCap <= uint32_t(arena::kCompScratchCap),
              "staging capacity");

// Shared scratch in the ring (idle after P4), behind the P5 scalars: candidate lists and counts,
// local / offset / rank / above digit histograms, the selected digit, scan totals, the AND / OR
// reduction and the per-query state.
constexpr uint32_t kSList = 0x6000;
constexpr uint32_t kSCnt = kSList + kSeqLen * kListCap * 8;
constexpr uint32_t kSHist = kSCnt + 64;
constexpr uint32_t kSOff = kSHist + kSeqLen * kHistBins * 4;
constexpr uint32_t kSRank = kSOff + kSeqLen * kHistBins * 4;
constexpr uint32_t kSAbove = kSRank + kSeqLen * kHistBins * 4;
constexpr uint32_t kSPick = kSAbove + kSeqLen * kHistBins * 4;
constexpr uint32_t kSTot = kSPick + kSeqLen * 16;
constexpr uint32_t kSRed = kSTot + kSeqLen * 8 * 4;
constexpr uint32_t kSState = kSRed + kConsumerWarps * kSeqLen * 16;
constexpr uint32_t kSEnd = kSState + kSeqLen * 128;
static_assert(kSList >= kP5NSure + kSeqLen * 4 && kSList % 16 == 0 && kSRed % 16 == 0 &&
                  kSState % 16 == 0 && kSEnd <= smem::kRingEnd,
              "refine shared memory fits the idle ring behind the P5 scratch");
}  // namespace dsel

__device__ __forceinline__ void p5_merge_raw_offsets(OpCtx& ctx, const uint32_t* hist,
                                                     uint32_t* hist_blk, uint32_t (&off)[kSeqLen]) {
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    const uint32_t* hq = hist + q * kHistBlockDwords;
    const uint32_t v = hq[ctx.tid] - hq[ctx.tid + 1];
    off[q] = v != 0u ? atomicAdd(hist_blk + q * kHistBlockDwords + ctx.tid, v) : 0u;
  }
}

__device__ __forceinline__ void p5_threshold_bases(OpCtx& ctx, const uint32_t* hist,
                                                   uint32_t* hist_blk, uint16_t* thr_bin,
                                                   const uint32_t (&off)[kSeqLen]) {
  uint32_t sum[kSeqLen];
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) sum[q] = __ldcg(hist_blk + q * kHistBlockDwords + ctx.tid);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, sum[q], offset);
      if (ctx.lane + offset < 32) sum[q] += v;
    }
  }
  uint32_t* warp_totals = smem_at<uint32_t>(ctx.smem, smem::kG2bHist2);
  if (ctx.lane == 0) {
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) warp_totals[q * 8 + ctx.warp] = sum[q];
  }
  consumer_sync();
  if (ctx.warp == 0) {
    const int warp_in_query = ctx.lane & 7;
    const uint32_t own = warp_totals[ctx.lane];
    uint32_t suffix = own;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, suffix, offset, 8);
      if (warp_in_query + offset < 8) suffix += v;
    }
    warp_totals[ctx.lane] = suffix - own;
  }
  consumer_sync();
  uint32_t next[kSeqLen];
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    sum[q] += warp_totals[q * 8 + ctx.warp];
    next[q] = __shfl_down_sync(0xffffffffu, sum[q], 1);
  }
  if (ctx.lane == 31) {
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) next[q] = warp_totals[q * 8 + ctx.warp];
  }
  uint32_t* base_sure = smem_at<uint32_t>(ctx.smem, kP5BaseSure);
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    if (sum[q] >= kP5SureThreshold && next[q] <= kP5SureLimit)
      thr_bin[q] = static_cast<uint16_t>(ctx.tid);
    base_sure[q * kHistBins + ctx.tid] =
        next[q] + off[q] - hist[q * kHistBlockDwords + ctx.tid + 1];
  }
  consumer_sync();
  uint32_t* cand_off = smem_at<uint32_t>(ctx.smem, kP5CandOff);
  uint32_t* n_cand_s = smem_at<uint32_t>(ctx.smem, kP5NCand);
  uint32_t* n_sure_s = smem_at<uint32_t>(ctx.smem, kP5NSure);
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    if (static_cast<uint32_t>(ctx.tid) == static_cast<uint32_t>(thr_bin[q])) {
      cand_off[q] = off[q];
      n_cand_s[q] = sum[q] - next[q];
      n_sure_s[q] = next[q];  // rows above the threshold bin: dsel's first free slot
      if (blockIdx.x == 0) {
        hist_blk[q * kHistBlockDwords + kHistSureIndex] = next[q];
        hist_blk[q * kHistBlockDwords + kHistCandIndex] = sum[q] - next[q];
      }
    }
  }
  consumer_sync();
}

__device__ __forceinline__ void p5_scatter_ticketfree(const IndexerArgs& args, OpCtx& ctx,
                                                      uint32_t* hist, const CtxRow4& row,
                                                      const uint16_t* thr_bin) {
  const int logits_stride = logits_row_stride(args.flags, ctx.cur_pos);
  const int ws_stride = ws_row_stride(args.flags, ctx.cur_pos);
  const int lane_row = ctx.tid & 0x7f;
  const uint32_t* base_sure = smem_at<uint32_t>(ctx.smem, kP5BaseSure);
  const uint32_t* cand_off = smem_at<uint32_t>(ctx.smem, kP5CandOff);
  const uint32_t* n_cand_s = smem_at<uint32_t>(ctx.smem, kP5NCand);
  const bool stage_comp = g2b_refine_fast(args, ctx.cur_pos);
  // Row i of query q (tile i >> 7, lane row i & 127, i < n = this CTA's counted rows) is visited
  // exactly once; a warp takes one (query, tile) pair per step and each lane checks four
  // consecutive rows from one 32-bit word of the bin cache.
  (void)lane_row;
  const int n_tiles = k_tiles_of_cta(ctx.cur_pos, ctx.cta);
  const uint64_t tag_bits = comp_tag(ctx.epoch) << kCompTagShift;
  // dsel_run collects a distributed query's threshold-bin rows itself, so phase 0 does not stage
  // them.
  const bool dsel_on = stage_comp;
  // Phase 0 writes the threshold-bin candidates (what the Group2b refine polls for), phase 1 the
  // sure IDs (published after C4).
#pragma unroll 1
  for (int phase = 0; phase < 2; ++phase) {
    for (int pair = ctx.warp; pair < kSeqLen * n_tiles; pair += kConsumerWarps) {
      const int q = pair / n_tiles;
      const int tile = pair - q * n_tiles;
      if (phase == 0 && dsel_on && n_cand_s[q] > dsel::kDistMin) continue;
      uint32_t* hq = hist + q * kHistBlockDwords;
      const int n = static_cast<int>(hq[0]);
      const int i0 = tile * kKTileRows + 4 * ctx.lane;
      if (i0 >= n) continue;
      const int last = ctx.cur_pos + q;
      const uint32_t thr = thr_bin[q];
      const uint32_t* base_q = base_sure + q * kHistBins;
      int32_t* const sel_q = row.selects + q * kTopk;
      int32_t* const cand_q = row.cand_ws + static_cast<size_t>(q) * ws_stride + cand_off[q];
      const float* logits_q = row.logits + static_cast<size_t>(q) * logits_stride;
      uint32_t bins4 = 0u;
      if (tile < static_cast<int>(smem::kBin8Tiles)) {
        bins4 = *smem_at<uint32_t>(
            ctx.smem,
            smem::kBin8 + static_cast<uint32_t>((tile * kSeqLen + q) * kKTileRows + 4 * ctx.lane));
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = i0 + u;
        if (i >= n) break;
        const int lane_row_u = 4 * ctx.lane + u;
        const int pos = (tile * kP4Ctas + ctx.cta) * kKTileRows + lane_row_u;
        if (pos > last) continue;
        uint32_t b;
        if (tile < static_cast<int>(smem::kBin8Tiles)) {
          b = (bins4 >> (8 * u)) & 0xffu;
        } else {
          b = score_bin16(logits_q[pos]);
        }
        if (phase != 0) {
          if (b > thr) {
            const uint32_t k = atomicAdd(hq + b + 1, 1u);
            sel_q[base_q[b] + k] = pos;
          }
        } else if (b == thr) {
          const uint32_t j = atomicAdd(hq + kHistCandIndex, 1u);
          cand_q[j] = pos;
          if (stage_comp && n_cand_s[q] <= static_cast<uint32_t>(arena::kCompScratchCap)) {
            // Same composite the refine builds from the logit, tagged with this launch
            // (op_g2b_refine_fast polls for it).
            const float v =
                tile < static_cast<int>(smem::kLogitCacheTiles)
                    ? smem_at<const float>(
                          ctx.smem,
                          smem::kLogitCache)[(tile * kSeqLen + q) * kKTileRows + lane_row_u]
                    : logits_q[pos];
            st_relaxed_gpu_u64(reinterpret_cast<uint64_t*>(ctx.arena + arena::kCompScratch) +
                                   size_t(q) * arena::kCompScratchCap + cand_off[q] + j,
                               (static_cast<uint64_t>(score_key32(v)) << 18) |
                                   ((1u << 18) - 1u - static_cast<uint32_t>(pos)) | tag_bits);
          }
        }
      }
    }
  }
}

// Resets the dsel global scratch (round histograms, AND = ~0 / OR = 0, done counters).  Every
// active CTA clears a slice in stage B, before the S barrier; dsel_run only touches it after that
// barrier.
__device__ __forceinline__ void dsel_clear_scratch(const OpCtx& ctx, int n_active) {
  uint32_t* g = reinterpret_cast<uint32_t*>(ctx.arena + dsel::kHistG);
  for (int i = ctx.cta * kConsumerThreads + ctx.tid; i < static_cast<int>(dsel::kHistGWords);
       i += n_active * kConsumerThreads)
    g[i] = 0u;
  if (ctx.cta == 0 && ctx.tid < 2 * kSeqLen) {
    reinterpret_cast<uint64_t*>(ctx.arena + dsel::kAndOr)[ctx.tid] =
        (ctx.tid & 1) ? uint64_t{0} : ~uint64_t{0};
    if (ctx.tid < kSeqLen) reinterpret_cast<uint32_t*>(ctx.arena + dsel::kDone)[ctx.tid] = 0u;
  }
}

template <bool SparseComputeOnly, bool EarlyClear = false>
static __device__ void op_p5_select_stage_b(const IndexerArgs& args, OpCtx& ctx) {
  static_assert(!(SparseComputeOnly && EarlyClear),
                "the compute-only form keeps its own histogram clear");
  if constexpr (!SparseComputeOnly) {
    if (!ctx.sparse) return;  // dense launches never enter (consumer_main: sparse steps only)
  }
  const CtxRow4& row = ctx_score(ctx.smem);
  if constexpr (!EarlyClear) {
    if (blockIdx.x == 0) hist_block_clear(ctx, row.hist_blk);
  }
  const int n_active = n_active_ctas(ctx.cur_pos);  // min((cur_pos + 0x83) >> 7, 0x80)
  if constexpr (!SparseComputeOnly) {
    if (ctx.cta >= n_active || ctx.cur_pos < kSelectCurPos)
      return;  // -> C4 arrival in consumer_main
  }
  uint32_t* hist = smem_at<uint32_t>(ctx.smem, smem::kHist);           // [4][258] window 0x30030
  uint16_t* thr_bin = smem_at<uint16_t>(ctx.smem, smem::kThrBin);      // u16[4] window 0x32070
  uint32_t* sure_base = smem_at<uint32_t>(ctx.smem, smem::kSureBase);  // u32[4] window 0x32080
  uint32_t* cand_base = smem_at<uint32_t>(ctx.smem, smem::kCandBase);  // u32[4] window 0x320a0
  uint32_t sense_word = 0u;
  (void)sense_word;
  uint32_t off[kSeqLen] = {0u, 0u, 0u, 0u};
  (void)off;
  if constexpr (SparseComputeOnly) {
    sense_barrier(ctx, n_active, blockIdx.x == 0);
    p5_accumulate_hist_block(ctx, hist,
                             row.hist_blk);  // merge raw bins before the local suffix transform
  } else {
    if constexpr (EarlyClear) {
      if (ctx.tid == 0) sense_word = load_relaxed_gpu(sync_word(ctx, kSyncS));
    }
    if (!ctx.null_math) p5_suffix_scan(ctx, hist);  // step 3
    if constexpr (!EarlyClear) {
      sense_barrier(ctx, n_active, blockIdx.x == 0);
    } else {
      consumer_sync();
    }
    if (!ctx.null_math) {
      if constexpr (EarlyClear) {
        p5_merge_raw_offsets(ctx, hist, row.hist_blk, off);
        if (g2b_refine_fast(args, ctx.cur_pos))
          dsel_clear_scratch(ctx, n_active);  // this launch's dsel scratch
      } else {
        p5_accumulate_hist_block(ctx, hist, row.hist_blk);
      }
    }
  }
  if constexpr (EarlyClear && !SparseComputeOnly) {
    sense_barrier_presensed(ctx, n_active, blockIdx.x == 0, sense_word);
  } else {
    sense_barrier(ctx, n_active, blockIdx.x == 0);
  }
  if constexpr (!SparseComputeOnly) {
    if (ctx.null_math) return;
  }
  if constexpr (!SparseComputeOnly && EarlyClear) {
    p5_threshold_bases(ctx, hist, row.hist_blk, thr_bin, off);
    p5_scatter_ticketfree(args, ctx, hist, row, thr_bin);
    return;
  }
  if constexpr (SparseComputeOnly) {
    p5_dual_suffix_threshold(ctx, hist, row.hist_blk, thr_bin);
  } else if constexpr (EarlyClear) {
    p5_threshold_bins_raw(ctx, row.hist_blk, thr_bin);
  } else {
    p5_threshold_bins(ctx, row.hist_blk, thr_bin);  // step 5  (+ barrier)
  }
  const int n_dense =
      SparseComputeOnly ? 0 : kTopk - ctx.cur_pos;  // 2048 - cur_pos; <= 0 when every q is sparse
  if constexpr (!SparseComputeOnly) p5_dense_lengths(ctx, row.hist_blk, n_dense);  // step 6
  p5_tickets(ctx, hist, row.hist_blk, thr_bin, sure_base, cand_base,
             n_dense);  // step 7  (+ barrier)
  p5_scatter<SparseComputeOnly>(args, ctx, hist, row, thr_bin, sure_base, cand_base,
                                n_dense);  // step 8
  // consumer_main's C4 arrive follows
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

// Group2b: selection stage C, exact selection inside the threshold bin (CTA 132 + q refines query
// q; 136..147 idle; all 384 threads; sparse launches only).  Entered after the C4 wait;
// group2b_main arrives C5 (+0x10) afterwards.

namespace flashinfer::glm5_tp8::indexer_q4 {

static_assert(smem::kG2bList1 == smem::kG2bList0 + kRefineMaxCand * 4,
              "list[1] = list[0] + 16384 int32 (window 0x4430 / 0x14430: (r&1) << 14 slots , "
              "(r&1)*0x10000 B )");
static_assert(smem::kG2bThr + 2 <= smem::kG2bHist2 &&
                  smem::kG2bHist2 + (kHistBins + 1) * 4 <= smem::kG2bCnt &&
                  smem::kG2bCnt + 8 <= smem::kG2bList0,
              "thr u16  | hist2[257] ..0x4414 | cnt[2]  | list0");

static_assert(kMaxSeqLen <= 0x01000000, "packed G2b candidate IDs must fit in 24 bits");

// ---- helpers (prefix g2b_)
// -----------------------------------------------------------------------------------------------------
__device__ __forceinline__ float g2b_load_logit(const float* address) { return __ldg(address); }

__device__ __forceinline__ void g2b_insert_bottom3(uint64_t x, uint64_t (&best)[3]) {
  if (x >= best[2]) return;
  if (x < best[0]) {
    best[2] = best[1];
    best[1] = best[0];
    best[0] = x;
  } else if (x < best[1]) {
    best[2] = best[1];
    best[1] = x;
  } else {
    best[2] = x;
  }
}

__device__ __forceinline__ uint64_t g2b_shfl_down_u64(uint64_t x, int delta) {
  const uint32_t lo = __shfl_down_sync(0xffffffffu, static_cast<uint32_t>(x), delta);
  const uint32_t hi = __shfl_down_sync(0xffffffffu, static_cast<uint32_t>(x >> 32), delta);
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// For cur_pos 2044..2047 each causal row contains at most three more than
// kTopk elements. Exact top-k is the ascending ID range with the globally
// smallest 0..3 (score-key, id) pairs removed.
template <bool SparseComputeOnly>
static __device__ __noinline__ void op_g2b_near_topk(const IndexerArgs& args, OpCtx& ctx_in) {
  const OpCtx ctx = ctx_in;  // register copy (never written here), see op_g2b_refine_fast
  if constexpr (!SparseComputeOnly) {
    if (ctx.cta >= kGroup2bWorkCtas || ctx.null_math) return;
  }
  constexpr int kWarps = kBlock / 32;
  const int q = ctx.cta;
  const int len = ctx.cur_pos + q + 1;
  const int n_bottom = len - kTopk;
  const CtxRow5& row = ctx_refine(ctx.smem);
  int32_t* out = row.selects + static_cast<size_t>(q) * kTopk;

  if (n_bottom <= 0) {
    for (int id = ctx.tid; id < len; id += kBlock) out[id] = id;
    return;
  }

  const int logits_stride =
      SparseComputeOnly ? total_len(ctx.cur_pos) : logits_row_stride(args.flags, ctx.cur_pos);
  const float* logits_q = row.logits + static_cast<size_t>(q) * logits_stride;
  constexpr uint64_t kInvalid = ~uint64_t{0};
  uint64_t best[3] = {kInvalid, kInvalid, kInvalid};
  for (int id = ctx.tid; id < len; id += kBlock) {
    const uint64_t packed =
        (static_cast<uint64_t>(score_key32(g2b_load_logit(logits_q + id))) << 32) |
        static_cast<uint32_t>(id);
    g2b_insert_bottom3(packed, best);
  }

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const uint64_t x0 = g2b_shfl_down_u64(best[0], delta);
    const uint64_t x1 = g2b_shfl_down_u64(best[1], delta);
    const uint64_t x2 = g2b_shfl_down_u64(best[2], delta);
    if (ctx.lane + delta < 32) {
      g2b_insert_bottom3(x0, best);
      g2b_insert_bottom3(x1, best);
      g2b_insert_bottom3(x2, best);
    }
  }

  uint64_t* warp_bottom = smem_at<uint64_t>(ctx.smem, smem::kG2bList0);
  if (ctx.lane == 0) {
    warp_bottom[ctx.warp * 3 + 0] = best[0];
    warp_bottom[ctx.warp * 3 + 1] = best[1];
    warp_bottom[ctx.warp * 3 + 2] = best[2];
  }
  group_sync(ctx);

  uint64_t block_best[3] = {kInvalid, kInvalid, kInvalid};
  if (ctx.warp == 0) {
    if (ctx.lane < kWarps) {
      block_best[0] = warp_bottom[ctx.lane * 3 + 0];
      block_best[1] = warp_bottom[ctx.lane * 3 + 1];
      block_best[2] = warp_bottom[ctx.lane * 3 + 2];
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      const uint64_t x0 = g2b_shfl_down_u64(block_best[0], delta);
      const uint64_t x1 = g2b_shfl_down_u64(block_best[1], delta);
      const uint64_t x2 = g2b_shfl_down_u64(block_best[2], delta);
      if (ctx.lane + delta < 32) {
        g2b_insert_bottom3(x0, block_best);
        g2b_insert_bottom3(x1, block_best);
        g2b_insert_bottom3(x2, block_best);
      }
    }
  }

  int32_t* excluded = reinterpret_cast<int32_t*>(warp_bottom + kWarps * 3);
  if (ctx.tid == 0) {
    int ids[3] = {static_cast<int>(static_cast<uint32_t>(block_best[0])),
                  static_cast<int>(static_cast<uint32_t>(block_best[1])),
                  static_cast<int>(static_cast<uint32_t>(block_best[2]))};
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
      for (int j = i + 1; j < 3; ++j) {
        if (j < n_bottom && ids[j] < ids[i]) {
          const int tmp = ids[i];
          ids[i] = ids[j];
          ids[j] = tmp;
        }
      }
      excluded[i] = i < n_bottom ? ids[i] : 0x7fffffff;
    }
  }
  group_sync(ctx);

  const int e0 = excluded[0], e1 = excluded[1], e2 = excluded[2];
  for (int slot = ctx.tid; slot < kTopk; slot += kBlock) {
    int id = slot;
    if (id >= e0) ++id;
    if (id >= e1) ++id;
    if (id >= e2) ++id;
    out[slot] = id;
  }
}

// Per-CTA state of one refine (CTA-uniform values, loaded once after the C4 wait).
struct G2bState {
  int q;
  int cp;
  int n_sure;             // H[q][256]
  int len;                // (cp >= 0x7fd) ? H[q][257] : cp + q + 1
  int k_rem0;             // 0x800 - n_sure
  int32_t* out;           // &selects[q][n_sure]
  const float* logits_q;  // logits + q*stride
  const int32_t* cand_q;  // cand_ws + q*stride
  uint32_t* hist2;        // smem kG2bHist2 [257] (window 0x4010)
  uint32_t* cnt;          // smem kG2bCnt [2] (window 0x4420 / 0x4424)
};

// Dense q (len <= k_rem0): selects[q][n_sure + i] = i for i = tid; i < len; i += 384; [len, 2048)
// is NOT written. n_sure == 0 for a dense q (CTA 0 cleared H, no ticket: P5 steps 1 / 7).
__device__ __forceinline__ void g2b_dense_arange(const OpCtx& ctx, const G2bState& s) {
  for (int i = ctx.tid; i < s.len; i += kBlock) s.out[i] = i;
}

// Named-barrier protocol shared by both refine paths.
struct G2bExactSync {
  const OpCtx& ctx;
  __device__ __forceinline__ void operator()() const { group_sync(ctx); }
};
namespace g2b_smem {
constexpr int kCap = 20480;
constexpr int kIdBits = 18;
constexpr uint32_t kIdMask = (1u << kIdBits) - 1u;
constexpr int kWarps = kBlock / 32;
constexpr uint32_t kComp = smem::kG2bThr;
constexpr uint32_t kIdx = kComp + kCap * 8;
constexpr uint32_t kHist = kIdx + kCap * 2;
constexpr uint32_t kRed = kHist + 256 * 4;
constexpr uint32_t kTot = kRed + 2 * kWarps * 8;
constexpr uint32_t kSel = kTot + 8 * 4;
constexpr uint32_t kEnd = kSel + 8 * 4;
// mbarrier of the staged-composite bulk copy (refine CTAs only).
constexpr uint32_t kStageBar = (kEnd + 7u) & ~7u;
static_assert(kComp % 16 == 0 && kStageBar + 8 <= smem::kDynamicBytes,
              "bulk-copy destination / barrier fit the refine pool");
static_assert(kComp % 8 == 0 && kRed % 8 == 0, "u64 smem arrays");
static_assert(kCap <= 65536, "u16 list indices");
static_assert(kEnd <= smem::kDynamicBytes, "smem refine pool fits the dynamic buffer");
static_assert(kMaxSeqLen <= (1 << kIdBits), "candidate IDs fit the composite's ID field");
}  // namespace g2b_smem

__device__ __forceinline__ uint32_t g2b_smem_append(uint32_t* cursor, bool take, int lane) {
  const uint32_t ballot = __ballot_sync(0xffffffffu, take);
  uint32_t base = 0u;
  if (lane == 0 && ballot != 0u) base = atomicAdd(cursor, static_cast<uint32_t>(__popc(ballot)));
  base = __shfl_sync(0xffffffffu, base, 0);
  return base + static_cast<uint32_t>(__popc(ballot & ((1u << lane) - 1u)));
}

// kStaged: the candidates' composites were staged by P5 (comp_g[i], arena::kCompScratch, tagged
// with the launch) -- one load per candidate instead of the ID then logit gather -- and every
// selected ID is also written to out_s (shared) for the publication. kPreloaded: comp[0, n) already
// holds the composites (g2b_large_select stages the in-digit candidates there); nothing is loaded.
template <bool kStaged = false, bool kPreloaded = false>
__device__ __forceinline__ void g2b_smem_select(const OpCtx& ctx, const float* logits_q,
                                                const int32_t* cand_q, int n, int k, int32_t* out,
                                                const uint64_t* comp_g = nullptr,
                                                int32_t* out_s = nullptr,
                                                uint32_t stage_parity = 0u) {
  using namespace g2b_smem;
  uint64_t* comp = smem_at<uint64_t>(ctx.smem, kComp);
  uint16_t* idx = smem_at<uint16_t>(ctx.smem, kIdx);
  uint32_t* hist = smem_at<uint32_t>(ctx.smem, kHist);
  uint64_t* red = smem_at<uint64_t>(ctx.smem, kRed);
  uint32_t* tot = smem_at<uint32_t>(ctx.smem, kTot);
  uint32_t* sel = smem_at<uint32_t>(ctx.smem, kSel);
  const int tid = ctx.tid, lane = ctx.lane, warp = ctx.warp;

  uint64_t and_acc = ~uint64_t{0}, or_acc = 0;
  if constexpr (kStaged) {
    // Each composite carries this launch's tag (bits [50, 64)) once its P5 store has landed; slots
    // still holding 0 / another launch's value are reloaded until tagged.  Every thread keeps up to
    // kPoll loads in flight.
    (void)stage_parity;
    const uint64_t tag = comp_tag(ctx.epoch);
    constexpr int kPoll = 8;
    uint32_t rounds = 0u;
    for (int base = tid; base < n; base += kBlock * kPoll) {
      uint64_t c[kPoll];
      uint32_t pending = 0u;
#pragma unroll
      for (int u = 0; u < kPoll; ++u) {
        c[u] = 0u;
        if (base + u * kBlock < n) {
          c[u] = ld_relaxed_gpu_u64(comp_g + base + u * kBlock);
          pending |= 1u << u;
        }
      }
      for (;;) {
#pragma unroll
        for (int u = 0; u < kPoll; ++u) {
          if (((pending >> u) & 1u) != 0u && (c[u] >> kCompTagShift) == tag) pending &= ~(1u << u);
        }
        if (pending == 0u) break;
        // P5 writes exactly the n bin rows counted in H after S; a slot that never gets this
        // launch's tag is a broken invariant.
        if (++rounds > (1u << 22)) __trap();
#pragma unroll
        for (int u = 0; u < kPoll; ++u) {
          if (((pending >> u) & 1u) != 0u) c[u] = ld_relaxed_gpu_u64(comp_g + base + u * kBlock);
        }
      }
#pragma unroll
      for (int u = 0; u < kPoll; ++u) {
        if (base + u * kBlock < n) {
          comp[base + u * kBlock] = c[u];
          and_acc &= c[u];
          or_acc |= c[u];
        }
      }
    }
  } else if constexpr (kPreloaded) {
    for (int i = tid; i < n; i += kBlock) {
      const uint64_t c = comp[i];
      and_acc &= c;
      or_acc |= c;
    }
  } else {
    constexpr int kUnroll = 16;
    for (int base = tid; base < n; base += kBlock * kUnroll) {
      int32_t id[kUnroll];
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        const int i = base + j * kBlock;
        id[j] = i < n ? __ldcg(cand_q + i) : 0;
      }
      float v[kUnroll];
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        const int i = base + j * kBlock;
        v[j] = i < n ? __ldcg(logits_q + id[j]) : 0.0f;
      }
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        const int i = base + j * kBlock;
        if (i < n) {
          const uint64_t c = (static_cast<uint64_t>(score_key32(v[j])) << kIdBits) |
                             (kIdMask - static_cast<uint32_t>(id[j]));
          comp[i] = c;
          and_acc &= c;
          or_acc |= c;
        }
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    and_acc &= __shfl_xor_sync(0xffffffffu, and_acc, off);
    or_acc |= __shfl_xor_sync(0xffffffffu, or_acc, off);
  }
  if (lane == 0) {
    red[warp] = and_acc;
    red[kWarps + warp] = or_acc;
  }
  if (tid == 0) {
    sel[3] = 0u;
    sel[4] = 0u;
  }
  group_sync(ctx);
  uint64_t all_and = ~uint64_t{0}, all_or = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    all_and &= red[w];
    all_or |= red[kWarps + w];
  }

  int hi = 63 - __clzll(static_cast<long long>(all_and ^ all_or));
  uint64_t mask = ~((uint64_t{2} << hi) - 1u);
  uint64_t prefix = all_and & mask;
  int remaining = k;
  int m = n;
  bool listed = false;
  for (;;) {
    const int lo = hi > 7 ? hi - 7 : 0;
    const uint32_t dmask = (2u << (hi - lo)) - 1u;
    if (tid < 256) hist[tid] = 0u;
    group_sync(ctx);
    for (int i0 = warp * 32; i0 < m; i0 += kBlock) {
      const int i = i0 + lane;
      const uint64_t c = i < m ? comp[listed ? static_cast<int>(idx[i]) : i] : 0u;
      // One shared-memory reduction per participating composite (same counts as a match_any +
      // per-group add, without the warp-wide match latency on every histogram step).
      if (i < m && (c & mask) == prefix)
        atomicAdd(hist + (static_cast<uint32_t>(c >> lo) & dmask), 1u);
    }
    group_sync(ctx);
    uint32_t h = 0u, incl = 0u;
    if (tid < 256) {
      h = hist[tid];
      incl = h;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_down_sync(0xffffffffu, incl, off);
        if (lane + off < 32) incl += v;
      }
      if (lane == 0) tot[warp] = incl;
    }
    group_sync(ctx);
    if (tid < 256) {
      for (int w = warp + 1; w < 8; ++w) incl += tot[w];
      const uint32_t above = incl - h;
      if (above < static_cast<uint32_t>(remaining) && static_cast<uint32_t>(remaining) <= incl) {
        sel[0] = static_cast<uint32_t>(tid);
        sel[1] = above;
        sel[2] = h;
      }
    }
    group_sync(ctx);
    const uint32_t t = sel[0];
    remaining -= static_cast<int>(sel[1]);
    const uint32_t ht = sel[2];
    prefix |= static_cast<uint64_t>(t) << lo;
    mask |= static_cast<uint64_t>(dmask) << lo;
    const bool done = ht == static_cast<uint32_t>(remaining) || lo == 0;
    if (!listed) {
      for (int i0 = warp * 32; i0 < n; i0 += kBlock) {
        const int i = i0 + lane;
        const uint64_t c = i < n ? comp[i] : 0u;
        const uint64_t mc = c & mask;
        const bool in_digit = i < n && mc == prefix;
        const bool take = i < n && (mc > prefix || (done && in_digit));
        const uint32_t slot = g2b_smem_append(sel + 3, take, lane);
        if (take && slot < static_cast<uint32_t>(k)) {
          const int32_t id_v = static_cast<int32_t>(kIdMask - static_cast<uint32_t>(c & kIdMask));
          out[slot] = id_v;
          if constexpr (kStaged) out_s[slot] = id_v;
        }
        if (!done) {
          const uint32_t pos = g2b_smem_append(sel + 4, in_digit, lane);
          if (in_digit && pos < static_cast<uint32_t>(kCap)) idx[pos] = static_cast<uint16_t>(i);
        }
      }
      if (done) return;
      group_sync(ctx);
      listed = true;
      m = static_cast<int>(ht);
    } else if (done) {
      for (int i0 = warp * 32; i0 < m; i0 += kBlock) {
        const int i = i0 + lane;
        const uint64_t c = i < m ? comp[static_cast<int>(idx[i])] : 0u;
        const bool take = i < m && (c & mask) >= prefix;
        const uint32_t slot = g2b_smem_append(sel + 3, take, lane);
        if (take && slot < static_cast<uint32_t>(k)) {
          const int32_t id_v = static_cast<int32_t>(kIdMask - static_cast<uint32_t>(c & kIdMask));
          out[slot] = id_v;
          if constexpr (kStaged) out_s[slot] = id_v;
        }
      }
      return;
    }
    hi = lo - 1;
  }
}

// Exact top-k of a threshold bin larger than the shared-memory staging (n > g2b_smem::kCap; long
// contexts, e.g. layer 1 whose whole context can fall into one fp16-key bin), one CTA per query, on
// the same unique composites as g2b_smem_select (key32 << kIdBits | kIdMask - id, so ties go to the
// smaller id).  The candidates are streamed from global memory instead of the single-CTA rescans of
// exact_select::exact_topk:
//   pass 1   AND/OR of all composites + an unfiltered 8-bit histogram of composite bits [31, 38]
//   (key32 bits 13..20).  Inside one
//            fp16-key bin key32 bits 31..21 are constant for normal values, so this is already the
//            first useful digit; it is used only when the AND/OR confirms composite bits >= 39 are
//            constant, otherwise the digit pass is repeated below.
//   more     8-bit histogram passes under the decided prefix while the in-digit population exceeds
//   the staging. collect  every candidate strictly above the prefix goes to out; the in-digit ones
//   are staged in comp[] and
//            g2b_smem_select<false, true> finishes them (or all are taken when the digit holds
//            exactly the remaining count).
// Histograms are per-warp private copies in the idle composite staging; a warp whose lanes all hit
// one bin adds once.
namespace g2b_large {
constexpr int kSpecLo = 31;           // composite bit of key32 bit 13
constexpr int kSpecHi = kSpecLo + 7;  // composite bit of key32 bit 20
constexpr int kUnroll = 8;
static_assert(g2b_smem::kIdBits + 13 == kSpecLo, "speculative digit = key32 bits 13..20");
static_assert(g2b_smem::kWarps * 256 * 4 <= g2b_smem::kCap * 8,
              "private histograms fit the composite staging");
}  // namespace g2b_large

__device__ __forceinline__ uint64_t g2b_large_comp(float v, int32_t id) {
  return (static_cast<uint64_t>(score_key32(v)) << g2b_smem::kIdBits) |
         (g2b_smem::kIdMask - static_cast<uint32_t>(id));
}

// One streaming pass over the n candidates: per-warp histograms of composite bits [lo, lo + 7] over
// those with (c & mask) == prefix (and, when kAndOr, the AND/OR of all composites).  Leaves the CTA
// total in hist[256].
template <bool kAndOr>
__device__ __forceinline__ void g2b_large_hist_pass(const OpCtx& ctx, const float* logits_q,
                                                    const int32_t* cand_q, int n, uint64_t mask,
                                                    uint64_t prefix, int lo, uint64_t& and_acc,
                                                    uint64_t& or_acc) {
  using namespace g2b_smem;
  using namespace g2b_large;
  uint32_t* priv = smem_at<uint32_t>(ctx.smem, kComp);
  uint32_t* hist = smem_at<uint32_t>(ctx.smem, kHist);
  uint32_t* mine = priv + ctx.warp * 256;
  for (int b = ctx.tid; b < kWarps * 256; b += kBlock) priv[b] = 0u;
  group_sync(ctx);
  for (int base0 = ctx.warp * 32; base0 < n;
       base0 += kBlock * kUnroll) {  // warp-uniform: the body uses warp collectives
    const int base = base0 + ctx.lane;
    int32_t id[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      id[j] = i < n ? __ldcg(cand_q + i) : 0;
    }
    float v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      v[j] = i < n ? __ldcg(logits_q + id[j]) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      const bool valid = i < n;
      const uint64_t c = g2b_large_comp(v[j], id[j]);
      if constexpr (kAndOr) {
        if (valid) {
          and_acc &= c;
          or_acc |= c;
        }
      }
      const bool hit = valid && (c & mask) == prefix;
      const uint32_t bin = static_cast<uint32_t>(c >> lo) & 0xffu;
      const uint32_t hits = __ballot_sync(0xffffffffu, hit);
      const uint32_t bin0 = __shfl_sync(0xffffffffu, bin, hits != 0u ? __ffs(hits) - 1 : 0);
      if (__all_sync(0xffffffffu, !hit || bin == bin0)) {
        if (ctx.lane == 0 && hits != 0u)
          atomicAdd(mine + bin0, static_cast<uint32_t>(__popc(hits)));
      } else if (hit) {
        atomicAdd(mine + bin, 1u);
      }
    }
  }
  group_sync(ctx);
  if (ctx.tid < 256) {
    uint32_t total = 0u;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) total += priv[w * 256 + ctx.tid];
    hist[ctx.tid] = total;
  }
  group_sync(ctx);
}

// Threshold digit of hist[256] for `remaining`: t with above(t) < remaining <= above(t) + hist[t]
// (above = bins > t). Returns t, above(t) and hist[t] (CTA-uniform, through sel[0..2]).
__device__ __forceinline__ void g2b_large_pick(const OpCtx& ctx, int remaining, uint32_t& t,
                                               uint32_t& above, uint32_t& ht) {
  using namespace g2b_smem;
  const uint32_t* hist = smem_at<uint32_t>(ctx.smem, kHist);
  uint32_t* tot = smem_at<uint32_t>(ctx.smem, kTot);
  uint32_t* sel = smem_at<uint32_t>(ctx.smem, kSel);
  uint32_t h = 0u, incl = 0u;
  if (ctx.tid < 256) {
    h = hist[ctx.tid];
    incl = h;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t v = __shfl_down_sync(0xffffffffu, incl, off);
      if (ctx.lane + off < 32) incl += v;
    }
    if (ctx.lane == 0) tot[ctx.warp] = incl;
  }
  group_sync(ctx);
  if (ctx.tid < 256) {
    for (int w = ctx.warp + 1; w < 8; ++w) incl += tot[w];
    const uint32_t ab = incl - h;
    if (ab < static_cast<uint32_t>(remaining) && static_cast<uint32_t>(remaining) <= incl) {
      sel[0] = static_cast<uint32_t>(ctx.tid);
      sel[1] = ab;
      sel[2] = h;
    }
  }
  group_sync(ctx);
  t = sel[0];
  above = sel[1];
  ht = sel[2];
  group_sync(ctx);
}

static __device__ __noinline__ void g2b_large_select(const OpCtx& ctx, const float* logits_q,
                                                     const int32_t* cand_q, int n, int k,
                                                     int32_t* out) {
  using namespace g2b_smem;
  using namespace g2b_large;
  uint64_t* comp = smem_at<uint64_t>(ctx.smem, kComp);
  uint64_t* red = smem_at<uint64_t>(ctx.smem, kRed);
  uint32_t* sel = smem_at<uint32_t>(ctx.smem, kSel);

  // pass 1: AND/OR + speculative digit [kSpecLo, kSpecHi], unfiltered
  uint64_t and_acc = ~uint64_t{0}, or_acc = 0;
  g2b_large_hist_pass<true>(ctx, logits_q, cand_q, n, 0u, 0u, kSpecLo, and_acc, or_acc);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    and_acc &= __shfl_xor_sync(0xffffffffu, and_acc, off);
    or_acc |= __shfl_xor_sync(0xffffffffu, or_acc, off);
  }
  if (ctx.lane == 0) {
    red[ctx.warp] = and_acc;
    red[kWarps + ctx.warp] = or_acc;
  }
  group_sync(ctx);
  uint64_t all_and = ~uint64_t{0}, all_or = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    all_and &= red[w];
    all_or |= red[kWarps + w];
  }
  group_sync(ctx);
  const int hi_all = 63 - __clzll(static_cast<long long>(
                              all_and ^ all_or));  // composites are unique and n > 1: hi_all >= 0

  // first digit: the speculative one when every composite bit above it is constant, else a filtered
  // pass at [hi_all - 7, hi_all]
  int lo;
  uint64_t mask, prefix;
  if (hi_all <= kSpecHi) {
    lo = kSpecLo;
    mask = ~((uint64_t{2} << kSpecHi) - 1u);
  } else {
    lo = hi_all - 7;
    mask = ~((uint64_t{2} << hi_all) - 1u);
    g2b_large_hist_pass<false>(ctx, logits_q, cand_q, n, mask, all_and & mask, lo, and_acc, or_acc);
  }
  prefix = all_and & mask;
  int remaining = k;
  uint32_t t, above, ht;
  for (;;) {
    g2b_large_pick(ctx, remaining, t, above, ht);
    remaining -= static_cast<int>(above);
    prefix |= static_cast<uint64_t>(t) << lo;
    mask |= uint64_t{0xff} << lo;
    if (ht == static_cast<uint32_t>(remaining) || lo == 0 || ht <= static_cast<uint32_t>(kCap))
      break;
    // next digit below lo, skipping bits constant over all candidates (they cannot split the
    // in-digit population either)
    const int hi = (lo - 1 < hi_all) ? lo - 1 : hi_all;
    if (hi < 0) break;
    const int nlo = hi > 7 ? hi - 7 : 0;
    const uint64_t gap = (lo > hi + 1) ? (((uint64_t{1} << (lo - hi - 1)) - 1u) << (hi + 1)) : 0u;
    mask |= gap;
    prefix |= all_and & gap;
    lo = nlo;
    g2b_large_hist_pass<false>(ctx, logits_q, cand_q, n, mask, prefix, lo, and_acc, or_acc);
  }
  const bool done = ht == static_cast<uint32_t>(remaining) || lo == 0;
  const int sure = k - remaining;  // candidates strictly above the prefix: exactly this many
  // collect: strictly above -> out[0, sure); in-digit -> out[sure, k) when done, else staged in
  // comp[0, ht)
  if (ctx.tid == 0) {
    sel[3] = 0u;
    sel[4] = 0u;
  }
  group_sync(ctx);
  for (int base0 = ctx.warp * 32; base0 < n;
       base0 += kBlock * kUnroll) {  // warp-uniform: the body uses warp collectives
    const int base = base0 + ctx.lane;
    int32_t id[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      id[j] = i < n ? __ldcg(cand_q + i) : 0;
    }
    float v[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      v[j] = i < n ? __ldcg(logits_q + id[j]) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      const int i = base + j * kBlock;
      const uint64_t c = g2b_large_comp(v[j], id[j]);
      const uint64_t mc = c & mask;
      const bool in_digit = i < n && mc == prefix;
      const bool take = i < n && (mc > prefix || (done && in_digit));
      const uint32_t slot = g2b_smem_append(sel + 3, take, ctx.lane);
      if (take && slot < static_cast<uint32_t>(k)) out[slot] = id[j];
      if (!done) {
        const uint32_t pos = g2b_smem_append(sel + 4, in_digit, ctx.lane);
        if (in_digit && pos < static_cast<uint32_t>(kCap)) comp[pos] = c;
      }
    }
  }
  group_sync(ctx);
  if (!done)
    g2b_smem_select<false, true>(ctx, logits_q, cand_q, static_cast<int>(ht), remaining,
                                 out + sure);
}

__device__ __forceinline__ void g2b_refine_candidates(const OpCtx& ctx, const G2bState& s) {
  if (s.len <= s.k_rem0) {
    g2b_dense_arange(ctx, s);
  } else if (s.k_rem0 > 0 && s.len <= g2b_smem::kCap) {
    g2b_smem_select(ctx, s.logits_q, s.cand_q, s.len, s.k_rem0, s.out);
  } else if (s.k_rem0 > 0) {
    g2b_large_select(ctx, s.logits_q, s.cand_q, s.len, s.k_rem0, s.out);
  } else {
    exact_select::exact_topk(ctx.tid, kBlock, s.logits_q, s.cand_q, s.len, s.k_rem0, s.out, s.hist2,
                             s.cnt, G2bExactSync{ctx});
  }
  group_sync(ctx);  // barrier; group2b_main's C5 arrive is the barrier + atomic
}

__device__ __forceinline__ void g2b_publish_records(const OpCtx& ctx, int q, int j0, int j1) {
  const CtxRow6& row = ctx_bcast(ctx.smem);
  const int32_t* ids = row.ids_src;
  const uint64_t* table = row.peer_table;
  const uint32_t tag = row.tag;
  const int n_peers = static_cast<int>(row.n_peers);
  static_assert(kTopk == ll_wire::kIdsPerQuery, "LL publication/consumer query stride");

  constexpr int kPerThread = (ll_wire::kRecordsPerQuery + kBlock - 1) / kBlock;
  int2 pair[kPerThread];
#pragma unroll
  for (int m = 0; m < kPerThread; ++m) {
    const int j = j0 + ctx.tid + m * kBlock;
    pair[m] =
        j < j1
            ? __ldcg(reinterpret_cast<const int2*>(ids + 2 * (q * ll_wire::kRecordsPerQuery + j)))
            : make_int2(0, 0);
  }
#pragma unroll
  for (int m = 0; m < kPerThread; ++m) {
    const int j = j0 + ctx.tid + m * kBlock;
    if (j >= j1) continue;
    const int r = q * ll_wire::kRecordsPerQuery + j;
    for (int k = 0; k < n_peers; ++k) {
      uint8_t* peer = reinterpret_cast<uint8_t*>(table[k]);
      st_relaxed_sys_b128(peer + static_cast<size_t>(r) * kIdsRecordBytes,
                          static_cast<uint32_t>(pair[m].x), tag, static_cast<uint32_t>(pair[m].y),
                          tag);
    }
  }
}

static_assert(arena::kCompScratchCap == g2b_smem::kCap && g2b_smem::kIdBits == 18,
              "P5 stages exactly the composites the refine builds");
// Shared-memory staging of the refined IDs of one query (Group2b work CTAs only run Group2b;
// [kCtxEnd, kRmsPartial) is free).
constexpr uint32_t kG2bOutS = smem::kCtxEnd;
static_assert(kG2bOutS % 16 == 0 && kG2bOutS + kTopk * 4 <= smem::kRmsPartial,
              "refined-ID staging fits below the RMS partials");

// Instruction-cache warm-up of the staged refine select on full sparse layers.  The four Group2b
// work CTAs idle from kernel entry until P5's S barrier, while the refine's code is otherwise first
// fetched (cold, from DRAM) on the critical path after C4.  So op_g2b_refine_fast runs its (single,
// inlined) staged select twice: pass 0 right away on synthetic composites (distinct ids, 10-bit
// keys, so the radix passes, the list compaction and both take loops execute) kept in this CTA's
// slice of an otherwise unused arena area and writing only the shared-memory staging that pass 1
// re-initializes; pass 1 is the refine itself after S and C4.  The pass flag is opaque to the
// compiler so both passes execute the same instructions.  No value, ID or synchronization of the
// operation is affected.
constexpr int kDryWarmN = 4096;
constexpr int kDryWarmK = 1024;
constexpr size_t kDryWarmScratch = arena::kCompScratch + arena::kCompScratchBytes;
static_assert(kDryWarmScratch % 128 == 0 &&
                  kDryWarmScratch + size_t(kGroup2bWorkCtas) * kDryWarmN * 8 <=
                      arena::kScratch + arena::kScratchBytes,
              "dry-run composites fit the scratch region after the refine staging");
static_assert(kDryWarmK <= kTopk && kDryWarmN <= g2b_smem::kCap,
              "dry-run select fits the refine staging");

// Refine of query q = ctx.cta for g2b_refine_fast launches.  P5's S barrier completes once every
// scoring CTA has merged its raw bins into H, so the threshold bin, n_sure and n_cand follow from H
// exactly as in P5's p5_threshold_bases; this CTA derives them while P5 is still scattering and
// then, without waiting for C4, takes the threshold-bin candidates as the launch-tagged composites
// P5 stores before its sure IDs (one load each instead of the ID -> logit gather), the radix select
// is the same, and the refined IDs are published from shared memory (the sure ID in slot n_sure -
// 1, when it shares an LL record, from selected_ids once C4 is reached).  The consumed composite
// slots are cleared.  Returns false (the composites cleared after C4 when nothing is left to
// select) when the bin exceeds the staging capacity or nothing is left to select; the caller then
// runs op_g2b_refine. A bin above dsel::kDistMin takes the distributed refine instead: the scoring
// CTAs (dsel_run) finalize every slot above the last digit, and this CTA only finishes the digit
// reported in dsel_run's result record and publishes the remaining records.
static __device__ __noinline__ bool op_g2b_refine_fast(const IndexerArgs& args, OpCtx& ctx_in,
                                                       uint32_t sense_word) {
  // Register copy of the caller's context (never written here): the reference would otherwise keep
  // every ctx field in local memory and reload it after each barrier / "memory"-clobbering asm on
  // this critical-path select.
  const OpCtx ctx = ctx_in;
  const int q = ctx.cta;
  const CtxRow5& row = ctx_refine(ctx.smem);
  int32_t* const selects_q = row.selects + static_cast<size_t>(q) * kTopk;
  int32_t* const out_s = smem_at<int32_t>(ctx.smem, kG2bOutS);

  uint64_t* const dry =
      reinterpret_cast<uint64_t*>(ctx.arena + kDryWarmScratch) + size_t(q) * kDryWarmN;
  const uint64_t dry_tag = comp_tag(ctx.epoch) << kCompTagShift;
  for (int i = ctx.tid; i < kDryWarmN; i += kBlock) {
    const uint32_t key = 0x40000000u | ((static_cast<uint32_t>(i) * 2654435761u) >> 22);
    st_relaxed_gpu_u64(dry + i, (static_cast<uint64_t>(key) << g2b_smem::kIdBits) |
                                    (g2b_smem::kIdMask - static_cast<uint32_t>(i)) | dry_tag);
  }
  group_sync(ctx);

  int n_sure = 0;
  int n_sel = kDryWarmN;
  int k_sel = kDryWarmK;
  int pub_base = 0;
  bool distributed = false;
  int32_t* sel_out = out_s;
  const uint64_t* comp = dry;
  uint32_t real_pass = 0u;
#pragma unroll 1
  for (;;) {
    asm volatile("" : "+r"(real_pass));
    if (real_pass != 0u) {
      if (ctx.tid == 0) {
        const uint32_t* word = sync_word(ctx, kSyncS);
        while (static_cast<int32_t>(load_relaxed_gpu(word) ^ sense_word) >= 0) spin_backoff();
        fence_acquire_gpu();
      }
      group_sync(ctx);

      // Inclusive suffix sums of the merged raw bins; the threshold bin has S(thr) > 2048 >= S(thr
      // + 1) (p5_threshold_bases rule).
      uint32_t* const totals = smem_at<uint32_t>(ctx.smem, g2b_smem::kTot);
      uint32_t* const scal = smem_at<uint32_t>(ctx.smem, g2b_smem::kSel);
      uint32_t h = 0u, sum = 0u;
      if (ctx.tid < kHistBins) {
        h = __ldcg(row.hist_blk + q * kHistBlockDwords + ctx.tid);
        sum = h;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
          const uint32_t v = __shfl_down_sync(0xffffffffu, sum, offset);
          if (ctx.lane + offset < 32) sum += v;
        }
        if (ctx.lane == 0) totals[ctx.warp] = sum;
      }
      group_sync(ctx);
      if (ctx.tid < kHistBins) {
        for (int w = ctx.warp + 1; w < kHistBins / 32; ++w) sum += totals[w];
        const uint32_t next = sum - h;
        if (sum >= kP5SureThreshold && next <= kP5SureLimit) {
          scal[0] = next;
          scal[1] = h;
        }
      }
      group_sync(ctx);
      n_sure = static_cast<int>(scal[0]);
      const int n_cand = static_cast<int>(scal[1]);
      group_sync(ctx);
      k_sel = kTopk - n_sure;
      uint64_t* const staged = reinterpret_cast<uint64_t*>(ctx.arena + arena::kCompScratch) +
                               size_t(q) * arena::kCompScratchCap;
      if (n_cand > static_cast<int>(dsel::kDistMin)) {
        // Distributed refine (dsel_run): P5 staged nothing for this bin, and with no pick left the
        // scoring CTAs publish every record.  Otherwise wait for CTA 0's result record of this
        // launch (~epoch tag): digit population to finish, picks left, first slot of those picks.
        // The scoring CTAs have already written every slot before it.
        if (k_sel <= 0) return true;
        if (ctx.tid == 0) {
          const uint8_t* rec = ctx.arena + dsel::kResult + size_t(q) * 16;
          uint64_t lo = 0u, hi = 0u;
          ld_relaxed_sys_b128(rec, lo, hi);
          while (static_cast<uint32_t>(hi >> 32) != ~ctx.epoch) {
            spin_backoff();
            ld_relaxed_sys_b128(rec, lo, hi);
          }
          scal[0] = static_cast<uint32_t>(lo);
          scal[1] = static_cast<uint32_t>(lo >> 32);
          scal[2] = static_cast<uint32_t>(hi);
        }
        group_sync(ctx);
        n_sel = static_cast<int>(scal[0]);
        k_sel = static_cast<int>(scal[1]);
        pub_base = static_cast<int>(scal[2]);
        group_sync(ctx);

        distributed = true;
      } else {
        if (n_cand > arena::kCompScratchCap) return false;
        if (k_sel <= 0) {
          // Nothing to refine, but P5 still staged this bin's tagged composites: clear them once
          // all are written (C4).
          boundary_wait(ctx, kSyncC4);
          for (int i = ctx.tid; i < n_cand; i += kBlock) staged[i] = 0u;
          return false;
        }

        n_sel = n_cand;
        pub_base = n_sure;
      }

      // No C4 wait: the staged select polls these n_cand composites for this launch's tag (P5
      // writes them before its sure IDs).
      sel_out = selects_q + pub_base;
      comp = staged;
    }
    if (n_sel > 0)
      g2b_smem_select<true>(ctx, nullptr, nullptr, n_sel, k_sel, sel_out, comp, out_s, real_pass);
    group_sync(ctx);
    if (real_pass != 0u) break;
    real_pass = 1u;
  }

  if (distributed) {
    // Records [pub_base >> 1, end) are published here.  With an odd pub_base the first one also
    // holds slot pub_base - 1, written by a scoring CTA: wait until every scoring CTA has finished
    // this query (done counter, release / acquire).
    if ((pub_base & 1) != 0) {
      if (ctx.tid == 0) {
        const uint32_t* done = reinterpret_cast<const uint32_t*>(ctx.arena + dsel::kDone) + q;
        const uint32_t target = static_cast<uint32_t>(n_active_ctas(ctx.cur_pos));
        while (load_relaxed_gpu(done) < target) spin_backoff();
        fence_acquire_gpu();
      }
      group_sync(ctx);
    }
    if (mode_runs_comm(args.mode)) {
      const CtxRow6& brow = ctx_bcast(ctx.smem);
      const uint64_t* table = brow.peer_table;
      const uint32_t tag = brow.tag;
      const int n_peers = static_cast<int>(brow.n_peers);
      for (int r = (pub_base >> 1) + ctx.tid; r < ll_wire::kRecordsPerQuery; r += kBlock) {
        const int s0 = 2 * r;
        const int32_t id0 = s0 < pub_base ? __ldcg(selects_q + s0) : out_s[s0 - pub_base];
        const int32_t id1 = out_s[s0 + 1 - pub_base];
        const int rec = q * ll_wire::kRecordsPerQuery + r;
        for (int k = 0; k < n_peers; ++k) {
          uint8_t* peer = reinterpret_cast<uint8_t*>(table[k]);
          st_relaxed_sys_b128(peer + static_cast<size_t>(rec) * kIdsRecordBytes,
                              static_cast<uint32_t>(id0), tag, static_cast<uint32_t>(id1), tag);
        }
      }
    }
  } else if (mode_runs_comm(args.mode)) {
    const CtxRow6& brow = ctx_bcast(ctx.smem);
    const uint64_t* table = brow.peer_table;
    const uint32_t tag = brow.tag;
    const int n_peers = static_cast<int>(brow.n_peers);
    for (int r = (n_sure >> 1) + ctx.tid; r < ll_wire::kRecordsPerQuery; r += kBlock) {
      const int s0 = 2 * r - n_sure;
      const int rec = q * ll_wire::kRecordsPerQuery + r;
      int32_t id0;
      if (s0 >= 0) {
        id0 = out_s[s0];
      } else {
        // Sure slot n_sure - 1 (odd n_sure) shares this record; P5 writes it to selected_ids before
        // its C4 arrival.
        const uint32_t* c4 = sync_word(ctx, kSyncC4);
        while (!word_reached(c4, word_target(ctx.epoch))) spin_backoff();
        fence_acquire_gpu();
        id0 = __ldcg(selects_q + n_sure - 1);
      }
      const int32_t id1 = out_s[s0 + 1];
      for (int k = 0; k < n_peers; ++k) {
        uint8_t* peer = reinterpret_cast<uint8_t*>(table[k]);
        st_relaxed_sys_b128(peer + static_cast<size_t>(rec) * kIdsRecordBytes,
                            static_cast<uint32_t>(id0), tag, static_cast<uint32_t>(id1), tag);
      }
    }
  }
  // The staged composites were copied to shared memory inside the select: clear the consumed slots
  // (see comp_tag).
  {
    uint64_t* const staged = reinterpret_cast<uint64_t*>(ctx.arena + arena::kCompScratch) +
                             size_t(q) * arena::kCompScratchCap;
    for (int i = ctx.tid; i < n_sel; i += kBlock) staged[i] = 0u;
  }
  return true;
}

template <bool SparseComputeOnly, bool Publish = false>
static __device__ __noinline__ void op_g2b_refine(const IndexerArgs& args, OpCtx& ctx_in) {
  const OpCtx ctx = ctx_in;  // register copy (never written here), see op_g2b_refine_fast
  if constexpr (!SparseComputeOnly) {
    if (!ctx.sparse) return;  // the caller still arrives C5
    if constexpr (Publish) {
      if (ctx.cta >= kGroup2bWorkCtas && !ctx.null_math && mode_runs_comm(args.mode)) {
        constexpr int kHelpers = kGroup2bCtas / kGroup2bWorkCtas - 1;
        const int q = ctx.cta & (kGroup2bWorkCtas - 1);
        const int part = ctx.cta / kGroup2bWorkCtas - 1;
        // The sure records are published by the scoring CTAs after C4 (p5_publish_sure_records).
        (void)kHelpers;
        (void)q;
        (void)part;
        return;
      }
    }
    if (ctx.cta >= kGroup2bWorkCtas) return;
    if (ctx.null_math) return;  // protocol only: no memory traffic
  }

  G2bState s;
  s.q = ctx.cta;
  const CtxRow5& row = ctx_refine(ctx.smem);
  s.cp = ctx.cur_pos;
  const uint32_t* hist_q = row.hist_blk + s.q * kHistBlockDwords;  // H + q*0x408
  s.n_sure = static_cast<int>(hist_q[kHistSureIndex]);
  s.len = SparseComputeOnly
              ? static_cast<int>(hist_q[kHistCandIndex])
              : ((s.cp >= kSelectCurPos) ? static_cast<int>(hist_q[kHistCandIndex])  //  /
                                         : s.cp + s.q + 1);
  s.k_rem0 = kTopk - s.n_sure;
  s.out = row.selects + static_cast<size_t>(s.q) * kTopk + s.n_sure;  // &selects[q][n_sure]
  const int logits_stride =
      SparseComputeOnly ? total_len(s.cp) : logits_row_stride(args.flags, s.cp);
  const int ws_stride = SparseComputeOnly ? total_len(s.cp) : ws_row_stride(args.flags, s.cp);
  s.logits_q = row.logits + static_cast<size_t>(s.q) * logits_stride;
  s.cand_q = row.cand_ws + static_cast<size_t>(s.q) * ws_stride;
  s.hist2 = smem_at<uint32_t>(ctx.smem, smem::kG2bHist2);
  s.cnt = smem_at<uint32_t>(ctx.smem, smem::kG2bCnt);

  constexpr bool kPublish = Publish && !SparseComputeOnly;
  const bool publish = kPublish && mode_runs_comm(args.mode);
  const int sure_records = s.n_sure >> 1;
  g2b_refine_candidates(ctx, s);
  if (publish) g2b_publish_records(ctx, s.q, sure_records, ll_wire::kRecordsPerQuery);
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

static __device__ void op_p6_broadcast_ids_ll(const IndexerArgs& args, OpCtx& ctx) {
  if (ctx.null_math) return;
  if (!ctx.sparse && (args.flags & kFlagDenseSkipBroadcast)) return;
  const CtxRow6& row = ctx_bcast(ctx.smem);
  const int32_t* ids = row.ids_src;
  const uint64_t* table = row.peer_table;
  const uint32_t tag = row.tag;
  const int n_records = static_cast<int>(row.n_ids >> 1);
  const int n_peers = static_cast<int>(row.n_peers);
  static_assert(kTopk == ll_wire::kIdsPerQuery, "LL publication/consumer query stride");
  for (int r = ctx.cta * kConsumerThreads + ctx.tid; r < n_records;
       r += kP6Ctas * kConsumerThreads) {  // +0x800 per round
    // Filter records, not CTAs or phase arrivals: a boundary launch can have
    // both dense and sparse queries.  Every sparse query still publishes all
    // 1024 records, and every P6 CTA still reaches C6 and the P7 collective.
    if (!ll_wire::should_publish_record(ctx.cur_pos, r, kSeqLen, true)) continue;
    const uint32_t id0 = static_cast<uint32_t>(ids[2 * r]);
    const uint32_t id1 = static_cast<uint32_t>(ids[2 * r + 1]);
    for (int k = 0; k < n_peers; ++k) {
      uint8_t* peer = reinterpret_cast<uint8_t*>(table[k]);
      st_relaxed_sys_b128(peer + static_cast<size_t>(r) * kIdsRecordBytes, id0, tag, id1, tag);
    }
  }
}
static __device__ void op_p6_wait_acks(const IndexerArgs& args, OpCtx& ctx) {
  (void)args;
  (void)ctx;
}

// P6 entry point: LL-record publication of the selected IDs.
static __device__ void op_p6_broadcast_ids(const IndexerArgs& args, OpCtx& ctx) {
  op_p6_broadcast_ids_ll(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// The IndexerArgs endpoint / x overrides take precedence over ctx row 7.

// ---- P7: rank 0 sums the seven attention ranks' records with the residual (the same reduction
// as op_k0_rank0_reduce in mla_q4.cu)
// ---------------------------------------------------------
static __device__ void op_p7_rank0_reduce(const IndexerArgs& args, OpCtx& ctx) {
  if (!args.rank0_reduce) return;
  if (ctx.cta >= mega_wire::kRxCtas) return;          // kOProjCtas = 128
  constexpr int kPairs = mega_wire::kRowsPerCta / 2;  // 24 bf16 pairs per token per CTA
  if (ctx.tid >= kSeqLen * kPairs) return;            // 96 workers
  const int q = ctx.tid / kPairs;
  const int p = ctx.tid - q * kPairs;
  const int k = p / 3;
  const int w = p - k * 3;
  const int pair_index = q * kHidden + mega_wire::kRowsPerCta * ctx.cta + 2 * p;
  const CtxRow7& row = ctx_collective(ctx.smem);
  const __nv_bfloat16* x = args.x != nullptr ? args.x : row.residual;
  __nv_bfloat16* endpoint = args.endpoint != nullptr ? args.endpoint : row.hidden_out;
  const uint32_t x_bits = *reinterpret_cast<const uint32_t*>(x + pair_index);
  constexpr int kSources = kWorld - 1;
  uint64_t lo[kSources], hi[kSources];
#pragma unroll
  for (int i = 0; i < kSources; ++i) {
    ld_relaxed_sys_b128(ctx.arena + mega_wire::k5_rx_offset(ctx.epoch, i, ctx.cta, q, k), lo[i],
                        hi[i]);
  }
  for (;;) {
    uint32_t stale = 0u;
#pragma unroll
    for (int i = 0; i < kSources; ++i) {
      if (static_cast<uint32_t>(hi[i] >> 32) != ctx.epoch) stale |= 1u << i;
    }
    if (stale == 0u) break;
    spin_backoff();
#pragma unroll
    for (int i = 0; i < kSources; ++i) {
      if (stale & (1u << i))
        ld_relaxed_sys_b128(ctx.arena + mega_wire::k5_rx_offset(ctx.epoch, i, ctx.cta, q, k), lo[i],
                            hi[i]);
    }
  }
  uint64_t total = 0ull;
#pragma unroll
  for (int i = 0; i < kSources; ++i) {
    const uint32_t term =
        w == 0 ? static_cast<uint32_t>(lo[i])
               : (w == 1 ? static_cast<uint32_t>(lo[i] >> 32) : static_cast<uint32_t>(hi[i]));
    total = add_f32x2(total, pack_f32x2(bf16_pair_low(term), bf16_pair_high(term)));
  }
  F32Pair t;
  t.bits = total;
  *reinterpret_cast<uint32_t*>(endpoint + pair_index) = pack_bf16_pair(
      __fadd_rn(bf16_pair_low(x_bits), t.value[0]), __fadd_rn(bf16_pair_high(x_bits), t.value[1]));
}

// P7 entry point: rank 0 reduces the o_proj records of the attention ranks.
static __device__ void op_p7_collective(const IndexerArgs& args, OpCtx& ctx) {
  op_p7_rank0_reduce(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// Prologue: copy n_rows x 16 u64 of the host ctx table into smem rows of stride 0x1c0 at 0x400.
template <bool CompactDense, bool SparseComputeOnly>
__device__ __forceinline__ void ctx_table_to_smem(const IndexerArgs& args, uint8_t* smem,
                                                  int persistent_cta) {
  if constexpr (CompactDense) {
    // A compact dense launch contains only four indexer.wk workers and Group2a.
    // Each role consumes one 64-B inline row prefix, so eight lanes suffice.
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < 8) {
      const int row = persistent_cta == kGroup2aWorker ? 2 : 0;
      *reinterpret_cast<uint64_t*>(smem + smem::kCtx + row * smem::kCtxRowStride + tid * 8) =
          args.inline_rows[row][tid];
    }
    return;
  }
  // The compute-only persistent launch carries six live row prefixes in
  // grid-constant parameters, but each virtual CTA consumes only a subset.
  // Pack that subset into warp 0: every participating lane performs one
  // aligned u64 parameter load and one shared-memory store.
  if ((SparseComputeOnly || has_inline_ctx(args)) && persistent_cta >= 0) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid >= 32) return;

    const int slot = tid >> 3;
    const int col = tid & 7;
    int row = -1;
    if (persistent_cta < kGroup1Ctas) {
      int next = 0;
      if (persistent_cta < kP1AllCtas) {
        if (slot == next) row = 0;
        ++next;
      }
      if (slot == next) row = 1;
      ++next;
      if (persistent_cta < kP3CtasDefault) {
        if (slot == next) row = 3;
        ++next;
      }
      if (persistent_cta < n_active_ctas(args.cur_pos)) {
        if (slot == next) row = 4;
      }
    } else if (persistent_cta == kGroup2aWorker) {
      if (slot == 0) row = 2;
    } else if (persistent_cta >= kGroup2bBase && persistent_cta < kGroup2bBase + kGroup2bWorkCtas) {
      if (slot == 0) row = 5;
    }

    if (row >= 0) {
      *reinterpret_cast<uint64_t*>(smem + smem::kCtx + row * smem::kCtxRowStride + col * 8) =
          args.inline_rows[row][col];
    }
    return;
  }
  if (has_inline_ctx(args)) {
    constexpr int kLiveWordsPerRow = 8;  // the first 64 B of each inline compute row
    const int n = static_cast<int>(args.n_rows) * kLiveWordsPerRow;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
      const int row = i / kLiveWordsPerRow;
      const int col = i % kLiveWordsPerRow;
      *reinterpret_cast<uint64_t*>(smem + smem::kCtx + row * smem::kCtxRowStride + col * 8) =
          args.inline_rows[row][col];
    }
    return;
  }
  const uint64_t* src = reinterpret_cast<const uint64_t*>(args.ctx_table);
  const int n = static_cast<int>(args.n_rows) * 16;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const int row = i >> 4;
    const int col = i & 15;
    *reinterpret_cast<uint64_t*>(smem + smem::kCtx + row * smem::kCtxRowStride + col * 8) =
        src[row * 16 + col];
  }
}

// cur_pos is reachable through THREE ctx pointers -- row 4 +0x20 (prologue, P4, Group2b exit gate),
// row 2 +0x08 (Group2a's indexer_k_cache row index) and row 5 +0x10 (Group2b's row length and its
// cur_pos >= kSelectCurPos check).  The
// implementation loads cur_pos once from row 4 +0x20 and uses that value everywhere, so the host
// table must make the other two pointers observe the same value.  Asserted here: thread 0 of the
// CTA that would consume the pointer compares and traps on a mismatch.
__device__ __forceinline__ void assert_cur_pos_contract(const int32_t* other_ptr, int cur_pos) {
  if (threadIdx.x == 0 && (other_ptr == nullptr || *other_ptr != cur_pos)) __trap();
}

__device__ __forceinline__ void assert_ids_src_contract(const uint8_t* smem) {
  const int32_t* ids = ctx_bcast(smem).ids_src;
  if (threadIdx.x == 0 &&
      (ids == nullptr || ids != ctx_score(smem).selects || ids != ctx_refine(smem).selects))
    __trap();
}

// Hands the exit-record slot of epoch `done` back to the attention ranks (mega_wire::kExitCredit,
// one line per collective CTA).  Called after griddepcontrol.wait: every earlier launch on this
// stream, including the one that read epoch `done`, has completed.  The stores are not waited on;
// they drain while this launch computes.
__device__ __forceinline__ void publish_exit_credit(const IndexerArgs& args, int cta,
                                                    uint32_t done) {
  if (!args.rank0_reduce || args.single_rank || cta >= mega_wire::kRxCtas) return;
#pragma unroll
  for (int r = 1; r < kWorld; ++r) {
    st_relaxed_sys_b128(
        args.peers[r] + mega_wire::kExitCredit + size_t(cta) * mega_wire::kExitCreditLineBytes,
        done, done, done, done);
  }
}
// Device-side load of the two per-launch scalars into the
// dynamic smem words smem::kEpochWord / kCurPosWord so the buffer keeps window 0x400 and there is
// still no static smem).  Thread 0 of every CTA calls it before the prologue barrier; every role
// reads the words after it (smem_epoch / smem_cur_pos).
//   cur_pos = cur_pos_dev ? *cur_pos_dev: *ctx_table->score.cur_pos_ptr
//   epoch   = device_epoch && epoch_ctr ? ++epoch_ctr[cta]                  (per-CTA counter, plain
//   load/store)
//           device_epoch && epoch_dev ? atomicAdd(*epoch_dev, 1) / kGrid + 1  (launch ordinal)
//           args.epoch                                                     (explicit fallback)
// Public calls use workspace-owned per-CTA counters. All eight ranks execute
// the same serialized operator sequence.
template <bool InlineCompute>
__device__ __forceinline__ void load_epoch_cur_pos(const IndexerArgs& args, uint8_t* smem, int cta,
                                                   bool device_epoch) {
  uint32_t epoch = args.epoch;
  int32_t cur_pos;
  if constexpr (InlineCompute) {
    cur_pos = args.cur_pos;
  } else {
    cur_pos = has_inline_ctx(args)
                  ? args.cur_pos
                  : (args.cur_pos_dev != nullptr ? *args.cur_pos_dev
                                                 : *args.ctx_table->score.cur_pos_ptr);
    if (device_epoch && args.epoch_ctr != nullptr) {
      const uint32_t done = args.epoch_ctr[cta];
      epoch = done + 1u;
      args.epoch_ctr[cta] = epoch;
      publish_exit_credit(args, cta, done);
    } else if (device_epoch && args.epoch_dev != nullptr) {
      epoch =
          atomicAdd(const_cast<uint32_t*>(args.epoch_dev), 1u) / static_cast<uint32_t>(kGrid) + 1u;
    }
  }
  *smem_at<uint32_t>(smem, smem::kEpochWord) = epoch;
  *smem_at<int32_t>(smem, smem::kCurPosWord) = cur_pos;
}
__device__ __forceinline__ uint32_t smem_epoch(const uint8_t* smem) {
  return *reinterpret_cast<const uint32_t*>(smem + smem::kEpochWord);
}
__device__ __forceinline__ int32_t smem_cur_pos(const uint8_t* smem) {
  return *reinterpret_cast<const int32_t*>(smem + smem::kCurPosWord);
}

template <bool StaticMath, bool StaticSparse>
__device__ __forceinline__ OpCtx make_op_ctx(const IndexerArgs& args, int op, int cta, int num_ctas,
                                             uint8_t* smem, int cur_pos, uint32_t epoch,
                                             int sync_threads) {
  static_assert(!StaticSparse || StaticMath,
                "a statically sparse context is also a static-math context");
  OpCtx ctx;
  ctx.op = op;
  ctx.cta = cta;
  ctx.num_ctas = num_ctas;
  ctx.tid = static_cast<int>(threadIdx.x);
  ctx.warp = ctx.tid >> 5;
  ctx.lane = ctx.tid & 31;
  ctx.sync_threads = sync_threads;
  ctx.prefetcher = ctx.warp >= kConsumerWarps;
  if constexpr (StaticMath) {
    ctx.null_math = false;
    ctx.sparse = StaticSparse;
  } else {
    ctx.null_math = args.mode == kModeNull;
    ctx.sparse = is_sparse(cur_pos);
  }
  ctx.cur_pos = cur_pos;
  ctx.epoch = epoch;  // loaded: smem::kEpochWord in the kernels
  ctx.smem = smem;
  ctx.smem_s = smem_u32(smem);
  ctx.arena = args.arena;
  ctx.sync = args.sync_words;
  ctx.item = 0;
  ctx.reserved_ptr = nullptr;
  return ctx;
}
// Sure part [0, n_sure >> 1) of query (cta & 3)'s LL records, split over the scoring CTAs of that
// query, once C4 has made every scattered sure ID visible in selected_ids.  Same records, tags and
// peers as the Group2b helper publication it replaces. query_mask limits it to some queries;
// slot_count >= 0 replaces n_sure (dsel: the slots finalized by the scoring CTAs).
__device__ __forceinline__ void p5_publish_sure_records(const IndexerArgs& args, const OpCtx& ctx,
                                                        int n_ctas, uint32_t query_mask = 0xfu,
                                                        int slot_count = -1) {
  const CtxRow6& brow = ctx_bcast(ctx.smem);
  const int q = ctx.cta & (kSeqLen - 1);
  if (((query_mask >> q) & 1u) == 0u) return;
  const int parts = (n_ctas - q + kSeqLen - 1) / kSeqLen;
  const int part = ctx.cta / kSeqLen;
  const int n_sure = slot_count >= 0
                         ? slot_count
                         : static_cast<int>(__ldcg(ctx_score(ctx.smem).hist_blk +
                                                   q * kHistBlockDwords + kHistSureIndex));
  const int records = n_sure >> 1;
  const int j0 = records * part / parts;
  const int j1 = records * (part + 1) / parts;
  const int32_t* ids = brow.ids_src;
  const uint64_t* table = brow.peer_table;
  const uint32_t tag = brow.tag;
  const int n_peers = static_cast<int>(brow.n_peers);
  for (int j = j0 + ctx.tid; j < j1; j += ctx.sync_threads) {
    const int2 pair =
        __ldcg(reinterpret_cast<const int2*>(ids + 2 * (q * ll_wire::kRecordsPerQuery + j)));
    const int r = q * ll_wire::kRecordsPerQuery + j;
    for (int k = 0; k < n_peers; ++k) {
      uint8_t* peer = reinterpret_cast<uint8_t*>(table[k]);
      st_relaxed_sys_b128(peer + static_cast<size_t>(r) * kIdsRecordBytes,
                          static_cast<uint32_t>(pair.x), tag, static_cast<uint32_t>(pair.y), tag);
    }
  }
  (void)args;
}

// Grid barrier of the active scoring CTAs on dsel::kSyncWord, with the S barrier's sense-flip
// counter (CTA 0 adds kSenseCta0Base - n_active, the others 1).
__device__ __forceinline__ void dsel_barrier(const OpCtx& ctx, int n_active) {
  consumer_sync();
  if (ctx.tid == 0) {
    uint32_t* word = sync_word(ctx, dsel::kSyncWord);
    const uint32_t old = atom_acq_rel_gpu_add(
        word, ctx.cta == 0 ? (kSenseCta0Base - static_cast<uint32_t>(n_active)) : 1u);
    while (static_cast<int32_t>(load_relaxed_gpu(word) ^ old) >= 0) spin_backoff();
    fence_acquire_gpu();
  }
  consumer_sync();
}

// Per-query refine state (thread 0 writes, every thread reads): current / next digit mask, prefix
// and window (lowest bit, digit mask), the AND over all candidates and the highest bit where they
// differ, next free slot (base), picks left (r), phase (0 first round, 1 refining, 2 done), this
// round's action, the selected digit and the slot of its first row.
struct DselQ {
  uint64_t m_cur, p_cur;
  uint64_t m_nxt, p_nxt;
  uint64_t all_and;
  int32_t lo_cur, dm_cur, lo_nxt, dm_nxt;
  int32_t hi_all, base, r, phase;
  int32_t action, t, take_base, pad;
};
static_assert(sizeof(DselQ) <= 128, "dsel state stride");
enum DselAction : int32_t {
  kDselNone = 0,
  kDselHist = 1,
  kDselContinue = 2,
  kDselStage = 3,
  kDselTakeAll = 4
};

// Distributed exact refine (see namespace dsel), run by every scoring CTA in place of the plain
// sure-record publication.  Queries with more than kDistMin threshold-bin candidates and picks left
// form dist_mask.  Returns false when there is none (the caller then publishes the sure records
// after C4); otherwise it publishes every query's LL records itself: the other queries' sure
// records after C4 in round 0, the distributed queries' records [0, base >> 1) once all their
// slots are final.
static __device__ __noinline__ bool dsel_run(const IndexerArgs& args, OpCtx& ctx_in, int n_active) {
  using namespace dsel;
  const OpCtx ctx = ctx_in;
  const int tid = ctx.tid, lane = ctx.lane, warp = ctx.warp;
  const uint32_t* n_cand_s = smem_at<uint32_t>(ctx.smem, kP5NCand);
  const uint32_t* n_sure_s = smem_at<uint32_t>(ctx.smem, kP5NSure);
  const uint16_t* thr_bin = smem_at<uint16_t>(ctx.smem, smem::kThrBin);
  uint32_t dist_mask = 0u;
#pragma unroll
  for (int q = 0; q < kSeqLen; ++q) {
    if (n_cand_s[q] > kDistMin && n_sure_s[q] < static_cast<uint32_t>(kTopk)) dist_mask |= 1u << q;
  }
  if (dist_mask == 0u) return false;

  DselQ* st = smem_at<DselQ>(ctx.smem, kSState);
  uint64_t* list = smem_at<uint64_t>(ctx.smem, kSList);
  uint32_t* cnt = smem_at<uint32_t>(ctx.smem, kSCnt);
  uint32_t* lhist = smem_at<uint32_t>(ctx.smem, kSHist);
  uint32_t* soff = smem_at<uint32_t>(ctx.smem, kSOff);
  uint32_t* srank = smem_at<uint32_t>(ctx.smem, kSRank);
  uint32_t* sabove = smem_at<uint32_t>(ctx.smem, kSAbove);
  uint32_t* spick = smem_at<uint32_t>(ctx.smem, kSPick);
  uint32_t* stot = smem_at<uint32_t>(ctx.smem, kSTot);
  uint64_t* sred = smem_at<uint64_t>(ctx.smem, kSRed);
  uint32_t* const gh = reinterpret_cast<uint32_t*>(ctx.arena + kHistG);
  unsigned long long* const gandor = reinterpret_cast<unsigned long long*>(ctx.arena + kAndOr);
  uint64_t* const staged = reinterpret_cast<uint64_t*>(ctx.arena + arena::kCompScratch);
  const CtxRow4& row = ctx_score(ctx.smem);
  const uint32_t hist_s = ctx.smem_s + kSHist;

  for (int i = tid; i < kSeqLen * kHistBins; i += kConsumerThreads) lhist[i] = 0u;
  if (tid < kSeqLen) {
    cnt[tid] = 0u;
    DselQ& s = st[tid];
    s.phase = ((dist_mask >> tid) & 1u) ? 0 : 2;
    s.base = static_cast<int>(n_sure_s[tid]);
    s.r = kTopk - static_cast<int>(n_sure_s[tid]);
    s.action = kDselNone;
  }
  consumer_sync();

  // Collect this CTA's threshold-bin rows of each distributed query from the stage-A bin / logit
  // caches into its shared list, with the speculative digit histogram and the AND / OR over the
  // composites.  Rows beyond the logit cache come from the logits row, whose compact pitch (cur_pos
  // + 4) may start it at any 4-byte offset: guarded scalar loads.
  {
    const int n_tiles = k_tiles_of_cta(ctx.cur_pos, ctx.cta);
    const int stride = logits_row_stride(args.flags, ctx.cur_pos);
    uint64_t and_acc[kSeqLen], or_acc[kSeqLen];
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
      and_acc[q] = ~uint64_t{0};
      or_acc[q] = 0u;
    }
    for (int pair = warp; pair < kSeqLen * n_tiles; pair += kConsumerWarps) {
      const int q = pair / n_tiles;
      const int tile = pair - q * n_tiles;
      if (((dist_mask >> q) & 1u) == 0u) continue;
      const int lr0 = 4 * lane;
      const int pos0 = (tile * kP4Ctas + ctx.cta) * kKTileRows + lr0;
      const int last = ctx.cur_pos + q;
      const uint32_t thr = thr_bin[q];
      const uint32_t bins4 = *smem_at<uint32_t>(
          ctx.smem, smem::kBin8 + static_cast<uint32_t>((tile * kSeqLen + q) * kKTileRows + lr0));
      float4 lg;
      if (tile < static_cast<int>(smem::kLogitCacheTiles)) {
        lg = *smem_at<float4>(
            ctx.smem, smem::kLogitCache +
                          static_cast<uint32_t>(((tile * kSeqLen + q) * kKTileRows + lr0) * 4));
      } else {
        const float* lrow = row.logits + static_cast<size_t>(q) * stride;
        lg.x = pos0 + 0 <= last ? __ldcg(lrow + pos0 + 0) : 0.0f;
        lg.y = pos0 + 1 <= last ? __ldcg(lrow + pos0 + 1) : 0.0f;
        lg.z = pos0 + 2 <= last ? __ldcg(lrow + pos0 + 2) : 0.0f;
        lg.w = pos0 + 3 <= last ? __ldcg(lrow + pos0 + 3) : 0.0f;
      }
      const float lv[4] = {lg.x, lg.y, lg.z, lg.w};
      uint64_t* lq = list + q * kListCap;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int pos = pos0 + u;
        const bool in = pos <= last && ((bins4 >> (8 * u)) & 0xffu) == thr;
        const uint64_t c = (static_cast<uint64_t>(score_key32(lv[u])) << g2b_smem::kIdBits) |
                           (g2b_smem::kIdMask - (static_cast<uint32_t>(pos) & g2b_smem::kIdMask));
        const uint32_t slot = g2b_smem_append(cnt + q, in, lane);
        if (in) {
          lq[slot] = c;
          p4_hist_inc(
              hist_s +
              static_cast<uint32_t>(q * kHistBins + static_cast<int>((c >> kSpecLo) & 0xffu)) * 4u);
#pragma unroll
          for (int qq = 0; qq < kSeqLen;
               ++qq) {  // static indexing keeps the accumulators in registers
            if (qq == q) {
              and_acc[qq] &= c;
              or_acc[qq] |= c;
            }
          }
        }
      }
    }
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        and_acc[q] &= __shfl_xor_sync(0xffffffffu, and_acc[q], off);
        or_acc[q] |= __shfl_xor_sync(0xffffffffu, or_acc[q], off);
      }
      if (lane == 0) {
        sred[(warp * kSeqLen + q) * 2] = and_acc[q];
        sred[(warp * kSeqLen + q) * 2 + 1] = or_acc[q];
      }
    }
  }
  consumer_sync();
  if (tid < kSeqLen && ((dist_mask >> tid) & 1u) != 0u) {
    uint64_t a = ~uint64_t{0}, o = 0u;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      a &= sred[(w * kSeqLen + tid) * 2];
      o |= sred[(w * kSeqLen + tid) * 2 + 1];
    }
    if (o != 0u || a != ~uint64_t{0}) {  // this CTA saw candidates of the query
      atomicAnd(gandor + 2 * tid, static_cast<unsigned long long>(a));
      atomicOr(gandor + 2 * tid + 1, static_cast<unsigned long long>(o));
    }
  }

  const uint64_t tag_bits = comp_tag(ctx.epoch) << kCompTagShift;
  uint32_t merged = dist_mask;
  int round = 0;
  // One round per digit.  Every active CTA adds its local digit histogram to the round's global
  // histogram (the returned old value is this CTA's offset inside each digit), meets the others at
  // the grid barrier and reads the merged histogram; thread t owns digit t.
#pragma unroll 1
  for (;;) {
    if (round >= kMaxRounds) __trap();
    uint32_t* const gr = gh + static_cast<size_t>(round) * kSeqLen * kHistBins;
    uint32_t off_reg[kSeqLen];
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
      off_reg[q] = 0u;
      if ((merged >> q) & 1u) {
        const uint32_t v = lhist[q * kHistBins + tid];
        if (v != 0u) off_reg[q] = atomicAdd(gr + q * kHistBins + tid, v);
      }
    }
    dsel_barrier(ctx, n_active);

    uint32_t g[kSeqLen], s[kSeqLen];
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q)
      g[q] = ((merged >> q) & 1u) ? __ldcg(gr + q * kHistBins + tid) : 0u;
    if (round == 0 && tid < 2 * kSeqLen)
      sred[tid] = __ldcg(reinterpret_cast<const uint64_t*>(gandor) + tid);
    if (round == 0 && dist_mask != 0xfu) {
      // The other queries of this launch keep the plain path: publish their sure records once C4 is
      // reached.
      boundary_wait(ctx, kSyncC4);
      p5_publish_sure_records(args, ctx, n_active, ~dist_mask & 0xfu);
    }
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
      if ((merged >> q) & 1u) {
        soff[q * kHistBins + tid] = off_reg[q];
        srank[q * kHistBins + tid] = 0u;
        lhist[q * kHistBins + tid] = 0u;
      }
      s[q] = g[q];
    }
    // Suffix sums over the digits (a larger digit is a larger key): inclusive within each warp,
    // exclusive across the 8 warps.
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
#pragma unroll
      for (int q = 0; q < kSeqLen; ++q) {
        const uint32_t v = __shfl_down_sync(0xffffffffu, s[q], off);
        if (lane + off < 32) s[q] += v;
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < kSeqLen; ++q) stot[q * 8 + warp] = s[q];
    }
    consumer_sync();
    if (warp == 0) {
      const int wq = lane & 7;
      const uint32_t own = stot[lane];
      uint32_t suf = own;
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) {
        const uint32_t v = __shfl_down_sync(0xffffffffu, suf, off, 8);
        if (wq + off < 8) suf += v;
      }
      stot[lane] = suf - own;
    }
    consumer_sync();
#pragma unroll
    for (int q = 0; q < kSeqLen; ++q) {
      if ((merged >> q) & 1u) {
        // incl = candidates with digit >= tid, above = digit > tid; the digit with above < r <=
        // incl holds the r-th pick.
        const uint32_t incl = s[q] + stot[q * 8 + warp];
        const uint32_t above = incl - g[q];
        sabove[q * kHistBins + tid] = above;
        const uint32_t r = static_cast<uint32_t>(st[q].r);
        if (above < r && r <= incl) {
          spick[q * 4 + 0] = static_cast<uint32_t>(tid);
          spick[q * 4 + 1] = above;
          spick[q * 4 + 2] = g[q];
        }
      }
    }
    consumer_sync();

    // Thread 0 sets each active query's action for the pass below and the digit window of the next
    // round.
    if (tid == 0) {
#pragma unroll 1
      for (int q = 0; q < kSeqLen; ++q) {
        DselQ& sq = st[q];
        sq.action = kDselNone;
        if (((merged >> q) & 1u) == 0u) continue;
        if (sq.phase == 0) {
          // First round: the speculative digit (bits kSpecLo..kSpecHi) is usable only if every
          // candidate agrees on the bits above kSpecHi.
          const uint64_t a = sred[2 * q];
          const uint64_t o = sred[2 * q + 1];
          const int hi_all = 63 - __clzll(static_cast<long long>(a ^ o));
          sq.all_and = a;
          sq.hi_all = hi_all;
          sq.phase = 1;
          if (hi_all > kSpecHi) {
            // They differ above it: spend this round on a histogram of the 8 bits below the highest
            // differing bit.
            const uint64_t m = ~((uint64_t{2} << hi_all) - 1u);
            sq.m_nxt = m;
            sq.p_nxt = a & m;
            sq.lo_nxt = hi_all > 7 ? hi_all - 7 : 0;
            sq.dm_nxt = static_cast<int32_t>((2u << (hi_all - sq.lo_nxt)) - 1u);
            sq.action = kDselHist;
            continue;
          }
          const uint64_t m = ~((uint64_t{2} << kSpecHi) - 1u);
          sq.m_cur = m;
          sq.p_cur = a & m;
          sq.lo_cur = kSpecLo;
          sq.dm_cur = 0xff;
        } else {
          sq.m_cur = sq.m_nxt;
          sq.p_cur = sq.p_nxt;
          sq.lo_cur = sq.lo_nxt;
          sq.dm_cur = sq.dm_nxt;
        }
        const uint32_t t = spick[q * 4 + 0];
        const int above = static_cast<int>(spick[q * 4 + 1]);
        const int h = static_cast<int>(spick[q * 4 + 2]);
        sq.t = static_cast<int32_t>(t);
        sq.take_base = sq.base;
        sq.base += above;
        sq.r -= above;
        // Rows above digit t take slots [take_base, take_base + above).  Then the whole digit fits
        // the picks left (take it), or it is small enough for the Group2b select (stage it), or the
        // next round refines it.
        if (h == sq.r) {
          sq.action = kDselTakeAll;
          sq.base += h;
          sq.r = 0;
          sq.phase = 2;
        } else if (h <= static_cast<int>(kFinalCap)) {
          sq.action = kDselStage;
          sq.phase = 2;
        } else {
          // Next window: the 8 bits below this digit, skipping the bits above hi_all (common to
          // every candidate).
          sq.action = kDselContinue;
          const int lo = sq.lo_cur;
          uint64_t m = sq.m_cur | (static_cast<uint64_t>(static_cast<uint32_t>(sq.dm_cur)) << lo);
          uint64_t p = sq.p_cur | (static_cast<uint64_t>(t) << lo);
          const int hi = (lo - 1 < sq.hi_all) ? lo - 1 : sq.hi_all;
          const uint64_t gap =
              (lo - 1 > hi) ? (((uint64_t{1} << (lo - 1 - hi)) - 1u) << (hi + 1)) : uint64_t{0};
          m |= gap;
          p |= sq.all_and & gap;
          sq.m_nxt = m;
          sq.p_nxt = p;
          sq.lo_nxt = hi > 7 ? hi - 7 : 0;
          sq.dm_nxt = static_cast<int32_t>((2u << (hi - sq.lo_nxt)) - 1u);
        }
        if (ctx.cta == 0 && (sq.action == kDselStage || sq.action == kDselTakeAll)) {
          // Result record for the Group2b CTA: (digit population, picks left, first slot of the
          // picks) when staged, (0, 0, filled slots) when the digit is taken whole; tagged with
          // ~epoch.
          const bool stage = sq.action == kDselStage;
          st_relaxed_sys_b128(ctx.arena + kResult + static_cast<size_t>(q) * 16,
                              stage ? static_cast<uint32_t>(h) : 0u,
                              stage ? static_cast<uint32_t>(sq.r) : 0u,
                              static_cast<uint32_t>(sq.base), ~ctx.epoch);
        }
      }
    }
    consumer_sync();

    // Apply the actions to this CTA's candidates: histogram the next digit (kDselHist, or the
    // selected digit's rows for kDselContinue); write rows above the selected digit (and the digit
    // itself for kDselTakeAll) to slot take_base + above[d] + this CTA's offset in digit d + rank;
    // stage the selected digit's rows as launch-tagged composites for kDselStage.
    uint32_t next_merged = 0u;
#pragma unroll 1
    for (int q = 0; q < kSeqLen; ++q) {
      const DselQ& sq = st[q];
      const int action = sq.action;
      if (action == kDselNone) continue;
      if (action == kDselHist || action == kDselContinue) next_merged |= 1u << q;
      const uint64_t m_cur = sq.m_cur, p_cur = sq.p_cur, m_nxt = sq.m_nxt, p_nxt = sq.p_nxt;
      const int lo_cur = sq.lo_cur, lo_nxt = sq.lo_nxt;
      const uint32_t dm_cur = static_cast<uint32_t>(sq.dm_cur),
                     dm_nxt = static_cast<uint32_t>(sq.dm_nxt);
      const uint32_t t = static_cast<uint32_t>(sq.t);
      const int take_base = sq.take_base;
      const uint64_t* lq = list + q * kListCap;
      const int n = static_cast<int>(cnt[q]);
      const uint32_t hq_s = hist_s + static_cast<uint32_t>(q * kHistBins) * 4u;
      uint32_t* const off_q = soff + q * kHistBins;
      uint32_t* const rank_q = srank + q * kHistBins;
      const uint32_t* const above_q = sabove + q * kHistBins;
      for (int i = tid; i < n; i += kConsumerThreads) {
        const uint64_t c = lq[i];
        if (action == kDselHist) {
          if ((c & m_nxt) == p_nxt)
            p4_hist_inc(hq_s + (static_cast<uint32_t>(c >> lo_nxt) & dm_nxt) * 4u);
          continue;
        }
        if ((c & m_cur) != p_cur) continue;
        const uint32_t d = static_cast<uint32_t>(c >> lo_cur) & dm_cur;
        if (d < t) continue;
        if (d == t) {
          if (action == kDselContinue) {
            p4_hist_inc(hq_s + (static_cast<uint32_t>(c >> lo_nxt) & dm_nxt) * 4u);
            continue;
          }
          if (action == kDselStage) {
            const uint32_t idx = off_q[t] + atomicAdd(rank_q + t, 1u);
            st_relaxed_gpu_u64(staged + static_cast<size_t>(q) * arena::kCompScratchCap + idx,
                               c | tag_bits);
            continue;
          }
        }
        const int slot =
            take_base + static_cast<int>(above_q[d] + off_q[d] + atomicAdd(rank_q + d, 1u));
        row.selects[q * kTopk + slot] =
            static_cast<int32_t>(g2b_smem::kIdMask - static_cast<uint32_t>(c & g2b_smem::kIdMask));
      }
    }
    consumer_sync();

    // This CTA's slots of a finished query are written: release them to the query's done counter.
    if (tid < kSeqLen && (st[tid].action == kDselStage || st[tid].action == kDselTakeAll)) {
      red_release_gpu_add(reinterpret_cast<uint32_t*>(ctx.arena + kDone) + tid, 1u);
    }
    if (next_merged == 0u) break;
    merged = next_merged;
    ++round;
  }

  // The scoring CTAs of query (cta & 3) publish its records [0, base >> 1) once every CTA's slots
  // are visible; the Group2b CTA publishes the rest after its select.
  const int pq = ctx.cta & (kSeqLen - 1);
  if ((dist_mask >> pq) & 1u) {
    if (tid == 0) {
      const uint32_t* done = reinterpret_cast<const uint32_t*>(ctx.arena + kDone) + pq;
      while (load_relaxed_gpu(done) < static_cast<uint32_t>(n_active)) spin_backoff();
      fence_acquire_gpu();
    }
    consumer_sync();
    p5_publish_sure_records(args, ctx, n_active, dist_mask, st[pq].base);
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Group1 consumers (CTAs 0..127, threads 0..255): P1/P1b -> C0 -> P2 -> C1 -> P3 -> C3 -> [sparse:
// wait C3,C2 -> P4 -> P5]
// -> C4 -> [CTAs 0..7: (sparse: wait C5) -> P6 -> C6 (-> CTA 0: kFlagAck pacing)] -> wait
// C6 -> P7. kModeComputeOnly keeps every wait / arrive of this table and skips only the P6 / P7
// bodies and the ack pacing (mode_runs_comm): the compute-only form, the communications not being
// exercised.
// ---------------------------------------------------------------------------------------------------------------
template <bool CompactDense, bool SparseComputeOnly>
__device__ __forceinline__ void consumer_main(const IndexerArgs& args, OpCtx& ctx) {
  const int cta = ctx.cta;
  // At a fully dense step the public contract checks only the four newly
  // written indexer_k_cache rows. Compute indexer.wk with CTAs 64..67 and publish C0 with
  // the same per-launch total; CTA 0 advances otherwise-unused phase words so
  // a later sparse launch can keep using the global epoch target.
  if constexpr (CompactDense) {
    if (cta >= 64 && cta < kP1Ctas) {
      op_p1_rmsnorm_gemv_q_a_indexer_wk<true, true>(args, ctx);
      boundary_arrive(ctx, kSyncC0, kWordSum / 4u);
      // In the compact five-CTA dense launch virtual CTA 0 is absent.  Let
      // the first indexer.wk worker keep the otherwise-unused phase words aligned
      // with the host epoch so a later sparse call starts from the right
      // target.
      if (cta == 64 && ctx.tid == 0) {
        red_release_gpu_add(sync_word(ctx, kSyncC1), kWordSum);
        red_release_gpu_add(sync_word(ctx, kSyncC3), kWordSum);
        red_release_gpu_add(sync_word(ctx, kSyncC4), kWordSum);
      }
    }
    return;
  }
  // P1: CTAs 0..67 q_a_proj/indexer.wk GEMV, 68..71 indexer.weights_proj, 72..127 idle; the idle
  // CTAs still arrive C0 (+2) at 0x6320.
  if (cta < kQaIndexerWkCtas) {
    op_p1_rmsnorm_gemv_q_a_indexer_wk<SparseComputeOnly>(args, ctx);
  } else if (cta < kP1AllCtas) {
    op_p1b_gemv_indexer_weights_proj<SparseComputeOnly>(args, ctx);
  }
  // Only the 72 CTAs that actually produce q_a/ki/idx_scores publish C0.
  // CTA 0 carries the complement so the monotonic word still advances by
  // exactly 0x100, and the target cannot be reached until every producer
  // has completed its CTA-local stores.
  if (cta < kP1AllCtas) {
    const uint32_t weight = cta == 0 ? kWordSum - (kP1AllCtas - 1) : 1u;
    boundary_arrive(ctx, kSyncC0, weight);
  }
  if constexpr (!CompactDense && !SparseComputeOnly) {
    if (cta == kP1AllCtas && ctx.sparse && ctx.cur_pos > kLastDenseCurPos &&
        !long_context(ctx.cur_pos))
      hist_block_clear(ctx, ctx_score(ctx.smem).hist_blk);
  }

  // P2: every Group1 CTA needs all q_a blocks: wait C0.
  boundary_wait(ctx, kSyncC0);
  op_p2_rmsnorm_gemv_indexer_wq_b<SparseComputeOnly>(args, ctx);
  boundary_arrive(ctx, kSyncC1, kArriveC1);  // atomic +2 [B+0x10]

  // P3: only 4*ceil(N/16) CTAs consume IQ.  In compute-only mode the other
  // scoring CTAs acquire the completed P2 state transitively through C3, so
  // they can skip the redundant C1 poll/fence/barrier.
  const int n_heads =
      SparseComputeOnly
          ? kIdxHeads
          : (has_inline_ctx(args) ? kIdxHeads : static_cast<int>(ctx_p3(ctx.smem).n_heads));
  const bool fused_p3 = !SparseComputeOnly && !ctx.null_math && p3_fused(args, ctx);
  const int p3_producers = fused_p3 ? 0 : p3_ctas(n_heads);
  if constexpr (SparseComputeOnly) {
    if (cta < p3_producers) {
      boundary_wait(ctx, kSyncC1);
    }
  } else {
    if (!fused_p3 && (args.mode != kModeComputeOnly || cta < p3_producers)) {
      boundary_wait(ctx, kSyncC1);
    }
  }
  if (cta < p3_producers) op_p3_rotate_q<SparseComputeOnly>(args, ctx);
  // P3 has only p3_ctas(n_heads) producers (8 in this configuration).  Idle
  // Group1 CTAs need not contend on the release word.
  if (cta < p3_producers) {
    const uint32_t weight = cta == 0 ? kWordSum - (p3_producers - 1) : 1u;
    boundary_arrive(ctx, kSyncC3, weight);
  }

  // P2 needs every Group1 CTA, but P4/P5 need only CTAs that own a K tile.
  // Once an inactive compute-only CTA has published C1 it has no remaining
  // producer or consumer role.
  const int score_producers = n_active_ctas(ctx.cur_pos);
  if constexpr (SparseComputeOnly) {
    if (cta >= score_producers) {
      return;
    }
  } else {
    if (args.mode == kModeComputeOnly && cta >= score_producers) {
      return;
    }
  }

  // Sparse steps only (is_sparse: cur_pos >= 2044).
  if (SparseComputeOnly || ctx.sparse) {
    if constexpr (SparseComputeOnly || CompactDense) {
      boundary_wait2(ctx, kSyncC3, kSyncC2);
      op_p4_score_tiles<SparseComputeOnly>(args, ctx);
    } else {
      op_p4_score_tiles<false, true>(args, ctx);
    }
    // For cur_pos 2044..2047 each query has at most 2051 causal rows, so
    // Group2b can select exactly by dropping at most three minima. Avoid the
    // histogram/S-barrier/radix pipeline in that short sparse regime.
    if (ctx.cur_pos > kLastDenseCurPos && !long_context(ctx.cur_pos))
      op_p5_select_stage_b<SparseComputeOnly, !SparseComputeOnly && !CompactDense>(args, ctx);
  }
  // C4 depends only on CTAs that own at least one K tile.  Use the same
  // last-producer completion encoding as C0/C3; all inactive scoring CTAs
  // can leave without a CTA barrier or global atomic.
  if (cta < score_producers) {
    const uint32_t weight = cta == 0 ? kWordSum - (score_producers - 1) : 1u;
    boundary_arrive(ctx, kSyncC4, weight);
  }
  if constexpr (!CompactDense && !SparseComputeOnly) {
    if (ctx.sparse && ctx.cur_pos > kLastDenseCurPos && !long_context(ctx.cur_pos) &&
        cta < score_producers && mode_runs_comm(args.mode) && !ctx.null_math) {
      // A launch with a distributed refine publishes inside dsel_run; otherwise the sure records go
      // out after C4.
      const bool published =
          g2b_refine_fast(args, ctx.cur_pos) && dsel_run(args, ctx, score_producers);
      if (!published) {
        boundary_wait(ctx, kSyncC4);
        p5_publish_sure_records(args, ctx, score_producers);
      }
    }
  }

  // Compute-only has no consumer after Group2b. Kernel completion already
  // orders the refined selected_ids stores, so keeping the C5 -> empty P6 ->
  // C6 -> empty P7 chain only adds two grid-wide word round trips.
  if (args.mode == kModeComputeOnly) {
    return;
  }

  // P6: CTAs 0..7 push the ids; sparse launches wait C5 first.
  if (cta < kP6Ctas) {
    const bool published_by_g2b = !CompactDense && !SparseComputeOnly && ctx.sparse &&
                                  ctx.cur_pos > kLastDenseCurPos && !long_context(ctx.cur_pos);
    if (ctx.sparse && !published_by_g2b) boundary_wait(ctx, kSyncC5);
    if (cta == 0)
      assert_ids_src_contract(
          ctx.smem);  // row 6 +0x00 == row 4 +0x30 == row 5 +0x08 (one selected_ids buffer)
    if (mode_runs_comm(args.mode) && !published_by_g2b) op_p6_broadcast_ids(args, ctx);
    boundary_arrive(ctx, kSyncC6, kArriveC6);  // barrier; atomic +0x20 [B+0x60]
    if (mode_runs_comm(args.mode)) op_p6_wait_acks(args, ctx);
    // end of publish, before reduce); placed after this CTA's C6 arrive so the
    // other 127 CTAs are not held; compute_only: skipped
  }
  // P7: all 128 wait C6 then run the collective and exit.
  boundary_wait(ctx, kSyncC6);
  if (args.mode == kModePersistent)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // PDL: release the MoE
  if (mode_runs_comm(args.mode))
    op_p7_collective(
        args, ctx);  // compute_only: no reduce / NaN fills, the kernel exits after the C6 wait
}
__device__ __forceinline__ void g2a_idle_warm(const IndexerArgs& args, int cta, int cur_pos) {
  if (args.mode != kModePersistent || has_inline_ctx(args) || cta != kGroup2aBase + 1) return;
  const int ptid = static_cast<int>(threadIdx.x) - kConsumerThreads;
  const bool sparse_select = cur_pos > kLastDenseCurPos;
  const uint8_t* src = nullptr;
  uint32_t bytes = 0;
  if (ptid == 0) {
    src = args.arena + arena::kP3HeadCounters;
    bytes = 128u;
  } else if (ptid == 1 && sparse_select) {
    src = args.arena + arena::kHist;
    bytes = uint32_t(arena::kHistBytes);
  } else if (ptid == 2 && sparse_select) {
    src = args.arena + arena::kSelects;
    bytes = uint32_t(arena::kSelectsBytes);
  } else if (ptid >= 3 && ptid < 3 + kSeqLen && sparse_select) {
    src = args.arena + arena::kCompScratch + size_t(ptid - 3) * arena::kCompScratchCap * 8;
    bytes = 16384u;
  }
  if (src != nullptr) prefetch_l2_bytes(src, bytes);
}

// ---------------------------------------------------------------------------------------------------------------
// Group2a (CTAs 128..131): prefetch warps of CTA 128 feed slot 0 (others exit);
// consumers wait C0, only CTA 128 computes, all four arrive C2 (+0x40).
// ---------------------------------------------------------------------------------------------------------------
template <bool CompactDense, bool SparseComputeOnly>
__device__ __forceinline__ void group2a_main(const IndexerArgs& args, uint8_t* smem, int cta,
                                             int cur_pos, uint32_t epoch) {
  if (threadIdx.x >= kConsumerThreads) {
    if (cta == kGroup2aWorker) {
      PrefetchCtx p = make_prefetch_ctx(args, cta - kGroup2aBase, smem, epoch);
      g2a_prefetch_feed(p);
    } else if constexpr (!CompactDense && !SparseComputeOnly) {
      g2a_idle_warm(args, cta, cur_pos);
    }
    return;
  }
  // Compute-only has a single real Group2a producer.
  if constexpr (!CompactDense && !SparseComputeOnly) {
    if (args.mode == kModeComputeOnly && cta != kGroup2aWorker) return;
  }
  OpCtx ctx = make_op_ctx < CompactDense || SparseComputeOnly,
        SparseComputeOnly > (args, kOpG2a, cta - kGroup2aBase, kGroup2aCtas, smem, cur_pos, epoch,
                             kConsumerThreads);
  if constexpr (!CompactDense && !SparseComputeOnly) {
    if (cta == kGroup2aWorker && !has_inline_ctx(args)) {
      assert_cur_pos_contract(ctx_g2a(smem).cur_pos_ptr, cur_pos);  // row 2 +0x08 == row 4 +0x20
    }
  }
  boundary_wait(ctx, kSyncC0);  //  (ki of P1 CTAs 64..67)
  if (cta == kGroup2aWorker) op_g2a_layernorm_rope_k<CompactDense || SparseComputeOnly>(args, ctx);
  // generic persistent path: all four Group2a CTAs arrive on every launch (dense or sparse), so the
  // weight must be kArriveC2 (4 x 0x40 = kWordSum).  Using kWordSum on dense launches would
  // over-count C2 by 0x300 per dense launch and lets later sparse launches pass boundary_wait2(C3,
  // C2) before this launch's Group2a is done.
  boundary_arrive(
      ctx, kSyncC2,
      (CompactDense || SparseComputeOnly || args.mode == kModeComputeOnly) ? kWordSum : kArriveC2);
}

// ---------------------------------------------------------------------------------------------------------------
// Group2b (CTAs 132..147, all 384 threads): dense -> arrive C5 and exit;
// wait C4 (named barrier 4, 384 threads); CTA 132 + q (q < 4) refines; all 16 arrive C5 (+0x10).
// ---------------------------------------------------------------------------------------------------------------
template <bool SparseComputeOnly>
__device__ __forceinline__ void group2b_main(const IndexerArgs& args, uint8_t* smem, int cta,
                                             int cur_pos, uint32_t epoch) {
  OpCtx ctx = make_op_ctx<SparseComputeOnly, SparseComputeOnly>(
      args, kOpG2b, cta - kGroup2bBase, kGroup2bCtas, smem, cur_pos, epoch, kBlock);
  if constexpr (!SparseComputeOnly) {
    if (!ctx.sparse) {
      if (args.mode != kModeComputeOnly) boundary_arrive(ctx, kSyncC5, kArriveC5);
      return;
    }
    // Only CTAs 132..135 own queries, and compute-only has no C5 consumer.
    if (args.mode == kModeComputeOnly && ctx.cta >= kGroup2bWorkCtas) return;
    if (ctx.cta < kGroup2bWorkCtas && !has_inline_ctx(args)) {
      assert_cur_pos_contract(ctx_refine(smem).cur_pos_ptr, cur_pos);
    }
  }
  if (long_context(cur_pos)) {
    boundary_wait(ctx, kSyncC4);
    if (ctx.cta < kGroup2bWorkCtas && !ctx.null_math) {
      // Long rows always use the workspace-allocated compact-pitch score/candidate
      // tensors; the fixed-stride arena regions keep their kMaxSeqLen capacity.
      if (has_inline_ctx(args) || !(args.flags & kFlagCompactPitch)) __trap();
      const CtxRow5& row = ctx_refine(smem);
      const int q = ctx.cta;
      const int length = total_len(cur_pos);
      const int stride = logits_row_stride(args.flags, cur_pos);
      const auto* maxima = reinterpret_cast<const uint64_t*>(row.cand_ws) +
                           size_t(q) * long_topk::summary_stride(length);
      long_topk::select(row.logits + size_t(q) * stride, cur_pos + q + 1, maxima,
                        row.selects + size_t(q) * kTopk, smem + smem::kRing, true);
    }
    if (args.mode != kModeComputeOnly) boundary_arrive(ctx, kSyncC5, kArriveC5);
    return;
  }
  bool refined = false;
  if constexpr (!SparseComputeOnly) {
    if (ctx.cta < kGroup2bWorkCtas && g2b_refine_fast(args, ctx.cur_pos) && !ctx.null_math) {
      // P5's S generation of this launch cannot have completed yet (it follows P4), so this is the
      // pre-arrival sense value.
      uint32_t sense_word = 0u;
      if (ctx.tid == 0) sense_word = load_relaxed_gpu(sync_word(ctx, kSyncS));
      refined = op_g2b_refine_fast(args, ctx, sense_word);
    }
  }
  if (!refined) {
    boundary_wait(ctx, kSyncC4);
    if (ctx.cur_pos <= kLastDenseCurPos) {
      op_g2b_near_topk<SparseComputeOnly>(args, ctx);
    } else {
      op_g2b_refine<SparseComputeOnly, !SparseComputeOnly>(args, ctx);
    }
  }
  // No publish phase consumes C5 in compute-only mode. The launch itself is
  // the completion fence for the output tensor.
  if (args.mode == kModeComputeOnly) return;
  boundary_arrive(ctx, kSyncC5, kArriveC5);  // barrier 4,0x180; atomic +0x10 [B+0x50]
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

namespace flashinfer::glm5_tp8::indexer_q4 {

// Rank 0's persistent kernel.  launch_indexer_q4 launches indexer_persistent<false, false>; the
// CompactDense and SparseComputeOnly arguments select reduced-grid compute-only forms of the same
// body.
template <bool CompactDense, bool SparseComputeOnly>
__global__ __launch_bounds__(kBlock,
                             1) void indexer_persistent(const __grid_constant__ IndexerArgs args) {
  asm volatile("griddepcontrol.wait;" ::
                   : "memory");  // PDL: launched behind the previous layer's MoE; nothing before
                                 // this touches memory
  extern __shared__ __align__(1024) uint8_t dyn[];
  const int physical_cta = static_cast<int>(blockIdx.x);
  // Dense compute-only needs exactly four indexer.wk CTAs and the Group2a worker. Sparse
  // compute-only additionally needs all 128 Group1 CTAs and four Group2b query workers, but not the
  // other 15 idle Group2 CTAs.
  const int cta =
      CompactDense
          ? (physical_cta < 4 ? 64 + physical_cta : kGroup2aBase)
          : (SparseComputeOnly && physical_cta > kGroup2aBase ? physical_cta + 3 : physical_cta);
  if (cta >= kGrid)
    __trap();  // every CTA at entry
               // both unreachable at grid 148
  const bool p1_preissue = !CompactDense && !SparseComputeOnly && p1_preissue_enabled(args, cta);
  if (p1_preissue) p1_preissue_pages(args, dyn, cta);

  // The epoch / cur_pos chain (cur_pos is reached through the ctx table's cur_pos pointer: two
  // dependent global loads) runs on the last prefetch-warp thread, concurrently with the ctx-table
  // copy (threads 0..127), instead of after thread 0's copy word.
  if (threadIdx.x == kBlock - 1)
    load_epoch_cur_pos<CompactDense || SparseComputeOnly>(args, dyn, cta, true);
  ctx_table_to_smem<CompactDense, SparseComputeOnly>(args, dyn, cta);
  if (!p1_preissue) ring_init_barriers(dyn);
  __syncthreads();  // the only block-wide barrier
  const int cur_pos = smem_cur_pos(dyn);
  if (long_context(cur_pos) && (has_inline_ctx(args) || !(args.flags & kFlagCompactPitch) ||
                                total_len(cur_pos) > long_topk::kLogicalMax)) {
    __trap();
  }
  const uint32_t epoch = smem_epoch(dyn);

  if (cta >= kGroup2bBase) {
    group2b_main<SparseComputeOnly>(args, dyn, cta, cur_pos, epoch);
    return;
  }
  if (cta >= kGroup2aBase) {
    group2a_main<CompactDense, SparseComputeOnly>(args, dyn, cta, cur_pos, epoch);
    return;
  }
  if (threadIdx.x >= kConsumerThreads) {
    setmaxnreg_prefetcher();
    PrefetchCtx p = make_prefetch_ctx(args, cta, dyn, epoch);
    prefetch_main<CompactDense, SparseComputeOnly>(p, cur_pos);
    return;
  }
  setmaxnreg_consumer();
  OpCtx ctx = make_op_ctx < CompactDense || SparseComputeOnly,
        SparseComputeOnly > (args, kOpP1, cta, kGroup1Ctas, dyn, cur_pos, epoch, kConsumerThreads);
  consumer_main<CompactDense, SparseComputeOnly>(args, ctx);
}

// Shared-index mode: the caller's selection stays bound as the selected-ID buffer of rows 4/5/6.
// Each call republishes it through the LL records; consumed LL tags are never reused.  The
// attention ranks still run their full MLA kernels for this layer and send their partials.
__global__ __launch_bounds__(kBlock,
                             1) void indexer_shared(const __grid_constant__ IndexerArgs args) {
  asm volatile("griddepcontrol.wait;" ::
                   : "memory");  // PDL: launched behind the previous layer's MoE; nothing before
                                 // this touches memory
  extern __shared__ __align__(1024) uint8_t dyn[];
  const int cta = static_cast<int>(blockIdx.x);
  if (threadIdx.x == kBlock - 1) load_epoch_cur_pos<false>(args, dyn, cta, true);
  ctx_table_to_smem<false, false>(args, dyn, cta);
  __syncthreads();
  if (threadIdx.x >= kConsumerThreads) return;
  OpCtx ctx = make_op_ctx<false, false>(args, kOpP6, cta, kGrid, dyn, smem_cur_pos(dyn),
                                        smem_epoch(dyn), kConsumerThreads);
  // Full calls wait for 0x100 * epoch. Advance skipped computation words so
  // a later full layer remains live after any number of shared calls.
  if (cta == 0 && ctx.tid == 0) {
    for (int word = kSyncC0; word <= kSyncC5; word += 0x10) {
      red_release_gpu_add(sync_word(ctx, word), kWordSum);
    }
  }
  if (cta < kP6Ctas) {
    if (cta == 0) assert_ids_src_contract(ctx.smem);
    op_p6_broadcast_ids(args, ctx);
    boundary_arrive(ctx, kSyncC6, kArriveC6);
    op_p6_wait_acks(args, ctx);
  }
  if (cta < kP7Ctas) {
    boundary_wait(ctx, kSyncC6);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // PDL: release the MoE
    ctx.op = kOpP7;
    op_p7_collective(args, ctx);
  }
}

}  // namespace flashinfer::glm5_tp8::indexer_q4

#include "launch_util.cuh"
namespace flashinfer::glm5_tp8 {
KernelResources initialize_indexer_q4(int device) {
  using namespace indexer_q4;
  attention_detail::configure(reinterpret_cast<const void*>(&indexer_shared), kGrid, kBlock,
                              smem::kDynamicBytes, device);
  return attention_detail::configure(
      reinterpret_cast<const void*>(&indexer_persistent<false, false>), kGrid, kBlock,
      smem::kDynamicBytes, device);
}
cudaError_t launch_indexer_q4(const AttentionArguments& p, cudaStream_t stream, bool pdl,
                              bool cooperative) {
  using namespace indexer_q4;
  IndexerArgs a{};
  a.n_rows = 8;
  a.rank = 0;
  a.ctx_table = reinterpret_cast<const CtxTable*>(p.context);
  a.arena = p.peer_arenas[0];
  a.sync_words = reinterpret_cast<uint32_t*>(a.arena + arena::kSync);
  a.epoch_ctr = p.epoch_ctr;
  for (int r = 0; r < 8; ++r) a.peers[r] = p.peer_arenas[r];
  a.cur_pos = -1;
  a.mode = kModePersistent;
  a.ids_direct = a.skip_done_wait = a.rank0_reduce = a.dense_ids = 1;
  a.flags = kFlagCompactPitch;
  const void* kernel = p.shared_indexer
                           ? reinterpret_cast<const void*>(&indexer_shared)
                           : reinterpret_cast<const void*>(&indexer_persistent<false, false>);
  return attention_detail::launch(kernel, a, kGrid, kBlock, smem::kDynamicBytes, stream, pdl,
                                  cooperative);
}
}  // namespace flashinfer::glm5_tp8
