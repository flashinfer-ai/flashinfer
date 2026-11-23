/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include "internal.cuh"
#include "q4_types.cuh"

// MoE layer for four decode tokens on one TP rank of eight: one persistent launch of kGrid = 148
// CTAs x kBlock = 320 threads (warps 0..7 compute, warp 8 streams weight pages, warp 9 is the
// auxiliary warp of the routed CTAs).  CTA roles (q4_types.cuh):
//   CTAs 0..127   routed, token t = cta >> 5, output slice c = cta & 31: x RMSNorm and the router
//                 GEMV of kExpertsPerRouterCta experts, top-8 over the gathered expert scores,
//                 gate/up of the selected experts, then the Down GEMV of the slice's
//                 kDownColsPerCta columns plus the shared-expert term, the TP all-reduce through
//                 the peers' inboxes and the residual add.
//   CTAs 128..143 shared expert: RMSNorm (kSharedNormCtas CTAs, one token each), gate/up, Down.
//   CTAs 144..147 sentinel helpers: refill the readiness sentinels of the score and mid outputs,
//                 then arrive on the kSyncInit word.
// Weight pages stream through a kRingSlots-slot shared-memory ring guarded by full/empty
// mbarriers; a page is one cp.async.bulk, a Down page two (weights and scales).  Cross-CTA values
// are polled in place: a score or mid word is ready once it no longer holds its pending encoding
// (kScorePending / kMidPending).  Each CTA's launch_ctr entry counts its launches; the helpers'
// refill for launch `generation` is complete when the kSyncInit word reaches
// kSentinelHelperCtas * generation.  launch_ctr entries and the kSyncInit word only grow; the
// helpers re-arm the score/mid words every launch, and CTA 0's auxiliary warp clears the next
// launch's kSyncProgress bank (two banks alternate by epoch parity).

namespace flashinfer::glm5_tp8::moe_q4 {

constexpr uint32_t kScorePending = 0xffffffffu;
constexpr uint32_t kMidPending = 0xffffu;

// Readiness reserves exact encodings, not the NaN class. Only published NaNs
// are canonicalized; finite values, infinities and signed zero keep their bits.
__device__ __forceinline__ uint32_t score_payload_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : bits;
}
__device__ __forceinline__ uint16_t mid_payload_bits(uint16_t bits) {
  return (bits & 0x7fffu) > 0x7f80u ? uint16_t(0x7fc0u) : bits;
}
__device__ __forceinline__ bool mid_word_pending(uint32_t bits) {
  return (bits & 0xffffu) == kMidPending || (bits >> 16) == kMidPending;
}

__device__ __forceinline__ void consumer_sync() {
  asm volatile("bar.sync 4, %0;" ::"n"(kConsumerThreads) : "memory");
}
__device__ __forceinline__ void spin_backoff() {}

__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

__device__ __forceinline__ void grid_dependency_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void publish_score_quad(void* address, float a, float b, float c,
                                                   float d) {
  // Each scalar component matches the b32 polling loads. Consumers require all
  // components ready, so vector atomicity is neither needed nor assumed.
  asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address),
               "r"(score_payload_bits(a)), "r"(score_payload_bits(b)), "r"(score_payload_bits(c)),
               "r"(score_payload_bits(d))
               : "memory");
}

__device__ __forceinline__ void ld_relaxed_gpu_v2u32(const void* address, uint32_t& lo,
                                                     uint32_t& hi) {
  asm volatile("ld.relaxed.gpu.global.v2.b32 {%0, %1}, [%2];"
               : "=r"(lo), "=r"(hi)
               : "l"(address)
               : "memory");
}

__device__ __forceinline__ void st_relaxed_gpu_u32(void* address, uint32_t value) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void st_release_gpu_u32(void* address, uint32_t value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}
__device__ __forceinline__ void st_relaxed_gpu_v4u32(void* address, uint32_t a, uint32_t b,
                                                     uint32_t c, uint32_t d) {
  asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(a),
               "r"(b), "r"(c), "r"(d)
               : "memory");
}

__device__ __forceinline__ uint32_t ld_relaxed_gpu_u32(const void* address) {
  uint32_t value;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ uint32_t word_arrive_release(uint32_t* word, uint32_t add) {
  uint32_t old;
  asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(word), "r"(add)
               : "memory");
  return old;
}

__device__ __forceinline__ void word_arrive_release_only(uint32_t* word, uint32_t add) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(word), "r"(add) : "memory");
}

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

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ bool mbar_try_wait_parity(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred P1;\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\tselp.u32 "
      "%0, 1, 0, P1;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ bool mbar_test_wait_parity(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred P1;\n\tmbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n\tselp.u32 "
      "%0, 1, 0, P1;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait_parity(bar, parity)) {
  }
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ uint64_t l2_policy(const void* src_global, uint32_t bytes) {
  uint64_t policy = 0;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  (void)src_global;
  (void)bytes;
  return policy;
}
__device__ __forceinline__ uint64_t l2_policy_reuse(const void* src_global, uint32_t bytes) {
  uint64_t policy;
  asm volatile("createpolicy.range.global.L2::evict_last.L2::evict_first.b64 %0, [%1], %2, %3;"
               : "=l"(policy)
               : "l"(src_global), "r"(bytes), "r"(bytes));
  return policy;
}
__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst_smem, const void* src_global,
                                              uint32_t bytes, uint32_t bar, bool retain_l2) {
  if (retain_l2) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1], %2, [%3], %4;" ::"r"(dst_smem),
        "l"(src_global), "r"(bytes), "r"(bar), "l"(l2_policy_reuse(src_global, bytes))
        : "memory");
    return;
  }
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], %4;" ::"r"(dst_smem),
      "l"(src_global), "r"(bytes), "r"(bar), "l"(l2_policy(src_global, bytes))
      : "memory");
}

__device__ __forceinline__ void l2_prefetch_bulk(const void* src_global, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src_global),
               "r"(bytes), "l"(l2_policy(src_global, bytes))
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
__device__ __forceinline__ void ldgsts_wait_1() {
  asm volatile("cp.async.wait_group 1;" ::: "memory");
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
__device__ __forceinline__ void lds_u32x2(uint32_t address, uint32_t& lo, uint32_t& hi) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(lo), "=r"(hi) : "r"(address));
}
__device__ __forceinline__ void lds_u32x4(uint32_t address, uint32_t& a, uint32_t& b, uint32_t& c,
                                          uint32_t& d) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(address));
}
__device__ __forceinline__ uint32_t lds_u16(uint32_t address) {
  uint16_t value;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(value) : "r"(address));
  return value;
}
__device__ __forceinline__ void sts_u32(uint32_t address, uint32_t value) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(address), "r"(value) : "memory");
}
__device__ __forceinline__ void sts_f32(uint32_t address, float value) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(address), "f"(value) : "memory");
}
__device__ __forceinline__ void sts_u16(uint32_t address, uint16_t value) {
  asm volatile("st.shared.u16 [%0], %1;" ::"r"(address), "h"(value) : "memory");
}
__device__ __forceinline__ void sts_u32x2(uint32_t address, uint32_t lo, uint32_t hi) {
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(address), "r"(lo), "r"(hi) : "memory");
}
__device__ __forceinline__ void sts_u32x4(uint32_t address, uint32_t a, uint32_t b, uint32_t c,
                                          uint32_t d) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(address), "r"(a), "r"(b), "r"(c),
               "r"(d)
               : "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* generic) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(generic));
}

__device__ __forceinline__ float bf16_bits_to_f32(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}
__device__ __forceinline__ float bf16_lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t pair) {
  return __uint_as_float(pair & 0xffff0000u);
}
__device__ __forceinline__ uint32_t pack_bf16_pair(float low, float high) {
  uint32_t bits;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bits) : "f"(high), "f"(low));
  return bits;
}
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float value) {
  return static_cast<uint16_t>(pack_bf16_pair(value, 0.0f) & 0xffffu);
}
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigmoid_approx(float x) {
  return fmaf(tanh_approx(0.5f * x), 0.5f, 0.5f);
}

__device__ __forceinline__ float silu_approx(float g) {
  const float h = 0.5f * g;
  return fmaf(h, tanh_approx(h), h);
}

__device__ __forceinline__ uint32_t f32_monotonic_key(float v) {
  const uint32_t bits = __float_as_uint(v);
  return (v >= 0.0f) ? (bits | 0x80000000u) : ~bits;
}

struct HmmaAcc {
  float v[4];
};

struct GuQuant128 {
  uint32_t q;
  float scale;
};

__device__ __forceinline__ GuQuant128 gu_quant128_regs(uint32_t x01, uint32_t x23) {
  float x0 = bf16_lo(x01), x1 = bf16_hi(x01);
  float x2 = bf16_lo(x23), x3 = bf16_hi(x23);
  float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
  const float scale = __fmul_rn(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
  const float inv = __fdiv_rn(1.0f, scale);
  uint16_t q01, q23;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(q01)
               : "f"(__fmul_rn(x1, inv)), "f"(__fmul_rn(x0, inv)));
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(q23)
               : "f"(__fmul_rn(x3, inv)), "f"(__fmul_rn(x2, inv)));
  return {uint32_t(q01) | (uint32_t(q23) << 16), scale};
}
__device__ __forceinline__ GuQuant128 gu_quant128(uint32_t block_s, int lane) {
  return gu_quant128_regs(lds_u32(block_s + lane * 8), lds_u32(block_s + lane * 8 + 4));
}

__device__ __forceinline__ void gu_weight_k32(uint32_t tile_s, int lane, uint32_t (&a)[4]) {
  const int source_lane = (lane & ~3) + 2 * (lane & 1);
  const uint32_t addr = tile_s + source_lane * 8 + ((lane & 2) ? 4 : 0);
  const uint32_t x0 = lds_u32(addr), x1 = lds_u32(addr + 8);
  const uint32_t x2 = lds_u32(addr + kGateUpKStepBytes), x3 = lds_u32(addr + kGateUpKStepBytes + 8);
  a[0] = (x0 & 0xffffu) | (x1 << 16);
  a[1] = (x0 >> 16) | (x1 & 0xffff0000u);
  a[2] = (x2 & 0xffffu) | (x3 << 16);
  a[3] = (x2 >> 16) | (x3 & 0xffff0000u);
}
__device__ __forceinline__ void gu_mma_fp8(HmmaAcc& acc, const uint32_t (&a)[4],
                                           const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(acc.v[0]), "+f"(acc.v[1]), "+f"(acc.v[2]), "+f"(acc.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v16) {
  uint32_t r;
  asm("{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}"
      : "=r"(r)
      : "r"(v16));
  return r;
}
__device__ __forceinline__ void fp8x8_to_f16_fragment(uint32_t lo, uint32_t hi, uint32_t (&a)[4]) {
  asm volatile(
      "{\n\t.reg .b16 h0,h1,h2,h3;\n\tmov.b32 {h0,h1}, %4;\n\tmov.b32 {h2,h3}, %5;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, h0;\n\tcvt.rn.f16x2.e4m3x2 %1, h1;\n\tcvt.rn.f16x2.e4m3x2 %2, "
      "h2;\n\tcvt.rn.f16x2.e4m3x2 %3, h3;\n\t}"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(lo), "r"(hi));
}
__device__ __forceinline__ void hmma_f16(HmmaAcc& acc, const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(acc.v[0]), "+f"(acc.v[1]), "+f"(acc.v[2]), "+f"(acc.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void codes4_to_f16(uint32_t q, uint32_t& lo, uint32_t& hi) {
  lo = e4m3x2_to_f16x2(q & 0xffffu);
  hi = e4m3x2_to_f16x2(q >> 16);
}

__device__ __forceinline__ void down_weight_k32(uint32_t tile_s, int lane, uint32_t (&a)[4]) {
  const int source_lane = (lane & ~3) + 2 * (lane & 1);
  const int half = (lane >> 1) & 1;
  uint32_t v0, v1, v2, v3;
  lds_u32x4(tile_s + (source_lane + half) * 16, v0, v1, v2, v3);
  const uint32_t cross01 = __shfl_xor_sync(0xffffffffu, half ? v0 : v1, 2);
  const uint32_t cross23 = __shfl_xor_sync(0xffffffffu, half ? v2 : v3, 2);
  const uint32_t x0 = half ? cross01 : v0;
  const uint32_t x1 = half ? v1 : cross01;
  const uint32_t x2 = half ? cross23 : v2;
  const uint32_t x3 = half ? v3 : cross23;
  a[0] = (x0 & 0xffffu) | (x1 << 16);
  a[1] = (x0 >> 16) | (x1 & 0xffff0000u);
  a[2] = (x2 & 0xffffu) | (x3 << 16);
  a[3] = (x2 >> 16) | (x3 & 0xffff0000u);
}

__device__ __forceinline__ float warp_sum_down(float v) {
  v += __shfl_down_sync(0xffffffffu, v, 16);
  v += __shfl_down_sync(0xffffffffu, v, 8);
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  return v;
}

__device__ __forceinline__ float fold8(const float (&p)[8]) {
  return ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
}

__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

struct OpCtx {
  int cta;
  int tid, warp, lane;
  int t;
  int c;
  int k;
  int quarter;
  int rgS;
  bool routed;
  int rank;
  uint32_t epoch;
  uint8_t* smem;
  uint32_t smem_s;
  uint32_t item;
  const MoeArgs* args;

  __device__ __forceinline__ uint32_t s(uint32_t off) const { return smem_s + off; }
  template <typename T>
  __device__ __forceinline__ T* p(uint32_t off) const {
    return reinterpret_cast<T*>(smem + off);
  }
  __device__ __forceinline__ void sync() const { consumer_sync(); }
  __device__ __forceinline__ uint32_t acquire_page() {
    const uint32_t slot = item & (smem::kRingSlots - 1);
    mbar_wait_parity(s(smem::kFullBar) + slot * 8, (item >> smem::kRingLog2) & 1u);
    return s(smem::kRing) + slot * smem::kRingSlotBytes;
  }

  __device__ __forceinline__ void release_page() {
    const uint32_t slot = item & (smem::kRingSlots - 1);
    __syncwarp();
    if (lane == 0) mbar_arrive(s(smem::kEmptyBar) + slot * 8);
    ++item;
  }
};

struct PrefetchCtx {
  int cta;
  int ptid, pwarp, plane;
  int t, c, k, quarter, rgS;
  bool routed;
  uint32_t item;
  uint8_t* smem;
  uint32_t smem_s;
  const int32_t* sel_idx_src;
  const MoeArgs* args;
  __device__ __forceinline__ uint32_t s(uint32_t off) const { return smem_s + off; }
};

template <typename Row>
__device__ __forceinline__ const Row& ctx_row(const uint8_t* smem_base, int row) {
  return *reinterpret_cast<const Row*>(smem_base + smem::kCtx + row * smem::kCtxRowStride);
}
__device__ __forceinline__ const CtxRow0& ctx0(const uint8_t* s) { return ctx_row<CtxRow0>(s, 0); }
__device__ __forceinline__ const CtxRow1& ctx1(const uint8_t* s) { return ctx_row<CtxRow1>(s, 1); }
__device__ __forceinline__ const CtxRow2& ctx2(const uint8_t* s) { return ctx_row<CtxRow2>(s, 2); }
__device__ __forceinline__ const CtxRow3& ctx3(const uint8_t* s) { return ctx_row<CtxRow3>(s, 3); }
__device__ __forceinline__ const CtxRow4& ctx4(const uint8_t* s) { return ctx_row<CtxRow4>(s, 4); }

__device__ __forceinline__ void ctx_table_to_smem(const MoeArgs& args, uint8_t* smem) {
  const uint32_t idx = threadIdx.x;
  if (idx < 20) {
    const uint32_t row = (idx >= 2) + (idx >= 4) + (idx >= 10) + (idx >= 14);
    const uint32_t first = 2 * row + 4 * (row >= 3) + 2 * (row >= 4);
    const uint32_t col = idx - first;
    const uint64_t* src = reinterpret_cast<const uint64_t*>(&args.context.direct);
    uint64_t* dst =
        reinterpret_cast<uint64_t*>(smem + smem::kCtx + row * smem::kCtxRowStride + col * 8);
    *dst = src[idx];
  }
}

__device__ __forceinline__ void initialize_token_sentinels(const MoeArgs& args, int token) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < kCtasPerToken * 2) {
    const int score_cta = tid >> 1;
    const int half = tid & 1;
    uint8_t* dst = reinterpret_cast<uint8_t*>(args.context.direct.r1_scores_out) +
                   (size_t(token) * kCtasPerToken + score_cta) * kScoresSlotBytes + half * 16;
    st_relaxed_gpu_v4u32(dst, kScorePending, kScorePending, kScorePending, kScorePending);
  }
  if (tid < static_cast<int>(kHiddenMidTokenBytes / 16)) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(args.context.direct.r2_hidden_mid_out) +
                   size_t(token) * kHiddenMidTokenBytes + tid * 16;
    constexpr uint32_t pending = kMidPending | (kMidPending << 16);
    st_relaxed_gpu_v4u32(dst, pending, pending, pending, pending);
  }
}

__device__ __forceinline__ void wait_for_sentinel_helpers(const MoeArgs& args, const OpCtx& ctx) {
  if (ctx.tid == 0) {
    uint32_t* word = args.sync_words + (kSyncInit / sizeof(uint32_t));
    uint32_t v = ld_relaxed_gpu_u32(word);
    const uint32_t generation = ctx.routed ? ctx.epoch : lds_u32(ctx.s(smem::kSharedEpoch));
    const uint32_t target = kSentinelHelperCtas * generation;
    while (static_cast<int32_t>(v - target) < 0) {
      spin_backoff();
      v = ld_relaxed_gpu_u32(word);
    }
    fence_acquire_gpu();
  }
  ctx.sync();
}

__device__ __forceinline__ void ring_init_barriers(uint8_t* smem) {
  const uint32_t base = smem_u32(smem);
  if (threadIdx.x < smem::kRingSlots) {
    mbar_init(base + smem::kFullBar + threadIdx.x * 8, kPrefetchWarps);
    mbar_init(base + smem::kEmptyBar + threadIdx.x * 8, kConsumerWarps);
  }
  if (threadIdx.x == 0) {
    mbar_init(base + smem::kSelAllBar, 1);
    mbar_init(base + smem::kSelMineBar, 1);
    for (int e = 0; e < kTopk; ++e) mbar_init(base + smem::kMidReadyBar + e * 8, 1);
    mbar_fence_init();
  }
}

__device__ __forceinline__ void routed_early_router_pages(const MoeArgs& args, uint8_t* smem_base,
                                                          int cta) {
  const uint32_t base = smem_u32(smem_base);
#pragma unroll
  for (uint32_t i = 0; i < smem::kRingSlots; ++i) {
    mbar_init(base + smem::kFullBar + i * 8, kPrefetchWarps);
    mbar_init(base + smem::kEmptyBar + i * 8, kConsumerWarps);
  }
  mbar_init(base + smem::kSelAllBar, 1);
  mbar_init(base + smem::kSelMineBar, 1);
  mbar_init(base + smem::kSharedReadyBar, 1);
  mbar_init(base + smem::kFoldsDoneBar, 1);
#pragma unroll
  for (int u = 0; u < kRowGroupsPerCta; ++u)
    mbar_init(base + smem::kUnitBar + u * 8, kConsumerWarps);
#pragma unroll
  for (int e = 0; e < kTopk; ++e) mbar_init(base + smem::kMidReadyBar + e * 8, 1);
#pragma unroll
  for (uint32_t d = 0; d < smem::kDownRingSlots; ++d) {
    mbar_init(base + smem::kDownFullBar + d * 8, 1);
    mbar_init(base + smem::kDownEmptyBar + d * 8, kConsumerWarps);
  }
  mbar_fence_init();
  sts_u32(base + smem::kProgOwn, 0u);
  sts_u32(base + smem::kCapFlag, 0u);
  sts_u32(base + smem::kPfDone, 0u);
  const uint8_t* const src = reinterpret_cast<const uint8_t*>(args.context.direct.r1_router_w) +
                             size_t(cta & (kCtasPerToken - 1)) * kRouterCtaStride;
#pragma unroll
  for (uint32_t i = 0; i < kRouterPages; ++i) {
    const uint32_t full = base + smem::kFullBar + i * 8;
    mbar_expect_tx(full, kRouterPageBytes);
    bulk_copy_g2s(base + smem::kRing + i * smem::kRingSlotBytes, src + size_t(i) * kRouterPageBytes,
                  kRouterPageBytes, full, false);
    mbar_arrive(full);
  }
}

__device__ __forceinline__ OpCtx make_op_ctx(const MoeArgs& args, int cta, uint8_t* smem) {
  OpCtx ctx;
  ctx.cta = cta;
  ctx.tid = static_cast<int>(threadIdx.x);
  ctx.warp = ctx.tid >> 5;
  ctx.lane = ctx.tid & 31;
  ctx.routed = cta < kRoutedCtas;
  ctx.t = cta >> 5;
  ctx.c = cta & 31;
  ctx.k = ctx.c >> 4;
  ctx.quarter = ctx.c & 15;
  ctx.rgS = cta - kSharedBase;
  ctx.rank = args.rank;
  ctx.epoch = static_cast<uint32_t>(ctx4(smem).epoch);
  ctx.smem = smem;
  ctx.smem_s = smem_u32(smem);
  ctx.item = 0;
  ctx.args = &args;
  return ctx;
}
__device__ __forceinline__ PrefetchCtx make_prefetch_ctx(const MoeArgs& args, int cta,
                                                         uint8_t* smem) {
  PrefetchCtx p;
  p.cta = cta;
  p.ptid = static_cast<int>(threadIdx.x) - kConsumerThreads;
  p.pwarp = p.ptid >> 5;
  p.plane = p.ptid & 31;
  p.routed = cta < kRoutedCtas;
  p.t = cta >> 5;
  p.c = cta & 31;
  p.k = p.c >> 4;
  p.quarter = p.c & 15;
  p.rgS = cta - kSharedBase;
  p.item = 0;
  p.smem = smem;
  p.smem_s = smem_u32(smem);
  p.sel_idx_src = reinterpret_cast<const int32_t*>(smem + smem::kRSelIdx);
  p.args = &args;
  return p;
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

struct PageDesc {
  const uint8_t* src[4];
  uint32_t bytes[4];
  uint32_t dst_off[4];
  uint32_t n;
  uint32_t total;
};

__device__ __forceinline__ PageDesc page_single(const void* src, uint32_t bytes) {
  PageDesc d;
  d.src[0] = static_cast<const uint8_t*>(src);
  d.bytes[0] = bytes;
  d.dst_off[0] = 0;
  d.n = 1;
  d.total = bytes;
  return d;
}

__device__ __forceinline__ PageDesc page_gamma(const uint8_t* smem) {
  return page_single(ctx0(smem).gamma, kGammaPageBytes);
}

__device__ __forceinline__ PageDesc page_gate_up_routed(const uint8_t* smem, int E, int rg, int p) {
  return page_single(ctx2(smem).gate_up_w + size_t(E + 1) * kGateUpExpertStride +
                         size_t(rg) * kGateUpRowGroupStride + size_t(p) * kGateUpPageStride,
                     kGateUpPageCopyBytes);
}
__device__ __forceinline__ PageDesc page_gate_up_shared(const uint8_t* smem, int rgS, int p) {
  return page_single(ctx3(smem).gate_up_w_shared + size_t(rgS) * kGateUpRowGroupStride +
                         size_t(p) * kGateUpPageStride,
                     kGateUpPageCopyBytes);
}

__device__ __forceinline__ PageDesc page_down_half(const uint8_t* smem, const int32_t* sel, int c,
                                                   int half, int e) {
  PageDesc d;
  const int Ew = (e == 0) ? 0 : sel[e - 1] + 1;
  const int s0 = c * kDownBlocksPerCta + half * kDownHalfBlocks;
  d.src[0] = ctx4(smem).down_w + size_t(Ew) * kDownExpertStride + size_t(s0) * kDownSliceBytes;
  d.bytes[0] = kDownHalfMatBytes;
  d.dst_off[0] = 0;
  d.src[1] = reinterpret_cast<const uint8_t*>(ctx4(smem).down_scale) +
             size_t(Ew) * kDownScaleExpertStride + size_t(s0) * kDownScaleSliceBytes;
  d.bytes[1] = kDownHalfScaleBytes;
  d.dst_off[1] = kDownHalfMatBytes;
  d.n = 2;
  d.total = kDownHalfStageBytes;
  return d;
}

__device__ __forceinline__ void l2_prefetch_page(const PageDesc& d) {
#pragma unroll
  for (uint32_t i = 0; i < d.n; ++i) l2_prefetch_bulk(d.src[i], d.bytes[i]);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void pf_issue_page(PrefetchCtx& p, const PageDesc& d,
                                              bool force_retain_l2 = false, bool one_use = false) {
  const bool retain_l2 =
      !one_use && (force_retain_l2 || (d.n == 1 && d.bytes[0] == kGammaPageBytes));
  const uint32_t slot = p.item & (smem::kRingSlots - 1);
  const uint32_t full = p.s(smem::kFullBar) + slot * 8;
  const uint32_t dst = p.s(smem::kRing) + slot * smem::kRingSlotBytes;

  mbar_wait_parity(p.s(smem::kEmptyBar) + slot * 8, ((p.item >> smem::kRingLog2) & 1u) ^ 1u);
  if (p.plane == 0) {
    if (p.pwarp == 0) mbar_expect_tx(full, d.total);
    if (p.pwarp == 0) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        if (i < d.n) bulk_copy_g2s(dst + d.dst_off[i], d.src[i], d.bytes[i], full, retain_l2);
    }
    mbar_arrive(full);
  }
  ++p.item;
}

__device__ __forceinline__ void pf_stream_gamma(PrefetchCtx& p) {
  pf_issue_page(p, page_gamma(p.smem));
}

__device__ __forceinline__ void pf_wait_sel_mine(PrefetchCtx& p) {
  mbar_wait_parity(p.s(smem::kSelMineBar), 0u);
}

__device__ __forceinline__ void pf_wait_sel_all(PrefetchCtx& p) {
  mbar_wait_parity(p.s(smem::kSelAllBar), 0u);
}

__device__ __forceinline__ void sts_vol_u32(uint32_t address, uint32_t value) {
  asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(address), "r"(value) : "memory");
}
__device__ __forceinline__ uint32_t lds_vol_u32(uint32_t address) {
  uint32_t v;
  asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(address) : "memory");
  return v;
}
__device__ __forceinline__ bool pf_ahead(const PrefetchCtx& p, uint32_t consumed_bytes) {
  sts_vol_u32(p.s(smem::kProgOwn), consumed_bytes >> 8);
  return lds_vol_u32(p.s(smem::kCapFlag)) != 0u;
}
__device__ __forceinline__ void pf_stream_gate_up_routed(PrefetchCtx& p,
                                                         bool first_page_issued = false) {
#pragma unroll 1
  for (int q = 0; q < kRowGroupsPerCta * kGateUpPages; ++q) {
    const int i = q / kGateUpPages;
    const int page = q - i * kGateUpPages;
    if (q == kGateUpPages) pf_wait_sel_all(p);
    if (first_page_issued && q == 0) {
      ++p.item;
      continue;
    }
    const int E = p.sel_idx_src[2 * i + p.k];

    if (300 > 0 && q > (first_page_issued ? 1 : 0) && q < static_cast<int>(smem::kRingSlots))
      __nanosleep(300);
    if (first_page_issued && q >= static_cast<int>(smem::kRingSlots)) {
      const uint32_t m = p.item;
      mbar_wait_parity(p.s(smem::kEmptyBar) + (m & (smem::kRingSlots - 1)) * 8,
                       ((m >> smem::kRingLog2) & 1u) ^ 1u);
      if (pf_ahead(p, static_cast<uint32_t>(q - 3) * kGateUpPageStride)) {
        const uint32_t n = m - 3;
        mbar_wait_parity(p.s(smem::kEmptyBar) + (n & (smem::kRingSlots - 1)) * 8,
                         (n >> smem::kRingLog2) & 1u);
      }
    }
    pf_issue_page(p, page_gate_up_routed(p.smem, E, p.quarter, page));
  }
}

constexpr int kRoutedGateUpItemsAC = kRowGroupsPerCta * kGateUpPages;
__device__ __forceinline__ void pf_stream_down_routed5(PrefetchCtx& p) {
  const int32_t* sel = p.sel_idx_src;
  const uint32_t ring_end = p.item;
#pragma unroll 1
  for (int q = 0; q < kRoutedPersistentDownStages; ++q) {
    const uint32_t d = static_cast<uint32_t>(q) % smem::kDownRingSlots;
    const uint32_t u = static_cast<uint32_t>(q) / smem::kDownRingSlots;
    if (u == 0) {
      const uint32_t s_lo = (d * kDownHalfStageBytes) / smem::kRingSlotBytes;
      const uint32_t s_hi = ((d + 1) * kDownHalfStageBytes - 1) / smem::kRingSlotBytes;
      for (uint32_t s = s_lo; s <= s_hi; ++s) {
        const uint32_t n = ring_end - smem::kRingSlots +
                           ((s - (ring_end - smem::kRingSlots)) & (smem::kRingSlots - 1));
        mbar_wait_parity(p.s(smem::kEmptyBar) + s * 8, (n >> smem::kRingLog2) & 1u);
      }
    } else {
      mbar_wait_parity(p.s(smem::kDownEmptyBar) + d * 8, (u - 1) & 1u);
      if (pf_ahead(p, static_cast<uint32_t>(kRoutedGateUpItemsAC) * kGateUpPageStride +
                          static_cast<uint32_t>(q - 4) * kDownHalfStageBytes)) {
        const uint32_t qn = static_cast<uint32_t>(q) - 4;
        mbar_wait_parity(p.s(smem::kDownEmptyBar) + (qn % smem::kDownRingSlots) * 8,
                         (qn / smem::kDownRingSlots) & 1u);
      }
    }
    const PageDesc pd = page_down_half(p.smem, sel, p.c, q & 1, 1 + (q >> 1));
    const uint32_t full = p.s(smem::kDownFullBar) + d * 8;
    const uint32_t dst = p.s(smem::kRing) + d * kDownHalfStageBytes;
    mbar_expect_tx(full, pd.total);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i)
      bulk_copy_g2s(dst + pd.dst_off[i], pd.src[i], pd.bytes[i], full, false);
    mbar_arrive(full);
  }
  sts_vol_u32(p.s(smem::kPfDone), 1u);
}

constexpr int kRoutedGateUpItems = kRowGroupsPerCta * kGateUpPages;
constexpr int kRoutedPostItems = kRoutedGateUpItems + kRoutedPersistentDownStages;
constexpr int kSharedWeightItems = kGateUpPages + kSharedPersistentDownStages;

__device__ __forceinline__ PageDesc shared_weight_page(const PrefetchCtx& p, int j) {
  if (j < kGateUpPages) return page_gate_up_shared(p.smem, p.rgS, j);
  const int32_t zero_sel[kTopk] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int stage = j - kGateUpPages;
  return page_down_half(p.smem, zero_sel, 2 * p.rgS + (stage >> 1), stage & 1, 0);
}

__device__ __forceinline__ void pf_main_routed(PrefetchCtx& p) {
  if (p.ptid != 0) return;

  p.item += kRouterPages;
  pf_wait_sel_mine(p);
  pf_stream_gate_up_routed(p, true);
  pf_wait_sel_all(p);
  pf_stream_down_routed5(p);
}

__device__ __forceinline__ void pf_main_shared(PrefetchCtx& p) {
  if (p.ptid != 0) return;
  const bool skip_gamma = p.rgS >= kSharedNormCtas;
  if (!skip_gamma) pf_stream_gamma(p);

  const int lead = static_cast<int>(smem::kRingSlots) - (skip_gamma ? 0 : 1);
  for (int j = 0; j < kSharedWeightItems; ++j) {
    pf_issue_page(p, shared_weight_page(p, j));
    if (j + 1 == lead) {
      for (int q = lead; q < kSharedWeightItems; ++q) l2_prefetch_page(shared_weight_page(p, q));
    }
  }
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void load_x_row(OpCtx& ctx, const uint16_t* x_row_global,
                                           uint32_t dst_window) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(x_row_global) + ctx.tid * 16;
  const uint32_t dst = ctx.s(dst_window) + ctx.tid * 16;
#pragma unroll
  for (int i = 0; i < 3; ++i) ldgsts_16(dst + i * 0x1000, src + i * 0x1000);
  ldgsts_commit();
  ldgsts_wait_all();
  ctx.sync();
}

__device__ __forceinline__ float rmsnorm_rstd(OpCtx& ctx, uint32_t x_s) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const uint32_t w = lds_u32(x_s + (ctx.tid + kConsumerThreads * j) * 4);
    const float hi = bf16_hi(w);
    const float lo = bf16_lo(w);
    float p = hi * hi;
    p = fmaf(lo, lo, p);
    acc = acc + p;
  }
  const float wsum = warp_sum_down(acc);
  if (ctx.lane == 0) sts_f32(ctx.s(smem::kRedScratch) + ctx.warp * 4, wsum);
  ctx.sync();
  float p8[8];
#pragma unroll
  for (int w = 0; w < 8; ++w) p8[w] = lds_f32(ctx.s(smem::kRedScratch) + w * 4);
  const float sumsq = fold8(p8);
  const float mean = __fdiv_rn(sumsq, kRmsDenominator);
  const float rstd = rsqrtf(mean + kRmsEps);
  ctx.sync();
  return rstd;
}

template <bool ToGlobal>
__device__ __forceinline__ void rmsnorm_apply(OpCtx& ctx, uint32_t x_s, uint32_t gamma_slot,
                                              float rstd, uint32_t* out_global) {
  float g[kWordsPerThread][2];
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const uint32_t word = ctx.tid + kConsumerThreads * j;
    g[j][0] = lds_f32(gamma_slot + (2 * word) * 4);
    g[j][1] = lds_f32(gamma_slot + (2 * word + 1) * 4);
  }
  ctx.release_page();
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const uint32_t word = ctx.tid + kConsumerThreads * j;
    const uint32_t xw = lds_u32(x_s + word * 4);
    float y0, y1;
    {
      y0 = bf16_lo(xw) * (rstd * g[j][0]);
      y1 = bf16_hi(xw) * (rstd * g[j][1]);
    }
    const uint32_t packed = pack_bf16_pair(y0, y1);
    if (ToGlobal)
      out_global[word] = packed;
    else
      sts_u32(x_s + word * 4, packed);
  }
  if (!ToGlobal) ctx.sync();
}

__device__ __forceinline__ void fma_bf16_pair(float& acc_lo, float& acc_hi, uint32_t a,
                                              uint32_t b) {
  asm("{\n\t.reg .b16 al, ah, bl, bh;\n\tmov.b32 {al, ah}, %2;\n\tmov.b32 {bl, bh}, %3;\n\t"
      "fma.rn.f32.bf16 %0, al, bl, %0;\n\tfma.rn.f32.bf16 %1, ah, bh, %1;\n\t}"
      : "+f"(acc_lo), "+f"(acc_hi)
      : "r"(a), "r"(b));
}

__device__ __forceinline__ void mul_f32x2(float& d_lo, float& d_hi, float a_lo, float a_hi,
                                          float b_lo, float b_hi) {
  asm("{\n\t.reg .b64 pa, pb, pd;\n\tmov.b64 pa, {%2, %3};\n\tmov.b64 pb, {%4, %5};\n\t"
      "mul.rn.f32x2 pd, pa, pb;\n\tmov.b64 {%0, %1}, pd;\n\t}"
      : "=f"(d_lo), "=f"(d_hi)
      : "f"(a_lo), "f"(a_hi), "f"(b_lo), "f"(b_hi));
}

__device__ __forceinline__ float sumsq_bf16_pair(uint32_t w) {
  float p;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tfma.rn.f32.bf16 %0, h, h, 0f00000000;\n\t"
      "fma.rn.f32.bf16 %0, l, l, %0;\n\t}"
      : "=f"(p)
      : "r"(w));
  return p;
}
struct RoutedGamma {
  float2 g[kWordsPerThread];
  uint32_t x[kWordsPerThread];
  uint32_t epoch;
};

__device__ __forceinline__ uint32_t routed_chunk(uint32_t tid, int jc) {
  return tid + kConsumerThreads * uint32_t(jc);
}
__device__ __forceinline__ void ld_cg_u32x4(const void* address, uint32_t& a, uint32_t& b,
                                            uint32_t& c, uint32_t& d) {
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "l"(address)
               : "memory");
}
__device__ __forceinline__ void routed_load_gamma(const MoeArgs& args, RoutedGamma& rg);
__device__ __forceinline__ void routed_issue_entry_loads(const MoeArgs& args, uint8_t* smem_base,
                                                         int cta, RoutedGamma& rg) {
  const uint32_t base = smem_u32(smem_base);
  const uint32_t tid = threadIdx.x;
  const uint8_t* x =
      reinterpret_cast<const uint8_t*>(args.context.direct.r0_x + size_t(cta >> 5) * kDim);

  if (tid < 64)
    ldgsts_16(base + smem::kRBias + tid * 16,
              reinterpret_cast<const uint8_t*>(args.context.direct.r2_bias) + tid * 16);
  ldgsts_commit();
  routed_load_gamma(args, rg);
  grid_dependency_wait();

  rg.epoch = ld_relaxed_gpu_u32(args.launch_ctr + cta);
#pragma unroll
  for (int jc = 0; jc < kWordsPerThread / 4; ++jc)
    ld_cg_u32x4(x + size_t(routed_chunk(tid, jc)) * 16, rg.x[4 * jc], rg.x[4 * jc + 1],
                rg.x[4 * jc + 2], rg.x[4 * jc + 3]);
}
__device__ __forceinline__ void routed_load_gamma(const MoeArgs& args, RoutedGamma& rg) {
  const float4* gamma4 = reinterpret_cast<const float4*>(args.context.direct.r0_gamma);
#pragma unroll
  for (int jc = 0; jc < kWordsPerThread / 4; ++jc) {
    const float4 a = __ldg(gamma4 + 2 * routed_chunk(threadIdx.x, jc));
    const float4 b = __ldg(gamma4 + 2 * routed_chunk(threadIdx.x, jc) + 1);
    rg.g[4 * jc + 0] = make_float2(a.x, a.y);
    rg.g[4 * jc + 1] = make_float2(a.z, a.w);
    rg.g[4 * jc + 2] = make_float2(b.x, b.y);
    rg.g[4 * jc + 3] = make_float2(b.z, b.w);
  }
}

__device__ __forceinline__ void routed_norm_gamma_regs(OpCtx& ctx, const RoutedGamma& rg,
                                                       uint32_t (&xnw)[kWordsPerThread]) {
  const float2* g = rg.g;
  const uint32_t x_s = ctx.s(smem::kRxRow);
  uint32_t xw[kWordsPerThread];
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    xw[j] = rg.x[j];
    acc = acc + sumsq_bf16_pair(xw[j]);
  }
  const float wsum = warp_sum_down(acc);
  if (ctx.lane == 0) sts_f32(ctx.s(smem::kRedScratch) + ctx.warp * 4, wsum);
  ldgsts_wait_all();
  ctx.sync();
  float p8[8];
  static_assert(smem::kRedScratch % 16 == 0, "16 B aligned eight warp partials");
  {
    uint32_t a, b, c, d, e, f, g2, h;
    lds_u32x4(ctx.s(smem::kRedScratch), a, b, c, d);
    lds_u32x4(ctx.s(smem::kRedScratch) + 16, e, f, g2, h);
    p8[0] = __uint_as_float(a);
    p8[1] = __uint_as_float(b);
    p8[2] = __uint_as_float(c);
    p8[3] = __uint_as_float(d);
    p8[4] = __uint_as_float(e);
    p8[5] = __uint_as_float(f);
    p8[6] = __uint_as_float(g2);
    p8[7] = __uint_as_float(h);
  }
  const float mean = __fdiv_rn(fold8(p8), kRmsDenominator);
  const float rstd = rsqrtf(mean + kRmsEps);

  (void)x_s;
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    float m0, m1, y0, y1;
    mul_f32x2(m0, m1, rstd, rstd, g[j].x, g[j].y);
    mul_f32x2(y0, y1, bf16_lo(xw[j]), bf16_hi(xw[j]), m0, m1);
    xnw[j] = pack_bf16_pair(y0, y1);
  }
}
__device__ __forceinline__ void routed_store_xnorm(OpCtx& ctx,
                                                   const uint32_t (&xnw)[kWordsPerThread]) {
  const uint32_t x_s = ctx.s(smem::kRxRow);
#pragma unroll
  for (int jc = 0; jc < kWordsPerThread / 4; ++jc)
    sts_u32x4(x_s + routed_chunk(ctx.tid, jc) * 16, xnw[4 * jc], xnw[4 * jc + 1], xnw[4 * jc + 2],
              xnw[4 * jc + 3]);
}

__device__ __forceinline__ void routed_norm_from_page(OpCtx& ctx, bool x_preloaded = false) {
  if (x_preloaded) {
    ldgsts_wait_all();
    ctx.sync();
  } else {
    load_x_row(ctx, ctx0(ctx.smem).x + size_t(ctx.t) * kDim, smem::kRxRow);
  }
  const float rstd = rmsnorm_rstd(ctx, ctx.s(smem::kRxRow));
  const uint32_t gamma_slot = ctx.acquire_page();
  rmsnorm_apply<false>(ctx, ctx.s(smem::kRxRow), gamma_slot, rstd, nullptr);
}

__device__ __forceinline__ void op_rmsnorm_router_routed(const MoeArgs& args, OpCtx& ctx,
                                                         bool wait_for_helpers,
                                                         const RoutedGamma* entry_gamma = nullptr) {
  float* scores_out = ctx1(ctx.smem).scores_out;
  uint32_t xnw[kWordsPerThread];
  if (wait_for_helpers)
    routed_norm_gamma_regs(ctx, *entry_gamma, xnw);
  else
    routed_norm_from_page(ctx);
  const uint32_t x_s = ctx.s(smem::kRxRow);

  const uint32_t red_s =
      wait_for_helpers ? ctx.s(smem::kRouterRedScratch) : ctx.s(smem::kRedScratch);
  if (wait_for_helpers) {
    float rsum[kExpertsPerRouterCta];
#pragma unroll
    for (int i = 0; i < kRouterPages; ++i) {
      const uint32_t slot = ctx.acquire_page();
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint32_t row = slot + r * kRouterRowBytes;
        float even = 0.0f, odd = 0.0f;
#pragma unroll
        for (int jc = 0; jc < kWordsPerThread / 4; ++jc) {
          uint32_t w0, w1, w2, w3;
          lds_u32x4(row + routed_chunk(ctx.tid, jc) * 16, w0, w1, w2, w3);
          fma_bf16_pair(even, odd, w0, xnw[4 * jc + 0]);
          fma_bf16_pair(even, odd, w1, xnw[4 * jc + 1]);
          fma_bf16_pair(even, odd, w2, xnw[4 * jc + 2]);
          fma_bf16_pair(even, odd, w3, xnw[4 * jc + 3]);
        }
        rsum[2 * i + r] = even + odd;
      }
      ctx.release_page();
    }

    {
      const bool b16 = (ctx.lane & 16) != 0, b8 = (ctx.lane & 8) != 0, b4 = (ctx.lane & 4) != 0;
      float k4[4], k2[2];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float send = b16 ? rsum[m] : rsum[4 + m];
        const float keep = b16 ? rsum[4 + m] : rsum[m];
        k4[m] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const float send = b8 ? k4[m] : k4[2 + m];
        const float keep = b8 ? k4[2 + m] : k4[m];
        k2[m] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
      }
      const float send = b4 ? k2[0] : k2[1];
      const float keep = b4 ? k2[1] : k2[0];
      float t = keep + __shfl_xor_sync(0xffffffffu, send, 4);
      t = t + __shfl_xor_sync(0xffffffffu, t, 2);
      t = t + __shfl_xor_sync(0xffffffffu, t, 1);
      if ((ctx.lane & 3) == 0) sts_f32(red_s + ((ctx.lane >> 2) * 8 + ctx.warp) * 4, t);
    }
  }
#pragma unroll
  for (int i = 0; i < (wait_for_helpers ? 0 : kRouterPages); ++i) {
    const uint32_t slot = ctx.acquire_page();
    float sum[2] = {0.0f, 0.0f};
    {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const uint32_t row = slot + r * kRouterRowBytes;
        float even = 0.0f, odd = 0.0f;
#pragma unroll
        for (int j = 0; j < kWordsPerThread; ++j) {
          const uint32_t word = ctx.tid + kConsumerThreads * j;
          const uint32_t w = lds_u32(row + word * 4);
          const uint32_t xw = lds_u32(x_s + word * 4);
          even = fmaf(bf16_lo(w), bf16_lo(xw), even);
          odd = fmaf(bf16_hi(w), bf16_hi(xw), odd);
        }
        sum[r] = even + odd;
      }
    }
    ctx.release_page();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float wsum = warp_sum_down(sum[r]);
      const int e_local = 2 * i + r;
      if (ctx.lane == 0) sts_f32(red_s + (e_local * 8 + ctx.warp) * 4, wsum);
    }
  }

  ctx.sync();

  if (wait_for_helpers) wait_for_sentinel_helpers(args, ctx);

  if (ctx.warp == 0) {
    float logit = 0.0f;
    if (ctx.lane < kExpertsPerRouterCta) {
      const int e_local = ctx.lane;
      float p8[8];
      static_assert(smem::kRouterRedScratch % 16 == 0 && smem::kRedScratch % 16 == 0,
                    "16 B aligned warp sums of a logit");
      {
        uint32_t a, b, c, d, e, f, g2, h;
        lds_u32x4(red_s + e_local * 32, a, b, c, d);
        lds_u32x4(red_s + e_local * 32 + 16, e, f, g2, h);
        p8[0] = __uint_as_float(a);
        p8[1] = __uint_as_float(b);
        p8[2] = __uint_as_float(c);
        p8[3] = __uint_as_float(d);
        p8[4] = __uint_as_float(e);
        p8[5] = __uint_as_float(f);
        p8[6] = __uint_as_float(g2);
        p8[7] = __uint_as_float(h);
      }
      logit = fold8(p8);
    }
    const float l1 = __shfl_down_sync(0xffffffffu, logit, 1);
    const float l2 = __shfl_down_sync(0xffffffffu, logit, 2);
    const float l3 = __shfl_down_sync(0xffffffffu, logit, 3);
    if (ctx.lane == 0 || ctx.lane == 4)
      publish_score_quad(
          reinterpret_cast<uint8_t*>(scores_out) + ctx.cta * kScoresSlotBytes + ctx.lane * 4, logit,
          l1, l2, l3);
  }

  if (wait_for_helpers) routed_store_xnorm(ctx, xnw);
}

__device__ __forceinline__ void op_rmsnorm_router_shared(const MoeArgs& args, OpCtx& ctx) {
  if (ctx.rgS >= kSharedNormCtas) return;
  const int tok = ctx.rgS;
  load_x_row(ctx, ctx3(ctx.smem).x_shared + size_t(tok) * kDim, smem::kSxRow);
  const float rstd = rmsnorm_rstd(ctx, ctx.s(smem::kSxRow));
  const uint32_t gamma_slot = ctx.acquire_page();
  uint32_t* out = reinterpret_cast<uint32_t*>(ctx3(ctx.smem).x_norm_buf + size_t(tok) * kDim);
  rmsnorm_apply<true>(ctx, ctx.s(smem::kSxRow), gamma_slot, rstd, out);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void rc3_gather_scores(const MoeArgs& args, OpCtx& ctx,
                                                  bool bias_preloaded) {
  (void)args;
  const CtxRow2& r2 = ctx2(ctx.smem);
  if (ctx.tid < 128) {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(r2.scores_in) +
        (size_t(ctx.t) * 0x400 + size_t(ctx.tid >> 2) * 32 + size_t(ctx.tid & 3) * 2) * 4;
    uint32_t lo, hi;
    for (;;) {
      ld_relaxed_gpu_v2u32(src, lo, hi);
      if (lo != kScorePending && hi != kScorePending) break;
      spin_backoff();
    }
    if (bias_preloaded) {
      const float s0 = sigmoid_approx(__uint_as_float(lo)),
                  s1 = sigmoid_approx(__uint_as_float(hi));
      uint32_t b0, b1;
      lds_u32x2(ctx.s(smem::kRBias) + ctx.tid * 8, b0, b1);
      sts_u32x2(ctx.s(smem::kRBias) + ctx.tid * 8, f32_monotonic_key(s0 + __uint_as_float(b0)),
                f32_monotonic_key(s1 + __uint_as_float(b1)));
      sts_u32x2(ctx.s(smem::kRScores) + ctx.tid * 8, __float_as_uint(s0), __float_as_uint(s1));
    } else {
      sts_u32x2(ctx.s(smem::kRScores) + ctx.tid * 8, lo, hi);
    }
  }
  if (!bias_preloaded && ctx.tid < 64) {
    ldgsts_16(ctx.s(smem::kRBias) + ctx.tid * 16,
              reinterpret_cast<const uint8_t*>(r2.bias) + ctx.tid * 16);
    ldgsts_commit();
    ldgsts_wait_all();
  }
  ctx.sync();
}

__device__ __forceinline__ uint32_t umax_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }
__device__ __forceinline__ void rc5_top8_warp0(OpCtx& ctx, bool keys_ready) {
  const uint32_t scores_s = ctx.s(smem::kRScores);
  const uint32_t bias_s = ctx.s(smem::kRBias);
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int e = ctx.lane + 32 * j;
    if (keys_ready) {
      key[j] = lds_u32(bias_s + e * 4);
    } else {
      const float v = sigmoid_approx(lds_f32(scores_s + e * 4)) + lds_f32(bias_s + e * 4);
      key[j] = f32_monotonic_key(v);
    }
  }
  int my_idx = 0;

#pragma unroll 1
  for (int k = 0; k < kTopk; ++k) {
    uint64_t best = (uint64_t(key[0]) << 32) | uint32_t(255 - ctx.lane);
#pragma unroll
    for (int j = 1; j < 8; ++j) {
      const uint64_t c = (uint64_t(key[j]) << 32) | uint32_t(255 - (ctx.lane + 32 * j));
      best = c > best ? c : best;
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
      const uint64_t o = __shfl_xor_sync(0xffffffffu, best, mask);
      best = o > best ? o : best;
    }
    const int idx = 255 - static_cast<int>(static_cast<uint32_t>(best));
    const uint32_t zero_bits = (ctx.lane == (idx & 31)) ? (1u << (idx >> 5)) : 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) key[j] = ((zero_bits >> j) & 1u) ? 0u : key[j];
    if (ctx.lane == k) my_idx = idx;
    if (ctx.lane == 0) sts_u32(ctx.s(smem::kRSelIdx) + k * 4, static_cast<uint32_t>(idx));
    if (k == ctx.k) {
      __syncwarp(0xffffffffu);
      if (ctx.lane == 31) mbar_arrive(ctx.s(smem::kSelMineBar));
      if (keys_ready && ctx.lane == 31) {
        mbar_wait_parity(ctx.s(smem::kEmptyBar), 0u);
        const uint32_t full = ctx.s(smem::kFullBar);
        mbar_expect_tx(full, kGateUpPageCopyBytes);
        const uint8_t* const src = ctx2(ctx.smem).gate_up_w +
                                   size_t(idx + 1) * kGateUpExpertStride +
                                   size_t(ctx.quarter) * kGateUpRowGroupStride;
        bulk_copy_g2s(ctx.s(smem::kRing), src, kGateUpPageCopyBytes, full, false);
        mbar_arrive(full);
      }
    }
  }
  const float mine = ctx.lane < kTopk
                         ? (keys_ready ? lds_f32(scores_s + my_idx * 4)
                                       : sigmoid_approx(lds_f32(scores_s + my_idx * 4)))
                         : 0.0f;
  float v = mine;
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  const float sum = __shfl_sync(0xffffffffu, v, 0);
  const float scale = __fdiv_rn(kRouteScale, sum);
  if (ctx.lane < kTopk) sts_f32(ctx.s(smem::kRSelProb) + ctx.lane * 4, mine * scale);
}

constexpr uint32_t kRSelTop2 = smem::kRDownPartial + 0x800;
__device__ __forceinline__ void top2_merge(uint64_t& a1, uint64_t& a2, uint64_t b1, uint64_t b2) {
  const uint64_t hi = a1 > b1 ? a1 : b1;
  const uint64_t lo = a1 > b1 ? b1 : a1;
  const uint64_t s2 = a2 > b2 ? a2 : b2;
  a1 = hi;
  a2 = lo > s2 ? lo : s2;
}
__device__ __forceinline__ void rc3_gather_scores_early_m1(OpCtx& ctx) {
  const CtxRow2& r2 = ctx2(ctx.smem);
  if (ctx.tid < 128) {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(r2.scores_in) +
        (size_t(ctx.t) * 0x400 + size_t(ctx.tid >> 2) * 32 + size_t(ctx.tid & 3) * 2) * 4;
    uint32_t lo, hi;
    for (;;) {
      ld_relaxed_gpu_v2u32(src, lo, hi);
      if (lo != kScorePending && hi != kScorePending) break;
      spin_backoff();
    }
    const float s0 = sigmoid_approx(__uint_as_float(lo)), s1 = sigmoid_approx(__uint_as_float(hi));
    uint32_t b0, b1;
    lds_u32x2(ctx.s(smem::kRBias) + ctx.tid * 8, b0, b1);
    uint32_t k0 = f32_monotonic_key(s0 + __uint_as_float(b0));
    uint32_t k1 = f32_monotonic_key(s1 + __uint_as_float(b1));
    sts_u32x2(ctx.s(smem::kRBias) + ctx.tid * 8, k0, k1);
    sts_u32x2(ctx.s(smem::kRScores) + ctx.tid * 8, __float_as_uint(s0), __float_as_uint(s1));
    __syncwarp(0xffffffffu);
    const uint32_t e0 = 2u * static_cast<uint32_t>(ctx.tid);
    const uint32_t m1 = __reduce_max_sync(0xffffffffu, umax_u32(k0, k1));
    const uint32_t i1 = __reduce_min_sync(0xffffffffu, k0 == m1 ? e0 : (k1 == m1 ? e0 + 1u : 256u));
    k0 = (i1 == e0) ? 0u : k0;
    k1 = (i1 == e0 + 1u) ? 0u : k1;
    const uint32_t m2 = __reduce_max_sync(0xffffffffu, umax_u32(k0, k1));
    const uint32_t i2 = __reduce_min_sync(0xffffffffu, k0 == m2 ? e0 : (k1 == m2 ? e0 + 1u : 256u));
    if (ctx.lane == 0) sts_u32x4(ctx.s(kRSelTop2) + ctx.warp * 16, m1, i1, m2, i2);
  }
  ctx.sync();
  if (ctx.warp == 0) {
    uint64_t g1 = 0, g2 = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t a, b, c, d;
      lds_u32x4(ctx.s(kRSelTop2) + w * 16, a, b, c, d);
      const uint64_t p1 = (uint64_t(a) << 32) | (255u - b), p2 = (uint64_t(c) << 32) | (255u - d);
      if (w == 0) {
        g1 = p1;
        g2 = p2;
      } else
        top2_merge(g1, g2, p1, p2);
    }
    const uint32_t id0 = 255u - static_cast<uint32_t>(g1), id1 = 255u - static_cast<uint32_t>(g2);
    if (ctx.lane == 0) sts_u32x2(ctx.s(smem::kRSelIdx), id0, id1);
    __syncwarp(0xffffffffu);
    if (ctx.lane == 31) {
      const uint32_t e_unit0 = ctx.k == 0 ? id0 : id1;
      mbar_wait_parity(ctx.s(smem::kEmptyBar), 0u);
      const uint32_t full = ctx.s(smem::kFullBar);
      mbar_expect_tx(full, kGateUpPageCopyBytes);
      const uint8_t* const src = ctx2(ctx.smem).gate_up_w +
                                 size_t(e_unit0 + 1) * kGateUpExpertStride +
                                 size_t(ctx.quarter) * kGateUpRowGroupStride;
      bulk_copy_g2s(ctx.s(smem::kRing), src, kGateUpPageCopyBytes, full, false);
      mbar_arrive(full);
      mbar_arrive(ctx.s(smem::kSelMineBar));
    }
  }
}

__device__ __forceinline__ void rc5_top8_warp0_p(OpCtx& ctx) {
  const uint32_t scores_s = ctx.s(smem::kRScores);
  uint32_t key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) key[j] = lds_u32(ctx.s(smem::kRBias) + (ctx.lane + 32 * j) * 4);
  int my_idx = 0;
#pragma unroll 1
  for (int k = 0; k < kTopk; ++k) {
    const uint32_t lmax = umax_u32(umax_u32(umax_u32(key[0], key[1]), umax_u32(key[2], key[3])),
                                   umax_u32(umax_u32(key[4], key[5]), umax_u32(key[6], key[7])));
    const uint32_t kmax = __reduce_max_sync(0xffffffffu, lmax);
    uint32_t cand = 256u;
#pragma unroll
    for (int j = 7; j >= 0; --j)
      cand = (key[j] == kmax) ? static_cast<uint32_t>(ctx.lane + 32 * j) : cand;
    const int idx = static_cast<int>(__reduce_min_sync(0xffffffffu, cand));
    const uint32_t zero_bits = (ctx.lane == (idx & 31)) ? (1u << (idx >> 5)) : 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) key[j] = ((zero_bits >> j) & 1u) ? 0u : key[j];
    if (ctx.lane == k) my_idx = idx;
    if (ctx.lane == 0 && k >= 2) sts_u32(ctx.s(smem::kRSelIdx) + k * 4, static_cast<uint32_t>(idx));
  }
  const float mine = ctx.lane < kTopk ? lds_f32(scores_s + my_idx * 4) : 0.0f;
  float v = mine;
  v += __shfl_down_sync(0xffffffffu, v, 4);
  v += __shfl_down_sync(0xffffffffu, v, 2);
  v += __shfl_down_sync(0xffffffffu, v, 1);
  const float sum = __shfl_sync(0xffffffffu, v, 0);
  const float scale = __fdiv_rn(kRouteScale, sum);
  if (ctx.lane < kTopk) sts_f32(ctx.s(smem::kRSelProb) + ctx.lane * 4, mine * scale);
}

struct MidPrefetch {
  uint32_t a0, a1, b0, b1;
  bool issued;
  bool resolved;
};
__device__ __forceinline__ bool quantize_routed_mid_slot(OpCtx& ctx, uint32_t a0, uint32_t a1,
                                                         uint32_t b0, uint32_t b1) {
  const bool ok = !(mid_word_pending(a0) || mid_word_pending(a1) || mid_word_pending(b0) ||
                    mid_word_pending(b1));
  if (!__all_sync(0xffffffffu, ok)) return false;
  const int qe = ctx.warp + 1;
  const GuQuant128 q0 = gu_quant128_regs(a0, a1);
  const GuQuant128 q1 = gu_quant128_regs(b0, b1);
  uint32_t c0, c1, c2, c3;
  codes4_to_f16(q0.q, c0, c1);
  codes4_to_f16(q1.q, c2, c3);
  sts_u32x2(ctx.s(smem::kRDownActCodesP) + qe * smem::kDownActSlotBytes + ctx.lane * 8, c0, c1);
  sts_u32x2(ctx.s(smem::kRDownActCodesP) + qe * smem::kDownActSlotBytes + 256 + ctx.lane * 8, c2,
            c3);
  if (ctx.lane == 0)
    sts_u32x2(ctx.s(smem::kRDownActScalesP) + qe * 8, __float_as_uint(q0.scale),
              __float_as_uint(q1.scale));
  __syncwarp(0xffffffffu);
  if (ctx.lane == 0) mbar_arrive(ctx.s(smem::kMidReadyBar) + ctx.warp * 8);
  return true;
}

template <bool kBulkMid>
__device__ __forceinline__ void rc7_gate_up(OpCtx& ctx, MidPrefetch* mp) {
  const uint32_t x_s = ctx.s(smem::kRxRow);

  const uint32_t qscales_s = ctx.s(smem::kRQScales);
  if (ctx.warp != 0) {
#pragma unroll 1
    for (int block = ctx.warp - 1; block < kGateUpPages * kConsumerWarps;
         block += kConsumerWarps - 1) {
      const GuQuant128 q = gu_quant128(x_s + block * 256, ctx.lane);
      uint32_t lo, hi;
      codes4_to_f16(q.q, lo, hi);
      sts_u32x2(x_s + block * 256 + ctx.lane * 8, lo, hi);
      if (ctx.lane == 0) sts_f32(qscales_s + block * 4, q.scale);
    }
  }

  if (!kBulkMid) ctx.sync();

  const uint32_t partial_s = ctx.s(smem::kRDownPartial);
  const int g = ctx.lane >> 2;
  for (int rg_i = 0; rg_i < kRowGroupsPerCta; ++rg_i) {
    if (mp != nullptr && rg_i == kRowGroupsPerCta - 1) {
      const uint8_t* const mid_src =
          reinterpret_cast<const uint8_t*>(ctx4(ctx.smem).hidden_mid_in) +
          size_t(ctx.t) * kHiddenMidTokenBytes + size_t(ctx.warp + 1) * kHiddenMidSlotBytes +
          ctx.lane * 8;
      ld_relaxed_gpu_v2u32(mid_src, mp->a0, mp->a1);
      ld_relaxed_gpu_v2u32(mid_src + 256, mp->b0, mp->b1);
      mp->issued = true;
    }
    float acc_g0 = 0.0f, acc_g8 = 0.0f, acc_u0 = 0.0f, acc_u8 = 0.0f;

#pragma unroll 1
    for (int page = (kBulkMid && rg_i == 0 && ctx.warp != 0) ? -1 : 0; page < kGateUpPages;
         ++page) {
      const bool warm = kBulkMid && page < 0;
      if (kBulkMid && rg_i == 0 && page == 0) ctx.sync();
      const uint32_t slot = warm ? x_s : ctx.acquire_page();
      {
        const int qblock = (warm ? 0 : page) * kConsumerWarps + ctx.warp;
        const float xscale = lds_f32(qscales_s + qblock * 4);
        const uint32_t a_g = slot + ctx.warp * kGateUpWarpKBytes + ctx.lane * 8;
        const uint32_t a_u = a_g + kGateUpUpHalf;
        const uint32_t b_s = x_s + qblock * 256 + ctx.lane * 4;
        HmmaAcc dg = {{0.f, 0.f, 0.f, 0.f}}, du = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          uint32_t glo, ghi, ulo, uhi;
          lds_u32x2(a_g + k * kGateUpKStepBytes, glo, ghi);
          lds_u32x2(a_u + k * kGateUpKStepBytes, ulo, uhi);
          uint32_t b[2] = {0u, 0u};
          if (ctx.lane < 4) {
            b[0] = lds_u32(b_s + k * 32);
            b[1] = lds_u32(b_s + k * 32 + 16);
          }
          uint32_t ag[4], au[4];
          fp8x8_to_f16_fragment(glo, ghi, ag);
          fp8x8_to_f16_fragment(ulo, uhi, au);
          hmma_f16(dg, ag, b);
          hmma_f16(du, au, b);
        }
        const float sg = __fmul_rn(lds_f32(slot + kGateUpScaleGate + ctx.warp * 4), xscale);
        const float su = __fmul_rn(lds_f32(slot + kGateUpScaleUp + ctx.warp * 4), xscale);
        if (!warm) ctx.release_page();
        if (mp != nullptr && rg_i == kRowGroupsPerCta - 1 && page == 2 && !mp->resolved)
          mp->resolved = quantize_routed_mid_slot(ctx, mp->a0, mp->a1, mp->b0, mp->b1);
        acc_g0 = fmaf(sg, dg.v[0], acc_g0);
        acc_g8 = fmaf(sg, dg.v[2], acc_g8);
        acc_u0 = fmaf(su, du.v[0], acc_u0);
        acc_u8 = fmaf(su, du.v[2], acc_u8);
        if (warm) {
          acc_g0 = 0.0f;
          acc_g8 = 0.0f;
          acc_u0 = 0.0f;
          acc_u8 = 0.0f;
        }
      }
    }

    const uint32_t rg_partial_s = partial_s + rg_i * 0x400;
    if ((ctx.lane & 3) == 0) {
      sts_f32(rg_partial_s + 0x000 + (g * 8 + ctx.warp) * 4, acc_g0);
      sts_f32(rg_partial_s + 0x100 + (g * 8 + ctx.warp) * 4, acc_g8);
      sts_f32(rg_partial_s + 0x200 + (g * 8 + ctx.warp) * 4, acc_u0);
      sts_f32(rg_partial_s + 0x300 + (g * 8 + ctx.warp) * 4, acc_u8);
    }
    if (kBulkMid) {
      __syncwarp(0xffffffffu);
      if (ctx.lane == 0) mbar_arrive(ctx.s(smem::kUnitBar) + rg_i * 8);
      continue;
    }

    ctx.sync();
    if (ctx.warp == rg_i) {
      const int kind = ctx.lane >> 4;
      const int row = ctx.lane & 15;
      const uint32_t base = rg_partial_s + (kind * 2 + (row >> 3)) * 0x100 + (row & 7) * 32;
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) sum = sum + lds_f32(base + w * 4);
      const float up = __shfl_down_sync(0xffffffffu, sum, 16);
      const int slot_e = 1 + 2 * rg_i + ctx.k;
      float val;
      val = up * silu_approx(sum);
      const uint32_t hidden = mid_payload_bits(f32_to_bf16_bits(val));
      const uint32_t next = __shfl_down_sync(0xffffffffu, hidden, 1);
      if (ctx.lane < 16 && (ctx.lane & 1) == 0) {
        uint16_t* dst = ctx2(ctx.smem).hidden_mid_out + size_t(ctx.t) * (kHiddenMidTokenBytes / 2) +
                        size_t(slot_e) * kInterPerRank + ctx.quarter * kRowsPerGroup + ctx.lane;
        st_relaxed_gpu_u32(dst, hidden | (next << 16));
      }
      if (rg_i == 0 && ctx.lane < kTopk)
        sts_f32(ctx.s(smem::kRProbCopy) + ctx.lane * 4,
                lds_f32(ctx.s(smem::kRSelProb) + ctx.lane * 4));
    }
  }
}

__device__ __forceinline__ void op_select_gate_up(const MoeArgs& args, OpCtx& ctx,
                                                  bool recompute_norm, MidPrefetch* mp = nullptr) {
  if (recompute_norm) routed_norm_from_page(ctx);
  if (recompute_norm)
    rc3_gather_scores(args, ctx, false);
  else
    rc3_gather_scores_early_m1(ctx);
  if (ctx.warp == 0) {
    if (recompute_norm)
      rc5_top8_warp0(ctx, false);
    else
      rc5_top8_warp0_p(ctx);
    __syncwarp(0xffffffffu);
  }
  if (ctx.tid == 31) mbar_arrive(ctx.s(smem::kSelAllBar));
  if (recompute_norm && ctx.c == 0 && ctx.tid < kTopk) {
    const CtxRow2& r2 = ctx2(ctx.smem);
    r2.sel_probs[ctx.t * kTopk + ctx.tid] = lds_f32(ctx.s(smem::kRSelProb) + ctx.tid * 4);
    r2.sel_indices[ctx.t * kTopk + ctx.tid] =
        static_cast<int32_t>(lds_u32(ctx.s(smem::kRSelIdx) + ctx.tid * 4));
  }
  if (recompute_norm)
    rc7_gate_up<false>(ctx, mp);
  else
    rc7_gate_up<true>(ctx, mp);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void sc3_issue_xnorm_page(OpCtx& ctx, const uint16_t* xnorm, int page,
                                                     int buf) {
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    const int idx = ctx.lane * 2 + v;
    const int tok = idx >> 4;
    const int chunk = idx & 15;
    const size_t k_byte = size_t(page) * (kPageK * 2) + ctx.warp * 256 + chunk * 16;
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(xnorm) + size_t(tok) * kXRowBytes + k_byte;
    ldgsts_16(ctx.s(smem::kSxNorm) + buf * smem::kSxNormBufStride + tok * smem::kSxNormTokStride +
                  ctx.warp * 256 + chunk * 16,
              src);
  }
  ldgsts_commit();
}

__device__ __forceinline__ void sc2_word_w1(const MoeArgs& args, OpCtx& ctx) {
  uint32_t* word = args.sync_words + (kSyncW1 / 4);
  ctx.sync();
  if (ctx.tid == 0 && ctx.rgS < kSharedNormCtas) word_arrive_release_only(word, 1u);
  if (ctx.warp == 0) {
    uint32_t v = 0u;
    if (ctx.lane == 0) {
      const uint32_t target = w1_target(lds_u32(ctx.s(smem::kSharedEpoch)));
      while (v != target) {
        spin_backoff();
        v = ld_relaxed_gpu_u32(word);
      }
    }
    __syncwarp(0xffffffffu);
    fence_acquire_gpu();
  }
  ctx.sync();
}

__device__ __forceinline__ void shared_mid_rendezvous(const MoeArgs& args, OpCtx& ctx) {
  uint32_t* const word = args.sync_words + (kSyncSharedMid / sizeof(uint32_t));
  ctx.sync();
  uint32_t old = 0;
  if (ctx.tid == 0) old = word_arrive_release(word, 1u);
  if (ctx.warp == 0) {
    const uint32_t mine = __shfl_sync(0xffffffffu, old, 0);
    const uint32_t target = (mine | (kSharedCtas - 1u)) + 1u;
    uint32_t v = mine + 1u;
    while (v != target) {
      spin_backoff();
      v = ld_relaxed_gpu_u32(word);
    }
    fence_acquire_gpu();
  }
  ctx.sync();
}

__device__ __forceinline__ void op_shared_down(const MoeArgs& args, OpCtx& ctx) {
  shared_mid_rendezvous(args, ctx);
  const uint16_t* const hm = ctx3(ctx.smem).hidden_mid_shared_out;
  const uint32_t mid_s = ctx.s(smem::kSxNorm);
  if (ctx.tid < 128) {
    const int tok = ctx.tid >> 5;
    const int chunk = ctx.tid & 31;
    const uint8_t* const src =
        reinterpret_cast<const uint8_t*>(hm) + size_t(tok) * kHiddenMidTokenBytes + chunk * 16;
    ldgsts_16(mid_s + tok * kHiddenMidSlotBytes + chunk * 16, src);
    ldgsts_commit();
    ldgsts_wait_all();
  }
  ctx.sync();

  const uint32_t codes_s = ctx.s(smem::kDownActCodes);
  const uint32_t scales_s = codes_s + kSeqLen * kInterPerRank;
  const int tok_q = ctx.warp >> 1;
  const int half_q = ctx.warp & 1;
  const GuQuant128 q = gu_quant128(mid_s + tok_q * kHiddenMidSlotBytes + half_q * 256, ctx.lane);
  sts_u32(codes_s + tok_q * kInterPerRank + half_q * 128 + ctx.lane * 4, q.q);
  if (ctx.lane == 0) sts_f32(scales_s + (tok_q * 2 + half_q) * 4, q.scale);
  ctx.sync();

  constexpr int kSharedOutputs = kSeqLen * kDownHalfBlocks * kDownRowsPerCta;

#pragma unroll 1
  for (int c_i = 0; c_i < 2; ++c_i) {
#pragma unroll 1
    for (int half = 0; half < kDownHalfGroups; ++half) {
      const int c = 2 * ctx.rgS + c_i;
      const uint32_t slot = ctx.acquire_page();
      uint32_t bq[2] = {0u, 0u};
      if (ctx.lane < 16) {
        const int tok = ctx.lane >> 2;
        const uint32_t qb = codes_s + tok * kInterPerRank + ctx.warp * 32 + (ctx.lane & 3) * 4;
        bq[0] = lds_u32(qb);
        bq[1] = lds_u32(qb + 16);
      }
#pragma unroll 1
      for (int bi = 0; bi < kDownHalfBlocks; ++bi) {
        const uint32_t blk = slot + bi * kDownSliceBytes;
        uint32_t A[3][4];
#pragma unroll
        for (int tile = 0; tile < 3; ++tile)
          down_weight_k32(blk + ctx.warp * kDownWarpKBytes + tile * kDownTileBytes, ctx.lane,
                          A[tile]);
        float sc[6];
        const uint32_t scale = slot + kDownHalfMatBytes + bi * kDownScaleSliceBytes;
#pragma unroll
        for (int z = 0; z < 6; ++z) sc[z] = lds_f32(scale + z * 8 + (ctx.warp >> 2) * 4);
        if (bi == kDownHalfBlocks - 1) ctx.release_page();
        HmmaAcc D[3] = {{{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}};
#pragma unroll
        for (int tile = 0; tile < 3; ++tile) gu_mma_fp8(D[tile], A[tile], bq);
        if ((ctx.lane & 3) < 2) {
          const int g = ctx.lane >> 2;
          const int tok0 = 2 * (ctx.lane & 3);
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = tok0 + (i & 1);
              const int row = tile * 16 + g + ((i >> 1) ? 8 : 0);
              const float mid_scale = lds_f32(scales_s + (tok * 2 + (ctx.warp >> 2)) * 4);
              const float s = __fmul_rn(__fmul_rn(sc[tile * 2 + (i >> 1)], 1.0f), mid_scale);
              const float v = fmaf(s, D[tile].v[i], 0.0f);
              const int out = (tok * kDownHalfBlocks + bi) * kDownRowsPerCta + row;
              const uint32_t off = (ctx.warp * kSharedOutputs + out) * sizeof(float);
              sts_f32(mid_s + off, v);
            }
          }
        }
      }
      ctx.sync();
      for (int idx = ctx.tid; idx < kSharedOutputs; idx += kConsumerThreads) {
        const int tok = idx / (kDownHalfBlocks * kDownRowsPerCta);
        const int rem = idx % (kDownHalfBlocks * kDownRowsPerCta);
        const int bi = rem / kDownRowsPerCta;
        const int row = rem % kDownRowsPerCta;
        const uint32_t base = mid_s + idx * sizeof(float);
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kConsumerWarps; ++w)
          sum = sum + lds_f32(base + w * kSharedOutputs * sizeof(float));
        const int col = c * kDownColsPerCta + half * kDownHalfBlocks * kDownRowsPerCta +
                        bi * kDownRowsPerCta + row;
        st_relaxed_gpu_u32(args.shared_down_scratch + size_t(tok) * kDim + col,
                           __float_as_uint(sum));
      }
      ctx.sync();
    }

    if (ctx.tid == 0) {
      const int ready_c = 2 * ctx.rgS + c_i;
      uint32_t* const ready = args.sync_words + (kSyncSharedDown / sizeof(uint32_t)) + ready_c;
      st_release_gpu_u32(ready, lds_u32(ctx.s(smem::kSharedEpoch)));
    }
  }
}

__device__ __forceinline__ void op_shared_gate_up(const MoeArgs& args, OpCtx& ctx,
                                                  bool wait_for_helpers) {
  sc2_word_w1(args, ctx);
  uint16_t* hm = ctx3(ctx.smem).hidden_mid_shared_out;
  const uint16_t* xnorm = ctx3(ctx.smem).x_norm_buf;
  const uint32_t partial_s = ctx.s(smem::kSPartial);
  float acc_g[4] = {0.f, 0.f, 0.f, 0.f}, acc_u[4] = {0.f, 0.f, 0.f, 0.f};
  sc3_issue_xnorm_page(ctx, xnorm, 0, 0);
  for (int page = 0; page < kGateUpPages; ++page) {
    if (page + 1 < kGateUpPages) {
      sc3_issue_xnorm_page(ctx, xnorm, page + 1, (page + 1) & 1);
      ldgsts_wait_1();
    } else {
      ldgsts_wait_all();
    }
    __syncwarp(0xffffffffu);
    const uint32_t slot = ctx.acquire_page();
    {
      GuQuant128 xq[4];
#pragma unroll
      for (int tok = 0; tok < 4; ++tok)
        xq[tok] = gu_quant128(ctx.s(smem::kSxNorm) + (page & 1) * smem::kSxNormBufStride +
                                  tok * smem::kSxNormTokStride + ctx.warp * 256,
                              ctx.lane);
      HmmaAcc dg = {{0.f, 0.f, 0.f, 0.f}}, du = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint32_t ag[4], au[4];
        const uint32_t tile = slot + ctx.warp * kGateUpWarpKBytes + k * 2 * kGateUpKStepBytes;
        gu_weight_k32(tile, ctx.lane, ag);
        gu_weight_k32(tile + kGateUpUpHalf, ctx.lane, au);
        uint32_t b[2] = {0u, 0u};
#pragma unroll
        for (int tok = 0; tok < 4; ++tok) {
          const uint32_t b0 = __shfl_sync(0xffffffffu, xq[tok].q, k * 8 + (ctx.lane & 3));
          const uint32_t b1 = __shfl_sync(0xffffffffu, xq[tok].q, k * 8 + 4 + (ctx.lane & 3));
          if ((ctx.lane >> 2) == tok) {
            b[0] = b0;
            b[1] = b1;
          }
        }
        gu_mma_fp8(dg, ag, b);
        gu_mma_fp8(du, au, b);
      }
      const float sg = lds_f32(slot + kGateUpScaleGate + ctx.warp * 4);
      const float su = lds_f32(slot + kGateUpScaleUp + ctx.warp * 4);

      const int l3 = ctx.lane & 3;
      float sx[2];
      sx[0] = l3 == 0 ? xq[0].scale : (l3 == 1 ? xq[2].scale : 0.f);
      sx[1] = l3 == 0 ? xq[1].scale : (l3 == 1 ? xq[3].scale : 0.f);
      ctx.release_page();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc_g[i] = fmaf(__fmul_rn(sg, sx[i & 1]), dg.v[i], acc_g[i]);
        acc_u[i] = fmaf(__fmul_rn(su, sx[i & 1]), du.v[i], acc_u[i]);
      }
    }
    __syncwarp(0xffffffffu);
  }

  if ((ctx.lane & 3) < 2) {
    const int g = ctx.lane >> 2;
    const int tok0 = 2 * (ctx.lane & 3);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = g + ((i >> 1) ? 8 : 0);
      const int tok = tok0 + (i & 1);
      sts_f32(
          partial_s + (((0 * kSeqLen + tok) * kRowsPerGroup + row) * kConsumerWarps + ctx.warp) * 4,
          acc_g[i]);
      sts_f32(
          partial_s + (((1 * kSeqLen + tok) * kRowsPerGroup + row) * kConsumerWarps + ctx.warp) * 4,
          acc_u[i]);
    }
  }
  ctx.sync();
  uint16_t hidden = 0;
  if (ctx.tid < 64) {
    const int row = ctx.tid >> 2;
    const int tok = ctx.tid & 3;
    float val;
    {
      float gate = 0.0f, up = 0.0f;
      const uint32_t gb =
          partial_s + (((0 * kSeqLen + tok) * kRowsPerGroup + row) * kConsumerWarps) * 4;
      const uint32_t ub =
          partial_s + (((1 * kSeqLen + tok) * kRowsPerGroup + row) * kConsumerWarps) * 4;
#pragma unroll
      for (int w = 0; w < kConsumerWarps; ++w) {
        gate = gate + lds_f32(gb + w * 4);
        up = up + lds_f32(ub + w * 4);
      }
      val = up * silu_approx(gate);
    }
    hidden = mid_payload_bits(f32_to_bf16_bits(val));
  }

  if (wait_for_helpers) wait_for_sentinel_helpers(args, ctx);
  if (ctx.tid < 64) {
    const int row = ctx.tid >> 2;
    const int tok = ctx.tid & 3;
    // Adjacent rows of the same token are four lanes apart; even rows never
    // cross a warp. Publish the pair with the b32 width used by mid polling.
    const uint32_t next = __shfl_down_sync(0xffffffffu, uint32_t(hidden), 4);
    if ((row & 1) == 0)
      st_relaxed_gpu_u32(
          hm + size_t(tok) * (kHiddenMidTokenBytes / 2) + ctx.rgS * kRowsPerGroup + row,
          uint32_t(hidden) | (next << 16));
  }
  if (wait_for_helpers) op_shared_down(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void collective_push_receive_reduce(const MoeArgs& args, OpCtx& ctx) {
  if (args.peers[0] == nullptr) {
    if (ctx.tid < kDownColsPerCta) {
      const int b = ctx.tid / kDownRowsPerCta;
      const int row = ctx.tid - b * kDownRowsPerCta;
      const float s = bf16_bits_to_f32(lds_u16(ctx.s(smem::kRDownOut) + ctx.tid * 2));
      const float x = bf16_bits_to_f32(lds_u16(ctx.s(smem::kRResidual) + b * 0x60 + row * 2));
      ctx4(ctx.smem).hidden_out[size_t(ctx.t) * kDim + size_t(ctx.c) * kDownColsPerCta + ctx.tid] =
          f32_to_bf16_bits(s + x);
    }
    return;
  }
  const uint32_t epoch_lo = ctx.epoch;
  const uint32_t parity = epoch_lo & 1u;
  const int sub = ctx.tid >> 6;
  const int peer_idx = (ctx.warp & 1) * 4 + (ctx.tid & 3);
  const int line = ctx.lane >> 2;
  const int chunk = ctx.c * kDownBlocksPerCta + sub;
  const size_t rec_off = size_t(ctx.t) * kInboxTokenStride + size_t(parity) * kInboxParityStride +
                         size_t(chunk) * kInboxChunkStride + size_t(line) * kInboxLineStride;
  const uint32_t own_s = ctx.s(smem::kRDownOut) + sub * smem::kRRxPeerStride;

  if (peer_idx != kWorld - 1) {
    const int dst = peer_idx + (peer_idx >= ctx.rank ? 1 : 0);
    int slot;
    slot = dst > ctx.rank ? ctx.rank : ctx.rank - 1;
    const uint32_t w0 = lds_u32(own_s + line * 12), w1 = lds_u32(own_s + line * 12 + 4),
                   w2 = lds_u32(own_s + line * 12 + 8);
    st_relaxed_sys_b128(args.peers[dst] + size_t(slot) * kInboxSlotStride + rec_off, w0, w1, w2,
                        epoch_lo);
  }

  if (peer_idx != kWorld - 1) {
    const uint8_t* rec = args.peers[ctx.rank] + size_t(peer_idx) * kInboxSlotStride + rec_off;
    uint64_t lo, hi;
    for (;;) {
      ld_relaxed_sys_b128(rec, lo, hi);
      if (static_cast<uint32_t>(hi >> 32) == epoch_lo) break;
      spin_backoff();
    }
    const uint32_t dst_s = ctx.s(smem::kRCollectiveRx) + sub * smem::kRRxSubStride +
                           peer_idx * smem::kRRxPeerStride + line * 12;
    sts_u32(dst_s, static_cast<uint32_t>(lo));
    sts_u32(dst_s + 4, static_cast<uint32_t>(lo >> 32));
    sts_u32(dst_s + 8, static_cast<uint32_t>(hi));
  }
  ctx.sync();

  if ((ctx.warp & 1) == 0 && ctx.lane < 12) {
    const int rsub = ctx.warp >> 1;
    const int e0 = ctx.lane * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int r = 0; r < kWorld; ++r) {
      uint32_t src;
      if (r == ctx.rank)
        src = ctx.s(smem::kRDownOut) + rsub * smem::kRRxPeerStride + e0 * 2;
      else
        src = ctx.s(smem::kRCollectiveRx) + rsub * smem::kRRxSubStride +
              (r < ctx.rank ? r : r - 1) * smem::kRRxPeerStride + e0 * 2;
      uint32_t p0, p1;
      lds_u32x2(src, p0, p1);
      acc[0] = acc[0] + bf16_lo(p0);
      acc[1] = acc[1] + bf16_hi(p0);
      acc[2] = acc[2] + bf16_lo(p1);
      acc[3] = acc[3] + bf16_hi(p1);
    }
    uint32_t xr0, xr1;
    lds_u32x2(ctx.s(smem::kRResidual) + rsub * 0x60 + e0 * 2, xr0, xr1);
    const uint32_t y01 = pack_bf16_pair(acc[0] + bf16_lo(xr0), acc[1] + bf16_hi(xr0));
    const uint32_t y23 = pack_bf16_pair(acc[2] + bf16_lo(xr1), acc[3] + bf16_hi(xr1));
    uint16_t* out = ctx4(ctx.smem).hidden_out + size_t(ctx.t) * kDim +
                    size_t(ctx.c) * kDownColsPerCta + rsub * kDownRowsPerCta + e0;
    st_relaxed_gpu_u32(out, y01);
    st_relaxed_gpu_u32(out + 2, y23);
  }
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__device__ __forceinline__ void rc10_f2_read_page(OpCtx& ctx, int j) {
  const int n = (j == kDownPages - 1) ? 128 : 256;
  if (ctx.tid < n) {
    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(ctx4(ctx.smem).hidden_mid_in) +
        (size_t(ctx.t) * (kHiddenMidTokenBytes / 2) + size_t(j) * 2 * kInterPerRank) * 2 +
        ctx.tid * 4;
    uint32_t v = ld_relaxed_gpu_u32(src);
    while (mid_word_pending(v)) {
      spin_backoff();
      v = ld_relaxed_gpu_u32(src);
    }
    sts_u32(ctx.s(smem::kRHiddenMid) + j * (2 * kHiddenMidSlotBytes) + ctx.tid * 4, v);
  }
  ctx.sync();
}

struct TailAddrs {
  uint8_t* dst[kWorld - 1];
  const uint8_t* inbox;
  bool collective;
};
__device__ __forceinline__ TailAddrs tail_addrs(const MoeArgs& args, const OpCtx& ctx) {
  const int sub = ctx.lane >> 3;
  const int line = ctx.lane & 7;
  const uint32_t parity = ctx.epoch & 1u;
  const int chunk = ctx.c * kDownBlocksPerCta + sub;
  const size_t rec_off = size_t(ctx.t) * kInboxTokenStride + size_t(parity) * kInboxParityStride +
                         size_t(chunk) * kInboxChunkStride + size_t(line) * kInboxLineStride;
  TailAddrs a;
#pragma unroll
  for (int pi = 0; pi < kWorld - 1; ++pi) {
    const int dst = pi + (pi >= ctx.rank ? 1 : 0);
    const int slot = dst > ctx.rank ? ctx.rank : ctx.rank - 1;
    a.dst[pi] = args.peers[dst] + size_t(slot) * kInboxSlotStride + rec_off;
    asm volatile("" : "+l"(a.dst[pi]));
  }
  a.inbox = args.peers[ctx.rank] + rec_off;
  asm volatile("" : "+l"(a.inbox));
  a.collective = args.peers[0] != nullptr;
  return a;
}

template <int R>
__device__ __forceinline__ uint32_t tail_pick(int rank, uint32_t own,
                                              const uint32_t (&rx)[kWorld - 1]) {
  return R == rank ? own : (R < rank ? rx[R < kWorld - 1 ? R : kWorld - 2] : rx[R > 0 ? R - 1 : 0]);
}
__device__ __forceinline__ void fused_tail_warp0(const MoeArgs& args, OpCtx& ctx,
                                                 uint32_t partial_s, size_t out_col0,
                                                 bool term_in_smem = false,
                                                 const TailAddrs* pre = nullptr) {
  const int sub = ctx.lane >> 3;
  const int line = ctx.lane & 7;
  const int col0 = sub * kDownRowsPerCta + line * 6;
  const bool term_ready = term_in_smem || lds_u32(ctx.s(smem::kRSharedTermReady)) != 0u;
  float v[6];

  static_assert(smem::kRDownPartialWarpStride % 8 == 0 && smem::kRDownPartialBlockStride % 8 == 0 &&
                    smem::kRDownPartial % 8 == 0 && smem::kRSharedTerm % 8 == 0,
                "8 B alignment of the tail fold");
#pragma unroll
  for (int i = 0; i < 6; i += 2) {
    const int row = line * 6 + i;
    float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      uint32_t a, b;
      lds_u32x2(partial_s + w * smem::kRDownPartialWarpStride +
                    sub * smem::kRDownPartialBlockStride + row * 4,
                a, b);
      sum0 = sum0 + __uint_as_float(a);
      sum1 = sum1 + __uint_as_float(b);
    }
    float t0, t1;
    if (term_ready) {
      uint32_t a, b;
      lds_u32x2(ctx.s(smem::kRSharedTerm) + (col0 + i) * 4, a, b);
      t0 = __uint_as_float(a);
      t1 = __uint_as_float(b);
    } else {
      const size_t p = size_t(ctx.t) * kDim + out_col0 + col0 + i;
      t0 = __uint_as_float(ld_relaxed_gpu_u32(args.shared_down_scratch + p));
      t1 = __uint_as_float(ld_relaxed_gpu_u32(args.shared_down_scratch + p + 1));
    }
    v[i] = t0 + sum0;
    v[i + 1] = t1 + sum1;
  }

  const uint32_t w0 = pack_bf16_pair(v[0], v[1]), w1 = pack_bf16_pair(v[2], v[3]),
                 w2 = pack_bf16_pair(v[4], v[5]);
  float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (pre != nullptr ? pre->collective : args.peers[0] != nullptr) {
    const uint32_t epoch_lo = ctx.epoch;
    const uint32_t parity = epoch_lo & 1u;
    const int chunk = ctx.c * kDownBlocksPerCta + sub;
    const size_t rec_off = size_t(ctx.t) * kInboxTokenStride + size_t(parity) * kInboxParityStride +
                           size_t(chunk) * kInboxChunkStride + size_t(line) * kInboxLineStride;
#pragma unroll
    for (int pi = 0; pi < kWorld - 1; ++pi) {
      const int dst = pi + (pi >= ctx.rank ? 1 : 0);
      const int slot = dst > ctx.rank ? ctx.rank : ctx.rank - 1;
      uint8_t* const target = pre != nullptr
                                  ? pre->dst[pi]
                                  : args.peers[dst] + size_t(slot) * kInboxSlotStride + rec_off;
      st_relaxed_sys_b128(target, w0, w1, w2, epoch_lo);
    }
    uint32_t r0[kWorld - 1], r1[kWorld - 1], r2[kWorld - 1];
    uint32_t pending = (1u << (kWorld - 1)) - 1u;
    const uint8_t* const inbox = pre != nullptr ? pre->inbox : args.peers[ctx.rank] + rec_off;
    while (pending != 0u) {
      uint64_t lo[kWorld - 1], hi[kWorld - 1];
#pragma unroll
      for (int pi = 0; pi < kWorld - 1; ++pi)
        if ((pending >> pi) & 1u)
          ld_relaxed_sys_b128(inbox + size_t(pi) * kInboxSlotStride, lo[pi], hi[pi]);
#pragma unroll
      for (int pi = 0; pi < kWorld - 1; ++pi) {
        if (((pending >> pi) & 1u) && static_cast<uint32_t>(hi[pi] >> 32) == epoch_lo) {
          r0[pi] = static_cast<uint32_t>(lo[pi]);
          r1[pi] = static_cast<uint32_t>(lo[pi] >> 32);
          r2[pi] = static_cast<uint32_t>(hi[pi]);
          pending &= ~(1u << pi);
        }
      }
    }

    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#define MOE_TAIL_ADD(R)                                                                    \
  {                                                                                        \
    const uint32_t a = tail_pick<R>(ctx.rank, w0, r0), b = tail_pick<R>(ctx.rank, w1, r1), \
                   c = tail_pick<R>(ctx.rank, w2, r2);                                     \
    acc[0] = acc[0] + bf16_lo(a);                                                          \
    acc[1] = acc[1] + bf16_hi(a);                                                          \
    acc[2] = acc[2] + bf16_lo(b);                                                          \
    acc[3] = acc[3] + bf16_hi(b);                                                          \
    acc[4] = acc[4] + bf16_lo(c);                                                          \
    acc[5] = acc[5] + bf16_hi(c);                                                          \
  }
    MOE_TAIL_ADD(0)
    MOE_TAIL_ADD(1)
    MOE_TAIL_ADD(2) MOE_TAIL_ADD(3) MOE_TAIL_ADD(4) MOE_TAIL_ADD(5) MOE_TAIL_ADD(6) MOE_TAIL_ADD(7)
#undef MOE_TAIL_ADD
  } else {
    acc[0] = bf16_lo(w0);
    acc[1] = bf16_hi(w0);
    acc[2] = bf16_lo(w1);
    acc[3] = bf16_hi(w1);
    acc[4] = bf16_lo(w2);
    acc[5] = bf16_hi(w2);
  }
  const uint32_t res_s = ctx.s(smem::kRResidual) + sub * 0x60 + line * 12;
  const uint32_t x0 = lds_u32(res_s), x1 = lds_u32(res_s + 4), x2 = lds_u32(res_s + 8);
  uint16_t* const out = ctx4(ctx.smem).hidden_out + size_t(ctx.t) * kDim + out_col0 + col0;
  st_relaxed_gpu_u32(out, pack_bf16_pair(acc[0] + bf16_lo(x0), acc[1] + bf16_hi(x0)));
  st_relaxed_gpu_u32(out + 2, pack_bf16_pair(acc[2] + bf16_lo(x1), acc[3] + bf16_hi(x1)));
  st_relaxed_gpu_u32(out + 4, pack_bf16_pair(acc[4] + bf16_lo(x2), acc[5] + bf16_hi(x2)));
}

__device__ __forceinline__ void op_down_allreduce(const MoeArgs& args, OpCtx& ctx,
                                                  bool probs_from_global,
                                                  const MidPrefetch* mp = nullptr,
                                                  bool aux_warp = false) {
  const CtxRow4& r4 = ctx4(ctx.smem);
  if (probs_from_global) {
    if (ctx.tid < kTopk)
      sts_f32(ctx.s(smem::kRProbCopy) + ctx.tid * 4,
              ctx2(ctx.smem).sel_probs[ctx.t * kTopk + ctx.tid]);
    ctx.sync();
  }

  const size_t out_col0 = size_t(ctx.c) * kDownColsPerCta;
  const uint8_t* mid_src = nullptr;
  uint32_t ma0 = 0u, ma1 = 0u, mb0 = 0u, mb1 = 0u;
  bool mid_pending = false;
  if (!aux_warp && ctx.tid < 6) {
    const uint8_t* xr = reinterpret_cast<const uint8_t*>(r4.x_residual) +
                        (size_t(ctx.t) * kDim + out_col0) * 2 + ctx.tid * 16;
#pragma unroll
    for (int b = 0; b < kDownBlocksPerCta; ++b)
      ldgsts_16(ctx.s(smem::kRResidual) + b * 0x60 + ctx.tid * 16, xr + b * 0x60);
    ldgsts_commit();
  }
  if (!probs_from_global && !aux_warp) {
    mid_src = reinterpret_cast<const uint8_t*>(ctx4(ctx.smem).hidden_mid_in) +
              size_t(ctx.t) * kHiddenMidTokenBytes + size_t(ctx.warp + 1) * kHiddenMidSlotBytes +
              ctx.lane * 8;
    if (mp != nullptr && mp->issued) {
      ma0 = mp->a0;
      ma1 = mp->a1;
      mb0 = mp->b0;
      mb1 = mp->b1;
    } else {
      ld_relaxed_gpu_v2u32(mid_src, ma0, ma1);
      ld_relaxed_gpu_v2u32(mid_src + 256, mb0, mb1);
    }
    mid_pending = !(mp != nullptr && mp->resolved);
  }
  const uint32_t partial_s = ctx.s(smem::kRDownPartial);
  const uint32_t hm_s = ctx.s(smem::kRHiddenMid);
  const int g = ctx.lane >> 2;
  const int khalf = ctx.tid >> 7;
  const int expert_begin = probs_from_global ? 0 : 1;
  TailAddrs tail_pre;
  if (!probs_from_global && aux_warp && ctx.warp == 0) tail_pre = tail_addrs(args, ctx);

  bool shared_prefetched = false;

  uint32_t shared_flag = 0u;
  if (!probs_from_global && !aux_warp && ctx.warp == 0)
    shared_flag =
        ld_relaxed_gpu_u32(args.sync_words + (kSyncSharedDown / sizeof(uint32_t)) + ctx.c);
  if (!probs_from_global) {
    float acc[kDownHalfGroups][kDownHalfBlocks][6] = {};
#pragma unroll 1
    for (int e = 1; e < kExpertSlots; ++e) {
      if (!aux_warp && e == 5 && ctx.warp == 0) {
        if (__all_sync(0xffffffffu, shared_flag != 0u)) {
          fence_acquire_gpu();
          const uint8_t* const src = reinterpret_cast<const uint8_t*>(
              args.shared_down_scratch + size_t(ctx.t) * kDim + out_col0);
          for (int i = ctx.lane; i < kDownColsPerCta * 4 / 16; i += 32)
            ldgsts_16(ctx.s(smem::kRSharedTerm) + i * 16, src + i * 16);
          ldgsts_commit();
          shared_prefetched = true;
        }
      }

      if (mid_pending) {
        for (;;) {
          const bool ok = !(mid_word_pending(ma0) || mid_word_pending(ma1) ||
                            mid_word_pending(mb0) || mid_word_pending(mb1));
          if (quantize_routed_mid_slot(ctx, ma0, ma1, mb0, mb1)) {
            mid_pending = false;
            break;
          }
          if (!ok) {
            ld_relaxed_gpu_v2u32(mid_src, ma0, ma1);
            ld_relaxed_gpu_v2u32(mid_src + 256, mb0, mb1);
          }
          if (ctx.warp + 1 != e) break;
          spin_backoff();
        }
      }
      mbar_wait_parity(ctx.s(smem::kMidReadyBar) + (e - 1) * 8, 0u);
      uint32_t b0[2] = {0u, 0u}, b1[2] = {0u, 0u};
      if (ctx.lane < 4) {
        const uint32_t hb = ctx.s(smem::kRDownActCodesP) + e * smem::kDownActSlotBytes +
                            ctx.warp * 64 + ctx.lane * 4;
        b0[0] = lds_u32(hb);
        b0[1] = lds_u32(hb + 16);
        b1[0] = lds_u32(hb + 32);
        b1[1] = lds_u32(hb + 48);
      }
      const float mid_scale = lds_f32(ctx.s(smem::kRDownActScalesP) + e * 8 + khalf * 4);
      const float prob = lds_f32(ctx.s(smem::kRSelProb) + (e - 1) * 4);
#pragma unroll
      for (int group = 0; group < kDownHalfGroups; ++group) {
        const uint32_t dq = static_cast<uint32_t>(2 * (e - 1) + group);
        const uint32_t dd = dq % smem::kDownRingSlots;
        mbar_wait_parity(ctx.s(smem::kDownFullBar) + dd * 8, (dq / smem::kDownRingSlots) & 1u);
        const uint32_t slot = ctx.s(smem::kRing) + dd * kDownHalfStageBytes;
#pragma unroll
        for (int bi = 0; bi < kDownHalfBlocks; ++bi) {
          const uint32_t blk = slot + bi * kDownSliceBytes;
          const uint32_t a_base = blk + ctx.warp * kDownWarpKBytes + ctx.lane * 16;
          uint32_t A[3][4];
#pragma unroll
          for (int tile = 0; tile < 3; ++tile)
            lds_u32x4(a_base + tile * kDownTileBytes, A[tile][0], A[tile][1], A[tile][2],
                      A[tile][3]);
          float sc[6];
          const uint32_t scale = slot + kDownHalfMatBytes + bi * kDownScaleSliceBytes;
#pragma unroll
          for (int q = 0; q < 6; ++q) sc[q] = lds_f32(scale + q * 8 + khalf * 4);
          if (bi == kDownHalfBlocks - 1) {
            __syncwarp();
            if (ctx.lane == 0) mbar_arrive(ctx.s(smem::kDownEmptyBar) + dd * 8);
          }
          HmmaAcc D[3] = {{{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}};
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            uint32_t a0[4], a1[4];
            fp8x8_to_f16_fragment(A[tile][0], A[tile][1], a0);
            fp8x8_to_f16_fragment(A[tile][2], A[tile][3], a1);
            hmma_f16(D[tile], a0, b0);
            hmma_f16(D[tile], a1, b1);
          }
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            const float s0 = __fmul_rn(__fmul_rn(sc[tile * 2], prob), mid_scale);
            const float s8 = __fmul_rn(__fmul_rn(sc[tile * 2 + 1], prob), mid_scale);
            acc[group][bi][tile * 2] = fmaf(s0, D[tile].v[0], acc[group][bi][tile * 2]);
            acc[group][bi][tile * 2 + 1] = fmaf(s8, D[tile].v[2], acc[group][bi][tile * 2 + 1]);
          }
        }
      }
    }
    if (aux_warp) mbar_wait_parity(ctx.s(smem::kFoldsDoneBar), 0u);
    if ((ctx.lane & 3) == 0) {
#pragma unroll
      for (int group = 0; group < kDownHalfGroups; ++group) {
#pragma unroll
        for (int bi = 0; bi < kDownHalfBlocks; ++bi) {
          const int b = group * kDownHalfBlocks + bi;
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            sts_f32(partial_s + ctx.warp * smem::kRDownPartialWarpStride +
                        b * smem::kRDownPartialBlockStride + (tile * 16 + g) * 4,
                    acc[group][bi][tile * 2]);
            sts_f32(partial_s + ctx.warp * smem::kRDownPartialWarpStride +
                        b * smem::kRDownPartialBlockStride + (tile * 16 + g + 8) * 4,
                    acc[group][bi][tile * 2 + 1]);
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int group = 0; group < kDownHalfGroups; ++group) {
      if (!probs_from_global && group == 1 && ctx.warp == 0) {
        if (__all_sync(0xffffffffu, shared_flag != 0u)) {
          fence_acquire_gpu();
          const uint8_t* const src = reinterpret_cast<const uint8_t*>(
              args.shared_down_scratch + size_t(ctx.t) * kDim + out_col0);
          for (int i = ctx.lane; i < kDownColsPerCta * 4 / 16; i += 32)
            ldgsts_16(ctx.s(smem::kRSharedTerm) + i * 16, src + i * 16);
          ldgsts_commit();
          shared_prefetched = true;
        }
      }
      float acc[kDownHalfBlocks][6] = {};
#pragma unroll 1
      for (int e = expert_begin; e < kExpertSlots; ++e) {
        if (!probs_from_global && group == 0)
          mbar_wait_parity(ctx.s(smem::kMidReadyBar) + (e - 1) * 8, 0u);
        const uint32_t slot = ctx.acquire_page();
        if (probs_from_global && group == 0 && (e & 1) == 0) rc10_f2_read_page(ctx, e >> 1);

        if (probs_from_global && group == 0 && ((e - expert_begin) & 1) == 0) {
          if ((ctx.warp & 3) < 2) {
            const int qe = e + (ctx.warp & 1);
            if (qe < kExpertSlots) {
              const GuQuant128 q =
                  gu_quant128(hm_s + qe * kHiddenMidSlotBytes + khalf * 256, ctx.lane);
              uint32_t c0, c1;
              codes4_to_f16(q.q, c0, c1);
              sts_u32x2(ctx.s(smem::kDownActCodes) + qe * smem::kDownActSlotBytes + khalf * 256 +
                            ctx.lane * 8,
                        c0, c1);
              if (ctx.lane == 0) sts_f32(ctx.s(smem::kDownActScales) + qe * 8 + khalf * 4, q.scale);
            }
          }
          ctx.sync();
        }
        uint32_t b0[2] = {0u, 0u}, b1[2] = {0u, 0u};
        if (ctx.lane < 4) {
          const uint32_t hb = ctx.s(smem::kDownActCodes) + e * smem::kDownActSlotBytes +
                              ctx.warp * 64 + ctx.lane * 4;
          b0[0] = lds_u32(hb);
          b0[1] = lds_u32(hb + 16);
          b1[0] = lds_u32(hb + 32);
          b1[1] = lds_u32(hb + 48);
        }
        const float mid_scale = lds_f32(ctx.s(smem::kDownActScales) + e * 8 + khalf * 4);

        const float prob =
            (e == 0) ? 1.0f
                     : lds_f32(ctx.s(probs_from_global ? smem::kRProbCopy : smem::kRSelProb) +
                               (e - 1) * 4);

#pragma unroll
        for (int bi = 0; bi < kDownHalfBlocks; ++bi) {
          const uint32_t blk = slot + bi * kDownSliceBytes;
          const uint32_t a_base = blk + ctx.warp * kDownWarpKBytes + ctx.lane * 16;
          uint32_t A[3][4];
#pragma unroll
          for (int tile = 0; tile < 3; ++tile)
            lds_u32x4(a_base + tile * kDownTileBytes, A[tile][0], A[tile][1], A[tile][2],
                      A[tile][3]);

          float sc[6];
          const uint32_t scale = slot + kDownHalfMatBytes + bi * kDownScaleSliceBytes;
#pragma unroll
          for (int q = 0; q < 6; ++q) sc[q] = lds_f32(scale + q * 8 + khalf * 4);
          if (bi == kDownHalfBlocks - 1) ctx.release_page();

          HmmaAcc D[3] = {{{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}};
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            uint32_t a0[4], a1[4];
            fp8x8_to_f16_fragment(A[tile][0], A[tile][1], a0);
            fp8x8_to_f16_fragment(A[tile][2], A[tile][3], a1);
            hmma_f16(D[tile], a0, b0);
            hmma_f16(D[tile], a1, b1);
          }
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            const float s0 = __fmul_rn(__fmul_rn(sc[tile * 2], prob), mid_scale);
            const float s8 = __fmul_rn(__fmul_rn(sc[tile * 2 + 1], prob), mid_scale);
            acc[bi][tile * 2] = fmaf(s0, D[tile].v[0], acc[bi][tile * 2]);
            acc[bi][tile * 2 + 1] = fmaf(s8, D[tile].v[2], acc[bi][tile * 2 + 1]);
          }
        }
      }

      if ((ctx.lane & 3) == 0) {
#pragma unroll
        for (int bi = 0; bi < kDownHalfBlocks; ++bi) {
          const int b = group * kDownHalfBlocks + bi;
#pragma unroll
          for (int tile = 0; tile < 3; ++tile) {
            sts_f32(partial_s + ctx.warp * smem::kRDownPartialWarpStride +
                        b * smem::kRDownPartialBlockStride + (tile * 16 + g) * 4,
                    acc[bi][tile * 2]);
            sts_f32(partial_s + ctx.warp * smem::kRDownPartialWarpStride +
                        b * smem::kRDownPartialBlockStride + (tile * 16 + g + 8) * 4,
                    acc[bi][tile * 2 + 1]);
          }
        }
      }
    }
  }
  if (!aux_warp && ctx.warp == 0) ldgsts_wait_all();
  if (!probs_from_global && !aux_warp) {
    if (ctx.tid == 0) {
      if (!shared_prefetched) {
        const uint32_t* const word = args.sync_words + (kSyncSharedDown / sizeof(uint32_t)) + ctx.c;
        uint32_t v = ld_relaxed_gpu_u32(word);
        while (v == 0u) {
          spin_backoff();
          v = ld_relaxed_gpu_u32(word);
        }
        fence_acquire_gpu();
      }
      sts_u32(ctx.s(smem::kRSharedTermReady), shared_prefetched ? 1u : 0u);
    }
  }
  ctx.sync();
  if (!probs_from_global) {
    if (ctx.warp == 0) {
      if (aux_warp) mbar_wait_parity(ctx.s(smem::kSharedReadyBar), 0u);
      fused_tail_warp0(args, ctx, partial_s, out_col0, aux_warp, aux_warp ? &tail_pre : nullptr);
    }
    return;
  }
  if (ctx.tid < kDownBlocksPerCta * kDownRowsPerCta) {
    float sum = 0.0f;
    {
      const int b = ctx.tid / kDownRowsPerCta;
      const int row = ctx.tid - b * kDownRowsPerCta;
#pragma unroll
      for (int w = 0; w < kConsumerWarps; ++w) {
        sum = sum + lds_f32(partial_s + w * smem::kRDownPartialWarpStride +
                            b * smem::kRDownPartialBlockStride + row * 4);
      }
      if (!probs_from_global) {
        const size_t p = size_t(ctx.t) * kDim + out_col0 + b * kDownRowsPerCta + row;
        const float shared_term =
            lds_u32(ctx.s(smem::kRSharedTermReady)) != 0u
                ? lds_f32(ctx.s(smem::kRSharedTerm) + ctx.tid * 4)
                : __uint_as_float(ld_relaxed_gpu_u32(args.shared_down_scratch + p));
        sum = shared_term + sum;
      }
    }
    sts_u16(ctx.s(smem::kRDownOut) + ctx.tid * 2, f32_to_bf16_bits(sum));
  }
  ctx.sync();
  collective_push_receive_reduce(args, ctx);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

struct AuxProgress {
  uint32_t pushed, own_at_load, v;
  unsigned long long last;
  bool have;
};
__device__ __forceinline__ void aux_progress(AuxProgress& a, uint32_t sbase, uint32_t* prog, int t,
                                             int lane) {
  const unsigned long long now = globaltimer_ns();
  if (now - a.last < 400) return;
  a.last = now;
  if (a.have) {
    const uint32_t sum = __reduce_add_sync(0xffffffffu, a.v);
    const bool ahead = static_cast<unsigned long long>(a.own_at_load) * kRoutedCtas >
                       static_cast<unsigned long long>(sum) +
                           static_cast<unsigned long long>(28672 >> 8) * kRoutedCtas;
    if (lane == 0) sts_vol_u32(sbase + smem::kCapFlag, ahead ? 1u : 0u);
  }
  const uint32_t own = lds_vol_u32(sbase + smem::kProgOwn);
  if (lane == 0 && own > a.pushed)
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(prog + t), "r"(own - a.pushed)
                 : "memory");
  a.pushed = own;
  a.v = lane < kSeqLen ? ld_relaxed_gpu_u32(prog + lane) : 0u;
  a.own_at_load = own;
  a.have = true;
}
__device__ __forceinline__ void aux_main_routed(const MoeArgs& args, uint8_t* smem, int cta) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int t = cta >> 5;
  const int c = cta & (kCtasPerToken - 1);
  const uint32_t sbase = smem_u32(smem);
  const size_t out_col0 = size_t(c) * kDownColsPerCta;
  grid_dependency_wait();

  if (lane < 24) {
    const uint8_t* xr = reinterpret_cast<const uint8_t*>(args.context.direct.r4_x_residual) +
                        (size_t(t) * kDim + out_col0) * 2;
    ldgsts_16(sbase + smem::kRResidual + lane * 16, xr + lane * 16);
  }
  ldgsts_commit();

  const uint32_t epoch = ld_relaxed_gpu_u32(args.launch_ctr + cta);

  const int k = c >> 4;
  const uint8_t* const hm = reinterpret_cast<const uint8_t*>(args.context.direct.r4_hidden_mid_in) +
                            size_t(t) * kHiddenMidTokenBytes + lane * 8;
  uint16_t* const mid_out = args.context.direct.r2_hidden_mid_out +
                            size_t(t) * (kHiddenMidTokenBytes / 2) + size_t(c & 15) * kRowsPerGroup;
  int u = 0;
  int e = 1;
  uint32_t* const prog =
      args.sync_words + (kSyncProgress + (epoch & 1u) * kSyncProgressBankStride) / sizeof(uint32_t);

  if (cta == 0 && lane == 0)
    st_relaxed_gpu_v4u32(
        args.sync_words +
            (kSyncProgress + ((epoch + 1u) & 1u) * kSyncProgressBankStride) / sizeof(uint32_t),
        0u, 0u, 0u, 0u);
  AuxProgress ap = {0u, 0u, 0u, 0ull, false};
#pragma unroll 1
  while (u < kRowGroupsPerCta || e < kExpertSlots) {
    aux_progress(ap, sbase, prog, t, lane);
    if (u < kRowGroupsPerCta &&
        (e >= kExpertSlots ? (mbar_wait_parity(sbase + smem::kUnitBar + u * 8, 0u), true)
                           : mbar_test_wait_parity(sbase + smem::kUnitBar + u * 8, 0u))) {
      const int kind = lane >> 4;
      const int row = lane & 15;
      const uint32_t pbase = sbase + smem::kRDownPartial + u * 0x400 +
                             (kind * 2 + (row >> 3)) * 0x100 + (row & 7) * 32;
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < kConsumerWarps; ++w) sum = sum + lds_f32(pbase + w * 4);
      const float up = __shfl_down_sync(0xffffffffu, sum, 16);
      const float val = up * silu_approx(sum);
      const uint32_t hidden = mid_payload_bits(f32_to_bf16_bits(val));
      const uint32_t next = __shfl_down_sync(0xffffffffu, hidden, 1);
      // A generic b32 publication matches the polling loads. An asynchronous
      // bulk copy cannot itself signal completion through its numeric payload.
      if (lane < 16 && (lane & 1) == 0)
        st_relaxed_gpu_u32(mid_out + size_t(1 + 2 * u + k) * kInterPerRank + lane,
                           hidden | (next << 16));
      __syncwarp(0xffffffffu);
      ++u;
      if (u == kRowGroupsPerCta && lane == 0) mbar_arrive(sbase + smem::kFoldsDoneBar);
      continue;
    }
    if (e < kExpertSlots) {
      const uint8_t* const src = hm + size_t(e) * kHiddenMidSlotBytes;
      uint32_t a0, a1, b0, b1;
      ld_relaxed_gpu_v2u32(src, a0, a1);
      ld_relaxed_gpu_v2u32(src + 256, b0, b1);
      const bool ok = !(mid_word_pending(a0) || mid_word_pending(a1) || mid_word_pending(b0) ||
                        mid_word_pending(b1));
      if (__all_sync(0xffffffffu, ok)) {
        const GuQuant128 q0 = gu_quant128_regs(a0, a1);
        const GuQuant128 q1 = gu_quant128_regs(b0, b1);
        uint32_t c0, c1, c2, c3;
        codes4_to_f16(q0.q, c0, c1);
        codes4_to_f16(q1.q, c2, c3);
        sts_u32x2(sbase + smem::kRDownActCodesP + e * smem::kDownActSlotBytes + lane * 8, c0, c1);
        sts_u32x2(sbase + smem::kRDownActCodesP + e * smem::kDownActSlotBytes + 256 + lane * 8, c2,
                  c3);
        if (lane == 0)
          sts_u32x2(sbase + smem::kRDownActScalesP + e * 8, __float_as_uint(q0.scale),
                    __float_as_uint(q1.scale));
        __syncwarp(0xffffffffu);
        if (lane == 0) mbar_arrive(sbase + smem::kMidReadyBar + (e - 1) * 8);
        ++e;
        continue;
      }
    }
    __nanosleep(128);
  }
  const uint32_t* const word = args.sync_words + (kSyncSharedDown / sizeof(uint32_t)) + c;
  for (;;) {
    const uint32_t v = ld_relaxed_gpu_u32(word);
    if (__all_sync(0xffffffffu, v == epoch)) break;
    aux_progress(ap, sbase, prog, t, lane);
    __nanosleep(128);
  }
  fence_acquire_gpu();
  const uint8_t* const st =
      reinterpret_cast<const uint8_t*>(args.shared_down_scratch + size_t(t) * kDim + out_col0);
#pragma unroll
  for (int i = lane; i < kDownColsPerCta * 4 / 16; i += 32)
    ldgsts_16(sbase + smem::kRSharedTerm + i * 16, st + i * 16);
  ldgsts_commit();
  ldgsts_wait_all();
  __syncwarp(0xffffffffu);
  if (lane == 0) mbar_arrive(sbase + smem::kSharedReadyBar);

#pragma unroll 1
  while (lds_vol_u32(sbase + smem::kPfDone) == 0u) {
    aux_progress(ap, sbase, prog, t, lane);
    __nanosleep(128);
  }
  if (lane == 0) st_relaxed_gpu_u32(args.launch_ctr + cta, epoch + 1u);
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_q4 {

__global__ __launch_bounds__(kBlock, 1) void moe_persistent(const __grid_constant__ MoeArgs args) {
  extern __shared__ __align__(1024) uint8_t dyn[];
  const int cta = static_cast<int>(blockIdx.x);
  if (cta >= kWorkCtas) {
    grid_dependency_wait();
    uint32_t generation = 0;
    if (threadIdx.x == 0) generation = ld_relaxed_gpu_u32(args.launch_ctr + cta);
    initialize_token_sentinels(args, cta - kWorkCtas);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t* word = args.sync_words + (kSyncInit / sizeof(uint32_t));
      word_arrive_release_only(word, 1u);
      st_relaxed_gpu_u32(args.launch_ctr + cta, generation + 1u);
    }
    return;
  }
  RoutedGamma entry_gamma;
  if (cta < kRoutedCtas && threadIdx.x == kConsumerThreads) {
    routed_early_router_pages(args, dyn, cta);
    grid_dependency_wait();
  }
  if (cta < kRoutedCtas && threadIdx.x < kConsumerThreads)
    routed_issue_entry_loads(args, dyn, cta, entry_gamma);
  ctx_table_to_smem(args, dyn);
  if (cta >= kRoutedCtas) ring_init_barriers(dyn);
  __syncthreads();

  if (threadIdx.x >= kConsumerThreads) {
    if (threadIdx.x >= kAuxBase) {
      if (cta < kRoutedCtas) aux_main_routed(args, dyn, cta);
      return;
    }
    PrefetchCtx p = make_prefetch_ctx(args, cta, dyn);
    if (p.routed)
      pf_main_routed(p);
    else
      pf_main_shared(p);
    return;
  }

  OpCtx ctx = make_op_ctx(args, cta, dyn);
  if (ctx.routed) {
    ctx.epoch = entry_gamma.epoch;
    op_rmsnorm_router_routed(args, ctx, true, &entry_gamma);
    op_select_gate_up(args, ctx, false, nullptr);
    op_down_allreduce(args, ctx, false, nullptr, true);
  } else {
    grid_dependency_wait();

    const uint32_t epoch = ld_relaxed_gpu_u32(args.launch_ctr + cta);
    op_rmsnorm_router_shared(args, ctx);
    if (ctx.tid == 0) sts_u32(ctx.s(smem::kSharedEpoch), epoch);
    op_shared_gate_up(args, ctx, true);
    if (ctx.tid == 0)
      st_relaxed_gpu_u32(args.launch_ctr + cta, lds_u32(ctx.s(smem::kSharedEpoch)) + 1u);
  }
}

}  // namespace flashinfer::glm5_tp8::moe_q4

namespace flashinfer::glm5_tp8::moe_detail {
const void* q4_kernel() { return reinterpret_cast<const void*>(&moe_q4::moe_persistent); }
cudaError_t launch_q4(const MoeArguments& p, cudaStream_t stream, bool pdl, bool cooperative) {
  moe_q4::MoeArgs a{};
  a.rank = p.rank;
  a.sync_words = reinterpret_cast<uint32_t*>(p.state);
  a.launch_ctr = reinterpret_cast<uint32_t*>(p.state + kMoeQ4CounterOffset);
  a.shared_down_scratch = p.shared_down_scratch;
  for (int rank = 0; rank < 8; ++rank) a.peers[rank] = p.peers[rank];
  auto& c = a.context.direct;
  c.r0_x = c.r3_x_shared = c.r4_x_residual = p.x;
  c.r0_gamma = p.gamma;
  c.r1_router_w = p.router;
  c.r1_scores_out = p.scores;
  c.r2_scores_in = p.scores;
  c.r2_gate_up_w = c.r3_gate_up_w_shared = p.gate_up;
  c.r2_bias = p.bias;
  c.r2_sel_probs = p.probabilities;
  c.r2_sel_indices = p.indices;
  c.r2_hidden_mid_out = c.r3_hidden_mid_shared_out = p.mid;
  c.r4_hidden_mid_in = p.mid;
  c.r3_x_norm_buf = p.x_norm;
  c.r4_down_w = p.down;
  c.r4_down_scale = p.down_scale;
  c.r4_hidden_out = p.out;
  return launch(q4_kernel(), &a, 320, 193536, stream, pdl, cooperative);
}
}  // namespace flashinfer::glm5_tp8::moe_detail
