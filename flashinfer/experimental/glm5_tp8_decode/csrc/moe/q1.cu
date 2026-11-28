/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include "internal.cuh"
#include "q1_types.cuh"

// MoE layer for one decode token on one TP rank of eight: one launch of kGrid = 148 CTAs x
// kQ1Block = 384 threads (q1_types.cuh).  CTA roles:
//   CTAs 0..127   routed (q1n_routed): x RMSNorm and the router GEMV of their share of the experts,
//                 top-8 over the published scores, gate/up of the selected experts, then the Down
//                 GEMV of their output columns over all kExpertSlots expert slots (slot 0 is the
//                 shared expert) with the TP all-reduce through the peers' inboxes.
//   CTAs 128..143 shared expert (q1n_shared): RMSNorm and gate/up of the shared expert; its Down
//                 runs in the routed CTAs as expert slot 0.
//   CTAs 144..147 sentinel helpers (initialize_sentinels): refill the readiness sentinels of the
//                 score and mid outputs, then arrive on the kSyncInit word.
// Thread kConsumerThreads (q1n_produce) streams the weight pages into mbarrier-guarded
// shared-memory slots; the eight consumer warps compute.  Cross-CTA values are polled in place: a
// score or mid word is ready once it no longer holds its pending encoding (kScorePending /
// kMidPending).  Each CTA's launch_ctr entry counts its launches; the helpers' refill for launch
// `generation` is complete when the kSyncInit word reaches kSentinelHelperCtas * generation.

namespace flashinfer::glm5_tp8::moe_q1 {

constexpr uint32_t kScorePending = 0xffffffffu;
constexpr uint32_t kMidPending = 0xffffu;

// Reserve one NaN encoding for readiness; computed NaNs remain NaNs, but cannot
// collide with it. Finite values, infinities and signed zero retain their bits.
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

// F1 publishes the two logits with a matching scalar b64 store.
__device__ __forceinline__ void load_relaxed_gpu_b64_words(const void* address, uint32_t& lo,
                                                           uint32_t& hi) {
  uint64_t pair;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(pair) : "l"(address) : "memory");
  lo = static_cast<uint32_t>(pair);
  hi = static_cast<uint32_t>(pair >> 32);
}

// Mid values are published in independent b32 words. Check both BF16 components
// of every word, without imposing a mixed-width b64 polling operation.
__device__ __forceinline__ void load_relaxed_gpu_mid_words(const void* address, uint32_t& lo,
                                                           uint32_t& hi) {
  asm volatile("ld.relaxed.gpu.global.v2.b32 {%0, %1}, [%2];"
               : "=r"(lo), "=r"(hi)
               : "l"(address)
               : "memory");
}

__device__ __forceinline__ void st_relaxed_gpu_u32(void* address, uint32_t value) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
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
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait_parity(bar, parity)) {
  }
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
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

__device__ __forceinline__ void sts_u32(uint32_t address, uint32_t value) {
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(address), "r"(value) : "memory");
}
__device__ __forceinline__ void sts_f32(uint32_t address, float value) {
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(address), "f"(value) : "memory");
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

// Four helpers refill only the live publication fields. Work CTAs overlap
// their independent input/math work and acquire completion before publishing.
__device__ __forceinline__ void initialize_sentinels(const LaunchArgs& args, int helper) {
  const CtxTable& rows = *reinterpret_cast<const CtxTable*>(args.context);
  const int tid = threadIdx.x;
  uint32_t generation = 0;
  if (tid == 0) {
    generation = args.launch_ctr[blockIdx.x] + 1u;
    args.launch_ctr[blockIdx.x] = generation;
  }
  if (tid < 32) {
    uint8_t* score = reinterpret_cast<uint8_t*>(rows.r1.scores_out) +
                     size_t(helper * 32 + tid) * kScoresSlotBytes;
    const uint64_t pending = uint64_t(kScorePending) | (uint64_t(kScorePending) << 32);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(score), "l"(pending) : "memory");
  }
  for (int i = helper * int(blockDim.x) + tid; i < 288; i += 4 * int(blockDim.x)) {
    const uint32_t pending = kMidPending | (kMidPending << 16);
    st_relaxed_gpu_v4u32(reinterpret_cast<uint8_t*>(rows.r2.hidden_mid_out) + i * 16, pending,
                         pending, pending, pending);
  }
  __syncthreads();
  if (tid == 0) word_arrive_release_only(args.sync_words + kSyncInit / 4, 1u);
}
__device__ __forceinline__ void wait_sentinels(const LaunchArgs& args, uint32_t generation) {
  const uint32_t target = 4u * generation;
  while (static_cast<int32_t>(ld_relaxed_gpu_u32(args.sync_words + kSyncInit / 4) - target) < 0) {
  }
  fence_acquire_gpu();
}

__device__ __forceinline__ void store_score_pair(void* address, float a, float b) {
  const uint64_t pair = static_cast<uint64_t>(score_payload_bits(a)) |
                        (static_cast<uint64_t>(score_payload_bits(b)) << 32);
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(address), "l"(pair) : "memory");
}

struct Q1Quant128 {
  uint32_t q;
  float scale;
};

__device__ __forceinline__ void q1_mma_fp8(HmmaAcc& acc, const uint32_t (&a)[4],
                                           const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(acc.v[0]), "+f"(acc.v[1]), "+f"(acc.v[2]), "+f"(acc.v[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

namespace q1n {
constexpr uint32_t kRows = 0x0000;
constexpr uint32_t kBars = 0x0280;
constexpr uint32_t kRed = 0x0340;
constexpr uint32_t kSelIdx = 0x0440;
constexpr uint32_t kSelProb = 0x0460;
constexpr uint32_t kSelMine = 0x0480;
constexpr uint32_t kXqScale = 0x0500;
constexpr uint32_t kMqScale = 0x05c0;
constexpr uint32_t kOwn = 0x0610;
constexpr uint32_t kRx = 0x0670;
constexpr uint32_t kUgPart = 0x0910;
constexpr uint32_t kDownPart = 0x0d10;
constexpr uint32_t kMq = 0x1310;
constexpr uint32_t kXq = 0x1c10;
constexpr uint32_t kPage0 = 0x3480;
constexpr uint32_t kPageStride = 0x8040;
constexpr uint32_t kX = kPage0 + 5 * kPageStride;
constexpr uint32_t kDynBytes = kPage0 + 6 * kPageStride;
enum : uint32_t {
  kBarFull0 = 0,
  kBarCons0 = 6,
  kBarDown0 = 11,
  kBarSelMine = 20,
  kBarSelAll = 21,
  kBarXFree = 22,
  kBarP1 = 23,
  kNumBars = 24,
};
static_assert(kBars + kNumBars * 8 <= kRed && kRed + 64 * 4 <= kSelIdx &&
                  kSelProb + 32 <= kSelMine && kSelMine + 4 <= kXqScale &&
                  kXqScale + 48 * 4 <= kMqScale && kMqScale + 18 * 4 <= kOwn && kOwn + 96 <= kRx &&
                  kRx + 7 * 96 <= kUgPart && kUgPart + 0x400 <= kDownPart &&
                  kDownPart + 8 * 48 * 4 <= kMq && kMq + 18 * 128 <= kXq &&
                  kXq + 48 * 128 <= kPage0,
              "q1n smem layout");
static_assert(kPage0 % 128 == 0 && kPageStride % 16 == 0 && kX % 16 == 0,
              "bulk-copy destinations are 16 B aligned");
static_assert(kGateUpPageCopyBytes == kPageStride &&
                  2 * (kDownSliceBytes + kDownScaleSliceBytes) <= kPageStride,
              "slot sizes");
static_assert(kX + kXRowBytes <= kDynBytes, "x_norm fits page slot 5");
static_assert(kDynBytes <= kQ1DynamicSmem,
              "the launch configuration must provide the kernel's dynamic shared memory");
constexpr uint32_t kDownSlotBytes = kDownSliceBytes + kDownScaleSliceBytes;
}  // namespace q1n

__device__ __forceinline__ void q1n_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx)
               : "memory");
}
__device__ __forceinline__ void q1n_bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                             uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void q1n_prefetch_l2(const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint64_t q1n_policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void q1n_fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ uint32_t q1n_ldg_u32(const void* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}
__device__ __forceinline__ float q1n_ldg_f32(const void* p) {
  return __ldg(reinterpret_cast<const float*>(p));
}
__device__ __forceinline__ uint32_t q1n_ldg_u16(const void* p) {
  return __ldg(reinterpret_cast<const unsigned short*>(p));
}

__device__ __forceinline__ Q1Quant128 q1n_quant128_regs(uint32_t x01, uint32_t x23) {
  const float x0 = bf16_lo(x01), x1 = bf16_hi(x01);
  const float x2 = bf16_lo(x23), x3 = bf16_hi(x23);
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

__device__ __forceinline__ uint32_t q1n_perm_lane(int lane) {
  const int c = lane & 3;
  return static_cast<uint32_t>((lane & ~3) + ((c & 1) << 1) + (c >> 1));
}
__device__ __forceinline__ void q1n_frag_pack(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                                              uint32_t (&a)[4]) {
  a[0] = (x0 & 0xffffu) | (x1 << 16);
  a[1] = (x0 >> 16) | (x1 & 0xffff0000u);
  a[2] = (x2 & 0xffffu) | (x3 << 16);
  a[3] = (x2 >> 16) | (x3 & 0xffff0000u);
}

__device__ __forceinline__ void q1n_gate_up_frag_regs(int lane, uint32_t v0, uint32_t v1,
                                                      uint32_t w0, uint32_t w1, uint32_t (&a)[4]) {
  const bool hi = (lane & 2) != 0;
  const uint32_t sv = __shfl_xor_sync(0xffffffffu, hi ? v0 : v1, 2);
  const uint32_t sw = __shfl_xor_sync(0xffffffffu, hi ? w0 : w1, 2);
  q1n_frag_pack(hi ? sv : v0, hi ? v1 : sv, hi ? sw : w0, hi ? w1 : sw, a);
}

__device__ __forceinline__ void q1n_down_frag_regs(int lane, uint32_t w0, uint32_t w1, uint32_t w2,
                                                   uint32_t w3, uint32_t (&a)[4]) {
  const bool hi = (lane & 2) != 0;
  const uint32_t s1 = __shfl_xor_sync(0xffffffffu, hi ? w0 : w1, 2);
  const uint32_t s2 = __shfl_xor_sync(0xffffffffu, hi ? w2 : w3, 2);
  q1n_frag_pack(hi ? s1 : w0, hi ? w1 : s1, hi ? s2 : w2, hi ? w3 : s2, a);
}

__device__ __forceinline__ void q1n_issue_page(uint32_t sb, int p, const uint8_t* src,
                                               uint64_t pol) {
  const uint32_t bar = sb + q1n::kBars + (q1n::kBarFull0 + p) * 8;
  q1n_arrive_expect_tx(bar, kGateUpPageCopyBytes);
  q1n_bulk_g2s(sb + q1n::kPage0 + p * q1n::kPageStride, src, kGateUpPageCopyBytes, bar, pol);
}

__device__ __forceinline__ void q1n_produce(const CtxTable& R, int cta, uint32_t sb) {
  const uint64_t pol = q1n_policy_evict_first();
  if (cta < kRoutedCtas) {
    const int cc = cta;
    const uint8_t* down_w = R.r4.down_w;
    const uint8_t* down_sc = reinterpret_cast<const uint8_t*>(R.r4.down_scale);

    mbar_wait_parity(sb + q1n::kBars + q1n::kBarP1 * 8, 0u);
    q1n_prefetch_l2(down_w + size_t(cc) * kDownSliceBytes, kDownSliceBytes);
    q1n_prefetch_l2(down_sc + size_t(cc) * kDownScaleSliceBytes, kDownScaleSliceBytes);

    mbar_wait_parity(sb + q1n::kBars + q1n::kBarSelMine * 8, 0u);
    const int E = static_cast<int>(lds_u32(sb + q1n::kSelMine));
    const uint8_t* rgb = R.r2.gate_up_w + size_t(E + 1) * kGateUpExpertStride +
                         size_t(cc % 16) * kGateUpRowGroupStride;
#pragma unroll
    for (int p = 0; p < kGateUpPages - 1; ++p)
      q1n_issue_page(sb, p, rgb + size_t(p) * kGateUpPageStride, pol);
    mbar_wait_parity(sb + q1n::kBars + q1n::kBarXFree * 8, 0u);
    q1n_fence_proxy_async();
    q1n_issue_page(sb, kGateUpPages - 1, rgb + size_t(kGateUpPages - 1) * kGateUpPageStride, pol);

    mbar_wait_parity(sb + q1n::kBars + q1n::kBarSelAll * 8, 0u);
    int ew[kExpertSlots];
    ew[0] = 0;
#pragma unroll
    for (int e = 1; e < kExpertSlots; ++e)
      ew[e] = static_cast<int>(lds_u32(sb + q1n::kSelIdx + (e - 1) * 4)) + 1;
#pragma unroll
    for (int e = 1; e < kExpertSlots; ++e) {
      q1n_prefetch_l2(down_w + size_t(ew[e]) * kDownExpertStride + size_t(cc) * kDownSliceBytes,
                      kDownSliceBytes);
      q1n_prefetch_l2(
          down_sc + size_t(ew[e]) * kDownScaleExpertStride + size_t(cc) * kDownScaleSliceBytes,
          kDownScaleSliceBytes);
    }
#pragma unroll
    for (int p = 0; p < 5; ++p) {
      mbar_wait_parity(sb + q1n::kBars + (q1n::kBarCons0 + p) * 8, 0u);
      q1n_fence_proxy_async();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int e = 2 * p + h;
        if (e < kExpertSlots) {
          const uint32_t bar = sb + q1n::kBars + (q1n::kBarDown0 + e) * 8;
          const uint32_t dst = sb + q1n::kPage0 + p * q1n::kPageStride + h * q1n::kDownSlotBytes;
          q1n_arrive_expect_tx(bar, q1n::kDownSlotBytes);
          q1n_bulk_g2s(dst,
                       down_w + size_t(ew[e]) * kDownExpertStride + size_t(cc) * kDownSliceBytes,
                       kDownSliceBytes, bar, pol);
          q1n_bulk_g2s(
              dst + kDownSliceBytes,
              down_sc + size_t(ew[e]) * kDownScaleExpertStride + size_t(cc) * kDownScaleSliceBytes,
              kDownScaleSliceBytes, bar, pol);
        }
      }
    }
  } else {
    const int rgS = cta - kSharedBase;
    const uint8_t* rgb = R.r3.gate_up_w_shared + size_t(rgS) * kGateUpRowGroupStride;
    mbar_wait_parity(sb + q1n::kBars + q1n::kBarP1 * 8, 0u);
#pragma unroll
    for (int p = 0; p < kGateUpPages - 1; ++p)
      q1n_issue_page(sb, p, rgb + size_t(p) * kGateUpPageStride, pol);
    mbar_wait_parity(sb + q1n::kBars + q1n::kBarXFree * 8, 0u);
    q1n_fence_proxy_async();
    q1n_issue_page(sb, kGateUpPages - 1, rgb + size_t(kGateUpPages - 1) * kGateUpPageStride, pol);
  }
}

__device__ __forceinline__ void q1n_rmsnorm_regs(uint32_t sb, int warp, int lane,
                                                 const uint32_t (&xw)[kWordsPerThread],
                                                 const float2 (&gm)[kWordsPerThread],
                                                 uint32_t (&xn)[kWordsPerThread]) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const float hi = bf16_hi(xw[j]);
    const float lo = bf16_lo(xw[j]);
    float p = hi * hi;
    p = fmaf(lo, lo, p);
    acc = acc + p;
  }
  const float wsum = warp_sum_down(acc);
  if (lane == 0) sts_f32(sb + q1n::kRed + warp * 4, wsum);
  consumer_sync();
  float p8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) p8[i] = lds_f32(sb + q1n::kRed + i * 4);
  const float sumsq = fold8(p8);
  const float mean = __fdiv_rn(sumsq, kRmsDenominator);
  const float rstd = rsqrtf(mean + kRmsEps);
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j) {
    const float y0 = bf16_lo(xw[j]) * (rstd * gm[j].x);
    const float y1 = bf16_hi(xw[j]) * (rstd * gm[j].y);
    xn[j] = pack_bf16_pair(y0, y1);
  }
}

__device__ __forceinline__ void q1n_quant_xnorm(uint32_t sb, int lane, int first, int stride) {
#pragma unroll 2
  for (int b = first; b < 48; b += stride) {
    const uint32_t src = sb + q1n::kX + b * 256 + lane * 8;
    const Q1Quant128 q = q1n_quant128_regs(lds_u32(src), lds_u32(src + 4));
    sts_u32(sb + q1n::kXq + b * 128 + lane * 4, q.q);
    if (lane == 0) sts_f32(sb + q1n::kXqScale + b * 4, q.scale);
  }
}

template <bool kRelease>
__device__ __forceinline__ void q1n_gate_up(uint32_t sb, int warp, int lane, uint16_t* hm_slot,
                                            int rg) {
  float acc_g0 = 0.f, acc_g8 = 0.f, acc_u0 = 0.f, acc_u8 = 0.f;
#pragma unroll 1
  for (int page = 0; page < kGateUpPages; ++page) {
    const uint32_t pg = sb + q1n::kPage0 + page * q1n::kPageStride;
    const int blk = page * kConsumerWarps + warp;
    const uint32_t xq = sb + q1n::kXq + blk * 128 + (lane & 3) * 4;
    mbar_wait_parity(sb + q1n::kBars + (q1n::kBarFull0 + page) * 8, 0u);

    uint32_t v[4][2][2], w[4][2][2];
    const uint32_t a_base = pg + warp * kGateUpWarpKBytes + q1n_perm_lane(lane) * 8;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int gu = 0; gu < 2; ++gu) {
        const uint32_t t = a_base + k * 2 * kGateUpKStepBytes + gu * kGateUpUpHalf;
        lds_u32x2(t, v[k][gu][0], v[k][gu][1]);
        lds_u32x2(t + kGateUpKStepBytes, w[k][gu][0], w[k][gu][1]);
      }
    }
    uint32_t bfr[4][2];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      bfr[k][0] = 0u;
      bfr[k][1] = 0u;
      if (lane < 4) {
        bfr[k][0] = lds_u32(xq + k * 32);
        bfr[k][1] = lds_u32(xq + k * 32 + 16);
      }
    }
    HmmaAcc dg = {{0.f, 0.f, 0.f, 0.f}}, du = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      uint32_t ag[4], au[4];
      q1n_gate_up_frag_regs(lane, v[k][0][0], v[k][0][1], w[k][0][0], w[k][0][1], ag);
      q1n_gate_up_frag_regs(lane, v[k][1][0], v[k][1][1], w[k][1][0], w[k][1][1], au);
      const uint32_t b[2] = {bfr[k][0], bfr[k][1]};
      q1_mma_fp8(dg, ag, b);
      q1_mma_fp8(du, au, b);
    }
    const float xs = lds_f32(sb + q1n::kXqScale + blk * 4);
    const float sg = __fmul_rn(lds_f32(pg + kGateUpScaleGate + warp * 4), xs);
    const float su = __fmul_rn(lds_f32(pg + kGateUpScaleUp + warp * 4), xs);
    if (kRelease && page < kGateUpPages - 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(sb + q1n::kBars + (q1n::kBarCons0 + page) * 8);
    }
    acc_g0 = fmaf(sg, dg.v[0], acc_g0);
    acc_g8 = fmaf(sg, dg.v[2], acc_g8);
    acc_u0 = fmaf(su, du.v[0], acc_u0);
    acc_u8 = fmaf(su, du.v[2], acc_u8);
  }
  const int g = lane >> 2;
  if ((lane & 3) == 0) {
    sts_f32(sb + q1n::kUgPart + 0x000 + (g * 8 + warp) * 4, acc_g0);
    sts_f32(sb + q1n::kUgPart + 0x100 + (g * 8 + warp) * 4, acc_g8);
    sts_f32(sb + q1n::kUgPart + 0x200 + (g * 8 + warp) * 4, acc_u0);
    sts_f32(sb + q1n::kUgPart + 0x300 + (g * 8 + warp) * 4, acc_u8);
  }
  consumer_sync();
  if (warp == 0) {
    const int kind = lane >> 4;
    const int row = lane & 15;
    const uint32_t base = sb + q1n::kUgPart + ((kind * 2 + (row >> 3)) * 64 + (row & 7) * 8) * 4;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) sum = sum + lds_f32(base + w * 4);
    const float up = __shfl_sync(0xffffffffu, sum, (lane & 15) + 16);
    const uint32_t h = mid_payload_bits(f32_to_bf16_bits(up * silu_approx(sum)));
    const uint32_t h_next = __shfl_down_sync(0xffffffffu, h, 1);
    if (lane < kRowsPerGroup && (lane & 1) == 0)
      st_relaxed_gpu_u32(hm_slot + rg * kRowsPerGroup + lane, h | (h_next << 16));
  }
}

__device__ __forceinline__ void q1n_ce(uint64_t& a, uint64_t& b) {
  const uint64_t hi = a > b ? a : b;
  const uint64_t lo = a > b ? b : a;
  a = hi;
  b = lo;
}
__device__ __forceinline__ void q1n_sort8_desc(uint64_t (&v)[8]) {
  q1n_ce(v[0], v[2]);
  q1n_ce(v[1], v[3]);
  q1n_ce(v[4], v[6]);
  q1n_ce(v[5], v[7]);
  q1n_ce(v[0], v[4]);
  q1n_ce(v[1], v[5]);
  q1n_ce(v[2], v[6]);
  q1n_ce(v[3], v[7]);
  q1n_ce(v[0], v[1]);
  q1n_ce(v[2], v[3]);
  q1n_ce(v[4], v[5]);
  q1n_ce(v[6], v[7]);
  q1n_ce(v[2], v[4]);
  q1n_ce(v[3], v[5]);
  q1n_ce(v[1], v[4]);
  q1n_ce(v[3], v[6]);
  q1n_ce(v[1], v[2]);
  q1n_ce(v[3], v[4]);
  q1n_ce(v[5], v[6]);
}

__device__ __forceinline__ void q1n_routed(const LaunchArgs& args, const CtxTable& R, int cc,
                                           uint32_t sb, int tid, int warp, int lane) {
  uint32_t xw[kWordsPerThread], rw0[kWordsPerThread], rw1[kWordsPerThread];
  float2 gm[kWordsPerThread];
  {
    const uint8_t* xg = reinterpret_cast<const uint8_t*>(R.r0.x);
    const uint8_t* gg = reinterpret_cast<const uint8_t*>(R.r0.gamma);
    const uint8_t* r0 = reinterpret_cast<const uint8_t*>(R.r1.router_w + size_t(cc * 2) * kDim);
    const uint8_t* r1 = r0 + kXRowBytes;
#pragma unroll
    for (int j = 0; j < kWordsPerThread; ++j) {
      const int word = tid + kConsumerThreads * j;
      xw[j] = q1n_ldg_u32(xg + word * 4);
      gm[j] = make_float2(q1n_ldg_f32(gg + word * 8), q1n_ldg_f32(gg + word * 8 + 4));
      rw0[j] = q1n_ldg_u32(r0 + word * 4);
      rw1[j] = q1n_ldg_u32(r1 + word * 4);
    }
  }
  float bias_r[8];
  uint32_t xres0 = 0, xres1 = 0;
  if (warp == 0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) bias_r[j] = q1n_ldg_f32(R.r2.bias + lane * 8 + j);
  }
  if (warp == 0 && lane < kDownRowsPerCta / 2) {
    const uint16_t* xr = R.r4.x_residual + size_t(cc) * kDownRowsPerCta + 2 * lane;
    xres0 = q1n_ldg_u16(xr);
    xres1 = q1n_ldg_u16(xr + 1);
  }

  uint32_t xn[kWordsPerThread];
  q1n_rmsnorm_regs(sb, warp, lane, xw, gm, xn);
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j)
    sts_u32(sb + q1n::kX + (tid + kConsumerThreads * j) * 4, xn[j]);

  {
    float even0 = 0.0f, odd0 = 0.0f, even1 = 0.0f, odd1 = 0.0f;
#pragma unroll
    for (int j = 0; j < kWordsPerThread; ++j) {
      even0 = fmaf(bf16_lo(rw0[j]), bf16_lo(xn[j]), even0);
      odd0 = fmaf(bf16_hi(rw0[j]), bf16_hi(xn[j]), odd0);
    }
#pragma unroll
    for (int j = 0; j < kWordsPerThread; ++j) {
      even1 = fmaf(bf16_lo(rw1[j]), bf16_lo(xn[j]), even1);
      odd1 = fmaf(bf16_hi(rw1[j]), bf16_hi(xn[j]), odd1);
    }
    const float w0 = warp_sum_down(even0 + odd0);
    const float w1 = warp_sum_down(even1 + odd1);
    if (lane == 0) {
      sts_f32(sb + q1n::kRed + (8 + 0 * 8 + warp) * 4, w0);
      sts_f32(sb + q1n::kRed + (8 + 1 * 8 + warp) * 4, w1);
    }
  }
  consumer_sync();

  int myidx = 0;
  float myprob = 0.0f;
  if (warp == 0) {
    if (lane == 0) mbar_arrive(sb + q1n::kBars + q1n::kBarP1 * 8);
    if (lane == 0) wait_sentinels(args, static_cast<uint32_t>(R.r4.epoch));
    __syncwarp();

    float logit = 0.0f;
    if (lane < 2) {
      float p8[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) p8[w] = lds_f32(sb + q1n::kRed + (8 + lane * 8 + w) * 4);
      logit = fold8(p8);
    }
    const float logit1 = __shfl_sync(0xffffffffu, logit, 1);
    if (lane == 0)
      store_score_pair(reinterpret_cast<uint8_t*>(R.r1.scores_out) + size_t(cc) * kScoresSlotBytes,
                       logit, logit1);

    const uint8_t* sin =
        reinterpret_cast<const uint8_t*>(R.r2.scores_in) + size_t(lane) * 4 * kScoresSlotBytes;
    uint32_t lo[4], hi[4];
    for (;;) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        load_relaxed_gpu_b64_words(sin + i * kScoresSlotBytes, lo[i], hi[i]);
      bool ok = true;
#pragma unroll
      for (int i = 0; i < 4; ++i) ok = ok && lo[i] != kScorePending && hi[i] != kScorePending;
      if (__all_sync(0xffffffffu, ok)) break;
      spin_backoff();
    }

    float s[8];
    uint64_t kv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s[2 * i] = sigmoid_approx(__uint_as_float(lo[i]));
      s[2 * i + 1] = sigmoid_approx(__uint_as_float(hi[i]));
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      kv[j] = (uint64_t(f32_monotonic_key(s[j] + bias_r[j])) << 32) | uint64_t(7 - j);
    q1n_sort8_desc(kv);

    const int kmine = cc / 16;
    float mine = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopk; ++k) {
      const uint32_t hk = static_cast<uint32_t>(kv[0] >> 32);
      const uint32_t wmax = __reduce_max_sync(0xffffffffu, hk);
      const uint32_t bal = __ballot_sync(0xffffffffu, hk == wmax);
      const int wl = __ffs(static_cast<int>(bal)) - 1;
      const int hj = 7 - static_cast<int>(static_cast<uint32_t>(kv[0]) & 7u);
      float sv = s[0];
#pragma unroll
      for (int j = 1; j < 8; ++j) sv = (j == hj) ? s[j] : sv;
      const float s_sel = __shfl_sync(0xffffffffu, sv, wl);
      const int j_sel = __shfl_sync(0xffffffffu, hj, wl);
      const int idx = wl * 8 + j_sel;
      if (lane == wl) {
#pragma unroll
        for (int j = 0; j < 7; ++j) kv[j] = kv[j + 1];
        kv[7] = 0ull;
      }
      if (lane == k) {
        mine = s_sel;
        myidx = idx;
      }
      if (k == kmine && lane == 0) {
        sts_u32(sb + q1n::kSelMine, static_cast<uint32_t>(idx));
        mbar_arrive(sb + q1n::kBars + q1n::kBarSelMine * 8);
      }
    }
    float v = mine;
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    const float sum = __shfl_sync(0xffffffffu, v, 0);
    const float scale = __fdiv_rn(kRouteScale, sum);
    myprob = mine * scale;
    if (lane < kTopk) {
      sts_u32(sb + q1n::kSelIdx + lane * 4, static_cast<uint32_t>(myidx));
      sts_f32(sb + q1n::kSelProb + lane * 4, myprob);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(sb + q1n::kBars + q1n::kBarSelAll * 8);

    if (cc == 0 && lane < kTopk) {
      R.r2.sel_probs[lane] = myprob;
      R.r2.sel_indices[lane] = myidx;
    }
  } else {
    q1n_quant_xnorm(sb, lane, warp - 1, kConsumerWarps - 1);
    __syncwarp();
    if (lane == 0) mbar_arrive(sb + q1n::kBars + q1n::kBarXFree * 8);
  }
  consumer_sync();

  {
    const int k = cc / 16;
    uint16_t* hm_slot = R.r2.hidden_mid_out + size_t(1 + k) * kInterPerRank;
    q1n_gate_up<true>(sb, warp, lane, hm_slot, cc % 16);
  }

  {
    const uint8_t* hm = reinterpret_cast<const uint8_t*>(R.r4.hidden_mid_in);
    const int nb = (warp < 2) ? 3 : 2;
    uint32_t w01[3], w23[3];
    for (;;) {
#pragma unroll
      for (int i = 0; i < 3; ++i)
        if (i < nb)
          load_relaxed_gpu_mid_words(hm + (warp + 8 * i) * 256 + lane * 8, w01[i], w23[i]);
      bool ok = true;
#pragma unroll
      for (int i = 0; i < 3; ++i)
        if (i < nb) ok = ok && !mid_word_pending(w01[i]) && !mid_word_pending(w23[i]);
      if (__all_sync(0xffffffffu, ok)) break;
      spin_backoff();
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (i < nb) {
        const int b2 = warp + 8 * i;
        const Q1Quant128 q = q1n_quant128_regs(w01[i], w23[i]);
        sts_u32(sb + q1n::kMq + b2 * 128 + lane * 4, q.q);
        if (lane == 0) sts_f32(sb + q1n::kMqScale + b2 * 4, q.scale);
      }
    }
  }
  consumer_sync();

  {
    const int khalf = warp >> 2;
    const int k32 = warp & 3;
    const int g = lane >> 2;
    float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const uint32_t a_off = warp * kDownWarpKBytes + q1n_perm_lane(lane) * 16;
#pragma unroll 1
    for (int e0 = 0; e0 < kExpertSlots; e0 += 3) {
      uint32_t W[3][3][4];
      float sc[3][6], prob[3], msc[3];
      uint32_t bq[3][2];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const int e = e0 + i;
        const int b2 = e * 2 + khalf;
        prob[i] = (e == 0) ? 1.0f : lds_f32(sb + q1n::kSelProb + (e - 1) * 4);
        msc[i] = lds_f32(sb + q1n::kMqScale + b2 * 4);
        bq[i][0] = 0u;
        bq[i][1] = 0u;
        if (lane < 4) {
          bq[i][0] = lds_u32(sb + q1n::kMq + b2 * 128 + k32 * 32 + lane * 4);
          bq[i][1] = lds_u32(sb + q1n::kMq + b2 * 128 + k32 * 32 + 16 + lane * 4);
        }
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const int e = e0 + i;
        const uint32_t sl =
            sb + q1n::kPage0 + (e >> 1) * q1n::kPageStride + (e & 1) * q1n::kDownSlotBytes;
        mbar_wait_parity(sb + q1n::kBars + (q1n::kBarDown0 + e) * 8, 0u);
#pragma unroll
        for (int tile = 0; tile < 3; ++tile)
          lds_u32x4(sl + tile * kDownTileBytes + a_off, W[i][tile][0], W[i][tile][1], W[i][tile][2],
                    W[i][tile][3]);
#pragma unroll
        for (int q = 0; q < 6; ++q) sc[i][q] = lds_f32(sl + kDownSliceBytes + (q * 2 + khalf) * 4);
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int tile = 0; tile < 3; ++tile) {
          uint32_t a[4];
          q1n_down_frag_regs(lane, W[i][tile][0], W[i][tile][1], W[i][tile][2], W[i][tile][3], a);
          HmmaAcc D = {{0.f, 0.f, 0.f, 0.f}};
          q1_mma_fp8(D, a, bq[i]);
          const float s0 = __fmul_rn(__fmul_rn(sc[i][tile * 2], prob[i]), msc[i]);
          const float s8 = __fmul_rn(__fmul_rn(sc[i][tile * 2 + 1], prob[i]), msc[i]);
          acc[tile * 2] = fmaf(s0, D.v[0], acc[tile * 2]);
          acc[tile * 2 + 1] = fmaf(s8, D.v[2], acc[tile * 2 + 1]);
        }
      }
    }
    if ((lane & 3) == 0) {
#pragma unroll
      for (int tile = 0; tile < 3; ++tile) {
        sts_f32(sb + q1n::kDownPart + (warp * kDownRowsPerCta + tile * 16 + g) * 4, acc[tile * 2]);
        sts_f32(sb + q1n::kDownPart + (warp * kDownRowsPerCta + tile * 16 + g + 8) * 4,
                acc[tile * 2 + 1]);
      }
    }
  }
  consumer_sync();
  if (warp != 0) return;

  uint32_t ownw = 0;
  if (lane < kDownRowsPerCta / 2) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) {
      s0 = s0 + lds_f32(sb + q1n::kDownPart + (w * kDownRowsPerCta + 2 * lane) * 4);
      s1 = s1 + lds_f32(sb + q1n::kDownPart + (w * kDownRowsPerCta + 2 * lane + 1) * 4);
    }
    ownw = pack_bf16_pair(s0, s1);
  }

  const int line = lane & 7;
  uint16_t* out = R.r4.hidden_out + size_t(cc) * kDownRowsPerCta;
  float red0, red1;
  if (args.peers[0] != nullptr) {
    const uint64_t epoch = R.r4.epoch;
    const uint32_t epoch_lo = static_cast<uint32_t>(epoch);
    const long long e64 = static_cast<long long>(epoch);
    const long long parity = e64 - 2 * (e64 / 2);
    const int rank = args.rank;
    const ptrdiff_t rec_off = size_t(cc) * kInboxChunkStride +
                              static_cast<ptrdiff_t>(parity * kInboxParityStride) +
                              line * kInboxLineStride;

    uint32_t d[4];
#pragma unroll
    for (int i = 0; i < 3; ++i) d[i] = __shfl_sync(0xffffffffu, ownw, 3 * line + i);
    d[3] = epoch_lo;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int pi = 4 * r + (lane >> 3);
      if (pi < kWorld - 1) {
        const int dst = pi + (pi >= rank ? 1 : 0);
        const int slot = dst > rank ? rank : rank - 1;
        st_relaxed_sys_b128(args.peers[dst] + size_t(slot) * kInboxSlotStride + rec_off, d[0], d[1],
                            d[2], d[3]);
      }
    }

    const uint8_t* inbox = args.peers[rank] + rec_off;
    uint64_t lo[2], hi[2];
    for (;;) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        lo[r] = hi[r] = 0;
        const int pi = 4 * r + (lane >> 3);
        if (pi < kWorld - 1)
          ld_relaxed_sys_b128(inbox + size_t(pi) * kInboxSlotStride, lo[r], hi[r]);
      }
      bool ok = true;
#pragma unroll
      for (int r = 0; r < 2; ++r)
        ok = ok && (static_cast<uint32_t>(hi[r] >> 32) == epoch_lo ||
                    (4 * r + (lane >> 3)) >= kWorld - 1);
      if (__all_sync(0xffffffffu, ok)) break;
      spin_backoff();
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int pi = 4 * r + (lane >> 3);
      if (pi < kWorld - 1) {
        const uint32_t rx = sb + q1n::kRx + pi * 96 + line * 12;
        sts_u32(rx, static_cast<uint32_t>(lo[r]));
        sts_u32(rx + 4, static_cast<uint32_t>(lo[r] >> 32));
        sts_u32(rx + 8, static_cast<uint32_t>(hi[r]));
      }
    }
    __syncwarp();

    red0 = 0.0f;
    red1 = 0.0f;
    if (lane < kDownRowsPerCta / 2) {
#pragma unroll
      for (int r = 0; r < kWorld; ++r) {
        const uint32_t v =
            (r == rank) ? ownw : lds_u32(sb + q1n::kRx + (r < rank ? r : r - 1) * 96 + lane * 4);
        red0 = red0 + bf16_lo(v);
        red1 = red1 + bf16_hi(v);
      }
    }
  } else {
    red0 = bf16_lo(ownw);
    red1 = bf16_hi(ownw);
  }

  if (lane < kDownRowsPerCta / 2) {
    out[2 * lane] = f32_to_bf16_bits(red0 + bf16_bits_to_f32(xres0));
    out[2 * lane + 1] = f32_to_bf16_bits(red1 + bf16_bits_to_f32(xres1));
  }
}

__device__ __forceinline__ void q1n_shared(const LaunchArgs& args, const CtxTable& R, int rgS,
                                           uint32_t sb, int tid, int warp, int lane) {
  uint32_t xw[kWordsPerThread];
  float2 gm[kWordsPerThread];
  {
    const uint8_t* xg = reinterpret_cast<const uint8_t*>(R.r3.x_shared);
    const uint8_t* gg = reinterpret_cast<const uint8_t*>(R.r0.gamma);
#pragma unroll
    for (int j = 0; j < kWordsPerThread; ++j) {
      const int word = tid + kConsumerThreads * j;
      xw[j] = q1n_ldg_u32(xg + word * 4);
      gm[j] = make_float2(q1n_ldg_f32(gg + word * 8), q1n_ldg_f32(gg + word * 8 + 4));
    }
  }
  uint32_t xn[kWordsPerThread];
  q1n_rmsnorm_regs(sb, warp, lane, xw, gm, xn);
#pragma unroll
  for (int j = 0; j < kWordsPerThread; ++j)
    sts_u32(sb + q1n::kX + (tid + kConsumerThreads * j) * 4, xn[j]);
  if (rgS == 0) {
    uint32_t* o = reinterpret_cast<uint32_t*>(R.r3.x_norm_buf);
#pragma unroll
    for (int j = 0; j < kWordsPerThread; ++j) o[tid + kConsumerThreads * j] = xn[j];
    __threadfence();
  }
  consumer_sync();
  if (tid == 0) {
    mbar_arrive(sb + q1n::kBars + q1n::kBarP1 * 8);
  }
  q1n_quant_xnorm(sb, lane, warp, kConsumerWarps);
  __syncwarp();
  if (lane == 0) mbar_arrive(sb + q1n::kBars + q1n::kBarXFree * 8);
  consumer_sync();
  if (tid == 0) wait_sentinels(args, static_cast<uint32_t>(R.r4.epoch));
  consumer_sync();
  q1n_gate_up<false>(sb, warp, lane, R.r3.hidden_mid_shared_out, rgS);
}

__device__ __forceinline__ void q1n_kernel(const LaunchArgs& args, uint8_t* dyn) {
  const int cta = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const uint32_t sb = smem_u32(dyn);
  {
    const int n_u64 =
        static_cast<int>(args.n_rows < uint32_t(kCtxRows) ? args.n_rows : uint32_t(kCtxRows)) *
        (kCtxRowBytes / 8);
    if (tid < kCtxRows * (kCtxRowBytes / 8)) {
      const uint64_t* src = reinterpret_cast<const uint64_t*>(args.context);
      uint64_t value;
      if (tid == 68) {
        const uint32_t generation = args.launch_ctr[cta] + 1u;
        args.launch_ctr[cta] = generation;
        value = generation;
      } else {
        value = tid < n_u64 ? src[tid] : 0ull;
      }
      reinterpret_cast<uint64_t*>(dyn + q1n::kRows)[tid] = value;
    }
    if (tid == kCtxRows * (kCtxRowBytes / 8)) {
      uint32_t dsz;
      asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
      if (dsz < q1n::kDynBytes) __trap();
#pragma unroll
      for (int i = 0; i < 6; ++i) mbar_init(sb + q1n::kBars + (q1n::kBarFull0 + i) * 8, 1);
#pragma unroll
      for (int i = 0; i < 5; ++i)
        mbar_init(sb + q1n::kBars + (q1n::kBarCons0 + i) * 8, kConsumerWarps);
#pragma unroll
      for (int i = 0; i < kExpertSlots; ++i)
        mbar_init(sb + q1n::kBars + (q1n::kBarDown0 + i) * 8, 1);
      mbar_init(sb + q1n::kBars + q1n::kBarSelMine * 8, 1);
      mbar_init(sb + q1n::kBars + q1n::kBarSelAll * 8, 1);
      mbar_init(sb + q1n::kBars + q1n::kBarXFree * 8,
                cta < kRoutedCtas ? kConsumerWarps - 1 : kConsumerWarps);
      mbar_init(sb + q1n::kBars + q1n::kBarP1 * 8, 1);
      mbar_fence_init();
    }
  }
  __syncthreads();
  const CtxTable& R = *reinterpret_cast<const CtxTable*>(dyn + q1n::kRows);
  if (tid >= kConsumerThreads) {
    if (tid == kConsumerThreads) q1n_produce(R, cta, sb);
    return;
  }
  const int warp = tid >> 5, lane = tid & 31;
  if (cta < kRoutedCtas)
    q1n_routed(args, R, cta, sb, tid, warp, lane);
  else
    q1n_shared(args, R, cta - kSharedBase, sb, tid, warp, lane);
}

template <int kS>
__global__ __launch_bounds__(kQ1Block,
                             1) void moe_q1_kernel(const __grid_constant__ LaunchArgs args) {
  static_assert(kS == 1, "q1 specialization");
  extern __shared__ __align__(1024) uint8_t q1_dyn[];
  const int cta = static_cast<int>(blockIdx.x);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (cta >= kWorkCtas) {
    initialize_sentinels(args, cta - kWorkCtas);
    return;
  }
  q1n_kernel(args, q1_dyn);
}

template __global__ void moe_q1_kernel<1>(const __grid_constant__ LaunchArgs args);

}  // namespace flashinfer::glm5_tp8::moe_q1

namespace flashinfer::glm5_tp8::moe_detail {
const void* q1_kernel() { return reinterpret_cast<const void*>(&moe_q1::moe_q1_kernel<1>); }
cudaError_t launch_q1(const MoeArguments& p, cudaStream_t stream, bool pdl, bool cooperative) {
  moe_q1::LaunchArgs a{};
  a.n_rows = 5;
  a.rank = p.rank;
  a.context = reinterpret_cast<const uint8_t*>(p.context);
  a.sync_words = reinterpret_cast<uint32_t*>(p.state);
  a.launch_ctr = reinterpret_cast<uint32_t*>(p.state + kMoeQ1CounterOffset);
  a.reset_token =
      reinterpret_cast<uint32_t*>(p.state + kMoeQ1CounterOffset + 148 * sizeof(uint32_t));
  for (int rank = 0; rank < 8; ++rank) a.peers[rank] = p.peers[rank];
  return launch(q1_kernel(), &a, 384, 214016, stream, pdl, cooperative);
}
}  // namespace flashinfer::glm5_tp8::moe_detail
