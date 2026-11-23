/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "ll_poll.cuh"
#include "mla_q1_helpers.cuh"
#include "mla_q1_types.cuh"

namespace flashinfer::glm5_tp8::mla_q1 {

using q1_detail::bulk_copy_g2s_l2;
using q1_detail::fence_acquire_gpu;
using q1_detail::l2_policy_evict_first;
using q1_detail::ld_relaxed_sys_b128;
using q1_detail::load_ll_record;
using q1_detail::load_relaxed_gpu;
using q1_detail::mbar_arrive;
using q1_detail::mbar_arrive_expect_tx;
using q1_detail::mbar_fence_init;
using q1_detail::mbar_init;
using q1_detail::mbar_wait_parity;
using q1_detail::red_release_gpu_add;
using q1_detail::smem_u32;
using q1_detail::st_relaxed_sys_b128;

// ---- bf16 bit helpers (raw uint16_t storage, RN rounding)
// ---------------------------------------------------------------
__device__ __forceinline__ float bf(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}
__device__ __forceinline__ uint16_t to_bf16_bits(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// ---- sync-word idiom: arrive = block barrier, thread 0 fence + atomic add (return value dropped);
//      wait = thread 0 acquire-polls, fence, block barrier broadcasts the acquisition
//      ---------------------------------------
__device__ __forceinline__ void word_arrive(uint32_t* word, uint32_t weight) {
  __syncthreads();
  if (threadIdx.x == 0) red_release_gpu_add(word, weight);
}
__device__ __forceinline__ void word_wait_ge(const uint32_t* word, uint32_t target) {
  if (threadIdx.x == 0) {
    while (load_relaxed_gpu(word) < target) {
    }
    fence_acquire_gpu();
  }
  __syncthreads();
}
__device__ __forceinline__ void word_wait_sign(const uint32_t* word) {
  if (threadIdx.x == 0) {
    while ((load_relaxed_gpu(word) & 0x80000000u) == 0u) {
    }
    fence_acquire_gpu();
  }
  __syncthreads();
}

__device__ __forceinline__ float sumsq8(const float* e) {
  const float p01 = fmaf(e[0], e[0], __fmul_rn(e[1], e[1]));
  const float p23 = fmaf(e[2], e[2], __fmul_rn(e[3], e[3]));
  const float p45 = fmaf(e[4], e[4], __fmul_rn(e[5], e[5]));
  const float p67 = fmaf(e[6], e[6], __fmul_rn(e[7], e[7]));
  return __fadd_rn(__fadd_rn(__fadd_rn(p01, p23), p45), p67);
}
// Returns the block sum to every thread. `red` = 8 floats of shared scratch.
__device__ __forceinline__ float norm_block_sum(float v, float* red) {
  const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  if (tid < 256) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float o = __shfl_down_sync(0xffffffffu, v, off);
      if (lane < off) v = __fadd_rn(v, o);
    }
    if (lane == 0) red[warp] = v;
  }
  __syncthreads();
  float acc = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) acc = __fadd_rn(acc, red[w]);
  // red[] is not overwritten before each caller's next CTA barrier.  The
  // trailing barrier only delayed consumers after every RMS reduction.
  return acc;
}
// RoPE on one adjacent pair (x[2j], x[2j+1]) with (cos_j, sin_j): o0 = fma(x0, c, -(x1*s)), o1 =
// fma(x0, s, x1*c); bf16 in / out.
__device__ __forceinline__ void rope_pair(uint16_t* v, const float* cs) {
  const float x0 = bf(v[0]), x1 = bf(v[1]), c = cs[0], s = cs[1];
  v[0] = to_bf16_bits(fmaf(x0, c, -__fmul_rn(x1, s)));
  v[1] = to_bf16_bits(fmaf(x0, s, __fmul_rn(x1, c)));
}

// ---- dynamic shared memory carve-out (bytes)
// --------------------------------------------------------------------------

}  // namespace flashinfer::glm5_tp8::mla_q1
namespace flashinfer::glm5_tp8::mla_q1 {

using q1_detail::fp8_to_bf16_fragment;
using q1_detail::hmma;
using q1_detail::hmma_ldsm_x4;
using q1_detail::hmma_load_a_16x16;
using q1_detail::hmma_load_a_16x16_trans;
using q1_detail::HmmaAcc;
using q1_detail::lds_f32;
using q1_detail::lds_fp8x8;
using q1_detail::lds_u32;
using q1_detail::RawFp8x8;

// ---- cp.async helpers
// -----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t s_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_pending(int n) {
  switch (n) {
    case 0:
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      break;
    case 1:
      asm volatile("cp.async.wait_group 1;" ::: "memory");
      break;
    case 2:
      asm volatile("cp.async.wait_group 2;" ::: "memory");
      break;
    case 3:
      asm volatile("cp.async.wait_group 3;" ::: "memory");
      break;
    case 4:
      asm volatile("cp.async.wait_group 4;" ::: "memory");
      break;
    default:
      asm volatile("cp.async.wait_group 5;" ::: "memory");
      break;
  }
}

// The release variant stalls the arriving thread on a full store drain (~0.5 us / word).
__device__ __forceinline__ void fast_arrive(uint32_t* word, uint32_t weight) {
  __syncthreads();
  if (threadIdx.x == 0) q1_detail::red_relaxed_gpu_add(word, weight);
}

// K5 receive, serial 7-thread form (narrow-block fallback; the 384-thread path overlaps this with
// the o_proj GEMV on warps 8..11).
__device__ __forceinline__ void k5_receive_serial(const LaunchArgs& args, int rank, size_t parity,
                                                  uint32_t e32, int cta, uint16_t* rx, int nt,
                                                  int tid) {
  for (int item = tid; item < (kWorld - 1); item += nt) {
    const int s = item;
    const int src = s + (s >= rank ? 1 : 0);
    const uint8_t* blk = args.peers[rank] + static_cast<size_t>(s) * kInboxSlotStride +
                         parity * kInboxParityStride + static_cast<size_t>(cta) * kInboxCtaStride;
    uint16_t* dstp = rx + src * kOProjRows;
    for (int line = 0; line < 8; ++line) {  // every line carries its own tag
      uint64_t lo = 0, hi = 0;
      for (;;) {
        ld_relaxed_sys_b128(blk + line * 16, lo, hi);
        if (static_cast<uint32_t>(hi >> 32) == e32) break;
        __nanosleep(64);
      }
      uint16_t* d = dstp + line * 6;
      d[0] = static_cast<uint16_t>(lo);
      d[1] = static_cast<uint16_t>(lo >> 16);
      d[2] = static_cast<uint16_t>(lo >> 32);
      d[3] = static_cast<uint16_t>(lo >> 48);
      d[4] = static_cast<uint16_t>(hi);
      d[5] = static_cast<uint16_t>(hi >> 16);
    }
  }
}

// Three-word wait with one polling thread per word: the reference kernel's serial three-load poll
// pays up to three L2 round trips per iteration on the attention critical path; three parallel
// pollers discover the last arrival after one.
__device__ __forceinline__ void fast_wait_ge3(const uint32_t* w0, const uint32_t* w1,
                                              const uint32_t* w2, uint32_t target) {
  const int t = static_cast<int>(threadIdx.x);
  if (t < 3) {
    const uint32_t* w = (t == 0) ? w0 : (t == 1) ? w1 : w2;
    while (load_relaxed_gpu(w) < target) {
    }
    fence_acquire_gpu();
  }
  __syncthreads();
}

constexpr int kFastBlock = 384;

// The globally normalized probabilities are rounded to BF16 before PV.  The
// rendezvous for the global statistics uses two words (0x58, 0x5c) of the
// 256-byte sync block, so they share its per-launch reset.
constexpr int kQ1StatsReady = 0x58 / 4;
constexpr int kQ1StatsRead = 0x5c / 4;
static_assert(kQ1StatsRead * sizeof(uint32_t) + sizeof(uint32_t) <= kSyncBlockBytes,
              "global probability rendezvous stays inside the per-launch reset block");

// One contiguous early weight page per mbarrier.
__device__ __forceinline__ void issue_bulk_page(uint8_t* dst_smem, const uint8_t* src, int bytes,
                                                uint64_t* bar) {
  // Only the matrix and live scale words are consumed.  Page strides stay at
  // their ABI sizes; trimming the copied tail avoids 132 KiB of useless traffic
  // per launch while retaining 16-byte transfer alignment.
  if (bytes == 0x8080)
    bytes = 0x8020;
  else if (bytes == 0x3080)
    bytes = 0x3010;
  else if (bytes == 0x4080)
    bytes = 0x4010;
  else if (bytes == 0x6080)
    bytes = 0x6030;
  // Keep the asynchronous bulk-copy issuer off tid 0, which owns the global
  // phase-counter REDs.  The final warp is otherwise idle at each call site.
  if (threadIdx.x == kFastBlock - 32) {
    const uint32_t b = smem_u32(bar);
    if (bytes != 0) {
      mbar_arrive_expect_tx(b, static_cast<uint32_t>(bytes));
      bulk_copy_g2s_l2(s_addr(dst_smem), src, static_cast<uint32_t>(bytes), b,
                       l2_policy_evict_first());
    } else {
      mbar_arrive(b);
    }
  }
}
__device__ __forceinline__ void wait_bulk_page(uint64_t* bar) {
  mbar_wait_parity(smem_u32(bar), 0u);
}

// ---- dynamic shared memory carve-out (fast kernel)
// --------------------------------------------------------------------
struct FastSmem {
  static constexpr int kQStride =
      584;  // bf16 elements per staged activation row (576 + 8 pad: ldmatrix bank spread)
  static constexpr int kPStride = 72;  // bf16 elements per probability row (64 + 8 pad)
  static constexpr int kAct = 0;       // xn bf16[6144] | qn bf16[2048] | q A [16][584] |
  static constexpr int kActBytes =
      16 * kQStride * 2;  //   P6 act[512] | P7 act[2560] + local[48] + rx[8][48]
  static constexpr int kRed = kAct + kActBytes;  // f32[8] norm partials (+pad)
  static constexpr int kPartial = kRed + 128;    // f32 [8 warp][48 row] GEMV partials
  static constexpr int kPartialBytes = 8 * 48 * 4;
  static constexpr int kScores = kPartial + kPartialBytes;    // f32 [10][64]
  static constexpr int kProb = kScores + 10 * 64 * 4;         // bf16 [16][kPStride]
  static constexpr int kIds = kProb + 16 * kPStride * 2;      // i32 [64]
  static constexpr int kStats = kIds + 256;                   // f32 m[10] l[10] rl[10] | w[32]
  static constexpr int kArena = (kStats + 256 + 127) & ~127;  // page arena (128 B aligned)
  static constexpr int kArenaBytes = 5 * 0x8080;              // 164,480: 5 qkv_a_proj ring slots
  static constexpr int kBytes = kArena + kArenaBytes;
  // arena offsets (bytes from kArena)
  static constexpr int kSlot = 0x8080;  // P1 page p -> slot (p % 5); P2 w0 -> 0, w1 -> 1; P3 -> 2
  static constexpr int kKvStrideB =
      584 * 2;  // gathered kv rows [64][584] bf16 (512 kv ++ 64 pe ++ pad)
  static constexpr int kKvBytes = 64 * kKvStrideB;  // 74,752
  static constexpr int kUvPage = 124160;  // W_UV page (all CTAs), clear of every o_proj layout
  // o_proj pages: CTAs >= 32 prefetch all five at p * 0x6080 (ends 123,520 <= kUvPage); attention
  // CTAs keep the kv area free: u0/u1 above the kv rows, u2..u4 in the kv area after PV.
  static __device__ __forceinline__ int o_proj_off(int cta, int p) {
    if (cta >= kSplits) return p * 0x6080;
    if (p == 0) return kKvBytes;
    if (p == 1) return kKvBytes + 0x6080;
    return (p - 2) * 0x6080;
  }
};
static_assert(FastSmem::kKvBytes + 2 * 0x6080 <= FastSmem::kUvPage,
              "attention u0/u1 fit under the W_UV slot");
static_assert(FastSmem::kUvPage + 0x4080 <= FastSmem::kArenaBytes, "W_UV page fits the arena");
static_assert(FastSmem::kBytes <= kLaunchDynSmem,
              "fast kernel fits the dynamic smem requested at launch");

// ---- B fragment of one k16 step for q = 1 (token 0 on lanes 0..3, zero elsewhere)
// -------------------------------------
__device__ __forceinline__ void act_frag_q1(uint32_t act_s, int k, int lane, uint32_t (&b)[2]) {
  b[0] = 0u;
  b[1] = 0u;
  if (lane < 4) {
    const uint32_t a = act_s + static_cast<uint32_t>(k + 2 * lane) * 2u;
    b[0] = lds_u32(a);
    b[1] = lds_u32(a + 16u);
  }
}

// ---- 32-row x 1024-K page (qkv_a_proj / q_b_proj), warp w = 128-K block w of both 16-row halves
// ----------------------
template <bool kQkvAProjTiles>
__device__ __forceinline__ void consume_page32(uint32_t slot_s, uint32_t act_s, int k0_page,
                                               int warp, int lane, float2 (&run)[2]) {
  const uint32_t a0 = slot_s + static_cast<uint32_t>(warp) * (kQkvAProjTiles ? 0x100u : 0x800u) +
                      static_cast<uint32_t>(lane) * 8u;
  const uint32_t kstep = kQkvAProjTiles ? 0x800u : 0x100u;
  // The activation fragments are shared by both 16-row halves; the raw weight fragments are
  // consumed one half at a time so the live register set stays small (raw[8] + b[8][2] instead of
  // raw[16]).
  uint32_t b[8][2];
#pragma unroll
  for (int t = 0; t < 8; ++t) act_frag_q1(act_s, k0_page + warp * 128 + t * 16, lane, b[t]);
  const float scale = lds_f32(slot_s + 0x8000u + static_cast<uint32_t>(warp) * 4u);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    RawFp8x8 raw[8];
    const uint32_t ah = a0 + static_cast<uint32_t>(half) * 0x4000u;
#pragma unroll
    for (int t = 0; t < 8; ++t) raw[t] = lds_fp8x8(ah + static_cast<uint32_t>(t) * kstep);
    HmmaAcc d{{0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      uint32_t A[4];
      fp8_to_bf16_fragment(raw[t], A);
      hmma(d, A, b[t]);
    }
    run[half].x = fmaf(scale, d.value[0], run[half].x);
    run[half].y = fmaf(scale, d.value[2], run[half].y);
  }
}

// ---- scatter one warp's 32-row running sums into the ordered-fold buffer
// ----------------------------------------------
__device__ __forceinline__ void scatter_partial32(float* partial, int warp, int lane,
                                                  const float2 (&run)[2]) {
  if ((lane & 3) == 0) {
    float* p = partial + warp * 48;
    const int r = lane >> 2;
    p[r] = run[0].x;
    p[r + 8] = run[0].y;
    p[16 + r] = run[1].x;
    p[16 + 8 + r] = run[1].y;
  }
}

__global__ void __launch_bounds__(kFastBlock)
    q1_attention_mode_fast_kernel(const __grid_constant__ LaunchArgs args) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  using S = FastSmem;
  extern __shared__ __align__(128) uint8_t smem[];
  // 0..5 qkv_a_proj pages, 6..7 q_b_proj, 8 W_UK, 9..10 o_proj u0/u1, 11 W_UV, 12..14 o_proj
  // u2..u4, 15 x row.
  __shared__ __align__(16) uint64_t s_early_bars[16];
  const int cta = static_cast<int>(blockIdx.x), tid = static_cast<int>(threadIdx.x);
  constexpr int nt = kFastBlock;
  const int warp = tid >> 5, lane = tid & 31;
  constexpr int rank = 1;
  __shared__ uint32_t s_generation;
  if (tid == 0) {
    s_generation = args.launch_ctr[cta] + 1u;
    args.launch_ctr[cta] = s_generation;
    if (cta == 0) {
      auto* words = reinterpret_cast<uint4*>(args.sync_words);
      const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int i = 0; i < kSyncBlockBytes / 16; ++i) words[i] = zero;
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(args.reset_token), "r"(s_generation)
                   : "memory");
    }
  }
  if (cta < kGroup1Ctas && tid < 16) {
    mbar_init(smem_u32(s_early_bars + tid), 1u);
    mbar_fence_init();
  }
  if (tid == 0) {
    uint32_t observed;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                   : "=r"(observed)
                   : "l"(args.reset_token)
                   : "memory");
    } while (observed != s_generation);
  }
  __syncthreads();
  const MlaTable& T = *reinterpret_cast<const MlaTable*>(
      args.context);  // immutable operator descriptor prepared by the caller
  uint32_t* B = args.sync_words;
  const int cur_pos = *T.r0.cur_pos;
  const uint32_t e32 = s_generation;
  const size_t parity = static_cast<size_t>(e32 & 1u);
  float* red = reinterpret_cast<float*>(smem + S::kRed);
  float* partial = reinterpret_cast<float*>(smem + S::kPartial);
  uint8_t* arena = smem + S::kArena;
  const uint32_t arena_s = s_addr(arena);
  const uint32_t act_s = s_addr(smem + S::kAct);

  // ------------------------------------------------------------------ tail group: CTAs 128..131
  // kv-norm, 132..147 RoPE
  if (cta >= kGroup1Ctas) {
    if (cta < kRopeTail) {
      const int t = cta - kKvNormTail;
      // kv_gamma is a launch input: issue its loads under the C0 wait (both tails are on the
      // attention critical path through C3 / C2, so every cold DRAM trip removed here shortens C4
      // -> C5).
      float4 kga = {0.0f, 0.0f, 0.0f, 0.0f}, kgb = {0.0f, 0.0f, 0.0f, 0.0f};
      if (t < 1 && tid < kKvlora / 8) {
        kga = *reinterpret_cast<const float4*>(T.r3.kv_gamma + tid * 8);
        kgb = *reinterpret_cast<const float4*>(T.r3.kv_gamma + tid * 8 + 4);
      }
      word_wait_ge(B + kWordC0, kSyncFull);
      if (t < 1) {
        const uint16_t* kv = T.r3.alias_post_kv;
        float v = 0.0f;
        uint4 kvr = {0u, 0u, 0u, 0u};
        if (tid <
            kKvlora / 8) {  // one 16 B load per thread covers both the sumsq and the write below
          kvr = __ldcg(reinterpret_cast<const uint4*>(kv + 8 * tid));
          float e[8];
          e[0] = bf(static_cast<uint16_t>(kvr.x));
          e[1] = bf(static_cast<uint16_t>(kvr.x >> 16));
          e[2] = bf(static_cast<uint16_t>(kvr.y));
          e[3] = bf(static_cast<uint16_t>(kvr.y >> 16));
          e[4] = bf(static_cast<uint16_t>(kvr.z));
          e[5] = bf(static_cast<uint16_t>(kvr.z >> 16));
          e[6] = bf(static_cast<uint16_t>(kvr.w));
          e[7] = bf(static_cast<uint16_t>(kvr.w >> 16));
          v = sumsq8(e);
        }
        const float sum = norm_block_sum(v, red);
        const float rstd = rsqrtf(fmaf(sum, 0.001953125f, kRmsEps));
        uint16_t* dst = T.r3.kv_cache + static_cast<size_t>(cur_pos) * kKvlora;
        if (tid < kKvlora / 8) {
          const int i = tid * 8;
          const uint4 kvv = kvr;
          const float4 g0 = kga;
          const float4 g1 = kgb;
          uint4 o;
          o.x = static_cast<uint32_t>(to_bf16_bits(
                    __fmul_rn(bf(static_cast<uint16_t>(kvv.x)), __fmul_rn(g0.x, rstd)))) |
                (static_cast<uint32_t>(to_bf16_bits(
                     __fmul_rn(bf(static_cast<uint16_t>(kvv.x >> 16)), __fmul_rn(g0.y, rstd))))
                 << 16);
          o.y = static_cast<uint32_t>(to_bf16_bits(
                    __fmul_rn(bf(static_cast<uint16_t>(kvv.y)), __fmul_rn(g0.z, rstd)))) |
                (static_cast<uint32_t>(to_bf16_bits(
                     __fmul_rn(bf(static_cast<uint16_t>(kvv.y >> 16)), __fmul_rn(g0.w, rstd))))
                 << 16);
          o.z = static_cast<uint32_t>(to_bf16_bits(
                    __fmul_rn(bf(static_cast<uint16_t>(kvv.z)), __fmul_rn(g1.x, rstd)))) |
                (static_cast<uint32_t>(to_bf16_bits(
                     __fmul_rn(bf(static_cast<uint16_t>(kvv.z >> 16)), __fmul_rn(g1.y, rstd))))
                 << 16);
          o.w = static_cast<uint32_t>(to_bf16_bits(
                    __fmul_rn(bf(static_cast<uint16_t>(kvv.w)), __fmul_rn(g1.z, rstd)))) |
                (static_cast<uint32_t>(to_bf16_bits(
                     __fmul_rn(bf(static_cast<uint16_t>(kvv.w >> 16)), __fmul_rn(g1.w, rstd))))
                 << 16);
          *reinterpret_cast<uint4*>(dst + i) = o;
        }
      }
      fast_arrive(B + kWordC3, kWeightKvNorm);
    } else {
      const int t = cta - kRopeTail;
      // The Q_PE rotation happens in the P2 stores (the producers hold the raw pair in registers),
      // so this tail rotates only the pe_cache row and arrives C2 without waiting for C1, off the
      // attention critical path.  The RoPE table row (cold 256 B input) is fetched under the wait.
      float2 cs_reg = make_float2(0.0f, 0.0f);
      if (t < 1 && tid < kRope / 2)
        cs_reg = *reinterpret_cast<const float2*>(T.r2.rope + size_t(cur_pos) * kRope + 2 * tid);
      word_wait_ge(B + kWordC0, kSyncFull);
      if (t < 1) {
        const float cs2[2] = {cs_reg.x, cs_reg.y};
        uint16_t* pe = T.r2.alias_pe_cache + static_cast<size_t>(cur_pos) * kRope;
        if (tid < kRope / 2) {
          uint16_t v[2] = {__ldcg(reinterpret_cast<const unsigned short*>(pe + 2 * tid)),
                           __ldcg(reinterpret_cast<const unsigned short*>(pe + 2 * tid + 1))};
          rope_pair(v, cs2);
          pe[2 * tid] = v[0];
          pe[2 * tid + 1] = v[1];
        }
      }
      fast_arrive(B + kWordC2, kWeightRope);
    }
    return;
  }

  // ------------------------------------------------------------------ P1: x-RMSNorm + qkv_a_proj
  // GEMV (CTAs 0..81) -> C0
  {
    const bool work = cta < kQkvAProjCtas;
    const uint8_t* blob = T.r0.qkv_a_proj;
    // qkv_a_proj pages 0..1 at entry, 2..4 after the norm reduce: the x / gamma loads then race
    // a 5.4 MB page burst instead of 13.5 MB, and pages 2..4 still lead their consume by ~2 us. The
    // issue order does not affect the numerics.
    uint16_t* xn = reinterpret_cast<uint16_t*>(smem + S::kAct);
    issue_bulk_page(reinterpret_cast<uint8_t*>(xn),
                    work ? reinterpret_cast<const uint8_t*>(T.r0.x) : nullptr,
                    work ? kHidden * 2 : 0, s_early_bars + 15);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      issue_bulk_page(arena + p * S::kSlot,
                      work ? blob + (static_cast<size_t>(cta) * 6 + p) * 0x8080 : nullptr,
                      work ? 0x8080 : 0, s_early_bars + p);
    }
    if (work) {
      float4 ga_r[3], gb_r[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int i8 = tid + k * nt;
        if (i8 < kHidden / 8) {
          ga_r[k] = *reinterpret_cast<const float4*>(T.r0.x_gamma + i8 * 8);
          gb_r[k] = *reinterpret_cast<const float4*>(T.r0.x_gamma + i8 * 8 + 4);
        }
      }
      wait_bulk_page(s_early_bars + 15);
      float v = 0.0f;
      if (tid < 256) {
        float ch[3];
#pragma unroll
        for (int band = 0; band < 3; ++band) {
          const uint4 xv = *reinterpret_cast<const uint4*>(xn + band * 2048 + 8 * tid);
          float e[8];
          e[0] = bf(static_cast<uint16_t>(xv.x));
          e[1] = bf(static_cast<uint16_t>(xv.x >> 16));
          e[2] = bf(static_cast<uint16_t>(xv.y));
          e[3] = bf(static_cast<uint16_t>(xv.y >> 16));
          e[4] = bf(static_cast<uint16_t>(xv.z));
          e[5] = bf(static_cast<uint16_t>(xv.z >> 16));
          e[6] = bf(static_cast<uint16_t>(xv.w));
          e[7] = bf(static_cast<uint16_t>(xv.w >> 16));
          ch[band] = sumsq8(e);
        }
        v = __fadd_rn(__fadd_rn(ch[0], ch[1]), ch[2]);
      }
      const float sum = norm_block_sum(v, red);
      const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(sum, 6144.0f), kRmsEps));
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int i8 = tid + k * nt;
        if (i8 >= kHidden / 8) break;
        const int i = i8 * 8;
        const uint4 xv = *reinterpret_cast<const uint4*>(xn + i);
        const float4 g0 = ga_r[k];
        const float4 g1 = gb_r[k];
        uint4 o;
        o.x = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(xv.x)), __fmul_rn(g0.x, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(xv.x >> 16)), __fmul_rn(g0.y, rstd))))
               << 16);
        o.y = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(xv.y)), __fmul_rn(g0.z, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(xv.y >> 16)), __fmul_rn(g0.w, rstd))))
               << 16);
        o.z = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(xv.z)), __fmul_rn(g1.x, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(xv.z >> 16)), __fmul_rn(g1.y, rstd))))
               << 16);
        o.w = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(xv.w)), __fmul_rn(g1.z, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(xv.w >> 16)), __fmul_rn(g1.w, rstd))))
               << 16);
        *reinterpret_cast<uint4*>(xn + i) = o;
      }
    }
    // pages 2..4 follow once the norm's loads are in flight (empty pages for the non-work CTAs)
#pragma unroll
    for (int p = 2; p < 5; ++p) {
      issue_bulk_page(arena + p * S::kSlot,
                      work ? blob + (static_cast<size_t>(cta) * 6 + p) * 0x8080 : nullptr,
                      work ? 0x8080 : 0, s_early_bars + p);
    }
    float2 run[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    // Pages consumed in PAIRS: the two consumes' shared-load/convert/MMA chains are independent, so
    // back-to-back calls give the scheduler a second in-flight chain to hide the shared-load and
    // MMA latencies. The mbarrier completion already publishes the TMA payload; CTA barriers remain
    // only for the xn hazard and before a slot is re-filled.
#pragma unroll 1
    for (int pp = 0; pp < 3; ++pp) {
      const int p0 = pp * 2;
      wait_bulk_page(s_early_bars + p0);
      wait_bulk_page(s_early_bars + p0 + 1);
      if (pp == 0) __syncthreads();
      if (work && warp < 8) {
        consume_page32<true>(arena_s + (p0 % 5) * S::kSlot, act_s, p0 * 1024, warp, lane, run);
        consume_page32<true>(arena_s + ((p0 + 1) % 5) * S::kSlot, act_s, (p0 + 1) * 1024, warp,
                             lane, run);
      }
      if (pp < 2) {
        __syncthreads();
        if (pp == 0) {  // page 5 into the freed slot 0, q_b_proj page 0 into slot 1
          issue_bulk_page(arena,
                          work ? blob + (static_cast<size_t>(cta) * 6 + 5) * 0x8080 : nullptr,
                          work ? 0x8080 : 0, s_early_bars + 5);
          const bool w = cta < kQbProjCtas;
          issue_bulk_page(arena + S::kSlot,
                          w ? T.r1.q_b_proj + static_cast<size_t>(cta) * 2 * 0x8080 : nullptr,
                          w ? 0x8080 : 0, s_early_bars + 6);
        } else {  // q_b_proj page 1 into freed slot 2, W_UK into slot 3
          const bool w = cta < kQbProjCtas;
          issue_bulk_page(arena + 2 * S::kSlot,
                          w ? T.r1.q_b_proj + (static_cast<size_t>(cta) * 2 + 1) * 0x8080 : nullptr,
                          w ? 0x8080 : 0, s_early_bars + 7);
          const bool wq = cta < kUkCtas;
          issue_bulk_page(arena + 3 * S::kSlot,
                          wq ? T.r4.w_uk + static_cast<size_t>(cta) * 0x3080 : nullptr,
                          wq ? 0x3080 : 0, s_early_bars + 8);
        }
      }
    }
    if (work && warp < 8) scatter_partial32(partial, warp, lane, run);
    __syncthreads();
    if (work && tid < 32) {
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) acc = __fadd_rn(acc, partial[w * 48 + tid]);
      const int row = cta * 32 + tid;
      const uint16_t v = to_bf16_bits(acc);
      if (row < kQlora)
        T.r0.post_q[row] = v;
      else if (row < kQlora + kKvlora)
        T.r0.post_kv[row - kQlora] = v;
      else
        T.r0.pe_cache[static_cast<size_t>(cur_pos) * kRope + (row - kQlora - kKvlora)] = v;
    }
  }
  fast_arrive(B + kWordC0, kWeightGroup1);
  // q_gamma is a launch input: issue its loads under the C0 wait so the q-norm pays only the post_q
  // trip.
  float4 qga = {0.0f, 0.0f, 0.0f, 0.0f}, qgb = {0.0f, 0.0f, 0.0f, 0.0f};
  if (cta < kQbProjCtas && tid < kQlora / 8) {
    qga = *reinterpret_cast<const float4*>(T.r1.q_gamma + tid * 8);
    qgb = *reinterpret_cast<const float4*>(T.r1.q_gamma + tid * 8 + 4);
  }
  // The Q_PE RoPE is folded into this phase's stores (the producing thread pair already holds the
  // raw bf16 values): prefetch this thread's (cos, sin) pair.  Rows 1920..2559 are CTAs 60..79; the
  // pair index of thread tid is tid>>1 (even CTA) or 16 + (tid>>1) (odd CTA).
  float2 qcs = make_float2(0.0f, 0.0f);
  if (cta >= 60 && cta < kQbProjCtas && tid < 32) {
    const int j = ((cta & 1) << 4) + (tid >> 1);
    qcs = *reinterpret_cast<const float2*>(T.r2.rope + size_t(cur_pos) * kRope + 2 * j);
  }
  word_wait_ge(B + kWordC0, kSyncFull);
  // CTAs 82..127 never touch the page arena: stream their W_UV + five o_proj pages from C0 on,
  // spread over the quiet P2..P3 DRAM window (the P1 window carries the 13.5 MB qkv_a_proj burst,
  // the C4 window the kv gather; this group's 6.4 MB fits between them).
  if (cta >= kQkvAProjCtas) {
    issue_bulk_page(arena + S::o_proj_off(cta, 0),
                    T.r7.o_proj + static_cast<size_t>(cta) * 5 * 0x6080, 0x6080, s_early_bars + 9);
    issue_bulk_page(arena + S::o_proj_off(cta, 1),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 1) * 0x6080, 0x6080,
                    s_early_bars + 10);
    issue_bulk_page(arena + S::kUvPage,
                    cta < kUvCtas ? T.r6.w_uv + static_cast<size_t>(cta) * 0x4080 : nullptr,
                    cta < kUvCtas ? 0x4080 : 0, s_early_bars + 11);
    issue_bulk_page(arena + S::o_proj_off(cta, 2),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 2) * 0x6080, 0x6080,
                    s_early_bars + 12);
    issue_bulk_page(arena + S::o_proj_off(cta, 3),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 3) * 0x6080, 0x6080,
                    s_early_bars + 13);
    issue_bulk_page(arena + S::o_proj_off(cta, 4),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 4) * 0x6080, 0x6080,
                    s_early_bars + 14);
  }

  // ------------------------------------------------------------------ P2: q-RMSNorm + q_b_proj
  // (CTAs 0..79) -> C1
  {
    const bool work = cta < kQbProjCtas;
    uint16_t* qn = reinterpret_cast<uint16_t*>(smem + S::kAct);
    if (work) {
      const uint16_t* q = T.r1.alias_post_q;
      float v = 0.0f;
      uint4 qv = {0u, 0u, 0u, 0u};
      if (tid < kQlora / 8) {  // one 16 B load per thread covers both the sumsq and the write below
        qv = __ldcg(reinterpret_cast<const uint4*>(q + 8 * tid));
        float e[8];
        e[0] = bf(static_cast<uint16_t>(qv.x));
        e[1] = bf(static_cast<uint16_t>(qv.x >> 16));
        e[2] = bf(static_cast<uint16_t>(qv.y));
        e[3] = bf(static_cast<uint16_t>(qv.y >> 16));
        e[4] = bf(static_cast<uint16_t>(qv.z));
        e[5] = bf(static_cast<uint16_t>(qv.z >> 16));
        e[6] = bf(static_cast<uint16_t>(qv.w));
        e[7] = bf(static_cast<uint16_t>(qv.w >> 16));
        v = sumsq8(e);
      }
      const float sum = norm_block_sum(v, red);
      const float rstd = rsqrtf(fmaf(sum, 0.00048828125f, kRmsEps));
      if (tid < kQlora / 8) {
        const float4 g0 = qga;
        const float4 g1 = qgb;
        uint4 o;
        o.x = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(qv.x)), __fmul_rn(g0.x, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(qv.x >> 16)), __fmul_rn(g0.y, rstd))))
               << 16);
        o.y = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(qv.y)), __fmul_rn(g0.z, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(qv.y >> 16)), __fmul_rn(g0.w, rstd))))
               << 16);
        o.z = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(qv.z)), __fmul_rn(g1.x, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(qv.z >> 16)), __fmul_rn(g1.y, rstd))))
               << 16);
        o.w = static_cast<uint32_t>(
                  to_bf16_bits(__fmul_rn(bf(static_cast<uint16_t>(qv.w)), __fmul_rn(g1.z, rstd)))) |
              (static_cast<uint32_t>(to_bf16_bits(
                   __fmul_rn(bf(static_cast<uint16_t>(qv.w >> 16)), __fmul_rn(g1.w, rstd))))
               << 16);
        *reinterpret_cast<uint4*>(qn + tid * 8) = o;
      }
    }
    float2 run[2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    wait_bulk_page(s_early_bars + 6);
    wait_bulk_page(s_early_bars +
                   7);  // both q_b_proj pages landed during P1: consume back-to-back for ILP
    __syncthreads();    // qn (smem) is written by the norm above
    if (work && warp < 8) {
      consume_page32<false>(arena_s + S::kSlot, act_s, 0, warp, lane, run);
      consume_page32<false>(arena_s + 2 * S::kSlot, act_s, 1024, warp, lane, run);
    }
    if (work && warp < 8) scatter_partial32(partial, warp, lane, run);
    __syncthreads();
    if (work && tid < 32) {
      float acc = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) acc = __fadd_rn(acc, partial[w * 48 + tid]);
      const int row = cta * 32 + tid;
      const uint16_t v = to_bf16_bits(acc);
      if (row < kQbProjNopeRows) {
        T.r1.post_qnd[row] = v;
      } else {
        // Q_PE RoPE folded into the store (the tail CTA rotates only the pe_cache row and arrives
        // C2 right after C0).  Same arithmetic as a separate RoPE pass: pairs of
        // bf16(acc),

        const int e = row - kQbProjNopeRows;
        const uint32_t mine = v;
        const uint32_t peer = __shfl_xor_sync(0xffffffffu, mine, 1);
        const float xe = bf(static_cast<uint16_t>((tid & 1) ? peer : mine));
        const float xo = bf(static_cast<uint16_t>((tid & 1) ? mine : peer));
        const float r = (e & 1) ? fmaf(xe, qcs.y, __fmul_rn(xo, qcs.x))
                                : fmaf(xe, qcs.x, -__fmul_rn(xo, qcs.y));
        T.r1.post_qpe[e] = to_bf16_bits(r);
      }
    }
  }
  fast_arrive(B + kWordC1, kWeightGroup1);

  // CTAs 32..81 are finished with the low arena after P2.  Pull the pages
  // which cannot overlap W_UK into this otherwise quiet interval instead of
  // competing with the attention gather after C4.  CTAs 80..81 have no W_UK
  // payload, so their final two pages are safe here as well.
  if (cta >= kSplits && cta < kQkvAProjCtas) {
    issue_bulk_page(arena + S::o_proj_off(cta, 0),
                    T.r7.o_proj + static_cast<size_t>(cta) * 5 * 0x6080, 0x6080, s_early_bars + 9);
    issue_bulk_page(arena + S::o_proj_off(cta, 1),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 1) * 0x6080, 0x6080,
                    s_early_bars + 10);
    issue_bulk_page(arena + S::kUvPage,
                    cta < kUvCtas ? T.r6.w_uv + static_cast<size_t>(cta) * 0x4080 : nullptr,
                    cta < kUvCtas ? 0x4080 : 0, s_early_bars + 11);
    issue_bulk_page(arena + S::o_proj_off(cta, 2),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 2) * 0x6080, 0x6080,
                    s_early_bars + 12);
    if (cta >= kUkCtas) {
      issue_bulk_page(arena + S::o_proj_off(cta, 3),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 3) * 0x6080, 0x6080,
                      s_early_bars + 13);
      issue_bulk_page(arena + S::o_proj_off(cta, 4),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 4) * 0x6080, 0x6080,
                      s_early_bars + 14);
    }
  }
  word_wait_ge(B + kWordC1, kSyncFull);

  // ------------------------------------------------------------------ P3: W_UK (CTAs 0..79, 64
  // rows, K = 192) -> C4
  {
    const bool work = cta < kUkCtas;
    if (work && tid < 24) {  // stage post_qnd[head][192] (bf16, 24 x 16 B)
      const int head = cta >> 3;
      *reinterpret_cast<uint4*>(smem + S::kAct + tid * 16) =
          __ldcg(reinterpret_cast<const uint4*>(T.r4.alias_post_qnd + head * kNope + tid * 8));
    }
    wait_bulk_page(s_early_bars + 8);
    __syncthreads();
    if (work && warp < 4) {
      const uint32_t slot = arena_s + 3 * S::kSlot;
      const uint32_t a_rg =
          slot + static_cast<uint32_t>(warp) * 0xC00u + static_cast<uint32_t>(lane) * 8u;
      float2 rq = {0.0f, 0.0f};
#pragma unroll
      for (int bblk = 0; bblk < 3; ++bblk) {
        RawFp8x8 raw[4];
        uint32_t bb[4][2];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          raw[t] = lds_fp8x8(a_rg + static_cast<uint32_t>(bblk * 4 + t) * 0x100u);
          act_frag_q1(act_s, bblk * 64 + t * 16, lane, bb[t]);
        }
        const float sc = lds_f32(slot + 0x3000u + static_cast<uint32_t>(bblk) * 4u);
        HmmaAcc d{{0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          uint32_t A[4];
          fp8_to_bf16_fragment(raw[t], A);
          hmma(d, A, bb[t]);
        }
        rq.x = fmaf(sc, d.value[0], rq.x);
        rq.y = fmaf(sc, d.value[2], rq.y);
      }
      if ((lane & 3) == 0) {
        const int r0 = cta * 64 + warp * 16 + (lane >> 2);
        T.r4.post_qnope[r0] = to_bf16_bits(rq.x);
        T.r4.post_qnope[r0 + 8] = to_bf16_bits(rq.y);
      }
    }
  }
  fast_arrive(B + kWordC4, kWeightGroup1);

  // ------------------------------------------------------------------ P4/P5: split-KV attention +
  // combine (CTAs 0..31)
  const int kv_len = cur_pos + 1;
  const int att_len = kv_len < kSelected ? kv_len : kSelected;
  const int n_blocks = (att_len + kSplitRows - 1) / kSplitRows;
  if (cta < kSplits) {
    const int split = cta;
    int32_t* ids_s = reinterpret_cast<int32_t*>(smem + S::kIds);
    const bool att_work = split < n_blocks;
    const int rows_this_split = att_work ? min(kSplitRows, att_len - split * kSplitRows) : 0;
    const int mma_rows = (rows_this_split + 15) & ~15;
    // ids first: the LL records / local table / dense ids do not depend on the sync words, so the
    // cache-row gather below overlaps the C2/C3/C4 wait.
    if (att_work) {
      const bool ll_mode = kv_len > kSelected;
      if (!ll_mode && tid < kSplitRows) {
        const int s = split * kSplitRows + tid;
        const int lim = (kv_len - 1 < kSelected - 1) ? kv_len - 1 : kSelected - 1;
        ids_s[tid] = (s <= lim) ? s : kIdSentinel;
      } else if (ll_mode && T.r5.ll_recv != nullptr && tid < kSplitRows / 2) {
        // Rank zero publishes the complete {id0,tag,id1,tag} record concurrently.
        // Read its payload and tags together with the matching scalar b128 load.
        const int r = split * kLlRecordsPerCta + tid;
        const uint32_t lltag = static_cast<uint32_t>(T.r5.ll_tag);
        const uint8_t* ll_record =
            reinterpret_cast<const uint8_t*>(T.r5.ll_recv) + static_cast<size_t>(r) * 16;
        uint4 rec = load_ll_record(ll_record);
        uint32_t llspins = 0;
        while (rec.y != lltag || rec.w != lltag) {
          __nanosleep(64);
          rec = load_ll_record(ll_record);
          if (++llspins == (1u << 24)) {
            if (ll_wire::deadline_missing_tags(llspins, 1u << 24, rec.y, rec.w, lltag))
              ll_wire::timeout_trap();
            llspins =
                0;  // final reload was valid: preserve the ordinary consume/clear-tag path below
            break;
          }
        }
        ids_s[2 * tid] = static_cast<int32_t>(rec.x);
        ids_s[2 * tid + 1] = static_cast<int32_t>(rec.z);
        // The constant tag must be cleared after each consumption. Rank zero
        // cannot publish the next call before this CTA's collective push, so
        // this thread retires only the record it just consumed.
        if (llspins != (1u << 24)) {
          asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(
                           reinterpret_cast<uint4*>(const_cast<uint8_t*>(
                               reinterpret_cast<const uint8_t*>(T.r5.ll_recv))) +
                           r),
                       "r"(rec.x), "r"(0u), "r"(rec.z), "r"(0u)
                       : "memory");
        }
      } else if (ll_mode && T.r5.ll_recv == nullptr && tid < kSplitRows) {
        ids_s[tid] = __ldcg(T.r5.ids + split * kSplitRows + tid);
      }
    }
    __syncthreads();
    // g9 -> TMA: one 1 KB + one 128 B bulk copy per valid row (thread = key), tx-counted on bar 15;
    // the invalid slots are zeroed by a flat store loop.  Issued before the wait so the copies fly
    // under it.
    if (att_work) {
      const uint8_t* kvc = reinterpret_cast<const uint8_t*>(T.r5.alias_kv_cache);
      const uint8_t* pec = reinterpret_cast<const uint8_t*>(T.r5.alias_pe_cache);
      // g9: gather every kv/pe row except row cur_pos (stable cache inputs; masked slots zeroed)
      // before the wait.  16 B cp.async from threads 32..383 ONLY: warp 0 holds the sync-word
      // pollers, and an issuing thread can stall for microseconds behind a full memory-instruction
      // queue.
      if (tid >= 32) {
        for (int item = tid - 32; item < mma_rows * 72; item += nt - 32) {
          const int key = item / 72, l = item - key * 72;
          const uint32_t dst = arena_s + static_cast<uint32_t>(key * S::kKvStrideB + l * 16);
          const int32_t id = ids_s[key];
          if (id >= 0 && id < cur_pos) {
            const uint8_t* src =
                (l < 64) ? kvc + static_cast<size_t>(id) * 1024 + static_cast<size_t>(l) * 16
                         : pec + static_cast<size_t>(id) * 128 + static_cast<size_t>(l - 64) * 16;
            cp16(dst, src);
          } else if (id != cur_pos) {
            *reinterpret_cast<uint4*>(arena + key * S::kKvStrideB + l * 16) =
                make_uint4(0u, 0u, 0u, 0u);
          }
        }
        // Q_PE is complete at C1 and independent of W_UK.  Stage it in the
        // same pre-wait copy group so its L2 latency is hidden by P3/C4.
        uint16_t* qsm = reinterpret_cast<uint16_t*>(smem + S::kAct);
        const uint16_t* qp = T.r5.alias_post_qpe;
        for (int i = tid - 32; i < kHeads * 8; i += nt - 32) {
          const int h = i >> 3, l = i & 7;
          cp16(s_addr(qsm + h * S::kQStride + (64 + l) * 8),
               reinterpret_cast<const uint8_t*>(qp + h * kRope + l * 8));
        }
      }
    }
    cp_commit();  // g9
    fast_wait_ge3(B + kWordC2, B + kWordC3, B + kWordC4, kSyncFull);
    if (att_work) {
      float* sc = reinterpret_cast<float*>(smem + S::kScores);
      // g10: the freshly-written cache row (kv-norm / RoPE tails) + the q rows [16][kQStride], 16 B
      // cp.async from threads 32..383 (warp 0 stays off the issue path).
      {
        const uint8_t* kvc = reinterpret_cast<const uint8_t*>(T.r5.alias_kv_cache);
        const uint8_t* pec = reinterpret_cast<const uint8_t*>(T.r5.alias_pe_cache);
        uint16_t* qsm = reinterpret_cast<uint16_t*>(smem + S::kAct);
        const uint16_t* qn = T.r5.alias_post_qnope;
        if (tid >= 32) {
          for (int item = tid - 32; item < mma_rows * 72; item += nt - 32) {
            const int key = item / 72, l = item - key * 72;
            if (ids_s[key] == cur_pos) {
              const uint32_t dst = arena_s + static_cast<uint32_t>(key * S::kKvStrideB + l * 16);
              const uint8_t* src =
                  (l < 64)
                      ? kvc + static_cast<size_t>(cur_pos) * 1024 + static_cast<size_t>(l) * 16
                      : pec + static_cast<size_t>(cur_pos) * 128 + static_cast<size_t>(l - 64) * 16;
              cp16(dst, src);
            }
          }
          for (int i = tid - 32; i < kHeads * 64; i += nt - 32) {
            const int h = i >> 6, l = i & 63;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(qn + h * kKvlora + l * 8);
            cp16(s_addr(qsm + h * S::kQStride + l * 8), src);
          }
        }
      }
      cp_commit();  // g10

      issue_bulk_page(arena + S::o_proj_off(cta, 0),
                      T.r7.o_proj + static_cast<size_t>(cta) * 5 * 0x6080, 0x6080,
                      s_early_bars + 9);
      issue_bulk_page(arena + S::o_proj_off(cta, 1),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 1) * 0x6080, 0x6080,
                      s_early_bars + 10);
      issue_bulk_page(arena + S::kUvPage, T.r6.w_uv + static_cast<size_t>(cta) * 0x4080, 0x4080,
                      s_early_bars + 11);
      cp_wait_pending(0);
      __syncthreads();
      // QK MMA: warps 0..3 (key tile = warp * 16), A = q rows, B = kv rows; scores fp32 [10][64].
      // Software-pipelined: the next k16 step's ldmatrix fragments are fetched while the current
      // MMAs run, so the ~30-cycle ldmatrix latency no longer serializes the 36-step loop.
      if (warp < (mma_rows >> 4)) {
        const uint16_t* qsm = reinterpret_cast<const uint16_t*>(smem + S::kAct);
        const uint16_t* kvs = reinterpret_cast<const uint16_t*>(arena);
        HmmaAcc acc2[2] = {{{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 0.0f}}};
        // One x4 ldmatrix covers both 8-key halves of the B fragment; lanes 8g..8g+7 provide the
        // row addresses of matrix g.
        const __nv_bfloat16* kb4 = reinterpret_cast<const __nv_bfloat16*>(kvs) +
                                   (warp * 16 + (lane & 7) + ((lane >> 4) << 3)) * S::kQStride +
                                   (((lane >> 3) & 1) << 3);
        uint32_t fq[4], fkb[2][2];
        hmma_load_a_16x16(reinterpret_cast<const __nv_bfloat16*>(qsm), S::kQStride, fq);
        hmma_ldsm_x4(fkb, kb4);
#pragma unroll 5
        for (int kk = 0; kk < kKvlora + kRope - 16;
             kk += 16) {  // last step peeled: no guards in the loop
          uint32_t nfq[4], nfkb[2][2];
          hmma_load_a_16x16(reinterpret_cast<const __nv_bfloat16*>(qsm) + kk + 16, S::kQStride,
                            nfq);
          hmma_ldsm_x4(nfkb, kb4 + kk + 16);
          hmma(acc2[0], fq, fkb[0]);
          hmma(acc2[1], fq, fkb[1]);
#pragma unroll
          for (int r = 0; r < 4; ++r) fq[r] = nfq[r];
          fkb[0][0] = nfkb[0][0];
          fkb[0][1] = nfkb[0][1];
          fkb[1][0] = nfkb[1][0];
          fkb[1][1] = nfkb[1][1];
        }
        hmma(acc2[0], fq, fkb[0]);
        hmma(acc2[1], fq, fkb[1]);
        const int orow = lane >> 2, opair = lane & 3;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int ocol = warp * 16 + half * 8 + 2 * opair;
          *reinterpret_cast<float2*>(sc + orow * kSplitRows + ocol) =
              make_float2(acc2[half].value[0], acc2[half].value[1]);
          if (orow + 8 < kHeads) {
            *reinterpret_cast<float2*>(sc + (orow + 8) * kSplitRows + ocol) =
                make_float2(acc2[half].value[2], acc2[half].value[3]);
          }
        }
      }
      // zero the probability pad rows 10..15 once
      {
        uint16_t* Ps = reinterpret_cast<uint16_t*>(smem + S::kProb);
        for (int i = tid; i < 6 * S::kPStride; i += nt) Ps[kHeads * S::kPStride + i] = 0;
      }
      __syncthreads();
      // Keep local exponentials in FP32 until every active split's (max,sum)
      // is available. O_ACC has no PV result yet: its first twenty floats
      // temporarily hold these statistics and are overwritten by this
      // split's PV below. Inactive O_ACC slabs remain untouched.
      if (warp < kHeads) {
        const int head = warp;
        const int k0 = lane;
        const int k1 = lane + 32;
        const bool valid0 = ids_s[k0] >= 0 && ids_s[k0] <= cur_pos;
        const bool valid1 = ids_s[k1] >= 0 && ids_s[k1] <= cur_pos;
        const float s0 = valid0 ? sc[head * kSplitRows + k0] : neg_inf();
        const float s1 = valid1 ? sc[head * kSplitRows + k1] : neg_inf();
        const float m = q1_detail::k3_warp_max_broadcast(fmaxf(s0, s1));
        const bool live = m > neg_inf();
        const float e0 = (live && valid0) ? exp2f(__fmul_rn(__fadd_rn(s0, -m), kExp2Scale)) : 0.0f;
        const float e1 = (live && valid1) ? exp2f(__fmul_rn(__fadd_rn(s1, -m), kExp2Scale)) : 0.0f;
        sc[head * kSplitRows + k0] = e0;
        sc[head * kSplitRows + k1] = e1;
        float l = __fadd_rn(e0, e1);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, off));
        if (lane == 0) {
          *reinterpret_cast<float2*>(T.r5.o_acc + static_cast<size_t>(split) * kOAccSplitElems +
                                     2 * head) = make_float2(m, l);
          T.r5.lse_acc[static_cast<size_t>(split) * kLseStride + head] =
              live ? fmaf(m, kExp2Scale, log2f(l)) : neg_inf();
        }
      }
      word_arrive(B + kQ1StatsReady, 1u);
      word_wait_ge(B + kQ1StatsReady, static_cast<uint32_t>(n_blocks));
      if (warp < kHeads) {
        const int head = warp;
        float2 stat = make_float2(neg_inf(), 0.0f);
        if (lane < n_blocks) {
          stat = __ldcg(reinterpret_cast<const float2*>(
              T.r5.o_acc + static_cast<size_t>(lane) * kOAccSplitElems + 2 * head));
        }
        const float global_max = q1_detail::k3_warp_max_broadcast(stat.x);
        const bool live = stat.x > neg_inf();
        const float weight =
            live ? exp2f(__fmul_rn(__fsub_rn(stat.x, global_max), kExp2Scale)) : 0.0f;
        const float term = live ? __fmul_rn(stat.y, weight) : 0.0f;
        const float denominator = q1_detail::k3_warp_sum_broadcast(term);
        const float inverse = denominator > 0.0f ? __frcp_rn(denominator) : 0.0f;
        const float own_weight = __shfl_sync(0xffffffffu, weight, split);
        const int k0 = lane, k1 = lane + 32;
        const float p0 = inverse > 0.0f
                             ? __fmul_rn(__fmul_rn(sc[head * kSplitRows + k0], own_weight), inverse)
                             : 0.0f;
        const float p1 = inverse > 0.0f
                             ? __fmul_rn(__fmul_rn(sc[head * kSplitRows + k1], own_weight), inverse)
                             : 0.0f;
        uint16_t* Ps = reinterpret_cast<uint16_t*>(smem + S::kProb);
        Ps[head * S::kPStride + k0] = to_bf16_bits(p0);
        Ps[head * S::kPStride + k1] = to_bf16_bits(p1);
      }
      // Every active split must finish reading the temporary statistics
      // before any PV writer reuses its O_ACC slab.
      word_arrive(B + kQ1StatsRead, 1u);
      word_wait_ge(B + kQ1StatsRead, static_cast<uint32_t>(n_blocks));
      // PV MMA: warps 0..7, A = kv^T (dims), B = globally normalized BF16 P.
      HmmaAcc out[4][2] = {};
      if (warp < 8) {
        const uint16_t* kvs = reinterpret_cast<const uint16_t*>(arena);
        const uint16_t* Ps = reinterpret_cast<const uint16_t*>(smem + S::kProb);
#pragma unroll
        for (int kk = 0; kk < mma_rows; kk += 16) {
          uint32_t fp2[2][2];
          const int shead = lane & 7;
          const int skh = ((lane >> 3) & 1) * 8;
          const int shh = lane >> 4;
          hmma_ldsm_x4(fp2, reinterpret_cast<const __nv_bfloat16*>(Ps) +
                                (shh * 8 + shead) * S::kPStride + kk + skh);
          uint32_t fv[4][4];
#pragma unroll
          for (int tile = 0; tile < 4; ++tile) {
            hmma_load_a_16x16_trans(reinterpret_cast<const __nv_bfloat16*>(kvs) + kk * S::kQStride +
                                        (warp + tile * 8) * 16,
                                    S::kQStride, fv[tile]);
          }
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int tile = 0; tile < 4; ++tile) hmma(out[tile][hh], fv[tile], fp2[hh]);
          }
        }
      }
      __syncthreads();  // the kv rows are read by the PV warps until here; the arena below is dead
      // O_ACC via a [10][512] f32 stage in the dead kv area: the MMA-fragment scatter lands in
      // shared

      // fewer write-allocate round trips and full-sector lines for the combine's reads).
      {
        float* ostage = reinterpret_cast<float*>(arena);
        if (warp < 8) {
          const int drow = lane >> 2, hpair = 2 * (lane & 3);
#pragma unroll
          for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int h0 = hh * 8 + hpair;
              const int d0 = (warp + tile * 8) * 16 + drow;
              const HmmaAcc& v = out[tile][hh];
              if (h0 < kHeads) {
                ostage[h0 * kKvlora + d0] = v.value[0];
                ostage[h0 * kKvlora + d0 + 8] = v.value[2];
              }
              if (h0 + 1 < kHeads) {
                ostage[(h0 + 1) * kKvlora + d0] = v.value[1];
                ostage[(h0 + 1) * kKvlora + d0 + 8] = v.value[3];
              }
            }
          }
        }
        __syncthreads();
        float* oacc = T.r5.o_acc + static_cast<size_t>(split) * kOAccSplitElems;
        if (n_blocks == 1) {
          // CTA 0 owns all final normalized heads in the one-split regime.
          // Store the required FP32 partial and its BF16 projection together,
          // eliminating the C5 wait and O_ACC reload on the short-context path.
          for (int i = tid; i < (kHeads * kKvlora) / 4; i += nt) {
            const float4 v = reinterpret_cast<const float4*>(ostage)[i];
            reinterpret_cast<float4*>(oacc)[i] = v;
            reinterpret_cast<uint2*>(T.r5.post_o)[i] =
                make_uint2(static_cast<uint32_t>(to_bf16_bits(v.x)) |
                               (static_cast<uint32_t>(to_bf16_bits(v.y)) << 16),
                           static_cast<uint32_t>(to_bf16_bits(v.z)) |
                               (static_cast<uint32_t>(to_bf16_bits(v.w)) << 16));
          }
        } else {
          for (int i = tid; i < (kHeads * kKvlora) / 4; i += nt) {
            reinterpret_cast<float4*>(oacc)[i] = reinterpret_cast<const float4*>(ostage)[i];
          }
        }
      }
    } else {
      issue_bulk_page(arena + S::o_proj_off(cta, 0),
                      T.r7.o_proj + static_cast<size_t>(cta) * 5 * 0x6080, 0x6080,
                      s_early_bars + 9);
      issue_bulk_page(arena + S::o_proj_off(cta, 1),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 1) * 0x6080, 0x6080,
                      s_early_bars + 10);
      issue_bulk_page(arena + S::kUvPage, T.r6.w_uv + static_cast<size_t>(cta) * 0x4080, 0x4080,
                      s_early_bars + 11);
    }
    fast_arrive(B + kWordC5, cta == 0 ? kC5Head<1> : 1u);
    if (n_blocks != 1) word_wait_sign(B + kWordC5);
    // The remaining o_proj pages (the kv rows are dead after PV): short context
    // issues immediately after C5 arrival; multi-split waits for the C5 sign.
    issue_bulk_page(arena + S::o_proj_off(cta, 2),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 2) * 0x6080, 0x6080,
                    s_early_bars + 12);
    issue_bulk_page(arena + S::o_proj_off(cta, 3),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 3) * 0x6080, 0x6080,
                    s_early_bars + 13);
    issue_bulk_page(arena + S::o_proj_off(cta, 4),
                    T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 4) * 0x6080, 0x6080,
                    s_early_bars + 14);
    if (n_blocks != 1) {
      // Multi-split combine on 20 CTAs (cta = head * 2 + half, each covering 256 of the 512 dims):
      // the PV operands already contain global normalization, so combine sums the FP32 partials
      // without a second softmax weight.
      const int head = cta >> 1, half = cta & 1;
      if (head < kHeads) {
        uint16_t* po = T.r5.post_o + head * kKvlora;
        const float* oa = T.r5.o_acc + static_cast<size_t>(head) * kKvlora;
        const int d = half * (kKvlora / 2) + tid * 2;
        float2 staged[kSplits];
        if (tid < 128) {
#pragma unroll
          for (int s = 0; s < kSplits; ++s) {
            staged[s] = (s < n_blocks) ? __ldcg(reinterpret_cast<const float2*>(
                                             oa + static_cast<size_t>(s) * kOAccSplitElems + d))
                                       : make_float2(0.0f, 0.0f);
          }
        }
        if (tid < 128) {
          float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
          for (int s = 0; s < kSplits; ++s) {
            if (s < n_blocks) {
              acc.x = __fadd_rn(acc.x, staged[s].x);
              acc.y = __fadd_rn(acc.y, staged[s].y);
            }
          }
          *reinterpret_cast<uint32_t*>(po + d) = static_cast<uint32_t>(to_bf16_bits(acc.x)) |
                                                 (static_cast<uint32_t>(to_bf16_bits(acc.y)) << 16);
        }
      }
    }
    // CTA 0 is the sole one-split post_o writer, so its release RMW heads the C6 publication chain;
    // the other CTAs keep a relaxed arrival.
    if (n_blocks == 1 && cta == 0)
      word_arrive(B + kWordC6, kWeightC6<1>);
    else
      fast_arrive(B + kWordC6, kWeightC6<1>);
  } else {
    // CTAs 32..79 delayed only u3/u4 because those ranges overlap the W_UK
    // slot.  The other four pages were already launched just after C1.
    if (cta < kUkCtas) {
      issue_bulk_page(arena + S::o_proj_off(cta, 3),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 3) * 0x6080, 0x6080,
                      s_early_bars + 13);
      issue_bulk_page(arena + S::o_proj_off(cta, 4),
                      T.r7.o_proj + (static_cast<size_t>(cta) * 5 + 4) * 0x6080, 0x6080,
                      s_early_bars + 14);
    }
  }
  word_wait_ge(B + kWordC6, kSyncFull);

  // ------------------------------------------------------------------ P6: W_UV (CTAs 0..79, 32
  // rows, K = 512) -> C7
  {
    const bool work = cta < kUvCtas;
    if (work && tid < 64) {  // stage post_o[head][512]
      const int head = cta >> 3;
      *reinterpret_cast<uint4*>(smem + S::kAct + tid * 16) =
          __ldcg(reinterpret_cast<const uint4*>(T.r6.alias_post_o + head * kKvlora + tid * 8));
    }
    wait_bulk_page(s_early_bars + 11);  // W_UV page landed (o_proj pages may still be in flight)
    __syncthreads();
    if (work && warp < 8) {
      const uint32_t slot = arena_s + S::kUvPage;
      const int bblk = warp >> 1;
      const int half = warp & 1;
      const uint32_t a_h =
          slot + static_cast<uint32_t>(half) * 0x2000u + static_cast<uint32_t>(lane) * 8u;
      RawFp8x8 raw[8];
      uint32_t bb[8][2];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        raw[t] = lds_fp8x8(a_h + static_cast<uint32_t>(bblk * 8 + t) * 0x100u);
        act_frag_q1(act_s, bblk * 128 + t * 16, lane, bb[t]);
      }
      HmmaAcc d{{0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        uint32_t A[4];
        fp8_to_bf16_fragment(raw[t], A);
        hmma(d, A, bb[t]);
      }
      if ((lane & 3) == 0) {
        const int row = half * 16 + (lane >> 2);
        partial[bblk * 48 + row] = d.value[0];
        partial[bblk * 48 + row + 8] = d.value[2];
      }
    }
    __syncthreads();
    if (work && tid < 32) {
      float acc = 0.0f;
#pragma unroll
      for (int bblk = 0; bblk < 4; ++bblk) {
        const float scv =
            lds_f32(arena_s + S::kUvPage + 0x4000u + static_cast<uint32_t>(bblk) * 4u);
        acc = fmaf(scv, partial[bblk * 48 + tid], acc);
      }
      T.r6.post_uv[cta * 32 + tid] = to_bf16_bits(acc);
    }
  }
  fast_arrive(B + kWordC7, kWeightGroup1);
  word_wait_ge(B + kWordC7, kSyncFull);

  uint16_t residual_reg = 0;
  if (tid < kOProjRows) {
    residual_reg = __ldcg(T.r7.residual + cta * kOProjRows + tid);
  }

  // ------------------------------------------------------------------ P7: o_proj (48 rows, K =
  // 2560) + K5 collective
  uint16_t* act16 = reinterpret_cast<uint16_t*>(smem + S::kAct);
  uint16_t* local = act16 + kHeads * kVdim;  // bf16 [48]
  uint16_t* rx = local + kOProjRows;         // bf16 [8][48]
  {
    // Start both independent inputs before waiting for the weight pages.  The
    // compute warps stage post_uv while warps 8..11 fetch the already
    // complete single-rank inbox image; both overlap the tail of o_proj TMA.
    if (warp < 8) {
      for (int i = tid; i < (kHeads * kVdim) / 8; i += 256) {
        *reinterpret_cast<uint4*>(act16 + i * 8) =
            __ldcg(reinterpret_cast<const uint4*>(T.r7.alias_post_uv + i * 8));
      }
    } else if (nt >= 320) {
      // The receive is done AFTER the push: a live multi-rank K5 all-reduce must wait for every
      // peer to publish before reading.  Warps 8..11 idle here; the poll-until-tag receive follows
      // below.
    }
    wait_bulk_page(s_early_bars + 9);
    wait_bulk_page(s_early_bars + 10);
    wait_bulk_page(s_early_bars + 12);
    wait_bulk_page(s_early_bars + 13);
    wait_bulk_page(s_early_bars + 14);
    __syncthreads();  // all five o_proj pages are CTA-visible
    if (warp < 8 || nt < 320) {
      // GEMV on the 8 consumer warps (warp = K-block slot; act fragments shared across the three
      // row tiles, raw fragments per tile).  Named barrier 1 orders stage -> MMA -> fold across
      // exactly these 256 threads so the concurrent K5 receive on warps 8..11 never gates the math.
      if (warp < 8) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        float2 ru[3] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
#pragma unroll
        for (int jj = 0; jj < 3; ++jj) {
          const int j = warp + jj * 8;
          if (j < 20) {
            const int p = j >> 2, bblk = j & 3;
            const uint32_t slot = arena_s + static_cast<uint32_t>(S::o_proj_off(cta, p));
            uint32_t bb[8][2];
#pragma unroll
            for (int t = 0; t < 8; ++t)
              act_frag_q1(act_s, p * 512 + bblk * 128 + t * 16, lane, bb[t]);
#pragma unroll
            for (int rt = 0; rt < 3; ++rt) {
              const uint32_t base = slot + static_cast<uint32_t>(rt) * 0x2000u;
              RawFp8x8 raw[8];
#pragma unroll
              for (int t = 0; t < 8; ++t) {
                raw[t] = lds_fp8x8(base + static_cast<uint32_t>(bblk * 8 + t) * 0x100u +
                                   static_cast<uint32_t>(lane) * 8u);
              }
              const float scl = lds_f32(slot + 0x6000u + static_cast<uint32_t>(4 * rt + bblk) * 4u);
              HmmaAcc d{{0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
              for (int t = 0; t < 8; ++t) {
                uint32_t A[4];
                fp8_to_bf16_fragment(raw[t], A);
                hmma(d, A, bb[t]);
              }
              ru[rt].x = fmaf(scl, d.value[0], ru[rt].x);
              ru[rt].y = fmaf(scl, d.value[2], ru[rt].y);
            }
          }
        }
        if ((lane & 3) == 0) {
          float* p = partial + warp * 48;
          const int r = lane >> 2;
#pragma unroll
          for (int rt = 0; rt < 3; ++rt) {
            p[rt * 16 + r] = ru[rt].x;
            p[rt * 16 + 8 + r] = ru[rt].y;
          }
        }
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        if (tid < kOProjRows) {
          float acc = 0.0f;
#pragma unroll
          for (int w = 0; w < 8; ++w) acc = __fadd_rn(acc, partial[w * 48 + tid]);
          local[tid] = to_bf16_bits(acc);
        }
      }
      if (nt < 320) {  // narrow-block fallback: serial receive after the GEMV
        __syncthreads();
        k5_receive_serial(args, rank, parity, e32, cta, rx, nt, tid);
      }
    }
  }
  __syncthreads();  // local[] and rx[] are read below

  const int arank = static_cast<int>(args.rank);

  if (tid < (kWorld - 1) * 8) {
    const int line = tid & 7;
    const int p = tid >> 3;
    const int dst = p + (p >= arank ? 1 : 0);
    const int slot = arank - (arank > dst ? 1 : 0);
    uint8_t* blk = args.peers[dst] + static_cast<size_t>(slot) * kInboxSlotStride +
                   parity * kInboxParityStride + static_cast<size_t>(cta) * kInboxCtaStride;
    const uint16_t* srcp = local + line * 6;
    const uint32_t w0 = static_cast<uint32_t>(srcp[0]) | (static_cast<uint32_t>(srcp[1]) << 16);
    const uint32_t w1 = static_cast<uint32_t>(srcp[2]) | (static_cast<uint32_t>(srcp[3]) << 16);
    const uint32_t w2 = static_cast<uint32_t>(srcp[4]) | (static_cast<uint32_t>(srcp[5]) << 16);
    const uint32_t w3 = e32;

    st_relaxed_sys_b128(blk + line * 16, w0, w1, w2, w3);
  }
  __syncthreads();
  // receive: own inbox peers[arank], slot s <- rank s + (s >= arank); lane (s, line) polls its own
  // line until dword 3 == E and keeps the six bf16 of that same 16-B load, so no line is used
  // before its own write is visible
  if (tid < (kWorld - 1) * 8) {
    const int line = tid & 7;
    const int s = tid >> 3;
    const int src = s + (s >= arank ? 1 : 0);
    const uint8_t* lp = args.peers[arank] + static_cast<size_t>(s) * kInboxSlotStride +
                        parity * kInboxParityStride + static_cast<size_t>(cta) * kInboxCtaStride +
                        line * 16;
    uint64_t lo = 0, hi = 0;
    do {
      ld_relaxed_sys_b128(lp, lo, hi);
    } while (static_cast<uint32_t>(hi >> 32) != e32);
    uint16_t* dstp = rx + src * kOProjRows + line * 6;
    dstp[0] = static_cast<uint16_t>(lo);
    dstp[1] = static_cast<uint16_t>(lo >> 16);
    dstp[2] = static_cast<uint16_t>(lo >> 32);
    dstp[3] = static_cast<uint16_t>(lo >> 48);
    dstp[4] = static_cast<uint16_t>(hi);
    dstp[5] = static_cast<uint16_t>(hi >> 16);
  }
  __syncthreads();
  for (int i = tid; i < kOProjRows; i += nt) {
    float acc = 0.0f;
    for (int r = 0; r < kWorld; ++r) {
      const uint16_t v = (r == arank) ? local[i] : rx[r * kOProjRows + i];
      acc = __fadd_rn(acc, bf(v));
    }
    const int col = cta * kOProjRows + i;
    T.r7.hidden_out[col] = to_bf16_bits(__fadd_rn(acc, bf(residual_reg)));
  }
}

}  // namespace flashinfer::glm5_tp8::mla_q1

#include "launch_util.cuh"
namespace flashinfer::glm5_tp8 {
KernelResources initialize_mla_q1(int device) {
  using namespace mla_q1;
  return attention_detail::configure(reinterpret_cast<const void*>(&q1_attention_mode_fast_kernel),
                                     kGrid, kFastBlock, kLaunchDynSmem, device);
}
cudaError_t launch_mla_q1(const AttentionArguments& p, cudaStream_t stream, bool pdl,
                          bool cooperative) {
  using namespace mla_q1;
  LaunchArgs a{};
  a.n_rows = 8;
  a.rank = p.rank;
  a.context = reinterpret_cast<const uint8_t*>(p.context);
  a.sync_words = reinterpret_cast<uint32_t*>(p.q1_state);
  a.launch_ctr = reinterpret_cast<uint32_t*>(p.q1_state + attention_q1::kQ1Counters);
  a.reset_token = reinterpret_cast<uint32_t*>(p.q1_state + attention_q1::kQ1ResetToken);
  for (int r = 0; r < 8; ++r) a.peers[r] = p.peer_records[r];
  return attention_detail::launch(reinterpret_cast<const void*>(&q1_attention_mode_fast_kernel), a,
                                  kGrid, kFastBlock, kLaunchDynSmem, stream, pdl, cooperative);
}
}  // namespace flashinfer::glm5_tp8
