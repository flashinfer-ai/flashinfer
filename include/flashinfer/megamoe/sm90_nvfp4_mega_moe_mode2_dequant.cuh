#pragma once

#include <cutlass/arch/barrier.h>

#include <flashinfer/megamoe/nvfp4_dequant.cuh>

namespace flashinfer::megamoe::nvfp4 {

#ifndef FI_MEGAMOE_NVFP4_LUT_COMPACT
#define FI_MEGAMOE_NVFP4_LUT_COMPACT 0
#endif

// Scale-LUT gather. Compact mode (FI_MEGAMOE_NVFP4_LUT_COMPACT) halves the gathered entry
// from 8 B to 4 B: the E2M1 magnitudes hide two doubling chains (0.5->1->2->4 and
// 1.5->3->6), one doubling = one e4m3 exponent step = +0x08, so the high word is the
// low word's bytes (2,3,2,3) plus (0x08,0x08,0x10,0x10). The largest byte is
// 0x7f + 0x10 = 0x8f < 0x100, so the packed add cannot carry across byte lanes, and a
// per-byte clamp to 0x7f reproduces the table's saturation exactly. Bit-exact for every
// scale code >= 8 (verified exhaustively against all 128 rows); the eight
// subnormal-scale rows -- where e4m3 spacing is arithmetic, not geometric, and the
// chain breaks -- come verbatim from a 32-B side table stored right after lo[128],
// loaded only when some lane in the warp actually holds such a code. Callers are
// whole warps (the dequant teams), so the warp-wide vote is safe.
// Measured motivation: the 8-B gather over the 1024-B table costs ~64 base + ~20
// conflict wavefronts per K-stage, 11.3% of the L1 SMEM pipe (dose-response probe).
__device__ __forceinline__ uint2 load_nvfp4_lut(const uint2* __restrict__ lut_smem,
                                                const uint32_t idx) {
#if FI_MEGAMOE_NVFP4_LUT_COMPACT
  const auto* lo_tbl = reinterpret_cast<const uint32_t*>(lut_smem);
  const uint32_t lo = lo_tbl[idx];
  uint32_t hi = __byte_perm(lo, 0u, 0x3232u) + 0x10100808u;
  const uint32_t sat = hi & 0x80808080u;
  hi = (hi | (sat - (sat >> 7u))) & ~sat;
  if (__any_sync(0xffffffffu, idx < 8u)) {
    const uint32_t exc = lo_tbl[128u + (idx & 7u)];
    hi = idx < 8u ? exc : hi;
  }
  return make_uint2(lo, hi);
#else
  return lut_smem[idx];
#endif
}

// Reconstruct the straight NVFP4-scale -> E4M3 magnitude table in registers.
// For normal UE4M3 codes the low four entries are
//   {0, round_even(scale / 2), scale, round_even(3 * scale / 2)}
// and the high four are the same two non-zero chains advanced by one/two E4M3
// exponent steps.  This is the direct-scale MegaMoE table, not Humming's
// deliberately /6-adjusted helper.  Codes 0..7 are UE4M3 subnormals, where
// exponent stepping is not valid, so they retain the exact shared-LUT path.
// Code 0x7f is MegaMoE's documented saturating-448 fallback.

__device__ __forceinline__ uint2 dequant_mode2_lop3_word(const uint32_t packed, const uint2& lut) {
  return detail::decode_fp4_word(packed, lut);
}

template <int kQuad, bool kHalfStream = false, bool kProducerMBarrier = false,
          bool kWarpLeaderMBarrier = false, bool kPostPublishLut = false>
__device__ __forceinline__ void dequant_mode2_lop3_two_rows_lut_window(
    uint8_t* __restrict__ fp8_dst0, uint8_t* __restrict__ fp8_dst1, const uint4 (&fp4_quads0)[4],
    const uint4 (&fp4_quads1)[4], const uint2& scale_words0, const uint2& scale_words1,
    const uint32_t row_swizzle, const uint2* __restrict__ lut_smem, const uint2 lut00,
    const uint2 lut10, const uint2 lut01, const uint2 lut11,
    const uint32_t half_ready_barrier_0 = 0, const uint32_t half_ready_barrier_1 = 0,
    const uint32_t half_ready_threads = 0, const uint32_t producer_barrier_idx = 0,
    const uint32_t producer_threads = 0, const uint32_t dequant_tid = 0,
    const cutlass::arch::ClusterTransactionBarrier* half_ready_mbarrier_0 = nullptr,
    const cutlass::arch::ClusterTransactionBarrier* half_ready_mbarrier_1 = nullptr) {
  uint2 next_lut00;
  uint2 next_lut10;
  uint2 next_lut01;
  uint2 next_lut11;
  if constexpr (kQuad + 1 < 4 and not(kPostPublishLut and kQuad == 1)) {
    constexpr int kNextScaleI0 = (kQuad + 1) * 2;
    constexpr int kNextScaleI1 = kNextScaleI0 + 1;
    const uint32_t next_scale_word0 = kQuad + 1 < 2 ? scale_words0.x : scale_words0.y;
    const uint32_t next_scale_word1 = kQuad + 1 < 2 ? scale_words1.x : scale_words1.y;
    const uint32_t next_scale00 = (next_scale_word0 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
    const uint32_t next_scale10 = (next_scale_word1 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
    const uint32_t next_scale01 = (next_scale_word0 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
    const uint32_t next_scale11 = (next_scale_word1 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
    next_lut00 = load_nvfp4_lut(lut_smem, next_scale00);
    next_lut10 = load_nvfp4_lut(lut_smem, next_scale10);
    next_lut01 = load_nvfp4_lut(lut_smem, next_scale01);
    next_lut11 = load_nvfp4_lut(lut_smem, next_scale11);
  }

  const uint4 q0 = fp4_quads0[kQuad];
  const uint4 q1 = fp4_quads1[kQuad];
  constexpr int kScaleI0 = kQuad * 2;
  constexpr int kScaleI1 = kScaleI0 + 1;
  const uint2 q0x = dequant_mode2_lop3_word(q0.x, lut00);
  const uint2 q0y = dequant_mode2_lop3_word(q0.y, lut00);
  const uint2 q1x = dequant_mode2_lop3_word(q1.x, lut10);
  const uint2 q1y = dequant_mode2_lop3_word(q1.y, lut10);
  *reinterpret_cast<uint4*>(fp8_dst0 + ((kScaleI0 * 16) ^ row_swizzle)) =
      make_uint4(q0x.x, q0x.y, q0y.x, q0y.y);
  *reinterpret_cast<uint4*>(fp8_dst1 + ((kScaleI0 * 16) ^ row_swizzle)) =
      make_uint4(q1x.x, q1x.y, q1y.x, q1y.y);

  const uint2 q0z = dequant_mode2_lop3_word(q0.z, lut01);
  const uint2 q0w = dequant_mode2_lop3_word(q0.w, lut01);
  const uint2 q1z = dequant_mode2_lop3_word(q1.z, lut11);
  const uint2 q1w = dequant_mode2_lop3_word(q1.w, lut11);
  *reinterpret_cast<uint4*>(fp8_dst0 + ((kScaleI1 * 16) ^ row_swizzle)) =
      make_uint4(q0z.x, q0z.y, q0w.x, q0w.y);
  *reinterpret_cast<uint4*>(fp8_dst1 + ((kScaleI1 * 16) ^ row_swizzle)) =
      make_uint4(q1z.x, q1z.y, q1w.x, q1w.y);

  if constexpr ((kHalfStream or kProducerMBarrier) and kQuad == 1) {
    // The FP8 stores above use the generic proxy while WGMMA reads through
    // the async proxy. Modes 2/3 rendezvous with all math threads; mode 4
    // rendezvous only the 64 writers and lets one writer release an
    // mbarrier. Mode 5 replaces that CTA barrier with one warp barrier per
    // producer warp, then each warp leader contributes one release arrival
    // to an expected-count-2 mbarrier. Mode 6 publishes before loading the
    // first four second-half LUT entries; the compiler memory clobber and
    // the resulting LUT-to-PRMT dependency keep all second-half work after
    // publication without the identity MOV gate used by modes 2--5.
    cutlass::arch::fence_view_async_shared();
    if constexpr (kPostPublishLut) {
      asm volatile("barrier.sync %0, %1;"
                   :
                   : "r"(producer_barrier_idx), "r"(producer_threads)
                   : "memory");
      if (dequant_tid == 0u) half_ready_mbarrier_0->arrive();

      // kQuad==1 deferred exactly the kQuad==2 LUT window. Load it only
      // after the first-half release so math can begin consuming K0:64.
      constexpr int kNextScaleI0 = 4;
      constexpr int kNextScaleI1 = 5;
      const uint32_t next_scale_word0 = scale_words0.y;
      const uint32_t next_scale_word1 = scale_words1.y;
      const uint32_t next_scale00 = (next_scale_word0 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
      const uint32_t next_scale10 = (next_scale_word1 >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
      const uint32_t next_scale01 = (next_scale_word0 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
      const uint32_t next_scale11 = (next_scale_word1 >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
      next_lut00 = load_nvfp4_lut(lut_smem, next_scale00);
      next_lut10 = load_nvfp4_lut(lut_smem, next_scale10);
      next_lut01 = load_nvfp4_lut(lut_smem, next_scale01);
      next_lut11 = load_nvfp4_lut(lut_smem, next_scale11);
    } else {
      uint2 gated_lut00, gated_lut10, gated_lut01, gated_lut11;
      if constexpr (kWarpLeaderMBarrier) {
        // bar.warp.sync orders every lane's preceding stores before
        // its warp leader's release arrival. Completion after both
        // leaders arrive is acquired by all math consumers.
        asm volatile(
            "bar.warp.sync 0xffffffff;\n"
            "mov.b32 %0, %8;\n"
            "mov.b32 %1, %9;\n"
            "mov.b32 %2, %10;\n"
            "mov.b32 %3, %11;\n"
            "mov.b32 %4, %12;\n"
            "mov.b32 %5, %13;\n"
            "mov.b32 %6, %14;\n"
            "mov.b32 %7, %15;"
            : "=r"(gated_lut00.x), "=r"(gated_lut00.y), "=r"(gated_lut10.x), "=r"(gated_lut10.y),
              "=r"(gated_lut01.x), "=r"(gated_lut01.y), "=r"(gated_lut11.x), "=r"(gated_lut11.y)
            : "r"(next_lut00.x), "r"(next_lut00.y), "r"(next_lut10.x), "r"(next_lut10.y),
              "r"(next_lut01.x), "r"(next_lut01.y), "r"(next_lut11.x), "r"(next_lut11.y)
            : "memory");
      } else {
        const uint32_t sync_barrier =
            kProducerMBarrier ? producer_barrier_idx : half_ready_barrier_0;
        const uint32_t sync_threads = kProducerMBarrier ? producer_threads : half_ready_threads;
        asm volatile(
            "barrier.sync %8, %9;\n"
            "mov.b32 %0, %10;\n"
            "mov.b32 %1, %11;\n"
            "mov.b32 %2, %12;\n"
            "mov.b32 %3, %13;\n"
            "mov.b32 %4, %14;\n"
            "mov.b32 %5, %15;\n"
            "mov.b32 %6, %16;\n"
            "mov.b32 %7, %17;"
            : "=r"(gated_lut00.x), "=r"(gated_lut00.y), "=r"(gated_lut10.x), "=r"(gated_lut10.y),
              "=r"(gated_lut01.x), "=r"(gated_lut01.y), "=r"(gated_lut11.x), "=r"(gated_lut11.y)
            : "r"(sync_barrier), "r"(sync_threads), "r"(next_lut00.x), "r"(next_lut00.y),
              "r"(next_lut10.x), "r"(next_lut10.y), "r"(next_lut01.x), "r"(next_lut01.y),
              "r"(next_lut11.x), "r"(next_lut11.y));
      }
      next_lut00 = gated_lut00;
      next_lut10 = gated_lut10;
      next_lut01 = gated_lut01;
      next_lut11 = gated_lut11;
      if constexpr (kProducerMBarrier) {
        if constexpr (kWarpLeaderMBarrier) {
          if ((dequant_tid & 31u) == 0u) half_ready_mbarrier_0->arrive();
        } else if (dequant_tid == 0u) {
          half_ready_mbarrier_0->arrive();
        }
      }
    }
  } else if constexpr ((kHalfStream or kProducerMBarrier) and kQuad == 3) {
    cutlass::arch::fence_view_async_shared();
    if constexpr (kWarpLeaderMBarrier) {
      asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
    } else {
      const uint32_t sync_barrier = kProducerMBarrier ? producer_barrier_idx : half_ready_barrier_1;
      const uint32_t sync_threads = kProducerMBarrier ? producer_threads : half_ready_threads;
      asm volatile("barrier.sync %0, %1;" : : "r"(sync_barrier), "r"(sync_threads));
    }
    if constexpr (kProducerMBarrier) {
      if constexpr (kWarpLeaderMBarrier) {
        if ((dequant_tid & 31u) == 0u) half_ready_mbarrier_1->arrive();
      } else if (dequant_tid == 0u) {
        half_ready_mbarrier_1->arrive();
      }
    }
  }

  if constexpr (kQuad + 1 < 4) {
    dequant_mode2_lop3_two_rows_lut_window<kQuad + 1, kHalfStream, kProducerMBarrier,
                                           kWarpLeaderMBarrier, kPostPublishLut>(
        fp8_dst0, fp8_dst1, fp4_quads0, fp4_quads1, scale_words0, scale_words1, row_swizzle,
        lut_smem, next_lut00, next_lut10, next_lut01, next_lut11, half_ready_barrier_0,
        half_ready_barrier_1, half_ready_threads, producer_barrier_idx, producer_threads,
        dequant_tid, half_ready_mbarrier_0, half_ready_mbarrier_1);
  }
}

template <uint32_t kNumDequantThreads, uint32_t kBarIdx, bool kSyncAfter = false,
          uint32_t kFusedRowBytes = 80, bool kHalfStream = false, bool kProducerMBarrier = false,
          bool kWarpLeaderMBarrier = false, bool kPostPublishLut = false>
__device__ __forceinline__ void dequant_smem_b_inplace_two_rows_mode2_lop3(
    uint8_t* __restrict__ smem_b, const uint32_t tid, const uint2* __restrict__ lut_smem,
    const uint32_t half_ready_barrier_0 = 0, const uint32_t half_ready_barrier_1 = 0,
    const uint32_t half_ready_threads = 0,
    const cutlass::arch::ClusterTransactionBarrier* half_ready_mbarrier_0 = nullptr,
    const cutlass::arch::ClusterTransactionBarrier* half_ready_mbarrier_1 = nullptr) {
  static_assert(not(kSyncAfter and (kHalfStream or kProducerMBarrier)),
                "half-ready handoff replaces the final writer rendezvous");
  static_assert(not(kHalfStream and kProducerMBarrier), "select exactly one half-ready protocol");
  static_assert(not kWarpLeaderMBarrier or kProducerMBarrier,
                "warp-leader publication requires producer mbarriers");
  static_assert(not kPostPublishLut or (kProducerMBarrier and not kWarpLeaderMBarrier),
                "post-publish LUT scheduling requires mode-4 mbarriers");
  const uint32_t row0 = tid;
  const uint32_t row1 = tid + kNumDequantThreads;
  const uint8_t* __restrict__ row_ptr0 = smem_b + row0 * kFusedRowBytes;
  const uint8_t* __restrict__ row_ptr1 = smem_b + row1 * kFusedRowBytes;
  const uint2 scale_words0 = *reinterpret_cast<const uint2*>(row_ptr0 + 64);
  const uint2 scale_words1 = *reinterpret_cast<const uint2*>(row_ptr1 + 64);
  const uint32_t scale00 = scale_words0.x & 0x7fu;
  const uint32_t scale10 = scale_words1.x & 0x7fu;
  const uint32_t scale01 = (scale_words0.x >> 8) & 0x7fu;
  const uint32_t scale11 = (scale_words1.x >> 8) & 0x7fu;
  const uint2 lut00 = load_nvfp4_lut(lut_smem, scale00);
  const uint2 lut10 = load_nvfp4_lut(lut_smem, scale10);
  const uint2 lut01 = load_nvfp4_lut(lut_smem, scale01);
  const uint2 lut11 = load_nvfp4_lut(lut_smem, scale11);
  const uint4* __restrict__ fp4_src0 = reinterpret_cast<const uint4*>(row_ptr0);
  const uint4* __restrict__ fp4_src1 = reinterpret_cast<const uint4*>(row_ptr1);
  uint4 fp4_quads0[4];
  uint4 fp4_quads1[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fp4_quads0[i] = fp4_src0[i];
    fp4_quads1[i] = fp4_src1[i];
  }

  asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));

  uint8_t* __restrict__ fp8_dst0 = smem_b + row0 * 128;
  uint8_t* __restrict__ fp8_dst1 = smem_b + row1 * 128;
  const uint32_t row_swizzle = (tid & 7u) << 4;
  dequant_mode2_lop3_two_rows_lut_window<0, kHalfStream, kProducerMBarrier, kWarpLeaderMBarrier,
                                         kPostPublishLut>(
      fp8_dst0, fp8_dst1, fp4_quads0, fp4_quads1, scale_words0, scale_words1, row_swizzle, lut_smem,
      lut00, lut10, lut01, lut11, half_ready_barrier_0, half_ready_barrier_1, half_ready_threads,
      kBarIdx, kNumDequantThreads, tid, half_ready_mbarrier_0, half_ready_mbarrier_1);
  if constexpr (kSyncAfter) {
    // Every writer must publish its generic-proxy stores before the
    // writer-set rendezvous allows one thread to signal the mbarrier.
    cutlass::arch::fence_view_async_shared();
    asm volatile("bar.sync %0, %1;" : : "n"(kBarIdx), "n"(kNumDequantThreads));
  }
}

// ============================================================================
// RS-form register-target decode (L1 RS-swapAB arm)
// ----------------------------------------------------------------------------
// Decodes the two packed FP4 words a math lane owns in one k32 slice straight
// into its four WGMMA A-fragment registers, with no SMEM write-back. For
// wgmma.m64nNk32 e4m3 the lane's fragment is (per k32 slice):
//   a0 = row r,     K nibble-group 4*(l%4)      a2 = a0 at K offset +16
//   a1 = row r + 8, K nibble-group 4*(l%4)      a3 = a1 at K offset +16
// `dequant_mode2_lop3_word` returns out_hi = K-consecutive elements 0..3 and
// out_lo = 4..7, so each half IS one fragment register. Lane pairs (l ^ 1)
// need the same two packed words but complementary rows: the even lane
// decodes row r, the odd lane row r + 8, and the discarded halves are
// exchanged over one shfl per word, so every word is decoded exactly once.
// `keep_hi` must be true on even lanes (own K half = elements 0..3) and
// false on odd lanes (elements 4..7).
__device__ __forceinline__ void dequant_mode2_lop3_rs_word_pair(
    const uint32_t w_lo, const uint32_t w_hi, const uint2& lut_lo, const uint2& lut_hi,
    const bool keep_hi, uint32_t (&a_frag)[4]) {
  detail::assemble_rs_word_pair(dequant_mode2_lop3_word(w_lo, lut_lo),
                                dequant_mode2_lop3_word(w_hi, lut_hi), keep_hi, a_frag);
}

// Debug fallback (decode-and-discard): the lane decodes both of its fragment
// rows' words itself and discards the complementary halves: no shfl, ~2x
// decode ALU and LUT traffic. Kept for bisecting shfl/register-lifetime
// issues; never the production arm.
__device__ __forceinline__ void dequant_mode2_lop3_rs_word_pair_noshfl(
    const uint32_t w_lo_r0, const uint32_t w_hi_r0, const uint32_t w_lo_r8, const uint32_t w_hi_r8,
    const uint2& lut_lo_r0, const uint2& lut_hi_r0, const uint2& lut_lo_r8, const uint2& lut_hi_r8,
    const bool keep_hi, uint32_t (&a_frag)[4]) {
  const uint2 d_lo_r0 = dequant_mode2_lop3_word(w_lo_r0, lut_lo_r0);
  const uint2 d_lo_r8 = dequant_mode2_lop3_word(w_lo_r8, lut_lo_r8);
  a_frag[0] = keep_hi ? d_lo_r0.x : d_lo_r0.y;
  a_frag[1] = keep_hi ? d_lo_r8.x : d_lo_r8.y;
  const uint2 d_hi_r0 = dequant_mode2_lop3_word(w_hi_r0, lut_hi_r0);
  const uint2 d_hi_r8 = dequant_mode2_lop3_word(w_hi_r8, lut_hi_r8);
  a_frag[2] = keep_hi ? d_hi_r0.x : d_hi_r0.y;
  a_frag[3] = keep_hi ? d_hi_r8.x : d_hi_r8.y;
}

}  // namespace flashinfer::megamoe::nvfp4
