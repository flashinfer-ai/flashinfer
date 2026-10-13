#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cstdint>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_atom.hpp>
#include <flashinfer/megamoe/barrier.cuh>
#include <flashinfer/megamoe/detail/common.cuh>
#include <flashinfer/megamoe/detail/mma_sm90.cuh>
#include <flashinfer/megamoe/detail/ptx.cuh>
#include <flashinfer/megamoe/detail/tma_copy.cuh>
#include <flashinfer/megamoe/layout.cuh>
#include <flashinfer/megamoe/nvfp4_dequant.cuh>
#include <flashinfer/megamoe/scheduler.cuh>
#include <flashinfer/megamoe/sym_buffer.cuh>
#include <type_traits>

// The persistent fused kernel launches one CTA per SM and derives its
// dispatch/combine strides and grid barriers from the SM count.  The JIT host
// defines MEGAMOE_NUM_SMS from the live device (132 on H200, 78 on H20-3e).
#ifndef MEGAMOE_NUM_SMS
#define MEGAMOE_NUM_SMS 132
#endif

namespace flashinfer::megamoe {
namespace nvfp4 {

__device__ __forceinline__ uint2 dequant_mode2_nibble_word(const uint32_t packed,
                                                           const uint2& lut) {
  return detail::decode_fp4_word(packed, lut);
}

// Decode the two packed words owned by one lane pair directly into the four
// register-source WGMMA A operands. This is the archived mode-5 mapping: the
// exact shared LUT is retained and packed-row padding bytes 72..79 are never
// read.
__device__ __forceinline__ void dequant_mode2_rs_word_pair(const uint32_t w_lo, const uint32_t w_hi,
                                                           const uint2& lut_lo, const uint2& lut_hi,
                                                           const bool keep_hi,
                                                           uint32_t (&a_frag)[4]) {
  detail::assemble_rs_word_pair(dequant_mode2_nibble_word(w_lo, lut_lo),
                                dequant_mode2_nibble_word(w_hi, lut_hi), keep_hi, a_frag);
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(uint8_t* __restrict__ fp8_dst,
                                                              const uint4 (&fp4_quads)[4],
                                                              const uint2& scale_words,
                                                              const uint32_t row_swizzle,
                                                              const uint2* __restrict__ lut_smem) {
#pragma unroll
  for (int quad_i = 0; quad_i < 4; ++quad_i) {
    const uint4 q = fp4_quads[quad_i];
    const uint32_t scale_word = quad_i < 2 ? scale_words.x : scale_words.y;
    const int scale_i0 = quad_i * 2;
    const int scale_i1 = scale_i0 + 1;
    const uint32_t scale0 = (scale_word >> ((scale_i0 & 3) * 8)) & 0x7fu;
    const uint32_t scale1 = (scale_word >> ((scale_i1 & 3) * 8)) & 0x7fu;
    const uint2 lut0 = lut_smem[scale0];
    const uint2 lut1 = lut_smem[scale1];

    const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
    const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
    if constexpr (!kQuadILP) {
      *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
          make_uint4(q0.x, q0.y, q1.x, q1.y);
    }

    const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
    const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
    if constexpr (kQuadILP) {
      *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
          make_uint4(q0.x, q0.y, q1.x, q1.y);
    }
    *reinterpret_cast<uint4*>(fp8_dst + ((scale_i1 * 16) ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
  }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
    uint8_t* __restrict__ smem_b, const uint8_t* __restrict__ packed_b, const uint32_t row,
    const uint2* __restrict__ lut_smem) {
  const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
  const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
  uint4 fp4_quads[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) fp4_quads[i] = fp4_src[i];
  const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
  dequant_mode2_nibble_row_regs<kQuadILP>(smem_b + row * 128, fp4_quads, scale_words,
                                          (row & 7u) << 4, lut_smem);
}

// Threads 0-127 and 128-255 each decode one K64 half of the same N128 tile,
// allowing the two M64 warpgroups to reuse the decoded weights.
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble_split_m(
    uint8_t* __restrict__ smem_b, const uint8_t* __restrict__ packed_b, const uint32_t thread_idx,
    const uint2* __restrict__ lut_smem) {
  const uint32_t row = thread_idx & 127u;
  const uint32_t k_half_idx = thread_idx >> 7;
  const uint8_t* __restrict__ row_ptr = packed_b + row * 80u;
  const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr + k_half_idx * 32u);
  const uint32_t scale_word =
      *reinterpret_cast<const uint32_t*>(row_ptr + 64u + k_half_idx * sizeof(uint32_t));
  uint8_t* __restrict__ fp8_dst = smem_b + row * 128u;
  const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
  for (uint32_t quad_i = 0; quad_i < 2; ++quad_i) {
    const uint4 q = fp4_src[quad_i];
    const uint32_t scale_i0 = quad_i * 2u;
    const uint32_t scale_i1 = scale_i0 + 1u;
    const uint32_t scale0 = (scale_word >> (scale_i0 * 8u)) & 0x7fu;
    const uint32_t scale1 = (scale_word >> (scale_i1 * 8u)) & 0x7fu;
    const uint2 lut0 = lut_smem[scale0];
    const uint2 lut1 = lut_smem[scale1];
    const uint2 q0 = dequant_mode2_nibble_word(q.x, lut0);
    const uint2 q1 = dequant_mode2_nibble_word(q.y, lut0);
    const uint2 q2 = dequant_mode2_nibble_word(q.z, lut1);
    const uint2 q3 = dequant_mode2_nibble_word(q.w, lut1);
    const uint32_t k_offset0 = k_half_idx * 64u + scale_i0 * 16u;
    const uint32_t k_offset1 = k_half_idx * 64u + scale_i1 * 16u;
    *reinterpret_cast<uint4*>(fp8_dst + (k_offset0 ^ row_swizzle)) =
        make_uint4(q0.x, q0.y, q1.x, q1.y);
    *reinterpret_cast<uint4*>(fp8_dst + (k_offset1 ^ row_swizzle)) =
        make_uint4(q2.x, q2.y, q3.x, q3.y);
  }
}

__device__ __forceinline__ uint2 dequant_braided_selector_word(const uint32_t braided,
                                                               const uint2& lut) {
  const uint32_t sel0 = braided & 0x00007777u;
  const uint32_t sel1 = (braided >> 16) & 0x00007777u;
  uint32_t out0 = byte_perm_unchecked(lut.x, lut.y, sel0);
  uint32_t out1 = byte_perm_unchecked(lut.x, lut.y, sel1);
  out0 |= braided & 0x80808080u;
  out1 |= (braided << 4) & 0x80808080u;
  return make_uint2(out0, out1);
}

__device__ __forceinline__ void dequant_braided_quad(uint8_t* __restrict__ fp8_dst, const uint4& q,
                                                     const uint2& lut0, const uint2& lut1,
                                                     const int scale_i0,
                                                     const uint32_t row_swizzle) {
  const uint2 q0 = dequant_braided_selector_word(q.x, lut0);
  const uint2 q1 = dequant_braided_selector_word(q.y, lut0);
  *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
      make_uint4(q0.x, q0.y, q1.x, q1.y);

  const uint2 q2 = dequant_braided_selector_word(q.z, lut1);
  const uint2 q3 = dequant_braided_selector_word(q.w, lut1);
  *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
      make_uint4(q2.x, q2.y, q3.x, q3.y);
}

__device__ __forceinline__ void dequant_braided_quad_ilp(uint8_t* __restrict__ fp8_dst,
                                                         const uint4& q, const uint2& lut0,
                                                         const uint2& lut1, const int scale_i0,
                                                         const uint32_t row_swizzle) {
  // Expose all four independent PRMT chains together so ptxas can overlap
  // their selector/sign work before either 128-bit shared-memory store.
  const uint32_t q0_sel0 = q.x & 0x00007777u;
  const uint32_t q0_sel1 = (q.x >> 16) & 0x00007777u;
  const uint32_t q1_sel0 = q.y & 0x00007777u;
  const uint32_t q1_sel1 = (q.y >> 16) & 0x00007777u;
  const uint32_t q2_sel0 = q.z & 0x00007777u;
  const uint32_t q2_sel1 = (q.z >> 16) & 0x00007777u;
  const uint32_t q3_sel0 = q.w & 0x00007777u;
  const uint32_t q3_sel1 = (q.w >> 16) & 0x00007777u;

  uint32_t q0_out0 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel0);
  uint32_t q0_out1 = byte_perm_unchecked(lut0.x, lut0.y, q0_sel1);
  uint32_t q1_out0 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel0);
  uint32_t q1_out1 = byte_perm_unchecked(lut0.x, lut0.y, q1_sel1);
  uint32_t q2_out0 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel0);
  uint32_t q2_out1 = byte_perm_unchecked(lut1.x, lut1.y, q2_sel1);
  uint32_t q3_out0 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel0);
  uint32_t q3_out1 = byte_perm_unchecked(lut1.x, lut1.y, q3_sel1);

  q0_out0 |= q.x & 0x80808080u;
  q0_out1 |= (q.x << 4) & 0x80808080u;
  q1_out0 |= q.y & 0x80808080u;
  q1_out1 |= (q.y << 4) & 0x80808080u;
  q2_out0 |= q.z & 0x80808080u;
  q2_out1 |= (q.z << 4) & 0x80808080u;
  q3_out0 |= q.w & 0x80808080u;
  q3_out1 |= (q.w << 4) & 0x80808080u;

  *reinterpret_cast<uint4*>(fp8_dst + ((scale_i0 * 16) ^ row_swizzle)) =
      make_uint4(q0_out0, q0_out1, q1_out0, q1_out1);
  *reinterpret_cast<uint4*>(fp8_dst + (((scale_i0 + 1) * 16) ^ row_swizzle)) =
      make_uint4(q2_out0, q2_out1, q3_out0, q3_out1);
}

template <int kQuad, bool kQuadIlp>
__device__ __forceinline__ void dequant_braided_quad_lut_window(
    uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4], const uint32_t scale_word_lo,
    const uint32_t scale_word_hi, const uint2* __restrict__ lut_smem, const uint2 lut0,
    const uint2 lut1, const uint32_t row_swizzle) {
  uint2 next_lut0;
  uint2 next_lut1;
  if constexpr (kQuad + 1 < 4) {
    constexpr int kNextScaleI0 = (kQuad + 1) * 2;
    constexpr int kNextScaleI1 = kNextScaleI0 + 1;
    const uint32_t next_scale_word = kQuad + 1 < 2 ? scale_word_lo : scale_word_hi;
    const uint32_t next_scale0 = (next_scale_word >> ((kNextScaleI0 & 3) * 8)) & 0x7fu;
    const uint32_t next_scale1 = (next_scale_word >> ((kNextScaleI1 & 3) * 8)) & 0x7fu;
    next_lut0 = lut_smem[next_scale0];
    next_lut1 = lut_smem[next_scale1];
  }

  if constexpr (kQuadIlp) {
    dequant_braided_quad_ilp(fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
  } else {
    dequant_braided_quad(fp8_dst, fp4_quads[kQuad], lut0, lut1, kQuad * 2, row_swizzle);
  }

  if constexpr (kQuad + 1 < 4) {
    dequant_braided_quad_lut_window<kQuad + 1, kQuadIlp>(fp8_dst, fp4_quads, scale_word_lo,
                                                         scale_word_hi, lut_smem, next_lut0,
                                                         next_lut1, row_swizzle);
  }
}

template <bool kQuadIlp = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_braided_lut_window(
    uint8_t* __restrict__ smem_b, const uint8_t* __restrict__ packed_b, const uint32_t row,
    const uint2* __restrict__ lut_smem) {
  const uint8_t* __restrict__ row_ptr = packed_b + row * 80;
  const uint4* __restrict__ fp4_src = reinterpret_cast<const uint4*>(row_ptr);
  uint4 fp4_quads[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) fp4_quads[i] = fp4_src[i];

  const uint2 scale_words = *reinterpret_cast<const uint2*>(row_ptr + 64);
  const uint2 lut0 = lut_smem[scale_words.x & 0x7fu];
  const uint2 lut1 = lut_smem[(scale_words.x >> 8) & 0x7fu];
  dequant_braided_quad_lut_window<0, kQuadIlp>(smem_b + row * 128, fp4_quads, scale_words.x,
                                               scale_words.y, lut_smem, lut0, lut1,
                                               (row & 7u) << 4);
}

}  // namespace nvfp4

// Push dispatch (see the kernel body, `kPushDispatch`): the routed rows of this
// rank are written straight into the destination ranks' fixed-stride pools.
// Force-inlined: an ABI call (`__noinline__`) anywhere in the persistent kernel
// changes the register allocation of the whole kernel and costs ~15 % on the
// math warps (measured on H20).
template <uint32_t kHidden, uint32_t kNumTopk, uint32_t kNumExpertsPerRank,
          uint32_t kNumPaddedSFPoolTokens, uint32_t BLOCK_M, uint32_t kPushBlocksPerExpert,
          uint32_t kNumGlobalWarps, uint32_t kNumRanks, bool kGenericRows>
__device__ __forceinline__ void sm90_nvfp4_push_dispatch_rows(
    const layout::SymBuffer<kNumRanks>& sym_buffer, void* smem_row_buffer,
    cutlass::arch::ClusterTransactionBarrier* row_mbarrier,
    const int64_t* __restrict__ input_topk_idx, const uint8_t* __restrict__ input_tokens,
    const float* __restrict__ input_sf, const float* __restrict__ input_topk_weights,
    uint8_t* l1_tokens, float* l1_sf, float* l1_topk_weights,
    layout::TokenSrcMetadata* token_src_metadata, uint64_t* recv_count_sum,
    const uint32_t pool_block_base, const uint32_t num_tokens, const uint32_t global_warp_idx,
    const uint32_t lane_idx) {
  constexpr uint32_t kNumSFFloats = kHidden / 128;
  constexpr uint32_t kNumTokenChunksPerLane = kHidden / (16 * 32);
  constexpr uint32_t kChunkGroup = 2;
  static_assert(kHidden % 512 == 0 && kNumTokenChunksPerLane % kChunkGroup == 0,
                "Invalid token shape for push dispatch");
  // Rows (token, top-k slot) in contiguous chunks of R = ceil(rows / warps) per
  // warp (<= 32 per ticket batch): every lane takes the remote ticket of one row
  // of the batch at once (the NVLink round trips overlap), then the warp streams
  // the batch's rows (16 B per lane per store) into the destination pools.
  const uint32_t num_rows = num_tokens * kNumTopk;
  const uint32_t rows_per_warp = (num_rows + kNumGlobalWarps - 1) / kNumGlobalWarps;
  const uint32_t row_begin = global_warp_idx * rows_per_warp;
  const uint32_t row_end = min(row_begin + rows_per_warp, num_rows);
  uint32_t row_mbarrier_phase = 0;
#pragma unroll 1
  for (uint32_t batch_begin = row_begin; batch_begin < row_end; batch_begin += 32) {
    const uint32_t batch_size = min(row_end - batch_begin, 32u);
    const uint32_t r = batch_begin + lane_idx;
    int expert_idx = -1;
    if (lane_idx < batch_size) expert_idx = static_cast<int>(__ldg(input_topk_idx + r));
    uint32_t lane_row_idx = 0;
    if (expert_idx >= 0) {
      const uint32_t dr = static_cast<uint32_t>(expert_idx) / kNumExpertsPerRank;
      const uint32_t de = static_cast<uint32_t>(expert_idx) % kNumExpertsPerRank;
      lane_row_idx =
          static_cast<uint32_t>(ptx::atomic_add_sys(sym_buffer.map(recv_count_sum + de, dr), 1ull));
    }
    const uint32_t valid_mask = __ballot_sync(0xffffffff, expert_idx >= 0);
#pragma unroll 1
    for (uint32_t mask = valid_mask; mask != 0; mask &= mask - 1) {
      const uint32_t src_lane = __ffs(mask) - 1;
      const uint32_t row_r = batch_begin + src_lane;
      const uint32_t e = static_cast<uint32_t>(__shfl_sync(0xffffffff, expert_idx, src_lane));
      const uint32_t row_idx = __shfl_sync(0xffffffff, lane_row_idx, src_lane);
      const uint32_t dr = e / kNumExpertsPerRank;
      const uint32_t de = e % kNumExpertsPerRank;
      const uint32_t src_token_idx = row_r / kNumTopk, src_topk_idx = row_r % kNumTopk;
      FI_MEGAMOE_TRAP_ONLY_DEVICE_ASSERT(row_idx < kPushBlocksPerExpert * BLOCK_M);
      const uint32_t pool_token_idx =
          (pool_block_base + de * kPushBlocksPerExpert) * BLOCK_M + row_idx;
      const auto* src_token = reinterpret_cast<const uint4*>(
          input_tokens + static_cast<uint64_t>(src_token_idx) * kHidden);
      auto* dst_token = sym_buffer.map(
          reinterpret_cast<uint4*>(l1_tokens + static_cast<uint64_t>(pool_token_idx) * kHidden),
          dr);
      if constexpr (kGenericRows) {
#pragma unroll
        for (uint32_t g = 0; g < kNumTokenChunksPerLane; g += kChunkGroup) {
          uint4 row[kChunkGroup];
#pragma unroll
          for (uint32_t c = 0; c < kChunkGroup; ++c)
            row[c] = __ldg(src_token + (g + c) * 32 + lane_idx);
#pragma unroll
          for (uint32_t c = 0; c < kChunkGroup; ++c) dst_token[(g + c) * 32 + lane_idx] = row[c];
        }
      } else {
        // Bulk copy (async proxy) local input row -> smem -> destination pool,
        // the mirror image of the pull path's TMA pull.
        if (cute::elect_one_sync()) {
          ptx::tma_load_1d(smem_row_buffer, src_token, row_mbarrier, kHidden);
          ptx::mbarrier_arrive_and_set_tx(row_mbarrier, kHidden);
          ptx::mbarrier_wait_and_flip_phase(row_mbarrier, row_mbarrier_phase);
          ptx::tma_store_1d(dst_token, smem_row_buffer, kHidden);
          cute::tma_store_arrive();
          ptx::tma_store_wait<0>();
        }
        __syncwarp();
      }
      const float* src_sf = input_sf + static_cast<uint64_t>(src_token_idx) * kNumSFFloats;
      float* dst_sf = sym_buffer.map(l1_sf, dr);
#pragma unroll
      for (uint32_t j = lane_idx; j < kNumSFFloats; j += 32)
        dst_sf[static_cast<uint64_t>(j) * kNumPaddedSFPoolTokens + pool_token_idx] =
            __ldg(src_sf + j);
      if (lane_idx == 0) {
        *sym_buffer.map(l1_topk_weights + pool_token_idx, dr) = __ldg(input_topk_weights + row_r);
        *sym_buffer.map(token_src_metadata + pool_token_idx, dr) = {
            static_cast<uint32_t>(sym_buffer.rank_idx), src_token_idx, src_topk_idx};
      }
    }
    __syncwarp();
  }
}

template <uint32_t kNumMaxTokensPerRank, uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kHidden,
          uint32_t kIntermediateHidden, uint32_t kNumExpertsPerWave, uint32_t BLOCK_M,
          uint32_t BLOCK_N, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages, float kActivationClamp, bool kFastMath, bool kSwapABRequested,
          bool kRSSwapABRequested, bool kSingleActiveDispatchWarp, bool kUseMode2RowDecoder,
          bool kUseInterleavedScheduler,
          // Counter-based synchronisation (see the body): push dispatch + DONE flags
          // (replaces NVLink barrier #1) and the rotated pool without the
          // workspace-clean barrier (#3).
          bool kPushDispatchRequested = false, bool kNoCleanBarrierRequested = false,
          // Diagnostics: strided pool addressing under the PULL protocol (isolates the
          // layout's cost), and the generic->async proxy fence in the push A loader.
          bool kStridedPoolDebug = false, bool kPushProxyFence = false,
          // Push rows with 16 B generic stores instead of TMA bulk copies (diagnostic).
          bool kPushGenericRows = false,
          // Phase stamps (globaltimer) compiled in; host env FI_MEGAMOE_NVFP4_PHASE_STAMPS_PTR.
          bool kPhaseStamps = false, uint32_t kNumRanks = 8>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void sm90_nvfp4_mega_moe_fused_impl(
    void* y, int* cumulative_local_expert_recv_stats, unsigned long long* phase_stamps,
    const uint32_t num_tokens, const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,
    const float* __restrict__ l1_global_scales, const float* __restrict__ l2_global_scales) {
  constexpr uint32_t BLOCK_K = 128;
  constexpr uint32_t kNumDispatchThreads = 64;
  constexpr uint32_t kNumNonEpilogueThreads = 64;
  constexpr uint32_t kNumEpilogueThreads = 256;
  constexpr uint32_t kNumSMs = MEGAMOE_NUM_SMS;
  static_assert(kNumRanks > 0 && kNumRanks <= 32, "Invalid EP width");
  static_assert(kNumExperts % kNumRanks == 0, "Experts must divide evenly across ranks");
  constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
  constexpr uint32_t L1_SHAPE_K = kHidden;
  constexpr uint32_t L2_SHAPE_N = kHidden;
  constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
  constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
  constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
  constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
  constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
  constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <flashinfer/megamoe/sm90_nvfp4_mega_moe_fused_body.inl>
}

}  // namespace flashinfer::megamoe

#pragma clang diagnostic pop
