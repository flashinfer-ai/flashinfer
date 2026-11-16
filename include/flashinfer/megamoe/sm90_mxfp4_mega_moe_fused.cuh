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
#include <flashinfer/megamoe/mxfp4_dequant.cuh>
#include <flashinfer/megamoe/scheduler.cuh>
#include <flashinfer/megamoe/sym_buffer.cuh>
#include <type_traits>

namespace flashinfer::megamoe {
namespace mxfp4 {

// MXFP4 decoders for the SM90 fused MegaMoE mainloop.
//
// Two differences from the NVFP4 decoders they replace, both consequences of
// the scale format:
//
//  * Group width. NVFP4 groups 16 elements, so a 32-element quad spans two
//    scales and its four packed words split across lut0/lut1. MXFP4 groups 32,
//    so one quad is exactly one scale group and all four words share a table.
//    A BK128 row therefore carries 4 E8M0 bytes (one uint32) instead of 8
//    UE4M3 bytes (a uint2).
//  * Table provenance and size. Both formats look the scaled magnitudes up in
//    shared memory. Combined with the wider group, MXFP4 needs half as many
//    lookups per row -- see kScaledLutSize in mxfp4_fused_scale.cuh.

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_mode2_nibble_row_regs(
    uint8_t* __restrict__ fp8_dst, const uint4 (&fp4_quads)[4], const uint32_t scale_word,
    const uint32_t row_swizzle, const ScaledLut* __restrict__ lut_smem) {
#pragma unroll
  for (int quad_i = 0; quad_i < 4; ++quad_i) {
    const uint4 q = fp4_quads[quad_i];
    const ScaledLut lut = load_scaled_lut(lut_smem, (scale_word >> (quad_i * 8)) & 0xffu);

    const uint2 w0 = dequant_word(q.x, lut);
    const uint2 w1 = dequant_word(q.y, lut);
    if constexpr (!kQuadILP) {
      *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
          make_uint4(w0.x, w0.y, w1.x, w1.y);
    }
    const uint2 w2 = dequant_word(q.z, lut);
    const uint2 w3 = dequant_word(q.w, lut);
    if constexpr (kQuadILP) {
      *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
          make_uint4(w0.x, w0.y, w1.x, w1.y);
    }
    *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2 + 1) * 16) ^ row_swizzle)) =
        make_uint4(w2.x, w2.y, w3.x, w3.y);
  }
}

template <bool kQuadILP = false>
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble(
    uint8_t* __restrict__ smem_b, const uint8_t* __restrict__ packed_b, const uint32_t row,
    const ScaledLut* __restrict__ lut_smem) {
  const DecodeTileRow src = decode_tile_row(packed_b, row);
  uint4 fp4_quads[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    fp4_quads[i] = *reinterpret_cast<const uint4*>(src.chunk + i * kBChunkStride);
  const uint32_t scale_word = *reinterpret_cast<const uint32_t*>(src.scale);
  dequant_mode2_nibble_row_regs<kQuadILP>(smem_b + row * 128, fp4_quads, scale_word,
                                          (row & 7u) << 4, lut_smem);
}

// Threads 0-127 and 128-255 each decode one K64 half of the same N128 tile,
// letting the two M64 warpgroups reuse the decoded weights. A K64 half is two
// quads, i.e. two E8M0 bytes.
__device__ __forceinline__ void dequant_smem_b_from_packed_mode2_nibble_split_m(
    uint8_t* __restrict__ smem_b, const uint8_t* __restrict__ packed_b, const uint32_t thread_idx,
    const ScaledLut* __restrict__ lut_smem) {
  const uint32_t row = thread_idx & 127u;
  const uint32_t k_half_idx = thread_idx >> 7;
  const DecodeTileRow src = decode_tile_row(packed_b, row);
  const uint8_t* __restrict__ fp4_src = src.chunk + k_half_idx * 2u * kBChunkStride;
  const uint32_t scale_pair =
      *reinterpret_cast<const uint16_t*>(src.scale + k_half_idx * sizeof(uint16_t));
  uint8_t* __restrict__ fp8_dst = smem_b + row * 128u;
  const uint32_t row_swizzle = (row & 7u) << 4;

#pragma unroll
  for (uint32_t quad_i = 0; quad_i < 2; ++quad_i) {
    const uint4 q = *reinterpret_cast<const uint4*>(fp4_src + quad_i * kBChunkStride);
    const ScaledLut lut = load_scaled_lut(lut_smem, (scale_pair >> (quad_i * 8u)) & 0xffu);
    const uint2 w0 = dequant_word(q.x, lut);
    const uint2 w1 = dequant_word(q.y, lut);
    const uint2 w2 = dequant_word(q.z, lut);
    const uint2 w3 = dequant_word(q.w, lut);
    const uint32_t off0 = k_half_idx * 64u + (quad_i * 2u) * 16u;
    const uint32_t off1 = k_half_idx * 64u + (quad_i * 2u + 1u) * 16u;
    *reinterpret_cast<uint4*>(fp8_dst + (off0 ^ row_swizzle)) = make_uint4(w0.x, w0.y, w1.x, w1.y);
    *reinterpret_cast<uint4*>(fp8_dst + (off1 ^ row_swizzle)) = make_uint4(w2.x, w2.y, w3.x, w3.y);
  }
}

// Decode the two packed words one lane pair owns directly into the four
// register-source WGMMA A operands, bypassing shared memory entirely.
//
// A K32 slice is 16 packed bytes. Lanes pair up: the even lane keeps the high
// half of each decoded word and ships the low half to its partner, the odd lane
// does the reverse, so one shuffle per word assembles both lanes' fragments.
// MXFP4's group of 32 is exactly the WGMMA's K, so both words of a slice share
// one table; NVFP4's group of 16 needs two.
__device__ __forceinline__ void dequant_rs_word_pair(const uint32_t w_lo, const uint32_t w_hi,
                                                     const ScaledLut& lut, const bool keep_hi,
                                                     uint32_t (&a_frag)[4]) {
  detail::assemble_rs_word_pair(dequant_word(w_lo, lut), dequant_word(w_hi, lut), keep_hi, a_frag);
}

__device__ __forceinline__ void dequant_braided_quad(uint8_t* __restrict__ fp8_dst, const uint4& q,
                                                     const ScaledLut& lut, const int quad_i,
                                                     const uint32_t row_swizzle) {
  const uint2 w0 = dequant_word(q.x, lut);
  const uint2 w1 = dequant_word(q.y, lut);
  *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2) * 16) ^ row_swizzle)) =
      make_uint4(w0.x, w0.y, w1.x, w1.y);
  const uint2 w2 = dequant_word(q.z, lut);
  const uint2 w3 = dequant_word(q.w, lut);
  *reinterpret_cast<uint4*>(fp8_dst + (((quad_i * 2 + 1) * 16) ^ row_swizzle)) =
      make_uint4(w2.x, w2.y, w3.x, w3.y);
}

}  // namespace mxfp4

// Decode occupies its own warps only on a shared-memory swapAB tile tall
// enough for the WGMMA to hide it; the register-source path has no decoded-B
// ring to fill, and BLOCK_M 8 is faster decoding on the math warps. An
// unconditional 640 would also cut the compiler's per-thread register budget
// for the narrow tiers. Must stay in step with
// SM90MXFP4H200FusedConfig::num_threads().
constexpr bool sm90_mxfp4_split_decode(const bool swap_ab, const bool use_rs,
                                       const uint32_t block_m) {
  return swap_ab and not use_rs and block_m >= 16;
}

constexpr uint32_t sm90_mxfp4_num_threads(const bool swap_ab, const bool use_rs,
                                          const uint32_t block_m) {
  return 64 + 64 + 256 + (sm90_mxfp4_split_decode(swap_ab, use_rs, block_m) ? 256 : 0);
}

template <uint32_t kNumSMs,
          // Expert-parallel width. SymBuffer<N> has one layout for every N, and the
          // barriers, scheduler and workspace were already templated on it.
          uint32_t kNumRanks,
          // Shape is a template parameter, not a baked-in constant. The kernel body
          // was already written against these names; only the wrapper hardcoded the
          // H200 384-expert / 6144-hidden model, which locked out every other shape
          // (e.g. DeepSeek-V4-Flash: 4096 hidden, 256 experts, topk 6).
          uint32_t kHidden, uint32_t kIntermediateHidden, uint32_t kNumExperts, uint32_t kNumTopk,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumExpertsPerWave, uint32_t BLOCK_M,
          uint32_t BLOCK_N, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages, float kActivationClamp, bool kFastMath, bool kSwapABRequested,
          bool kSingleActiveDispatchWarp, bool kUseMode2RowDecoder, bool kUseInterleavedScheduler,
          // Feed the WGMMA's A operand from registers instead of decoding the weight
          // tile into shared memory first. swapAB only; removes the decoded-B ring.
          bool kUseRSOperand,
          // Phase timestamps are a compile-time variant, not a runtime flag. Checking
          // a pointer per task is not free -- the same accumulators cost ~5% per task
          // on the NVFP4 side when they were runtime-gated -- so the shipped kernel
          // must not contain them at all. Defaulted, so every existing instantiation
          // (and every compile-gate row) keeps its current meaning.
          bool kPhaseStamps = false>
CUTLASS_GLOBAL __launch_bounds__(
    sm90_mxfp4_num_threads(kSwapABRequested, kUseRSOperand, BLOCK_M),
    1) void sm90_mxfp4_mega_moe_fused_impl(void* y, int* cumulative_local_expert_recv_stats,
                                           const uint32_t num_tokens,
                                           const __grid_constant__ layout::SymBuffer<kNumRanks>
                                               sym_buffer,
                                           const __grid_constant__ cute::TmaDescriptor
                                               tensor_map_l1_acts,
                                           const __grid_constant__ cute::TmaDescriptor
                                               tensor_map_l1_acts_sf,
                                           const __grid_constant__ cute::TmaDescriptor
                                               tensor_map_l1_output,
                                           const __grid_constant__ cute::TmaDescriptor
                                               tensor_map_l2_acts,
                                           const __grid_constant__ cute::TmaDescriptor
                                               tensor_map_l2_acts_sf,
                                           // Weights are bulk-copied a tile at a time, not
                                           // addressed through a tensor map: the tile is
                                           // contiguous, so there is no shape to describe.
                                           const uint8_t* __restrict__ l1_weights,
                                           const uint8_t* __restrict__ l2_weights,
                                           const float* __restrict__ l1_global_scales,
                                           const float* __restrict__ l2_global_scales,
                                           // Only written when kPhaseStamps; see the slot map in
                                           // the body.
                                           unsigned long long* __restrict__ phase_stamps) {
  constexpr uint32_t BLOCK_K = 128;
  constexpr uint32_t kNumDispatchThreads = 64;
  constexpr uint32_t kNumNonEpilogueThreads = 64;
  // Decode gets its own warps so it overlaps the WGMMA rather than
  // serialising with it on the math warps.
  constexpr bool kSplitDecodeWarps =
      sm90_mxfp4_split_decode(kSwapABRequested, kUseRSOperand, BLOCK_M);
  constexpr uint32_t kNumDecodeThreads = kSplitDecodeWarps ? 256 : 0;
  constexpr uint32_t kNumEpilogueThreads = 256;
  constexpr uint32_t kNumThreads =
      kNumDispatchThreads + kNumNonEpilogueThreads + kNumDecodeThreads + kNumEpilogueThreads;
  FI_MEGAMOE_STATIC_ASSERT(
      kNumThreads == sm90_mxfp4_num_threads(kSwapABRequested, kUseRSOperand, BLOCK_M),
      "Launch bounds and the role partition disagree");
  FI_MEGAMOE_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Experts must divide evenly across ranks");
  FI_MEGAMOE_STATIC_ASSERT(kHidden % 128 == 0, "Hidden must be a multiple of BLOCK_K");
  FI_MEGAMOE_STATIC_ASSERT(kIntermediateHidden % 128 == 0,
                           "Intermediate hidden must be a multiple of BLOCK_K");
  constexpr uint32_t L1_SHAPE_N = kIntermediateHidden * 2;
  constexpr uint32_t L1_SHAPE_K = kHidden;
  constexpr uint32_t L2_SHAPE_N = kHidden;
  constexpr uint32_t L2_SHAPE_K = kIntermediateHidden;
  constexpr uint32_t kNumDispatchWarps = kNumDispatchThreads / 32;
  constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32;
  constexpr uint32_t kNumDecodeWarps = kNumDecodeThreads / 32;
  constexpr uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32;
  // Warp map: [dispatch][A loader][B loader][decode][math]
  constexpr uint32_t kFirstDecodeWarp = kNumDispatchWarps + kNumMMANonEpilogueWarps;
  constexpr uint32_t kFirstMathWarp = kFirstDecodeWarp + kNumDecodeWarps;
  constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
  constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopk;
  constexpr uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks;
#include <flashinfer/megamoe/sm90_mxfp4_mega_moe_fused_body.inl>
}

}  // namespace flashinfer::megamoe

#pragma clang diagnostic pop
