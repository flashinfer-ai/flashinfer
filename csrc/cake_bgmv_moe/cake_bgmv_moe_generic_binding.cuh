/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * TVM-FFI binding for the generic-shape Cake BGMV MoE bundle: one compile-time
 * LoRA rank (CAKE_BGMV_MOE_RANK), any hidden size that is a multiple of 8 at
 * run time. The consuming JIT spec defines:
 *   CAKE_BGMV_MOE_BODY_FILE           generated body to include
 *   CAKE_BGMV_MOE_RANK                8, 16, 32 or 64
 *   CAKE_BGMV_MOE_INPUT_DTYPE         dl_bfloat16 or dl_float16
 *   CAKE_BGMV_MOE_CC_MAJOR / _MINOR   compute capability this module was built for
 *   CAKE_BGMV_MOE_SHRINK_DECODE       generated decode shrink kernel (PPB=4, 3 stages)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL      generated prefill shrink kernel (PPB=1, 2 stages)
 *   CAKE_BGMV_MOE_SHRINK_DECODE_PDL   decode shrink + griddepcontrol.launch_dependents (PDL)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL_PDL  prefill shrink + griddepcontrol.launch_dependents (PDL)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL_S3   prefill shrink, 3-stage ring (small grids, lever 11c)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL_S3_PDL  3-stage prefill shrink, PDL form
 *   CAKE_BGMV_MOE_SHRINK_PREFILL_REMAP   2-stage prefill shrink reading its route through the
 *                                        lever-27 permutation (SM90 bin-ordered dispatch)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL_REMAP_PDL  its PDL form
 *   CAKE_BGMV_MOE_EXPAND_T64          generated 64-lane token-owned expand kernel
 *   CAKE_BGMV_MOE_EXPAND_T128         generated 128-lane token-owned expand kernel
 *   CAKE_BGMV_MOE_EXPAND_T64_PF       64-lane expand, B rows register-prefetched before PDL wait
 *   CAKE_BGMV_MOE_EXPAND_T128_PF      128-lane expand, B rows register-prefetched before PDL wait
 *   CAKE_BGMV_MOE_ORDER_BUILD         single-CTA route-order prologue of the SM90 per-route shrink
 *   (the grouped-pipeline kernels CAKE_BGMV_MOE_GROUP_* / *_GROUPED are listed below)
 */
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

#include CAKE_BGMV_MOE_BODY_FILE

namespace flashinfer {
namespace cake_bgmv_moe_generic {

constexpr int32_t kRank = CAKE_BGMV_MOE_RANK;
constexpr int32_t kRankTile = 8;
constexpr int32_t kVec = 8;
constexpr int32_t kShrinkThreads = 128;
constexpr int32_t kShrinkTileElements = 128 * kVec;  // one vec8 per lane per tile
constexpr int32_t kShrinkDecodePairsPerBlock = 4;
// Dynamic shared memory per launch; the generated body records the values its
// kernels were scheduled with (x/weight cp.async rings plus FP32 partials for
// the shrink kernels, routed activations plus the route list for expand).
constexpr int32_t kShrinkDecodeSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE;
constexpr int32_t kShrinkPrefillSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL;
static_assert(CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE_PDL == kShrinkDecodeSmemBytes,
              "decode shrink forms must share smem");
static_assert(CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_PDL == kShrinkPrefillSmemBytes,
              "prefill shrink forms must share smem");
// Lever 11c: three-stage prefill ring (one more x + weight stage) for grids of at most
// kShrinkDeepRingMaxCtas CTAs; shrink form 2 (0 = two-stage prefill, 1 = decode).
constexpr int32_t kShrinkPrefillS3SmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3;
static_assert(CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_S3_PDL == kShrinkPrefillS3SmemBytes,
              "three-stage prefill shrink forms must share smem");
static_assert(kShrinkPrefillS3SmemBytes > kShrinkPrefillSmemBytes,
              "the three-stage prefill ring must be deeper than the two-stage one");
constexpr int32_t kShrinkPrefillRemapSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP;
static_assert(kShrinkPrefillRemapSmemBytes == kShrinkPrefillSmemBytes &&
                  CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL_REMAP_PDL == kShrinkPrefillSmemBytes,
              "the bin-ordered prefill shrink forms share the two-stage prefill smem");
constexpr int32_t kShrinkFormPrefill = 0;
constexpr int32_t kShrinkFormDecode = 1;
constexpr int32_t kShrinkFormPrefillS3 = 2;
constexpr int32_t kExpandT64SmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64;
constexpr int32_t kExpandT128SmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128;
constexpr int32_t kExpandT64PfSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64_PF;
constexpr int32_t kExpandT128PfSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128_PF;
static_assert(kExpandT64PfSmemBytes == kExpandT64SmemBytes, "expand T64 forms must share smem");
static_assert(kExpandT128PfSmemBytes == kExpandT128SmemBytes, "expand T128 forms must share smem");
static_assert(kShrinkDecodeSmemBytes == 221824, "decode shrink smem layout changed");
static_assert(kShrinkPrefillSmemBytes == 37120, "prefill shrink smem layout changed");
static_assert(kRank % kRankTile == 0, "rank must be a multiple of the 8-row shrink tile");
// Token->pair route index published by the shrink kernels and consumed by the
// expand kernels for arbitrary pair order (u32 words): a 4-word header (launch
// counter, parity used by the current shrink), per token a monotonic route
// count, two launch-parity base counts and kRouteIndexMaxRoutes pair slots.
// The plan allocates it zeroed once; the kernels never reset it.
constexpr int32_t kRouteIndexMaxRoutes = 16;
constexpr int32_t kRouteIndexHeaderWords = 4;
constexpr int32_t kRouteIndexWordsPerToken = 3 + kRouteIndexMaxRoutes;
// Hidden-split shrink workspace appended to the route index (see
// flashinfer/jit/cake_bgmv_moe.py): FP32 partials [split][pair][64] as raw
// 32-bit words, then one arrival counter per (pair block, rank block).
constexpr int32_t kShrinkSplitMax = 8;
constexpr int32_t kShrinkSplitMaxPairs = 128;
constexpr int32_t kShrinkSplitPartialWords = kShrinkSplitMax * kShrinkSplitMaxPairs * 64;
constexpr int32_t kShrinkSplitCounterWords = kShrinkSplitMaxPairs * (64 / kRankTile);
// Pair-grouped pipeline (round 5): routes grouped by their unique (LoRA,
// expert) pair so each pair's weights are streamed once per tile of
// kGroupTileTokens routes.  Plan-owned int32 workspace (see
// flashinfer/jit/cake_bgmv_moe.py ``cake_bgmv_moe_grouped_workspace_words``):
// header, bin counts, bin offsets, bin fill counters, tile table, grouped route
// ids, per-token route counts and per-token route lists; plus FP32 per-route
// expand partials [num_pairs][hidden].  Rebuilt by the grouping kernels on every
// launch (no memset node); the deterministic combine sums each token's route
// partials in ascending pair order.
constexpr int32_t kGroupThreads = 1024;     // group_scan (one CTA)
constexpr int32_t kGroupHistThreads = 256;  // group_hist / group_scatter CTAs
constexpr int32_t kGroupHistPairsPerLane = 4;
constexpr int32_t kGroupHistCtasMax = 64;
constexpr int32_t kGroupTileTokens = 16;
constexpr int32_t kGroupShrinkRankTile = 8;  // rank rows per grouped-shrink CTA
constexpr int32_t kGroupShrinkRoutes = 4;    // routes per grouped-shrink CTA
constexpr int32_t kGroupShrinkRankTiles = kRank / kGroupShrinkRankTile;
// Rank tiles one grouped-shrink CTA walks (the same policy the kernel generator's host launcher
// applies). Blackwell: largest power of two <= min(rank tiles, 16 / K
// tiles).  sm_90 pays 2-5 % for the rank-tile loop form itself, so it loops only where the H100
// sweep showed net gains: one K tile -> the whole rank, two or three K tiles with >= 4 rank tiles
// -> 4, otherwise one (straight-line path).
inline int32_t GroupShrinkRankTilesPerCta(int32_t num_tiles) {
  static thread_local int cached_device = -1;
  static thread_local int cached_major = 0;
  int device = 0;
  cudaGetDevice(&device);
  if (device != cached_device) {
    cudaDeviceGetAttribute(&cached_major, cudaDevAttrComputeCapabilityMajor, device);
    cached_device = device;
  }
  if (cached_major == 9) {
    if (num_tiles >= 4 || (num_tiles > 1 && kGroupShrinkRankTiles < 4)) return 1;
    return num_tiles == 1 ? kGroupShrinkRankTiles
                          : (kGroupShrinkRankTiles < 4 ? kGroupShrinkRankTiles : 4);
  }
  const int32_t budget_raw = 16 / num_tiles;
  const int32_t budget =
      budget_raw < 1 ? 1
                     : (budget_raw > kGroupShrinkRankTiles ? kGroupShrinkRankTiles : budget_raw);
  int32_t per_cta = 1;  // largest power of two <= budget (a divisor of the rank tiles)
  while (per_cta * 2 <= budget) per_cta *= 2;
  return per_cta;
}
// Lever 3: the cp.async operand-ring grouped shrink (tile k+1's x and weight rows land in shared
// memory while tile k computes) serves rows whose hidden spans more than one 1024-element K tile;
// on Blackwell a single-tile CTA has nothing to overlap and pays the ring's two barriers (768-wide
// rows lost 1.2-2.2 %), so it keeps the register-direct form there; on sm_90 the ring wins on every
// row including the single-tile ones (768x16 0.983, 768x64 0.976), so the jit sets
// CAKE_BGMV_MOE_GROUP_SHRINK_RING_SINGLE_TILE to 1 for sm90a (lever 3c).  Lever 36b: where the
// lever-34 mixed-precision form exists (CAKE_BGMV_MOE_GROUP_SHRINK_MIXED, bf16 sm100a/sm103a) the
// ring form -- barrier-free, register-direct x rows, weights-only ring at 6 CTAs/SM -- also wins
// on single-tile rows (GB300 768x16/32/64 x 4096 tokens 0.94-0.95 of the direct form), so it
// serves them too.  Both forms store bitwise-identical rows.
inline bool GroupShrinkRing(int32_t num_tiles) {
  return CAKE_BGMV_MOE_GROUP_SHRINK_RING_SINGLE_TILE != 0 || num_tiles > 1 ||
         CAKE_BGMV_MOE_GROUP_SHRINK_MIXED != 0;
}
// Lever 34: the bf16 bundles of sm_100a/sm_103a run the mixed-precision grouped shrink
// (fma.rn.f32.bf16 on the packed BF16 halves, weights-only cp.async ring, x rows register-direct)
// wherever the ring form is selected: the products, their order and the rounding points are those
// of the widened chain, so the rows are bitwise identical.  The jit sets
// CAKE_BGMV_MOE_GROUP_SHRINK_MIXED to 1 for those bundles (0 elsewhere: sm_90 lacks the
// instruction, fp16 rows keep the widened chain).
inline bool GroupShrinkMixed(bool ring) { return CAKE_BGMV_MOE_GROUP_SHRINK_MIXED != 0 && ring; }
// Column blocks (kGroupExpandThreads columns each) one grouped-expand CTA walks.
// Blackwell: 8 at hidden >= 4096, 4 below (the expand is a per-CTA latency chain; 8 blocks at
// hidden 2048 leaves too few CTAs); sm_90 rank 64 the same, rank 8-32 2 (neutral at 2-8 on H100).
inline int32_t GroupExpandColBlocksPerCta(int32_t hidden, int32_t col_blocks) {
  static thread_local int cached_device = -1;
  static thread_local int cached_major = 0;
  int device = 0;
  cudaGetDevice(&device);
  if (device != cached_device) {
    cudaDeviceGetAttribute(&cached_major, cudaDevAttrComputeCapabilityMajor, device);
    cached_device = device;
  }
  int32_t per_cta = hidden >= 4096 ? 8 : 4;
  if (cached_major == 9 && kRank < 32) per_cta = 2;
  if (per_cta > col_blocks) per_cta = col_blocks;
  return per_cta < 1 ? 1 : per_cta;
}
static_assert(kGroupTileTokens % kGroupShrinkRoutes == 0,
              "route tile must split into whole grouped-shrink parts");
static_assert(kRank % kGroupShrinkRankTile == 0,
              "rank must be a multiple of the grouped shrink rank tile");
constexpr int32_t kGroupBinsMax = 4096;
constexpr int32_t kGroupHeaderWords = 4;
constexpr int32_t kGroupExpandThreads = 256;
constexpr int32_t kGroupCombineThreads = 256;
constexpr int32_t kGroupHistSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_HIST;
constexpr int32_t kGroupScanSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCAN;
constexpr int32_t kGroupScatterSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_GROUP_SCATTER;
constexpr int32_t kShrinkGroupedSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED;
// Lever 3: the cp.async operand-ring grouped shrink stages two K tiles of x and weight rows.
constexpr int32_t kShrinkGroupedRingSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING;
// Lever 34: the mixed-precision form's ring holds the weight rows only (fp16 bundles render no
// mixed form; their aliases resolve to the ring kernels and share its smem).
#ifdef CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED
constexpr int32_t kShrinkGroupedRingMixedSmemBytes =
    CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED;
#else
constexpr int32_t kShrinkGroupedRingMixedSmemBytes = kShrinkGroupedRingSmemBytes;
#endif
#ifdef CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED_SINGLE
static_assert(CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_GROUPED_RING_MIXED_SINGLE ==
                  kShrinkGroupedRingMixedSmemBytes,
              "the mixed-precision grouped shrink forms must share smem");
#endif
constexpr int32_t kExpandGroupedSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_GROUPED;
constexpr int32_t kCombineGroupedSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_COMBINE_GROUPED;

// Lever 27: bin-ordered dispatch of the per-route shrink.  One CTA of kOrderBuildThreads lanes,
// each holding kOrderRoutesPerLane routes in registers between the bin-count and the scatter
// pass, writes the route permutation (routes sorted by (LoRA, expert) bin, invalid routes last)
// that the shrink kernels map their CTA slot through when route_remap != 0.  Every route is still
// computed by the same code on the same operands in the same order: bitwise identical to the
// identity dispatch; only the CTA order (and so the L2 hit rate of repeated A rows) changes.
constexpr int32_t kOrderBuildThreads = 1024;
constexpr int32_t kOrderRoutesPerLane = 4;
constexpr int32_t kOrderRemapMaxPairs = kOrderBuildThreads * kOrderRoutesPerLane;
constexpr int32_t kOrderBuildSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_ORDER_BUILD;
static_assert(kOrderBuildSmemBytes <= 48 * 1024, "order_build must fit the default smem carveout");

// Lever 22: programmatic dependent launch helper for the grouped chain.  Every kernel launched
// through it executes griddepcontrol.wait before reading its predecessor's outputs, so the
// launch and prologue of each stage overlap the previous stage's drain (a programmatic
// dependency edge under graph capture).
template <typename Kernel, typename... Args>
inline cudaError_t LaunchGroupedPdl(Kernel kernel, dim3 grid, dim3 block, int32_t smem_bytes,
                                    cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, args...);
}

struct GroupedOffsets {
  int64_t group_offset;
  int64_t tile_table;
  int64_t sorted_routes;
  int64_t token_count;
  int64_t token_routes;
  int64_t hist;  // per histogram CTA: route count per bin
  int64_t base;  // per histogram CTA: first slot per bin in sorted_routes
  int64_t numel;
  int64_t max_tiles;
  int64_t hist_ctas;
};

inline int64_t GroupHistCtas(int64_t num_pairs) {
  const int64_t per_cta = int64_t{kGroupHistThreads} * kGroupHistPairsPerLane;
  const int64_t ctas = (num_pairs + per_cta - 1) / per_cta;
  return ctas < 1 ? 1 : (ctas > kGroupHistCtasMax ? kGroupHistCtasMax : ctas);
}

inline GroupedOffsets ComputeGroupedOffsets(int64_t num_pairs, int64_t num_tokens, int64_t bins) {
  GroupedOffsets off{};
  off.max_tiles = (num_pairs + kGroupTileTokens - 1) / kGroupTileTokens + bins;
  off.hist_ctas = GroupHistCtas(num_pairs);
  int64_t cursor = kGroupHeaderWords;
  off.group_offset = cursor;
  cursor += bins + 1;
  off.tile_table = cursor;
  cursor += off.max_tiles;
  off.sorted_routes = cursor;
  cursor += num_pairs;
  off.token_count = cursor;
  cursor += num_tokens;
  off.token_routes = cursor;
  cursor += num_tokens * kRouteIndexMaxRoutes;
  off.hist = cursor;
  cursor += off.hist_ctas * bins;
  off.base = cursor;
  cursor += off.hist_ctas * bins;
  off.numel = cursor;
  return off;
}

enum class Schedule : int32_t {
  kTokenOwnedT64 = 0,
  kTokenOwned = 1,
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

// Each module is compiled for exactly one target (sm_90a for H100/H200,
// sm_100a for B200/GB200, sm_103a for B300/GB300). The device must match it;
// anything else fails closed instead of silently running another cubin.
inline void CheckCompiledArch(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == CAKE_BGMV_MOE_CC_MAJOR && minor == CAKE_BGMV_MOE_CC_MINOR)
      << "Cake BGMV MoE generic module was compiled for compute capability "
      << CAKE_BGMV_MOE_CC_MAJOR << "." << CAKE_BGMV_MOE_CC_MINOR << ", got " << major << "."
      << minor;
}

// Called once per loaded module: the device must match the compiled target and
// the decode shrink needs its opt-in dynamic shared memory.
void Configure() {
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  CheckCompiledArch(device_id);

  int32_t max_dynamic_smem = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
      "cudaDeviceGetAttribute(max opt-in shared memory)");
  TVM_FFI_ICHECK(max_dynamic_smem >= kShrinkDecodeSmemBytes)
      << "Cake BGMV MoE decode shrink requires " << kShrinkDecodeSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_DECODE, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kShrinkDecodeSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE decode shrink)");
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_DECODE_PDL,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkDecodeSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE decode shrink, PDL form)");
  // Lever 11c: the three-stage prefill ring exceeds the default 48 KiB carveout.
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_PREFILL_S3,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkPrefillS3SmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE three-stage prefill shrink)");
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_PREFILL_S3_PDL,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkPrefillS3SmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE three-stage prefill shrink, PDL form)");
  TVM_FFI_ICHECK(max_dynamic_smem >= kShrinkGroupedSmemBytes)
      << "Cake BGMV MoE grouped shrink requires " << kShrinkGroupedSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkGroupedSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink)");
  // Lever 20k: the one-rank-tile-per-CTA form shares the launch constants (rank 8: same kernel).
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED_SINGLE,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkGroupedSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink, single rank tile)");
  // Lever 3: the operand-ring forms exceed the default 48 KiB carveout.
  TVM_FFI_ICHECK(max_dynamic_smem >= kShrinkGroupedRingSmemBytes)
      << "Cake BGMV MoE grouped shrink (operand ring) requires " << kShrinkGroupedRingSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED_RING,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kShrinkGroupedRingSmemBytes),
            "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink, operand ring)");
  CheckCuda(cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED_RING_SINGLE,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kShrinkGroupedRingSmemBytes),
            "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink, operand ring, single rank tile)");
  // Lever 34: the mixed-precision ring forms (bf16 bundles; fp16 aliases resolve to the ring
  // kernels).
  if (CAKE_BGMV_MOE_GROUP_SHRINK_MIXED != 0) {
    CheckCuda(cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED_RING_MIXED,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kShrinkGroupedRingMixedSmemBytes),
              "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink, mixed-precision ring)");
    CheckCuda(cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_GROUPED_RING_MIXED_SINGLE,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   kShrinkGroupedRingMixedSmemBytes),
              "cudaFuncSetAttribute(Cake BGMV MoE grouped shrink, mixed-precision ring, single "
              "rank tile)");
  }
  // Lever 30: the rank-64 grouped expand double-buffers its weight stage (two 36 KiB column
  // blocks) and exceeds the default carveout; lower ranks keep register fragments.
  TVM_FFI_ICHECK(max_dynamic_smem >= kExpandGroupedSmemBytes)
      << "Cake BGMV MoE grouped expand requires " << kExpandGroupedSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_EXPAND_GROUPED,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kExpandGroupedSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE grouped expand)");
}

inline void CheckCompact(const TensorView& tensor, const char* name) {
  CHECK_CONTIGUOUS(tensor);
  TVM_FFI_ICHECK(tensor.numel() <= std::numeric_limits<int32_t>::max())
      << name << " exceeds the generated kernel's int32 index range";
}

// The expand kernels take the output row stride at runtime, so the FP32
// accumulator may be a column slice of a wider row-major buffer: unit column
// stride, row stride at least the hidden size, and every row offset within
// the kernels' int32 index range.
inline int32_t OutputRowStride(const TensorView& y_accum, int64_t num_tokens, int64_t hidden) {
  TVM_FFI_ICHECK(y_accum.stride(1) == 1) << "y_accum must be contiguous along hidden";
  const int64_t row_stride = y_accum.stride(0);
  TVM_FFI_ICHECK(row_stride >= hidden)
      << "y_accum row stride " << row_stride << " is smaller than hidden " << hidden;
  TVM_FFI_ICHECK((num_tokens - 1) * row_stride + hidden <= std::numeric_limits<int32_t>::max())
      << "y_accum exceeds the generated kernel's int32 index range";
  return static_cast<int32_t>(row_stride);
}

void Run(TensorView y_accum, TensorView shrink_out, TensorView x, TensorView lora_a,
         TensorView lora_b, TensorView sorted_token_ids, TensorView expert_ids,
         TensorView lora_indices, TensorView topk_weights, TensorView route_index,
         int64_t schedule_value, int64_t shrink_form, int64_t shrink_splits, int64_t grouped,
         TensorView group_workspace, TensorView group_partials, int64_t order_remap,
         TensorView order_workspace, int64_t pdl_mode, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(pdl_mode >= 0 && pdl_mode <= 2)
      << "pdl_mode must be 0 (off), 1 (early trigger) or 2 (late trigger), got " << pdl_mode;
  TVM_FFI_ICHECK(order_remap == 0 || grouped == 0)
      << "order_remap applies to the per-route pipeline only (grouped must be 0)";
  TVM_FFI_ICHECK(order_remap == 0 || shrink_form == kShrinkFormPrefill)
      << "order_remap applies to the two-stage prefill shrink form only (shrink_form must be 0)";
  // Programmatic dependent launch of the expand behind the shrink: the shrink
  // kernels trigger their dependents at entry (pdl_early) or after the tile
  // loop; every expand kernel executes griddepcontrol.wait before it reads the
  // shrink output or the route index.  Without the launch attribute both
  // instructions are no-ops, so mode 0 is the plain two-launch pipeline.
  const int32_t pdl_early = pdl_mode == 1 ? 1 : 0;
  CHECK_CUDA(x);
  // The device/module match is checked once in Configure (module load); the
  // Python side already routes each device to the module compiled for it.
  ffi::CUDADeviceGuard device_guard(x.device().device_id);

  CHECK_CUDA(y_accum);
  CHECK_CUDA(shrink_out);
  CHECK_CUDA(lora_a);
  CHECK_CUDA(lora_b);
  CHECK_CUDA(sorted_token_ids);
  CHECK_CUDA(expert_ids);
  CHECK_CUDA(lora_indices);
  CHECK_CUDA(topk_weights);
  CHECK_CUDA(route_index);
  CHECK_DEVICE(x, y_accum);
  CHECK_DEVICE(x, shrink_out);
  CHECK_DEVICE(x, lora_a);
  CHECK_DEVICE(x, lora_b);
  CHECK_DEVICE(x, sorted_token_ids);
  CHECK_DEVICE(x, expert_ids);
  CHECK_DEVICE(x, lora_indices);
  CHECK_DEVICE(x, topk_weights);
  CHECK_DEVICE(x, route_index);

  CHECK_INPUT_TYPE(x, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(shrink_out, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_a, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_b, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(y_accum, dl_float32);
  CHECK_INPUT_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_TYPE(sorted_token_ids, dl_int64);
  CHECK_INPUT_TYPE(expert_ids, dl_int64);
  CHECK_INPUT_TYPE(lora_indices, dl_int64);
  CHECK_INPUT_TYPE(route_index, dl_int32);

  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(0) > 0 && x.size(1) > 0 && x.size(1) % kVec == 0)
      << "x must have shape [num_tokens, hidden] with hidden a positive multiple of 8";
  const int32_t num_tokens = static_cast<int32_t>(x.size(0));
  const int32_t hidden = static_cast<int32_t>(x.size(1));
  TVM_FFI_ICHECK(sorted_token_ids.ndim() == 1 && sorted_token_ids.size(0) > 0)
      << "sorted_token_ids must be a non-empty rank-1 tensor";
  const int32_t num_pairs = static_cast<int32_t>(sorted_token_ids.size(0));
  TVM_FFI_ICHECK(expert_ids.ndim() == 1 && expert_ids.size(0) == num_pairs)
      << "expert_ids must have shape [num_pairs]";
  TVM_FFI_ICHECK(topk_weights.ndim() == 1 && topk_weights.size(0) == num_pairs)
      << "topk_weights must have shape [num_pairs]";
  TVM_FFI_ICHECK(lora_indices.ndim() == 1 && lora_indices.size(0) == num_tokens)
      << "lora_indices must have shape [num_tokens]";
  TVM_FFI_ICHECK(shrink_out.ndim() == 3 && shrink_out.size(0) == 1 &&
                 shrink_out.size(1) == num_pairs && shrink_out.size(2) == kRank)
      << "shrink_out must have shape [1, num_pairs, " << kRank << "]";
  TVM_FFI_ICHECK(y_accum.ndim() == 2 && y_accum.size(0) == num_tokens && y_accum.size(1) == hidden)
      << "y_accum must have shape [num_tokens, " << hidden << "]";
  TVM_FFI_ICHECK(lora_a.ndim() == 4 && lora_a.size(0) > 0 && lora_a.size(1) > 0 &&
                 lora_a.size(2) == kRank && lora_a.size(3) == hidden)
      << "lora_a must have shape [num_loras, num_experts, " << kRank << ", " << hidden << "]";
  const int32_t num_experts = static_cast<int32_t>(lora_a.size(1));
  TVM_FFI_ICHECK(lora_b.ndim() == 4 && lora_b.size(0) == lora_a.size(0) &&
                 lora_b.size(1) == num_experts && lora_b.size(2) == hidden &&
                 lora_b.size(3) == kRank)
      << "lora_b must have shape [num_loras, num_experts, " << hidden << ", " << kRank << "]";

  CheckCompact(x, "x");
  CheckCompact(shrink_out, "shrink_out");
  CheckCompact(lora_a, "lora_a");
  CheckCompact(lora_b, "lora_b");
  CheckCompact(sorted_token_ids, "sorted_token_ids");
  CheckCompact(expert_ids, "expert_ids");
  CheckCompact(lora_indices, "lora_indices");
  CheckCompact(topk_weights, "topk_weights");
  CheckCompact(route_index, "route_index");
  const int64_t route_words =
      kRouteIndexHeaderWords + static_cast<int64_t>(num_tokens) * kRouteIndexWordsPerToken;
  TVM_FFI_ICHECK(route_index.ndim() == 1 && route_index.size(0) >= route_words +
                                                                       kShrinkSplitPartialWords +
                                                                       kShrinkSplitCounterWords)
      << "route_index must hold at least " << kRouteIndexHeaderWords << " + num_tokens * "
      << kRouteIndexWordsPerToken << " + " << (kShrinkSplitPartialWords + kShrinkSplitCounterWords)
      << " int32 words";

  TVM_FFI_ICHECK(schedule_value >= static_cast<int64_t>(Schedule::kTokenOwnedT64) &&
                 schedule_value <= static_cast<int64_t>(Schedule::kTokenOwned))
      << "invalid Cake BGMV MoE generic schedule id: " << schedule_value;
  const auto schedule = static_cast<Schedule>(schedule_value);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  auto* y_ptr = static_cast<float*>(y_accum.data_ptr());
  auto* shrink_ptr = static_cast<unsigned short*>(shrink_out.data_ptr());
  auto* x_ptr = static_cast<unsigned short*>(x.data_ptr());
  auto* a_ptr = static_cast<unsigned short*>(lora_a.data_ptr());
  auto* b_ptr = static_cast<unsigned short*>(lora_b.data_ptr());
  auto* token_ptr = static_cast<long long*>(sorted_token_ids.data_ptr());
  auto* expert_ptr = static_cast<long long*>(expert_ids.data_ptr());
  auto* lora_ptr = static_cast<long long*>(lora_indices.data_ptr());
  auto* weight_ptr = static_cast<float*>(topk_weights.data_ptr());
  auto* route_ptr = static_cast<unsigned int*>(route_index.data_ptr());
  constexpr int32_t kRouteBuild = 1;
  constexpr int32_t kRouteLookup = 1;
  constexpr int32_t kRouteAdvance = 1;

  const int32_t num_tiles = (hidden + kShrinkTileElements - 1) / kShrinkTileElements;
  if (grouped != 0) {
    // Pair-grouped pipeline: group_hist -> group_scan -> group_scatter -> grouped shrink
    // -> grouped expand -> combine.
    CHECK_CUDA(group_workspace);
    CHECK_CUDA(group_partials);
    CHECK_DEVICE(x, group_workspace);
    CHECK_DEVICE(x, group_partials);
    CHECK_INPUT_TYPE(group_workspace, dl_int32);
    CHECK_INPUT_TYPE(group_partials, dl_float32);
    CheckCompact(group_workspace, "group_workspace");
    CheckCompact(group_partials, "group_partials");
    const int64_t num_loras = lora_a.size(0);
    const int64_t bins = num_loras * static_cast<int64_t>(num_experts);
    TVM_FFI_ICHECK(bins >= 1 && bins <= kGroupBinsMax)
        << "the grouped pipeline supports at most " << kGroupBinsMax << " (lora, expert) bins, got "
        << bins;
    const GroupedOffsets off = ComputeGroupedOffsets(num_pairs, num_tokens, bins);
    TVM_FFI_ICHECK(group_workspace.ndim() == 1 && group_workspace.size(0) >= off.numel)
        << "group_workspace must hold at least " << off.numel << " int32 words";
    TVM_FFI_ICHECK(group_partials.ndim() == 1 &&
                   group_partials.size(0) >= static_cast<int64_t>(num_pairs) * hidden)
        << "group_partials must hold at least num_pairs * hidden = "
        << static_cast<int64_t>(num_pairs) * hidden << " floats";
    TVM_FFI_ICHECK(off.max_tiles < (int64_t{1} << 31) &&
                   (num_pairs + kGroupTileTokens - 1) / kGroupTileTokens < 65536)
        << "grouped tile table out of range for num_pairs=" << num_pairs;
    auto* ws_ptr = static_cast<unsigned int*>(group_workspace.data_ptr());
    auto* partials_ptr = static_cast<float*>(group_partials.data_ptr());
    const int32_t max_tiles = static_cast<int32_t>(off.max_tiles);
    const int32_t hist_ctas = static_cast<int32_t>(off.hist_ctas);
    CAKE_BGMV_MOE_GROUP_HIST<<<hist_ctas, kGroupHistThreads, kGroupHistSmemBytes, stream>>>(
        token_ptr, expert_ptr, lora_ptr, shrink_ptr, num_pairs, num_tokens, num_experts,
        static_cast<int32_t>(num_loras), hist_ctas, ws_ptr, static_cast<int32_t>(off.hist));
    CheckCuda(cudaGetLastError(), "Cake BGMV MoE grouped group_hist launch");
    CheckCuda(LaunchGroupedPdl(CAKE_BGMV_MOE_GROUP_SCAN, dim3(1, 1, 1), dim3(kGroupThreads, 1, 1),
                               kGroupScanSmemBytes, stream, num_pairs, num_tokens, num_experts,
                               static_cast<int32_t>(num_loras), hist_ctas, ws_ptr,
                               static_cast<int32_t>(off.group_offset),
                               static_cast<int32_t>(off.tile_table),
                               static_cast<int32_t>(off.token_count),
                               static_cast<int32_t>(off.hist), static_cast<int32_t>(off.base)),
              "Cake BGMV MoE grouped group_scan launch");
    CheckCuda(LaunchGroupedPdl(
                  CAKE_BGMV_MOE_GROUP_SCATTER, dim3(hist_ctas, 1, 1), dim3(kGroupHistThreads, 1, 1),
                  kGroupScatterSmemBytes, stream, token_ptr, expert_ptr, lora_ptr, num_pairs,
                  num_tokens, num_experts, static_cast<int32_t>(num_loras), hist_ctas, ws_ptr,
                  static_cast<int32_t>(off.sorted_routes), static_cast<int32_t>(off.token_count),
                  static_cast<int32_t>(off.token_routes), static_cast<int32_t>(off.base)),
              "Cake BGMV MoE grouped group_scatter launch");
    // Route-tile parts interleaved in grid.x so the parts of a tile co-schedule
    // (repeated reads of the group weight rows hit L2).
    // Flat grid, rank tile fastest (the kernel decodes rank tile, part and route tile from
    // blockIdx.x). Rank tiles one CTA walks sequentially (lever 20; see
    // GroupShrinkRankTilesPerCta): short-K CTAs amortize the lookup chain and the x row reads; the
    // FMA chains and reductions are unchanged (bitwise identical stores for any value).
    const int32_t shrink_rt_per_cta = GroupShrinkRankTilesPerCta(num_tiles);
    const int32_t shrink_rt_groups =
        (kGroupShrinkRankTiles + shrink_rt_per_cta - 1) / shrink_rt_per_cta;
    const dim3 shrink_grid(max_tiles * (kGroupTileTokens / kGroupShrinkRoutes) * shrink_rt_groups,
                           1, 1);
    // Lever 20k: one rank tile per CTA takes the straight-line form (sm_90a pays ~10 % per shrink
    // CTA for the loop form at its 128-register budget; the stores are bitwise identical).
    const bool shrink_ring = GroupShrinkRing(num_tiles);
    // Lever 34: mixed-precision ring form on the bf16 Blackwell bundles (bitwise identical rows).
    const bool shrink_mixed = GroupShrinkMixed(shrink_ring);
    const auto shrink_kernel =
        shrink_mixed  ? (shrink_rt_per_cta == 1 ? CAKE_BGMV_MOE_SHRINK_GROUPED_RING_MIXED_SINGLE
                                                : CAKE_BGMV_MOE_SHRINK_GROUPED_RING_MIXED)
        : shrink_ring ? (shrink_rt_per_cta == 1 ? CAKE_BGMV_MOE_SHRINK_GROUPED_RING_SINGLE
                                                : CAKE_BGMV_MOE_SHRINK_GROUPED_RING)
                      : (shrink_rt_per_cta == 1 ? CAKE_BGMV_MOE_SHRINK_GROUPED_SINGLE
                                                : CAKE_BGMV_MOE_SHRINK_GROUPED);
    const int32_t shrink_smem = shrink_mixed  ? kShrinkGroupedRingMixedSmemBytes
                                : shrink_ring ? kShrinkGroupedRingSmemBytes
                                              : kShrinkGroupedSmemBytes;
    CheckCuda(LaunchGroupedPdl(shrink_kernel, shrink_grid, dim3(kShrinkThreads, 1, 1), shrink_smem,
                               stream, shrink_ptr, x_ptr, a_ptr, token_ptr, num_pairs, num_experts,
                               hidden, num_tiles, shrink_rt_per_cta, shrink_rt_groups, ws_ptr,
                               static_cast<int32_t>(off.group_offset),
                               static_cast<int32_t>(off.tile_table),
                               static_cast<int32_t>(off.sorted_routes)),
              "Cake BGMV MoE grouped shrink launch");
    // Column blocks one grouped-expand CTA walks (see GroupExpandColBlocksPerCta): the tile's route
    // ids and shrink rows are staged once per CTA and the expand's dependent lookup chain is paid
    // once per group of blocks; the per-(route, column) MMA and stores are unchanged (bitwise
    // identical partials for any value).
    const int32_t expand_col_blocks = (hidden + kGroupExpandThreads - 1) / kGroupExpandThreads;
    const int32_t expand_cbpc = GroupExpandColBlocksPerCta(hidden, expand_col_blocks);
    const dim3 expand_grid(max_tiles, (expand_col_blocks + expand_cbpc - 1) / expand_cbpc, 1);
    CheckCuda(LaunchGroupedPdl(CAKE_BGMV_MOE_EXPAND_GROUPED, expand_grid,
                               dim3(kGroupExpandThreads, 1, 1), kExpandGroupedSmemBytes, stream,
                               partials_ptr, shrink_ptr, b_ptr, num_pairs, num_experts, hidden,
                               ws_ptr, static_cast<int32_t>(off.group_offset),
                               static_cast<int32_t>(off.tile_table),
                               static_cast<int32_t>(off.sorted_routes), expand_cbpc),
              "Cake BGMV MoE grouped expand launch");
    const int32_t output_stride = hidden;
    const int32_t output_offset = 0;
    CheckCuda(LaunchGroupedPdl(CAKE_BGMV_MOE_COMBINE_GROUPED, dim3(num_tokens, 1, 1),
                               dim3(kGroupCombineThreads, 1, 1), kCombineGroupedSmemBytes, stream,
                               y_ptr, partials_ptr, token_ptr, lora_ptr, weight_ptr, num_pairs,
                               num_tokens, hidden, output_stride, output_offset, ws_ptr,
                               static_cast<int32_t>(off.token_count),
                               static_cast<int32_t>(off.token_routes)),
              "Cake BGMV MoE grouped combine launch");
    return;
  }
  TVM_FFI_ICHECK(shrink_splits >= 1 && shrink_splits <= kShrinkSplitMax &&
                 shrink_splits <= num_tiles)
      << "shrink_splits must be in [1, min(" << kShrinkSplitMax << ", num_tiles=" << num_tiles
      << ")], got " << shrink_splits;
  TVM_FFI_ICHECK(shrink_splits == 1 || num_pairs <= kShrinkSplitMaxPairs)
      << "hidden-split shrink supports at most " << kShrinkSplitMaxPairs << " pairs, got "
      << num_pairs;
  TVM_FFI_ICHECK(shrink_form >= kShrinkFormPrefill && shrink_form <= kShrinkFormPrefillS3)
      << "shrink_form must be 0 (two-stage prefill), 1 (decode) or 2 (three-stage prefill), got "
      << shrink_form;
  TVM_FFI_ICHECK(shrink_form != kShrinkFormDecode || num_pairs <= 32)
      << "the decode shrink kernel supports at most 32 pairs, got " << num_pairs;
  const int32_t splits = static_cast<int32_t>(shrink_splits);
  auto* split_partials = reinterpret_cast<float*>(route_ptr + route_words);
  auto* split_counters = route_ptr + route_words + kShrinkSplitPartialWords;
  // Lever 27: bin-ordered dispatch.  The order_build prologue writes the route
  // permutation into order_workspace and the shrink kernels read their route
  // through it (route_remap = 1).  Without the remap the kernels never
  // dereference the pointer (the plan passes a 1-word dummy).
  CHECK_CUDA(order_workspace);
  CHECK_DEVICE(x, order_workspace);
  CHECK_INPUT_TYPE(order_workspace, dl_int32);
  CheckCompact(order_workspace, "order_workspace");
  auto* order_ptr = static_cast<unsigned int*>(order_workspace.data_ptr());
  const int32_t route_remap = order_remap != 0 ? 1 : 0;
  constexpr int32_t kOrderOffset = 0;
  if (route_remap != 0) {
    const int64_t num_loras = lora_a.size(0);
    const int64_t bins = num_loras * static_cast<int64_t>(num_experts);
    TVM_FFI_ICHECK(bins >= 1 && bins <= kGroupBinsMax)
        << "order_remap supports at most " << kGroupBinsMax << " (lora, expert) bins, got " << bins;
    TVM_FFI_ICHECK(num_pairs <= kOrderRemapMaxPairs)
        << "order_remap supports at most " << kOrderRemapMaxPairs << " routes, got " << num_pairs;
    TVM_FFI_ICHECK(order_workspace.ndim() == 1 && order_workspace.size(0) >= num_pairs)
        << "order_workspace must hold at least num_pairs = " << num_pairs << " int32 words";
    CAKE_BGMV_MOE_ORDER_BUILD<<<dim3(1, 1, 1), dim3(kOrderBuildThreads, 1, 1), kOrderBuildSmemBytes,
                                stream>>>(token_ptr, expert_ptr, lora_ptr, num_pairs, num_tokens,
                                          num_experts, static_cast<int32_t>(num_loras), order_ptr,
                                          kOrderOffset);
    CheckCuda(cudaGetLastError(), "Cake BGMV MoE order_build launch");
  }
  const dim3 shrink_block(kShrinkThreads, 1, 1);
  // Kernel forms by launch mode: the PDL forms carry griddepcontrol (shrink:
  // launch_dependents, early or late by pdl_early; expand: wait after the
  // register B-row prefetch); the plain forms carry no PDL instruction.
  const bool prefetch_form = pdl_mode != 0;
  if (shrink_form == kShrinkFormDecode) {
    const dim3 shrink_grid(
        (num_pairs + kShrinkDecodePairsPerBlock - 1) / kShrinkDecodePairsPerBlock,
        kRank / kRankTile, splits);
    if (prefetch_form) {
      CAKE_BGMV_MOE_SHRINK_DECODE_PDL<<<shrink_grid, shrink_block, kShrinkDecodeSmemBytes,
                                        stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    } else {
      CAKE_BGMV_MOE_SHRINK_DECODE<<<shrink_grid, shrink_block, kShrinkDecodeSmemBytes, stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    }
  } else if (shrink_form == kShrinkFormPrefillS3) {
    // Lever 11c: three-stage ring for small grids (same FMA chain per lane: bitwise identical
    // to the two-stage form).
    const dim3 shrink_grid(num_pairs, kRank / kRankTile, splits);
    if (prefetch_form) {
      CAKE_BGMV_MOE_SHRINK_PREFILL_S3_PDL<<<shrink_grid, shrink_block, kShrinkPrefillS3SmemBytes,
                                            stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    } else {
      CAKE_BGMV_MOE_SHRINK_PREFILL_S3<<<shrink_grid, shrink_block, kShrinkPrefillS3SmemBytes,
                                        stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    }
  } else if (route_remap != 0) {
    // Lever 27: bin-ordered dispatch forms (the identity forms never read the permutation).
    const dim3 shrink_grid(num_pairs, kRank / kRankTile, splits);
    if (prefetch_form) {
      CAKE_BGMV_MOE_SHRINK_PREFILL_REMAP_PDL<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes,
                                               stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    } else {
      CAKE_BGMV_MOE_SHRINK_PREFILL_REMAP<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes,
                                           stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    }
  } else {
    const dim3 shrink_grid(num_pairs, kRank / kRankTile, splits);
    if (prefetch_form) {
      CAKE_BGMV_MOE_SHRINK_PREFILL_PDL<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes,
                                         stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    } else {
      CAKE_BGMV_MOE_SHRINK_PREFILL<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes, stream>>>(
          shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
          num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
          splits, pdl_early, order_ptr, kOrderOffset, route_remap);
    }
  }
  CheckCuda(cudaGetLastError(), "Cake BGMV MoE generic shrink launch");

  const int32_t output_stride = OutputRowStride(y_accum, num_tokens, hidden);
  const int32_t output_offset = 0;
  // Programmatic dependent launch: the shrink kernel triggers its dependents at
  // entry and the expand kernel executes griddepcontrol.wait before it reads the
  // shrink output or the route index, so the expand grid's launch and its B
  // weight prefetch overlap the shrink grid's tail.  Captured into a graph this
  // becomes a programmatic dependency edge.
  const int32_t expand_threads = schedule == Schedule::kTokenOwnedT64 ? 64 : 128;
  // PDL launches take the register-prefetch expand forms (both routes' B rows
  // are in flight before griddepcontrol.wait); plain launches take the round-4
  // interleaved forms, which keep the plain-launch occupancy.
  const int32_t col_blocks_total = (hidden + expand_threads - 1) / expand_threads;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(num_tokens, col_blocks_total, 1);
  config.blockDim = dim3(expand_threads, 1, 1);
  config.dynamicSmemBytes =
      schedule == Schedule::kTokenOwnedT64 ? kExpandT64SmemBytes : kExpandT128SmemBytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl_mode != 0 ? 1 : 0;
  config.attrs = attrs;
  config.numAttrs = 1;
  auto expand_kernel_t64 = prefetch_form ? CAKE_BGMV_MOE_EXPAND_T64_PF : CAKE_BGMV_MOE_EXPAND_T64;
  auto expand_kernel_t128 =
      prefetch_form ? CAKE_BGMV_MOE_EXPAND_T128_PF : CAKE_BGMV_MOE_EXPAND_T128;
  const cudaError_t expand_status =
      schedule == Schedule::kTokenOwnedT64
          ? cudaLaunchKernelEx(&config, expand_kernel_t64, y_ptr, shrink_ptr, b_ptr, token_ptr,
                               expert_ptr, lora_ptr, weight_ptr, num_pairs, num_experts, num_tokens,
                               output_stride, output_offset, route_ptr, kRouteLookup, kRouteAdvance,
                               hidden)
          : cudaLaunchKernelEx(&config, expand_kernel_t128, y_ptr, shrink_ptr, b_ptr, token_ptr,
                               expert_ptr, lora_ptr, weight_ptr, num_pairs, num_experts, num_tokens,
                               output_stride, output_offset, route_ptr, kRouteLookup, kRouteAdvance,
                               hidden);
  CheckCuda(expand_status, "Cake BGMV MoE generic expand launch");
}

}  // namespace cake_bgmv_moe_generic
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, flashinfer::cake_bgmv_moe_generic::Configure);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_bgmv_moe_generic::Run);
