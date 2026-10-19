// SM90 MXFP4 MegaMoE fused kernel body.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or \
    defined(__CLION_IDE__)
using Barrier = cutlass::arch::ClusterTransactionBarrier;

// =====================================================================
// Template checks
// =====================================================================
FI_MEGAMOE_STATIC_ASSERT(BLOCK_M == 8 || BLOCK_M == 16 || BLOCK_M == 24 || BLOCK_M == 32 ||
                             BLOCK_M == 64 || BLOCK_M == 128,
                         "H200 fused kernel requires BM8/BM16/BM24/BM32/BM64/BM128");
// Pipeline depth used to be pinned to one value per BLOCK_M, which made the
// shipped table the only reachable schedule. The binding constraint is
// shared memory, and the layout assertions below already enforce it
// exactly, so only the pipelining minimum is asserted here.
FI_MEGAMOE_STATIC_ASSERT(kNumStages >= 2, "Pipelining needs at least two stages");
FI_MEGAMOE_STATIC_ASSERT(BLOCK_M != 128 || kNumStages == 6,
                         "BM128 split-M is only validated at stage depth 6");
FI_MEGAMOE_STATIC_ASSERT((BLOCK_M == 128) == (BLOCK_N == 128),
                         "BM128 is paired with the BN128 split-M topology");
// FP8MMASelector takes N=32, and the N_SWAP ladder below has a BM32 arm.
FI_MEGAMOE_STATIC_ASSERT(!kSwapABRequested || BLOCK_M <= 32,
                         "swap-AB packs tokens into the WGMMA N dimension");

// =====================================================================
// Thread / warp identification
// =====================================================================
const uint32_t sm_idx = blockIdx.x;
const uint32_t thread_idx = threadIdx.x;
const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
const uint32_t lane_idx = ptx::get_lane_idx();

// Phase timestamps (globaltimer, ns), written only when kPhaseStamps.
// Compiled out entirely otherwise -- see the note on the template parameter.
//
//   0  min  kernel entry              1  max  dispatch barrier #1 done
//   2  max  pool ready                3  min  first math task
//   4  max  last L1 task end          5  max  last L2 task end
//   6  max  combine barrier #2 done   7  max  combine end
//   8  max  cleanup barrier #3 done
//
// Accumulators, summed over every CTA (value, then count):
//  20/21 L1 task ns     22/23 L2 task ns
//  24/25 decode warps waiting on the weight TMA (decode starved by the load)
//  26/27 math warps waiting on the decoded ring (WGMMA starved by decode)
// 24 and 26 are the pair that says whether the mainloop is stream-bound or
// decode-bound, which is the question the end-to-end numbers cannot answer.
const auto stamp_min = [&](const uint32_t slot) {
  if constexpr (kPhaseStamps) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    atomicMin(phase_stamps + slot, t);
  }
};
const auto stamp_max = [&](const uint32_t slot) {
  if constexpr (kPhaseStamps) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    atomicMax(phase_stamps + slot, t);
  }
};
const auto stamp_now = [&]() -> unsigned long long {
  unsigned long long t = 0;
  if constexpr (kPhaseStamps) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
};
const auto stamp_add = [&](const uint32_t slot, const unsigned long long delta) {
  if constexpr (kPhaseStamps) {
    atomicAdd(phase_stamps + slot, delta);
    atomicAdd(phase_stamps + slot + 1, 1ull);
  }
};
(void)stamp_now;
(void)stamp_add;

if (warp_idx == 0 and cute::elect_one_sync()) {
  stamp_min(0);
  cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
  cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
  cute::prefetch_tma_descriptor(&tensor_map_l1_output);
  cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
  cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
}

// =====================================================================
// Workspaces and symmetric buffer slicing. The framework reserves
// per-64 SF capacity; this per-128 path uses its first half
// as a dense per-128 layout so no framework allocation change is needed.
// =====================================================================
#include <flashinfer/megamoe/detail/buffers.inl>

// =====================================================================
// GEMM data types and shape constants
// =====================================================================
using a_dtype_t = cutlass::float_e4m3_t;
using b_dtype_t = cutlass::float_e4m3_t;
using task_info_t = sched::TaskInfo;
using interleaved_scheduler_t =
    sched::InterleavedMegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                       L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank, kNumSMs,
                                       kNumRanks>;
constexpr uint32_t kNumRoutedL1BlockNs = L1_SHAPE_N / BLOCK_N;
constexpr bool kSplitMDecodedWeightReuse =
    BLOCK_M == 128 && BLOCK_N == 128 && kNumEpilogueWarpgroups == 2;
constexpr uint32_t WG_BLOCK_M = kSplitMDecodedWeightReuse ? BLOCK_M / 2 : BLOCK_M;
constexpr uint32_t WG_BLOCK_N = kSplitMDecodedWeightReuse ? BLOCK_N : BLOCK_N / 2;
constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;        // post-SwiGLU tile N
constexpr uint32_t WG_L1_OUT_BLOCK_N = WG_BLOCK_N / 2;  // post-SwiGLU per-WG N
constexpr uint32_t kSwapABTokenChunks = BLOCK_M / 8;
constexpr uint32_t kSwapABWeightHalves = WG_BLOCK_N / 64;
constexpr uint32_t kSwapABHalfAccumPerThread = 64 * 64 / 128;
FI_MEGAMOE_STATIC_ASSERT(!kSwapABRequested || WG_L1_OUT_BLOCK_N == 64,
                         "swapAB expects BN256 split-N with 64 L1 output columns per WG");
// Both dispatch warps participate in CTA-wide barriers. Selected plans may
// use one warp for routing and token pulls, leaving the other warp's send
// buffer available for an additional GEMM stage.
constexpr uint32_t kNumActiveDispatchWarps = kSingleActiveDispatchWarp ? 1u : kNumDispatchWarps;
constexpr uint32_t kNumActiveDispatchThreads = kNumActiveDispatchWarps * 32;
constexpr bool kQuadDequantIlp = BLOCK_M == 8 && kNumStages == 4;
using L1WGMMA = typename mma::sm90::FP8MMASelector<WG_BLOCK_N>::type;
static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
              "Unexpected WGMMA shape");
// A and B are CTA-local in the fixed cluster-size-one plan.
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);  // 128
constexpr uint32_t kL2ActsSFGranK = kSplitMDecodedWeightReuse ? 64u : 128u;
FI_MEGAMOE_STATIC_ASSERT(kSplitMDecodedWeightReuse || WG_L1_OUT_BLOCK_N < kL2ActsSFGranK,
                         "split-N warpgroups must share one L2 activation scale");

// =====================================================================
// Shared memory layout
// =====================================================================
constexpr uint32_t kSharedMemoryAlignment = 1024;
extern __shared__ __align__(kSharedMemoryAlignment)
uint8_t smem_buffer[];

constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
    math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
constexpr uint32_t SMEM_SEND_BUFFER_SIZE = math::constexpr_align(
    fp8_token_layout.get_num_bytes() * kNumActiveDispatchWarps, kSharedMemoryAlignment);
// Scaled-magnitude tables for all 256 E8M0 codes; see kScaledLutSize.
constexpr uint32_t SMEM_MXFP4_LUT_SIZE = math::constexpr_align<uint32_t>(
    mxfp4::kScaledLutSize * sizeof(mxfp4::ScaledLut), kSharedMemoryAlignment);
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
// BM128 split-M alternates two decoded-B slots. This lets one WG begin
// decoding K+1 after its K WGMMA completes without overwriting the slot
// that the paired WG may still be consuming.
//
// Every other plan needs exactly one. Decode and consume are serialized
// inside a K-block iteration -- the warpgroup drains its own WGMMA with
// warpgroup_wait<0> before it loops -- and the two warpgroups decode and
// read disjoint N halves of the same BN256 tile, so no slot is ever live
// across iterations. Sizing this ring by kNumStages spent
// (kNumStages - 1) * BN256 * BK128 bytes (64 KB at three stages) to hold
// tiles nobody could read, and that is what capped the TMA pipeline at
// three or four stages.
// One slot is enough outside split-M. Decode and consume are serialized
// inside a K-block iteration -- the warpgroup drains its own WGMMA with
// warpgroup_wait<0> before it loops -- and the two warpgroups decode and
// read disjoint N halves of the same BN256 tile, so no slot is live across
// iterations.
//
// Rotating two slots to decode K+1 under K's in-flight WGMMAs was measured
// and is worse: at MiMo/H20 M=64 it cost +5.9 / +7.1 / +10.8 % over three
// runs. The prefetch has to wait on stage K+1's TMA barrier, and that wait
// sits between the WGMMA issue and arrive_empty(K), so it delays the loader
// by exactly as much as it saves on the decode.
FI_MEGAMOE_STATIC_ASSERT(!kUseRSOperand || kSwapABRequested,
                         "The register-source A path is swapAB-only");
constexpr uint32_t kNumDecodedBStages =
    (kSplitMDecodedWeightReuse || kSplitDecodeWarps) ? 2u : (kUseRSOperand ? 0u : 1u);
// Only the split hands decoded tiles between warps, so only it needs the
// ring's barriers; every other plan decodes and consumes in one warp.
constexpr uint32_t kNumBRingStages = kSplitDecodeWarps ? 2u : 0u;
// Tile geometry (chunk-major, unpadded) lives with the decoders that read
// it; the loader only needs the tile size and a fixed row count, so the
// layout does not depend on the selected BLOCK_N.
// Local copies: `tma_load_1d` takes its byte count by reference, which
// odr-uses the operand, and a namespace-scope constant is host-side.
constexpr uint32_t kBTileRows = mxfp4::kBTileRows;
constexpr uint32_t kBTileBytes = mxfp4::kBTileBytes;
constexpr uint32_t SMEM_PACKED_B_SIZE_PER_STAGE =
    (LOAD_BLOCK_N / kBTileRows) * kBTileBytes * sizeof(b_dtype_t);
FI_MEGAMOE_STATIC_ASSERT(LOAD_BLOCK_N % kBTileRows == 0,
                         "BLOCK_N must be a whole number of weight tiles");
FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_N % kBTileRows == 0 && L2_SHAPE_N % kBTileRows == 0,
                         "Weight N must be a whole number of weight tiles");
// L1 and L2 each consume one per-128 activation scale per row and K tile.
constexpr uint32_t kL2SFAHalfStride =
    math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
constexpr uint32_t kNumL2SFAGroups = kSplitMDecodedWeightReuse ? 2u : 1u;
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kNumL2SFAGroups * kL2SFAHalfStride * sizeof(float);
// CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
// L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
constexpr uint32_t SMEM_CD_L1_SIZE =
    kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
constexpr uint32_t SMEM_CD_L2_SIZE =
    kSwapABRequested ? BLOCK_M * BLOCK_N * sizeof(nv_bfloat16) : 0u;
constexpr uint32_t SMEM_CD_OUTPUT_BASE_SIZE =
    SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
constexpr uint32_t SMEM_CD_L1_SHARED_SF_SLOTS = kNumEpilogueWarpgroups * BLOCK_M;
constexpr uint32_t SMEM_CD_L1_SWAP_AMAX_SLOTS = kSwapABRequested ? BLOCK_M * kNumEpilogueWarps : 0u;
constexpr uint32_t SMEM_CD_L1_EXTRA_FLOAT_SLOTS =
    SMEM_CD_L1_SHARED_SF_SLOTS > SMEM_CD_L1_SWAP_AMAX_SLOTS ? SMEM_CD_L1_SHARED_SF_SLOTS
                                                            : SMEM_CD_L1_SWAP_AMAX_SLOTS;
constexpr uint32_t SMEM_CD_L1_SHARED_SF_SIZE = SMEM_CD_L1_EXTRA_FLOAT_SLOTS * sizeof(float);
constexpr uint32_t SMEM_CD_OUTPUT_UNALIGNED_SIZE =
    SMEM_CD_OUTPUT_BASE_SIZE + SMEM_CD_L1_SHARED_SF_SIZE;
constexpr uint32_t SMEM_CD_SIZE =
    math::constexpr_align(SMEM_CD_OUTPUT_UNALIGNED_SIZE, kSharedMemoryAlignment);

constexpr uint32_t SMEM_GEMM_END_SIZE =
    SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_MXFP4_LUT_SIZE + SMEM_CD_SIZE +
    kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
    kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
// The combine phase reuses this whole region as its own ring, so it sets a
// floor on it that has nothing to do with the GEMM. Deriving the chunk
// count from kHidden alone keeps that floor non-circular; the static
// assertion further down re-derives it from the final region size and
// would fire if the two ever disagreed.
constexpr uint32_t kCombineNumChunks = kHidden <= 4096 ? 1u : 2u;
constexpr uint32_t SMEM_COMBINE_MIN_SIZE =
    3u * kNumEpilogueWarps * kHidden * sizeof(nv_bfloat16) / kCombineNumChunks;
constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
    SMEM_GEMM_END_SIZE > SMEM_COMBINE_MIN_SIZE ? SMEM_GEMM_END_SIZE : SMEM_COMBINE_MIN_SIZE;

// SMEM pointers
auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
const auto smem_send_buffers =
    layout::Buffer(fp8_token_layout, kNumActiveDispatchWarps, 1,
                   math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
auto smem_mxfp4_lut = reinterpret_cast<mxfp4::ScaledLut*>(
    math::advance_ptr<uint8_t>(smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));
auto smem_gemm_base = math::advance_ptr(
    smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_MXFP4_LUT_SIZE);

auto smem_cd_base = smem_gemm_base;
// CD output is shared by L1 (FP8) and L2 (BF16); reinterpret-cast as needed.
auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);
auto smem_cd_l1_shared_sf = math::advance_ptr<float>(smem_cd_base, SMEM_CD_OUTPUT_BASE_SIZE);
auto smem_cd_l2 = reinterpret_cast<nv_bfloat16*>(smem_cd_base);

auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
});
auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
  // Callers pass a decoded-slot index, not a pipeline stage. The RS path
  // allocates no decoded tile at all, so the index degenerates to zero and
  // the pointer is never dereferenced.
  const uint32_t decoded_stage = kSplitMDecodedWeightReuse
                                     ? (i & 1u)
                                     : (kNumDecodedBStages <= 1u ? 0u : (i % kNumDecodedBStages));
  return math::advance_ptr<b_dtype_t>(
      smem_gemm_base,
      SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + decoded_stage * SMEM_B_SIZE_PER_STAGE);
});
auto smem_packed_b = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<b_dtype_t>(smem_gemm_base,
                                      SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE +
                                          kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE +
                                          i * SMEM_PACKED_B_SIZE_PER_STAGE);
});
auto sf_start_ptr = math::advance_ptr<uint8_t>(
    smem_gemm_base, SMEM_CD_SIZE +
                        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_PACKED_B_SIZE_PER_STAGE) +
                        kNumDecodedBStages * SMEM_B_SIZE_PER_STAGE);
auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
  return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
});
// Barriers live after SF.
auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_BEFORE_BARRIER_SIZE);
auto dispatch_barriers =
    utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
auto full_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
auto empty_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
auto combine_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
// The decoded-B ring is a two-stage pipeline in its own right, so it gets
// the same barrier pair the stage pipeline uses.
constexpr uint32_t kNumRingBarriers = kNumDispatchWarps + kNumStages * 2 + kNumEpilogueWarps * 2;
auto decoded_full_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumRingBarriers + i; });
auto decoded_empty_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumRingBarriers + kNumBRingStages + i; });
constexpr uint32_t kNumBaseBarriers = kNumRingBarriers + kNumBRingStages * 2;
auto task_info_full_barriers = barrier_start_ptr + kNumBaseBarriers;
auto task_info_empty_barriers =
    task_info_full_barriers + interleaved_scheduler_t::kNumScheduleStages;
auto task_infos = reinterpret_cast<task_info_t*>(task_info_empty_barriers +
                                                 interleaved_scheduler_t::kNumScheduleStages);
constexpr uint32_t kInterleavedSchedulerSMEMBytes =
    2 * interleaved_scheduler_t::kNumScheduleStages * sizeof(Barrier) +
    interleaved_scheduler_t::kNumScheduleStages * sizeof(task_info_t);
FI_MEGAMOE_STATIC_ASSERT(kInterleavedSchedulerSMEMBytes ==
                             layout::kSM90InterleavedSchedulerSMEMBytes,
                         "Host and device scheduler shared-memory layouts disagree");
constexpr uint32_t kInterleavedSMEMEnd =
    SMEM_BEFORE_BARRIER_SIZE + kNumBaseBarriers * sizeof(Barrier) + kInterleavedSchedulerSMEMBytes;
FI_MEGAMOE_STATIC_ASSERT(!kUseInterleavedScheduler || kInterleavedSMEMEnd <= 232448,
                         "Interleaved scheduler exceeds the SM90 shared-memory capacity");

// =====================================================================
// Initialization
// =====================================================================
mxfp4::init_scaled_lut(smem_mxfp4_lut, thread_idx, kNumThreads);

if (warp_idx == 0) {
// Clean expert-count shared memory
#pragma unroll
  for (uint32_t i = lane_idx; i < kNumExperts; i += 32) ptx::st_shared(smem_expert_count + i, 0u);
} else if (warp_idx == 1) {
// Init dispatch m-barriers
#pragma unroll
  for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32) dispatch_barriers[i]->init(1);
  cutlass::arch::fence_barrier_init();
} else if (warp_idx == 2) {
  // Init GEMM full/empty barriers and combine barriers
  if (cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
      // cp.async.bulk and counted as B-loader transaction bytes, so
      // it does not need a separate producer arrival.
      full_barriers[i]->init(2);
      // A stage holds both the packed weights and the activations.
      // With split decode the two are read by different warps and
      // both must release it, or the A loader refills the stage
      // under a running WGMMA.
      empty_barriers[i]->init(kNumEpilogueWarps + (kSplitDecodeWarps ? kNumDecodeWarps : 0u));
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++i) combine_barriers[i]->init(1);
#pragma unroll
    for (uint32_t i = 0; i < kNumBRingStages; ++i) {
      // One arrival per warp on each side: the tile is not readable
      // until every decode warp has written its rows.
      decoded_full_barriers[i]->init(kNumDecodeWarps);
      decoded_empty_barriers[i]->init(kNumEpilogueWarps);
    }
    if constexpr (kUseInterleavedScheduler) {
#pragma unroll
      for (uint32_t i = 0; i < interleaved_scheduler_t::kNumScheduleStages; ++i) {
        task_info_full_barriers[i].init(1);
        task_info_empty_barriers[i].init(kNumEpilogueWarps +
                                         (kSplitDecodeWarps ? kNumDecodeWarps : 0u));
      }
    }
  }
  cutlass::arch::fence_barrier_init();
}
__syncthreads();

// =====================================================================
// Scheduler (cluster=1)
// =====================================================================
auto scheduler = sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                         L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank,
                                         kNumExpertsPerWave, kNumSMs, kNumRanks, 1>(workspace);
auto interleaved_scheduler = interleaved_scheduler_t(workspace, task_info_full_barriers,
                                                     task_info_empty_barriers, task_infos);

// Pipeline state shared by TMA loaders and math warpgroups
uint32_t stage_idx = 0, phase = 0;
// The ring advances with the stage pipeline and, like it, runs continuously
// across blocks: its barriers carry state, so resetting the slot per block
// would leave the producer a slot ahead of its consumer.
uint32_t b_ring_idx = 0, b_ring_phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
  ++k_block_idx;
  stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
  phase ^= stage_idx == 0;
  if constexpr (kSplitDecodeWarps) {
    b_ring_idx = b_ring_idx == kNumBRingStages - 1 ? 0 : b_ring_idx + 1;
    b_ring_phase ^= b_ring_idx == 0;
  }
};
// Intra-SM barrier indices (mirroring SM100)
constexpr uint32_t kDispatchBarrierIdx = 0;
constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
constexpr uint32_t kEpilogueFullBarrierIdx = 2;
constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;
constexpr uint32_t kSplitMDecodeBarrierIdx = 8;

// Cross-rank NVLink barrier tags
constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

// Register reconfiguration counts (chosen to fit in 64512 reg budget).
constexpr uint32_t kNumDispatchRegisters = 48;
constexpr uint32_t kNumNonEpilogueRegisters = kUseInterleavedScheduler ? 64 : 40;
constexpr uint32_t kNumDecodeRegisters = 48;
// The decode warps take their share out of the epilogue's budget, so the
// split runs the math warps leaner.
constexpr uint32_t kNumEpilogueRegisters = kSplitDecodeWarps ? 152 : 208;
// The register file is handed out per warp in 256-register granules, so
// what setmaxnreg can redistribute is not 64K but what the launch could
// actually reserve at this width -- 64512 at 384 threads, 61440 at 640.
constexpr uint32_t kNumWarps = kNumThreads / 32;
constexpr uint32_t kRegisterBudget = 65536 / kNumWarps / 256 * 256 * kNumWarps;
// setmaxnreg acts per warpgroup, so the budget has to be counted that way:
// dispatch and the two loaders share one warpgroup, which is held at the
// larger of their two requests. Counting per role instead understates the
// total, and an over-subscribed warpgroup_reg_alloc does not fail -- it
// blocks, so the kernel hangs rather than refusing to launch.
constexpr uint32_t kNumLoaderWarpgroupRegisters = kNumDispatchRegisters > kNumNonEpilogueRegisters
                                                      ? kNumDispatchRegisters
                                                      : kNumNonEpilogueRegisters;
FI_MEGAMOE_STATIC_ASSERT(kNumLoaderWarpgroupRegisters * 128 +
                                 kNumDecodeRegisters * kNumDecodeThreads +
                                 kNumEpilogueRegisters * kNumEpilogueThreads <=
                             kRegisterBudget,
                         "Too many registers");

constexpr uint32_t kDispatchGridSyncIndex = 0;
constexpr uint32_t kEpilogueGridSyncIndex = 1;

const auto for_each_static_selected_block = [&](auto&& func) {
  scheduler.fetch_expert_recv_count();
  scheduler.set_expert_idx(0);
  while (true) {
    CUTE_TIE_DECL(scheduler.get_next_block(), block_phase, local_expert_idx, m_block_idx,
                  n_block_idx);
    if (block_phase == sched::BlockPhase::None) break;
    if (block_phase == sched::BlockPhase::Linear1) {
      func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
           local_expert_idx, L1_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
           scheduler.get_current_pool_block_offset() + m_block_idx,
           scheduler.template get_valid_m<false>());
    } else {
      func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
           local_expert_idx, L2_SHAPE_K / BLOCK_K, m_block_idx, n_block_idx,
           scheduler.get_current_pool_block_offset() + m_block_idx,
           scheduler.template get_valid_m<false>());
    }
  }
};

const auto invoke_interleaved_task = [&](const task_info_t& task_info, auto&& func) {
  if (task_info.block_phase == sched::BlockPhase::Linear1) {
    func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{},
         task_info.local_expert_idx, L1_SHAPE_K / BLOCK_K, task_info.m_block_idx,
         task_info.n_block_idx, task_info.pool_block_idx, task_info.valid_m);
  } else {
    func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{},
         task_info.local_expert_idx, L2_SHAPE_K / BLOCK_K, task_info.m_block_idx,
         task_info.n_block_idx, task_info.pool_block_idx, task_info.valid_m);
  }
};

const auto for_each_published_block = [&](auto&& func) {
  task_info_t task_info;
  while (interleaved_scheduler.get_published_task(task_info))
    invoke_interleaved_task(task_info, func);
};

const auto produce_interleaved_blocks = [&](auto&& func) {
  interleaved_scheduler.fetch_expert_recv_count();
  while (true) {
    interleaved_scheduler.wait_task_slot_empty();
    const auto task_info = interleaved_scheduler.claim_next_task();
    interleaved_scheduler.publish_task(task_info);
    if (!task_info.is_valid()) break;
    invoke_interleaved_task(task_info, func);
  }
};

const auto cleanup_workspace = [&]() {
  FI_MEGAMOE_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
  if (sm_idx == 0) {
#pragma unroll
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
      *workspace.get_expert_send_count_ptr(i) = 0;
    if constexpr (kUseInterleavedScheduler) {
      if (thread_idx == 0) {
        *workspace.get_l1_task_count_ptr() = 0;
        *workspace.get_l2_task_count_ptr() = 0;
      }
    }
  } else {
    for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
      const auto num_recv_tokens =
          static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
      const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
      const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

      FI_MEGAMOE_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
      if (warp_idx == 0) {
        *workspace.get_expert_recv_count_sum_ptr(i) = 0;
      } else if (warp_idx == 1) {
        if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
          ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
        __syncwarp();
      }

      for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
        *workspace.get_expert_recv_count_ptr(j, i) = 0;
      __syncwarp();

      for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
        *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
        *workspace.get_l2_arrival_mask_ptr(cleanup_pool_block_offset + j) = 0;
      }
      __syncwarp();
    }
  }
};

// =====================================================================
// ROLE 1: DISPATCH WARPS
//   Mirrors SM100 dispatch with two changes:
//     * SF is per-128 channel float (no UTCCP transpose). We store the
//       remote per-token SF directly into the local L1 SF buffer in
//       MN-major layout: `local_sf[k_chunk * num_padded_sf_pool_tokens + token_idx]`.
//     * The "token_idx_in_expert" → SF token index is now the simple
//       per-block linear mapping (no 4×32 transpose).
// =====================================================================
if (warp_idx < kNumDispatchWarps) {
  cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

  FI_MEGAMOE_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
  constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
  const auto read_topk_idx = [&](const auto& process) {
    if (warp_idx < kNumActiveDispatchWarps) {
#pragma unroll
      for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
           i < num_tokens; i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
        int expert_idx = -1;
        if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
          expert_idx = static_cast<int>(
              __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
          if (expert_idx >= 0) process(i * kNumTopk + lane_idx, expert_idx);
        }
        __syncwarp();
      }
    }
  };

  // Count tokens per expert
  read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    atomicAdd_block(smem_expert_count + expert_idx, 1);
  });
  ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

// Stake out per-expert SM offsets via global atomic
#pragma unroll
  for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
    smem_expert_count[i] =
        static_cast<uint32_t>(ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
  }
  ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

  // Write source token-topk indices to remote ranks
  read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(expert_idx % kNumExpertsPerRank,
                                                              sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
  });

  comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(workspace, sm_idx, thread_idx, [=]() {
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
  });

  if (sm_idx == 0 and thread_idx < kNumActiveDispatchThreads) {
#pragma unroll
    for (uint32_t i = thread_idx; i < kNumExperts; i += kNumActiveDispatchThreads) {
      const auto dst_rank_idx = i / kNumExpertsPerRank;
      const auto dst_local_expert_idx = i % kNumExpertsPerRank;
      const auto expert_status = *workspace.get_expert_send_count_ptr(i);
      *sym_buffer.map(
          workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
          dst_rank_idx) = expert_status & 0xffffffff;
      ptx::atomic_add_sys(
          sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx),
                         dst_rank_idx),
          expert_status);
    }
  }
  ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

  comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, kDispatchGridSyncIndex,
                       kBeforeDispatchPullBarrierTag>(
      workspace, sym_buffer, sm_idx, thread_idx,
      [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }, false, true);
  if (thread_idx == 0) stamp_max(1);

  // Sync with epilogue warps before pulling tokens
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  // Token / SF pull loop
  if (warp_idx < kNumActiveDispatchWarps) {
    uint32_t pull_mbarrier_phase = 0;
    const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
    const auto pull_mbarrier = dispatch_barriers[warp_idx];

    scheduler.fetch_expert_recv_count();

    constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
    int current_expert_idx = -1;
    uint32_t stored_rank_count[kNumRanksPerLane] = {};
    uint32_t expert_start_idx = 0, expert_end_idx = 0;
    uint32_t expert_pool_block_offset = 0;

    constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumActiveDispatchWarps;
    for (uint32_t token_idx = sm_idx * kNumActiveDispatchWarps + warp_idx;;
         token_idx += kNumGlobalWarps) {
      int old_expert_idx = current_expert_idx;
      while (token_idx >= expert_end_idx) {
        if (++current_expert_idx >= kNumExpertsPerRank) break;
        expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
        expert_start_idx = expert_end_idx;
        expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
      }
      if (current_expert_idx >= kNumExpertsPerRank) break;

      if (old_expert_idx != current_expert_idx) {
        old_expert_idx = current_expert_idx;
#pragma unroll
        for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
          const uint32_t j = i * 32 + lane_idx;
          stored_rank_count[i] = j < kNumRanks
                                     ? static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(
                                           j, current_expert_idx))
                                     : 0;
        }
      }

      // Round-robin rank selection (identical to SM100)

#include <flashinfer/megamoe/detail/dispatch_rank_selection.inl>
      const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
          current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
      const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
      const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

      // TMA pull token data into SMEM
      if (cute::elect_one_sync()) {
        ptx::tma_load_1d(
            pull_buffer.get_base_ptr(),
            sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                           current_rank_in_expert_idx),
            pull_mbarrier, kHidden);
      }
      __syncwarp();

      // Copy SF: per-128 K floats, written linearly (no UTCCP transpose).
      constexpr uint32_t kNumSFFloats = kHidden / 128;
      FI_MEGAMOE_STATIC_ASSERT(kNumSFFloats > 0 and kHidden % 128 == 0, "Invalid SF");
      const auto remote_sf_ptr =
          sym_buffer.map(input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<float>(),
                         current_rank_in_expert_idx);
      const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<float>();
      const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
#pragma unroll
      for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++i) {
        const uint32_t j = i * 32 + lane_idx;
        if (j < kNumSFFloats)
          local_sf_ptr[j * kNumPaddedSFPoolTokens + pool_token_idx] = remote_sf_ptr[j];
      }
      __syncwarp();

      if (cute::elect_one_sync()) {
        const auto weight =
            *sym_buffer.map(input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                            current_rank_in_expert_idx);
        *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
        ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

        ptx::tma_store_1d(l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                          pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

        *workspace.get_token_src_metadata_ptr(pool_token_idx) = {current_rank_in_expert_idx,
                                                                 src_token_idx, src_topk_idx};

        cute::tma_store_arrive();
        ptx::tma_store_wait<0>();
        ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(expert_pool_block_offset +
                                                            token_idx_in_expert / BLOCK_M),
                         1u);
      }
      __syncwarp();
    }
  }

  // Cleanup workspace, overlapping with combine
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  cleanup_workspace();
  comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads, kDispatchGridSyncIndex,
                       kAfterWorkspaceCleanBarrierTag>(
      workspace, sym_buffer, sm_idx, thread_idx,
      [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }, true, false);
  if (thread_idx == 0) stamp_max(8);
} else if (warp_idx == kNumDispatchWarps) {
  // =====================================================================
  // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
  //   The two warps inside `kNumNonEpilogueThreads` load A + SFA and
  //   B + SFB, respectively.
  // =====================================================================
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  const auto load_a_task = [&](const auto& block_phase, const uint32_t& local_expert_idx,
                               const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                               const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
                               const uint32_t& valid_m) {
    using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
    constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
    const auto tensor_map_a_ptr = kBlockIsL2 ? &tensor_map_l2_acts : &tensor_map_l1_acts;
    const auto tensor_map_sfa_ptr = kBlockIsL2 ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

    const bool has_valid_m = valid_m > 0;

    // Wait for the pool to be ready.
    if (has_valid_m) {
      if constexpr (!kBlockIsL2) {
        const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
        while (ptx::ld_acq(ptr) != valid_m) {
        }
      } else {
        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
        const uint64_t expected =
            (kNumRoutedL1BlockNs >= 64) ? ~0ull : ((1ull << kNumRoutedL1BlockNs) - 1ull);
        while (ptx::ld_acq_gpu(ptr) != expected) {
        }
      }
    }
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      if (cute::elect_one_sync()) {
        if (has_valid_m) {
          const uint32_t m_idx = pool_block_idx * BLOCK_M;
          const uint32_t k_idx = k_block_idx * BLOCK_K;

          // TMA load A
          tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
              tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1);

          if constexpr (!kBlockIsL2 || !kSplitMDecodedWeightReuse) {
            tma::copy<BLOCK_M, 1, 0, float>(tensor_map_sfa_ptr, full_barriers[stage_idx],
                                            smem_sfa[stage_idx], m_idx, k_block_idx, 1);
            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE +
                                                           BLOCK_M * sizeof(float));
          } else {
            // BN128 L1 produces per-64 activation scales. L2
            // consumes both scale groups for each BK128 tile.
            tma::copy<BLOCK_M, 1, 0, float>(tensor_map_sfa_ptr, full_barriers[stage_idx],
                                            smem_sfa[stage_idx], m_idx, k_block_idx * 2, 1);
            tma::copy<BLOCK_M, 1, 0, float>(tensor_map_sfa_ptr, full_barriers[stage_idx],
                                            smem_sfa[stage_idx] + kL2SFAHalfStride, m_idx,
                                            k_block_idx * 2 + 1, 1);
            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE +
                                                           2 * BLOCK_M * sizeof(float));
          }
        } else {
          full_barriers[stage_idx]->arrive();
        }
      }
      __syncwarp();
    }
  };
  if constexpr (kUseInterleavedScheduler)
    for_each_published_block(load_a_task);
  else
    for_each_static_selected_block(load_a_task);

} else if (warp_idx == kNumDispatchWarps + 1) {
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  const auto load_b_task = [&](const auto& block_phase, const uint32_t& local_expert_idx,
                               const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                               const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
                               const uint32_t& valid_m) {
    using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
    constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
    const uint8_t* b_base = kBlockIsL2 ? l2_weights : l1_weights;
    constexpr uint32_t shape_n = kBlockIsL2 ? L2_SHAPE_N : L1_SHAPE_N;
    // One tile is `kBTileRows` fused rows laid out contiguously, so a
    // stage is `BLOCK_N / kBTileRows` bulk copies of adjacent tiles.
    constexpr uint32_t kNumTileLoads = LOAD_BLOCK_N / kBTileRows;
    constexpr uint32_t kNumNTiles = shape_n / kBTileRows;

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      const uint32_t n_tile_idx = local_expert_idx * kNumNTiles + n_block_idx * kNumTileLoads;
      if (cute::elect_one_sync()) {
#pragma unroll
        for (uint32_t i = 0; i < kNumTileLoads; ++i) {
          const uint64_t tile_idx =
              static_cast<uint64_t>(n_tile_idx + i) * num_k_blocks + k_block_idx;
          // EVICT_NORMAL, not tma_load_1d's streaming default:
          // above the swapAB tiers an expert spans several
          // m-blocks, and those CTAs share these tiles via L2.
          ptx::tma_load_1d(smem_packed_b[stage_idx] + i * kBTileBytes,
                           b_base + tile_idx * kBTileBytes, full_barriers[stage_idx], kBTileBytes,
                           cute::TMA::CacheHintSm90::EVICT_NORMAL);
        }
        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_PACKED_B_SIZE_PER_STAGE);
      }
      __syncwarp();
    }
  };
  if constexpr (kUseInterleavedScheduler)
    produce_interleaved_blocks(load_b_task);
  else
    for_each_static_selected_block(load_b_task);

} else if (kSplitDecodeWarps and warp_idx < kFirstMathWarp) {
  // =====================================================================
  // ROLE 3a: DECODE WARPS (FP4 -> FP8 into the decoded ring)
  // =====================================================================
  cutlass::arch::warpgroup_reg_dealloc<kNumDecodeRegisters>();
  const uint32_t decode_thread_idx = (warp_idx - kFirstDecodeWarp) * 32 + lane_idx;

  const auto run_decode_task = [&](const auto& block_phase, const uint32_t& local_expert_idx,
                                   const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                                   const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
                                   const uint32_t& valid_m) {
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      {
        const unsigned long long w0 = stamp_now();
        full_barriers[stage_idx]->wait(phase);
        if (decode_thread_idx == 0) stamp_add(24, stamp_now() - w0);
      }
      if constexpr (kUseInterleavedScheduler) {
        if (k_block_idx == 0) interleaved_scheduler.release_task_info(lane_idx);
      }
      decoded_empty_barriers[b_ring_idx]->wait(b_ring_phase ^ 1);
      flashinfer::megamoe::mxfp4::dequant_smem_b_from_packed_mode2_nibble<kQuadDequantIlp>(
          reinterpret_cast<uint8_t*>(smem_b[b_ring_idx]),
          reinterpret_cast<const uint8_t*>(smem_packed_b[stage_idx]), decode_thread_idx,
          smem_mxfp4_lut);
      cutlass::arch::fence_view_async_shared();
      if (lane_idx == 0) {
        // Publishing does not block, so this warp runs on into the
        // other slot while the math warps consume this one.
        decoded_full_barriers[b_ring_idx]->arrive();
        // The packed stage is ours to release once it is decoded.
        empty_barriers[stage_idx]->arrive();
      }
    }
  };
  if constexpr (kUseInterleavedScheduler)
    for_each_published_block(run_decode_task);
  else
    for_each_static_selected_block(run_decode_task);

} else {
  // =====================================================================
  // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
  // =====================================================================
  cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

  const uint32_t epilogue_warp_idx = warp_idx - kFirstMathWarp;
  const uint32_t epilogue_wg_idx = epilogue_warp_idx / 4;
  const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
  const uint32_t warp_idx_in_wg = epilogue_warp_idx % 4;

  const auto arrive_empty_barrier = [&](const uint32_t& s) {
    if (lane_idx == 0) {
      // Hand the decoded slot back, then release the stage itself:
      // its activations were read by these warps.
      if constexpr (kSplitDecodeWarps) decoded_empty_barriers[b_ring_idx]->arrive();
      empty_barriers[s]->arrive();
    }
  };

  const auto notify_l1_ready = [&](const uint32_t& ready_pool_block_idx,
                                   const uint32_t& ready_n_block_idx) {
    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
      ptx::red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(ready_pool_block_idx),
                          1ull << ready_n_block_idx);
    }
    __syncwarp();
  };

  // WGMMA-output register layout helpers
  const uint32_t row_idx = lane_idx / 4;
  const uint32_t col_idx = lane_idx % 4;
  const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
  const uint32_t r_1 = r_0 + 8;

  FI_MEGAMOE_STATIC_ASSERT(
      kSwapABRequested || (WG_BLOCK_M == L1WGMMA::M and WG_BLOCK_N == L1WGMMA::N),
      "Split-N WGs must each run one M64N128 WGMMA per K-block");

  // Sync with dispatch
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  const auto run_math_task = [&](const auto& block_phase, const uint32_t& local_expert_idx,
                                 const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                                 const uint32_t& n_block_idx, const uint32_t& pool_block_idx,
                                 const uint32_t& valid_m) {
    const unsigned long long task_t0 = stamp_now();
    if (epilogue_thread_idx == 0) stamp_min(3);
    const uint32_t m_idx = pool_block_idx * BLOCK_M;
    const uint32_t wg_n_idx = kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_BLOCK_N;
    const uint32_t wg_l1_out_n_idx =
        kSplitMDecodedWeightReuse ? 0u : epilogue_wg_idx * WG_L1_OUT_BLOCK_N;
    const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
    const uint32_t row_block_offset = kSplitMDecodedWeightReuse ? epilogue_wg_idx * WG_BLOCK_M : 0u;
    const uint32_t row_offset_r0 = row_block_offset + r_0;
    const uint32_t row_offset_r1 = row_block_offset + r_1;
    using BlockPhaseTag = std::remove_cv_t<std::remove_reference_t<decltype(block_phase)>>;
    constexpr bool kBlockIsL2 = BlockPhaseTag::value == sched::BlockPhase::Linear2;
    const float l2_global_scale =
        l2_global_scales == nullptr ? 1.0f : __ldg(l2_global_scales + local_expert_idx);
    const auto cast_l2_scaled_bf16_pair = [&](float x, float y) -> uint32_t {
      x *= l2_global_scale;
      y *= l2_global_scale;
      return math::cast_into_bf16_and_pack(x, y);
    };

    // ---------------- GEMM ----------------
    using WGMMA = L1WGMMA;
    constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
    float final_accum[kAccumPerThread] = {};
    const auto decode_b_stage = [&](const uint32_t& packed_stage, const uint32_t& decoded_slot) {
      if constexpr (kSplitMDecodedWeightReuse) {
        // Both M64 consumers share the same decoded N128 tile.
        // The two physical decoded slots remove the overwrite
        // hazard; the trailing pair barrier publishes K+1 before
        // either consumer issues its WGMMA.
        FI_MEGAMOE_STATIC_ASSERT(kUseMode2RowDecoder,
                                 "BM128 split-M uses the cooperative Mode2 decoder");
        flashinfer::megamoe::mxfp4::dequant_smem_b_from_packed_mode2_nibble_split_m(
            reinterpret_cast<uint8_t*>(smem_b[decoded_slot]),
            reinterpret_cast<const uint8_t*>(smem_packed_b[packed_stage]), epilogue_thread_idx,
            smem_mxfp4_lut);
        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.sync %0, %1;" : : "n"(kSplitMDecodeBarrierIdx), "n"(256) : "memory");
      } else {
        flashinfer::megamoe::mxfp4::dequant_smem_b_from_packed_mode2_nibble<kQuadDequantIlp>(
            reinterpret_cast<uint8_t*>(smem_b[decoded_slot]),
            reinterpret_cast<const uint8_t*>(smem_packed_b[packed_stage]), epilogue_thread_idx,
            smem_mxfp4_lut);
        cutlass::arch::fence_view_async_shared();
        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
      }
    };

    // Register-source decode: one lane pair owns one weight row of the
    // half-tile, and each slice of four A registers is produced from two
    // packed words plus one shuffle. Nothing lands in shared memory, so
    // this path allocates no decoded-B tile at all.
    const auto decode_rs_half = [&](const uint32_t& rs_stage, const uint32_t& half,
                                    uint32_t(&a_frag)[4][4]) {
      const uint32_t frag_row0 = wg_n_idx + half * 64u + warp_idx_in_wg * 16u + row_idx;
      const uint32_t decode_row = frag_row0 + ((lane_idx & 1u) << 3);
      const uint32_t word_sel = (lane_idx >> 1) & 1u;
      const bool keep_hi = (lane_idx & 1u) == 0;
      const auto src = mxfp4::decode_tile_row(
          reinterpret_cast<const uint8_t*>(smem_packed_b[rs_stage]), decode_row);
      const uint32_t scale_word = ptx::ld_shared(reinterpret_cast<const uint32_t*>(src.scale));

#pragma unroll
      for (uint32_t slice = 0; slice < 4; ++slice) {
        const auto* __restrict__ chunk =
            src.chunk + slice * mxfp4::kBChunkStride + word_sel * sizeof(uint32_t);
        const uint32_t w_lo = ptx::ld_shared(reinterpret_cast<const uint32_t*>(chunk));
        const uint32_t w_hi = ptx::ld_shared(reinterpret_cast<const uint32_t*>(chunk + 8u));
        const auto lut =
            mxfp4::load_scaled_lut(smem_mxfp4_lut, (scale_word >> (slice * 8u)) & 0xffu);
        mxfp4::dequant_rs_word_pair(w_lo, w_hi, lut, keep_hi, a_frag[slice]);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) mma::sm90::warpgroup_fence_operand(a_frag[slice][i]);
      }
    };
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      // BM128 split-M alternates two decoded slots keyed by the
      // pipeline stage, because its paired warpgroups share the same
      // N range and one must not overwrite a tile the other is still
      // reading. Every other plan decodes and consumes inside one
      // iteration and has a single slot.
      const uint32_t b_slot =
          kSplitDecodeWarps ? b_ring_idx : (kSplitMDecodedWeightReuse ? stage_idx : 0u);
      {
        const unsigned long long w0 = stamp_now();
        if constexpr (kSplitDecodeWarps) {
          // The decode warps own the packed stage and its release,
          // so wait on the decoded tile rather than the TMA.
          decoded_full_barriers[b_ring_idx]->wait(b_ring_phase);
        } else {
          full_barriers[stage_idx]->wait(phase);
        }
        if (epilogue_thread_idx == 0) stamp_add(26, stamp_now() - w0);
      }
      if constexpr (kUseInterleavedScheduler) {
        if (k_block_idx == 0) interleaved_scheduler.release_task_info(lane_idx);
      }
      if constexpr (!kUseRSOperand and !kSplitDecodeWarps) decode_b_stage(stage_idx, b_slot);

      // Read SF (must precede warpgroup_arrive)
      const float scale_a_0_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
      const float scale_a_1_lo = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);
      float scale_a_0_hi = 0.0f;
      float scale_a_1_hi = 0.0f;
      if constexpr (kBlockIsL2 && kSplitMDecodedWeightReuse) {
        scale_a_0_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r0);
        scale_a_1_hi = ptx::ld_shared(smem_sfa[stage_idx] + kL2SFAHalfStride + row_offset_r1);
      }

      // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
      // expansion, so the WGMMA accumulator only needs activation SF.

      if constexpr (!kBlockIsL2) {
        if constexpr (kSwapABRequested) {
          auto run_swap_ab_l1 = [&]<uint32_t N_SWAP>() {
            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
            FI_MEGAMOE_STATIC_ASSERT(kSwapABWeightHalves == 2,
                                     "BN256 swapAB splits the weight tile into two halves");
            float swap_accum[2][kSwapAccum];

            // The two halves are independent WGMMA groups, so issue both and only
            // then wait. The window between the last issue and the first wait is
            // where the next K-block's dequant hides.
            using RSWGMMA = typename mma::sm90::FP8RSMMASelector<N_SWAP>::type;
            // One fragment set per half. Under RS the decode sits
            // between the stage barrier and the WGMMA, so giving
            // each half its own registers lets half 1's decode run
            // under half 0's MMA. Unlike prefetching the next
            // K-block, this stays inside one stage and so never
            // puts a TMA wait in front of arrive_empty.
            uint32_t a_frags[kUseRSOperand ? 2 : 1][4][4];
            const auto issue_half = [&](const uint32_t half) {
              auto& a_frag = a_frags[kUseRSOperand ? half : 0];
              if constexpr (kUseRSOperand) decode_rs_half(stage_idx, half, a_frag);
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum; ++i)
                ptx::warpgroup_fence_operand(swap_accum[half][i]);
              ptx::warpgroup_arrive();
#pragma unroll
              for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++k) {
                if constexpr (kUseRSOperand) {
                  auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + k * RSWGMMA::K, 1);
                  RSWGMMA::wgmma(a_frag[k], desc_b, swap_accum[half], k);
                } else {
                  auto desc_a = mma::sm90::make_smem_desc(
                      smem_b[b_slot] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                  auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                  SwapWGMMA::wgmma(desc_a, desc_b, swap_accum[half], k);
                }
              }
              ptx::warpgroup_commit_batch();
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum; ++i)
                ptx::warpgroup_fence_operand(swap_accum[half][i]);
            };
            // The next issue_half overwrites a_frag, so the
            // fragments have to be re-fenced once the WGMMA that
            // reads them has drained.
            const auto fence_frags = [&](const uint32_t half) {
              if constexpr (kUseRSOperand) {
                auto& a_frag = a_frags[half];
#pragma unroll
                for (uint32_t slice = 0; slice < 4; ++slice)
#pragma unroll
                  for (uint32_t i = 0; i < 4; ++i)
                    mma::sm90::warpgroup_fence_operand(a_frag[slice][i]);
              }
            };
            const auto accumulate_half = [&](const uint32_t half) {
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum / 4; ++i) {
                const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                const uint32_t token_0 = i * 8 + col_idx * 2;
                const uint32_t token_1 = token_0 + 1;
                if (token_0 < valid_m) {
                  const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                  final_accum[accum_offset + 0] += scale_0 * swap_accum[half][i * 4 + 0];
                  final_accum[accum_offset + 2] += scale_0 * swap_accum[half][i * 4 + 2];
                }
                if (token_1 < valid_m) {
                  const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                  final_accum[accum_offset + 1] += scale_1 * swap_accum[half][i * 4 + 1];
                  final_accum[accum_offset + 3] += scale_1 * swap_accum[half][i * 4 + 3];
                }
              }
            };

            if constexpr (kUseRSOperand) {
              issue_half(0);
              issue_half(1);  // decodes under half 0's MMA
              ptx::warpgroup_wait<1>();
              fence_frags(0);
              accumulate_half(0);
              ptx::warpgroup_wait<0>();
              fence_frags(1);
              accumulate_half(1);
            } else {
              issue_half(0);
              ptx::warpgroup_wait<0>();
              accumulate_half(0);
              issue_half(1);
              ptx::warpgroup_wait<0>();
              accumulate_half(1);
            }

            arrive_empty_barrier(stage_idx);
          };

          if constexpr (BLOCK_M == 8) {
            run_swap_ab_l1.template operator()<8>();
          } else if constexpr (BLOCK_M == 16) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l1.template operator()<8>();
            } else {
              run_swap_ab_l1.template operator()<16>();
            }
          } else if constexpr (BLOCK_M == 24) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l1.template operator()<8>();
            } else if (n_swap <= 16) {
              run_swap_ab_l1.template operator()<16>();
            } else {
              run_swap_ab_l1.template operator()<24>();
            }
          } else if constexpr (BLOCK_M == 32) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l1.template operator()<8>();
            } else if (n_swap <= 16) {
              run_swap_ab_l1.template operator()<16>();
            } else if (n_swap <= 24) {
              run_swap_ab_l1.template operator()<24>();
            } else {
              run_swap_ab_l1.template operator()<32>();
            }
          } else {
            // A tile with no arm here never runs the mainloop,
            // so nothing releases the stage and the loader
            // hangs. Fail at compile time instead.
            FI_MEGAMOE_STATIC_ASSERT(
                BLOCK_M == 8 || BLOCK_M == 16 || BLOCK_M == 24 || BLOCK_M == 32,
                "swapAB tile has no N_SWAP arm");
          }
        } else {
          float accum[kAccumPerThread];
// Single per-128 K-block WGMMA group
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
          ptx::warpgroup_arrive();
#pragma unroll
          for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            auto desc_a = mma::sm90::make_smem_desc(
                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
            auto desc_b =
                mma::sm90::make_smem_desc(smem_b[b_slot] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
            WGMMA::wgmma(desc_a, desc_b, accum, k);
          }
          ptx::warpgroup_commit_batch();
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
          ptx::warpgroup_wait<0>();

          arrive_empty_barrier(stage_idx);

// L1: gate/up alternate at gran=8 along N; each `i` block
// of 8 cols belongs entirely to one of {gate, up}, so .x
// and .y share the same scalar.
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
            final_accum[i * 4 + 0] += scale_a_0_lo * accum[i * 4 + 0];
            final_accum[i * 4 + 1] += scale_a_0_lo * accum[i * 4 + 1];
            final_accum[i * 4 + 2] += scale_a_1_lo * accum[i * 4 + 2];
            final_accum[i * 4 + 3] += scale_a_1_lo * accum[i * 4 + 3];
          }
        }
      } else {
        if constexpr (kSwapABRequested) {
          FI_MEGAMOE_STATIC_ASSERT(kL2ActsSFGranK == 128,
                                   "L2 swap-AB requires per-128 activation scales");
          auto run_swap_ab_l2 = [&]<uint32_t N_SWAP>() {
            using SwapWGMMA = typename mma::sm90::FP8MMASelector<N_SWAP>::type;
            constexpr uint32_t kSwapAccum = SwapWGMMA::kNumAccum;
            FI_MEGAMOE_STATIC_ASSERT(kSwapABWeightHalves == 2,
                                     "BN256 swapAB splits the weight tile into two halves");
            float swap_accum[2][kSwapAccum];

            // The two halves are independent WGMMA groups, so issue both and only
            // then wait. The window between the last issue and the first wait is
            // where the next K-block's dequant hides.
            using RSWGMMA = typename mma::sm90::FP8RSMMASelector<N_SWAP>::type;
            // One fragment set per half. Under RS the decode sits
            // between the stage barrier and the WGMMA, so giving
            // each half its own registers lets half 1's decode run
            // under half 0's MMA. Unlike prefetching the next
            // K-block, this stays inside one stage and so never
            // puts a TMA wait in front of arrive_empty.
            uint32_t a_frags[kUseRSOperand ? 2 : 1][4][4];
            const auto issue_half = [&](const uint32_t half) {
              auto& a_frag = a_frags[kUseRSOperand ? half : 0];
              if constexpr (kUseRSOperand) decode_rs_half(stage_idx, half, a_frag);
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum; ++i)
                ptx::warpgroup_fence_operand(swap_accum[half][i]);
              ptx::warpgroup_arrive();
#pragma unroll
              for (uint32_t k = 0; k < BLOCK_K / SwapWGMMA::K; ++k) {
                if constexpr (kUseRSOperand) {
                  auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + k * RSWGMMA::K, 1);
                  RSWGMMA::wgmma(a_frag[k], desc_b, swap_accum[half], k);
                } else {
                  auto desc_a = mma::sm90::make_smem_desc(
                      smem_b[b_slot] + (wg_n_idx + half * 64u) * BLOCK_K + k * SwapWGMMA::K, 1);
                  auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + k * SwapWGMMA::K, 1);
                  SwapWGMMA::wgmma(desc_a, desc_b, swap_accum[half], k);
                }
              }
              ptx::warpgroup_commit_batch();
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum; ++i)
                ptx::warpgroup_fence_operand(swap_accum[half][i]);
            };
            // The next issue_half overwrites a_frag, so the
            // fragments have to be re-fenced once the WGMMA that
            // reads them has drained.
            const auto fence_frags = [&](const uint32_t half) {
              if constexpr (kUseRSOperand) {
                auto& a_frag = a_frags[half];
#pragma unroll
                for (uint32_t slice = 0; slice < 4; ++slice)
#pragma unroll
                  for (uint32_t i = 0; i < 4; ++i)
                    mma::sm90::warpgroup_fence_operand(a_frag[slice][i]);
              }
            };
            const auto accumulate_half = [&](const uint32_t half) {
#pragma unroll
              for (uint32_t i = 0; i < kSwapAccum / 4; ++i) {
                const uint32_t accum_offset = half * kSwapABHalfAccumPerThread + i * 4;
                const uint32_t token_0 = i * 8 + col_idx * 2;
                const uint32_t token_1 = token_0 + 1;
                if (token_0 < valid_m) {
                  const float scale_0 = ptx::ld_shared(smem_sfa[stage_idx] + token_0);
                  final_accum[accum_offset + 0] += scale_0 * swap_accum[half][i * 4 + 0];
                  final_accum[accum_offset + 2] += scale_0 * swap_accum[half][i * 4 + 2];
                }
                if (token_1 < valid_m) {
                  const float scale_1 = ptx::ld_shared(smem_sfa[stage_idx] + token_1);
                  final_accum[accum_offset + 1] += scale_1 * swap_accum[half][i * 4 + 1];
                  final_accum[accum_offset + 3] += scale_1 * swap_accum[half][i * 4 + 3];
                }
              }
            };

            if constexpr (kUseRSOperand) {
              issue_half(0);
              issue_half(1);  // decodes under half 0's MMA
              ptx::warpgroup_wait<1>();
              fence_frags(0);
              accumulate_half(0);
              ptx::warpgroup_wait<0>();
              fence_frags(1);
              accumulate_half(1);
            } else {
              issue_half(0);
              ptx::warpgroup_wait<0>();
              accumulate_half(0);
              issue_half(1);
              ptx::warpgroup_wait<0>();
              accumulate_half(1);
            }

            arrive_empty_barrier(stage_idx);
          };

          if constexpr (BLOCK_M == 8) {
            run_swap_ab_l2.template operator()<8>();
          } else if constexpr (BLOCK_M == 16) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l2.template operator()<8>();
            } else {
              run_swap_ab_l2.template operator()<16>();
            }
          } else if constexpr (BLOCK_M == 24) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l2.template operator()<8>();
            } else if (n_swap <= 16) {
              run_swap_ab_l2.template operator()<16>();
            } else {
              run_swap_ab_l2.template operator()<24>();
            }
          } else if constexpr (BLOCK_M == 32) {
            const uint32_t n_swap = ((valid_m + 7u) / 8u) * 8u;
            if (n_swap <= 8) {
              run_swap_ab_l2.template operator()<8>();
            } else if (n_swap <= 16) {
              run_swap_ab_l2.template operator()<16>();
            } else if (n_swap <= 24) {
              run_swap_ab_l2.template operator()<24>();
            } else {
              run_swap_ab_l2.template operator()<32>();
            }
          } else {
            // A tile with no arm here never runs the mainloop,
            // so nothing releases the stage and the loader
            // hangs. Fail at compile time instead.
            FI_MEGAMOE_STATIC_ASSERT(
                BLOCK_M == 8 || BLOCK_M == 16 || BLOCK_M == 24 || BLOCK_M == 32,
                "swapAB tile has no N_SWAP arm");
          }
        } else {
          float accum[kAccumPerThread];
          const auto promote_l2_accum = [&](const float& scale_r0, const float& scale_r1) {
#pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
              final_accum[i * 4 + 0] += scale_r0 * accum[i * 4 + 0];
              final_accum[i * 4 + 1] += scale_r0 * accum[i * 4 + 1];
              final_accum[i * 4 + 2] += scale_r1 * accum[i * 4 + 2];
              final_accum[i * 4 + 3] += scale_r1 * accum[i * 4 + 3];
            }
          };
          if constexpr (kSplitMDecodedWeightReuse) {
#pragma unroll
            for (uint32_t sf_group = 0; sf_group < 2; ++sf_group) {
#pragma unroll
              for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
              ptx::warpgroup_arrive();
#pragma unroll
              for (uint32_t k = 0; k < (BLOCK_K / 2) / WGMMA::K; ++k) {
                const uint32_t k_off = sf_group * (BLOCK_K / 2) + k * WGMMA::K;
                auto desc_a = mma::sm90::make_smem_desc(
                    smem_a[stage_idx] + row_block_offset * BLOCK_K + k_off, 1);
                auto desc_b =
                    mma::sm90::make_smem_desc(smem_b[b_slot] + wg_n_idx * BLOCK_K + k_off, 1);
                WGMMA::wgmma(desc_a, desc_b, accum, k);
              }
              ptx::warpgroup_commit_batch();
#pragma unroll
              for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
              ptx::warpgroup_wait<0>();
              if (sf_group == 0)
                promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
              else
                promote_l2_accum(scale_a_0_hi, scale_a_1_hi);
            }
            arrive_empty_barrier(stage_idx);
          } else {
// One per-128 scale permits a single four-instruction
// WGMMA group and one accumulator promotion per K tile.
#pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
#pragma unroll
            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
              auto desc_a = mma::sm90::make_smem_desc(
                  smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
              auto desc_b =
                  mma::sm90::make_smem_desc(smem_b[b_slot] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
              WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            ptx::warpgroup_commit_batch();
#pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_wait<0>();
            // All four warps must finish reading decoded B before the next K tile overwrites it.
            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
            arrive_empty_barrier(stage_idx);
            promote_l2_accum(scale_a_0_lo, scale_a_1_lo);
          }
        }
      }
    }

    // Skip epilogue when block is past valid M (the GEMM loop already
    // released its pipeline stages). Drain any prior L1 async store.
    if (valid_m == 0) {
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
      return;
    }

#include <flashinfer/megamoe/detail/fused_epilogue.inl>

    if (epilogue_thread_idx == 0) {
      stamp_add(kBlockIsL2 ? 22u : 20u, stamp_now() - task_t0);
      stamp_max(kBlockIsL2 ? 5u : 4u);
    }
  };
  if constexpr (kUseInterleavedScheduler)
    for_each_published_block(run_math_task);
  else
    for_each_static_selected_block(run_math_task);

  // ---------------- COMBINE ----------------
  // NVLink barrier first: signals remote ranks that this rank's GEMM
  // outputs (NVLink scatter targets) are fully written.
  comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads, kEpilogueGridSyncIndex,
                       kBeforeCombineReduceBarrierTag>(
      workspace, sym_buffer, sm_idx, epilogue_thread_idx,
      [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
  if (epilogue_thread_idx == 0) stamp_max(6);

  // Sync with dispatch (paired with dispatch's pre-cleanup sync) so that
  // dispatch may now safely clean workspace state.
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

// These are function fragments: sizing declarations must precede the combine body.
// clang-format off
#include <flashinfer/megamoe/detail/combine_config.inl>
#include <flashinfer/megamoe/detail/combine.inl>
  // clang-format on
}
#else
if (blockIdx.x == 0 and threadIdx.x == 0)
  FI_MEGAMOE_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
