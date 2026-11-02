// Independent SM90 NVFP4 MegaMoE l1 kernel body.
// RS-swapAB compile-time flag, emitted into the generated TU by the host
// codegen (csrc/jit_kernels/impls/sm90_nvfp4_mega_moe.hpp) from the
// FI_MEGAMOE_NVFP4_L1_RS_SWAPAB env var. Default off: the SS path below must stay
// byte-identical in behaviour when the flag is unset.
#ifndef FI_MEGAMOE_NVFP4_L1_RS_SWAPAB
#define FI_MEGAMOE_NVFP4_L1_RS_SWAPAB 0
#endif
#ifndef FI_MEGAMOE_NVFP4_L1_SINGLE_ACCUM
#define FI_MEGAMOE_NVFP4_L1_SINGLE_ACCUM 0
#endif
#ifndef FI_MEGAMOE_NVFP4_LUT_COMPACT
#define FI_MEGAMOE_NVFP4_LUT_COMPACT 0
#endif
#ifndef FI_MEGAMOE_NVFP4_L1_SFA_LOOKAHEAD
#define FI_MEGAMOE_NVFP4_L1_SFA_LOOKAHEAD 0
#endif
#ifndef FI_MEGAMOE_NVFP4_L1_DEQUANT_HALF_STREAM
#define FI_MEGAMOE_NVFP4_L1_DEQUANT_HALF_STREAM 0
#endif
#ifndef FI_MEGAMOE_NVFP4_L1_DEQUANT_WARP_REMAP
#define FI_MEGAMOE_NVFP4_L1_DEQUANT_WARP_REMAP 0
#endif
// Debug arm of the RS decoder: decode-and-discard (no shfl pairing) with
// fully serialized WGMMA groups, for bisecting shfl/register-lifetime issues.
#ifndef FI_MEGAMOE_NVFP4_L1_RS_DEBUG_DECODE
#define FI_MEGAMOE_NVFP4_L1_RS_DEBUG_DECODE 0
#endif
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or \
    defined(__CLION_IDE__)
using Barrier = cutlass::arch::ClusterTransactionBarrier;

// =====================================================================
// Template checks
// =====================================================================
FI_MEGAMOE_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
FI_MEGAMOE_STATIC_ASSERT(kNumSMs % 2 == 0, "SM count must be divisible by the L1 cluster size");

// =====================================================================
// Thread / warp identification
// =====================================================================
const uint32_t sm_idx = blockIdx.x;
const uint32_t thread_idx = threadIdx.x;
const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
const uint32_t lane_idx = ptx::get_lane_idx();

if (warp_idx == 0 and cute::elect_one_sync()) {
  cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
  cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
  cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
  cute::prefetch_tma_descriptor(&tensor_map_l1_output);
}

// =====================================================================
// Workspaces and symmetric buffer slicing. The L2 activation SF allocation
// intentionally retains its old per-64 physical capacity in stage 1, while
// the active logical layout densely addresses only per-128 groups.
// =====================================================================
#include <flashinfer/megamoe/detail/buffers.inl>

// =====================================================================
// GEMM data types and shape constants
// =====================================================================
using a_dtype_t = cutlass::float_e4m3_t;
using b_dtype_t = cutlass::float_e4m3_t;
constexpr uint32_t WG_BLOCK_M = 64;
constexpr uint32_t WG_BLOCK_N = 128;
constexpr uint32_t L1_OUT_BLOCK_N = 64;
constexpr uint32_t WG_L1_OUT_BLOCK_N = 64;
constexpr bool kL2ArrivalCounter = kL2ArrivalCounterRequested;
// Use two active dispatch warps. The other two dispatch warps form the
// even-stage dequant team when dispatch-assisted dequant is on.
constexpr uint32_t kNumActiveDispatchWarps = 2;
constexpr uint32_t kNumActiveDispatchThreads = 64;
// RS-swapAB: swap the WGMMA A/B roles and decode the FP4 weights straight
// into A-register fragments (RS form). The FP8 SMEM expansion and both
// dequant teams compile out, removing the FP8 write-back and the WGMMA
// weight re-read from the SMEM pipe entirely.
constexpr bool kRSSwapAB = FI_MEGAMOE_NVFP4_L1_RS_SWAPAB != 0;
// Single-accumulator form. Microscaling forces a per-k-block flush -- the WGMMA
// output must be multiplied by the token's activation SF for THAT k range before it
// can join the running total -- which is why the default path carries two 64-register
// arrays. Keeping the running total in the WGMMA accumulator itself, expressed in
// units of the current block's SF and rescaled in place when the SF changes, needs
// only one. That frees 64 registers per math thread, which is the binding constraint
// on BLOCK_M * BLOCK_N (accum bytes = 2 * BLOCK_M * BLOCK_N / 256 threads).
constexpr bool kSingleAccum = FI_MEGAMOE_NVFP4_L1_SINGLE_ACCUM != 0 and not kRSSwapAB;
constexpr bool kLutCompact = FI_MEGAMOE_NVFP4_LUT_COMPACT != 0;
constexpr uint32_t kSFALookahead = FI_MEGAMOE_NVFP4_L1_SFA_LOOKAHEAD;
FI_MEGAMOE_STATIC_ASSERT(kSFALookahead <= 2,
                         "SFA lookahead modes: 0=OFF, 1=EMPTY full-wait, 2=REAL prefetch");
// 0: original whole-stage mbarrier handoff. 1: whole-stage decode followed
// by the same two named ready barriers as the real arm (EMPTY/control).
// 2: publish K64 halves so the first two K32 WGMMAs overlap the second half.
// 3: producer is identical to mode 2, but math waits for both halves before
// issuing any WGMMA. This isolates split/fence cost from SMEM contention.
// 4: keep the mode-2 overlap, but rendezvous only the 64 dequant writers;
// one writer releases each half through a stage-local mbarrier, so math
// waits without forcing the producer to rendezvous with 256 consumers.
// 5: replace each mode-4 producer CTA barrier with per-warp rendezvous;
// both producer warp leaders release the expected-count-2 mbarrier.
// 6: retain mode-4 publication, but move the first four second-half LUT
// loads after the K0:64 release and remove its identity-MOV dependency gate.
constexpr uint32_t kDequantHalfStream = FI_MEGAMOE_NVFP4_L1_DEQUANT_HALF_STREAM;
FI_MEGAMOE_STATIC_ASSERT(kDequantHalfStream <= 6,
                         "dequant half stream modes: 0=OFF, 1=EMPTY, 2=NAMED, 3=NO_OVERLAP, "
                         "4=MBARRIER, 5=WARP_LEADER, 6=POST_PUBLISH_LUT");
FI_MEGAMOE_STATIC_ASSERT(not(kDequantHalfStream != 0 and kSFALookahead != 0),
                         "half stream owns the full/dequant handoff ordering");
FI_MEGAMOE_STATIC_ASSERT(not(kDequantHalfStream != 0 and kRSSwapAB),
                         "RS-swapAB has no shared-memory dequant producer");
// The RS arm gathers full-width uint2 entries straight off smem_nvfp4_lut and
// never went through the shared loader; the two layouts cannot coexist.
FI_MEGAMOE_STATIC_ASSERT(not(kRSSwapAB and kLutCompact),
                         "RS-swapAB reads the full-width LUT layout");
constexpr bool kDispatchDequant = kDispatchDequantRequested and not kRSSwapAB;
constexpr bool kDequantWarpRemap = FI_MEGAMOE_NVFP4_L1_DEQUANT_WARP_REMAP != 0;
FI_MEGAMOE_STATIC_ASSERT(not kDequantWarpRemap or (kDispatchDequant and kDequantHalfStream == 4),
                         "dequant warp remap is calibrated only for dispatch mode4");
using L1WGMMA = typename mma::sm90::FP8MMASelector<128>::type;
// Same M64N128K32 shape, A sourced from per-thread fragments instead of
// an SMEM descriptor. Under swapAB each math WG's "M" is its 64 weight
// rows and "N" is the 128 activation tokens.
using L1WGMMARS = typename mma::sm90::FP8RSMMASelector<128>::type;
static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
              "Unexpected WGMMA shape");

// A is always CTA-local.  When kClusterSize=2 the scheduler pairs adjacent
// M blocks with identical expert/N/K coordinates so the B TMA can multicast.
constexpr uint32_t LOAD_BLOCK_M = 128;
constexpr uint32_t LOAD_BLOCK_N = 128;
constexpr uint32_t kSwizzleAMode = 128;
constexpr uint32_t kLocalL1ActsSFGranK = 64;  // each CTA's local half
constexpr uint32_t kL2ActsSFGranK = 128;      // final L1 output / L2 input
FI_MEGAMOE_STATIC_ASSERT(L1_SHAPE_N / BLOCK_N <= 64,
                         "paired-N readiness must fit one 64-bit pool-block mask");
FI_MEGAMOE_STATIC_ASSERT((L1_SHAPE_N / BLOCK_N) % 2 == 0,
                         "every L1 N block must have a cluster peer");
FI_MEGAMOE_STATIC_ASSERT(kIntermediateHidden / kL2ActsSFGranK <= kIntermediateHidden / 64,
                         "logical per-128 SF groups must fit retained physical capacity");

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
constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
    math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
constexpr uint32_t SMEM_B_LOAD_SIZE_PER_STAGE = LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW;
// Under RS-swapAB nothing ever expands the packed rows to FP8 in SMEM, so
// each B stage shrinks to the 80-byte packed rows (10240 B, still a
// multiple of the 1024 B shared-memory alignment, so the SFA/barrier
// bases keep their alignment). This is safe for TMA: the B tensormap box
// is 80 B x 128 rows with no swizzle, written contiguously at the stage
// base; the 80-byte row stride lives in the tensormap, not in the stage
// stride. NOTE: the host smem-size heuristics are deliberately unchanged;
// the RS device layout uses less shared memory than the host allocates,
// which always fits and keeps num_stages/config selection identical
// between the two arms.
constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
    kRSSwapAB ? SMEM_B_LOAD_SIZE_PER_STAGE : LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
// Keep a two-slot SFA stage allocation so the six-stage ring retains its
// proven shared-memory/barrier placement. L1 reads only the first per-128
// slot.
constexpr uint32_t kL2SFAHalfStride =
    math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = 2 * kL2SFAHalfStride * sizeof(float);
// CD output: max of L1 FP8 (BLOCK_M * (BLOCK_N/2) * 1 byte * num_wg) and
// L2 BF16 (BLOCK_M * BLOCK_N * 2 bytes * num_wg).
constexpr uint32_t SMEM_CD_L1_SIZE =
    kNumEpilogueWarpgroups * WG_BLOCK_M * WG_L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t);
// RS-swapAB epilogue scratch: per-token amax slots (BLOCK_M tokens x
// kNumEpilogueWarps floats), appended to the CD tile. The DSM peer amax
// slots keep aliasing the CD tile itself, exactly as in the default path.
constexpr uint32_t SMEM_CD_RS_AMAX_SIZE =
    kRSSwapAB ? BLOCK_M * kNumEpilogueWarps * static_cast<uint32_t>(sizeof(float)) : 0u;
constexpr uint32_t SMEM_CD_OUTPUT_SIZE =
    math::constexpr_align(SMEM_CD_L1_SIZE + SMEM_CD_RS_AMAX_SIZE, kSharedMemoryAlignment);
constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_OUTPUT_SIZE;

// SMEM pointers
auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
const auto smem_send_buffers =
    layout::Buffer(fp8_token_layout, kNumActiveDispatchWarps, 1,
                   math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));
auto smem_nvfp4_lut = reinterpret_cast<uint2*>(
    math::advance_ptr<uint8_t>(smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE));

auto smem_gemm_base = math::advance_ptr(
    smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_NVFP4_LUT_SIZE);

auto smem_cd_base = smem_gemm_base;
auto smem_cd_l1 = reinterpret_cast<cutlass::float_e4m3_t*>(smem_cd_base);

auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
});
auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<b_dtype_t>(
      smem_gemm_base,
      SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
});
auto sf_start_ptr = math::advance_ptr<uint8_t>(
    smem_gemm_base, SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
  return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
});
constexpr uint32_t kNumDequantBarriers = kNumStages;
constexpr uint32_t kNumDequantHalf1Barriers = kDequantHalfStream >= 4 ? kNumStages : 0u;

// Barriers live after SF.
auto barrier_start_ptr =
    reinterpret_cast<Barrier*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
auto dispatch_barriers =
    utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
auto full_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + i; });
auto empty_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages + i; });
auto dequant_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + i; });
auto dequant_half1_barriers = utils::PatternVisitor([=](const uint32_t& i) {
  return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumDequantBarriers + i;
});
auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) {
  return barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumDequantBarriers +
         kNumDequantHalf1Barriers + i;
});

// =====================================================================
// Initialization
// =====================================================================
if constexpr (kLutCompact) {
  // Compact layout: lo[128] (each row's low word), then the eight
  // subnormal-scale rows' high words verbatim -- 544 B of the 1024-B region.
  // The high words of rows >= 8 are reconstructed inline by load_nvfp4_lut.
  if (thread_idx < 32) {
    const auto* lut_src = flashinfer::megamoe::nvfp4::kE2M1AndUe4m3ToFp8Lut + thread_idx * 4;
    reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
        make_uint4(lut_src[0].x, lut_src[1].x, lut_src[2].x, lut_src[3].x);
    if (thread_idx < 8)
      reinterpret_cast<uint32_t*>(smem_nvfp4_lut)[128 + thread_idx] =
          flashinfer::megamoe::nvfp4::kE2M1AndUe4m3ToFp8Lut[thread_idx].y;
  }
} else if (thread_idx < 64) {
  reinterpret_cast<uint4*>(smem_nvfp4_lut)[thread_idx] =
      reinterpret_cast<const uint4*>(flashinfer::megamoe::nvfp4::kE2M1AndUe4m3ToFp8Lut)[thread_idx];
}

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
      // With cluster multicast the leader CTA's TMA warp waits on peer
      // empty barriers too, so every math warp releases both CTAs.
      empty_barriers[i]->init(kClusterSize * kNumEpilogueWarpgroups);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
      dequant_barriers[i]->init(kDequantHalfStream == 5 ? 2 : 1);
    if constexpr (kDequantHalfStream >= 4) {
#pragma unroll
      for (uint32_t i = 0; i < kNumStages; ++i)
        dequant_half1_barriers[i]->init(kDequantHalfStream == 5 ? 2 : 1);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++i) combine_barriers[i]->init(1);
  }
  cutlass::arch::fence_barrier_init();
  if (cute::elect_one_sync()) {
    // Arm the first DSM amax exchange before cluster peers leave the
    // initialization rendezvous. Every tile completes one float
    // transaction for each of its BLOCK_M rows.
    combine_barriers[0]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
  }
}
cute::cluster_sync();

// =====================================================================
// Cluster-aware L1 scheduler
// =====================================================================
auto scheduler =
    sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K, L2_SHAPE_N,
                            L2_SHAPE_K, kNumExpertsPerRank, kNumExpertsPerWave, kNumSMs, kNumRanks,
                            kClusterSize>(workspace);

// Pipeline state shared by TMA loaders and math warpgroups
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
  ++k_block_idx;
  stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
  phase ^= stage_idx == 0;
};
// Separate named barriers let the two dequant teams publish K stages independently.
constexpr uint32_t kDequantBarrierIdx = 8;
constexpr uint32_t kAlternateDequantBarrierIdx = 9;
// With dispatch-assisted dequant, even and odd K stages have independent
// producer teams which may run concurrently. Give each team its own pair
// of K64 ready barriers. Indices 10..13 do not overlap dispatch/epilogue
// barriers (0..4) or the two producer-only in-place barriers (8..9).
constexpr uint32_t kDequantHalfReadyEven0 = 10;
constexpr uint32_t kDequantHalfReadyEven1 = 11;
constexpr uint32_t kDequantHalfReadyOdd0 = 12;
constexpr uint32_t kDequantHalfReadyOdd1 = 13;
constexpr uint32_t kDequantHalfReadyThreads = 64u + kNumEpilogueThreads;
const auto dequant_loaded_b_stage = [&](const uint32_t& s, const uint32_t& p,
                                        const uint32_t& k_block_idx,
                                        const uint32_t& non_epilogue_thread_idx) {
  // RS-swapAB: math lanes decode the packed FP4 straight into A-register
  // fragments, no SMEM FP8 expansion exists and no dequant team runs.
  // The freed warps deliberately stay idle: the post-change stage floor
  // still sits on the SMEM/MIO pipe, and any repurposing would add
  // traffic to exactly that pipe (issue-active is only ~10%).
  if constexpr (kRSSwapAB) return;
  if constexpr (kDispatchDequant) {
    if (non_epilogue_thread_idx >= 64u && (k_block_idx & 1u)) {
      full_barriers[s]->wait(p);
      const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
      if constexpr (kDequantHalfStream == 0) {
        flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
            64u, kAlternateDequantBarrierIdx, true>(reinterpret_cast<uint8_t*>(smem_b[s]),
                                                    dequant_tid, smem_nvfp4_lut);
        if (dequant_tid == 0) dequant_barriers[s]->arrive();
      } else if constexpr (kDequantHalfStream == 1) {
        flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
            64u, kAlternateDequantBarrierIdx>(reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid,
                                              smem_nvfp4_lut);
        cutlass::arch::fence_view_async_shared();
        ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyOdd0);
        ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyOdd1);
      } else if constexpr (kDequantHalfStream >= 4) {
        flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
            64u, kAlternateDequantBarrierIdx, false, 80u, false, true, kDequantHalfStream == 5,
            kDequantHalfStream == 6>(reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid,
                                     smem_nvfp4_lut, 0u, 0u, 0u, dequant_barriers[s],
                                     dequant_half1_barriers[s]);
      } else {
        flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
            64u, kAlternateDequantBarrierIdx, false, 80u, true>(
            reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut,
            kDequantHalfReadyOdd0, kDequantHalfReadyOdd1, kDequantHalfReadyThreads);
      }
    }
  } else if (non_epilogue_thread_idx >= 64u) {
    full_barriers[s]->wait(p);
    const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
    if constexpr (kDequantHalfStream == 0) {
      flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<64u,
                                                                             kDequantBarrierIdx>(
          reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
      cutlass::arch::fence_view_async_shared();
      ptx::sync_aligned(64, kDequantBarrierIdx);
      if (dequant_tid == 0) dequant_barriers[s]->arrive();
    } else if constexpr (kDequantHalfStream == 1) {
      flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<64u,
                                                                             kDequantBarrierIdx>(
          reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
      cutlass::arch::fence_view_async_shared();
      ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyEven0);
      ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyEven1);
    } else if constexpr (kDequantHalfStream >= 4) {
      flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
          64u, kDequantBarrierIdx, false, 80u, false, true, kDequantHalfStream == 5,
          kDequantHalfStream == 6>(reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid,
                                   smem_nvfp4_lut, 0u, 0u, 0u, dequant_barriers[s],
                                   dequant_half1_barriers[s]);
    } else {
      flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
          64u, kDequantBarrierIdx, false, 80u, true>(
          reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut,
          kDequantHalfReadyEven0, kDequantHalfReadyEven1, kDequantHalfReadyThreads);
    }
  }
};

// Intra-SM barrier indices (mirroring SM100)
constexpr uint32_t kDispatchBarrierIdx = 0;
constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
constexpr uint32_t kEpilogueFullBarrierIdx = 2;
constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

// Cross-rank NVLink barrier tags
constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;

// Register reconfiguration counts (chosen to fit in 64512 reg budget).
// Dispatch-assisted dequant keeps the same 63488-register CTA budget:
// 128*80 + 128*80 + 256*168.
constexpr uint32_t kNumDispatchRegisters = kDispatchDequant ? 80 : 48;
constexpr uint32_t kNumNonEpilogueRegisters = kDispatchDequant ? 80 : 64;
constexpr uint32_t kNumEpilogueRegisters = kDispatchDequant ? 168 : 192;
FI_MEGAMOE_STATIC_ASSERT(kNumDispatchRegisters* kNumDispatchThreads +
                                 kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                                 kNumEpilogueRegisters * kNumEpilogueThreads <=
                             64512,
                         "Too many registers");

constexpr uint32_t kDispatchGridSyncIndex = 0;

const auto for_each_selected_block = [&](auto&& func) {
  scheduler.for_each_linear1_block([&](const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                                       const uint32_t& n_block_idx) {
    func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear1>{}, local_expert_idx,
         num_k_blocks, m_block_idx, n_block_idx);
  });
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
      const uint32_t sf_pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
#pragma unroll
      for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFFloats, 32u); ++i) {
        const uint32_t j = i * 32 + lane_idx;
        if (j < kNumSFFloats)
          local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
      }
      __syncwarp();

      const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
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
                         1);
      }
      __syncwarp();
    }

  } else if constexpr (kDispatchDequant) {
    const uint32_t dequant_tid = (warp_idx - kNumActiveDispatchWarps) * 32 + lane_idx;
    for_each_selected_block([&](const auto&, const uint32_t&, const uint32_t& num_k_blocks,
                                const uint32_t&, const uint32_t&) {
      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        if ((k_block_idx & 1u) == 0) {
          full_barriers[stage_idx]->wait(phase);
          if constexpr (kDequantHalfStream == 0) {
            flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
                64u, kDequantBarrierIdx, true>(reinterpret_cast<uint8_t*>(smem_b[stage_idx]),
                                               dequant_tid, smem_nvfp4_lut);
            if (dequant_tid == 0) dequant_barriers[stage_idx]->arrive();
          } else if constexpr (kDequantHalfStream == 1) {
            flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
                64u, kDequantBarrierIdx>(reinterpret_cast<uint8_t*>(smem_b[stage_idx]), dequant_tid,
                                         smem_nvfp4_lut);
            cutlass::arch::fence_view_async_shared();
            ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyEven0);
            ptx::sync_unaligned(kDequantHalfReadyThreads, kDequantHalfReadyEven1);
          } else if constexpr (kDequantHalfStream >= 4) {
            flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
                64u, kDequantBarrierIdx, false, 80u, false, true, kDequantHalfStream == 5,
                kDequantHalfStream == 6>(reinterpret_cast<uint8_t*>(smem_b[stage_idx]), dequant_tid,
                                         smem_nvfp4_lut, 0u, 0u, 0u, dequant_barriers[stage_idx],
                                         dequant_half1_barriers[stage_idx]);
          } else {
            flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<
                64u, kDequantBarrierIdx, false, 80u, true>(
                reinterpret_cast<uint8_t*>(smem_b[stage_idx]), dequant_tid, smem_nvfp4_lut,
                kDequantHalfReadyEven0, kDequantHalfReadyEven1, kDequantHalfReadyThreads);
          }
        }
      }
    });
  }
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  // =====================================================================
  // ROLE 2: GEMM TMA LOAD warps (load A+SFA, B+SFB)
  //   Default logical mapping is loaders 0/1 and odd-K dequant 2/3.
  //   The scheduler-remap experiment swaps those physical warp pairs while
  //   preserving the same logical TMA/dequant work and register budgets.
  // =====================================================================
} else if (warp_idx == kNumDispatchWarps + (kDequantWarpRemap ? 2u : 0u)) {
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const auto tensor_map_a_ptr = [&]() { return &tensor_map_l1_acts; }();
    const auto tensor_map_sfa_ptr = [&]() { return &tensor_map_l1_acts_sf; }();

    const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
    const uint32_t valid_m = scheduler.template get_valid_m<false>();
    const bool has_valid_m = valid_m > 0;

    // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
    // the tail M unit when an expert has an odd number of M blocks.
    if (has_valid_m) {
      const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
      const auto expected = valid_m;
      while (ptx::ld_acq(ptr) != expected);
    }

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      if (cute::elect_one_sync()) {
        if (has_valid_m) {
          const uint32_t m_idx = pool_block_idx * BLOCK_M;
          const uint32_t k_idx = k_block_idx * BLOCK_K;

          // Paired N CTAs share the same M/K coordinates, so the
          // cluster leader multicasts A and SFA to both CTAs.
          tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
              tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx,
              kClusterSize);

          // TMA load SFA

          // L1 SFA per-128: load (BLOCK_M, 1) at K=k_block_idx
          tma::copy<BLOCK_M, 1, 0, float>(tensor_map_sfa_ptr, full_barriers[stage_idx],
                                          smem_sfa[stage_idx], m_idx, k_block_idx, kClusterSize);
          full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE +
                                                         BLOCK_M * sizeof(float));

        } else {
          full_barriers[stage_idx]->arrive();
        }
      }
      __syncwarp();
      dequant_loaded_b_stage(stage_idx, phase, k_block_idx, lane_idx);
    }
  });

} else if (warp_idx == kNumDispatchWarps + (kDequantWarpRemap ? 3u : 1u)) {
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const auto tensor_map_b_ptr = [&]() { return &tensor_map_l1_weights; }();
    constexpr uint32_t shape_n = L1_SHAPE_N;

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
      // NVFP4 fused B+scale layout stores 64B packed FP4 + 16B
      // UE4M3 scale per BK128 row.
      const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;
      if (cute::elect_one_sync()) {
        // The peer owns the adjacent N block, therefore B must be
        // loaded independently rather than multicast.
        tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_LOAD_SIZE_PER_STAGE);
      }
      __syncwarp();
      dequant_loaded_b_stage(stage_idx, phase, k_block_idx, 32u + lane_idx);
    }
  });

} else if (warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {
  // The two remaining non-epilogue warps form the odd-K dequant team.
  // Default: physical warps 6/7 (logical 2/3). Remap: physical warps
  // 4/5 retain logical dequant IDs 2/3 while the TMA loaders move to 6/7.
  // All four warps still participate in the warpgroup-collective
  // `setmaxnreg.dec.sync.aligned` required by the math warpgroups.
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  const uint32_t non_epilogue_warp_idx = warp_idx - kNumDispatchWarps;
  const uint32_t dequant_warp_idx = non_epilogue_warp_idx + (kDequantWarpRemap ? 2u : 0u);
  const uint32_t non_epilogue_thread_idx = dequant_warp_idx * 32 + lane_idx;
  for_each_selected_block([&](const auto& block_phase, const uint32_t&,
                              const uint32_t& num_k_blocks, const uint32_t&, const uint32_t&) {
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      dequant_loaded_b_stage(stage_idx, phase, k_block_idx, non_epilogue_thread_idx);
      __syncwarp();
    }
  });
} else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
  // =====================================================================
  // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
  // =====================================================================
  cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

  const uint32_t epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps);
  const uint32_t epilogue_wg_idx = epilogue_warp_idx / 4;
  const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
  const uint32_t warp_idx_in_wg = epilogue_warp_idx % 4;

  const auto arrive_empty_barrier = [&](const uint32_t& s) {
    if (warp_idx_in_wg != 0) return;
    if (lane_idx < 2) empty_barriers[s]->arrive(lane_idx);
  };

  // WGMMA-output register layout helpers
  const uint32_t row_idx = lane_idx / 4;
  const uint32_t col_idx = lane_idx % 4;
  const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
  const uint32_t r_1 = r_0 + 8;

  FI_MEGAMOE_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                           "Each warpgroup must run exactly one WGMMA per K-block");

  // Sync with dispatch
  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  uint32_t pair_scale_phase = 0;

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const uint32_t valid_m = scheduler.template get_valid_m<false>();
    const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
    const uint32_t m_idx = pool_block_idx * BLOCK_M;
    constexpr uint32_t wg_n_idx = 0;
    constexpr uint32_t wg_l1_out_n_idx = 0;
    const uint32_t n_idx = n_block_idx * BLOCK_N + wg_n_idx;
    const uint32_t row_block_offset = epilogue_wg_idx * WG_BLOCK_M;
    const uint32_t row_offset_r0 = row_block_offset + r_0;
    const uint32_t row_offset_r1 = row_block_offset + r_1;
    const bool valid_r0 = row_offset_r0 < valid_m;
    const bool valid_r1 = row_offset_r1 < valid_m;
    // Under RS-swapAB each warpgroup owns 64 weight rows that are
    // valid for every valid token, so a warpgroup may only drain when
    // the whole CTA is a token-less dummy cluster peer (valid_m == 0),
    // NOT when its per-WG row offset exceeds valid_m.
    const bool inactive_math_wg = kRSSwapAB ? (valid_m == 0) : (row_block_offset >= valid_m);

    if (inactive_math_wg) {
      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        // RS-swapAB gates directly on the producer barrier: the
        // dequant barriers stay allocated and initialized but are
        // never armed in that arm.
        if constexpr (kRSSwapAB)
          full_barriers[stage_idx]->wait(phase);
        else if constexpr (kDequantHalfStream == 0)
          dequant_barriers[stage_idx]->wait(phase);
        else {
          full_barriers[stage_idx]->wait(phase);
          if constexpr (kDequantHalfStream >= 4) {
            dequant_barriers[stage_idx]->wait(phase);
            dequant_half1_barriers[stage_idx]->wait(phase);
          } else {
            const bool use_odd_ready = kDispatchDequant and (k_block_idx & 1u);
            ptx::sync_unaligned(kDequantHalfReadyThreads,
                                use_odd_ready ? kDequantHalfReadyOdd0 : kDequantHalfReadyEven0);
            ptx::sync_unaligned(kDequantHalfReadyThreads,
                                use_odd_ready ? kDequantHalfReadyOdd1 : kDequantHalfReadyEven1);
          }
        }
        arrive_empty_barrier(stage_idx);
        __syncwarp();
      }
    }

    // ---------------- GEMM ----------------
    using WGMMA = L1WGMMA;
    constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
    float final_accum[kAccumPerThread] = {};
    // In single-accumulator mode `final_accum` IS the WGMMA accumulator and
    // `accum` is unused (and optimised away); the running total is held in units
    // of `unit_scale_*`, the most recent NON-ZERO activation SF for each of the
    // thread's two token rows.
    float accum[kSingleAccum ? 1 : kAccumPerThread];
    float unit_scale_0 = 0.0f, unit_scale_1 = 0.0f;

    const auto run_default_gemm_loop = [&]() {
      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        const bool use_odd_ready = kDispatchDequant and (k_block_idx & 1u);
        const uint32_t half_ready_barrier_0 =
            use_odd_ready ? kDequantHalfReadyOdd0 : kDequantHalfReadyEven0;
        const uint32_t half_ready_barrier_1 =
            use_odd_ready ? kDequantHalfReadyOdd1 : kDequantHalfReadyEven1;
        // The full barrier tracks TMA A + SFA + packed-B completion.
        // Its successful wait is an acquire, so SFA is safe to read.
        // The dequant barrier is a later acquire that publishes the
        // generic-proxy FP8 stores into smem_b. Modes 2 and 4 use
        // the gap between those dependency edges for useful work.
        // Mode 1 executes the same extra full wait but keeps the
        // original ordering, isolating wait/control perturbation.
        if constexpr (kDequantHalfStream != 0)
          full_barriers[stage_idx]->wait(phase);
        else if constexpr (kSFALookahead != 0)
          full_barriers[stage_idx]->wait(phase);
        if constexpr (kDequantHalfStream == 0 and kSFALookahead != 2)
          dequant_barriers[stage_idx]->wait(phase);

        // Read SF (must precede warpgroup_arrive). In mode 2 these
        // loads can issue while the producer warps dequantize B.
        const float scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
        const float scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

        if constexpr (kDequantHalfStream == 0 and kSFALookahead == 2) {
          // Do not issue WGMMA before this acquire: smem_b is
          // overwritten through the generic proxy by all 64
          // dequant threads, and their writer rendezvous precedes
          // the sole release arrival on dequant_barriers[s].
          dequant_barriers[stage_idx]->wait(phase);
        }

        // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8
        // smem expansion, so WGMMA only needs activation SF.

        if constexpr (kSingleAccum) {
          // Rescale the running total from the previous block's units into this
          // block's. A zero SF means the token's activations over this k range
          // are all zero, so the block contributes nothing and the units must
          // NOT change -- carrying the previous unit forward instead of dividing
          // is what keeps a zero SF from producing inf. When the unit is still
          // zero the running total is zero too, so a ratio of 1 is correct.
          const float u_0 = scale_a_0 != 0.0f ? scale_a_0 : unit_scale_0;
          const float u_1 = scale_a_1 != 0.0f ? scale_a_1 : unit_scale_1;
          if (k_block_idx > 0) {
            const float r_0 = u_0 != 0.0f ? unit_scale_0 / u_0 : 1.0f;
            const float r_1 = u_1 != 0.0f ? unit_scale_1 / u_1 : 1.0f;
#pragma unroll
            for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
              final_accum[i * 4 + 0] *= r_0;
              final_accum[i * 4 + 1] *= r_0;
              final_accum[i * 4 + 2] *= r_1;
              final_accum[i * 4 + 3] *= r_1;
            }
          }
          unit_scale_0 = u_0;
          unit_scale_1 = u_1;

#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i)
            ptx::warpgroup_fence_operand(final_accum[i]);
          ptx::warpgroup_arrive();
#pragma unroll
          for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            if constexpr (kDequantHalfStream >= 4) {
              if (k == 0)
                dequant_barriers[stage_idx]->wait(phase);
              else if (k == 2)
                dequant_half1_barriers[stage_idx]->wait(phase);
            } else if constexpr (kDequantHalfStream != 0) {
              if (k == 0) {
                ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_0);
                if constexpr (kDequantHalfStream == 3)
                  ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_1);
              } else if constexpr (kDequantHalfStream != 3) {
                if (k == 2) ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_1);
              }
            }
            auto desc_a = mma::sm90::make_smem_desc(
                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
            auto desc_b =
                mma::sm90::make_smem_desc(smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K, 1);
            // Accumulate in place across the whole K loop: only the very
            // first WGMMA of the very first block may overwrite.
            WGMMA::wgmma(desc_a, desc_b, final_accum, k > 0 or k_block_idx > 0);
          }
          ptx::warpgroup_commit_batch();
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i)
            ptx::warpgroup_fence_operand(final_accum[i]);
          ptx::warpgroup_wait<0>();

          arrive_empty_barrier(stage_idx);
        } else {
// Single per-128 K-block WGMMA group
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
          ptx::warpgroup_arrive();
#pragma unroll
          for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
            if constexpr (kDequantHalfStream >= 4) {
              if (k == 0)
                dequant_barriers[stage_idx]->wait(phase);
              else if (k == 2)
                dequant_half1_barriers[stage_idx]->wait(phase);
            } else if constexpr (kDequantHalfStream != 0) {
              if (k == 0) {
                ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_0);
                if constexpr (kDequantHalfStream == 3)
                  ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_1);
              } else if constexpr (kDequantHalfStream != 3) {
                if (k == 2) ptx::sync_unaligned(kDequantHalfReadyThreads, half_ready_barrier_1);
              }
            }
            auto desc_a = mma::sm90::make_smem_desc(
                smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
            auto desc_b =
                mma::sm90::make_smem_desc(smem_b[stage_idx] + wg_n_idx * BLOCK_K + k * WGMMA::K,
                                          1);  // NVFP4: no swizzle on B
            WGMMA::wgmma(desc_a, desc_b, accum, k);
          }
          ptx::warpgroup_commit_batch();
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
          ptx::warpgroup_wait<0>();

          arrive_empty_barrier(stage_idx);

// L1: gate/up alternate at gran=8 along N; each `i` block of 8
// cols belongs entirely to one of {gate, up}, so .x and .y
// share the same scalar.
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
            final_accum[i * 4 + 0] += scale_a_0 * accum[i * 4 + 0];
            final_accum[i * 4 + 1] += scale_a_0 * accum[i * 4 + 1];
            final_accum[i * 4 + 2] += scale_a_1 * accum[i * 4 + 2];
            final_accum[i * 4 + 3] += scale_a_1 * accum[i * 4 + 3];
          }
        }
      }
    };

    // RS-swapAB main loop: math gates directly on `full_barriers`
    // (packed-FP4 arrival AND activation/SFA arrival), decodes the
    // packed weights into WGMMA A-register fragments per k32 slice,
    // and runs weights-as-A / activations-as-B WGMMAs. No FP8 SMEM
    // write-back, no WGMMA weight re-read.
    const auto run_rs_swapab_gemm_loop = [&]() {
      using WGMMARS = L1WGMMARS;
      FI_MEGAMOE_STATIC_ASSERT(WGMMARS::kNumAccum == kAccumPerThread,
                               "RS and SS accumulator budgets must match");
      FI_MEGAMOE_STATIC_ASSERT(BLOCK_K / WGMMARS::K == 4,
                               "RS loop expects four k32 slices per stage");
      // Proven A-fragment map for wgmma.m64n128k32 e4m3: per k32
      // slice, thread (warp w, lane l) holds rows 16w + l/4 and
      // 16w + l/4 + 8 of its WG's 64-row A tile at K nibble-group
      // 4*(l%4) (+16 for a2/a3). Lane pairs (l ^ 1) share packed
      // words: the even lane decodes row r, the odd lane row r + 8,
      // and the discarded uint2 halves are exchanged over shfl so
      // every word is decoded exactly once.
      const uint32_t frag_row0 = epilogue_wg_idx * WG_BLOCK_M + warp_idx_in_wg * 16 + row_idx;
      const uint32_t decode_row = frag_row0 + ((lane_idx & 1u) << 3);
      const uint32_t word_sel = (lane_idx >> 1) & 1u;
      const bool keep_hi = (lane_idx & 1u) == 0;

      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        // Producer arrivals: TMA-B (packed FP4 + fused scales) and
        // TMA-A (+SFA), both counted on the same barrier.
        full_barriers[stage_idx]->wait(phase);

        const auto* packed_base = reinterpret_cast<const uint8_t*>(smem_b[stage_idx]);
#if FI_MEGAMOE_NVFP4_L1_RS_DEBUG_DECODE
        const auto* prow_r0 = packed_base + frag_row0 * B_LOAD_BYTES_PER_ROW;
        const auto* prow_r8 = prow_r0 + 8 * B_LOAD_BYTES_PER_ROW;
        const uint2 scale_words_r0 = *reinterpret_cast<const uint2*>(prow_r0 + 64);
        const uint2 scale_words_r8 = *reinterpret_cast<const uint2*>(prow_r8 + 64);
#else
        const auto* prow = packed_base + decode_row * B_LOAD_BYTES_PER_ROW;
        const uint2 scale_words = *reinterpret_cast<const uint2*>(prow + 64);
#endif

        // Slice `s` consumes packed words 4s..4s+3; this lane's
        // words are 4s + word_sel (K-lo half, UE4M3 scale byte 2s)
        // and 4s + 2 + word_sel (K-hi half, scale byte 2s + 1).
        uint32_t a_frag[2][4];
        const auto decode_slice = [&](const uint32_t& s, uint32_t(&a)[4]) {
#if FI_MEGAMOE_NVFP4_L1_RS_DEBUG_DECODE
          const uint4 q_r0 = reinterpret_cast<const uint4*>(prow_r0)[s];
          const uint4 q_r8 = reinterpret_cast<const uint4*>(prow_r8)[s];
          const uint32_t sw_r0 = s < 2 ? scale_words_r0.x : scale_words_r0.y;
          const uint32_t sw_r8 = s < 2 ? scale_words_r8.x : scale_words_r8.y;
          flashinfer::megamoe::nvfp4::dequant_mode2_lop3_rs_word_pair_noshfl(
              word_sel ? q_r0.y : q_r0.x, word_sel ? q_r0.w : q_r0.z, word_sel ? q_r8.y : q_r8.x,
              word_sel ? q_r8.w : q_r8.z, smem_nvfp4_lut[(sw_r0 >> ((s & 1u) * 16u)) & 0x7fu],
              smem_nvfp4_lut[(sw_r0 >> ((s & 1u) * 16u + 8u)) & 0x7fu],
              smem_nvfp4_lut[(sw_r8 >> ((s & 1u) * 16u)) & 0x7fu],
              smem_nvfp4_lut[(sw_r8 >> ((s & 1u) * 16u + 8u)) & 0x7fu], keep_hi, a);
#else
          const uint4 q = reinterpret_cast<const uint4*>(prow)[s];
          const uint32_t sw = s < 2 ? scale_words.x : scale_words.y;
          flashinfer::megamoe::nvfp4::dequant_mode2_lop3_rs_word_pair(
              word_sel ? q.y : q.x, word_sel ? q.w : q.z,
              smem_nvfp4_lut[(sw >> ((s & 1u) * 16u)) & 0x7fu],
              smem_nvfp4_lut[(sw >> ((s & 1u) * 16u + 8u)) & 0x7fu], keep_hi, a);
#endif
        };

        decode_slice(0, a_frag[0]);
#pragma unroll
        for (uint32_t s = 0; s < BLOCK_K / WGMMARS::K; ++s) {
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) mma::sm90::warpgroup_fence_operand(a_frag[s & 1u][i]);
#pragma unroll
          for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
          // wgmma.fence is mandatory before EVERY wgmma here:
          // the A registers are rewritten between groups (an
          // accumulator-only chain would need only one).
          ptx::warpgroup_arrive();
          // Swapped-role B descriptor: the full 128-token
          // activation tile, B128 swizzle, no row offset (both
          // WGs read the whole tile, the swapped-role cost).
          const auto desc_b = mma::sm90::make_smem_desc(smem_a[stage_idx] + s * WGMMARS::K, 1);
          WGMMARS::wgmma(a_frag[s & 1u], desc_b, accum, s > 0);
          ptx::warpgroup_commit_batch();
          if (s + 1 < BLOCK_K / WGMMARS::K) {
#if FI_MEGAMOE_NVFP4_L1_RS_DEBUG_DECODE
            // Debug arm: fully serialize the WGMMA groups.
            ptx::warpgroup_wait<0>();
#else
            // Retire group s-1 before rewriting its A-fragment
            // buffer (the load-bearing register-WAR gate);
            // group s stays in flight under the next decode.
            if (s >= 1) ptx::warpgroup_wait<1>();
#endif
// Pin the retired buffer: the compiler must not sink
// the decode's register writes above the wait that
// retires the wgmma group still reading them.
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) ptx::warpgroup_fence_operand(a_frag[(s + 1) & 1u][r]);
            decode_slice(s + 1, a_frag[(s + 1) & 1u]);
          }
        }
#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
        // WAR-2: all groups retired, so the async smem_a
        // descriptor reads are finished. WAR-1: group completion
        // is warpgroup-collective, so every lane issued its wgmma
        // and, by register data dependence through a_frag, every
        // packed smem_b staging LDS has retired.
        ptx::warpgroup_wait<0>();

// Per-token activation SF apply (the transposed axis: SFA
// now scales WGMMA N columns). Must complete before the
// empty release below: TMA-A rewrites smem_sfa together
// with smem_a under ring wraparound (the third WAR
// hazard). This mirrors the fused swapAB precedent.
#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
          const uint32_t token_0 = i * 8 + col_idx * 2;
          const float2 sfa =
              ptx::ld_shared(reinterpret_cast<const float2*>(smem_sfa[stage_idx] + token_0));
          final_accum[i * 4 + 0] += sfa.x * accum[i * 4 + 0];
          final_accum[i * 4 + 1] += sfa.y * accum[i * 4 + 1];
          final_accum[i * 4 + 2] += sfa.x * accum[i * 4 + 2];
          final_accum[i * 4 + 3] += sfa.y * accum[i * 4 + 3];
        }

        // The release is issued by warp 0 lanes 0-1 only, while the
        // SFA loads above run per-thread across the whole warpgroup:
        // without a rendezvous, warp 0 can release the stage while
        // warps 1-3 are still reading smem_sfa. Both adversarial
        // reviews found this independently.
        ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

        // Release LAST, after the SFA reads (NOT at the default
        // path's position right after the WGMMA wait).
        arrive_empty_barrier(stage_idx);
      }
    };

    if (!inactive_math_wg) {
      if constexpr (kRSSwapAB)
        run_rs_swapab_gemm_loop();
      else
        run_default_gemm_loop();
    }
    if constexpr (kSingleAccum) {
// The running total is in units of the last non-zero SF per row; restore
// absolute units. An inactive warpgroup never entered the loop, so its
// unit scales are still zero and its zero accumulators stay zero.
#pragma unroll
      for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
        final_accum[i * 4 + 0] *= unit_scale_0;
        final_accum[i * 4 + 1] *= unit_scale_0;
        final_accum[i * 4 + 2] *= unit_scale_1;
        final_accum[i * 4 + 3] *= unit_scale_1;
      }
    }

    // Even a fully invalid tail warpgroup must participate in the
    // CTA-wide scale rendezvous. It carries zero accumulators and only
    // writes padding rows, which L2 never scatters.

    const float l1_global_scale =
        l1_global_scales == nullptr ? 1.0f : __ldg(l1_global_scales + local_expert_idx);
    if constexpr (kRSSwapAB) {
      // ---------------- RS-swapAB L1 EPILOGUE ----------------
      // D is transposed relative to the default path: WGMMA M is 64
      // weight rows, WGMMA N is 128 tokens. Gate/up alternate at
      // granularity 8 along the weight rows, so each thread's row
      // pair (r_0 = gate, r_1 = r_0 + 8 = up) is exactly one SwiGLU
      // pair, and chunk `i` covers tokens i*8 + col_idx*2 + {0,1}.
      // Weight rows 16g..16g+15 of this 128-row block produce
      // output I-columns 8g..8g+7, hence:
      const uint32_t out_col = epilogue_wg_idx * 32 + warp_idx_in_wg * 8 + row_idx;
      constexpr uint32_t kNumTokenChunks = kAccumPerThread / 4;
      auto* smem_rs_amax = math::advance_ptr<float>(smem_cd_base, SMEM_CD_L1_SIZE);
      auto* l2_sf_base = l2_sf_buffer.get_base_ptr<float>();
      const uint32_t dense_sf_group_idx = n_block_idx / 2;
      const uint32_t peer_cta_rank = cute::block_rank_in_cluster() ^ 1u;

      auto silu = [](float x) -> float {
        const float e = kFastMath ? __expf(-x) : expf(-x);
        const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
        return x * sig;
      };
      auto clamp_gate = [](float& x) {
        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
          x = cute::min(x, kActivationClamp);
      };
      auto clamp_up = [](float& x) {
        if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
          x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
      };

      // SwiGLU + per-token amax. Invalid tokens contribute exact
      // zeros (their accumulators may hold NaN/Inf from garbage SFA
      // in padding rows), so the DSM exchange below never ships
      // garbage: a dummy peer CTA sends all-zero amax.
      float v_0[kNumTokenChunks], v_1[kNumTokenChunks];
#pragma unroll
      for (uint32_t i = 0; i < kNumTokenChunks; ++i) {
        const uint32_t token_0 = i * 8 + col_idx * 2;
        const uint32_t token_1 = token_0 + 1;
        v_0[i] = 0.0f;
        v_1[i] = 0.0f;
        if (token_0 < valid_m) {
          float g = final_accum[i * 4 + 0] * l1_global_scale;
          clamp_gate(g);
          float u = final_accum[i * 4 + 2] * l1_global_scale;
          clamp_up(u);
          const float weight = *l1_topk_weights_buffer.get_data_buffer(m_idx + token_0)
                                    .template get_base_ptr<float>();
          v_0[i] = silu(g) * u * weight;
        }
        if (token_1 < valid_m) {
          float g = final_accum[i * 4 + 1] * l1_global_scale;
          clamp_gate(g);
          float u = final_accum[i * 4 + 3] * l1_global_scale;
          clamp_up(u);
          const float weight = *l1_topk_weights_buffer.get_data_buffer(m_idx + token_1)
                                    .template get_base_ptr<float>();
          v_1[i] = silu(g) * u * weight;
        }
        // Reduce over the 8 lanes sharing col_idx (the 8 I-columns
        // this warp owns for these two tokens).
        const float amax_0 =
            math::warp_reduce<4, true>(cute::abs(v_0[i]), math::ReduceMax<float>());
        const float amax_1 =
            math::warp_reduce<4, true>(cute::abs(v_1[i]), math::ReduceMax<float>());
        if (row_idx == 0) {
          smem_rs_amax[token_0 * kNumEpilogueWarps + epilogue_warp_idx] = amax_0;
          smem_rs_amax[token_1 * kNumEpilogueWarps + epilogue_warp_idx] = amax_1;
        }
      }
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

      // Cross-warp per-token reduction and adjacent-N cluster DSM
      // exchange. Token t is owned by epilogue thread t; even
      // threads ship one float2 token pair (64 senders x 8 B =
      // 512 B), matching the pre-armed BLOCK_M * sizeof(float)
      // transaction. The peer slots alias the CD tile exactly as in
      // the default path: the exchange completes before the
      // quantize below overwrites the tile.
      float token_amax = 0.0f;
      if (epilogue_thread_idx < BLOCK_M) {
#pragma unroll
        for (uint32_t w = 0; w < kNumEpilogueWarps; ++w)
          token_amax =
              cute::max(token_amax, smem_rs_amax[epilogue_thread_idx * kNumEpilogueWarps + w]);
        const float token_amax_next = __shfl_down_sync(0xffffffff, token_amax, 1);
        if ((epilogue_thread_idx & 1u) == 0) {
          ptx::st_shared_cluster_async(
              reinterpret_cast<float2*>(smem_cd_l1) + epilogue_thread_idx / 2,
              make_float2(token_amax, token_amax_next), combine_barriers[0], peer_cta_rank);
        }
      }
      if (epilogue_thread_idx == 0) {
        combine_barriers[0]->wait(pair_scale_phase);
        pair_scale_phase ^= 1u;
        // Pre-arm the next phase. The current quantize/store tail
        // and the following GEMM separate this from peer stores.
        combine_barriers[0]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
      }
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

      // Both CTAs derive the same per-token scale; only the even-N
      // owner publishes the dense logical per-128 SF entry.
      if (epilogue_thread_idx < BLOCK_M) {
        const uint32_t token = epilogue_thread_idx;
        const float peer_amax = ptx::ld_shared(reinterpret_cast<const float*>(smem_cd_l1) + token);
        const float common_amax = cute::max(token_amax, peer_amax);
        float2 sf_pair, sf_inv_pair;
        math::get_e4m3_sf_and_sf_inv(make_float2(common_amax, common_amax), sf_pair, sf_inv_pair);
        if ((n_block_idx & 1u) == 0 and token < valid_m)
          l2_sf_base[dense_sf_group_idx * kNumPaddedSFPoolTokens + pool_block_idx * BLOCK_M +
                     token] = sf_pair.x;
        smem_rs_amax[token * kNumEpilogueWarps] = sf_inv_pair.x;
      }
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

// Quantize and write to smem_cd_l1 ([token][I], row-major, no
// swizzle) so L2 consumes the pool buffer unchanged.
#pragma unroll
      for (uint32_t i = 0; i < kNumTokenChunks; ++i) {
        const uint32_t token_0 = i * 8 + col_idx * 2;
        const uint32_t token_1 = token_0 + 1;
        if (token_0 < valid_m) {
          const float sf_inv = smem_rs_amax[token_0 * kNumEpilogueWarps];
          const __nv_fp8_e4m3 q(v_0[i] * sf_inv);
          reinterpret_cast<uint8_t*>(smem_cd_l1)[token_0 * L1_OUT_BLOCK_N + out_col] =
              *reinterpret_cast<const uint8_t*>(&q);
        }
        if (token_1 < valid_m) {
          const float sf_inv = smem_rs_amax[token_1 * kNumEpilogueWarps];
          const __nv_fp8_e4m3 q(v_1[i] * sf_inv);
          reinterpret_cast<uint8_t*>(smem_cd_l1)[token_1 * L1_OUT_BLOCK_N + out_col] =
              *reinterpret_cast<const uint8_t*>(&q);
        }
      }

      // Both warpgroups wrote into both token halves, so the store
      // must wait on the whole CTA rather than the default path's
      // per-WG barrier. WG wg's CD sub-tile still holds token rows
      // [64*wg, 64*wg + 64) x 64 I-columns, which is exactly what
      // the unchanged per-WG TMA store below expects.
      auto* smem_cd_l1_wg = smem_cd_l1 + epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N;
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
      if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
        cute::tma_store_fence();
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1_wg, out_n_idx,
                                      m_idx + row_block_offset);
        cute::tma_store_arrive();
      }
      __syncwarp();
      ptx::tma_store_wait<0>();
      if constexpr (!kL2ArrivalCounter)
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    } else {
      // ---------------- L1 EPILOGUE: SwiGLU + FP8 quantize + TMA store ----------------
      // Layout in `final_accum`:
      //   16 chunks of 8 N-cols, each chunk = 4 floats per thread = (r0c0, r0c1, r1c0, r1c1).
      //   Gate chunks: even (0, 2, ..., 14). Up chunks: odd (1, 3, ..., 15).
      //   Pair `p` ∈ [0, 8): gate chunk = 2p, up chunk = 2p+1.
      //
      // For each pair we produce 4 post-SwiGLU floats per thread, mapped to
      // output cols (p*8 + col_idx*2 + {0,1}) for both r0 and r1.

      constexpr uint32_t kNumPairs = kAccumPerThread / 8;
      constexpr uint32_t kNumSFGroups = WG_L1_OUT_BLOCK_N / kLocalL1ActsSFGranK;
      FI_MEGAMOE_STATIC_ASSERT(kNumSFGroups == 1,
                               "each paired CTA must own one local 64-column SF group");
      float swiglu_r0[kNumPairs][2];
      float swiglu_r1[kNumPairs][2];

      // Each CTA computes one local 64-column amax for all 128 rows.
      // Its adjacent-N cluster peer owns the other half.
      float amax_r0[kNumSFGroups] = {};
      float amax_r1[kNumSFGroups] = {};

// Compute SwiGLU + per-group amax.
#pragma unroll
      for (uint32_t p = 0; p < kNumPairs; ++p) {
        const uint32_t gate = 2 * p, up = 2 * p + 1;
        const uint32_t sf_group = p / 8;

        auto clamp_gate = [](float& x) {
          if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
            x = cute::min(x, kActivationClamp);
        };
        auto clamp_up = [](float& x) {
          if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity())
            x = cute::min(cute::max(x, -kActivationClamp), kActivationClamp);
        };
        float g_r0_c0 = final_accum[gate * 4 + 0] * l1_global_scale;
        clamp_gate(g_r0_c0);
        float g_r0_c1 = final_accum[gate * 4 + 1] * l1_global_scale;
        clamp_gate(g_r0_c1);
        float g_r1_c0 = final_accum[gate * 4 + 2] * l1_global_scale;
        clamp_gate(g_r1_c0);
        float g_r1_c1 = final_accum[gate * 4 + 3] * l1_global_scale;
        clamp_gate(g_r1_c1);
        float u_r0_c0 = final_accum[up * 4 + 0] * l1_global_scale;
        clamp_up(u_r0_c0);
        float u_r0_c1 = final_accum[up * 4 + 1] * l1_global_scale;
        clamp_up(u_r0_c1);
        float u_r1_c0 = final_accum[up * 4 + 2] * l1_global_scale;
        clamp_up(u_r1_c0);
        float u_r1_c1 = final_accum[up * 4 + 3] * l1_global_scale;
        clamp_up(u_r1_c1);

        auto silu = [](float x) -> float {
          const float e = kFastMath ? __expf(-x) : expf(-x);
          const float sig = kFastMath ? math::fast_rcp(1.0f + e) : 1.0f / (1.0f + e);
          return x * sig;
        };

        if (valid_r0) {
          swiglu_r0[p][0] = silu(g_r0_c0) * u_r0_c0;
          swiglu_r0[p][1] = silu(g_r0_c1) * u_r0_c1;
          amax_r0[sf_group] = cute::max(
              amax_r0[sf_group], cute::max(cute::abs(swiglu_r0[p][0]), cute::abs(swiglu_r0[p][1])));
        } else {
          swiglu_r0[p][0] = 0.0f;
          swiglu_r0[p][1] = 0.0f;
        }
        if (valid_r1) {
          swiglu_r1[p][0] = silu(g_r1_c0) * u_r1_c0;
          swiglu_r1[p][1] = silu(g_r1_c1) * u_r1_c1;
          amax_r1[sf_group] = cute::max(
              amax_r1[sf_group], cute::max(cute::abs(swiglu_r1[p][0]), cute::abs(swiglu_r1[p][1])));
        } else {
          swiglu_r1[p][0] = 0.0f;
          swiglu_r1[p][1] = 0.0f;
        }
      }

      const float weight_r0 = valid_r0
                                  ? *l1_topk_weights_buffer.get_data_buffer(m_idx + row_offset_r0)
                                         .template get_base_ptr<float>()
                                  : 0.0f;
      const float weight_r1 = valid_r1
                                  ? *l1_topk_weights_buffer.get_data_buffer(m_idx + row_offset_r1)
                                         .template get_base_ptr<float>()
                                  : 0.0f;
#pragma unroll
      for (uint32_t p = 0; p < kNumPairs; ++p) {
        swiglu_r0[p][0] *= weight_r0;
        swiglu_r0[p][1] *= weight_r0;
        swiglu_r1[p][0] *= weight_r1;
        swiglu_r1[p][1] *= weight_r1;
      }
#pragma unroll
      for (uint32_t g = 0; g < kNumSFGroups; ++g) {
        amax_r0[g] *= cute::abs(weight_r0);
        amax_r1[g] *= cute::abs(weight_r1);
      }
#pragma unroll
      for (uint32_t g = 0; g < kNumSFGroups; ++g) {
        amax_r0[g] = math::warp_reduce<4, false>(amax_r0[g], math::ReduceMax<float>());
        amax_r1[g] = math::warp_reduce<4, false>(amax_r1[g], math::ReduceMax<float>());
      }

      // Direct per-128 rendezvous across adjacent-N cluster CTAs.
      // Pack the two output rows owned by each four-lane group into
      // one contiguous float2 DSM slot. This keeps the same 512-byte
      // transaction/barrier protocol while halving DSM store and
      // shared-load instructions versus row-strided scalar slots.
      auto* l2_sf_base = l2_sf_buffer.get_base_ptr<float>();
      const uint32_t token_r0 = pool_block_idx * BLOCK_M + row_offset_r0;
      const uint32_t token_r1 = pool_block_idx * BLOCK_M + row_offset_r1;
      const uint32_t dense_sf_group_idx = n_block_idx / 2;
      const uint32_t peer_cta_rank = cute::block_rank_in_cluster() ^ 1u;
      auto* smem_cd_l1_wg = smem_cd_l1 + epilogue_wg_idx * WG_BLOCK_M * L1_OUT_BLOCK_N;
      const uint32_t row_group_idx = epilogue_thread_idx / 4;
      auto* peer_amax_pair_slot = reinterpret_cast<float2*>(smem_cd_l1) + row_group_idx;

      if (col_idx == 0) {
        ptx::st_shared_cluster_async(peer_amax_pair_slot, make_float2(amax_r0[0], amax_r1[0]),
                                     combine_barriers[0], peer_cta_rank);
      }
      if (epilogue_thread_idx == 0) {
        combine_barriers[0]->wait(pair_scale_phase);
        pair_scale_phase ^= 1u;
        // Pre-arm the next phase. The current quantize/store tail
        // and the following GEMM separate this from peer stores.
        combine_barriers[0]->arrive_and_expect_tx(BLOCK_M * sizeof(float));
      }
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);

      float sf_r0 = 0.0f, sf_inv_r0 = 0.0f;
      float sf_r1 = 0.0f, sf_inv_r1 = 0.0f;
      if (col_idx == 0) {
        const float2 peer_amax_pair = ptx::ld_shared(peer_amax_pair_slot);
        const float2 common_amax_pair = {cute::max(amax_r0[0], peer_amax_pair.x),
                                         cute::max(amax_r1[0], peer_amax_pair.y)};
        float2 sf_pair, sf_inv_pair;
        math::get_e4m3_sf_and_sf_inv(common_amax_pair, sf_pair, sf_inv_pair);
        sf_r0 = sf_pair.x;
        sf_inv_r0 = sf_inv_pair.x;
        sf_r1 = sf_pair.y;
        sf_inv_r1 = sf_inv_pair.y;
      }
      const uint32_t row_group_leader = lane_idx & ~3u;
      sf_inv_r0 = __shfl_sync(0xffffffff, sf_inv_r0, row_group_leader);
      sf_inv_r1 = __shfl_sync(0xffffffff, sf_inv_r1, row_group_leader);

      // Both CTAs derive the same scale; only the even-N owner
      // publishes the dense logical per-128 SF entry.
      if ((n_block_idx & 1u) == 0 && col_idx == 0) {
        if (valid_r0) l2_sf_base[dense_sf_group_idx * kNumPaddedSFPoolTokens + token_r0] = sf_r0;
        if (valid_r1) l2_sf_base[dense_sf_group_idx * kNumPaddedSFPoolTokens + token_r1] = sf_r1;
      }

      // Quantize and write to smem_cd_l1 (row-major, no swizzle).
      constexpr uint32_t l1_store_stage = 0u;
      smem_cd_l1_wg += l1_store_stage * kNumEpilogueWarpgroups * WG_BLOCK_M * L1_OUT_BLOCK_N;
#pragma unroll
      for (uint32_t p = 0; p < kNumPairs; ++p) {
        const float v00 = swiglu_r0[p][0] * sf_inv_r0;
        const float v01 = swiglu_r0[p][1] * sf_inv_r0;
        const float v10 = swiglu_r1[p][0] * sf_inv_r1;
        const float v11 = swiglu_r1[p][1] * sf_inv_r1;

        const __nv_fp8x2_e4m3 r0_pair(make_float2(v00, v01));
        const __nv_fp8x2_e4m3 r1_pair(make_float2(v10, v11));

        const uint32_t col = p * 8 + col_idx * 2;
        auto* p0 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + r_0 * L1_OUT_BLOCK_N +
                                               wg_l1_out_n_idx + col);
        auto* p1 = reinterpret_cast<uint16_t*>(smem_cd_l1_wg + r_1 * L1_OUT_BLOCK_N +
                                               wg_l1_out_n_idx + col);
        if (valid_r0) *p0 = r0_pair.__x;
        if (valid_r1) *p1 = r1_pair.__x;
      }

      // Issue TMA store of the entire tile. Padding rows beyond
      // `valid_m` are written with stale/garbage FP8 to the L1-output
      // pool buffer, but they are never consumed downstream: the L2
      // GEMM tile loads them, but its NVLink-scatter epilogue is
      // gated by `m_idx_in_block >= valid_m`, and stale SF in the
      // padding rows can produce NaN accumulators that simply stay
      // in registers (only valid rows are converted to BF16 and
      // STSM'd into smem). Using TMA for partial tiles is a large
      // win for low-batch / decode where every tile is partial.
      ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);
      if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
        const uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
        cute::tma_store_fence();
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_l1_output, smem_cd_l1_wg, out_n_idx,
                                      m_idx + row_block_offset);
        cute::tma_store_arrive();
      }
      __syncwarp();
      ptx::tma_store_wait<0>();
      if constexpr (!kL2ArrivalCounter)
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    }
  });

  ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
  return;
}
#else
if (blockIdx.x == 0 and threadIdx.x == 0)
  FI_MEGAMOE_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
