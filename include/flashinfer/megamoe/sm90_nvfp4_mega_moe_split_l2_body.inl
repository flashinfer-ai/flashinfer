#ifndef FI_MEGAMOE_NVFP4_LUT_COMPACT
#define FI_MEGAMOE_NVFP4_LUT_COMPACT 0
#endif
// Independent SM90 NVFP4 MegaMoE l2 kernel body.
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900) and (__CUDA_ARCH__ < 1000)) or \
    defined(__CLION_IDE__)
using Barrier = cutlass::arch::ClusterTransactionBarrier;

// =====================================================================
// Template checks
// =====================================================================
FI_MEGAMOE_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

// =====================================================================
// Thread / warp identification
// =====================================================================
const uint32_t sm_idx = blockIdx.x;
const uint32_t thread_idx = threadIdx.x;
const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
const uint32_t lane_idx = ptx::get_lane_idx();

if (warp_idx == 0 and cute::elect_one_sync()) {
  cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
  cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
  cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
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
using L1WGMMA = typename mma::sm90::FP8MMASelector<128>::type;
static_assert(L1WGMMA::M == 64 and L1WGMMA::N == WG_BLOCK_N and L1WGMMA::K == 32,
              "Unexpected WGMMA shape");

// Split L2 uses independent CTAs and CTA-local A/B tiles.
constexpr uint32_t LOAD_BLOCK_M = 128;
constexpr uint32_t LOAD_BLOCK_N = 128;
constexpr uint32_t kSwizzleAMode = 128;
constexpr uint32_t kL2ActsSFGranK = 128;  // L1 output and L2 input SF granularity
// Split L1 and L2 share the same canonical BM128 physical pool-block unit,
// so scheduling and pool-prefix accounting use the same granularity.
constexpr uint32_t kPoolBlockM = 128;
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
constexpr uint32_t SMEM_NVFP4_LUT_SIZE =
    math::constexpr_align<uint32_t>(128u * sizeof(uint2), kSharedMemoryAlignment);
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
constexpr uint32_t B_LOAD_BYTES_PER_ROW = 80u;
constexpr uint32_t SMEM_B_LOAD_SIZE_PER_STAGE = LOAD_BLOCK_N * B_LOAD_BYTES_PER_ROW;
// One per-128 activation scale is loaded for each row and BK128 tile.
constexpr uint32_t kL2SFAHalfStride =
    math::constexpr_align<uint32_t>(BLOCK_M * sizeof(float), 128u) / sizeof(float);
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kL2SFAHalfStride * sizeof(float);
constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
    SMEM_EXPERT_COUNT_SIZE + SMEM_NVFP4_LUT_SIZE +
    kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);

// SMEM pointers
auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
auto smem_nvfp4_lut =
    reinterpret_cast<uint2*>(math::advance_ptr<uint8_t>(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

// L2 remote-scatter staging. Epilogue warp `w` owns rows `[16w, 16w + 16)`
// of its warpgroup tile, so the tile is warp-private and only one 8-row
// half (`r_0` or `r_1`) needs to be resident at a time: a full 16-row tile
// would cost 32768 B and drop `max_num_stages` from 6 to 5. The row stride
// is padded by 8 BF16 so the 4-byte staging stores are bank-conflict free
// (`row * 68` words, all distinct modulo 32) while every staged row start
// stays 16-byte aligned for the `ld.shared.v4` read-back.
constexpr uint32_t kL2StageRows = 8;
constexpr uint32_t kL2StageRowPad = 8;
constexpr uint32_t kL2StageRowStride = WG_BLOCK_N + kL2StageRowPad;
constexpr uint32_t SMEM_CD_L2_PER_WARP = kL2StageRows * kL2StageRowStride * sizeof(nv_bfloat16);
constexpr uint32_t SMEM_CD_L2_SIZE = math::constexpr_align<uint32_t>(
    kNumEpilogueWarps * SMEM_CD_L2_PER_WARP, kSharedMemoryAlignment);
FI_MEGAMOE_STATIC_ASSERT(kL2StageRowStride % 8 == 0, "Staged rows must stay 16-byte aligned");
FI_MEGAMOE_STATIC_ASSERT(SMEM_CD_L2_SIZE == kNumEpilogueWarps * SMEM_CD_L2_PER_WARP,
                         "Host and device `smem_cd_l2` must agree exactly");

auto smem_cd_l2_base =
    math::advance_ptr<uint8_t>(smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_NVFP4_LUT_SIZE);

auto smem_gemm_base =
    math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_NVFP4_LUT_SIZE + SMEM_CD_L2_SIZE);

auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<a_dtype_t>(smem_gemm_base, i * SMEM_A_SIZE_PER_STAGE);
});
auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
  return math::advance_ptr<b_dtype_t>(
      smem_gemm_base, kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
});
auto sf_start_ptr = math::advance_ptr<uint8_t>(
    smem_gemm_base, kNumStages*(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
  return reinterpret_cast<float*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
});
constexpr uint32_t kNumDequantBarriers = kNumStages;

// Barriers live after SF.
auto barrier_start_ptr =
    reinterpret_cast<Barrier*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE);
auto full_barriers =
    utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
auto empty_barriers =
    utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
auto dequant_barriers = utils::PatternVisitor(
    [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
auto combine_barriers = utils::PatternVisitor([=](const uint32_t& i) {
  return barrier_start_ptr + kNumStages * 2 + kNumDequantBarriers + i;
});

// =====================================================================
// Initialization
// =====================================================================
if constexpr (FI_MEGAMOE_NVFP4_LUT_COMPACT != 0) {
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
} else if (warp_idx == 2) {
  // Init GEMM full/empty barriers and combine barriers
  if (cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      // Producer arrivals: A(+SFA) + B(TMA+SFB). SFB is copied with
      // cp.async.bulk and counted as B-loader transaction bytes, so
      // it does not need a separate producer arrival.
      full_barriers[i]->init(2);
      empty_barriers[i]->init(kNumEpilogueWarps);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) dequant_barriers[i]->init(1);
#pragma unroll
    for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++i) combine_barriers[i]->init(1);
  }
  cutlass::arch::fence_barrier_init();
}
__syncthreads();

// =====================================================================
// Scheduler (cluster=1)
// =====================================================================
auto scheduler =
    sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K, L2_SHAPE_N,
                            L2_SHAPE_K, kNumExpertsPerRank, kNumExpertsPerWave, kNumSMs, kNumRanks,
                            1, kPoolBlockM>(workspace);

// Pipeline state shared by TMA loaders and math warpgroups
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
  ++k_block_idx;
  stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
  phase ^= stage_idx == 0;
};
const auto dequant_loaded_b_stage = [&](const uint32_t& s, const uint32_t& p,
                                        const uint32_t& non_epilogue_thread_idx) {
  if (non_epilogue_thread_idx >= 64u) {
    full_barriers[s]->wait(p);
    const uint32_t dequant_tid = non_epilogue_thread_idx - 64u;
    flashinfer::megamoe::nvfp4::dequant_smem_b_inplace_two_rows_mode2_lop3<64u, 8u>(
        reinterpret_cast<uint8_t*>(smem_b[s]), dequant_tid, smem_nvfp4_lut);
    cutlass::arch::fence_view_async_shared();
    ptx::sync_aligned(64, 8);
    if (dequant_tid == 0) dequant_barriers[s]->arrive();
  }
};

// Intra-SM barrier indices (mirroring SM100)
constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
constexpr uint32_t kEpilogueFullBarrierIdx = 2;
constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

// Cross-rank NVLink barrier tags
constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

// Register reconfiguration counts (chosen to fit in 64512 reg budget).
// For the 256-epilogue-thread loader-dequant case (block_m=128, 2 math WGs),
// give non-epilogue warps more room for two-row NVFP4 dequant while staying
// inside the 64K register file: 128*48 + 128*64 + 256*192 = 63488.
constexpr uint32_t kNumNonEpilogueRegisters = 112;
constexpr uint32_t kNumEpilogueRegisters = 192;
FI_MEGAMOE_STATIC_ASSERT(kNumNonEpilogueRegisters* kNumNonEpilogueThreads +
                                 kNumEpilogueRegisters * kNumEpilogueThreads <=
                             64512,
                         "Too many registers");

constexpr uint32_t kDispatchGridSyncIndex = 0;
constexpr uint32_t kEpilogueGridSyncIndex = 1;

const auto for_each_selected_block = [&](auto&& func) {
  scheduler.for_each_linear2_block([&](const uint32_t& local_expert_idx,
                                       const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                                       const uint32_t& n_block_idx) {
    func(std::integral_constant<sched::BlockPhase, sched::BlockPhase::Linear2>{}, local_expert_idx,
         num_k_blocks, m_block_idx, n_block_idx);
  });
};

// =====================================================================
// ROLE 1: GEMM TMA LOAD warps
// =====================================================================
if (warp_idx == 0) {
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const auto tensor_map_a_ptr = [&]() { return &tensor_map_l2_acts; }();
    const auto tensor_map_sfa_ptr = [&]() { return &tensor_map_l2_acts_sf; }();

    const uint32_t pool_token_idx = scheduler.get_current_block_pool_token_idx();
    const uint32_t valid_m = scheduler.template get_valid_m<false>();
    const bool has_valid_m = valid_m > 0;

    // Wait for the pool to be ready. Cluster peers can be dummy CTAs for
    // the tail M unit when an expert has an odd number of M blocks.
    (void)has_valid_m;

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      if (cute::elect_one_sync()) {
        if (has_valid_m) {
          const uint32_t m_idx = pool_token_idx;
          const uint32_t k_idx = k_block_idx * BLOCK_K;

          // TMA load A
          tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
              tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 1);

          // One dense SFA group per BK128. The physical global
          // allocation is still twice as large, but the descriptor
          // and transaction address only its logical first half.
          tma::copy<BLOCK_M, 1, 0, float>(tensor_map_sfa_ptr, full_barriers[stage_idx],
                                          smem_sfa[stage_idx], m_idx, k_block_idx, 1);
          full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE +
                                                         BLOCK_M * sizeof(float));
        } else {
          full_barriers[stage_idx]->arrive();
        }
      }
      __syncwarp();
      dequant_loaded_b_stage(stage_idx, phase, lane_idx);
    }
  });

} else if (warp_idx == 1) {
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const auto tensor_map_b_ptr = [&]() { return &tensor_map_l2_weights; }();
    constexpr uint32_t shape_n = L2_SHAPE_N;

    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      empty_barriers[stage_idx]->wait(phase ^ 1);

      const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
      // NVFP4 fused B+scale layout stores 64B packed FP4 + 16B
      // UE4M3 scale per BK128 row.
      const uint32_t k_idx = k_block_idx * B_LOAD_BYTES_PER_ROW;
      if (cute::elect_one_sync()) {
        tma::copy<B_LOAD_BYTES_PER_ROW, LOAD_BLOCK_N, 0, b_dtype_t>(
            tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_LOAD_SIZE_PER_STAGE);
      }
      __syncwarp();
      dequant_loaded_b_stage(stage_idx, phase, 32u + lane_idx);
    }
  });

} else if (warp_idx < kNumMMANonEpilogueWarps) {
  // Idle non-epilogue warps (2 and 3). They must still
  // participate in the warpgroup-collective `setmaxnreg.dec.sync.aligned`
  // so that the math warpgroup's `warpgroup_reg_alloc` can succeed.
  cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

  const uint32_t non_epilogue_thread_idx = warp_idx * 32 + lane_idx;
  for_each_selected_block([&](const auto& block_phase, const uint32_t&,
                              const uint32_t& num_k_blocks, const uint32_t&, const uint32_t&) {
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
      dequant_loaded_b_stage(stage_idx, phase, non_epilogue_thread_idx);
      __syncwarp();
    }
  });
} else {
  // =====================================================================
  // ROLE 3: MATH WARPGROUPS (WGMMA + epilogue + combine)
  // =====================================================================
  cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

  const uint32_t epilogue_warp_idx = warp_idx - kNumMMANonEpilogueWarps;
  const uint32_t epilogue_wg_idx = epilogue_warp_idx / 4;
  const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
  const uint32_t warp_idx_in_wg = epilogue_warp_idx % 4;

  const auto arrive_empty_barrier = [&](const uint32_t& s) {
    if (lane_idx == 0) empty_barriers[s]->arrive();
  };

  const auto cleanup_workspace_from_epilogue = [&]() {
    FI_MEGAMOE_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
    if (sm_idx == 0) {
#pragma unroll
      for (uint32_t i = epilogue_thread_idx; i < kNumExperts; i += kNumEpilogueThreads)
        *workspace.get_expert_send_count_ptr(i) = 0;
    } else {
      for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
        const auto num_recv_tokens =
            static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
        const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, kPoolBlockM);
        const auto cleanup_pool_block_offset = scheduler.get_pool_block_offset(i);

        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        if (epilogue_thread_idx == 0) {
          *workspace.get_expert_recv_count_sum_ptr(i) = 0;
          if (cumulative_local_expert_recv_stats != nullptr)
            ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
        }

        for (uint32_t j = epilogue_thread_idx; j < kNumRanks; j += kNumEpilogueThreads)
          *workspace.get_expert_recv_count_ptr(j, i) = 0;

        for (uint32_t j = epilogue_thread_idx; j < num_recv_m_blocks; j += kNumEpilogueThreads)
          *workspace.get_l1_arrival_count_ptr(cleanup_pool_block_offset + j) = 0;
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
      }
    }
  };

  const auto finish_no_dispatch_cleanup = [&]() {
    ptx::sync_unaligned(kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);
    cleanup_workspace_from_epilogue();
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads, kDispatchGridSyncIndex,
                         kAfterWorkspaceCleanBarrierTag>(
        workspace, sym_buffer, sm_idx, epilogue_thread_idx,
        [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }, true, false);
  };

  // WGMMA-output register layout helpers
  const uint32_t row_idx = lane_idx / 4;
  const uint32_t col_idx = lane_idx % 4;
  const uint32_t r_0 = warp_idx_in_wg * 16 + row_idx;
  const uint32_t r_1 = r_0 + 8;

  FI_MEGAMOE_STATIC_ASSERT(WG_BLOCK_M == L1WGMMA::M,
                           "Each warpgroup must run exactly one WGMMA per K-block");

  // The staged scatter writes 16 bytes at a time. The combine row pitch
  // (`kHidden * 2`) is 16-byte aligned, so every destination row inherits
  // the buffer base alignment: check it once, not once per row.
  FI_MEGAMOE_DEVICE_ASSERT(
      reinterpret_cast<uint64_t>(
          combine_token_buffer.get_rank_buffer(0u).get_data_buffer(0u).get_base_ptr()) %
          16 ==
      0);

  ptx::sync_unaligned(kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  for_each_selected_block([&](const auto& block_phase, const uint32_t& local_expert_idx,
                              const uint32_t& num_k_blocks, const uint32_t& m_block_idx,
                              const uint32_t& n_block_idx) {
    const uint32_t valid_m = scheduler.template get_valid_m<false>();
    const uint32_t m_idx = scheduler.get_current_block_pool_token_idx();
    const uint32_t n_idx = n_block_idx * BLOCK_N;
    const uint32_t row_block_offset = epilogue_wg_idx * WG_BLOCK_M;
    const uint32_t row_offset_r0 = row_block_offset + r_0;
    const uint32_t row_offset_r1 = row_block_offset + r_1;
    const bool valid_r0 = row_offset_r0 < valid_m;
    const bool valid_r1 = row_offset_r1 < valid_m;
    const float l2_global_scale =
        l2_global_scales == nullptr ? 1.0f : __ldg(l2_global_scales + local_expert_idx);
    const auto cast_l2_scaled_bf16_pair = [&](float x, float y) -> uint32_t {
      x *= l2_global_scale;
      y *= l2_global_scale;
      return math::cast_into_bf16_and_pack(x, y);
    };

    if (row_block_offset >= valid_m) {
      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        dequant_barriers[stage_idx]->wait(phase);
        arrive_empty_barrier(stage_idx);
        __syncwarp();
      }
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
      return;
    }

    // ---------------- GEMM ----------------
    using WGMMA = L1WGMMA;
    constexpr uint32_t kAccumPerThread = WGMMA::kNumAccum;  // 64 for M=64,N=128
    float final_accum[kAccumPerThread] = {};
    float accum[kAccumPerThread];

    const auto run_default_gemm_loop = [&]() {
      for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
        dequant_barriers[stage_idx]->wait(phase);

        // One activation scale covers all four K32 WGMMAs in BK128.
        // Read it before warpgroup_arrive as required by WGMMA.
        const float scale_a_0 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r0);
        const float scale_a_1 = ptx::ld_shared(smem_sfa[stage_idx] + row_offset_r1);

        // NVFP4 UE4M3 weight scales are applied during FP4 -> FP8 smem
        // expansion, so the WGMMA accumulator only needs activation SF.

#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
        ptx::warpgroup_arrive();
#pragma unroll
        for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
          auto desc_a = mma::sm90::make_smem_desc(
              smem_a[stage_idx] + row_block_offset * BLOCK_K + k * WGMMA::K, 1);
          auto desc_b = mma::sm90::make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
          WGMMA::wgmma(desc_a, desc_b, accum, k);
        }
        ptx::warpgroup_commit_batch();
#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread; ++i) ptx::warpgroup_fence_operand(accum[i]);
        ptx::warpgroup_wait<0>();

        arrive_empty_barrier(stage_idx);

#pragma unroll
        for (uint32_t i = 0; i < kAccumPerThread / 4; ++i) {
          final_accum[i * 4 + 0] += scale_a_0 * accum[i * 4 + 0];
          final_accum[i * 4 + 1] += scale_a_0 * accum[i * 4 + 1];
          final_accum[i * 4 + 2] += scale_a_1 * accum[i * 4 + 2];
          final_accum[i * 4 + 3] += scale_a_1 * accum[i * 4 + 3];
        }
      }
    };

    run_default_gemm_loop();

    // ---------------- L2 EPILOGUE: BF16 cast + staged NVLink scatter ----------------
    // The direct path stored 4 bytes per lane, so a warp's 8 row groups
    // produced 8 scattered 16-byte requests per instruction, each billed as
    // a 32-byte sector. Staging one 8-row half in warp-private shared memory
    // lets every destination row leave as one 256-byte contiguous burst.
    auto smem_cd_l2 =
        math::advance_ptr<nv_bfloat16>(smem_cd_l2_base, epilogue_warp_idx * SMEM_CD_L2_PER_WARP);

    // Cast and pack one 8-row half into this warp's private tile. The
    // arithmetic is unchanged, so the same values reach the same addresses.
    const auto stage_rows = [&](const bool& valid_row, const uint32_t& row_accum_offset) {
      // This guard must stay textually equivalent to the scatter guard
      // below: invalid rows keep stale tile bytes and are skipped there.
      if (not valid_row) return;
#pragma unroll
      for (uint32_t i = 0; i < kAccumPerThread / 8; ++i) {
        const uint32_t chunk_lo = 2 * i, chunk_hi = 2 * i + 1;
        const uint32_t col_lo = chunk_lo * 8 + col_idx * 2;
        const uint32_t col_hi = chunk_hi * 8 + col_idx * 2;
        const uint32_t packed_lo =
            cast_l2_scaled_bf16_pair(final_accum[chunk_lo * 4 + row_accum_offset + 0],
                                     final_accum[chunk_lo * 4 + row_accum_offset + 1]);
        const uint32_t packed_hi =
            cast_l2_scaled_bf16_pair(final_accum[chunk_hi * 4 + row_accum_offset + 0],
                                     final_accum[chunk_hi * 4 + row_accum_offset + 1]);
        ptx::st_shared(
            reinterpret_cast<uint32_t*>(smem_cd_l2 + row_idx * kL2StageRowStride + col_lo),
            packed_lo);
        ptx::st_shared(
            reinterpret_cast<uint32_t*>(smem_cd_l2 + row_idx * kL2StageRowStride + col_hi),
            packed_hi);
      }
    };

    // Read the tile back with 16 lanes per destination row, 16 bytes each.
    constexpr uint32_t kNumScatterLanesPerRow = 16;
    constexpr uint32_t kScatterBytesPerLane =
        WG_BLOCK_N * sizeof(nv_bfloat16) / kNumScatterLanesPerRow;
    FI_MEGAMOE_STATIC_ASSERT(kScatterBytesPerLane == sizeof(uint4), "Expect one `uint4` per lane");
    const uint32_t scatter_row_in_pair = lane_idx / kNumScatterLanesPerRow;
    const uint32_t lane_in_row = lane_idx % kNumScatterLanesPerRow;
    const uint32_t scatter_group_leader = lane_idx - lane_in_row;
    const uint32_t scatter_group_mask = 0xffffu << scatter_group_leader;

    const auto scatter_staged_rows = [&](const uint32_t& row_base) {
#pragma unroll
      for (uint32_t j = 0; j < kL2StageRows / 2; ++j) {
        const uint32_t stage_row = j * 2 + scatter_row_in_pair;
        const uint32_t row_offset = row_base + stage_row;
        // Uniform within each 16-lane group, so the shuffles stay converged.
        if (row_offset >= valid_m) continue;
        uint32_t dst_rank_idx = 0;
        uint32_t dst_token_idx = 0;
        uint32_t dst_topk_idx = 0;
        if (lane_in_row == 0) {
          const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + row_offset);
          dst_rank_idx = src_metadata.rank_idx;
          dst_token_idx = src_metadata.token_idx;
          dst_topk_idx = src_metadata.topk_idx;
        }
        dst_rank_idx = __shfl_sync(scatter_group_mask, dst_rank_idx, scatter_group_leader);
        dst_token_idx = __shfl_sync(scatter_group_mask, dst_token_idx, scatter_group_leader);
        dst_topk_idx = __shfl_sync(scatter_group_mask, dst_topk_idx, scatter_group_leader);
        const auto dst_token =
            combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx);
        auto dst_ptr = math::advance_ptr<uint8_t>(
            dst_token.get_base_ptr(),
            n_idx * sizeof(nv_bfloat16) + lane_in_row * kScatterBytesPerLane);
        auto mapped_dst_ptr = sym_buffer.map(dst_ptr, dst_rank_idx);
        const auto packed = ptx::ld_shared(reinterpret_cast<const uint4*>(
            smem_cd_l2 + stage_row * kL2StageRowStride +
            lane_in_row * (kScatterBytesPerLane / sizeof(nv_bfloat16))));
        *reinterpret_cast<uint4*>(mapped_dst_ptr) = packed;
      }
    };

    // `row_offset_r0 - row_idx`: the first of this warp's 16 rows.
    const uint32_t warp_row_base = row_block_offset + warp_idx_in_wg * 16;

    // The staging tile is warp-private (producer and consumer are the same
    // 32 lanes), so `__syncwarp` is a sufficient shared-memory ordering
    // point and no named barrier index is consumed. The trailing
    // `kEpilogueFullBarrierIdx` sync also covers the write-after-read on
    // the tile across scheduled blocks.
    stage_rows(valid_r0, 0);
    __syncwarp();
    scatter_staged_rows(warp_row_base);
    __syncwarp();
    stage_rows(valid_r1, 2);
    __syncwarp();
    scatter_staged_rows(warp_row_base + kL2StageRows);
    ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
  });

  // ---------------- COMBINE ----------------
  // NVLink barrier first: signals remote ranks that this rank's GEMM
  // outputs (NVLink scatter targets) are fully written.
  comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads, kEpilogueGridSyncIndex,
                       kBeforeCombineReduceBarrierTag>(
      workspace, sym_buffer, sm_idx, epilogue_thread_idx,
      [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });

  // Rendezvous before workspace cleanup.
  ptx::sync_unaligned(kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

  constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
  constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);

  constexpr uint32_t kNumChunkSlots = 3;
  constexpr uint32_t kNumMaxRegistersForBuffer = 128;
  constexpr uint32_t kNumDefaultChunks =
      (kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and
       kHidden <= 32 * kNumMaxRegistersForBuffer)
          ? 1
          : 2;
  constexpr uint32_t kNumChunks = kNumDefaultChunks;
  constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
  constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
  constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32;
  FI_MEGAMOE_STATIC_ASSERT(kHidden % kNumChunks == 0,
                           "Hidden must be divisible by number of chunks");
  FI_MEGAMOE_STATIC_ASSERT(
      kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE,
      "Hidden is too large");
  FI_MEGAMOE_STATIC_ASSERT(kNumChunkBytes % 16 == 0,
                           "Combine chunk must be TMA-aligned (16 bytes)");
  FI_MEGAMOE_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0,
                           "Combine chunk must be divisible by 16 bytes");
  FI_MEGAMOE_STATIC_ASSERT(kNumChunkUint4 % 32 == 0,
                           "Combine chunk must be a multiple of 32 16-byte elements");
  FI_MEGAMOE_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

  FI_MEGAMOE_DEVICE_ASSERT(
      kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <=
      static_cast<uint32_t>(reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

#include <flashinfer/megamoe/detail/combine.inl>

  finish_no_dispatch_cleanup();
}
#else
if (blockIdx.x == 0 and threadIdx.x == 0)
  FI_MEGAMOE_DEVICE_ASSERT(false and "This kernel only supports sm_90");
#endif
