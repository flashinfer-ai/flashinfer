constexpr bool kReturnRowCTA = kWeightTileBytes != 8704;

// Inter-node part of the combine (MXFP4 fused kernel), included inside ROLE 3
// after the node-local combine barrier. The mirror duty (reduce the outbox rows
// of the mirrored source and put them back) ran on the dispatch warps during
// the GEMM (`remote_early_return` in detail/remote_dispatch.inl); here:
//
//   remote_wait_return source duty: wait for the return rows of every remote
//                      node (after the node-local reduction, which hides what
//                      is left of the return latency).
//   remote_add_returns  source duty: add the returned rows into the output.
//
// The node-local reduction (detail/combine.inl) masks out the slots whose
// experts are on other nodes. With one node the helpers compile to nothing.

const auto remote_wait_return = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if ((kM4M8Stripe8ReturnLocalJoin ? (sm_idx < num_tokens && sm_idx < kNumSMs) : sm_idx == 0) &&
        epilogue_thread_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j)
        io.wait(j, remote.return_flag(j), launch_epoch + 1);
    }
    if constexpr (kReturnRowCTA) {
      // Full completion, not .read: all first-pass y writes complete before
      // the original pre-grid CTA barrier and GPU release/acquire grid.
      if (cute::elect_one_sync()) ptx::tma_store_wait<0>();
      __syncwarp();
    }
    if constexpr (kM4M8Stripe8ReturnLocalJoin) {
      // Full TMA completion and this CTA join cover the cross-warp row ownership.
      if (sm_idx < num_tokens && sm_idx < kNumSMs)
        ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
      return;
    }
    // M1 keeps every first-pass y store and return consumer in CTA0. Preserve the
    // flag wait and each warp's full TMA completion above; relay them across
    // CTA0's math warps without joining unrelated CTAs.
    if constexpr (kWeightTileBytes == 8448 && kNumSMs == 78 && kNumRanks == 16 && kNumNodes == 2 &&
                  kNumLocalRanks == 8 && kNumExperts == 256 && kHidden == 4096 &&
                  kIntermediateHidden == 2048 && kNumTopk == 6 && kNumMaxTokensPerRank == 2 &&
                  BLOCK_M == 8 && BLOCK_N == 256 && BLOCK_K == 128 && kNumStages == 4 &&
                  kNumExpertsPerWave == 16 && kNumThreads == 384 && kNumEpilogueThreads == 256 &&
                  kNumDecodeThreads == 0 && kSwapABRequested && kUseRSOperand &&
                  kSingleActiveDispatchWarp && kUseMode2RowDecoder && kUseInterleavedScheduler &&
                  !kInKernelWeightCopy && kActivationClamp == 10.0f && kFastMath) {
      if (num_tokens == 1) {
        if (sm_idx == 0) ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
        return;
      }
    }
    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(workspace, sm_idx, epilogue_thread_idx, [&]() {
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    });
  }
};

// Prepared formats assign each row to one CTA after all first-pass TMA stores
// complete and the grid synchronizes. Canonical weights retain same-warp row
// ownership and wait for those stores after the grid. Both use the same
// ordered per-column BF16/FP32 accumulation.
const auto remote_add_returns = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kUint4PerRow = kRowBytes / sizeof(uint4);
    if constexpr (!kReturnRowCTA) {
      if (cute::elect_one_sync()) ptx::tma_store_wait<0>();
      __syncwarp();
    }
    const uint32_t first_token =
        kReturnRowCTA ? sm_idx : sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
    constexpr uint32_t token_stride = kReturnRowCTA ? kNumSMs : kNumSMs * kNumEpilogueWarps;
    for (uint32_t token_idx = first_token; token_idx < num_tokens; token_idx += token_stride) {
      uint32_t mask = 0;
      int32_t slots[kNumRemoteNodes];
#pragma unroll
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
        slots[j] = remote.packed_slot_map(j)[token_idx];
        if (slots[j] >= 0) mask |= 1u << j;
      }
      if (mask == 0) continue;
      auto* row = reinterpret_cast<uint4*>(
          math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kRowBytes));
      const uint32_t first_col = kReturnRowCTA ? epilogue_warp_idx * 32u + lane_idx : lane_idx;
      constexpr uint32_t col_stride = kReturnRowCTA ? 256u : 32u;
      for (uint32_t c = first_col; c < kUint4PerRow; c += col_stride) {
        uint4 v = row[c];
        auto* pairs = reinterpret_cast<nv_bfloat162*>(&v);
        float2 acc[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) acc[q] = __bfloat1622float2(pairs[q]);
        uint32_t m = mask;
        while (m) {
          const uint32_t j = __ffs(m) - 1;
          m ^= 1u << j;
          const auto r =
              reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(remote.return_inbox(j)) +
                                             static_cast<uint64_t>(slots[j]) * kRowBytes)[c];
          const auto* rp = reinterpret_cast<const nv_bfloat162*>(&r);
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q) ptx::accumulate(acc[q], rp[q]);
        }
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) pairs[q] = __float22bfloat162_rn(acc[q]);
        row[c] = v;
      }
      __syncwarp();
    }
  }
};
