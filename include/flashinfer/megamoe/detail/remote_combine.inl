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
    if (sm_idx == 0 and epilogue_thread_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j)
        io.wait(j, remote.return_flag(j), comm::remote_return_signal(j), launch_epoch + 1);
      stamp_max(14);
    }
    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(workspace, sm_idx, epilogue_thread_idx, [&]() {
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    });
  }
};

// Second combine pass: y[token] += returned row of every remote node that
// holds a slot for the token. Same token-to-warp map as the first pass, so
// each warp only touches rows it wrote itself.
const auto remote_add_returns = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kUint4PerRow = kRowBytes / sizeof(uint4);
    if (cute::elect_one_sync()) ptx::tma_store_wait<0>();
    __syncwarp();
    for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
         token_idx < num_tokens; token_idx += kNumSMs * kNumEpilogueWarps) {
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
      for (uint32_t c = lane_idx; c < kUint4PerRow; c += 32) {
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
