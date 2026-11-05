// Inter-node part of the combine (MXFP4 fused kernel), included inside ROLE 3
// after the node-local combine barrier. Two steps, in this order on every rank
// so that no two ranks wait on each other:
//
//   remote_return      mirror duty: for every packed token of the remote source
//                      this rank mirrors, reduce the FC2 rows that this node's
//                      owners stored in the outbox over the top-k slots whose
//                      experts live on this node, and put the BF16 rows into the
//                      source's return inbox with a signal.
//   remote_wait_return source duty: wait for the return rows of every remote
//                      node before the final reduction adds them.
//
// The final reduction itself (detail/combine.inl) masks out the slots whose
// experts are on other nodes and reads the returned rows as extra sources.
// With one node both helpers compile to nothing.

const auto remote_return = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    constexpr uint32_t kRowBytes = kHidden * sizeof(nv_bfloat16);
    constexpr uint32_t kUint4PerRow = kRowBytes / sizeof(uint4);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      auto* header = remote.inbox_header(j);
      const uint32_t num_packed = remote.header_control(header)[0];
      const auto* inbox_idx = reinterpret_cast<const int64_t*>(
          static_cast<uint8_t*>(remote.inbox_input_block(j)) + idx_block_offset);
      auto* outbox = static_cast<const uint8_t*>(remote.outbox(j));
      auto* out_rows = static_cast<uint8_t*>(remote.return_outbound(j));
      // One warp per packed token; lanes over 16-byte columns of the row.
      for (uint32_t p = sm_idx * kNumEpilogueWarps + epilogue_warp_idx; p < num_packed;
           p += kNumSMs * kNumEpilogueWarps) {
        uint32_t mask = 0;
        if (lane_idx < kNumTopk) {
          const int64_t expert_idx = inbox_idx[p * kNumTopk + lane_idx];
          mask = (expert_idx >= 0 and node_of_expert(static_cast<uint32_t>(expert_idx)) == my_node)
                     ? 1u
                     : 0u;
        }
        mask = __ballot_sync(0xffffffff, mask);
        for (uint32_t c = lane_idx; c < kUint4PerRow; c += 32) {
          float2 acc[4] = {};
          uint32_t m = mask;
          while (m) {
            const uint32_t k = __ffs(m) - 1;
            m ^= 1u << k;
            // Combine layout: [top-k slot][packed token][row].
            const auto v = reinterpret_cast<const uint4*>(
                outbox + (static_cast<uint64_t>(k) * remote.capacity + p) * kRowBytes)[c];
            const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&v);
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) ptx::accumulate(acc[q], pairs[q]);
          }
          uint4 packed;
          auto* out_pairs = reinterpret_cast<nv_bfloat162*>(&packed);
#pragma unroll
          for (uint32_t q = 0; q < 4; ++q) out_pairs[q] = __float22bfloat162_rn(acc[q]);
          reinterpret_cast<uint4*>(out_rows + static_cast<uint64_t>(p) * kRowBytes)[c] = packed;
        }
        __syncwarp();
      }
    }
    // Every return row of this rank is written before SM 0 ships them.
    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(workspace, sm_idx, epilogue_thread_idx, [&]() {
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    });
    if (sm_idx == 0 and epilogue_warp_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
        const uint32_t num_packed = remote.header_control(remote.inbox_header(j))[0];
        const uint32_t at = remote_links.peers[j].node_at_mirror;
        io.put_warp(j, remote.return_inbox(at), remote.return_outbound(j),
                    static_cast<uint64_t>(cute::max(num_packed, 1u)) * kRowBytes, lane_idx, true,
                    remote.return_flag(at), comm::remote_return_signal(at));
      }
    }
  }
};

const auto remote_wait_return = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx == 0 and epilogue_thread_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j)
        io.wait(j, remote.return_flag(j), comm::remote_return_signal(j), launch_epoch + 1);
    }
    comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(workspace, sm_idx, epilogue_thread_idx, [&]() {
      ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
    });
  }
};
