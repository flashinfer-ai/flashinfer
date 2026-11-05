// Inter-node part of the dispatch warps (MXFP4 fused kernel). Included inside
// ROLE 1 after the per-expert staking pass; every name below is a body local.
// With one node this file compiles to nothing.
//
//   remote_pack_tokens        one pass over this rank's routing: a token with an
//                             expert on another node takes a packed slot per
//                             such node and is copied once into the outbound
//                             staging of that node (row, scales, ids, weights).
//   remote_list_entry         the index pass writes `slot * topk + k` for an
//                             expert on another node into the list scratch.
//   remote_send               SM 0, after the grid sync: compacts the lists into
//                             the outbound header and puts header and rows to
//                             the mirror with a signal.
//   remote_publish_by_proxy   SM 0, before the node-local barrier: waits for the
//                             inbox of every remote node and performs the
//                             mirror's publication into this node's owners.
// Pass over the routing: packed slots and the outbound copies. Mirrors the
// iteration of `read_topk_idx`: `kNumTokensPerWarp` tokens per warp step, the
// top-k entries of one token on consecutive lanes.
const auto remote_pack_tokens = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (warp_idx >= kNumActiveDispatchWarps) return;
    constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
    constexpr uint32_t kTokenMask = (1u << kNumTopk) - 1;
    for (uint32_t i = (sm_idx * kNumActiveDispatchWarps + warp_idx) * kNumTokensPerWarp;
         i < num_tokens; i += kNumSMs * kNumActiveDispatchWarps * kNumTokensPerWarp) {
      int expert_idx = -1;
      const uint32_t token_in_warp = lane_idx / kNumTopk;
      if (i + token_in_warp < num_tokens and lane_idx < kNumActivateLanes)
        expert_idx = static_cast<int>(
            __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
        const uint32_t node = node_of_remote_index(j);
        const uint32_t remote_mask =
            __ballot_sync(0xffffffff, expert_idx >= 0 and node_of_expert(expert_idx) == node);
        for (uint32_t t = 0; t < kNumTokensPerWarp and i + t < num_tokens; ++t) {
          const uint32_t token_idx = i + t;
          int32_t slot = -1;
          if ((remote_mask >> (t * kNumTopk)) & kTokenMask) {
            if (lane_idx == 0) slot = static_cast<int32_t>(atomicAdd(remote.packed_token_count(j), 1u));
            slot = __shfl_sync(0xffffffff, slot, 0);
            // Copy the token into the outbound staging of node `node`.
            auto* out_block = static_cast<uint8_t*>(remote.outbound_input_block(j));
            const auto* src_row = input_token_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>();
            comm::copy_16b_warp(out_block + static_cast<uint64_t>(slot) * kHidden, src_row, kHidden,
                                lane_idx);
            comm::copy_16b_warp(out_block + sf_block_offset + static_cast<uint64_t>(slot) * kSFBytes,
                                input_sf_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>(),
                                kSFBytes, lane_idx);
            if (lane_idx < kNumTopk) {
              reinterpret_cast<int64_t*>(out_block + idx_block_offset)[slot * kNumTopk + lane_idx] =
                  __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk +
                        lane_idx);
              reinterpret_cast<float*>(out_block + weights_block_offset)[slot * kNumTopk + lane_idx] =
                  __ldg(input_topk_weights_buffer.get_base_ptr<float>() + token_idx * kNumTopk +
                        lane_idx);
            }
          }
          if (lane_idx == 0) remote.packed_slot_map(j)[token_idx] = slot;
          __syncwarp();
        }
      }
      __syncwarp();
    }
  }
};

// Index pass entry for an expert on another node: the owner's list gets the
// packed index of the token, staged in the list scratch until the totals are known.
const auto remote_list_entry = [&](const uint32_t& token_topk_idx, const uint32_t& expert_idx,
                                   const uint32_t& dst_slot_idx) {
  if constexpr (kNumRemoteNodes > 0) {
    const uint32_t node = node_of_expert(expert_idx);
    const uint32_t j = remote_index_of_node(node);
    const uint32_t token_idx = token_topk_idx / kNumTopk;
    const int32_t slot = remote.packed_slot_map(j)[token_idx];
    remote.list_scratch(j, expert_idx - node * kNumExpertsPerNode)[dst_slot_idx] =
        static_cast<uint32_t>(slot) * kNumTopk + token_topk_idx % kNumTopk;
  }
};

// SM 0 after the grid sync: compact header + puts, one remote node at a time.
const auto remote_send = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx != 0) return;
    comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      const uint32_t node = node_of_remote_index(j);
      auto* header = remote.outbound_header(j);
      auto* counts = remote.header_expert_counts(header);
      auto* starts = remote.header_expert_starts(header);
      auto* entries = remote.header_entries(header);
      for (uint32_t e = thread_idx; e < kNumExpertsPerNode; e += kNumDispatchThreads)
        counts[e] = static_cast<uint32_t>(
            *workspace.get_expert_send_count_ptr(node * kNumExpertsPerNode + e) & 0xffffffffu);
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      if (thread_idx == 0) {
        uint32_t running = 0;
        for (uint32_t e = 0; e < kNumExpertsPerNode; ++e) {
          starts[e] = running;
          running += counts[e];
        }
        remote.header_control(header)[0] = *remote.packed_token_count(j);
        remote.header_control(header)[1] = running;
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      const uint32_t num_entries = remote.header_control(header)[1];
      const uint32_t num_packed = remote.header_control(header)[0];
      // Compaction: one warp per expert, lanes over its entries.
      for (uint32_t e = warp_idx; e < kNumExpertsPerNode; e += kNumDispatchWarps) {
        const auto* src = remote.list_scratch(j, e);
        for (uint32_t k = lane_idx; k < counts[e]; k += 32) entries[starts[e] + k] = src[k];
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      if (warp_idx == 0) {
        const uint32_t at = remote_links.peers[j].node_at_mirror;
        auto* dst_header = remote.inbox_header(at);
        auto* dst_block = static_cast<uint8_t*>(remote.inbox_input_block(at));
        auto* src_block = static_cast<uint8_t*>(remote.outbound_input_block(j));
        io.put_warp(j, dst_header, header, remote.header_used_bytes(num_entries), lane_idx, false,
                    nullptr, 0);
        if (num_packed > 0) {
          io.put_warp(j, dst_block, src_block, static_cast<uint64_t>(num_packed) * kHidden, lane_idx,
                      false, nullptr, 0);
          io.put_warp(j, dst_block + sf_block_offset, src_block + sf_block_offset,
                      static_cast<uint64_t>(num_packed) * kSFBytes, lane_idx, false, nullptr, 0);
          io.put_warp(j, dst_block + idx_block_offset, src_block + idx_block_offset,
                      math::align<uint64_t>(static_cast<uint64_t>(num_packed) * kIdxBytes, 16),
                      lane_idx, false, nullptr, 0);
        }
        // The weights put carries the signal; with no packed token it is a
        // 16-byte put of the (unused) first weight slot so the signal still lands.
        io.put_warp(j, dst_block + weights_block_offset, src_block + weights_block_offset,
                    math::align<uint64_t>(static_cast<uint64_t>(cute::max(num_packed, 1u)) *
                                              kWeightBytes,
                                          16),
                    lane_idx, true, remote.inbox_flag(at), comm::remote_inbox_signal(at));
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    }
  }
};

// SM 0 before the node-local dispatch barrier: the mirror's publication for
// every remote source into this node's owners, from the inbox header.
const auto remote_publish_by_proxy = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx != 0) return;
    comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      const uint32_t source_rank = remote_links.peers[j].rank;
      if (thread_idx == 0)
        io.wait(j, remote.inbox_flag(j), comm::remote_inbox_signal(j), launch_epoch + 1);
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      auto* header = remote.inbox_header(j);
      const auto* counts = remote.header_expert_counts(header);
      const auto* starts = remote.header_expert_starts(header);
      const auto* entries = remote.header_entries(header);
      // Counts: one entry per expert of this node, as the source would have written.
      for (uint32_t e = thread_idx; e < kNumExpertsPerNode; e += kNumDispatchThreads) {
        const uint32_t expert_idx = my_node * kNumExpertsPerNode + e;
        const uint32_t owner = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert = expert_idx % kNumExpertsPerRank;
        const uint64_t count = counts[e];
        *sym_buffer.map(workspace.get_expert_recv_count_ptr(source_rank, local_expert), owner) =
            count;
        ptx::atomic_add_sys(
            sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(local_expert), owner),
            (static_cast<uint64_t>(kNumSMs) << 32) | count);
      }
      // Lists: one warp per expert.
      for (uint32_t e = warp_idx; e < kNumExpertsPerNode; e += kNumDispatchWarps) {
        const uint32_t expert_idx = my_node * kNumExpertsPerNode + e;
        const uint32_t owner = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert = expert_idx % kNumExpertsPerRank;
        for (uint32_t k = lane_idx; k < counts[e]; k += 32)
          *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(local_expert, source_rank, k),
                          owner) = entries[starts[e] + k];
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    }
  }
};
