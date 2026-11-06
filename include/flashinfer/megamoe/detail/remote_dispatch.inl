// Inter-node part of the dispatch warps (MXFP4 fused kernel). Included inside
// ROLE 1 after the per-expert staking pass; every name below is a body local.
// With one node this file compiles to nothing.
//
//   remote_pack_tokens        one pass over this rank's routing: a token with an
//                             expert on another node takes a packed slot per
//                             such node and is copied once into the outbound
//                             staging of that node (row, scales, ids, weights).
//   remote_list_entry         the index pass writes `slot * topk + k` for an
//                             expert on another node into the outbound header.
//   remote_send               SM 0, right after the grid sync: puts the rows,
//                             then scales, routing, weights and the header
//                             (counts and lists) in one signalled put.
//   remote_publish_by_proxy   every SM, before the node-local barrier: waits for
//                             the inbox of every remote node and publishes its
//                             share of the experts into this node's owners.
//   remote_early_return       every SM, after the token pull: the mirror duty of
//                             the combine, run while the GEMM still works on the
//                             blocks that need no return.
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
            if (lane_idx == 0)
              slot = static_cast<int32_t>(atomicAdd(remote.packed_token_count(j), 1u));
            slot = __shfl_sync(0xffffffff, slot, 0);
            // Copy the token into the outbound staging of node `node`.
            auto* out_block = static_cast<uint8_t*>(remote.outbound_input_block(j));
            const auto* src_row =
                input_token_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>();
            comm::copy_16b_warp(out_block + static_cast<uint64_t>(slot) * kHidden, src_row, kHidden,
                                lane_idx);
            comm::copy_16b_warp(
                out_block + sf_block_offset + static_cast<uint64_t>(slot) * kSFBytes,
                input_sf_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>(), kSFBytes,
                lane_idx);
            if (lane_idx < kNumTopk) {
              reinterpret_cast<int64_t*>(out_block + idx_block_offset)[slot * kNumTopk + lane_idx] =
                  __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk +
                        lane_idx);
              reinterpret_cast<float*>(out_block +
                                       weights_block_offset)[slot * kNumTopk + lane_idx] =
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

// Index pass entry for an expert on another node: the owner's list in the
// outbound header gets the packed index of the token.
const auto remote_list_entry = [&](const uint32_t& token_topk_idx, const uint32_t& expert_idx,
                                   const uint32_t& dst_slot_idx) {
  if constexpr (kNumRemoteNodes > 0) {
    const uint32_t node = node_of_expert(expert_idx);
    const uint32_t j = remote_index_of_node(node);
    const uint32_t token_idx = token_topk_idx / kNumTopk;
    const int32_t slot = remote.packed_slot_map(j)[token_idx];
    remote.outbound_list(j, expert_idx - node * kNumExpertsPerNode)[dst_slot_idx] =
        static_cast<uint32_t>(slot) * kNumTopk + token_topk_idx % kNumTopk;
  }
};

// SM 0, warp 0, right after the index-pass grid sync (every SM's packed rows
// and list entries are complete): two puts per remote node. The rows leave
// first; the scales, routing, weights and the header (the per-expert counts
// and the lists the index pass wrote in place) are adjacent in the layout and
// leave as one put that carries the signal, which orders the rows put before
// it on the same context. The lists are not compacted: at a capacity of 512
// tokens they are 32 KB per node, under a microsecond on the wire, and the
// compaction kept the header put behind the count publication.
const auto remote_send = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx != 0 or warp_idx != 0) return;
    comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      const uint32_t node = node_of_remote_index(j);
      const uint32_t at = remote_links.peers[j].node_at_mirror;
      auto* header = remote.outbound_header(j);
      auto* counts = remote.header_expert_counts(header);
      auto* dst_block = static_cast<uint8_t*>(remote.inbox_input_block(at));
      auto* src_block = static_cast<uint8_t*>(remote.outbound_input_block(j));
      const uint32_t num_packed = *remote.packed_token_count(j);
      if (lane_idx == 0) remote.header_control(header)[0] = num_packed;
      for (uint32_t e = lane_idx; e < kNumExpertsPerNode; e += 32)
        counts[e] = static_cast<uint32_t>(
            *workspace.get_expert_send_count_ptr(node * kNumExpertsPerNode + e) & 0xffffffffu);
      __syncwarp();
      if (num_packed > 0)
        io.put_warp(j, dst_block, src_block, static_cast<uint64_t>(num_packed) * kHidden, lane_idx,
                    false, nullptr, 0);
      // Scales, routing ids, weights (whole regions) and the header, contiguous.
      const uint64_t tail =
          static_cast<uint64_t>(remote.input_block_bytes) - sf_block_offset + remote.header_bytes();
      io.put_warp(j, dst_block + sf_block_offset, src_block + sf_block_offset, tail, lane_idx, true,
                  remote.inbox_flag(at), comm::remote_inbox_signal(at));
    }
  }
};

// SM 0 before the node-local dispatch barrier: the mirror's publication for
// every remote source into this node's owners, from the inbox header.
// The experts of this node are spread over the SMs (on one SM the publication
// took 35-40 us, a chain of peer stores issued by two warps); every SM waits
// for the inbox signal itself, and the node-local barrier that follows orders
// all of these peer writes before the owners read them, exactly as it orders
// the local count publication and the epilogue's peer stores.
const auto remote_publish_by_proxy = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      const uint32_t source_rank = remote_links.peers[j].rank;
      if (thread_idx == 0)
        io.wait(j, remote.inbox_flag(j), comm::remote_inbox_signal(j), launch_epoch + 1);
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      if (thread_idx == 0) stamp_max(11);
      auto* header = remote.inbox_header(j);
      const auto* counts = remote.header_expert_counts(header);
      // Per expert: one thread writes the count as the source would have, every
      // thread copies a share of the list.
      for (uint32_t e = sm_idx; e < kNumExpertsPerNode; e += kNumSMs) {
        const uint32_t expert_idx = my_node * kNumExpertsPerNode + e;
        const uint32_t owner = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert = expert_idx % kNumExpertsPerRank;
        const uint32_t count = counts[e];
        if (thread_idx == 0) {
          *sym_buffer.map(workspace.get_expert_recv_count_ptr(source_rank, local_expert), owner) =
              static_cast<uint64_t>(count);
          // Release: the task producer reads the per-source counts after one
          // acquire of the complete sum (`fetch_remote_rows`).
          ptx::red_add_rel_sys(
              sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(local_expert), owner),
              (static_cast<uint64_t>(kNumSMs) << 32) | count);
        }
        const auto* list = remote.header_list(header, e);
        for (uint32_t k = thread_idx; k < count; k += kNumDispatchThreads)
          *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(local_expert, source_rank, k),
                          owner) = list[k];
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    }
  }
};

// Mirror duty, ahead of the combine barrier. The rows pulled from other nodes
// sit first within every expert and the scheduler claims their blocks first,
// so their FC2 tasks are done while the blocks of this node's own rows still
// run; every such task releases one count. SM 0 waits for this rank's count and
// tells the ranks of the node; once every rank of the node has reported, the
// outbox rows of this rank's packed tokens are complete, and the dispatch warps
// (idle until the cleanup) reduce them over the top-k slots whose experts live
// on this node and put the BF16 rows into the source's return inbox, hidden
// behind the remaining GEMM tasks. The counter and the flag are reset in the
// cleanup: every arrival precedes the mirror's read of the flag, which precedes
// its put and its cleanup.
const auto remote_early_return = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx == 0 and warp_idx == 0) {
      // Remote-dependent FC2 tasks of this rank: the first ceil(remote rows /
      // BLOCK_M) blocks of every local expert, one task per N block.
      uint32_t num_blocks = 0;
      for (uint32_t e = lane_idx; e < kNumExpertsPerRank; e += 32) {
        uint32_t rows = 0;
        for (uint32_t r = 0; r < kNumRanks; ++r) {
          if (r / kNumLocalRanks != my_node)
            rows += static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(r, e));
        }
        num_blocks += math::ceil_div(rows, BLOCK_M);
      }
      const uint32_t expected = __reduce_add_sync(0xffffffff, num_blocks) * kNumRoutedL2BlockNs;
      if (lane_idx == 0) {
        while (ptx::ld_acq(workspace.get_remote_l2_done_ptr()) < expected) __nanosleep(256);
        stamp_max(2);
        const uint32_t local_rank_base = my_node * kNumLocalRanks;
        for (uint32_t r = 0; r < kNumLocalRanks; ++r)
          ptx::red_add_rel_sys(
              sym_buffer.map(workspace.get_remote_rows_ready_ptr(), local_rank_base + r), 1);
      }
      __syncwarp();
    }
    // Every rank of the node has stored its FC2 rows of this rank's packed tokens.
    if (thread_idx == 0) {
      while (ptx::ld_acq_sys(workspace.get_remote_rows_ready_ptr()) <
             static_cast<int>(kNumLocalRanks))
        __nanosleep(256);
      stamp_max(28);
    }
    ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    // One acquire per thread orders its outbox loads after the arrivals.
    ptx::ld_acq_sys(workspace.get_remote_rows_ready_ptr());

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
      for (uint32_t p = sm_idx * kNumDispatchWarps + warp_idx; p < num_packed;
           p += kNumSMs * kNumDispatchWarps) {
        uint32_t mask = 0;
        if (lane_idx < kNumTopk) {
          const int64_t expert_idx = inbox_idx[p * kNumTopk + lane_idx];
          mask = (expert_idx >= 0 and node_of_expert(static_cast<uint32_t>(expert_idx)) == my_node)
                     ? 1u
                     : 0u;
        }
        mask = __ballot_sync(0xffffffff, mask);
        for (uint32_t c = lane_idx; c < kUint4PerRow; c += 32) {
          // All slot rows of this column are loaded before any is summed, so the
          // k loads overlap instead of paying one memory latency each.
          uint4 v[kNumTopk];
#pragma unroll
          for (uint32_t k = 0; k < kNumTopk; ++k) {
            if (mask & (1u << k))
              // Combine layout: [top-k slot][packed token][row].
              v[k] = reinterpret_cast<const uint4*>(
                  outbox + (static_cast<uint64_t>(k) * remote.capacity + p) * kRowBytes)[c];
          }
          float2 acc[4] = {};
#pragma unroll
          for (uint32_t k = 0; k < kNumTopk; ++k) {
            if (mask & (1u << k)) {
              const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&v[k]);
#pragma unroll
              for (uint32_t q = 0; q < 4; ++q) ptx::accumulate(acc[q], pairs[q]);
            }
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
    if (thread_idx == 0) stamp_max(16);
    // Every return row of this rank is written before SM 0 ships them.
    comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(workspace, sm_idx, thread_idx, [=]() {
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    });
    if (sm_idx == 0 and thread_idx == 0) stamp_max(17);
    if (sm_idx == 0 and warp_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
        const uint32_t num_packed = remote.header_control(remote.inbox_header(j))[0];
        const uint32_t at = remote_links.peers[j].node_at_mirror;
        io.put_warp(j, remote.return_inbox(at), remote.return_outbound(j),
                    static_cast<uint64_t>(cute::max(num_packed, 1u)) * kRowBytes, lane_idx, true,
                    remote.return_flag(at), comm::remote_return_signal(at));
      }
      if (lane_idx == 0) stamp_max(13);
    }
  }
};
