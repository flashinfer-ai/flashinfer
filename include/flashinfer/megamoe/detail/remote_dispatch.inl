// Inter-node part of the dispatch (MXFP4 fused kernel). Included at body level
// before the warp roles; every name below is a body local. With one node this
// file compiles to nothing.
//
//   remote_pack_tokens        the math warps, from kernel entry: one pass over
//                             this SM's share of the routing, a token with an
//                             expert on another node takes a packed slot per
//                             such node and is copied once into the outbound
//                             staging of that node (row, scales, ids, weights).
//   remote_list_entry         the index pass writes `slot * topk + k` for an
//                             expert on another node into the outbound header.
//   remote_send               SM 0, right after the grid sync: puts the rows,
//                             then scales, routing, weights and the header
//                             (counts and lists) in one signalled put.
//   remote_publish_by_proxy   every SM, after the local pull pass: waits for the
//                             inbox of every remote node and publishes its share
//                             of the experts into this node's owners (lists,
//                             then the count with a release on the remote sum).
//   remote_early_return       every SM, after the token pull: the mirror duty of
//                             the combine, run while the GEMM still works on the
//                             blocks that need no return.
// Pass over the routing: packed slots and the outbound copies. Mirrors the
// iteration of `read_topk_idx` (`kNumTokensPerWarp` tokens per warp step, the
// top-k entries of one token on consecutive lanes); the caller hands the warp
// the chunks (`first_token`, every `token_step`) that one dispatch warp of the
// same SM will index, so the index pass reads a slot map its own SM wrote.
const auto remote_pack_tokens = [&](const uint32_t& first_token, const uint32_t& token_step) {
  if constexpr (kNumRemoteNodes > 0) {
    constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
    constexpr uint32_t kTokenMask = (1u << kNumTopk) - 1;
    for (uint32_t i = first_token; i < num_tokens; i += token_step) {
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
            if constexpr (kSFBytes % sizeof(uint4) == 0) {
              comm::copy_16b_warp(
                  out_block + sf_block_offset + static_cast<uint64_t>(slot) * kSFBytes,
                  input_sf_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>(), kSFBytes,
                  lane_idx);
            } else {
              auto* dst_sf = reinterpret_cast<uint32_t*>(out_block + sf_block_offset +
                                                         static_cast<uint64_t>(slot) * kSFBytes);
              const auto* src_sf =
                  input_sf_buffer.get_data_buffer(token_idx).get_base_ptr<uint32_t>();
              for (uint32_t s = lane_idx; s < kSFBytes / sizeof(uint32_t); s += 32)
                dst_sf[s] = src_sf[s];
            }
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

// SM 0, warp 0, after the index-pass grid sync. M1 can send the padded input
// block and header in one put; other paths send rows before the signalled
// scales, routing, weights, and header.
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
      const bool single_input_put = kWeightTileBytes == 8448 && kNumRanks == 16 && kNumNodes == 2 &&
                                    kNumMaxTokensPerRank == 2 && num_tokens == 1 &&
                                    num_packed == 1 && !io.is_lsa(j);
      if (num_packed > 0 && !single_input_put)
        io.put_warp(j, dst_block, src_block, static_cast<uint64_t>(num_packed) * kHidden, lane_idx,
                    false, nullptr);
      if (lane_idx == 0) remote.header_control(header)[0] = num_packed;
      for (uint32_t e = lane_idx; e < kNumExpertsPerNode; e += 32)
        counts[e] = static_cast<uint32_t>(
            *workspace.get_expert_send_count_ptr(node * kNumExpertsPerNode + e) & 0xffffffffu);
      __syncwarp();
      // Scales, routing ids, weights (whole regions) and the header, contiguous.
      const uint64_t tail =
          static_cast<uint64_t>(remote.input_block_bytes) - sf_block_offset + remote.header_bytes();
      if (single_input_put) {
        const uint64_t bytes = remote.input_block_bytes + remote.header_bytes();
        io.put_warp(j, dst_block, src_block, bytes, lane_idx, true, remote.inbox_flag(at));
      } else {
        io.put_warp(j, dst_block + sf_block_offset, src_block + sf_block_offset, tail, lane_idx,
                    true, remote.inbox_flag(at));
      }
    }
  }
};

// Every SM, after the local pull pass: the mirror's publication for every
// remote source into this node's owners, from the inbox header. The experts of
// this node are spread over the SMs, and each SM waits for the inbox signal.
// Per expert the list goes first, then, after a CTA barrier,
// one thread stores the count and adds it to the owner's remote count sum with
// a sys-scope release: the owner's task producer and its second pull pass
// acquire that sum, which orders the list and the count behind it.
const auto remote_publish_by_proxy = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
    for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
      const uint32_t source_rank = remote_links.peers[j].rank;
      if (thread_idx == 0) io.wait(j, remote.inbox_flag(j), launch_epoch + 1);
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
      auto* header = remote.inbox_header(j);
      const auto* counts = remote.header_expert_counts(header);
      for (uint32_t e = sm_idx; e < kNumExpertsPerNode; e += kNumSMs) {
        const uint32_t expert_idx = my_node * kNumExpertsPerNode + e;
        const uint32_t owner = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert = expert_idx % kNumExpertsPerRank;
        const uint32_t count = counts[e];
        const auto* list = remote.header_list(header, e);
        for (uint32_t k = thread_idx; k < count; k += kNumDispatchThreads)
          *sym_buffer.map(workspace.get_src_token_topk_idx_ptr(local_expert, source_rank, k),
                          owner) = list[k];
        // Complete every publishing thread before the per-expert release.
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
        if (thread_idx == 0) {
          *sym_buffer.map(workspace.get_expert_recv_count_ptr(source_rank, local_expert), owner) =
              static_cast<uint64_t>(count);
          // Publish list and count before the release sum; consumers acquire
          // the completed sum before reading any owned expert list/count.
          ptx::red_add_rel_sys(
              sym_buffer.map(workspace.get_expert_recv_count_remote_sum_ptr(local_expert), owner),
              (static_cast<uint64_t>(kNumSMs) << 32) | count);
        }
      }
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    }
  }
};

// Mirror duty, ahead of the combine barrier. The rows pulled from other nodes
// sit in the remote segment of every expert and the scheduler claims those
// blocks before the last local blocks, so their FC2 tasks are done while
// blocks of this node's own rows still run; every such task releases one count. SM 0 waits for this
// rank's count and tells the ranks of the node; once every rank of the node has reported, the
// outbox rows of this rank's packed tokens are complete, and the dispatch warps
// (idle until the cleanup) reduce them over the top-k slots whose experts live
// on this node and put the BF16 rows into the source's return inbox, hidden
// behind the remaining GEMM tasks. The counter and the flag are reset in the
// cleanup: every arrival precedes the mirror's read of the flag, which precedes
// its put and its cleanup.
const auto remote_early_return = [&]() {
  if constexpr (kNumRemoteNodes > 0) {
    if (sm_idx == 0 and warp_idx == 0) {
      // Remote-dependent FC2 tasks of this rank: one per N block of every
      // remote-segment block (the dispatch warps laid out both waves).
      const uint32_t expected =
          interleaved_scheduler.num_remote_dependent_m_blocks * kNumRoutedL2BlockNs;
      if (lane_idx == 0) {
        while (ptx::ld_acq(workspace.get_remote_l2_done_ptr()) < expected) __nanosleep(256);
      }
      // Relay lane0's original acquire to every publisher in this full warp.
      // Each destination still receives exactly one original SYS release-add.
      __syncwarp();
      static_assert(kNumLocalRanks <= 32, "Node-local ranks must fit a warp");
      const uint32_t local_rank_base = my_node * kNumLocalRanks;
      if (lane_idx < kNumLocalRanks)
        ptx::red_add_rel_sys(
            sym_buffer.map(workspace.get_remote_rows_ready_ptr(), local_rank_base + lane_idx), 1);
      __syncwarp();
    }
    // Every rank of the node has stored its FC2 rows of this rank's packed tokens.
    if (thread_idx == 0) {
      while (ptx::ld_acq_sys(workspace.get_remote_rows_ready_ptr()) <
             static_cast<int>(kNumLocalRanks))
        __nanosleep(256);
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
      // Split returned-row columns across dispatch warps without changing
      // the ascending top-k accumulation order.
      constexpr uint32_t kReturnWarpsPerRow = 2;
      static_assert(kNumDispatchWarps % kReturnWarpsPerRow == 0);
      constexpr uint32_t kReturnRowsPerCTA = kNumDispatchWarps / kReturnWarpsPerRow;
      for (uint32_t p = sm_idx * kReturnRowsPerCTA + warp_idx / kReturnWarpsPerRow; p < num_packed;
           p += kNumSMs * kReturnRowsPerCTA) {
        uint32_t mask = 0;
        if (lane_idx < kNumTopk) {
          const int64_t expert_idx = inbox_idx[p * kNumTopk + lane_idx];
          mask = (expert_idx >= 0 and node_of_expert(static_cast<uint32_t>(expert_idx)) == my_node)
                     ? 1u
                     : 0u;
        }
        mask = __ballot_sync(0xffffffff, mask);
        // Expose independent column loads before using
        // either. Every output column retains the original ascending top-k
        // accumulation and BF16 rounding. Hidden4096 gives eight columns per
        // lane with two warps/row, so this batch introduces no padded loads.
        constexpr uint32_t kColumnStride = 32 * kReturnWarpsPerRow;
        constexpr uint32_t kColumnBatch = kUint4PerRow % (kColumnStride * 2) == 0 ? 2 : 1;
        static_assert(kUint4PerRow % 32 == 0);
        for (uint32_t c = (warp_idx % kReturnWarpsPerRow) * 32 + lane_idx; c < kUint4PerRow;
             c += kColumnStride * kColumnBatch) {
          uint4 values[kNumTopk][kColumnBatch];
#pragma unroll
          for (uint32_t k = 0; k < kNumTopk; ++k) {
            if (mask & (1u << k)) {
#pragma unroll
              for (uint32_t b = 0; b < kColumnBatch; ++b)
                values[k][b] = reinterpret_cast<const uint4*>(
                    outbox + (static_cast<uint64_t>(k) * remote.capacity + p) *
                                 kRowBytes)[c + b * kColumnStride];
            }
          }
#pragma unroll
          for (uint32_t b = 0; b < kColumnBatch; ++b) {
            float2 acc[4] = {};
#pragma unroll
            for (uint32_t k = 0; k < kNumTopk; ++k) {
              if (mask & (1u << k)) {
                const auto* pairs = reinterpret_cast<const nv_bfloat162*>(&values[k][b]);
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) ptx::accumulate(acc[q], pairs[q]);
              }
            }
            uint4 packed;
            auto* out_pairs = reinterpret_cast<nv_bfloat162*>(&packed);
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) out_pairs[q] = __float22bfloat162_rn(acc[q]);
            reinterpret_cast<uint4*>(out_rows + static_cast<uint64_t>(p) *
                                                    kRowBytes)[c + b * kColumnStride] = packed;
          }
        }
        __syncwarp();
      }
    }
    // Every return row of this rank is written before SM 0 ships them.
    comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(workspace, sm_idx, thread_idx, [=]() {
      ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
    });
    if (sm_idx == 0 and warp_idx == 0) {
      comm::RemoteIO<kNumRemoteNodes> io(remote_links);
#pragma unroll 1
      for (uint32_t j = 0; j < kNumRemoteNodes; ++j) {
        const uint32_t num_packed = remote.header_control(remote.inbox_header(j))[0];
        const uint32_t at = remote_links.peers[j].node_at_mirror;
        io.put_warp(j, remote.return_inbox(at), remote.return_outbound(j),
                    static_cast<uint64_t>(cute::max(num_packed, 1u)) * kRowBytes, lane_idx, true,
                    remote.return_flag(at));
      }
    }
  }
};
