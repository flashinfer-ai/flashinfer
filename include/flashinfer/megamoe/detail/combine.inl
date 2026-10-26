// Shared top-k reduction: all three SM90 FP4 MegaMoE paths.
const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
  return math::advance_ptr<uint4>(smem_buffer,
                                  (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
});
const auto combine_store_buffer = math::advance_ptr<uint4>(
    smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

auto combine_load_barriers = utils::PatternVisitor(
    [&](const uint32_t& i) { return combine_barriers[i + epilogue_warp_idx * 2]; });

uint32_t combine_phase = 0;
uint32_t load_stage_idx = 0;
for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx; token_idx < num_tokens;
     token_idx += kNumSMs * kNumEpilogueWarps) {
  const int stored_topk_slot_idx =
      lane_idx < kNumTopk ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                                   token_idx * kNumTopk + lane_idx))
                          : -1;
  const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

  for (uint32_t chunk = 0; chunk < kNumChunks; ++chunk) {
    const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

    uint32_t mask = total_mask;
    const auto move_mask_and_load = [&](const uint32_t& i) {
      if (mask) {
        const uint32_t slot_idx = __ffs(mask) - 1;
        mask ^= 1 << slot_idx;
        if (cute::elect_one_sync()) {
          const auto src_ptr =
              math::advance_ptr<uint8_t>(combine_token_buffer.get_rank_buffer(slot_idx)
                                             .get_data_buffer(token_idx)
                                             .get_base_ptr(),
                                         chunk_byte_offset);
          ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i],
                           kNumChunkBytes);
          ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
        }
        __syncwarp();
        return true;
      }
      return false;
    };

    bool do_reduce = move_mask_and_load(load_stage_idx);

    float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
    while (do_reduce) {
      do_reduce = move_mask_and_load(load_stage_idx ^ 1);
      combine_load_barriers[load_stage_idx]->wait(combine_phase);
#pragma unroll
      for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
#pragma unroll
        for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)
          ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
      }
      combine_phase ^= load_stage_idx;
      load_stage_idx ^= 1;
    }

#pragma unroll
    for (uint32_t j = 0; j < kNumUint4PerLane; ++j) {
      uint4 casted;
      auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
#pragma unroll
      for (uint32_t l = 0; l < kNumElemsPerUint4; ++l)
        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

      if (j == 0) {
        ptx::tma_store_wait<0>();
        __syncwarp();
      }
      ptx::st_shared(combine_store_buffer + j * 32 + lane_idx, casted.x, casted.y, casted.z,
                     casted.w);
    }
    __syncwarp();

    if (cute::elect_one_sync()) {
      cute::tma_store_fence();
      ptx::tma_store_1d(math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes +
                                                 chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
      cute::tma_store_arrive();
    }
    __syncwarp();
  }
}
