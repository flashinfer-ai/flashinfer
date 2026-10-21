// Warp-wide round-robin selection from per-rank expert counts.
uint32_t current_rank_in_expert_idx;
uint32_t remaining[kNumRanksPerLane];
#pragma unroll
for (uint32_t i = 0; i < kNumRanksPerLane; ++i) remaining[i] = stored_rank_count[i];
uint32_t offset = 0;
uint32_t token_idx_in_expert = token_idx - expert_start_idx;
uint32_t slot_idx = token_idx_in_expert;
uint32_t token_idx_in_rank;
while (true) {
  uint32_t num_actives_in_lane = 0;
  uint32_t min_in_lane = 0xffffffff;
#pragma unroll
  for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
    num_actives_in_lane += remaining[i] > 0;
    if (remaining[i] > 0) min_in_lane = cute::min(min_in_lane, remaining[i]);
  }
  const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
  const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

  const uint32_t num_round_tokens = length * num_active_ranks;
  if (slot_idx < num_round_tokens) {
    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
    uint32_t num_seen_ranks = 0;
    current_rank_in_expert_idx = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumRanksPerLane; ++i) {
      const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
      const uint32_t num_active_lanes = __popc(mask);
      if (slot_idx_in_round >= num_seen_ranks and
          slot_idx_in_round < num_seen_ranks + num_active_lanes)
        current_rank_in_expert_idx =
            i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
      num_seen_ranks += num_active_lanes;
    }
    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
    break;
  }
  slot_idx -= num_round_tokens;
  offset += length;
#pragma unroll
  for (uint32_t i = 0; i < kNumRanksPerLane; ++i) remaining[i] -= cute::min(remaining[i], length);
}
