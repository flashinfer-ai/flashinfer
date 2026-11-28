// Node geometry and inter-node region handles shared by the dispatch and the
// combine of the MXFP4 fused kernel (included once at body level after the
// buffer set). With one node the values are unused.
const uint32_t my_node = sym_buffer.rank_idx / kNumLocalRanks;
constexpr uint32_t kNumExpertsPerNode = kNumExpertsPerRank * kNumLocalRanks;
const auto node_of_expert = [&](const uint32_t& expert_idx) {
  return expert_idx / kNumExpertsPerNode;
};
// Index of `node` among this rank's remote nodes, and back.
const auto remote_index_of_node = [&](const uint32_t& node) {
  return node < my_node ? node : node - 1;
};
const auto node_of_remote_index = [&](const uint32_t& j) { return j < my_node ? j : j + 1; };
const auto& remote = buffers.remote;
const uint32_t launch_epoch = kNumRemoteNodes > 0 ? ptx::ld_volatile(remote.launch_epoch()) : 0u;

// Byte offsets of the four input buffers inside the input block; the inbox
// repeats them.
const auto input_block_base = reinterpret_cast<int64_t>(input_token_buffer.get_base_ptr());
const int64_t sf_block_offset =
    reinterpret_cast<int64_t>(input_sf_buffer.get_base_ptr()) - input_block_base;
const int64_t idx_block_offset =
    reinterpret_cast<int64_t>(input_topk_idx_buffer.get_base_ptr()) - input_block_base;
const int64_t weights_block_offset =
    reinterpret_cast<int64_t>(input_topk_weights_buffer.get_base_ptr()) - input_block_base;
constexpr uint32_t kSFBytes = kHidden / 32;
constexpr uint32_t kIdxBytes = kNumTopk * sizeof(int64_t);
constexpr uint32_t kWeightBytes = kNumTopk * sizeof(float);
