// Shared host/device buffer ABI; shape constants remain visible to SMEM sizing.
constexpr auto fp8_token_layout = layout::Data(kHidden);
const auto buffers = layout::MegaMoEBuffers(
    sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk, kHidden,
    kIntermediateHidden, kNumMaxPoolTokens, kNumPaddedSFPoolTokens);
const auto workspace = buffers.workspace;
const auto input_token_buffer = buffers.input_token_buffer;
const auto input_sf_buffer = buffers.input_sf_buffer;
const auto input_topk_idx_buffer = buffers.input_topk_idx_buffer;
const auto input_topk_weights_buffer = buffers.input_topk_weights_buffer;
const auto l1_token_buffer = buffers.l1_token_buffer;
const auto l1_sf_buffer = buffers.l1_sf_buffer;
const auto l1_topk_weights_buffer = buffers.l1_topk_weights_buffer;
const auto l2_token_buffer = buffers.l2_token_buffer;
const auto l2_sf_buffer = buffers.l2_sf_buffer;
const auto combine_token_buffer = buffers.combine_token_buffer;
