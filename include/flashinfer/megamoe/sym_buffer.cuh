#pragma once

#include <flashinfer/megamoe/detail/common.cuh>

namespace flashinfer::megamoe::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
  int64_t base;
  int64_t offsets[kNumMaxRanks];
  uint32_t rank_idx;

  FI_MEGAMOE_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

  SymBuffer() = default;

  template <typename Container>
  explicit SymBuffer(const Container& c, const uint32_t& rank_idx) : rank_idx(rank_idx) {
    const auto size = static_cast<uint32_t>(c.size());
    base = c[rank_idx];
    for (uint32_t i = 0; i < kNumMaxRanks; ++i) offsets[i] = i < size ? (c[i] - base) : 0;
  }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
  template <typename ptr_t = void*>
  CUTLASS_DEVICE ptr_t get_base_ptr() const {
    return reinterpret_cast<ptr_t>(base);
  }

  template <typename ptr_t>
  CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
    int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
    return *reinterpret_cast<ptr_t*>(&mapped_ptr);
  }
#endif
};

// Per-rank offsets of the two regions a peer touches besides the workspace: the
// input rows / scales / routing a puller reads from a source rank and the combine
// rows an owner stores into a source rank. Inside one NVLink domain both are the
// peer's own buffer, so the tables equal `SymBuffer::offsets`; a rank on another
// node is reached through the inbox and outbox of its mirror rank on this node,
// whose offsets differ per region. Kept apart from `SymBuffer` so the other
// kernels and the EPLB arena keep their parameter size.
template <uint32_t kNumRanks = kNumMaxRanks>
struct SymRegions {
  int64_t input_offsets[kNumMaxRanks];
  int64_t combine_offsets[kNumMaxRanks];

  FI_MEGAMOE_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

  SymRegions() = default;

  // `input` and `combine` hold, per rank, the address this rank uses for that
  // peer's input block and combine block; the bases are this rank's own blocks.
  template <typename Container>
  SymRegions(const Container& input, const Container& combine, const int64_t& input_base,
             const int64_t& combine_base) {
    const auto size = static_cast<uint32_t>(input.size());
    for (uint32_t i = 0; i < kNumMaxRanks; ++i) {
      input_offsets[i] = i < size ? (input[i] - input_base) : 0;
      combine_offsets[i] = i < size ? (combine[i] - combine_base) : 0;
    }
  }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
  template <typename ptr_t>
  CUTLASS_DEVICE ptr_t map_input(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
    int64_t mapped_ptr = input_offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
    return *reinterpret_cast<ptr_t*>(&mapped_ptr);
  }

  template <typename ptr_t>
  CUTLASS_DEVICE ptr_t map_combine(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
    int64_t mapped_ptr = combine_offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
    return *reinterpret_cast<ptr_t*>(&mapped_ptr);
  }
#endif
};

}  // namespace flashinfer::megamoe::layout
