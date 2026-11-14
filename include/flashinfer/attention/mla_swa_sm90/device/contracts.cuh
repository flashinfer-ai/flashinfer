
template <class C>
struct PVRoleOps;
template <class T>
struct SplitOutputTraits : T {
  using InputT = std::conditional_t<T::Config::Backend == mla_swa::Reduce::GlobalFP32, float,
                                    typename T::InputT>;
};
template <typename T>
struct DsmDirectAccumLayout {
  static constexpr int kPvGroups = T::Config::PvWG;
  static constexpr int kSlicesPerPvGroup = mla_swa::PVOwnership<typename T::Config>::MaxSlices;
  static constexpr int kPayloadWordsPerThreadPerSlice = 4;
  static constexpr int kThreadsPerWarpgroup = 128;
  static constexpr int kMailboxWords =
      kPvGroups * kSlicesPerPvGroup * kPayloadWordsPerThreadPerSlice * kThreadsPerWarpgroup;
  static constexpr int kRowsPerOwner = T::BLOCK_M / 2;
  static constexpr int kStateFloats =
      kRowsPerOwner * ((T::Config::PartialFormat == mla_swa::PartialForm::Raw) ? 2 : 1);
};
struct PVTask {
  const int* block_table;
  int batch_idx, m_block_idx, total_rows, s_q, num_heads, max_pages, cache_len;
  int start_page, end_page, window_size;
  float softmax_scale;
  int warpgroup, idx_in_warpgroup, warp_in_group, lane;
  int64_t block_table_stride_b, out_stride_b, out_stride_s, out_stride_h;
};
