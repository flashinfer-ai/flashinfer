#pragma once

template <typename T, typename TmaParamsT, typename TensorSKV, typename TensorSP, typename TensorSV,
          typename Plan>
__forceinline__ __device__ void run_pv_role(
    TensorSKV& sKV, TensorSP const& sP, TensorSV const& sV, Plan& plan, const int* block_table,
    typename T::InputT* out, int batch_idx, int m_block_idx, int total_rows, int s_q, int num_heads,
    int max_pages, int cache_len, int start_page, int end_page, int window_size,
    float softmax_scale, int warpgroup, int idx_in_warpgroup, int warp_in_group, int lane,
    int64_t block_table_stride_b, int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    TmaParamsT const& tma_params) {
  using C = typename T::Config;
  using Ownership = mla_swa::PVOwnership<C>;
  constexpr int MaxSlices = Ownership::MaxSlices;
  const int pv_group = warpgroup - 1;
  const int slice_begin = Ownership::begin(pv_group);
  const int slice_end = Ownership::end(pv_group);
  constexpr int kSliceStride = Ownership::Stride;
  Tensor output_fragment_all = partition_fragment_C(
      typename T::TiledMMAPV{}, Shape<Int<MaxSlices * T::TILE_K>, Int<T::BLOCK_M>>{});
  cute::fill(output_fragment_all, 0.0f);
  const PVTask task{block_table,
                    batch_idx,
                    m_block_idx,
                    total_rows,
                    s_q,
                    num_heads,
                    max_pages,
                    cache_len,
                    start_page,
                    end_page,
                    window_size,
                    softmax_scale,
                    warpgroup,
                    idx_in_warpgroup,
                    warp_in_group,
                    lane,
                    block_table_stride_b,
                    out_stride_b,
                    out_stride_s,
                    out_stride_h};
  bool p_ready_phase = 1;
  for (int page_idx : mla_swa::PageRange<C>{start_page, end_page}) {
    PVRoleOps<C>::template step<MaxSlices, T>(output_fragment_all, slice_begin, slice_end,
                                              kSliceStride, sKV, sP, sV, plan, task, pv_group,
                                              page_idx, p_ready_phase, tma_params);
  }
  PVRoleOps<C>::template finish<MaxSlices, T>(output_fragment_all, slice_begin, slice_end,
                                              kSliceStride, plan, out, task, pv_group);
}
