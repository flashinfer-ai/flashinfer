#pragma once

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void softmax_one_row(TensorSP const& sP, Plan& plan, int row, int lane,
                                                int physical_page, int page_idx, int m_block_idx,
                                                int total_rows, int s_q, int num_heads,
                                                int cache_len, int window_size,
                                                float softmax_scale) {
  Tensor sScore =
      make_tensor(make_smem_ptr(plan.softmax_score.data()), typename T::SmemLayoutScore{});
  const int global_row = m_block_idx * T::BLOCK_M + row;
  const bool row_valid = physical_page >= 0 && global_row < total_rows;
  int q_pos = 0, left = 0;
  if (lane == 0 && row_valid) {
    q_pos = cache_len - s_q + global_row / num_heads;
    left = max(q_pos - window_size + 1, 0);
  }
  q_pos = __shfl_sync(0xffffffff, q_pos, 0);
  left = __shfl_sync(0xffffffff, left, 0);
  const int token0 = page_idx * T::BLOCK_N + lane;
  const int token1 = token0 + 32;
  const float scale = softmax_scale * 1.4426950408889634f;
  const float score0 =
      row_valid && token0 >= left && token0 <= q_pos ? sScore(row, lane) * scale : -CUDART_INF_F;
  const float score1 = row_valid && token1 >= left && token1 <= q_pos
                           ? sScore(row, lane + 32) * scale
                           : -CUDART_INF_F;
  float tile_max = fmaxf(score0, score1);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
  float new_max = -CUDART_INF_F, row_scale = 0.0f;
  if (lane == 0) {
    const float old_max = plan.row_max[row];
    new_max = fmaxf(old_max, tile_max);
    row_scale = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
    plan.row_scale[row] = row_scale;
    plan.row_max[row] = new_max;
  }
  new_max = __shfl_sync(0xffffffff, new_max, 0);
  row_scale = __shfl_sync(0xffffffff, row_scale, 0);
  const float p0 = isfinite(score0) ? exp2f(score0 - new_max) : 0.0f;
  if constexpr (T::Config::StoreP0First) sP(row, lane) = static_cast<typename T::InputT>(p0);
  const float p1 = isfinite(score1) ? exp2f(score1 - new_max) : 0.0f;
  if constexpr (!T::Config::StoreP0First) sP(row, lane) = static_cast<typename T::InputT>(p0);
  sP(row, lane + 32) = static_cast<typename T::InputT>(p1);
  float tile_sum = p0 + p1;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
  if (lane == 0) plan.row_sum[row] = plan.row_sum[row] * row_scale + tile_sum;
}

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void run_m32_p0_softmax_helper(TensorSP const& sP, Plan& plan,
                                                          int warpgroup, int warp_in_group,
                                                          int lane, int physical_page, int page_idx,
                                                          int m_block_idx, int total_rows, int s_q,
                                                          int num_heads, int cache_len,
                                                          int window_size, float softmax_scale) {
  const int warp = warpgroup * 4 + warp_in_group;
#pragma unroll
  for (int row_in_warp = 0; row_in_warp < 2; ++row_in_warp) {
    const int row = warp * 2 + row_in_warp;
    softmax_one_row<T>(sP, plan, row, lane, physical_page, page_idx, m_block_idx, total_rows, s_q,
                       num_heads, cache_len, window_size, softmax_scale);
  }
  // Every writer fences its P stores before the cross-WG publication barrier.
  cutlass::arch::fence_view_async_shared();
}

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void run_softmax_helper(
    TensorSP const& sP, Plan& plan, int pv_group, int warp_in_group, int lane, int physical_page,
    int page_idx, int m_block_idx, int total_rows, int s_q, int num_heads, int cache_len,
    int window_size, float softmax_scale, int pstate_slot, float* deferred_p0 = nullptr,
    float* deferred_p1 = nullptr, float* deferred_row_scale = nullptr) {
  const int row = pv_group * 4 + warp_in_group;
  softmax_one_row<T>(sP, plan, row, lane, physical_page, page_idx, m_block_idx, total_rows, s_q,
                     num_heads, cache_len, window_size, softmax_scale);
  cutlass::arch::fence_view_async_shared();
}
