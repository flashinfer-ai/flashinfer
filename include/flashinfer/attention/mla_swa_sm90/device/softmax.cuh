#pragma once

template <class T, class Plan>
__forceinline__ __device__ float* m32_dead_rope_score(Plan& plan) {
  static_assert(T::BLOCK_M == 32 && T::BLOCK_N == 64);
  // The final KV feature is QK-only. Its complete 8-KiB swizzle atom is
  // dead after QK wait0 and remains dead until the next feature16 TMA.
  constexpr int offset = typename T::SmemLayoutKV{}(cute::make_coord(0, 1024));
  return reinterpret_cast<float*>(plan.kv.data() + offset);
}

__forceinline__ __device__ float softmax_ordered_warp_max(float value,
                                                          unsigned mask = 0xffffffffu) {
  // Monotonic signed-int encoding, including negative values and -Inf.
  // Ignore NaNs as fmaxf does; nonfinite probabilities are still zeroed below.
  int key = __float_as_int(fmaxf(value, -CUDART_INF_F));
  key ^= key < 0 ? 0x7fffffff : 0;
  key = __reduce_max_sync(mask, key);
  key ^= key < 0 ? 0x7fffffff : 0;
  return __int_as_float(key);
}

template <typename T, bool RegisterState = false, typename TensorSP, typename Plan>
__forceinline__ __device__ void softmax_one_row(TensorSP const& sP, Plan& plan, int row, int lane,
                                                int physical_page, int page_idx, int m_block_idx,
                                                int total_rows, int s_q, int num_heads,
                                                int cache_len, int window_size, float softmax_scale,
                                                SoftmaxRowState* state = nullptr,
                                                bool last_page = false) {
  Tensor sScore =
      make_tensor(make_smem_ptr(plan.softmax_score.data()), typename T::SmemLayoutScore{});
  const int global_row = m_block_idx * T::BLOCK_M + row;
  const bool row_valid = physical_page >= 0 && global_row < total_rows;
  // Hq=64 and M divides64, so every helper in this CTA uses one query
  // position. Avoid runtime head division and two per-row broadcasts.
  static_assert(64 % T::BLOCK_M == 0);
  const int q_pos = cache_len - s_q + m_block_idx / (64 / T::BLOCK_M);
  const int left = max(q_pos - window_size + 1, 0);
  const int token0 = page_idx * T::BLOCK_N + lane;
  const int token1 = token0 + 32;
  const float scale = softmax_scale * 1.4426950408889634f;
  const float score0 =
      row_valid && token0 >= left && token0 <= q_pos ? sScore(row, lane) * scale : -CUDART_INF_F;
  const float score1 = row_valid && token1 >= left && token1 <= q_pos
                           ? sScore(row, lane + 32) * scale
                           : -CUDART_INF_F;
  const float tile_max = softmax_ordered_warp_max(fmaxf(score0, score1));
  float new_max = -CUDART_INF_F, row_scale = 0.0f;
  if constexpr (RegisterState) {
    const float old_max = state->maximum;
    new_max = fmaxf(old_max, tile_max);
    state->maximum = new_max;
    if (lane == 0) {
      row_scale = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
      plan.row_scale[row] = row_scale;
    }
  } else {
    if (lane == 0) {
      const float old_max = plan.row_max[row];
      new_max = fmaxf(old_max, tile_max);
      row_scale = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
      plan.row_scale[row] = row_scale;
      plan.row_max[row] = new_max;
    }
    new_max = __shfl_sync(0xffffffff, new_max, 0);
    row_scale = __shfl_sync(0xffffffff, row_scale, 0);
  }
  const float p0 = isfinite(score0) ? exp2f(score0 - new_max) : 0.0f;
  if constexpr (T::Config::StoreP0First) sP(row, lane) = static_cast<typename T::InputT>(p0);
  const float p1 = isfinite(score1) ? exp2f(score1 - new_max) : 0.0f;
  if constexpr (!T::Config::StoreP0First) sP(row, lane) = static_cast<typename T::InputT>(p0);
  sP(row, lane + 32) = static_cast<typename T::InputT>(p1);
  float tile_sum = p0 + p1;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
  if (lane == 0) {
    if constexpr (RegisterState) {
      state->sum = state->sum * row_scale + tile_sum;
      if (last_page) {
        plan.row_max[row] = new_max;
        plan.row_sum[row] = state->sum;
      }
    } else {
      plan.row_sum[row] = plan.row_sum[row] * row_scale + tile_sum;
    }
  }
}

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void softmax_two_rows_halfwarp(TensorSP const& sP, Plan& plan, int warp,
                                                          int lane, int physical_page, int page_idx,
                                                          int m_block_idx, int total_rows, int s_q,
                                                          int cache_len, int window_size,
                                                          float softmax_scale,
                                                          SoftmaxRowState& state, bool last_page) {
  static_assert(T::BLOCK_M == 32);
  Tensor sScore =
      make_tensor(make_smem_ptr(m32_dead_rope_score<T>(plan)), typename T::SmemLayoutScore{});
  const int half_lane = lane % 16;
  const int row = warp * 2 + lane / 16;
  const int column = half_lane * 4;
  const unsigned mask = lane < 16 ? 0x0000ffffu : 0xffff0000u;
  const int q_pos = cache_len - s_q + m_block_idx / 2;
  const int left = max(q_pos - window_size + 1, 0);
  const bool row_valid = physical_page >= 0 && m_block_idx * 32 + row < total_rows;
  const int token = page_idx * 64 + column;
  const float scale = softmax_scale * 1.4426950408889634f;
  const float4 values = *reinterpret_cast<float4 const*>(&sScore(row, column));
  const float a = row_valid && token >= left && token <= q_pos ? values.x * scale : -CUDART_INF_F;
  const float b =
      row_valid && token + 1 >= left && token + 1 <= q_pos ? values.y * scale : -CUDART_INF_F;
  const float c =
      row_valid && token + 2 >= left && token + 2 <= q_pos ? values.z * scale : -CUDART_INF_F;
  const float d =
      row_valid && token + 3 >= left && token + 3 <= q_pos ? values.w * scale : -CUDART_INF_F;
  const float maximum = softmax_ordered_warp_max(fmaxf(fmaxf(a, b), fmaxf(c, d)), mask);
  float new_max = -CUDART_INF_F, row_scale = 0.0f;
  if constexpr (T::Config::S == 1) {
    const float old_max = state.maximum;
    new_max = fmaxf(old_max, maximum);
    state.maximum = new_max;
    if (half_lane == 0) {
      row_scale = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
      plan.row_scale[row] = row_scale;
    }
  } else {
    if (half_lane == 0) {
      const float old_max = plan.row_max[row];
      new_max = fmaxf(old_max, maximum);
      row_scale = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
      plan.row_max[row] = new_max;
      plan.row_scale[row] = row_scale;
    }
    new_max = __shfl_sync(mask, new_max, 0, 16);
    row_scale = __shfl_sync(mask, row_scale, 0, 16);
  }
  const float p0 = isfinite(a) ? exp2f(a - new_max) : 0.0f;
  const float p1 = isfinite(b) ? exp2f(b - new_max) : 0.0f;
  const float p2 = isfinite(c) ? exp2f(c - new_max) : 0.0f;
  const float p3 = isfinite(d) ? exp2f(d - new_max) : 0.0f;
  cutlass::Array<typename T::InputT, 4> packed;
  packed[0] = typename T::InputT(p0);
  packed[1] = typename T::InputT(p1);
  packed[2] = typename T::InputT(p2);
  packed[3] = typename T::InputT(p3);
  *reinterpret_cast<cutlass::Array<typename T::InputT, 4>*>(&sP(row, column)) = packed;
  float sum = (p0 + p1) + (p2 + p3);
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) sum += __shfl_down_sync(mask, sum, offset, 16);
  if (half_lane == 0) {
    if constexpr (T::Config::S == 1) {
      state.sum = state.sum * row_scale + sum;
      if (last_page) {
        plan.row_max[row] = new_max;
        plan.row_sum[row] = state.sum;
      }
    } else {
      plan.row_sum[row] = plan.row_sum[row] * row_scale + sum;
    }
  }
}

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void run_m32_p0_softmax_helper(
    TensorSP const& sP, Plan& plan, int warpgroup, int warp_in_group, int lane, int physical_page,
    int page_idx, int m_block_idx, int total_rows, int s_q, int num_heads, int cache_len,
    int window_size, float softmax_scale, SoftmaxRowState& state, bool last_page) {
  const int warp = warpgroup * 4 + warp_in_group;
  softmax_two_rows_halfwarp<T>(sP, plan, warp, lane, physical_page, page_idx, m_block_idx,
                               total_rows, s_q, cache_len, window_size, softmax_scale, state,
                               last_page);
  // Every writer fences its P stores before the cross-WG publication barrier.
  cutlass::arch::fence_view_async_shared();
}

template <typename T, typename TensorSP, typename Plan>
__forceinline__ __device__ void run_softmax_helper(
    TensorSP const& sP, Plan& plan, int pv_group, int warp_in_group, int lane, int physical_page,
    int page_idx, int m_block_idx, int total_rows, int s_q, int num_heads, int cache_len,
    int window_size, float softmax_scale, int pstate_slot, SoftmaxRowState& state, bool last_page) {
  const int row = pv_group * 4 + warp_in_group;
  softmax_one_row<T, true>(sP, plan, row, lane, physical_page, page_idx, m_block_idx, total_rows,
                           s_q, num_heads, cache_len, window_size, softmax_scale, &state,
                           last_page);
  cutlass::arch::fence_view_async_shared();
}
