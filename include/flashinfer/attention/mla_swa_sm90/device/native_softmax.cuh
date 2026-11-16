
template <class T, class Score, class P, class Plan>
__forceinline__ __device__ void register_fragment_softmax(
    Score& score, P const& sP, Plan& plan, int physical_page, int page_idx, int m_block_idx,
    int total_rows, int s_q, int num_heads, int cache_len, int window_size, float softmax_scale,
    int idx_in_warpgroup, int warp_in_group, int lane) {
  const int m0 = fragment_m0(warp_in_group, lane);
  const int m1 = m0 + 8;
  static_assert(T::BLOCK_M == 8);
  // Public contract Hq=64 and M8 keeps the CTA inside one query token.
  const int q_pos = cache_len - s_q + m_block_idx / 8;
  const int left = max(q_pos - window_size + 1, 0);
#pragma unroll
  for (int group = 0; group < T::BLOCK_M / 8; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
    const int base = group * 4;
    const int token0 = page_idx * T::BLOCK_N + m0;
    const int token1 = page_idx * T::BLOCK_N + m1;
    const int global_row0 = m_block_idx * T::BLOCK_M + n0;
    const int global_row1 = global_row0 + 1;
    const bool row0_valid = physical_page >= 0 && global_row0 < total_rows;
    const bool row1_valid = physical_page >= 0 && global_row1 < total_rows;
    const bool valid00 = row0_valid && token0 >= left && token0 <= q_pos;
    const bool valid01 = row1_valid && token0 >= left && token0 <= q_pos;
    const bool valid10 = row0_valid && token1 >= left && token1 <= q_pos;
    const bool valid11 = row1_valid && token1 >= left && token1 <= q_pos;
    const float scale = softmax_scale * 1.4426950408889634f;
    score(base + 0) = valid00 ? score(base + 0) * scale : -CUDART_INF_F;
    score(base + 1) = valid01 ? score(base + 1) * scale : -CUDART_INF_F;
    score(base + 2) = valid10 ? score(base + 2) * scale : -CUDART_INF_F;
    score(base + 3) = valid11 ? score(base + 3) * scale : -CUDART_INF_F;
    float max0 = fmaxf(score(base + 0), score(base + 2));
    float max1 = fmaxf(score(base + 1), score(base + 3));
    max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 4));
    max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 4));
    max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 8));
    max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 8));
    max0 = fmaxf(max0, __shfl_xor_sync(0xffffffff, max0, 16));
    max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, 16));
    if (lane < 4) {
      plan.reduction[warp_in_group * T::BLOCK_M + n0] = max0;
      plan.reduction[warp_in_group * T::BLOCK_M + n1] = max1;
    }
  }
  if constexpr (T::Config::S == 9)
    role_barrier();
  else
    qk_warpgroup_barrier();
  if (idx_in_warpgroup < T::BLOCK_M) {
    const int row = idx_in_warpgroup;
    float tile_max = -CUDART_INF_F;
#pragma unroll
    for (int warp = 0; warp < 4; ++warp) {
      tile_max = fmaxf(tile_max, plan.reduction[warp * T::BLOCK_M + row]);
    }
    const float old_max = plan.row_max[row];
    const float new_max = fmaxf(old_max, tile_max);
    plan.row_scale[row] = isfinite(old_max) ? exp2f(old_max - new_max) : 0.0f;
    plan.row_max[row] = new_max;
  }
  if constexpr (T::Config::S == 9)
    role_barrier();
  else
    qk_warpgroup_barrier();
#pragma unroll
  for (int group = 0; group < T::BLOCK_M / 8; ++group) {
    const int n0 = group * 8 + (lane % 4) * 2;
    const int n1 = n0 + 1;
    const int base = group * 4;
    const float score00 = score(base + 0);
    const float score01 = score(base + 1);
    const float score10 = score(base + 2);
    const float score11 = score(base + 3);
    const float p00 = isfinite(score00) ? exp2f(score00 - plan.row_max[n0]) : 0.0f;
    const float p01 = isfinite(score01) ? exp2f(score01 - plan.row_max[n1]) : 0.0f;
    const float p10 = isfinite(score10) ? exp2f(score10 - plan.row_max[n0]) : 0.0f;
    const float p11 = isfinite(score11) ? exp2f(score11 - plan.row_max[n1]) : 0.0f;
    sP(n0, m0) = static_cast<typename T::InputT>(p00);
    sP(n1, m0) = static_cast<typename T::InputT>(p01);
    sP(n0, m1) = static_cast<typename T::InputT>(p10);
    sP(n1, m1) = static_cast<typename T::InputT>(p11);
    float sum0 = p00 + p10;
    float sum1 = p01 + p11;
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 4);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 4);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 8);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 8);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 16);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 16);
    if (lane < 4) {
      plan.reduction[warp_in_group * T::BLOCK_M + n0] = sum0;
      plan.reduction[warp_in_group * T::BLOCK_M + n1] = sum1;
    }
  }
  // Publish generic P stores to the WGMMA async proxy before the handoff.
  cutlass::arch::fence_view_async_shared();
  if constexpr (T::Config::S == 9)
    role_barrier();
  else
    qk_warpgroup_barrier();
  if (idx_in_warpgroup < T::BLOCK_M) {
    const int row = idx_in_warpgroup;
    float tile_sum = 0.0f;
#pragma unroll
    for (int warp = 0; warp < 4; ++warp) {
      tile_sum += plan.reduction[warp * T::BLOCK_M + row];
    }
    plan.row_sum[row] = plan.row_sum[row] * plan.row_scale[row] + tile_sum;
  }
  role_barrier();
}
