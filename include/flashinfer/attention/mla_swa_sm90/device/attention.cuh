#pragma once
template <class T, class KV, class Plan, class TMA>
__forceinline__ __device__ void bootstrap_regular_kv(KV& sKV, Plan& plan, int physical_page,
                                                     TMA const& tma_params) {
  if (threadIdx.x == 128 && physical_page >= 0) {
    Tensor gkv = tma_params.tma_k.get_tma_tensor(tma_params.shape_k)(_, _, physical_page);
    launch_kv_tma<16, T::NUM_K_TILES>(gkv, sKV, tma_params.tma_k, plan.kv_barriers, 0);
    launch_kv_tma<0, 16>(gkv, sKV, tma_params.tma_k, plan.kv_barriers, 0);
#pragma unroll
    for (int tile = 0; tile < T::NUM_K_TILES; ++tile)
      plan.kv_barriers[tile].arrive_and_expect_tx(T::BLOCK_N * T::TILE_K *
                                                  sizeof(typename T::InputT));
  }
}

template <class T, class Q, class Plan, class TMA>
__forceinline__ __device__ void bootstrap_q(Q& sQ, Plan& plan, int batch_idx, int m_block_idx,
                                            TMA const& tma_params) {
  if (threadIdx.x == 0) {
    Tensor gq = tma_params.tma_q.get_tma_tensor(tma_params.shape_q)(_, _, batch_idx);
    auto thr_tma = tma_params.tma_q.get_slice(_0{});
    Tensor q_tile =
        flat_divide(gq, Shape<Int<T::BLOCK_M>, Int<T::HEAD_DIM_K>>{})(_, _, m_block_idx, _0{});
    cute::copy(
        tma_params.tma_q.with(reinterpret_cast<typename TMABarrier::ValueType&>(plan.q_barrier),
                              uint16_t{0}, cute::TMA::CacheHintSm90::EVICT_FIRST),
        thr_tma.partition_S(q_tile), thr_tma.partition_D(sQ));
    plan.q_barrier.arrive_and_expect_tx(T::BLOCK_M * T::HEAD_DIM_K * sizeof(typename T::InputT));
  }
}

template <class T, class TMA>
__forceinline__ __device__ void attention_body(
    const typename T::InputT* q, const typename T::InputT* kv, const int* block_table,
    const int* cache_seqlens, typename T::InputT* out, float* lse, int num_splits,
    int64_t out_stride_split, int batch_size, int s_q, int num_heads, int max_pages,
    int window_size, float softmax_scale, int64_t q_stride_b, int64_t q_stride_s,
    int64_t q_stride_h, int64_t kv_stride_token, int64_t block_table_stride_b, int64_t out_stride_b,
    int64_t out_stride_s, int64_t out_stride_h, TMA const& tma_params) {
  using C = typename T::Config;
  using InputT = typename T::InputT;
  extern __shared__ char shared_raw[];
  auto& plan = *reinterpret_cast<typename T::SharedMemoryPlan*>(shared_raw);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), typename T::SmemLayoutQ{});
  Tensor sKV = make_tensor(make_smem_ptr(plan.kv.data()), typename T::SmemLayoutKV{});
  Tensor sP = make_tensor(make_smem_ptr(plan.p.data()), typename T::SmemLayoutP{});
  Tensor sV = make_tensor(make_smem_ptr(plan.kv.data()), typename T::SmemLayoutV{});
  const int warpgroup = threadIdx.x / 128;
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_in_group = idx_in_warpgroup / 32;
  const int lane = idx_in_warpgroup % 32;
  const int batch_idx = C::Backend == mla_swa::Reduce::DSM ? blockIdx.x / C::S : blockIdx.x;
  const int split_idx = C::Backend == mla_swa::Reduce::DSM ? blockIdx.x % C::S : blockIdx.z;
  const int grid_m_block_idx = blockIdx.y;
  const int total_rows = s_q * num_heads;
  const int total_m_blocks = (total_rows + T::BLOCK_M - 1) / T::BLOCK_M;
  if (threadIdx.x == 0) {
    cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_params.tma_k.get_tma_descriptor());
  }
  role_barrier();
#pragma unroll
  for (int row_group_idx = 0; row_group_idx < 1; ++row_group_idx) {
    const int m_block_idx = grid_m_block_idx + row_group_idx;
    if (m_block_idx >= total_m_blocks) break;
    if (threadIdx.x == 0) {
      plan.q_barrier.init(1);
      if constexpr (C::LookAhead > 0) {
        plan.pv_done_barrier.init(T::NUM_PV_WARPGROUPS);
        plan.softmax_score_ready.init(1);
        plan.softmax_done.init(T::NUM_PV_WARPGROUPS);
      }
      if constexpr (C::M == 32 && C::Backend == mla_swa::Reduce::DSM) {
        for (int event = 0; event < 4; ++event) {
          plan.p1_ready[event].init(1);
          plan.p1_ack[event].init(event == 0 ? 1 : 4);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
#pragma unroll
      for (int tile = 0; tile < T::NUM_K_TILES; ++tile) plan.kv_barriers[tile].init(1);
      if constexpr (C::LookAhead > 0) {
#pragma unroll
        for (int slot = 0; slot < T::LOOKAHEAD_K + T::AUX_GUARD_SLICES; ++slot)
          plan.aux_kv_barriers[slot].init(1);
      }
#pragma unroll
      for (int tile = 0; tile < T::NUM_V_TILES; ++tile) plan.kv_free_barriers[tile].init(1);
      cutlass::arch::fence_view_async_shared();
    }
    role_barrier();
    if constexpr (C::M == 32 && C::Backend == mla_swa::Reduce::DSM) cute::cluster_arrive();
    // Preserve original prologue order: M16 issues Q before reading scheduler
    // metadata; the other protocols compute their first page before issuing Q.
    if constexpr (C::LookAhead > 0) bootstrap_q<T>(sQ, plan, batch_idx, m_block_idx, tma_params);
    const int cache_len = cache_seqlens[batch_idx];
    const int union_left = max(cache_len - window_size - s_q + 1, 0);
    const int full_start_page = union_left / T::BLOCK_N;
    // Graph padding may have length zero and an entirely sentinel page table.
    // Do not run masked PV against uninitialized KV: 0 * NaN is still NaN.
    const int full_end_page = (max(cache_len, 0) + T::BLOCK_N - 1) / T::BLOCK_N;
    const int total_pages = full_end_page - full_start_page;
    const int start_page =
        full_start_page +
        static_cast<int>((static_cast<int64_t>(total_pages) * split_idx) / num_splits);
    const int end_page =
        full_start_page +
        static_cast<int>((static_cast<int64_t>(total_pages) * (split_idx + 1)) / num_splits);
    InputT* role_out = out;
    if constexpr (C::Backend == mla_swa::Reduce::GlobalFP32)
      role_out += split_idx * out_stride_split;
    if constexpr (C::LookAhead == 0) {
      const int first_physical_page =
          start_page < end_page && start_page < max_pages
              ? block_table[batch_idx * block_table_stride_b + start_page]
              : -1;
      bootstrap_q<T>(sQ, plan, batch_idx, m_block_idx, tma_params);
      bootstrap_regular_kv<T>(sKV, plan, first_physical_page, tma_params);
    }
    if constexpr (C::LookAhead > 0) {
      plan.q_barrier.wait(0);
      if (threadIdx.x < T::BLOCK_M) {
        plan.row_max[threadIdx.x] = -CUDART_INF_F;
        plan.row_sum[threadIdx.x] = 0.0f;
        plan.row_scale[threadIdx.x] = 0.0f;
      }
    } else {
      if (threadIdx.x >= 128 && threadIdx.x < 128 + T::BLOCK_M) {
        const int row = threadIdx.x - 128;
        plan.row_max[row] = -CUDART_INF_F;
        plan.row_sum[row] = 0.0f;
        plan.row_scale[row] = 0.0f;
      }
      plan.q_barrier.wait(0);
    }
    role_barrier();
    if (warpgroup == 0) {
      run_qk_role<C::QkPV, T>(sQ, sKV, sP, sV, plan, block_table, role_out, batch_idx, m_block_idx,
                              total_rows, s_q, num_heads, max_pages, cache_len, start_page,
                              end_page, window_size, softmax_scale, block_table_stride_b,
                              out_stride_b, out_stride_s, out_stride_h, idx_in_warpgroup,
                              warp_in_group, lane, tma_params);
      if constexpr (C::M == 16) {
        if (split_idx == 0)
          dsm_direct_accum_qk_s2<0, T>(plan, idx_in_warpgroup);
        else
          dsm_direct_accum_qk_s2<1, T>(plan, idx_in_warpgroup);
      }
    } else {
      run_pv_role<T>(sKV, sP, sV, plan, block_table, role_out, batch_idx, m_block_idx, total_rows,
                     s_q, num_heads, max_pages, cache_len, start_page, end_page, window_size,
                     softmax_scale, warpgroup, idx_in_warpgroup, warp_in_group, lane,
                     block_table_stride_b, out_stride_b, out_stride_s, out_stride_h, tma_params);
    }
    if constexpr (C::Backend == mla_swa::Reduce::GlobalFP32) {
      if (warpgroup == 0 && idx_in_warpgroup < T::BLOCK_M) {
        const int global_row = m_block_idx * T::BLOCK_M + idx_in_warpgroup;
        if (global_row < total_rows) {
          const float sum = plan.row_sum[idx_in_warpgroup];
          const float row_lse =
              sum > 0.0f ? plan.row_max[idx_in_warpgroup] + log2f(sum) : -CUDART_INF_F;
          lse[(batch_idx * total_rows + global_row) * num_splits + split_idx] = row_lse;
        }
      }
    }
    if constexpr (C::M == 16 || C::Backend == mla_swa::Reduce::GlobalFP32) role_barrier();
  }
}
