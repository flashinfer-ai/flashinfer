#pragma once
// A single slice executor. Buffer selection and TMA producer differ, not
// accumulator rescaling, fragment layout or the PV GEMM itself.
template <int MaxSlices, class T, class Output, class V, class P, class KV, class Plan, class TMA>
__forceinline__ __device__ void pv_step(Output& output_fragment_all, int output_slice_offset,
                                        int slice_begin, int slice_end, int slice_stride,
                                        V const& sV, P const& sP, Plan& plan, int idx_in_warpgroup,
                                        int warp_in_group, int lane, bool signal_free, KV& sKV,
                                        int next_physical_page, int page_plus2_physical,
                                        int local_page, int pstate_slot, int tma_lane,
                                        TMA const& tma_params) {
  static_assert(MaxSlices > 0);
  // P publication has completed; the current row scales remain
  // immutable until every PV slice is done. Retain them across each
  // WGMMA/TMA compiler memory clobber instead of reloading per slice.
  cutlass::Array<float, T::BLOCK_M / 4> cached_scale;
  constexpr bool CacheScale = T::Config::M == 16 || (T::Config::M == 8 && T::Config::S != 9);
  if constexpr (CacheScale) {
#pragma unroll
    for (int group = 0; group < T::BLOCK_M / 8; ++group) {
      const int n0 = group * 8 + (lane % 4) * 2;
      cached_scale[group * 2] = plan.row_scale[n0];
      cached_scale[group * 2 + 1] = plan.row_scale[n0 + 1];
    }
  }
#pragma unroll
  for (int slice = 0; slice < MaxSlices; ++slice) {
    if (slice_begin + slice * slice_stride >= slice_end) continue;
    Tensor output_fragment_flat = output_fragment_all(_, output_slice_offset + slice, _0{});
    Tensor output_fragment =
        make_tensor(output_fragment_flat.data(), typename T::OutputSliceFragmentLayout{});
    // Hq64 single-query PageRange has at most nine physical pages for
    // W513. S9 therefore gives each CTA at most one page: initial +0
    // never needs online rescaling. Keep other tactics exactly unchanged.
    if constexpr (CacheScale) {
#pragma unroll
      for (int group = 0; group < T::BLOCK_M / 8; ++group) {
        const int base = group * 4;
        output_fragment(base + 0) *= cached_scale[group * 2];
        output_fragment(base + 1) *= cached_scale[group * 2 + 1];
        output_fragment(base + 2) *= cached_scale[group * 2];
        output_fragment(base + 3) *= cached_scale[group * 2 + 1];
      }
    } else if constexpr (!(T::Config::M == 8 && T::Config::S == 9))
      rescale_output_fragment<T>(output_fragment, plan, lane);
    const int global_slice = slice_begin + slice * slice_stride;
    Tensor v_tiles = flat_divide(sV, Shape<Int<T::TILE_K>, Int<T::BLOCK_N>>{});
    Tensor v_tile = v_tiles(_, _, global_slice, _0{});
    bool used_aux = false;
    if constexpr (T::Config::LookAhead > 0) {
      if ((local_page & 1) != 0 && global_slice >= T::NUM_V_TILES - T::LOOKAHEAD_K) {
        issue_aux_pv_dynamic<T>(output_fragment, sP, plan,
                                global_slice - (T::NUM_V_TILES - T::LOOKAHEAD_K), idx_in_warpgroup);
        used_aux = true;
      }
    }
    if (!used_aux) {
      issue_pv_gemm<T>(output_fragment, v_tile, sP, idx_in_warpgroup);
    }
    if constexpr (T::Config::LookAhead > 0) {
      if (signal_free)
        issue_owner_tma_for_slice<T>(global_slice, sKV, plan, next_physical_page,
                                     page_plus2_physical, local_page, tma_lane, tma_params);
    } else if constexpr (T::Config::M == 8 && T::Config::S != 9) {
      // This owner has completed its WGMMA wait0 for exactly this slice.
      // No other PV group reads that feature slice: overwrite it directly.
      if (signal_free)
        issue_base_kv_tma_dynamic<T>(sKV, plan, next_physical_page, global_slice, tma_lane,
                                     tma_params);
    } else {
      if (signal_free && idx_in_warpgroup == 0) plan.kv_free_barriers[global_slice].arrive();
    }
  }
}

template <class C>
struct PVRoleOps {
  template <int kMaxSlices, class T, class Output, class KV, class P, class V, class Plan,
            class Task, class TMA>
  __forceinline__ __device__ static void step(Output& output, int begin, int end, int stride,
                                              KV& sKV, P const& sP, V const& sV, Plan& plan,
                                              Task const& task, int pv_group, int page_idx,
                                              bool& p_ready_phase, TMA const& tma,
                                              SoftmaxRowState& row_state) {
    if constexpr (C::M == 8) {
      const bool has_next = C::S != 9 && page_idx + 1 < task.end_page;
      const int next_physical_page =
          has_next && page_idx + 1 < task.max_pages
              ? task.block_table[task.batch_idx * task.block_table_stride_b + page_idx + 1]
              : -1;
      role_barrier();
      if constexpr (C::S == 9) {
        role_barrier();
        role_barrier();
        role_barrier();
      }
      role_barrier();
      pv_step<kMaxSlices, T>(output, 0, begin, end, stride, sV, sP, plan, task.idx_in_warpgroup,
                             task.warp_in_group, task.lane, has_next, sKV, next_physical_page, -1,
                             page_idx - task.start_page, 0, task.idx_in_warpgroup, tma);
      // The next iteration's first rendezvous protects old P/row_scale.
      // Last-page DSM still gathers all PV groups before reusing KV.
      if (!has_next && C::Backend == mla_swa::Reduce::DSM) role_barrier();
    } else {
      const int physical_page =
          page_idx < task.max_pages
              ? task.block_table[task.batch_idx * task.block_table_stride_b + page_idx]
              : -1;
      if constexpr (C::M == 32) {
        m32_p0_score_ready_consumer();
        run_m32_p0_softmax_helper<T>(sP, plan, pv_group + 1, task.warp_in_group, task.lane,
                                     physical_page, page_idx, task.m_block_idx, task.total_rows,
                                     task.s_q, task.num_heads, task.cache_len, task.window_size,
                                     task.softmax_scale, row_state, page_idx + 1 == task.end_page);
        m32_p0_p_ready();
      }
      const bool has_next = page_idx + 1 < task.end_page;
      const int next_page = page_idx + 1;
      const int next_physical_page =
          has_next && next_page < task.max_pages
              ? task.block_table[task.batch_idx * task.block_table_stride_b + next_page]
              : -1;
      if constexpr (C::M == 16) {
        const int local_page = page_idx - task.start_page;
        const int pstate_slot = local_page & 1;
        Tensor page_sP = sP;
        const int page_plus2_physical =
            page_idx + 2 < task.end_page && page_idx + 2 < task.max_pages
                ? task.block_table[task.batch_idx * task.block_table_stride_b + page_idx + 2]
                : -1;
        v0_wait_score_ready();
        run_softmax_helper<T>(page_sP, plan, pv_group, task.warp_in_group, task.lane, physical_page,
                              page_idx, task.m_block_idx, task.total_rows, task.s_q, task.num_heads,
                              task.cache_len, task.window_size, task.softmax_scale, pstate_slot,
                              row_state, !has_next);
        v0_pv_warpgroups_barrier();
        if (!has_next && m16_terminal_page_is_masked(page_idx, task.m_block_idx, task.s_q,
                                                     task.cache_len, task.window_size))
          v0_wait_helper_done();
        else
          v0_helper_done();
        pv_step<kMaxSlices, T>(output, 0, begin, end, stride, sV, page_sP, plan,
                               task.idx_in_warpgroup, task.warp_in_group, task.lane, has_next, sKV,
                               next_physical_page, page_plus2_physical, local_page, pstate_slot,
                               task.idx_in_warpgroup, tma);
        // No following page can overwrite P/scale after the final slice.
        // Each role's existing Global/DSM epilogue owns its output and
        // still performs the original mailbox/lifetime synchronization.
        if (has_next) v0_pv_warpgroups_barrier();
      } else {
        if constexpr (C::LookAhead > 0) {
          const int page_plus2 =
              page_idx + 2 < task.end_page && page_idx + 2 < task.max_pages
                  ? task.block_table[task.batch_idx * task.block_table_stride_b + page_idx + 2]
                  : -1;
          pv_step_owner_lookahead_m32<T>(output, begin, sV, sP, sKV, plan, task.idx_in_warpgroup,
                                         task.lane, page_idx - task.start_page, next_physical_page,
                                         page_plus2, has_next, tma);
        } else {
          pv_step_owner_tma_static<T>(output, begin, sV, sP, sKV, plan, task.idx_in_warpgroup,
                                      task.lane, next_physical_page, has_next, tma);
        }
        if (has_next || C::S == 1) m32_p0_p_free();
      }
    }
  }

  template <int MaxSlices, class T, class Output, class Plan, class Task>
  __forceinline__ __device__ static void finish(Output& output, int begin, int end, int stride,
                                                Plan& plan, typename T::InputT* out,
                                                Task const& task, int pv_group) {
    if constexpr (C::Backend == mla_swa::Reduce::GlobalFP32) {
      write_output_slices<MaxSlices, SplitOutputTraits<T>>(
          output, 0, begin, end, plan, reinterpret_cast<float*>(out), task.batch_idx,
          task.m_block_idx, task.total_rows, task.num_heads, task.out_stride_b, task.out_stride_s,
          task.out_stride_h, task.warp_in_group, task.lane);
    } else if constexpr (C::Backend == mla_swa::Reduce::Direct) {
      write_output_slices<MaxSlices, T>(output, 0, begin, end, plan, out, task.batch_idx,
                                        task.m_block_idx, task.total_rows, task.num_heads,
                                        task.out_stride_b, task.out_stride_s, task.out_stride_h,
                                        task.warp_in_group, task.lane);
    } else if constexpr (C::M == 8) {
      dsm_direct_accum_m8_s3_slices<false, MaxSlices, T>(
          output, 0, begin, end, plan, out, task.batch_idx, task.m_block_idx, task.total_rows,
          task.num_heads, task.out_stride_b, task.out_stride_s, task.out_stride_h,
          task.idx_in_warpgroup, task.warp_in_group, task.lane);
    } else if constexpr (C::M == 32) {
      if (task.start_page >= task.end_page) role_barrier();
      dsm_direct_accum_m32_pv_s2<MaxSlices, T>(output, begin, end, plan, out, task.batch_idx,
                                               task.m_block_idx, task.total_rows, task.num_heads,
                                               task.out_stride_b, task.out_stride_s,
                                               task.out_stride_h, task.warp_in_group, task.lane);
    } else {
      // Keep the owner rank a template constant: dynamic ownership
      // previously enlarged accumulator live ranges and spilled.
      const int rank = cute::block_id_in_cluster().x;
      if (rank == 0) {
        dsm_direct_accum_pv_s2<0, 1, MaxSlices, T>(
            output, begin, end, stride, pv_group, plan, out, task.batch_idx, task.m_block_idx,
            task.total_rows, task.num_heads, task.out_stride_b, task.out_stride_s,
            task.out_stride_h, task.idx_in_warpgroup, task.warp_in_group, task.lane);
      } else {
        dsm_direct_accum_pv_s2<1, 0, MaxSlices, T>(
            output, begin, end, stride, pv_group, plan, out, task.batch_idx, task.m_block_idx,
            task.total_rows, task.num_heads, task.out_stride_b, task.out_stride_s,
            task.out_stride_h, task.idx_in_warpgroup, task.warp_in_group, task.lane);
      }
    }
  }
};
