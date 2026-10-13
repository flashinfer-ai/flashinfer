#pragma once
template<class T, class MMA, class A, class B, class Score, class Plan>
__forceinline__ __device__ void execute_qk_program(
    MMA& mma, A const& a, B const& b, Score& score, Plan& plan,
    int idx, bool use_aux, bool aux_phase, bool kv_phase) {
    if constexpr (T::Config::LookAhead > 0) {
        issue_qk_ready_order_lookahead<T>(mma, a, b, score, plan, idx, use_aux, aux_phase, kv_phase);
    } else {
        QKExecutor<typename T::Config::QKProgram>::template execute<T>(mma, a, b, score, plan, kv_phase);
    }
}

// One common role loop. Accumulators are scoped to their actual role; the
// three protocols only select publication/wait/producer operations.
template<int QKSlices, class T, class TMA, class SQ, class KV, class P,
         class V, class Plan>
__forceinline__ __device__ void run_qk_role(
    SQ const& sQ, KV& sKV, P const& sP, V const& sV, Plan& plan,
    const int* block_table, typename T::InputT* out,
    int batch_idx, int m_block_idx, int total_rows, int s_q, int num_heads,
    int max_pages, int cache_len, int start_page, int end_page, int window_size,
    float softmax_scale, int64_t block_table_stride_b,
    int64_t out_stride_b, int64_t out_stride_s, int64_t out_stride_h,
    int idx_in_warpgroup, int warp_in_group, int lane, TMA const& tma_params) {
    using C = typename T::Config;
    bool kv_phase = 0;
    {
        auto qk_output = [&]() {
            if constexpr (QKSlices > 0) {
                return partition_fragment_C(typename T::TiledMMAPV{},
                    Shape<Int<QKSlices * T::TILE_K>, Int<T::BLOCK_M>>{});
            } else return cute::tuple<>{};
        }();
        if constexpr (QKSlices > 0) cute::fill(qk_output, 0.0f);
        Tensor score = partition_fragment_C(typename T::TiledMMAQK{},
            Shape<Int<T::BLOCK_N>, Int<T::BLOCK_M>>{});
        typename T::TiledMMAQK qk_mma;
        auto qk_thread = qk_mma.get_slice(idx_in_warpgroup);
        Tensor qk_a = qk_thread.partition_fragment_A(sKV);
        Tensor qk_b = qk_thread.partition_fragment_B(sQ);
        bool free_phase = 0;
        const int first_physical_page = start_page < end_page && start_page < max_pages ?
            block_table[batch_idx * block_table_stride_b + start_page] : -1;
        if (first_physical_page >= 0) {
            if constexpr (C::LookAhead > 0) {
                const int second_page = start_page + 1;
                const int second_physical_page = second_page < end_page && second_page < max_pages ?
                    block_table[batch_idx * block_table_stride_b + second_page] : -1;
                issue_lookahead_prologue<T>(sKV, plan, first_physical_page,
                    second_physical_page, idx_in_warpgroup, tma_params);
            }
            execute_qk_program<T>(qk_mma, qk_a, qk_b, score, plan,
                idx_in_warpgroup, false, false, kv_phase);
            kv_phase ^= 1;
        } else cute::fill(score, 0.0f);
        for (int page_idx : mla_swa::PageRange<C>{start_page, end_page}) {
            const int physical_page = page_idx < max_pages ?
                block_table[batch_idx * block_table_stride_b + page_idx] : -1;
            const bool has_next = page_idx + 1 < end_page;
            const int next_page = page_idx + 1;
            const int next_physical_page = has_next && next_page < max_pages ?
                block_table[batch_idx * block_table_stride_b + next_page] : -1;
            if constexpr (C::M == 8) role_barrier();
            if (has_next && warp_in_group == 0) {
                issue_kv_tma_async<16, false, T>(sKV, plan, next_physical_page,
                    lane, free_phase, tma_params);
            }
            if constexpr (C::M == 8) {
                register_fragment_softmax<T>(score, sP, plan, physical_page, page_idx,
                    m_block_idx, total_rows, s_q, num_heads, cache_len, window_size,
                    softmax_scale, idx_in_warpgroup, warp_in_group, lane);
                pv_step<QKSlices, T>(qk_output, 0, 0, QKSlices, 1, sV, sP, plan,
                    idx_in_warpgroup, warp_in_group, lane, has_next, sKV,
                    -1, -1, page_idx - start_page, 0, idx_in_warpgroup, tma_params);
            } else {
                if constexpr (C::M == 16) {
                    if (page_idx > start_page) v0_wait_helper_done();
                }
                Tensor sScore = make_tensor(make_smem_ptr(plan.softmax_score.data()), typename T::SmemLayoutScore{});
                store_score_fragment<T>(score, sScore, warp_in_group, lane);
                if constexpr (C::M == 16) {
                    qk_warpgroup_barrier();
                    v0_score_ready();
                } else {
                    m32_p0_qk_warpgroup_barrier();
                    m32_p0_score_ready_producer();
                    run_m32_p0_softmax_helper<T>(sP, plan, 0, warp_in_group, lane,
                        physical_page, page_idx, m_block_idx, total_rows,
                        s_q, num_heads, cache_len, window_size, softmax_scale);
                    m32_p0_p_ready();
                }
            }
            if constexpr (C::M == 8) {
                if (has_next) {
                    if (warp_in_group == 0) issue_kv_tma_stream<0, 1, T>(sKV, plan, next_physical_page, lane, free_phase, tma_params);
                    else if (warp_in_group == 1) issue_kv_tma_stream<1, 6, T>(sKV, plan, next_physical_page, lane, free_phase, tma_params);
                    else if (warp_in_group == 2) issue_kv_tma_stream<6, 11, T>(sKV, plan, next_physical_page, lane, free_phase, tma_params);
                    else issue_kv_tma_stream<11, 16, T>(sKV, plan, next_physical_page, lane, free_phase, tma_params);
                }
                if (has_next || C::Backend == mla_swa::Reduce::DSM) role_barrier();
            }
            if (has_next) {
                if (next_physical_page >= 0) {
                    const int next_local_page = page_idx - start_page + 1;
                    execute_qk_program<T>(qk_mma, qk_a, qk_b, score, plan, idx_in_warpgroup,
                        (next_local_page & 1) != 0, ((next_local_page >> 1) & 1) != 0, kv_phase);
                } else cute::fill(score, 0.0f);
                kv_phase ^= 1;
                free_phase ^= 1;
            }
            if constexpr (C::M == 32) m32_p0_p_free();
        }
        if constexpr (C::M == 8) {
            if constexpr (C::Backend == mla_swa::Reduce::GlobalFP32) {
                write_output_slices<QKSlices, SplitOutputTraits<T>>(qk_output, 0, 0, QKSlices, plan,
                    reinterpret_cast<float*>(out), batch_idx, m_block_idx, total_rows, num_heads,
                    out_stride_b, out_stride_s, out_stride_h, warp_in_group, lane);
            } else {
                dsm_direct_accum_m8_s3_slices<true, QKSlices, T>(qk_output, 0, 0, QKSlices, plan, out,
                    batch_idx, m_block_idx, total_rows, num_heads, out_stride_b, out_stride_s, out_stride_h,
                    idx_in_warpgroup, warp_in_group, lane);
            }
        } else if constexpr (C::M == 16) {
            if (start_page < end_page) v0_wait_helper_done();
        }
    }
    if constexpr (C::M == 32 && C::Backend == mla_swa::Reduce::DSM) {
        if (start_page >= end_page) role_barrier();
        dsm_direct_accum_m32_qk_s2<T>(plan, idx_in_warpgroup, out,
            batch_idx, m_block_idx, total_rows, num_heads, out_stride_b, out_stride_s, out_stride_h);
    }
}
