/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_kimi_k3_vision_tower_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SQ_SMEM_OFF 230400
#define SMEM_SQ_SMEM_STAGE_BYTES 1024
#define SMEM_SQ_SMEM_STRIDE 1024
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_H_OFF 17408
#define SMEM_SMEM_B_H_STAGE_BYTES 8192
#define SMEM_SMEM_B_H_STRIDE 32768
#define SMEM_TOTAL 231424
#define THREADS 320
#define num_cluster_tiles (m_tiles * 8)
#define tiles_per_group 256

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_kimi_k3_vision_tower_e850b9cf3fb4e092ca76(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ C2, __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ R, float* __restrict__ COS, float* __restrict__ SIN, __nv_bfloat16* __restrict__ CS, float* __restrict__ SQ, __nv_bfloat16* __restrict__ XW, __nv_bfloat16* __restrict__ WN, float* __restrict__ WS, unsigned int* __restrict__ FLAGS, const __grid_constant__ CUtensorMap RT, const __grid_constant__ CUtensorMap CT, const __grid_constant__ CUtensorMap XWT, int M, int m_tiles, int full_tiles, int tail_split, int pf_l2, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* sq_smem = reinterpret_cast<float*>(smem_raw + 230400);
    const int sq_smem_addr = smem + 230400;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    __nv_bfloat16* smem_b_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_h_addr = smem + 17408;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int cluster_id_1 = bid;
            unsigned int num_clusters_1 = num_bids;
            int k_begin = 0;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int h_item = cluster_id_1;
                int h_rel = h_item - full_tiles;
                int h_is = ((h_rel >= 0) ? 1 : 0);
                int h_relc = h_rel * h_is;
                int h_t = h_relc / 2;
                int h_half = h_relc - h_t * 2;
                int h_ct = h_item + h_is * (full_tiles + h_t - h_item);
                int group = h_ct / tiles_per_group;
                int first_m = group * 32;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= 32) ? 32 : remaining);
                int local = h_ct % tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int e_off_m = bid_m * 128;
                int e_wrow = bid_n * 128;
                #pragma unroll 1
                for (int e_s = 0; e_s < 7; e_s++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int e_k = k_begin + e_s;
                    tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768, (&B), 0, e_wrow, e_k, tma_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, e_wrow + 64, e_k, tma_full_addr + (load_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll
                for (int e_s2 = 0; e_s2 < 7; e_s2++) {
                    tma_3d_gmem2smem(smem_a_addr + (unsigned int)(e_s2 * 32768), (&A), 0, e_off_m, k_begin + e_s2, tma_full_addr + (e_s2) * 8);
                }
                int h_item_0 = cluster_id_1;
                int h_rel_1 = h_item_0 - full_tiles;
                int h_is_2 = ((h_rel_1 >= 0) ? 1 : 0);
                int h_relc_3 = h_rel_1 * h_is_2;
                int h_t_4 = h_relc_3 / 2;
                int h_half_5 = h_relc_3 - h_t_4 * 2;
                int h_ct_6 = h_item_0 + h_is_2 * (full_tiles + h_t_4 - h_item_0);
                int group_7 = h_ct_6 / tiles_per_group;
                int first_m_8 = group_7 * 32;
                int remaining_9 = m_tiles - first_m_8;
                int group_size_10 = ((remaining_9 >= 32) ? 32 : remaining_9);
                int local_11 = h_ct_6 % tiles_per_group;
                int bid_m_12 = first_m_8 + local_11 % group_size_10;
                int bid_n_13 = local_11 / group_size_10;
                int off_m = bid_m_12 * 128;
                int weight_row = bid_n_13 * 128;
                if (pf_l2 != 0) {
                    #pragma unroll 1
                    for (int pk = 7; pk < 24; pk++) {
                        int pk_step = k_begin + pk;
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m)), "r"((int)(pk_step)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row)), "r"((int)(pk_step)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int iter_k = 7; iter_k < 24; iter_k++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int k_step = k_begin + iter_k;
                    tma_3d_gmem2smem(smem_a_addr + load_stage * 32768, (&A), 0, off_m, k_step, tma_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row, k_step, tma_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, weight_row + 64, k_step, tma_full_addr + (load_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
                #pragma unroll 1
                for (unsigned int item = cluster_id_1 + num_clusters_1; item < full_tiles + 2 * (num_cluster_tiles - full_tiles); item += num_clusters_1) {
                    int h_item_1 = item;
                    int h_rel_2 = h_item_1 - full_tiles;
                    int h_is_3 = ((h_rel_2 >= 0) ? 1 : 0);
                    int h_relc_4 = h_rel_2 * h_is_3;
                    int h_t_5 = h_relc_4 / 2;
                    int h_half_6 = h_relc_4 - h_t_5 * 2;
                    int h_ct_7 = h_item_1 + h_is_3 * (full_tiles + h_t_5 - h_item_1);
                    if (h_is_3 == 0) {
                        int h_item_2 = item;
                        int h_rel_3 = h_item_2 - full_tiles;
                        int h_is_4 = ((h_rel_3 >= 0) ? 1 : 0);
                        int h_relc_5 = h_rel_3 * h_is_4;
                        int h_t_6 = h_relc_5 / 2;
                        int h_half_7 = h_relc_5 - h_t_6 * 2;
                        int h_ct_8 = h_item_2 + h_is_4 * (full_tiles + h_t_6 - h_item_2);
                        int group_9 = h_ct_8 / tiles_per_group;
                        int first_m_10 = group_9 * 32;
                        int remaining_11 = m_tiles - first_m_10;
                        int group_size_12 = ((remaining_11 >= 32) ? 32 : remaining_11);
                        int local_13 = h_ct_8 % tiles_per_group;
                        int bid_m_14 = first_m_10 + local_13 % group_size_12;
                        int bid_n_15 = local_13 / group_size_12;
                        int off_m_16 = bid_m_14 * 128;
                        int weight_row_17 = bid_n_15 * 128;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk_1 = 7; pk_1 < 24; pk_1++) {
                                int pk_step_1 = k_begin + pk_1;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m_16)), "r"((int)(pk_step_1)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row_17)), "r"((int)(pk_step_1)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < 24; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step_1 = k_begin + iter_k_1;
                            tma_3d_gmem2smem(smem_a_addr + load_stage * 32768, (&A), 0, off_m_16, k_step_1, tma_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row_17, k_step_1, tma_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, weight_row_17 + 64, k_step_1, tma_full_addr + (load_stage) * 8);
                            mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int h_item_2_1 = item;
                        int h_rel_3_1 = h_item_2_1 - full_tiles;
                        int h_is_4_1 = ((h_rel_3_1 >= 0) ? 1 : 0);
                        int h_relc_5_1 = h_rel_3_1 * h_is_4_1;
                        int h_t_6_1 = h_relc_5_1 / 2;
                        int h_half_7_1 = h_relc_5_1 - h_t_6_1 * 2;
                        int h_ct_8_1 = h_item_2_1 + h_is_4_1 * (full_tiles + h_t_6_1 - h_item_2_1);
                        int group_9_1 = h_ct_8_1 / tiles_per_group;
                        int first_m_10_1 = group_9_1 * 32;
                        int remaining_11_1 = m_tiles - first_m_10_1;
                        int group_size_12_1 = ((remaining_11_1 >= 32) ? 32 : remaining_11_1);
                        int local_13_1 = h_ct_8_1 % tiles_per_group;
                        int bid_m_14_1 = first_m_10_1 + local_13_1 % group_size_12_1;
                        int bid_n_15_1 = local_13_1 / group_size_12_1;
                        int off_m_16_1 = bid_m_14_1 * 128;
                        int weight_row_17_1 = bid_n_15_1 * 128 + h_half_6 * 64;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk_2 = 7; pk_2 < 24; pk_2++) {
                                int pk_step_2 = k_begin + pk_2;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m_16_1)), "r"((int)(pk_step_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row_17_1)), "r"((int)(pk_step_2)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < 24; iter_k_2++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step_2 = k_begin + iter_k_2;
                            tma_3d_gmem2smem(smem_a_addr + load_stage * 32768, (&A), 0, off_m_16_1, k_step_2, tma_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row_17_1, k_step_2, tma_full_addr + (load_stage) * 8);
                            mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 24576);
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int mma_cluster_id = bid;
            unsigned int mma_num_clusters = num_bids;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int item_1 = mma_cluster_id; item_1 < full_tiles + 2 * (num_cluster_tiles - full_tiles); item_1 += mma_num_clusters) {
                int h_item_3 = item_1;
                int h_rel_4 = h_item_3 - full_tiles;
                int h_is_1 = ((h_rel_4 >= 0) ? 1 : 0);
                int h_relc_1 = h_rel_4 * h_is_1;
                int h_t_1 = h_relc_1 / 2;
                int h_half_1 = h_relc_1 - h_t_1 * 2;
                int h_ct_1 = h_item_3 + h_is_1 * (full_tiles + h_t_1 - h_item_3);
                if (h_is_1 == 0) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_3 = 0; iter_k_3 < 24; iter_k_3++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_3 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, ((init_flag) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                            }
                        }
                        elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                } else {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_4 = 0; iter_k_4 < 24; iter_k_4++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1 = ((iter_k_4 == 0) ? 1 : 0);
                        int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_b_h_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, ((init_flag_1) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                            }
                        }
                        elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            const int epi_half = (warp - 2) / 4;
            const int grp_lo = epi_half;
            const int grp_lo_h = epi_half;
            unsigned int epi_cluster_id = bid;
            unsigned int epi_num_clusters = num_bids;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int item_2 = epi_cluster_id; item_2 < full_tiles + 2 * (num_cluster_tiles - full_tiles); item_2 += epi_num_clusters) {
                int h_item_4 = item_2;
                int h_rel_5 = h_item_4 - full_tiles;
                int h_is_5 = ((h_rel_5 >= 0) ? 1 : 0);
                int h_relc_2 = h_rel_5 * h_is_5;
                int h_t_2 = h_relc_2 / 2;
                int h_half_2 = h_relc_2 - h_t_2 * 2;
                int h_ct_2 = h_item_4 + h_is_5 * (full_tiles + h_t_2 - h_item_4);
                if (h_is_5 == 0) {
                    int group_1 = h_ct_2 / tiles_per_group;
                    int first_m_1 = group_1 * 32;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= 32) ? 32 : remaining_1);
                    int local_1 = h_ct_2 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int global_row = bid_m_1 * 128 + local_row;
                    int off_n = bid_n_1 * 128;
                    int safe_row = ((global_row < M) ? global_row : M - 1);
                    int store_row = global_row;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                    unsigned long long pf_row_in = (unsigned long long)safe_row * 1024 + (unsigned long long)off_n;
                    unsigned int res_pf[32];
                    #pragma unroll
                    for (int g = 0; g < 1; g++) {
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(res_pf[g * 32 + q * 8 + 0]), "=r"(res_pf[g * 32 + q * 8 + 1]), "=r"(res_pf[g * 32 + q * 8 + 2]), "=r"(res_pf[g * 32 + q * 8 + 3]), "=r"(res_pf[g * 32 + q * 8 + 4]), "=r"(res_pf[g * 32 + q * 8 + 5]), "=r"(res_pf[g * 32 + q * 8 + 6]), "=r"(res_pf[g * 32 + q * 8 + 7]) : "l"((const void*)((const char*)(R + (pf_row_in + (unsigned long long)((grp_lo + g) * 64 + q * 16)) + 0) + 0)) : "memory");
                            }
                        }
                    }
                    unsigned int wn_pf[32];
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 4; q_1++) {
                        {
                            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(wn_pf[q_1 * 8 + 0]), "=r"(wn_pf[q_1 * 8 + 1]), "=r"(wn_pf[q_1 * 8 + 2]), "=r"(wn_pf[q_1 * 8 + 3]), "=r"(wn_pf[q_1 * 8 + 4]), "=r"(wn_pf[q_1 * 8 + 5]), "=r"(wn_pf[q_1 * 8 + 6]), "=r"(wn_pf[q_1 * 8 + 7]) : "l"((const void*)((const char*)(WN + ((unsigned long long)off_n + (unsigned long long)(grp_lo * 64 + q_1 * 16)) + 0) + 0)) : "memory");
                        }
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out = (unsigned long long)global_row * 1024 + (unsigned long long)off_n;
                    unsigned long long row_in = (unsigned long long)safe_row * 1024 + (unsigned long long)off_n;
                    unsigned long long sq_out_off = (unsigned long long)global_row * 16 + (unsigned long long)(off_n / 64);
                    unsigned long long wn_off = (unsigned long long)off_n;
                    #pragma unroll 1
                    for (int g_i = 0; g_i < 1; g_i++) {
                        int grp = grp_lo + g_i;
                        int gcol = grp * 64;
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(lane_addr + gcol));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                            : "r"(lane_addr + gcol + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float res[64];
                        unsigned int wn_w[32];
                        float part_sq = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            float vals[16];
                            unsigned int rw[8];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                rw[j] = res_pf[g_i * 32 + c * 8 + j];
                            }
                            float rw_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&rw_f32[_pair * 2])[0]), "=f"((&rw_f32[_pair * 2])[1])
                                    : "r"(rw[_pair]));
                            }
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 16; j_1++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[c * 16 + j_1]);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                vals[j_1] = rw_f32[j_1] + _cvt_f32_0;
                            }
                            uint32_t vals_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                                vals_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            float vals_bf16_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&vals_bf16_f32[_pair * 2])[0]), "=f"((&vals_bf16_f32[_pair * 2])[1])
                                    : "r"(vals_bf16[_pair]));
                            }
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 16; j_2++) {
                                part_sq += vals_bf16_f32[j_2] * vals_bf16_f32[j_2];
                            }
                            if (store_row < M) {
                                {
                                    const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(vals_bf16);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(C + (row_out + (unsigned long long)(gcol + c * 16)))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                                }
                            }
                            unsigned int xw_w[8];
                            #pragma unroll
                            for (int q_2 = 0; q_2 < 8; q_2++) {
                                uint32_t _bf16x2_mul_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(vals_bf16[q_2]), "r"(wn_pf[c * 8 + q_2]));
                                xw_w[q_2] = _bf16x2_mul_0;
                            }
                            if (store_row < M) {
                                {
                                    const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(xw_w);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(XW + (row_out + (unsigned long long)(gcol + c * 16)))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                                }
                            }
                        }
                        if (store_row < M) {
                            SQ[sq_out_off + (unsigned long long)grp] = part_sq;
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                } else {
                    int group_2 = h_ct_2 / tiles_per_group;
                    int first_m_2 = group_2 * 32;
                    int remaining_2 = m_tiles - first_m_2;
                    int group_size_2 = ((remaining_2 >= 32) ? 32 : remaining_2);
                    int local_2 = h_ct_2 % tiles_per_group;
                    int bid_m_2 = first_m_2 + local_2 % group_size_2;
                    int bid_n_2 = local_2 / group_size_2;
                    int global_row_1 = bid_m_2 * 128 + local_row;
                    int off_n_1 = bid_n_2 * 128 + h_half_2 * 64;
                    int safe_row_1 = ((global_row_1 < M) ? global_row_1 : M - 1);
                    int store_row_1 = global_row_1;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                    unsigned long long pf_row_in_1 = (unsigned long long)safe_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned int res_pf_1[16];
                    #pragma unroll
                    for (int g_1 = 0; g_1 < 1; g_1++) {
                        #pragma unroll
                        for (int q_3 = 0; q_3 < 2; q_3++) {
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 0]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 1]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 2]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 3]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 4]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 5]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 6]), "=r"(res_pf_1[g_1 * 16 + q_3 * 8 + 7]) : "l"((const void*)((const char*)(R + (pf_row_in_1 + (unsigned long long)((grp_lo_h + g_1) * 32 + q_3 * 16)) + 0) + 0)) : "memory");
                            }
                        }
                    }
                    unsigned int wn_pf_1[16];
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 2; q_4++) {
                        {
                            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(wn_pf_1[q_4 * 8 + 0]), "=r"(wn_pf_1[q_4 * 8 + 1]), "=r"(wn_pf_1[q_4 * 8 + 2]), "=r"(wn_pf_1[q_4 * 8 + 3]), "=r"(wn_pf_1[q_4 * 8 + 4]), "=r"(wn_pf_1[q_4 * 8 + 5]), "=r"(wn_pf_1[q_4 * 8 + 6]), "=r"(wn_pf_1[q_4 * 8 + 7]) : "l"((const void*)((const char*)(WN + ((unsigned long long)off_n_1 + (unsigned long long)(grp_lo_h * 32 + q_4 * 16)) + 0) + 0)) : "memory");
                        }
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out_1 = (unsigned long long)global_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned long long row_in_1 = (unsigned long long)safe_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned long long sq_out_off_1 = (unsigned long long)global_row_1 * 16 + (unsigned long long)(off_n_1 / 64);
                    unsigned long long wn_off_1 = (unsigned long long)off_n_1;
                    #pragma unroll 1
                    for (int g_i_1 = 0; g_i_1 < 1; g_i_1++) {
                        int grp_1 = grp_lo_h + g_i_1;
                        int gcol_1 = grp_1 * 32;
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(lane_addr_1 + gcol_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float res_1[32];
                        unsigned int wn_w_1[16];
                        float part_sq_1 = 0.0f;
                        unsigned int keep_w[16];
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 2; c_1++) {
                            float vals_1[16];
                            unsigned int rw_1[8];
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 8; j_3++) {
                                rw_1[j_3] = res_pf_1[g_i_1 * 16 + c_1 * 8 + j_3];
                            }
                            float rw_f32_1[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&rw_f32_1[_pair * 2])[0]), "=f"((&rw_f32_1[_pair * 2])[1])
                                    : "r"(rw_1[_pair]));
                            }
                            #pragma unroll
                            for (int j_4 = 0; j_4 < 16; j_4++) {
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_1[c_1 * 16 + j_4]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                vals_1[j_4] = rw_f32_1[j_4] + _cvt_f32_1;
                            }
                            uint32_t vals_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                                vals_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            float vals_bf16_f32_1[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&vals_bf16_f32_1[_pair * 2])[0]), "=f"((&vals_bf16_f32_1[_pair * 2])[1])
                                    : "r"(vals_bf16_1[_pair]));
                            }
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 16; j_5++) {
                                part_sq_1 += vals_bf16_f32_1[j_5] * vals_bf16_f32_1[j_5];
                            }
                            #pragma unroll
                            for (int q_5 = 0; q_5 < 8; q_5++) {
                                keep_w[c_1 * 8 + q_5] = vals_bf16_1[q_5];
                            }
                            if (store_row_1 < M) {
                                {
                                    const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(vals_bf16_1);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(C + (row_out_1 + (unsigned long long)(gcol_1 + c_1 * 16)))), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
                                }
                            }
                            unsigned int xw_w_1[8];
                            #pragma unroll
                            for (int q_6 = 0; q_6 < 8; q_6++) {
                                uint32_t _bf16x2_mul_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(vals_bf16_1[q_6]), "r"(wn_pf_1[c_1 * 8 + q_6]));
                                xw_w_1[q_6] = _bf16x2_mul_1;
                            }
                            if (store_row_1 < M) {
                                {
                                    const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(xw_w_1);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(XW + (row_out_1 + (unsigned long long)(gcol_1 + c_1 * 16)))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                                }
                            }
                        }
                        if (epi_half == 0) {
                            float p0 = 0.0f;
                            #pragma unroll
                            for (int c_2 = 0; c_2 < 2; c_2++) {
                                unsigned int w0[8];
                                #pragma unroll
                                for (int q_7 = 0; q_7 < 8; q_7++) {
                                    w0[q_7] = keep_w[c_2 * 8 + q_7];
                                }
                                float w0_f32[16];
                                #pragma unroll
                                for (int _pair = 0; _pair < 8; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&w0_f32[_pair * 2])[0]), "=f"((&w0_f32[_pair * 2])[1])
                                        : "r"(w0[_pair]));
                                }
                                #pragma unroll
                                for (int j_6 = 0; j_6 < 16; j_6++) {
                                    p0 += w0_f32[j_6] * w0_f32[j_6];
                                }
                            }
                            sq_smem[epi_stage * 128 + (unsigned int)local_row] = p0;
                        }
                        asm volatile("bar.sync 1, 256;" ::: "memory");
                        if (epi_half == 1) {
                            float p1 = sq_smem[epi_stage * 128 + (unsigned int)local_row];
                            #pragma unroll
                            for (int c_3 = 0; c_3 < 2; c_3++) {
                                unsigned int w1[8];
                                #pragma unroll
                                for (int q_8 = 0; q_8 < 8; q_8++) {
                                    w1[q_8] = keep_w[c_3 * 8 + q_8];
                                }
                                float w1_f32[16];
                                #pragma unroll
                                for (int _pair = 0; _pair < 8; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&w1_f32[_pair * 2])[0]), "=f"((&w1_f32[_pair * 2])[1])
                                        : "r"(w1[_pair]));
                                }
                                #pragma unroll
                                for (int j_7 = 0; j_7 < 16; j_7++) {
                                    p1 += w1_f32[j_7] * w1_f32[j_7];
                                }
                            }
                            if (store_row_1 < M) {
                                SQ[sq_out_off_1] = p1;
                            }
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
