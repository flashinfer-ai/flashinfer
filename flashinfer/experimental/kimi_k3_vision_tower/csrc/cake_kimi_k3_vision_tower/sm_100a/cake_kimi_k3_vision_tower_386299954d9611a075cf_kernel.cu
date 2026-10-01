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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_H_OFF 17408
#define SMEM_SMEM_B_H_STAGE_BYTES 8192
#define SMEM_SMEM_B_H_STRIDE 32768
#define SMEM_TOTAL 230400
#define THREADS 320
#define num_cluster_tiles ((m_tiles / 2) * 4)
#define tiles_per_group 128

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_vision_tower_386299954d9611a075cf(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ C2, __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ R, float* __restrict__ COS, float* __restrict__ SIN, __nv_bfloat16* __restrict__ CS, float* __restrict__ SQ, __nv_bfloat16* __restrict__ XW, __nv_bfloat16* __restrict__ WN, float* __restrict__ WS, unsigned int* __restrict__ FLAGS, const __grid_constant__ CUtensorMap RT, const __grid_constant__ CUtensorMap CT, const __grid_constant__ CUtensorMap XWT, int M, int m_tiles, int full_tiles, int tail_split, int pf_l2, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
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
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int cluster_id_1 = bid / 2;
            unsigned int num_clusters_1 = num_bids / 2;
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
                int pair_bid = h_ct * 2 + cta_rank;
                int group = pair_bid / tiles_per_group;
                int first_m = group * 32;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= 32) ? 32 : remaining);
                int local = pair_bid % tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int e_off_m = bid_m * 128;
                int e_wrow = bid_n * 256 + cta_rank * 128;
                #pragma unroll 1
                for (int e_s = 0; e_s < 7; e_s++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int e_k = k_begin + e_s;
                    tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768, (&B), 0, e_wrow, e_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, e_wrow + 64, e_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll
                for (int e_s2 = 0; e_s2 < 7; e_s2++) {
                    tma_3d_gmem2smem_cta2(smem_a_addr + (unsigned int)(e_s2 * 32768), (&A), 0, e_off_m, k_begin + e_s2, ((tma_full_addr + (e_s2) * 8) & 0xFEFFFFFF));
                }
                int h_item_0 = cluster_id_1;
                int h_rel_1 = h_item_0 - full_tiles;
                int h_is_2 = ((h_rel_1 >= 0) ? 1 : 0);
                int h_relc_3 = h_rel_1 * h_is_2;
                int h_t_4 = h_relc_3 / 2;
                int h_half_5 = h_relc_3 - h_t_4 * 2;
                int h_ct_6 = h_item_0 + h_is_2 * (full_tiles + h_t_4 - h_item_0);
                int pair_bid_7 = h_ct_6 * 2 + cta_rank;
                int group_8 = pair_bid_7 / tiles_per_group;
                int first_m_9 = group_8 * 32;
                int remaining_10 = m_tiles - first_m_9;
                int group_size_11 = ((remaining_10 >= 32) ? 32 : remaining_10);
                int local_12 = pair_bid_7 % tiles_per_group;
                int bid_m_13 = first_m_9 + local_12 % group_size_11;
                int bid_n_14 = local_12 / group_size_11;
                int off_m = bid_m_13 * 128;
                int weight_row = bid_n_14 * 256 + cta_rank * 128;
                if (pf_l2 != 0) {
                    #pragma unroll 1
                    for (int pk = 7; pk < 64; pk++) {
                        int pk_step = k_begin + pk;
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m)), "r"((int)(pk_step)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row)), "r"((int)(pk_step)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int iter_k = 7; iter_k < 64; iter_k++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int k_step = k_begin + iter_k;
                    tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, weight_row + 64, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
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
                        int pair_bid_9 = h_ct_8 * 2 + cta_rank;
                        int group_10 = pair_bid_9 / tiles_per_group;
                        int first_m_11 = group_10 * 32;
                        int remaining_12 = m_tiles - first_m_11;
                        int group_size_13 = ((remaining_12 >= 32) ? 32 : remaining_12);
                        int local_14 = pair_bid_9 % tiles_per_group;
                        int bid_m_15 = first_m_11 + local_14 % group_size_13;
                        int bid_n_16 = local_14 / group_size_13;
                        int off_m_17 = bid_m_15 * 128;
                        int weight_row_18 = bid_n_16 * 256 + cta_rank * 128;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk_1 = 7; pk_1 < 64; pk_1++) {
                                int pk_step_1 = k_begin + pk_1;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m_17)), "r"((int)(pk_step_1)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row_18)), "r"((int)(pk_step_1)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < 64; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step_1 = k_begin + iter_k_1;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m_17, k_step_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row_18, k_step_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768 + 8192, (&B), 0, weight_row_18 + 64, k_step_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
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
                        int pair_bid_9_1 = h_ct_8_1 * 2 + cta_rank;
                        int group_10_1 = pair_bid_9_1 / tiles_per_group;
                        int first_m_11_1 = group_10_1 * 32;
                        int remaining_12_1 = m_tiles - first_m_11_1;
                        int group_size_13_1 = ((remaining_12_1 >= 32) ? 32 : remaining_12_1);
                        int local_14_1 = pair_bid_9_1 % tiles_per_group;
                        int bid_m_15_1 = first_m_11_1 + local_14_1 % group_size_13_1;
                        int bid_n_16_1 = local_14_1 / group_size_13_1;
                        int off_m_17_1 = bid_m_15_1 * 128;
                        int weight_row_18_1 = bid_n_16_1 * 256 + h_half_6 * 128 + cta_rank * 64;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk_2 = 7; pk_2 < 64; pk_2++) {
                                int pk_step_2 = k_begin + pk_2;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m_17_1)), "r"((int)(pk_step_2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row_18_1)), "r"((int)(pk_step_2)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < 64; iter_k_2++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step_2 = k_begin + iter_k_2;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m_17_1, k_step_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 32768, (&B), 0, weight_row_18_1, k_step_2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(24576)) : "memory");
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
            unsigned int mma_cluster_id = bid / 2;
            unsigned int mma_num_clusters = num_bids / 2;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
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
                        for (int iter_k_3 = 0; iter_k_3 < 64; iter_k_3++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_3 == 0) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    } else {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_4 = 0; iter_k_4 < 64; iter_k_4++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag_1 = ((iter_k_4 == 0) ? 1 : 0);
                            int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            int _mma_b_lo_1 = (((smem_b_h_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag_1) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    }
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
            const int grp_lo = epi_half * 2;
            const int grp_lo_h = epi_half;
            unsigned int epi_cluster_id = bid / 2;
            unsigned int epi_num_clusters = num_bids / 2;
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
                    int pair_bid_1 = h_ct_2 * 2 + cta_rank;
                    int group_1 = pair_bid_1 / tiles_per_group;
                    int first_m_1 = group_1 * 32;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= 32) ? 32 : remaining_1);
                    int local_1 = pair_bid_1 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int global_row = bid_m_1 * 128 + local_row;
                    int off_n = bid_n_1 * 256;
                    int safe_row = ((global_row < M) ? global_row : M - 1);
                    int store_row = global_row;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 256;
                    unsigned long long pf_row_in = (unsigned long long)safe_row * 1024 + (unsigned long long)off_n;
                    unsigned int res_pf[64];
                    #pragma unroll
                    for (int g = 0; g < 2; g++) {
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(res_pf[g * 32 + q * 8 + 0]), "=r"(res_pf[g * 32 + q * 8 + 1]), "=r"(res_pf[g * 32 + q * 8 + 2]), "=r"(res_pf[g * 32 + q * 8 + 3]), "=r"(res_pf[g * 32 + q * 8 + 4]), "=r"(res_pf[g * 32 + q * 8 + 5]), "=r"(res_pf[g * 32 + q * 8 + 6]), "=r"(res_pf[g * 32 + q * 8 + 7]) : "l"((const void*)((const char*)(R + (pf_row_in + (unsigned long long)((grp_lo + g) * 64 + q * 16)) + 0) + 0)) : "memory");
                            }
                        }
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out = (unsigned long long)global_row * 1024 + (unsigned long long)off_n;
                    unsigned long long row_in = (unsigned long long)safe_row * 1024 + (unsigned long long)off_n;
                    #pragma unroll 1
                    for (int g_i = 0; g_i < 2; g_i++) {
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
                            if (store_row < M) {
                                {
                                    const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(vals_bf16);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(C + (row_out + (unsigned long long)(gcol + c * 16)))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                } else {
                    int pair_bid_2 = h_ct_2 * 2 + cta_rank;
                    int group_2 = pair_bid_2 / tiles_per_group;
                    int first_m_2 = group_2 * 32;
                    int remaining_2 = m_tiles - first_m_2;
                    int group_size_2 = ((remaining_2 >= 32) ? 32 : remaining_2);
                    int local_2 = pair_bid_2 % tiles_per_group;
                    int bid_m_2 = first_m_2 + local_2 % group_size_2;
                    int bid_n_2 = local_2 / group_size_2;
                    int global_row_1 = bid_m_2 * 128 + local_row;
                    int off_n_1 = bid_n_2 * 256 + h_half_2 * 128;
                    int safe_row_1 = ((global_row_1 < M) ? global_row_1 : M - 1);
                    int store_row_1 = global_row_1;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 256;
                    unsigned long long pf_row_in_1 = (unsigned long long)safe_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned int res_pf_1[32];
                    #pragma unroll
                    for (int g_1 = 0; g_1 < 1; g_1++) {
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 4; q_1++) {
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 0]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 1]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 2]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 3]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 4]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 5]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 6]), "=r"(res_pf_1[g_1 * 32 + q_1 * 8 + 7]) : "l"((const void*)((const char*)(R + (pf_row_in_1 + (unsigned long long)((grp_lo_h + g_1) * 64 + q_1 * 16)) + 0) + 0)) : "memory");
                            }
                        }
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out_1 = (unsigned long long)global_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned long long row_in_1 = (unsigned long long)safe_row_1 * 1024 + (unsigned long long)off_n_1;
                    #pragma unroll 1
                    for (int g_i_1 = 0; g_i_1 < 1; g_i_1++) {
                        int grp_1 = grp_lo_h + g_i_1;
                        int gcol_1 = grp_1 * 64;
                        float _tmem_load_1[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(lane_addr_1 + gcol_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                            : "r"(lane_addr_1 + gcol_1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float res_1[64];
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 4; c_1++) {
                            float vals_1[16];
                            unsigned int rw_1[8];
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 8; j_2++) {
                                rw_1[j_2] = res_pf_1[g_i_1 * 32 + c_1 * 8 + j_2];
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
                            for (int j_3 = 0; j_3 < 16; j_3++) {
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_1[c_1 * 16 + j_3]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                vals_1[j_3] = rw_f32_1[j_3] + _cvt_f32_1;
                            }
                            uint32_t vals_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                                vals_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            if (store_row_1 < M) {
                                {
                                    const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(vals_bf16_1);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(C + (row_out_1 + (unsigned long long)(gcol_1 + c_1 * 16)))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
