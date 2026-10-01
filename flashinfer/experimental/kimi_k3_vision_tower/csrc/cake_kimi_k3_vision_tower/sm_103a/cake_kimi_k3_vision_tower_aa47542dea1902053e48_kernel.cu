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
#define NUM_TMA_PIPE_STAGES 8
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 24576
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 24576
#define SMEM_SMEM_B_H_OFF 17408
#define SMEM_SMEM_B_H_STAGE_BYTES 4096
#define SMEM_SMEM_B_H_STRIDE 24576
#define SMEM_EPI_RES_OFF 197632
#define SMEM_EPI_RES_STAGE_BYTES 16384
#define SMEM_EPI_RES_STRIDE 16384
#define SMEM_EPI_RES_W_OFF 197632
#define SMEM_EPI_RES_W_STAGE_BYTES 32768
#define SMEM_EPI_RES_W_STRIDE 32768
#define SMEM_TOTAL 230400
#define THREADS 192
#define num_cluster_tiles ((m_tiles / 2) * 8)
#define tiles_per_group 256

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_vision_tower_aa47542dea1902053e48(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ C2, __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ R, float* __restrict__ COS, float* __restrict__ SIN, __nv_bfloat16* __restrict__ CS, float* __restrict__ SQ, __nv_bfloat16* __restrict__ XW, __nv_bfloat16* __restrict__ WN, float* __restrict__ WS, unsigned int* __restrict__ FLAGS, const __grid_constant__ CUtensorMap RT, const __grid_constant__ CUtensorMap CT, const __grid_constant__ CUtensorMap XWT, int M, int m_tiles, int full_tiles, int tail_split, int pf_l2, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 64)
    #define mainloop_done_addr (mbar_base + 128)
    #define epilogue_done_addr (mbar_base + 144)
    #define res_full_addr (mbar_base + 160)

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
    __nv_bfloat16* epi_res = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int epi_res_addr = smem + 197632;
    unsigned int* epi_res_w = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int epi_res_w_addr = smem + 197632;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 8 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // mma_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            // res_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 0) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
                #pragma unroll 1
                for (unsigned int item = cluster_id_1; item < full_tiles + 2 * (num_cluster_tiles - full_tiles); item += num_clusters_1) {
                    int h_item = item;
                    int h_rel = h_item - full_tiles;
                    int h_is = ((h_rel >= 0) ? 1 : 0);
                    int h_relc = h_rel * h_is;
                    int h_t = h_relc / 2;
                    int h_half = h_relc - h_t * 2;
                    int h_ct = h_item + h_is * (full_tiles + h_t - h_item);
                    if (h_is == 0) {
                        int h_item_0 = item;
                        int h_rel_1 = h_item_0 - full_tiles;
                        int h_is_2 = ((h_rel_1 >= 0) ? 1 : 0);
                        int h_relc_3 = h_rel_1 * h_is_2;
                        int h_t_4 = h_relc_3 / 2;
                        int h_half_5 = h_relc_3 - h_t_4 * 2;
                        int h_ct_6 = h_item_0 + h_is_2 * (full_tiles + h_t_4 - h_item_0);
                        int pair_bid = h_ct_6 * 2 + cta_rank;
                        int group = pair_bid / tiles_per_group;
                        int first_m = group * 32;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= 32) ? 32 : remaining);
                        int local = pair_bid % tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * 128;
                        int weight_row = bid_n * 128 + cta_rank * 64;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk = 8; pk < 24; pk++) {
                                int pk_step = k_begin + pk;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m)), "r"((int)(pk_step)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row)), "r"((int)(pk_step)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < 24; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step = k_begin + iter_k;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 24576, (&A), 0, off_m, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 24576, (&B), 0, weight_row, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 24576 + 4096, (&B), 0, weight_row + 32, k_step, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(24576)) : "memory");
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int h_item_0_1 = item;
                        int h_rel_1_1 = h_item_0_1 - full_tiles;
                        int h_is_2_1 = ((h_rel_1_1 >= 0) ? 1 : 0);
                        int h_relc_3_1 = h_rel_1_1 * h_is_2_1;
                        int h_t_4_1 = h_relc_3_1 / 2;
                        int h_half_5_1 = h_relc_3_1 - h_t_4_1 * 2;
                        int h_ct_6_1 = h_item_0_1 + h_is_2_1 * (full_tiles + h_t_4_1 - h_item_0_1);
                        int pair_bid_1 = h_ct_6_1 * 2 + cta_rank;
                        int group_1 = pair_bid_1 / tiles_per_group;
                        int first_m_1 = group_1 * 32;
                        int remaining_1 = m_tiles - first_m_1;
                        int group_size_1 = ((remaining_1 >= 32) ? 32 : remaining_1);
                        int local_1 = pair_bid_1 % tiles_per_group;
                        int bid_m_1 = first_m_1 + local_1 % group_size_1;
                        int bid_n_1 = local_1 / group_size_1;
                        int off_m_1 = bid_m_1 * 128;
                        int weight_row_1 = bid_n_1 * 128 + h_half * 64 + cta_rank * 32;
                        if (pf_l2 != 0) {
                            #pragma unroll 1
                            for (int pk_1 = 8; pk_1 < 24; pk_1++) {
                                int pk_step_1 = k_begin + pk_1;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m_1)), "r"((int)(pk_step_1)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row_1)), "r"((int)(pk_step_1)) : "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < 24; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_step_1 = k_begin + iter_k_1;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 24576, (&A), 0, off_m_1, k_step_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_h_addr + load_stage * 24576, (&B), 0, weight_row_1, k_step_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(20480)) : "memory");
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; _phase_mma_done ^= 1; }
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
                    int h_item_1 = item_1;
                    int h_rel_2 = h_item_1 - full_tiles;
                    int h_is_1 = ((h_rel_2 >= 0) ? 1 : 0);
                    int h_relc_1 = h_rel_2 * h_is_1;
                    int h_t_1 = h_relc_1 / 2;
                    int h_half_1 = h_relc_1 - h_t_1 * 2;
                    int h_ct_1 = h_item_1 + h_is_1 * (full_tiles + h_t_1 - h_item_1);
                    if (h_is_1 == 0) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < 24; iter_k_2++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_2 == 0) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 128))), "r"(((init_flag) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 8) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    } else {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_3 = 0; iter_k_3 < 24; iter_k_3++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag_1 = ((iter_k_3 == 0) ? 1 : 0);
                            int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
                            int _mma_b_lo_1 = (((smem_b_h_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 1536;
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
                    "mov.b32 id, 269485200;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (mma_epi_stage * 128))), "r"(((init_flag_1) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 8) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
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
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            const int epi_half = (warp - 2) / 4;
            const int grp_lo = epi_half * 2;
            const int grp_lo_h = epi_half;
            unsigned int epi_cluster_id = bid / 2;
            unsigned int epi_num_clusters = num_bids / 2;
            if (warp == 2) {
                if (elect_sync()) {
                    if (epi_cluster_id < (unsigned int)(full_tiles + 2 * (num_cluster_tiles - full_tiles))) {
                        int h_item_2 = epi_cluster_id;
                        int h_rel_3 = h_item_2 - full_tiles;
                        int h_is_3 = ((h_rel_3 >= 0) ? 1 : 0);
                        int h_relc_2 = h_rel_3 * h_is_3;
                        int h_t_2 = h_relc_2 / 2;
                        int h_half_2 = h_relc_2 - h_t_2 * 2;
                        int h_ct_2 = h_item_2 + h_is_3 * (full_tiles + h_t_2 - h_item_2);
                        int pair_bid_2 = h_ct_2 * 2 + cta_rank;
                        int group_2 = pair_bid_2 / tiles_per_group;
                        int first_m_2 = group_2 * 32;
                        int remaining_2 = m_tiles - first_m_2;
                        int group_size_2 = ((remaining_2 >= 32) ? 32 : remaining_2);
                        int local_2 = pair_bid_2 % tiles_per_group;
                        int bid_m_2 = first_m_2 + local_2 % group_size_2;
                        int bid_n_2 = local_2 / group_size_2;
                        int n_off_m = bid_m_2 * 128;
                        int n_chunk0 = (bid_n_2 * 128 + h_half_2 * 64) / 64;
                        if (h_is_3 == 0) {
                            mbarrier_arrive_expect_tx(res_full_addr, 32768);
                            #pragma unroll
                            for (int j = 0; j < 2; j++) {
                                tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j * 16384), (&RT), 0, n_off_m, n_chunk0 + j, res_full_addr);
                            }
                        } else {
                            mbarrier_arrive_expect_tx(res_full_addr, 16384);
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 1; j_1++) {
                                tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j_1 * 16384), (&RT), 0, n_off_m, n_chunk0 + j_1, res_full_addr);
                            }
                        }
                    }
                }
            }
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_res_full_0 = 0;
            #pragma unroll 1
            for (unsigned int item_2 = epi_cluster_id; item_2 < full_tiles + 2 * (num_cluster_tiles - full_tiles); item_2 += epi_num_clusters) {
                int h_item_3 = item_2;
                int h_rel_4 = h_item_3 - full_tiles;
                int h_is_4 = ((h_rel_4 >= 0) ? 1 : 0);
                int h_relc_4 = h_rel_4 * h_is_4;
                int h_t_3 = h_relc_4 / 2;
                int h_half_3 = h_relc_4 - h_t_3 * 2;
                int h_ct_3 = h_item_3 + h_is_4 * (full_tiles + h_t_3 - h_item_3);
                if (h_is_4 == 0) {
                    int pair_bid_3 = h_ct_3 * 2 + cta_rank;
                    int group_3 = pair_bid_3 / tiles_per_group;
                    int first_m_3 = group_3 * 32;
                    int remaining_3 = m_tiles - first_m_3;
                    int group_size_3 = ((remaining_3 >= 32) ? 32 : remaining_3);
                    int local_3 = pair_bid_3 % tiles_per_group;
                    int bid_m_3 = first_m_3 + local_3 % group_size_3;
                    int bid_n_3 = local_3 / group_size_3;
                    int global_row = bid_m_3 * 128 + local_row;
                    int off_n = bid_n_3 * 128;
                    int safe_row = ((global_row < M) ? global_row : M - 1);
                    int store_row = global_row;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(res_full_addr, _phase_res_full_0);
                    _phase_res_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out = (unsigned long long)global_row * 1024 + (unsigned long long)off_n;
                    unsigned long long row_in = (unsigned long long)safe_row * 1024 + (unsigned long long)off_n;
                    unsigned long long sq_out_off = (unsigned long long)global_row * 16 + (unsigned long long)(off_n / 64);
                    unsigned long long wn_off = (unsigned long long)off_n;
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
                        unsigned int wn_w[32];
                        float part_sq = 0.0f;
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            float vals[16];
                            int tb = (gcol + c * 16) / 64;
                            int tcb = (gcol + c * 16) % 64 * 2;
                            int tsw0 = (tcb ^ (local_row & 7) << 4);
                            int tsw1 = (tcb + 16 ^ (local_row & 7) << 4);
                            int tbase = tb * 4096 + local_row * 32;
                            unsigned int rw[8];
                            uint32_t _epi_res_w_reg_0[4];
                            __int128_t _smem_b128_0;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_0) : "r"(epi_res_w_addr + (tbase + tsw0 / 4) * 4));
                            _epi_res_w_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[0];
                            _epi_res_w_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[1];
                            _epi_res_w_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[2];
                            _epi_res_w_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_0)[3];
                            uint32_t _epi_res_w_reg_1[4];
                            __int128_t _smem_b128_1;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_1) : "r"(epi_res_w_addr + (tbase + tsw1 / 4) * 4));
                            _epi_res_w_reg_1[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[0];
                            _epi_res_w_reg_1[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[1];
                            _epi_res_w_reg_1[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[2];
                            _epi_res_w_reg_1[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_1)[3];
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 4; j_2++) {
                                rw[j_2] = _epi_res_w_reg_0[j_2];
                                rw[4 + j_2] = _epi_res_w_reg_1[j_2];
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
                            for (int j_3 = 0; j_3 < 16; j_3++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[c * 16 + j_3]);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                vals[j_3] = rw_f32[j_3] + _cvt_f32_0;
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
                            for (int j_4 = 0; j_4 < 16; j_4++) {
                                part_sq += vals_bf16_f32[j_4] * vals_bf16_f32[j_4];
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((tb * 128 + local_row) * 128 + tcb ^ ((tb * 128 + local_row) * 128 + tcb >> 7 & 7) << 4))), "r"(vals_bf16[0]), "r"(vals_bf16[1]), "r"(vals_bf16[2]), "r"(vals_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((tb * 128 + local_row) * 128 + (tcb + 16) ^ ((tb * 128 + local_row) * 128 + (tcb + 16) >> 7 & 7) << 4))), "r"(vals_bf16[4]), "r"(vals_bf16[5]), "r"(vals_bf16[6]), "r"(vals_bf16[7]) : "memory");
                        }
                        if (store_row < M) {
                            SQ[sq_out_off + (unsigned long long)grp] = part_sq;
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            int st_off_m = bid_m_3 * 128;
                            int st_chunk0 = off_n / 64;
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 2; j_5++) {
                                tma_store_3d((&CT), 0, st_off_m, st_chunk0 + j_5, epi_res_addr + (unsigned int)(j_5 * 16384));
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    unsigned long long pw_off = (unsigned long long)off_n;
                    #pragma unroll
                    for (int pg = 0; pg < 2; pg++) {
                        #pragma unroll
                        for (int pc = 0; pc < 4; pc++) {
                            int pcol = (grp_lo + pg) * 64 + pc * 16;
                            int ptb = pcol / 64;
                            int ptcb = pcol % 64 * 2;
                            int ptsw0 = (ptcb ^ (local_row & 7) << 4);
                            int ptsw1 = (ptcb + 16 ^ (local_row & 7) << 4);
                            int ptbase = ptb * 4096 + local_row * 32;
                            uint32_t _epi_res_w_reg_2[4];
                            __int128_t _smem_b128_2;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(epi_res_w_addr + (ptbase + ptsw0 / 4) * 4));
                            _epi_res_w_reg_2[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                            _epi_res_w_reg_2[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                            _epi_res_w_reg_2[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                            _epi_res_w_reg_2[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                            uint32_t _epi_res_w_reg_3[4];
                            __int128_t _smem_b128_3;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(epi_res_w_addr + (ptbase + ptsw1 / 4) * 4));
                            _epi_res_w_reg_3[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                            _epi_res_w_reg_3[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                            _epi_res_w_reg_3[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                            _epi_res_w_reg_3[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                            unsigned int pwn[8];
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(pwn[0 + 0]), "=r"(pwn[0 + 1]), "=r"(pwn[0 + 2]), "=r"(pwn[0 + 3]), "=r"(pwn[0 + 4]), "=r"(pwn[0 + 5]), "=r"(pwn[0 + 6]), "=r"(pwn[0 + 7]) : "l"((const void*)((const char*)(WN + (pw_off + (unsigned long long)pcol) + 0) + 0)) : "memory");
                            }
                            unsigned int pxw[8];
                            #pragma unroll
                            for (int q = 0; q < 4; q++) {
                                uint32_t _bf16x2_mul_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_epi_res_w_reg_2[q]), "r"(pwn[q]));
                                pxw[q] = _bf16x2_mul_0;
                                uint32_t _bf16x2_mul_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_epi_res_w_reg_3[q]), "r"(pwn[4 + q]));
                                pxw[4 + q] = _bf16x2_mul_1;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((ptb * 128 + local_row) * 128 + ptcb ^ ((ptb * 128 + local_row) * 128 + ptcb >> 7 & 7) << 4))), "r"(pxw[0]), "r"(pxw[1]), "r"(pxw[2]), "r"(pxw[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((ptb * 128 + local_row) * 128 + (ptcb + 16) ^ ((ptb * 128 + local_row) * 128 + (ptcb + 16) >> 7 & 7) << 4))), "r"(pxw[4]), "r"(pxw[5]), "r"(pxw[6]), "r"(pxw[7]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            int xw_off_m = bid_m_3 * 128;
                            int xw_chunk0 = off_n / 64;
                            #pragma unroll
                            for (int j_6 = 0; j_6 < 2; j_6++) {
                                tma_store_3d((&XWT), 0, xw_off_m, xw_chunk0 + j_6, epi_res_addr + (unsigned int)(j_6 * 16384));
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            unsigned int xw_next = item_2 + epi_num_clusters;
                            if (xw_next < (unsigned int)(full_tiles + 2 * (num_cluster_tiles - full_tiles))) {
                                int h_item_0_2 = xw_next;
                                int h_rel_1_2 = h_item_0_2 - full_tiles;
                                int h_is_2_2 = ((h_rel_1_2 >= 0) ? 1 : 0);
                                int h_relc_3_2 = h_rel_1_2 * h_is_2_2;
                                int h_t_4_2 = h_relc_3_2 / 2;
                                int h_half_5_2 = h_relc_3_2 - h_t_4_2 * 2;
                                int h_ct_6_2 = h_item_0_2 + h_is_2_2 * (full_tiles + h_t_4_2 - h_item_0_2);
                                int pair_bid_7 = h_ct_6_2 * 2 + cta_rank;
                                int group_8 = pair_bid_7 / tiles_per_group;
                                int first_m_9 = group_8 * 32;
                                int remaining_10 = m_tiles - first_m_9;
                                int group_size_11 = ((remaining_10 >= 32) ? 32 : remaining_10);
                                int local_12 = pair_bid_7 % tiles_per_group;
                                int bid_m_13 = first_m_9 + local_12 % group_size_11;
                                int bid_n_14 = local_12 / group_size_11;
                                int n_off_m_1 = bid_m_13 * 128;
                                int n_chunk0_1 = (bid_n_14 * 128 + h_half_5_2 * 64) / 64;
                                if (h_is_2_2 == 0) {
                                    mbarrier_arrive_expect_tx(res_full_addr, 32768);
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 2; j_7++) {
                                        tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j_7 * 16384), (&RT), 0, n_off_m_1, n_chunk0_1 + j_7, res_full_addr);
                                    }
                                } else {
                                    mbarrier_arrive_expect_tx(res_full_addr, 16384);
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 1; j_8++) {
                                        tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j_8 * 16384), (&RT), 0, n_off_m_1, n_chunk0_1 + j_8, res_full_addr);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    int pair_bid_4 = h_ct_3 * 2 + cta_rank;
                    int group_4 = pair_bid_4 / tiles_per_group;
                    int first_m_4 = group_4 * 32;
                    int remaining_4 = m_tiles - first_m_4;
                    int group_size_4 = ((remaining_4 >= 32) ? 32 : remaining_4);
                    int local_4 = pair_bid_4 % tiles_per_group;
                    int bid_m_4 = first_m_4 + local_4 % group_size_4;
                    int bid_n_4 = local_4 / group_size_4;
                    int global_row_1 = bid_m_4 * 128 + local_row;
                    int off_n_1 = bid_n_4 * 128 + h_half_3 * 64;
                    int safe_row_1 = ((global_row_1 < M) ? global_row_1 : M - 1);
                    int store_row_1 = global_row_1;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(res_full_addr, _phase_res_full_0);
                    _phase_res_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned long long row_out_1 = (unsigned long long)global_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned long long row_in_1 = (unsigned long long)safe_row_1 * 1024 + (unsigned long long)off_n_1;
                    unsigned long long sq_out_off_1 = (unsigned long long)global_row_1 * 16 + (unsigned long long)(off_n_1 / 64);
                    unsigned long long wn_off_1 = (unsigned long long)off_n_1;
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
                        unsigned int wn_w_1[32];
                        float part_sq_1 = 0.0f;
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 4; c_1++) {
                            float vals_1[16];
                            int tb_1 = (gcol_1 + c_1 * 16) / 64;
                            int tcb_1 = (gcol_1 + c_1 * 16) % 64 * 2;
                            int tsw0_1 = (tcb_1 ^ (local_row & 7) << 4);
                            int tsw1_1 = (tcb_1 + 16 ^ (local_row & 7) << 4);
                            int tbase_1 = tb_1 * 4096 + local_row * 32;
                            unsigned int rw_1[8];
                            uint32_t _epi_res_w_reg_4[4];
                            __int128_t _smem_b128_5;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_5) : "r"(epi_res_w_addr + (tbase_1 + tsw0_1 / 4) * 4));
                            _epi_res_w_reg_4[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[0];
                            _epi_res_w_reg_4[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[1];
                            _epi_res_w_reg_4[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[2];
                            _epi_res_w_reg_4[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_5)[3];
                            uint32_t _epi_res_w_reg_5[4];
                            __int128_t _smem_b128_6;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_6) : "r"(epi_res_w_addr + (tbase_1 + tsw1_1 / 4) * 4));
                            _epi_res_w_reg_5[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[0];
                            _epi_res_w_reg_5[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[1];
                            _epi_res_w_reg_5[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[2];
                            _epi_res_w_reg_5[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_6)[3];
                            #pragma unroll
                            for (int j_9 = 0; j_9 < 4; j_9++) {
                                rw_1[j_9] = _epi_res_w_reg_4[j_9];
                                rw_1[4 + j_9] = _epi_res_w_reg_5[j_9];
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
                            for (int j_10 = 0; j_10 < 16; j_10++) {
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_1[c_1 * 16 + j_10]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                vals_1[j_10] = rw_f32_1[j_10] + _cvt_f32_1;
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
                            for (int j_11 = 0; j_11 < 16; j_11++) {
                                part_sq_1 += vals_bf16_f32_1[j_11] * vals_bf16_f32_1[j_11];
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((tb_1 * 128 + local_row) * 128 + tcb_1 ^ ((tb_1 * 128 + local_row) * 128 + tcb_1 >> 7 & 7) << 4))), "r"(vals_bf16_1[0]), "r"(vals_bf16_1[1]), "r"(vals_bf16_1[2]), "r"(vals_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((tb_1 * 128 + local_row) * 128 + (tcb_1 + 16) ^ ((tb_1 * 128 + local_row) * 128 + (tcb_1 + 16) >> 7 & 7) << 4))), "r"(vals_bf16_1[4]), "r"(vals_bf16_1[5]), "r"(vals_bf16_1[6]), "r"(vals_bf16_1[7]) : "memory");
                        }
                        if (store_row_1 < M) {
                            SQ[sq_out_off_1 + (unsigned long long)grp_1] = part_sq_1;
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            int st_off_m_1 = bid_m_4 * 128;
                            int st_chunk0_1 = off_n_1 / 64;
                            #pragma unroll
                            for (int j_12 = 0; j_12 < 1; j_12++) {
                                tma_store_3d((&CT), 0, st_off_m_1, st_chunk0_1 + j_12, epi_res_addr + (unsigned int)(j_12 * 16384));
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    unsigned long long pw_off_1 = (unsigned long long)off_n_1;
                    #pragma unroll
                    for (int pg_1 = 0; pg_1 < 1; pg_1++) {
                        #pragma unroll
                        for (int pc_1 = 0; pc_1 < 4; pc_1++) {
                            int pcol_1 = (grp_lo_h + pg_1) * 64 + pc_1 * 16;
                            int ptb_1 = pcol_1 / 64;
                            int ptcb_1 = pcol_1 % 64 * 2;
                            int ptsw0_1 = (ptcb_1 ^ (local_row & 7) << 4);
                            int ptsw1_1 = (ptcb_1 + 16 ^ (local_row & 7) << 4);
                            int ptbase_1 = ptb_1 * 4096 + local_row * 32;
                            uint32_t _epi_res_w_reg_6[4];
                            __int128_t _smem_b128_7;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_7) : "r"(epi_res_w_addr + (ptbase_1 + ptsw0_1 / 4) * 4));
                            _epi_res_w_reg_6[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[0];
                            _epi_res_w_reg_6[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[1];
                            _epi_res_w_reg_6[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[2];
                            _epi_res_w_reg_6[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_7)[3];
                            uint32_t _epi_res_w_reg_7[4];
                            __int128_t _smem_b128_8;
                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_8) : "r"(epi_res_w_addr + (ptbase_1 + ptsw1_1 / 4) * 4));
                            _epi_res_w_reg_7[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[0];
                            _epi_res_w_reg_7[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[1];
                            _epi_res_w_reg_7[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[2];
                            _epi_res_w_reg_7[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_8)[3];
                            unsigned int pwn_1[8];
                            {
                                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(pwn_1[0 + 0]), "=r"(pwn_1[0 + 1]), "=r"(pwn_1[0 + 2]), "=r"(pwn_1[0 + 3]), "=r"(pwn_1[0 + 4]), "=r"(pwn_1[0 + 5]), "=r"(pwn_1[0 + 6]), "=r"(pwn_1[0 + 7]) : "l"((const void*)((const char*)(WN + (pw_off_1 + (unsigned long long)pcol_1) + 0) + 0)) : "memory");
                            }
                            unsigned int pxw_1[8];
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 4; q_1++) {
                                uint32_t _bf16x2_mul_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_epi_res_w_reg_6[q_1]), "r"(pwn_1[q_1]));
                                pxw_1[q_1] = _bf16x2_mul_2;
                                uint32_t _bf16x2_mul_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_epi_res_w_reg_7[q_1]), "r"(pwn_1[4 + q_1]));
                                pxw_1[4 + q_1] = _bf16x2_mul_3;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((ptb_1 * 128 + local_row) * 128 + ptcb_1 ^ ((ptb_1 * 128 + local_row) * 128 + ptcb_1 >> 7 & 7) << 4))), "r"(pxw_1[0]), "r"(pxw_1[1]), "r"(pxw_1[2]), "r"(pxw_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_res_addr + (unsigned int)((ptb_1 * 128 + local_row) * 128 + (ptcb_1 + 16) ^ ((ptb_1 * 128 + local_row) * 128 + (ptcb_1 + 16) >> 7 & 7) << 4))), "r"(pxw_1[4]), "r"(pxw_1[5]), "r"(pxw_1[6]), "r"(pxw_1[7]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            int xw_off_m_1 = bid_m_4 * 128;
                            int xw_chunk0_1 = off_n_1 / 64;
                            #pragma unroll
                            for (int j_13 = 0; j_13 < 1; j_13++) {
                                tma_store_3d((&XWT), 0, xw_off_m_1, xw_chunk0_1 + j_13, epi_res_addr + (unsigned int)(j_13 * 16384));
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            unsigned int xw_next_1 = item_2 + epi_num_clusters;
                            if (xw_next_1 < (unsigned int)(full_tiles + 2 * (num_cluster_tiles - full_tiles))) {
                                int h_item_0_3 = xw_next_1;
                                int h_rel_1_3 = h_item_0_3 - full_tiles;
                                int h_is_2_3 = ((h_rel_1_3 >= 0) ? 1 : 0);
                                int h_relc_3_3 = h_rel_1_3 * h_is_2_3;
                                int h_t_4_3 = h_relc_3_3 / 2;
                                int h_half_5_3 = h_relc_3_3 - h_t_4_3 * 2;
                                int h_ct_6_3 = h_item_0_3 + h_is_2_3 * (full_tiles + h_t_4_3 - h_item_0_3);
                                int pair_bid_7_1 = h_ct_6_3 * 2 + cta_rank;
                                int group_8_1 = pair_bid_7_1 / tiles_per_group;
                                int first_m_9_1 = group_8_1 * 32;
                                int remaining_10_1 = m_tiles - first_m_9_1;
                                int group_size_11_1 = ((remaining_10_1 >= 32) ? 32 : remaining_10_1);
                                int local_12_1 = pair_bid_7_1 % tiles_per_group;
                                int bid_m_13_1 = first_m_9_1 + local_12_1 % group_size_11_1;
                                int bid_n_14_1 = local_12_1 / group_size_11_1;
                                int n_off_m_2 = bid_m_13_1 * 128;
                                int n_chunk0_2 = (bid_n_14_1 * 128 + h_half_5_3 * 64) / 64;
                                if (h_is_2_3 == 0) {
                                    mbarrier_arrive_expect_tx(res_full_addr, 32768);
                                    #pragma unroll
                                    for (int j_14 = 0; j_14 < 2; j_14++) {
                                        tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j_14 * 16384), (&RT), 0, n_off_m_2, n_chunk0_2 + j_14, res_full_addr);
                                    }
                                } else {
                                    mbarrier_arrive_expect_tx(res_full_addr, 16384);
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 1; j_15++) {
                                        tma_3d_gmem2smem(epi_res_addr + (unsigned int)(j_15 * 16384), (&RT), 0, n_off_m_2, n_chunk0_2 + j_15, res_full_addr);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (warp == 2) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
