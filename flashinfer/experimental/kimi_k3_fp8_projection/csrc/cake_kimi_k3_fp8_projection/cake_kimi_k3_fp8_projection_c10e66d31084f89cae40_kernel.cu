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
#include "cake_kimi_k3_fp8_projection_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 144
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 128
#define TMEM_TMEM_SFX_OFFSET 136
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 67584
#define SMEM_SMEM_SFW0_OFF 66560
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 67584
#define SMEM_SMEM_SFW1_OFF 67072
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 67584
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 32768
#define SMEM_SMEM_X_STRIDE 67584
#define SMEM_SMEM_SFX_ALL_OFF 67584
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 67584
#define SMEM_SMEM_SFX0_OFF 67584
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 67584
#define SMEM_SMEM_SFX1_OFF 68096
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 67584
#define SMEM_SMEM_EPI_OFF 203776
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_TOTAL 220160
#define THREADS 192

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_fp8_projection_c10e66d31084f89cae40(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB, const __grid_constant__ CUtensorMap OUT)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_W_OFF);
    const int smem_w_addr = smem + SMEM_SMEM_W_OFF;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFW0_OFF);
    const int smem_sfw0_addr = smem + SMEM_SMEM_SFW0_OFF;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFW1_OFF);
    const int smem_sfw1_addr = smem + SMEM_SMEM_SFW1_OFF;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_X_OFF);
    const int smem_x_addr = smem + SMEM_SMEM_X_OFF;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFX_ALL_OFF);
    const int smem_sfx_all_addr = smem + SMEM_SMEM_SFX_ALL_OFF;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFX0_OFF);
    const int smem_sfx0_addr = smem + SMEM_SMEM_SFX0_OFF;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFX1_OFF);
    const int smem_sfx1_addr = smem + SMEM_SMEM_SFX1_OFF;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_EPI_OFF);
    const int smem_epi_addr = smem + SMEM_SMEM_EPI_OFF;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // mma_done: 3 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 56, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 144 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfw = taddr + 128;
    const int tmem_tmem_sfx = taddr + 136;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int w0 = bid / 2;
                int ws = num_bids / 2;
                int n_items = (total_work - bid / 2 + num_bids / 2 - 1) / (num_bids / 2);
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int it = 0; it < n_items; it++) {
                    int work = w0 + it * ws;
                    int n_tile_c = work % n_tiles;
                    int tile = (work / n_tiles * 2 + cta_rank) * n_tiles + n_tile_c;
                    int rank = 0;
                    int k_begin = 0;
                    int k_count = num_k_iters;
                    int n_tile = tile % n_tiles;
                    int m_tile = tile / n_tiles;
                    int w_tile0 = n_tile * (2 * num_k_iters);
                    int x_row = m_tile * 128;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        if (i == 0) {
                            if (k_begin < k_begin + k_count) {
                                if (cta_rank == 0) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_begin * 2)) : "memory");
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_begin * 2 + 1)) : "memory");
                                }
                                if (cta_rank == 1) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(64)), "r"((int)(w_tile0 + k_begin * 2)) : "memory");
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(64)), "r"((int)(w_tile0 + k_begin * 2 + 1)) : "memory");
                                }
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_begin * 4)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_begin * 4 + 2)) : "memory");
                            }
                        }
                        int k_pf = iter_k + 1;
                        if (k_pf < k_begin + k_count) {
                            if (cta_rank == 0) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_pf * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(0)), "r"((int)(w_tile0 + k_pf * 2 + 1)) : "memory");
                            }
                            if (cta_rank == 1) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(64)), "r"((int)(w_tile0 + k_pf * 2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W))), "r"((int)(0)), "r"((int)(64)), "r"((int)(w_tile0 + k_pf * 2 + 1)) : "memory");
                            }
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_pf * 4)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFW))), "r"((int)(0)), "r"((int)(0)), "r"((int)(sfw_unit0 + k_pf * 4 + 2)) : "memory");
                        }
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        if (cta_rank == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_w_addr + load_stage * 67584), "l"((&W)), "r"(0), "r"(0), "r"(w_tile0 + k_group),
                                   "r"(tma_full_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_w_addr + load_stage * 67584 + 16384), "l"((&W)), "r"(0), "r"(0), "r"(w_tile0 + k_group + 1),
                                   "r"(tma_full_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                        }
                        if (cta_rank == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_w_addr + load_stage * 67584 + 8192), "l"((&W)), "r"(0), "r"(64), "r"(w_tile0 + k_group),
                                   "r"(tma_full_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_w_addr + load_stage * 67584 + 24576), "l"((&W)), "r"(0), "r"(64), "r"(w_tile0 + k_group + 1),
                                   "r"(tma_full_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                        }
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 67584, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 67584, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        if (it == 0 && i == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_x_addr + load_stage * 67584, (&X), 0, x_row, k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfx_all_addr + load_stage * 67584, (&SFX), 0, 0, sfx_unit0 + k_group, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 67584);
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = bid / 2;
            int ws_m = num_bids / 2;
            int n_items_m = (total_work - bid / 2 + num_bids / 2 - 1) / (num_bids / 2);
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int n_tile_c_1 = work_m % n_tiles;
                int tile_1 = (work_m / n_tiles * 2 + cta_rank) * n_tiles + n_tile_c_1;
                int rank_1 = 0;
                int k_begin_1 = 0;
                int k_count_1 = num_k_iters;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 4224)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 4224)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 4224)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 4224)));
                        int _mma_a_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 4224;
                        int _mma_b_lo_0 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                        }
                        int init2 = 0;
                        int _mma_a_lo_1 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 4224;
                        int _mma_b_lo_1 = (((smem_x_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                        }
                    }
                    elect_commit_cg1_multicast(mma_done_addr + (mma_stage) * 8, (uint16_t)(3));
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (acc_stage) * 8);
                _phase_epilogue_done ^= 1;
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 128;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = bid / 2;
            int ws_e = num_bids / 2;
            int n_items_e = (total_work - bid / 2 + num_bids / 2 - 1) / (num_bids / 2);
            unsigned int acc_stage_e = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int it_e = 0; it_e < n_items_e; it_e++) {
                int work_e = w0_e + it_e * ws_e;
                int n_tile_c_2 = work_e % n_tiles;
                int tile_2 = (work_e / n_tiles * 2 + cta_rank) * n_tiles + n_tile_c_2;
                int rank_2 = 0;
                int k_begin_2 = 0;
                int k_count_2 = num_k_iters;
                int n_tile_e = tile_2 % n_tiles;
                int m_tile_e = tile_2 / n_tiles;
                int feature = n_tile_e * 128 + lane_row;
                int tok0 = m_tile_e * 128;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (acc_stage_e) * 8);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                if (split == 1) {
                    unsigned int slot_rel = (unsigned int)(epi_group * 16384);
                    unsigned int slot_addr = smem_epi_addr + slot_rel;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    #pragma unroll
                    for (int t = 0; t < 32; t++) {
                        {
                            __nv_bfloat16 _bval_0 = __float2bfloat16_rn(_tmem_load_0[t]);
                            uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                            uint32_t _addr_0 = static_cast<uint32_t>(smem_epi_addr + (slot_rel + (unsigned int)(t * 256 + lane_row * 2)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), n_tile_e * 128, tok0 + tok_g0, slot_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    unsigned int slot_rel_0 = (unsigned int)(epi_group * 16384 + 8192);
                    unsigned int slot_addr_1 = smem_epi_addr + slot_rel_0;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    #pragma unroll
                    for (int t_1 = 0; t_1 < 32; t_1++) {
                        {
                            __nv_bfloat16 _bval_1 = __float2bfloat16_rn(_tmem_load_0[32 + t_1]);
                            uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_epi_addr + (slot_rel_0 + (unsigned int)(t_1 * 256 + lane_row * 2)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), n_tile_e * 128, tok0 + tok_g0 + 32, slot_addr_1);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    unsigned int slot_rel_2 = (unsigned int)(epi_group * 16384);
                    unsigned int slot_addr_3 = smem_epi_addr + slot_rel_2;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    #pragma unroll
                    for (int t_2 = 0; t_2 < 32; t_2++) {
                        {
                            __nv_bfloat16 _bval_2 = __float2bfloat16_rn(_tmem_load_0[64 + t_2]);
                            uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                            uint32_t _addr_2 = static_cast<uint32_t>(smem_epi_addr + (slot_rel_2 + (unsigned int)(t_2 * 256 + lane_row * 2)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), n_tile_e * 128, tok0 + tok_g0 + 64, slot_addr_3);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    unsigned int slot_rel_4 = (unsigned int)(epi_group * 16384 + 8192);
                    unsigned int slot_addr_5 = smem_epi_addr + slot_rel_4;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    #pragma unroll
                    for (int t_3 = 0; t_3 < 32; t_3++) {
                        {
                            __nv_bfloat16 _bval_3 = __float2bfloat16_rn(_tmem_load_0[96 + t_3]);
                            uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                            uint32_t _addr_3 = static_cast<uint32_t>(smem_epi_addr + (slot_rel_4 + (unsigned int)(t_3 * 256 + lane_row * 2)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 128;" :: "r"(2 + epi_group) : "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&OUT), n_tile_e * 128, tok0 + tok_g0 + 96, slot_addr_5);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else {
                    unsigned long long pbase = (unsigned long long)(tile_2 * split + rank_2) * 16384 + (unsigned long long)lane_row + (unsigned long long)tok_g0 * 128;
                    #pragma unroll
                    for (int t_4 = 0; t_4 < 128; t_4++) {
                        partials[pbase + (unsigned long long)(t_4 * 128)] = _tmem_load_0[t_4];
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (tid == 64) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (tile_2 * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (tile_2 * 2))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>((unsigned int)split)) >= static_cast<unsigned int>(1));
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int t_first = rank_2 * tok_per_cta;
                    int _min_1 = ((128) < (t_first + tok_per_cta) ? (128) : (t_first + tok_per_cta));
                    int t_last = _min_1;
                    unsigned long long rbase = (unsigned long long)(tile_2 * split) * 16384 + (unsigned long long)lane_row;
                    #pragma unroll
                    for (int tb = 0; tb < 128; tb += 8) {
                        if (t_first < tb + 8 && t_last > tb && tok_g0 <= tb && tb < tok_g0 + 128) {
                            float total[8];
                            #pragma unroll
                            for (int u = 0; u < 8; u++) {
                                total[u] = 0.0f;
                            }
                            #pragma unroll 1
                            for (int s_0 = 0; s_0 < split; s_0 += 8) {
                                float parts[64];
                                #pragma unroll
                                for (int j = 0; j < 8; j++) {
                                    int _min_2 = ((s_0 + j) < (split - 1) ? (s_0 + j) : (split - 1));
                                    int s_j = _min_2;
                                    unsigned long long rank_off = rbase + (unsigned long long)s_j * 16384;
                                    #pragma unroll
                                    for (int u_1 = 0; u_1 < 8; u_1++) {
                                        parts[u_1 * 8 + j] = partials[rank_off + (unsigned long long)((tb + u_1) * 128)];
                                    }
                                }
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 8; j_1++) {
                                    if (s_0 + j_1 < split) {
                                        #pragma unroll
                                        for (int u_2 = 0; u_2 < 8; u_2++) {
                                            total[u_2] = total[u_2] + parts[u_2 * 8 + j_1];
                                        }
                                    }
                                }
                            }
                            if (feature < n_valid) {
                                #pragma unroll
                                for (int u_3 = 0; u_3 < 8; u_3++) {
                                    int tok_r = tok0 + tb + u_3;
                                    if (t_first <= tb + u_3 && t_last > tb + u_3 && tok_r < M) {
                                        *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_r * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total[u_3]);
                                    }
                                }
                            }
                        }
                    }
                    if (tid == 64) {
                        unsigned int _atomic_old_0 = atomicAdd(&counters[tile_2 * 2 + 1], 1);
                        unsigned int finished = _atomic_old_0;
                        if ((int)finished == split - 1) {
                            counters[(unsigned long long)(tile_2 * 2)] = 0;
                            counters[(unsigned long long)(tile_2 * 2 + 1)] = 0;
                        }
                    }
                }
            }
            if (epi_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
