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
#include "cake_grouped_fp8_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 268
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 260
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 50688
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 50688
#define SMEM_SMEM_SFA_OFF 50176
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 50688
#define SMEM_SMEM_SFB_OFF 50688
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 50688
#define SMEM_EPI_STAGING_OFF 203776
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_TOTAL 220160
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) void
kernel_cake_grouped_fp8_gemm_1035a4c29fa0da40de08(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A64, const __grid_constant__ CUtensorMap A32, const __grid_constant__ CUtensorMap B, int* __restrict__ SFA, int* __restrict__ SFB, int* __restrict__ m_indices, const __grid_constant__ CUtensorMap C_tma, unsigned int shape_m, unsigned int shape_n, unsigned int grid_n, unsigned int k_tiles, unsigned int sfa_row_stride, unsigned int sfa_col_stride, unsigned int sfb_group_stride, unsigned int sfb_row_stride, unsigned int sfb_col_stride)
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
    #define tma_free_addr (mbar_base + 32)
    #define sf_full_addr (mbar_base + 64)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfa_addr = smem + 50176;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 50688);
    const int smem_sfb_addr = smem + 50688;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 203776);
    const int epi_staging_addr = smem + 203776;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A64))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A32))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // tma_free: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // sf_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 104, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 268 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 260;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int epi_panel = 0;
            const int epi_warp = warp;
            unsigned int num_workers = (unsigned int)num_bids;
            unsigned int worker_idx = (unsigned int)bid;
            unsigned int m_blocks = (shape_m + 128 - 1) / 128;
            unsigned int total_tiles = m_blocks * grid_n;
            unsigned int pairs_per_l2 = m_blocks * 16;
            unsigned int n_l2_limit = (unsigned int)16;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = worker_idx; tile_idx < total_tiles; tile_idx += num_workers) {
                unsigned int l2_group = tile_idx / pairs_per_l2;
                unsigned int first_n = l2_group * 16;
                unsigned int in_l2 = tile_idx % pairs_per_l2;
                unsigned int remaining_n = grid_n - first_n;
                unsigned int n_in_l2 = ((remaining_n > n_l2_limit) ? n_l2_limit : remaining_n);
                unsigned int m_block = in_l2 / n_in_l2;
                unsigned int n_block = first_n + in_l2 % n_in_l2;
                unsigned int off_m = m_block * 128;
                unsigned int off_n = n_block * 256;
                unsigned int last_row = shape_m - 1;
                int group = m_indices[off_m];
                if (group >= 0) {
                    unsigned int nsub = (unsigned int)1;
                    unsigned int srow = off_m + 32;
                    unsigned int srow_safe = ((srow < shape_m) ? srow : last_row);
                    int sg = m_indices[srow_safe];
                    if (srow >= shape_m) {
                        sg = -1;
                    }
                    if (sg >= 0) {
                        nsub = nsub + 1;
                    }
                    unsigned int srow_0 = off_m + 64;
                    unsigned int srow_safe_1 = ((srow_0 < shape_m) ? srow_0 : last_row);
                    int sg_2 = m_indices[srow_safe_1];
                    if (srow_0 >= shape_m) {
                        sg_2 = -1;
                    }
                    if (sg_2 >= 0) {
                        nsub = nsub + 1;
                    }
                    unsigned int srow_3 = off_m + 96;
                    unsigned int srow_safe_4 = ((srow_3 < shape_m) ? srow_3 : last_row);
                    int sg_5 = m_indices[srow_safe_4];
                    if (srow_3 >= shape_m) {
                        sg_5 = -1;
                    }
                    if (sg_5 >= 0) {
                        nsub = nsub + 1;
                    }
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int slab_row0 = epi_warp * 32;
                    unsigned int warp_row0 = (unsigned int)(epi_warp * 32);
                    unsigned int store_row0 = off_m + warp_row0;
                    int slab_in_run = 1;
                    if (warp_row0 < (unsigned int)0 || warp_row0 >= nsub * 32) {
                        slab_in_run = 0;
                    }
                    #pragma unroll
                    for (int panel = 0; panel < 4; panel++) {
                        unsigned int panel_col0 = off_n + (unsigned int)(panel * 64);
                        int staging_row0 = slab_row0 + (int)0 * 32;
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int row = epi_warp * 32;
                            int col = (int)epi_stage * 256 + panel * 64 + n_chunk * 8;
                            int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                            float _tmem_load_0[8];
                            tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            uint32_t _tmem_load_0_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                                _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int staging_row = staging_row0 + lane;
                            int swz_col = n_chunk * 8 ^ (staging_row & 7) << 3;
                            __nv_bfloat16* _sv_ptr_0 = reinterpret_cast<__nv_bfloat16*>(epi_staging + (staging_row * 64 + swz_col));
                            reinterpret_cast<int4*>(_sv_ptr_0 + 0)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            if (panel_col0 < shape_n && slab_in_run != 0) {
                                tma_store_2d((&C_tma), panel_col0, store_row0, epi_staging_addr + (unsigned int)(staging_row0 * 128));
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        __syncwarp();
                        epi_panel = epi_panel + 1;
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                    }
                    _phase_mainloop_done ^= 1;
                }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int num_workers_1 = (unsigned int)num_bids;
            unsigned int worker_idx_1 = (unsigned int)bid;
            unsigned int m_blocks_1 = (shape_m + 128 - 1) / 128;
            unsigned int total_tiles_1 = m_blocks_1 * grid_n;
            unsigned int pairs_per_l2_1 = m_blocks_1 * 16;
            unsigned int n_l2_limit_1 = (unsigned int)16;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = worker_idx_1; tile_idx_1 < total_tiles_1; tile_idx_1 += num_workers_1) {
                unsigned int l2_group_1 = tile_idx_1 / pairs_per_l2_1;
                unsigned int first_n_1 = l2_group_1 * 16;
                unsigned int in_l2_1 = tile_idx_1 % pairs_per_l2_1;
                unsigned int remaining_n_1 = grid_n - first_n_1;
                unsigned int n_in_l2_1 = ((remaining_n_1 > n_l2_limit_1) ? n_l2_limit_1 : remaining_n_1);
                unsigned int m_block_1 = in_l2_1 / n_in_l2_1;
                unsigned int off_m_1 = m_block_1 * 128;
                unsigned int last_row_1 = shape_m - 1;
                int group_1 = m_indices[off_m_1];
                if (group_1 >= 0) {
                    unsigned int nsub_1 = (unsigned int)1;
                    unsigned int srow_1 = off_m_1 + 32;
                    unsigned int srow_safe_2 = ((srow_1 < shape_m) ? srow_1 : last_row_1);
                    int sg_1 = m_indices[srow_safe_2];
                    if (srow_1 >= shape_m) {
                        sg_1 = -1;
                    }
                    if (sg_1 >= 0) {
                        nsub_1 = nsub_1 + 1;
                    }
                    unsigned int srow_0_1 = off_m_1 + 64;
                    unsigned int srow_safe_1_1 = ((srow_0_1 < shape_m) ? srow_0_1 : last_row_1);
                    int sg_2_1 = m_indices[srow_safe_1_1];
                    if (srow_0_1 >= shape_m) {
                        sg_2_1 = -1;
                    }
                    if (sg_2_1 >= 0) {
                        nsub_1 = nsub_1 + 1;
                    }
                    unsigned int srow_3_1 = off_m_1 + 96;
                    unsigned int srow_safe_4_1 = ((srow_3_1 < shape_m) ? srow_3_1 : last_row_1);
                    int sg_5_1 = m_indices[srow_safe_4_1];
                    if (srow_3_1 >= shape_m) {
                        sg_5_1 = -1;
                    }
                    if (sg_5_1 >= 0) {
                        nsub_1 = nsub_1 + 1;
                    }
                    mbarrier_wait(epilogue_done_addr + (epi_stage_1) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_tiles; iter_k++) {
                        mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
                        mbarrier_wait(sf_full_addr + (tma_stage) * 8, _phase_sf_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 3168)));
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 3168)));
                            tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 3168 + 32)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 3168;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (tma_stage) * 3168;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 256)), a_desc + 0, b_desc + 0,
                                    0x8c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 256)), a_desc + 2, b_desc + 2,
                                    0x28c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 256)), a_desc + 4, b_desc + 4,
                                    0x48c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 256)), a_desc + 6, b_desc + 6,
                                    0x68c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                        }
                        elect_commit(tma_free_addr + (tma_stage) * 8);
                        tma_stage += 1;
                        if (tma_stage == 4) { tma_stage = 0; _phase_tma_full ^= 1; _phase_sf_full ^= 1; }
                    }
                    elect_commit(mainloop_done_addr + (epi_stage_1) * 8);
                    _phase_epilogue_done ^= 1;
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int num_workers_2 = (unsigned int)num_bids;
            unsigned int worker_idx_2 = (unsigned int)bid;
            unsigned int m_blocks_2 = (shape_m + 128 - 1) / 128;
            unsigned int total_tiles_2 = m_blocks_2 * grid_n;
            unsigned int pairs_per_l2_2 = m_blocks_2 * 16;
            unsigned int n_l2_limit_2 = (unsigned int)16;
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = worker_idx_2; tile_idx_2 < total_tiles_2; tile_idx_2 += num_workers_2) {
                unsigned int l2_group_2 = tile_idx_2 / pairs_per_l2_2;
                unsigned int first_n_2 = l2_group_2 * 16;
                unsigned int in_l2_2 = tile_idx_2 % pairs_per_l2_2;
                unsigned int remaining_n_2 = grid_n - first_n_2;
                unsigned int n_in_l2_2 = ((remaining_n_2 > n_l2_limit_2) ? n_l2_limit_2 : remaining_n_2);
                unsigned int m_block_2 = in_l2_2 / n_in_l2_2;
                unsigned int n_block_1 = first_n_2 + in_l2_2 % n_in_l2_2;
                unsigned int off_m_2 = m_block_2 * 128;
                unsigned int off_n_1 = n_block_1 * 256;
                unsigned int last_row_2 = shape_m - 1;
                int group_2 = m_indices[off_m_2];
                if (group_2 >= 0) {
                    unsigned int nsub_2 = (unsigned int)1;
                    unsigned int srow_2 = off_m_2 + 32;
                    unsigned int srow_safe_3 = ((srow_2 < shape_m) ? srow_2 : last_row_2);
                    int sg_3 = m_indices[srow_safe_3];
                    if (srow_2 >= shape_m) {
                        sg_3 = -1;
                    }
                    if (sg_3 >= 0) {
                        nsub_2 = nsub_2 + 1;
                    }
                    unsigned int srow_0_2 = off_m_2 + 64;
                    unsigned int srow_safe_1_2 = ((srow_0_2 < shape_m) ? srow_0_2 : last_row_2);
                    int sg_2_2 = m_indices[srow_safe_1_2];
                    if (srow_0_2 >= shape_m) {
                        sg_2_2 = -1;
                    }
                    if (sg_2_2 >= 0) {
                        nsub_2 = nsub_2 + 1;
                    }
                    unsigned int srow_3_2 = off_m_2 + 96;
                    unsigned int srow_safe_4_2 = ((srow_3_2 < shape_m) ? srow_3_2 : last_row_2);
                    int sg_5_2 = m_indices[srow_safe_4_2];
                    if (srow_3_2 >= shape_m) {
                        sg_5_2 = -1;
                    }
                    if (sg_5_2 >= 0) {
                        nsub_2 = nsub_2 + 1;
                    }
                    unsigned int run_rows = nsub_2 * 32 - (unsigned int)0;
                    unsigned int a_box = (unsigned int)128;
                    if (run_rows <= 64) {
                        a_box = 64;
                    }
                    if (run_rows <= 32) {
                        a_box = 32;
                    }
                    unsigned int a_row0 = off_m_2;
                    unsigned int a_dst_off = (unsigned int)0;
                    if (a_box < 128) {
                        a_row0 = off_m_2 + (unsigned int)0;
                        a_dst_off = (unsigned int)0 * 128;
                    }
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_tiles; iter_k_1++) {
                        mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                        if (elect_sync()) {
                            if (a_box == 128) {
                                tma_3d_gmem2smem(smem_a_addr + load_stage * 50688, (&A), 0, off_m_2, (unsigned int)iter_k_1, tma_full_addr + (load_stage) * 8);
                            } else if (a_box == 64) {
                                tma_3d_gmem2smem(smem_a_addr + load_stage * 50688 + (unsigned int)(int)a_dst_off, (&A64), 0, a_row0, (unsigned int)iter_k_1, tma_full_addr + (load_stage) * 8);
                            } else {
                                tma_3d_gmem2smem(smem_a_addr + load_stage * 50688 + (unsigned int)(int)a_dst_off, (&A32), 0, a_row0, (unsigned int)iter_k_1, tma_full_addr + (load_stage) * 8);
                            }
                            tma_4d_gmem2smem(smem_b_addr + load_stage * 50688, (&B), 0, off_n_1, (unsigned int)iter_k_1, (unsigned int)group_2, tma_full_addr + (load_stage) * 8);
                            mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, a_box * 128 + 32768);
                        }
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_tma_free ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: scales ----
    if (warp == 6) {
        { // scales_main
            unsigned int sf_stage = 0;
            unsigned int num_workers_3 = (unsigned int)num_bids;
            unsigned int worker_idx_3 = (unsigned int)bid;
            unsigned int m_blocks_3 = (shape_m + 128 - 1) / 128;
            unsigned int total_tiles_3 = m_blocks_3 * grid_n;
            unsigned int pairs_per_l2_3 = m_blocks_3 * 16;
            unsigned int n_l2_limit_3 = (unsigned int)16;
            unsigned int last_row_3 = shape_m - 1;
            unsigned int last_sfb_block = shape_n / 128 - 1;
            unsigned int _phase_tma_free_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = worker_idx_3; tile_idx_3 < total_tiles_3; tile_idx_3 += num_workers_3) {
                unsigned int l2_group_3 = tile_idx_3 / pairs_per_l2_3;
                unsigned int first_n_3 = l2_group_3 * 16;
                unsigned int in_l2_3 = tile_idx_3 % pairs_per_l2_3;
                unsigned int remaining_n_3 = grid_n - first_n_3;
                unsigned int n_in_l2_3 = ((remaining_n_3 > n_l2_limit_3) ? n_l2_limit_3 : remaining_n_3);
                unsigned int m_block_3 = in_l2_3 / n_in_l2_3;
                unsigned int n_block_2 = first_n_3 + in_l2_3 % n_in_l2_3;
                unsigned int off_m_3 = m_block_3 * 128;
                unsigned int off_n_2 = n_block_2 * 256;
                unsigned int last_row_0 = shape_m - 1;
                int group_3 = m_indices[off_m_3];
                if (group_3 >= 0) {
                    unsigned int nsub_3 = (unsigned int)1;
                    unsigned int srow_4 = off_m_3 + 32;
                    unsigned int srow_safe_5 = ((srow_4 < shape_m) ? srow_4 : last_row_0);
                    int sg_4 = m_indices[srow_safe_5];
                    if (srow_4 >= shape_m) {
                        sg_4 = -1;
                    }
                    if (sg_4 >= 0) {
                        nsub_3 = nsub_3 + 1;
                    }
                    unsigned int srow_0_3 = off_m_3 + 64;
                    unsigned int srow_safe_1_3 = ((srow_0_3 < shape_m) ? srow_0_3 : last_row_0);
                    int sg_2_3 = m_indices[srow_safe_1_3];
                    if (srow_0_3 >= shape_m) {
                        sg_2_3 = -1;
                    }
                    if (sg_2_3 >= 0) {
                        nsub_3 = nsub_3 + 1;
                    }
                    unsigned int srow_3_3 = off_m_3 + 96;
                    unsigned int srow_safe_4_3 = ((srow_3_3 < shape_m) ? srow_3_3 : last_row_0);
                    int sg_5_3 = m_indices[srow_safe_4_3];
                    if (srow_3_3 >= shape_m) {
                        sg_5_3 = -1;
                    }
                    if (sg_5_3 >= 0) {
                        nsub_3 = nsub_3 + 1;
                    }
                    unsigned int sfb_group_base = (unsigned int)group_3 * sfb_group_stride;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < k_tiles; iter_k_2++) {
                        mbarrier_wait(tma_free_addr + (sf_stage) * 8, _phase_tma_free_1);
                        int sfa_base = smem_sfa_addr + sf_stage * 50688;
                        int sfb_base = smem_sfb_addr + sf_stage * 50688;
                        unsigned int sf_lane = (unsigned int)lane;
                        unsigned int kb0 = (unsigned int)iter_k_2;
                        unsigned int sf_row = off_m_3 + (unsigned int)0 + sf_lane;
                        unsigned int sf_row_safe = ((sf_row < shape_m) ? sf_row : last_row_3);
                        unsigned int sfa_row_base = sf_row_safe * sfa_row_stride;
                        unsigned int sfa_lane_dst = sf_lane * 16;
                        unsigned int kb = kb0;
                        unsigned int sfa_word = (unsigned int)SFA[sfa_row_base + kb / 4 * sfa_col_stride];
                        unsigned int sfa_v = sfa_word >> kb % 4 * 8 & 255;
                        unsigned int sfa_rep = sfa_v | sfa_v << 8 | sfa_v << 16 | sfa_v << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst), "r"(sfa_rep));
                        unsigned int sf_row_0 = off_m_3 + (unsigned int)32 + sf_lane;
                        unsigned int sf_row_safe_1 = ((sf_row_0 < shape_m) ? sf_row_0 : last_row_3);
                        unsigned int sfa_row_base_2 = sf_row_safe_1 * sfa_row_stride;
                        unsigned int sfa_lane_dst_3 = sf_lane * 16 + 4;
                        unsigned int kb_4 = kb0;
                        unsigned int sfa_word_5 = (unsigned int)SFA[sfa_row_base_2 + kb_4 / 4 * sfa_col_stride];
                        unsigned int sfa_v_6 = sfa_word_5 >> kb_4 % 4 * 8 & 255;
                        unsigned int sfa_rep_7 = sfa_v_6 | sfa_v_6 << 8 | sfa_v_6 << 16 | sfa_v_6 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_3), "r"(sfa_rep_7));
                        unsigned int sf_row_8 = off_m_3 + (unsigned int)64 + sf_lane;
                        unsigned int sf_row_safe_9 = ((sf_row_8 < shape_m) ? sf_row_8 : last_row_3);
                        unsigned int sfa_row_base_10 = sf_row_safe_9 * sfa_row_stride;
                        unsigned int sfa_lane_dst_11 = sf_lane * 16 + 8;
                        unsigned int kb_12 = kb0;
                        unsigned int sfa_word_13 = (unsigned int)SFA[sfa_row_base_10 + kb_12 / 4 * sfa_col_stride];
                        unsigned int sfa_v_14 = sfa_word_13 >> kb_12 % 4 * 8 & 255;
                        unsigned int sfa_rep_15 = sfa_v_14 | sfa_v_14 << 8 | sfa_v_14 << 16 | sfa_v_14 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_11), "r"(sfa_rep_15));
                        unsigned int sf_row_16 = off_m_3 + (unsigned int)96 + sf_lane;
                        unsigned int sf_row_safe_17 = ((sf_row_16 < shape_m) ? sf_row_16 : last_row_3);
                        unsigned int sfa_row_base_18 = sf_row_safe_17 * sfa_row_stride;
                        unsigned int sfa_lane_dst_19 = sf_lane * 16 + 12;
                        unsigned int kb_20 = kb0;
                        unsigned int sfa_word_21 = (unsigned int)SFA[sfa_row_base_18 + kb_20 / 4 * sfa_col_stride];
                        unsigned int sfa_v_22 = sfa_word_21 >> kb_20 % 4 * 8 & 255;
                        unsigned int sfa_rep_23 = sfa_v_22 | sfa_v_22 << 8 | sfa_v_22 << 16 | sfa_v_22 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_19), "r"(sfa_rep_23));
                        unsigned int sfb_block_idx = 0;
                        unsigned int n_glob = off_n_2 + (unsigned int)0 + sf_lane;
                        unsigned int n_sf_block = n_glob / 128;
                        unsigned int n_sf_block_safe = ((n_glob < shape_n) ? n_sf_block : last_sfb_block);
                        unsigned int sfb_row_base = sfb_group_base + n_sf_block_safe * sfb_row_stride;
                        unsigned int sfb_lane_dst = sfb_block_idx * 512 + sf_lane * 16;
                        unsigned int kb_24 = kb0;
                        unsigned int sfb_word = (unsigned int)SFB[sfb_row_base + kb_24 / 4 * sfb_col_stride];
                        unsigned int sfb_v = sfb_word >> kb_24 % 4 * 8 & 255;
                        unsigned int sfb_rep = sfb_v | sfb_v << 8 | sfb_v << 16 | sfb_v << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst), "r"(sfb_rep));
                        unsigned int sfb_block_idx_25 = 0;
                        unsigned int n_glob_26 = off_n_2 + (unsigned int)32 + sf_lane;
                        unsigned int n_sf_block_27 = n_glob_26 / 128;
                        unsigned int n_sf_block_safe_28 = ((n_glob_26 < shape_n) ? n_sf_block_27 : last_sfb_block);
                        unsigned int sfb_row_base_29 = sfb_group_base + n_sf_block_safe_28 * sfb_row_stride;
                        unsigned int sfb_lane_dst_30 = sfb_block_idx_25 * 512 + sf_lane * 16 + 4;
                        unsigned int kb_31 = kb0;
                        unsigned int sfb_word_32 = (unsigned int)SFB[sfb_row_base_29 + kb_31 / 4 * sfb_col_stride];
                        unsigned int sfb_v_33 = sfb_word_32 >> kb_31 % 4 * 8 & 255;
                        unsigned int sfb_rep_34 = sfb_v_33 | sfb_v_33 << 8 | sfb_v_33 << 16 | sfb_v_33 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_30), "r"(sfb_rep_34));
                        unsigned int sfb_block_idx_35 = 0;
                        unsigned int n_glob_36 = off_n_2 + (unsigned int)64 + sf_lane;
                        unsigned int n_sf_block_37 = n_glob_36 / 128;
                        unsigned int n_sf_block_safe_38 = ((n_glob_36 < shape_n) ? n_sf_block_37 : last_sfb_block);
                        unsigned int sfb_row_base_39 = sfb_group_base + n_sf_block_safe_38 * sfb_row_stride;
                        unsigned int sfb_lane_dst_40 = sfb_block_idx_35 * 512 + sf_lane * 16 + 8;
                        unsigned int kb_41 = kb0;
                        unsigned int sfb_word_42 = (unsigned int)SFB[sfb_row_base_39 + kb_41 / 4 * sfb_col_stride];
                        unsigned int sfb_v_43 = sfb_word_42 >> kb_41 % 4 * 8 & 255;
                        unsigned int sfb_rep_44 = sfb_v_43 | sfb_v_43 << 8 | sfb_v_43 << 16 | sfb_v_43 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_40), "r"(sfb_rep_44));
                        unsigned int sfb_block_idx_45 = 0;
                        unsigned int n_glob_46 = off_n_2 + (unsigned int)96 + sf_lane;
                        unsigned int n_sf_block_47 = n_glob_46 / 128;
                        unsigned int n_sf_block_safe_48 = ((n_glob_46 < shape_n) ? n_sf_block_47 : last_sfb_block);
                        unsigned int sfb_row_base_49 = sfb_group_base + n_sf_block_safe_48 * sfb_row_stride;
                        unsigned int sfb_lane_dst_50 = sfb_block_idx_45 * 512 + sf_lane * 16 + 12;
                        unsigned int kb_51 = kb0;
                        unsigned int sfb_word_52 = (unsigned int)SFB[sfb_row_base_49 + kb_51 / 4 * sfb_col_stride];
                        unsigned int sfb_v_53 = sfb_word_52 >> kb_51 % 4 * 8 & 255;
                        unsigned int sfb_rep_54 = sfb_v_53 | sfb_v_53 << 8 | sfb_v_53 << 16 | sfb_v_53 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_50), "r"(sfb_rep_54));
                        unsigned int sfb_block_idx_55 = 1;
                        unsigned int n_glob_56 = off_n_2 + (unsigned int)128 + sf_lane;
                        unsigned int n_sf_block_57 = n_glob_56 / 128;
                        unsigned int n_sf_block_safe_58 = ((n_glob_56 < shape_n) ? n_sf_block_57 : last_sfb_block);
                        unsigned int sfb_row_base_59 = sfb_group_base + n_sf_block_safe_58 * sfb_row_stride;
                        unsigned int sfb_lane_dst_60 = sfb_block_idx_55 * 512 + sf_lane * 16;
                        unsigned int kb_61 = kb0;
                        unsigned int sfb_word_62 = (unsigned int)SFB[sfb_row_base_59 + kb_61 / 4 * sfb_col_stride];
                        unsigned int sfb_v_63 = sfb_word_62 >> kb_61 % 4 * 8 & 255;
                        unsigned int sfb_rep_64 = sfb_v_63 | sfb_v_63 << 8 | sfb_v_63 << 16 | sfb_v_63 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_60), "r"(sfb_rep_64));
                        unsigned int sfb_block_idx_65 = 1;
                        unsigned int n_glob_66 = off_n_2 + (unsigned int)160 + sf_lane;
                        unsigned int n_sf_block_67 = n_glob_66 / 128;
                        unsigned int n_sf_block_safe_68 = ((n_glob_66 < shape_n) ? n_sf_block_67 : last_sfb_block);
                        unsigned int sfb_row_base_69 = sfb_group_base + n_sf_block_safe_68 * sfb_row_stride;
                        unsigned int sfb_lane_dst_70 = sfb_block_idx_65 * 512 + sf_lane * 16 + 4;
                        unsigned int kb_71 = kb0;
                        unsigned int sfb_word_72 = (unsigned int)SFB[sfb_row_base_69 + kb_71 / 4 * sfb_col_stride];
                        unsigned int sfb_v_73 = sfb_word_72 >> kb_71 % 4 * 8 & 255;
                        unsigned int sfb_rep_74 = sfb_v_73 | sfb_v_73 << 8 | sfb_v_73 << 16 | sfb_v_73 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_70), "r"(sfb_rep_74));
                        unsigned int sfb_block_idx_75 = 1;
                        unsigned int n_glob_76 = off_n_2 + (unsigned int)192 + sf_lane;
                        unsigned int n_sf_block_77 = n_glob_76 / 128;
                        unsigned int n_sf_block_safe_78 = ((n_glob_76 < shape_n) ? n_sf_block_77 : last_sfb_block);
                        unsigned int sfb_row_base_79 = sfb_group_base + n_sf_block_safe_78 * sfb_row_stride;
                        unsigned int sfb_lane_dst_80 = sfb_block_idx_75 * 512 + sf_lane * 16 + 8;
                        unsigned int kb_81 = kb0;
                        unsigned int sfb_word_82 = (unsigned int)SFB[sfb_row_base_79 + kb_81 / 4 * sfb_col_stride];
                        unsigned int sfb_v_83 = sfb_word_82 >> kb_81 % 4 * 8 & 255;
                        unsigned int sfb_rep_84 = sfb_v_83 | sfb_v_83 << 8 | sfb_v_83 << 16 | sfb_v_83 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_80), "r"(sfb_rep_84));
                        unsigned int sfb_block_idx_85 = 1;
                        unsigned int n_glob_86 = off_n_2 + (unsigned int)224 + sf_lane;
                        unsigned int n_sf_block_87 = n_glob_86 / 128;
                        unsigned int n_sf_block_safe_88 = ((n_glob_86 < shape_n) ? n_sf_block_87 : last_sfb_block);
                        unsigned int sfb_row_base_89 = sfb_group_base + n_sf_block_safe_88 * sfb_row_stride;
                        unsigned int sfb_lane_dst_90 = sfb_block_idx_85 * 512 + sf_lane * 16 + 12;
                        unsigned int kb_91 = kb0;
                        unsigned int sfb_word_92 = (unsigned int)SFB[sfb_row_base_89 + kb_91 / 4 * sfb_col_stride];
                        unsigned int sfb_v_93 = sfb_word_92 >> kb_91 % 4 * 8 & 255;
                        unsigned int sfb_rep_94 = sfb_v_93 | sfb_v_93 << 8 | sfb_v_93 << 16 | sfb_v_93 << 24;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_90), "r"(sfb_rep_94));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(sf_full_addr + (sf_stage) * 8);
                        }
                        sf_stage += 1;
                        if (sf_stage == 4) { sf_stage = 0; _phase_tma_free_1 ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
