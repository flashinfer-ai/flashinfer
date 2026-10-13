/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
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
 *
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, licensed under
 * the MIT License; see DEEPGEMM_NOTICE.txt in this directory.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_fp4_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 472
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 448
#define TMEM_TMEM_SFB_OFFSET 456
#define NUM_TMA_PIPE_STAGES 6
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 33792
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 14336
#define SMEM_SMEM_B_STRIDE 33792
#define SMEM_SMEM_SFA_OFF 31744
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 33792
#define SMEM_SMEM_SFB_OFF 32768
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 33792
#define SMEM_EPI_STAGING_OFF 203776
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_TOTAL 220160
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_fp4_gemm_7f6481216fbfb264acf6(const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, int M, int N, int K, int grid_m, int grid_n, int K_tiles, float alpha)
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
    #define sf_ready_addr (mbar_base + 48)
    #define mma_done_addr (mbar_base + 96)
    #define mainloop_done_addr (mbar_base + 144)
    #define epilogue_done_addr (mbar_base + 160)
    #define overlap_done_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 31744);
    const int smem_sfa_addr = smem + 31744;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 32768);
    const int smem_sfb_addr = smem + 32768;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 203776);
    const int epi_staging_addr = smem + 203776;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // sf_ready: 6 barriers, init_count=130
            mbarrier_init(smem + 48, 130);
            mbarrier_init(smem + 56, 130);
            mbarrier_init(smem + 64, 130);
            mbarrier_init(smem + 72, 130);
            mbarrier_init(smem + 80, 130);
            mbarrier_init(smem + 88, 130);
            // mma_done: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // epilogue_done: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // overlap_done: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 472 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 2) {
        int _tmem_hold = smem + 192;
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
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 448;
    const int tmem_tmem_sfb = taddr + 456;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int num_tiles = grid_m * grid_n;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (unsigned int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
                int swizzle_group = this_bid / (unsigned int)(grid_n * 16);
                int first_m = swizzle_group * 16;
                int in_group = this_bid % (unsigned int)(grid_n * 16);
                int _min_0 = ((16) < (grid_m - first_m) ? (16) : (grid_m - first_m));
                int m_blocks = _min_0;
                int bid_m = first_m + in_group % m_blocks;
                int bid_n = in_group / m_blocks;
                int off_m = bid_m * 128;
                int off_n = bid_n * 224;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    if (elect_sync()) {
                        {
                            tma_2d_gmem2smem(smem_sfa_addr + load_stage * 33792, (&SFA), off_m, iter_k * 2, tma_full_addr + (load_stage) * 8);
                        }
                        tma_2d_gmem2smem(smem_sfb_addr + load_stage * 33792, (&SFB), off_n, iter_k * 2, tma_full_addr + (load_stage) * 8);
                        {
                            mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 2048 + ((1) ? 1024 : 0));
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 33792, (&A), 0, off_m, iter_k, ((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 33792, (&B), 0, off_n + cta_rank * 112, iter_k, ((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30720)) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int mma_iteration = 0;
            int num_tiles_1 = grid_m * grid_n;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_sf_ready = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int this_bid_1 = bid; this_bid_1 < num_tiles_1; this_bid_1 += num_bids) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                        mbarrier_wait(sf_ready_addr + (mma_tma_stage) * 8, _phase_sf_ready);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 2112)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 2112 + 32)));
                            }
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2112)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2112 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2112 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2112 + 96)));
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            {
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2112;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2112;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 224)), a_desc + 0, b_desc + 0,
                                        0x10b80480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                                    tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 224)), a_desc + 2, b_desc + 2,
                                        0x50b804a0U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1);
                                }
                                int _mma_a_lo_1 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2112;
                                int _mma_b_lo_1 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2112;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 224)), a_desc + 0, b_desc + 0,
                                        0x10b80480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                                    tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 224)), a_desc + 2, b_desc + 2,
                                        0x50b804a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                                }
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 6) { mma_tma_stage = 0; _phase_sf_ready ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mma_iteration += 1;
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp >= 2 && warp <= 3) {
        { // transpose_main
            int sf_subblock = warp - 2;
            unsigned int transpose_stage = 0;
            int num_tiles_2 = grid_m * grid_n;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int this_bid_2 = bid; this_bid_2 < num_tiles_2; this_bid_2 += num_bids) {
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                    mbarrier_wait(tma_full_addr + (transpose_stage) * 8, _phase_tma_full);
                    int sfa_smem = smem_sfa_addr + transpose_stage * 33792 + (unsigned int)(sf_subblock * 512);
                    int sfb_smem = smem_sfb_addr + transpose_stage * 33792 + (unsigned int)(sf_subblock * 256 * 4);
                    {
                        unsigned int words[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[i])) : "r"(sfa_smem + (i * 32 + lane) * 4));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(sfa_smem + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                    }
                    #pragma unroll
                    for (int sf_block = 0; sf_block < 2; sf_block++) {
                        unsigned int words_1[4];
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_1[i_1])) : "r"(sfb_smem + sf_block * 512 + (i_1 * 32 + lane) * 4));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(sfb_smem + sf_block * 512 + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((sf_ready_addr + (transpose_stage) * 8) & 0xFEFFFFFF) : "memory");
                    transpose_stage += 1;
                    if (transpose_stage == 6) { transpose_stage = 0; _phase_tma_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            int num_tiles_3 = grid_m * grid_n;
            unsigned int store_stage = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int this_bid_3 = bid; this_bid_3 < num_tiles_3; this_bid_3 += num_bids) {
                int swizzle_group_1 = this_bid_3 / (unsigned int)(grid_n * 16);
                int first_m_1 = swizzle_group_1 * 16;
                int in_group_1 = this_bid_3 % (unsigned int)(grid_n * 16);
                int _min_1 = ((16) < (grid_m - first_m_1) ? (16) : (grid_m - first_m_1));
                int m_blocks_1 = _min_1;
                int bid_m_1 = first_m_1 + in_group_1 % m_blocks_1;
                int bid_n_1 = in_group_1 / m_blocks_1;
                int off_m_1 = bid_m_1 * 128;
                int off_n_1 = bid_n_1 * 224;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int epi_pass = 0; epi_pass < 7; epi_pass++) {
                    int store_idx = ((0) ? 6 - epi_pass : epi_pass);
                    int col_start = store_idx * 32;
                    int staging_smem = epi_staging_addr + store_stage * 8192;
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    #pragma unroll
                    for (int n = 0; n < 4; n++) {
                        int row = cta_rank * 128 + epi_warp * 32;
                        int load_idx = ((0) ? 3 - n : n);
                        int col = epi_stage * 224 + (unsigned int)col_start + (unsigned int)(load_idx * 8);
                        int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            _tmem_load_0[element] = _tmem_load_0[element] * alpha;
                        }
                        if (epi_pass == 6 && n == 3) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        uint32_t _tmem_load_0_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int c_stage_row = store_stage * 128 + (unsigned int)(epi_warp * 32) + (unsigned int)lane;
                        unsigned int c_abs = epi_staging_addr + (unsigned int)(c_stage_row * 64);
                        unsigned int c_swz = c_abs / 8 & 48;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + ((unsigned int)(c_stage_row * 64) + ((unsigned int)(load_idx * 16) ^ c_swz)))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_2d((&C_tma), off_n_1 + col_start, off_m_1, staging_smem);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    store_stage = (store_stage + 1) % 2;
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
            if (warp == 4) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 15, 128;" ::: "memory");
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
