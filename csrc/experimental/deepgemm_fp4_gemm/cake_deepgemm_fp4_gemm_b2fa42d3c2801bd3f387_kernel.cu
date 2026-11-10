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
#define TMEM_NCOLS 344
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 320
#define TMEM_SCALE_B_OFFSET 328
#define NUM_PIPE_STAGES 7
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_EPI_OFF 0
#define SMEM_EPI_STAGE_BYTES 8192
#define SMEM_EPI_STRIDE 8192
#define SMEM_SMEM_A_OFF 16384
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 131072
#define SMEM_SMEM_B_STAGE_BYTES 10240
#define SMEM_SMEM_B_STRIDE 10240
#define SMEM_SMEM_SFA_OFF 202752
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 1024
#define SMEM_SMEM_SFB_OFF 209920
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 2048
#define SMEM_TOTAL 224512
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_fp4_gemm_b2fa42d3c2801bd3f387(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, int M, int N, int K, int grid_m, int grid_n, int K_tiles, float alpha)
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

    const int mbar_base = smem + 224256;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 56)
    #define empty_addr (mbar_base + 112)
    #define tmem_full_addr (mbar_base + 168)
    #define tmem_empty_addr (mbar_base + 184)
    #define tmem_overlap_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* epi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_EPI_OFF);
    const int epi_addr = smem + SMEM_EPI_OFF;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFA_OFF);
    const int smem_sfa_addr = smem + SMEM_SMEM_SFA_OFF;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFB_OFF);
    const int smem_sfb_addr = smem + SMEM_SMEM_SFB_OFF;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[224256..224472)

    if (warp == 1) {
        // --- pipeline 'pipe' ---
        // full: 7 barriers, init_count=130
        // sf_full: 7 barriers, init_count=1
        // empty: 7 barriers, init_count=1
        // --- pipeline 'epi_pipe' ---
        // tmem_full: 2 barriers, init_count=1
        // tmem_empty: 2 barriers, init_count=256
        // tmem_overlap: 2 barriers, init_count=256
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_1_0 = 256;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_1_0) : "r"(lane), "n"(23), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_1_0) : "r"(lane), "n"(7), "r"((uint32_t)(130)));
        if (lane < 27) {
            mbarrier_init(smem + 224256 + lane * 8, _mbarrier_init_count_1_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 344 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224472);
    if (warp == 2) {
        int _tmem_hold = smem + 224472;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    // Partial post-allocation TMEM rendezvous (224 threads on named barrier 9)
    if (warp >= 1 && warp <= 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(9), "r"(224) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp >= 1 && warp <= 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 320;
    const int tmem_scale_b = taddr + 328;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (unsigned int bid_1 = bid; bid_1 < grid_m * grid_n; bid_1 += num_bids) {
                unsigned int blocks_per_group = grid_n * 16;
                unsigned int group = bid_1 / blocks_per_group;
                unsigned int first_m = group * 16;
                unsigned int in_group = bid_1 % blocks_per_group;
                int _min_0 = ((16) < ((unsigned int)grid_m - first_m) ? (16) : ((unsigned int)grid_m - first_m));
                unsigned int m_in_group = _min_0;
                unsigned int off_m = (first_m + in_group % m_in_group) * 128;
                unsigned int off_n = in_group / m_in_group * 160;
                for (int k = 0; k < K_tiles; k++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(smem_sfa_addr + stage * 1024, (&SFA), off_m, k * 2, sf_full_addr + (stage) * 8);
                        tma_2d_gmem2smem(smem_sfb_addr + stage * 2048, (&SFB), off_n, k * 2, sf_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, 3072);
                        tma_2d_gmem2smem_cta2(smem_a_addr + stage * 16384, (&A), k * 256, off_m, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_b_addr + stage * 10240, (&B), k * 256, off_n + (unsigned int)(cta_rank * 80), ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 53248);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 7) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int accum_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int bid_2 = bid; bid_2 < grid_m * grid_n; bid_2 += num_bids) {
                    mbarrier_wait(tmem_empty_addr + (accum_stage) * 8, _phase_tmem_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int k_1 = 0; k_1 < K_tiles; k_1++) {
                        mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_a + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 64 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 128)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_b + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 128 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_b + 8), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 128 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_b + 12), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 128 + 96)));
                        }
                        __syncwarp();
                        int init_flag = ((k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            {
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 640;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 160)), a_desc + 0, b_desc + 0,
                                        0x10a80480U, tmem_scale_a + 0, tmem_scale_b + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                                    tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 160)), a_desc + 2, b_desc + 2,
                                        0x50a804a0U, tmem_scale_a + 0, tmem_scale_b + 0, 1);
                                }
                                int _mma_a_lo_1 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_1) * 1024;
                                int _mma_b_lo_1 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_1) * 640;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 160)), a_desc + 0, b_desc + 0,
                                        0x10a80480U, tmem_scale_a + 4 + 0, tmem_scale_b + 8 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                                    tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 160)), a_desc + 2, b_desc + 2,
                                        0x50a804a0U, tmem_scale_a + 4 + 0, tmem_scale_b + 8 + 0, 1);
                                }
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (stage_1) * 8, (uint16_t)(3));
                        if (k_1 == K_tiles - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        stage_1 += 1;
                        if (stage_1 == 7) { stage_1 = 0; _phase_full ^= 1; }
                    }
                    accum_stage += 1;
                    if (accum_stage == 2) { accum_stage = 0; _phase_tmem_empty ^= 1; }
                    completed = completed + 1;
                }
                if (completed > 0) {
                    mbarrier_wait(tmem_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 % 2);
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp >= 2 && warp <= 3) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int sf_subblock = warp - 2;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int bid_3 = bid; bid_3 < grid_m * grid_n; bid_3 += num_bids) {
                for (int k_2 = 0; k_2 < K_tiles; k_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    #pragma unroll
                    for (int sf_block = 0; sf_block < 1; sf_block++) {
                        unsigned int words[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[i])) : "r"(smem_sfa_addr + stage_2 * 1024 + sf_subblock * 128 * 4 + (unsigned int)(sf_block * 512) + (unsigned int)((i * 32 + lane) * 4)));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_sfa_addr + stage_2 * 1024 + sf_subblock * 128 * 4 + (unsigned int)(sf_block * 512) + (unsigned int)(lane * 16)), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                    }
                    #pragma unroll
                    for (int sf_block_1 = 0; sf_block_1 < 2; sf_block_1++) {
                        unsigned int words_1[4];
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_1[i_1])) : "r"(smem_sfb_addr + stage_2 * 2048 + sf_subblock * 256 * 4 + (unsigned int)(sf_block_1 * 512) + (unsigned int)((i_1 * 32 + lane) * 4)));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_sfb_addr + stage_2 * 2048 + sf_subblock * 256 * 4 + (unsigned int)(sf_block_1 * 512) + (unsigned int)(lane * 16)), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 7) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int accum_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (unsigned int bid_4 = bid; bid_4 < grid_m * grid_n; bid_4 += num_bids) {
                unsigned int blocks_per_group_1 = grid_n * 16;
                unsigned int group_1 = bid_4 / blocks_per_group_1;
                unsigned int first_m_1 = group_1 * 16;
                unsigned int in_group_1 = bid_4 % blocks_per_group_1;
                int _min_1 = ((16) < ((unsigned int)grid_m - first_m_1) ? (16) : ((unsigned int)grid_m - first_m_1));
                unsigned int m_in_group_1 = _min_1;
                unsigned int off_m_1 = (first_m_1 + in_group_1 % m_in_group_1) * 128;
                unsigned int off_n_1 = in_group_1 / m_in_group_1 * 160;
                mbarrier_wait(tmem_full_addr + (accum_stage_1) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int row = (unsigned int)(cta_rank * 128) + warp_0 * 32;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (row << 16) + accum_stage_1 * 160));
                if (warp_0 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int stage_base = epi_addr + store_stage * 8192;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + (row << 16) + accum_stage_1 * 160 + 32));
                _tmem_load_0[0] = _tmem_load_0[0] * alpha;
                _tmem_load_0[1] = _tmem_load_0[1] * alpha;
                _tmem_load_0[2] = _tmem_load_0[2] * alpha;
                _tmem_load_0[3] = _tmem_load_0[3] * alpha;
                _tmem_load_0[4] = _tmem_load_0[4] * alpha;
                _tmem_load_0[5] = _tmem_load_0[5] * alpha;
                _tmem_load_0[6] = _tmem_load_0[6] * alpha;
                _tmem_load_0[7] = _tmem_load_0[7] * alpha;
                _tmem_load_0[8] = _tmem_load_0[8] * alpha;
                _tmem_load_0[9] = _tmem_load_0[9] * alpha;
                _tmem_load_0[10] = _tmem_load_0[10] * alpha;
                _tmem_load_0[11] = _tmem_load_0[11] * alpha;
                _tmem_load_0[12] = _tmem_load_0[12] * alpha;
                _tmem_load_0[13] = _tmem_load_0[13] * alpha;
                _tmem_load_0[14] = _tmem_load_0[14] * alpha;
                _tmem_load_0[15] = _tmem_load_0[15] * alpha;
                _tmem_load_0[16] = _tmem_load_0[16] * alpha;
                _tmem_load_0[17] = _tmem_load_0[17] * alpha;
                _tmem_load_0[18] = _tmem_load_0[18] * alpha;
                _tmem_load_0[19] = _tmem_load_0[19] * alpha;
                _tmem_load_0[20] = _tmem_load_0[20] * alpha;
                _tmem_load_0[21] = _tmem_load_0[21] * alpha;
                _tmem_load_0[22] = _tmem_load_0[22] * alpha;
                _tmem_load_0[23] = _tmem_load_0[23] * alpha;
                _tmem_load_0[24] = _tmem_load_0[24] * alpha;
                _tmem_load_0[25] = _tmem_load_0[25] * alpha;
                _tmem_load_0[26] = _tmem_load_0[26] * alpha;
                _tmem_load_0[27] = _tmem_load_0[27] * alpha;
                _tmem_load_0[28] = _tmem_load_0[28] * alpha;
                _tmem_load_0[29] = _tmem_load_0[29] * alpha;
                _tmem_load_0[30] = _tmem_load_0[30] * alpha;
                _tmem_load_0[31] = _tmem_load_0[31] * alpha;
                uint32_t _tmem_load_0_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int shifted = lane * 4;
                unsigned int swz_row = shifted / 8;
                unsigned int swz_col = shifted % 8 ^ swz_row % 4;
                unsigned int write_addr = stage_base + warp_0 * 32 * 32 * 2 + swz_row * 128 + swz_col * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(0) + 3])));
                unsigned int shifted_0 = lane * 4 + 1;
                unsigned int swz_row_1 = shifted_0 / 8;
                unsigned int swz_col_2 = shifted_0 % 8 ^ swz_row_1 % 4;
                unsigned int write_addr_3 = stage_base + warp_0 * 32 * 32 * 2 + swz_row_1 * 128 + swz_col_2 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_3), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(4) + 3])));
                unsigned int shifted_4 = lane * 4 + 2;
                unsigned int swz_row_5 = shifted_4 / 8;
                unsigned int swz_col_6 = shifted_4 % 8 ^ swz_row_5 % 4;
                unsigned int write_addr_7 = stage_base + warp_0 * 32 * 32 * 2 + swz_row_5 * 128 + swz_col_6 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_7), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(8) + 3])));
                unsigned int shifted_8 = lane * 4 + 3;
                unsigned int swz_row_9 = shifted_8 / 8;
                unsigned int swz_col_10 = shifted_8 % 8 ^ swz_row_9 % 4;
                unsigned int write_addr_11 = stage_base + warp_0 * 32 * 32 * 2 + swz_row_9 * 128 + swz_col_10 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_11), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0_bf16[(12) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&C_tma), off_n_1, off_m_1, stage_base);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                __syncwarp();
                store_stage = (store_stage + 1) % 2;
                if (warp_0 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int stage_base_12 = epi_addr + store_stage * 8192;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(taddr + (row << 16) + accum_stage_1 * 160 + 64));
                _tmem_load_1[0] = _tmem_load_1[0] * alpha;
                _tmem_load_1[1] = _tmem_load_1[1] * alpha;
                _tmem_load_1[2] = _tmem_load_1[2] * alpha;
                _tmem_load_1[3] = _tmem_load_1[3] * alpha;
                _tmem_load_1[4] = _tmem_load_1[4] * alpha;
                _tmem_load_1[5] = _tmem_load_1[5] * alpha;
                _tmem_load_1[6] = _tmem_load_1[6] * alpha;
                _tmem_load_1[7] = _tmem_load_1[7] * alpha;
                _tmem_load_1[8] = _tmem_load_1[8] * alpha;
                _tmem_load_1[9] = _tmem_load_1[9] * alpha;
                _tmem_load_1[10] = _tmem_load_1[10] * alpha;
                _tmem_load_1[11] = _tmem_load_1[11] * alpha;
                _tmem_load_1[12] = _tmem_load_1[12] * alpha;
                _tmem_load_1[13] = _tmem_load_1[13] * alpha;
                _tmem_load_1[14] = _tmem_load_1[14] * alpha;
                _tmem_load_1[15] = _tmem_load_1[15] * alpha;
                _tmem_load_1[16] = _tmem_load_1[16] * alpha;
                _tmem_load_1[17] = _tmem_load_1[17] * alpha;
                _tmem_load_1[18] = _tmem_load_1[18] * alpha;
                _tmem_load_1[19] = _tmem_load_1[19] * alpha;
                _tmem_load_1[20] = _tmem_load_1[20] * alpha;
                _tmem_load_1[21] = _tmem_load_1[21] * alpha;
                _tmem_load_1[22] = _tmem_load_1[22] * alpha;
                _tmem_load_1[23] = _tmem_load_1[23] * alpha;
                _tmem_load_1[24] = _tmem_load_1[24] * alpha;
                _tmem_load_1[25] = _tmem_load_1[25] * alpha;
                _tmem_load_1[26] = _tmem_load_1[26] * alpha;
                _tmem_load_1[27] = _tmem_load_1[27] * alpha;
                _tmem_load_1[28] = _tmem_load_1[28] * alpha;
                _tmem_load_1[29] = _tmem_load_1[29] * alpha;
                _tmem_load_1[30] = _tmem_load_1[30] * alpha;
                _tmem_load_1[31] = _tmem_load_1[31] * alpha;
                uint32_t _tmem_load_1_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int shifted_13 = lane * 4;
                unsigned int swz_row_14 = shifted_13 / 8;
                unsigned int swz_col_15 = shifted_13 % 8 ^ swz_row_14 % 4;
                unsigned int write_addr_16 = stage_base_12 + warp_0 * 32 * 32 * 2 + swz_row_14 * 128 + swz_col_15 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_16), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(0) + 3])));
                unsigned int shifted_17 = lane * 4 + 1;
                unsigned int swz_row_18 = shifted_17 / 8;
                unsigned int swz_col_19 = shifted_17 % 8 ^ swz_row_18 % 4;
                unsigned int write_addr_20 = stage_base_12 + warp_0 * 32 * 32 * 2 + swz_row_18 * 128 + swz_col_19 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_20), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(4) + 3])));
                unsigned int shifted_21 = lane * 4 + 2;
                unsigned int swz_row_22 = shifted_21 / 8;
                unsigned int swz_col_23 = shifted_21 % 8 ^ swz_row_22 % 4;
                unsigned int write_addr_24 = stage_base_12 + warp_0 * 32 * 32 * 2 + swz_row_22 * 128 + swz_col_23 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_24), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[8])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(8) + 3])));
                unsigned int shifted_25 = lane * 4 + 3;
                unsigned int swz_row_26 = shifted_25 / 8;
                unsigned int swz_col_27 = shifted_25 % 8 ^ swz_row_26 % 4;
                unsigned int write_addr_28 = stage_base_12 + warp_0 * 32 * 32 * 2 + swz_row_26 * 128 + swz_col_27 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_28), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[12])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_bf16[(12) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&C_tma), off_n_1 + 32, off_m_1, stage_base_12);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                __syncwarp();
                store_stage = (store_stage + 1) % 2;
                if (warp_0 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int stage_base_29 = epi_addr + store_stage * 8192;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + (row << 16) + accum_stage_1 * 160 + 96));
                _tmem_load_2[0] = _tmem_load_2[0] * alpha;
                _tmem_load_2[1] = _tmem_load_2[1] * alpha;
                _tmem_load_2[2] = _tmem_load_2[2] * alpha;
                _tmem_load_2[3] = _tmem_load_2[3] * alpha;
                _tmem_load_2[4] = _tmem_load_2[4] * alpha;
                _tmem_load_2[5] = _tmem_load_2[5] * alpha;
                _tmem_load_2[6] = _tmem_load_2[6] * alpha;
                _tmem_load_2[7] = _tmem_load_2[7] * alpha;
                _tmem_load_2[8] = _tmem_load_2[8] * alpha;
                _tmem_load_2[9] = _tmem_load_2[9] * alpha;
                _tmem_load_2[10] = _tmem_load_2[10] * alpha;
                _tmem_load_2[11] = _tmem_load_2[11] * alpha;
                _tmem_load_2[12] = _tmem_load_2[12] * alpha;
                _tmem_load_2[13] = _tmem_load_2[13] * alpha;
                _tmem_load_2[14] = _tmem_load_2[14] * alpha;
                _tmem_load_2[15] = _tmem_load_2[15] * alpha;
                _tmem_load_2[16] = _tmem_load_2[16] * alpha;
                _tmem_load_2[17] = _tmem_load_2[17] * alpha;
                _tmem_load_2[18] = _tmem_load_2[18] * alpha;
                _tmem_load_2[19] = _tmem_load_2[19] * alpha;
                _tmem_load_2[20] = _tmem_load_2[20] * alpha;
                _tmem_load_2[21] = _tmem_load_2[21] * alpha;
                _tmem_load_2[22] = _tmem_load_2[22] * alpha;
                _tmem_load_2[23] = _tmem_load_2[23] * alpha;
                _tmem_load_2[24] = _tmem_load_2[24] * alpha;
                _tmem_load_2[25] = _tmem_load_2[25] * alpha;
                _tmem_load_2[26] = _tmem_load_2[26] * alpha;
                _tmem_load_2[27] = _tmem_load_2[27] * alpha;
                _tmem_load_2[28] = _tmem_load_2[28] * alpha;
                _tmem_load_2[29] = _tmem_load_2[29] * alpha;
                _tmem_load_2[30] = _tmem_load_2[30] * alpha;
                _tmem_load_2[31] = _tmem_load_2[31] * alpha;
                uint32_t _tmem_load_2_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int shifted_30 = lane * 4;
                unsigned int swz_row_31 = shifted_30 / 8;
                unsigned int swz_col_32 = shifted_30 % 8 ^ swz_row_31 % 4;
                unsigned int write_addr_33 = stage_base_29 + warp_0 * 32 * 32 * 2 + swz_row_31 * 128 + swz_col_32 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_33), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(0) + 3])));
                unsigned int shifted_34 = lane * 4 + 1;
                unsigned int swz_row_35 = shifted_34 / 8;
                unsigned int swz_col_36 = shifted_34 % 8 ^ swz_row_35 % 4;
                unsigned int write_addr_37 = stage_base_29 + warp_0 * 32 * 32 * 2 + swz_row_35 * 128 + swz_col_36 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_37), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(4) + 3])));
                unsigned int shifted_38 = lane * 4 + 2;
                unsigned int swz_row_39 = shifted_38 / 8;
                unsigned int swz_col_40 = shifted_38 % 8 ^ swz_row_39 % 4;
                unsigned int write_addr_41 = stage_base_29 + warp_0 * 32 * 32 * 2 + swz_row_39 * 128 + swz_col_40 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_41), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[8])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(8) + 3])));
                unsigned int shifted_42 = lane * 4 + 3;
                unsigned int swz_row_43 = shifted_42 / 8;
                unsigned int swz_col_44 = shifted_42 % 8 ^ swz_row_43 % 4;
                unsigned int write_addr_45 = stage_base_29 + warp_0 * 32 * 32 * 2 + swz_row_43 * 128 + swz_col_44 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_45), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[12])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2_bf16[(12) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&C_tma), off_n_1 + 64, off_m_1, stage_base_29);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                __syncwarp();
                store_stage = (store_stage + 1) % 2;
                if (warp_0 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int stage_base_46 = epi_addr + store_stage * 8192;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float _tmem_load_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                    : "r"(taddr + (row << 16) + accum_stage_1 * 160 + 128));
                _tmem_load_3[0] = _tmem_load_3[0] * alpha;
                _tmem_load_3[1] = _tmem_load_3[1] * alpha;
                _tmem_load_3[2] = _tmem_load_3[2] * alpha;
                _tmem_load_3[3] = _tmem_load_3[3] * alpha;
                _tmem_load_3[4] = _tmem_load_3[4] * alpha;
                _tmem_load_3[5] = _tmem_load_3[5] * alpha;
                _tmem_load_3[6] = _tmem_load_3[6] * alpha;
                _tmem_load_3[7] = _tmem_load_3[7] * alpha;
                _tmem_load_3[8] = _tmem_load_3[8] * alpha;
                _tmem_load_3[9] = _tmem_load_3[9] * alpha;
                _tmem_load_3[10] = _tmem_load_3[10] * alpha;
                _tmem_load_3[11] = _tmem_load_3[11] * alpha;
                _tmem_load_3[12] = _tmem_load_3[12] * alpha;
                _tmem_load_3[13] = _tmem_load_3[13] * alpha;
                _tmem_load_3[14] = _tmem_load_3[14] * alpha;
                _tmem_load_3[15] = _tmem_load_3[15] * alpha;
                _tmem_load_3[16] = _tmem_load_3[16] * alpha;
                _tmem_load_3[17] = _tmem_load_3[17] * alpha;
                _tmem_load_3[18] = _tmem_load_3[18] * alpha;
                _tmem_load_3[19] = _tmem_load_3[19] * alpha;
                _tmem_load_3[20] = _tmem_load_3[20] * alpha;
                _tmem_load_3[21] = _tmem_load_3[21] * alpha;
                _tmem_load_3[22] = _tmem_load_3[22] * alpha;
                _tmem_load_3[23] = _tmem_load_3[23] * alpha;
                _tmem_load_3[24] = _tmem_load_3[24] * alpha;
                _tmem_load_3[25] = _tmem_load_3[25] * alpha;
                _tmem_load_3[26] = _tmem_load_3[26] * alpha;
                _tmem_load_3[27] = _tmem_load_3[27] * alpha;
                _tmem_load_3[28] = _tmem_load_3[28] * alpha;
                _tmem_load_3[29] = _tmem_load_3[29] * alpha;
                _tmem_load_3[30] = _tmem_load_3[30] * alpha;
                _tmem_load_3[31] = _tmem_load_3[31] * alpha;
                uint32_t _tmem_load_3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int shifted_47 = lane * 4;
                unsigned int swz_row_48 = shifted_47 / 8;
                unsigned int swz_col_49 = shifted_47 % 8 ^ swz_row_48 % 4;
                unsigned int write_addr_50 = stage_base_46 + warp_0 * 32 * 32 * 2 + swz_row_48 * 128 + swz_col_49 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_50), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(0) + 3])));
                unsigned int shifted_51 = lane * 4 + 1;
                unsigned int swz_row_52 = shifted_51 / 8;
                unsigned int swz_col_53 = shifted_51 % 8 ^ swz_row_52 % 4;
                unsigned int write_addr_54 = stage_base_46 + warp_0 * 32 * 32 * 2 + swz_row_52 * 128 + swz_col_53 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_54), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(4) + 3])));
                unsigned int shifted_55 = lane * 4 + 2;
                unsigned int swz_row_56 = shifted_55 / 8;
                unsigned int swz_col_57 = shifted_55 % 8 ^ swz_row_56 % 4;
                unsigned int write_addr_58 = stage_base_46 + warp_0 * 32 * 32 * 2 + swz_row_56 * 128 + swz_col_57 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_58), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[8])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(8) + 3])));
                unsigned int shifted_59 = lane * 4 + 3;
                unsigned int swz_row_60 = shifted_59 / 8;
                unsigned int swz_col_61 = shifted_59 % 8 ^ swz_row_60 % 4;
                unsigned int write_addr_62 = stage_base_46 + warp_0 * 32 * 32 * 2 + swz_row_60 * 128 + swz_col_61 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_62), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[12])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3_bf16[(12) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&C_tma), off_n_1 + 96, off_m_1, stage_base_46);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                __syncwarp();
                store_stage = (store_stage + 1) % 2;
                if (warp_0 == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int stage_base_63 = epi_addr + store_stage * 8192;
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                _tmem_load_4[0] = _tmem_load_4[0] * alpha;
                _tmem_load_4[1] = _tmem_load_4[1] * alpha;
                _tmem_load_4[2] = _tmem_load_4[2] * alpha;
                _tmem_load_4[3] = _tmem_load_4[3] * alpha;
                _tmem_load_4[4] = _tmem_load_4[4] * alpha;
                _tmem_load_4[5] = _tmem_load_4[5] * alpha;
                _tmem_load_4[6] = _tmem_load_4[6] * alpha;
                _tmem_load_4[7] = _tmem_load_4[7] * alpha;
                _tmem_load_4[8] = _tmem_load_4[8] * alpha;
                _tmem_load_4[9] = _tmem_load_4[9] * alpha;
                _tmem_load_4[10] = _tmem_load_4[10] * alpha;
                _tmem_load_4[11] = _tmem_load_4[11] * alpha;
                _tmem_load_4[12] = _tmem_load_4[12] * alpha;
                _tmem_load_4[13] = _tmem_load_4[13] * alpha;
                _tmem_load_4[14] = _tmem_load_4[14] * alpha;
                _tmem_load_4[15] = _tmem_load_4[15] * alpha;
                _tmem_load_4[16] = _tmem_load_4[16] * alpha;
                _tmem_load_4[17] = _tmem_load_4[17] * alpha;
                _tmem_load_4[18] = _tmem_load_4[18] * alpha;
                _tmem_load_4[19] = _tmem_load_4[19] * alpha;
                _tmem_load_4[20] = _tmem_load_4[20] * alpha;
                _tmem_load_4[21] = _tmem_load_4[21] * alpha;
                _tmem_load_4[22] = _tmem_load_4[22] * alpha;
                _tmem_load_4[23] = _tmem_load_4[23] * alpha;
                _tmem_load_4[24] = _tmem_load_4[24] * alpha;
                _tmem_load_4[25] = _tmem_load_4[25] * alpha;
                _tmem_load_4[26] = _tmem_load_4[26] * alpha;
                _tmem_load_4[27] = _tmem_load_4[27] * alpha;
                _tmem_load_4[28] = _tmem_load_4[28] * alpha;
                _tmem_load_4[29] = _tmem_load_4[29] * alpha;
                _tmem_load_4[30] = _tmem_load_4[30] * alpha;
                _tmem_load_4[31] = _tmem_load_4[31] * alpha;
                uint32_t _tmem_load_4_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int shifted_64 = lane * 4;
                unsigned int swz_row_65 = shifted_64 / 8;
                unsigned int swz_col_66 = shifted_64 % 8 ^ swz_row_65 % 4;
                unsigned int write_addr_67 = stage_base_63 + warp_0 * 32 * 32 * 2 + swz_row_65 * 128 + swz_col_66 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_67), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(0) + 3])));
                unsigned int shifted_68 = lane * 4 + 1;
                unsigned int swz_row_69 = shifted_68 / 8;
                unsigned int swz_col_70 = shifted_68 % 8 ^ swz_row_69 % 4;
                unsigned int write_addr_71 = stage_base_63 + warp_0 * 32 * 32 * 2 + swz_row_69 * 128 + swz_col_70 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_71), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(4) + 3])));
                unsigned int shifted_72 = lane * 4 + 2;
                unsigned int swz_row_73 = shifted_72 / 8;
                unsigned int swz_col_74 = shifted_72 % 8 ^ swz_row_73 % 4;
                unsigned int write_addr_75 = stage_base_63 + warp_0 * 32 * 32 * 2 + swz_row_73 * 128 + swz_col_74 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_75), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[8])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(8) + 3])));
                unsigned int shifted_76 = lane * 4 + 3;
                unsigned int swz_row_77 = shifted_76 / 8;
                unsigned int swz_col_78 = shifted_76 % 8 ^ swz_row_77 % 4;
                unsigned int write_addr_79 = stage_base_63 + warp_0 * 32 * 32 * 2 + swz_row_77 * 128 + swz_col_78 * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(write_addr_79), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[12])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4_bf16[(12) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&C_tma), off_n_1 + 128, off_m_1, stage_base_63);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                __syncwarp();
                store_stage = (store_stage + 1) % 2;
                accum_stage_1 += 1;
                if (accum_stage_1 == 2) { accum_stage_1 = 0; _phase_tmem_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

} // extern "C"
