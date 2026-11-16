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
#define TMEM_NCOLS 416
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 384
#define TMEM_TMEM_SFB_OFFSET 392
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_OUT_OFF 217088
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 43008
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 24576
#define SMEM_SMEM_B_STRIDE 43008
#define SMEM_SMEM_SFA_ALL_OFF 41984
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 512
#define SMEM_SMEM_SFA_ALL_STRIDE 43008
#define SMEM_SMEM_SFB_ALL_OFF 42496
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 1536
#define SMEM_SMEM_SFB_ALL_STRIDE 43008
#define SMEM_SMEM_SFB_P0_OFF 42496
#define SMEM_SMEM_SFB_P0_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P0_STRIDE 43008
#define SMEM_SMEM_SFB_P1_OFF 43008
#define SMEM_SMEM_SFB_P1_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P1_STRIDE 43008
#define SMEM_SMEM_SFB_P2_OFF 43520
#define SMEM_SMEM_SFB_P2_STAGE_BYTES 512
#define SMEM_SMEM_SFB_P2_STRIDE 43008
#define SMEM_WORK_RESPONSE_OFF 216064
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 225280
#define THREADS 320
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 3
#define GROUP_M 32
#define WORK_STAGES 4

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_fp8_projection_4da95d45aada36775de5(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp)
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
    #define mma_done_addr (mbar_base + 40)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_OUT_OFF);
    const int smem_out_addr = smem + SMEM_SMEM_OUT_OFF;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA_ALL_OFF);
    const int smem_sfa_all_addr = smem + SMEM_SMEM_SFA_ALL_OFF;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_ALL_OFF);
    const int smem_sfb_all_addr = smem + SMEM_SMEM_SFB_ALL_OFF;
    uint8_t* smem_sfb_p0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_P0_OFF);
    const int smem_sfb_p0_addr = smem + SMEM_SMEM_SFB_P0_OFF;
    uint8_t* smem_sfb_p1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_P1_OFF);
    const int smem_sfb_p1_addr = smem + SMEM_SMEM_SFB_P1_OFF;
    uint8_t* smem_sfb_p2 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_P2_OFF);
    const int smem_sfb_p2_addr = smem + SMEM_SMEM_SFB_P2_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // mma_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 88, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            mbarrier_init(smem + 144, 546);
            mbarrier_init(smem + 152, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 416 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
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
    const int tmem_tmem_sfa = taddr + 384;
    const int tmem_tmem_sfb = taddr + 392;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int tiles_per_group = GROUP_M * n_tiles;
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * 384;
                    int weight_tile0 = (off_n + cta_rank * 192) / 128 * (2 * num_k_iters);
                    int b_row = cta_rank * 64;
                    int b_tile0 = bid_n * 3 * (2 * num_k_iters);
                    int sfa_tile_row = bid_m * sf_k_tiles;
                    int sfb_tile_row = bid_n * sf_k_tiles;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
                        int k_group = iter_k * 2;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 43008, (&A), 0, off_m, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 43008, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008, (&B), 0, b_row, b_tile0 + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008 + 8192, (&B), 0, b_row, b_tile0 + 2 * num_k_iters + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008 + 16384, (&B), 0, b_row, b_tile0 + 2 * (2 * num_k_iters) + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 43008, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(43008)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 43008, (&A), 0, off_m, k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 43008, (&SFA), 0, 0, sfa_tile_row + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008, (&B), 0, b_row, b_tile0 + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008 + 8192, (&B), 0, b_row, b_tile0 + 2 * num_k_iters + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 43008 + 16384, (&B), 0, b_row, b_tile0 + 2 * (2 * num_k_iters) + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 43008, (&SFB), 0, 0, sfb_tile_row + k_group + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(43008)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int num_cluster_tiles_1 = m_tiles / CTA_GROUP * n_tiles;
            unsigned int mma_tma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles_1; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_k_iters; iter_k_1++) {
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_all_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_p0_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 4, make_sf_cp_desc_lo_sbo128((((smem_sfb_p1_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb_p2_addr) >> 4) + (mma_tma_stage) * 2688)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_1 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb + 4, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb + 4, 1);
                            }
                            int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_2 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb + 8, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa_all_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 12, make_sf_cp_desc_lo_sbo128((((smem_sfb_p0_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_sfb_p1_addr) >> 4) + (mma_tma_stage) * 2688)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 20, make_sf_cp_desc_lo_sbo128((((smem_sfb_p2_addr) >> 4) + (mma_tma_stage) * 2688)));
                            int _mma_a_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 12, 1);
                            }
                            int _mma_a_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_4 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1);
                            }
                            int _mma_a_lo_5 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            int _mma_b_lo_5 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 2688;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (256)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 20, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            int num_cluster_tiles_2 = m_tiles / CTA_GROUP * n_tiles;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            int sk_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 192);
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles_2; _tile_iter_2++) {
                int tiles_per_group_1 = GROUP_M * n_tiles;
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group_1;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group_1;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * 384;
                int global_row = off_m_1 + local_row;
                int col0 = col_part * 192;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                int col_base = off_n_1 + col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
                mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int lane_row = taddr + (unsigned int)(epi_warp * 32 << 16);
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(lane_row + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(lane_row + col_part * 64 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_0_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(lane_row + 128 + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(lane_row + 128 + col_part * 64 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_1_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_2[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_row + 256 + col_part * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(lane_row + 256 + col_part * 64 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                uint32_t _tmem_load_2_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int ts_base = smem_out_addr;
                unsigned int ts_row = ts_base + (unsigned int)(local_row * 64);
                int swz = local_row >> 1 & 3;
                int rd_half = local_row & 7;
                int rd_q = rd_half >> 1;
                int rd_sub = (rd_half & 1) * 8;
                int rd_row0 = local_row >> 3;
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int s0 = c * 16 + q * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row + (unsigned int)((q ^ swz) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        int col_c = off_n_1 + c * 32 + rd_half * 4;
                        #pragma unroll
                        for (int p = 0; p < 8; p++) {
                            int row_p = rd_row0 + p * 16;
                            int swz_p = row_p >> 1 & 3;
                            unsigned int words[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1]))
                                : "r"(ts_base + (unsigned int)(row_p * 64 + (rd_q ^ swz_p) * 16 + rd_sub)));
                            int grow = off_m_1 + row_p;
                            if (grow < M) {
                                unsigned long long dst_off = (unsigned long long)grow * (unsigned long long)ldo + (unsigned long long)col_c;
                                if (col_c + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words[0 + 0], words[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off + 0) = _iv2;
                                    }
                                } else if (col_c + 2 <= n_valid) {
                                    unsigned int w0 = words[0];
                                    *(reinterpret_cast<int*>(out + dst_off) + (0)) = w0;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 4; q_1++) {
                            int s0_1 = c * 16 + q_1 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row + (unsigned int)((q_1 ^ swz) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0_1)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0_1)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0_1)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0_bf16 + s0_1)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        int col_c_1 = off_n_1 + 64 + c * 32 + rd_half * 4;
                        #pragma unroll
                        for (int p_1 = 0; p_1 < 8; p_1++) {
                            int row_p_1 = rd_row0 + p_1 * 16;
                            int swz_p_1 = row_p_1 >> 1 & 3;
                            unsigned int words_1[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1]))
                                : "r"(ts_base + (unsigned int)(row_p_1 * 64 + (rd_q ^ swz_p_1) * 16 + rd_sub)));
                            int grow_1 = off_m_1 + row_p_1;
                            if (grow_1 < M) {
                                unsigned long long dst_off_1 = (unsigned long long)grow_1 * (unsigned long long)ldo + (unsigned long long)col_c_1;
                                if (col_c_1 + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words_1[0 + 0], words_1[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off_1 + 0) = _iv2;
                                    }
                                } else if (col_c_1 + 2 <= n_valid) {
                                    unsigned int w0_1 = words_1[0];
                                    *(reinterpret_cast<int*>(out + dst_off_1) + (0)) = w0_1;
                                }
                            }
                        }
                    }
                }
                unsigned int ts_base_0 = smem_out_addr;
                unsigned int ts_row_1 = ts_base_0 + (unsigned int)(local_row * 64);
                int swz_2 = local_row >> 1 & 3;
                int rd_half_3 = local_row & 7;
                int rd_q_4 = rd_half_3 >> 1;
                int rd_sub_5 = (rd_half_3 & 1) * 8;
                int rd_row0_6 = local_row >> 3;
                #pragma unroll
                for (int c_1 = 0; c_1 < 2; c_1++) {
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q_2 = 0; q_2 < 4; q_2++) {
                            int s0_2 = c_1 * 16 + q_2 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row_1 + (unsigned int)((q_2 ^ swz_2) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_2)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_2)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_2)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_2)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        int col_c_2 = off_n_1 + 128 + c_1 * 32 + rd_half_3 * 4;
                        #pragma unroll
                        for (int p_2 = 0; p_2 < 8; p_2++) {
                            int row_p_2 = rd_row0_6 + p_2 * 16;
                            int swz_p_2 = row_p_2 >> 1 & 3;
                            unsigned int words_2[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1]))
                                : "r"(ts_base_0 + (unsigned int)(row_p_2 * 64 + (rd_q_4 ^ swz_p_2) * 16 + rd_sub_5)));
                            int grow_2 = off_m_1 + row_p_2;
                            if (grow_2 < M) {
                                unsigned long long dst_off_2 = (unsigned long long)grow_2 * (unsigned long long)ldo + (unsigned long long)col_c_2;
                                if (col_c_2 + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words_2[0 + 0], words_2[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off_2 + 0) = _iv2;
                                    }
                                } else if (col_c_2 + 2 <= n_valid) {
                                    unsigned int w0_2 = words_2[0];
                                    *(reinterpret_cast<int*>(out + dst_off_2) + (0)) = w0_2;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        #pragma unroll
                        for (int q_3 = 0; q_3 < 4; q_3++) {
                            int s0_3 = c_1 * 16 + q_3 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row_1 + (unsigned int)((q_3 ^ swz_2) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_3)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_3)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_3)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_1_bf16 + s0_3)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        int col_c_3 = off_n_1 + 192 + c_1 * 32 + rd_half_3 * 4;
                        #pragma unroll
                        for (int p_3 = 0; p_3 < 8; p_3++) {
                            int row_p_3 = rd_row0_6 + p_3 * 16;
                            int swz_p_3 = row_p_3 >> 1 & 3;
                            unsigned int words_3[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1]))
                                : "r"(ts_base_0 + (unsigned int)(row_p_3 * 64 + (rd_q_4 ^ swz_p_3) * 16 + rd_sub_5)));
                            int grow_3 = off_m_1 + row_p_3;
                            if (grow_3 < M) {
                                unsigned long long dst_off_3 = (unsigned long long)grow_3 * (unsigned long long)ldo + (unsigned long long)col_c_3;
                                if (col_c_3 + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words_3[0 + 0], words_3[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off_3 + 0) = _iv2;
                                    }
                                } else if (col_c_3 + 2 <= n_valid) {
                                    unsigned int w0_3 = words_3[0];
                                    *(reinterpret_cast<int*>(out + dst_off_3) + (0)) = w0_3;
                                }
                            }
                        }
                    }
                }
                unsigned int ts_base_7 = smem_out_addr;
                unsigned int ts_row_8 = ts_base_7 + (unsigned int)(local_row * 64);
                int swz_9 = local_row >> 1 & 3;
                int rd_half_10 = local_row & 7;
                int rd_q_11 = rd_half_10 >> 1;
                int rd_sub_12 = (rd_half_10 & 1) * 8;
                int rd_row0_13 = local_row >> 3;
                #pragma unroll
                for (int c_2 = 0; c_2 < 2; c_2++) {
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        #pragma unroll
                        for (int q_4 = 0; q_4 < 4; q_4++) {
                            int s0_4 = c_2 * 16 + q_4 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row_8 + (unsigned int)((q_4 ^ swz_9) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_4)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_4)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_4)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_4)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 0) {
                        int col_c_4 = off_n_1 + 256 + c_2 * 32 + rd_half_10 * 4;
                        #pragma unroll
                        for (int p_4 = 0; p_4 < 8; p_4++) {
                            int row_p_4 = rd_row0_13 + p_4 * 16;
                            int swz_p_4 = row_p_4 >> 1 & 3;
                            unsigned int words_4[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 1]))
                                : "r"(ts_base_7 + (unsigned int)(row_p_4 * 64 + (rd_q_11 ^ swz_p_4) * 16 + rd_sub_12)));
                            int grow_4 = off_m_1 + row_p_4;
                            if (grow_4 < M) {
                                unsigned long long dst_off_4 = (unsigned long long)grow_4 * (unsigned long long)ldo + (unsigned long long)col_c_4;
                                if (col_c_4 + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words_4[0 + 0], words_4[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off_4 + 0) = _iv2;
                                    }
                                } else if (col_c_4 + 2 <= n_valid) {
                                    unsigned int w0_4 = words_4[0];
                                    *(reinterpret_cast<int*>(out + dst_off_4) + (0)) = w0_4;
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        #pragma unroll
                        for (int q_5 = 0; q_5 < 4; q_5++) {
                            int s0_5 = c_2 * 16 + q_5 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row_8 + (unsigned int)((q_5 ^ swz_9) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_5)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_5)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_5)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_2_bf16 + s0_5)[(0) + 3])));
                        }
                    }
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (col_part == 1) {
                        int col_c_5 = off_n_1 + 320 + c_2 * 32 + rd_half_10 * 4;
                        #pragma unroll
                        for (int p_5 = 0; p_5 < 8; p_5++) {
                            int row_p_5 = rd_row0_13 + p_5 * 16;
                            int swz_p_5 = row_p_5 >> 1 & 3;
                            unsigned int words_5[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 1]))
                                : "r"(ts_base_7 + (unsigned int)(row_p_5 * 64 + (rd_q_11 ^ swz_p_5) * 16 + rd_sub_12)));
                            int grow_5 = off_m_1 + row_p_5;
                            if (grow_5 < M) {
                                unsigned long long dst_off_5 = (unsigned long long)grow_5 * (unsigned long long)ldo + (unsigned long long)col_c_5;
                                if (col_c_5 + 4 <= n_valid) {
                                    {
                                        int2 _iv2 = make_int2(words_5[0 + 0], words_5[0 + 1]);
                                        *reinterpret_cast<int2*>(out + dst_off_5 + 0) = _iv2;
                                    }
                                } else if (col_c_5 + 2 <= n_valid) {
                                    unsigned int w0_5 = words_5[0];
                                    *(reinterpret_cast<int*>(out + dst_off_5) + (0)) = w0_5;
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
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
