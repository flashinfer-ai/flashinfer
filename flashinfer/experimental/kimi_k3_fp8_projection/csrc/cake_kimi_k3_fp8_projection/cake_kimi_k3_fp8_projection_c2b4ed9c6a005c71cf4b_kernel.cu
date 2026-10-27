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
#define TMEM_NCOLS 216
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 192
#define TMEM_TMEM_SFB_OFFSET 200
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_OUT_OFF 183296
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 60416
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 24576
#define SMEM_SMEM_B_STRIDE 60416
#define SMEM_SMEM_SFA_ALL_OFF 58368
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_ALL_STRIDE 60416
#define SMEM_SMEM_SFB_ALL_OFF 59392
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_ALL_STRIDE 60416
#define SMEM_SMEM_SFA0_OFF 58368
#define SMEM_SMEM_SFA0_STAGE_BYTES 512
#define SMEM_SMEM_SFA0_STRIDE 60416
#define SMEM_SMEM_SFA1_OFF 58880
#define SMEM_SMEM_SFA1_STAGE_BYTES 512
#define SMEM_SMEM_SFA1_STRIDE 60416
#define SMEM_SMEM_SFB0_OFF 59392
#define SMEM_SMEM_SFB0_STAGE_BYTES 1024
#define SMEM_SMEM_SFB0_STRIDE 60416
#define SMEM_SMEM_SFB1_OFF 60416
#define SMEM_SMEM_SFB1_STAGE_BYTES 1024
#define SMEM_SMEM_SFB1_STRIDE 60416
#define SMEM_WORK_RESPONSE_OFF 182272
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_SMEM_XCH_OFF 1024
#define SMEM_SMEM_XCH_STAGE_BYTES 102400
#define SMEM_SMEM_XCH_STRIDE 102400
#define SMEM_TOTAL 199680
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

__global__ __launch_bounds__(THREADS) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fp8_projection_c2b4ed9c6a005c71cf4b(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap OUT, int M, int m_tiles, int n_tiles, int n_valid, int ldo, int store_vec, int num_k_iters, int sf_k_tiles, __nv_bfloat16* __restrict__ x, int K, float* __restrict__ sk_partials, unsigned int* __restrict__ sk_flags, int sk_pairs, int sk_rem, int sk_ksplit, int sk_dp)
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
    #define tma_full_h0_addr (mbar_base + 0)
    #define tma_full_h1_addr (mbar_base + 24)
    #define mma_done_h0_addr (mbar_base + 48)
    #define mma_done_h1_addr (mbar_base + 72)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 104)
    #define red_full_addr (mbar_base + 112)
    #define ring_dead_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
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
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA0_OFF);
    const int smem_sfa0_addr = smem + SMEM_SMEM_SFA0_OFF;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA1_OFF);
    const int smem_sfa1_addr = smem + SMEM_SMEM_SFA1_OFF;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB0_OFF);
    const int smem_sfb0_addr = smem + SMEM_SMEM_SFB0_OFF;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB1_OFF);
    const int smem_sfb1_addr = smem + SMEM_SMEM_SFB1_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;
    float* smem_xch = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_XCH_OFF);
    const int smem_xch_addr = smem + SMEM_SMEM_XCH_OFF;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full_h0: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // tma_full_h1: 3 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // mma_done_h0: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // mma_done_h1: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 104, 16);
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // ring_dead: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 112, 51200);
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 216 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
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
    const int tmem_tmem_sfa = taddr + 192;
    const int tmem_tmem_sfb = taddr + 200;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_cluster_tiles = m_tiles / CTA_GROUP * n_tiles;
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_done_h0 = 1;
            unsigned int _phase_mma_done_h1 = 1;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                unsigned int cs_tile_bid = (unsigned int)(bid / 4 * 2) + (unsigned int)(cta_rank & 1);
                int cs_p = cta_rank >> 1;
                int cs_k0 = num_k_iters * cs_p / 2;
                int cs_k1 = num_k_iters * (cs_p + 1) / 2;
                int tiles_per_group = GROUP_M * n_tiles;
                int group = cs_tile_bid / (unsigned int)tiles_per_group;
                int first_m = group * GROUP_M;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                int local = cs_tile_bid % (unsigned int)tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int off_m = bid_m * BLOCK_M;
                int off_n = bid_n * 192;
                int weight_tile0 = (off_n + (cta_rank & 1) * 96) / 128 * (2 * num_k_iters);
                int b_row0 = off_n + (cta_rank & 1) * 96;
                int sfa_tile_row = bid_m * sf_k_tiles;
                int sfb_tile_row = bid_n * sf_k_tiles;
                #pragma unroll 1
                for (int iter_k = cs_k0; iter_k < cs_k1; iter_k++) {
                    int k_group = iter_k * 2;
                    mbarrier_wait(mma_done_h0_addr + (load_stage) * 8, _phase_mma_done_h0);
                    tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 60416, (&A), 0, off_m, k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 60416, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416, (&B), 0, b_row0 % 128, b_row0 / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 4096, (&B), 0, (b_row0 + 32) % 128, (b_row0 + 32) / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 8192, (&B), 0, (b_row0 + 64) % 128, (b_row0 + 64) / 128 * (2 * num_k_iters) + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 60416, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_h0_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30208)) : "memory");
                    mbarrier_wait(mma_done_h1_addr + (load_stage) * 8, _phase_mma_done_h1);
                    tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 60416 + 16384, (&A), 0, off_m, k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 60416 + 512, (&SFA), 0, 0, sfa_tile_row + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 12288, (&B), 0, b_row0 % 128, b_row0 / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 16384, (&B), 0, (b_row0 + 32) % 128, (b_row0 + 32) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 60416 + 20480, (&B), 0, (b_row0 + 64) % 128, (b_row0 + 64) / 128 * (2 * num_k_iters) + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 60416 + 1024, (&SFB), 0, 0, sfb_tile_row + k_group + 1, ((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_h1_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30208)) : "memory");
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done_h0 ^= 1; _phase_mma_done_h1 ^= 1; }
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
            unsigned int _phase_tma_full_h0 = 0;
            unsigned int _phase_tma_full_h1 = 0;
            if ((cta_rank & 1) == 0) {
                int cs_p_1 = cta_rank >> 1;
                int cs_k0_1 = num_k_iters * cs_p_1 / 2;
                int cs_k1_1 = num_k_iters * (cs_p_1 + 1) / 2;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                mbarrier_wait(tma_full_h0_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 3776)));
                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776)));
                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776 + 32)));
                    int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10b00000U, tmem_tmem_sfa, tmem_tmem_sfb, 0);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                            0x30b00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                            0x50b00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                            0x70b00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                    }
                }
                elect_commit_cg2_multicast(mma_done_h0_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                mbarrier_wait(tma_full_h1_addr + (mma_tma_stage) * 8, _phase_tma_full_h1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 3776)));
                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776)));
                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776 + 32)));
                    int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    int _mma_b_lo_1 = (((smem_b_addr + 12288) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10b00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                            0x30b00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                            0x50b00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                        tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                            0x70b00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                    }
                }
                elect_commit_cg2_multicast(mma_done_h1_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                mma_tma_stage += 1;
                if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full_h0 ^= 1; _phase_tma_full_h1 ^= 1; }
                #pragma unroll 1
                for (int _iter_k = cs_k0_1 + 1; _iter_k < cs_k1_1; _iter_k++) {
                    mbarrier_wait(tma_full_h0_addr + (mma_tma_stage) * 8, _phase_tma_full_h0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 3776)));
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 3776 + 32)));
                        int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                        int _mma_b_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10b00000U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                0x30b00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                0x50b00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                0x70b00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        }
                    }
                    elect_commit_cg2_multicast(mma_done_h0_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                    mbarrier_wait(tma_full_h1_addr + (mma_tma_stage) * 8, _phase_tma_full_h1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 3776)));
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 3776 + 32)));
                        int _mma_a_lo_3 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                        int _mma_b_lo_3 = (((smem_b_addr + 12288) >> 4) & 0x3FFF) + (mma_tma_stage) * 3776;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10b00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                0x30b00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                0x50b00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                0x70b00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                        }
                    }
                    elect_commit_cg2_multicast(mma_done_h1_addr + (mma_tma_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full_h0 ^= 1; _phase_tma_full_h1 ^= 1; }
                }
                elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                _phase_epilogue_done ^= 1;
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
            int sk_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 96);
            unsigned int cs_tile_bid_1 = (unsigned int)(bid / 4 * 2) + (unsigned int)(cta_rank & 1);
            int tiles_per_group_1 = GROUP_M * n_tiles;
            int group_1 = cs_tile_bid_1 / (unsigned int)tiles_per_group_1;
            int first_m_1 = group_1 * GROUP_M;
            int remaining_1 = m_tiles - first_m_1;
            int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
            int local_1 = cs_tile_bid_1 % (unsigned int)tiles_per_group_1;
            int bid_m_1 = first_m_1 + local_1 % group_size_1;
            int bid_n_1 = local_1 / group_size_1;
            int off_m_1 = bid_m_1 * BLOCK_M;
            int off_n_1 = bid_n_1 * 192;
            int global_row = off_m_1 + local_row;
            int col0 = col_part * 96;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
            int col_base = off_n_1 + col0;
            unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
            unsigned int _phase_mainloop_done = 0;
            mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[96];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                : "r"(lane_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                : "r"(lane_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                : "r"(lane_addr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (elect_sync()) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
            }
            _phase_mainloop_done ^= 1;
            int cs_p_2 = cta_rank >> 1;
            int cs_r = cta_rank & 1;
            unsigned int xch_row = (unsigned int)(local_row * 400);
            if (tid == 64) {
                if (cs_r == 0) {
                    if (cs_p_2 == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(2) : "memory");
                    }
                    if (cs_p_2 == 1) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(0) : "memory");
                    }
                }
                if (cs_r == 1) {
                    if (cs_p_2 == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(3) : "memory");
                    }
                    if (cs_p_2 == 1) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(ring_dead_addr), "r"(1) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (cs_p_2 == 0) {
                if (col_part == 1) {
                    unsigned int stg_b = smem_xch_addr + 51200 + xch_row;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 16), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 32), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 48), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 64), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 80), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 96), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 112), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 128), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 144), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 160), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 176), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 192), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 208), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 224), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 240), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 256), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 272), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 288), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 304), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 320), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 336), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 352), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b + 368), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 3])));
                }
            }
            if (cs_p_2 == 1) {
                if (col_part == 0) {
                    unsigned int stg_b_1 = smem_xch_addr + 51200 + xch_row;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 0)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 16), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 4)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 32), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 8)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 48), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 12)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 64), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 16)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 80), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 20)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 96), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 24)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 112), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 28)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 128), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 32)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 144), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 36)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 160), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 40)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 176), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 44)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 192), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 48)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 208), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 52)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 224), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 56)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 240), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 60)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 256), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 64)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 272), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 68)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 288), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 72)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 304), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 76)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 320), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 80)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 336), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 84)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 352), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 88)[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(stg_b_1 + 368), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[0])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(_tmem_load_0 + 92)[(0) + 3])));
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            if (warp == 2) {
                mbarrier_wait_cluster_hint(ring_dead_addr, 0, 10000000);
                if (elect_sync()) {
                    if (cs_r == 0) {
                        if (cs_p_2 == 0) {
                            uint32_t _mapa_0;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_0) : "r"(smem_xch_addr), "r"(2));
                            uint32_t _mapa_1;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_1) : "r"(red_full_addr), "r"(2));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_mapa_0), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mapa_1)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (cs_p_2 == 1) {
                            uint32_t _mapa_2;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_2) : "r"(smem_xch_addr), "r"(0));
                            uint32_t _mapa_3;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_3) : "r"(red_full_addr), "r"(0));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_mapa_2), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mapa_3)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (cs_r == 1) {
                        if (cs_p_2 == 0) {
                            uint32_t _mapa_4;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_4) : "r"(smem_xch_addr), "r"(3));
                            uint32_t _mapa_5;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_5) : "r"(red_full_addr), "r"(3));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_mapa_4), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mapa_5)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        if (cs_p_2 == 1) {
                            uint32_t _mapa_6;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_6) : "r"(smem_xch_addr), "r"(1));
                            uint32_t _mapa_7;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_7) : "r"(red_full_addr), "r"(1));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                                " [%0], [%1], %2, [%3];"
                                :: "r"(_mapa_6), "r"(smem_xch_addr + 51200), "r"((uint32_t)(51200)), "r"(_mapa_7)
                                : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    mbarrier_arrive(red_full_addr);
                }
            }
            mbarrier_wait_cluster_hint(red_full_addr, 0, 10000000);
            if (col_part == 0) {
                if (cs_p_2 == 0) {
                    float total[96];
                    total[0] = _tmem_load_0[0];
                    total[1] = _tmem_load_0[1];
                    total[2] = _tmem_load_0[2];
                    total[3] = _tmem_load_0[3];
                    total[4] = _tmem_load_0[4];
                    total[5] = _tmem_load_0[5];
                    total[6] = _tmem_load_0[6];
                    total[7] = _tmem_load_0[7];
                    total[8] = _tmem_load_0[8];
                    total[9] = _tmem_load_0[9];
                    total[10] = _tmem_load_0[10];
                    total[11] = _tmem_load_0[11];
                    total[12] = _tmem_load_0[12];
                    total[13] = _tmem_load_0[13];
                    total[14] = _tmem_load_0[14];
                    total[15] = _tmem_load_0[15];
                    total[16] = _tmem_load_0[16];
                    total[17] = _tmem_load_0[17];
                    total[18] = _tmem_load_0[18];
                    total[19] = _tmem_load_0[19];
                    total[20] = _tmem_load_0[20];
                    total[21] = _tmem_load_0[21];
                    total[22] = _tmem_load_0[22];
                    total[23] = _tmem_load_0[23];
                    total[24] = _tmem_load_0[24];
                    total[25] = _tmem_load_0[25];
                    total[26] = _tmem_load_0[26];
                    total[27] = _tmem_load_0[27];
                    total[28] = _tmem_load_0[28];
                    total[29] = _tmem_load_0[29];
                    total[30] = _tmem_load_0[30];
                    total[31] = _tmem_load_0[31];
                    total[32] = _tmem_load_0[32];
                    total[33] = _tmem_load_0[33];
                    total[34] = _tmem_load_0[34];
                    total[35] = _tmem_load_0[35];
                    total[36] = _tmem_load_0[36];
                    total[37] = _tmem_load_0[37];
                    total[38] = _tmem_load_0[38];
                    total[39] = _tmem_load_0[39];
                    total[40] = _tmem_load_0[40];
                    total[41] = _tmem_load_0[41];
                    total[42] = _tmem_load_0[42];
                    total[43] = _tmem_load_0[43];
                    total[44] = _tmem_load_0[44];
                    total[45] = _tmem_load_0[45];
                    total[46] = _tmem_load_0[46];
                    total[47] = _tmem_load_0[47];
                    total[48] = _tmem_load_0[48];
                    total[49] = _tmem_load_0[49];
                    total[50] = _tmem_load_0[50];
                    total[51] = _tmem_load_0[51];
                    total[52] = _tmem_load_0[52];
                    total[53] = _tmem_load_0[53];
                    total[54] = _tmem_load_0[54];
                    total[55] = _tmem_load_0[55];
                    total[56] = _tmem_load_0[56];
                    total[57] = _tmem_load_0[57];
                    total[58] = _tmem_load_0[58];
                    total[59] = _tmem_load_0[59];
                    total[60] = _tmem_load_0[60];
                    total[61] = _tmem_load_0[61];
                    total[62] = _tmem_load_0[62];
                    total[63] = _tmem_load_0[63];
                    total[64] = _tmem_load_0[64];
                    total[65] = _tmem_load_0[65];
                    total[66] = _tmem_load_0[66];
                    total[67] = _tmem_load_0[67];
                    total[68] = _tmem_load_0[68];
                    total[69] = _tmem_load_0[69];
                    total[70] = _tmem_load_0[70];
                    total[71] = _tmem_load_0[71];
                    total[72] = _tmem_load_0[72];
                    total[73] = _tmem_load_0[73];
                    total[74] = _tmem_load_0[74];
                    total[75] = _tmem_load_0[75];
                    total[76] = _tmem_load_0[76];
                    total[77] = _tmem_load_0[77];
                    total[78] = _tmem_load_0[78];
                    total[79] = _tmem_load_0[79];
                    total[80] = _tmem_load_0[80];
                    total[81] = _tmem_load_0[81];
                    total[82] = _tmem_load_0[82];
                    total[83] = _tmem_load_0[83];
                    total[84] = _tmem_load_0[84];
                    total[85] = _tmem_load_0[85];
                    total[86] = _tmem_load_0[86];
                    total[87] = _tmem_load_0[87];
                    total[88] = _tmem_load_0[88];
                    total[89] = _tmem_load_0[89];
                    total[90] = _tmem_load_0[90];
                    total[91] = _tmem_load_0[91];
                    total[92] = _tmem_load_0[92];
                    total[93] = _tmem_load_0[93];
                    total[94] = _tmem_load_0[94];
                    total[95] = _tmem_load_0[95];
                    float inb_b[96];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(0) + 3]))
                        : "r"(smem_xch_addr + xch_row));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(4) + 3]))
                        : "r"(smem_xch_addr + xch_row + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[8])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(8) + 3]))
                        : "r"(smem_xch_addr + xch_row + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[12])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(12) + 3]))
                        : "r"(smem_xch_addr + xch_row + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[16])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(16) + 3]))
                        : "r"(smem_xch_addr + xch_row + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[20])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(20) + 3]))
                        : "r"(smem_xch_addr + xch_row + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[24])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(24) + 3]))
                        : "r"(smem_xch_addr + xch_row + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[28])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(28) + 3]))
                        : "r"(smem_xch_addr + xch_row + 112));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[32])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(32) + 3]))
                        : "r"(smem_xch_addr + xch_row + 128));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[36])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(36) + 3]))
                        : "r"(smem_xch_addr + xch_row + 144));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[40])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(40) + 3]))
                        : "r"(smem_xch_addr + xch_row + 160));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[44])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(44) + 3]))
                        : "r"(smem_xch_addr + xch_row + 176));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[48])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(48) + 3]))
                        : "r"(smem_xch_addr + xch_row + 192));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[52])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(52) + 3]))
                        : "r"(smem_xch_addr + xch_row + 208));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[56])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(56) + 3]))
                        : "r"(smem_xch_addr + xch_row + 224));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[60])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(60) + 3]))
                        : "r"(smem_xch_addr + xch_row + 240));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[64])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(64) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(64) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(64) + 3]))
                        : "r"(smem_xch_addr + xch_row + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[68])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(68) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(68) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(68) + 3]))
                        : "r"(smem_xch_addr + xch_row + 272));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[72])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(72) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(72) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(72) + 3]))
                        : "r"(smem_xch_addr + xch_row + 288));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[76])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(76) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(76) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(76) + 3]))
                        : "r"(smem_xch_addr + xch_row + 304));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[80])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(80) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(80) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(80) + 3]))
                        : "r"(smem_xch_addr + xch_row + 320));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[84])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(84) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(84) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(84) + 3]))
                        : "r"(smem_xch_addr + xch_row + 336));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[88])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(88) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(88) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(88) + 3]))
                        : "r"(smem_xch_addr + xch_row + 352));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb_b[92])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(92) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(92) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb_b[(92) + 3]))
                        : "r"(smem_xch_addr + xch_row + 368));
                    total[0] = total[0] + inb_b[0];
                    total[1] = total[1] + inb_b[1];
                    total[2] = total[2] + inb_b[2];
                    total[3] = total[3] + inb_b[3];
                    total[4] = total[4] + inb_b[4];
                    total[5] = total[5] + inb_b[5];
                    total[6] = total[6] + inb_b[6];
                    total[7] = total[7] + inb_b[7];
                    total[8] = total[8] + inb_b[8];
                    total[9] = total[9] + inb_b[9];
                    total[10] = total[10] + inb_b[10];
                    total[11] = total[11] + inb_b[11];
                    total[12] = total[12] + inb_b[12];
                    total[13] = total[13] + inb_b[13];
                    total[14] = total[14] + inb_b[14];
                    total[15] = total[15] + inb_b[15];
                    total[16] = total[16] + inb_b[16];
                    total[17] = total[17] + inb_b[17];
                    total[18] = total[18] + inb_b[18];
                    total[19] = total[19] + inb_b[19];
                    total[20] = total[20] + inb_b[20];
                    total[21] = total[21] + inb_b[21];
                    total[22] = total[22] + inb_b[22];
                    total[23] = total[23] + inb_b[23];
                    total[24] = total[24] + inb_b[24];
                    total[25] = total[25] + inb_b[25];
                    total[26] = total[26] + inb_b[26];
                    total[27] = total[27] + inb_b[27];
                    total[28] = total[28] + inb_b[28];
                    total[29] = total[29] + inb_b[29];
                    total[30] = total[30] + inb_b[30];
                    total[31] = total[31] + inb_b[31];
                    total[32] = total[32] + inb_b[32];
                    total[33] = total[33] + inb_b[33];
                    total[34] = total[34] + inb_b[34];
                    total[35] = total[35] + inb_b[35];
                    total[36] = total[36] + inb_b[36];
                    total[37] = total[37] + inb_b[37];
                    total[38] = total[38] + inb_b[38];
                    total[39] = total[39] + inb_b[39];
                    total[40] = total[40] + inb_b[40];
                    total[41] = total[41] + inb_b[41];
                    total[42] = total[42] + inb_b[42];
                    total[43] = total[43] + inb_b[43];
                    total[44] = total[44] + inb_b[44];
                    total[45] = total[45] + inb_b[45];
                    total[46] = total[46] + inb_b[46];
                    total[47] = total[47] + inb_b[47];
                    total[48] = total[48] + inb_b[48];
                    total[49] = total[49] + inb_b[49];
                    total[50] = total[50] + inb_b[50];
                    total[51] = total[51] + inb_b[51];
                    total[52] = total[52] + inb_b[52];
                    total[53] = total[53] + inb_b[53];
                    total[54] = total[54] + inb_b[54];
                    total[55] = total[55] + inb_b[55];
                    total[56] = total[56] + inb_b[56];
                    total[57] = total[57] + inb_b[57];
                    total[58] = total[58] + inb_b[58];
                    total[59] = total[59] + inb_b[59];
                    total[60] = total[60] + inb_b[60];
                    total[61] = total[61] + inb_b[61];
                    total[62] = total[62] + inb_b[62];
                    total[63] = total[63] + inb_b[63];
                    total[64] = total[64] + inb_b[64];
                    total[65] = total[65] + inb_b[65];
                    total[66] = total[66] + inb_b[66];
                    total[67] = total[67] + inb_b[67];
                    total[68] = total[68] + inb_b[68];
                    total[69] = total[69] + inb_b[69];
                    total[70] = total[70] + inb_b[70];
                    total[71] = total[71] + inb_b[71];
                    total[72] = total[72] + inb_b[72];
                    total[73] = total[73] + inb_b[73];
                    total[74] = total[74] + inb_b[74];
                    total[75] = total[75] + inb_b[75];
                    total[76] = total[76] + inb_b[76];
                    total[77] = total[77] + inb_b[77];
                    total[78] = total[78] + inb_b[78];
                    total[79] = total[79] + inb_b[79];
                    total[80] = total[80] + inb_b[80];
                    total[81] = total[81] + inb_b[81];
                    total[82] = total[82] + inb_b[82];
                    total[83] = total[83] + inb_b[83];
                    total[84] = total[84] + inb_b[84];
                    total[85] = total[85] + inb_b[85];
                    total[86] = total[86] + inb_b[86];
                    total[87] = total[87] + inb_b[87];
                    total[88] = total[88] + inb_b[88];
                    total[89] = total[89] + inb_b[89];
                    total[90] = total[90] + inb_b[90];
                    total[91] = total[91] + inb_b[91];
                    total[92] = total[92] + inb_b[92];
                    total[93] = total[93] + inb_b[93];
                    total[94] = total[94] + inb_b[94];
                    total[95] = total[95] + inb_b[95];
                    uint32_t total_bf16[48];
                    #pragma unroll
                    for (int _lp = 0; _lp < 48; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(total[_lp*2 + 0], total[_lp*2+1 + 0]));
                        total_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int ts_base = smem_out_addr + (unsigned int)(col_part * 8192);
                    unsigned int ts_row = ts_base + (unsigned int)(local_row * 64);
                    int swz = local_row >> 1 & 3;
                    int rd_half = local_row & 7;
                    int rd_q = rd_half >> 1;
                    int rd_sub = (rd_half & 1) * 8;
                    int rd_row0 = local_row >> 3;
                    #pragma unroll
                    for (int c = 0; c < 3; c++) {
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            int s0 = c * 16 + q * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row + (unsigned int)((q ^ swz) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16 + s0)[0])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16 + s0)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16 + s0)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16 + s0)[(0) + 3])));
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        int col_c = col_base + c * 32 + rd_half * 4;
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
                }
            }
            if (col_part == 1) {
                if (cs_p_2 == 1) {
                    float total_1[96];
                    float inb0[96];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[0])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(0) + 3]))
                        : "r"(smem_xch_addr + xch_row));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[4])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(4) + 3]))
                        : "r"(smem_xch_addr + xch_row + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[8])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(8) + 3]))
                        : "r"(smem_xch_addr + xch_row + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[12])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(12) + 3]))
                        : "r"(smem_xch_addr + xch_row + 48));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[16])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(16) + 3]))
                        : "r"(smem_xch_addr + xch_row + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[20])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(20) + 3]))
                        : "r"(smem_xch_addr + xch_row + 80));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[24])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(24) + 3]))
                        : "r"(smem_xch_addr + xch_row + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[28])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(28) + 3]))
                        : "r"(smem_xch_addr + xch_row + 112));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[32])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(32) + 3]))
                        : "r"(smem_xch_addr + xch_row + 128));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[36])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(36) + 3]))
                        : "r"(smem_xch_addr + xch_row + 144));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[40])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(40) + 3]))
                        : "r"(smem_xch_addr + xch_row + 160));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[44])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(44) + 3]))
                        : "r"(smem_xch_addr + xch_row + 176));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[48])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(48) + 3]))
                        : "r"(smem_xch_addr + xch_row + 192));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[52])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(52) + 3]))
                        : "r"(smem_xch_addr + xch_row + 208));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[56])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(56) + 3]))
                        : "r"(smem_xch_addr + xch_row + 224));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[60])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(60) + 3]))
                        : "r"(smem_xch_addr + xch_row + 240));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[64])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(64) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(64) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(64) + 3]))
                        : "r"(smem_xch_addr + xch_row + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[68])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(68) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(68) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(68) + 3]))
                        : "r"(smem_xch_addr + xch_row + 272));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[72])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(72) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(72) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(72) + 3]))
                        : "r"(smem_xch_addr + xch_row + 288));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[76])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(76) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(76) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(76) + 3]))
                        : "r"(smem_xch_addr + xch_row + 304));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[80])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(80) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(80) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(80) + 3]))
                        : "r"(smem_xch_addr + xch_row + 320));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[84])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(84) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(84) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(84) + 3]))
                        : "r"(smem_xch_addr + xch_row + 336));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[88])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(88) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(88) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(88) + 3]))
                        : "r"(smem_xch_addr + xch_row + 352));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&inb0[92])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(92) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(92) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inb0[(92) + 3]))
                        : "r"(smem_xch_addr + xch_row + 368));
                    total_1[0] = inb0[0];
                    total_1[1] = inb0[1];
                    total_1[2] = inb0[2];
                    total_1[3] = inb0[3];
                    total_1[4] = inb0[4];
                    total_1[5] = inb0[5];
                    total_1[6] = inb0[6];
                    total_1[7] = inb0[7];
                    total_1[8] = inb0[8];
                    total_1[9] = inb0[9];
                    total_1[10] = inb0[10];
                    total_1[11] = inb0[11];
                    total_1[12] = inb0[12];
                    total_1[13] = inb0[13];
                    total_1[14] = inb0[14];
                    total_1[15] = inb0[15];
                    total_1[16] = inb0[16];
                    total_1[17] = inb0[17];
                    total_1[18] = inb0[18];
                    total_1[19] = inb0[19];
                    total_1[20] = inb0[20];
                    total_1[21] = inb0[21];
                    total_1[22] = inb0[22];
                    total_1[23] = inb0[23];
                    total_1[24] = inb0[24];
                    total_1[25] = inb0[25];
                    total_1[26] = inb0[26];
                    total_1[27] = inb0[27];
                    total_1[28] = inb0[28];
                    total_1[29] = inb0[29];
                    total_1[30] = inb0[30];
                    total_1[31] = inb0[31];
                    total_1[32] = inb0[32];
                    total_1[33] = inb0[33];
                    total_1[34] = inb0[34];
                    total_1[35] = inb0[35];
                    total_1[36] = inb0[36];
                    total_1[37] = inb0[37];
                    total_1[38] = inb0[38];
                    total_1[39] = inb0[39];
                    total_1[40] = inb0[40];
                    total_1[41] = inb0[41];
                    total_1[42] = inb0[42];
                    total_1[43] = inb0[43];
                    total_1[44] = inb0[44];
                    total_1[45] = inb0[45];
                    total_1[46] = inb0[46];
                    total_1[47] = inb0[47];
                    total_1[48] = inb0[48];
                    total_1[49] = inb0[49];
                    total_1[50] = inb0[50];
                    total_1[51] = inb0[51];
                    total_1[52] = inb0[52];
                    total_1[53] = inb0[53];
                    total_1[54] = inb0[54];
                    total_1[55] = inb0[55];
                    total_1[56] = inb0[56];
                    total_1[57] = inb0[57];
                    total_1[58] = inb0[58];
                    total_1[59] = inb0[59];
                    total_1[60] = inb0[60];
                    total_1[61] = inb0[61];
                    total_1[62] = inb0[62];
                    total_1[63] = inb0[63];
                    total_1[64] = inb0[64];
                    total_1[65] = inb0[65];
                    total_1[66] = inb0[66];
                    total_1[67] = inb0[67];
                    total_1[68] = inb0[68];
                    total_1[69] = inb0[69];
                    total_1[70] = inb0[70];
                    total_1[71] = inb0[71];
                    total_1[72] = inb0[72];
                    total_1[73] = inb0[73];
                    total_1[74] = inb0[74];
                    total_1[75] = inb0[75];
                    total_1[76] = inb0[76];
                    total_1[77] = inb0[77];
                    total_1[78] = inb0[78];
                    total_1[79] = inb0[79];
                    total_1[80] = inb0[80];
                    total_1[81] = inb0[81];
                    total_1[82] = inb0[82];
                    total_1[83] = inb0[83];
                    total_1[84] = inb0[84];
                    total_1[85] = inb0[85];
                    total_1[86] = inb0[86];
                    total_1[87] = inb0[87];
                    total_1[88] = inb0[88];
                    total_1[89] = inb0[89];
                    total_1[90] = inb0[90];
                    total_1[91] = inb0[91];
                    total_1[92] = inb0[92];
                    total_1[93] = inb0[93];
                    total_1[94] = inb0[94];
                    total_1[95] = inb0[95];
                    total_1[0] = total_1[0] + _tmem_load_0[0];
                    total_1[1] = total_1[1] + _tmem_load_0[1];
                    total_1[2] = total_1[2] + _tmem_load_0[2];
                    total_1[3] = total_1[3] + _tmem_load_0[3];
                    total_1[4] = total_1[4] + _tmem_load_0[4];
                    total_1[5] = total_1[5] + _tmem_load_0[5];
                    total_1[6] = total_1[6] + _tmem_load_0[6];
                    total_1[7] = total_1[7] + _tmem_load_0[7];
                    total_1[8] = total_1[8] + _tmem_load_0[8];
                    total_1[9] = total_1[9] + _tmem_load_0[9];
                    total_1[10] = total_1[10] + _tmem_load_0[10];
                    total_1[11] = total_1[11] + _tmem_load_0[11];
                    total_1[12] = total_1[12] + _tmem_load_0[12];
                    total_1[13] = total_1[13] + _tmem_load_0[13];
                    total_1[14] = total_1[14] + _tmem_load_0[14];
                    total_1[15] = total_1[15] + _tmem_load_0[15];
                    total_1[16] = total_1[16] + _tmem_load_0[16];
                    total_1[17] = total_1[17] + _tmem_load_0[17];
                    total_1[18] = total_1[18] + _tmem_load_0[18];
                    total_1[19] = total_1[19] + _tmem_load_0[19];
                    total_1[20] = total_1[20] + _tmem_load_0[20];
                    total_1[21] = total_1[21] + _tmem_load_0[21];
                    total_1[22] = total_1[22] + _tmem_load_0[22];
                    total_1[23] = total_1[23] + _tmem_load_0[23];
                    total_1[24] = total_1[24] + _tmem_load_0[24];
                    total_1[25] = total_1[25] + _tmem_load_0[25];
                    total_1[26] = total_1[26] + _tmem_load_0[26];
                    total_1[27] = total_1[27] + _tmem_load_0[27];
                    total_1[28] = total_1[28] + _tmem_load_0[28];
                    total_1[29] = total_1[29] + _tmem_load_0[29];
                    total_1[30] = total_1[30] + _tmem_load_0[30];
                    total_1[31] = total_1[31] + _tmem_load_0[31];
                    total_1[32] = total_1[32] + _tmem_load_0[32];
                    total_1[33] = total_1[33] + _tmem_load_0[33];
                    total_1[34] = total_1[34] + _tmem_load_0[34];
                    total_1[35] = total_1[35] + _tmem_load_0[35];
                    total_1[36] = total_1[36] + _tmem_load_0[36];
                    total_1[37] = total_1[37] + _tmem_load_0[37];
                    total_1[38] = total_1[38] + _tmem_load_0[38];
                    total_1[39] = total_1[39] + _tmem_load_0[39];
                    total_1[40] = total_1[40] + _tmem_load_0[40];
                    total_1[41] = total_1[41] + _tmem_load_0[41];
                    total_1[42] = total_1[42] + _tmem_load_0[42];
                    total_1[43] = total_1[43] + _tmem_load_0[43];
                    total_1[44] = total_1[44] + _tmem_load_0[44];
                    total_1[45] = total_1[45] + _tmem_load_0[45];
                    total_1[46] = total_1[46] + _tmem_load_0[46];
                    total_1[47] = total_1[47] + _tmem_load_0[47];
                    total_1[48] = total_1[48] + _tmem_load_0[48];
                    total_1[49] = total_1[49] + _tmem_load_0[49];
                    total_1[50] = total_1[50] + _tmem_load_0[50];
                    total_1[51] = total_1[51] + _tmem_load_0[51];
                    total_1[52] = total_1[52] + _tmem_load_0[52];
                    total_1[53] = total_1[53] + _tmem_load_0[53];
                    total_1[54] = total_1[54] + _tmem_load_0[54];
                    total_1[55] = total_1[55] + _tmem_load_0[55];
                    total_1[56] = total_1[56] + _tmem_load_0[56];
                    total_1[57] = total_1[57] + _tmem_load_0[57];
                    total_1[58] = total_1[58] + _tmem_load_0[58];
                    total_1[59] = total_1[59] + _tmem_load_0[59];
                    total_1[60] = total_1[60] + _tmem_load_0[60];
                    total_1[61] = total_1[61] + _tmem_load_0[61];
                    total_1[62] = total_1[62] + _tmem_load_0[62];
                    total_1[63] = total_1[63] + _tmem_load_0[63];
                    total_1[64] = total_1[64] + _tmem_load_0[64];
                    total_1[65] = total_1[65] + _tmem_load_0[65];
                    total_1[66] = total_1[66] + _tmem_load_0[66];
                    total_1[67] = total_1[67] + _tmem_load_0[67];
                    total_1[68] = total_1[68] + _tmem_load_0[68];
                    total_1[69] = total_1[69] + _tmem_load_0[69];
                    total_1[70] = total_1[70] + _tmem_load_0[70];
                    total_1[71] = total_1[71] + _tmem_load_0[71];
                    total_1[72] = total_1[72] + _tmem_load_0[72];
                    total_1[73] = total_1[73] + _tmem_load_0[73];
                    total_1[74] = total_1[74] + _tmem_load_0[74];
                    total_1[75] = total_1[75] + _tmem_load_0[75];
                    total_1[76] = total_1[76] + _tmem_load_0[76];
                    total_1[77] = total_1[77] + _tmem_load_0[77];
                    total_1[78] = total_1[78] + _tmem_load_0[78];
                    total_1[79] = total_1[79] + _tmem_load_0[79];
                    total_1[80] = total_1[80] + _tmem_load_0[80];
                    total_1[81] = total_1[81] + _tmem_load_0[81];
                    total_1[82] = total_1[82] + _tmem_load_0[82];
                    total_1[83] = total_1[83] + _tmem_load_0[83];
                    total_1[84] = total_1[84] + _tmem_load_0[84];
                    total_1[85] = total_1[85] + _tmem_load_0[85];
                    total_1[86] = total_1[86] + _tmem_load_0[86];
                    total_1[87] = total_1[87] + _tmem_load_0[87];
                    total_1[88] = total_1[88] + _tmem_load_0[88];
                    total_1[89] = total_1[89] + _tmem_load_0[89];
                    total_1[90] = total_1[90] + _tmem_load_0[90];
                    total_1[91] = total_1[91] + _tmem_load_0[91];
                    total_1[92] = total_1[92] + _tmem_load_0[92];
                    total_1[93] = total_1[93] + _tmem_load_0[93];
                    total_1[94] = total_1[94] + _tmem_load_0[94];
                    total_1[95] = total_1[95] + _tmem_load_0[95];
                    uint32_t total_bf16_1[48];
                    #pragma unroll
                    for (int _lp = 0; _lp < 48; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(total_1[_lp*2 + 0], total_1[_lp*2+1 + 0]));
                        total_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int ts_base_1 = smem_out_addr + (unsigned int)(col_part * 8192);
                    unsigned int ts_row_1 = ts_base_1 + (unsigned int)(local_row * 64);
                    int swz_1 = local_row >> 1 & 3;
                    int rd_half_1 = local_row & 7;
                    int rd_q_1 = rd_half_1 >> 1;
                    int rd_sub_1 = (rd_half_1 & 1) * 8;
                    int rd_row0_1 = local_row >> 3;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 3; c_1++) {
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 4; q_1++) {
                            int s0_1 = c_1 * 16 + q_1 * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(ts_row_1 + (unsigned int)((q_1 ^ swz_1) * 16)), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16_1 + s0_1)[0])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16_1 + s0_1)[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16_1 + s0_1)[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&(total_bf16_1 + s0_1)[(0) + 3])));
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + col_part) : "memory");
                        int col_c_1 = col_base + c_1 * 32 + rd_half_1 * 4;
                        #pragma unroll
                        for (int p_1 = 0; p_1 < 8; p_1++) {
                            int row_p_1 = rd_row0_1 + p_1 * 16;
                            int swz_p_1 = row_p_1 >> 1 & 3;
                            unsigned int words_1[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1]))
                                : "r"(ts_base_1 + (unsigned int)(row_p_1 * 64 + (rd_q_1 ^ swz_p_1) * 16 + rd_sub_1)));
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
            }
            if (warp == 2) {
                asm volatile("cp.async.bulk.wait_group 0;");
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
