/*
 * Copyright (c) 2026 by FlashInfer team.
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
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, licensed
 * under the MIT License; see DEEPGEMM_NOTICE.txt in the parent directory.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_kgroup_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 488
#define TMEM_TMEM_SFB_OFFSET 496
#define NUM_TMA_PIPE_STAGES 6
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 35840
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 35840
#define SMEM_SMEM_SFA_OFF 33792
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 35840
#define SMEM_SMEM_SFB_OFF 34816
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 35840
#define SMEM_EPI_STAGING_OFF 216064
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_TOTAL 232448
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_kgroup_gemm_29d7fd24d60cd168073d(const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, int* __restrict__ grouped_layout, int M, int N, int K, int grid_m, int grid_n, int num_groups)
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
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFA_OFF);
    const int smem_sfa_addr = smem + SMEM_SMEM_SFA_OFF;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFB_OFF);
    const int smem_sfb_addr = smem + SMEM_SMEM_SFB_OFF;
    float* epi_staging = reinterpret_cast<float*>(smem_raw + SMEM_EPI_STAGING_OFF);
    const int epi_staging_addr = smem + SMEM_EPI_STAGING_OFF;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 1) {
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

    // TMEM alloc (512 columns, 512 used)
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
    const int tmem_tmem_sfa = taddr + 488;
    const int tmem_tmem_sfb = taddr + 496;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int num_tiles = num_groups * grid_m * grid_n;
            int current_group = 0;
            int current_k_start = 0;
            int current_sf_start = 0;
            int current_shape_k = grouped_layout[0];
            int current_k_end = current_shape_k;
            unsigned int fd_tiles_mult = (unsigned int)grouped_layout[num_groups];
            unsigned int fd_tiles_shift = (unsigned int)grouped_layout[num_groups + 1];
            unsigned int fd_swizzle_mult = (unsigned int)grouped_layout[num_groups + 2];
            unsigned int fd_swizzle_shift = (unsigned int)grouped_layout[num_groups + 3];
            unsigned int fd_tail_mult = (unsigned int)grouped_layout[num_groups + 4];
            unsigned int fd_tail_shift = (unsigned int)grouped_layout[num_groups + 5];
            int rot_b = num_bids;
            int rot_t = grid_m * grid_n;
            int rot_q = rot_t / rot_b;
            int rot_r = rot_t - rot_q * rot_b;
            int rot_bid = bid;
            int rot_first = 0;
            int rot_iters = 0;
            #pragma unroll 1
            for (int rot_g = 0; rot_g < num_groups; rot_g++) {
                int rot_gd = num_groups - 1 - rot_g;
                int _max_0 = ((rot_gd - 1) > (0) ? (rot_gd - 1) : (0));
                int _min_0 = ((rot_gd) < (1) ? (rot_gd) : (1));
                int prev_end = grouped_layout[_max_0] * _min_0;
                int rot_k_start = (prev_end + 256 - 1) / 256 * 256;
                int rot_shape_k = grouped_layout[rot_gd] - rot_k_start;
                int _max_1 = ((1) > ((rot_shape_k + 256 - 1) / 256) ? (1) : ((rot_shape_k + 256 - 1) / 256));
                int cost_g = _max_1;
                int rot_rank = 0;
                #pragma unroll 1
                for (int rot_h = 0; rot_h < num_groups; rot_h++) {
                    int _max_2 = ((rot_h - 1) > (0) ? (rot_h - 1) : (0));
                    int _min_1 = ((rot_h) < (1) ? (rot_h) : (1));
                    int prev_end_0 = grouped_layout[_max_2] * _min_1;
                    int rot_k_start_1 = (prev_end_0 + 256 - 1) / 256 * 256;
                    int rot_shape_k_2 = grouped_layout[rot_h] - rot_k_start_1;
                    int _max_3 = ((1) > ((rot_shape_k_2 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2 + 256 - 1) / 256));
                    int cost_h = _max_3;
                    int _max_4 = ((0) > (cost_h - cost_g) ? (0) : (cost_h - cost_g));
                    int _min_2 = ((1) < (_max_4) ? (1) : (_max_4));
                    int _max_5 = ((cost_h - cost_g) > (cost_g - cost_h) ? (cost_h - cost_g) : (cost_g - cost_h));
                    int _min_3 = ((1) < (_max_5) ? (1) : (_max_5));
                    int _max_6 = ((0) > (rot_gd - rot_h) ? (0) : (rot_gd - rot_h));
                    int _min_4 = ((1) < (_max_6) ? (1) : (_max_6));
                    rot_rank += _min_2 + (1 - _min_3) * _min_4;
                }
                rot_first = (rot_bid + rot_b - rot_rank * rot_r % rot_b) % rot_b;
                int _max_7 = ((0) > (rot_r - rot_first) ? (0) : (rot_r - rot_first));
                int _min_5 = ((1) < (_max_7) ? (1) : (_max_7));
                rot_iters += rot_q + _min_5;
            }
            int rot_loop_end = rot_bid + rot_iters * rot_b;
            int rot_cum_begin = 0;
            int _max_8 = ((0) > (rot_r - rot_first) ? (0) : (rot_r - rot_first));
            int _min_6 = ((1) < (_max_8) ? (1) : (_max_8));
            int rot_cum_end = rot_q + _min_6;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (unsigned int this_bid = bid; this_bid < rot_loop_end; this_bid += num_bids) {
                int rot_k = (this_bid - (unsigned int)rot_bid) / (unsigned int)rot_b;
                int _max_9 = ((0) > (rot_k - rot_cum_end + 1) ? (0) : (rot_k - rot_cum_end + 1));
                int _min_7 = ((1) < (_max_9) ? (1) : (_max_9));
                int rot_adv_n = _min_7;
                int _min_8 = ((1) < (rot_q) ? (1) : (rot_q));
                int rot_scan_n = num_groups * (1 - _min_8);
                int rot_scan_g = current_group;
                int rot_scan_cum = rot_cum_end;
                #pragma unroll 1
                for (int rot_s = 0; rot_s < rot_scan_n; rot_s++) {
                    int _max_10 = ((0) > (rot_k - rot_scan_cum + 1) ? (0) : (rot_k - rot_scan_cum + 1));
                    int _min_9 = ((1) < (_max_10) ? (1) : (_max_10));
                    int _max_11 = ((0) > (num_groups - 1 - rot_scan_g) ? (0) : (num_groups - 1 - rot_scan_g));
                    int _min_10 = ((1) < (_max_11) ? (1) : (_max_11));
                    int rot_more = _min_9 * _min_10;
                    rot_scan_g += rot_more;
                    int _max_12 = ((rot_scan_g - 1) > (0) ? (rot_scan_g - 1) : (0));
                    int _min_11 = ((rot_scan_g) < (1) ? (rot_scan_g) : (1));
                    int prev_end_1 = grouped_layout[_max_12] * _min_11;
                    int rot_k_start_2 = (prev_end_1 + 256 - 1) / 256 * 256;
                    int rot_shape_k_1 = grouped_layout[rot_scan_g] - rot_k_start_2;
                    int _max_13 = ((1) > ((rot_shape_k_1 + 256 - 1) / 256) ? (1) : ((rot_shape_k_1 + 256 - 1) / 256));
                    int cost_g_1 = _max_13;
                    int rot_rank_1 = 0;
                    #pragma unroll 1
                    for (int rot_h_1 = 0; rot_h_1 < num_groups; rot_h_1++) {
                        int _max_14 = ((rot_h_1 - 1) > (0) ? (rot_h_1 - 1) : (0));
                        int _min_12 = ((rot_h_1) < (1) ? (rot_h_1) : (1));
                        int prev_end_0_1 = grouped_layout[_max_14] * _min_12;
                        int rot_k_start_1_1 = (prev_end_0_1 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_1 = grouped_layout[rot_h_1] - rot_k_start_1_1;
                        int _max_15 = ((1) > ((rot_shape_k_2_1 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_1 + 256 - 1) / 256));
                        int cost_h_1 = _max_15;
                        int _max_16 = ((0) > (cost_h_1 - cost_g_1) ? (0) : (cost_h_1 - cost_g_1));
                        int _min_13 = ((1) < (_max_16) ? (1) : (_max_16));
                        int _max_17 = ((cost_h_1 - cost_g_1) > (cost_g_1 - cost_h_1) ? (cost_h_1 - cost_g_1) : (cost_g_1 - cost_h_1));
                        int _min_14 = ((1) < (_max_17) ? (1) : (_max_17));
                        int _max_18 = ((0) > (rot_scan_g - rot_h_1) ? (0) : (rot_scan_g - rot_h_1));
                        int _min_15 = ((1) < (_max_18) ? (1) : (_max_18));
                        rot_rank_1 += _min_13 + (1 - _min_14) * _min_15;
                    }
                    int rot_scan_first = (rot_bid + rot_b - rot_rank_1 * rot_r % rot_b) % rot_b;
                    int _max_19 = ((0) > (rot_r - rot_scan_first) ? (0) : (rot_r - rot_scan_first));
                    int _min_16 = ((1) < (_max_19) ? (1) : (_max_19));
                    rot_scan_cum += rot_more * (rot_q + _min_16);
                }
                int _min_17 = ((1) < (rot_q) ? (1) : (rot_q));
                int _min_18 = ((1) < (rot_q) ? (1) : (rot_q));
                rot_adv_n = rot_adv_n * _min_17 + (rot_scan_g - current_group) * (1 - _min_18);
                #pragma unroll 1
                for (int rot_adv = 0; rot_adv < rot_adv_n; rot_adv++) {
                    int aligned_k = (current_shape_k + 256 - 1) / 256 * 256;
                    current_sf_start += (aligned_k + 127) / 128;
                    current_group += 1;
                    {
                        current_k_start = (current_k_end + 256 - 1) / 256 * 256;
                        current_k_end = grouped_layout[current_group];
                        current_shape_k = current_k_end - current_k_start;
                    }
                    int _max_20 = ((current_group - 1) > (0) ? (current_group - 1) : (0));
                    int _min_19 = ((current_group) < (1) ? (current_group) : (1));
                    int prev_end_2 = grouped_layout[_max_20] * _min_19;
                    int rot_k_start_3 = (prev_end_2 + 256 - 1) / 256 * 256;
                    int rot_shape_k_3 = grouped_layout[current_group] - rot_k_start_3;
                    int _max_21 = ((1) > ((rot_shape_k_3 + 256 - 1) / 256) ? (1) : ((rot_shape_k_3 + 256 - 1) / 256));
                    int cost_g_2 = _max_21;
                    int rot_rank_2 = 0;
                    #pragma unroll 1
                    for (int rot_h_2 = 0; rot_h_2 < num_groups; rot_h_2++) {
                        int _max_22 = ((rot_h_2 - 1) > (0) ? (rot_h_2 - 1) : (0));
                        int _min_20 = ((rot_h_2) < (1) ? (rot_h_2) : (1));
                        int prev_end_0_2 = grouped_layout[_max_22] * _min_20;
                        int rot_k_start_1_2 = (prev_end_0_2 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_2 = grouped_layout[rot_h_2] - rot_k_start_1_2;
                        int _max_23 = ((1) > ((rot_shape_k_2_2 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_2 + 256 - 1) / 256));
                        int cost_h_2 = _max_23;
                        int _max_24 = ((0) > (cost_h_2 - cost_g_2) ? (0) : (cost_h_2 - cost_g_2));
                        int _min_21 = ((1) < (_max_24) ? (1) : (_max_24));
                        int _max_25 = ((cost_h_2 - cost_g_2) > (cost_g_2 - cost_h_2) ? (cost_h_2 - cost_g_2) : (cost_g_2 - cost_h_2));
                        int _min_22 = ((1) < (_max_25) ? (1) : (_max_25));
                        int _max_26 = ((0) > (current_group - rot_h_2) ? (0) : (current_group - rot_h_2));
                        int _min_23 = ((1) < (_max_26) ? (1) : (_max_26));
                        rot_rank_2 += _min_21 + (1 - _min_22) * _min_23;
                    }
                    rot_first = (rot_bid + rot_b - rot_rank_2 * rot_r % rot_b) % rot_b;
                    rot_cum_begin = rot_cum_end;
                    int _max_27 = ((0) > (rot_r - rot_first) ? (0) : (rot_r - rot_first));
                    int _min_24 = ((1) < (_max_27) ? (1) : (_max_27));
                    rot_cum_end += rot_q + _min_24;
                }
                int rot_tile = current_group * rot_t + rot_first + (rot_k - rot_cum_begin) * rot_b;
                int _max_28 = ((1) > ((current_shape_k + 256 - 1) / 256) ? (1) : ((current_shape_k + 256 - 1) / 256));
                int current_k_tiles = _max_28;
                int block_idx = rot_tile - current_group * (grid_m * grid_n);
                int swizzle_group = (int)((unsigned long long)block_idx * (unsigned long long)fd_swizzle_mult >> (unsigned long long)fd_swizzle_shift);
                int first_m = swizzle_group * 16;
                int in_group = block_idx - swizzle_group * (grid_n * 16);
                int _min_25 = ((16) < (grid_m - first_m) ? (16) : (grid_m - first_m));
                int m_blocks = _min_25;
                int _max_29 = ((0) > (first_m + 16 - grid_m) ? (0) : (first_m + 16 - grid_m));
                int _min_26 = ((1) < (_max_29) ? (1) : (_max_29));
                int is_tail = _min_26;
                int full_n = in_group / 16;
                int tail_n = (int)((unsigned long long)in_group * (unsigned long long)fd_tail_mult >> (unsigned long long)fd_tail_shift);
                int bid_n = full_n + (tail_n - full_n) * is_tail;
                int bid_m = first_m + in_group - bid_n * m_blocks;
                int off_m = bid_m * 128;
                int off_n = bid_n * 256;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < current_k_tiles; iter_k++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    if (elect_sync()) {
                        {
                            tma_2d_gmem2smem(smem_sfa_addr + load_stage * 35840, (&SFA), off_m, current_sf_start + iter_k * 2, tma_full_addr + (load_stage) * 8);
                        }
                        tma_2d_gmem2smem(smem_sfb_addr + load_stage * 35840, (&SFB), off_n, current_sf_start + iter_k * 2, tma_full_addr + (load_stage) * 8);
                        {
                            mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 2048 + ((1) ? 1024 : 0));
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 35840, (&A), 0, off_m, current_k_start / 256 + iter_k, ((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 35840, (&B), 0, off_n + cta_rank * 128, current_k_start / 256 + iter_k, ((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (cta_rank == 0) {
                                mbarrier_arrive_expect_tx(sf_ready_addr + (load_stage) * 8, 65536);
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((sf_ready_addr + (load_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
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
            int num_tiles_1 = num_groups * grid_m * grid_n;
            int current_group_1 = 0;
            int current_k_start_1 = 0;
            int current_sf_start_1 = 0;
            int current_shape_k_1 = grouped_layout[0];
            int current_k_end_1 = current_shape_k_1;
            unsigned int fd_tiles_mult_1 = (unsigned int)grouped_layout[num_groups];
            unsigned int fd_tiles_shift_1 = (unsigned int)grouped_layout[num_groups + 1];
            unsigned int fd_swizzle_mult_1 = (unsigned int)grouped_layout[num_groups + 2];
            unsigned int fd_swizzle_shift_1 = (unsigned int)grouped_layout[num_groups + 3];
            unsigned int fd_tail_mult_1 = (unsigned int)grouped_layout[num_groups + 4];
            unsigned int fd_tail_shift_1 = (unsigned int)grouped_layout[num_groups + 5];
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_sf_ready = 0;
            if (cta_rank == 0) {
                int rot_b_1 = num_bids;
                int rot_t_1 = grid_m * grid_n;
                int rot_q_1 = rot_t_1 / rot_b_1;
                int rot_r_1 = rot_t_1 - rot_q_1 * rot_b_1;
                int rot_bid_1 = bid;
                int rot_first_1 = 0;
                int rot_iters_1 = 0;
                #pragma unroll 1
                for (int rot_g_1 = 0; rot_g_1 < num_groups; rot_g_1++) {
                    int rot_gd_1 = num_groups - 1 - rot_g_1;
                    int _max_59 = ((rot_gd_1 - 1) > (0) ? (rot_gd_1 - 1) : (0));
                    int _min_52 = ((rot_gd_1) < (1) ? (rot_gd_1) : (1));
                    int prev_end_3 = grouped_layout[_max_59] * _min_52;
                    int rot_k_start_4 = (prev_end_3 + 256 - 1) / 256 * 256;
                    int rot_shape_k_4 = grouped_layout[rot_gd_1] - rot_k_start_4;
                    int _max_60 = ((1) > ((rot_shape_k_4 + 256 - 1) / 256) ? (1) : ((rot_shape_k_4 + 256 - 1) / 256));
                    int cost_g_3 = _max_60;
                    int rot_rank_3 = 0;
                    #pragma unroll 1
                    for (int rot_h_3 = 0; rot_h_3 < num_groups; rot_h_3++) {
                        int _max_61 = ((rot_h_3 - 1) > (0) ? (rot_h_3 - 1) : (0));
                        int _min_53 = ((rot_h_3) < (1) ? (rot_h_3) : (1));
                        int prev_end_0_3 = grouped_layout[_max_61] * _min_53;
                        int rot_k_start_1_3 = (prev_end_0_3 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_3 = grouped_layout[rot_h_3] - rot_k_start_1_3;
                        int _max_62 = ((1) > ((rot_shape_k_2_3 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_3 + 256 - 1) / 256));
                        int cost_h_3 = _max_62;
                        int _max_63 = ((0) > (cost_h_3 - cost_g_3) ? (0) : (cost_h_3 - cost_g_3));
                        int _min_54 = ((1) < (_max_63) ? (1) : (_max_63));
                        int _max_64 = ((cost_h_3 - cost_g_3) > (cost_g_3 - cost_h_3) ? (cost_h_3 - cost_g_3) : (cost_g_3 - cost_h_3));
                        int _min_55 = ((1) < (_max_64) ? (1) : (_max_64));
                        int _max_65 = ((0) > (rot_gd_1 - rot_h_3) ? (0) : (rot_gd_1 - rot_h_3));
                        int _min_56 = ((1) < (_max_65) ? (1) : (_max_65));
                        rot_rank_3 += _min_54 + (1 - _min_55) * _min_56;
                    }
                    rot_first_1 = (rot_bid_1 + rot_b_1 - rot_rank_3 * rot_r_1 % rot_b_1) % rot_b_1;
                    int _max_66 = ((0) > (rot_r_1 - rot_first_1) ? (0) : (rot_r_1 - rot_first_1));
                    int _min_57 = ((1) < (_max_66) ? (1) : (_max_66));
                    rot_iters_1 += rot_q_1 + _min_57;
                }
                int rot_loop_end_1 = rot_bid_1 + rot_iters_1 * rot_b_1;
                int rot_cum_begin_1 = 0;
                int _max_67 = ((0) > (rot_r_1 - rot_first_1) ? (0) : (rot_r_1 - rot_first_1));
                int _min_58 = ((1) < (_max_67) ? (1) : (_max_67));
                int rot_cum_end_1 = rot_q_1 + _min_58;
                #pragma unroll 1
                for (unsigned int this_bid_1 = bid; this_bid_1 < rot_loop_end_1; this_bid_1 += num_bids) {
                    int rot_k_1 = (this_bid_1 - (unsigned int)rot_bid_1) / (unsigned int)rot_b_1;
                    int _max_68 = ((0) > (rot_k_1 - rot_cum_end_1 + 1) ? (0) : (rot_k_1 - rot_cum_end_1 + 1));
                    int _min_59 = ((1) < (_max_68) ? (1) : (_max_68));
                    int rot_adv_n_1 = _min_59;
                    int _min_60 = ((1) < (rot_q_1) ? (1) : (rot_q_1));
                    int rot_scan_n_1 = num_groups * (1 - _min_60);
                    int rot_scan_g_1 = current_group_1;
                    int rot_scan_cum_1 = rot_cum_end_1;
                    #pragma unroll 1
                    for (int rot_s_1 = 0; rot_s_1 < rot_scan_n_1; rot_s_1++) {
                        int _max_69 = ((0) > (rot_k_1 - rot_scan_cum_1 + 1) ? (0) : (rot_k_1 - rot_scan_cum_1 + 1));
                        int _min_61 = ((1) < (_max_69) ? (1) : (_max_69));
                        int _max_70 = ((0) > (num_groups - 1 - rot_scan_g_1) ? (0) : (num_groups - 1 - rot_scan_g_1));
                        int _min_62 = ((1) < (_max_70) ? (1) : (_max_70));
                        int rot_more_1 = _min_61 * _min_62;
                        rot_scan_g_1 += rot_more_1;
                        int _max_71 = ((rot_scan_g_1 - 1) > (0) ? (rot_scan_g_1 - 1) : (0));
                        int _min_63 = ((rot_scan_g_1) < (1) ? (rot_scan_g_1) : (1));
                        int prev_end_4 = grouped_layout[_max_71] * _min_63;
                        int rot_k_start_5 = (prev_end_4 + 256 - 1) / 256 * 256;
                        int rot_shape_k_5 = grouped_layout[rot_scan_g_1] - rot_k_start_5;
                        int _max_72 = ((1) > ((rot_shape_k_5 + 256 - 1) / 256) ? (1) : ((rot_shape_k_5 + 256 - 1) / 256));
                        int cost_g_4 = _max_72;
                        int rot_rank_4 = 0;
                        #pragma unroll 1
                        for (int rot_h_4 = 0; rot_h_4 < num_groups; rot_h_4++) {
                            int _max_73 = ((rot_h_4 - 1) > (0) ? (rot_h_4 - 1) : (0));
                            int _min_64 = ((rot_h_4) < (1) ? (rot_h_4) : (1));
                            int prev_end_0_4 = grouped_layout[_max_73] * _min_64;
                            int rot_k_start_1_4 = (prev_end_0_4 + 256 - 1) / 256 * 256;
                            int rot_shape_k_2_4 = grouped_layout[rot_h_4] - rot_k_start_1_4;
                            int _max_74 = ((1) > ((rot_shape_k_2_4 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_4 + 256 - 1) / 256));
                            int cost_h_4 = _max_74;
                            int _max_75 = ((0) > (cost_h_4 - cost_g_4) ? (0) : (cost_h_4 - cost_g_4));
                            int _min_65 = ((1) < (_max_75) ? (1) : (_max_75));
                            int _max_76 = ((cost_h_4 - cost_g_4) > (cost_g_4 - cost_h_4) ? (cost_h_4 - cost_g_4) : (cost_g_4 - cost_h_4));
                            int _min_66 = ((1) < (_max_76) ? (1) : (_max_76));
                            int _max_77 = ((0) > (rot_scan_g_1 - rot_h_4) ? (0) : (rot_scan_g_1 - rot_h_4));
                            int _min_67 = ((1) < (_max_77) ? (1) : (_max_77));
                            rot_rank_4 += _min_65 + (1 - _min_66) * _min_67;
                        }
                        int rot_scan_first_1 = (rot_bid_1 + rot_b_1 - rot_rank_4 * rot_r_1 % rot_b_1) % rot_b_1;
                        int _max_78 = ((0) > (rot_r_1 - rot_scan_first_1) ? (0) : (rot_r_1 - rot_scan_first_1));
                        int _min_68 = ((1) < (_max_78) ? (1) : (_max_78));
                        rot_scan_cum_1 += rot_more_1 * (rot_q_1 + _min_68);
                    }
                    int _min_69 = ((1) < (rot_q_1) ? (1) : (rot_q_1));
                    int _min_70 = ((1) < (rot_q_1) ? (1) : (rot_q_1));
                    rot_adv_n_1 = rot_adv_n_1 * _min_69 + (rot_scan_g_1 - current_group_1) * (1 - _min_70);
                    #pragma unroll 1
                    for (int rot_adv_1 = 0; rot_adv_1 < rot_adv_n_1; rot_adv_1++) {
                        int aligned_k_1 = (current_shape_k_1 + 256 - 1) / 256 * 256;
                        current_sf_start_1 += (aligned_k_1 + 127) / 128;
                        current_group_1 += 1;
                        {
                            current_k_start_1 = (current_k_end_1 + 256 - 1) / 256 * 256;
                            current_k_end_1 = grouped_layout[current_group_1];
                            current_shape_k_1 = current_k_end_1 - current_k_start_1;
                        }
                        int _max_79 = ((current_group_1 - 1) > (0) ? (current_group_1 - 1) : (0));
                        int _min_71 = ((current_group_1) < (1) ? (current_group_1) : (1));
                        int prev_end_5 = grouped_layout[_max_79] * _min_71;
                        int rot_k_start_6 = (prev_end_5 + 256 - 1) / 256 * 256;
                        int rot_shape_k_6 = grouped_layout[current_group_1] - rot_k_start_6;
                        int _max_80 = ((1) > ((rot_shape_k_6 + 256 - 1) / 256) ? (1) : ((rot_shape_k_6 + 256 - 1) / 256));
                        int cost_g_5 = _max_80;
                        int rot_rank_5 = 0;
                        #pragma unroll 1
                        for (int rot_h_5 = 0; rot_h_5 < num_groups; rot_h_5++) {
                            int _max_81 = ((rot_h_5 - 1) > (0) ? (rot_h_5 - 1) : (0));
                            int _min_72 = ((rot_h_5) < (1) ? (rot_h_5) : (1));
                            int prev_end_0_5 = grouped_layout[_max_81] * _min_72;
                            int rot_k_start_1_5 = (prev_end_0_5 + 256 - 1) / 256 * 256;
                            int rot_shape_k_2_5 = grouped_layout[rot_h_5] - rot_k_start_1_5;
                            int _max_82 = ((1) > ((rot_shape_k_2_5 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_5 + 256 - 1) / 256));
                            int cost_h_5 = _max_82;
                            int _max_83 = ((0) > (cost_h_5 - cost_g_5) ? (0) : (cost_h_5 - cost_g_5));
                            int _min_73 = ((1) < (_max_83) ? (1) : (_max_83));
                            int _max_84 = ((cost_h_5 - cost_g_5) > (cost_g_5 - cost_h_5) ? (cost_h_5 - cost_g_5) : (cost_g_5 - cost_h_5));
                            int _min_74 = ((1) < (_max_84) ? (1) : (_max_84));
                            int _max_85 = ((0) > (current_group_1 - rot_h_5) ? (0) : (current_group_1 - rot_h_5));
                            int _min_75 = ((1) < (_max_85) ? (1) : (_max_85));
                            rot_rank_5 += _min_73 + (1 - _min_74) * _min_75;
                        }
                        rot_first_1 = (rot_bid_1 + rot_b_1 - rot_rank_5 * rot_r_1 % rot_b_1) % rot_b_1;
                        rot_cum_begin_1 = rot_cum_end_1;
                        int _max_86 = ((0) > (rot_r_1 - rot_first_1) ? (0) : (rot_r_1 - rot_first_1));
                        int _min_76 = ((1) < (_max_86) ? (1) : (_max_86));
                        rot_cum_end_1 += rot_q_1 + _min_76;
                    }
                    int rot_tile_1 = current_group_1 * rot_t_1 + rot_first_1 + (rot_k_1 - rot_cum_begin_1) * rot_b_1;
                    int _max_87 = ((1) > ((current_shape_k_1 + 256 - 1) / 256) ? (1) : ((current_shape_k_1 + 256 - 1) / 256));
                    int current_k_tiles_1 = _max_87;
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int iter_k_1 = 0; iter_k_1 < current_k_tiles_1; iter_k_1++) {
                        mbarrier_wait(sf_ready_addr + (mma_tma_stage) * 8, _phase_sf_ready);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 2240)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 2240 + 32)));
                            }
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2240)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2240 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2240 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 2240 + 96)));
                        }
                        __syncwarp();
                        if (iter_k_1 == 0 && mma_iteration > 0) {
                            unsigned int preceding_stage = (mma_iteration - 1) % 2;
                            unsigned int preceding_phase = (mma_iteration - 1) / 2 & 1;
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_wait(overlap_done_addr + (preceding_stage) * 8, preceding_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        if (elect_sync()) {
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2240;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2240;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 232)), a_desc + 0, b_desc + 0,
                                    0x10c00480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                                tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 232)), a_desc + 2, b_desc + 2,
                                    0x50c004a0U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2240;
                            int _mma_b_lo_1 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2240;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 232)), a_desc + 0, b_desc + 0,
                                    0x10c00480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                                tcgen05_mma_mxf4_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 232)), a_desc + 2, b_desc + 2,
                                    0x50c004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
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
            int num_tiles_2 = num_groups * grid_m * grid_n;
            int current_group_2 = 0;
            int current_k_start_2 = 0;
            int current_sf_start_2 = 0;
            int current_shape_k_2 = grouped_layout[0];
            int current_k_end_2 = current_shape_k_2;
            unsigned int fd_tiles_mult_2 = (unsigned int)grouped_layout[num_groups];
            unsigned int fd_tiles_shift_2 = (unsigned int)grouped_layout[num_groups + 1];
            unsigned int fd_swizzle_mult_2 = (unsigned int)grouped_layout[num_groups + 2];
            unsigned int fd_swizzle_shift_2 = (unsigned int)grouped_layout[num_groups + 3];
            unsigned int fd_tail_mult_2 = (unsigned int)grouped_layout[num_groups + 4];
            unsigned int fd_tail_shift_2 = (unsigned int)grouped_layout[num_groups + 5];
            int rot_b_2 = num_bids;
            int rot_t_2 = grid_m * grid_n;
            int rot_q_2 = rot_t_2 / rot_b_2;
            int rot_r_2 = rot_t_2 - rot_q_2 * rot_b_2;
            int rot_bid_2 = bid;
            int rot_first_2 = 0;
            int rot_iters_2 = 0;
            #pragma unroll 1
            for (int rot_g_2 = 0; rot_g_2 < num_groups; rot_g_2++) {
                int rot_gd_2 = num_groups - 1 - rot_g_2;
                int _max_30 = ((rot_gd_2 - 1) > (0) ? (rot_gd_2 - 1) : (0));
                int _min_27 = ((rot_gd_2) < (1) ? (rot_gd_2) : (1));
                int prev_end_6 = grouped_layout[_max_30] * _min_27;
                int rot_k_start_7 = (prev_end_6 + 256 - 1) / 256 * 256;
                int rot_shape_k_7 = grouped_layout[rot_gd_2] - rot_k_start_7;
                int _max_31 = ((1) > ((rot_shape_k_7 + 256 - 1) / 256) ? (1) : ((rot_shape_k_7 + 256 - 1) / 256));
                int cost_g_6 = _max_31;
                int rot_rank_6 = 0;
                #pragma unroll 1
                for (int rot_h_6 = 0; rot_h_6 < num_groups; rot_h_6++) {
                    int _max_32 = ((rot_h_6 - 1) > (0) ? (rot_h_6 - 1) : (0));
                    int _min_28 = ((rot_h_6) < (1) ? (rot_h_6) : (1));
                    int prev_end_0_6 = grouped_layout[_max_32] * _min_28;
                    int rot_k_start_1_6 = (prev_end_0_6 + 256 - 1) / 256 * 256;
                    int rot_shape_k_2_6 = grouped_layout[rot_h_6] - rot_k_start_1_6;
                    int _max_33 = ((1) > ((rot_shape_k_2_6 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_6 + 256 - 1) / 256));
                    int cost_h_6 = _max_33;
                    int _max_34 = ((0) > (cost_h_6 - cost_g_6) ? (0) : (cost_h_6 - cost_g_6));
                    int _min_29 = ((1) < (_max_34) ? (1) : (_max_34));
                    int _max_35 = ((cost_h_6 - cost_g_6) > (cost_g_6 - cost_h_6) ? (cost_h_6 - cost_g_6) : (cost_g_6 - cost_h_6));
                    int _min_30 = ((1) < (_max_35) ? (1) : (_max_35));
                    int _max_36 = ((0) > (rot_gd_2 - rot_h_6) ? (0) : (rot_gd_2 - rot_h_6));
                    int _min_31 = ((1) < (_max_36) ? (1) : (_max_36));
                    rot_rank_6 += _min_29 + (1 - _min_30) * _min_31;
                }
                rot_first_2 = (rot_bid_2 + rot_b_2 - rot_rank_6 * rot_r_2 % rot_b_2) % rot_b_2;
                int _max_37 = ((0) > (rot_r_2 - rot_first_2) ? (0) : (rot_r_2 - rot_first_2));
                int _min_32 = ((1) < (_max_37) ? (1) : (_max_37));
                rot_iters_2 += rot_q_2 + _min_32;
            }
            int rot_loop_end_2 = rot_bid_2 + rot_iters_2 * rot_b_2;
            int rot_cum_begin_2 = 0;
            int _max_38 = ((0) > (rot_r_2 - rot_first_2) ? (0) : (rot_r_2 - rot_first_2));
            int _min_33 = ((1) < (_max_38) ? (1) : (_max_38));
            int rot_cum_end_2 = rot_q_2 + _min_33;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int this_bid_2 = bid; this_bid_2 < rot_loop_end_2; this_bid_2 += num_bids) {
                int rot_k_2 = (this_bid_2 - (unsigned int)rot_bid_2) / (unsigned int)rot_b_2;
                int _max_39 = ((0) > (rot_k_2 - rot_cum_end_2 + 1) ? (0) : (rot_k_2 - rot_cum_end_2 + 1));
                int _min_34 = ((1) < (_max_39) ? (1) : (_max_39));
                int rot_adv_n_2 = _min_34;
                int _min_35 = ((1) < (rot_q_2) ? (1) : (rot_q_2));
                int rot_scan_n_2 = num_groups * (1 - _min_35);
                int rot_scan_g_2 = current_group_2;
                int rot_scan_cum_2 = rot_cum_end_2;
                #pragma unroll 1
                for (int rot_s_2 = 0; rot_s_2 < rot_scan_n_2; rot_s_2++) {
                    int _max_40 = ((0) > (rot_k_2 - rot_scan_cum_2 + 1) ? (0) : (rot_k_2 - rot_scan_cum_2 + 1));
                    int _min_36 = ((1) < (_max_40) ? (1) : (_max_40));
                    int _max_41 = ((0) > (num_groups - 1 - rot_scan_g_2) ? (0) : (num_groups - 1 - rot_scan_g_2));
                    int _min_37 = ((1) < (_max_41) ? (1) : (_max_41));
                    int rot_more_2 = _min_36 * _min_37;
                    rot_scan_g_2 += rot_more_2;
                    int _max_42 = ((rot_scan_g_2 - 1) > (0) ? (rot_scan_g_2 - 1) : (0));
                    int _min_38 = ((rot_scan_g_2) < (1) ? (rot_scan_g_2) : (1));
                    int prev_end_7 = grouped_layout[_max_42] * _min_38;
                    int rot_k_start_8 = (prev_end_7 + 256 - 1) / 256 * 256;
                    int rot_shape_k_8 = grouped_layout[rot_scan_g_2] - rot_k_start_8;
                    int _max_43 = ((1) > ((rot_shape_k_8 + 256 - 1) / 256) ? (1) : ((rot_shape_k_8 + 256 - 1) / 256));
                    int cost_g_7 = _max_43;
                    int rot_rank_7 = 0;
                    #pragma unroll 1
                    for (int rot_h_7 = 0; rot_h_7 < num_groups; rot_h_7++) {
                        int _max_44 = ((rot_h_7 - 1) > (0) ? (rot_h_7 - 1) : (0));
                        int _min_39 = ((rot_h_7) < (1) ? (rot_h_7) : (1));
                        int prev_end_0_7 = grouped_layout[_max_44] * _min_39;
                        int rot_k_start_1_7 = (prev_end_0_7 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_7 = grouped_layout[rot_h_7] - rot_k_start_1_7;
                        int _max_45 = ((1) > ((rot_shape_k_2_7 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_7 + 256 - 1) / 256));
                        int cost_h_7 = _max_45;
                        int _max_46 = ((0) > (cost_h_7 - cost_g_7) ? (0) : (cost_h_7 - cost_g_7));
                        int _min_40 = ((1) < (_max_46) ? (1) : (_max_46));
                        int _max_47 = ((cost_h_7 - cost_g_7) > (cost_g_7 - cost_h_7) ? (cost_h_7 - cost_g_7) : (cost_g_7 - cost_h_7));
                        int _min_41 = ((1) < (_max_47) ? (1) : (_max_47));
                        int _max_48 = ((0) > (rot_scan_g_2 - rot_h_7) ? (0) : (rot_scan_g_2 - rot_h_7));
                        int _min_42 = ((1) < (_max_48) ? (1) : (_max_48));
                        rot_rank_7 += _min_40 + (1 - _min_41) * _min_42;
                    }
                    int rot_scan_first_2 = (rot_bid_2 + rot_b_2 - rot_rank_7 * rot_r_2 % rot_b_2) % rot_b_2;
                    int _max_49 = ((0) > (rot_r_2 - rot_scan_first_2) ? (0) : (rot_r_2 - rot_scan_first_2));
                    int _min_43 = ((1) < (_max_49) ? (1) : (_max_49));
                    rot_scan_cum_2 += rot_more_2 * (rot_q_2 + _min_43);
                }
                int _min_44 = ((1) < (rot_q_2) ? (1) : (rot_q_2));
                int _min_45 = ((1) < (rot_q_2) ? (1) : (rot_q_2));
                rot_adv_n_2 = rot_adv_n_2 * _min_44 + (rot_scan_g_2 - current_group_2) * (1 - _min_45);
                #pragma unroll 1
                for (int rot_adv_2 = 0; rot_adv_2 < rot_adv_n_2; rot_adv_2++) {
                    int aligned_k_2 = (current_shape_k_2 + 256 - 1) / 256 * 256;
                    current_sf_start_2 += (aligned_k_2 + 127) / 128;
                    current_group_2 += 1;
                    {
                        current_k_start_2 = (current_k_end_2 + 256 - 1) / 256 * 256;
                        current_k_end_2 = grouped_layout[current_group_2];
                        current_shape_k_2 = current_k_end_2 - current_k_start_2;
                    }
                    int _max_50 = ((current_group_2 - 1) > (0) ? (current_group_2 - 1) : (0));
                    int _min_46 = ((current_group_2) < (1) ? (current_group_2) : (1));
                    int prev_end_8 = grouped_layout[_max_50] * _min_46;
                    int rot_k_start_9 = (prev_end_8 + 256 - 1) / 256 * 256;
                    int rot_shape_k_9 = grouped_layout[current_group_2] - rot_k_start_9;
                    int _max_51 = ((1) > ((rot_shape_k_9 + 256 - 1) / 256) ? (1) : ((rot_shape_k_9 + 256 - 1) / 256));
                    int cost_g_8 = _max_51;
                    int rot_rank_8 = 0;
                    #pragma unroll 1
                    for (int rot_h_8 = 0; rot_h_8 < num_groups; rot_h_8++) {
                        int _max_52 = ((rot_h_8 - 1) > (0) ? (rot_h_8 - 1) : (0));
                        int _min_47 = ((rot_h_8) < (1) ? (rot_h_8) : (1));
                        int prev_end_0_8 = grouped_layout[_max_52] * _min_47;
                        int rot_k_start_1_8 = (prev_end_0_8 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_8 = grouped_layout[rot_h_8] - rot_k_start_1_8;
                        int _max_53 = ((1) > ((rot_shape_k_2_8 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_8 + 256 - 1) / 256));
                        int cost_h_8 = _max_53;
                        int _max_54 = ((0) > (cost_h_8 - cost_g_8) ? (0) : (cost_h_8 - cost_g_8));
                        int _min_48 = ((1) < (_max_54) ? (1) : (_max_54));
                        int _max_55 = ((cost_h_8 - cost_g_8) > (cost_g_8 - cost_h_8) ? (cost_h_8 - cost_g_8) : (cost_g_8 - cost_h_8));
                        int _min_49 = ((1) < (_max_55) ? (1) : (_max_55));
                        int _max_56 = ((0) > (current_group_2 - rot_h_8) ? (0) : (current_group_2 - rot_h_8));
                        int _min_50 = ((1) < (_max_56) ? (1) : (_max_56));
                        rot_rank_8 += _min_48 + (1 - _min_49) * _min_50;
                    }
                    rot_first_2 = (rot_bid_2 + rot_b_2 - rot_rank_8 * rot_r_2 % rot_b_2) % rot_b_2;
                    rot_cum_begin_2 = rot_cum_end_2;
                    int _max_57 = ((0) > (rot_r_2 - rot_first_2) ? (0) : (rot_r_2 - rot_first_2));
                    int _min_51 = ((1) < (_max_57) ? (1) : (_max_57));
                    rot_cum_end_2 += rot_q_2 + _min_51;
                }
                int rot_tile_2 = current_group_2 * rot_t_2 + rot_first_2 + (rot_k_2 - rot_cum_begin_2) * rot_b_2;
                int _max_58 = ((1) > ((current_shape_k_2 + 256 - 1) / 256) ? (1) : ((current_shape_k_2 + 256 - 1) / 256));
                int current_k_tiles_2 = _max_58;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < current_k_tiles_2; iter_k_2++) {
                    mbarrier_wait(tma_full_addr + (transpose_stage) * 8, _phase_tma_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int sfa_smem = smem_sfa_addr + transpose_stage * 35840 + (unsigned int)(sf_subblock * 512);
                    int sfb_smem = smem_sfb_addr + transpose_stage * 35840 + (unsigned int)(sf_subblock * 1024);
                    {
                        unsigned int _sf_v[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(sfa_smem + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(sfa_smem + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(sfa_smem + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(sfa_smem + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_smem + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_smem + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_smem + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_smem + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v[3])));
                    }
                    #pragma unroll
                    for (int sf_block = 0; sf_block < 2; sf_block++) {
                        unsigned int _sf_v_1[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[0])) : "r"(sfb_smem + sf_block * 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[1])) : "r"(sfb_smem + sf_block * 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[2])) : "r"(sfb_smem + sf_block * 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[3])) : "r"(sfb_smem + sf_block * 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_smem + sf_block * 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_smem + sf_block * 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_smem + sf_block * 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_smem + sf_block * 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_1[3])));
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
            int num_tiles_3 = num_groups * grid_m * grid_n;
            int current_group_3 = 0;
            int current_k_start_3 = 0;
            int current_sf_start_3 = 0;
            int current_shape_k_3 = grouped_layout[0];
            int current_k_end_3 = current_shape_k_3;
            unsigned int fd_tiles_mult_3 = (unsigned int)grouped_layout[num_groups];
            unsigned int fd_tiles_shift_3 = (unsigned int)grouped_layout[num_groups + 1];
            unsigned int fd_swizzle_mult_3 = (unsigned int)grouped_layout[num_groups + 2];
            unsigned int fd_swizzle_shift_3 = (unsigned int)grouped_layout[num_groups + 3];
            unsigned int fd_tail_mult_3 = (unsigned int)grouped_layout[num_groups + 4];
            unsigned int fd_tail_shift_3 = (unsigned int)grouped_layout[num_groups + 5];
            unsigned int store_stage = 0;
            int rot_b_3 = num_bids;
            int rot_t_3 = grid_m * grid_n;
            int rot_q_3 = rot_t_3 / rot_b_3;
            int rot_r_3 = rot_t_3 - rot_q_3 * rot_b_3;
            int rot_bid_3 = bid;
            int rot_first_3 = 0;
            int rot_iters_3 = 0;
            #pragma unroll 1
            for (int rot_g_3 = 0; rot_g_3 < num_groups; rot_g_3++) {
                int rot_gd_3 = num_groups - 1 - rot_g_3;
                int _max_88 = ((rot_gd_3 - 1) > (0) ? (rot_gd_3 - 1) : (0));
                int _min_77 = ((rot_gd_3) < (1) ? (rot_gd_3) : (1));
                int prev_end_9 = grouped_layout[_max_88] * _min_77;
                int rot_k_start_10 = (prev_end_9 + 256 - 1) / 256 * 256;
                int rot_shape_k_10 = grouped_layout[rot_gd_3] - rot_k_start_10;
                int _max_89 = ((1) > ((rot_shape_k_10 + 256 - 1) / 256) ? (1) : ((rot_shape_k_10 + 256 - 1) / 256));
                int cost_g_9 = _max_89;
                int rot_rank_9 = 0;
                #pragma unroll 1
                for (int rot_h_9 = 0; rot_h_9 < num_groups; rot_h_9++) {
                    int _max_90 = ((rot_h_9 - 1) > (0) ? (rot_h_9 - 1) : (0));
                    int _min_78 = ((rot_h_9) < (1) ? (rot_h_9) : (1));
                    int prev_end_0_9 = grouped_layout[_max_90] * _min_78;
                    int rot_k_start_1_9 = (prev_end_0_9 + 256 - 1) / 256 * 256;
                    int rot_shape_k_2_9 = grouped_layout[rot_h_9] - rot_k_start_1_9;
                    int _max_91 = ((1) > ((rot_shape_k_2_9 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_9 + 256 - 1) / 256));
                    int cost_h_9 = _max_91;
                    int _max_92 = ((0) > (cost_h_9 - cost_g_9) ? (0) : (cost_h_9 - cost_g_9));
                    int _min_79 = ((1) < (_max_92) ? (1) : (_max_92));
                    int _max_93 = ((cost_h_9 - cost_g_9) > (cost_g_9 - cost_h_9) ? (cost_h_9 - cost_g_9) : (cost_g_9 - cost_h_9));
                    int _min_80 = ((1) < (_max_93) ? (1) : (_max_93));
                    int _max_94 = ((0) > (rot_gd_3 - rot_h_9) ? (0) : (rot_gd_3 - rot_h_9));
                    int _min_81 = ((1) < (_max_94) ? (1) : (_max_94));
                    rot_rank_9 += _min_79 + (1 - _min_80) * _min_81;
                }
                rot_first_3 = (rot_bid_3 + rot_b_3 - rot_rank_9 * rot_r_3 % rot_b_3) % rot_b_3;
                int _max_95 = ((0) > (rot_r_3 - rot_first_3) ? (0) : (rot_r_3 - rot_first_3));
                int _min_82 = ((1) < (_max_95) ? (1) : (_max_95));
                rot_iters_3 += rot_q_3 + _min_82;
            }
            int rot_loop_end_3 = rot_bid_3 + rot_iters_3 * rot_b_3;
            int rot_cum_begin_3 = 0;
            int _max_96 = ((0) > (rot_r_3 - rot_first_3) ? (0) : (rot_r_3 - rot_first_3));
            int _min_83 = ((1) < (_max_96) ? (1) : (_max_96));
            int rot_cum_end_3 = rot_q_3 + _min_83;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int this_bid_3 = bid; this_bid_3 < rot_loop_end_3; this_bid_3 += num_bids) {
                int rot_k_3 = (this_bid_3 - (unsigned int)rot_bid_3) / (unsigned int)rot_b_3;
                int _max_97 = ((0) > (rot_k_3 - rot_cum_end_3 + 1) ? (0) : (rot_k_3 - rot_cum_end_3 + 1));
                int _min_84 = ((1) < (_max_97) ? (1) : (_max_97));
                int rot_adv_n_3 = _min_84;
                int _min_85 = ((1) < (rot_q_3) ? (1) : (rot_q_3));
                int rot_scan_n_3 = num_groups * (1 - _min_85);
                int rot_scan_g_3 = current_group_3;
                int rot_scan_cum_3 = rot_cum_end_3;
                #pragma unroll 1
                for (int rot_s_3 = 0; rot_s_3 < rot_scan_n_3; rot_s_3++) {
                    int _max_98 = ((0) > (rot_k_3 - rot_scan_cum_3 + 1) ? (0) : (rot_k_3 - rot_scan_cum_3 + 1));
                    int _min_86 = ((1) < (_max_98) ? (1) : (_max_98));
                    int _max_99 = ((0) > (num_groups - 1 - rot_scan_g_3) ? (0) : (num_groups - 1 - rot_scan_g_3));
                    int _min_87 = ((1) < (_max_99) ? (1) : (_max_99));
                    int rot_more_3 = _min_86 * _min_87;
                    rot_scan_g_3 += rot_more_3;
                    int _max_100 = ((rot_scan_g_3 - 1) > (0) ? (rot_scan_g_3 - 1) : (0));
                    int _min_88 = ((rot_scan_g_3) < (1) ? (rot_scan_g_3) : (1));
                    int prev_end_10 = grouped_layout[_max_100] * _min_88;
                    int rot_k_start_11 = (prev_end_10 + 256 - 1) / 256 * 256;
                    int rot_shape_k_11 = grouped_layout[rot_scan_g_3] - rot_k_start_11;
                    int _max_101 = ((1) > ((rot_shape_k_11 + 256 - 1) / 256) ? (1) : ((rot_shape_k_11 + 256 - 1) / 256));
                    int cost_g_10 = _max_101;
                    int rot_rank_10 = 0;
                    #pragma unroll 1
                    for (int rot_h_10 = 0; rot_h_10 < num_groups; rot_h_10++) {
                        int _max_102 = ((rot_h_10 - 1) > (0) ? (rot_h_10 - 1) : (0));
                        int _min_89 = ((rot_h_10) < (1) ? (rot_h_10) : (1));
                        int prev_end_0_10 = grouped_layout[_max_102] * _min_89;
                        int rot_k_start_1_10 = (prev_end_0_10 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_10 = grouped_layout[rot_h_10] - rot_k_start_1_10;
                        int _max_103 = ((1) > ((rot_shape_k_2_10 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_10 + 256 - 1) / 256));
                        int cost_h_10 = _max_103;
                        int _max_104 = ((0) > (cost_h_10 - cost_g_10) ? (0) : (cost_h_10 - cost_g_10));
                        int _min_90 = ((1) < (_max_104) ? (1) : (_max_104));
                        int _max_105 = ((cost_h_10 - cost_g_10) > (cost_g_10 - cost_h_10) ? (cost_h_10 - cost_g_10) : (cost_g_10 - cost_h_10));
                        int _min_91 = ((1) < (_max_105) ? (1) : (_max_105));
                        int _max_106 = ((0) > (rot_scan_g_3 - rot_h_10) ? (0) : (rot_scan_g_3 - rot_h_10));
                        int _min_92 = ((1) < (_max_106) ? (1) : (_max_106));
                        rot_rank_10 += _min_90 + (1 - _min_91) * _min_92;
                    }
                    int rot_scan_first_3 = (rot_bid_3 + rot_b_3 - rot_rank_10 * rot_r_3 % rot_b_3) % rot_b_3;
                    int _max_107 = ((0) > (rot_r_3 - rot_scan_first_3) ? (0) : (rot_r_3 - rot_scan_first_3));
                    int _min_93 = ((1) < (_max_107) ? (1) : (_max_107));
                    rot_scan_cum_3 += rot_more_3 * (rot_q_3 + _min_93);
                }
                int _min_94 = ((1) < (rot_q_3) ? (1) : (rot_q_3));
                int _min_95 = ((1) < (rot_q_3) ? (1) : (rot_q_3));
                rot_adv_n_3 = rot_adv_n_3 * _min_94 + (rot_scan_g_3 - current_group_3) * (1 - _min_95);
                #pragma unroll 1
                for (int rot_adv_3 = 0; rot_adv_3 < rot_adv_n_3; rot_adv_3++) {
                    int aligned_k_3 = (current_shape_k_3 + 256 - 1) / 256 * 256;
                    current_sf_start_3 += (aligned_k_3 + 127) / 128;
                    current_group_3 += 1;
                    {
                        current_k_start_3 = (current_k_end_3 + 256 - 1) / 256 * 256;
                        current_k_end_3 = grouped_layout[current_group_3];
                        current_shape_k_3 = current_k_end_3 - current_k_start_3;
                    }
                    int _max_108 = ((current_group_3 - 1) > (0) ? (current_group_3 - 1) : (0));
                    int _min_96 = ((current_group_3) < (1) ? (current_group_3) : (1));
                    int prev_end_11 = grouped_layout[_max_108] * _min_96;
                    int rot_k_start_12 = (prev_end_11 + 256 - 1) / 256 * 256;
                    int rot_shape_k_12 = grouped_layout[current_group_3] - rot_k_start_12;
                    int _max_109 = ((1) > ((rot_shape_k_12 + 256 - 1) / 256) ? (1) : ((rot_shape_k_12 + 256 - 1) / 256));
                    int cost_g_11 = _max_109;
                    int rot_rank_11 = 0;
                    #pragma unroll 1
                    for (int rot_h_11 = 0; rot_h_11 < num_groups; rot_h_11++) {
                        int _max_110 = ((rot_h_11 - 1) > (0) ? (rot_h_11 - 1) : (0));
                        int _min_97 = ((rot_h_11) < (1) ? (rot_h_11) : (1));
                        int prev_end_0_11 = grouped_layout[_max_110] * _min_97;
                        int rot_k_start_1_11 = (prev_end_0_11 + 256 - 1) / 256 * 256;
                        int rot_shape_k_2_11 = grouped_layout[rot_h_11] - rot_k_start_1_11;
                        int _max_111 = ((1) > ((rot_shape_k_2_11 + 256 - 1) / 256) ? (1) : ((rot_shape_k_2_11 + 256 - 1) / 256));
                        int cost_h_11 = _max_111;
                        int _max_112 = ((0) > (cost_h_11 - cost_g_11) ? (0) : (cost_h_11 - cost_g_11));
                        int _min_98 = ((1) < (_max_112) ? (1) : (_max_112));
                        int _max_113 = ((cost_h_11 - cost_g_11) > (cost_g_11 - cost_h_11) ? (cost_h_11 - cost_g_11) : (cost_g_11 - cost_h_11));
                        int _min_99 = ((1) < (_max_113) ? (1) : (_max_113));
                        int _max_114 = ((0) > (current_group_3 - rot_h_11) ? (0) : (current_group_3 - rot_h_11));
                        int _min_100 = ((1) < (_max_114) ? (1) : (_max_114));
                        rot_rank_11 += _min_98 + (1 - _min_99) * _min_100;
                    }
                    rot_first_3 = (rot_bid_3 + rot_b_3 - rot_rank_11 * rot_r_3 % rot_b_3) % rot_b_3;
                    rot_cum_begin_3 = rot_cum_end_3;
                    int _max_115 = ((0) > (rot_r_3 - rot_first_3) ? (0) : (rot_r_3 - rot_first_3));
                    int _min_101 = ((1) < (_max_115) ? (1) : (_max_115));
                    rot_cum_end_3 += rot_q_3 + _min_101;
                }
                int rot_tile_3 = current_group_3 * rot_t_3 + rot_first_3 + (rot_k_3 - rot_cum_begin_3) * rot_b_3;
                int _max_116 = ((1) > ((current_shape_k_3 + 256 - 1) / 256) ? (1) : ((current_shape_k_3 + 256 - 1) / 256));
                int current_k_tiles_3 = _max_116;
                int block_idx_1 = rot_tile_3 - current_group_3 * (grid_m * grid_n);
                int swizzle_group_1 = (int)((unsigned long long)block_idx_1 * (unsigned long long)fd_swizzle_mult_3 >> (unsigned long long)fd_swizzle_shift_3);
                int first_m_1 = swizzle_group_1 * 16;
                int in_group_1 = block_idx_1 - swizzle_group_1 * (grid_n * 16);
                int _min_102 = ((16) < (grid_m - first_m_1) ? (16) : (grid_m - first_m_1));
                int m_blocks_1 = _min_102;
                int _max_117 = ((0) > (first_m_1 + 16 - grid_m) ? (0) : (first_m_1 + 16 - grid_m));
                int _min_103 = ((1) < (_max_117) ? (1) : (_max_117));
                int is_tail_1 = _min_103;
                int full_n_1 = in_group_1 / 16;
                int tail_n_1 = (int)((unsigned long long)in_group_1 * (unsigned long long)fd_tail_mult_3 >> (unsigned long long)fd_tail_shift_3);
                int bid_n_1 = full_n_1 + (tail_n_1 - full_n_1) * is_tail_1;
                int bid_m_1 = first_m_1 + in_group_1 - bid_n_1 * m_blocks_1;
                int off_m_1 = bid_m_1 * 128;
                int off_n_1 = bid_n_1 * 256;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (current_shape_k_3 != 0) {
                    #pragma unroll
                    for (int epi_pass = 0; epi_pass < 8; epi_pass++) {
                        int store_idx = ((epi_stage == 0) ? 7 - epi_pass : epi_pass);
                        int col_start = store_idx * 32;
                        int staging_smem = epi_staging_addr + store_stage * 16384;
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        #pragma unroll
                        for (int n = 0; n < 8; n++) {
                            int row = cta_rank * 128 + epi_warp * 32;
                            int load_idx = ((epi_stage == 0) ? 7 - n : n);
                            int col = epi_stage * 232 + (unsigned int)col_start + (unsigned int)(load_idx * 4);
                            int tmem_addr = col;
                            float _tmem_load_0[4];
                            tmem_ld_x4(&_tmem_load_0[0], tmem_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #pragma unroll
                            for (int element = 0; element < 4; element++) {
                                _tmem_load_0[element] = ((current_shape_k_3 == 0) ? 0.0f : _tmem_load_0[element]);
                            }
                            if (epi_pass == 0 && n + 1 == 6) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((overlap_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            if (epi_pass == 7 && n == 7) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            int c_stage_row = store_stage * 128 + (unsigned int)(epi_warp * 32) + (unsigned int)lane;
                            unsigned int c_abs = epi_staging_addr + (unsigned int)(c_stage_row * 128);
                            unsigned int c_swz = c_abs / 8 & 112;
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + ((unsigned int)(c_stage_row * 128) + ((unsigned int)(load_idx * 16) ^ c_swz)))), "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]), "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                    #error "TmaReduceAdd3d requires SM90 or newer"
                                    #endif
                                    asm volatile(
                                        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
                                        " [%0, {%1, %2, %3}], [%4];"
                                        :: "l"((&C_tma)), "r"(off_n_1 + col_start), "r"(off_m_1), "r"(current_group_3), "r"(staging_smem) : "memory");
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                        store_stage = 0;
                    }
                } else {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((overlap_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
            if (warp == 4) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

} // extern "C"
