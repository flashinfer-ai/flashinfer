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
#define TMEM_NCOLS 48
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 32
#define TMEM_TMEM_SFX_OFFSET 40
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_XQ_PIPE_STAGES 4
#define NUM_XB_PIPE_STAGES 4
#define NUM_RED_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 33792
#define SMEM_SMEM_SFW0_OFF 33792
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 33792
#define SMEM_SMEM_SFW1_OFF 34304
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 33792
#define SMEM_SMEM_X_OFF 136192
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 9216
#define SMEM_SMEM_SFX_ALL_OFF 144384
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 9216
#define SMEM_SMEM_SFX0_OFF 144384
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 9216
#define SMEM_SMEM_SFX1_OFF 144896
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 9216
#define SMEM_SMEM_XB_OFF 173056
#define SMEM_SMEM_XB_STAGE_BYTES 8192
#define SMEM_SMEM_XB_STRIDE 8192
#define SMEM_SMEM_EPI_OFF 205824
#define SMEM_SMEM_EPI_STAGE_BYTES 8192
#define SMEM_SMEM_EPI_STRIDE 8192
#define SMEM_SMEM_RED_OFF 205824
#define SMEM_SMEM_RED_STAGE_BYTES 10240
#define SMEM_SMEM_RED_STRIDE 10240
#define SMEM_SMEM_STG_OFF 216064
#define SMEM_SMEM_STG_STAGE_BYTES 7680
#define SMEM_SMEM_STG_STRIDE 7680
#define SMEM_TOTAL 223744
#define THREADS 480


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fp8_projection_287eadf0f85f36cea16a(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
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
    #define mma_done_addr (mbar_base + 32)
    #define xq_full_addr (mbar_base + 64)
    #define xq_empty_addr (mbar_base + 96)
    #define xb_full_addr (mbar_base + 128)
    #define xb_empty_addr (mbar_base + 160)
    #define mainloop_done_addr (mbar_base + 192)
    #define epilogue_done_addr (mbar_base + 200)
    #define red_full_addr (mbar_base + 208)
    #define peers_free_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
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
    uint8_t* smem_xb = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_XB_OFF);
    const int smem_xb_addr = smem + SMEM_SMEM_XB_OFF;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_EPI_OFF);
    const int smem_epi_addr = smem + SMEM_SMEM_EPI_OFF;
    float* smem_red = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_RED_OFF);
    const int smem_red_addr = smem + SMEM_SMEM_RED_OFF;
    float* smem_stg = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_STG_OFF);
    const int smem_stg_addr = smem + SMEM_SMEM_STG_OFF;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'xq_pipe' ---
            // xq_full: 4 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // xq_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'xb_pipe' ---
            // xb_full: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // xb_empty: 4 barriers, init_count=8
            mbarrier_init(smem + 160, 8);
            mbarrier_init(smem + 168, 8);
            mbarrier_init(smem + 176, 8);
            mbarrier_init(smem + 184, 8);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 200, 4);
            // --- pipeline 'red_pipe' ---
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // peers_free: 1 barriers, init_count=3
            mbarrier_init(smem + 216, 3);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
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
    const int tmem_tmem_sfw = taddr + 32;
    const int tmem_tmem_sfx = taddr + 40;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int w0 = bid;
                int ws = num_bids;
                int n_items = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int it = 0; it < n_items; it++) {
                    int work = w0 + it * ws;
                    int tile = work / split;
                    int rank = work - tile * split;
                    int k_begin = num_k_iters * rank / split;
                    int k_end = num_k_iters * (rank + 1) / split;
                    int k_count = k_end - k_begin;
                    int n_tile = tile % n_tiles;
                    int m_tile = tile / n_tiles;
                    int w_tile0 = n_tile * (2 * num_k_iters);
                    int x_row = m_tile * 16;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 33792, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 33792, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 33792, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 33792);
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = bid;
            int ws_m = num_bids;
            int n_items_m = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_xq_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int tile_1 = work_m / split;
                int rank_1 = work_m - tile_1 * split;
                int k_begin_1 = num_k_iters * rank_1 / split;
                int k_end_1 = num_k_iters * (rank_1 + 1) / split;
                int k_count_1 = k_end_1 - k_begin_1;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    mbarrier_wait(xq_full_addr + (xq_m) * 8, _phase_xq_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 2112)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (xq_m) * 576)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 2112)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (xq_m) * 576)));
                        int _mma_a_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 2112;
                        int _mma_b_lo_0 = (((smem_x_addr) >> 4) & 0x3FFF) + (xq_m) * 576;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8840000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28840010U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48840020U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68840030U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                        }
                        int init2 = 0;
                        int _mma_a_lo_1 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 2112;
                        int _mma_b_lo_1 = (((smem_x_addr + 4096) >> 4) & 0x3FFF) + (xq_m) * 576;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8840000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28840010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48840020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68840030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    elect_commit(xq_empty_addr + (xq_m) * 8);
                    xq_m += 1;
                    if (xq_m == 4) { xq_m = 0; _phase_xq_full ^= 1; }
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (acc_stage) * 8);
                _phase_epilogue_done ^= 1;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 16;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = bid;
            int ws_e = num_bids;
            int n_items_e = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int acc_stage_e = 0;
            unsigned int red_stage = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_peers_free = 1;
            unsigned int _phase_red_full = 0;
            #pragma unroll 1
            for (int it_e = 0; it_e < n_items_e; it_e++) {
                int work_e = w0_e + it_e * ws_e;
                int tile_2 = work_e / split;
                int rank_2 = work_e - tile_2 * split;
                int k_begin_2 = num_k_iters * rank_2 / split;
                int k_end_2 = num_k_iters * (rank_2 + 1) / split;
                int k_count_2 = k_end_2 - k_begin_2;
                int n_tile_e = tile_2 % n_tiles;
                int m_tile_e = tile_2 / n_tiles;
                int feature = n_tile_e * 128 + lane_row;
                int tok0 = m_tile_e * 16;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (acc_stage_e) * 8);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 8192);
                int my_rank = cta_rank;
                int _min_2 = ((16) < ((my_rank + 1) * 4) ? (16) : ((my_rank + 1) * 4));
                int rows_mine = _min_2 - my_rank * 4;
                unsigned int line_b = (unsigned int)(lane_row * 20);
                unsigned int slot_b = (unsigned int)(my_rank * 2560);
                mbarrier_wait_cluster_hint(peers_free_addr + (red_stage) * 8, _phase_peers_free, 10000000);
                if (tid == 64) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        {
                            uint32_t _addr_0 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_2 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_3 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(_tmem_load_0[3]) : "memory");
                        }
                    } else {
                        int _max_6 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_3 = ((1) < (_max_6) ? (1) : (_max_6));
                        int o_idx = -_min_3;
                        unsigned int stg_b = (unsigned int)(o_idx * 2560) + line_b;
                        {
                            uint32_t _addr_4 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_5 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_6 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_7 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(_tmem_load_0[3]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_8 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_9 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_10 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(_tmem_load_0[7]) : "memory");
                        }
                    } else {
                        int _max_7 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_4 = ((1) < (_max_7) ? (1) : (_max_7));
                        int o_idx_1 = 1 - _min_4;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 2560) + line_b;
                        {
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[7]) : "memory");
                        }
                    }
                    if (2 == my_rank) {
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_17 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_18 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[11]) : "memory");
                        }
                    } else {
                        int _max_8 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_5 = ((1) < (_max_8) ? (1) : (_max_8));
                        int o_idx_2 = 2 - _min_5;
                        unsigned int stg_b_2 = (unsigned int)(o_idx_2 * 2560) + line_b;
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_stg_addr + stg_b_2);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[11]) : "memory");
                        }
                    }
                    if (3 == my_rank) {
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(_tmem_load_0[15]) : "memory");
                        }
                    } else {
                        int _max_9 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_6 = ((1) < (_max_9) ? (1) : (_max_9));
                        int o_idx_3 = 3 - _min_6;
                        unsigned int stg_b_3 = (unsigned int)(o_idx_3 * 2560) + line_b;
                        {
                            uint32_t _addr_28 = static_cast<uint32_t>(smem_stg_addr + stg_b_3);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[15]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_7 = ((4) < (rows_mine) ? (4) : (rows_mine));
                int _max_10 = ((0) > (_min_7) ? (0) : (_min_7));
                int rows_round = _max_10;
                if (tid == 64) {
                    if (0 != my_rank) {
                        int _max_11 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_8 = ((1) < (_max_11) ? (1) : (_max_11));
                        int o_idx_c = -_min_8;
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(smem_red_addr + slot_b), "r"(0));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(red_full_addr), "r"(0));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_0), "r"(smem_stg_addr + (unsigned int)(o_idx_c * 2560)), "r"((uint32_t)(2560)), "r"(_mapa_1)
                            : "memory");
                    }
                    if (1 != my_rank) {
                        int _max_12 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_9 = ((1) < (_max_12) ? (1) : (_max_12));
                        int o_idx_c_1 = 1 - _min_9;
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(smem_red_addr + slot_b), "r"(1));
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(red_full_addr), "r"(1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_2), "r"(smem_stg_addr + (unsigned int)(o_idx_c_1 * 2560)), "r"((uint32_t)(2560)), "r"(_mapa_3)
                            : "memory");
                    }
                    if (2 != my_rank) {
                        int _max_13 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_10 = ((1) < (_max_13) ? (1) : (_max_13));
                        int o_idx_c_2 = 2 - _min_10;
                        uint32_t _mapa_4;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_4) : "r"(smem_red_addr + slot_b), "r"(2));
                        uint32_t _mapa_5;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_5) : "r"(red_full_addr), "r"(2));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_4), "r"(smem_stg_addr + (unsigned int)(o_idx_c_2 * 2560)), "r"((uint32_t)(2560)), "r"(_mapa_5)
                            : "memory");
                    }
                    if (3 != my_rank) {
                        int _max_14 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_11 = ((1) < (_max_14) ? (1) : (_max_14));
                        int o_idx_c_3 = 3 - _min_11;
                        uint32_t _mapa_6;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_6) : "r"(smem_red_addr + slot_b), "r"(3));
                        uint32_t _mapa_7;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_7) : "r"(red_full_addr), "r"(3));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_6), "r"(smem_stg_addr + (unsigned int)(o_idx_c_3 * 2560)), "r"((uint32_t)(2560)), "r"(_mapa_7)
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    int _min_12 = ((1) < (rows_round) ? (1) : (rows_round));
                    mbarrier_arrive_expect_tx(red_full_addr + (red_stage) * 8, _min_12 * 7680);
                }
                mbarrier_wait_cluster_hint(red_full_addr + (red_stage) * 8, _phase_red_full, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total = 0.0f;
                        #pragma unroll
                        for (int src = 0; src < 4; src++) {
                            total = total + smem_red[src * 640 + lane_row * 5];
                        }
                        int tok_c = tok0;
                        if (feature < n_valid && tok_c < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total);
                        }
                        float total_0 = 0.0f;
                        #pragma unroll
                        for (int src_1 = 0; src_1 < 4; src_1++) {
                            total_0 = total_0 + smem_red[src_1 * 640 + lane_row * 5 + 1];
                        }
                        int tok_c_1 = tok0 + 1;
                        if (feature < n_valid && tok_c_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0);
                        }
                        float total_2 = 0.0f;
                        #pragma unroll
                        for (int src_2 = 0; src_2 < 4; src_2++) {
                            total_2 = total_2 + smem_red[src_2 * 640 + lane_row * 5 + 2];
                        }
                        int tok_c_3 = tok0 + 2;
                        if (feature < n_valid && tok_c_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2);
                        }
                        float total_4 = 0.0f;
                        #pragma unroll
                        for (int src_3 = 0; src_3 < 4; src_3++) {
                            total_4 = total_4 + smem_red[src_3 * 640 + lane_row * 5 + 3];
                        }
                        int tok_c_5 = tok0 + 3;
                        if (feature < n_valid && tok_c_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4);
                        }
                    }
                    if (1 == my_rank) {
                        float total_1 = 0.0f;
                        #pragma unroll
                        for (int src_4 = 0; src_4 < 4; src_4++) {
                            total_1 = total_1 + smem_red[src_4 * 640 + lane_row * 5];
                        }
                        int tok_c_2 = tok0 + 4;
                        if (feature < n_valid && tok_c_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_1);
                        }
                        float total_0_1 = 0.0f;
                        #pragma unroll
                        for (int src_5 = 0; src_5 < 4; src_5++) {
                            total_0_1 = total_0_1 + smem_red[src_5 * 640 + lane_row * 5 + 1];
                        }
                        int tok_c_1_1 = tok0 + 5;
                        if (feature < n_valid && tok_c_1_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_1);
                        }
                        float total_2_1 = 0.0f;
                        #pragma unroll
                        for (int src_6 = 0; src_6 < 4; src_6++) {
                            total_2_1 = total_2_1 + smem_red[src_6 * 640 + lane_row * 5 + 2];
                        }
                        int tok_c_3_1 = tok0 + 6;
                        if (feature < n_valid && tok_c_3_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_1);
                        }
                        float total_4_1 = 0.0f;
                        #pragma unroll
                        for (int src_7 = 0; src_7 < 4; src_7++) {
                            total_4_1 = total_4_1 + smem_red[src_7 * 640 + lane_row * 5 + 3];
                        }
                        int tok_c_5_1 = tok0 + 7;
                        if (feature < n_valid && tok_c_5_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_1);
                        }
                    }
                    if (2 == my_rank) {
                        float total_3 = 0.0f;
                        #pragma unroll
                        for (int src_8 = 0; src_8 < 4; src_8++) {
                            total_3 = total_3 + smem_red[src_8 * 640 + lane_row * 5];
                        }
                        int tok_c_4 = tok0 + 8;
                        if (feature < n_valid && tok_c_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_3);
                        }
                        float total_0_2 = 0.0f;
                        #pragma unroll
                        for (int src_9 = 0; src_9 < 4; src_9++) {
                            total_0_2 = total_0_2 + smem_red[src_9 * 640 + lane_row * 5 + 1];
                        }
                        int tok_c_1_2 = tok0 + 9;
                        if (feature < n_valid && tok_c_1_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_2);
                        }
                        float total_2_2 = 0.0f;
                        #pragma unroll
                        for (int src_10 = 0; src_10 < 4; src_10++) {
                            total_2_2 = total_2_2 + smem_red[src_10 * 640 + lane_row * 5 + 2];
                        }
                        int tok_c_3_2 = tok0 + 10;
                        if (feature < n_valid && tok_c_3_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_2);
                        }
                        float total_4_2 = 0.0f;
                        #pragma unroll
                        for (int src_11 = 0; src_11 < 4; src_11++) {
                            total_4_2 = total_4_2 + smem_red[src_11 * 640 + lane_row * 5 + 3];
                        }
                        int tok_c_5_2 = tok0 + 11;
                        if (feature < n_valid && tok_c_5_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_2);
                        }
                    }
                    if (3 == my_rank) {
                        float total_5 = 0.0f;
                        #pragma unroll
                        for (int src_12 = 0; src_12 < 4; src_12++) {
                            total_5 = total_5 + smem_red[src_12 * 640 + lane_row * 5];
                        }
                        int tok_c_6 = tok0 + 12;
                        if (feature < n_valid && tok_c_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_5);
                        }
                        float total_0_3 = 0.0f;
                        #pragma unroll
                        for (int src_13 = 0; src_13 < 4; src_13++) {
                            total_0_3 = total_0_3 + smem_red[src_13 * 640 + lane_row * 5 + 1];
                        }
                        int tok_c_1_3 = tok0 + 13;
                        if (feature < n_valid && tok_c_1_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_3);
                        }
                        float total_2_3 = 0.0f;
                        #pragma unroll
                        for (int src_14 = 0; src_14 < 4; src_14++) {
                            total_2_3 = total_2_3 + smem_red[src_14 * 640 + lane_row * 5 + 2];
                        }
                        int tok_c_3_3 = tok0 + 14;
                        if (feature < n_valid && tok_c_3_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_3);
                        }
                        float total_4_3 = 0.0f;
                        #pragma unroll
                        for (int src_15 = 0; src_15 < 4; src_15++) {
                            total_4_3 = total_4_3 + smem_red[src_15 * 640 + lane_row * 5 + 3];
                        }
                        int tok_c_5_3 = tok0 + 15;
                        if (feature < n_valid && tok_c_5_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_3);
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 4; p++) {
                            if (p != my_rank) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(peers_free_addr), "r"(p) : "memory");
                            }
                        }
                    }
                }
                _phase_peers_free ^= 1;
                _phase_red_full ^= 1;
            }
        }
    }
    // ---- Role: quant ----
    if (warp >= 6 && warp <= 13) {
        { // quant_main
            int w0_q = bid;
            int ws_q = num_bids;
            int n_items_q = (total_work - bid + num_bids - 1) / num_bids;
            const int qwarp = warp - 6;
            const int half = lane >> 4;
            const int half_lane = lane & 15;
            const int lane_q = lane;
            unsigned int q_stage = 0;
            unsigned int xb_q = 0;
            unsigned int xq_q = 0;
            unsigned int _phase_xq_empty = 1;
            unsigned int _phase_xb_full = 0;
            #pragma unroll 1
            for (int it_q = 0; it_q < n_items_q; it_q++) {
                int work_q = w0_q + it_q * ws_q;
                int tile_3 = work_q / split;
                int rank_3 = work_q - tile_3 * split;
                int k_begin_3 = num_k_iters * rank_3 / split;
                int k_end_3 = num_k_iters * (rank_3 + 1) / split;
                int k_count_3 = k_end_3 - k_begin_3;
                int m_tile_q = tile_3 / n_tiles;
                int tok0_q = m_tile_q * 16;
                #pragma unroll 1
                for (int i_2 = 0; i_2 < k_count_3; i_2++) {
                    mbarrier_wait(xq_empty_addr + (xq_q) * 8, _phase_xq_empty);
                    mbarrier_wait(xb_full_addr + (xb_q) * 8, _phase_xb_full);
                    unsigned int x_stage = smem_x_addr + xq_q * 9216;
                    unsigned int sf_stage_off = xq_q * 9216;
                    unsigned int xb_stage = smem_xb_addr + xb_q * 8192;
                    unsigned int words_all[8];
                    #pragma unroll
                    for (int it_1 = 0; it_1 < 2; it_1++) {
                        int unit_l = (it_1 * 8 + qwarp) * 2 + half;
                        int tok_l = unit_l >> 1;
                        int kb_l = unit_l & 1;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&words_all[4 * it_1])), "=r"(*reinterpret_cast<uint32_t*>(&words_all[(4 * it_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_all[(4 * it_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_all[(4 * it_1) + 3]))
                            : "r"(xb_stage + (unsigned int)(kb_l * 4096 + tok_l * 256 + half_lane * 16)));
                    }
                    float absmax_all[2];
                    #pragma unroll
                    for (int it_2 = 0; it_2 < 2; it_2++) {
                        float mags[8];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            unsigned int lo_bits_m = words_all[4 * it_2 + j] << 16;
                            unsigned int hi_bits_m = words_all[4 * it_2 + j] & 4294901760u;
                            float lo_m = 0.0f;
                            float hi_m = 0.0f;
                            lo_m = reinterpret_cast<float*>(&lo_bits_m)[0];
                            hi_m = reinterpret_cast<float*>(&hi_bits_m)[0];
                            mags[2 * j] = lo_m;
                            mags[2 * j + 1] = hi_m;
                        }
                        float _fabs_0 = fabsf(mags[0]);
                        mags[0] = _fabs_0;
                        float _fabs_1 = fabsf(mags[1]);
                        mags[1] = _fabs_1;
                        float _fabs_2 = fabsf(mags[2]);
                        mags[2] = _fabs_2;
                        float _fabs_3 = fabsf(mags[3]);
                        mags[3] = _fabs_3;
                        float _fabs_4 = fabsf(mags[4]);
                        mags[4] = _fabs_4;
                        float _fabs_5 = fabsf(mags[5]);
                        mags[5] = _fabs_5;
                        float _fabs_6 = fabsf(mags[6]);
                        mags[6] = _fabs_6;
                        float _fabs_7 = fabsf(mags[7]);
                        mags[7] = _fabs_7;
                        float mags_max = mags[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 8; _lr++) {
                            mags_max = max_noftz(mags_max, mags[_lr]);
                        }
                        absmax_all[it_2] = mags_max;
                    }
                    #pragma unroll
                    for (int it_3 = 0; it_3 < 2; it_3++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, absmax_all[it_3], 1);
                        float o_sh = _shfl_xor_0;
                        float _max_0 = max_noftz(absmax_all[it_3], o_sh);
                        absmax_all[it_3] = _max_0;
                    }
                    #pragma unroll
                    for (int it_4 = 0; it_4 < 2; it_4++) {
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, absmax_all[it_4], 2);
                        float o_sh_1 = _shfl_xor_1;
                        float _max_1 = max_noftz(absmax_all[it_4], o_sh_1);
                        absmax_all[it_4] = _max_1;
                    }
                    #pragma unroll
                    for (int it_5 = 0; it_5 < 2; it_5++) {
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, absmax_all[it_5], 4);
                        float o_sh_2 = _shfl_xor_2;
                        float _max_2 = max_noftz(absmax_all[it_5], o_sh_2);
                        absmax_all[it_5] = _max_2;
                    }
                    #pragma unroll
                    for (int it_6 = 0; it_6 < 2; it_6++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, absmax_all[it_6], 8);
                        float o_sh_3 = _shfl_xor_3;
                        float _max_3 = max_noftz(absmax_all[it_6], o_sh_3);
                        absmax_all[it_6] = _max_3;
                    }
                    #pragma unroll
                    for (int it_7 = 0; it_7 < 2; it_7++) {
                        int unit = (it_7 * 8 + qwarp) * 2 + half;
                        int tok_local = unit >> 1;
                        int kb = unit & 1;
                        int tok = tok0_q + tok_local;
                        float vals[8];
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            unsigned int lo_bits = words_all[4 * it_7 + j_1] << 16;
                            unsigned int hi_bits = words_all[4 * it_7 + j_1] & 4294901760u;
                            float lo = 0.0f;
                            float hi = 0.0f;
                            lo = reinterpret_cast<float*>(&lo_bits)[0];
                            hi = reinterpret_cast<float*>(&hi_bits)[0];
                            vals[2 * j_1] = lo;
                            vals[2 * j_1 + 1] = hi;
                        }
                        float _max_4 = max_noftz(absmax_all[it_7], 0.0001f);
                        float absmax = _max_4;
                        float scale = absmax / 448.0f;
                        unsigned int scale_bits = __as_u32(scale);
                        unsigned int exponent = scale_bits >> 23 & 255;
                        unsigned int mantissa = scale_bits & 8388607;
                        unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
                        unsigned int _max_5 = ((exponent + has_mantissa) > (1) ? (exponent + has_mantissa) : (1));
                        unsigned int _min_1 = ((_max_5) < (254) ? (_max_5) : (254));
                        unsigned int scale_byte = _min_1;
                        unsigned int inverse_bits = 254 - scale_byte << 23;
                        float inverse = 0.0f;
                        inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {inverse, inverse};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++) {
                            vals[_ls] = vals[_ls] * inverse;
                        }
                        #endif
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vals[1]), "f"(vals[0]));
                        uint16_t p0 = _e4m3x2_f32_0;
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(vals[3]), "f"(vals[2]));
                        uint16_t p1 = _e4m3x2_f32_1;
                        uint16_t _e4m3x2_f32_2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(vals[5]), "f"(vals[4]));
                        uint16_t p2 = _e4m3x2_f32_2;
                        uint16_t _e4m3x2_f32_3;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(vals[7]), "f"(vals[6]));
                        uint16_t p3 = _e4m3x2_f32_3;
                        unsigned int w0_1 = (unsigned int)p0 | (unsigned int)p1 << 16;
                        unsigned int w1 = (unsigned int)p2 | (unsigned int)p3 << 16;
                        unsigned int panel = x_stage + (unsigned int)kb * 4096;
                        unsigned int sf4 = scale_byte * 16843009;
                        float sf4_bits = 0.0f;
                        sf4_bits = reinterpret_cast<float*>(&sf4)[0];
                        unsigned int sf_off = sf_stage_off + (unsigned int)((tok_local & 31) * 16 + (tok_local >> 5) * 4);
                        if (tok < M) {
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel + (unsigned int)(tok_local * 128 + half_lane * 8 ^ (tok_local * 128 + half_lane * 8 >> 7 & 7) << 4))), "r"(w0_1), "r"(w1) : "memory");
                            if (half_lane == 0) {
                                if (kb == 0) {
                                    {
                                        uint32_t _addr_1 = static_cast<uint32_t>(smem_sfx0_addr + sf_off);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(sf4_bits) : "memory");
                                    }
                                } else {
                                    {
                                        uint32_t _addr_2 = static_cast<uint32_t>(smem_sfx1_addr + sf_off);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(sf4_bits) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(xb_empty_addr + (xb_q) * 8);
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(xq_full_addr + (xq_q) * 8);
                    }
                    xq_q += 1;
                    if (xq_q == 4) { xq_q = 0; _phase_xq_empty ^= 1; }
                    xb_q += 1;
                    if (xb_q == 4) { xb_q = 0; _phase_xb_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: xload ----
    if (warp == 14) {
        { // xload_main
            unsigned int _phase_xb_empty = 1;
            if (elect_sync()) {
                int w0_x = bid;
                int ws_x = num_bids;
                int n_items_x = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int xb_stage_l = 0;
                #pragma unroll 1
                for (int it_x = 0; it_x < n_items_x; it_x++) {
                    int work_x = w0_x + it_x * ws_x;
                    int tile_4 = work_x / split;
                    int rank_4 = work_x - tile_4 * split;
                    int k_begin_4 = num_k_iters * rank_4 / split;
                    int k_end_4 = num_k_iters * (rank_4 + 1) / split;
                    int k_count_4 = k_end_4 - k_begin_4;
                    int m_tile_x = tile_4 / n_tiles;
                    int x_row_x = m_tile_x * 16;
                    #pragma unroll 1
                    for (int i_3 = 0; i_3 < k_count_4; i_3++) {
                        int k_group_x = (k_begin_4 + i_3) * 2;
                        mbarrier_wait(xb_empty_addr + (xb_stage_l) * 8, _phase_xb_empty);
                        if (it_x == 0 && i_3 == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_xb_addr + xb_stage_l * 8192, (&XB), 0, x_row_x, k_group_x, xb_full_addr + (xb_stage_l) * 8);
                        mbarrier_arrive_expect_tx(xb_full_addr + (xb_stage_l) * 8, 8192);
                        xb_stage_l += 1;
                        if (xb_stage_l == 4) { xb_stage_l = 0; _phase_xb_empty ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"
