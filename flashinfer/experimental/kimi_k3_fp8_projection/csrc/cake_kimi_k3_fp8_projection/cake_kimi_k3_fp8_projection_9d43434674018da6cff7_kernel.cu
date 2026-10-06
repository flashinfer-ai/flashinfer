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
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_RES_PIPE_STAGES 4
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 34816
#define SMEM_SMEM_SFW0_OFF 33792
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 34816
#define SMEM_SMEM_SFW1_OFF 34304
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 34816
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 34816
#define SMEM_SMEM_SFX_ALL_OFF 34816
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 34816
#define SMEM_SMEM_SFX0_OFF 34816
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 34816
#define SMEM_SMEM_SFX1_OFF 35328
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 34816
#define SMEM_SMEM_XRES_OFF 105472
#define SMEM_SMEM_XRES_STAGE_BYTES 8192
#define SMEM_SMEM_XRES_STRIDE 9216
#define SMEM_SMEM_SFXRES0_OFF 113664
#define SMEM_SMEM_SFXRES0_STAGE_BYTES 512
#define SMEM_SMEM_SFXRES0_STRIDE 9216
#define SMEM_SMEM_SFXRES1_OFF 114176
#define SMEM_SMEM_SFXRES1_STAGE_BYTES 512
#define SMEM_SMEM_SFXRES1_STRIDE 9216
#define SMEM_SMEM_EPI_OFF 142336
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_TOTAL 158720
#define THREADS 448


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

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_fp8_projection_9d43434674018da6cff7(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB, const __grid_constant__ CUtensorMap OUT)
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
    #define res_full_addr (mbar_base + 48)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfw0_addr = smem + 33792;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + 34304);
    const int smem_sfw1_addr = smem + 34304;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_sfx_all_addr = smem + 34816;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_sfx0_addr = smem + 34816;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + 35328);
    const int smem_sfx1_addr = smem + 35328;
    uint8_t* smem_xres = reinterpret_cast<uint8_t*>(smem_raw + 105472);
    const int smem_xres_addr = smem + 105472;
    uint8_t* smem_sfxres0 = reinterpret_cast<uint8_t*>(smem_raw + 113664);
    const int smem_sfxres0_addr = smem + 113664;
    uint8_t* smem_sfxres1 = reinterpret_cast<uint8_t*>(smem_raw + 114176);
    const int smem_sfxres1_addr = smem + 114176;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 142336);
    const int smem_epi_addr = smem + 142336;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'res_pipe' ---
            // res_full: 4 barriers, init_count=8
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 88, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
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
                int w0 = total_work * bid / num_bids;
                int ws = 1;
                int n_items = total_work * (bid + 1) / num_bids - w0;
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
                    int x_row = m_tile * 32;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 34816, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 34816, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 34816, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 33792);
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = total_work * bid / num_bids;
            int ws_m = 1;
            int n_items_m = total_work * (bid + 1) / num_bids - w0_m;
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_res_full = 0;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int tile_1 = work_m / split;
                int rank_1 = work_m - tile_1 * split;
                int k_begin_1 = num_k_iters * rank_1 / split;
                int k_end_1 = num_k_iters * (rank_1 + 1) / split;
                int k_count_1 = k_end_1 - k_begin_1;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                int m_tile_m = tile_1 / n_tiles;
                if (m_tile_m != m_prev_m) {
                    mbarrier_wait(res_full_addr + (res_stage_m) * 8, _phase_res_full);
                    res_stage_m += 1;
                    if (res_stage_m == 4) { res_stage_m = 0; _phase_res_full ^= 1; }
                }
                m_prev_m = m_tile_m;
                unsigned int cur_slot = res_stage_m - 1;
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 2176)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfxres0_addr) >> 4) + (cur_slot) * 576)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 2176)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfxres1_addr) >> 4) + (cur_slot) * 576)));
                        int _mma_a_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_0 = (((smem_xres_addr) >> 4) & 0x3FFF) + (cur_slot) * 576;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                        }
                        int _mma_a_lo_1 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 2176;
                        int _mma_b_lo_1 = (((smem_xres_addr + 4096) >> 4) & 0x3FFF) + (cur_slot) * 576;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_tma_full ^= 1; }
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
            int tok_g0 = epi_group * 32;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = total_work * bid / num_bids;
            int ws_e = 1;
            int n_items_e = total_work * (bid + 1) / num_bids - w0_e;
            unsigned int acc_stage_e = 0;
            unsigned int _phase_mainloop_done = 0;
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
                int tok0 = m_tile_e * 32;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
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
                        asm volatile("cp.async.bulk.wait_group.read 0;");
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
                } else {
                    unsigned long long pbase = (unsigned long long)(tile_2 * split + rank_2) * 4096 + (unsigned long long)lane_row + (unsigned long long)tok_g0 * 128;
                    #pragma unroll
                    for (int t_1 = 0; t_1 < 32; t_1++) {
                        partials[pbase + (unsigned long long)(t_1 * 128)] = _tmem_load_0[t_1];
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
                    int _min_3 = ((32) < (t_first + tok_per_cta) ? (32) : (t_first + tok_per_cta));
                    int t_last = _min_3;
                    unsigned long long rbase = (unsigned long long)(tile_2 * split) * 4096 + (unsigned long long)lane_row;
                    #pragma unroll
                    for (int tb = 0; tb < 32; tb++) {
                        if (t_first < tb + 1 && t_last > tb && tok_g0 <= tb && tb < tok_g0 + 32) {
                            float total[1];
                            #pragma unroll
                            for (int u = 0; u < 1; u++) {
                                total[u] = 0.0f;
                            }
                            #pragma unroll 1
                            for (int s_0 = 0; s_0 < split; s_0 += 16) {
                                float parts[16];
                                #pragma unroll
                                for (int j = 0; j < 16; j++) {
                                    int _min_4 = ((s_0 + j) < (split - 1) ? (s_0 + j) : (split - 1));
                                    int s_j = _min_4;
                                    unsigned long long rank_off = rbase + (unsigned long long)s_j * 4096;
                                    #pragma unroll
                                    for (int u_1 = 0; u_1 < 1; u_1++) {
                                        parts[u_1 * 16 + j] = partials[rank_off + (unsigned long long)((tb + u_1) * 128)];
                                    }
                                }
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 16; j_1++) {
                                    if (s_0 + j_1 < split) {
                                        #pragma unroll
                                        for (int u_2 = 0; u_2 < 1; u_2++) {
                                            total[u_2] = total[u_2] + parts[u_2 * 16 + j_1];
                                        }
                                    }
                                }
                            }
                            if (feature < n_valid) {
                                #pragma unroll
                                for (int u_3 = 0; u_3 < 1; u_3++) {
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
        }
    }
    // ---- Role: quant ----
    if (warp >= 6 && warp <= 13) {
        { // quant_main
            int w0_q = total_work * bid / num_bids;
            int ws_q = 1;
            int n_items_q = total_work * (bid + 1) / num_bids - w0_q;
            const int qwarp = warp - 6;
            const int half = lane >> 4;
            const int half_lane = lane & 15;
            const int lane_q = lane;
            unsigned int q_stage = 0;
            unsigned int xb_q = 0;
            unsigned int xq_q = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int res_stage_q = 0;
            int m_prev_r = -1;
            #pragma unroll 1
            for (int it_r = 0; it_r < n_items_q; it_r++) {
                int work_r = w0_q + it_r * ws_q;
                int tile_3 = work_r / split;
                int rank_3 = work_r - tile_3 * split;
                int k_begin_3 = num_k_iters * rank_3 / split;
                int k_end_3 = num_k_iters * (rank_3 + 1) / split;
                int k_count_3 = k_end_3 - k_begin_3;
                int m_tile_r = tile_3 / n_tiles;
                int tok0_r = m_tile_r * 32;
                if (m_tile_r != m_prev_r) {
                    unsigned int x_slot = smem_xres_addr + res_stage_q * 9216;
                    unsigned int sf_slot_off = res_stage_q * 9216;
                    float xv_all[32];
                    #pragma unroll
                    for (int it_1 = 0; it_1 < 4; it_1++) {
                        int unit_l = (it_1 * 8 + qwarp) * 2 + half;
                        int tok_l = unit_l >> 1;
                        int kb_l = unit_l & 1;
                        int _min_0 = ((tok0_r + tok_l) < (M - 1) ? (tok0_r + tok_l) : (M - 1));
                        int tok_ld = _min_0;
                        int _min_1 = ((kb_l * 128) < (K - 128) ? (kb_l * 128) : (K - 128));
                        int k_ld = _min_1 + half_lane * 8;
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + ((unsigned long long)tok_ld * (unsigned long long)K + (unsigned long long)k_ld) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 8; j_2++) {
                            xv_all[8 * it_1 + j_2] = _vec_load_0[j_2];
                        }
                    }
                    float absmax_r[4];
                    #pragma unroll
                    for (int it_2 = 0; it_2 < 4; it_2++) {
                        float mags_r[8];
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            mags_r[j_3] = xv_all[8 * it_2 + j_3];
                        }
                        float _fabs_0 = fabsf(mags_r[0]);
                        mags_r[0] = _fabs_0;
                        float _fabs_1 = fabsf(mags_r[1]);
                        mags_r[1] = _fabs_1;
                        float _fabs_2 = fabsf(mags_r[2]);
                        mags_r[2] = _fabs_2;
                        float _fabs_3 = fabsf(mags_r[3]);
                        mags_r[3] = _fabs_3;
                        float _fabs_4 = fabsf(mags_r[4]);
                        mags_r[4] = _fabs_4;
                        float _fabs_5 = fabsf(mags_r[5]);
                        mags_r[5] = _fabs_5;
                        float _fabs_6 = fabsf(mags_r[6]);
                        mags_r[6] = _fabs_6;
                        float _fabs_7 = fabsf(mags_r[7]);
                        mags_r[7] = _fabs_7;
                        float mags_r_max = mags_r[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 8; _lr++) {
                            mags_r_max = max_noftz(mags_r_max, mags_r[_lr]);
                        }
                        absmax_r[it_2] = mags_r_max;
                    }
                    #pragma unroll
                    for (int it_3 = 0; it_3 < 4; it_3++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, absmax_r[it_3], 1);
                        float o_r = _shfl_xor_0;
                        float _max_0 = max_noftz(absmax_r[it_3], o_r);
                        absmax_r[it_3] = _max_0;
                    }
                    #pragma unroll
                    for (int it_4 = 0; it_4 < 4; it_4++) {
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, absmax_r[it_4], 2);
                        float o_r_1 = _shfl_xor_1;
                        float _max_1 = max_noftz(absmax_r[it_4], o_r_1);
                        absmax_r[it_4] = _max_1;
                    }
                    #pragma unroll
                    for (int it_5 = 0; it_5 < 4; it_5++) {
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, absmax_r[it_5], 4);
                        float o_r_2 = _shfl_xor_2;
                        float _max_2 = max_noftz(absmax_r[it_5], o_r_2);
                        absmax_r[it_5] = _max_2;
                    }
                    #pragma unroll
                    for (int it_6 = 0; it_6 < 4; it_6++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, absmax_r[it_6], 8);
                        float o_r_3 = _shfl_xor_3;
                        float _max_3 = max_noftz(absmax_r[it_6], o_r_3);
                        absmax_r[it_6] = _max_3;
                    }
                    #pragma unroll
                    for (int it_7 = 0; it_7 < 4; it_7++) {
                        int unit_r = (it_7 * 8 + qwarp) * 2 + half;
                        int tok_local_r = unit_r >> 1;
                        int kb_r = unit_r & 1;
                        int tok_r2 = tok0_r + tok_local_r;
                        float vals_r[8];
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            vals_r[j_4] = xv_all[8 * it_7 + j_4];
                        }
                        float _max_4 = max_noftz(absmax_r[it_7], 0.0001f);
                        float amax_r = _max_4;
                        float scale_r = amax_r / 448.0f;
                        unsigned int scale_bits_r = __as_u32(scale_r);
                        unsigned int exponent_r = scale_bits_r >> 23 & 255;
                        unsigned int mantissa_r = scale_bits_r & 8388607;
                        unsigned int has_mantissa_r = ((mantissa_r != 0) ? 1 : 0);
                        unsigned int _max_5 = ((exponent_r + has_mantissa_r) > (1) ? (exponent_r + has_mantissa_r) : (1));
                        unsigned int _min_2 = ((_max_5) < (254) ? (_max_5) : (254));
                        unsigned int scale_byte_r = _min_2;
                        unsigned int inverse_bits_r = 254 - scale_byte_r << 23;
                        float inverse_r = 0.0f;
                        inverse_r = reinterpret_cast<float*>(&inverse_bits_r)[0];
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_1 = {inverse_r, inverse_r};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(vals_r)[_ls], _scale2_1);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++) {
                            vals_r[_ls] = vals_r[_ls] * inverse_r;
                        }
                        #endif
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vals_r[1]), "f"(vals_r[0]));
                        uint16_t q0 = _e4m3x2_f32_0;
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(vals_r[3]), "f"(vals_r[2]));
                        uint16_t q1 = _e4m3x2_f32_1;
                        uint16_t _e4m3x2_f32_2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(vals_r[5]), "f"(vals_r[4]));
                        uint16_t q2 = _e4m3x2_f32_2;
                        uint16_t _e4m3x2_f32_3;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(vals_r[7]), "f"(vals_r[6]));
                        uint16_t q3 = _e4m3x2_f32_3;
                        unsigned int wr0 = (unsigned int)q0 | (unsigned int)q1 << 16;
                        unsigned int wr1 = (unsigned int)q2 | (unsigned int)q3 << 16;
                        unsigned int panel_r = x_slot + (unsigned int)kb_r * 4096;
                        unsigned int sf4_r = scale_byte_r * 16843009;
                        float sf4_bits_r = 0.0f;
                        sf4_bits_r = reinterpret_cast<float*>(&sf4_r)[0];
                        unsigned int sf_off_r = sf_slot_off + (unsigned int)((tok_local_r & 31) * 16 + (tok_local_r >> 5) * 4);
                        if (tok_r2 < M) {
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((panel_r + (unsigned int)(tok_local_r * 128 + half_lane * 8 ^ (tok_local_r * 128 + half_lane * 8 >> 7 & 7) << 4))), "r"(wr0), "r"(wr1) : "memory");
                            if (half_lane == 0) {
                                if (kb_r == 0) {
                                    {
                                        uint32_t _addr_2 = static_cast<uint32_t>(smem_sfxres0_addr + sf_off_r);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(sf4_bits_r) : "memory");
                                    }
                                } else {
                                    {
                                        uint32_t _addr_3 = static_cast<uint32_t>(smem_sfxres1_addr + sf_off_r);
                                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(sf4_bits_r) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(res_full_addr + (res_stage_q) * 8);
                    }
                    res_stage_q += 1;
                    if (res_stage_q == 4) { res_stage_q = 0; }
                }
                m_prev_r = m_tile_r;
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
