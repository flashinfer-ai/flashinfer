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
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_TOTAL 230400
#define THREADS 192
#define num_cluster_tiles (m_tiles * 56)
#define tiles_per_group 1792

extern "C" {

__global__ __launch_bounds__(192) void
kernel_cake_kimi_k3_vision_tower_9da90bee5b5f093d2621(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A2, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ C2, __nv_bfloat16* __restrict__ C3, __nv_bfloat16* __restrict__ R, float* __restrict__ COS, float* __restrict__ SIN, __nv_bfloat16* __restrict__ CS, float* __restrict__ SQ, __nv_bfloat16* __restrict__ XW, __nv_bfloat16* __restrict__ WN, float* __restrict__ WS, unsigned int* __restrict__ FLAGS, const __grid_constant__ CUtensorMap RT, const __grid_constant__ CUtensorMap CT, const __grid_constant__ CUtensorMap XWT, int M, int m_tiles, int full_tiles, int tail_split, int pf_l2, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
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
            unsigned int cluster_id_1 = bid;
            unsigned int num_clusters_1 = num_bids;
            int k_begin = 0;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int item = cluster_id_1; item < num_cluster_tiles; item += num_clusters_1) {
                    int group = item / (unsigned int)tiles_per_group;
                    int first_m = group * 32;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= 32) ? 32 : remaining);
                    int local = item % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * 128;
                    int weight_row = bid_n * 128;
                    if (pf_l2 != 0) {
                        #pragma unroll 1
                        for (int pk = 7; pk < 64; pk++) {
                            int pk_step = k_begin + pk;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(off_m)), "r"((int)(pk_step)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(weight_row)), "r"((int)(pk_step)) : "memory");
                        }
                    }
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < 64; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_step = k_begin + iter_k;
                        tma_3d_gmem2smem(smem_a_addr + load_stage * 32768, (&A), 0, off_m, k_step, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_b_addr + load_stage * 32768, (&B), 0, weight_row, k_step, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 32768);
                        load_stage += 1;
                        if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
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
            unsigned int mma_cluster_id = bid;
            unsigned int mma_num_clusters = num_bids;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int item_1 = mma_cluster_id; item_1 < num_cluster_tiles; item_1 += mma_num_clusters) {
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < 64; iter_k_1++) {
                    mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, ((init_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
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
            unsigned int epi_cluster_id = bid;
            unsigned int epi_num_clusters = num_bids;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int item_2 = epi_cluster_id; item_2 < num_cluster_tiles; item_2 += epi_num_clusters) {
                int group_1 = item_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * 32;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= 32) ? 32 : remaining_1);
                int local_1 = item_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int global_row = bid_m_1 * 128 + local_row;
                int off_n = bid_n_1 * 128;
                int safe_row = ((global_row < M) ? global_row : M - 1);
                int store_row = global_row;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 128;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned long long row_out = (unsigned long long)global_row * 7168 + (unsigned long long)off_n;
                unsigned long long row_in = (unsigned long long)safe_row * 7168 + (unsigned long long)off_n;
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
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        float vals[16];
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            vals[j] = _tmem_load_0[c * 16 + j];
                        }
                        if (store_row < M) {
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(vals[0 + 8], vals[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(vals[0 + 10], vals[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(vals[0 + 12], vals[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(vals[0 + 14], vals[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(C + (row_out + (unsigned long long)(gcol + c * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
