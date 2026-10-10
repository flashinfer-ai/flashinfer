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
#include "cake_dense_projection_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 8
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 28672
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 12288
#define SMEM_SMEM_B_STRIDE 28672
#define SMEM_STT_SMEM_OFF 1024
#define SMEM_STT_SMEM_STAGE_BYTES 6144
#define SMEM_STT_SMEM_STRIDE 6144
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 230528
#define THREADS 320

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_e9b4d2f1e9a14e42293a(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap OUT32, const __grid_constant__ CUtensorMap OUT16, __nv_bfloat16* __restrict__ out, float* __restrict__ out32, float* __restrict__ ws, unsigned int* __restrict__ counters, int M, int N, int m_tiles, int n_tiles, int group_m, int promo_code, int k_iters, int ldo, int out_l, int num_cluster_tiles, int num_l, int num_full, int iters_per_unit, int sk_iters)
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
    #define mma_done_addr (mbar_base + 64)
    #define mainloop_done_addr (mbar_base + 128)
    #define epilogue_done_addr (mbar_base + 144)
    #define work_full_addr (mbar_base + 160)
    #define work_empty_addr (mbar_base + 192)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* stt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_STT_SMEM_OFF);
    const int stt_smem_addr = smem + SMEM_STT_SMEM_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 8 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // mma_done: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 144, 16);
            mbarrier_init(smem + 152, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
            mbarrier_init(smem + 208, 546);
            mbarrier_init(smem + 216, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
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

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
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
                    int us = (int)this_bid / 2 - num_full;
                    int nt_s = sk_iters;
                    int cs = num_cluster_tiles - num_full - nt_s;
                    int cc = us - nt_s;
                    int n_coll = (nt_s - cc + cs - 1) / cs;
                    int n_s = ((us < nt_s) ? 1 : n_coll);
                    int nseg = n_s;
                    #pragma unroll 1
                    for (int seg = 0; seg < nseg; seg++) {
                        int us_0 = (int)this_bid / 2 - num_full;
                        int nt_s_1 = sk_iters;
                        int cs_2 = num_cluster_tiles - num_full - nt_s_1;
                        int cc_3 = us_0 - nt_s_1;
                        int t_s = ((us_0 >= nt_s_1) ? cc_3 + seg * cs_2 : us_0);
                        int tile_bid_s = ((us_0 < 0) ? (int)this_bid : 2 * (num_full + t_s) + cta_rank);
                        int kbeg_s = ((us_0 >= nt_s_1) ? iters_per_unit : 0);
                        int kend_m = ((us_0 < nt_s_1) ? iters_per_unit : k_iters);
                        int kend_s = ((us_0 < 0) ? k_iters : kend_m);
                        int tail_s = ((us_0 < 0) ? -1 : t_s);
                        int tiles_per_l = m_tiles * n_tiles;
                        int tiles_per_group = group_m * n_tiles;
                        int head = tile_bid_s / tiles_per_l;
                        int rem = tile_bid_s - head * tiles_per_l;
                        int group = rem / tiles_per_group;
                        int first_m = group * group_m;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= group_m) ? group_m : remaining);
                        int local = rem % tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * 128;
                        int off_n = bid_n * 192;
                        int b_col = off_n + cta_rank * 96;
                        int klen = kend_s - kbeg_s;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < klen; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k0 = (kbeg_s + iter_k) * 64;
                            #pragma unroll
                            for (int p = 0; p < 2; p++) {
                                #pragma unroll
                                for (int kh = 0; kh < 1; kh++) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_a_addr + load_stage * 28672 + (unsigned int)(p * 8192 + kh * 8192)), "l"((&A)), "r"(off_m + 64 * p), "r"(k0 + 64 * kh), "r"(head),
                                           "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            #pragma unroll
                            for (int p_1 = 0; p_1 < 3; p_1++) {
                                #pragma unroll
                                for (int kh_1 = 0; kh_1 < 1; kh_1++) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(smem_b_addr + load_stage * 28672 + (unsigned int)(p_1 * 4096 + kh_1 * 4096)), "l"((&B)), "r"(b_col + 32 * p_1), "r"(k0 + 64 * kh_1), "r"(head),
                                           "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(28672)) : "memory");
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int this_bid_1 = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    int us_1 = (int)this_bid_1 / 2 - num_full;
                    int nt_s_2 = sk_iters;
                    int cs_1 = num_cluster_tiles - num_full - nt_s_2;
                    int cc_1 = us_1 - nt_s_2;
                    int n_coll_1 = (nt_s_2 - cc_1 + cs_1 - 1) / cs_1;
                    int n_s_1 = ((us_1 < nt_s_2) ? 1 : n_coll_1);
                    int nseg_1 = n_s_1;
                    #pragma unroll 1
                    for (int seg_1 = 0; seg_1 < nseg_1; seg_1++) {
                        int us_0_1 = (int)this_bid_1 / 2 - num_full;
                        int nt_s_1_1 = sk_iters;
                        int cs_2_1 = num_cluster_tiles - num_full - nt_s_1_1;
                        int cc_3_1 = us_0_1 - nt_s_1_1;
                        int t_s_1 = ((us_0_1 >= nt_s_1_1) ? cc_3_1 + seg_1 * cs_2_1 : us_0_1);
                        int tile_bid_s_1 = ((us_0_1 < 0) ? (int)this_bid_1 : 2 * (num_full + t_s_1) + cta_rank);
                        int kbeg_s_1 = ((us_0_1 >= nt_s_1_1) ? iters_per_unit : 0);
                        int kend_m_1 = ((us_0_1 < nt_s_1_1) ? iters_per_unit : k_iters);
                        int kend_s_1 = ((us_0_1 < 0) ? k_iters : kend_m_1);
                        int tail_s_1 = ((us_0_1 < 0) ? -1 : t_s_1);
                        int klen_1 = kend_s_1 - kbeg_s_1;
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < klen_1; iter_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                            int _mma_a_lo_0 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 1792;
                            int _mma_b_lo_0 = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_tma_stage) * 1792;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 271680656;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 8) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    }
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
                    this_bid_1 = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int part = (warp - 2) / 4;
            const int row_half = 0;
            const int col_part = part;
            const int warp_row0 = row_half * 128 + epi_warp * 32;
            const int slice_col0 = col_part * 96;
            const int tmem_col0 = col_part * 96;
            const int local_row = warp_row0 + lane;
            unsigned int this_bid_2 = bid;
            unsigned int valid_st = 1;
            unsigned int next_st = 0;
            unsigned int zero_u32 = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int us_2 = (int)this_bid_2 / 2 - num_full;
                int nt_s_3 = sk_iters;
                int cs_3 = num_cluster_tiles - num_full - nt_s_3;
                int cc_2 = us_2 - nt_s_3;
                int n_coll_2 = (nt_s_3 - cc_2 + cs_3 - 1) / cs_3;
                int n_s_2 = ((us_2 < nt_s_3) ? 1 : n_coll_2);
                int nseg_2 = n_s_2;
                #pragma unroll 1
                for (int seg_2 = 0; seg_2 < nseg_2; seg_2++) {
                    int us_0_2 = (int)this_bid_2 / 2 - num_full;
                    int nt_s_1_2 = sk_iters;
                    int cs_2_2 = num_cluster_tiles - num_full - nt_s_1_2;
                    int cc_3_2 = us_0_2 - nt_s_1_2;
                    int t_s_2 = ((us_0_2 >= nt_s_1_2) ? cc_3_2 + seg_2 * cs_2_2 : us_0_2);
                    int tile_bid_s_2 = ((us_0_2 < 0) ? (int)this_bid_2 : 2 * (num_full + t_s_2) + cta_rank);
                    int kbeg_s_2 = ((us_0_2 >= nt_s_1_2) ? iters_per_unit : 0);
                    int kend_m_2 = ((us_0_2 < nt_s_1_2) ? iters_per_unit : k_iters);
                    int kend_s_2 = ((us_0_2 < 0) ? k_iters : kend_m_2);
                    int tail_s_2 = ((us_0_2 < 0) ? -1 : t_s_2);
                    int tiles_per_l_1 = m_tiles * n_tiles;
                    int tiles_per_group_1 = group_m * n_tiles;
                    int head_1 = tile_bid_s_2 / tiles_per_l_1;
                    int rem_1 = tile_bid_s_2 - head_1 * tiles_per_l_1;
                    int group_1 = rem_1 / tiles_per_group_1;
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = rem_1 % tiles_per_group_1;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * 128;
                    int off_n_1 = bid_n_1 * 192;
                    int bidx = head_1;
                    int part_4 = ((kbeg_s_2 > 0) ? 1 : ((kend_s_2 < k_iters) ? 1 : 0));
                    int slab = tail_s_2;
                    int global_row = off_m_1 + local_row;
                    int row0 = off_m_1 + warp_row0;
                    int col0 = slice_col0;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 256 + (unsigned int)(row_half * 256) + (unsigned int)tmem_col0;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (nseg_2 <= seg_2 + 1) {
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
                        valid_st = _clc_valid_2;
                        next_st = _clc_ctaid_2 + (unsigned int)cta_rank;
                        work_stage_2 += 1;
                        if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                    }
                    float acc[96];
                    if (valid_st == 0) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[15])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[16])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[17])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[18])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[19])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[20])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[21])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[22])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[23])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[24])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[25])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[26])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[27])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[28])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[29])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[30])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 0))[31]))
                            : "r"(lane_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 32))[15]))
                            : "r"(lane_addr + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[15])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[16])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[17])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[18])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[19])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[20])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[21])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[22])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[23])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[24])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[25])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[26])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[27])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[28])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[29])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[30])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 48))[31]))
                            : "r"(lane_addr + 1048576));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((acc + 80))[15]))
                            : "r"(lane_addr + 1048640));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                            : "r"(lane_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[32]), "=f"(acc[33]), "=f"(acc[34]), "=f"(acc[35]), "=f"(acc[36]), "=f"(acc[37]), "=f"(acc[38]), "=f"(acc[39]), "=f"(acc[40]), "=f"(acc[41]), "=f"(acc[42]), "=f"(acc[43]), "=f"(acc[44]), "=f"(acc[45]), "=f"(acc[46]), "=f"(acc[47]), "=f"(acc[48]), "=f"(acc[49]), "=f"(acc[50]), "=f"(acc[51]), "=f"(acc[52]), "=f"(acc[53]), "=f"(acc[54]), "=f"(acc[55]), "=f"(acc[56]), "=f"(acc[57]), "=f"(acc[58]), "=f"(acc[59]), "=f"(acc[60]), "=f"(acc[61]), "=f"(acc[62]), "=f"(acc[63])
                            : "r"(lane_addr + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[64]), "=f"(acc[65]), "=f"(acc[66]), "=f"(acc[67]), "=f"(acc[68]), "=f"(acc[69]), "=f"(acc[70]), "=f"(acc[71]), "=f"(acc[72]), "=f"(acc[73]), "=f"(acc[74]), "=f"(acc[75]), "=f"(acc[76]), "=f"(acc[77]), "=f"(acc[78]), "=f"(acc[79]), "=f"(acc[80]), "=f"(acc[81]), "=f"(acc[82]), "=f"(acc[83]), "=f"(acc[84]), "=f"(acc[85]), "=f"(acc[86]), "=f"(acc[87]), "=f"(acc[88]), "=f"(acc[89]), "=f"(acc[90]), "=f"(acc[91]), "=f"(acc[92]), "=f"(acc[93]), "=f"(acc[94]), "=f"(acc[95])
                            : "r"(lane_addr + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    int cidx_s = tail_s_2 * 16 + cta_rank * 8 + (warp - 2);
                    unsigned int role = 0;
                    if (part_4 != 0) {
                        __syncwarp();
                        unsigned int inc = ((lane == 0) ? 1 : 0);
                        int aidx = cidx_s;
                        aidx = ((lane == 0) ? cidx_s : 4096 + (cta_rank * 8 + (warp - 2)) * 32 + lane);
                        unsigned int _atomic_old_0;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_0) : "l"(&counters[aidx]), "r"(static_cast<uint32_t>(inc)) : "memory");
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, _atomic_old_0, 0);
                        if (_shfl_0 == 0) {
                            role = 1;
                        } else {
                            role = 2;
                            if (_shfl_0 == 1) {
                                #pragma unroll 1
                                for (int _spin = 0; _spin < 1073741824; _spin++) {
                                    unsigned int inc_0 = ((lane == 0) ? 0 : 0);
                                    int aidx_1 = cidx_s;
                                    aidx_1 = ((lane == 0) ? cidx_s : 4096 + (cta_rank * 8 + (warp - 2)) * 32 + lane);
                                    unsigned int _atomic_old_1;
                                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                        : "=r"(_atomic_old_1) : "l"(&counters[aidx_1]), "r"(static_cast<uint32_t>(inc_0)) : "memory");
                                    unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, _atomic_old_1, 0);
                                    if (_shfl_1 >= 3) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (role == 0) {
                        unsigned long long row_base = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row;
                        if (valid_st == 0) {
                            if (off_n_1 + col0 < N) {
                                unsigned int stt_slot = stt_smem_addr + (unsigned int)((warp - 2) * 6144);
                                int stt_v = lane >> 3 ^ lane >> 1 & 3;
                                unsigned int stt_a = stt_slot + (unsigned int)((lane & 7) * 64 + stt_v * 16);
                                __syncwarp();
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                uint32_t acc_bf16[48];
                                #pragma unroll
                                for (int _lp = 0; _lp < 48; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                    acc_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int b = 0; b < 12; b++) {
                                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(stt_a + (unsigned int)(512 * b));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16[2 * b])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16[2 * b + 1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16[24 + 2 * b])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16[24 + 2 * b + 1]))
                                        : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                __syncwarp();
                                if (lane == 0) {
                                    tma_store_3d((&OUT16), row0, off_n_1 + col0, bidx, stt_slot);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        } else if (global_row < M) {
                            #pragma unroll
                            for (int j = 0; j < 96; j++) {
                                int n_idx = off_n_1 + col0 + j;
                                if (n_idx < N) {
                                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(acc[j]);
                                    out[row_base + (unsigned long long)n_idx * (unsigned long long)ldo] = _cvt_bf16_0;
                                }
                            }
                        }
                    }
                    if (role == 1) {
                        if (part_4 == 0) {
                            unsigned long long row_base_1 = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row;
                            if (valid_st == 0) {
                                if (off_n_1 + col0 < N) {
                                    unsigned int stt_slot_1 = stt_smem_addr + (unsigned int)((warp - 2) * 6144);
                                    int stt_v_1 = lane >> 3 ^ lane >> 1 & 3;
                                    unsigned int stt_a_1 = stt_slot_1 + (unsigned int)((lane & 7) * 64 + stt_v_1 * 16);
                                    __syncwarp();
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    uint32_t acc_bf16_1[48];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 48; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                        acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int b_1 = 0; b_1 < 12; b_1++) {
                                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(stt_a_1 + (unsigned int)(512 * b_1));
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_1[2 * b_1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_1[2 * b_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_1[24 + 2 * b_1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_1[24 + 2 * b_1 + 1]))
                                            : "memory");
                                    }
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (lane == 0) {
                                        tma_store_3d((&OUT16), row0, off_n_1 + col0, bidx, stt_slot_1);
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            } else if (global_row < M) {
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 96; j_1++) {
                                    int n_idx_1 = off_n_1 + col0 + j_1;
                                    if (n_idx_1 < N) {
                                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(acc[j_1]);
                                        out[row_base_1 + (unsigned long long)n_idx_1 * (unsigned long long)ldo] = _cvt_bf16_1;
                                    }
                                }
                            }
                        } else {
                            int slice_id = cta_rank * 8 + (warp - 2);
                            unsigned long long off = (unsigned long long)(slab * 16 + slice_id) * 3072 + (unsigned long long)((col0 - slice_col0) * 32 + lane * 4);
                            unsigned long long off_0 = off;
                            if (valid_st == 0) {
                                int stt_lo = ((lane & 3) >> 1) * 128 + (lane >> 2) * 4 + (lane & 1) * 2;
                                unsigned long long stt_off = off_0 - (unsigned long long)(lane * 4) + (unsigned long long)stt_lo;
                                int stt_nv = N - (off_n_1 + col0);
                                if (stt_nv >= 96) {
                                    #pragma unroll
                                    for (int stt_q = 0; stt_q < 12; stt_q++) {
                                        {
                                            float2 _v2 = make_float2(acc[stt_q * 4 + 0], acc[stt_q * 4 + 1]);
                                            *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(stt_q * 256))) + 0) = _v2;
                                        }
                                    }
                                    #pragma unroll
                                    for (int stt_q_1 = 0; stt_q_1 < 12; stt_q_1++) {
                                        {
                                            float2 _v2 = make_float2(acc[stt_q_1 * 4 + 2 + 0], acc[stt_q_1 * 4 + 2 + 1]);
                                            *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(stt_q_1 * 256 + 32))) + 0) = _v2;
                                        }
                                    }
                                    #pragma unroll
                                    for (int stt_q_2 = 0; stt_q_2 < 12; stt_q_2++) {
                                        {
                                            float2 _v2 = make_float2(acc[stt_q_2 * 4 + 48 + 0], acc[stt_q_2 * 4 + 48 + 1]);
                                            *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(stt_q_2 * 256 + 64))) + 0) = _v2;
                                        }
                                    }
                                    #pragma unroll
                                    for (int stt_q_3 = 0; stt_q_3 < 12; stt_q_3++) {
                                        {
                                            float2 _v2 = make_float2(acc[stt_q_3 * 4 + 50 + 0], acc[stt_q_3 * 4 + 50 + 1]);
                                            *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(stt_q_3 * 256 + 96))) + 0) = _v2;
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int g = 0; g < 6; g++) {
                                        if (stt_nv > g * 16) {
                                            {
                                                float2 _v2 = make_float2(acc[2 * g * 4 + 0], acc[2 * g * 4 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(2 * g * 256))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[(2 * g + 1) * 4 + 0], acc[(2 * g + 1) * 4 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)((2 * g + 1) * 256))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[2 * g * 4 + 2 + 0], acc[2 * g * 4 + 2 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(2 * g * 256 + 32))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[(2 * g + 1) * 4 + 2 + 0], acc[(2 * g + 1) * 4 + 2 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)((2 * g + 1) * 256 + 32))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[2 * g * 4 + 48 + 0], acc[2 * g * 4 + 48 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(2 * g * 256 + 64))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[(2 * g + 1) * 4 + 48 + 0], acc[(2 * g + 1) * 4 + 48 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)((2 * g + 1) * 256 + 64))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[2 * g * 4 + 50 + 0], acc[2 * g * 4 + 50 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)(2 * g * 256 + 96))) + 0) = _v2;
                                            }
                                            {
                                                float2 _v2 = make_float2(acc[(2 * g + 1) * 4 + 50 + 0], acc[(2 * g + 1) * 4 + 50 + 1]);
                                                *reinterpret_cast<float2*>((ws + (stt_off + (unsigned long long)((2 * g + 1) * 256 + 96))) + 0) = _v2;
                                            }
                                        }
                                    }
                                }
                            } else {
                                int nvalid = N - (off_n_1 + col0);
                                if (nvalid >= 96) {
                                    #pragma unroll
                                    for (int j_2 = 0; j_2 < 24; j_2++) {
                                        {
                                            float4 _v4 = make_float4(acc[j_2 * 4 + 0], acc[j_2 * 4 + 1], acc[j_2 * 4 + 2], acc[j_2 * 4 + 3]);
                                            *reinterpret_cast<float4*>((ws + (off_0 + (unsigned long long)(j_2 * 128))) + 0) = _v4;
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int g_1 = 0; g_1 < 6; g_1++) {
                                        if (nvalid > g_1 * 16) {
                                            #pragma unroll
                                            for (int q = 0; q < 4; q++) {
                                                {
                                                    float4 _v4 = make_float4(acc[(g_1 * 4 + q) * 4 + 0], acc[(g_1 * 4 + q) * 4 + 1], acc[(g_1 * 4 + q) * 4 + 2], acc[(g_1 * 4 + q) * 4 + 3]);
                                                    *reinterpret_cast<float4*>((ws + (off_0 + (unsigned long long)((g_1 * 4 + q) * 128))) + 0) = _v4;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        {
                            __syncwarp();
                            unsigned int inc_1 = ((lane == 0) ? 1 : 0);
                            int aidx_2 = cidx_s;
                            aidx_2 = ((lane == 0) ? cidx_s : 4096 + (cta_rank * 8 + (warp - 2)) * 32 + lane);
                            unsigned int _atomic_old_2;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_2) : "l"(&counters[aidx_2]), "r"(static_cast<uint32_t>(inc_1)) : "memory");
                            unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, _atomic_old_2, 0);
                        }
                    }
                    if (role == 2) {
                        int slice_id_1 = cta_rank * 8 + (warp - 2);
                        unsigned long long off_1 = (unsigned long long)(slab * 16 + slice_id_1) * 3072 + (unsigned long long)((col0 - slice_col0) * 32 + lane * 4);
                        unsigned long long off_0_1 = off_1;
                        if (valid_st == 0) {
                            int stt_lo_1 = ((lane & 3) >> 1) * 128 + (lane >> 2) * 4 + (lane & 1) * 2;
                            unsigned long long stt_off_1 = off_0_1 - (unsigned long long)(lane * 4) + (unsigned long long)stt_lo_1;
                            int stt_nv_1 = N - (off_n_1 + col0);
                            if (stt_nv_1 >= 96) {
                                #pragma unroll
                                for (int stt_q_4 = 0; stt_q_4 < 12; stt_q_4++) {
                                    float _vec_load_0[2];
                                    {
                                        float2 _v2_2 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(stt_q_4 * 256)) + 0);
                                        _vec_load_0[0] = _v2_2.x;
                                        _vec_load_0[0 + 1] = _v2_2.y;
                                    }
                                    #pragma unroll
                                    for (int i = 0; i < 2; i++) {
                                        acc[stt_q_4 * 4 + i] = acc[stt_q_4 * 4 + i] + _vec_load_0[i];
                                    }
                                }
                                #pragma unroll
                                for (int stt_q_5 = 0; stt_q_5 < 12; stt_q_5++) {
                                    float _vec_load_1[2];
                                    {
                                        float2 _v2_3 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(stt_q_5 * 256 + 32)) + 0);
                                        _vec_load_1[0] = _v2_3.x;
                                        _vec_load_1[0 + 1] = _v2_3.y;
                                    }
                                    #pragma unroll
                                    for (int i_1 = 0; i_1 < 2; i_1++) {
                                        acc[stt_q_5 * 4 + 2 + i_1] = acc[stt_q_5 * 4 + 2 + i_1] + _vec_load_1[i_1];
                                    }
                                }
                                #pragma unroll
                                for (int stt_q_6 = 0; stt_q_6 < 12; stt_q_6++) {
                                    float _vec_load_2[2];
                                    {
                                        float2 _v2_4 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(stt_q_6 * 256 + 64)) + 0);
                                        _vec_load_2[0] = _v2_4.x;
                                        _vec_load_2[0 + 1] = _v2_4.y;
                                    }
                                    #pragma unroll
                                    for (int i_2 = 0; i_2 < 2; i_2++) {
                                        acc[stt_q_6 * 4 + 48 + i_2] = acc[stt_q_6 * 4 + 48 + i_2] + _vec_load_2[i_2];
                                    }
                                }
                                #pragma unroll
                                for (int stt_q_7 = 0; stt_q_7 < 12; stt_q_7++) {
                                    float _vec_load_3[2];
                                    {
                                        float2 _v2_5 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(stt_q_7 * 256 + 96)) + 0);
                                        _vec_load_3[0] = _v2_5.x;
                                        _vec_load_3[0 + 1] = _v2_5.y;
                                    }
                                    #pragma unroll
                                    for (int i_3 = 0; i_3 < 2; i_3++) {
                                        acc[stt_q_7 * 4 + 50 + i_3] = acc[stt_q_7 * 4 + 50 + i_3] + _vec_load_3[i_3];
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (int g_2 = 0; g_2 < 6; g_2++) {
                                    if (stt_nv_1 > g_2 * 16) {
                                        float _vec_load_4[2];
                                        {
                                            float2 _v2_6 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(2 * g_2 * 256)) + 0);
                                            _vec_load_4[0] = _v2_6.x;
                                            _vec_load_4[0 + 1] = _v2_6.y;
                                        }
                                        #pragma unroll
                                        for (int i_4 = 0; i_4 < 2; i_4++) {
                                            acc[2 * g_2 * 4 + i_4] = acc[2 * g_2 * 4 + i_4] + _vec_load_4[i_4];
                                        }
                                        float _vec_load_5[2];
                                        {
                                            float2 _v2_7 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)((2 * g_2 + 1) * 256)) + 0);
                                            _vec_load_5[0] = _v2_7.x;
                                            _vec_load_5[0 + 1] = _v2_7.y;
                                        }
                                        #pragma unroll
                                        for (int i_5 = 0; i_5 < 2; i_5++) {
                                            acc[(2 * g_2 + 1) * 4 + i_5] = acc[(2 * g_2 + 1) * 4 + i_5] + _vec_load_5[i_5];
                                        }
                                        float _vec_load_6[2];
                                        {
                                            float2 _v2_8 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(2 * g_2 * 256 + 32)) + 0);
                                            _vec_load_6[0] = _v2_8.x;
                                            _vec_load_6[0 + 1] = _v2_8.y;
                                        }
                                        #pragma unroll
                                        for (int i_6 = 0; i_6 < 2; i_6++) {
                                            acc[2 * g_2 * 4 + 2 + i_6] = acc[2 * g_2 * 4 + 2 + i_6] + _vec_load_6[i_6];
                                        }
                                        float _vec_load_7[2];
                                        {
                                            float2 _v2_9 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)((2 * g_2 + 1) * 256 + 32)) + 0);
                                            _vec_load_7[0] = _v2_9.x;
                                            _vec_load_7[0 + 1] = _v2_9.y;
                                        }
                                        #pragma unroll
                                        for (int i_7 = 0; i_7 < 2; i_7++) {
                                            acc[(2 * g_2 + 1) * 4 + 2 + i_7] = acc[(2 * g_2 + 1) * 4 + 2 + i_7] + _vec_load_7[i_7];
                                        }
                                        float _vec_load_8[2];
                                        {
                                            float2 _v2_10 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(2 * g_2 * 256 + 64)) + 0);
                                            _vec_load_8[0] = _v2_10.x;
                                            _vec_load_8[0 + 1] = _v2_10.y;
                                        }
                                        #pragma unroll
                                        for (int i_8 = 0; i_8 < 2; i_8++) {
                                            acc[2 * g_2 * 4 + 48 + i_8] = acc[2 * g_2 * 4 + 48 + i_8] + _vec_load_8[i_8];
                                        }
                                        float _vec_load_9[2];
                                        {
                                            float2 _v2_11 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)((2 * g_2 + 1) * 256 + 64)) + 0);
                                            _vec_load_9[0] = _v2_11.x;
                                            _vec_load_9[0 + 1] = _v2_11.y;
                                        }
                                        #pragma unroll
                                        for (int i_9 = 0; i_9 < 2; i_9++) {
                                            acc[(2 * g_2 + 1) * 4 + 48 + i_9] = acc[(2 * g_2 + 1) * 4 + 48 + i_9] + _vec_load_9[i_9];
                                        }
                                        float _vec_load_10[2];
                                        {
                                            float2 _v2_12 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)(2 * g_2 * 256 + 96)) + 0);
                                            _vec_load_10[0] = _v2_12.x;
                                            _vec_load_10[0 + 1] = _v2_12.y;
                                        }
                                        #pragma unroll
                                        for (int i_10 = 0; i_10 < 2; i_10++) {
                                            acc[2 * g_2 * 4 + 50 + i_10] = acc[2 * g_2 * 4 + 50 + i_10] + _vec_load_10[i_10];
                                        }
                                        float _vec_load_11[2];
                                        {
                                            float2 _v2_13 = *reinterpret_cast<const float2*>(ws + (stt_off_1 + (unsigned long long)((2 * g_2 + 1) * 256 + 96)) + 0);
                                            _vec_load_11[0] = _v2_13.x;
                                            _vec_load_11[0 + 1] = _v2_13.y;
                                        }
                                        #pragma unroll
                                        for (int i_11 = 0; i_11 < 2; i_11++) {
                                            acc[(2 * g_2 + 1) * 4 + 50 + i_11] = acc[(2 * g_2 + 1) * 4 + 50 + i_11] + _vec_load_11[i_11];
                                        }
                                    }
                                }
                            }
                        } else {
                            int nvalid_1 = N - (off_n_1 + col0);
                            if (nvalid_1 >= 96) {
                                #pragma unroll
                                for (int j_3 = 0; j_3 < 24; j_3++) {
                                    float _vec_load_12[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_1 + (unsigned long long)(j_3 * 128)) + 0);
                                        _vec_load_12[0 + 0] = _v4.x;
                                        _vec_load_12[0 + 1] = _v4.y;
                                        _vec_load_12[0 + 2] = _v4.z;
                                        _vec_load_12[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int i_12 = 0; i_12 < 4; i_12++) {
                                        acc[j_3 * 4 + i_12] = acc[j_3 * 4 + i_12] + _vec_load_12[i_12];
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (int g_3 = 0; g_3 < 6; g_3++) {
                                    if (nvalid_1 > g_3 * 16) {
                                        #pragma unroll
                                        for (int q_1 = 0; q_1 < 4; q_1++) {
                                            float _vec_load_13[4];
                                            {
                                                float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_1 + (unsigned long long)((g_3 * 4 + q_1) * 128)) + 0);
                                                _vec_load_13[0 + 0] = _v4.x;
                                                _vec_load_13[0 + 1] = _v4.y;
                                                _vec_load_13[0 + 2] = _v4.z;
                                                _vec_load_13[0 + 3] = _v4.w;
                                            }
                                            #pragma unroll
                                            for (int i_13 = 0; i_13 < 4; i_13++) {
                                                acc[(g_3 * 4 + q_1) * 4 + i_13] = acc[(g_3 * 4 + q_1) * 4 + i_13] + _vec_load_13[i_13];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        unsigned long long row_base_2 = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row;
                        if (valid_st == 0) {
                            if (off_n_1 + col0 < N) {
                                unsigned int stt_slot_2 = stt_smem_addr + (unsigned int)((warp - 2) * 6144);
                                int stt_v_2 = lane >> 3 ^ lane >> 1 & 3;
                                unsigned int stt_a_2 = stt_slot_2 + (unsigned int)((lane & 7) * 64 + stt_v_2 * 16);
                                __syncwarp();
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                uint32_t acc_bf16_2[48];
                                #pragma unroll
                                for (int _lp = 0; _lp < 48; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                    acc_bf16_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int b_2 = 0; b_2 < 12; b_2++) {
                                    uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(stt_a_2 + (unsigned int)(512 * b_2));
                                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                        :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_2[2 * b_2])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_2[2 * b_2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_2[24 + 2 * b_2])), "r"(*reinterpret_cast<const uint32_t*>(&acc_bf16_2[24 + 2 * b_2 + 1]))
                                        : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                __syncwarp();
                                if (lane == 0) {
                                    tma_store_3d((&OUT16), row0, off_n_1 + col0, bidx, stt_slot_2);
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        } else if (global_row < M) {
                            #pragma unroll
                            for (int j_4 = 0; j_4 < 96; j_4++) {
                                int n_idx_2 = off_n_1 + col0 + j_4;
                                if (n_idx_2 < N) {
                                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(acc[j_4]);
                                    out[row_base_2 + (unsigned long long)n_idx_2 * (unsigned long long)ldo] = _cvt_bf16_2;
                                }
                            }
                        }
                        {
                            if (lane == 0) {
                                counters[(unsigned long long)cidx_s] = zero_u32;
                            }
                        }
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                }
                if (valid_st == 0) {
                    break;
                }
                this_bid_2 = next_st;
            }
            if (lane == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
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
