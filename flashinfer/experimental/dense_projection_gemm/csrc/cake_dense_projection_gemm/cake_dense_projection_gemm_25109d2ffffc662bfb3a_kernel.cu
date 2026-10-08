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
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_A1_OFF 17408
#define SMEM_SMEM_A1_STAGE_BYTES 16384
#define SMEM_SMEM_A1_STRIDE 49152
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 49152
#define SMEM_WORK_RESPONSE_OFF 197632
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 197760
#define THREADS 320

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_25109d2ffffc662bfb3a(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap OUT32, const __grid_constant__ CUtensorMap OUT16, __nv_bfloat16* __restrict__ out, float* __restrict__ out32, float* __restrict__ ws, unsigned int* __restrict__ counters, int M, int N, int m_tiles, int n_tiles, int group_m, int promo_code, int k_iters, int ldo, int out_l, int num_cluster_tiles, int num_l, int num_full, int iters_per_unit, int sk_iters)
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
    #define mainloop_done_addr (mbar_base + 64)
    #define epilogue_done_addr (mbar_base + 72)
    #define work_full_addr (mbar_base + 80)
    #define work_empty_addr (mbar_base + 112)

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
    __nv_bfloat16* smem_a1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A1_OFF);
    const int smem_a1_addr = smem + SMEM_SMEM_A1_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 72, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 112, 546);
            mbarrier_init(smem + 120, 546);
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
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
                    int u = (int)this_bid / 2 - num_full;
                    int lin0 = u * iters_per_unit;
                    int lin1_raw = lin0 + iters_per_unit;
                    int lin1 = ((lin1_raw > sk_iters) ? sk_iters : lin1_raw);
                    int n_tail = (lin1 - 1) / k_iters - lin0 / k_iters + 1;
                    int n = ((u < 0) ? 1 : n_tail);
                    int nseg = n;
                    #pragma unroll 1
                    for (int seg = 0; seg < nseg; seg++) {
                        int u_0 = (int)this_bid / 2 - num_full;
                        int lin0_1 = u_0 * iters_per_unit;
                        int lin1_raw_2 = lin0_1 + iters_per_unit;
                        int lin1_3 = ((lin1_raw_2 > sk_iters) ? sk_iters : lin1_raw_2);
                        int t = lin0_1 / k_iters + seg;
                        int tb = t * k_iters;
                        int kb0 = lin0_1 - tb;
                        int kbeg_t = ((kb0 > 0) ? kb0 : 0);
                        int ke0 = lin1_3 - tb;
                        int kend_t = ((ke0 > k_iters) ? k_iters : ke0);
                        int tile_bid = ((u_0 < 0) ? (int)this_bid : 2 * (num_full + t / 2) + cta_rank);
                        int kbeg = ((u_0 < 0) ? 0 : kbeg_t);
                        int kend = ((u_0 < 0) ? k_iters : kend_t);
                        int tail_t = ((u_0 < 0) ? -1 : t);
                        int tiles_per_l = m_tiles * n_tiles;
                        int tiles_per_group = group_m * n_tiles;
                        int head = tile_bid / tiles_per_l;
                        int rem = tile_bid - head * tiles_per_l;
                        int group = rem / tiles_per_group;
                        int first_m = group * group_m;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= group_m) ? group_m : remaining);
                        int local = rem % tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * 256;
                        int off_n = bid_n * 256;
                        int u_4 = (int)this_bid / 2 - num_full;
                        int is_half = ((u_4 >= 0) ? 1 : 0);
                        int hh = ((u_4 >= 0) ? u_4 % 2 : 0);
                        int off_m_h = off_m + hh * 256 - is_half * cta_rank * 128;
                        int tx_h = 49152 - is_half * 16384;
                        int b_col = off_n + cta_rank * 128;
                        int klen = kend - kbeg;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < klen; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k0 = (kbeg + iter_k) * 64;
                            #pragma unroll
                            for (int p = 0; p < 4; p++) {
                                #pragma unroll
                                for (int kh = 0; kh < 1; kh++) {
                                    if (p < 2) {
                                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + (unsigned int)(p * 8192 + kh * 8192), (&A), off_m_h + 64 * p, k0 + 64 * kh, head, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                    } else if (is_half == 0) {
                                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + (unsigned int)(p * 8192 + kh * 8192), (&A), off_m_h + 64 * p, k0 + 64 * kh, head, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                    }
                                }
                            }
                            #pragma unroll
                            for (int p_1 = 0; p_1 < 2; p_1++) {
                                #pragma unroll
                                for (int kh_1 = 0; kh_1 < 1; kh_1++) {
                                    tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 49152 + (unsigned int)(p_1 * 8192 + kh_1 * 8192), (&B), b_col + 64 * p_1, k0 + 64 * kh_1, head, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                }
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(tx_h)) : "memory");
                            load_stage += 1;
                            if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
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
                    int u_1 = (int)this_bid_1 / 2 - num_full;
                    int lin0_2 = u_1 * iters_per_unit;
                    int lin1_raw_1 = lin0_2 + iters_per_unit;
                    int lin1_1 = ((lin1_raw_1 > sk_iters) ? sk_iters : lin1_raw_1);
                    int n_tail_1 = (lin1_1 - 1) / k_iters - lin0_2 / k_iters + 1;
                    int n_1 = ((u_1 < 0) ? 1 : n_tail_1);
                    int nseg_1 = n_1;
                    #pragma unroll 1
                    for (int seg_1 = 0; seg_1 < nseg_1; seg_1++) {
                        int u_0_1 = (int)this_bid_1 / 2 - num_full;
                        int lin0_1_1 = u_0_1 * iters_per_unit;
                        int lin1_raw_2_1 = lin0_1_1 + iters_per_unit;
                        int lin1_3_1 = ((lin1_raw_2_1 > sk_iters) ? sk_iters : lin1_raw_2_1);
                        int t_1 = lin0_1_1 / k_iters + seg_1;
                        int tb_1 = t_1 * k_iters;
                        int kb0_1 = lin0_1_1 - tb_1;
                        int kbeg_t_1 = ((kb0_1 > 0) ? kb0_1 : 0);
                        int ke0_1 = lin1_3_1 - tb_1;
                        int kend_t_1 = ((ke0_1 > k_iters) ? k_iters : ke0_1);
                        int tile_bid_1 = ((u_0_1 < 0) ? (int)this_bid_1 : 2 * (num_full + t_1 / 2) + cta_rank);
                        int kbeg_1 = ((u_0_1 < 0) ? 0 : kbeg_t_1);
                        int kend_1 = ((u_0_1 < 0) ? k_iters : kend_t_1);
                        int tail_t_1 = ((u_0_1 < 0) ? -1 : t_1);
                        int klen_1 = kend_1 - kbeg_1;
                        int u_4_1 = (int)this_bid_1 / 2 - num_full;
                        int is_half_1 = ((u_4_1 >= 0) ? 1 : 0);
                        int hh_1 = ((u_4_1 >= 0) ? u_4_1 % 2 : 0);
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < klen_1; iter_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                            int _mma_a_lo_0 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                            int _mma_b_lo_0 = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                            if (is_half_1 == 0) {
                                int _mma_a_lo_1 = ((((smem_a1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                                int _mma_b_lo_1 = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(((init_flag) ? 0 : 1)));
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        _phase_epilogue_done ^= 1;
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
            const int row_half = part;
            const int col_part = 0;
            const int warp_row0 = row_half * 128 + epi_warp * 32;
            const int slice_col0 = col_part * 128;
            const int tmem_col0 = col_part * 128;
            const int local_row = warp_row0 + lane;
            unsigned int this_bid_2 = bid;
            unsigned int zero_u32 = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int u_2 = (int)this_bid_2 / 2 - num_full;
                int lin0_3 = u_2 * iters_per_unit;
                int lin1_raw_3 = lin0_3 + iters_per_unit;
                int lin1_2 = ((lin1_raw_3 > sk_iters) ? sk_iters : lin1_raw_3);
                int n_tail_2 = (lin1_2 - 1) / k_iters - lin0_3 / k_iters + 1;
                int n_2 = ((u_2 < 0) ? 1 : n_tail_2);
                int nseg_2 = n_2;
                #pragma unroll 1
                for (int seg_2 = 0; seg_2 < nseg_2; seg_2++) {
                    int u_0_2 = (int)this_bid_2 / 2 - num_full;
                    int lin0_1_2 = u_0_2 * iters_per_unit;
                    int lin1_raw_2_2 = lin0_1_2 + iters_per_unit;
                    int lin1_3_2 = ((lin1_raw_2_2 > sk_iters) ? sk_iters : lin1_raw_2_2);
                    int t_2 = lin0_1_2 / k_iters + seg_2;
                    int tb_2 = t_2 * k_iters;
                    int kb0_2 = lin0_1_2 - tb_2;
                    int kbeg_t_2 = ((kb0_2 > 0) ? kb0_2 : 0);
                    int ke0_2 = lin1_3_2 - tb_2;
                    int kend_t_2 = ((ke0_2 > k_iters) ? k_iters : ke0_2);
                    int tile_bid_2 = ((u_0_2 < 0) ? (int)this_bid_2 : 2 * (num_full + t_2 / 2) + cta_rank);
                    int kbeg_2 = ((u_0_2 < 0) ? 0 : kbeg_t_2);
                    int kend_2 = ((u_0_2 < 0) ? k_iters : kend_t_2);
                    int tail_t_2 = ((u_0_2 < 0) ? -1 : t_2);
                    int tiles_per_l_1 = m_tiles * n_tiles;
                    int tiles_per_group_1 = group_m * n_tiles;
                    int head_1 = tile_bid_2 / tiles_per_l_1;
                    int rem_1 = tile_bid_2 - head_1 * tiles_per_l_1;
                    int group_1 = rem_1 / tiles_per_group_1;
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = rem_1 % tiles_per_group_1;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * 256;
                    int off_n_1 = bid_n_1 * 256;
                    int bidx = head_1;
                    int part_4 = ((kbeg_2 > 0) ? 1 : ((kend_2 < k_iters) ? 1 : 0));
                    int slab = u_0_2 + tail_t_2;
                    int u_5 = (int)this_bid_2 / 2 - num_full;
                    int is_half_2 = ((u_5 >= 0) ? 1 : 0);
                    int hh_2 = ((u_5 >= 0) ? u_5 % 2 : 0);
                    int off_m_h_1 = off_m_1 + hh_2 * 256 - is_half_2 * cta_rank * 128;
                    int idle = is_half_2 * row_half;
                    int global_row = ((idle != 0) ? M : off_m_h_1 + local_row);
                    int row0 = ((idle != 0) ? M : off_m_h_1 + warp_row0);
                    int col0 = slice_col0;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 256 + (unsigned int)(row_half * 256) + (unsigned int)tmem_col0;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(lane_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(lane_addr + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (part_4 == 0) {
                        unsigned long long row_base = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)(off_n_1 + col0);
                        if (global_row < M) {
                            int mis32 = (int)((unsigned long long)(out + row_base) & 31);
                            #pragma unroll
                            for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                                int col = n_chunk * 16;
                                float out_vals[16];
                                #pragma unroll
                                for (int j = 0; j < 16; j++) {
                                    out_vals[j] = _tmem_load_0[n_chunk * 16 + j];
                                }
                                if (off_n_1 + col0 + col + 16 <= N) {
                                    if (mis32 == 0) {
                                        {
                                            {
                                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals[0 + 8], out_vals[0 + 9]);
                                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals[0 + 10], out_vals[0 + 11]);
                                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals[0 + 12], out_vals[0 + 13]);
                                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals[0 + 14], out_vals[0 + 15]);
                                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                            }
                                        }
                                    } else {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals[8 + 0], out_vals[8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals[8 + 2], out_vals[8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals[8 + 4], out_vals[8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals[8 + 6], out_vals[8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)(col + 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                } else if (off_n_1 + col0 + col + 8 <= N) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    } else {
                        int slice_id = cta_rank * 8 + (warp - 2);
                        unsigned long long off = (unsigned long long)(slab * 16 + slice_id) * 8192 + (unsigned long long)((col0 - slice_col0) * 32 + lane * 4);
                        unsigned long long off_0 = off;
                        int nvalid = N - (off_n_1 + col0);
                        if (nvalid >= 128) {
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 32; j_1++) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_0[j_1 * 4 + 0], _tmem_load_0[j_1 * 4 + 1], _tmem_load_0[j_1 * 4 + 2], _tmem_load_0[j_1 * 4 + 3]);
                                    *reinterpret_cast<float4*>((ws + (off_0 + (unsigned long long)(j_1 * 128))) + 0) = _v4;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int g = 0; g < 8; g++) {
                                if (nvalid > g * 16) {
                                    #pragma unroll
                                    for (int q = 0; q < 4; q++) {
                                        {
                                            float4 _v4 = make_float4(_tmem_load_0[(g * 4 + q) * 4 + 0], _tmem_load_0[(g * 4 + q) * 4 + 1], _tmem_load_0[(g * 4 + q) * 4 + 2], _tmem_load_0[(g * 4 + q) * 4 + 3]);
                                            *reinterpret_cast<float4*>((ws + (off_0 + (unsigned long long)((g * 4 + q) * 128))) + 0) = _v4;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    float _tmem_load_1[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(lane_addr + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95]), "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                        : "r"(lane_addr + 128 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    if (part_4 == 0) {
                        unsigned long long row_base_1 = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)(off_n_1 + col0 + 128);
                        if (global_row < M) {
                            int mis32_1 = (int)((unsigned long long)(out + row_base_1) & 31);
                            #pragma unroll
                            for (int n_chunk_1 = 0; n_chunk_1 < 8; n_chunk_1++) {
                                int col_1 = n_chunk_1 * 16;
                                float out_vals_1[16];
                                #pragma unroll
                                for (int j_2 = 0; j_2 < 16; j_2++) {
                                    out_vals_1[j_2] = _tmem_load_1[n_chunk_1 * 16 + j_2];
                                }
                                if (off_n_1 + col0 + 128 + col_1 + 16 <= N) {
                                    if (mis32_1 == 0) {
                                        {
                                            {
                                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_1[0 + 0], out_vals_1[0 + 1]);
                                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_1[0 + 2], out_vals_1[0 + 3]);
                                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_1[0 + 4], out_vals_1[0 + 5]);
                                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_1[0 + 6], out_vals_1[0 + 7]);
                                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_1[0 + 8], out_vals_1[0 + 9]);
                                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_1[0 + 10], out_vals_1[0 + 11]);
                                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_1[0 + 12], out_vals_1[0 + 13]);
                                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_1[0 + 14], out_vals_1[0 + 15]);
                                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base_1 + (unsigned long long)col_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                            }
                                        }
                                    } else {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals_1[0 + 0], out_vals_1[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals_1[0 + 2], out_vals_1[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals_1[0 + 4], out_vals_1[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals_1[0 + 6], out_vals_1[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1 + (unsigned long long)col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals_1[8 + 0], out_vals_1[8 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals_1[8 + 2], out_vals_1[8 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals_1[8 + 4], out_vals_1[8 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals_1[8 + 6], out_vals_1[8 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1 + (unsigned long long)(col_1 + 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                } else if (off_n_1 + col0 + 128 + col_1 + 8 <= N) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(out_vals_1[0 + 0], out_vals_1[0 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(out_vals_1[0 + 2], out_vals_1[0 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(out_vals_1[0 + 4], out_vals_1[0 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(out_vals_1[0 + 6], out_vals_1[0 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1 + (unsigned long long)col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    } else {
                        int slice_id_1 = cta_rank * 8 + (warp - 2);
                        unsigned long long off_1 = (unsigned long long)(slab * 16 + slice_id_1) * 8192 + (unsigned long long)((col0 + 128 - slice_col0) * 32 + lane * 4);
                        unsigned long long off_0_1 = off_1;
                        int nvalid_1 = N - (off_n_1 + col0 + 128);
                        if (nvalid_1 >= 128) {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 32; j_3++) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[j_3 * 4 + 0], _tmem_load_1[j_3 * 4 + 1], _tmem_load_1[j_3 * 4 + 2], _tmem_load_1[j_3 * 4 + 3]);
                                    *reinterpret_cast<float4*>((ws + (off_0_1 + (unsigned long long)(j_3 * 128))) + 0) = _v4;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int g_1 = 0; g_1 < 8; g_1++) {
                                if (nvalid_1 > g_1 * 16) {
                                    #pragma unroll
                                    for (int q_1 = 0; q_1 < 4; q_1++) {
                                        {
                                            float4 _v4 = make_float4(_tmem_load_1[(g_1 * 4 + q_1) * 4 + 0], _tmem_load_1[(g_1 * 4 + q_1) * 4 + 1], _tmem_load_1[(g_1 * 4 + q_1) * 4 + 2], _tmem_load_1[(g_1 * 4 + q_1) * 4 + 3]);
                                            *reinterpret_cast<float4*>((ws + (off_0_1 + (unsigned long long)((g_1 * 4 + q_1) * 128))) + 0) = _v4;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (part_4 != 0) {
                        __syncwarp();
                        int cidx = tail_t_2 * 16 + cta_rank * 8 + (warp - 2);
                        int u_first = tail_t_2 * k_iters / iters_per_unit;
                        int u_last = ((tail_t_2 + 1) * k_iters - 1) / iters_per_unit;
                        int nparts = u_last - u_first + 1;
                        unsigned int inc = ((lane == 0) ? 1 : 0);
                        int aidx = cidx;
                        aidx = ((lane == 0) ? cidx : 4096 + (cta_rank * 8 + (warp - 2)) * 32 + lane);
                        unsigned int _atomic_old_0;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_0) : "l"(&counters[aidx]), "r"(static_cast<uint32_t>(inc)) : "memory");
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, _atomic_old_0, 0);
                        if (_shfl_0 == (unsigned int)(nparts - 1)) {
                            #pragma unroll
                            for (int j_4 = 0; j_4 < 128; j_4++) {
                                _tmem_load_0[j_4] = 0.0f;
                            }
                            int nvalid_2 = N - (off_n_1 + col0);
                            #pragma unroll 1
                            for (int sidx = 0; sidx < nparts; sidx++) {
                                int slice_id_2 = cta_rank * 8 + (warp - 2);
                                unsigned long long off_2 = (unsigned long long)((u_first + sidx + tail_t_2) * 16 + slice_id_2) * 8192 + (unsigned long long)((col0 - slice_col0) * 32 + lane * 4);
                                unsigned long long off_0_2 = off_2;
                                if (nvalid_2 >= 128) {
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 32; j_5++) {
                                        float _vec_load_0[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_2 + (unsigned long long)(j_5 * 128)) + 0);
                                            _vec_load_0[0 + 0] = _v4.x;
                                            _vec_load_0[0 + 1] = _v4.y;
                                            _vec_load_0[0 + 2] = _v4.z;
                                            _vec_load_0[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int i = 0; i < 4; i++) {
                                            _tmem_load_0[j_5 * 4 + i] = _tmem_load_0[j_5 * 4 + i] + _vec_load_0[i];
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int g_2 = 0; g_2 < 8; g_2++) {
                                        if (nvalid_2 > g_2 * 16) {
                                            #pragma unroll
                                            for (int q_2 = 0; q_2 < 4; q_2++) {
                                                float _vec_load_1[4];
                                                {
                                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_2 + (unsigned long long)((g_2 * 4 + q_2) * 128)) + 0);
                                                    _vec_load_1[0 + 0] = _v4.x;
                                                    _vec_load_1[0 + 1] = _v4.y;
                                                    _vec_load_1[0 + 2] = _v4.z;
                                                    _vec_load_1[0 + 3] = _v4.w;
                                                }
                                                #pragma unroll
                                                for (int i_1 = 0; i_1 < 4; i_1++) {
                                                    _tmem_load_0[(g_2 * 4 + q_2) * 4 + i_1] = _tmem_load_0[(g_2 * 4 + q_2) * 4 + i_1] + _vec_load_1[i_1];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            unsigned long long row_base_2 = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)(off_n_1 + col0);
                            if (global_row < M) {
                                int mis32_2 = (int)((unsigned long long)(out + row_base_2) & 31);
                                #pragma unroll
                                for (int n_chunk_2 = 0; n_chunk_2 < 8; n_chunk_2++) {
                                    int col_2 = n_chunk_2 * 16;
                                    float out_vals_2[16];
                                    #pragma unroll
                                    for (int j_6 = 0; j_6 < 16; j_6++) {
                                        out_vals_2[j_6] = _tmem_load_0[n_chunk_2 * 16 + j_6];
                                    }
                                    if (off_n_1 + col0 + col_2 + 16 <= N) {
                                        if (mis32_2 == 0) {
                                            {
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_2[0 + 0], out_vals_2[0 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_2[0 + 2], out_vals_2[0 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_2[0 + 4], out_vals_2[0 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_2[0 + 6], out_vals_2[0 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_2[0 + 8], out_vals_2[0 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_2[0 + 10], out_vals_2[0 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_2[0 + 12], out_vals_2[0 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_2[0 + 14], out_vals_2[0 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base_2 + (unsigned long long)col_2)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        } else {
                                            {
                                                __nv_bfloat162 _pk[4];
                                                _pk[0] = __floats2bfloat162_rn(out_vals_2[0 + 0], out_vals_2[0 + 1]);
                                                _pk[1] = __floats2bfloat162_rn(out_vals_2[0 + 2], out_vals_2[0 + 3]);
                                                _pk[2] = __floats2bfloat162_rn(out_vals_2[0 + 4], out_vals_2[0 + 5]);
                                                _pk[3] = __floats2bfloat162_rn(out_vals_2[0 + 6], out_vals_2[0 + 7]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_2 + (unsigned long long)col_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                            }
                                            {
                                                __nv_bfloat162 _pk[4];
                                                _pk[0] = __floats2bfloat162_rn(out_vals_2[8 + 0], out_vals_2[8 + 1]);
                                                _pk[1] = __floats2bfloat162_rn(out_vals_2[8 + 2], out_vals_2[8 + 3]);
                                                _pk[2] = __floats2bfloat162_rn(out_vals_2[8 + 4], out_vals_2[8 + 5]);
                                                _pk[3] = __floats2bfloat162_rn(out_vals_2[8 + 6], out_vals_2[8 + 7]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_2 + (unsigned long long)(col_2 + 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                            }
                                        }
                                    } else if (off_n_1 + col0 + col_2 + 8 <= N) {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals_2[0 + 0], out_vals_2[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals_2[0 + 2], out_vals_2[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals_2[0 + 4], out_vals_2[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals_2[0 + 6], out_vals_2[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_2 + (unsigned long long)col_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                            }
                            #pragma unroll
                            for (int j_7 = 0; j_7 < 128; j_7++) {
                                _tmem_load_0[j_7] = 0.0f;
                            }
                            int nvalid_0 = N - (off_n_1 + col0 + 128);
                            #pragma unroll 1
                            for (int sidx_1 = 0; sidx_1 < nparts; sidx_1++) {
                                int slice_id_3 = cta_rank * 8 + (warp - 2);
                                unsigned long long off_3 = (unsigned long long)((u_first + sidx_1 + tail_t_2) * 16 + slice_id_3) * 8192 + (unsigned long long)((col0 + 128 - slice_col0) * 32 + lane * 4);
                                unsigned long long off_0_3 = off_3;
                                if (nvalid_0 >= 128) {
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 32; j_8++) {
                                        float _vec_load_2[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_3 + (unsigned long long)(j_8 * 128)) + 0);
                                            _vec_load_2[0 + 0] = _v4.x;
                                            _vec_load_2[0 + 1] = _v4.y;
                                            _vec_load_2[0 + 2] = _v4.z;
                                            _vec_load_2[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int i_2 = 0; i_2 < 4; i_2++) {
                                            _tmem_load_0[j_8 * 4 + i_2] = _tmem_load_0[j_8 * 4 + i_2] + _vec_load_2[i_2];
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int g_3 = 0; g_3 < 8; g_3++) {
                                        if (nvalid_0 > g_3 * 16) {
                                            #pragma unroll
                                            for (int q_3 = 0; q_3 < 4; q_3++) {
                                                float _vec_load_3[4];
                                                {
                                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (off_0_3 + (unsigned long long)((g_3 * 4 + q_3) * 128)) + 0);
                                                    _vec_load_3[0 + 0] = _v4.x;
                                                    _vec_load_3[0 + 1] = _v4.y;
                                                    _vec_load_3[0 + 2] = _v4.z;
                                                    _vec_load_3[0 + 3] = _v4.w;
                                                }
                                                #pragma unroll
                                                for (int i_3 = 0; i_3 < 4; i_3++) {
                                                    _tmem_load_0[(g_3 * 4 + q_3) * 4 + i_3] = _tmem_load_0[(g_3 * 4 + q_3) * 4 + i_3] + _vec_load_3[i_3];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            unsigned long long row_base_1_1 = (unsigned long long)bidx * (unsigned long long)out_l + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)(off_n_1 + col0 + 128);
                            if (global_row < M) {
                                int mis32_3 = (int)((unsigned long long)(out + row_base_1_1) & 31);
                                #pragma unroll
                                for (int n_chunk_3 = 0; n_chunk_3 < 8; n_chunk_3++) {
                                    int col_3 = n_chunk_3 * 16;
                                    float out_vals_3[16];
                                    #pragma unroll
                                    for (int j_9 = 0; j_9 < 16; j_9++) {
                                        out_vals_3[j_9] = _tmem_load_0[n_chunk_3 * 16 + j_9];
                                    }
                                    if (off_n_1 + col0 + 128 + col_3 + 16 <= N) {
                                        if (mis32_3 == 0) {
                                            {
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_3[0 + 0], out_vals_3[0 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_3[0 + 2], out_vals_3[0 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_3[0 + 4], out_vals_3[0 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_3[0 + 6], out_vals_3[0 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_3[0 + 8], out_vals_3[0 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_3[0 + 10], out_vals_3[0 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_3[0 + 12], out_vals_3[0 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_3[0 + 14], out_vals_3[0 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base_1_1 + (unsigned long long)col_3)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        } else {
                                            {
                                                __nv_bfloat162 _pk[4];
                                                _pk[0] = __floats2bfloat162_rn(out_vals_3[0 + 0], out_vals_3[0 + 1]);
                                                _pk[1] = __floats2bfloat162_rn(out_vals_3[0 + 2], out_vals_3[0 + 3]);
                                                _pk[2] = __floats2bfloat162_rn(out_vals_3[0 + 4], out_vals_3[0 + 5]);
                                                _pk[3] = __floats2bfloat162_rn(out_vals_3[0 + 6], out_vals_3[0 + 7]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1_1 + (unsigned long long)col_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                            }
                                            {
                                                __nv_bfloat162 _pk[4];
                                                _pk[0] = __floats2bfloat162_rn(out_vals_3[8 + 0], out_vals_3[8 + 1]);
                                                _pk[1] = __floats2bfloat162_rn(out_vals_3[8 + 2], out_vals_3[8 + 3]);
                                                _pk[2] = __floats2bfloat162_rn(out_vals_3[8 + 4], out_vals_3[8 + 5]);
                                                _pk[3] = __floats2bfloat162_rn(out_vals_3[8 + 6], out_vals_3[8 + 7]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1_1 + (unsigned long long)(col_3 + 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                            }
                                        }
                                    } else if (off_n_1 + col0 + 128 + col_3 + 8 <= N) {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(out_vals_3[0 + 0], out_vals_3[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_vals_3[0 + 2], out_vals_3[0 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(out_vals_3[0 + 4], out_vals_3[0 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(out_vals_3[0 + 6], out_vals_3[0 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (row_base_1_1 + (unsigned long long)col_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                }
                            }
                            if (lane == 0) {
                                counters[(unsigned long long)cidx] = zero_u32;
                            }
                        }
                    }
                    _phase_mainloop_done ^= 1;
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
                this_bid_2 = _clc_ctaid_2 + (unsigned int)cta_rank;
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
