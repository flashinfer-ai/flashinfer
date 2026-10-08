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
#include "cake_dsa_indexer_topk_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 4
#define NUM_TMEM_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 65536
#define SMEM_SMEM_Q_STRIDE 65536
#define SMEM_SMEM_W_OFF 66560
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_K_OFF 67584
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_S_HIST_OFF 198656
#define SMEM_S_HIST_STAGE_BYTES 8192
#define SMEM_S_HIST_STRIDE 8192
#define SMEM_S_COUNT_OFF 206848
#define SMEM_S_COUNT_STAGE_BYTES 64
#define SMEM_S_COUNT_STRIDE 64
#define SMEM_S_TAU_OFF 206912
#define SMEM_S_TAU_STAGE_BYTES 128
#define SMEM_S_TAU_STRIDE 128
#define SMEM_S_FAIL_OFF 207040
#define SMEM_S_FAIL_STAGE_BYTES 32
#define SMEM_S_FAIL_STRIDE 32
#define SMEM_S_WSC_OFF 207104
#define SMEM_S_WSC_STAGE_BYTES 512
#define SMEM_S_WSC_STRIDE 512
#define SMEM_TOTAL 208128
#define TILE_UNROLL 1
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_indexer_topk_2fbd0bcee8bd4fc571c4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap W, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, long long* __restrict__ q_offsets, int* __restrict__ Indices, float* __restrict__ Scores, long long* __restrict__ Cand, int num_segments, int top_k, int ratio, int has_offsets, int cand_cap, int first_cap, int sample_tiles_max, int sample_shift_permille, int check_period, int grid_ctas, float softmax_scale, int n_split)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 48)
    #define umma_full_addr (mbar_base + 80)
    #define umma_empty_addr (mbar_base + 96)
    #define verdict_bar_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    float* smem_w = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_W_OFF);
    const int smem_w_addr = smem + SMEM_SMEM_W_OFF;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_K_OFF);
    const int smem_k_addr = smem + SMEM_SMEM_K_OFF;
    int* s_hist = reinterpret_cast<int*>(smem_raw + SMEM_S_HIST_OFF);
    const int s_hist_addr = smem + SMEM_S_HIST_OFF;
    int* s_count = reinterpret_cast<int*>(smem_raw + SMEM_S_COUNT_OFF);
    const int s_count_addr = smem + SMEM_S_COUNT_OFF;
    int* s_tau = reinterpret_cast<int*>(smem_raw + SMEM_S_TAU_OFF);
    const int s_tau_addr = smem + SMEM_S_TAU_OFF;
    int* s_fail = reinterpret_cast<int*>(smem_raw + SMEM_S_FAIL_OFF);
    const int s_fail_addr = smem + SMEM_S_FAIL_OFF;
    float* s_wsc = reinterpret_cast<float*>(smem_raw + SMEM_S_WSC_OFF);
    const int s_wsc_addr = smem + SMEM_S_WSC_OFF;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=288
            mbarrier_init(smem + 8, 288);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // umma_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            // verdict_bar: 1 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 10) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load_q ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                int unit_base_q = 0;
                unsigned int vphase_q = 0;
                #pragma unroll 1
                for (int seg = 0; seg < num_segments; seg++) {
                    int q0 = cu_seqlens_q[seg];
                    int lq = cu_seqlens_q[seg + 1] - q0;
                    int nb = (lq + 8 - 1) / 8;
                    #pragma unroll 1
                    for (int rnd = unit_base_q / grid_ctas; rnd < (unit_base_q + nb - 1) / grid_ctas + 1; rnd++) {
                        int bid_i = (int)bid;
                        int pos_e = bid_i;
                        if ((rnd & 1) != 0) {
                            pos_e = grid_ctas - 1 - pos_e;
                        }
                        int unit = rnd * grid_ctas + pos_e;
                        int _min_0 = ((unit - unit_base_q) < (unit_base_q + nb - 1 - unit) ? (unit - unit_base_q) : (unit_base_q + nb - 1 - unit));
                        if (_min_0 >= 0) {
                            int blk = nb - 1 - (unit - unit_base_q);
                            #pragma unroll 1
                            for (int attempt = 0; attempt < 2; attempt++) {
                                mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                                tma_3d_gmem2smem(smem_q_addr + load_q_stage * 65536, (&Q), 0, (q0 + blk * 8) * 32, 0, q_full_addr + (load_q_stage) * 8);
                                tma_2d_gmem2smem(smem_w_addr + load_q_stage * 1024, (&W), 0, q0 + blk * 8, q_full_addr + (load_q_stage) * 8);
                                mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 66560);
                                _phase_q_empty ^= 1;
                                if (attempt == 0) {
                                    mbarrier_wait(verdict_bar_addr, vphase_q);
                                    vphase_q ^= 1;
                                    int f = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                    f = f | s_fail[4];
                                    f = f | s_fail[5];
                                    f = f | s_fail[6];
                                    f = f | s_fail[7];
                                    if (f == 0) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    unit_base_q += nb;
                }
            }
            __syncwarp();
        }
    // ---- Role: load_k ----
    } else if (warp == 9) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // load_k_main
            unsigned int load_k_stage = 0;
            unsigned int _phase_k_empty = 1;
            if (elect_sync()) {
                int unit_base_k = 0;
                unsigned int vphase_k = 0;
                #pragma unroll 1
                for (int seg_1 = 0; seg_1 < num_segments; seg_1++) {
                    int q0_1 = cu_seqlens_q[seg_1];
                    int lq_1 = cu_seqlens_q[seg_1 + 1] - q0_1;
                    int k0 = cu_seqlens_k[seg_1];
                    int nb_1 = (lq_1 + 8 - 1) / 8;
                    #pragma unroll 1
                    for (int rnd_1 = unit_base_k / grid_ctas; rnd_1 < (unit_base_k + nb_1 - 1) / grid_ctas + 1; rnd_1++) {
                        int bid_i_1 = (int)bid;
                        int pos_e_1 = bid_i_1;
                        if ((rnd_1 & 1) != 0) {
                            pos_e_1 = grid_ctas - 1 - pos_e_1;
                        }
                        int unit_1 = rnd_1 * grid_ctas + pos_e_1;
                        int _min_1 = ((unit_1 - unit_base_k) < (unit_base_k + nb_1 - 1 - unit_1) ? (unit_1 - unit_base_k) : (unit_base_k + nb_1 - 1 - unit_1));
                        if (_min_1 >= 0) {
                            int blk_1 = nb_1 - 1 - (unit_1 - unit_base_k);
                            int q0_0 = cu_seqlens_q[seg_1];
                            int lq_1_1 = cu_seqlens_q[seg_1 + 1] - q0_0;
                            int lk = cu_seqlens_k[seg_1 + 1] - cu_seqlens_k[seg_1];
                            long long off = 0;
                            if (has_offsets != 0) {
                                off = q_offsets[seg_1];
                            } else if (ratio == 1) {
                                off = (long long)lk - (long long)lq_1_1;
                            }
                            long long off_2 = off;
                            int _min_2 = ((blk_1 * 8 + 7) < (lq_1_1 - 1) ? (blk_1 * 8 + 7) : (lq_1_1 - 1));
                            int last_u = _min_2;
                            long long num = off_2 + (long long)last_u + 1;
                            int vis = 0;
                            if (num > 0) {
                                long long quotient = num / (long long)ratio;
                                long long lk64 = (long long)lk;
                                long long bounded = ((quotient < lk64) ? quotient : lk64);
                                vis = (int)bounded;
                            }
                            int v_max = vis;
                            int n_tiles = (v_max + 128 - 1) / 128;
                            int n_tiles_3 = n_tiles;
                            int _min_3 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
                            int max_tiles = _min_3;
                            int stride = 1;
                            int n_sample = 0;
                            if (max_tiles >= 2) {
                                if (n_tiles_3 * 128 + 128 > cand_cap) {
                                    stride = (n_tiles_3 + max_tiles - 1) / max_tiles;
                                    if (stride >= 2) {
                                        n_sample = (n_tiles_3 + stride - 1) / stride;
                                    }
                                }
                            }
                            if (max_tiles >= 2) {
                                if (n_tiles_3 * 128 + 128 <= cand_cap) {
                                    if (n_tiles_3 * 64 > top_k) {
                                        int _min_4 = ((max_tiles) < (8) ? (max_tiles) : (8));
                                        int cap_fit = _min_4;
                                        if (cap_fit >= 2) {
                                            stride = (n_tiles_3 + cap_fit - 1) / cap_fit;
                                            if (stride >= 2) {
                                                n_sample = (n_tiles_3 + stride - 1) / stride;
                                            }
                                        }
                                    }
                                }
                            }
                            #pragma unroll 1
                            for (int attempt_1 = 0; attempt_1 < 2; attempt_1++) {
                                int n_s_k = n_sample;
                                if (attempt_1 != 0) {
                                    n_s_k = 0;
                                }
                                int d_k = 0;
                                int r_k = 0;
                                #pragma unroll 1
                                for (int ti = 0; ti < n_tiles_3; ti++) {
                                    int tile = d_k;
                                    mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                                    tma_3d_gmem2smem(smem_k_addr + load_k_stage * 32768, (&K), 0, k0 + tile * 128, 0, k_full_addr + (load_k_stage) * 8);
                                    mbarrier_arrive_expect_tx(k_full_addr + (load_k_stage) * 8, 32768);
                                    load_k_stage += 1;
                                    if (load_k_stage == 4) { load_k_stage = 0; _phase_k_empty ^= 1; }
                                    if (n_s_k > 0) {
                                        if (n_s_k > ti + 1) {
                                            d_k += stride;
                                        } else if (ti + 1 == n_s_k) {
                                            d_k = 1;
                                            r_k = 0;
                                        } else {
                                            d_k += 1;
                                            r_k += 1;
                                            if (r_k == stride - 1) {
                                                r_k = 0;
                                                d_k += 1;
                                            }
                                        }
                                    } else {
                                        d_k += 1;
                                    }
                                }
                                if (attempt_1 == 0) {
                                    mbarrier_wait(verdict_bar_addr, vphase_k);
                                    vphase_k ^= 1;
                                    int f_1 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                    f_1 = f_1 | s_fail[4];
                                    f_1 = f_1 | s_fail[5];
                                    f_1 = f_1 | s_fail[6];
                                    f_1 = f_1 | s_fail[7];
                                    if (f_1 == 0) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    unit_base_k += nb_1;
                }
            }
            __syncwarp();
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_k_stage = 0;
            unsigned int mma_tmem_stage = 0;
            int unit_base_m = 0;
            unsigned int vphase_m = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (int seg_2 = 0; seg_2 < num_segments; seg_2++) {
                int q0_2 = cu_seqlens_q[seg_2];
                int lq_2 = cu_seqlens_q[seg_2 + 1] - q0_2;
                int nb_2 = (lq_2 + 8 - 1) / 8;
                #pragma unroll 1
                for (int rnd_2 = unit_base_m / grid_ctas; rnd_2 < (unit_base_m + nb_2 - 1) / grid_ctas + 1; rnd_2++) {
                    int bid_i_2 = (int)bid;
                    int pos_e_2 = bid_i_2;
                    if ((rnd_2 & 1) != 0) {
                        pos_e_2 = grid_ctas - 1 - pos_e_2;
                    }
                    int unit_2 = rnd_2 * grid_ctas + pos_e_2;
                    int _min_5 = ((unit_2 - unit_base_m) < (unit_base_m + nb_2 - 1 - unit_2) ? (unit_2 - unit_base_m) : (unit_base_m + nb_2 - 1 - unit_2));
                    if (_min_5 >= 0) {
                        int blk_2 = nb_2 - 1 - (unit_2 - unit_base_m);
                        int q0_0_1 = cu_seqlens_q[seg_2];
                        int lq_1_2 = cu_seqlens_q[seg_2 + 1] - q0_0_1;
                        int lk_1 = cu_seqlens_k[seg_2 + 1] - cu_seqlens_k[seg_2];
                        long long off_1 = 0;
                        if (has_offsets != 0) {
                            off_1 = q_offsets[seg_2];
                        } else if (ratio == 1) {
                            off_1 = (long long)lk_1 - (long long)lq_1_2;
                        }
                        long long off_2_1 = off_1;
                        int _min_6 = ((blk_2 * 8 + 7) < (lq_1_2 - 1) ? (blk_2 * 8 + 7) : (lq_1_2 - 1));
                        int last_u_1 = _min_6;
                        long long num_1 = off_2_1 + (long long)last_u_1 + 1;
                        int vis_1 = 0;
                        if (num_1 > 0) {
                            long long quotient_1 = num_1 / (long long)ratio;
                            long long lk64_1 = (long long)lk_1;
                            long long bounded_1 = ((quotient_1 < lk64_1) ? quotient_1 : lk64_1);
                            vis_1 = (int)bounded_1;
                        }
                        int v_max_1 = vis_1;
                        int n_tiles_1 = (v_max_1 + 128 - 1) / 128;
                        int n_tiles_3_1 = n_tiles_1;
                        #pragma unroll 1
                        for (int attempt_2 = 0; attempt_2 < 2; attempt_2++) {
                            mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full);
                            #pragma unroll 1
                            for (int ti_1 = 0; ti_1 < n_tiles_3_1; ti_1++) {
                                mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                                if (elect_sync()) {
                                    mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    int _mma_a_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
                                    int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 4096;
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138413200;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mma_tmem_stage * 256))), "r"(0));
                                    tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                                    tcgen05_commit(k_empty_addr + (mma_k_stage) * 8);
                                    mma_tmem_stage += 1;
                                    if (mma_tmem_stage == 2) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                                }
                                __syncwarp();
                                mma_k_stage += 1;
                                if (mma_k_stage == 4) { mma_k_stage = 0; _phase_k_full ^= 1; }
                            }
                            mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                            _phase_q_full ^= 1;
                            if (attempt_2 == 0) {
                                mbarrier_wait(verdict_bar_addr, vphase_m);
                                vphase_m ^= 1;
                                int f_2 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                f_2 = f_2 | s_fail[4];
                                f_2 = f_2 | s_fail[5];
                                f_2 = f_2 | s_fail[6];
                                f_2 = f_2 | s_fail[7];
                                if (f_2 == 0) {
                                    break;
                                }
                            }
                        }
                    }
                }
                unit_base_m += nb_2;
            }
        }
    // ---- Role: spare ----
    } else if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // spare_main
            __syncwarp();
        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
        { // math_main
            int warp_in_wg = warp % 4;
            int lane_0 = lane;
            int local_thread_idx = warp_in_wg * 32 + lane_0;
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int wg = (int)wg_idx;
            int pair = warp_in_wg / 2;
            int half = warp_in_wg % 2;
            int pair_bar = 4 + wg * 2 + pair;
            int bid_i_3 = (int)bid;
            unsigned int one_u = 1;
            unsigned int lower_lanes = (one_u << (unsigned int)lane) - 1;
            int hist_base = (wg * 2 + pair) * 256;
            unsigned int zero_u = 0;
            unsigned int sign_mask = 2147483648;
            unsigned int nan_floor = 2139095040;
            unsigned int never_u = 4294967295;
            unsigned int ninf_bits = 4286578688;
            unsigned int pinf_bits = 2139095040;
            unsigned int neg_key_floor = 8388607;
            unsigned int zero_gap_key = 2147483647;
            unsigned int pinf_key = 4286578688;
            unsigned int math_q_stage = 0;
            unsigned int math_tmem_stage = 0;
            unsigned int math_tmem_phase = 0;
            int unit_base = 0;
            int _min_7 = ((top_k + 64) < (cand_cap - 128 * check_period) ? (top_k + 64) : (cand_cap - 128 * check_period));
            int keep_cap = _min_7;
            int trigger_room = 128 * check_period;
            unsigned int _phase_q_full_1 = 0;
            #pragma unroll 1
            for (int seg_3 = 0; seg_3 < num_segments; seg_3++) {
                int q0_3 = cu_seqlens_q[seg_3];
                int lq_3 = cu_seqlens_q[seg_3 + 1] - q0_3;
                int lk_2 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                int nb_3 = (lq_3 + 8 - 1) / 8;
                long long off_3 = 0;
                if (has_offsets != 0) {
                    off_3 = q_offsets[seg_3];
                } else if (ratio == 1) {
                    off_3 = (long long)lk_2 - (long long)lq_3;
                }
                long long off_0 = off_3;
                #pragma unroll 1
                for (int rnd_3 = unit_base / grid_ctas; rnd_3 < (unit_base + nb_3 - 1) / grid_ctas + 1; rnd_3++) {
                    int bid_i_0 = (int)bid;
                    int pos_e_3 = bid_i_0;
                    if ((rnd_3 & 1) != 0) {
                        pos_e_3 = grid_ctas - 1 - pos_e_3;
                    }
                    int unit_3 = rnd_3 * grid_ctas + pos_e_3;
                    int _min_8 = ((unit_3 - unit_base) < (unit_base + nb_3 - 1 - unit_3) ? (unit_3 - unit_base) : (unit_base + nb_3 - 1 - unit_3));
                    if (_min_8 >= 0) {
                        int blk_3 = nb_3 - 1 - (unit_3 - unit_base);
                        int u_base = blk_3 * 8;
                        int q0_0_2 = cu_seqlens_q[seg_3];
                        int lq_1_3 = cu_seqlens_q[seg_3 + 1] - q0_0_2;
                        int lk_2_1 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                        long long off_3_1 = 0;
                        if (has_offsets != 0) {
                            off_3_1 = q_offsets[seg_3];
                        } else if (ratio == 1) {
                            off_3_1 = (long long)lk_2_1 - (long long)lq_1_3;
                        }
                        long long off_4 = off_3_1;
                        int _min_9 = ((blk_3 * 8 + 7) < (lq_1_3 - 1) ? (blk_3 * 8 + 7) : (lq_1_3 - 1));
                        int last_u_2 = _min_9;
                        long long num_2 = off_4 + (long long)last_u_2 + 1;
                        int vis_2 = 0;
                        if (num_2 > 0) {
                            long long quotient_2 = num_2 / (long long)ratio;
                            long long lk64_2 = (long long)lk_2_1;
                            long long bounded_2 = ((quotient_2 < lk64_2) ? quotient_2 : lk64_2);
                            vis_2 = (int)bounded_2;
                        }
                        int v_max_2 = vis_2;
                        int n_tiles_2 = (v_max_2 + 128 - 1) / 128;
                        int n_tiles_5 = n_tiles_2;
                        int _min_10 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
                        int max_tiles_1 = _min_10;
                        int stride_1 = 1;
                        int n_sample_1 = 0;
                        if (max_tiles_1 >= 2) {
                            if (n_tiles_5 * 128 + 128 > cand_cap) {
                                stride_1 = (n_tiles_5 + max_tiles_1 - 1) / max_tiles_1;
                                if (stride_1 >= 2) {
                                    n_sample_1 = (n_tiles_5 + stride_1 - 1) / stride_1;
                                }
                            }
                        }
                        if (max_tiles_1 >= 2) {
                            if (n_tiles_5 * 128 + 128 <= cand_cap) {
                                if (n_tiles_5 * 64 > top_k) {
                                    int _min_11 = ((max_tiles_1) < (8) ? (max_tiles_1) : (8));
                                    int cap_fit_1 = _min_11;
                                    if (cap_fit_1 >= 2) {
                                        stride_1 = (n_tiles_5 + cap_fit_1 - 1) / cap_fit_1;
                                        if (stride_1 >= 2) {
                                            n_sample_1 = (n_tiles_5 + stride_1 - 1) / stride_1;
                                        }
                                    }
                                }
                            }
                        }
                        int row_valid[4];
                        int visible[4];
                        unsigned long long tau[4];
                        float tau_t[4];
                        unsigned int tau_leq[4];
                        unsigned int tau_lnan[4];
                        int buf_base[4];
                        int buf_unit = (bid_i_3 * 8 + wg * 4) * cand_cap;
                        int u_q = u_base + wg * 4;
                        int valid_q = ((u_q < lq_3) ? 1 : 0);
                        row_valid[0] = valid_q;
                        int vis_q = 0;
                        if (valid_q != 0) {
                            long long num_0 = off_0 + (long long)u_q + 1;
                            int vis_1_1 = 0;
                            if (num_0 > 0) {
                                long long quotient_3 = num_0 / (long long)ratio;
                                long long lk64_3 = (long long)lk_2;
                                long long bounded_3 = ((quotient_3 < lk64_3) ? quotient_3 : lk64_3);
                                vis_1_1 = (int)bounded_3;
                            }
                            vis_q = vis_1_1;
                        }
                        visible[0] = vis_q;
                        buf_base[0] = buf_unit;
                        int u_q_6 = u_base + wg * 4 + 1;
                        int valid_q_7 = ((u_q_6 < lq_3) ? 1 : 0);
                        row_valid[1] = valid_q_7;
                        int vis_q_8 = 0;
                        if (valid_q_7 != 0) {
                            long long num_0_1 = off_0 + (long long)u_q_6 + 1;
                            int vis_1_2 = 0;
                            if (num_0_1 > 0) {
                                long long quotient_4 = num_0_1 / (long long)ratio;
                                long long lk64_4 = (long long)lk_2;
                                long long bounded_4 = ((quotient_4 < lk64_4) ? quotient_4 : lk64_4);
                                vis_1_2 = (int)bounded_4;
                            }
                            vis_q_8 = vis_1_2;
                        }
                        visible[1] = vis_q_8;
                        buf_base[1] = buf_unit + cand_cap;
                        int u_q_9 = u_base + wg * 4 + 2;
                        int valid_q_10 = ((u_q_9 < lq_3) ? 1 : 0);
                        row_valid[2] = valid_q_10;
                        int vis_q_11 = 0;
                        if (valid_q_10 != 0) {
                            long long num_0_2 = off_0 + (long long)u_q_9 + 1;
                            int vis_1_3 = 0;
                            if (num_0_2 > 0) {
                                long long quotient_5 = num_0_2 / (long long)ratio;
                                long long lk64_5 = (long long)lk_2;
                                long long bounded_5 = ((quotient_5 < lk64_5) ? quotient_5 : lk64_5);
                                vis_1_3 = (int)bounded_5;
                            }
                            vis_q_11 = vis_1_3;
                        }
                        visible[2] = vis_q_11;
                        buf_base[2] = buf_unit + 2 * cand_cap;
                        int u_q_12 = u_base + wg * 4 + 3;
                        int valid_q_13 = ((u_q_12 < lq_3) ? 1 : 0);
                        row_valid[3] = valid_q_13;
                        int vis_q_14 = 0;
                        if (valid_q_13 != 0) {
                            long long num_0_3 = off_0 + (long long)u_q_12 + 1;
                            int vis_1_4 = 0;
                            if (num_0_3 > 0) {
                                long long quotient_6 = num_0_3 / (long long)ratio;
                                long long lk64_6 = (long long)lk_2;
                                long long bounded_6 = ((quotient_6 < lk64_6) ? quotient_6 : lk64_6);
                                vis_1_4 = (int)bounded_6;
                            }
                            vis_q_14 = vis_1_4;
                        }
                        visible[3] = vis_q_14;
                        buf_base[3] = buf_unit + 3 * cand_cap;
                        int buf_pair = buf_unit + pair * cand_cap;
                        int slot_pair = wg * 4 + pair;
                        int valid_pair = row_valid[1];
                        int visible_pair = visible[1];
                        if (pair == 0) {
                            valid_pair = row_valid[0];
                            visible_pair = visible[0];
                        }
                        int buf_pairs[2];
                        int slot_pairs[2];
                        int valid_pairs[2];
                        int visible_pairs[2];
                        buf_pairs[0] = buf_unit + pair * 2 * cand_cap;
                        slot_pairs[0] = wg * 4 + pair * 2;
                        valid_pairs[0] = row_valid[2];
                        visible_pairs[0] = visible[2];
                        if (pair == 0) {
                            valid_pairs[0] = row_valid[0];
                            visible_pairs[0] = visible[0];
                        }
                        buf_pairs[1] = buf_unit + (pair * 2 + 1) * cand_cap;
                        slot_pairs[1] = wg * 4 + pair * 2 + 1;
                        valid_pairs[1] = row_valid[3];
                        visible_pairs[1] = visible[3];
                        if (pair == 0) {
                            valid_pairs[1] = row_valid[1];
                            visible_pairs[1] = visible[1];
                        }
                        #pragma unroll 1
                        for (int attempt_3 = 0; attempt_3 < 2; attempt_3++) {
                            int n_s = n_sample_1;
                            if (attempt_3 != 0) {
                                n_s = 0;
                            }
                            tau[0] = 0;
                            tau[1] = 0;
                            tau[2] = 0;
                            tau[3] = 0;
                            {
                                unsigned long long tq = tau[0];
                                unsigned int tau_hi = (unsigned int)(tq >> 32);
                                unsigned int tau_lo = (unsigned int)tq;
                                unsigned int t_bits = ~tau_hi;
                                unsigned int l_eq = tau_lo;
                                unsigned int l_nan = never_u;
                                if ((tau_hi & sign_mask) != 0) {
                                    t_bits = tau_hi ^ sign_mask;
                                }
                                if (tau_hi < neg_key_floor) {
                                    t_bits = ninf_bits;
                                    l_eq = zero_u;
                                }
                                if (tau_hi == 0) {
                                    l_nan = tau_lo;
                                }
                                if (tau_hi == zero_gap_key) {
                                    l_eq = zero_u;
                                }
                                if (tau_hi > pinf_key) {
                                    t_bits = pinf_bits;
                                    l_eq = never_u;
                                }
                                float t_f = 0.0f;
                                t_f = reinterpret_cast<float*>(&t_bits)[0];
                                tau_t[0] = t_f;
                                tau_leq[0] = l_eq;
                                tau_lnan[0] = l_nan;
                            }
                            {
                                unsigned long long tq_1 = tau[1];
                                unsigned int tau_hi_1 = (unsigned int)(tq_1 >> 32);
                                unsigned int tau_lo_1 = (unsigned int)tq_1;
                                unsigned int t_bits_1 = ~tau_hi_1;
                                unsigned int l_eq_1 = tau_lo_1;
                                unsigned int l_nan_1 = never_u;
                                if ((tau_hi_1 & sign_mask) != 0) {
                                    t_bits_1 = tau_hi_1 ^ sign_mask;
                                }
                                if (tau_hi_1 < neg_key_floor) {
                                    t_bits_1 = ninf_bits;
                                    l_eq_1 = zero_u;
                                }
                                if (tau_hi_1 == 0) {
                                    l_nan_1 = tau_lo_1;
                                }
                                if (tau_hi_1 == zero_gap_key) {
                                    l_eq_1 = zero_u;
                                }
                                if (tau_hi_1 > pinf_key) {
                                    t_bits_1 = pinf_bits;
                                    l_eq_1 = never_u;
                                }
                                float t_f_1 = 0.0f;
                                t_f_1 = reinterpret_cast<float*>(&t_bits_1)[0];
                                tau_t[1] = t_f_1;
                                tau_leq[1] = l_eq_1;
                                tau_lnan[1] = l_nan_1;
                            }
                            {
                                unsigned long long tq_2 = tau[2];
                                unsigned int tau_hi_2 = (unsigned int)(tq_2 >> 32);
                                unsigned int tau_lo_2 = (unsigned int)tq_2;
                                unsigned int t_bits_2 = ~tau_hi_2;
                                unsigned int l_eq_2 = tau_lo_2;
                                unsigned int l_nan_2 = never_u;
                                if ((tau_hi_2 & sign_mask) != 0) {
                                    t_bits_2 = tau_hi_2 ^ sign_mask;
                                }
                                if (tau_hi_2 < neg_key_floor) {
                                    t_bits_2 = ninf_bits;
                                    l_eq_2 = zero_u;
                                }
                                if (tau_hi_2 == 0) {
                                    l_nan_2 = tau_lo_2;
                                }
                                if (tau_hi_2 == zero_gap_key) {
                                    l_eq_2 = zero_u;
                                }
                                if (tau_hi_2 > pinf_key) {
                                    t_bits_2 = pinf_bits;
                                    l_eq_2 = never_u;
                                }
                                float t_f_2 = 0.0f;
                                t_f_2 = reinterpret_cast<float*>(&t_bits_2)[0];
                                tau_t[2] = t_f_2;
                                tau_leq[2] = l_eq_2;
                                tau_lnan[2] = l_nan_2;
                            }
                            {
                                unsigned long long tq_3 = tau[3];
                                unsigned int tau_hi_3 = (unsigned int)(tq_3 >> 32);
                                unsigned int tau_lo_3 = (unsigned int)tq_3;
                                unsigned int t_bits_3 = ~tau_hi_3;
                                unsigned int l_eq_3 = tau_lo_3;
                                unsigned int l_nan_3 = never_u;
                                if ((tau_hi_3 & sign_mask) != 0) {
                                    t_bits_3 = tau_hi_3 ^ sign_mask;
                                }
                                if (tau_hi_3 < neg_key_floor) {
                                    t_bits_3 = ninf_bits;
                                    l_eq_3 = zero_u;
                                }
                                if (tau_hi_3 == 0) {
                                    l_nan_3 = tau_lo_3;
                                }
                                if (tau_hi_3 == zero_gap_key) {
                                    l_eq_3 = zero_u;
                                }
                                if (tau_hi_3 > pinf_key) {
                                    t_bits_3 = pinf_bits;
                                    l_eq_3 = never_u;
                                }
                                float t_f_3 = 0.0f;
                                t_f_3 = reinterpret_cast<float*>(&t_bits_3)[0];
                                tau_t[3] = t_f_3;
                                tau_leq[3] = l_eq_3;
                                tau_lnan[3] = l_nan_3;
                            }
                            if (local_thread_idx < 4) {
                                s_count[wg * 4 + local_thread_idx] = 0;
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full_1);
                            int weight_stage_base_s = math_q_stage * 256 + (unsigned int)(wg * 128);
                            if (local_thread_idx < 128) {
                                float w_raw_s = smem_w[weight_stage_base_s + local_thread_idx];
                                s_wsc[wg * 128 + local_thread_idx] = w_raw_s * softmax_scale;
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            int wsc_addr = s_wsc_addr + wg_idx * 512;
                            int d_pos = 0;
                            int r_pos = 0;
                            int until_check = check_period - 1;
                            constexpr int ti_2_unroll = TILE_UNROLL;
                            #pragma unroll ti_2_unroll
                            for (int ti_2 = 0; ti_2 < n_tiles_5; ti_2++) {
                                int tile_1 = d_pos;
                                int kid = tile_1 * 128 + local_thread_idx;
                                if (n_s > 0) {
                                    if (n_s > ti_2 + 1) {
                                        d_pos += stride_1;
                                    } else if (ti_2 + 1 == n_s) {
                                        d_pos = 1;
                                        r_pos = 0;
                                    } else {
                                        d_pos += 1;
                                        r_pos += 1;
                                        if (r_pos == stride_1 - 1) {
                                            r_pos = 0;
                                            d_pos += 1;
                                        }
                                    }
                                } else {
                                    d_pos += 1;
                                }
                                mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, math_tmem_phase);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                unsigned int bits2[4];
                                int emit2[4];
                                unsigned int ballot2[4];
                                float weights_reg[64];
                                float _tmem_load_0[32];
                                tmem_ld_x16(&_tmem_load_0[0], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                tmem_ld_x16(&_tmem_load_0[16], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                float _tmem_load_1[32];
                                tmem_ld_x16(&_tmem_load_1[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                tmem_ld_x16(&_tmem_load_1[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[0])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 3]))
                                    : "r"(wsc_addr));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 3]))
                                    : "r"(wsc_addr + 16));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[8])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 3]))
                                    : "r"(wsc_addr + 32));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[12])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 3]))
                                    : "r"(wsc_addr + 48));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[16])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 3]))
                                    : "r"(wsc_addr + 64));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[20])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 3]))
                                    : "r"(wsc_addr + 80));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[24])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 3]))
                                    : "r"(wsc_addr + 96));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[28])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 3]))
                                    : "r"(wsc_addr + 112));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[32])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 3]))
                                    : "r"(wsc_addr + 128));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[36])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 3]))
                                    : "r"(wsc_addr + 144));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[40])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 3]))
                                    : "r"(wsc_addr + 160));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[44])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 3]))
                                    : "r"(wsc_addr + 176));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[48])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 3]))
                                    : "r"(wsc_addr + 192));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[52])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 3]))
                                    : "r"(wsc_addr + 208));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[56])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 3]))
                                    : "r"(wsc_addr + 224));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[60])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 3]))
                                    : "r"(wsc_addr + 240));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                {
                                    float _relu_wsum_0;
                                    {
                                        float2 _sum0 = make_float2(0.0f, 0.0f);
                                        float2 _sum1 = make_float2(0.0f, 0.0f);
                                        #pragma unroll
                                        for (int _j = 0; _j < 32; _j += 4) {
                                            float2 _a0_raw = make_float2(_tmem_load_0[0 + _j], _tmem_load_0[0 + _j + 1]);
                                            float2 _a0_abs = make_float2(fabsf(_tmem_load_0[0 + _j]), fabsf(_tmem_load_0[0 + _j + 1]));
                                            float2 _a0;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                            float2 _b0 = make_float2(weights_reg[0 + _j], weights_reg[0 + _j + 1]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                            float2 _a1_raw = make_float2(_tmem_load_0[0 + _j + 2], _tmem_load_0[0 + _j + 3]);
                                            float2 _a1_abs = make_float2(fabsf(_tmem_load_0[0 + _j + 2]), fabsf(_tmem_load_0[0 + _j + 3]));
                                            float2 _a1;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                            float2 _b1 = make_float2(weights_reg[0 + _j + 2], weights_reg[0 + _j + 3]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                        }
                                        float2 _sum;
                                        asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                        _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                                    }
                                    float score_w = _relu_wsum_0;
                                    unsigned int bits_w = 0;
                                    bits_w = reinterpret_cast<unsigned int*>(&score_w)[0];
                                    bits2[0] = bits_w;
                                    int in_prefix_w = ((kid < visible[0]) ? 1 : 0);
                                    int above_w = ((score_w > tau_t[0] || score_w == tau_t[0] && tau_leq[0] <= (unsigned int)kid || score_w != score_w && tau_lnan[0] <= (unsigned int)kid) ? 1 : 0);
                                    int emit_w = row_valid[0] & in_prefix_w & above_w;
                                    emit2[0] = emit_w;
                                    unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, emit_w != 0);
                                    ballot2[0] = _vote_0;
                                }
                                float _tmem_load_2[32];
                                tmem_ld_x16(&_tmem_load_2[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                tmem_ld_x16(&_tmem_load_2[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                float _tmem_load_3[32];
                                tmem_ld_x16(&_tmem_load_3[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                tmem_ld_x16(&_tmem_load_3[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                {
                                    float _relu_wsum_1;
                                    {
                                        float2 _sum0 = make_float2(0.0f, 0.0f);
                                        float2 _sum1 = make_float2(0.0f, 0.0f);
                                        #pragma unroll
                                        for (int _j = 0; _j < 32; _j += 4) {
                                            float2 _a0_raw = make_float2(_tmem_load_1[0 + _j], _tmem_load_1[0 + _j + 1]);
                                            float2 _a0_abs = make_float2(fabsf(_tmem_load_1[0 + _j]), fabsf(_tmem_load_1[0 + _j + 1]));
                                            float2 _a0;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                            float2 _b0 = make_float2(weights_reg[32 + _j], weights_reg[32 + _j + 1]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                            float2 _a1_raw = make_float2(_tmem_load_1[0 + _j + 2], _tmem_load_1[0 + _j + 3]);
                                            float2 _a1_abs = make_float2(fabsf(_tmem_load_1[0 + _j + 2]), fabsf(_tmem_load_1[0 + _j + 3]));
                                            float2 _a1;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                            float2 _b1 = make_float2(weights_reg[32 + _j + 2], weights_reg[32 + _j + 3]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                        }
                                        float2 _sum;
                                        asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                        _relu_wsum_1 = (_sum.x + _sum.y) * 0.5f;
                                    }
                                    float score_w_1 = _relu_wsum_1;
                                    unsigned int bits_w_1 = 0;
                                    bits_w_1 = reinterpret_cast<unsigned int*>(&score_w_1)[0];
                                    bits2[1] = bits_w_1;
                                    int in_prefix_w_1 = ((kid < visible[1]) ? 1 : 0);
                                    int above_w_1 = ((score_w_1 > tau_t[1] || score_w_1 == tau_t[1] && tau_leq[1] <= (unsigned int)kid || score_w_1 != score_w_1 && tau_lnan[1] <= (unsigned int)kid) ? 1 : 0);
                                    int emit_w_1 = row_valid[1] & in_prefix_w_1 & above_w_1;
                                    emit2[1] = emit_w_1;
                                    unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, emit_w_1 != 0);
                                    ballot2[1] = _vote_1;
                                }
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[0])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(0) + 3]))
                                    : "r"(wsc_addr + 256));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(4) + 3]))
                                    : "r"(wsc_addr + 272));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[8])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(8) + 3]))
                                    : "r"(wsc_addr + 288));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[12])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(12) + 3]))
                                    : "r"(wsc_addr + 304));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[16])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(16) + 3]))
                                    : "r"(wsc_addr + 320));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[20])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(20) + 3]))
                                    : "r"(wsc_addr + 336));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[24])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(24) + 3]))
                                    : "r"(wsc_addr + 352));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[28])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(28) + 3]))
                                    : "r"(wsc_addr + 368));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[32])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(32) + 3]))
                                    : "r"(wsc_addr + 384));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[36])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(36) + 3]))
                                    : "r"(wsc_addr + 400));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[40])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(40) + 3]))
                                    : "r"(wsc_addr + 416));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[44])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(44) + 3]))
                                    : "r"(wsc_addr + 432));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[48])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(48) + 3]))
                                    : "r"(wsc_addr + 448));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[52])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(52) + 3]))
                                    : "r"(wsc_addr + 464));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[56])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(56) + 3]))
                                    : "r"(wsc_addr + 480));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[60])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(60) + 3]))
                                    : "r"(wsc_addr + 496));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
                                {
                                    float _relu_wsum_2;
                                    {
                                        float2 _sum0 = make_float2(0.0f, 0.0f);
                                        float2 _sum1 = make_float2(0.0f, 0.0f);
                                        #pragma unroll
                                        for (int _j = 0; _j < 32; _j += 4) {
                                            float2 _a0_raw = make_float2(_tmem_load_2[0 + _j], _tmem_load_2[0 + _j + 1]);
                                            float2 _a0_abs = make_float2(fabsf(_tmem_load_2[0 + _j]), fabsf(_tmem_load_2[0 + _j + 1]));
                                            float2 _a0;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                            float2 _b0 = make_float2(weights_reg[0 + _j], weights_reg[0 + _j + 1]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                            float2 _a1_raw = make_float2(_tmem_load_2[0 + _j + 2], _tmem_load_2[0 + _j + 3]);
                                            float2 _a1_abs = make_float2(fabsf(_tmem_load_2[0 + _j + 2]), fabsf(_tmem_load_2[0 + _j + 3]));
                                            float2 _a1;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                            float2 _b1 = make_float2(weights_reg[0 + _j + 2], weights_reg[0 + _j + 3]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                        }
                                        float2 _sum;
                                        asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                        _relu_wsum_2 = (_sum.x + _sum.y) * 0.5f;
                                    }
                                    float score_w_2 = _relu_wsum_2;
                                    unsigned int bits_w_2 = 0;
                                    bits_w_2 = reinterpret_cast<unsigned int*>(&score_w_2)[0];
                                    bits2[2] = bits_w_2;
                                    int in_prefix_w_2 = ((kid < visible[2]) ? 1 : 0);
                                    int above_w_2 = ((score_w_2 > tau_t[2] || score_w_2 == tau_t[2] && tau_leq[2] <= (unsigned int)kid || score_w_2 != score_w_2 && tau_lnan[2] <= (unsigned int)kid) ? 1 : 0);
                                    int emit_w_2 = row_valid[2] & in_prefix_w_2 & above_w_2;
                                    emit2[2] = emit_w_2;
                                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, emit_w_2 != 0);
                                    ballot2[2] = _vote_2;
                                }
                                {
                                    float _relu_wsum_3;
                                    {
                                        float2 _sum0 = make_float2(0.0f, 0.0f);
                                        float2 _sum1 = make_float2(0.0f, 0.0f);
                                        #pragma unroll
                                        for (int _j = 0; _j < 32; _j += 4) {
                                            float2 _a0_raw = make_float2(_tmem_load_3[0 + _j], _tmem_load_3[0 + _j + 1]);
                                            float2 _a0_abs = make_float2(fabsf(_tmem_load_3[0 + _j]), fabsf(_tmem_load_3[0 + _j + 1]));
                                            float2 _a0;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                            float2 _b0 = make_float2(weights_reg[32 + _j], weights_reg[32 + _j + 1]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                            float2 _a1_raw = make_float2(_tmem_load_3[0 + _j + 2], _tmem_load_3[0 + _j + 3]);
                                            float2 _a1_abs = make_float2(fabsf(_tmem_load_3[0 + _j + 2]), fabsf(_tmem_load_3[0 + _j + 3]));
                                            float2 _a1;
                                            asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                            float2 _b1 = make_float2(weights_reg[32 + _j + 2], weights_reg[32 + _j + 3]);
                                            asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                        }
                                        float2 _sum;
                                        asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                        _relu_wsum_3 = (_sum.x + _sum.y) * 0.5f;
                                    }
                                    float score_w_3 = _relu_wsum_3;
                                    unsigned int bits_w_3 = 0;
                                    bits_w_3 = reinterpret_cast<unsigned int*>(&score_w_3)[0];
                                    bits2[3] = bits_w_3;
                                    int in_prefix_w_3 = ((kid < visible[3]) ? 1 : 0);
                                    int above_w_3 = ((score_w_3 > tau_t[3] || score_w_3 == tau_t[3] && tau_leq[3] <= (unsigned int)kid || score_w_3 != score_w_3 && tau_lnan[3] <= (unsigned int)kid) ? 1 : 0);
                                    int emit_w_3 = row_valid[3] & in_prefix_w_3 & above_w_3;
                                    emit2[3] = emit_w_3;
                                    unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, emit_w_3 != 0);
                                    ballot2[3] = _vote_3;
                                }
                                int _popc_0 = __popc(ballot2[3]);
                                int cnt_lane = _popc_0;
                                if (lane_0 == 2) {
                                    int _popc_1 = __popc(ballot2[2]);
                                    cnt_lane = _popc_1;
                                }
                                if (lane_0 == 1) {
                                    int _popc_2 = __popc(ballot2[1]);
                                    cnt_lane = _popc_2;
                                }
                                if (lane_0 == 0) {
                                    int _popc_3 = __popc(ballot2[0]);
                                    cnt_lane = _popc_3;
                                }
                                int reservation = 0;
                                if (lane_0 < 4) {
                                    int _atomic_old_0 = atomicAdd(s_count + (wg * 4 + lane_0), cnt_lane);
                                    reservation = _atomic_old_0;
                                }
                                {
                                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, reservation, 0);
                                    int slot_base = _shfl_0;
                                    if (emit2[0] != 0) {
                                        int _popc_4 = __popc(ballot2[0] & lower_lanes);
                                        int slot = buf_base[0] + slot_base + _popc_4;
                                        unsigned long long entry = (unsigned long long)bits2[0] << 32 | (unsigned long long)(unsigned int)kid;
                                        Cand[slot] = (long long)entry;
                                    }
                                }
                                {
                                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, reservation, 1);
                                    int slot_base_1 = _shfl_1;
                                    if (emit2[1] != 0) {
                                        int _popc_5 = __popc(ballot2[1] & lower_lanes);
                                        int slot_1 = buf_base[1] + slot_base_1 + _popc_5;
                                        unsigned long long entry_1 = (unsigned long long)bits2[1] << 32 | (unsigned long long)(unsigned int)kid;
                                        Cand[slot_1] = (long long)entry_1;
                                    }
                                }
                                {
                                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, reservation, 2);
                                    int slot_base_2 = _shfl_2;
                                    if (emit2[2] != 0) {
                                        int _popc_6 = __popc(ballot2[2] & lower_lanes);
                                        int slot_2 = buf_base[2] + slot_base_2 + _popc_6;
                                        unsigned long long entry_2 = (unsigned long long)bits2[2] << 32 | (unsigned long long)(unsigned int)kid;
                                        Cand[slot_2] = (long long)entry_2;
                                    }
                                }
                                {
                                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, reservation, 3);
                                    int slot_base_3 = _shfl_3;
                                    if (emit2[3] != 0) {
                                        int _popc_7 = __popc(ballot2[3] & lower_lanes);
                                        int slot_3 = buf_base[3] + slot_base_3 + _popc_7;
                                        unsigned long long entry_3 = (unsigned long long)bits2[3] << 32 | (unsigned long long)(unsigned int)kid;
                                        Cand[slot_3] = (long long)entry_3;
                                    }
                                }
                                math_tmem_stage += 1;
                                if (math_tmem_stage >= 2) {
                                    math_tmem_stage -= 2;
                                    math_tmem_phase ^= 1;
                                }
                                int sample_now = ((ti_2 + 1 == n_s) ? 1 : 0);
                                int do_check = ((until_check == 0) ? 1 : 0);
                                do_check = do_check | sample_now;
                                if (do_check == 0) {
                                    until_check -= 1;
                                } else {
                                    until_check = check_period - 1;
                                    asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                                    int need_any = 0;
                                    int need[4];
                                    int kw[4];
                                    int km[4];
                                    int c_q = s_count[wg * 4];
                                    int limit_q = ((tau[0] != 0) ? cand_cap : first_cap);
                                    int need_q = ((limit_q < c_q + trigger_room) ? 1 : 0);
                                    int kw_q = top_k;
                                    int km_q = keep_cap;
                                    if (need_q == 0) {
                                        if (sample_now != 0) {
                                            if (row_valid[0] != 0) {
                                                if (visible[0] > 0) {
                                                    int r_q = (top_k * c_q + visible[0] - 1) / visible[0];
                                                    int r_shift = r_q + r_q * sample_shift_permille / 1000 + 24;
                                                    if (r_shift < c_q) {
                                                        need_q = 1;
                                                        kw_q = r_shift;
                                                        int _min_12 = ((r_shift + 64) < (keep_cap) ? (r_shift + 64) : (keep_cap));
                                                        km_q = _min_12;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need[0] = need_q;
                                    kw[0] = kw_q;
                                    km[0] = km_q;
                                    need_any = need_any | need_q;
                                    int c_q_0 = s_count[wg * 4 + 1];
                                    int limit_q_1 = ((tau[1] != 0) ? cand_cap : first_cap);
                                    int need_q_2 = ((limit_q_1 < c_q_0 + trigger_room) ? 1 : 0);
                                    int kw_q_3 = top_k;
                                    int km_q_4 = keep_cap;
                                    if (need_q_2 == 0) {
                                        if (sample_now != 0) {
                                            if (row_valid[1] != 0) {
                                                if (visible[1] > 0) {
                                                    int r_q_1 = (top_k * c_q_0 + visible[1] - 1) / visible[1];
                                                    int r_shift_1 = r_q_1 + r_q_1 * sample_shift_permille / 1000 + 24;
                                                    if (r_shift_1 < c_q_0) {
                                                        need_q_2 = 1;
                                                        kw_q_3 = r_shift_1;
                                                        int _min_13 = ((r_shift_1 + 64) < (keep_cap) ? (r_shift_1 + 64) : (keep_cap));
                                                        km_q_4 = _min_13;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need[1] = need_q_2;
                                    kw[1] = kw_q_3;
                                    km[1] = km_q_4;
                                    need_any = need_any | need_q_2;
                                    int c_q_5 = s_count[wg * 4 + 2];
                                    int limit_q_6 = ((tau[2] != 0) ? cand_cap : first_cap);
                                    int need_q_7 = ((limit_q_6 < c_q_5 + trigger_room) ? 1 : 0);
                                    int kw_q_8 = top_k;
                                    int km_q_9 = keep_cap;
                                    if (need_q_7 == 0) {
                                        if (sample_now != 0) {
                                            if (row_valid[2] != 0) {
                                                if (visible[2] > 0) {
                                                    int r_q_2 = (top_k * c_q_5 + visible[2] - 1) / visible[2];
                                                    int r_shift_2 = r_q_2 + r_q_2 * sample_shift_permille / 1000 + 24;
                                                    if (r_shift_2 < c_q_5) {
                                                        need_q_7 = 1;
                                                        kw_q_8 = r_shift_2;
                                                        int _min_14 = ((r_shift_2 + 64) < (keep_cap) ? (r_shift_2 + 64) : (keep_cap));
                                                        km_q_9 = _min_14;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need[2] = need_q_7;
                                    kw[2] = kw_q_8;
                                    km[2] = km_q_9;
                                    need_any = need_any | need_q_7;
                                    int c_q_10 = s_count[wg * 4 + 3];
                                    int limit_q_11 = ((tau[3] != 0) ? cand_cap : first_cap);
                                    int need_q_12 = ((limit_q_11 < c_q_10 + trigger_room) ? 1 : 0);
                                    int kw_q_13 = top_k;
                                    int km_q_14 = keep_cap;
                                    if (need_q_12 == 0) {
                                        if (sample_now != 0) {
                                            if (row_valid[3] != 0) {
                                                if (visible[3] > 0) {
                                                    int r_q_3 = (top_k * c_q_10 + visible[3] - 1) / visible[3];
                                                    int r_shift_3 = r_q_3 + r_q_3 * sample_shift_permille / 1000 + 24;
                                                    if (r_shift_3 < c_q_10) {
                                                        need_q_12 = 1;
                                                        kw_q_13 = r_shift_3;
                                                        int _min_15 = ((r_shift_3 + 64) < (keep_cap) ? (r_shift_3 + 64) : (keep_cap));
                                                        km_q_14 = _min_15;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need[3] = need_q_12;
                                    kw[3] = kw_q_13;
                                    km[3] = km_q_14;
                                    need_any = need_any | need_q_12;
                                    asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                                    if (need_any != 0) {
                                        {
                                            int need_j = need[2];
                                            int kw_j = kw[2];
                                            int km_j = km[2];
                                            if (pair == 0) {
                                                need_j = need[0];
                                                kw_j = kw[0];
                                                km_j = km[0];
                                            }
                                            if (need_j != 0) {
                                                int c_j = s_count[slot_pairs[0]];
                                                int k_rem = kw_j;
                                                int kept_above = 0;
                                                unsigned long long prefix = 0;
                                                unsigned long long edge = 0;
                                                #pragma unroll 1
                                                for (int p = 0; p < 8; p++) {
                                                    int shift = 56 - 8 * p;
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    if (half == 0) {
                                                        s_hist[hist_base + lane_0 * 8] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 1] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 2] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 3] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 4] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 5] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 6] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 7] = 0;
                                                    }
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    unsigned long long ring[16];
                                                    #pragma unroll
                                                    for (int s_pro = 0; s_pro < 1; s_pro++) {
                                                        #pragma unroll
                                                        for (int u = 0; u < 8; u++) {
                                                            int idx_l = s_pro * 512 + half * 32 + lane_0 + 64 * u;
                                                            ring[s_pro * 8 + u] = 0;
                                                            if (idx_l < c_j) {
                                                                ring[s_pro * 8 + u] = (unsigned long long)Cand[buf_pairs[0] + idx_l];
                                                            }
                                                        }
                                                    }
                                                    int nbat = (c_j + 512 - 1) / 512;
                                                    #pragma unroll 1
                                                    for (int g = 0; g < nbat; g += 2) {
                                                        #pragma unroll
                                                        for (int s_st = 0; s_st < 2; s_st++) {
                                                            #pragma unroll
                                                            for (int u_1 = 0; u_1 < 8; u_1++) {
                                                                int valid_p = ((c_j > (g + s_st) * 512 + half * 32 + lane_0 + 64 * u_1) ? 1 : 0);
                                                                unsigned int bits = (unsigned int)(ring[s_st * 8 + u_1] >> 32);
                                                                int kid_0 = (int)(unsigned int)(ring[s_st * 8 + u_1] & 4294967295);
                                                                unsigned int magnitude = bits & 2147483647;
                                                                unsigned int m = ((magnitude != 0) ? bits : zero_u);
                                                                unsigned int key32 = (((m & sign_mask) != 0) ? ~m : m | sign_mask);
                                                                if (magnitude > nan_floor) {
                                                                    key32 = zero_u;
                                                                }
                                                                unsigned long long key64 = (unsigned long long)key32 << 32 | (unsigned long long)(unsigned int)kid_0;
                                                                unsigned long long key = key64;
                                                                unsigned long long key_p = key;
                                                                unsigned long long ks_p = key_p >> (unsigned long long)shift;
                                                                int digit_p = (int)(ks_p & 255);
                                                                int pm_p = ((ks_p >> 8 == prefix) ? 1 : 0);
                                                                int bin_p = (((valid_p & pm_p) != 0) ? hist_base + digit_p : hist_base + 1024);
                                                                atomicAdd(&s_hist[bin_p], 1);
                                                            }
                                                            #pragma unroll
                                                            for (int u_2 = 0; u_2 < 8; u_2++) {
                                                                int idx_l_1 = (g + s_st + 1) * 512 + half * 32 + lane_0 + 64 * u_2;
                                                                ring[(s_st + 1) % 2 * 8 + u_2] = 0;
                                                                if (idx_l_1 < c_j) {
                                                                    ring[(s_st + 1) % 2 * 8 + u_2] = (unsigned long long)Cand[buf_pairs[0] + idx_l_1];
                                                                }
                                                            }
                                                        }
                                                    }
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    int lane_bins[8];
                                                    int lane_sum = 0;
                                                    lane_bins[0] = s_hist[hist_base + lane_0 * 8];
                                                    lane_sum += lane_bins[0];
                                                    lane_bins[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                                    lane_sum += lane_bins[1];
                                                    lane_bins[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                                    lane_sum += lane_bins[2];
                                                    lane_bins[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                                    lane_sum += lane_bins[3];
                                                    lane_bins[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                                    lane_sum += lane_bins[4];
                                                    lane_bins[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                                    lane_sum += lane_bins[5];
                                                    lane_bins[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                                    lane_sum += lane_bins[6];
                                                    lane_bins[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                                    lane_sum += lane_bins[7];
                                                    int suffix = lane_sum;
                                                    int _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                                                    int above_part = _shfl_down_0;
                                                    if (lane_0 + 1 < 32) {
                                                        suffix += above_part;
                                                    }
                                                    int _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
                                                    int above_part_0 = _shfl_down_1;
                                                    if (lane_0 + 2 < 32) {
                                                        suffix += above_part_0;
                                                    }
                                                    int _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
                                                    int above_part_1 = _shfl_down_2;
                                                    if (lane_0 + 4 < 32) {
                                                        suffix += above_part_1;
                                                    }
                                                    int _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
                                                    int above_part_2 = _shfl_down_3;
                                                    if (lane_0 + 8 < 32) {
                                                        suffix += above_part_2;
                                                    }
                                                    int _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
                                                    int above_part_3 = _shfl_down_4;
                                                    if (lane_0 + 16 < 32) {
                                                        suffix += above_part_3;
                                                    }
                                                    int excl = suffix - lane_sum;
                                                    int is_target = ((excl < k_rem && k_rem <= excl + lane_sum) ? 1 : 0);
                                                    int d_sel = 0;
                                                    int above_sel = 0;
                                                    int count_sel = 0;
                                                    int found = 0;
                                                    int cum_above = excl;
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[7]) {
                                                            d_sel = 7;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[7];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[7];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[6]) {
                                                            d_sel = 6;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[6];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[6];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[5]) {
                                                            d_sel = 5;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[5];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[5];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[4]) {
                                                            d_sel = 4;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[4];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[4];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[3]) {
                                                            d_sel = 3;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[3];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[3];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[2]) {
                                                            d_sel = 2;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[2];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[2];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[1]) {
                                                            d_sel = 1;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[1];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[1];
                                                    if (found == 0) {
                                                        if (k_rem <= cum_above + lane_bins[0]) {
                                                            d_sel = 0;
                                                            above_sel = cum_above;
                                                            count_sel = lane_bins[0];
                                                            found = 1;
                                                        }
                                                    }
                                                    cum_above += lane_bins[0];
                                                    int _warp_redux_i32_0;
                                                    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(is_target * lane_0));
                                                    int target_lane = _warp_redux_i32_0;
                                                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
                                                    int digit_sel = _shfl_4;
                                                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
                                                    int above_cnt = _shfl_5;
                                                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
                                                    int bucket_cnt = _shfl_6;
                                                    k_rem = k_rem - above_cnt;
                                                    kept_above += above_cnt;
                                                    prefix = prefix << 8 | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
                                                    edge = prefix << (unsigned long long)shift;
                                                    if (km_j >= kept_above + bucket_cnt) {
                                                        break;
                                                    }
                                                }
                                                unsigned long long edge_0 = edge;
                                                if (half == 0) {
                                                    int write_pos = 0;
                                                    #pragma unroll 1
                                                    for (int e0 = 0; e0 < c_j; e0 += 256) {
                                                        unsigned long long ent[8];
                                                        #pragma unroll
                                                        for (int u_3 = 0; u_3 < 8; u_3++) {
                                                            ent[u_3] = 0;
                                                            if (c_j > e0 + lane_0 + 32 * u_3) {
                                                                ent[u_3] = (unsigned long long)Cand[buf_pairs[0] + e0 + lane_0 + 32 * u_3];
                                                            }
                                                        }
                                                        #pragma unroll
                                                        for (int u_4 = 0; u_4 < 8; u_4++) {
                                                            int keep = 0;
                                                            if (c_j > e0 + lane_0 + 32 * u_4) {
                                                                unsigned int bits_1 = (unsigned int)(ent[u_4] >> 32);
                                                                int kid_0_1 = (int)(unsigned int)(ent[u_4] & 4294967295);
                                                                unsigned int magnitude_1 = bits_1 & 2147483647;
                                                                unsigned int m_1 = ((magnitude_1 != 0) ? bits_1 : zero_u);
                                                                unsigned int key32_1 = (((m_1 & sign_mask) != 0) ? ~m_1 : m_1 | sign_mask);
                                                                if (magnitude_1 > nan_floor) {
                                                                    key32_1 = zero_u;
                                                                }
                                                                unsigned long long key64_1 = (unsigned long long)key32_1 << 32 | (unsigned long long)(unsigned int)kid_0_1;
                                                                unsigned long long key_1 = key64_1;
                                                                keep = ((key_1 >= edge_0) ? 1 : 0);
                                                            }
                                                            unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, keep != 0);
                                                            if (keep != 0) {
                                                                int _popc_8 = __popc(_vote_4 & lower_lanes);
                                                                Cand[buf_pairs[0] + write_pos + _popc_8] = (long long)ent[u_4];
                                                            }
                                                            int _popc_9 = __popc(_vote_4);
                                                            write_pos += _popc_9;
                                                        }
                                                    }
                                                    if (lane_0 == 0) {
                                                        s_count[slot_pairs[0]] = write_pos;
                                                        s_tau[slot_pairs[0] * 2] = (int)(unsigned int)edge_0;
                                                        s_tau[slot_pairs[0] * 2 + 1] = (int)(unsigned int)(edge_0 >> 32);
                                                    }
                                                }
                                                asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            }
                                        }
                                        {
                                            int need_j_1 = need[3];
                                            int kw_j_1 = kw[3];
                                            int km_j_1 = km[3];
                                            if (pair == 0) {
                                                need_j_1 = need[1];
                                                kw_j_1 = kw[1];
                                                km_j_1 = km[1];
                                            }
                                            if (need_j_1 != 0) {
                                                int c_j_1 = s_count[slot_pairs[1]];
                                                int k_rem_1 = kw_j_1;
                                                int kept_above_1 = 0;
                                                unsigned long long prefix_1 = 0;
                                                unsigned long long edge_1 = 0;
                                                #pragma unroll 1
                                                for (int p_1 = 0; p_1 < 8; p_1++) {
                                                    int shift_1 = 56 - 8 * p_1;
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    if (half == 0) {
                                                        s_hist[hist_base + lane_0 * 8] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 1] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 2] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 3] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 4] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 5] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 6] = 0;
                                                        s_hist[hist_base + lane_0 * 8 + 7] = 0;
                                                    }
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    unsigned long long ring_1[16];
                                                    #pragma unroll
                                                    for (int s_pro_1 = 0; s_pro_1 < 1; s_pro_1++) {
                                                        #pragma unroll
                                                        for (int u_5 = 0; u_5 < 8; u_5++) {
                                                            int idx_l_2 = s_pro_1 * 512 + half * 32 + lane_0 + 64 * u_5;
                                                            ring_1[s_pro_1 * 8 + u_5] = 0;
                                                            if (idx_l_2 < c_j_1) {
                                                                ring_1[s_pro_1 * 8 + u_5] = (unsigned long long)Cand[buf_pairs[1] + idx_l_2];
                                                            }
                                                        }
                                                    }
                                                    int nbat_1 = (c_j_1 + 512 - 1) / 512;
                                                    #pragma unroll 1
                                                    for (int g_1 = 0; g_1 < nbat_1; g_1 += 2) {
                                                        #pragma unroll
                                                        for (int s_st_1 = 0; s_st_1 < 2; s_st_1++) {
                                                            #pragma unroll
                                                            for (int u_6 = 0; u_6 < 8; u_6++) {
                                                                int valid_p_1 = ((c_j_1 > (g_1 + s_st_1) * 512 + half * 32 + lane_0 + 64 * u_6) ? 1 : 0);
                                                                unsigned int bits_2 = (unsigned int)(ring_1[s_st_1 * 8 + u_6] >> 32);
                                                                int kid_0_2 = (int)(unsigned int)(ring_1[s_st_1 * 8 + u_6] & 4294967295);
                                                                unsigned int magnitude_2 = bits_2 & 2147483647;
                                                                unsigned int m_2 = ((magnitude_2 != 0) ? bits_2 : zero_u);
                                                                unsigned int key32_2 = (((m_2 & sign_mask) != 0) ? ~m_2 : m_2 | sign_mask);
                                                                if (magnitude_2 > nan_floor) {
                                                                    key32_2 = zero_u;
                                                                }
                                                                unsigned long long key64_2 = (unsigned long long)key32_2 << 32 | (unsigned long long)(unsigned int)kid_0_2;
                                                                unsigned long long key_2 = key64_2;
                                                                unsigned long long key_p_1 = key_2;
                                                                unsigned long long ks_p_1 = key_p_1 >> (unsigned long long)shift_1;
                                                                int digit_p_1 = (int)(ks_p_1 & 255);
                                                                int pm_p_1 = ((ks_p_1 >> 8 == prefix_1) ? 1 : 0);
                                                                int bin_p_1 = (((valid_p_1 & pm_p_1) != 0) ? hist_base + digit_p_1 : hist_base + 1024);
                                                                atomicAdd(&s_hist[bin_p_1], 1);
                                                            }
                                                            #pragma unroll
                                                            for (int u_7 = 0; u_7 < 8; u_7++) {
                                                                int idx_l_3 = (g_1 + s_st_1 + 1) * 512 + half * 32 + lane_0 + 64 * u_7;
                                                                ring_1[(s_st_1 + 1) % 2 * 8 + u_7] = 0;
                                                                if (idx_l_3 < c_j_1) {
                                                                    ring_1[(s_st_1 + 1) % 2 * 8 + u_7] = (unsigned long long)Cand[buf_pairs[1] + idx_l_3];
                                                                }
                                                            }
                                                        }
                                                    }
                                                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                                    int lane_bins_1[8];
                                                    int lane_sum_1 = 0;
                                                    lane_bins_1[0] = s_hist[hist_base + lane_0 * 8];
                                                    lane_sum_1 += lane_bins_1[0];
                                                    lane_bins_1[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                                    lane_sum_1 += lane_bins_1[1];
                                                    lane_bins_1[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                                    lane_sum_1 += lane_bins_1[2];
                                                    lane_bins_1[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                                    lane_sum_1 += lane_bins_1[3];
                                                    lane_bins_1[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                                    lane_sum_1 += lane_bins_1[4];
                                                    lane_bins_1[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                                    lane_sum_1 += lane_bins_1[5];
                                                    lane_bins_1[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                                    lane_sum_1 += lane_bins_1[6];
                                                    lane_bins_1[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                                    lane_sum_1 += lane_bins_1[7];
                                                    int suffix_1 = lane_sum_1;
                                                    int _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 1, 32);
                                                    int above_part_4 = _shfl_down_5;
                                                    if (lane_0 + 1 < 32) {
                                                        suffix_1 += above_part_4;
                                                    }
                                                    int _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 2, 32);
                                                    int above_part_0_1 = _shfl_down_6;
                                                    if (lane_0 + 2 < 32) {
                                                        suffix_1 += above_part_0_1;
                                                    }
                                                    int _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 4, 32);
                                                    int above_part_1_1 = _shfl_down_7;
                                                    if (lane_0 + 4 < 32) {
                                                        suffix_1 += above_part_1_1;
                                                    }
                                                    int _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 8, 32);
                                                    int above_part_2_1 = _shfl_down_8;
                                                    if (lane_0 + 8 < 32) {
                                                        suffix_1 += above_part_2_1;
                                                    }
                                                    int _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 16, 32);
                                                    int above_part_3_1 = _shfl_down_9;
                                                    if (lane_0 + 16 < 32) {
                                                        suffix_1 += above_part_3_1;
                                                    }
                                                    int excl_1 = suffix_1 - lane_sum_1;
                                                    int is_target_1 = ((excl_1 < k_rem_1 && k_rem_1 <= excl_1 + lane_sum_1) ? 1 : 0);
                                                    int d_sel_1 = 0;
                                                    int above_sel_1 = 0;
                                                    int count_sel_1 = 0;
                                                    int found_1 = 0;
                                                    int cum_above_1 = excl_1;
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[7]) {
                                                            d_sel_1 = 7;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[7];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[7];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[6]) {
                                                            d_sel_1 = 6;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[6];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[6];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[5]) {
                                                            d_sel_1 = 5;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[5];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[5];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[4]) {
                                                            d_sel_1 = 4;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[4];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[4];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[3]) {
                                                            d_sel_1 = 3;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[3];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[3];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[2]) {
                                                            d_sel_1 = 2;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[2];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[2];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[1]) {
                                                            d_sel_1 = 1;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[1];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[1];
                                                    if (found_1 == 0) {
                                                        if (k_rem_1 <= cum_above_1 + lane_bins_1[0]) {
                                                            d_sel_1 = 0;
                                                            above_sel_1 = cum_above_1;
                                                            count_sel_1 = lane_bins_1[0];
                                                            found_1 = 1;
                                                        }
                                                    }
                                                    cum_above_1 += lane_bins_1[0];
                                                    int _warp_redux_i32_1;
                                                    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_1) : "r"(is_target_1 * lane_0));
                                                    int target_lane_1 = _warp_redux_i32_1;
                                                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, d_sel_1, target_lane_1);
                                                    int digit_sel_1 = _shfl_7;
                                                    int _shfl_8 = __shfl_sync(0xFFFFFFFF, above_sel_1, target_lane_1);
                                                    int above_cnt_1 = _shfl_8;
                                                    int _shfl_9 = __shfl_sync(0xFFFFFFFF, count_sel_1, target_lane_1);
                                                    int bucket_cnt_1 = _shfl_9;
                                                    k_rem_1 = k_rem_1 - above_cnt_1;
                                                    kept_above_1 += above_cnt_1;
                                                    prefix_1 = prefix_1 << 8 | (unsigned long long)(unsigned int)(target_lane_1 * 8 + digit_sel_1);
                                                    edge_1 = prefix_1 << (unsigned long long)shift_1;
                                                    if (km_j_1 >= kept_above_1 + bucket_cnt_1) {
                                                        break;
                                                    }
                                                }
                                                unsigned long long edge_0_1 = edge_1;
                                                if (half == 0) {
                                                    int write_pos_1 = 0;
                                                    #pragma unroll 1
                                                    for (int e0_1 = 0; e0_1 < c_j_1; e0_1 += 256) {
                                                        unsigned long long ent_1[8];
                                                        #pragma unroll
                                                        for (int u_8 = 0; u_8 < 8; u_8++) {
                                                            ent_1[u_8] = 0;
                                                            if (c_j_1 > e0_1 + lane_0 + 32 * u_8) {
                                                                ent_1[u_8] = (unsigned long long)Cand[buf_pairs[1] + e0_1 + lane_0 + 32 * u_8];
                                                            }
                                                        }
                                                        #pragma unroll
                                                        for (int u_9 = 0; u_9 < 8; u_9++) {
                                                            int keep_1 = 0;
                                                            if (c_j_1 > e0_1 + lane_0 + 32 * u_9) {
                                                                unsigned int bits_3 = (unsigned int)(ent_1[u_9] >> 32);
                                                                int kid_0_3 = (int)(unsigned int)(ent_1[u_9] & 4294967295);
                                                                unsigned int magnitude_3 = bits_3 & 2147483647;
                                                                unsigned int m_3 = ((magnitude_3 != 0) ? bits_3 : zero_u);
                                                                unsigned int key32_3 = (((m_3 & sign_mask) != 0) ? ~m_3 : m_3 | sign_mask);
                                                                if (magnitude_3 > nan_floor) {
                                                                    key32_3 = zero_u;
                                                                }
                                                                unsigned long long key64_3 = (unsigned long long)key32_3 << 32 | (unsigned long long)(unsigned int)kid_0_3;
                                                                unsigned long long key_3 = key64_3;
                                                                keep_1 = ((key_3 >= edge_0_1) ? 1 : 0);
                                                            }
                                                            unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, keep_1 != 0);
                                                            if (keep_1 != 0) {
                                                                int _popc_10 = __popc(_vote_5 & lower_lanes);
                                                                Cand[buf_pairs[1] + write_pos_1 + _popc_10] = (long long)ent_1[u_9];
                                                            }
                                                            int _popc_11 = __popc(_vote_5);
                                                            write_pos_1 += _popc_11;
                                                        }
                                                    }
                                                    if (lane_0 == 0) {
                                                        s_count[slot_pairs[1]] = write_pos_1;
                                                        s_tau[slot_pairs[1] * 2] = (int)(unsigned int)edge_0_1;
                                                        s_tau[slot_pairs[1] * 2 + 1] = (int)(unsigned int)(edge_0_1 >> 32);
                                                    }
                                                }
                                                asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            }
                                        }
                                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                                        if (need[0] != 0) {
                                            unsigned int lo = (unsigned int)s_tau[wg * 4 * 2];
                                            unsigned int hi = (unsigned int)s_tau[wg * 4 * 2 + 1];
                                            unsigned long long tau_new = (unsigned long long)hi << 32 | (unsigned long long)lo;
                                            unsigned long long _max_0 = ((tau_new) > (tau[0]) ? (tau_new) : (tau[0]));
                                            tau[0] = _max_0;
                                            {
                                                unsigned long long tq_4 = tau[0];
                                                unsigned int tau_hi_4 = (unsigned int)(tq_4 >> 32);
                                                unsigned int tau_lo_4 = (unsigned int)tq_4;
                                                unsigned int t_bits_4 = ~tau_hi_4;
                                                unsigned int l_eq_4 = tau_lo_4;
                                                unsigned int l_nan_4 = never_u;
                                                if ((tau_hi_4 & sign_mask) != 0) {
                                                    t_bits_4 = tau_hi_4 ^ sign_mask;
                                                }
                                                if (tau_hi_4 < neg_key_floor) {
                                                    t_bits_4 = ninf_bits;
                                                    l_eq_4 = zero_u;
                                                }
                                                if (tau_hi_4 == 0) {
                                                    l_nan_4 = tau_lo_4;
                                                }
                                                if (tau_hi_4 == zero_gap_key) {
                                                    l_eq_4 = zero_u;
                                                }
                                                if (tau_hi_4 > pinf_key) {
                                                    t_bits_4 = pinf_bits;
                                                    l_eq_4 = never_u;
                                                }
                                                float t_f_4 = 0.0f;
                                                t_f_4 = reinterpret_cast<float*>(&t_bits_4)[0];
                                                tau_t[0] = t_f_4;
                                                tau_leq[0] = l_eq_4;
                                                tau_lnan[0] = l_nan_4;
                                            }
                                        }
                                        if (need[1] != 0) {
                                            unsigned int lo_1 = (unsigned int)s_tau[(wg * 4 + 1) * 2];
                                            unsigned int hi_1 = (unsigned int)s_tau[(wg * 4 + 1) * 2 + 1];
                                            unsigned long long tau_new_1 = (unsigned long long)hi_1 << 32 | (unsigned long long)lo_1;
                                            unsigned long long _max_1 = ((tau_new_1) > (tau[1]) ? (tau_new_1) : (tau[1]));
                                            tau[1] = _max_1;
                                            {
                                                unsigned long long tq_5 = tau[1];
                                                unsigned int tau_hi_5 = (unsigned int)(tq_5 >> 32);
                                                unsigned int tau_lo_5 = (unsigned int)tq_5;
                                                unsigned int t_bits_5 = ~tau_hi_5;
                                                unsigned int l_eq_5 = tau_lo_5;
                                                unsigned int l_nan_5 = never_u;
                                                if ((tau_hi_5 & sign_mask) != 0) {
                                                    t_bits_5 = tau_hi_5 ^ sign_mask;
                                                }
                                                if (tau_hi_5 < neg_key_floor) {
                                                    t_bits_5 = ninf_bits;
                                                    l_eq_5 = zero_u;
                                                }
                                                if (tau_hi_5 == 0) {
                                                    l_nan_5 = tau_lo_5;
                                                }
                                                if (tau_hi_5 == zero_gap_key) {
                                                    l_eq_5 = zero_u;
                                                }
                                                if (tau_hi_5 > pinf_key) {
                                                    t_bits_5 = pinf_bits;
                                                    l_eq_5 = never_u;
                                                }
                                                float t_f_5 = 0.0f;
                                                t_f_5 = reinterpret_cast<float*>(&t_bits_5)[0];
                                                tau_t[1] = t_f_5;
                                                tau_leq[1] = l_eq_5;
                                                tau_lnan[1] = l_nan_5;
                                            }
                                        }
                                        if (need[2] != 0) {
                                            unsigned int lo_2 = (unsigned int)s_tau[(wg * 4 + 2) * 2];
                                            unsigned int hi_2 = (unsigned int)s_tau[(wg * 4 + 2) * 2 + 1];
                                            unsigned long long tau_new_2 = (unsigned long long)hi_2 << 32 | (unsigned long long)lo_2;
                                            unsigned long long _max_2 = ((tau_new_2) > (tau[2]) ? (tau_new_2) : (tau[2]));
                                            tau[2] = _max_2;
                                            {
                                                unsigned long long tq_6 = tau[2];
                                                unsigned int tau_hi_6 = (unsigned int)(tq_6 >> 32);
                                                unsigned int tau_lo_6 = (unsigned int)tq_6;
                                                unsigned int t_bits_6 = ~tau_hi_6;
                                                unsigned int l_eq_6 = tau_lo_6;
                                                unsigned int l_nan_6 = never_u;
                                                if ((tau_hi_6 & sign_mask) != 0) {
                                                    t_bits_6 = tau_hi_6 ^ sign_mask;
                                                }
                                                if (tau_hi_6 < neg_key_floor) {
                                                    t_bits_6 = ninf_bits;
                                                    l_eq_6 = zero_u;
                                                }
                                                if (tau_hi_6 == 0) {
                                                    l_nan_6 = tau_lo_6;
                                                }
                                                if (tau_hi_6 == zero_gap_key) {
                                                    l_eq_6 = zero_u;
                                                }
                                                if (tau_hi_6 > pinf_key) {
                                                    t_bits_6 = pinf_bits;
                                                    l_eq_6 = never_u;
                                                }
                                                float t_f_6 = 0.0f;
                                                t_f_6 = reinterpret_cast<float*>(&t_bits_6)[0];
                                                tau_t[2] = t_f_6;
                                                tau_leq[2] = l_eq_6;
                                                tau_lnan[2] = l_nan_6;
                                            }
                                        }
                                        if (need[3] != 0) {
                                            unsigned int lo_3 = (unsigned int)s_tau[(wg * 4 + 3) * 2];
                                            unsigned int hi_3 = (unsigned int)s_tau[(wg * 4 + 3) * 2 + 1];
                                            unsigned long long tau_new_3 = (unsigned long long)hi_3 << 32 | (unsigned long long)lo_3;
                                            unsigned long long _max_3 = ((tau_new_3) > (tau[3]) ? (tau_new_3) : (tau[3]));
                                            tau[3] = _max_3;
                                            {
                                                unsigned long long tq_7 = tau[3];
                                                unsigned int tau_hi_7 = (unsigned int)(tq_7 >> 32);
                                                unsigned int tau_lo_7 = (unsigned int)tq_7;
                                                unsigned int t_bits_7 = ~tau_hi_7;
                                                unsigned int l_eq_7 = tau_lo_7;
                                                unsigned int l_nan_7 = never_u;
                                                if ((tau_hi_7 & sign_mask) != 0) {
                                                    t_bits_7 = tau_hi_7 ^ sign_mask;
                                                }
                                                if (tau_hi_7 < neg_key_floor) {
                                                    t_bits_7 = ninf_bits;
                                                    l_eq_7 = zero_u;
                                                }
                                                if (tau_hi_7 == 0) {
                                                    l_nan_7 = tau_lo_7;
                                                }
                                                if (tau_hi_7 == zero_gap_key) {
                                                    l_eq_7 = zero_u;
                                                }
                                                if (tau_hi_7 > pinf_key) {
                                                    t_bits_7 = pinf_bits;
                                                    l_eq_7 = never_u;
                                                }
                                                float t_f_7 = 0.0f;
                                                t_f_7 = reinterpret_cast<float*>(&t_bits_7)[0];
                                                tau_t[3] = t_f_7;
                                                tau_leq[3] = l_eq_7;
                                                tau_lnan[3] = l_nan_7;
                                            }
                                        }
                                    }
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                            _phase_q_full_1 ^= 1;
                            int c_final = 0;
                            int c_finals[2];
                            c_finals[0] = 0;
                            c_finals[1] = 0;
                            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            c_finals[0] = s_count[slot_pairs[0]];
                            c_finals[1] = s_count[slot_pairs[1]];
                            int verdict = 0;
                            if (attempt_3 == 0) {
                                int fail_pair = 0;
                                int fails[2];
                                fails[0] = 0;
                                fails[1] = 0;
                                if (valid_pairs[0] != 0) {
                                    int _min_16 = ((top_k) < (visible_pairs[0]) ? (top_k) : (visible_pairs[0]));
                                    if (c_finals[0] < _min_16) {
                                        fails[0] = 1;
                                    }
                                }
                                if (valid_pairs[1] != 0) {
                                    int _min_17 = ((top_k) < (visible_pairs[1]) ? (top_k) : (visible_pairs[1]));
                                    if (c_finals[1] < _min_17) {
                                        fails[1] = 1;
                                    }
                                }
                                if (half == 0) {
                                    if (lane_0 == 0) {
                                        s_fail[slot_pairs[0]] = fails[0];
                                        s_fail[slot_pairs[1]] = fails[1];
                                    }
                                }
                                asm volatile("barrier.sync 3, 256;" ::: "memory");
                                int f_3 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                f_3 = f_3 | s_fail[4];
                                f_3 = f_3 | s_fail[5];
                                f_3 = f_3 | s_fail[6];
                                f_3 = f_3 | s_fail[7];
                                verdict = f_3;
                                if (elect_sync()) {
                                    mbarrier_arrive(verdict_bar_addr);
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            if (verdict == 0) {
                                {
                                    int c_fin = c_finals[0];
                                    int _min_18 = ((top_k) < (c_fin) ? (top_k) : (c_fin));
                                    int n_sel_j = _min_18;
                                    unsigned long long edge_j = 0;
                                    if (c_fin > top_k) {
                                        int k_rem_2 = top_k;
                                        int kept_above_2 = 0;
                                        unsigned long long prefix_2 = 0;
                                        unsigned long long edge_2 = 0;
                                        #pragma unroll 1
                                        for (int p_2 = 0; p_2 < 8; p_2++) {
                                            int shift_2 = 56 - 8 * p_2;
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            if (half == 0) {
                                                s_hist[hist_base + lane_0 * 8] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 1] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 2] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 3] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 4] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 5] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 6] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 7] = 0;
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            unsigned long long ring_2[24];
                                            #pragma unroll
                                            for (int s_pro_2 = 0; s_pro_2 < 2; s_pro_2++) {
                                                #pragma unroll
                                                for (int u_10 = 0; u_10 < 8; u_10++) {
                                                    int idx_l_4 = s_pro_2 * 512 + half * 32 + lane_0 + 64 * u_10;
                                                    ring_2[s_pro_2 * 8 + u_10] = 0;
                                                    if (idx_l_4 < c_fin) {
                                                        ring_2[s_pro_2 * 8 + u_10] = (unsigned long long)Cand[buf_pairs[0] + idx_l_4];
                                                    }
                                                }
                                            }
                                            int nbat_2 = (c_fin + 512 - 1) / 512;
                                            #pragma unroll 1
                                            for (int g_2 = 0; g_2 < nbat_2; g_2 += 3) {
                                                #pragma unroll
                                                for (int s_st_2 = 0; s_st_2 < 3; s_st_2++) {
                                                    #pragma unroll
                                                    for (int u_11 = 0; u_11 < 8; u_11++) {
                                                        int valid_p_2 = ((c_fin > (g_2 + s_st_2) * 512 + half * 32 + lane_0 + 64 * u_11) ? 1 : 0);
                                                        unsigned int bits_4 = (unsigned int)(ring_2[s_st_2 * 8 + u_11] >> 32);
                                                        int kid_1 = (int)(unsigned int)(ring_2[s_st_2 * 8 + u_11] & 4294967295);
                                                        unsigned int magnitude_4 = bits_4 & 2147483647;
                                                        unsigned int m_4 = ((magnitude_4 != 0) ? bits_4 : zero_u);
                                                        unsigned int key32_4 = (((m_4 & sign_mask) != 0) ? ~m_4 : m_4 | sign_mask);
                                                        if (magnitude_4 > nan_floor) {
                                                            key32_4 = zero_u;
                                                        }
                                                        unsigned long long key64_4 = (unsigned long long)key32_4 << 32 | (unsigned long long)(unsigned int)kid_1;
                                                        unsigned long long key_4 = key64_4;
                                                        unsigned long long key_p_2 = key_4;
                                                        unsigned long long ks_p_2 = key_p_2 >> (unsigned long long)shift_2;
                                                        int digit_p_2 = (int)(ks_p_2 & 255);
                                                        int pm_p_2 = ((ks_p_2 >> 8 == prefix_2) ? 1 : 0);
                                                        int bin_p_2 = (((valid_p_2 & pm_p_2) != 0) ? hist_base + digit_p_2 : hist_base + 1024);
                                                        atomicAdd(&s_hist[bin_p_2], 1);
                                                    }
                                                    #pragma unroll
                                                    for (int u_12 = 0; u_12 < 8; u_12++) {
                                                        int idx_l_5 = (g_2 + s_st_2 + 2) * 512 + half * 32 + lane_0 + 64 * u_12;
                                                        ring_2[(s_st_2 + 2) % 3 * 8 + u_12] = 0;
                                                        if (idx_l_5 < c_fin) {
                                                            ring_2[(s_st_2 + 2) % 3 * 8 + u_12] = (unsigned long long)Cand[buf_pairs[0] + idx_l_5];
                                                        }
                                                    }
                                                }
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            int lane_bins_2[8];
                                            int lane_sum_2 = 0;
                                            lane_bins_2[0] = s_hist[hist_base + lane_0 * 8];
                                            lane_sum_2 += lane_bins_2[0];
                                            lane_bins_2[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                            lane_sum_2 += lane_bins_2[1];
                                            lane_bins_2[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                            lane_sum_2 += lane_bins_2[2];
                                            lane_bins_2[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                            lane_sum_2 += lane_bins_2[3];
                                            lane_bins_2[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                            lane_sum_2 += lane_bins_2[4];
                                            lane_bins_2[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                            lane_sum_2 += lane_bins_2[5];
                                            lane_bins_2[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                            lane_sum_2 += lane_bins_2[6];
                                            lane_bins_2[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                            lane_sum_2 += lane_bins_2[7];
                                            int suffix_2 = lane_sum_2;
                                            int _shfl_down_10 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 1, 32);
                                            int above_part_5 = _shfl_down_10;
                                            if (lane_0 + 1 < 32) {
                                                suffix_2 += above_part_5;
                                            }
                                            int _shfl_down_11 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 2, 32);
                                            int above_part_0_2 = _shfl_down_11;
                                            if (lane_0 + 2 < 32) {
                                                suffix_2 += above_part_0_2;
                                            }
                                            int _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 4, 32);
                                            int above_part_1_2 = _shfl_down_12;
                                            if (lane_0 + 4 < 32) {
                                                suffix_2 += above_part_1_2;
                                            }
                                            int _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 8, 32);
                                            int above_part_2_2 = _shfl_down_13;
                                            if (lane_0 + 8 < 32) {
                                                suffix_2 += above_part_2_2;
                                            }
                                            int _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 16, 32);
                                            int above_part_3_2 = _shfl_down_14;
                                            if (lane_0 + 16 < 32) {
                                                suffix_2 += above_part_3_2;
                                            }
                                            int excl_2 = suffix_2 - lane_sum_2;
                                            int is_target_2 = ((excl_2 < k_rem_2 && k_rem_2 <= excl_2 + lane_sum_2) ? 1 : 0);
                                            int d_sel_2 = 0;
                                            int above_sel_2 = 0;
                                            int count_sel_2 = 0;
                                            int found_2 = 0;
                                            int cum_above_2 = excl_2;
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[7]) {
                                                    d_sel_2 = 7;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[7];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[7];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[6]) {
                                                    d_sel_2 = 6;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[6];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[6];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[5]) {
                                                    d_sel_2 = 5;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[5];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[5];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[4]) {
                                                    d_sel_2 = 4;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[4];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[4];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[3]) {
                                                    d_sel_2 = 3;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[3];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[3];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[2]) {
                                                    d_sel_2 = 2;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[2];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[2];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[1]) {
                                                    d_sel_2 = 1;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[1];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[1];
                                            if (found_2 == 0) {
                                                if (k_rem_2 <= cum_above_2 + lane_bins_2[0]) {
                                                    d_sel_2 = 0;
                                                    above_sel_2 = cum_above_2;
                                                    count_sel_2 = lane_bins_2[0];
                                                    found_2 = 1;
                                                }
                                            }
                                            cum_above_2 += lane_bins_2[0];
                                            int _warp_redux_i32_2;
                                            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_2) : "r"(is_target_2 * lane_0));
                                            int target_lane_2 = _warp_redux_i32_2;
                                            int _shfl_10 = __shfl_sync(0xFFFFFFFF, d_sel_2, target_lane_2);
                                            int digit_sel_2 = _shfl_10;
                                            int _shfl_11 = __shfl_sync(0xFFFFFFFF, above_sel_2, target_lane_2);
                                            int above_cnt_2 = _shfl_11;
                                            int _shfl_12 = __shfl_sync(0xFFFFFFFF, count_sel_2, target_lane_2);
                                            int bucket_cnt_2 = _shfl_12;
                                            k_rem_2 = k_rem_2 - above_cnt_2;
                                            kept_above_2 += above_cnt_2;
                                            prefix_2 = prefix_2 << 8 | (unsigned long long)(unsigned int)(target_lane_2 * 8 + digit_sel_2);
                                            edge_2 = prefix_2 << (unsigned long long)shift_2;
                                            if (kept_above_2 + bucket_cnt_2 <= top_k) {
                                                break;
                                            }
                                        }
                                        edge_j = edge_2;
                                    }
                                    int row_j = q0_3 + u_base + slot_pairs[0];
                                    long long row_base_j = (long long)row_j * (long long)top_k;
                                    if (valid_pairs[0] != 0) {
                                        int mid_w = (c_fin / 2 + 31) / 32 * 32;
                                        if (mid_w > c_fin) {
                                            mid_w = c_fin;
                                        }
                                        if (half == 0) {
                                            int front_w = 0;
                                            #pragma unroll 1
                                            for (int e0_2 = 0; e0_2 < mid_w; e0_2 += 256) {
                                                unsigned long long ent_wf[8];
                                                #pragma unroll
                                                for (int u_13 = 0; u_13 < 8; u_13++) {
                                                    ent_wf[u_13] = 0;
                                                    if (mid_w > e0_2 + lane_0 + 32 * u_13) {
                                                        ent_wf[u_13] = (unsigned long long)Cand[buf_pairs[0] + e0_2 + lane_0 + 32 * u_13];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_14 = 0; u_14 < 8; u_14++) {
                                                    int keep_wf = 0;
                                                    if (mid_w > e0_2 + lane_0 + 32 * u_14) {
                                                        unsigned int bits_5 = (unsigned int)(ent_wf[u_14] >> 32);
                                                        int kid_2 = (int)(unsigned int)(ent_wf[u_14] & 4294967295);
                                                        unsigned int magnitude_5 = bits_5 & 2147483647;
                                                        unsigned int m_5 = ((magnitude_5 != 0) ? bits_5 : zero_u);
                                                        unsigned int key32_5 = (((m_5 & sign_mask) != 0) ? ~m_5 : m_5 | sign_mask);
                                                        if (magnitude_5 > nan_floor) {
                                                            key32_5 = zero_u;
                                                        }
                                                        unsigned long long key64_5 = (unsigned long long)key32_5 << 32 | (unsigned long long)(unsigned int)kid_2;
                                                        unsigned long long key_5 = key64_5;
                                                        keep_wf = ((key_5 >= edge_j) ? 1 : 0);
                                                    }
                                                    unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, keep_wf != 0);
                                                    if (keep_wf != 0) {
                                                        int _popc_12 = __popc(_vote_6 & lower_lanes);
                                                        long long slot_wf = row_base_j + (long long)(front_w + _popc_12);
                                                        float score_wf = 0.0f;
                                                        unsigned int bits_6 = (unsigned int)(ent_wf[u_14] >> 32);
                                                        score_wf = reinterpret_cast<float*>(&bits_6)[0];
                                                        int kid_3 = (int)(unsigned int)(ent_wf[u_14] & 4294967295);
                                                        Indices[slot_wf] = kid_3;
                                                        Scores[slot_wf] = score_wf;
                                                    }
                                                    int _popc_13 = __popc(_vote_6);
                                                    front_w += _popc_13;
                                                }
                                            }
                                        } else {
                                            int back_w = 0;
                                            #pragma unroll 1
                                            for (int e0_3 = mid_w; e0_3 < c_fin; e0_3 += 256) {
                                                unsigned long long ent_wb[8];
                                                #pragma unroll
                                                for (int u_15 = 0; u_15 < 8; u_15++) {
                                                    ent_wb[u_15] = 0;
                                                    if (c_fin > e0_3 + lane_0 + 32 * u_15) {
                                                        ent_wb[u_15] = (unsigned long long)Cand[buf_pairs[0] + e0_3 + lane_0 + 32 * u_15];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_16 = 0; u_16 < 8; u_16++) {
                                                    int keep_wb = 0;
                                                    if (c_fin > e0_3 + lane_0 + 32 * u_16) {
                                                        unsigned int bits_7 = (unsigned int)(ent_wb[u_16] >> 32);
                                                        int kid_4 = (int)(unsigned int)(ent_wb[u_16] & 4294967295);
                                                        unsigned int magnitude_6 = bits_7 & 2147483647;
                                                        unsigned int m_6 = ((magnitude_6 != 0) ? bits_7 : zero_u);
                                                        unsigned int key32_6 = (((m_6 & sign_mask) != 0) ? ~m_6 : m_6 | sign_mask);
                                                        if (magnitude_6 > nan_floor) {
                                                            key32_6 = zero_u;
                                                        }
                                                        unsigned long long key64_6 = (unsigned long long)key32_6 << 32 | (unsigned long long)(unsigned int)kid_4;
                                                        unsigned long long key_6 = key64_6;
                                                        keep_wb = ((key_6 >= edge_j) ? 1 : 0);
                                                    }
                                                    unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, keep_wb != 0);
                                                    if (keep_wb != 0) {
                                                        int _popc_14 = __popc(_vote_7 & lower_lanes);
                                                        long long slot_wb = row_base_j + (long long)(n_sel_j - 1 - (back_w + _popc_14));
                                                        float score_wb = 0.0f;
                                                        unsigned int bits_8 = (unsigned int)(ent_wb[u_16] >> 32);
                                                        score_wb = reinterpret_cast<float*>(&bits_8)[0];
                                                        int kid_5 = (int)(unsigned int)(ent_wb[u_16] & 4294967295);
                                                        Indices[slot_wb] = kid_5;
                                                        Scores[slot_wb] = score_wb;
                                                    }
                                                    int _popc_15 = __popc(_vote_7);
                                                    back_w += _popc_15;
                                                }
                                            }
                                            #pragma unroll 1
                                            for (int pad_w = n_sel_j + lane_0; pad_w < top_k; pad_w += 32) {
                                                long long pad_slot_w = row_base_j + (long long)pad_w;
                                                Indices[pad_slot_w] = -1;
                                                Scores[pad_slot_w] = -CUDART_INF_F;
                                            }
                                        }
                                    }
                                }
                                {
                                    int c_fin_1 = c_finals[1];
                                    int _min_19 = ((top_k) < (c_fin_1) ? (top_k) : (c_fin_1));
                                    int n_sel_j_1 = _min_19;
                                    unsigned long long edge_j_1 = 0;
                                    if (c_fin_1 > top_k) {
                                        int k_rem_3 = top_k;
                                        int kept_above_3 = 0;
                                        unsigned long long prefix_3 = 0;
                                        unsigned long long edge_3 = 0;
                                        #pragma unroll 1
                                        for (int p_3 = 0; p_3 < 8; p_3++) {
                                            int shift_3 = 56 - 8 * p_3;
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            if (half == 0) {
                                                s_hist[hist_base + lane_0 * 8] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 1] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 2] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 3] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 4] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 5] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 6] = 0;
                                                s_hist[hist_base + lane_0 * 8 + 7] = 0;
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            unsigned long long ring_3[24];
                                            #pragma unroll
                                            for (int s_pro_3 = 0; s_pro_3 < 2; s_pro_3++) {
                                                #pragma unroll
                                                for (int u_17 = 0; u_17 < 8; u_17++) {
                                                    int idx_l_6 = s_pro_3 * 512 + half * 32 + lane_0 + 64 * u_17;
                                                    ring_3[s_pro_3 * 8 + u_17] = 0;
                                                    if (idx_l_6 < c_fin_1) {
                                                        ring_3[s_pro_3 * 8 + u_17] = (unsigned long long)Cand[buf_pairs[1] + idx_l_6];
                                                    }
                                                }
                                            }
                                            int nbat_3 = (c_fin_1 + 512 - 1) / 512;
                                            #pragma unroll 1
                                            for (int g_3 = 0; g_3 < nbat_3; g_3 += 3) {
                                                #pragma unroll
                                                for (int s_st_3 = 0; s_st_3 < 3; s_st_3++) {
                                                    #pragma unroll
                                                    for (int u_18 = 0; u_18 < 8; u_18++) {
                                                        int valid_p_3 = ((c_fin_1 > (g_3 + s_st_3) * 512 + half * 32 + lane_0 + 64 * u_18) ? 1 : 0);
                                                        unsigned int bits_9 = (unsigned int)(ring_3[s_st_3 * 8 + u_18] >> 32);
                                                        int kid_6 = (int)(unsigned int)(ring_3[s_st_3 * 8 + u_18] & 4294967295);
                                                        unsigned int magnitude_7 = bits_9 & 2147483647;
                                                        unsigned int m_7 = ((magnitude_7 != 0) ? bits_9 : zero_u);
                                                        unsigned int key32_7 = (((m_7 & sign_mask) != 0) ? ~m_7 : m_7 | sign_mask);
                                                        if (magnitude_7 > nan_floor) {
                                                            key32_7 = zero_u;
                                                        }
                                                        unsigned long long key64_7 = (unsigned long long)key32_7 << 32 | (unsigned long long)(unsigned int)kid_6;
                                                        unsigned long long key_7 = key64_7;
                                                        unsigned long long key_p_3 = key_7;
                                                        unsigned long long ks_p_3 = key_p_3 >> (unsigned long long)shift_3;
                                                        int digit_p_3 = (int)(ks_p_3 & 255);
                                                        int pm_p_3 = ((ks_p_3 >> 8 == prefix_3) ? 1 : 0);
                                                        int bin_p_3 = (((valid_p_3 & pm_p_3) != 0) ? hist_base + digit_p_3 : hist_base + 1024);
                                                        atomicAdd(&s_hist[bin_p_3], 1);
                                                    }
                                                    #pragma unroll
                                                    for (int u_19 = 0; u_19 < 8; u_19++) {
                                                        int idx_l_7 = (g_3 + s_st_3 + 2) * 512 + half * 32 + lane_0 + 64 * u_19;
                                                        ring_3[(s_st_3 + 2) % 3 * 8 + u_19] = 0;
                                                        if (idx_l_7 < c_fin_1) {
                                                            ring_3[(s_st_3 + 2) % 3 * 8 + u_19] = (unsigned long long)Cand[buf_pairs[1] + idx_l_7];
                                                        }
                                                    }
                                                }
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                            int lane_bins_3[8];
                                            int lane_sum_3 = 0;
                                            lane_bins_3[0] = s_hist[hist_base + lane_0 * 8];
                                            lane_sum_3 += lane_bins_3[0];
                                            lane_bins_3[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                            lane_sum_3 += lane_bins_3[1];
                                            lane_bins_3[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                            lane_sum_3 += lane_bins_3[2];
                                            lane_bins_3[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                            lane_sum_3 += lane_bins_3[3];
                                            lane_bins_3[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                            lane_sum_3 += lane_bins_3[4];
                                            lane_bins_3[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                            lane_sum_3 += lane_bins_3[5];
                                            lane_bins_3[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                            lane_sum_3 += lane_bins_3[6];
                                            lane_bins_3[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                            lane_sum_3 += lane_bins_3[7];
                                            int suffix_3 = lane_sum_3;
                                            int _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 1, 32);
                                            int above_part_6 = _shfl_down_15;
                                            if (lane_0 + 1 < 32) {
                                                suffix_3 += above_part_6;
                                            }
                                            int _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 2, 32);
                                            int above_part_0_3 = _shfl_down_16;
                                            if (lane_0 + 2 < 32) {
                                                suffix_3 += above_part_0_3;
                                            }
                                            int _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 4, 32);
                                            int above_part_1_3 = _shfl_down_17;
                                            if (lane_0 + 4 < 32) {
                                                suffix_3 += above_part_1_3;
                                            }
                                            int _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 8, 32);
                                            int above_part_2_3 = _shfl_down_18;
                                            if (lane_0 + 8 < 32) {
                                                suffix_3 += above_part_2_3;
                                            }
                                            int _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 16, 32);
                                            int above_part_3_3 = _shfl_down_19;
                                            if (lane_0 + 16 < 32) {
                                                suffix_3 += above_part_3_3;
                                            }
                                            int excl_3 = suffix_3 - lane_sum_3;
                                            int is_target_3 = ((excl_3 < k_rem_3 && k_rem_3 <= excl_3 + lane_sum_3) ? 1 : 0);
                                            int d_sel_3 = 0;
                                            int above_sel_3 = 0;
                                            int count_sel_3 = 0;
                                            int found_3 = 0;
                                            int cum_above_3 = excl_3;
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[7]) {
                                                    d_sel_3 = 7;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[7];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[7];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[6]) {
                                                    d_sel_3 = 6;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[6];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[6];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[5]) {
                                                    d_sel_3 = 5;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[5];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[5];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[4]) {
                                                    d_sel_3 = 4;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[4];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[4];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[3]) {
                                                    d_sel_3 = 3;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[3];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[3];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[2]) {
                                                    d_sel_3 = 2;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[2];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[2];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[1]) {
                                                    d_sel_3 = 1;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[1];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[1];
                                            if (found_3 == 0) {
                                                if (k_rem_3 <= cum_above_3 + lane_bins_3[0]) {
                                                    d_sel_3 = 0;
                                                    above_sel_3 = cum_above_3;
                                                    count_sel_3 = lane_bins_3[0];
                                                    found_3 = 1;
                                                }
                                            }
                                            cum_above_3 += lane_bins_3[0];
                                            int _warp_redux_i32_3;
                                            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_3) : "r"(is_target_3 * lane_0));
                                            int target_lane_3 = _warp_redux_i32_3;
                                            int _shfl_13 = __shfl_sync(0xFFFFFFFF, d_sel_3, target_lane_3);
                                            int digit_sel_3 = _shfl_13;
                                            int _shfl_14 = __shfl_sync(0xFFFFFFFF, above_sel_3, target_lane_3);
                                            int above_cnt_3 = _shfl_14;
                                            int _shfl_15 = __shfl_sync(0xFFFFFFFF, count_sel_3, target_lane_3);
                                            int bucket_cnt_3 = _shfl_15;
                                            k_rem_3 = k_rem_3 - above_cnt_3;
                                            kept_above_3 += above_cnt_3;
                                            prefix_3 = prefix_3 << 8 | (unsigned long long)(unsigned int)(target_lane_3 * 8 + digit_sel_3);
                                            edge_3 = prefix_3 << (unsigned long long)shift_3;
                                            if (kept_above_3 + bucket_cnt_3 <= top_k) {
                                                break;
                                            }
                                        }
                                        edge_j_1 = edge_3;
                                    }
                                    int row_j_1 = q0_3 + u_base + slot_pairs[1];
                                    long long row_base_j_1 = (long long)row_j_1 * (long long)top_k;
                                    if (valid_pairs[1] != 0) {
                                        int mid_w_1 = (c_fin_1 / 2 + 31) / 32 * 32;
                                        if (mid_w_1 > c_fin_1) {
                                            mid_w_1 = c_fin_1;
                                        }
                                        if (half == 0) {
                                            int front_w_1 = 0;
                                            #pragma unroll 1
                                            for (int e0_4 = 0; e0_4 < mid_w_1; e0_4 += 256) {
                                                unsigned long long ent_wf_1[8];
                                                #pragma unroll
                                                for (int u_20 = 0; u_20 < 8; u_20++) {
                                                    ent_wf_1[u_20] = 0;
                                                    if (mid_w_1 > e0_4 + lane_0 + 32 * u_20) {
                                                        ent_wf_1[u_20] = (unsigned long long)Cand[buf_pairs[1] + e0_4 + lane_0 + 32 * u_20];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_21 = 0; u_21 < 8; u_21++) {
                                                    int keep_wf_1 = 0;
                                                    if (mid_w_1 > e0_4 + lane_0 + 32 * u_21) {
                                                        unsigned int bits_10 = (unsigned int)(ent_wf_1[u_21] >> 32);
                                                        int kid_7 = (int)(unsigned int)(ent_wf_1[u_21] & 4294967295);
                                                        unsigned int magnitude_8 = bits_10 & 2147483647;
                                                        unsigned int m_8 = ((magnitude_8 != 0) ? bits_10 : zero_u);
                                                        unsigned int key32_8 = (((m_8 & sign_mask) != 0) ? ~m_8 : m_8 | sign_mask);
                                                        if (magnitude_8 > nan_floor) {
                                                            key32_8 = zero_u;
                                                        }
                                                        unsigned long long key64_8 = (unsigned long long)key32_8 << 32 | (unsigned long long)(unsigned int)kid_7;
                                                        unsigned long long key_8 = key64_8;
                                                        keep_wf_1 = ((key_8 >= edge_j_1) ? 1 : 0);
                                                    }
                                                    unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, keep_wf_1 != 0);
                                                    if (keep_wf_1 != 0) {
                                                        int _popc_16 = __popc(_vote_8 & lower_lanes);
                                                        long long slot_wf_1 = row_base_j_1 + (long long)(front_w_1 + _popc_16);
                                                        float score_wf_1 = 0.0f;
                                                        unsigned int bits_11 = (unsigned int)(ent_wf_1[u_21] >> 32);
                                                        score_wf_1 = reinterpret_cast<float*>(&bits_11)[0];
                                                        int kid_8 = (int)(unsigned int)(ent_wf_1[u_21] & 4294967295);
                                                        Indices[slot_wf_1] = kid_8;
                                                        Scores[slot_wf_1] = score_wf_1;
                                                    }
                                                    int _popc_17 = __popc(_vote_8);
                                                    front_w_1 += _popc_17;
                                                }
                                            }
                                        } else {
                                            int back_w_1 = 0;
                                            #pragma unroll 1
                                            for (int e0_5 = mid_w_1; e0_5 < c_fin_1; e0_5 += 256) {
                                                unsigned long long ent_wb_1[8];
                                                #pragma unroll
                                                for (int u_22 = 0; u_22 < 8; u_22++) {
                                                    ent_wb_1[u_22] = 0;
                                                    if (c_fin_1 > e0_5 + lane_0 + 32 * u_22) {
                                                        ent_wb_1[u_22] = (unsigned long long)Cand[buf_pairs[1] + e0_5 + lane_0 + 32 * u_22];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_23 = 0; u_23 < 8; u_23++) {
                                                    int keep_wb_1 = 0;
                                                    if (c_fin_1 > e0_5 + lane_0 + 32 * u_23) {
                                                        unsigned int bits_12 = (unsigned int)(ent_wb_1[u_23] >> 32);
                                                        int kid_9 = (int)(unsigned int)(ent_wb_1[u_23] & 4294967295);
                                                        unsigned int magnitude_9 = bits_12 & 2147483647;
                                                        unsigned int m_9 = ((magnitude_9 != 0) ? bits_12 : zero_u);
                                                        unsigned int key32_9 = (((m_9 & sign_mask) != 0) ? ~m_9 : m_9 | sign_mask);
                                                        if (magnitude_9 > nan_floor) {
                                                            key32_9 = zero_u;
                                                        }
                                                        unsigned long long key64_9 = (unsigned long long)key32_9 << 32 | (unsigned long long)(unsigned int)kid_9;
                                                        unsigned long long key_9 = key64_9;
                                                        keep_wb_1 = ((key_9 >= edge_j_1) ? 1 : 0);
                                                    }
                                                    unsigned int _vote_9 = __ballot_sync(0xFFFFFFFF, keep_wb_1 != 0);
                                                    if (keep_wb_1 != 0) {
                                                        int _popc_18 = __popc(_vote_9 & lower_lanes);
                                                        long long slot_wb_1 = row_base_j_1 + (long long)(n_sel_j_1 - 1 - (back_w_1 + _popc_18));
                                                        float score_wb_1 = 0.0f;
                                                        unsigned int bits_13 = (unsigned int)(ent_wb_1[u_23] >> 32);
                                                        score_wb_1 = reinterpret_cast<float*>(&bits_13)[0];
                                                        int kid_10 = (int)(unsigned int)(ent_wb_1[u_23] & 4294967295);
                                                        Indices[slot_wb_1] = kid_10;
                                                        Scores[slot_wb_1] = score_wb_1;
                                                    }
                                                    int _popc_19 = __popc(_vote_9);
                                                    back_w_1 += _popc_19;
                                                }
                                            }
                                            #pragma unroll 1
                                            for (int pad_w_1 = n_sel_j_1 + lane_0; pad_w_1 < top_k; pad_w_1 += 32) {
                                                long long pad_slot_w_1 = row_base_j_1 + (long long)pad_w_1;
                                                Indices[pad_slot_w_1] = -1;
                                                Scores[pad_slot_w_1] = -CUDART_INF_F;
                                            }
                                        }
                                    }
                                }
                            }
                            if (verdict == 0) {
                                break;
                            }
                        }
                    }
                }
                unit_base += nb_3;
            }
            asm volatile("barrier.sync 3, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }

    // Cleanup
}

} // extern "C"
