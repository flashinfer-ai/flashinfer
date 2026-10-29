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
#define NUM_TMEM_LAP_PIPE_STAGES 1
#define NUM_VERDICT_PIPE_STAGES 1
#define NUM_ACK_PIPE_STAGES 1
#define NUM_SCORE_PIPE_STAGES 3
#define NUM_REQ_PIPE_STAGES 4
#define NUM_DONE_PIPE_STAGES 1
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
#define SMEM_S_HIST_STAGE_BYTES 8320
#define SMEM_S_HIST_STRIDE 8320
#define SMEM_S_COUNT_OFF 206976
#define SMEM_S_COUNT_STAGE_BYTES 64
#define SMEM_S_COUNT_STRIDE 64
#define SMEM_S_TAU_OFF 207040
#define SMEM_S_TAU_STAGE_BYTES 128
#define SMEM_S_TAU_STRIDE 128
#define SMEM_S_FAIL_OFF 207168
#define SMEM_S_FAIL_STAGE_BYTES 32
#define SMEM_S_FAIL_STRIDE 32
#define SMEM_S_TK_OFF 221184
#define SMEM_S_TK_STAGE_BYTES 16
#define SMEM_S_TK_STRIDE 16
#define SMEM_S_WSC_OFF 207232
#define SMEM_S_WSC_STAGE_BYTES 512
#define SMEM_S_WSC_STRIDE 512
#define SMEM_S_SCORES_OFF 208256
#define SMEM_S_SCORES_STAGE_BYTES 4096
#define SMEM_S_SCORES_STRIDE 4096
#define SMEM_S_REQ_OFF 220544
#define SMEM_S_REQ_STAGE_BYTES 640
#define SMEM_S_REQ_STRIDE 640
#define SMEM_TOTAL 221312
#ifndef TILE_UNROLL
#define TILE_UNROLL 2
#endif
#define LAUNCH_MIN_BLOCKS 1

// Shared by every build.json entry that lists this translation unit in its sources: the entry's
// compile definitions select the tile unroll (-DTILE_UNROLL=<n>; the define above is guarded) and
// the kernel symbol (-DCAKE_PROGRAM_KERNEL=<its own symbol>; every library of the family exports
// a distinct kernel); the defaults name this file's own program.
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_dsa_indexer_topk_808a62a6315ae38034c6
#endif

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap W, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, long long* __restrict__ q_offsets, int* __restrict__ Indices, float* __restrict__ Scores, long long* __restrict__ Cand, int num_segments, int top_k, int ratio, int has_offsets, int cand_cap, int first_cap, int sample_tiles_max, int sample_shift_permille, int far_sample, int check_period, int grid_ctas, float softmax_scale, int n_split)
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
    #define verdict_ack_addr (mbar_base + 120)
    #define scores_full_addr (mbar_base + 128)
    #define scores_empty_addr (mbar_base + 152)
    #define comp_req_addr (mbar_base + 176)
    #define comp_done_addr (mbar_base + 208)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_w_addr = smem + 66560;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_k_addr = smem + 67584;
    int* s_hist = reinterpret_cast<int*>(smem_raw + 198656);
    const int s_hist_addr = smem + 198656;
    int* s_count = reinterpret_cast<int*>(smem_raw + 206976);
    const int s_count_addr = smem + 206976;
    int* s_tau = reinterpret_cast<int*>(smem_raw + 207040);
    const int s_tau_addr = smem + 207040;
    int* s_fail = reinterpret_cast<int*>(smem_raw + 207168);
    const int s_fail_addr = smem + 207168;
    int* s_tk = reinterpret_cast<int*>(smem_raw + 221184);
    const int s_tk_addr = smem + 221184;
    float* s_wsc = reinterpret_cast<float*>(smem_raw + 207232);
    const int s_wsc_addr = smem + 207232;
    unsigned int* s_scores = reinterpret_cast<unsigned int*>(smem_raw + 208256);
    const int s_scores_addr = smem + 208256;
    int* s_req = reinterpret_cast<int*>(smem_raw + 220544);
    const int s_req_addr = smem + 220544;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

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
            // --- pipeline 'verdict_pipe' ---
            // verdict_bar: 1 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            // --- pipeline 'ack_pipe' ---
            // verdict_ack: 1 barriers, init_count=3
            mbarrier_init(smem + 120, 3);
            // --- pipeline 'score_pipe' ---
            // scores_full: 3 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            mbarrier_init(smem + 144, 256);
            // scores_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // --- pipeline 'req_pipe' ---
            // comp_req: 4 barriers, init_count=4
            mbarrier_init(smem + 176, 4);
            mbarrier_init(smem + 184, 4);
            mbarrier_init(smem + 192, 4);
            mbarrier_init(smem + 200, 4);
            // --- pipeline 'done_pipe' ---
            // comp_done: 1 barriers, init_count=8
            mbarrier_init(smem + 208, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 10) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            int n_total_0 = 0;
            #pragma unroll 1
            for (int s_t = 0; s_t < num_segments; s_t++) {
                int lq_t = cu_seqlens_q[s_t + 1] - cu_seqlens_q[s_t];
                int nb_t = (lq_t + 8 - 1) / 8;
                n_total_0 += nb_t;
            }
            int tk_limit_0 = n_total_0 + grid_ctas - 1;
            s_tk[2] = tk_limit_0;
            s_tk[3] = n_total_0;
            long long tk_off_0 = (long long)grid_ctas * (long long)(8 * cand_cap);
            if (n_total_0 <= grid_ctas) {
                s_tk[0] = (int)bid;
                s_tk[1] = n_total_0;
            } else {
                uint32_t _atomic_fetch_inc_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_fetch_inc_0) : "l"(reinterpret_cast<unsigned int*>(Cand + tk_off_0)), "r"(static_cast<uint32_t>((unsigned int)tk_limit_0)) : "memory");
                s_tk[0] = (int)_atomic_fetch_inc_0;
            }
        }
    }
    __syncthreads();

    // ---- Role: load_q ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // load_q_main
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_verdict_bar = 0;
            if (elect_sync()) {
                int unit_base_q = 0;
                unsigned int vphase_q = 0;
                int tk_q = s_tk[0];
                int tkp_q = 1;
                #pragma unroll 1
                for (int seg = 0; seg < num_segments; seg++) {
                    int q0 = cu_seqlens_q[seg];
                    int lq = cu_seqlens_q[seg + 1] - q0;
                    int nb = (lq + 8 - 1) / 8;
                    int cnt_q = nb;
#if __CUDA_ARCH__ == 1000
#else
                    int q0_e = cu_seqlens_q[seg];
                    int lq_e = cu_seqlens_q[seg + 1] - q0_e;
                    int lk_e = cu_seqlens_k[seg + 1] - cu_seqlens_k[seg];
                    long long off_e = 0;
                    if (has_offsets != 0) {
                        off_e = q_offsets[seg];
                    } else if (ratio == 1) {
                        off_e = (long long)lk_e - (long long)lq_e;
                    }
                    int last_e = ((7) < (lq_e - 1) ? (7) : (lq_e - 1));
                    long long num_e = off_e + (long long)last_e + 1;
                    int vis_e = 0;
                    if (num_e > 0) {
                        long long quot_e = num_e / (long long)ratio;
                        long long lk64_e = (long long)lk_e;
                        long long bnd_e = ((quot_e < lk64_e) ? quot_e : lk64_e);
                        vis_e = (int)bnd_e;
                    }
                    int tiles_e = (vis_e + 128 - 1) / 128;
                    int t_lo_q = tiles_e;
                    int q0_e_0 = cu_seqlens_q[seg];
                    int lq_e_0 = cu_seqlens_q[seg + 1] - q0_e_0;
                    int lk_e_0 = cu_seqlens_k[seg + 1] - cu_seqlens_k[seg];
                    long long off_e_0 = 0;
                    if (has_offsets != 0) {
                        off_e_0 = q_offsets[seg];
                    } else if (ratio == 1) {
                        off_e_0 = (long long)lk_e_0 - (long long)lq_e_0;
                    }
                    int last_e_0 = (((nb - 1) * 8 + 7) < (lq_e_0 - 1) ? ((nb - 1) * 8 + 7) : (lq_e_0 - 1));
                    long long num_e_0 = off_e_0 + (long long)last_e_0 + 1;
                    int vis_e_0 = 0;
                    if (num_e_0 > 0) {
                        long long quot_e_1 = num_e_0 / (long long)ratio;
                        long long lk64_e_1 = (long long)lk_e_0;
                        long long bnd_e_1 = ((quot_e_1 < lk64_e_1) ? quot_e_1 : lk64_e_1);
                        vis_e_0 = (int)bnd_e_1;
                    }
                    int tiles_e_0 = (vis_e_0 + 128 - 1) / 128;
                    int t_hi_q = tiles_e_0;
                    int near_eq_q = ((t_hi_q * 4 <= t_lo_q * 5) ? 1 : 0);
#endif
                    #pragma unroll 1
                    for (int rnd = 0; rnd < cnt_q + 1; rnd++) {
                        int unit = tk_q;
                        if (unit >= unit_base_q + cnt_q) {
                            break;
                        }
                        tkp_q ^= 1;
                        int _min_0 = ((unit - unit_base_q) < (unit_base_q + nb - 1 - unit) ? (unit - unit_base_q) : (unit_base_q + nb - 1 - unit));
                        if (_min_0 >= 0) {
                            int blk = nb - 1 - (unit - unit_base_q);
#if __CUDA_ARCH__ == 1000
#else
                            if (sample_tiles_max != 0) {
                                if (near_eq_q != 0) {
                                    blk = unit - unit_base_q;
                                }
                            }
#endif
                            #pragma unroll 1
                            for (int attempt = 0; attempt < 2; attempt++) {
                                mbarrier_wait(q_empty_addr, _phase_q_empty);
                                tma_3d_gmem2smem(smem_q_addr + 0 * 65536, (&Q), 0, (q0 + blk * 8) * 32, 0, q_full_addr);
                                tma_2d_gmem2smem(smem_w_addr + 0 * 1024, (&W), 0, q0 + blk * 8, q_full_addr);
                                mbarrier_arrive_expect_tx(q_full_addr, 66560);
                                _phase_q_empty ^= 1;
                                if (attempt == 0) {
                                    mbarrier_wait(verdict_bar_addr, _phase_verdict_bar);
                                    _phase_verdict_bar ^= 1;
                                    int f = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                    f = f | s_fail[4];
                                    f = f | s_fail[5];
                                    f = f | s_fail[6];
                                    f = f | s_fail[7];
                                    int vword_q = f;
                                    tk_q = s_tk[tkp_q ^ 1];
                                    mbarrier_arrive(verdict_ack_addr);
                                    if (vword_q == 0) {
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // load_k_main
            unsigned int load_k_stage = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_verdict_bar_1 = 0;
            if (elect_sync()) {
                int unit_base_k = 0;
                unsigned int vphase_k = 0;
                int tk_k = s_tk[0];
                int tkp_k = 1;
                #pragma unroll 1
                for (int seg_1 = 0; seg_1 < num_segments; seg_1++) {
                    int q0_1 = cu_seqlens_q[seg_1];
                    int lq_1 = cu_seqlens_q[seg_1 + 1] - q0_1;
                    int k0 = cu_seqlens_k[seg_1];
                    int nb_1 = (lq_1 + 8 - 1) / 8;
                    int cnt_k = nb_1;
#if __CUDA_ARCH__ == 1000
#else
                    int q0_e_1 = cu_seqlens_q[seg_1];
                    int lq_e_1 = cu_seqlens_q[seg_1 + 1] - q0_e_1;
                    int lk_e_1 = cu_seqlens_k[seg_1 + 1] - cu_seqlens_k[seg_1];
                    long long off_e_1 = 0;
                    if (has_offsets != 0) {
                        off_e_1 = q_offsets[seg_1];
                    } else if (ratio == 1) {
                        off_e_1 = (long long)lk_e_1 - (long long)lq_e_1;
                    }
                    int last_e_1 = ((7) < (lq_e_1 - 1) ? (7) : (lq_e_1 - 1));
                    long long num_e_1 = off_e_1 + (long long)last_e_1 + 1;
                    int vis_e_1 = 0;
                    if (num_e_1 > 0) {
                        long long quot_e_2 = num_e_1 / (long long)ratio;
                        long long lk64_e_2 = (long long)lk_e_1;
                        long long bnd_e_2 = ((quot_e_2 < lk64_e_2) ? quot_e_2 : lk64_e_2);
                        vis_e_1 = (int)bnd_e_2;
                    }
                    int tiles_e_1 = (vis_e_1 + 128 - 1) / 128;
                    int t_lo_k = tiles_e_1;
                    int q0_e_1_1 = cu_seqlens_q[seg_1];
                    int lq_e_1_1 = cu_seqlens_q[seg_1 + 1] - q0_e_1_1;
                    int lk_e_1_1 = cu_seqlens_k[seg_1 + 1] - cu_seqlens_k[seg_1];
                    long long off_e_1_1 = 0;
                    if (has_offsets != 0) {
                        off_e_1_1 = q_offsets[seg_1];
                    } else if (ratio == 1) {
                        off_e_1_1 = (long long)lk_e_1_1 - (long long)lq_e_1_1;
                    }
                    int last_e_1_1 = (((nb_1 - 1) * 8 + 7) < (lq_e_1_1 - 1) ? ((nb_1 - 1) * 8 + 7) : (lq_e_1_1 - 1));
                    long long num_e_1_1 = off_e_1_1 + (long long)last_e_1_1 + 1;
                    int vis_e_1_1 = 0;
                    if (num_e_1_1 > 0) {
                        long long quot_e_3 = num_e_1_1 / (long long)ratio;
                        long long lk64_e_3 = (long long)lk_e_1_1;
                        long long bnd_e_3 = ((quot_e_3 < lk64_e_3) ? quot_e_3 : lk64_e_3);
                        vis_e_1_1 = (int)bnd_e_3;
                    }
                    int tiles_e_1_1 = (vis_e_1_1 + 128 - 1) / 128;
                    int t_hi_k = tiles_e_1_1;
                    int near_eq_k = ((t_hi_k * 4 <= t_lo_k * 5) ? 1 : 0);
#endif
                    #pragma unroll 1
                    for (int rnd_1 = 0; rnd_1 < cnt_k + 1; rnd_1++) {
                        int unit_1 = tk_k;
                        if (unit_1 >= unit_base_k + cnt_k) {
                            break;
                        }
                        tkp_k ^= 1;
                        int _min_0_1 = ((unit_1 - unit_base_k) < (unit_base_k + nb_1 - 1 - unit_1) ? (unit_1 - unit_base_k) : (unit_base_k + nb_1 - 1 - unit_1));
                        if (_min_0_1 >= 0) {
                            int blk_1 = nb_1 - 1 - (unit_1 - unit_base_k);
#if __CUDA_ARCH__ == 1000
#else
                            if (sample_tiles_max != 0) {
                                if (near_eq_k != 0) {
                                    blk_1 = unit_1 - unit_base_k;
                                }
                            }
#endif
                            int q0_0 = cu_seqlens_q[seg_1];
                            int lq_0 = cu_seqlens_q[seg_1 + 1] - q0_0;
                            int lk = cu_seqlens_k[seg_1 + 1] - cu_seqlens_k[seg_1];
                            long long off = 0;
                            if (has_offsets != 0) {
                                off = q_offsets[seg_1];
                            } else if (ratio == 1) {
                                off = (long long)lk - (long long)lq_0;
                            }
                            long long off_1 = off;
                            int last_u = ((blk_1 * 8 + 7) < (lq_0 - 1) ? (blk_1 * 8 + 7) : (lq_0 - 1));
                            long long num = off_1 + (long long)last_u + 1;
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
                            int max_tiles = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
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
                                        int cap_fit = ((max_tiles) < (8) ? (max_tiles) : (8));
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
#if __CUDA_ARCH__ == 1000
                                    tile = n_tiles_3 - 1 - d_k;
#endif
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
                                    mbarrier_wait(verdict_bar_addr, _phase_verdict_bar_1);
                                    _phase_verdict_bar_1 ^= 1;
                                    int f_1 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                    f_1 = f_1 | s_fail[4];
                                    f_1 = f_1 | s_fail[5];
                                    f_1 = f_1 | s_fail[6];
                                    f_1 = f_1 | s_fail[7];
                                    int vword_k = f_1;
                                    tk_k = s_tk[tkp_k ^ 1];
                                    mbarrier_arrive(verdict_ack_addr);
                                    if (vword_k == 0) {
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // mma_main
            unsigned int mma_k_stage = 0;
            unsigned int mma_tmem_stage = 0;
            int unit_base_m = 0;
            unsigned int vphase_m = 0;
            int tk_m = 0;
            if (lane == 0) {
                tk_m = s_tk[0];
            }
            int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, tk_m, 0);
            tk_m = _shfl_sync_0;
            int tkp_m = 1;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_umma_empty = 1;
            unsigned int _phase_verdict_bar_2 = 0;
            #pragma unroll 1
            for (int seg_2 = 0; seg_2 < num_segments; seg_2++) {
                int q0_2 = cu_seqlens_q[seg_2];
                int lq_2 = cu_seqlens_q[seg_2 + 1] - q0_2;
                int nb_2 = (lq_2 + 8 - 1) / 8;
                int cnt_m = nb_2;
#if __CUDA_ARCH__ == 1000
#else
                int q0_e_2 = cu_seqlens_q[seg_2];
                int lq_e_2 = cu_seqlens_q[seg_2 + 1] - q0_e_2;
                int lk_e_2 = cu_seqlens_k[seg_2 + 1] - cu_seqlens_k[seg_2];
                long long off_e_2 = 0;
                if (has_offsets != 0) {
                    off_e_2 = q_offsets[seg_2];
                } else if (ratio == 1) {
                    off_e_2 = (long long)lk_e_2 - (long long)lq_e_2;
                }
                int last_e_2 = ((7) < (lq_e_2 - 1) ? (7) : (lq_e_2 - 1));
                long long num_e_2 = off_e_2 + (long long)last_e_2 + 1;
                int vis_e_2 = 0;
                if (num_e_2 > 0) {
                    long long quot_e_4 = num_e_2 / (long long)ratio;
                    long long lk64_e_4 = (long long)lk_e_2;
                    long long bnd_e_4 = ((quot_e_4 < lk64_e_4) ? quot_e_4 : lk64_e_4);
                    vis_e_2 = (int)bnd_e_4;
                }
                int tiles_e_2 = (vis_e_2 + 128 - 1) / 128;
                int t_lo_m = tiles_e_2;
                int q0_e_2_1 = cu_seqlens_q[seg_2];
                int lq_e_2_1 = cu_seqlens_q[seg_2 + 1] - q0_e_2_1;
                int lk_e_2_1 = cu_seqlens_k[seg_2 + 1] - cu_seqlens_k[seg_2];
                long long off_e_2_1 = 0;
                if (has_offsets != 0) {
                    off_e_2_1 = q_offsets[seg_2];
                } else if (ratio == 1) {
                    off_e_2_1 = (long long)lk_e_2_1 - (long long)lq_e_2_1;
                }
                int last_e_2_1 = (((nb_2 - 1) * 8 + 7) < (lq_e_2_1 - 1) ? ((nb_2 - 1) * 8 + 7) : (lq_e_2_1 - 1));
                long long num_e_2_1 = off_e_2_1 + (long long)last_e_2_1 + 1;
                int vis_e_2_1 = 0;
                if (num_e_2_1 > 0) {
                    long long quot_e_5 = num_e_2_1 / (long long)ratio;
                    long long lk64_e_5 = (long long)lk_e_2_1;
                    long long bnd_e_5 = ((quot_e_5 < lk64_e_5) ? quot_e_5 : lk64_e_5);
                    vis_e_2_1 = (int)bnd_e_5;
                }
                int tiles_e_2_1 = (vis_e_2_1 + 128 - 1) / 128;
                int t_hi_m = tiles_e_2_1;
                int near_eq_m = ((t_hi_m * 4 <= t_lo_m * 5) ? 1 : 0);
#endif
                #pragma unroll 1
                for (int rnd_2 = 0; rnd_2 < cnt_m + 1; rnd_2++) {
                    int unit_2 = tk_m;
                    if (unit_2 >= unit_base_m + cnt_m) {
                        break;
                    }
                    tkp_m ^= 1;
                    int _min_1 = ((unit_2 - unit_base_m) < (unit_base_m + nb_2 - 1 - unit_2) ? (unit_2 - unit_base_m) : (unit_base_m + nb_2 - 1 - unit_2));
                    if (_min_1 >= 0) {
                        int blk_2 = nb_2 - 1 - (unit_2 - unit_base_m);
#if __CUDA_ARCH__ == 1000
#else
                        if (sample_tiles_max != 0) {
                            if (near_eq_m != 0) {
                                blk_2 = unit_2 - unit_base_m;
                            }
                        }
#endif
                        int q0_1_1 = cu_seqlens_q[seg_2];
                        int lq_2_1 = cu_seqlens_q[seg_2 + 1] - q0_1_1;
                        int lk_1 = cu_seqlens_k[seg_2 + 1] - cu_seqlens_k[seg_2];
                        long long off_2 = 0;
                        if (has_offsets != 0) {
                            off_2 = q_offsets[seg_2];
                        } else if (ratio == 1) {
                            off_2 = (long long)lk_1 - (long long)lq_2_1;
                        }
                        long long off_1_1 = off_2;
                        int last_u_1 = ((blk_2 * 8 + 7) < (lq_2_1 - 1) ? (blk_2 * 8 + 7) : (lq_2_1 - 1));
                        long long num_1 = off_1_1 + (long long)last_u_1 + 1;
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
                            mbarrier_wait(q_full_addr, _phase_q_full);
                            #pragma unroll 1
                            for (int ti_1 = 0; ti_1 < n_tiles_3_1; ti_1++) {
                                mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                                if (elect_sync()) {
                                    mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    int _smem_desc_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
                                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 4096;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_tmem_acc + (mma_tmem_stage * 256))), "r"(0) : "memory");
                                    tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                                    tcgen05_commit(k_empty_addr + (mma_k_stage) * 8);
                                    mma_tmem_stage += 1;
                                    if (mma_tmem_stage == 2) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                                }
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                __syncwarp();
                                mma_k_stage += 1;
                                if (mma_k_stage == 4) { mma_k_stage = 0; _phase_k_full ^= 1; }
                            }
                            mbarrier_arrive(q_empty_addr);
                            _phase_q_full ^= 1;
                            if (attempt_2 == 0) {
                                mbarrier_wait(verdict_bar_addr, _phase_verdict_bar_2);
                                _phase_verdict_bar_2 ^= 1;
                                int vread_m = 0;
                                int tread_m = 0;
                                if (lane == 0) {
                                    uint32_t shared_weak;
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(shared_weak) : "r"(static_cast<uint32_t>(s_fail_addr)));
                                    unsigned int f_2 = shared_weak;
                                    for (int fw = 1; fw < 8; fw++) {
                                        uint32_t shared_weak_1;
                                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(shared_weak_1) : "r"(static_cast<uint32_t>(s_fail_addr + (unsigned int)(4 * fw))));
                                        f_2 = f_2 | shared_weak_1;
                                    }
                                    vread_m = (int)f_2;
                                    uint32_t tk_word_m;
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(tk_word_m) : "r"(static_cast<uint32_t>(s_tk_addr + (unsigned int)((tkp_m ^ 1) * 4))));
                                    tread_m = (int)tk_word_m;
                                }
                                int vword_m = __shfl_sync(0xFFFFFFFF, vread_m, 0);
                                int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, tread_m, 0);
                                tk_m = _shfl_sync_4;
                                if (lane == 0) {
                                    mbarrier_arrive(verdict_ack_addr);
                                }
                                if (vword_m == 0) {
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // spare_main
            __syncwarp();
        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // math_main
            int warp_in_wg = warp % 4;
            int lane_0 = lane;
            int local_thread_idx = warp_in_wg * 32 + lane_0;
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int wg = (int)wg_idx;
            int pair = warp_in_wg / 2;
            int half = warp_in_wg % 2;
            int pair_bar = 4 + wg * 2 + pair;
            int bid_i = (int)bid;
            unsigned int one_u = 1;
            unsigned int lower_lanes = (one_u << (unsigned int)lane) - 1;
            int hist_base = (wg * 2 + pair) * 256;
            int pair_dummy = 2048 + wg * 2 + pair;
            unsigned int zero_u = 0;
            unsigned int sign_mask = 2147483648;
            unsigned int nan_floor = 2139095040;
            unsigned int math_sc = 0;
            unsigned int fold_rq = 0;
            int fold_done = 0;
            unsigned int math_tmem_stage = 0;
            unsigned int math_tmem_phase = 0;
            int unit_base = 0;
            int tk_u = 0;
            if (lane == 0) {
                tk_u = s_tk[0];
            }
            int _shfl_sync_0_1 = __shfl_sync(0xFFFFFFFF, tk_u, 0);
            tk_u = _shfl_sync_0_1;
            int tkp_u = 1;
            int tk_in_u = 0;
            int verdict_unit = 0;
            int keep_cap = ((top_k + 64) < (cand_cap - 128 * check_period) ? (top_k + 64) : (cand_cap - 128 * check_period));
            int trigger_room = 128 * check_period;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_umma_full = 0;
            unsigned int _phase_scores_empty = 1;
            unsigned int _phase_comp_req = 0;
            #pragma unroll 1
            for (int seg_3 = 0; seg_3 < num_segments; seg_3++) {
                int q0_3 = cu_seqlens_q[seg_3];
                int lq_3 = cu_seqlens_q[seg_3 + 1] - q0_3;
                int lk_2 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                int nb_3 = (lq_3 + 8 - 1) / 8;
                int cnt_u = nb_3;
#if __CUDA_ARCH__ == 1000
#else
                int q0_e_3 = cu_seqlens_q[seg_3];
                int lq_e_3 = cu_seqlens_q[seg_3 + 1] - q0_e_3;
                int lk_e_3 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                long long off_e_3 = 0;
                if (has_offsets != 0) {
                    off_e_3 = q_offsets[seg_3];
                } else if (ratio == 1) {
                    off_e_3 = (long long)lk_e_3 - (long long)lq_e_3;
                }
                int last_e_3 = ((7) < (lq_e_3 - 1) ? (7) : (lq_e_3 - 1));
                long long num_e_3 = off_e_3 + (long long)last_e_3 + 1;
                int vis_e_3 = 0;
                if (num_e_3 > 0) {
                    long long quot_e_6 = num_e_3 / (long long)ratio;
                    long long lk64_e_6 = (long long)lk_e_3;
                    long long bnd_e_6 = ((quot_e_6 < lk64_e_6) ? quot_e_6 : lk64_e_6);
                    vis_e_3 = (int)bnd_e_6;
                }
                int tiles_e_3 = (vis_e_3 + 128 - 1) / 128;
                int t_lo_u = tiles_e_3;
                int q0_e_3_1 = cu_seqlens_q[seg_3];
                int lq_e_3_1 = cu_seqlens_q[seg_3 + 1] - q0_e_3_1;
                int lk_e_3_1 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                long long off_e_3_1 = 0;
                if (has_offsets != 0) {
                    off_e_3_1 = q_offsets[seg_3];
                } else if (ratio == 1) {
                    off_e_3_1 = (long long)lk_e_3_1 - (long long)lq_e_3_1;
                }
                int last_e_3_1 = (((nb_3 - 1) * 8 + 7) < (lq_e_3_1 - 1) ? ((nb_3 - 1) * 8 + 7) : (lq_e_3_1 - 1));
                long long num_e_3_1 = off_e_3_1 + (long long)last_e_3_1 + 1;
                int vis_e_3_1 = 0;
                if (num_e_3_1 > 0) {
                    long long quot_e_7 = num_e_3_1 / (long long)ratio;
                    long long lk64_e_7 = (long long)lk_e_3_1;
                    long long bnd_e_7 = ((quot_e_7 < lk64_e_7) ? quot_e_7 : lk64_e_7);
                    vis_e_3_1 = (int)bnd_e_7;
                }
                int tiles_e_3_1 = (vis_e_3_1 + 128 - 1) / 128;
                int t_hi_u = tiles_e_3_1;
                int near_eq_u = ((t_hi_u * 4 <= t_lo_u * 5) ? 1 : 0);
#endif
                long long off_3 = 0;
                if (has_offsets != 0) {
                    off_3 = q_offsets[seg_3];
                } else if (ratio == 1) {
                    off_3 = (long long)lk_2 - (long long)lq_3;
                }
                long long dst = off_3;
                #pragma unroll 1
                for (int rnd_3 = 0; rnd_3 < cnt_u + 1; rnd_3++) {
                    int unit_3 = tk_u;
                    if (unit_3 >= unit_base + cnt_u) {
                        break;
                    }
                    tkp_u ^= 1;
                    int _min_1_1 = ((unit_3 - unit_base) < (unit_base + nb_3 - 1 - unit_3) ? (unit_3 - unit_base) : (unit_base + nb_3 - 1 - unit_3));
                    if (_min_1_1 >= 0) {
                        int blk_3 = nb_3 - 1 - (unit_3 - unit_base);
#if __CUDA_ARCH__ == 1000
#else
                        if (sample_tiles_max != 0) {
                            if (near_eq_u != 0) {
                                blk_3 = unit_3 - unit_base;
                            }
                        }
#endif
                        int u_base = blk_3 * 8;
                        int q0_2_1 = cu_seqlens_q[seg_3];
                        int lq_3_1 = cu_seqlens_q[seg_3 + 1] - q0_2_1;
                        int lk_0 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                        long long off_1_2 = 0;
                        if (has_offsets != 0) {
                            off_1_2 = q_offsets[seg_3];
                        } else if (ratio == 1) {
                            off_1_2 = (long long)lk_0 - (long long)lq_3_1;
                        }
                        long long off_1_0 = off_1_2;
                        int last_u_2 = ((blk_3 * 8 + 7) < (lq_3_1 - 1) ? (blk_3 * 8 + 7) : (lq_3_1 - 1));
                        long long num_2 = off_1_0 + (long long)last_u_2 + 1;
                        int vis_2 = 0;
                        if (num_2 > 0) {
                            long long quotient_2 = num_2 / (long long)ratio;
                            long long lk64_2 = (long long)lk_0;
                            long long bounded_2 = ((quotient_2 < lk64_2) ? quotient_2 : lk64_2);
                            vis_2 = (int)bounded_2;
                        }
                        int v_max_2 = vis_2;
                        int n_tiles_2 = (v_max_2 + 128 - 1) / 128;
                        int n_tiles_5 = n_tiles_2;
                        int buf_unit = (bid_i * 8 + wg * 4) * cand_cap;
                        int buf_pairs[2];
                        int slot_pairs[2];
                        int valid_pairs[2];
                        buf_pairs[0] = buf_unit + pair * 2 * cand_cap;
                        slot_pairs[0] = wg * 4 + pair * 2;
                        int u_qj_1_0 = u_base + wg * 4 + pair * 2;
                        int valid_j_1_0 = ((u_qj_1_0 < lq_3) ? 1 : 0);
                        valid_pairs[0] = valid_j_1_0;
                        buf_pairs[1] = buf_unit + (pair * 2 + 1) * cand_cap;
                        slot_pairs[1] = wg * 4 + pair * 2 + 1;
                        int u_qj_1_1 = u_base + wg * 4 + pair * 2 + 1;
                        int valid_j_1_1 = ((u_qj_1_1 < lq_3) ? 1 : 0);
                        valid_pairs[1] = valid_j_1_1;
                        int max_tiles_1 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
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
                                    int cap_fit_1 = ((max_tiles_1) < (8) ? (max_tiles_1) : (8));
                                    if (cap_fit_1 >= 2) {
                                        stride_1 = (n_tiles_5 + cap_fit_1 - 1) / cap_fit_1;
                                        if (stride_1 >= 2) {
                                            n_sample_1 = (n_tiles_5 + stride_1 - 1) / stride_1;
                                        }
                                    }
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int attempt_3 = 0; attempt_3 < 2; attempt_3++) {
                            mbarrier_wait(q_full_addr, _phase_q_full_1);
                            int weight_stage_base_s = 0 * 256 + (unsigned int)(wg * 128);
                            if (local_thread_idx < 128) {
                                float w_raw_s = smem_w[weight_stage_base_s + local_thread_idx];
                                s_wsc[wg * 128 + local_thread_idx] = w_raw_s * softmax_scale;
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            int wsc_addr = s_wsc_addr + wg_idx * 512;
                            float weights_a[64];
                            #pragma unroll
                            for (int wa = 0; wa < 64; wa++) {
                                float w_raw_a = smem_w[weight_stage_base_s + wa];
                                weights_a[wa] = w_raw_a * softmax_scale;
                            }
                            int n_s = n_sample_1;
                            if (attempt_3 != 0) {
                                n_s = 0;
                            }
                            int until_check = check_period - 1;
                            int pend = 0;
                            unsigned int acc_tok = 0;
                            int ti0 = 0;
                            while (ti0 < n_tiles_5) {
                                int run_c = ((until_check + 1) < (n_tiles_5 - ti0) ? (until_check + 1) : (n_tiles_5 - ti0));
                                if (n_s > ti0) {
                                    int _min_2 = ((run_c) < (n_s - ti0) ? (run_c) : (n_s - ti0));
                                    run_c = _min_2;
                                }
                                int head_c = 0;
                                if (pend != 0) {
                                    int _min_3 = ((run_c) < (3) ? (run_c) : (3));
                                    head_c = _min_3;
                                }
                                #pragma unroll 1
                                for (int ti_2 = ti0; ti_2 < ti0 + head_c; ti_2++) {
                                    if (acc_tok == 0) {
                                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    }
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    unsigned int bits2[4];
                                    float weights_reg[64];
                                    float tmem_load[32];
                                    tmem_ld_x16(&tmem_load[0], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load[16], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float tmem_load_0[32];
                                    tmem_ld_x16(&tmem_load_0[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_0[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    float2 sum0 = make_float2(0.0f, 0.0f);
                                    float2 sum1 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j = 0; j < 32; j += 4) {
                                        float2 doubled0 = add_f32x2_noftz(make_float2(tmem_load[j], tmem_load[j + 1]), make_float2(fabsf(tmem_load[j]), fabsf(tmem_load[j + 1])));
                                        sum0 = fma_f32x2_rn_noftz(doubled0, make_float2(weights_a[j], weights_a[j + 1]), sum0);
                                        float2 doubled1 = add_f32x2_noftz(make_float2(tmem_load[j + 2], tmem_load[j + 3]), make_float2(fabsf(tmem_load[j + 2]), fabsf(tmem_load[j + 3])));
                                        sum1 = fma_f32x2_rn_noftz(doubled1, make_float2(weights_a[j + 2], weights_a[j + 3]), sum1);
                                    }
                                    float2 total = add_f32x2_noftz(sum0, sum1);
                                    float relu_wsum = (total.x + total.y) * 0.5f;
                                    float score_w = relu_wsum;
                                    unsigned int bits_w = 0;
                                    bits_w = reinterpret_cast<unsigned int*>(&score_w)[0];
                                    bits2[0] = bits_w;
                                    float tmem_load_1[32];
                                    tmem_ld_x16(&tmem_load_1[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_1[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float tmem_load_2[32];
                                    tmem_ld_x16(&tmem_load_2[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_2[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float2 sum0_0 = make_float2(0.0f, 0.0f);
                                    float2 sum1_0 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_1 = 0; j_1 < 32; j_1 += 4) {
                                        float2 doubled0_1 = add_f32x2_noftz(make_float2(tmem_load_0[j_1], tmem_load_0[j_1 + 1]), make_float2(fabsf(tmem_load_0[j_1]), fabsf(tmem_load_0[j_1 + 1])));
                                        sum0_0 = fma_f32x2_rn_noftz(doubled0_1, make_float2(weights_a[32 + j_1], weights_a[32 + j_1 + 1]), sum0_0);
                                        float2 doubled1_1 = add_f32x2_noftz(make_float2(tmem_load_0[j_1 + 2], tmem_load_0[j_1 + 3]), make_float2(fabsf(tmem_load_0[j_1 + 2]), fabsf(tmem_load_0[j_1 + 3])));
                                        sum1_0 = fma_f32x2_rn_noftz(doubled1_1, make_float2(weights_a[32 + j_1 + 2], weights_a[32 + j_1 + 3]), sum1_0);
                                    }
                                    float2 total_0 = add_f32x2_noftz(sum0_0, sum1_0);
                                    float relu_wsum_0 = (total_0.x + total_0.y) * 0.5f;
                                    float score_w_0 = relu_wsum_0;
                                    unsigned int bits_w_0 = 0;
                                    bits_w_0 = reinterpret_cast<unsigned int*>(&score_w_0)[0];
                                    bits2[1] = bits_w_0;
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
#if __CUDA_ARCH__ == 1000
                                    math_tmem_stage += 1;
                                    if (math_tmem_stage == 2) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                                    uint32_t _mbar_token_0 = mbarrier_test_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    acc_tok = _mbar_token_0;
#endif
                                    float2 sum0_1 = make_float2(0.0f, 0.0f);
                                    float2 sum1_1 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_2 = 0; j_2 < 32; j_2 += 4) {
                                        float2 doubled0_2 = add_f32x2_noftz(make_float2(tmem_load_1[j_2], tmem_load_1[j_2 + 1]), make_float2(fabsf(tmem_load_1[j_2]), fabsf(tmem_load_1[j_2 + 1])));
                                        sum0_1 = fma_f32x2_rn_noftz(doubled0_2, make_float2(weights_reg[j_2], weights_reg[j_2 + 1]), sum0_1);
                                        float2 doubled1_2 = add_f32x2_noftz(make_float2(tmem_load_1[j_2 + 2], tmem_load_1[j_2 + 3]), make_float2(fabsf(tmem_load_1[j_2 + 2]), fabsf(tmem_load_1[j_2 + 3])));
                                        sum1_1 = fma_f32x2_rn_noftz(doubled1_2, make_float2(weights_reg[j_2 + 2], weights_reg[j_2 + 3]), sum1_1);
                                    }
                                    float2 total_1 = add_f32x2_noftz(sum0_1, sum1_1);
                                    float relu_wsum_1 = (total_1.x + total_1.y) * 0.5f;
                                    float score_w_1 = relu_wsum_1;
                                    unsigned int bits_w_1 = 0;
                                    bits_w_1 = reinterpret_cast<unsigned int*>(&score_w_1)[0];
                                    bits2[2] = bits_w_1;
                                    float2 sum0_2 = make_float2(0.0f, 0.0f);
                                    float2 sum1_2 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_3 = 0; j_3 < 32; j_3 += 4) {
                                        float2 doubled0_3 = add_f32x2_noftz(make_float2(tmem_load_2[j_3], tmem_load_2[j_3 + 1]), make_float2(fabsf(tmem_load_2[j_3]), fabsf(tmem_load_2[j_3 + 1])));
                                        sum0_2 = fma_f32x2_rn_noftz(doubled0_3, make_float2(weights_reg[32 + j_3], weights_reg[32 + j_3 + 1]), sum0_2);
                                        float2 doubled1_3 = add_f32x2_noftz(make_float2(tmem_load_2[j_3 + 2], tmem_load_2[j_3 + 3]), make_float2(fabsf(tmem_load_2[j_3 + 2]), fabsf(tmem_load_2[j_3 + 3])));
                                        sum1_2 = fma_f32x2_rn_noftz(doubled1_3, make_float2(weights_reg[32 + j_3 + 2], weights_reg[32 + j_3 + 3]), sum1_2);
                                    }
                                    float2 total_2 = add_f32x2_noftz(sum0_2, sum1_2);
                                    float relu_wsum_2 = (total_2.x + total_2.y) * 0.5f;
                                    float score_w_2 = relu_wsum_2;
                                    unsigned int bits_w_2 = 0;
                                    bits_w_2 = reinterpret_cast<unsigned int*>(&score_w_2)[0];
                                    bits2[3] = bits_w_2;
                                    mbarrier_wait(scores_empty_addr + (math_sc) * 8, _phase_scores_empty);
                                    if ((pend >> 2 - (ti_2 - ti0) & 1) != 0) {
                                        mbarrier_wait(comp_req_addr + (fold_rq) * 8, _phase_comp_req);
                                        int rq_base = (int)fold_rq * 40;
                                        fold_rq += 1;
                                        if (fold_rq == 4) { fold_rq = 0; _phase_comp_req ^= 1; }
                                        int need_any = 0;
                                        need_any = need_any | s_req[rq_base];
                                        need_any = need_any | s_req[rq_base + 5];
                                        need_any = need_any | s_req[rq_base + 10];
                                        need_any = need_any | s_req[rq_base + 15];
                                        need_any = need_any | s_req[rq_base + 20];
                                        need_any = need_any | s_req[rq_base + 25];
                                        need_any = need_any | s_req[rq_base + 30];
                                        need_any = need_any | s_req[rq_base + 35];
                                        int rq_j_1_0 = rq_base + slot_pairs[0] * 5;
                                        int need_j_2_0 = s_req[rq_j_1_0];
                                        if (need_j_2_0 != 0) {
                                            int kw_j_1_0 = s_req[rq_j_1_0 + 1];
                                            int km_j_1_0 = s_req[rq_j_1_0 + 2];
                                            int sk_j_1_0 = s_req[rq_j_1_0 + 3];
                                            int c_j_1_0 = s_req[rq_j_1_0 + 4];
#if __CUDA_ARCH__ == 1000
#else
                                            int c_rank = c_j_1_0 - sk_j_1_0;
#endif
                                            int k_rem = kw_j_1_0;
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
#if __CUDA_ARCH__ == 1000
                                                        if (idx_l < c_j_1_0 - sk_j_1_0) {
                                                            ring[s_pro * 8 + u] = (unsigned long long)Cand[buf_pairs[0] + sk_j_1_0 + idx_l];
#else
                                                        if (idx_l < c_rank) {
                                                            ring[s_pro * 8 + u] = (unsigned long long)Cand[buf_pairs[0] + idx_l];
#endif
                                                        }
                                                    }
                                                }
#if __CUDA_ARCH__ == 1000
                                                int nbat = (c_j_1_0 - sk_j_1_0 + 512 - 1) / 512;
#else
                                                int nbat = (c_rank + 512 - 1) / 512;
#endif
                                                #pragma unroll 1
                                                for (int g = 0; g < nbat; g += 2) {
                                                    #pragma unroll
                                                    for (int s_st = 0; s_st < 2; s_st++) {
                                                        #pragma unroll
                                                        for (int u_1 = 0; u_1 < 8; u_1++) {
#if __CUDA_ARCH__ == 1000
                                                            int valid_p = (((g + s_st) * 512 + half * 32 + lane_0 + 64 * u_1 < c_j_1_0 - sk_j_1_0) ? 1 : 0);
#else
                                                            int valid_p = ((c_rank > (g + s_st) * 512 + half * 32 + lane_0 + 64 * u_1) ? 1 : 0);
#endif
                                                            unsigned int bits = (unsigned int)(ring[s_st * 8 + u_1] >> 32);
                                                            int kid = (int)(unsigned int)(ring[s_st * 8 + u_1] & 4294967295);
                                                            unsigned int magnitude = bits & 2147483647;
                                                            unsigned int m = ((magnitude != 0) ? bits : zero_u);
                                                            unsigned int key32 = (((m & sign_mask) != 0) ? ~m : m | sign_mask);
                                                            if (magnitude > nan_floor) {
                                                                key32 = zero_u;
                                                            }
                                                            unsigned long long key64 = (unsigned long long)key32 << 32 | (unsigned long long)(unsigned int)kid;
                                                            unsigned long long key = key64;
                                                            unsigned long long key_p = key;
                                                            unsigned long long ks_p = key_p >> (unsigned long long)shift;
                                                            int digit_p = (int)(ks_p & 255);
                                                            int pm_p = ((ks_p >> 8 == prefix) ? 1 : 0);
                                                            int bin_p = (((valid_p & pm_p) != 0) ? hist_base + digit_p : pair_dummy);
                                                            atomicAdd(&s_hist[bin_p], 1);
                                                        }
                                                        #pragma unroll
                                                        for (int u_2 = 0; u_2 < 8; u_2++) {
                                                            int idx_l_1 = (g + s_st + 1) * 512 + half * 32 + lane_0 + 64 * u_2;
                                                            ring[(s_st + 1) % 2 * 8 + u_2] = 0;
#if __CUDA_ARCH__ == 1000
                                                            if (idx_l_1 < c_j_1_0 - sk_j_1_0) {
                                                                ring[(s_st + 1) % 2 * 8 + u_2] = (unsigned long long)Cand[buf_pairs[0] + sk_j_1_0 + idx_l_1];
#else
                                                            if (idx_l_1 < c_rank) {
                                                                ring[(s_st + 1) % 2 * 8 + u_2] = (unsigned long long)Cand[buf_pairs[0] + idx_l_1];
#endif
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
                                                int above_part_4_0 = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                                                if (lane_0 + 1 < 32) {
                                                    suffix += above_part_4_0;
                                                }
                                                int above_part_4_1 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
                                                if (lane_0 + 2 < 32) {
                                                    suffix += above_part_4_1;
                                                }
                                                int above_part_4_2 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
                                                if (lane_0 + 4 < 32) {
                                                    suffix += above_part_4_2;
                                                }
                                                int above_part_4_3 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
                                                if (lane_0 + 8 < 32) {
                                                    suffix += above_part_4_3;
                                                }
                                                int above_part_4_4 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
                                                if (lane_0 + 16 < 32) {
                                                    suffix += above_part_4_4;
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
                                                int target_lane;
                                                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane) : "r"(is_target * lane_0));
                                                int digit_sel = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
                                                int above_cnt = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
                                                int bucket_cnt = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
                                                k_rem = k_rem - above_cnt;
                                                kept_above += above_cnt;
                                                prefix = prefix << 8 | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
                                                edge = prefix << (unsigned long long)shift;
                                                if (km_j_1_0 >= kept_above + bucket_cnt) {
                                                    break;
                                                }
                                            }
                                            unsigned long long edge_0 = edge;
                                            if (half == 0) {
                                                int write_pos = 0;
                                                #pragma unroll 1
                                                for (int e0 = 0; e0 < c_j_1_0; e0 += 256) {
                                                    unsigned long long ent[8];
                                                    #pragma unroll
                                                    for (int u_3 = 0; u_3 < 8; u_3++) {
                                                        ent[u_3] = 0;
                                                        if (c_j_1_0 > e0 + lane_0 + 32 * u_3) {
                                                            ent[u_3] = (unsigned long long)Cand[buf_pairs[0] + e0 + lane_0 + 32 * u_3];
                                                        }
                                                    }
                                                    #pragma unroll
                                                    for (int u_4 = 0; u_4 < 8; u_4++) {
                                                        int keep = 0;
                                                        if (c_j_1_0 > e0 + lane_0 + 32 * u_4) {
                                                            unsigned int bits_1 = (unsigned int)(ent[u_4] >> 32);
                                                            int kid_1 = (int)(unsigned int)(ent[u_4] & 4294967295);
                                                            unsigned int magnitude_1 = bits_1 & 2147483647;
                                                            unsigned int m_1 = ((magnitude_1 != 0) ? bits_1 : zero_u);
                                                            unsigned int key32_1 = (((m_1 & sign_mask) != 0) ? ~m_1 : m_1 | sign_mask);
                                                            if (magnitude_1 > nan_floor) {
                                                                key32_1 = zero_u;
                                                            }
                                                            unsigned long long key64_1 = (unsigned long long)key32_1 << 32 | (unsigned long long)(unsigned int)kid_1;
                                                            unsigned long long key_1 = key64_1;
                                                            keep = ((key_1 >= edge_0) ? 1 : 0);
                                                        }
                                                        unsigned int vote = __ballot_sync(0xFFFFFFFF, keep != 0);
                                                        if (keep != 0) {
                                                            Cand[buf_pairs[0] + write_pos + (int)(unsigned int)__popc(vote & lower_lanes)] = (long long)ent[u_4];
                                                        }
                                                        write_pos += (int)(unsigned int)__popc(vote);
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
                                        int rq_j_1_1 = rq_base + slot_pairs[1] * 5;
                                        int need_j_2_1 = s_req[rq_j_1_1];
                                        if (need_j_2_1 != 0) {
                                            int kw_j_1_1 = s_req[rq_j_1_1 + 1];
                                            int km_j_1_1 = s_req[rq_j_1_1 + 2];
                                            int sk_j_1_1 = s_req[rq_j_1_1 + 3];
                                            int c_j_1_1 = s_req[rq_j_1_1 + 4];
#if __CUDA_ARCH__ == 1000
#else
                                            int c_rank_1 = c_j_1_1 - sk_j_1_1;
#endif
                                            int k_rem_1 = kw_j_1_1;
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
#if __CUDA_ARCH__ == 1000
                                                        if (idx_l_2 < c_j_1_1 - sk_j_1_1) {
                                                            ring_1[s_pro_1 * 8 + u_5] = (unsigned long long)Cand[buf_pairs[1] + sk_j_1_1 + idx_l_2];
#else
                                                        if (idx_l_2 < c_rank_1) {
                                                            ring_1[s_pro_1 * 8 + u_5] = (unsigned long long)Cand[buf_pairs[1] + idx_l_2];
#endif
                                                        }
                                                    }
                                                }
#if __CUDA_ARCH__ == 1000
                                                int nbat_1 = (c_j_1_1 - sk_j_1_1 + 512 - 1) / 512;
#else
                                                int nbat_1 = (c_rank_1 + 512 - 1) / 512;
#endif
                                                #pragma unroll 1
                                                for (int g_1 = 0; g_1 < nbat_1; g_1 += 2) {
                                                    #pragma unroll
                                                    for (int s_st_1 = 0; s_st_1 < 2; s_st_1++) {
                                                        #pragma unroll
                                                        for (int u_6 = 0; u_6 < 8; u_6++) {
#if __CUDA_ARCH__ == 1000
                                                            int valid_p_1 = (((g_1 + s_st_1) * 512 + half * 32 + lane_0 + 64 * u_6 < c_j_1_1 - sk_j_1_1) ? 1 : 0);
#else
                                                            int valid_p_1 = ((c_rank_1 > (g_1 + s_st_1) * 512 + half * 32 + lane_0 + 64 * u_6) ? 1 : 0);
#endif
                                                            unsigned int bits_2 = (unsigned int)(ring_1[s_st_1 * 8 + u_6] >> 32);
                                                            int kid_2 = (int)(unsigned int)(ring_1[s_st_1 * 8 + u_6] & 4294967295);
                                                            unsigned int magnitude_2 = bits_2 & 2147483647;
                                                            unsigned int m_2 = ((magnitude_2 != 0) ? bits_2 : zero_u);
                                                            unsigned int key32_2 = (((m_2 & sign_mask) != 0) ? ~m_2 : m_2 | sign_mask);
                                                            if (magnitude_2 > nan_floor) {
                                                                key32_2 = zero_u;
                                                            }
                                                            unsigned long long key64_2 = (unsigned long long)key32_2 << 32 | (unsigned long long)(unsigned int)kid_2;
                                                            unsigned long long key_2 = key64_2;
                                                            unsigned long long key_p_1 = key_2;
                                                            unsigned long long ks_p_1 = key_p_1 >> (unsigned long long)shift_1;
                                                            int digit_p_1 = (int)(ks_p_1 & 255);
                                                            int pm_p_1 = ((ks_p_1 >> 8 == prefix_1) ? 1 : 0);
                                                            int bin_p_1 = (((valid_p_1 & pm_p_1) != 0) ? hist_base + digit_p_1 : pair_dummy);
                                                            atomicAdd(&s_hist[bin_p_1], 1);
                                                        }
                                                        #pragma unroll
                                                        for (int u_7 = 0; u_7 < 8; u_7++) {
                                                            int idx_l_3 = (g_1 + s_st_1 + 1) * 512 + half * 32 + lane_0 + 64 * u_7;
                                                            ring_1[(s_st_1 + 1) % 2 * 8 + u_7] = 0;
#if __CUDA_ARCH__ == 1000
                                                            if (idx_l_3 < c_j_1_1 - sk_j_1_1) {
                                                                ring_1[(s_st_1 + 1) % 2 * 8 + u_7] = (unsigned long long)Cand[buf_pairs[1] + sk_j_1_1 + idx_l_3];
#else
                                                            if (idx_l_3 < c_rank_1) {
                                                                ring_1[(s_st_1 + 1) % 2 * 8 + u_7] = (unsigned long long)Cand[buf_pairs[1] + idx_l_3];
#endif
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
                                                int above_part_4_5 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 1, 32);
                                                if (lane_0 + 1 < 32) {
                                                    suffix_1 += above_part_4_5;
                                                }
                                                int above_part_4_6 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 2, 32);
                                                if (lane_0 + 2 < 32) {
                                                    suffix_1 += above_part_4_6;
                                                }
                                                int above_part_4_7 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 4, 32);
                                                if (lane_0 + 4 < 32) {
                                                    suffix_1 += above_part_4_7;
                                                }
                                                int above_part_4_8 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 8, 32);
                                                if (lane_0 + 8 < 32) {
                                                    suffix_1 += above_part_4_8;
                                                }
                                                int above_part_4_9 = __shfl_down_sync(0xFFFFFFFF, suffix_1, 16, 32);
                                                if (lane_0 + 16 < 32) {
                                                    suffix_1 += above_part_4_9;
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
                                                int target_lane_1;
                                                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane_1) : "r"(is_target_1 * lane_0));
                                                int digit_sel_1 = __shfl_sync(0xFFFFFFFF, d_sel_1, target_lane_1);
                                                int above_cnt_1 = __shfl_sync(0xFFFFFFFF, above_sel_1, target_lane_1);
                                                int bucket_cnt_1 = __shfl_sync(0xFFFFFFFF, count_sel_1, target_lane_1);
                                                k_rem_1 = k_rem_1 - above_cnt_1;
                                                kept_above_1 += above_cnt_1;
                                                prefix_1 = prefix_1 << 8 | (unsigned long long)(unsigned int)(target_lane_1 * 8 + digit_sel_1);
                                                edge_1 = prefix_1 << (unsigned long long)shift_1;
                                                if (km_j_1_1 >= kept_above_1 + bucket_cnt_1) {
                                                    break;
                                                }
                                            }
                                            unsigned long long edge_0_1 = edge_1;
                                            if (half == 0) {
                                                int write_pos_1 = 0;
                                                #pragma unroll 1
                                                for (int e0_1 = 0; e0_1 < c_j_1_1; e0_1 += 256) {
                                                    unsigned long long ent_1[8];
                                                    #pragma unroll
                                                    for (int u_8 = 0; u_8 < 8; u_8++) {
                                                        ent_1[u_8] = 0;
                                                        if (c_j_1_1 > e0_1 + lane_0 + 32 * u_8) {
                                                            ent_1[u_8] = (unsigned long long)Cand[buf_pairs[1] + e0_1 + lane_0 + 32 * u_8];
                                                        }
                                                    }
                                                    #pragma unroll
                                                    for (int u_9 = 0; u_9 < 8; u_9++) {
                                                        int keep_1 = 0;
                                                        if (c_j_1_1 > e0_1 + lane_0 + 32 * u_9) {
                                                            unsigned int bits_3 = (unsigned int)(ent_1[u_9] >> 32);
                                                            int kid_3 = (int)(unsigned int)(ent_1[u_9] & 4294967295);
                                                            unsigned int magnitude_3 = bits_3 & 2147483647;
                                                            unsigned int m_3 = ((magnitude_3 != 0) ? bits_3 : zero_u);
                                                            unsigned int key32_3 = (((m_3 & sign_mask) != 0) ? ~m_3 : m_3 | sign_mask);
                                                            if (magnitude_3 > nan_floor) {
                                                                key32_3 = zero_u;
                                                            }
                                                            unsigned long long key64_3 = (unsigned long long)key32_3 << 32 | (unsigned long long)(unsigned int)kid_3;
                                                            unsigned long long key_3 = key64_3;
                                                            keep_1 = ((key_3 >= edge_0_1) ? 1 : 0);
                                                        }
                                                        unsigned int vote_1 = __ballot_sync(0xFFFFFFFF, keep_1 != 0);
                                                        if (keep_1 != 0) {
                                                            Cand[buf_pairs[1] + write_pos_1 + (int)(unsigned int)__popc(vote_1 & lower_lanes)] = (long long)ent_1[u_9];
                                                        }
                                                        write_pos_1 += (int)(unsigned int)__popc(vote_1);
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
                                        if (need_any != 0) {
                                            if (lane == 0) {
                                                mbarrier_arrive(comp_done_addr);
                                            }
                                            fold_done += 1;
                                        }
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((s_scores_addr + math_sc * 4096 + (unsigned int)((wg * 128 + local_thread_idx) * 16))), "r"(bits2[0]), "r"(bits2[1]), "r"(bits2[2]), "r"(bits2[3]) : "memory");
                                    mbarrier_arrive(scores_full_addr + (math_sc) * 8);
                                    math_sc += 1;
                                    if (math_sc == 3) { math_sc = 0; _phase_scores_empty ^= 1; }
#if __CUDA_ARCH__ == 1000
#else
                                    math_tmem_stage += 1;
                                    if (math_tmem_stage >= 2) { math_tmem_stage -= 2; _phase_umma_full ^= 1; }
                                    uint32_t _mbar_token_0 = mbarrier_test_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    acc_tok = _mbar_token_0;
#endif
                                }
                                constexpr int ti_3_unroll = TILE_UNROLL;
                                #pragma unroll ti_3_unroll
                                for (int ti_3 = ti0 + head_c; ti_3 < ti0 + run_c; ti_3++) {
                                    if (acc_tok == 0) {
                                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    }
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    unsigned int bits2_1[4];
                                    float weights_reg_1[64];
                                    float tmem_load_3[32];
                                    tmem_ld_x16(&tmem_load_3[0], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_3[16], taddr + math_tmem_stage * 256 + (unsigned int)(wg * 4 * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float tmem_load_3_1[32];
                                    tmem_ld_x16(&tmem_load_3_1[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_3_1[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    float2 sum0_3 = make_float2(0.0f, 0.0f);
                                    float2 sum1_3 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_4 = 0; j_4 < 32; j_4 += 4) {
                                        float2 doubled0_4 = add_f32x2_noftz(make_float2(tmem_load_3[j_4], tmem_load_3[j_4 + 1]), make_float2(fabsf(tmem_load_3[j_4]), fabsf(tmem_load_3[j_4 + 1])));
                                        sum0_3 = fma_f32x2_rn_noftz(doubled0_4, make_float2(weights_a[j_4], weights_a[j_4 + 1]), sum0_3);
                                        float2 doubled1_4 = add_f32x2_noftz(make_float2(tmem_load_3[j_4 + 2], tmem_load_3[j_4 + 3]), make_float2(fabsf(tmem_load_3[j_4 + 2]), fabsf(tmem_load_3[j_4 + 3])));
                                        sum1_3 = fma_f32x2_rn_noftz(doubled1_4, make_float2(weights_a[j_4 + 2], weights_a[j_4 + 3]), sum1_3);
                                    }
                                    float2 total_3 = add_f32x2_noftz(sum0_3, sum1_3);
                                    float relu_wsum_3 = (total_3.x + total_3.y) * 0.5f;
                                    float score_w_3 = relu_wsum_3;
                                    unsigned int bits_w_3 = 0;
                                    bits_w_3 = reinterpret_cast<unsigned int*>(&score_w_3)[0];
                                    bits2_1[0] = bits_w_3;
                                    float tmem_load_4[32];
                                    tmem_ld_x16(&tmem_load_4[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_4[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 2) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float tmem_load_5[32];
                                    tmem_ld_x16(&tmem_load_5[0], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                                    tmem_ld_x16(&tmem_load_5[16], taddr + math_tmem_stage * 256 + (unsigned int)((wg * 4 + 3) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                                    float2 sum0_3_1 = make_float2(0.0f, 0.0f);
                                    float2 sum1_3_1 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 32; j_5 += 4) {
                                        float2 doubled0_5 = add_f32x2_noftz(make_float2(tmem_load_3_1[j_5], tmem_load_3_1[j_5 + 1]), make_float2(fabsf(tmem_load_3_1[j_5]), fabsf(tmem_load_3_1[j_5 + 1])));
                                        sum0_3_1 = fma_f32x2_rn_noftz(doubled0_5, make_float2(weights_a[32 + j_5], weights_a[32 + j_5 + 1]), sum0_3_1);
                                        float2 doubled1_5 = add_f32x2_noftz(make_float2(tmem_load_3_1[j_5 + 2], tmem_load_3_1[j_5 + 3]), make_float2(fabsf(tmem_load_3_1[j_5 + 2]), fabsf(tmem_load_3_1[j_5 + 3])));
                                        sum1_3_1 = fma_f32x2_rn_noftz(doubled1_5, make_float2(weights_a[32 + j_5 + 2], weights_a[32 + j_5 + 3]), sum1_3_1);
                                    }
                                    float2 total_3_1 = add_f32x2_noftz(sum0_3_1, sum1_3_1);
                                    float relu_wsum_3_1 = (total_3_1.x + total_3_1.y) * 0.5f;
                                    float score_w_3_1 = relu_wsum_3_1;
                                    unsigned int bits_w_3_1 = 0;
                                    bits_w_3_1 = reinterpret_cast<unsigned int*>(&score_w_3_1)[0];
                                    bits2_1[1] = bits_w_3_1;
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(0) + 3]))
                                        : "r"(wsc_addr + 256));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(4) + 3]))
                                        : "r"(wsc_addr + 272));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(8) + 3]))
                                        : "r"(wsc_addr + 288));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(12) + 3]))
                                        : "r"(wsc_addr + 304));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(16) + 3]))
                                        : "r"(wsc_addr + 320));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(20) + 3]))
                                        : "r"(wsc_addr + 336));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(24) + 3]))
                                        : "r"(wsc_addr + 352));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(28) + 3]))
                                        : "r"(wsc_addr + 368));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(32) + 3]))
                                        : "r"(wsc_addr + 384));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(36) + 3]))
                                        : "r"(wsc_addr + 400));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(40) + 3]))
                                        : "r"(wsc_addr + 416));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(44) + 3]))
                                        : "r"(wsc_addr + 432));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(48) + 3]))
                                        : "r"(wsc_addr + 448));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(52) + 3]))
                                        : "r"(wsc_addr + 464));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(56) + 3]))
                                        : "r"(wsc_addr + 480));
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg_1[(60) + 3]))
                                        : "r"(wsc_addr + 496));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
#if __CUDA_ARCH__ == 1000
                                    math_tmem_stage += 1;
                                    if (math_tmem_stage == 2) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                                    uint32_t _mbar_token_1 = mbarrier_test_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    acc_tok = _mbar_token_1;
#endif
                                    float2 sum0_4 = make_float2(0.0f, 0.0f);
                                    float2 sum1_4 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_6 = 0; j_6 < 32; j_6 += 4) {
                                        float2 doubled0_6 = add_f32x2_noftz(make_float2(tmem_load_4[j_6], tmem_load_4[j_6 + 1]), make_float2(fabsf(tmem_load_4[j_6]), fabsf(tmem_load_4[j_6 + 1])));
                                        sum0_4 = fma_f32x2_rn_noftz(doubled0_6, make_float2(weights_reg_1[j_6], weights_reg_1[j_6 + 1]), sum0_4);
                                        float2 doubled1_6 = add_f32x2_noftz(make_float2(tmem_load_4[j_6 + 2], tmem_load_4[j_6 + 3]), make_float2(fabsf(tmem_load_4[j_6 + 2]), fabsf(tmem_load_4[j_6 + 3])));
                                        sum1_4 = fma_f32x2_rn_noftz(doubled1_6, make_float2(weights_reg_1[j_6 + 2], weights_reg_1[j_6 + 3]), sum1_4);
                                    }
                                    float2 total_4 = add_f32x2_noftz(sum0_4, sum1_4);
                                    float relu_wsum_4 = (total_4.x + total_4.y) * 0.5f;
                                    float score_w_4 = relu_wsum_4;
                                    unsigned int bits_w_4 = 0;
                                    bits_w_4 = reinterpret_cast<unsigned int*>(&score_w_4)[0];
                                    bits2_1[2] = bits_w_4;
                                    float2 sum0_5 = make_float2(0.0f, 0.0f);
                                    float2 sum1_5 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 32; j_7 += 4) {
                                        float2 doubled0_7 = add_f32x2_noftz(make_float2(tmem_load_5[j_7], tmem_load_5[j_7 + 1]), make_float2(fabsf(tmem_load_5[j_7]), fabsf(tmem_load_5[j_7 + 1])));
                                        sum0_5 = fma_f32x2_rn_noftz(doubled0_7, make_float2(weights_reg_1[32 + j_7], weights_reg_1[32 + j_7 + 1]), sum0_5);
                                        float2 doubled1_7 = add_f32x2_noftz(make_float2(tmem_load_5[j_7 + 2], tmem_load_5[j_7 + 3]), make_float2(fabsf(tmem_load_5[j_7 + 2]), fabsf(tmem_load_5[j_7 + 3])));
                                        sum1_5 = fma_f32x2_rn_noftz(doubled1_7, make_float2(weights_reg_1[32 + j_7 + 2], weights_reg_1[32 + j_7 + 3]), sum1_5);
                                    }
                                    float2 total_5 = add_f32x2_noftz(sum0_5, sum1_5);
                                    float relu_wsum_5 = (total_5.x + total_5.y) * 0.5f;
                                    float score_w_5 = relu_wsum_5;
                                    unsigned int bits_w_5 = 0;
                                    bits_w_5 = reinterpret_cast<unsigned int*>(&score_w_5)[0];
                                    bits2_1[3] = bits_w_5;
                                    mbarrier_wait(scores_empty_addr + (math_sc) * 8, _phase_scores_empty);
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((s_scores_addr + math_sc * 4096 + (unsigned int)((wg * 128 + local_thread_idx) * 16))), "r"(bits2_1[0]), "r"(bits2_1[1]), "r"(bits2_1[2]), "r"(bits2_1[3]) : "memory");
                                    mbarrier_arrive(scores_full_addr + (math_sc) * 8);
                                    math_sc += 1;
                                    if (math_sc == 3) { math_sc = 0; _phase_scores_empty ^= 1; }
#if __CUDA_ARCH__ == 1000
#else
                                    math_tmem_stage += 1;
                                    if (math_tmem_stage >= 2) { math_tmem_stage -= 2; _phase_umma_full ^= 1; }
                                    uint32_t _mbar_token_1 = mbarrier_test_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                                    acc_tok = _mbar_token_1;
#endif
                                }
                                ti0 = ti0 + run_c;
                                int sample_c = ((ti0 == n_s) ? 1 : 0);
                                int check_c = 0;
                                if (run_c == until_check + 1 || sample_c != 0) {
                                    until_check = check_period - 1;
                                    check_c = 1;
                                } else {
                                    until_check = until_check - run_c;
                                }
                                int _min_4 = ((run_c) < (3) ? (run_c) : (3));
                                pend = (pend << _min_4 | check_c) & 7;
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(q_empty_addr);
                            _phase_q_full_1 ^= 1;
                            if (pend != 0) {
                                #pragma unroll 1
                                for (int kk = 0; kk < 3; kk++) {
                                    if ((pend >> 2 - kk & 1) != 0) {
                                        mbarrier_wait(comp_req_addr + (fold_rq) * 8, _phase_comp_req);
                                        int rq_base_1 = (int)fold_rq * 40;
                                        fold_rq += 1;
                                        if (fold_rq == 4) { fold_rq = 0; _phase_comp_req ^= 1; }
                                        int need_any_1 = 0;
                                        need_any_1 = need_any_1 | s_req[rq_base_1];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 5];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 10];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 15];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 20];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 25];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 30];
                                        need_any_1 = need_any_1 | s_req[rq_base_1 + 35];
                                        int rq_j_1_2 = rq_base_1 + slot_pairs[0] * 5;
                                        int need_j_2_2 = s_req[rq_j_1_2];
                                        if (need_j_2_2 != 0) {
                                            int kw_j_1_2 = s_req[rq_j_1_2 + 1];
                                            int km_j_1_2 = s_req[rq_j_1_2 + 2];
                                            int sk_j_1_2 = s_req[rq_j_1_2 + 3];
                                            int c_j_1_2 = s_req[rq_j_1_2 + 4];
#if __CUDA_ARCH__ == 1000
#else
                                            int c_rank_2 = c_j_1_2 - sk_j_1_2;
#endif
                                            int k_rem_2 = kw_j_1_2;
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
                                                unsigned long long ring_2[16];
                                                #pragma unroll
                                                for (int s_pro_2 = 0; s_pro_2 < 1; s_pro_2++) {
                                                    #pragma unroll
                                                    for (int u_10 = 0; u_10 < 8; u_10++) {
                                                        int idx_l_4 = s_pro_2 * 512 + half * 32 + lane_0 + 64 * u_10;
                                                        ring_2[s_pro_2 * 8 + u_10] = 0;
#if __CUDA_ARCH__ == 1000
                                                        if (idx_l_4 < c_j_1_2 - sk_j_1_2) {
                                                            ring_2[s_pro_2 * 8 + u_10] = (unsigned long long)Cand[buf_pairs[0] + sk_j_1_2 + idx_l_4];
#else
                                                        if (idx_l_4 < c_rank_2) {
                                                            ring_2[s_pro_2 * 8 + u_10] = (unsigned long long)Cand[buf_pairs[0] + idx_l_4];
#endif
                                                        }
                                                    }
                                                }
#if __CUDA_ARCH__ == 1000
                                                int nbat_2 = (c_j_1_2 - sk_j_1_2 + 512 - 1) / 512;
#else
                                                int nbat_2 = (c_rank_2 + 512 - 1) / 512;
#endif
                                                #pragma unroll 1
                                                for (int g_2 = 0; g_2 < nbat_2; g_2 += 2) {
                                                    #pragma unroll
                                                    for (int s_st_2 = 0; s_st_2 < 2; s_st_2++) {
                                                        #pragma unroll
                                                        for (int u_11 = 0; u_11 < 8; u_11++) {
#if __CUDA_ARCH__ == 1000
                                                            int valid_p_2 = (((g_2 + s_st_2) * 512 + half * 32 + lane_0 + 64 * u_11 < c_j_1_2 - sk_j_1_2) ? 1 : 0);
#else
                                                            int valid_p_2 = ((c_rank_2 > (g_2 + s_st_2) * 512 + half * 32 + lane_0 + 64 * u_11) ? 1 : 0);
#endif
                                                            unsigned int bits_4 = (unsigned int)(ring_2[s_st_2 * 8 + u_11] >> 32);
                                                            int kid_4 = (int)(unsigned int)(ring_2[s_st_2 * 8 + u_11] & 4294967295);
                                                            unsigned int magnitude_4 = bits_4 & 2147483647;
                                                            unsigned int m_4 = ((magnitude_4 != 0) ? bits_4 : zero_u);
                                                            unsigned int key32_4 = (((m_4 & sign_mask) != 0) ? ~m_4 : m_4 | sign_mask);
                                                            if (magnitude_4 > nan_floor) {
                                                                key32_4 = zero_u;
                                                            }
                                                            unsigned long long key64_4 = (unsigned long long)key32_4 << 32 | (unsigned long long)(unsigned int)kid_4;
                                                            unsigned long long key_4 = key64_4;
                                                            unsigned long long key_p_2 = key_4;
                                                            unsigned long long ks_p_2 = key_p_2 >> (unsigned long long)shift_2;
                                                            int digit_p_2 = (int)(ks_p_2 & 255);
                                                            int pm_p_2 = ((ks_p_2 >> 8 == prefix_2) ? 1 : 0);
                                                            int bin_p_2 = (((valid_p_2 & pm_p_2) != 0) ? hist_base + digit_p_2 : pair_dummy);
                                                            atomicAdd(&s_hist[bin_p_2], 1);
                                                        }
                                                        #pragma unroll
                                                        for (int u_12 = 0; u_12 < 8; u_12++) {
                                                            int idx_l_5 = (g_2 + s_st_2 + 1) * 512 + half * 32 + lane_0 + 64 * u_12;
                                                            ring_2[(s_st_2 + 1) % 2 * 8 + u_12] = 0;
#if __CUDA_ARCH__ == 1000
                                                            if (idx_l_5 < c_j_1_2 - sk_j_1_2) {
                                                                ring_2[(s_st_2 + 1) % 2 * 8 + u_12] = (unsigned long long)Cand[buf_pairs[0] + sk_j_1_2 + idx_l_5];
#else
                                                            if (idx_l_5 < c_rank_2) {
                                                                ring_2[(s_st_2 + 1) % 2 * 8 + u_12] = (unsigned long long)Cand[buf_pairs[0] + idx_l_5];
#endif
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
                                                int above_part_4_10 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 1, 32);
                                                if (lane_0 + 1 < 32) {
                                                    suffix_2 += above_part_4_10;
                                                }
                                                int above_part_4_11 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 2, 32);
                                                if (lane_0 + 2 < 32) {
                                                    suffix_2 += above_part_4_11;
                                                }
                                                int above_part_4_12 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 4, 32);
                                                if (lane_0 + 4 < 32) {
                                                    suffix_2 += above_part_4_12;
                                                }
                                                int above_part_4_13 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 8, 32);
                                                if (lane_0 + 8 < 32) {
                                                    suffix_2 += above_part_4_13;
                                                }
                                                int above_part_4_14 = __shfl_down_sync(0xFFFFFFFF, suffix_2, 16, 32);
                                                if (lane_0 + 16 < 32) {
                                                    suffix_2 += above_part_4_14;
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
                                                int target_lane_2;
                                                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane_2) : "r"(is_target_2 * lane_0));
                                                int digit_sel_2 = __shfl_sync(0xFFFFFFFF, d_sel_2, target_lane_2);
                                                int above_cnt_2 = __shfl_sync(0xFFFFFFFF, above_sel_2, target_lane_2);
                                                int bucket_cnt_2 = __shfl_sync(0xFFFFFFFF, count_sel_2, target_lane_2);
                                                k_rem_2 = k_rem_2 - above_cnt_2;
                                                kept_above_2 += above_cnt_2;
                                                prefix_2 = prefix_2 << 8 | (unsigned long long)(unsigned int)(target_lane_2 * 8 + digit_sel_2);
                                                edge_2 = prefix_2 << (unsigned long long)shift_2;
                                                if (km_j_1_2 >= kept_above_2 + bucket_cnt_2) {
                                                    break;
                                                }
                                            }
                                            unsigned long long edge_0_2 = edge_2;
                                            if (half == 0) {
                                                int write_pos_2 = 0;
                                                #pragma unroll 1
                                                for (int e0_2 = 0; e0_2 < c_j_1_2; e0_2 += 256) {
                                                    unsigned long long ent_2[8];
                                                    #pragma unroll
                                                    for (int u_13 = 0; u_13 < 8; u_13++) {
                                                        ent_2[u_13] = 0;
                                                        if (c_j_1_2 > e0_2 + lane_0 + 32 * u_13) {
                                                            ent_2[u_13] = (unsigned long long)Cand[buf_pairs[0] + e0_2 + lane_0 + 32 * u_13];
                                                        }
                                                    }
                                                    #pragma unroll
                                                    for (int u_14 = 0; u_14 < 8; u_14++) {
                                                        int keep_2 = 0;
                                                        if (c_j_1_2 > e0_2 + lane_0 + 32 * u_14) {
                                                            unsigned int bits_5 = (unsigned int)(ent_2[u_14] >> 32);
                                                            int kid_5 = (int)(unsigned int)(ent_2[u_14] & 4294967295);
                                                            unsigned int magnitude_5 = bits_5 & 2147483647;
                                                            unsigned int m_5 = ((magnitude_5 != 0) ? bits_5 : zero_u);
                                                            unsigned int key32_5 = (((m_5 & sign_mask) != 0) ? ~m_5 : m_5 | sign_mask);
                                                            if (magnitude_5 > nan_floor) {
                                                                key32_5 = zero_u;
                                                            }
                                                            unsigned long long key64_5 = (unsigned long long)key32_5 << 32 | (unsigned long long)(unsigned int)kid_5;
                                                            unsigned long long key_5 = key64_5;
                                                            keep_2 = ((key_5 >= edge_0_2) ? 1 : 0);
                                                        }
                                                        unsigned int vote_2 = __ballot_sync(0xFFFFFFFF, keep_2 != 0);
                                                        if (keep_2 != 0) {
                                                            Cand[buf_pairs[0] + write_pos_2 + (int)(unsigned int)__popc(vote_2 & lower_lanes)] = (long long)ent_2[u_14];
                                                        }
                                                        write_pos_2 += (int)(unsigned int)__popc(vote_2);
                                                    }
                                                }
                                                if (lane_0 == 0) {
                                                    s_count[slot_pairs[0]] = write_pos_2;
                                                    s_tau[slot_pairs[0] * 2] = (int)(unsigned int)edge_0_2;
                                                    s_tau[slot_pairs[0] * 2 + 1] = (int)(unsigned int)(edge_0_2 >> 32);
                                                }
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                        }
                                        int rq_j_1_3 = rq_base_1 + slot_pairs[1] * 5;
                                        int need_j_2_3 = s_req[rq_j_1_3];
                                        if (need_j_2_3 != 0) {
                                            int kw_j_1_3 = s_req[rq_j_1_3 + 1];
                                            int km_j_1_3 = s_req[rq_j_1_3 + 2];
                                            int sk_j_1_3 = s_req[rq_j_1_3 + 3];
                                            int c_j_1_3 = s_req[rq_j_1_3 + 4];
#if __CUDA_ARCH__ == 1000
#else
                                            int c_rank_3 = c_j_1_3 - sk_j_1_3;
#endif
                                            int k_rem_3 = kw_j_1_3;
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
                                                unsigned long long ring_3[16];
                                                #pragma unroll
                                                for (int s_pro_3 = 0; s_pro_3 < 1; s_pro_3++) {
                                                    #pragma unroll
                                                    for (int u_15 = 0; u_15 < 8; u_15++) {
                                                        int idx_l_6 = s_pro_3 * 512 + half * 32 + lane_0 + 64 * u_15;
                                                        ring_3[s_pro_3 * 8 + u_15] = 0;
#if __CUDA_ARCH__ == 1000
                                                        if (idx_l_6 < c_j_1_3 - sk_j_1_3) {
                                                            ring_3[s_pro_3 * 8 + u_15] = (unsigned long long)Cand[buf_pairs[1] + sk_j_1_3 + idx_l_6];
#else
                                                        if (idx_l_6 < c_rank_3) {
                                                            ring_3[s_pro_3 * 8 + u_15] = (unsigned long long)Cand[buf_pairs[1] + idx_l_6];
#endif
                                                        }
                                                    }
                                                }
#if __CUDA_ARCH__ == 1000
                                                int nbat_3 = (c_j_1_3 - sk_j_1_3 + 512 - 1) / 512;
#else
                                                int nbat_3 = (c_rank_3 + 512 - 1) / 512;
#endif
                                                #pragma unroll 1
                                                for (int g_3 = 0; g_3 < nbat_3; g_3 += 2) {
                                                    #pragma unroll
                                                    for (int s_st_3 = 0; s_st_3 < 2; s_st_3++) {
                                                        #pragma unroll
                                                        for (int u_16 = 0; u_16 < 8; u_16++) {
#if __CUDA_ARCH__ == 1000
                                                            int valid_p_3 = (((g_3 + s_st_3) * 512 + half * 32 + lane_0 + 64 * u_16 < c_j_1_3 - sk_j_1_3) ? 1 : 0);
#else
                                                            int valid_p_3 = ((c_rank_3 > (g_3 + s_st_3) * 512 + half * 32 + lane_0 + 64 * u_16) ? 1 : 0);
#endif
                                                            unsigned int bits_6 = (unsigned int)(ring_3[s_st_3 * 8 + u_16] >> 32);
                                                            int kid_6 = (int)(unsigned int)(ring_3[s_st_3 * 8 + u_16] & 4294967295);
                                                            unsigned int magnitude_6 = bits_6 & 2147483647;
                                                            unsigned int m_6 = ((magnitude_6 != 0) ? bits_6 : zero_u);
                                                            unsigned int key32_6 = (((m_6 & sign_mask) != 0) ? ~m_6 : m_6 | sign_mask);
                                                            if (magnitude_6 > nan_floor) {
                                                                key32_6 = zero_u;
                                                            }
                                                            unsigned long long key64_6 = (unsigned long long)key32_6 << 32 | (unsigned long long)(unsigned int)kid_6;
                                                            unsigned long long key_6 = key64_6;
                                                            unsigned long long key_p_3 = key_6;
                                                            unsigned long long ks_p_3 = key_p_3 >> (unsigned long long)shift_3;
                                                            int digit_p_3 = (int)(ks_p_3 & 255);
                                                            int pm_p_3 = ((ks_p_3 >> 8 == prefix_3) ? 1 : 0);
                                                            int bin_p_3 = (((valid_p_3 & pm_p_3) != 0) ? hist_base + digit_p_3 : pair_dummy);
                                                            atomicAdd(&s_hist[bin_p_3], 1);
                                                        }
                                                        #pragma unroll
                                                        for (int u_17 = 0; u_17 < 8; u_17++) {
                                                            int idx_l_7 = (g_3 + s_st_3 + 1) * 512 + half * 32 + lane_0 + 64 * u_17;
                                                            ring_3[(s_st_3 + 1) % 2 * 8 + u_17] = 0;
#if __CUDA_ARCH__ == 1000
                                                            if (idx_l_7 < c_j_1_3 - sk_j_1_3) {
                                                                ring_3[(s_st_3 + 1) % 2 * 8 + u_17] = (unsigned long long)Cand[buf_pairs[1] + sk_j_1_3 + idx_l_7];
#else
                                                            if (idx_l_7 < c_rank_3) {
                                                                ring_3[(s_st_3 + 1) % 2 * 8 + u_17] = (unsigned long long)Cand[buf_pairs[1] + idx_l_7];
#endif
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
                                                int above_part_4_15 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 1, 32);
                                                if (lane_0 + 1 < 32) {
                                                    suffix_3 += above_part_4_15;
                                                }
                                                int above_part_4_16 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 2, 32);
                                                if (lane_0 + 2 < 32) {
                                                    suffix_3 += above_part_4_16;
                                                }
                                                int above_part_4_17 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 4, 32);
                                                if (lane_0 + 4 < 32) {
                                                    suffix_3 += above_part_4_17;
                                                }
                                                int above_part_4_18 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 8, 32);
                                                if (lane_0 + 8 < 32) {
                                                    suffix_3 += above_part_4_18;
                                                }
                                                int above_part_4_19 = __shfl_down_sync(0xFFFFFFFF, suffix_3, 16, 32);
                                                if (lane_0 + 16 < 32) {
                                                    suffix_3 += above_part_4_19;
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
                                                int target_lane_3;
                                                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane_3) : "r"(is_target_3 * lane_0));
                                                int digit_sel_3 = __shfl_sync(0xFFFFFFFF, d_sel_3, target_lane_3);
                                                int above_cnt_3 = __shfl_sync(0xFFFFFFFF, above_sel_3, target_lane_3);
                                                int bucket_cnt_3 = __shfl_sync(0xFFFFFFFF, count_sel_3, target_lane_3);
                                                k_rem_3 = k_rem_3 - above_cnt_3;
                                                kept_above_3 += above_cnt_3;
                                                prefix_3 = prefix_3 << 8 | (unsigned long long)(unsigned int)(target_lane_3 * 8 + digit_sel_3);
                                                edge_3 = prefix_3 << (unsigned long long)shift_3;
                                                if (km_j_1_3 >= kept_above_3 + bucket_cnt_3) {
                                                    break;
                                                }
                                            }
                                            unsigned long long edge_0_3 = edge_3;
                                            if (half == 0) {
                                                int write_pos_3 = 0;
                                                #pragma unroll 1
                                                for (int e0_3 = 0; e0_3 < c_j_1_3; e0_3 += 256) {
                                                    unsigned long long ent_3[8];
                                                    #pragma unroll
                                                    for (int u_18 = 0; u_18 < 8; u_18++) {
                                                        ent_3[u_18] = 0;
                                                        if (c_j_1_3 > e0_3 + lane_0 + 32 * u_18) {
                                                            ent_3[u_18] = (unsigned long long)Cand[buf_pairs[1] + e0_3 + lane_0 + 32 * u_18];
                                                        }
                                                    }
                                                    #pragma unroll
                                                    for (int u_19 = 0; u_19 < 8; u_19++) {
                                                        int keep_3 = 0;
                                                        if (c_j_1_3 > e0_3 + lane_0 + 32 * u_19) {
                                                            unsigned int bits_7 = (unsigned int)(ent_3[u_19] >> 32);
                                                            int kid_7 = (int)(unsigned int)(ent_3[u_19] & 4294967295);
                                                            unsigned int magnitude_7 = bits_7 & 2147483647;
                                                            unsigned int m_7 = ((magnitude_7 != 0) ? bits_7 : zero_u);
                                                            unsigned int key32_7 = (((m_7 & sign_mask) != 0) ? ~m_7 : m_7 | sign_mask);
                                                            if (magnitude_7 > nan_floor) {
                                                                key32_7 = zero_u;
                                                            }
                                                            unsigned long long key64_7 = (unsigned long long)key32_7 << 32 | (unsigned long long)(unsigned int)kid_7;
                                                            unsigned long long key_7 = key64_7;
                                                            keep_3 = ((key_7 >= edge_0_3) ? 1 : 0);
                                                        }
                                                        unsigned int vote_3 = __ballot_sync(0xFFFFFFFF, keep_3 != 0);
                                                        if (keep_3 != 0) {
                                                            Cand[buf_pairs[1] + write_pos_3 + (int)(unsigned int)__popc(vote_3 & lower_lanes)] = (long long)ent_3[u_19];
                                                        }
                                                        write_pos_3 += (int)(unsigned int)__popc(vote_3);
                                                    }
                                                }
                                                if (lane_0 == 0) {
                                                    s_count[slot_pairs[1]] = write_pos_3;
                                                    s_tau[slot_pairs[1] * 2] = (int)(unsigned int)edge_0_3;
                                                    s_tau[slot_pairs[1] * 2 + 1] = (int)(unsigned int)(edge_0_3 >> 32);
                                                }
                                            }
                                            asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                        }
                                        if (need_any_1 != 0) {
                                            if (lane == 0) {
                                                mbarrier_arrive(comp_done_addr);
                                            }
                                            fold_done += 1;
                                        }
                                    }
                                }
                            }
                            asm volatile("barrier.sync 8, 384;" ::: "memory");
                            int c_finals[2];
                            c_finals[0] = s_count[slot_pairs[0]];
                            c_finals[1] = s_count[slot_pairs[1]];
                            int verdict = 0;
                            if (attempt_3 == 0) {
                                int f_3 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                f_3 = f_3 | s_fail[4];
                                f_3 = f_3 | s_fail[5];
                                f_3 = f_3 | s_fail[6];
                                f_3 = f_3 | s_fail[7];
                                verdict = f_3;
                            }
                            tk_u = s_tk[tkp_u ^ 1];
                            asm volatile("barrier.sync 8, 384;" ::: "memory");
                            if (verdict == 0) {
                                int c_fin_1_0 = c_finals[0];
                                int n_sel_j_2_0 = ((top_k) < (c_fin_1_0) ? (top_k) : (c_fin_1_0));
                                unsigned long long edge_j_1_0 = 0;
                                if (c_fin_1_0 > top_k) {
                                    int k_rem_4 = top_k;
                                    int kept_above_4 = 0;
                                    unsigned long long prefix_4 = 0;
                                    unsigned long long edge_4 = 0;
                                    #pragma unroll 1
                                    for (int p_4 = 0; p_4 < 8; p_4++) {
                                        int shift_4 = 56 - 8 * p_4;
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
                                        unsigned long long ring_4[24];
                                        #pragma unroll
                                        for (int s_pro_4 = 0; s_pro_4 < 2; s_pro_4++) {
                                            #pragma unroll
                                            for (int u_20 = 0; u_20 < 8; u_20++) {
                                                int idx_l_8 = s_pro_4 * 512 + half * 32 + lane_0 + 64 * u_20;
                                                ring_4[s_pro_4 * 8 + u_20] = 0;
                                                if (idx_l_8 < c_fin_1_0) {
                                                    ring_4[s_pro_4 * 8 + u_20] = (unsigned long long)Cand[buf_pairs[0] + idx_l_8];
                                                }
                                            }
                                        }
                                        int nbat_4 = (c_fin_1_0 + 512 - 1) / 512;
                                        #pragma unroll 1
                                        for (int g_4 = 0; g_4 < nbat_4; g_4 += 3) {
                                            #pragma unroll
                                            for (int s_st_4 = 0; s_st_4 < 3; s_st_4++) {
                                                #pragma unroll
                                                for (int u_21 = 0; u_21 < 8; u_21++) {
                                                    int valid_p_4 = ((c_fin_1_0 > (g_4 + s_st_4) * 512 + half * 32 + lane_0 + 64 * u_21) ? 1 : 0);
                                                    unsigned int bits_8 = (unsigned int)(ring_4[s_st_4 * 8 + u_21] >> 32);
                                                    int kid_8 = (int)(unsigned int)(ring_4[s_st_4 * 8 + u_21] & 4294967295);
                                                    unsigned int magnitude_8 = bits_8 & 2147483647;
                                                    unsigned int m_8 = ((magnitude_8 != 0) ? bits_8 : zero_u);
                                                    unsigned int key32_8 = (((m_8 & sign_mask) != 0) ? ~m_8 : m_8 | sign_mask);
                                                    if (magnitude_8 > nan_floor) {
                                                        key32_8 = zero_u;
                                                    }
                                                    unsigned long long key64_8 = (unsigned long long)key32_8 << 32 | (unsigned long long)(unsigned int)kid_8;
                                                    unsigned long long key_8 = key64_8;
                                                    unsigned long long key_p_4 = key_8;
                                                    unsigned long long ks_p_4 = key_p_4 >> (unsigned long long)shift_4;
                                                    int digit_p_4 = (int)(ks_p_4 & 255);
                                                    int pm_p_4 = ((ks_p_4 >> 8 == prefix_4) ? 1 : 0);
                                                    int bin_p_4 = (((valid_p_4 & pm_p_4) != 0) ? hist_base + digit_p_4 : pair_dummy);
                                                    atomicAdd(&s_hist[bin_p_4], 1);
                                                }
                                                #pragma unroll
                                                for (int u_22 = 0; u_22 < 8; u_22++) {
                                                    int idx_l_9 = (g_4 + s_st_4 + 2) * 512 + half * 32 + lane_0 + 64 * u_22;
                                                    ring_4[(s_st_4 + 2) % 3 * 8 + u_22] = 0;
                                                    if (idx_l_9 < c_fin_1_0) {
                                                        ring_4[(s_st_4 + 2) % 3 * 8 + u_22] = (unsigned long long)Cand[buf_pairs[0] + idx_l_9];
                                                    }
                                                }
                                            }
                                        }
                                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                        int lane_bins_4[8];
                                        int lane_sum_4 = 0;
                                        lane_bins_4[0] = s_hist[hist_base + lane_0 * 8];
                                        lane_sum_4 += lane_bins_4[0];
                                        lane_bins_4[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                        lane_sum_4 += lane_bins_4[1];
                                        lane_bins_4[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                        lane_sum_4 += lane_bins_4[2];
                                        lane_bins_4[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                        lane_sum_4 += lane_bins_4[3];
                                        lane_bins_4[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                        lane_sum_4 += lane_bins_4[4];
                                        lane_bins_4[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                        lane_sum_4 += lane_bins_4[5];
                                        lane_bins_4[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                        lane_sum_4 += lane_bins_4[6];
                                        lane_bins_4[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                        lane_sum_4 += lane_bins_4[7];
                                        int suffix_4 = lane_sum_4;
                                        int above_part_4_20 = __shfl_down_sync(0xFFFFFFFF, suffix_4, 1, 32);
                                        if (lane_0 + 1 < 32) {
                                            suffix_4 += above_part_4_20;
                                        }
                                        int above_part_4_21 = __shfl_down_sync(0xFFFFFFFF, suffix_4, 2, 32);
                                        if (lane_0 + 2 < 32) {
                                            suffix_4 += above_part_4_21;
                                        }
                                        int above_part_4_22 = __shfl_down_sync(0xFFFFFFFF, suffix_4, 4, 32);
                                        if (lane_0 + 4 < 32) {
                                            suffix_4 += above_part_4_22;
                                        }
                                        int above_part_4_23 = __shfl_down_sync(0xFFFFFFFF, suffix_4, 8, 32);
                                        if (lane_0 + 8 < 32) {
                                            suffix_4 += above_part_4_23;
                                        }
                                        int above_part_4_24 = __shfl_down_sync(0xFFFFFFFF, suffix_4, 16, 32);
                                        if (lane_0 + 16 < 32) {
                                            suffix_4 += above_part_4_24;
                                        }
                                        int excl_4 = suffix_4 - lane_sum_4;
                                        int is_target_4 = ((excl_4 < k_rem_4 && k_rem_4 <= excl_4 + lane_sum_4) ? 1 : 0);
                                        int d_sel_4 = 0;
                                        int above_sel_4 = 0;
                                        int count_sel_4 = 0;
                                        int found_4 = 0;
                                        int cum_above_4 = excl_4;
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[7]) {
                                                d_sel_4 = 7;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[7];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[7];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[6]) {
                                                d_sel_4 = 6;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[6];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[6];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[5]) {
                                                d_sel_4 = 5;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[5];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[5];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[4]) {
                                                d_sel_4 = 4;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[4];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[4];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[3]) {
                                                d_sel_4 = 3;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[3];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[3];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[2]) {
                                                d_sel_4 = 2;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[2];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[2];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[1]) {
                                                d_sel_4 = 1;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[1];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[1];
                                        if (found_4 == 0) {
                                            if (k_rem_4 <= cum_above_4 + lane_bins_4[0]) {
                                                d_sel_4 = 0;
                                                above_sel_4 = cum_above_4;
                                                count_sel_4 = lane_bins_4[0];
                                                found_4 = 1;
                                            }
                                        }
                                        cum_above_4 += lane_bins_4[0];
                                        int target_lane_4;
                                        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane_4) : "r"(is_target_4 * lane_0));
                                        int digit_sel_4 = __shfl_sync(0xFFFFFFFF, d_sel_4, target_lane_4);
                                        int above_cnt_4 = __shfl_sync(0xFFFFFFFF, above_sel_4, target_lane_4);
                                        int bucket_cnt_4 = __shfl_sync(0xFFFFFFFF, count_sel_4, target_lane_4);
                                        k_rem_4 = k_rem_4 - above_cnt_4;
                                        kept_above_4 += above_cnt_4;
                                        prefix_4 = prefix_4 << 8 | (unsigned long long)(unsigned int)(target_lane_4 * 8 + digit_sel_4);
                                        edge_4 = prefix_4 << (unsigned long long)shift_4;
                                        if (kept_above_4 + bucket_cnt_4 <= top_k) {
                                            break;
                                        }
                                    }
                                    edge_j_1_0 = edge_4;
                                }
                                int row_j_1_0 = q0_3 + u_base + slot_pairs[0];
                                long long row_base_j_1_0 = (long long)row_j_1_0 * (long long)top_k;
                                if (valid_pairs[0] != 0) {
                                    int mid_w = (c_fin_1_0 / 2 + 31) / 32 * 32;
                                    if (mid_w > c_fin_1_0) {
                                        mid_w = c_fin_1_0;
                                    }
                                    if (half == 0) {
                                        int front_w = 0;
                                        #pragma unroll 1
                                        for (int e0_4 = 0; e0_4 < mid_w; e0_4 += 256) {
                                            unsigned long long ent_wf[8];
                                            #pragma unroll
                                            for (int u_23 = 0; u_23 < 8; u_23++) {
                                                ent_wf[u_23] = 0;
                                                if (mid_w > e0_4 + lane_0 + 32 * u_23) {
                                                    ent_wf[u_23] = (unsigned long long)Cand[buf_pairs[0] + e0_4 + lane_0 + 32 * u_23];
                                                }
                                            }
                                            #pragma unroll
                                            for (int u_24 = 0; u_24 < 8; u_24++) {
                                                int keep_wf = 0;
                                                if (mid_w > e0_4 + lane_0 + 32 * u_24) {
                                                    unsigned int bits_9 = (unsigned int)(ent_wf[u_24] >> 32);
                                                    int kid_9 = (int)(unsigned int)(ent_wf[u_24] & 4294967295);
                                                    unsigned int magnitude_9 = bits_9 & 2147483647;
                                                    unsigned int m_9 = ((magnitude_9 != 0) ? bits_9 : zero_u);
                                                    unsigned int key32_9 = (((m_9 & sign_mask) != 0) ? ~m_9 : m_9 | sign_mask);
                                                    if (magnitude_9 > nan_floor) {
                                                        key32_9 = zero_u;
                                                    }
                                                    unsigned long long key64_9 = (unsigned long long)key32_9 << 32 | (unsigned long long)(unsigned int)kid_9;
                                                    unsigned long long key_9 = key64_9;
                                                    keep_wf = ((key_9 >= edge_j_1_0) ? 1 : 0);
                                                }
                                                unsigned int vote_4 = __ballot_sync(0xFFFFFFFF, keep_wf != 0);
                                                if (keep_wf != 0) {
                                                    long long slot_wf = row_base_j_1_0 + (long long)(front_w + (int)(unsigned int)__popc(vote_4 & lower_lanes));
                                                    float score_wf = 0.0f;
                                                    unsigned int bits_10 = (unsigned int)(ent_wf[u_24] >> 32);
                                                    score_wf = reinterpret_cast<float*>(&bits_10)[0];
                                                    int kid_10 = (int)(unsigned int)(ent_wf[u_24] & 4294967295);
                                                    Indices[slot_wf] = kid_10;
                                                    Scores[slot_wf] = score_wf;
                                                }
                                                front_w += (int)(unsigned int)__popc(vote_4);
                                            }
                                        }
                                    } else {
                                        int back_w = 0;
                                        #pragma unroll 1
                                        for (int e0_5 = mid_w; e0_5 < c_fin_1_0; e0_5 += 256) {
                                            unsigned long long ent_wb[8];
                                            #pragma unroll
                                            for (int u_25 = 0; u_25 < 8; u_25++) {
                                                ent_wb[u_25] = 0;
                                                if (c_fin_1_0 > e0_5 + lane_0 + 32 * u_25) {
                                                    ent_wb[u_25] = (unsigned long long)Cand[buf_pairs[0] + e0_5 + lane_0 + 32 * u_25];
                                                }
                                            }
                                            #pragma unroll
                                            for (int u_26 = 0; u_26 < 8; u_26++) {
                                                int keep_wb = 0;
                                                if (c_fin_1_0 > e0_5 + lane_0 + 32 * u_26) {
                                                    unsigned int bits_11 = (unsigned int)(ent_wb[u_26] >> 32);
                                                    int kid_11 = (int)(unsigned int)(ent_wb[u_26] & 4294967295);
                                                    unsigned int magnitude_10 = bits_11 & 2147483647;
                                                    unsigned int m_10 = ((magnitude_10 != 0) ? bits_11 : zero_u);
                                                    unsigned int key32_10 = (((m_10 & sign_mask) != 0) ? ~m_10 : m_10 | sign_mask);
                                                    if (magnitude_10 > nan_floor) {
                                                        key32_10 = zero_u;
                                                    }
                                                    unsigned long long key64_10 = (unsigned long long)key32_10 << 32 | (unsigned long long)(unsigned int)kid_11;
                                                    unsigned long long key_10 = key64_10;
                                                    keep_wb = ((key_10 >= edge_j_1_0) ? 1 : 0);
                                                }
                                                unsigned int vote_1_1 = __ballot_sync(0xFFFFFFFF, keep_wb != 0);
                                                if (keep_wb != 0) {
                                                    long long slot_wb = row_base_j_1_0 + (long long)(n_sel_j_2_0 - 1 - (back_w + (int)(unsigned int)__popc(vote_1_1 & lower_lanes)));
                                                    float score_wb = 0.0f;
                                                    unsigned int bits_12 = (unsigned int)(ent_wb[u_26] >> 32);
                                                    score_wb = reinterpret_cast<float*>(&bits_12)[0];
                                                    int kid_12 = (int)(unsigned int)(ent_wb[u_26] & 4294967295);
                                                    Indices[slot_wb] = kid_12;
                                                    Scores[slot_wb] = score_wb;
                                                }
                                                back_w += (int)(unsigned int)__popc(vote_1_1);
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int pad_w = n_sel_j_2_0 + lane_0; pad_w < top_k; pad_w += 32) {
                                            long long pad_slot_w = row_base_j_1_0 + (long long)pad_w;
                                            Indices[pad_slot_w] = -1;
                                            Scores[pad_slot_w] = -CUDART_INF_F;
                                        }
                                    }
                                }
                                int c_fin_1_1 = c_finals[1];
                                int n_sel_j_2_1 = ((top_k) < (c_fin_1_1) ? (top_k) : (c_fin_1_1));
                                unsigned long long edge_j_1_1 = 0;
                                if (c_fin_1_1 > top_k) {
                                    int k_rem_5 = top_k;
                                    int kept_above_5 = 0;
                                    unsigned long long prefix_5 = 0;
                                    unsigned long long edge_5 = 0;
                                    #pragma unroll 1
                                    for (int p_5 = 0; p_5 < 8; p_5++) {
                                        int shift_5 = 56 - 8 * p_5;
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
                                        unsigned long long ring_5[24];
                                        #pragma unroll
                                        for (int s_pro_5 = 0; s_pro_5 < 2; s_pro_5++) {
                                            #pragma unroll
                                            for (int u_27 = 0; u_27 < 8; u_27++) {
                                                int idx_l_10 = s_pro_5 * 512 + half * 32 + lane_0 + 64 * u_27;
                                                ring_5[s_pro_5 * 8 + u_27] = 0;
                                                if (idx_l_10 < c_fin_1_1) {
                                                    ring_5[s_pro_5 * 8 + u_27] = (unsigned long long)Cand[buf_pairs[1] + idx_l_10];
                                                }
                                            }
                                        }
                                        int nbat_5 = (c_fin_1_1 + 512 - 1) / 512;
                                        #pragma unroll 1
                                        for (int g_5 = 0; g_5 < nbat_5; g_5 += 3) {
                                            #pragma unroll
                                            for (int s_st_5 = 0; s_st_5 < 3; s_st_5++) {
                                                #pragma unroll
                                                for (int u_28 = 0; u_28 < 8; u_28++) {
                                                    int valid_p_5 = ((c_fin_1_1 > (g_5 + s_st_5) * 512 + half * 32 + lane_0 + 64 * u_28) ? 1 : 0);
                                                    unsigned int bits_13 = (unsigned int)(ring_5[s_st_5 * 8 + u_28] >> 32);
                                                    int kid_13 = (int)(unsigned int)(ring_5[s_st_5 * 8 + u_28] & 4294967295);
                                                    unsigned int magnitude_11 = bits_13 & 2147483647;
                                                    unsigned int m_11 = ((magnitude_11 != 0) ? bits_13 : zero_u);
                                                    unsigned int key32_11 = (((m_11 & sign_mask) != 0) ? ~m_11 : m_11 | sign_mask);
                                                    if (magnitude_11 > nan_floor) {
                                                        key32_11 = zero_u;
                                                    }
                                                    unsigned long long key64_11 = (unsigned long long)key32_11 << 32 | (unsigned long long)(unsigned int)kid_13;
                                                    unsigned long long key_11 = key64_11;
                                                    unsigned long long key_p_5 = key_11;
                                                    unsigned long long ks_p_5 = key_p_5 >> (unsigned long long)shift_5;
                                                    int digit_p_5 = (int)(ks_p_5 & 255);
                                                    int pm_p_5 = ((ks_p_5 >> 8 == prefix_5) ? 1 : 0);
                                                    int bin_p_5 = (((valid_p_5 & pm_p_5) != 0) ? hist_base + digit_p_5 : pair_dummy);
                                                    atomicAdd(&s_hist[bin_p_5], 1);
                                                }
                                                #pragma unroll
                                                for (int u_29 = 0; u_29 < 8; u_29++) {
                                                    int idx_l_11 = (g_5 + s_st_5 + 2) * 512 + half * 32 + lane_0 + 64 * u_29;
                                                    ring_5[(s_st_5 + 2) % 3 * 8 + u_29] = 0;
                                                    if (idx_l_11 < c_fin_1_1) {
                                                        ring_5[(s_st_5 + 2) % 3 * 8 + u_29] = (unsigned long long)Cand[buf_pairs[1] + idx_l_11];
                                                    }
                                                }
                                            }
                                        }
                                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                        int lane_bins_5[8];
                                        int lane_sum_5 = 0;
                                        lane_bins_5[0] = s_hist[hist_base + lane_0 * 8];
                                        lane_sum_5 += lane_bins_5[0];
                                        lane_bins_5[1] = s_hist[hist_base + lane_0 * 8 + 1];
                                        lane_sum_5 += lane_bins_5[1];
                                        lane_bins_5[2] = s_hist[hist_base + lane_0 * 8 + 2];
                                        lane_sum_5 += lane_bins_5[2];
                                        lane_bins_5[3] = s_hist[hist_base + lane_0 * 8 + 3];
                                        lane_sum_5 += lane_bins_5[3];
                                        lane_bins_5[4] = s_hist[hist_base + lane_0 * 8 + 4];
                                        lane_sum_5 += lane_bins_5[4];
                                        lane_bins_5[5] = s_hist[hist_base + lane_0 * 8 + 5];
                                        lane_sum_5 += lane_bins_5[5];
                                        lane_bins_5[6] = s_hist[hist_base + lane_0 * 8 + 6];
                                        lane_sum_5 += lane_bins_5[6];
                                        lane_bins_5[7] = s_hist[hist_base + lane_0 * 8 + 7];
                                        lane_sum_5 += lane_bins_5[7];
                                        int suffix_5 = lane_sum_5;
                                        int above_part_4_25 = __shfl_down_sync(0xFFFFFFFF, suffix_5, 1, 32);
                                        if (lane_0 + 1 < 32) {
                                            suffix_5 += above_part_4_25;
                                        }
                                        int above_part_4_26 = __shfl_down_sync(0xFFFFFFFF, suffix_5, 2, 32);
                                        if (lane_0 + 2 < 32) {
                                            suffix_5 += above_part_4_26;
                                        }
                                        int above_part_4_27 = __shfl_down_sync(0xFFFFFFFF, suffix_5, 4, 32);
                                        if (lane_0 + 4 < 32) {
                                            suffix_5 += above_part_4_27;
                                        }
                                        int above_part_4_28 = __shfl_down_sync(0xFFFFFFFF, suffix_5, 8, 32);
                                        if (lane_0 + 8 < 32) {
                                            suffix_5 += above_part_4_28;
                                        }
                                        int above_part_4_29 = __shfl_down_sync(0xFFFFFFFF, suffix_5, 16, 32);
                                        if (lane_0 + 16 < 32) {
                                            suffix_5 += above_part_4_29;
                                        }
                                        int excl_5 = suffix_5 - lane_sum_5;
                                        int is_target_5 = ((excl_5 < k_rem_5 && k_rem_5 <= excl_5 + lane_sum_5) ? 1 : 0);
                                        int d_sel_5 = 0;
                                        int above_sel_5 = 0;
                                        int count_sel_5 = 0;
                                        int found_5 = 0;
                                        int cum_above_5 = excl_5;
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[7]) {
                                                d_sel_5 = 7;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[7];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[7];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[6]) {
                                                d_sel_5 = 6;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[6];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[6];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[5]) {
                                                d_sel_5 = 5;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[5];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[5];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[4]) {
                                                d_sel_5 = 4;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[4];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[4];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[3]) {
                                                d_sel_5 = 3;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[3];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[3];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[2]) {
                                                d_sel_5 = 2;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[2];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[2];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[1]) {
                                                d_sel_5 = 1;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[1];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[1];
                                        if (found_5 == 0) {
                                            if (k_rem_5 <= cum_above_5 + lane_bins_5[0]) {
                                                d_sel_5 = 0;
                                                above_sel_5 = cum_above_5;
                                                count_sel_5 = lane_bins_5[0];
                                                found_5 = 1;
                                            }
                                        }
                                        cum_above_5 += lane_bins_5[0];
                                        int target_lane_5;
                                        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(target_lane_5) : "r"(is_target_5 * lane_0));
                                        int digit_sel_5 = __shfl_sync(0xFFFFFFFF, d_sel_5, target_lane_5);
                                        int above_cnt_5 = __shfl_sync(0xFFFFFFFF, above_sel_5, target_lane_5);
                                        int bucket_cnt_5 = __shfl_sync(0xFFFFFFFF, count_sel_5, target_lane_5);
                                        k_rem_5 = k_rem_5 - above_cnt_5;
                                        kept_above_5 += above_cnt_5;
                                        prefix_5 = prefix_5 << 8 | (unsigned long long)(unsigned int)(target_lane_5 * 8 + digit_sel_5);
                                        edge_5 = prefix_5 << (unsigned long long)shift_5;
                                        if (kept_above_5 + bucket_cnt_5 <= top_k) {
                                            break;
                                        }
                                    }
                                    edge_j_1_1 = edge_5;
                                }
                                int row_j_1_1 = q0_3 + u_base + slot_pairs[1];
                                long long row_base_j_1_1 = (long long)row_j_1_1 * (long long)top_k;
                                if (valid_pairs[1] != 0) {
                                    int mid_w_1 = (c_fin_1_1 / 2 + 31) / 32 * 32;
                                    if (mid_w_1 > c_fin_1_1) {
                                        mid_w_1 = c_fin_1_1;
                                    }
                                    if (half == 0) {
                                        int front_w_1 = 0;
                                        #pragma unroll 1
                                        for (int e0_6 = 0; e0_6 < mid_w_1; e0_6 += 256) {
                                            unsigned long long ent_wf_1[8];
                                            #pragma unroll
                                            for (int u_30 = 0; u_30 < 8; u_30++) {
                                                ent_wf_1[u_30] = 0;
                                                if (mid_w_1 > e0_6 + lane_0 + 32 * u_30) {
                                                    ent_wf_1[u_30] = (unsigned long long)Cand[buf_pairs[1] + e0_6 + lane_0 + 32 * u_30];
                                                }
                                            }
                                            #pragma unroll
                                            for (int u_31 = 0; u_31 < 8; u_31++) {
                                                int keep_wf_1 = 0;
                                                if (mid_w_1 > e0_6 + lane_0 + 32 * u_31) {
                                                    unsigned int bits_14 = (unsigned int)(ent_wf_1[u_31] >> 32);
                                                    int kid_14 = (int)(unsigned int)(ent_wf_1[u_31] & 4294967295);
                                                    unsigned int magnitude_12 = bits_14 & 2147483647;
                                                    unsigned int m_12 = ((magnitude_12 != 0) ? bits_14 : zero_u);
                                                    unsigned int key32_12 = (((m_12 & sign_mask) != 0) ? ~m_12 : m_12 | sign_mask);
                                                    if (magnitude_12 > nan_floor) {
                                                        key32_12 = zero_u;
                                                    }
                                                    unsigned long long key64_12 = (unsigned long long)key32_12 << 32 | (unsigned long long)(unsigned int)kid_14;
                                                    unsigned long long key_12 = key64_12;
                                                    keep_wf_1 = ((key_12 >= edge_j_1_1) ? 1 : 0);
                                                }
                                                unsigned int vote_5 = __ballot_sync(0xFFFFFFFF, keep_wf_1 != 0);
                                                if (keep_wf_1 != 0) {
                                                    long long slot_wf_1 = row_base_j_1_1 + (long long)(front_w_1 + (int)(unsigned int)__popc(vote_5 & lower_lanes));
                                                    float score_wf_1 = 0.0f;
                                                    unsigned int bits_15 = (unsigned int)(ent_wf_1[u_31] >> 32);
                                                    score_wf_1 = reinterpret_cast<float*>(&bits_15)[0];
                                                    int kid_15 = (int)(unsigned int)(ent_wf_1[u_31] & 4294967295);
                                                    Indices[slot_wf_1] = kid_15;
                                                    Scores[slot_wf_1] = score_wf_1;
                                                }
                                                front_w_1 += (int)(unsigned int)__popc(vote_5);
                                            }
                                        }
                                    } else {
                                        int back_w_1 = 0;
                                        #pragma unroll 1
                                        for (int e0_7 = mid_w_1; e0_7 < c_fin_1_1; e0_7 += 256) {
                                            unsigned long long ent_wb_1[8];
                                            #pragma unroll
                                            for (int u_32 = 0; u_32 < 8; u_32++) {
                                                ent_wb_1[u_32] = 0;
                                                if (c_fin_1_1 > e0_7 + lane_0 + 32 * u_32) {
                                                    ent_wb_1[u_32] = (unsigned long long)Cand[buf_pairs[1] + e0_7 + lane_0 + 32 * u_32];
                                                }
                                            }
                                            #pragma unroll
                                            for (int u_33 = 0; u_33 < 8; u_33++) {
                                                int keep_wb_1 = 0;
                                                if (c_fin_1_1 > e0_7 + lane_0 + 32 * u_33) {
                                                    unsigned int bits_16 = (unsigned int)(ent_wb_1[u_33] >> 32);
                                                    int kid_16 = (int)(unsigned int)(ent_wb_1[u_33] & 4294967295);
                                                    unsigned int magnitude_13 = bits_16 & 2147483647;
                                                    unsigned int m_13 = ((magnitude_13 != 0) ? bits_16 : zero_u);
                                                    unsigned int key32_13 = (((m_13 & sign_mask) != 0) ? ~m_13 : m_13 | sign_mask);
                                                    if (magnitude_13 > nan_floor) {
                                                        key32_13 = zero_u;
                                                    }
                                                    unsigned long long key64_13 = (unsigned long long)key32_13 << 32 | (unsigned long long)(unsigned int)kid_16;
                                                    unsigned long long key_13 = key64_13;
                                                    keep_wb_1 = ((key_13 >= edge_j_1_1) ? 1 : 0);
                                                }
                                                unsigned int vote_1_2 = __ballot_sync(0xFFFFFFFF, keep_wb_1 != 0);
                                                if (keep_wb_1 != 0) {
                                                    long long slot_wb_1 = row_base_j_1_1 + (long long)(n_sel_j_2_1 - 1 - (back_w_1 + (int)(unsigned int)__popc(vote_1_2 & lower_lanes)));
                                                    float score_wb_1 = 0.0f;
                                                    unsigned int bits_17 = (unsigned int)(ent_wb_1[u_33] >> 32);
                                                    score_wb_1 = reinterpret_cast<float*>(&bits_17)[0];
                                                    int kid_17 = (int)(unsigned int)(ent_wb_1[u_33] & 4294967295);
                                                    Indices[slot_wb_1] = kid_17;
                                                    Scores[slot_wb_1] = score_wb_1;
                                                }
                                                back_w_1 += (int)(unsigned int)__popc(vote_1_2);
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int pad_w_1 = n_sel_j_2_1 + lane_0; pad_w_1 < top_k; pad_w_1 += 32) {
                                            long long pad_slot_w_1 = row_base_j_1_1 + (long long)pad_w_1;
                                            Indices[pad_slot_w_1] = -1;
                                            Scores[pad_slot_w_1] = -CUDART_INF_F;
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
    // ---- Role: gate ----
    } else if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // gate_main
            int g_6 = (int)warp - 12;
            int lane_g = lane;
            int g_wg = g_6 / 2;
            int g_col = g_6 % 2 * 8;
            unsigned int gate_sc = 0;
            int bid_g = (int)bid;
            unsigned int tk_limit_g = (unsigned int)s_tk[2];
            int n_tot_g = s_tk[3];
            int g_units_g = grid_ctas;
            int tk_g = s_tk[0];
            int tkp_g = 1;
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            unsigned int zero_u_g = 0;
            unsigned int sign_mask_g = 2147483648;
            unsigned int nan_floor_g = 2139095040;
            unsigned int one_u_g = 1;
            unsigned int lower_lanes_g = (one_u_g << (unsigned int)lane) - 1;
            int keep_cap_g = ((top_k + 64) < (cand_cap - 128 * check_period) ? (top_k + 64) : (cand_cap - 128 * check_period));
            int trigger_room_g = 128 * check_period;
            unsigned int gate_rq = 0;
            int unit_base_g = 0;
            unsigned int _phase_scores_full = 0;
            unsigned int _phase_comp_done = 0;
            unsigned int _phase_verdict_ack = 1;
            #pragma unroll 1
            for (int seg_4 = 0; seg_4 < num_segments; seg_4++) {
                int q0_g = cu_seqlens_q[seg_4];
                int lq_g = cu_seqlens_q[seg_4 + 1] - q0_g;
                int lk_g = cu_seqlens_k[seg_4 + 1] - cu_seqlens_k[seg_4];
                int nb_g = (lq_g + 8 - 1) / 8;
#if __CUDA_ARCH__ == 1000
#else
                int q0_e_4 = cu_seqlens_q[seg_4];
                int lq_e_4 = cu_seqlens_q[seg_4 + 1] - q0_e_4;
                int lk_e_4 = cu_seqlens_k[seg_4 + 1] - cu_seqlens_k[seg_4];
                long long off_e_4 = 0;
                if (has_offsets != 0) {
                    off_e_4 = q_offsets[seg_4];
                } else if (ratio == 1) {
                    off_e_4 = (long long)lk_e_4 - (long long)lq_e_4;
                }
                int last_e_4 = ((7) < (lq_e_4 - 1) ? (7) : (lq_e_4 - 1));
                long long num_e_4 = off_e_4 + (long long)last_e_4 + 1;
                int vis_e_4 = 0;
                if (num_e_4 > 0) {
                    long long quot_e_8 = num_e_4 / (long long)ratio;
                    long long lk64_e_8 = (long long)lk_e_4;
                    long long bnd_e_8 = ((quot_e_8 < lk64_e_8) ? quot_e_8 : lk64_e_8);
                    vis_e_4 = (int)bnd_e_8;
                }
                int tiles_e_4 = (vis_e_4 + 128 - 1) / 128;
                int t_lo_g = tiles_e_4;
                int q0_e_4_1 = cu_seqlens_q[seg_4];
                int lq_e_4_1 = cu_seqlens_q[seg_4 + 1] - q0_e_4_1;
                int lk_e_4_1 = cu_seqlens_k[seg_4 + 1] - cu_seqlens_k[seg_4];
                long long off_e_4_1 = 0;
                if (has_offsets != 0) {
                    off_e_4_1 = q_offsets[seg_4];
                } else if (ratio == 1) {
                    off_e_4_1 = (long long)lk_e_4_1 - (long long)lq_e_4_1;
                }
                int last_e_4_1 = (((nb_g - 1) * 8 + 7) < (lq_e_4_1 - 1) ? ((nb_g - 1) * 8 + 7) : (lq_e_4_1 - 1));
                long long num_e_4_1 = off_e_4_1 + (long long)last_e_4_1 + 1;
                int vis_e_4_1 = 0;
                if (num_e_4_1 > 0) {
                    long long quot_e_9 = num_e_4_1 / (long long)ratio;
                    long long lk64_e_9 = (long long)lk_e_4_1;
                    long long bnd_e_9 = ((quot_e_9 < lk64_e_9) ? quot_e_9 : lk64_e_9);
                    vis_e_4_1 = (int)bnd_e_9;
                }
                int tiles_e_4_1 = (vis_e_4_1 + 128 - 1) / 128;
                int t_hi_g = tiles_e_4_1;
                int near_eq_g = ((t_hi_g * 4 <= t_lo_g * 5) ? 1 : 0);
#endif
                long long off_4 = 0;
                if (has_offsets != 0) {
                    off_4 = q_offsets[seg_4];
                } else if (ratio == 1) {
                    off_4 = (long long)lk_g - (long long)lq_g;
                }
                long long off_g = off_4;
                #pragma unroll 1
                for (int rnd_4 = 0; rnd_4 < nb_g + 1; rnd_4++) {
                    int unit_4 = tk_g;
                    if (unit_4 >= unit_base_g + nb_g) {
                        break;
                    }
                    tkp_g ^= 1;
                    int _min_0_2 = ((unit_4 - unit_base_g) < (unit_base_g + nb_g - 1 - unit_4) ? (unit_4 - unit_base_g) : (unit_base_g + nb_g - 1 - unit_4));
                    if (_min_0_2 >= 0) {
                        int blk_4 = nb_g - 1 - (unit_4 - unit_base_g);
#if __CUDA_ARCH__ == 1000
#else
                        if (sample_tiles_max != 0) {
                            if (near_eq_g != 0) {
                                blk_4 = unit_4 - unit_base_g;
                            }
                        }
#endif
                        int u_base_g = blk_4 * 8;
                        int q0_4 = cu_seqlens_q[seg_4];
                        int lq_4 = cu_seqlens_q[seg_4 + 1] - q0_4;
                        int lk_3 = cu_seqlens_k[seg_4 + 1] - cu_seqlens_k[seg_4];
                        long long off_2_1 = 0;
                        if (has_offsets != 0) {
                            off_2_1 = q_offsets[seg_4];
                        } else if (ratio == 1) {
                            off_2_1 = (long long)lk_3 - (long long)lq_4;
                        }
                        long long off_1_3 = off_2_1;
                        int last_u_3 = ((blk_4 * 8 + 7) < (lq_4 - 1) ? (blk_4 * 8 + 7) : (lq_4 - 1));
                        long long num_3 = off_1_3 + (long long)last_u_3 + 1;
                        int vis_3 = 0;
                        if (num_3 > 0) {
                            long long quotient_3 = num_3 / (long long)ratio;
                            long long lk64_3 = (long long)lk_3;
                            long long bounded_3 = ((quotient_3 < lk64_3) ? quotient_3 : lk64_3);
                            vis_3 = (int)bounded_3;
                        }
                        int v_max_3 = vis_3;
                        int n_tiles_4 = (v_max_3 + 128 - 1) / 128;
                        int n_tiles_2_1 = n_tiles_4;
                        int max_tiles_2 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
                        int stride_2 = 1;
                        int n_sample_2 = 0;
                        if (max_tiles_2 >= 2) {
                            if (n_tiles_2_1 * 128 + 128 > cand_cap) {
                                stride_2 = (n_tiles_2_1 + max_tiles_2 - 1) / max_tiles_2;
                                if (stride_2 >= 2) {
                                    n_sample_2 = (n_tiles_2_1 + stride_2 - 1) / stride_2;
                                }
                            }
                        }
                        if (max_tiles_2 >= 2) {
                            if (n_tiles_2_1 * 128 + 128 <= cand_cap) {
                                if (n_tiles_2_1 * 64 > top_k) {
                                    int cap_fit_2 = ((max_tiles_2) < (8) ? (max_tiles_2) : (8));
                                    if (cap_fit_2 >= 2) {
                                        stride_2 = (n_tiles_2_1 + cap_fit_2 - 1) / cap_fit_2;
                                        if (stride_2 >= 2) {
                                            n_sample_2 = (n_tiles_2_1 + stride_2 - 1) / stride_2;
                                        }
                                    }
                                }
                            }
                        }
                        int valid_g[2];
                        int vis_g[2];
                        int buf_g[2];
                        int buf_unit_g = (bid_g * 8 + 2 * g_6) * cand_cap;
                        int u_qj = u_base_g + 2 * g_6;
                        int valid_j = ((u_qj < lq_g) ? 1 : 0);
                        valid_g[0] = valid_j;
                        int vis_j = 0;
                        if (valid_j != 0) {
                            long long num_0 = off_g + (long long)u_qj + 1;
                            int vis_0 = 0;
                            if (num_0 > 0) {
                                long long quotient_4 = num_0 / (long long)ratio;
                                long long lk64_4 = (long long)lk_g;
                                long long bounded_4 = ((quotient_4 < lk64_4) ? quotient_4 : lk64_4);
                                vis_0 = (int)bounded_4;
                            }
                            vis_j = vis_0;
                        }
                        vis_g[0] = vis_j;
                        buf_g[0] = buf_unit_g;
                        int u_qj_3 = u_base_g + 2 * g_6 + 1;
                        int valid_j_4 = ((u_qj_3 < lq_g) ? 1 : 0);
                        valid_g[1] = valid_j_4;
                        int vis_j_5 = 0;
                        if (valid_j_4 != 0) {
                            long long num_1_1 = off_g + (long long)u_qj_3 + 1;
                            int vis_2_1 = 0;
                            if (num_1_1 > 0) {
                                long long quotient_5 = num_1_1 / (long long)ratio;
                                long long lk64_5 = (long long)lk_g;
                                long long bounded_5 = ((quotient_5 < lk64_5) ? quotient_5 : lk64_5);
                                vis_2_1 = (int)bounded_5;
                            }
                            vis_j_5 = vis_2_1;
                        }
                        vis_g[1] = vis_j_5;
                        buf_g[1] = buf_unit_g + cand_cap;
                        int drawn_g = 0;
                        if (warp == 12) {
                            if (lane == 0) {
                                drawn_g = n_tot_g;
                                if (n_tot_g > g_units_g) {
                                    uint32_t _atomic_fetch_inc_1;
                                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                                        : "=r"(_atomic_fetch_inc_1) : "l"(reinterpret_cast<unsigned int*>(Cand + ((long long)grid_ctas * (long long)(8 * cand_cap)))), "r"(static_cast<uint32_t>(tk_limit_g)) : "memory");
                                    drawn_g = (int)_atomic_fetch_inc_1;
                                }
                            }
                        }
                        int next_g = __shfl_sync(0xFFFFFFFF, drawn_g, 0);
                        #pragma unroll 1
                        for (int attempt_4 = 0; attempt_4 < 2; attempt_4++) {
                            int n_s_1 = n_sample_2;
                            if (attempt_4 != 0) {
                                n_s_1 = 0;
                            }
                            unsigned long long tau_g[2];
                            int cnt_g[2];
                            tau_g[0] = 0;
                            cnt_g[0] = 0;
                            tau_g[1] = 0;
                            cnt_g[1] = 0;
                            int d_pos = 0;
                            int r_pos = 0;
                            int until_check_1 = check_period - 1;
                            int ti0_1 = 0;
                            while (ti0_1 < n_tiles_2_1) {
                                int run_c_1 = ((until_check_1 + 1) < (n_tiles_2_1 - ti0_1) ? (until_check_1 + 1) : (n_tiles_2_1 - ti0_1));
                                if (n_s_1 > ti0_1) {
                                    int _min_2_1 = ((run_c_1) < (n_s_1 - ti0_1) ? (run_c_1) : (n_s_1 - ti0_1));
                                    run_c_1 = _min_2_1;
                                }
                                #pragma unroll 1
                                for (int ti_4 = ti0_1; ti_4 < ti0_1 + run_c_1; ti_4++) {
                                    int tile_1 = d_pos;
#if __CUDA_ARCH__ == 1000
                                    tile_1 = n_tiles_2_1 - 1 - d_pos;
#endif
                                    int nxt_w = ti_4 + 1;
                                    int r_inc_w = r_pos + 1;
                                    int wrap_w = ((r_inc_w == stride_2 - 1) ? 1 : 0);
                                    int d_gen_w = d_pos + 1 + wrap_w;
                                    int r_gen_w = ((wrap_w != 0) ? 0 : r_inc_w);
                                    int d_new_w = ((nxt_w < n_s_1) ? d_pos + stride_2 : ((nxt_w == n_s_1) ? 1 : d_gen_w));
                                    int r_new_w = ((nxt_w < n_s_1) ? r_pos : ((nxt_w == n_s_1) ? 0 : r_gen_w));
                                    d_pos = ((n_s_1 > 0) ? d_new_w : d_pos + 1);
                                    r_pos = ((n_s_1 > 0) ? r_new_w : r_pos);
                                    mbarrier_wait(scores_full_addr + (gate_sc) * 8, _phase_scores_full);
                                    unsigned int bq[8];
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&bq[0])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(0) + 1]))
                                        : "r"((s_scores_addr + gate_sc * 4096 + (unsigned int)((g_wg * 128 + lane_g) * 16 + g_col))));
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&bq[2])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(2) + 1]))
                                        : "r"((s_scores_addr + gate_sc * 4096 + (unsigned int)((g_wg * 128 + 32 + lane_g) * 16 + g_col))));
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&bq[4])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(4) + 1]))
                                        : "r"((s_scores_addr + gate_sc * 4096 + (unsigned int)((g_wg * 128 + 64 + lane_g) * 16 + g_col))));
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&bq[6])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(6) + 1]))
                                        : "r"((s_scores_addr + gate_sc * 4096 + (unsigned int)((g_wg * 128 + 96 + lane_g) * 16 + g_col))));
                                    mbarrier_arrive(scores_empty_addr + (gate_sc) * 8);
                                    gate_sc += 1;
                                    if (gate_sc == 3) { gate_sc = 0; _phase_scores_full ^= 1; }
                                    int kid_c_1_0 = tile_1 * 128 + lane_g;
                                    unsigned int bits_c_1_0 = bq[0];
                                    unsigned int magnitude_14 = bits_c_1_0 & 2147483647;
                                    unsigned int m_14 = ((magnitude_14 != 0) ? bits_c_1_0 : zero_u_g);
                                    unsigned int key32_14 = (((m_14 & sign_mask_g) != 0) ? ~m_14 : m_14 | sign_mask_g);
                                    if (magnitude_14 > nan_floor_g) {
                                        key32_14 = zero_u_g;
                                    }
                                    unsigned long long key64_14 = (unsigned long long)key32_14 << 32 | (unsigned long long)(unsigned int)kid_c_1_0;
                                    unsigned long long key_c_1_0 = key64_14;
                                    int in_prefix_c_1_0 = ((kid_c_1_0 < vis_g[0]) ? 1 : 0);
                                    int above_c_1_0 = ((key_c_1_0 >= tau_g[0]) ? 1 : 0);
                                    int emit_c_1_0 = valid_g[0] & in_prefix_c_1_0 & above_c_1_0;
                                    unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, emit_c_1_0 != 0);
                                    int emit_base_c_1_0 = make_warp_uniform(buf_g[0] + cnt_g[0]);
                                    if (emit_c_1_0 != 0) {
                                        int slot_c_1_0 = emit_base_c_1_0 + (int)(unsigned int)__popc(_vote_1 & lower_lanes_g);
                                        unsigned long long entry = (unsigned long long)bits_c_1_0 << 32 | (unsigned long long)(unsigned int)kid_c_1_0;
                                        Cand[slot_c_1_0] = (long long)entry;
                                    }
                                    cnt_g[0] = cnt_g[0] + (int)(unsigned int)__popc(_vote_1);
                                    unsigned int bits_c_1_1 = bq[1];
                                    unsigned int magnitude_0 = bits_c_1_1 & 2147483647;
                                    unsigned int m_0 = ((magnitude_0 != 0) ? bits_c_1_1 : zero_u_g);
                                    unsigned int key32_0 = (((m_0 & sign_mask_g) != 0) ? ~m_0 : m_0 | sign_mask_g);
                                    if (magnitude_0 > nan_floor_g) {
                                        key32_0 = zero_u_g;
                                    }
                                    unsigned long long key64_0 = (unsigned long long)key32_0 << 32 | (unsigned long long)(unsigned int)kid_c_1_0;
                                    unsigned long long key_c_1_1 = key64_0;
                                    int in_prefix_c_1_1 = ((kid_c_1_0 < vis_g[1]) ? 1 : 0);
                                    int above_c_1_1 = ((key_c_1_1 >= tau_g[1]) ? 1 : 0);
                                    int emit_c_1_1 = valid_g[1] & in_prefix_c_1_1 & above_c_1_1;
                                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, emit_c_1_1 != 0);
                                    int emit_base_c_1_1 = make_warp_uniform(buf_g[1] + cnt_g[1]);
                                    if (emit_c_1_1 != 0) {
                                        int slot_c_1_1 = emit_base_c_1_1 + (int)(unsigned int)__popc(_vote_2 & lower_lanes_g);
                                        unsigned long long entry_1 = (unsigned long long)bits_c_1_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_0;
                                        Cand[slot_c_1_1] = (long long)entry_1;
                                    }
                                    cnt_g[1] = cnt_g[1] + (int)(unsigned int)__popc(_vote_2);
                                    int kid_c_1_1 = tile_1 * 128 + 32 + lane_g;
                                    unsigned int bits_c_1_2 = bq[2];
                                    unsigned int magnitude_1_1 = bits_c_1_2 & 2147483647;
                                    unsigned int m_1_1 = ((magnitude_1_1 != 0) ? bits_c_1_2 : zero_u_g);
                                    unsigned int key32_1_1 = (((m_1_1 & sign_mask_g) != 0) ? ~m_1_1 : m_1_1 | sign_mask_g);
                                    if (magnitude_1_1 > nan_floor_g) {
                                        key32_1_1 = zero_u_g;
                                    }
                                    unsigned long long key64_1_1 = (unsigned long long)key32_1_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_1;
                                    unsigned long long key_c_1_2 = key64_1_1;
                                    int in_prefix_c_1_2 = ((kid_c_1_1 < vis_g[0]) ? 1 : 0);
                                    int above_c_1_2 = ((key_c_1_2 >= tau_g[0]) ? 1 : 0);
                                    int emit_c_1_2 = valid_g[0] & in_prefix_c_1_2 & above_c_1_2;
                                    unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, emit_c_1_2 != 0);
                                    int emit_base_c_1_2 = make_warp_uniform(buf_g[0] + cnt_g[0]);
                                    if (emit_c_1_2 != 0) {
                                        int slot_c_1_2 = emit_base_c_1_2 + (int)(unsigned int)__popc(_vote_3 & lower_lanes_g);
                                        unsigned long long entry_2 = (unsigned long long)bits_c_1_2 << 32 | (unsigned long long)(unsigned int)kid_c_1_1;
                                        Cand[slot_c_1_2] = (long long)entry_2;
                                    }
                                    cnt_g[0] = cnt_g[0] + (int)(unsigned int)__popc(_vote_3);
                                    unsigned int bits_c_1_3 = bq[3];
                                    unsigned int magnitude_2_1 = bits_c_1_3 & 2147483647;
                                    unsigned int m_2_1 = ((magnitude_2_1 != 0) ? bits_c_1_3 : zero_u_g);
                                    unsigned int key32_2_1 = (((m_2_1 & sign_mask_g) != 0) ? ~m_2_1 : m_2_1 | sign_mask_g);
                                    if (magnitude_2_1 > nan_floor_g) {
                                        key32_2_1 = zero_u_g;
                                    }
                                    unsigned long long key64_2_1 = (unsigned long long)key32_2_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_1;
                                    unsigned long long key_c_1_3 = key64_2_1;
                                    int in_prefix_c_1_3 = ((kid_c_1_1 < vis_g[1]) ? 1 : 0);
                                    int above_c_1_3 = ((key_c_1_3 >= tau_g[1]) ? 1 : 0);
                                    int emit_c_1_3 = valid_g[1] & in_prefix_c_1_3 & above_c_1_3;
                                    unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, emit_c_1_3 != 0);
                                    int emit_base_c_1_3 = make_warp_uniform(buf_g[1] + cnt_g[1]);
                                    if (emit_c_1_3 != 0) {
                                        int slot_c_1_3 = emit_base_c_1_3 + (int)(unsigned int)__popc(_vote_4 & lower_lanes_g);
                                        unsigned long long entry_3 = (unsigned long long)bits_c_1_3 << 32 | (unsigned long long)(unsigned int)kid_c_1_1;
                                        Cand[slot_c_1_3] = (long long)entry_3;
                                    }
                                    cnt_g[1] = cnt_g[1] + (int)(unsigned int)__popc(_vote_4);
                                    int kid_c_1_2 = tile_1 * 128 + 64 + lane_g;
                                    unsigned int bits_c_1_4 = bq[4];
                                    unsigned int magnitude_3_1 = bits_c_1_4 & 2147483647;
                                    unsigned int m_3_1 = ((magnitude_3_1 != 0) ? bits_c_1_4 : zero_u_g);
                                    unsigned int key32_3_1 = (((m_3_1 & sign_mask_g) != 0) ? ~m_3_1 : m_3_1 | sign_mask_g);
                                    if (magnitude_3_1 > nan_floor_g) {
                                        key32_3_1 = zero_u_g;
                                    }
                                    unsigned long long key64_3_1 = (unsigned long long)key32_3_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_2;
                                    unsigned long long key_c_1_4 = key64_3_1;
                                    int in_prefix_c_1_4 = ((kid_c_1_2 < vis_g[0]) ? 1 : 0);
                                    int above_c_1_4 = ((key_c_1_4 >= tau_g[0]) ? 1 : 0);
                                    int emit_c_1_4 = valid_g[0] & in_prefix_c_1_4 & above_c_1_4;
                                    unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, emit_c_1_4 != 0);
                                    int emit_base_c_1_4 = make_warp_uniform(buf_g[0] + cnt_g[0]);
                                    if (emit_c_1_4 != 0) {
                                        int slot_c_1_4 = emit_base_c_1_4 + (int)(unsigned int)__popc(_vote_5 & lower_lanes_g);
                                        unsigned long long entry_4 = (unsigned long long)bits_c_1_4 << 32 | (unsigned long long)(unsigned int)kid_c_1_2;
                                        Cand[slot_c_1_4] = (long long)entry_4;
                                    }
                                    cnt_g[0] = cnt_g[0] + (int)(unsigned int)__popc(_vote_5);
                                    unsigned int bits_c_1_5 = bq[5];
                                    unsigned int magnitude_4_1 = bits_c_1_5 & 2147483647;
                                    unsigned int m_4_1 = ((magnitude_4_1 != 0) ? bits_c_1_5 : zero_u_g);
                                    unsigned int key32_4_1 = (((m_4_1 & sign_mask_g) != 0) ? ~m_4_1 : m_4_1 | sign_mask_g);
                                    if (magnitude_4_1 > nan_floor_g) {
                                        key32_4_1 = zero_u_g;
                                    }
                                    unsigned long long key64_4_1 = (unsigned long long)key32_4_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_2;
                                    unsigned long long key_c_1_5 = key64_4_1;
                                    int in_prefix_c_1_5 = ((kid_c_1_2 < vis_g[1]) ? 1 : 0);
                                    int above_c_1_5 = ((key_c_1_5 >= tau_g[1]) ? 1 : 0);
                                    int emit_c_1_5 = valid_g[1] & in_prefix_c_1_5 & above_c_1_5;
                                    unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, emit_c_1_5 != 0);
                                    int emit_base_c_1_5 = make_warp_uniform(buf_g[1] + cnt_g[1]);
                                    if (emit_c_1_5 != 0) {
                                        int slot_c_1_5 = emit_base_c_1_5 + (int)(unsigned int)__popc(_vote_6 & lower_lanes_g);
                                        unsigned long long entry_5 = (unsigned long long)bits_c_1_5 << 32 | (unsigned long long)(unsigned int)kid_c_1_2;
                                        Cand[slot_c_1_5] = (long long)entry_5;
                                    }
                                    cnt_g[1] = cnt_g[1] + (int)(unsigned int)__popc(_vote_6);
                                    int kid_c_1_3 = tile_1 * 128 + 96 + lane_g;
                                    unsigned int bits_c_1_6 = bq[6];
                                    unsigned int magnitude_5_1 = bits_c_1_6 & 2147483647;
                                    unsigned int m_5_1 = ((magnitude_5_1 != 0) ? bits_c_1_6 : zero_u_g);
                                    unsigned int key32_5_1 = (((m_5_1 & sign_mask_g) != 0) ? ~m_5_1 : m_5_1 | sign_mask_g);
                                    if (magnitude_5_1 > nan_floor_g) {
                                        key32_5_1 = zero_u_g;
                                    }
                                    unsigned long long key64_5_1 = (unsigned long long)key32_5_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_3;
                                    unsigned long long key_c_1_6 = key64_5_1;
                                    int in_prefix_c_1_6 = ((kid_c_1_3 < vis_g[0]) ? 1 : 0);
                                    int above_c_1_6 = ((key_c_1_6 >= tau_g[0]) ? 1 : 0);
                                    int emit_c_1_6 = valid_g[0] & in_prefix_c_1_6 & above_c_1_6;
                                    unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, emit_c_1_6 != 0);
                                    int emit_base_c_1_6 = make_warp_uniform(buf_g[0] + cnt_g[0]);
                                    if (emit_c_1_6 != 0) {
                                        int slot_c_1_6 = emit_base_c_1_6 + (int)(unsigned int)__popc(_vote_7 & lower_lanes_g);
                                        unsigned long long entry_6 = (unsigned long long)bits_c_1_6 << 32 | (unsigned long long)(unsigned int)kid_c_1_3;
                                        Cand[slot_c_1_6] = (long long)entry_6;
                                    }
                                    cnt_g[0] = cnt_g[0] + (int)(unsigned int)__popc(_vote_7);
                                    unsigned int bits_c_1_7 = bq[7];
                                    unsigned int magnitude_6_1 = bits_c_1_7 & 2147483647;
                                    unsigned int m_6_1 = ((magnitude_6_1 != 0) ? bits_c_1_7 : zero_u_g);
                                    unsigned int key32_6_1 = (((m_6_1 & sign_mask_g) != 0) ? ~m_6_1 : m_6_1 | sign_mask_g);
                                    if (magnitude_6_1 > nan_floor_g) {
                                        key32_6_1 = zero_u_g;
                                    }
                                    unsigned long long key64_6_1 = (unsigned long long)key32_6_1 << 32 | (unsigned long long)(unsigned int)kid_c_1_3;
                                    unsigned long long key_c_1_7 = key64_6_1;
                                    int in_prefix_c_1_7 = ((kid_c_1_3 < vis_g[1]) ? 1 : 0);
                                    int above_c_1_7 = ((key_c_1_7 >= tau_g[1]) ? 1 : 0);
                                    int emit_c_1_7 = valid_g[1] & in_prefix_c_1_7 & above_c_1_7;
                                    unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, emit_c_1_7 != 0);
                                    int emit_base_c_1_7 = make_warp_uniform(buf_g[1] + cnt_g[1]);
                                    if (emit_c_1_7 != 0) {
                                        int slot_c_1_7 = emit_base_c_1_7 + (int)(unsigned int)__popc(_vote_8 & lower_lanes_g);
                                        unsigned long long entry_7 = (unsigned long long)bits_c_1_7 << 32 | (unsigned long long)(unsigned int)kid_c_1_3;
                                        Cand[slot_c_1_7] = (long long)entry_7;
                                    }
                                    cnt_g[1] = cnt_g[1] + (int)(unsigned int)__popc(_vote_8);
                                }
                                ti0_1 = ti0_1 + run_c_1;
                                int sample_c_1 = ((ti0_1 == n_s_1) ? 1 : 0);
                                if (run_c_1 == until_check_1 + 1 || sample_c_1 != 0) {
                                    until_check_1 = check_period - 1;
                                    int need_l[2];
                                    int kw_l[2];
                                    int km_l[2];
                                    int sk_l[2];
                                    int c_q_1_0 = cnt_g[0];
                                    int limit_q_2_0 = ((tau_g[0] != 0) ? cand_cap : first_cap);
                                    int need_q_1_0 = ((limit_q_2_0 < c_q_1_0 + trigger_room_g) ? 1 : 0);
                                    int kw_q_1_0 = top_k;
                                    int km_q_1_0 = keep_cap_g;
                                    int sk_q_1_0 = 0;
                                    if (need_q_1_0 == 0) {
                                        if (sample_c_1 != 0) {
                                            if (valid_g[0] != 0) {
#if __CUDA_ARCH__ == 1000
                                                int _max_3 = ((vis_g[0] - (n_tiles_2_1 - 1) * 128) > (0) ? (vis_g[0] - (n_tiles_2_1 - 1) * 128) : (0));
                                                int n_diag_1_0 = ((_max_3) < (128) ? (_max_3) : (128));
                                                int c_far_1_0 = c_q_1_0 - n_diag_1_0;
                                                int v_far_1_0 = vis_g[0] - n_diag_1_0;
#else
                                                int t_near_1_0 = (n_s_1 - 1) * stride_2;
                                                int _max_3 = ((vis_g[0] - t_near_1_0 * 128) > (0) ? (vis_g[0] - t_near_1_0 * 128) : (0));
                                                int n_near_1_0 = ((_max_3) < (128) ? (_max_3) : (128));
                                                int c_far_1_0 = c_q_1_0 - n_near_1_0;
                                                int v_far_1_0 = vis_g[0] - n_near_1_0;
#endif
                                                if (v_far_1_0 > 0) {
                                                    int r_q_1_0 = (top_k * c_far_1_0 + v_far_1_0 - 1) / v_far_1_0;
                                                    int r_shift_1_0 = r_q_1_0 + r_q_1_0 * sample_shift_permille / 1000 + 24;
                                                    if (r_shift_1_0 < c_far_1_0) {
                                                        need_q_1_0 = 1;
                                                        kw_q_1_0 = r_shift_1_0;
                                                        int _min_4_1 = ((r_shift_1_0 + 64) < (keep_cap_g) ? (r_shift_1_0 + 64) : (keep_cap_g));
                                                        km_q_1_0 = _min_4_1;
#if __CUDA_ARCH__ == 1000
                                                        sk_q_1_0 = n_diag_1_0;
#else
                                                        sk_q_1_0 = n_near_1_0;
#endif
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need_l[0] = need_q_1_0;
                                    kw_l[0] = kw_q_1_0;
                                    km_l[0] = km_q_1_0;
                                    sk_l[0] = sk_q_1_0;
                                    int c_q_1_1 = cnt_g[1];
                                    int limit_q_2_1 = ((tau_g[1] != 0) ? cand_cap : first_cap);
                                    int need_q_1_1 = ((limit_q_2_1 < c_q_1_1 + trigger_room_g) ? 1 : 0);
                                    int kw_q_1_1 = top_k;
                                    int km_q_1_1 = keep_cap_g;
                                    int sk_q_1_1 = 0;
                                    if (need_q_1_1 == 0) {
                                        if (sample_c_1 != 0) {
                                            if (valid_g[1] != 0) {
#if __CUDA_ARCH__ == 1000
                                                int _max_5 = ((vis_g[1] - (n_tiles_2_1 - 1) * 128) > (0) ? (vis_g[1] - (n_tiles_2_1 - 1) * 128) : (0));
                                                int n_diag_1_1 = ((_max_5) < (128) ? (_max_5) : (128));
                                                int c_far_1_1 = c_q_1_1 - n_diag_1_1;
                                                int v_far_1_1 = vis_g[1] - n_diag_1_1;
#else
                                                int t_near_1_1 = (n_s_1 - 1) * stride_2;
                                                int _max_5 = ((vis_g[1] - t_near_1_1 * 128) > (0) ? (vis_g[1] - t_near_1_1 * 128) : (0));
                                                int n_near_1_1 = ((_max_5) < (128) ? (_max_5) : (128));
                                                int c_far_1_1 = c_q_1_1 - n_near_1_1;
                                                int v_far_1_1 = vis_g[1] - n_near_1_1;
#endif
                                                if (v_far_1_1 > 0) {
                                                    int r_q_1_1 = (top_k * c_far_1_1 + v_far_1_1 - 1) / v_far_1_1;
                                                    int r_shift_1_1 = r_q_1_1 + r_q_1_1 * sample_shift_permille / 1000 + 24;
                                                    if (r_shift_1_1 < c_far_1_1) {
                                                        need_q_1_1 = 1;
                                                        kw_q_1_1 = r_shift_1_1;
                                                        int _min_6 = ((r_shift_1_1 + 64) < (keep_cap_g) ? (r_shift_1_1 + 64) : (keep_cap_g));
                                                        km_q_1_1 = _min_6;
#if __CUDA_ARCH__ == 1000
                                                        sk_q_1_1 = n_diag_1_1;
#else
                                                        sk_q_1_1 = n_near_1_1;
#endif
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    need_l[1] = need_q_1_1;
                                    kw_l[1] = kw_q_1_1;
                                    km_l[1] = km_q_1_1;
                                    sk_l[1] = sk_q_1_1;
                                    int rq_base_g = (int)gate_rq * 40;
                                    __syncwarp();
                                    if (lane == 0) {
                                        int rq_gj_1_0 = rq_base_g + 2 * g_6 * 5;
                                        s_req[rq_gj_1_0] = need_l[0];
                                        s_req[rq_gj_1_0 + 1] = kw_l[0];
                                        s_req[rq_gj_1_0 + 2] = km_l[0];
                                        s_req[rq_gj_1_0 + 3] = sk_l[0];
                                        s_req[rq_gj_1_0 + 4] = cnt_g[0];
                                        int rq_gj_1_1 = rq_base_g + (2 * g_6 + 1) * 5;
                                        s_req[rq_gj_1_1] = need_l[1];
                                        s_req[rq_gj_1_1 + 1] = kw_l[1];
                                        s_req[rq_gj_1_1 + 2] = km_l[1];
                                        s_req[rq_gj_1_1 + 3] = sk_l[1];
                                        s_req[rq_gj_1_1 + 4] = cnt_g[1];
                                        mbarrier_arrive(comp_req_addr + (gate_rq) * 8);
                                    }
                                    gate_rq += 1;
                                    if (gate_rq == 4) { gate_rq = 0; }
                                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                                    int need_any_g = 0;
                                    need_any_g = need_any_g | s_req[rq_base_g];
                                    need_any_g = need_any_g | s_req[rq_base_g + 5];
                                    need_any_g = need_any_g | s_req[rq_base_g + 10];
                                    need_any_g = need_any_g | s_req[rq_base_g + 15];
                                    need_any_g = need_any_g | s_req[rq_base_g + 20];
                                    need_any_g = need_any_g | s_req[rq_base_g + 25];
                                    need_any_g = need_any_g | s_req[rq_base_g + 30];
                                    need_any_g = need_any_g | s_req[rq_base_g + 35];
                                    if (need_any_g != 0) {
                                        mbarrier_wait(comp_done_addr, _phase_comp_done);
                                        _phase_comp_done ^= 1;
                                        if (need_l[0] != 0) {
                                            cnt_g[0] = s_count[2 * g_6];
                                            unsigned int lo_g = (unsigned int)s_tau[2 * g_6 * 2];
                                            unsigned int hi_g = (unsigned int)s_tau[2 * g_6 * 2 + 1];
                                            unsigned long long tau_new_g = (unsigned long long)hi_g << 32 | (unsigned long long)lo_g;
                                            unsigned long long _max_7 = ((tau_new_g) > (tau_g[0]) ? (tau_new_g) : (tau_g[0]));
                                            tau_g[0] = _max_7;
                                        }
                                        if (need_l[1] != 0) {
                                            cnt_g[1] = s_count[2 * g_6 + 1];
                                            unsigned int lo_g_1 = (unsigned int)s_tau[(2 * g_6 + 1) * 2];
                                            unsigned int hi_g_1 = (unsigned int)s_tau[(2 * g_6 + 1) * 2 + 1];
                                            unsigned long long tau_new_g_1 = (unsigned long long)hi_g_1 << 32 | (unsigned long long)lo_g_1;
                                            unsigned long long _max_8 = ((tau_new_g_1) > (tau_g[1]) ? (tau_new_g_1) : (tau_g[1]));
                                            tau_g[1] = _max_8;
                                        }
                                    }
                                } else {
                                    until_check_1 = until_check_1 - run_c_1;
                                }
                            }
                            if (attempt_4 == 0) {
                                int fails_g[2];
                                int fail_j_1_0 = 0;
                                if (valid_g[0] != 0) {
                                    int _min_9 = ((top_k) < (vis_g[0]) ? (top_k) : (vis_g[0]));
                                    if (cnt_g[0] < _min_9) {
                                        fail_j_1_0 = 1;
                                    }
                                }
                                fails_g[0] = fail_j_1_0;
                                int fail_j_1_1 = 0;
                                if (valid_g[1] != 0) {
                                    int _min_10 = ((top_k) < (vis_g[1]) ? (top_k) : (vis_g[1]));
                                    if (cnt_g[1] < _min_10) {
                                        fail_j_1_1 = 1;
                                    }
                                }
                                fails_g[1] = fail_j_1_1;
                                mbarrier_wait(verdict_ack_addr, _phase_verdict_ack);
                                _phase_verdict_ack ^= 1;
                                if (elect_sync()) {
                                    s_count[2 * g_6] = cnt_g[0];
                                    s_fail[2 * g_6] = fails_g[0];
                                    s_count[2 * g_6 + 1] = cnt_g[1];
                                    s_fail[2 * g_6 + 1] = fails_g[1];
                                    if (g_6 == 0) {
                                        s_tk[tkp_g ^ 1] = next_g;
                                    }
                                    mbarrier_arrive(verdict_bar_addr);
                                }
                            } else {
                                if (elect_sync()) {
                                    s_count[2 * g_6] = cnt_g[0];
                                    s_count[2 * g_6 + 1] = cnt_g[1];
                                }
                            }
                            asm volatile("barrier.sync 8, 384;" ::: "memory");
                            int verdict_g = 0;
                            if (attempt_4 == 0) {
                                int f_4 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                f_4 = f_4 | s_fail[4];
                                f_4 = f_4 | s_fail[5];
                                f_4 = f_4 | s_fail[6];
                                f_4 = f_4 | s_fail[7];
                                verdict_g = f_4;
                            }
                            tk_g = s_tk[tkp_g ^ 1];
                            asm volatile("barrier.sync 8, 384;" ::: "memory");
                            if (verdict_g == 0) {
                                break;
                            }
                        }
                    }
                }
                unit_base_g += nb_g;
            }
        }
    }

    // Cleanup
}

} // extern "C"
