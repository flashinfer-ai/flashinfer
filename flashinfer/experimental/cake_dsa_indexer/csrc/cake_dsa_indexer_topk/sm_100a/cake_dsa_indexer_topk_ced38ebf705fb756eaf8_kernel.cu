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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 2
#define NUM_K_PIPE_STAGES 4
#define NUM_TMEM_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_W_OFF 66560
#define SMEM_SMEM_W_STAGE_BYTES 512
#define SMEM_SMEM_W_STRIDE 512
#define SMEM_SMEM_K_OFF 67584
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_S_HIST_OFF 198656
#define SMEM_S_HIST_STAGE_BYTES 8192
#define SMEM_S_HIST_STRIDE 8192
#define SMEM_S_COUNT_OFF 206848
#define SMEM_S_COUNT_STAGE_BYTES 32
#define SMEM_S_COUNT_STRIDE 32
#define SMEM_S_TAU_OFF 206880
#define SMEM_S_TAU_STAGE_BYTES 64
#define SMEM_S_TAU_STRIDE 64
#define SMEM_S_FAIL_OFF 206944
#define SMEM_S_FAIL_STAGE_BYTES 16
#define SMEM_S_FAIL_STRIDE 16
#define SMEM_TOTAL 207104

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_dsa_indexer_topk_ced38ebf705fb756eaf8(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap W, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, long long* __restrict__ q_offsets, int* __restrict__ Indices, float* __restrict__ Scores, long long* __restrict__ Cand, int num_segments, int top_k, int ratio, int has_offsets, int cand_cap, int first_cap, int sample_tiles_max, int sample_shift_permille, int check_period, int grid_ctas, float softmax_scale)
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
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 32)
    #define k_empty_addr (mbar_base + 64)
    #define umma_full_addr (mbar_base + 96)
    #define umma_empty_addr (mbar_base + 128)
    #define verdict_bar_addr (mbar_base + 160)

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
    int* s_count = reinterpret_cast<int*>(smem_raw + 206848);
    const int s_count_addr = smem + 206848;
    int* s_tau = reinterpret_cast<int*>(smem_raw + 206880);
    const int s_tau_addr = smem + 206880;
    int* s_fail = reinterpret_cast<int*>(smem_raw + 206944);
    const int s_fail_addr = smem + 206944;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=288
            mbarrier_init(smem + 16, 288);
            mbarrier_init(smem + 24, 288);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // umma_empty: 4 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // verdict_bar: 1 barriers, init_count=8
            mbarrier_init(smem + 160, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 10) {
        int _tmem_hold = smem + 168;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
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
                    int nb = (lq + 4 - 1) / 4;
                    int bid_i = (int)bid;
                    int rel = (bid_i - unit_base_q) % grid_ctas;
                    if (rel < 0) {
                        rel += grid_ctas;
                    }
                    int first = unit_base_q + rel;
                    int first_0 = first;
                    #pragma unroll 1
                    for (int unit = first_0; unit < unit_base_q + nb; unit += grid_ctas) {
                        int blk = nb - 1 - (unit - unit_base_q);
                        #pragma unroll 1
                        for (int attempt = 0; attempt < 2; attempt++) {
                            mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                            tma_3d_gmem2smem(smem_q_addr + load_q_stage * 32768, (&Q), 0, (q0 + blk * 4) * 32, 0, q_full_addr + (load_q_stage) * 8);
                            tma_2d_gmem2smem(smem_w_addr + load_q_stage * 512, (&W), 0, q0 + blk * 4, q_full_addr + (load_q_stage) * 8);
                            mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 33280);
                            load_q_stage += 1;
                            if (load_q_stage == 2) { load_q_stage = 0; _phase_q_empty ^= 1; }
                            if (attempt == 0) {
                                mbarrier_wait(verdict_bar_addr, vphase_q);
                                vphase_q ^= 1;
                                int f = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                                if (f == 0) {
                                    break;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
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
                    int nb_1 = (lq_1 + 4 - 1) / 4;
                    int bid_i_1 = (int)bid;
                    int rel_1 = (bid_i_1 - unit_base_k) % grid_ctas;
                    if (rel_1 < 0) {
                        rel_1 += grid_ctas;
                    }
                    int first_1 = unit_base_k + rel_1;
                    int first_0_1 = first_1;
                    #pragma unroll 1
                    for (int unit_1 = first_0_1; unit_1 < unit_base_k + nb_1; unit_1 += grid_ctas) {
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
                        int _min_0 = ((blk_1 * 4 + 3) < (lq_1_1 - 1) ? (blk_1 * 4 + 3) : (lq_1_1 - 1));
                        int last_u = _min_0;
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
                        int _min_1 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
                        int max_tiles = _min_1;
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
                                {
                                    tile = n_tiles_3 - 1 - d_k;
                                }
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
                                if (f_1 == 0) {
                                    break;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
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
                int nb_2 = (lq_2 + 4 - 1) / 4;
                int bid_i_2 = (int)bid;
                int rel_2 = (bid_i_2 - unit_base_m) % grid_ctas;
                if (rel_2 < 0) {
                    rel_2 += grid_ctas;
                }
                int first_2 = unit_base_m + rel_2;
                int first_0_2 = first_2;
                #pragma unroll 1
                for (int unit_2 = first_0_2; unit_2 < unit_base_m + nb_2; unit_2 += grid_ctas) {
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
                    int _min_2 = ((blk_2 * 4 + 3) < (lq_1_2 - 1) ? (blk_2 * 4 + 3) : (lq_1_2 - 1));
                    int last_u_1 = _min_2;
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
                                int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 2048;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mma_tmem_stage * 128))), "r"(0));
                                tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                                tcgen05_commit(k_empty_addr + (mma_k_stage) * 8);
                                mma_tmem_stage += 1;
                                if (mma_tmem_stage == 4) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            }
                            __syncwarp();
                            mma_k_stage += 1;
                            if (mma_k_stage == 4) { mma_k_stage = 0; _phase_k_full ^= 1; }
                        }
                        mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                        mma_q_stage += 1;
                        if (mma_q_stage == 2) { mma_q_stage = 0; _phase_q_full ^= 1; }
                        if (attempt_2 == 0) {
                            mbarrier_wait(verdict_bar_addr, vphase_m);
                            vphase_m ^= 1;
                            int f_2 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                            if (f_2 == 0) {
                                break;
                            }
                        }
                    }
                }
                unit_base_m += nb_2;
            }
        }
    // ---- Role: spare ----
    } else if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // spare_main
            __syncwarp();
        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
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
            unsigned int math_q_stage = 0;
            unsigned int math_tmem_stage = 0;
            unsigned int math_tmem_phase = 0;
            int unit_base = 0;
            int _min_3 = ((top_k + 64) < (cand_cap - 128 * check_period) ? (top_k + 64) : (cand_cap - 128 * check_period));
            int keep_cap = _min_3;
            int trigger_room = 128 * check_period;
            unsigned int _phase_q_full_1 = 0;
            #pragma unroll 1
            for (int seg_3 = 0; seg_3 < num_segments; seg_3++) {
                int q0_3 = cu_seqlens_q[seg_3];
                int lq_3 = cu_seqlens_q[seg_3 + 1] - q0_3;
                int lk_2 = cu_seqlens_k[seg_3 + 1] - cu_seqlens_k[seg_3];
                int nb_3 = (lq_3 + 4 - 1) / 4;
                long long off_3 = 0;
                if (has_offsets != 0) {
                    off_3 = q_offsets[seg_3];
                } else if (ratio == 1) {
                    off_3 = (long long)lk_2 - (long long)lq_3;
                }
                long long off_0 = off_3;
                int bid_i_1_1 = (int)bid;
                int rel_3 = (bid_i_1_1 - unit_base) % grid_ctas;
                if (rel_3 < 0) {
                    rel_3 += grid_ctas;
                }
                int first_3 = unit_base + rel_3;
                int first_2_1 = first_3;
                #pragma unroll 1
                for (int unit_3 = first_2_1; unit_3 < unit_base + nb_3; unit_3 += grid_ctas) {
                    int blk_3 = nb_3 - 1 - (unit_3 - unit_base);
                    int u_base = blk_3 * 4;
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
                    int _min_4 = ((blk_3 * 4 + 3) < (lq_1_3 - 1) ? (blk_3 * 4 + 3) : (lq_1_3 - 1));
                    int last_u_2 = _min_4;
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
                    int _min_5 = ((sample_tiles_max) < (cand_cap / 128 - check_period) ? (sample_tiles_max) : (cand_cap / 128 - check_period));
                    int max_tiles_1 = _min_5;
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
                    int row_valid[2];
                    int visible[2];
                    unsigned long long tau[2];
                    int buf_base[2];
                    int buf_unit = (bid_i_3 * 4 + wg * 2) * cand_cap;
                    int u_q = u_base + wg * 2;
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
                    int u_q_6 = u_base + wg * 2 + 1;
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
                    int buf_pair = buf_unit + pair * cand_cap;
                    int slot_pair = wg * 2 + pair;
                    int valid_pair = row_valid[1];
                    int visible_pair = visible[1];
                    if (pair == 0) {
                        valid_pair = row_valid[0];
                        visible_pair = visible[0];
                    }
                    #pragma unroll 1
                    for (int attempt_3 = 0; attempt_3 < 2; attempt_3++) {
                        int n_s = n_sample_1;
                        if (attempt_3 != 0) {
                            n_s = 0;
                        }
                        tau[0] = 0;
                        tau[1] = 0;
                        if (local_thread_idx < 2) {
                            s_count[wg * 2 + local_thread_idx] = 0;
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                        mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full_1);
                        float weights_reg[64];
                        int weight_stage_base = math_q_stage * 128 + (unsigned int)(wg * 64);
                        #pragma unroll
                        for (int wi = 0; wi < 64; wi++) {
                            float w_raw = smem_w[weight_stage_base + wi];
                            weights_reg[wi] = w_raw * softmax_scale;
                        }
                        int d_pos = 0;
                        int r_pos = 0;
                        int until_check = check_period - 1;
                        #pragma unroll 1
                        for (int ti_2 = 0; ti_2 < n_tiles_5; ti_2++) {
                            int tile_1 = d_pos;
                            {
                                tile_1 = n_tiles_5 - 1 - d_pos;
                            }
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
                            float _tmem_load_0[32];
                            tmem_ld_x16(&_tmem_load_0[0], taddr + math_tmem_stage * 128 + (unsigned int)(wg * 2 * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                            tmem_ld_x16(&_tmem_load_0[16], taddr + math_tmem_stage * 128 + (unsigned int)(wg * 2 * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                            float _tmem_load_1[32];
                            tmem_ld_x16(&_tmem_load_1[0], taddr + math_tmem_stage * 128 + (unsigned int)((wg * 2 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16));
                            tmem_ld_x16(&_tmem_load_1[16], taddr + math_tmem_stage * 128 + (unsigned int)((wg * 2 + 1) * 32) + (unsigned int)(warp_in_wg * 32 << 16) + 16);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
                            unsigned int bits2[2];
                            int emit2[2];
                            unsigned int ballot2[2];
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
                                float score = _relu_wsum_0;
                                unsigned int bits = 0;
                                bits = reinterpret_cast<unsigned int*>(&score)[0];
                                bits2[0] = bits;
                                unsigned int magnitude = bits & 2147483647;
                                unsigned int m = ((magnitude != 0) ? bits : zero_u);
                                unsigned int key32 = (((m & sign_mask) != 0) ? ~m : m | sign_mask);
                                if (magnitude > nan_floor) {
                                    key32 = zero_u;
                                }
                                unsigned long long key64 = (unsigned long long)key32 << 32 | (unsigned long long)(unsigned int)kid;
                                unsigned long long key = key64;
                                int in_prefix = ((kid < visible[0]) ? 1 : 0);
                                int above = ((key >= tau[0]) ? 1 : 0);
                                int emit = row_valid[0] & in_prefix & above;
                                emit2[0] = emit;
                                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, emit != 0);
                                ballot2[0] = _vote_0;
                            }
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
                                float score_1 = _relu_wsum_1;
                                unsigned int bits_1 = 0;
                                bits_1 = reinterpret_cast<unsigned int*>(&score_1)[0];
                                bits2[1] = bits_1;
                                unsigned int magnitude_1 = bits_1 & 2147483647;
                                unsigned int m_1 = ((magnitude_1 != 0) ? bits_1 : zero_u);
                                unsigned int key32_1 = (((m_1 & sign_mask) != 0) ? ~m_1 : m_1 | sign_mask);
                                if (magnitude_1 > nan_floor) {
                                    key32_1 = zero_u;
                                }
                                unsigned long long key64_1 = (unsigned long long)key32_1 << 32 | (unsigned long long)(unsigned int)kid;
                                unsigned long long key_1 = key64_1;
                                int in_prefix_1 = ((kid < visible[1]) ? 1 : 0);
                                int above_1 = ((key_1 >= tau[1]) ? 1 : 0);
                                int emit_1 = row_valid[1] & in_prefix_1 & above_1;
                                emit2[1] = emit_1;
                                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, emit_1 != 0);
                                ballot2[1] = _vote_1;
                            }
                            int _popc_0 = __popc(ballot2[1]);
                            int cnt_lane = _popc_0;
                            if (lane_0 == 0) {
                                int _popc_1 = __popc(ballot2[0]);
                                cnt_lane = _popc_1;
                            }
                            int reservation = 0;
                            if (lane_0 < 2) {
                                int _atomic_old_0 = atomicAdd(s_count + (wg * 2 + lane_0), cnt_lane);
                                reservation = _atomic_old_0;
                            }
                            {
                                int _shfl_0 = __shfl_sync(0xFFFFFFFF, reservation, 0);
                                int slot_base = _shfl_0;
                                if (emit2[0] != 0) {
                                    int _popc_2 = __popc(ballot2[0] & lower_lanes);
                                    int slot = buf_base[0] + slot_base + _popc_2;
                                    unsigned long long entry = (unsigned long long)bits2[0] << 32 | (unsigned long long)(unsigned int)kid;
                                    Cand[slot] = (long long)entry;
                                }
                            }
                            {
                                int _shfl_1 = __shfl_sync(0xFFFFFFFF, reservation, 1);
                                int slot_base_1 = _shfl_1;
                                if (emit2[1] != 0) {
                                    int _popc_3 = __popc(ballot2[1] & lower_lanes);
                                    int slot_1 = buf_base[1] + slot_base_1 + _popc_3;
                                    unsigned long long entry_1 = (unsigned long long)bits2[1] << 32 | (unsigned long long)(unsigned int)kid;
                                    Cand[slot_1] = (long long)entry_1;
                                }
                            }
                            math_tmem_stage += 1;
                            if (math_tmem_stage >= 4) {
                                math_tmem_stage -= 4;
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
                                int need[2];
                                int kw[2];
                                int km[2];
                                int c_q = s_count[wg * 2];
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
                                                    int _min_6 = ((r_shift + 64) < (keep_cap) ? (r_shift + 64) : (keep_cap));
                                                    km_q = _min_6;
                                                }
                                            }
                                        }
                                    }
                                }
                                need[0] = need_q;
                                kw[0] = kw_q;
                                km[0] = km_q;
                                need_any = need_any | need_q;
                                int c_q_0 = s_count[wg * 2 + 1];
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
                                                    int _min_7 = ((r_shift_1 + 64) < (keep_cap) ? (r_shift_1 + 64) : (keep_cap));
                                                    km_q_4 = _min_7;
                                                }
                                            }
                                        }
                                    }
                                }
                                need[1] = need_q_2;
                                kw[1] = kw_q_3;
                                km[1] = km_q_4;
                                need_any = need_any | need_q_2;
                                if (need_any != 0) {
                                    int need_pair = need[1];
                                    int kw_pair = kw[1];
                                    int km_pair = km[1];
                                    if (pair == 0) {
                                        need_pair = need[0];
                                        kw_pair = kw[0];
                                        km_pair = km[0];
                                    }
                                    if (need_pair != 0) {
                                        int c_pair = s_count[slot_pair];
                                        int k_rem = kw_pair;
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
                                            #pragma unroll 1
                                            for (int e0 = 0; e0 < c_pair; e0 += 512) {
                                                unsigned long long ent[8];
                                                #pragma unroll
                                                for (int u = 0; u < 8; u++) {
                                                    ent[u] = 0;
                                                    if (c_pair > e0 + half * 32 + lane_0 + 64 * u) {
                                                        ent[u] = (unsigned long long)Cand[buf_pair + e0 + half * 32 + lane_0 + 64 * u];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_1 = 0; u_1 < 8; u_1++) {
                                                    if (c_pair > e0 + half * 32 + lane_0 + 64 * u_1) {
                                                        unsigned int bits_2 = (unsigned int)(ent[u_1] >> 32);
                                                        int kid_0 = (int)(unsigned int)(ent[u_1] & 4294967295);
                                                        unsigned int magnitude_2 = bits_2 & 2147483647;
                                                        unsigned int m_2 = ((magnitude_2 != 0) ? bits_2 : zero_u);
                                                        unsigned int key32_2 = (((m_2 & sign_mask) != 0) ? ~m_2 : m_2 | sign_mask);
                                                        if (magnitude_2 > nan_floor) {
                                                            key32_2 = zero_u;
                                                        }
                                                        unsigned long long key64_2 = (unsigned long long)key32_2 << 32 | (unsigned long long)(unsigned int)kid_0;
                                                        unsigned long long key_2 = key64_2;
                                                        unsigned long long key_1_1 = key_2;
                                                        int digit = (int)(key_1_1 >> (unsigned long long)shift & 255);
                                                        int matches = 1;
                                                        if (p > 0) {
                                                            matches = ((key_1_1 >> (unsigned long long)(shift + 8) == prefix) ? 1 : 0);
                                                        }
                                                        if (matches != 0) {
                                                            atomicAdd(&s_hist[hist_base + digit], 1);
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
                                            float _warp_reduce_0 = is_target * lane_0;
                                            #pragma unroll
                                            for (int offset = 16; offset > 0; offset >>= 1)
                                                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                                            int target_lane = _warp_reduce_0;
                                            int _shfl_2 = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
                                            int digit_sel = _shfl_2;
                                            int _shfl_3 = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
                                            int above_cnt = _shfl_3;
                                            int _shfl_4 = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
                                            int bucket_cnt = _shfl_4;
                                            k_rem = k_rem - above_cnt;
                                            kept_above += above_cnt;
                                            prefix = prefix << 8 | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
                                            edge = prefix << (unsigned long long)shift;
                                            if (km_pair >= kept_above + bucket_cnt) {
                                                break;
                                            }
                                        }
                                        unsigned long long edge_0 = edge;
                                        if (half == 0) {
                                            int write_pos = 0;
                                            #pragma unroll 1
                                            for (int e0_1 = 0; e0_1 < c_pair; e0_1 += 256) {
                                                unsigned long long ent_1[8];
                                                #pragma unroll
                                                for (int u_2 = 0; u_2 < 8; u_2++) {
                                                    ent_1[u_2] = 0;
                                                    if (c_pair > e0_1 + lane_0 + 32 * u_2) {
                                                        ent_1[u_2] = (unsigned long long)Cand[buf_pair + e0_1 + lane_0 + 32 * u_2];
                                                    }
                                                }
                                                #pragma unroll
                                                for (int u_3 = 0; u_3 < 8; u_3++) {
                                                    int keep = 0;
                                                    if (c_pair > e0_1 + lane_0 + 32 * u_3) {
                                                        unsigned int bits_3 = (unsigned int)(ent_1[u_3] >> 32);
                                                        int kid_0_1 = (int)(unsigned int)(ent_1[u_3] & 4294967295);
                                                        unsigned int magnitude_3 = bits_3 & 2147483647;
                                                        unsigned int m_3 = ((magnitude_3 != 0) ? bits_3 : zero_u);
                                                        unsigned int key32_3 = (((m_3 & sign_mask) != 0) ? ~m_3 : m_3 | sign_mask);
                                                        if (magnitude_3 > nan_floor) {
                                                            key32_3 = zero_u;
                                                        }
                                                        unsigned long long key64_3 = (unsigned long long)key32_3 << 32 | (unsigned long long)(unsigned int)kid_0_1;
                                                        unsigned long long key_3 = key64_3;
                                                        keep = ((key_3 >= edge_0) ? 1 : 0);
                                                    }
                                                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, keep != 0);
                                                    if (keep != 0) {
                                                        int _popc_4 = __popc(_vote_2 & lower_lanes);
                                                        Cand[buf_pair + write_pos + _popc_4] = (long long)ent_1[u_3];
                                                    }
                                                    int _popc_5 = __popc(_vote_2);
                                                    write_pos += _popc_5;
                                                }
                                            }
                                            if (lane_0 == 0) {
                                                s_count[slot_pair] = write_pos;
                                                s_tau[slot_pair * 2] = (int)(unsigned int)edge_0;
                                                s_tau[slot_pair * 2 + 1] = (int)(unsigned int)(edge_0 >> 32);
                                            }
                                        }
                                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                                    }
                                    asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                                    if (need[0] != 0) {
                                        unsigned int lo = (unsigned int)s_tau[wg * 2 * 2];
                                        unsigned int hi = (unsigned int)s_tau[wg * 2 * 2 + 1];
                                        unsigned long long tau_new = (unsigned long long)hi << 32 | (unsigned long long)lo;
                                        unsigned long long _max_0 = ((tau_new) > (tau[0]) ? (tau_new) : (tau[0]));
                                        tau[0] = _max_0;
                                    }
                                    if (need[1] != 0) {
                                        unsigned int lo_1 = (unsigned int)s_tau[(wg * 2 + 1) * 2];
                                        unsigned int hi_1 = (unsigned int)s_tau[(wg * 2 + 1) * 2 + 1];
                                        unsigned long long tau_new_1 = (unsigned long long)hi_1 << 32 | (unsigned long long)lo_1;
                                        unsigned long long _max_1 = ((tau_new_1) > (tau[1]) ? (tau_new_1) : (tau[1]));
                                        tau[1] = _max_1;
                                    }
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                        math_q_stage += 1;
                        if (math_q_stage == 2) { math_q_stage = 0; _phase_q_full_1 ^= 1; }
                        int c_final = 0;
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                        c_final = s_count[slot_pair];
                        int verdict = 0;
                        if (attempt_3 == 0) {
                            int fail_pair = 0;
                            if (valid_pair != 0) {
                                int _min_8 = ((top_k) < (visible_pair) ? (top_k) : (visible_pair));
                                if (c_final < _min_8) {
                                    fail_pair = 1;
                                }
                            }
                            if (half == 0) {
                                if (lane_0 == 0) {
                                    s_fail[slot_pair] = fail_pair;
                                }
                            }
                            asm volatile("barrier.sync 3, 256;" ::: "memory");
                            int f_3 = s_fail[0] | s_fail[1] | s_fail[2] | s_fail[3];
                            verdict = f_3;
                            if (elect_sync()) {
                                mbarrier_arrive(verdict_bar_addr);
                            }
                        }
                        int _min_9 = ((top_k) < (c_final) ? (top_k) : (c_final));
                        int n_sel = _min_9;
                        asm volatile("barrier.sync %0, 128;" :: "r"(1 + wg) : "memory");
                        if (verdict == 0) {
                            unsigned long long edge_out = 0;
                            if (c_final > top_k) {
                                int k_rem_1 = top_k;
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
                                    #pragma unroll 1
                                    for (int e0_2 = 0; e0_2 < c_final; e0_2 += 512) {
                                        unsigned long long ent_2[8];
                                        #pragma unroll
                                        for (int u_4 = 0; u_4 < 8; u_4++) {
                                            ent_2[u_4] = 0;
                                            if (c_final > e0_2 + half * 32 + lane_0 + 64 * u_4) {
                                                ent_2[u_4] = (unsigned long long)Cand[buf_pair + e0_2 + half * 32 + lane_0 + 64 * u_4];
                                            }
                                        }
                                        #pragma unroll
                                        for (int u_5 = 0; u_5 < 8; u_5++) {
                                            if (c_final > e0_2 + half * 32 + lane_0 + 64 * u_5) {
                                                unsigned int bits_4 = (unsigned int)(ent_2[u_5] >> 32);
                                                int kid_1 = (int)(unsigned int)(ent_2[u_5] & 4294967295);
                                                unsigned int magnitude_4 = bits_4 & 2147483647;
                                                unsigned int m_4 = ((magnitude_4 != 0) ? bits_4 : zero_u);
                                                unsigned int key32_4 = (((m_4 & sign_mask) != 0) ? ~m_4 : m_4 | sign_mask);
                                                if (magnitude_4 > nan_floor) {
                                                    key32_4 = zero_u;
                                                }
                                                unsigned long long key64_4 = (unsigned long long)key32_4 << 32 | (unsigned long long)(unsigned int)kid_1;
                                                unsigned long long key_4 = key64_4;
                                                unsigned long long key_0 = key_4;
                                                int digit_1 = (int)(key_0 >> (unsigned long long)shift_1 & 255);
                                                int matches_1 = 1;
                                                if (p_1 > 0) {
                                                    matches_1 = ((key_0 >> (unsigned long long)(shift_1 + 8) == prefix_1) ? 1 : 0);
                                                }
                                                if (matches_1 != 0) {
                                                    atomicAdd(&s_hist[hist_base + digit_1], 1);
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
                                    float _warp_reduce_1 = is_target_1 * lane_0;
                                    #pragma unroll
                                    for (int offset = 16; offset > 0; offset >>= 1)
                                        _warp_reduce_1 = max_noftz(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
                                    int target_lane_1 = _warp_reduce_1;
                                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, d_sel_1, target_lane_1);
                                    int digit_sel_1 = _shfl_5;
                                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, above_sel_1, target_lane_1);
                                    int above_cnt_1 = _shfl_6;
                                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, count_sel_1, target_lane_1);
                                    int bucket_cnt_1 = _shfl_7;
                                    k_rem_1 = k_rem_1 - above_cnt_1;
                                    kept_above_1 += above_cnt_1;
                                    prefix_1 = prefix_1 << 8 | (unsigned long long)(unsigned int)(target_lane_1 * 8 + digit_sel_1);
                                    edge_1 = prefix_1 << (unsigned long long)shift_1;
                                    if (kept_above_1 + bucket_cnt_1 <= top_k) {
                                        break;
                                    }
                                }
                                edge_out = edge_1;
                            }
                            int row = q0_3 + u_base + wg * 2 + pair;
                            long long row_base = (long long)row * (long long)top_k;
                            if (valid_pair != 0) {
                                if (half == 0) {
                                    int write_pos_1 = 0;
                                    #pragma unroll 1
                                    for (int e0_3 = 0; e0_3 < c_final; e0_3 += 256) {
                                        unsigned long long ent_o[8];
                                        #pragma unroll
                                        for (int u_6 = 0; u_6 < 8; u_6++) {
                                            ent_o[u_6] = 0;
                                            if (c_final > e0_3 + lane_0 + 32 * u_6) {
                                                ent_o[u_6] = (unsigned long long)Cand[buf_pair + e0_3 + lane_0 + 32 * u_6];
                                            }
                                        }
                                        #pragma unroll
                                        for (int u_7 = 0; u_7 < 8; u_7++) {
                                            int keep_o = 0;
                                            if (c_final > e0_3 + lane_0 + 32 * u_7) {
                                                unsigned int bits_5 = (unsigned int)(ent_o[u_7] >> 32);
                                                int kid_2 = (int)(unsigned int)(ent_o[u_7] & 4294967295);
                                                unsigned int magnitude_5 = bits_5 & 2147483647;
                                                unsigned int m_5 = ((magnitude_5 != 0) ? bits_5 : zero_u);
                                                unsigned int key32_5 = (((m_5 & sign_mask) != 0) ? ~m_5 : m_5 | sign_mask);
                                                if (magnitude_5 > nan_floor) {
                                                    key32_5 = zero_u;
                                                }
                                                unsigned long long key64_5 = (unsigned long long)key32_5 << 32 | (unsigned long long)(unsigned int)kid_2;
                                                unsigned long long key_5 = key64_5;
                                                keep_o = ((key_5 >= edge_out) ? 1 : 0);
                                            }
                                            unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, keep_o != 0);
                                            if (keep_o != 0) {
                                                int _popc_6 = __popc(_vote_3 & lower_lanes);
                                                long long out_slot = row_base + (long long)(write_pos_1 + _popc_6);
                                                float score_o = 0.0f;
                                                unsigned int bits_6 = (unsigned int)(ent_o[u_7] >> 32);
                                                score_o = reinterpret_cast<float*>(&bits_6)[0];
                                                int kid_3 = (int)(unsigned int)(ent_o[u_7] & 4294967295);
                                                Indices[out_slot] = kid_3;
                                                Scores[out_slot] = score_o;
                                            }
                                            int _popc_7 = __popc(_vote_3);
                                            write_pos_1 += _popc_7;
                                        }
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int pad = n_sel + lane_0; pad < top_k; pad += 32) {
                                        long long pad_slot = row_base + (long long)pad;
                                        Indices[pad_slot] = -1;
                                        Scores[pad_slot] = -CUDART_INF_F;
                                    }
                                }
                            }
                        }
                        if (verdict == 0) {
                            break;
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
