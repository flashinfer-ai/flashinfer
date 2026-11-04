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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_PARTIALS_OFFSET 0
#define NUM_AB_PIPE_STAGES 7
#define NUM_PARTIAL_PIPE_STAGES 4
#define NUM_SCALE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 115712
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_B_HALF0_OFF 115712
#define SMEM_SMEM_B_HALF0_STAGE_BYTES 8192
#define SMEM_SMEM_B_HALF0_STRIDE 16384
#define SMEM_SMEM_B_HALF1_OFF 123904
#define SMEM_SMEM_B_HALF1_STAGE_BYTES 8192
#define SMEM_SMEM_B_HALF1_STRIDE 16384
#define SMEM_ROWMAX_XCHG_OFF 1024
#define SMEM_ROWMAX_XCHG_STAGE_BYTES 114688
#define SMEM_ROWMAX_XCHG_STRIDE 114688
#define SMEM_SMEM_ASCALE_OFF 230400
#define SMEM_SMEM_ASCALE_STAGE_BYTES 2048
#define SMEM_SMEM_ASCALE_STRIDE 2048
#define SMEM_TOTAL 232448
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}




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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_grouped_fp8_fused_silu_quant_a54057945ffaff1016b1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, uint8_t* __restrict__ out_q, float* __restrict__ out_s, float* __restrict__ a_scale, float* __restrict__ b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
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
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 56)
    #define partial_full_addr (mbar_base + 112)
    #define partial_free_addr (mbar_base + 144)
    #define scale_full_addr (mbar_base + 176)
    #define scale_free_addr (mbar_base + 184)
    #define producers_done_addr (mbar_base + 192)
    #define pair_exit_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int smem_b_addr = smem + 115712;
    uint8_t* smem_b_half0 = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int smem_b_half0_addr = smem + 115712;
    uint8_t* smem_b_half1 = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_b_half1_addr = smem + 123904;
    float* rowmax_xchg = reinterpret_cast<float*>(smem_raw + 1024);
    const int rowmax_xchg_addr = smem + 1024;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_ascale_addr = smem + 230400;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    int taddr;
    int tmem_partials;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // ab_free: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'partial_pipe' ---
            // partial_full: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // partial_free: 4 barriers, init_count=16
            mbarrier_init(smem + 144, 16);
            mbarrier_init(smem + 152, 16);
            mbarrier_init(smem + 160, 16);
            mbarrier_init(smem + 168, 16);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 1 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            // scale_free: 1 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            // producers_done: 1 barriers, init_count=3
            mbarrier_init(smem + 192, 3);
            // pair_exit: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: acc_epi ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // acc_epi_main
            if (warp == 0) {
                int _tmem_hold_0 = smem + 208;
                asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_0), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            }
            unsigned int partial_stage = 0;
            unsigned int consumer_ab_cursor = 0;
            unsigned int epi_panel_cursor = 0;
            float b_scale_prefetch[4];
            float b_scale_prefetch_u[4];
            int b_scale_primed = 0;
            const int acc_wg = warp / 4;
            const int acc_warp_in_wg = warp % 4;
            int wg_col_base = acc_wg * 128;
            int row_base = acc_warp_in_wg * 32;
            int row = row_base + lane;
            int m_tiles = (M - 1) / 256 + 1;
            int n_tiles = N / 256;
            int total_tiles = m_tiles * n_tiles;
            int nblk = (M - 1) / 128 + 1;
            int g_blk_lo = -1;
            int g_blk_hi = -1;
            if (nblk > lane) {
                g_blk_lo = m_indices[lane * 128];
            }
            if (nblk > lane + 32) {
                g_blk_hi = m_indices[(lane + 32) * 128];
            }
            int pair_lo = 0;
            int pair_hi = 0;
            int sc_prev_lo = __shfl_sync(0xFFFFFFFF, g_blk_lo, lane + 31 & 31);
            int sc_prev_hi = __shfl_sync(0xFFFFFFFF, g_blk_hi, lane + 31 & 31);
            if (lane == 0) {
                sc_prev_hi = sc_prev_lo;
            }
            int sc_open_lo = 0;
            int sc_open_hi = 0;
            if (nblk > lane) {
                if (g_blk_lo != sc_prev_lo) {
                    sc_open_lo = 1;
                }
                if (lane == 0) {
                    sc_open_lo = 1;
                }
            }
            if (nblk > lane + 32) {
                if (g_blk_hi != sc_prev_hi) {
                    sc_open_hi = 1;
                }
            }
            unsigned int sc_runs_lo = __ballot_sync(0xFFFFFFFF, sc_open_lo != 0);
            unsigned int sc_runs_hi = __ballot_sync(0xFFFFFFFF, sc_open_hi != 0);
            unsigned int sc_upto = (2 << (unsigned int)lane) - 1;
            int _count_leading_zeros_1 = __clz(sc_runs_lo & sc_upto);
            int sc_start_lo = 31 - _count_leading_zeros_1;
            int _count_leading_zeros_2 = __clz(sc_runs_lo);
            int sc_start_hi = 31 - _count_leading_zeros_2;
            if ((sc_runs_hi & sc_upto) != 0) {
                int _count_leading_zeros_3 = __clz(sc_runs_hi & sc_upto);
                sc_start_hi = 63 - _count_leading_zeros_3;
            }
            int sc_odd_lo = 0;
            int sc_odd_hi = 0;
            if (nblk > lane) {
                sc_odd_lo = lane - sc_start_lo & 1;
            }
            if (nblk > lane + 32) {
                sc_odd_hi = lane + 32 - sc_start_hi & 1;
            }
            unsigned int sc_pairs_lo = __ballot_sync(0xFFFFFFFF, sc_odd_lo != 0);
            unsigned int sc_pairs_hi = __ballot_sync(0xFFFFFFFF, sc_odd_hi != 0);
            int sc_n_lo = (int)(unsigned int)__popc(sc_pairs_lo);
            int pairs_total = sc_n_lo + (int)(unsigned int)__popc(sc_pairs_hi);
            unsigned int sc_sel = sc_pairs_lo;
            int sc_rank = lane;
            int sc_base = 0;
            if (sc_n_lo <= lane) {
                sc_sel = sc_pairs_hi;
                sc_rank = lane - sc_n_lo;
                sc_base = 32;
            }
            unsigned int sc_pos = 0;
            unsigned int sc_step = 16;
            #pragma unroll
            for (int sc_i = 0; sc_i < 5; sc_i++) {
                int sc_below = (int)(unsigned int)__popc(sc_sel & (1 << sc_pos + sc_step) - 1);
                if (sc_below <= sc_rank) {
                    sc_pos = sc_pos + sc_step;
                }
                sc_step = sc_step >> 1;
            }
            if (pairs_total > lane) {
                pair_lo = sc_base + (int)sc_pos;
            }
            total_tiles = pairs_total * n_tiles;
            unsigned int _phase_scale_full = 0;
            unsigned int _phase_partial_full = 0;
            #pragma unroll 1
            for (int tile_id = cluster_id; tile_id < total_tiles; tile_id += num_clusters) {
                int pair_id = tile_id / n_tiles;
                int n_tile = tile_id % n_tiles;
                int b0 = __shfl_sync(0xFFFFFFFF, pair_lo, pair_id & 31);
                int b0_hi = __shfl_sync(0xFFFFFFFF, pair_hi, pair_id & 31);
                if (pair_id >= 32) {
                    b0 = b0_hi;
                }
                b0 = b0 - 1;
                int group = __shfl_sync(0xFFFFFFFF, g_blk_lo, b0 & 31);
                int group_hi = __shfl_sync(0xFFFFFFFF, g_blk_hi, b0 & 31);
                if (b0 >= 32) {
                    group = group_hi;
                }
                int bn = b0 + 1;
                int gn = __shfl_sync(0xFFFFFFFF, g_blk_lo, bn & 31);
                int gn_hi = __shfl_sync(0xFFFFFFFF, g_blk_hi, bn & 31);
                if (bn >= 32) {
                    gn = gn_hi;
                }
                int b1 = -1;
                if (bn < nblk) {
                    if (gn == group) {
                        b1 = bn;
                    }
                }
                int my_blk = b0;
                if (cta_rank == 1) {
                    my_blk = b1;
                }
                int m_tile = b0;
                int my_rows = 0;
                if (my_blk >= 0) {
                    m_tile = my_blk;
                    my_rows = M - my_blk * 128;
                    if (my_rows > 128) {
                        my_rows = 128;
                    }
                }
                int cluster_m_base = m_tile * 128 - cta_rank * 128;
                int tile_rows = 256;
                int run_begin = cta_rank * 128;
                int run_end = run_begin + my_rows;
                int next_tile = (unsigned int)tile_id + num_clusters;
                int has_next = 0;
                if (next_tile < total_tiles) {
                    has_next = 1;
                }
                int next_pair = next_tile / n_tiles;
                int next_n_tile = next_tile % n_tiles;
                int nb0 = __shfl_sync(0xFFFFFFFF, pair_lo, next_pair & 31);
                int nb0_hi = __shfl_sync(0xFFFFFFFF, pair_hi, next_pair & 31);
                if (next_pair >= 32) {
                    nb0 = nb0_hi;
                }
                nb0 = nb0 - 1;
                int next_group = __shfl_sync(0xFFFFFFFF, g_blk_lo, nb0 & 31);
                int next_group_hi = __shfl_sync(0xFFFFFFFF, g_blk_hi, nb0 & 31);
                if (nb0 >= 32) {
                    next_group = next_group_hi;
                }
                float acc0[32];
                float acc1[32];
                float acc2[32];
                float acc3[32];
                acc0[0] = 0.0f;
                acc0[1] = 0.0f;
                acc0[2] = 0.0f;
                acc0[3] = 0.0f;
                acc0[4] = 0.0f;
                acc0[5] = 0.0f;
                acc0[6] = 0.0f;
                acc0[7] = 0.0f;
                acc0[8] = 0.0f;
                acc0[9] = 0.0f;
                acc0[10] = 0.0f;
                acc0[11] = 0.0f;
                acc0[12] = 0.0f;
                acc0[13] = 0.0f;
                acc0[14] = 0.0f;
                acc0[15] = 0.0f;
                acc0[16] = 0.0f;
                acc0[17] = 0.0f;
                acc0[18] = 0.0f;
                acc0[19] = 0.0f;
                acc0[20] = 0.0f;
                acc0[21] = 0.0f;
                acc0[22] = 0.0f;
                acc0[23] = 0.0f;
                acc0[24] = 0.0f;
                acc0[25] = 0.0f;
                acc0[26] = 0.0f;
                acc0[27] = 0.0f;
                acc0[28] = 0.0f;
                acc0[29] = 0.0f;
                acc0[30] = 0.0f;
                acc0[31] = 0.0f;
                acc1[0] = 0.0f;
                acc1[1] = 0.0f;
                acc1[2] = 0.0f;
                acc1[3] = 0.0f;
                acc1[4] = 0.0f;
                acc1[5] = 0.0f;
                acc1[6] = 0.0f;
                acc1[7] = 0.0f;
                acc1[8] = 0.0f;
                acc1[9] = 0.0f;
                acc1[10] = 0.0f;
                acc1[11] = 0.0f;
                acc1[12] = 0.0f;
                acc1[13] = 0.0f;
                acc1[14] = 0.0f;
                acc1[15] = 0.0f;
                acc1[16] = 0.0f;
                acc1[17] = 0.0f;
                acc1[18] = 0.0f;
                acc1[19] = 0.0f;
                acc1[20] = 0.0f;
                acc1[21] = 0.0f;
                acc1[22] = 0.0f;
                acc1[23] = 0.0f;
                acc1[24] = 0.0f;
                acc1[25] = 0.0f;
                acc1[26] = 0.0f;
                acc1[27] = 0.0f;
                acc1[28] = 0.0f;
                acc1[29] = 0.0f;
                acc1[30] = 0.0f;
                acc1[31] = 0.0f;
                acc2[0] = 0.0f;
                acc2[1] = 0.0f;
                acc2[2] = 0.0f;
                acc2[3] = 0.0f;
                acc2[4] = 0.0f;
                acc2[5] = 0.0f;
                acc2[6] = 0.0f;
                acc2[7] = 0.0f;
                acc2[8] = 0.0f;
                acc2[9] = 0.0f;
                acc2[10] = 0.0f;
                acc2[11] = 0.0f;
                acc2[12] = 0.0f;
                acc2[13] = 0.0f;
                acc2[14] = 0.0f;
                acc2[15] = 0.0f;
                acc2[16] = 0.0f;
                acc2[17] = 0.0f;
                acc2[18] = 0.0f;
                acc2[19] = 0.0f;
                acc2[20] = 0.0f;
                acc2[21] = 0.0f;
                acc2[22] = 0.0f;
                acc2[23] = 0.0f;
                acc2[24] = 0.0f;
                acc2[25] = 0.0f;
                acc2[26] = 0.0f;
                acc2[27] = 0.0f;
                acc2[28] = 0.0f;
                acc2[29] = 0.0f;
                acc2[30] = 0.0f;
                acc2[31] = 0.0f;
                acc3[0] = 0.0f;
                acc3[1] = 0.0f;
                acc3[2] = 0.0f;
                acc3[3] = 0.0f;
                acc3[4] = 0.0f;
                acc3[5] = 0.0f;
                acc3[6] = 0.0f;
                acc3[7] = 0.0f;
                acc3[8] = 0.0f;
                acc3[9] = 0.0f;
                acc3[10] = 0.0f;
                acc3[11] = 0.0f;
                acc3[12] = 0.0f;
                acc3[13] = 0.0f;
                acc3[14] = 0.0f;
                acc3[15] = 0.0f;
                acc3[16] = 0.0f;
                acc3[17] = 0.0f;
                acc3[18] = 0.0f;
                acc3[19] = 0.0f;
                acc3[20] = 0.0f;
                acc3[21] = 0.0f;
                acc3[22] = 0.0f;
                acc3[23] = 0.0f;
                acc3[24] = 0.0f;
                acc3[25] = 0.0f;
                acc3[26] = 0.0f;
                acc3[27] = 0.0f;
                acc3[28] = 0.0f;
                acc3[29] = 0.0f;
                acc3[30] = 0.0f;
                acc3[31] = 0.0f;
                float partial_fragment[16];
                float partial_fragment2[16];
                int k_blocks = K / 128;
                int scale_groups = k_blocks / 4;
                long long b_scale_base = group;
                b_scale_base = (b_scale_base * (long long)(N / 128) + (long long)n_tile) * (long long)k_blocks;
                long long b_scale_base_u = b_scale_base + (long long)(N / 2 / 128 * k_blocks);
                if (b_scale_primed == 0) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base);
                        b_scale_prefetch[0 + 0] = _v4.x;
                        b_scale_prefetch[0 + 1] = _v4.y;
                        b_scale_prefetch[0 + 2] = _v4.z;
                        b_scale_prefetch[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base_u);
                        b_scale_prefetch_u[0 + 0] = _v4.x;
                        b_scale_prefetch_u[0 + 1] = _v4.y;
                        b_scale_prefetch_u[0 + 2] = _v4.z;
                        b_scale_prefetch_u[0 + 3] = _v4.w;
                    }
                }
                b_scale_primed = 0;
                #pragma unroll 1
                for (int scale_group = 0; scale_group < scale_groups; scale_group++) {
                    float scale_products[4];
                    float scale_products_u[4];
                    mbarrier_wait(scale_full_addr, _phase_scale_full);
                    int scale_base = 0 * 128 * 4 + (unsigned int)(row * 4);
                    #pragma unroll
                    for (int preload_k = 0; preload_k < 4; preload_k++) {
                        float a_s = smem_ascale[scale_base + preload_k];
                        scale_products[preload_k] = a_s * b_scale_prefetch[preload_k];
                        scale_products_u[preload_k] = a_s * b_scale_prefetch_u[preload_k];
                    }
                    mbarrier_arrive(scale_free_addr);
                    _phase_scale_full ^= 1;
                    if (scale_groups > scale_group + 1) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base + (long long)((scale_group + 1) * 4));
                            b_scale_prefetch[0 + 0] = _v4.x;
                            b_scale_prefetch[0 + 1] = _v4.y;
                            b_scale_prefetch[0 + 2] = _v4.z;
                            b_scale_prefetch[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base_u + (long long)((scale_group + 1) * 4));
                            b_scale_prefetch_u[0 + 0] = _v4.x;
                            b_scale_prefetch_u[0 + 1] = _v4.y;
                            b_scale_prefetch_u[0 + 2] = _v4.z;
                            b_scale_prefetch_u[0 + 3] = _v4.w;
                        }
                    }
                    if (scale_groups <= scale_group + 1) {
                        if (has_next != 0) {
                            long long next_b_scale_base = next_group;
                            next_b_scale_base = (next_b_scale_base * (long long)(N / 128) + (long long)next_n_tile) * (long long)k_blocks;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(b_scale + next_b_scale_base);
                                b_scale_prefetch[0 + 0] = _v4.x;
                                b_scale_prefetch[0 + 1] = _v4.y;
                                b_scale_prefetch[0 + 2] = _v4.z;
                                b_scale_prefetch[0 + 3] = _v4.w;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(b_scale + next_b_scale_base + (long long)(N / 2 / 128 * k_blocks));
                                b_scale_prefetch_u[0 + 0] = _v4.x;
                                b_scale_prefetch_u[0 + 1] = _v4.y;
                                b_scale_prefetch_u[0 + 2] = _v4.z;
                                b_scale_prefetch_u[0 + 3] = _v4.w;
                            }
                            b_scale_primed = 1;
                        }
                    }
                    #pragma unroll
                    for (int k_inner = 0; k_inner < 4; k_inner++) {
                        mbarrier_wait(partial_full_addr + (partial_stage) * 8, _phase_partial_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float combined = scale_products[k_inner];
                        float combined_u = scale_products_u[k_inner];
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64)) + (unsigned int)(row_base << 16));
                        tmem_ld_x16(&partial_fragment2[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned long long _fma_acc_scale2_10;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_10));
                        }
                        {
                            unsigned long long _fma_acc_scale2_11;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(partial_fragment2[0]), "f"(partial_fragment2[1]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(partial_fragment2[2]), "f"(partial_fragment2[3]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(partial_fragment2[4]), "f"(partial_fragment2[5]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(partial_fragment2[6]), "f"(partial_fragment2[7]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(partial_fragment2[8]), "f"(partial_fragment2[9]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(partial_fragment2[10]), "f"(partial_fragment2[11]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(partial_fragment2[12]), "f"(partial_fragment2[13]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(partial_fragment2[14]), "f"(partial_fragment2[15]), "l"(_fma_acc_scale2_11));
                        }
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32) + (unsigned int)(row_base << 16));
                        tmem_ld_x16(&partial_fragment2[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32 + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned long long _fma_acc_scale2_12;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_12));
                        }
                        {
                            unsigned long long _fma_acc_scale2_13;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(partial_fragment2[0]), "f"(partial_fragment2[1]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(partial_fragment2[2]), "f"(partial_fragment2[3]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(partial_fragment2[4]), "f"(partial_fragment2[5]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(partial_fragment2[6]), "f"(partial_fragment2[7]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(partial_fragment2[8]), "f"(partial_fragment2[9]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(partial_fragment2[10]), "f"(partial_fragment2[11]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(partial_fragment2[12]), "f"(partial_fragment2[13]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(partial_fragment2[14]), "f"(partial_fragment2[15]), "l"(_fma_acc_scale2_13));
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((partial_free_addr + (partial_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        partial_stage += 1;
                        if (partial_stage == 4) { partial_stage = 0; _phase_partial_full ^= 1; }
                        mbarrier_wait(partial_full_addr + (partial_stage) * 8, _phase_partial_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64)) + (unsigned int)(row_base << 16));
                        tmem_ld_x16(&partial_fragment2[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned long long _fma_acc_scale2_14;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[0]), "+f"(acc2[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[2]), "+f"(acc2[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[4]), "+f"(acc2[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[6]), "+f"(acc2[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[8]), "+f"(acc2[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[10]), "+f"(acc2[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[12]), "+f"(acc2[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[14]), "+f"(acc2[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_14));
                        }
                        {
                            unsigned long long _fma_acc_scale2_15;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[16]), "+f"(acc2[17]) : "f"(partial_fragment2[0]), "f"(partial_fragment2[1]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[18]), "+f"(acc2[19]) : "f"(partial_fragment2[2]), "f"(partial_fragment2[3]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[20]), "+f"(acc2[21]) : "f"(partial_fragment2[4]), "f"(partial_fragment2[5]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[22]), "+f"(acc2[23]) : "f"(partial_fragment2[6]), "f"(partial_fragment2[7]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[24]), "+f"(acc2[25]) : "f"(partial_fragment2[8]), "f"(partial_fragment2[9]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[26]), "+f"(acc2[27]) : "f"(partial_fragment2[10]), "f"(partial_fragment2[11]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[28]), "+f"(acc2[29]) : "f"(partial_fragment2[12]), "f"(partial_fragment2[13]), "l"(_fma_acc_scale2_15));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[30]), "+f"(acc2[31]) : "f"(partial_fragment2[14]), "f"(partial_fragment2[15]), "l"(_fma_acc_scale2_15));
                        }
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32) + (unsigned int)(row_base << 16));
                        tmem_ld_x16(&partial_fragment2[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32 + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            unsigned long long _fma_acc_scale2_16;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[0]), "+f"(acc3[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[2]), "+f"(acc3[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[4]), "+f"(acc3[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[6]), "+f"(acc3[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[8]), "+f"(acc3[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[10]), "+f"(acc3[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[12]), "+f"(acc3[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[14]), "+f"(acc3[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_16));
                        }
                        {
                            unsigned long long _fma_acc_scale2_17;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_17) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[16]), "+f"(acc3[17]) : "f"(partial_fragment2[0]), "f"(partial_fragment2[1]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[18]), "+f"(acc3[19]) : "f"(partial_fragment2[2]), "f"(partial_fragment2[3]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[20]), "+f"(acc3[21]) : "f"(partial_fragment2[4]), "f"(partial_fragment2[5]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[22]), "+f"(acc3[23]) : "f"(partial_fragment2[6]), "f"(partial_fragment2[7]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[24]), "+f"(acc3[25]) : "f"(partial_fragment2[8]), "f"(partial_fragment2[9]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[26]), "+f"(acc3[27]) : "f"(partial_fragment2[10]), "f"(partial_fragment2[11]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[28]), "+f"(acc3[29]) : "f"(partial_fragment2[12]), "f"(partial_fragment2[13]), "l"(_fma_acc_scale2_17));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[30]), "+f"(acc3[31]) : "f"(partial_fragment2[14]), "f"(partial_fragment2[15]), "l"(_fma_acc_scale2_17));
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((partial_free_addr + (partial_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        partial_stage += 1;
                        if (partial_stage == 4) { partial_stage = 0; _phase_partial_full ^= 1; }
                    }
                }
                consumer_ab_cursor += (unsigned int)(k_blocks - 1);
                unsigned int output_ab_stage = consumer_ab_cursor % 7;
                int cluster_row = cta_rank * 128 + row;
                int owns = 0;
                if (cluster_row >= run_begin) {
                    if (cluster_row < run_end) {
                        owns = 1;
                    }
                }
                float hvals[64];
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    unsigned int gp = __as_u32(__float22bfloat162_rn(make_float2(acc0[j], acc1[j])));
                    unsigned int up = __as_u32(__float22bfloat162_rn(make_float2(acc2[j], acc3[j])));
                    float g0 = __uint_as_float(gp << 16);
                    float u0 = __uint_as_float(up << 16);
                    float g1 = __uint_as_float(gp & 4294901760u);
                    float u1 = __uint_as_float(up & 4294901760u);
                    float _exp2_18 = approx_exp2((-g0) * 1.442695f);
                    float s0 = approx_rcp(1.0f + _exp2_18);
                    float _exp2_19 = approx_exp2((-g1) * 1.442695f);
                    float s1 = approx_rcp(1.0f + _exp2_19);
                    unsigned int hp = __as_u32(__float22bfloat162_rn(make_float2(g0 * s0 * u0, g1 * s1 * u1)));
                    hvals[j] = __uint_as_float(hp << 16);
                    hvals[32 + j] = __uint_as_float(hp & 4294901760u);
                }
                float mags[64];
                #pragma unroll
                for (int j_1 = 0; j_1 < 64; j_1++) {
                    mags[j_1] = hvals[j_1];
                }
                mags[0] = fabsf(mags[0]);
                mags[1] = fabsf(mags[1]);
                mags[2] = fabsf(mags[2]);
                mags[3] = fabsf(mags[3]);
                mags[4] = fabsf(mags[4]);
                mags[5] = fabsf(mags[5]);
                mags[6] = fabsf(mags[6]);
                mags[7] = fabsf(mags[7]);
                mags[8] = fabsf(mags[8]);
                mags[9] = fabsf(mags[9]);
                mags[10] = fabsf(mags[10]);
                mags[11] = fabsf(mags[11]);
                mags[12] = fabsf(mags[12]);
                mags[13] = fabsf(mags[13]);
                mags[14] = fabsf(mags[14]);
                mags[15] = fabsf(mags[15]);
                mags[16] = fabsf(mags[16]);
                mags[17] = fabsf(mags[17]);
                mags[18] = fabsf(mags[18]);
                mags[19] = fabsf(mags[19]);
                mags[20] = fabsf(mags[20]);
                mags[21] = fabsf(mags[21]);
                mags[22] = fabsf(mags[22]);
                mags[23] = fabsf(mags[23]);
                mags[24] = fabsf(mags[24]);
                mags[25] = fabsf(mags[25]);
                mags[26] = fabsf(mags[26]);
                mags[27] = fabsf(mags[27]);
                mags[28] = fabsf(mags[28]);
                mags[29] = fabsf(mags[29]);
                mags[30] = fabsf(mags[30]);
                mags[31] = fabsf(mags[31]);
                mags[32] = fabsf(mags[32]);
                mags[33] = fabsf(mags[33]);
                mags[34] = fabsf(mags[34]);
                mags[35] = fabsf(mags[35]);
                mags[36] = fabsf(mags[36]);
                mags[37] = fabsf(mags[37]);
                mags[38] = fabsf(mags[38]);
                mags[39] = fabsf(mags[39]);
                mags[40] = fabsf(mags[40]);
                mags[41] = fabsf(mags[41]);
                mags[42] = fabsf(mags[42]);
                mags[43] = fabsf(mags[43]);
                mags[44] = fabsf(mags[44]);
                mags[45] = fabsf(mags[45]);
                mags[46] = fabsf(mags[46]);
                mags[47] = fabsf(mags[47]);
                mags[48] = fabsf(mags[48]);
                mags[49] = fabsf(mags[49]);
                mags[50] = fabsf(mags[50]);
                mags[51] = fabsf(mags[51]);
                mags[52] = fabsf(mags[52]);
                mags[53] = fabsf(mags[53]);
                mags[54] = fabsf(mags[54]);
                mags[55] = fabsf(mags[55]);
                mags[56] = fabsf(mags[56]);
                mags[57] = fabsf(mags[57]);
                mags[58] = fabsf(mags[58]);
                mags[59] = fabsf(mags[59]);
                mags[60] = fabsf(mags[60]);
                mags[61] = fabsf(mags[61]);
                mags[62] = fabsf(mags[62]);
                mags[63] = fabsf(mags[63]);
                float2 _reg_reduce_max2_20 = {-CAKE_INF, -CAKE_INF};
                row_max_x32_accum(&mags[0], _reg_reduce_max2_20);
                row_max_x32_accum(&mags[32], _reg_reduce_max2_20);
                float absmax = row_max_reduce(_reg_reduce_max2_20);
                int xchg_base = output_ab_stage * 4096;
                rowmax_xchg[xchg_base + acc_wg * 128 + row] = absmax;
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                float other = rowmax_xchg[xchg_base + (1 - acc_wg) * 128 + row];
                float group_max = fmaxf(absmax, other);
                group_max = fmaxf(group_max, 1e-10f);
                float scale = group_max / 448.0f;
                float inv_scale = 1.0f / scale;
                const float2 _scale2_21 = {inv_scale, inv_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(hvals)[_ls], _scale2_21);
                #pragma unroll
                for (int j_2 = 0; j_2 < 64; j_2++) {
                    hvals[j_2] = fmaxf(fminf(hvals[j_2], 448.0f), -448.0f);
                }
                if (owns != 0) {
                    long long out_row = cluster_m_base;
                    out_row += cluster_row;
                    long long q_base = out_row * (long long)(N / 2) + (long long)(n_tile * 128) + (long long)(acc_wg * 64);
                    #pragma unroll
                    for (int vec = 0; vec < 64; vec += 16) {
                        unsigned int _packed_0[4];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"((hvals + vec)[0]), "f"((hvals + vec)[1]),
                                                   "f"((hvals + vec)[2]), "f"((hvals + vec)[3]));
                            _packed_0[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"((hvals + vec)[4]), "f"((hvals + vec)[5]),
                                                   "f"((hvals + vec)[6]), "f"((hvals + vec)[7]));
                            _packed_0[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"((hvals + vec)[8]), "f"((hvals + vec)[9]),
                                                   "f"((hvals + vec)[10]), "f"((hvals + vec)[11]));
                            _packed_0[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"((hvals + vec)[12]), "f"((hvals + vec)[13]),
                                                   "f"((hvals + vec)[14]), "f"((hvals + vec)[15]));
                            _packed_0[3] = _packed;
                        }
                        reinterpret_cast<int4*>(out_q + (q_base + (long long)vec))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    }
                    if (acc_wg == 0) {
                        *(reinterpret_cast<float*>(out_s + (out_row * (long long)(N / 2 / 128) + (long long)n_tile)) + (0)) = scale;
                    }
                }
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(ab_free_addr + (consumer_ab_cursor % 7) * 8);
                    }
                }
                consumer_ab_cursor += 1;
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            unsigned int _phase_producers_done_0 = 0;
            unsigned int _phase_pair_exit_0 = 0;
            if (warp == 0) {
                mbarrier_wait(producers_done_addr, _phase_producers_done_0);
                _phase_producers_done_0 ^= 1;
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(pair_exit_addr), "r"(1 - cta_rank) : "memory");
                }
                mbarrier_wait(pair_exit_addr, _phase_pair_exit_0);
                _phase_pair_exit_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: mma_role ----
    if (warp == 8) {
        { // mma_role_main
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            unsigned int ab_stage = 0;
            unsigned int partial_stage_1 = 0;
            int m_tiles_1 = (M - 1) / 256 + 1;
            int n_tiles_1 = N / 256;
            int total_tiles_1 = m_tiles_1 * n_tiles_1;
            unsigned int _phase_partial_free = 1;
            unsigned int _phase_ab_full = 0;
            if (cta_rank == 0) {
                int nblk_1 = (M - 1) / 128 + 1;
                int g_blk_lo_1 = -1;
                int g_blk_hi_1 = -1;
                if (nblk_1 > lane) {
                    g_blk_lo_1 = m_indices[lane * 128];
                }
                if (nblk_1 > lane + 32) {
                    g_blk_hi_1 = m_indices[(lane + 32) * 128];
                }
                int pair_lo_1 = 0;
                int pair_hi_1 = 0;
                int sc_prev_lo_1 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, lane + 31 & 31);
                int sc_prev_hi_1 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, lane + 31 & 31);
                if (lane == 0) {
                    sc_prev_hi_1 = sc_prev_lo_1;
                }
                int sc_open_lo_1 = 0;
                int sc_open_hi_1 = 0;
                if (nblk_1 > lane) {
                    if (g_blk_lo_1 != sc_prev_lo_1) {
                        sc_open_lo_1 = 1;
                    }
                    if (lane == 0) {
                        sc_open_lo_1 = 1;
                    }
                }
                if (nblk_1 > lane + 32) {
                    if (g_blk_hi_1 != sc_prev_hi_1) {
                        sc_open_hi_1 = 1;
                    }
                }
                unsigned int sc_runs_lo_1 = __ballot_sync(0xFFFFFFFF, sc_open_lo_1 != 0);
                unsigned int sc_runs_hi_1 = __ballot_sync(0xFFFFFFFF, sc_open_hi_1 != 0);
                unsigned int sc_upto_1 = (2 << (unsigned int)lane) - 1;
                int _count_leading_zeros_0 = __clz(sc_runs_lo_1 & sc_upto_1);
                int sc_start_lo_1 = 31 - _count_leading_zeros_0;
                int _count_leading_zeros_1_1 = __clz(sc_runs_lo_1);
                int sc_start_hi_1 = 31 - _count_leading_zeros_1_1;
                if ((sc_runs_hi_1 & sc_upto_1) != 0) {
                    int _count_leading_zeros_2_1 = __clz(sc_runs_hi_1 & sc_upto_1);
                    sc_start_hi_1 = 63 - _count_leading_zeros_2_1;
                }
                int sc_odd_lo_1 = 0;
                int sc_odd_hi_1 = 0;
                if (nblk_1 > lane) {
                    sc_odd_lo_1 = lane - sc_start_lo_1 & 1;
                }
                if (nblk_1 > lane + 32) {
                    sc_odd_hi_1 = lane + 32 - sc_start_hi_1 & 1;
                }
                unsigned int sc_pairs_lo_1 = __ballot_sync(0xFFFFFFFF, sc_odd_lo_1 != 0);
                unsigned int sc_pairs_hi_1 = __ballot_sync(0xFFFFFFFF, sc_odd_hi_1 != 0);
                int sc_n_lo_1 = (int)(unsigned int)__popc(sc_pairs_lo_1);
                int pairs_total_1 = sc_n_lo_1 + (int)(unsigned int)__popc(sc_pairs_hi_1);
                unsigned int sc_sel_1 = sc_pairs_lo_1;
                int sc_rank_1 = lane;
                int sc_base_1 = 0;
                if (sc_n_lo_1 <= lane) {
                    sc_sel_1 = sc_pairs_hi_1;
                    sc_rank_1 = lane - sc_n_lo_1;
                    sc_base_1 = 32;
                }
                unsigned int sc_pos_1 = 0;
                unsigned int sc_step_1 = 16;
                #pragma unroll
                for (int sc_i_1 = 0; sc_i_1 < 5; sc_i_1++) {
                    int sc_below_1 = (int)(unsigned int)__popc(sc_sel_1 & (1 << sc_pos_1 + sc_step_1) - 1);
                    if (sc_below_1 <= sc_rank_1) {
                        sc_pos_1 = sc_pos_1 + sc_step_1;
                    }
                    sc_step_1 = sc_step_1 >> 1;
                }
                if (pairs_total_1 > lane) {
                    pair_lo_1 = sc_base_1 + (int)sc_pos_1;
                }
                total_tiles_1 = pairs_total_1 * n_tiles_1;
                #pragma unroll 1
                for (int tile_id_1 = cluster_id; tile_id_1 < total_tiles_1; tile_id_1 += num_clusters) {
                    int pair_id_1 = tile_id_1 / n_tiles_1;
                    int n_tile_1 = tile_id_1 % n_tiles_1;
                    int b0_1 = __shfl_sync(0xFFFFFFFF, pair_lo_1, pair_id_1 & 31);
                    int b0_hi_1 = __shfl_sync(0xFFFFFFFF, pair_hi_1, pair_id_1 & 31);
                    if (pair_id_1 >= 32) {
                        b0_1 = b0_hi_1;
                    }
                    b0_1 = b0_1 - 1;
                    int group_1 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, b0_1 & 31);
                    int group_hi_1 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, b0_1 & 31);
                    if (b0_1 >= 32) {
                        group_1 = group_hi_1;
                    }
                    int bn_1 = b0_1 + 1;
                    int gn_1 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, bn_1 & 31);
                    int gn_hi_1 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, bn_1 & 31);
                    if (bn_1 >= 32) {
                        gn_1 = gn_hi_1;
                    }
                    int b1_1 = -1;
                    if (bn_1 < nblk_1) {
                        if (gn_1 == group_1) {
                            b1_1 = bn_1;
                        }
                    }
                    int my_blk_1 = b0_1;
                    if (cta_rank == 1) {
                        my_blk_1 = b1_1;
                    }
                    int m_tile_1 = b0_1;
                    int my_rows_1 = 0;
                    if (my_blk_1 >= 0) {
                        m_tile_1 = my_blk_1;
                        my_rows_1 = M - my_blk_1 * 128;
                        if (my_rows_1 > 128) {
                            my_rows_1 = 128;
                        }
                    }
                    int cluster_m_base_1 = m_tile_1 * 128 - cta_rank * 128;
                    int tile_rows_1 = 256;
                    int run_begin_1 = cta_rank * 128;
                    int run_end_1 = run_begin_1 + my_rows_1;
                    int k_blocks_1 = K / 128;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_blocks_1; iter_k++) {
                        mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                        mbarrier_wait(ab_full_addr + (ab_stage) * 8, _phase_ab_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
                        int _mma_b_lo_0 = (((smem_b_half0_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_partials + (partial_stage_1 * 128))), "r"(0) : "memory");
                        elect_commit_cg2_multicast(partial_full_addr + (partial_stage_1) * 8, (uint16_t)(3));
                        partial_stage_1 += 1;
                        if (partial_stage_1 == 4) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                        mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
                        int _mma_b_lo_1 = (((smem_b_half1_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
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
                    "mov.b32 id, 270532624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_partials + (partial_stage_1 * 128))), "r"(0) : "memory");
                        int release_ab = 1;
                        if (iter_k == k_blocks_1 - 1) {
                            release_ab = 0;
                        }
                        if (release_ab != 0) {
                            elect_commit_cg2_multicast(ab_free_addr + (ab_stage) * 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(partial_full_addr + (partial_stage_1) * 8, (uint16_t)(3));
                        ab_stage += 1;
                        if (ab_stage == 7) { ab_stage = 0; _phase_ab_full ^= 1; }
                        partial_stage_1 += 1;
                        if (partial_stage_1 == 4) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            if (lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
            unsigned int ab_stage_1 = 0;
            int m_tiles_2 = (M - 1) / 256 + 1;
            int n_tiles_2 = N / 256;
            int total_tiles_2 = m_tiles_2 * n_tiles_2;
            int nblk_2 = (M - 1) / 128 + 1;
            int g_blk_lo_2 = -1;
            int g_blk_hi_2 = -1;
            if (nblk_2 > lane) {
                g_blk_lo_2 = m_indices[lane * 128];
            }
            if (nblk_2 > lane + 32) {
                g_blk_hi_2 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_2 = 0;
            int pair_hi_2 = 0;
            int sc_prev_lo_2 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, lane + 31 & 31);
            int sc_prev_hi_2 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, lane + 31 & 31);
            if (lane == 0) {
                sc_prev_hi_2 = sc_prev_lo_2;
            }
            int sc_open_lo_2 = 0;
            int sc_open_hi_2 = 0;
            if (nblk_2 > lane) {
                if (g_blk_lo_2 != sc_prev_lo_2) {
                    sc_open_lo_2 = 1;
                }
                if (lane == 0) {
                    sc_open_lo_2 = 1;
                }
            }
            if (nblk_2 > lane + 32) {
                if (g_blk_hi_2 != sc_prev_hi_2) {
                    sc_open_hi_2 = 1;
                }
            }
            unsigned int sc_runs_lo_2 = __ballot_sync(0xFFFFFFFF, sc_open_lo_2 != 0);
            unsigned int sc_runs_hi_2 = __ballot_sync(0xFFFFFFFF, sc_open_hi_2 != 0);
            unsigned int sc_upto_2 = (2 << (unsigned int)lane) - 1;
            int _count_leading_zeros_0_1 = __clz(sc_runs_lo_2 & sc_upto_2);
            int sc_start_lo_2 = 31 - _count_leading_zeros_0_1;
            int _count_leading_zeros_1_2 = __clz(sc_runs_lo_2);
            int sc_start_hi_2 = 31 - _count_leading_zeros_1_2;
            if ((sc_runs_hi_2 & sc_upto_2) != 0) {
                int _count_leading_zeros_2_2 = __clz(sc_runs_hi_2 & sc_upto_2);
                sc_start_hi_2 = 63 - _count_leading_zeros_2_2;
            }
            int sc_odd_lo_2 = 0;
            int sc_odd_hi_2 = 0;
            if (nblk_2 > lane) {
                sc_odd_lo_2 = lane - sc_start_lo_2 & 1;
            }
            if (nblk_2 > lane + 32) {
                sc_odd_hi_2 = lane + 32 - sc_start_hi_2 & 1;
            }
            unsigned int sc_pairs_lo_2 = __ballot_sync(0xFFFFFFFF, sc_odd_lo_2 != 0);
            unsigned int sc_pairs_hi_2 = __ballot_sync(0xFFFFFFFF, sc_odd_hi_2 != 0);
            int sc_n_lo_2 = (int)(unsigned int)__popc(sc_pairs_lo_2);
            int pairs_total_2 = sc_n_lo_2 + (int)(unsigned int)__popc(sc_pairs_hi_2);
            unsigned int sc_sel_2 = sc_pairs_lo_2;
            int sc_rank_2 = lane;
            int sc_base_2 = 0;
            if (sc_n_lo_2 <= lane) {
                sc_sel_2 = sc_pairs_hi_2;
                sc_rank_2 = lane - sc_n_lo_2;
                sc_base_2 = 32;
            }
            unsigned int sc_pos_2 = 0;
            unsigned int sc_step_2 = 16;
            #pragma unroll
            for (int sc_i_2 = 0; sc_i_2 < 5; sc_i_2++) {
                int sc_below_2 = (int)(unsigned int)__popc(sc_sel_2 & (1 << sc_pos_2 + sc_step_2) - 1);
                if (sc_below_2 <= sc_rank_2) {
                    sc_pos_2 = sc_pos_2 + sc_step_2;
                }
                sc_step_2 = sc_step_2 >> 1;
            }
            if (pairs_total_2 > lane) {
                pair_lo_2 = sc_base_2 + (int)sc_pos_2;
            }
            total_tiles_2 = pairs_total_2 * n_tiles_2;
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int tile_id_2 = cluster_id; tile_id_2 < total_tiles_2; tile_id_2 += num_clusters) {
                int pair_id_2 = tile_id_2 / n_tiles_2;
                int n_tile_2 = tile_id_2 % n_tiles_2;
                int b0_2 = __shfl_sync(0xFFFFFFFF, pair_lo_2, pair_id_2 & 31);
                int b0_hi_2 = __shfl_sync(0xFFFFFFFF, pair_hi_2, pair_id_2 & 31);
                if (pair_id_2 >= 32) {
                    b0_2 = b0_hi_2;
                }
                b0_2 = b0_2 - 1;
                int group_2 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, b0_2 & 31);
                int group_hi_2 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, b0_2 & 31);
                if (b0_2 >= 32) {
                    group_2 = group_hi_2;
                }
                int bn_2 = b0_2 + 1;
                int gn_2 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, bn_2 & 31);
                int gn_hi_2 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, bn_2 & 31);
                if (bn_2 >= 32) {
                    gn_2 = gn_hi_2;
                }
                int b1_2 = -1;
                if (bn_2 < nblk_2) {
                    if (gn_2 == group_2) {
                        b1_2 = bn_2;
                    }
                }
                int my_blk_2 = b0_2;
                if (cta_rank == 1) {
                    my_blk_2 = b1_2;
                }
                int m_tile_2 = b0_2;
                int my_rows_2 = 0;
                if (my_blk_2 >= 0) {
                    m_tile_2 = my_blk_2;
                    my_rows_2 = M - my_blk_2 * 128;
                    if (my_rows_2 > 128) {
                        my_rows_2 = 128;
                    }
                }
                int cluster_m_base_2 = m_tile_2 * 128 - cta_rank * 128;
                int tile_rows_2 = 256;
                int run_begin_2 = cta_rank * 128;
                int run_end_2 = run_begin_2 + my_rows_2;
                int k_blocks_2 = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_blocks_2; iter_k_1++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                        tma_3d_gmem2smem_cta2(smem_a_addr + ab_stage_1 * 16384, (&A), 0, m_tile_2 * 128, iter_k_1, ((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        ab_stage_1 += 1;
                        if (ab_stage_1 == 7) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                    }
                }
            }
            if (elect_sync()) {
                #pragma unroll
                for (int _ab_drain = 0; _ab_drain < 7; _ab_drain++) {
                    mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                    ab_stage_1 += 1;
                    if (ab_stage_1 == 7) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                }
                mbarrier_arrive(producers_done_addr);
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp == 10) {
        { // scale_role_main
            int m_tiles_3 = (M - 1) / 256 + 1;
            int n_tiles_3 = N / 256;
            int total_tiles_3 = m_tiles_3 * n_tiles_3;
            int nblk_3 = (M - 1) / 128 + 1;
            int g_blk_lo_3 = -1;
            int g_blk_hi_3 = -1;
            if (nblk_3 > lane) {
                g_blk_lo_3 = m_indices[lane * 128];
            }
            if (nblk_3 > lane + 32) {
                g_blk_hi_3 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_3 = 0;
            int pair_hi_3 = 0;
            int sc_prev_lo_3 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, lane + 31 & 31);
            int sc_prev_hi_3 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, lane + 31 & 31);
            if (lane == 0) {
                sc_prev_hi_3 = sc_prev_lo_3;
            }
            int sc_open_lo_3 = 0;
            int sc_open_hi_3 = 0;
            if (nblk_3 > lane) {
                if (g_blk_lo_3 != sc_prev_lo_3) {
                    sc_open_lo_3 = 1;
                }
                if (lane == 0) {
                    sc_open_lo_3 = 1;
                }
            }
            if (nblk_3 > lane + 32) {
                if (g_blk_hi_3 != sc_prev_hi_3) {
                    sc_open_hi_3 = 1;
                }
            }
            unsigned int sc_runs_lo_3 = __ballot_sync(0xFFFFFFFF, sc_open_lo_3 != 0);
            unsigned int sc_runs_hi_3 = __ballot_sync(0xFFFFFFFF, sc_open_hi_3 != 0);
            unsigned int sc_upto_3 = (2 << (unsigned int)lane) - 1;
            int _count_leading_zeros_0_2 = __clz(sc_runs_lo_3 & sc_upto_3);
            int sc_start_lo_3 = 31 - _count_leading_zeros_0_2;
            int _count_leading_zeros_1_3 = __clz(sc_runs_lo_3);
            int sc_start_hi_3 = 31 - _count_leading_zeros_1_3;
            if ((sc_runs_hi_3 & sc_upto_3) != 0) {
                int _count_leading_zeros_2_3 = __clz(sc_runs_hi_3 & sc_upto_3);
                sc_start_hi_3 = 63 - _count_leading_zeros_2_3;
            }
            int sc_odd_lo_3 = 0;
            int sc_odd_hi_3 = 0;
            if (nblk_3 > lane) {
                sc_odd_lo_3 = lane - sc_start_lo_3 & 1;
            }
            if (nblk_3 > lane + 32) {
                sc_odd_hi_3 = lane + 32 - sc_start_hi_3 & 1;
            }
            unsigned int sc_pairs_lo_3 = __ballot_sync(0xFFFFFFFF, sc_odd_lo_3 != 0);
            unsigned int sc_pairs_hi_3 = __ballot_sync(0xFFFFFFFF, sc_odd_hi_3 != 0);
            int sc_n_lo_3 = (int)(unsigned int)__popc(sc_pairs_lo_3);
            int pairs_total_3 = sc_n_lo_3 + (int)(unsigned int)__popc(sc_pairs_hi_3);
            unsigned int sc_sel_3 = sc_pairs_lo_3;
            int sc_rank_3 = lane;
            int sc_base_3 = 0;
            if (sc_n_lo_3 <= lane) {
                sc_sel_3 = sc_pairs_hi_3;
                sc_rank_3 = lane - sc_n_lo_3;
                sc_base_3 = 32;
            }
            unsigned int sc_pos_3 = 0;
            unsigned int sc_step_3 = 16;
            #pragma unroll
            for (int sc_i_3 = 0; sc_i_3 < 5; sc_i_3++) {
                int sc_below_3 = (int)(unsigned int)__popc(sc_sel_3 & (1 << sc_pos_3 + sc_step_3) - 1);
                if (sc_below_3 <= sc_rank_3) {
                    sc_pos_3 = sc_pos_3 + sc_step_3;
                }
                sc_step_3 = sc_step_3 >> 1;
            }
            if (pairs_total_3 > lane) {
                pair_lo_3 = sc_base_3 + (int)sc_pos_3;
            }
            total_tiles_3 = pairs_total_3 * n_tiles_3;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int tile_id_3 = cluster_id; tile_id_3 < total_tiles_3; tile_id_3 += num_clusters) {
                int pair_id_3 = tile_id_3 / n_tiles_3;
                int n_tile_3 = tile_id_3 % n_tiles_3;
                int b0_3 = __shfl_sync(0xFFFFFFFF, pair_lo_3, pair_id_3 & 31);
                int b0_hi_3 = __shfl_sync(0xFFFFFFFF, pair_hi_3, pair_id_3 & 31);
                if (pair_id_3 >= 32) {
                    b0_3 = b0_hi_3;
                }
                b0_3 = b0_3 - 1;
                int group_3 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, b0_3 & 31);
                int group_hi_3 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, b0_3 & 31);
                if (b0_3 >= 32) {
                    group_3 = group_hi_3;
                }
                int bn_3 = b0_3 + 1;
                int gn_3 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, bn_3 & 31);
                int gn_hi_3 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, bn_3 & 31);
                if (bn_3 >= 32) {
                    gn_3 = gn_hi_3;
                }
                int b1_3 = -1;
                if (bn_3 < nblk_3) {
                    if (gn_3 == group_3) {
                        b1_3 = bn_3;
                    }
                }
                int my_blk_3 = b0_3;
                if (cta_rank == 1) {
                    my_blk_3 = b1_3;
                }
                int m_tile_3 = b0_3;
                int my_rows_3 = 0;
                if (my_blk_3 >= 0) {
                    m_tile_3 = my_blk_3;
                    my_rows_3 = M - my_blk_3 * 128;
                    if (my_rows_3 > 128) {
                        my_rows_3 = 128;
                    }
                }
                int cluster_m_base_3 = m_tile_3 * 128 - cta_rank * 128;
                int tile_rows_3 = 256;
                int run_begin_3 = cta_rank * 128;
                int run_end_3 = run_begin_3 + my_rows_3;
                int k_blocks_3 = K / 128;
                int scale_groups_1 = k_blocks_3 / 4;
                #pragma unroll 1
                for (int scale_group_1 = 0; scale_group_1 < scale_groups_1; scale_group_1++) {
                    mbarrier_wait(scale_free_addr, _phase_scale_free);
                    int stage_base = 0 * 128 * 4;
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; chunk++) {
                        int row_1 = chunk * 32 + lane;
                        int g_row = m_tile_3 * 128 + row_1;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_ascale_addr + (unsigned int)((stage_base + row_1 * 4) * 4)), "l"(a_scale + (g_row * k_blocks_3 + scale_group_1 * 4)), "r"((g_row < M) ? 16 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(scale_full_addr) : "memory");
                    _phase_scale_free ^= 1;
                }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 1; _drain++) {
                mbarrier_wait(scale_free_addr, _phase_scale_free);
                _phase_scale_free ^= 1;
            }
            if (elect_sync()) {
                mbarrier_arrive(producers_done_addr);
            }
        }
    }
    // ---- Role: tma_b_role ----
    if (warp == 11) {
        { // tma_b_role_main
            if (lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
            unsigned int ab_stage_b = 0;
            int m_tiles_b = (M - 1) / 256 + 1;
            int n_tiles_b = N / 256;
            int total_tiles_b = m_tiles_b * n_tiles_b;
            int nblk_4 = (M - 1) / 128 + 1;
            int g_blk_lo_4 = -1;
            int g_blk_hi_4 = -1;
            if (nblk_4 > lane) {
                g_blk_lo_4 = m_indices[lane * 128];
            }
            if (nblk_4 > lane + 32) {
                g_blk_hi_4 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_4 = 0;
            int pair_hi_4 = 0;
            int sc_prev_lo_4 = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, lane + 31 & 31);
            int sc_prev_hi_4 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, lane + 31 & 31);
            if (lane == 0) {
                sc_prev_hi_4 = sc_prev_lo_4;
            }
            int sc_open_lo_4 = 0;
            int sc_open_hi_4 = 0;
            if (nblk_4 > lane) {
                if (g_blk_lo_4 != sc_prev_lo_4) {
                    sc_open_lo_4 = 1;
                }
                if (lane == 0) {
                    sc_open_lo_4 = 1;
                }
            }
            if (nblk_4 > lane + 32) {
                if (g_blk_hi_4 != sc_prev_hi_4) {
                    sc_open_hi_4 = 1;
                }
            }
            unsigned int sc_runs_lo_4 = __ballot_sync(0xFFFFFFFF, sc_open_lo_4 != 0);
            unsigned int sc_runs_hi_4 = __ballot_sync(0xFFFFFFFF, sc_open_hi_4 != 0);
            unsigned int sc_upto_4 = (2 << (unsigned int)lane) - 1;
            int _count_leading_zeros_0_3 = __clz(sc_runs_lo_4 & sc_upto_4);
            int sc_start_lo_4 = 31 - _count_leading_zeros_0_3;
            int _count_leading_zeros_1_4 = __clz(sc_runs_lo_4);
            int sc_start_hi_4 = 31 - _count_leading_zeros_1_4;
            if ((sc_runs_hi_4 & sc_upto_4) != 0) {
                int _count_leading_zeros_2_4 = __clz(sc_runs_hi_4 & sc_upto_4);
                sc_start_hi_4 = 63 - _count_leading_zeros_2_4;
            }
            int sc_odd_lo_4 = 0;
            int sc_odd_hi_4 = 0;
            if (nblk_4 > lane) {
                sc_odd_lo_4 = lane - sc_start_lo_4 & 1;
            }
            if (nblk_4 > lane + 32) {
                sc_odd_hi_4 = lane + 32 - sc_start_hi_4 & 1;
            }
            unsigned int sc_pairs_lo_4 = __ballot_sync(0xFFFFFFFF, sc_odd_lo_4 != 0);
            unsigned int sc_pairs_hi_4 = __ballot_sync(0xFFFFFFFF, sc_odd_hi_4 != 0);
            int sc_n_lo_4 = (int)(unsigned int)__popc(sc_pairs_lo_4);
            int pairs_total_4 = sc_n_lo_4 + (int)(unsigned int)__popc(sc_pairs_hi_4);
            unsigned int sc_sel_4 = sc_pairs_lo_4;
            int sc_rank_4 = lane;
            int sc_base_4 = 0;
            if (sc_n_lo_4 <= lane) {
                sc_sel_4 = sc_pairs_hi_4;
                sc_rank_4 = lane - sc_n_lo_4;
                sc_base_4 = 32;
            }
            unsigned int sc_pos_4 = 0;
            unsigned int sc_step_4 = 16;
            #pragma unroll
            for (int sc_i_4 = 0; sc_i_4 < 5; sc_i_4++) {
                int sc_below_4 = (int)(unsigned int)__popc(sc_sel_4 & (1 << sc_pos_4 + sc_step_4) - 1);
                if (sc_below_4 <= sc_rank_4) {
                    sc_pos_4 = sc_pos_4 + sc_step_4;
                }
                sc_step_4 = sc_step_4 >> 1;
            }
            if (pairs_total_4 > lane) {
                pair_lo_4 = sc_base_4 + (int)sc_pos_4;
            }
            total_tiles_b = pairs_total_4 * n_tiles_b;
            unsigned int _phase_ab_free_1 = 1;
            #pragma unroll 1
            for (int tile_id_b = cluster_id; tile_id_b < total_tiles_b; tile_id_b += num_clusters) {
                int pair_id_4 = tile_id_b / n_tiles_b;
                int n_tile_b = tile_id_b % n_tiles_b;
                int b0_4 = __shfl_sync(0xFFFFFFFF, pair_lo_4, pair_id_4 & 31);
                int b0_hi_4 = __shfl_sync(0xFFFFFFFF, pair_hi_4, pair_id_4 & 31);
                if (pair_id_4 >= 32) {
                    b0_4 = b0_hi_4;
                }
                b0_4 = b0_4 - 1;
                int group_b = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, b0_4 & 31);
                int group_hi_4 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, b0_4 & 31);
                if (b0_4 >= 32) {
                    group_b = group_hi_4;
                }
                int bn_4 = b0_4 + 1;
                int gn_4 = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, bn_4 & 31);
                int gn_hi_4 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, bn_4 & 31);
                if (bn_4 >= 32) {
                    gn_4 = gn_hi_4;
                }
                int b1_4 = -1;
                if (bn_4 < nblk_4) {
                    if (gn_4 == group_b) {
                        b1_4 = bn_4;
                    }
                }
                int my_blk_4 = b0_4;
                if (cta_rank == 1) {
                    my_blk_4 = b1_4;
                }
                int m_tile_b = b0_4;
                int my_rows_4 = 0;
                if (my_blk_4 >= 0) {
                    m_tile_b = my_blk_4;
                    my_rows_4 = M - my_blk_4 * 128;
                    if (my_rows_4 > 128) {
                        my_rows_4 = 128;
                    }
                }
                int cluster_m_base_4 = m_tile_b * 128 - cta_rank * 128;
                int tile_rows_4 = 256;
                int run_begin_4 = cta_rank * 128;
                int run_end_4 = run_begin_4 + my_rows_4;
                int k_blocks_b = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int iter_k_b = 0; iter_k_b < k_blocks_b; iter_k_b++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_b) * 8, _phase_ab_free_1);
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_b_addr + ab_stage_b * 16384), "l"((&B)), "r"(0), "r"(n_tile_b * 128 + cta_rank * 64), "r"(iter_k_b), "r"(group_b),
                               "r"(((ab_full_addr + (ab_stage_b) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_b_addr + ab_stage_b * 16384 + 8192), "l"((&B)), "r"(0), "r"(N / 2 + n_tile_b * 128 + cta_rank * 64), "r"(iter_k_b), "r"(group_b),
                               "r"(((ab_full_addr + (ab_stage_b) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        ab_stage_b += 1;
                        if (ab_stage_b == 7) { ab_stage_b = 0; _phase_ab_free_1 ^= 1; }
                    }
                }
            }
            if (elect_sync()) {
                #pragma unroll
                for (int _ab_drain_b = 0; _ab_drain_b < 7; _ab_drain_b++) {
                    mbarrier_wait(ab_free_addr + (ab_stage_b) * 8, _phase_ab_free_1);
                    ab_stage_b += 1;
                    if (ab_stage_b == 7) { ab_stage_b = 0; _phase_ab_free_1 ^= 1; }
                }
                mbarrier_arrive(producers_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
