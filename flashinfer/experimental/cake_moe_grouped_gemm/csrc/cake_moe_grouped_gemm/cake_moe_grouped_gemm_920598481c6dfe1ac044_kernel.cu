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
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 49152
#define SMEM_SMEM_B1_OFF 33792
#define SMEM_SMEM_B1_STAGE_BYTES 16384
#define SMEM_SMEM_B1_STRIDE 49152
#define SMEM_TOTAL 197632
#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 4
#define NUM_EPILOGUE_WARPS 8
#define CLUSTER_M 256
#define EPI_CHUNK 16

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







__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_moe_grouped_gemm_920598481c6dfe1ac044(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, int* __restrict__ offs, int num_groups, int sum_m, int N, int K, int ldc, unsigned int* __restrict__ sched, int c0, int c1)
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
    #define epi_free0_addr (mbar_base + 80)
    #define acc_full0_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    __nv_bfloat16* smem_b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b1_addr = smem + 33792;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

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
            // epi_free0: 1 barriers, init_count=16
            mbarrier_init(smem + 80, 16);
            // acc_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int num_iters = K / BLOCK_K;
            unsigned int grid_n_u = (unsigned int)(N / 512);
            unsigned int num_groups_u = (unsigned int)num_groups;
            unsigned int num_clusters_u = (unsigned int)num_clusters;
            unsigned int cluster_u = (unsigned int)cluster_id;
            unsigned int total_tiles = 0;
            int p_start = 0;
            #pragma unroll 1
            for (int gp = 0; gp < num_groups_u; gp++) {
                int p_end = offs[gp];
                total_tiles = total_tiles + (unsigned int)((p_end - p_start + CLUSTER_M - 1) / CLUSTER_M) * grid_n_u;
                p_start = p_end;
            }
            unsigned int cur_g = 0;
            unsigned int cum_tiles = 0;
            int cur_start = 0;
            int g_end = offs[0];
            unsigned int cur_mblocks = (unsigned int)((g_end + CLUSTER_M - 1) / CLUSTER_M);
            unsigned int next_cum = cur_mblocks * grid_n_u;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                unsigned int lw_tail = total_tiles % num_clusters_u;
                if (lw_tail * 2 > num_clusters_u) {
                    lw_tail = 0;
                }
                if (cluster_u < lw_tail * 2) {
                    unsigned int li_rem = lw_tail - cluster_u / 2;
                    unsigned int li_g = (unsigned int)num_groups_u - 1;
                    unsigned int li_prev = li_g - 1;
                    int li_end = offs[li_g];
                    int li_start = 0;
                    if (li_g > 0) {
                        li_start = offs[li_prev];
                    }
                    unsigned int li_tiles = (unsigned int)((li_end - li_start + CLUSTER_M - 1) / CLUSTER_M) * grid_n_u;
                    while (li_tiles < li_rem) {
                        li_rem = li_rem - li_tiles;
                        li_g = li_g - 1;
                        li_prev = li_g - 1;
                        li_end = li_start;
                        li_start = 0;
                        if (li_g > 0) {
                            li_start = offs[li_prev];
                        }
                        li_tiles = (unsigned int)((li_end - li_start + CLUSTER_M - 1) / CLUSTER_M) * grid_n_u;
                    }
                    unsigned int li_local = li_tiles - li_rem;
                    unsigned int li_m = li_local / grid_n_u;
                    unsigned int li_n = li_local % grid_n_u;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, (&A), 0, li_start + (int)li_m * CLUSTER_M + cta_rank * BLOCK_M, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 49152, (&B), 0, (int)li_n * 512 + (int)(cluster_u % 2) * 256 + cta_rank * (BLOCK_N / CTA_GROUP), iter_k, (int)li_g, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                unsigned int wide_n = total_tiles - lw_tail;
                #pragma unroll 1
                for (unsigned int t = cluster_u; t < wide_n; t += num_clusters_u) {
                    while (next_cum <= t) {
                        cum_tiles = next_cum;
                        cur_start = g_end;
                        cur_g = cur_g + 1;
                        g_end = offs[cur_g];
                        cur_mblocks = (unsigned int)((g_end - cur_start + CLUSTER_M - 1) / CLUSTER_M);
                        next_cum = cum_tiles + cur_mblocks * grid_n_u;
                    }
                    unsigned int local_t = t - cum_tiles;
                    unsigned int m_block = 0;
                    unsigned int n_block = 0;
                    m_block = local_t / grid_n_u;
                    n_block = local_t % grid_n_u;
                    int off_m = cur_start + (int)m_block * CLUSTER_M + cta_rank * BLOCK_M;
                    int off_n = (int)n_block * 512 + cta_rank * (BLOCK_N / CTA_GROUP);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_iters; iter_k_1++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, (&A), 0, off_m, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 49152, (&B), 0, off_n, iter_k_1, (int)cur_g, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 49152 + 16384, (&B), 0, off_n + 256, iter_k_1, (int)cur_g, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            int num_iters_m = K / BLOCK_K;
            unsigned int grid_n_m = (unsigned int)(N / 512);
            unsigned int num_groups_m = (unsigned int)num_groups;
            unsigned int num_clusters_m = (unsigned int)num_clusters;
            unsigned int cluster_m = (unsigned int)cluster_id;
            unsigned int total_tiles_m = 0;
            int p_start_m = 0;
            #pragma unroll 1
            for (int gp_m = 0; gp_m < num_groups_m; gp_m++) {
                int p_end_m = offs[gp_m];
                total_tiles_m = total_tiles_m + (unsigned int)((p_end_m - p_start_m + CLUSTER_M - 1) / CLUSTER_M) * grid_n_m;
                p_start_m = p_end_m;
            }
            unsigned int cur_g_m = 0;
            unsigned int cum_tiles_m = 0;
            int cur_start_m = 0;
            int g_end_m = offs[0];
            unsigned int cur_mblocks_m = (unsigned int)((g_end_m + CLUSTER_M - 1) / CLUSTER_M);
            unsigned int next_cum_m = cur_mblocks_m * grid_n_m;
            unsigned int _phase_epi_free0 = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                unsigned int lw_tail_1 = total_tiles_m % num_clusters_m;
                if (lw_tail_1 * 2 > num_clusters_m) {
                    lw_tail_1 = 0;
                }
                if (cluster_m < lw_tail_1 * 2) {
                    mbarrier_wait(epi_free0_addr, _phase_epi_free0);
                    #pragma unroll 1
                    for (int iter_h = 0; iter_h < num_iters_m; iter_h++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_h = ((iter_h == 0) ? 1 : 0);
                        int _mma_a_lo_h = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_h = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_h), "r"(_mma_b_lo_h), "r"(tmem_accum), "r"(((init_h) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full0_addr, (uint16_t)(3));
                    mbarrier_wait(epilogue_done_addr, _phase_epi_free0);
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epi_free0 ^= 1;
                }
                unsigned int wide_m = total_tiles_m - lw_tail_1;
                #pragma unroll 1
                for (unsigned int t_m = cluster_m; t_m < wide_m; t_m += num_clusters_m) {
                    while (next_cum_m <= t_m) {
                        cum_tiles_m = next_cum_m;
                        cur_start_m = g_end_m;
                        cur_g_m = cur_g_m + 1;
                        g_end_m = offs[cur_g_m];
                        cur_mblocks_m = (unsigned int)((g_end_m - cur_start_m + CLUSTER_M - 1) / CLUSTER_M);
                        next_cum_m = cum_tiles_m + cur_mblocks_m * grid_n_m;
                    }
                    mbarrier_wait(epi_free0_addr, _phase_epi_free0);
                    if (0 < num_iters_m) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_accum), "r"(0) : "memory");
                    }
                    if (1 < num_iters_m) {
                        mbarrier_wait(tma_full_addr + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 8, ((mma_tma_stage + 1 < 4) ? _phase_tma_full : _phase_tma_full ^ 1));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x7FFF) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x7FFF) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    mbarrier_wait(epilogue_done_addr, _phase_epi_free0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (0 < num_iters_m) {
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_5 = (((smem_b1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_accum + (256))), "r"(0) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (1 < num_iters_m) {
                        int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_7 = (((smem_b1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    #pragma unroll 1
                    for (int iter_k_m = 2; iter_k_m < num_iters_m - 2; iter_k_m++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_1 = (((smem_b1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        uint32_t _elected_8 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(1), "r"(_elected_8) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(1), "r"(_elected_8) : "memory");
                        uint32_t _elected_9 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 2;\n\t"
                    "add.u32 blo, %1, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(1), "r"(_elected_9) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 2;\n\t"
                    "add.u32 blo, %1, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(1), "r"(_elected_9) : "memory");
                        uint32_t _elected_10 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 4;\n\t"
                    "add.u32 blo, %1, 4;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(1), "r"(_elected_10) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 4;\n\t"
                    "add.u32 blo, %1, 4;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(1), "r"(_elected_10) : "memory");
                        uint32_t _elected_11 = elect_sync();
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 6;\n\t"
                    "add.u32 blo, %1, 6;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(1), "r"(_elected_11) : "memory");
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "add.u32 alo, %0, 6;\n\t"
                    "add.u32 blo, %1, 6;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(1), "r"(_elected_11) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (2 < num_iters_m) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    if (3 < num_iters_m) {
                        mbarrier_wait(tma_full_addr + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 8, ((mma_tma_stage + 1 < 4) ? _phase_tma_full : _phase_tma_full ^ 1));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_14 = (((smem_a_addr) >> 4) & 0x7FFF) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
                        int _smem_desc_lo_15 = (((smem_b_addr) >> 4) & 0x7FFF) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    elect_commit_cg2_multicast(acc_full0_addr, (uint16_t)(3));
                    if (2 < num_iters_m) {
                        int _smem_desc_lo_16 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_17 = (((smem_b1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (3 < num_iters_m) {
                        int _smem_desc_lo_18 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_19 = (((smem_b1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epi_free0 ^= 1;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            const int epi_half = (warp - 2) / 4;
            unsigned int grid_n_e = (unsigned int)(N / 512);
            unsigned int num_groups_e = (unsigned int)num_groups;
            unsigned int num_clusters_e = (unsigned int)num_clusters;
            unsigned int cluster_e = (unsigned int)cluster_id;
            unsigned int total_tiles_e = 0;
            int p_start_e = 0;
            #pragma unroll 1
            for (int gp_e = 0; gp_e < num_groups_e; gp_e++) {
                int p_end_e = offs[gp_e];
                total_tiles_e = total_tiles_e + (unsigned int)((p_end_e - p_start_e + CLUSTER_M - 1) / CLUSTER_M) * grid_n_e;
                p_start_e = p_end_e;
            }
            unsigned int cur_g_e = 0;
            unsigned int cum_tiles_e = 0;
            int cur_start_e = 0;
            int g_end_e = offs[0];
            unsigned int cur_mblocks_e = (unsigned int)((g_end_e + CLUSTER_M - 1) / CLUSTER_M);
            unsigned int next_cum_e = cur_mblocks_e * grid_n_e;
            unsigned int lw_tail_2 = total_tiles_e % num_clusters_e;
            if (lw_tail_2 * 2 > num_clusters_e) {
                lw_tail_2 = 0;
            }
            unsigned int _phase_acc_full0 = 0;
            if (cluster_e < lw_tail_2 * 2) {
                unsigned int li_rem_1 = lw_tail_2 - cluster_e / 2;
                unsigned int li_g_1 = (unsigned int)num_groups_e - 1;
                unsigned int li_prev_1 = li_g_1 - 1;
                int li_end_1 = offs[li_g_1];
                int li_start_1 = 0;
                if (li_g_1 > 0) {
                    li_start_1 = offs[li_prev_1];
                }
                unsigned int li_tiles_1 = (unsigned int)((li_end_1 - li_start_1 + CLUSTER_M - 1) / CLUSTER_M) * grid_n_e;
                while (li_tiles_1 < li_rem_1) {
                    li_rem_1 = li_rem_1 - li_tiles_1;
                    li_g_1 = li_g_1 - 1;
                    li_prev_1 = li_g_1 - 1;
                    li_end_1 = li_start_1;
                    li_start_1 = 0;
                    if (li_g_1 > 0) {
                        li_start_1 = offs[li_prev_1];
                    }
                    li_tiles_1 = (unsigned int)((li_end_1 - li_start_1 + CLUSTER_M - 1) / CLUSTER_M) * grid_n_e;
                }
                unsigned int li_local_1 = li_tiles_1 - li_rem_1;
                unsigned int li_m_1 = li_local_1 / grid_n_e;
                unsigned int li_n_1 = li_local_1 % grid_n_e;
                mbarrier_wait(acc_full0_addr, _phase_acc_full0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int _packed_h[64];
                float _tmem_load_h[32];
                int tmem_row_h = taddr + (unsigned int)((cta_rank & 1) * 128 + epi_warp * 32 << 16) + (unsigned int)(epi_half * 128);
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_h[0]), "=f"(_tmem_load_h[1]), "=f"(_tmem_load_h[2]), "=f"(_tmem_load_h[3]), "=f"(_tmem_load_h[4]), "=f"(_tmem_load_h[5]), "=f"(_tmem_load_h[6]), "=f"(_tmem_load_h[7]), "=f"(_tmem_load_h[8]), "=f"(_tmem_load_h[9]), "=f"(_tmem_load_h[10]), "=f"(_tmem_load_h[11]), "=f"(_tmem_load_h[12]), "=f"(_tmem_load_h[13]), "=f"(_tmem_load_h[14]), "=f"(_tmem_load_h[15]), "=f"(_tmem_load_h[16]), "=f"(_tmem_load_h[17]), "=f"(_tmem_load_h[18]), "=f"(_tmem_load_h[19]), "=f"(_tmem_load_h[20]), "=f"(_tmem_load_h[21]), "=f"(_tmem_load_h[22]), "=f"(_tmem_load_h[23]), "=f"(_tmem_load_h[24]), "=f"(_tmem_load_h[25]), "=f"(_tmem_load_h[26]), "=f"(_tmem_load_h[27]), "=f"(_tmem_load_h[28]), "=f"(_tmem_load_h[29]), "=f"(_tmem_load_h[30]), "=f"(_tmem_load_h[31])
                    : "r"(tmem_row_h + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 0], _tmem_load_h[_lp*2+1 + 0]));
                    _packed_h[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 2], _tmem_load_h[_lp*2+1 + 2]));
                    _packed_h[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 4], _tmem_load_h[_lp*2+1 + 4]));
                    _packed_h[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 6], _tmem_load_h[_lp*2+1 + 6]));
                    _packed_h[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 8], _tmem_load_h[_lp*2+1 + 8]));
                    _packed_h[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 10], _tmem_load_h[_lp*2+1 + 10]));
                    _packed_h[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 12], _tmem_load_h[_lp*2+1 + 12]));
                    _packed_h[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 14], _tmem_load_h[_lp*2+1 + 14]));
                    _packed_h[_lp + 7] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 16], _tmem_load_h[_lp*2+1 + 16]));
                    _packed_h[_lp + 8] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 18], _tmem_load_h[_lp*2+1 + 18]));
                    _packed_h[_lp + 9] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 20], _tmem_load_h[_lp*2+1 + 20]));
                    _packed_h[_lp + 10] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 22], _tmem_load_h[_lp*2+1 + 22]));
                    _packed_h[_lp + 11] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 24], _tmem_load_h[_lp*2+1 + 24]));
                    _packed_h[_lp + 12] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 26], _tmem_load_h[_lp*2+1 + 26]));
                    _packed_h[_lp + 13] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 28], _tmem_load_h[_lp*2+1 + 28]));
                    _packed_h[_lp + 14] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 30], _tmem_load_h[_lp*2+1 + 30]));
                    _packed_h[_lp + 15] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_h[0]), "=f"(_tmem_load_h[1]), "=f"(_tmem_load_h[2]), "=f"(_tmem_load_h[3]), "=f"(_tmem_load_h[4]), "=f"(_tmem_load_h[5]), "=f"(_tmem_load_h[6]), "=f"(_tmem_load_h[7]), "=f"(_tmem_load_h[8]), "=f"(_tmem_load_h[9]), "=f"(_tmem_load_h[10]), "=f"(_tmem_load_h[11]), "=f"(_tmem_load_h[12]), "=f"(_tmem_load_h[13]), "=f"(_tmem_load_h[14]), "=f"(_tmem_load_h[15]), "=f"(_tmem_load_h[16]), "=f"(_tmem_load_h[17]), "=f"(_tmem_load_h[18]), "=f"(_tmem_load_h[19]), "=f"(_tmem_load_h[20]), "=f"(_tmem_load_h[21]), "=f"(_tmem_load_h[22]), "=f"(_tmem_load_h[23]), "=f"(_tmem_load_h[24]), "=f"(_tmem_load_h[25]), "=f"(_tmem_load_h[26]), "=f"(_tmem_load_h[27]), "=f"(_tmem_load_h[28]), "=f"(_tmem_load_h[29]), "=f"(_tmem_load_h[30]), "=f"(_tmem_load_h[31])
                    : "r"(tmem_row_h + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 0], _tmem_load_h[_lp*2+1 + 0]));
                    _packed_h[_lp + 16] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 2], _tmem_load_h[_lp*2+1 + 2]));
                    _packed_h[_lp + 17] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 4], _tmem_load_h[_lp*2+1 + 4]));
                    _packed_h[_lp + 18] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 6], _tmem_load_h[_lp*2+1 + 6]));
                    _packed_h[_lp + 19] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 8], _tmem_load_h[_lp*2+1 + 8]));
                    _packed_h[_lp + 20] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 10], _tmem_load_h[_lp*2+1 + 10]));
                    _packed_h[_lp + 21] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 12], _tmem_load_h[_lp*2+1 + 12]));
                    _packed_h[_lp + 22] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 14], _tmem_load_h[_lp*2+1 + 14]));
                    _packed_h[_lp + 23] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 16], _tmem_load_h[_lp*2+1 + 16]));
                    _packed_h[_lp + 24] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 18], _tmem_load_h[_lp*2+1 + 18]));
                    _packed_h[_lp + 25] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 20], _tmem_load_h[_lp*2+1 + 20]));
                    _packed_h[_lp + 26] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 22], _tmem_load_h[_lp*2+1 + 22]));
                    _packed_h[_lp + 27] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 24], _tmem_load_h[_lp*2+1 + 24]));
                    _packed_h[_lp + 28] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 26], _tmem_load_h[_lp*2+1 + 26]));
                    _packed_h[_lp + 29] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 28], _tmem_load_h[_lp*2+1 + 28]));
                    _packed_h[_lp + 30] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 30], _tmem_load_h[_lp*2+1 + 30]));
                    _packed_h[_lp + 31] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_h[0]), "=f"(_tmem_load_h[1]), "=f"(_tmem_load_h[2]), "=f"(_tmem_load_h[3]), "=f"(_tmem_load_h[4]), "=f"(_tmem_load_h[5]), "=f"(_tmem_load_h[6]), "=f"(_tmem_load_h[7]), "=f"(_tmem_load_h[8]), "=f"(_tmem_load_h[9]), "=f"(_tmem_load_h[10]), "=f"(_tmem_load_h[11]), "=f"(_tmem_load_h[12]), "=f"(_tmem_load_h[13]), "=f"(_tmem_load_h[14]), "=f"(_tmem_load_h[15]), "=f"(_tmem_load_h[16]), "=f"(_tmem_load_h[17]), "=f"(_tmem_load_h[18]), "=f"(_tmem_load_h[19]), "=f"(_tmem_load_h[20]), "=f"(_tmem_load_h[21]), "=f"(_tmem_load_h[22]), "=f"(_tmem_load_h[23]), "=f"(_tmem_load_h[24]), "=f"(_tmem_load_h[25]), "=f"(_tmem_load_h[26]), "=f"(_tmem_load_h[27]), "=f"(_tmem_load_h[28]), "=f"(_tmem_load_h[29]), "=f"(_tmem_load_h[30]), "=f"(_tmem_load_h[31])
                    : "r"(tmem_row_h + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 0], _tmem_load_h[_lp*2+1 + 0]));
                    _packed_h[_lp + 32] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 2], _tmem_load_h[_lp*2+1 + 2]));
                    _packed_h[_lp + 33] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 4], _tmem_load_h[_lp*2+1 + 4]));
                    _packed_h[_lp + 34] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 6], _tmem_load_h[_lp*2+1 + 6]));
                    _packed_h[_lp + 35] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 8], _tmem_load_h[_lp*2+1 + 8]));
                    _packed_h[_lp + 36] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 10], _tmem_load_h[_lp*2+1 + 10]));
                    _packed_h[_lp + 37] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 12], _tmem_load_h[_lp*2+1 + 12]));
                    _packed_h[_lp + 38] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 14], _tmem_load_h[_lp*2+1 + 14]));
                    _packed_h[_lp + 39] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 16], _tmem_load_h[_lp*2+1 + 16]));
                    _packed_h[_lp + 40] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 18], _tmem_load_h[_lp*2+1 + 18]));
                    _packed_h[_lp + 41] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 20], _tmem_load_h[_lp*2+1 + 20]));
                    _packed_h[_lp + 42] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 22], _tmem_load_h[_lp*2+1 + 22]));
                    _packed_h[_lp + 43] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 24], _tmem_load_h[_lp*2+1 + 24]));
                    _packed_h[_lp + 44] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 26], _tmem_load_h[_lp*2+1 + 26]));
                    _packed_h[_lp + 45] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 28], _tmem_load_h[_lp*2+1 + 28]));
                    _packed_h[_lp + 46] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 30], _tmem_load_h[_lp*2+1 + 30]));
                    _packed_h[_lp + 47] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_h[0]), "=f"(_tmem_load_h[1]), "=f"(_tmem_load_h[2]), "=f"(_tmem_load_h[3]), "=f"(_tmem_load_h[4]), "=f"(_tmem_load_h[5]), "=f"(_tmem_load_h[6]), "=f"(_tmem_load_h[7]), "=f"(_tmem_load_h[8]), "=f"(_tmem_load_h[9]), "=f"(_tmem_load_h[10]), "=f"(_tmem_load_h[11]), "=f"(_tmem_load_h[12]), "=f"(_tmem_load_h[13]), "=f"(_tmem_load_h[14]), "=f"(_tmem_load_h[15]), "=f"(_tmem_load_h[16]), "=f"(_tmem_load_h[17]), "=f"(_tmem_load_h[18]), "=f"(_tmem_load_h[19]), "=f"(_tmem_load_h[20]), "=f"(_tmem_load_h[21]), "=f"(_tmem_load_h[22]), "=f"(_tmem_load_h[23]), "=f"(_tmem_load_h[24]), "=f"(_tmem_load_h[25]), "=f"(_tmem_load_h[26]), "=f"(_tmem_load_h[27]), "=f"(_tmem_load_h[28]), "=f"(_tmem_load_h[29]), "=f"(_tmem_load_h[30]), "=f"(_tmem_load_h[31])
                    : "r"(tmem_row_h + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 0], _tmem_load_h[_lp*2+1 + 0]));
                    _packed_h[_lp + 48] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 2], _tmem_load_h[_lp*2+1 + 2]));
                    _packed_h[_lp + 49] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 4], _tmem_load_h[_lp*2+1 + 4]));
                    _packed_h[_lp + 50] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 6], _tmem_load_h[_lp*2+1 + 6]));
                    _packed_h[_lp + 51] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 8], _tmem_load_h[_lp*2+1 + 8]));
                    _packed_h[_lp + 52] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 10], _tmem_load_h[_lp*2+1 + 10]));
                    _packed_h[_lp + 53] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 12], _tmem_load_h[_lp*2+1 + 12]));
                    _packed_h[_lp + 54] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 14], _tmem_load_h[_lp*2+1 + 14]));
                    _packed_h[_lp + 55] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 16], _tmem_load_h[_lp*2+1 + 16]));
                    _packed_h[_lp + 56] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 18], _tmem_load_h[_lp*2+1 + 18]));
                    _packed_h[_lp + 57] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 20], _tmem_load_h[_lp*2+1 + 20]));
                    _packed_h[_lp + 58] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 22], _tmem_load_h[_lp*2+1 + 22]));
                    _packed_h[_lp + 59] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 24], _tmem_load_h[_lp*2+1 + 24]));
                    _packed_h[_lp + 60] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 26], _tmem_load_h[_lp*2+1 + 26]));
                    _packed_h[_lp + 61] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 28], _tmem_load_h[_lp*2+1 + 28]));
                    _packed_h[_lp + 62] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_h[_lp*2 + 30], _tmem_load_h[_lp*2+1 + 30]));
                    _packed_h[_lp + 63] = *(uint32_t*)&_bf2;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epi_free0_addr) & 0xFEFFFFFF) : "memory");
                }
                mbarrier_wait(mainloop_done_addr, _phase_acc_full0);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_acc_full0 ^= 1;
                if (li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid < li_end_1) {
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 0))), "r"(_packed_h[(0) + 0]), "r"(_packed_h[(0) + 1]), "r"(_packed_h[(0) + 2]), "r"(_packed_h[(0) + 3]), "r"(_packed_h[(0) + 4]), "r"(_packed_h[(0) + 5]), "r"(_packed_h[(0) + 6]), "r"(_packed_h[(0) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 16))), "r"(_packed_h[(8) + 0]), "r"(_packed_h[(8) + 1]), "r"(_packed_h[(8) + 2]), "r"(_packed_h[(8) + 3]), "r"(_packed_h[(8) + 4]), "r"(_packed_h[(8) + 5]), "r"(_packed_h[(8) + 6]), "r"(_packed_h[(8) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 32))), "r"(_packed_h[(16) + 0]), "r"(_packed_h[(16) + 1]), "r"(_packed_h[(16) + 2]), "r"(_packed_h[(16) + 3]), "r"(_packed_h[(16) + 4]), "r"(_packed_h[(16) + 5]), "r"(_packed_h[(16) + 6]), "r"(_packed_h[(16) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 48))), "r"(_packed_h[(24) + 0]), "r"(_packed_h[(24) + 1]), "r"(_packed_h[(24) + 2]), "r"(_packed_h[(24) + 3]), "r"(_packed_h[(24) + 4]), "r"(_packed_h[(24) + 5]), "r"(_packed_h[(24) + 6]), "r"(_packed_h[(24) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 64))), "r"(_packed_h[(32) + 0]), "r"(_packed_h[(32) + 1]), "r"(_packed_h[(32) + 2]), "r"(_packed_h[(32) + 3]), "r"(_packed_h[(32) + 4]), "r"(_packed_h[(32) + 5]), "r"(_packed_h[(32) + 6]), "r"(_packed_h[(32) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 80))), "r"(_packed_h[(40) + 0]), "r"(_packed_h[(40) + 1]), "r"(_packed_h[(40) + 2]), "r"(_packed_h[(40) + 3]), "r"(_packed_h[(40) + 4]), "r"(_packed_h[(40) + 5]), "r"(_packed_h[(40) + 6]), "r"(_packed_h[(40) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 96))), "r"(_packed_h[(48) + 0]), "r"(_packed_h[(48) + 1]), "r"(_packed_h[(48) + 2]), "r"(_packed_h[(48) + 3]), "r"(_packed_h[(48) + 4]), "r"(_packed_h[(48) + 5]), "r"(_packed_h[(48) + 6]), "r"(_packed_h[(48) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + ((li_start_1 + (int)li_m_1 * CLUSTER_M + cta_rank * BLOCK_M + epi_tid) * ldc + ((int)li_n_1 * 512 + (int)(cluster_e % 2) * 256 + epi_half * 128 + 112))), "r"(_packed_h[(56) + 0]), "r"(_packed_h[(56) + 1]), "r"(_packed_h[(56) + 2]), "r"(_packed_h[(56) + 3]), "r"(_packed_h[(56) + 4]), "r"(_packed_h[(56) + 5]), "r"(_packed_h[(56) + 6]), "r"(_packed_h[(56) + 7]) : "memory");
                }
            }
            unsigned int wide_e = total_tiles_e - lw_tail_2;
            #pragma unroll 1
            for (unsigned int t_e = cluster_e; t_e < wide_e; t_e += num_clusters_e) {
                while (next_cum_e <= t_e) {
                    cum_tiles_e = next_cum_e;
                    cur_start_e = g_end_e;
                    cur_g_e = cur_g_e + 1;
                    g_end_e = offs[cur_g_e];
                    cur_mblocks_e = (unsigned int)((g_end_e - cur_start_e + CLUSTER_M - 1) / CLUSTER_M);
                    next_cum_e = cum_tiles_e + cur_mblocks_e * grid_n_e;
                }
                unsigned int local_e = t_e - cum_tiles_e;
                unsigned int m_block_e = 0;
                unsigned int n_block_e = 0;
                m_block_e = local_e / grid_n_e;
                n_block_e = local_e % grid_n_e;
                int row_g = cur_start_e + (int)m_block_e * CLUSTER_M + cta_rank * BLOCK_M + epi_tid;
                int row_end = offs[cur_g_e];
                mbarrier_wait(acc_full0_addr, _phase_acc_full0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int _packed_0[128];
                float _tmem_load_0[32];
                int tmem_row = taddr + (unsigned int)((cta_rank & 1) * 128 + epi_warp * 32 << 16) + (unsigned int)(epi_half * 128);
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 8] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 9] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 10] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 11] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 12] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 13] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 14] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 15] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 16] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 17] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 18] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 19] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 20] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 21] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 22] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 23] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 24] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 25] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 26] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 27] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 28] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 29] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 30] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 31] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 32] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 33] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 34] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 35] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 36] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 37] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 38] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 39] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 40] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 41] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 42] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 43] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 44] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 45] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 46] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 47] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 48] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 49] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 50] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 51] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 52] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 53] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 54] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 55] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 56] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 57] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 58] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 59] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 60] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 61] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 62] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 63] = *(uint32_t*)&_bf2;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epi_free0_addr) & 0xFEFFFFFF) : "memory");
                }
                mbarrier_wait(mainloop_done_addr, _phase_acc_full0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 256 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 64] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 65] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 66] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 67] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 68] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 69] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 70] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 71] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 72] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 73] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 74] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 75] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 76] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 77] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 78] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 79] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 256 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 80] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 81] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 82] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 83] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 84] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 85] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 86] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 87] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 88] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 89] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 90] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 91] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 92] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 93] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 94] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 95] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 256 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 96] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 97] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 98] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 99] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 100] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 101] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 102] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 103] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 104] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 105] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 106] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 107] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 108] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 109] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 110] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 111] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_row + 256 + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _packed_0[_lp + 112] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                    _packed_0[_lp + 113] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                    _packed_0[_lp + 114] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                    _packed_0[_lp + 115] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                    _packed_0[_lp + 116] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                    _packed_0[_lp + 117] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                    _packed_0[_lp + 118] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                    _packed_0[_lp + 119] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 16], _tmem_load_0[_lp*2+1 + 16]));
                    _packed_0[_lp + 120] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 18], _tmem_load_0[_lp*2+1 + 18]));
                    _packed_0[_lp + 121] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 20], _tmem_load_0[_lp*2+1 + 20]));
                    _packed_0[_lp + 122] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 22], _tmem_load_0[_lp*2+1 + 22]));
                    _packed_0[_lp + 123] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 24], _tmem_load_0[_lp*2+1 + 24]));
                    _packed_0[_lp + 124] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 26], _tmem_load_0[_lp*2+1 + 26]));
                    _packed_0[_lp + 125] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 28], _tmem_load_0[_lp*2+1 + 28]));
                    _packed_0[_lp + 126] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 30], _tmem_load_0[_lp*2+1 + 30]));
                    _packed_0[_lp + 127] = *(uint32_t*)&_bf2;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_acc_full0 ^= 1;
                if (row_g < row_end) {
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 0))), "r"(_packed_0[(0) + 0]), "r"(_packed_0[(0) + 1]), "r"(_packed_0[(0) + 2]), "r"(_packed_0[(0) + 3]), "r"(_packed_0[(0) + 4]), "r"(_packed_0[(0) + 5]), "r"(_packed_0[(0) + 6]), "r"(_packed_0[(0) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 0))), "r"(_packed_0[(64) + 0]), "r"(_packed_0[(64) + 1]), "r"(_packed_0[(64) + 2]), "r"(_packed_0[(64) + 3]), "r"(_packed_0[(64) + 4]), "r"(_packed_0[(64) + 5]), "r"(_packed_0[(64) + 6]), "r"(_packed_0[(64) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 16))), "r"(_packed_0[(8) + 0]), "r"(_packed_0[(8) + 1]), "r"(_packed_0[(8) + 2]), "r"(_packed_0[(8) + 3]), "r"(_packed_0[(8) + 4]), "r"(_packed_0[(8) + 5]), "r"(_packed_0[(8) + 6]), "r"(_packed_0[(8) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 16))), "r"(_packed_0[(72) + 0]), "r"(_packed_0[(72) + 1]), "r"(_packed_0[(72) + 2]), "r"(_packed_0[(72) + 3]), "r"(_packed_0[(72) + 4]), "r"(_packed_0[(72) + 5]), "r"(_packed_0[(72) + 6]), "r"(_packed_0[(72) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 32))), "r"(_packed_0[(16) + 0]), "r"(_packed_0[(16) + 1]), "r"(_packed_0[(16) + 2]), "r"(_packed_0[(16) + 3]), "r"(_packed_0[(16) + 4]), "r"(_packed_0[(16) + 5]), "r"(_packed_0[(16) + 6]), "r"(_packed_0[(16) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 32))), "r"(_packed_0[(80) + 0]), "r"(_packed_0[(80) + 1]), "r"(_packed_0[(80) + 2]), "r"(_packed_0[(80) + 3]), "r"(_packed_0[(80) + 4]), "r"(_packed_0[(80) + 5]), "r"(_packed_0[(80) + 6]), "r"(_packed_0[(80) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 48))), "r"(_packed_0[(24) + 0]), "r"(_packed_0[(24) + 1]), "r"(_packed_0[(24) + 2]), "r"(_packed_0[(24) + 3]), "r"(_packed_0[(24) + 4]), "r"(_packed_0[(24) + 5]), "r"(_packed_0[(24) + 6]), "r"(_packed_0[(24) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 48))), "r"(_packed_0[(88) + 0]), "r"(_packed_0[(88) + 1]), "r"(_packed_0[(88) + 2]), "r"(_packed_0[(88) + 3]), "r"(_packed_0[(88) + 4]), "r"(_packed_0[(88) + 5]), "r"(_packed_0[(88) + 6]), "r"(_packed_0[(88) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 64))), "r"(_packed_0[(32) + 0]), "r"(_packed_0[(32) + 1]), "r"(_packed_0[(32) + 2]), "r"(_packed_0[(32) + 3]), "r"(_packed_0[(32) + 4]), "r"(_packed_0[(32) + 5]), "r"(_packed_0[(32) + 6]), "r"(_packed_0[(32) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 64))), "r"(_packed_0[(96) + 0]), "r"(_packed_0[(96) + 1]), "r"(_packed_0[(96) + 2]), "r"(_packed_0[(96) + 3]), "r"(_packed_0[(96) + 4]), "r"(_packed_0[(96) + 5]), "r"(_packed_0[(96) + 6]), "r"(_packed_0[(96) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 80))), "r"(_packed_0[(40) + 0]), "r"(_packed_0[(40) + 1]), "r"(_packed_0[(40) + 2]), "r"(_packed_0[(40) + 3]), "r"(_packed_0[(40) + 4]), "r"(_packed_0[(40) + 5]), "r"(_packed_0[(40) + 6]), "r"(_packed_0[(40) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 80))), "r"(_packed_0[(104) + 0]), "r"(_packed_0[(104) + 1]), "r"(_packed_0[(104) + 2]), "r"(_packed_0[(104) + 3]), "r"(_packed_0[(104) + 4]), "r"(_packed_0[(104) + 5]), "r"(_packed_0[(104) + 6]), "r"(_packed_0[(104) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 96))), "r"(_packed_0[(48) + 0]), "r"(_packed_0[(48) + 1]), "r"(_packed_0[(48) + 2]), "r"(_packed_0[(48) + 3]), "r"(_packed_0[(48) + 4]), "r"(_packed_0[(48) + 5]), "r"(_packed_0[(48) + 6]), "r"(_packed_0[(48) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 96))), "r"(_packed_0[(112) + 0]), "r"(_packed_0[(112) + 1]), "r"(_packed_0[(112) + 2]), "r"(_packed_0[(112) + 3]), "r"(_packed_0[(112) + 4]), "r"(_packed_0[(112) + 5]), "r"(_packed_0[(112) + 6]), "r"(_packed_0[(112) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + epi_half * 128 + 112))), "r"(_packed_0[(56) + 0]), "r"(_packed_0[(56) + 1]), "r"(_packed_0[(56) + 2]), "r"(_packed_0[(56) + 3]), "r"(_packed_0[(56) + 4]), "r"(_packed_0[(56) + 5]), "r"(_packed_0[(56) + 6]), "r"(_packed_0[(56) + 7]) : "memory");
                    asm volatile("st.global.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(C + (row_g * ldc + ((int)n_block_e * 512 + 256 + epi_half * 128 + 112))), "r"(_packed_0[(120) + 0]), "r"(_packed_0[(120) + 1]), "r"(_packed_0[(120) + 2]), "r"(_packed_0[(120) + 3]), "r"(_packed_0[(120) + 4]), "r"(_packed_0[(120) + 5]), "r"(_packed_0[(120) + 6]), "r"(_packed_0[(120) + 7]) : "memory");
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
