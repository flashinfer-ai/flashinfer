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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 6
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_EPI_STORE_PIPE_STAGES 2
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 32768
#define SMEM_SMEM_V1_OFF 17408
#define SMEM_SMEM_V1_STAGE_BYTES 16384
#define SMEM_SMEM_V1_STRIDE 32768
#define SMEM_SMEM_EPI_OFF 197632
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_CLC_RESP_OFF 230400
#define SMEM_CLC_RESP_STAGE_BYTES 16
#define SMEM_CLC_RESP_STRIDE 16
#define SMEM_TOTAL 230528
#define THREADS 224
#define num_pair_items ((m_tiles / 2) * 14)
#define num_items (num_pair_items * 3)
#define tiles_per_group 224

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





__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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


extern "C" {

__global__ __launch_bounds__(224) void
kernel_cake_lm_head_loss_3c22b54935d7781c3959(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, float* __restrict__ STATS_OUT, int M, int m_tiles, int k_iters, int first_chunk, const __grid_constant__ CUtensorMap WS, int ws_slab)
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
    #define mma_done_addr (mbar_base + 48)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 112)
    #define clc_full_addr (mbar_base + 128)
    #define clc_empty_addr (mbar_base + 144)
    #define thr_full_addr (mbar_base + 160)
    #define thr_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    unsigned int cluster_size;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cluster_size));

    // Kernel setup ops
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    __nv_bfloat16* smem_v1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v1_addr = smem + 17408;
    float* smem_epi = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_epi_addr = smem + 197632;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int clc_resp_addr = smem + 230400;
    if (tid == 0) {
        #pragma unroll
        for (int st = 0; st < 6; st++) {
            mbarrier_init(mma_done_addr + (st) * 8, (int)cluster_size >> 1);
        }
        #pragma unroll
        for (int st_1 = 0; st_1 < 2; st_1++) {
            mbarrier_init(clc_empty_addr + (st_1) * 8, (((int)cluster_size >> 1) * 12 + 1) * 32);
        }
    }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // Publish explicit kernel-setup mbarrier initialization.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
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
            int ld_pair_id = cta_rank >> 1;
            unsigned int ld_cluster_id = bid / 4;
            unsigned int ld_num_clusters = num_bids / 4;
            int ld_pairs = (int)cluster_size >> 1;
            unsigned int ld_rank_u = cta_rank;
            unsigned int load_clc_stage = 0;
            unsigned int load_thr_stage = 0;
            unsigned int ld_tile = bid;
            unsigned int _phase_thr_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            for (unsigned int _lp = 0; _lp < num_items; _lp++) {
                if (cta_rank == 0) {
                    mbarrier_wait(thr_empty_addr + (load_thr_stage) * 8, _phase_thr_empty);
                    mbarrier_arrive(thr_full_addr + (load_thr_stage) * 8);
                    load_thr_stage += 1;
                    if (load_thr_stage == 2) { load_thr_stage = 0; _phase_thr_empty ^= 1; }
                }
                if (elect_sync()) {
                    int ld_item_t = ld_tile >> 2;
                    int ld_col_t = (ld_tile & 3) >> 1;
                    int ld_item_i = ld_item_t;
                    int ld_slice = ld_item_i / num_pair_items;
                    int ld_pair_item = ld_item_i % num_pair_items;
                    int ld_k_base = ld_slice * 674;
                    int ld_k_count = ((ld_slice < 2) ? 674 : 672);
                    int ld_pair_bid = ld_pair_item * 2 + (cta_rank & 1);
                    int group = ld_pair_bid / tiles_per_group;
                    int first_m = group * 16;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= 16) ? 16 : remaining);
                    int local = ld_pair_bid % tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int ld_off_m = bid_m * 128;
                    int ld_off_n = (bid_n * 2 + ld_col_t) * 256 + (cta_rank & 1) * 128;
                    #pragma unroll 1
                    for (int ld_k = 0; ld_k < ld_k_count; ld_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int ld_kk = ld_k_base + ld_k;
                        int ld_k0 = ld_kk * 64;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v0_addr + load_stage * 32768 + (unsigned int)(ld_pair_id * 8192)), "l"((&A)), "r"(0), "r"(ld_off_m + ld_pair_id * 64), "r"(ld_kk),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)((21845 & (1 << (int)cluster_size) - 1) << (cta_rank & 1))) : "memory");
                        if (ld_pairs == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v0_addr + load_stage * 32768 + 8192), "l"((&A)), "r"(0), "r"(ld_off_m + 64), "r"(ld_kk),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)((21845 & (1 << (int)cluster_size) - 1) << (cta_rank & 1))) : "memory");
                        }
                        tma_3d_gmem2smem_cta2(smem_v1_addr + load_stage * 32768, (&B), ld_off_n, ld_k0, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v1_addr + load_stage * 32768 + 8192, (&B), ld_off_n + 64, ld_k0, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                mbarrier_wait(clc_full_addr + (load_clc_stage) * 8, _phase_clc_full);
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
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
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
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + load_clc_stage * 8), "r"(0) : "memory");
                load_clc_stage += 1;
                if (load_clc_stage == 2) { load_clc_stage = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                ld_tile = _clc_ctaid_0 + ld_rank_u;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int mma_cluster_id = bid / 4;
            unsigned int mma_num_clusters = num_bids / 4;
            unsigned int mm_rank_u = cta_rank;
            unsigned int mma_clc_stage = 0;
            unsigned int mm_tile = bid;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _mp = 0; _mp < num_items; _mp++) {
                if ((cta_rank & 1) == 0) {
                    int mm_item_t = mm_tile >> 2;
                    int mm_item_i = mm_item_t;
                    int mm_slice = mm_item_i / num_pair_items;
                    int mm_k_last = ((mm_slice < 2) ? 674 : 672);
                    #pragma unroll 1
                    for (int mm_kc = 0; mm_kc < 1; mm_kc++) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        int mm_steps = ((mm_kc < 0) ? 2020 : mm_k_last);
                        #pragma unroll 1
                        for (int mm_k = 0; mm_k < mm_steps; mm_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((mm_k == 0) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_v0_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            int _mma_b_lo_0 = ((((smem_v1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 2048;
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
                    "mov.b32 id, 272696464;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)((1 << (int)cluster_size) - 1));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 6) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3 << (cta_rank >> 1 << 1)));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    }
                }
                mbarrier_wait(clc_full_addr + (mma_clc_stage) * 8, _phase_clc_full_1);
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
                    : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
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
                    : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + mma_clc_stage * 8), "r"(0) : "memory");
                mma_clc_stage += 1;
                if (mma_clc_stage == 2) { mma_clc_stage = 0; _phase_clc_full_1 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                mm_tile = _clc_ctaid_1 + mm_rank_u;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            const int epi_col0 = (warp - 2) / 4 * 256;
            unsigned int ep_cluster_id = bid / 4;
            unsigned int ep_num_clusters = num_bids / 4;
            unsigned int ep_rank_u = cta_rank;
            unsigned int epi_clc_stage = 0;
            unsigned int ep_tile = bid;
            unsigned int _phase_clc_full_2 = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int _ep = 0; _ep < num_items; _ep++) {
                mbarrier_wait(clc_full_addr + (epi_clc_stage) * 8, _phase_clc_full_2);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
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
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + epi_clc_stage * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_2 ^= 1; }
                int ep_item_t = ep_tile >> 2;
                int ep_col_t = (ep_tile & 3) >> 1;
                int ep_item_i = ep_item_t;
                int ep_slice = ep_item_i / num_pair_items;
                int ep_pair_item = ep_item_i % num_pair_items;
                int ep_pair_bid = ep_pair_item * 2 + (cta_rank & 1);
                int group_1 = ep_pair_bid / tiles_per_group;
                int first_m_1 = group_1 * 16;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= 16) ? 16 : remaining_1);
                int local_1 = ep_pair_bid % tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int global_row = bid_m_1 * 128 + local_row;
                int ep_off_n = (bid_n_1 * 2 + ep_col_t) * 256;
                unsigned long long row_out = (unsigned long long)global_row * 7168 + (unsigned long long)ep_off_n;
                #pragma unroll 1
                for (int ep_kc = 0; ep_kc < 1; ep_kc++) {
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * 256;
                    int do_store = first_chunk * ((ep_kc == 0) ? 1 : 0);
                    float pmax = -3e+38f;
                    float psum = 0.0f;
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int epi_store_stage = 0;
                    int ep_row0x = bid_m_1 * 128;
                    #pragma unroll 1
                    for (int boxx = 0; boxx < 8; boxx++) {
                        int stage_row_x = epi_store_stage * 128 + (unsigned int)local_row;
                        int x_staging = smem_epi_addr + epi_store_stage * 16384;
                        if (warp == 2) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        unsigned int s_abs = smem_epi_addr + (unsigned int)(stage_row_x * 128);
                        unsigned int s_swz = s_abs / 8 & 112;
                        #pragma unroll
                        for (int f = 0; f < 2; f++) {
                            int fcolx = boxx * 32 + f * 16;
                            float _tmem_load_0[16];
                            tmem_ld_x16(&_tmem_load_0[0], lane_addr + fcolx);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_epi_addr + ((unsigned int)(stage_row_x * 128) + ((unsigned int)(f * 64) ^ s_swz)))), "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]), "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_epi_addr + ((unsigned int)(stage_row_x * 128) + ((unsigned int)(f * 64 + 16) ^ s_swz)))), "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]), "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_epi_addr + ((unsigned int)(stage_row_x * 128) + ((unsigned int)(f * 64 + 32) ^ s_swz)))), "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]), "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((smem_epi_addr + ((unsigned int)(stage_row_x * 128) + ((unsigned int)(f * 64 + 48) ^ s_swz)))), "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]), "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                if (ep_slice == 0) {
                                    tma_store_2d((&C), ep_off_n + boxx * 32, ep_row0x, x_staging);
                                } else {
                                    int ep_slabx = ep_slice - 1;
                                    tma_store_3d((&WS), ep_off_n + boxx * 32, ep_row0x, ep_slabx, x_staging);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        epi_store_stage += 1;
                        if (epi_store_stage == 2) { epi_store_stage = 0; }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                }
                if (_clc_valid_3 == 0) {
                    break;
                }
                ep_tile = _clc_ctaid_2 + ep_rank_u;
            }
            if (warp == 2) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: sched ----
    if (warp == 6) {
        { // sched_main
            unsigned int _phase_thr_full = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int sched_clc_stage = 0;
                unsigned int sched_thr_stage = 0;
                int sched_pairs = (int)cluster_size >> 1;
                int sched_ctas = sched_pairs * 2;
                #pragma unroll 1
                for (unsigned int _sp = 0; _sp < num_items; _sp++) {
                    mbarrier_wait(thr_full_addr + (sched_thr_stage) * 8, _phase_thr_full);
                    mbarrier_arrive(thr_empty_addr + (sched_thr_stage) * 8);
                    sched_thr_stage += 1;
                    if (sched_thr_stage == 2) { sched_thr_stage = 0; _phase_thr_full ^= 1; }
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    if (sched_ctas > lane) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(clc_full_addr + sched_clc_stage * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16), "r"(clc_full_addr + sched_clc_stage * 8)
                            : "memory");
                    }
                    mbarrier_wait(clc_full_addr + (sched_clc_stage) * 8, _phase_clc_full_3);
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
                        : "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(clc_empty_addr + sched_clc_stage * 8), "r"(0) : "memory");
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (int _st = 0; _st < 2; _st++) {
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                }
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
